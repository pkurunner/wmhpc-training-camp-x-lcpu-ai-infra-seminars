"""Shape-prepared launch path for the vendored sparse-decode Triton kernels.

This module does not modify the baseline.  It reuses the exact two JIT kernels
loaded by ``harness.triton_baseline`` while making their workspace lifetime and
split-K launch configuration explicit.  That is useful for vLLM decode, whose
shapes and tensor addresses are normally stable under CUDA graph capture.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from harness.data import DecodeProblem, PAGE_SIZE
from harness import triton_baseline as _baseline_loader


_BASELINE: dict[str, Any] = _baseline_loader._BASELINE
_DECODE_KERNEL = _BASELINE["_gqa_sparse_decode_kernel"]
_MERGE_KERNEL = _BASELINE["_merge_topk_attn_out_kernel"]
_KV_SCALE_ARGS = _BASELINE["_kv_scale_args"]
_FP8_DTYPES = _BASELINE["_FP8_DTYPES"]


def baseline_num_topk_chunks(total_q: int, num_kv_heads: int, max_topk: int) -> int:
    """Reproduce the pinned vLLM wrapper's TARGET_GRID=256 policy exactly."""
    target_grid = 256
    target = max(1, min(max_topk, target_grid // max(1, total_q * num_kv_heads)))
    return 1 << (target.bit_length() - 1)


def _validate_chunks(chunks: int, max_topk: int) -> None:
    if chunks < 1 or chunks > max_topk or chunks & (chunks - 1):
        raise ValueError("num_topk_chunks must be a power of two in [1, max_topk]")


def _validate_warps(name: str, warps: int) -> None:
    if warps not in (1, 2, 4, 8):
        raise ValueError(f"{name} must be one of 1, 2, 4, 8")


def _validate_problem_and_output(problem: DecodeProblem, output: torch.Tensor) -> None:
    """Reject layouts the pinned kernels do not explicitly support."""
    q = problem.q
    if q.device.type != "cuda" or q.ndim != 3 or q.dtype != torch.bfloat16:
        raise ValueError("q must be a rank-3 CUDA BF16 tensor")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    total_q, num_heads, head_dim = q.shape
    if (num_heads, problem.num_kv_heads, head_dim) != (64, 4, 128):
        raise ValueError("challenge acceptance shape is QH=64, KVH=4, D=128")
    if problem.decode_query_len <= 0:
        raise ValueError("decode_query_len must be positive")
    if total_q != problem.seq_lens.numel() * problem.decode_query_len:
        raise ValueError("q length must equal batch * decode_query_len")
    if not math.isfinite(problem.sm_scale) or problem.sm_scale <= 0:
        raise ValueError("sm_scale must be finite and positive")

    kv = problem.kv_cache
    expected_kv_tail = (problem.num_kv_heads, PAGE_SIZE, 2 * head_dim)
    if kv.ndim != 4 or tuple(kv.shape[1:]) != expected_kv_tail:
        raise ValueError("kv_cache must have shape [pages, 4, 128, 256]")
    if kv.device != q.device or kv.dtype not in (torch.bfloat16, *_FP8_DTYPES):
        raise ValueError("kv_cache must be CUDA BF16 or a supported FP8 dtype")
    if not kv.is_contiguous():
        raise ValueError("kv_cache must be contiguous")

    topk = problem.topk_idx
    if topk.shape != (problem.num_kv_heads, total_q, 16):
        raise ValueError("topk_idx must have shape [4, total_q, 16]")
    if topk.device != q.device or topk.dtype != torch.int32 or not topk.is_contiguous():
        raise ValueError("topk_idx must be contiguous CUDA int32")

    batch = int(problem.seq_lens.numel())
    block_table = problem.block_table
    if block_table.ndim != 2 or block_table.shape[0] != batch or block_table.shape[1] < 16:
        raise ValueError("block_table must have shape [batch, >=16]")
    if (
        block_table.device != q.device
        or block_table.dtype != torch.int32
        or not block_table.is_contiguous()
    ):
        raise ValueError("block_table must be contiguous CUDA int32")
    if (
        problem.seq_lens.shape != (batch,)
        or problem.seq_lens.device != q.device
        or problem.seq_lens.dtype != torch.int32
        or not problem.seq_lens.is_contiguous()
    ):
        raise ValueError("seq_lens must be contiguous CUDA int32 [batch]")

    if output.shape != q.shape or output.device != q.device or output.dtype != q.dtype:
        raise ValueError("caller-owned output must match q shape/device/dtype")
    if not output.is_contiguous():
        raise ValueError("caller-owned output must be contiguous")
    input_ptrs = {
        q.data_ptr(),
        kv.data_ptr(),
        topk.data_ptr(),
        block_table.data_ptr(),
        problem.seq_lens.data_ptr(),
    }
    if output.data_ptr() in input_ptrs:
        raise ValueError("caller-owned output must not alias an input tensor")

    scales = (problem.k_scale, problem.v_scale)
    if kv.dtype == torch.bfloat16:
        if problem.k_scale is not None or problem.v_scale is not None:
            raise ValueError("BF16 KV cache must not provide FP8 scales")
        return
    if problem.k_scale is None and problem.v_scale is None:
        return
    if scales[0] is None or scales[1] is None:
        raise ValueError("k_scale and v_scale must be both provided or both None")
    assert scales[0] is not None and scales[1] is not None
    for name, scale in zip(("k_scale", "v_scale"), scales, strict=True):
        if scale.device != q.device or scale.dtype != torch.float32 or not scale.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA float32")
    if scales[0].numel() == 1 and scales[1].numel() == 1:
        return
    expected_scale_shape = (problem.num_kv_heads, kv.shape[0] * PAGE_SIZE)
    if scales[0].shape != expected_scale_shape or scales[1].shape != expected_scale_shape:
        raise ValueError("per-token/head scales must have shape [4, pages*128]")


@dataclass(frozen=True)
class PreparedMetadata:
    num_topk_chunks: int
    decode_num_warps: int
    merge_num_warps: int
    caller_output_bytes: int
    logical_partial_output_bytes: int
    unique_partial_output_bytes: int
    partial_lse_bytes: int
    unique_workspace_bytes: int
    partial_output_aliases_caller_output: bool
    merge_bypassed: bool


class PreparedSparseDecode:
    """Bind one problem shape to persistent output/partial workspaces.

    The instance is intentionally not re-entrant: callers that use concurrent
    streams must create one instance per in-flight decode.  Input tensors may be
    updated in place between calls, matching CUDA-graph pointer stability.
    """

    def __init__(
        self,
        problem: DecodeProblem,
        output: torch.Tensor,
        *,
        num_topk_chunks: int | None = None,
        decode_num_warps: int = 4,
        merge_num_warps: int = 4,
    ) -> None:
        _validate_problem_and_output(problem, output)
        total_q, num_heads, head_dim = problem.q.shape
        max_topk = int(problem.topk_idx.shape[-1])
        chunks = (
            baseline_num_topk_chunks(total_q, problem.num_kv_heads, max_topk)
            if num_topk_chunks is None
            else num_topk_chunks
        )
        _validate_chunks(chunks, max_topk)
        _validate_warps("decode_num_warps", decode_num_warps)
        _validate_warps("merge_num_warps", merge_num_warps)

        self.problem = problem
        self.num_topk_chunks = chunks
        self.decode_num_warps = decode_num_warps
        self.merge_num_warps = merge_num_warps
        self.output = output
        # With one chunk the decode partial is already globally normalized, so
        # write it directly into output and omit the second kernel entirely.
        if chunks == 1:
            self.o_partial = self.output.unsqueeze(0)
        else:
            self.o_partial = torch.empty(
                chunks,
                total_q,
                num_heads,
                head_dim,
                dtype=problem.q.dtype,
                device=problem.q.device,
            )
        self.lse_partial = torch.empty(
            chunks,
            total_q,
            num_heads,
            dtype=torch.float32,
            device=problem.q.device,
        )

        use_fp8 = problem.kv_cache.dtype in _FP8_DTYPES
        self.use_fp8 = use_fp8
        (
            self.k_scale_arg,
            self.v_scale_arg,
            self.stride_ks_h,
            self.stride_ks_t,
            self.stride_vs_h,
            self.stride_vs_t,
            self.kv_scale_mode,
        ) = (
            _KV_SCALE_ARGS(
                self.output,
                problem.num_kv_heads,
                problem.k_scale,
                problem.v_scale,
            )
            if use_fp8
            else (self.output, self.output, 0, 0, 0, 0, 0)
        )
        current_platform = _BASELINE["current_platform"]
        self.use_pdl = bool(current_platform.is_arch_support_pdl())

    @property
    def metadata(self) -> PreparedMetadata:
        logical_partial_bytes = self.o_partial.numel() * self.o_partial.element_size()
        unique_partial_bytes = 0 if self.num_topk_chunks == 1 else logical_partial_bytes
        partial_lse_bytes = self.lse_partial.numel() * self.lse_partial.element_size()
        return PreparedMetadata(
            num_topk_chunks=self.num_topk_chunks,
            decode_num_warps=self.decode_num_warps,
            merge_num_warps=self.merge_num_warps,
            caller_output_bytes=self.output.numel() * self.output.element_size(),
            logical_partial_output_bytes=logical_partial_bytes,
            unique_partial_output_bytes=unique_partial_bytes,
            partial_lse_bytes=partial_lse_bytes,
            unique_workspace_bytes=unique_partial_bytes + partial_lse_bytes,
            partial_output_aliases_caller_output=self.num_topk_chunks == 1,
            merge_bypassed=self.num_topk_chunks == 1,
        )

    @torch.inference_mode()
    def __call__(self) -> torch.Tensor:
        problem = self.problem
        q = problem.q
        kv_cache = problem.kv_cache
        topk_idx = problem.topk_idx
        block_table = problem.block_table
        seq_lens = problem.seq_lens
        total_q, num_heads, head_dim = q.shape
        gqa_group_size = num_heads // problem.num_kv_heads
        pdl_launch = {"launch_pdl": True} if self.use_pdl else {}
        grid = (total_q * self.num_topk_chunks, problem.num_kv_heads)
        _DECODE_KERNEL[grid](
            q,
            kv_cache,
            self.k_scale_arg,
            self.v_scale_arg,
            topk_idx,
            self.o_partial,
            self.lse_partial,
            block_table,
            seq_lens,
            total_q,
            gqa_group_size,
            head_dim,
            topk_idx.shape[-1],
            problem.sm_scale,
            problem.decode_query_len,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            self.stride_ks_h,
            self.stride_ks_t,
            self.stride_vs_h,
            self.stride_vs_t,
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            self.o_partial.stride(0),
            self.o_partial.stride(1),
            self.o_partial.stride(2),
            self.o_partial.stride(3),
            self.lse_partial.stride(0),
            self.lse_partial.stride(1),
            self.lse_partial.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=PAGE_SIZE,
            NUM_TOPK_CHUNKS=self.num_topk_chunks,
            USE_FP8=self.use_fp8,
            KV_SCALE_MODE=self.kv_scale_mode,
            USE_PDL=self.use_pdl,
            num_warps=self.decode_num_warps,
            **pdl_launch,
        )
        if self.num_topk_chunks == 1:
            return self.output

        merge_grid = (total_q, num_heads)
        _MERGE_KERNEL[merge_grid](
            self.o_partial,
            self.lse_partial,
            self.output,
            head_dim,
            self.o_partial.stride(0),
            self.o_partial.stride(1),
            self.o_partial.stride(2),
            self.o_partial.stride(3),
            self.lse_partial.stride(0),
            self.lse_partial.stride(1),
            self.lse_partial.stride(2),
            self.output.stride(0),
            self.output.stride(1),
            self.output.stride(2),
            NUM_TOPK_CHUNKS=self.num_topk_chunks,
            USE_PDL=self.use_pdl,
            num_warps=self.merge_num_warps,
            **pdl_launch,
        )
        return self.output
