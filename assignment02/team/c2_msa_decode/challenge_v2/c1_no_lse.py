"""C=1 online-softmax decode that deliberately does not materialize LSE.

For one top-k chunk the output is already globally normalized.  The vendored
general split-K kernel nevertheless computes/stores log2 LSE for a later merge
that C=1 never launches.  This specialized kernel keeps the numerically stable
online softmax state ``(m, l, acc)`` and writes ``acc / l`` directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import triton
from triton import language as tl

from challenge.prepared_decode import _BASELINE, _FP8_DTYPES, _KV_SCALE_ARGS, _validate_problem_and_output
from harness.data import DecodeProblem, PAGE_SIZE


PdlMode = Literal["auto", "on", "off"]


@triton.jit(do_not_specialize=["decode_query_len"])
def _c1_online_softmax_kernel(
    q_ptr, kv_cache_ptr, k_scale_ptr, v_scale_ptr, t_ptr, out_ptr,
    block_table_ptr, seq_lens,
    total_q, gqa_group_size, head_dim, max_topk, sm_scale, decode_query_len,
    stride_qn, stride_qh, stride_qd,
    stride_kv_blk, stride_kv_h, stride_kv_pos, stride_kv_d,
    stride_ks_h, stride_ks_t, stride_vs_h, stride_vs_t,
    stride_th, stride_tn, stride_tk,
    stride_out_n, stride_out_h, stride_out_d, stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    GQA_SHARDS: tl.constexpr,
    USE_FP8: tl.constexpr,
    KV_SCALE_MODE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """One KV-head shard, C=1, with no LSE output/workspace side effect."""
    pid_b = tl.program_id(0)
    pid_kh_shard = tl.program_id(1)
    pid_kh = pid_kh_shard // GQA_SHARDS
    pid_shard = pid_kh_shard - pid_kh * GQA_SHARDS
    pid_h = pid_kh * gqa_group_size + pid_shard * BLOCK_SIZE_H
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    sm_scale_log2e = sm_scale * 1.4426950409

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    seq_len = tl.load(seq_lens + req_id)
    kv_len = tl.maximum(seq_len - decode_query_len + q_offset + 1, 0)
    real_topk = tl.minimum(max_topk, (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K)
    idx_ptr = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    bt_row = block_table_ptr + req_id * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(BLOCK_SIZE_H, head_dim), strides=(stride_qh, stride_qd),
        offsets=(0, 0), block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D), order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_H,), tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), tl.float32)
    for _ in tl.range(0, real_topk):
        blk = tl.load(idx_ptr).to(tl.int32)
        idx_ptr += stride_tk
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_n
        pos_mask = pos < kv_len
        k = tl.load(
            kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            + off_n[None, :] * stride_kv_pos + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :], other=0.0,
        )
        if USE_FP8:
            k = k.to(q.dtype)
            if KV_SCALE_MODE == 1:
                k = (k * tl.load(k_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                k_scale = tl.load(
                    k_scale_ptr + pid_kh * stride_ks_h + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                    mask=pos_mask, other=1.0,
                )
                k = (k * k_scale[None, :]).to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # A fully masked first selected page has m_i=m_new=-inf.  The raw
        # subtraction would form (-inf)-(-inf), i.e. NaN, even though its
        # exact online-softmax contribution is zero.  Make that empty-prefix
        # identity explicit before using alpha in the accumulator recurrence.
        alpha = tl.where(m_i > float("-inf"), tl.exp2(m_i - m_new), 0.0)
        p = tl.where(pos_mask[None, :], tl.exp2(qk - m_new[:, None]), 0.0)
        v = tl.load(
            kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            + off_n[:, None] * stride_kv_pos + (head_dim + off_d[None, :]) * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :], other=0.0,
        )
        if USE_FP8:
            v = v.to(q.dtype)
            if KV_SCALE_MODE == 1:
                v = (v * tl.load(v_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                v_scale = tl.load(
                    v_scale_ptr + pid_kh * stride_vs_h + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                    mask=pos_mask, other=1.0,
                )
                v = (v * v_scale[:, None]).to(q.dtype)
        acc_o = acc_o * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    out_ptrs = tl.make_block_ptr(
        base=out_ptr + pid_b * stride_out_n + pid_h * stride_out_h,
        shape=(BLOCK_SIZE_H, head_dim), strides=(stride_out_h, stride_out_d),
        offsets=(0, 0), block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D), order=(1, 0),
    )
    tl.store(out_ptrs, (acc_o * inv_l[:, None]).to(out_ptr.dtype.element_ty), boundary_check=(0, 1))


@dataclass(frozen=True)
class C1NoLseConfig:
    gqa_shards: int = 1
    num_warps: int = 4
    num_stages: int = 3
    pdl_mode: PdlMode = "off"
    maxnreg: int | None = None

    def validate(self) -> None:
        if self.gqa_shards not in (1, 2, 4):
            raise ValueError("gqa_shards must be 1,2,4")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("num_warps must be 1,2,4,8")
        if self.num_stages not in (1, 2, 3, 4, 5, 6):
            raise ValueError("num_stages must be 1..6")
        if self.pdl_mode not in ("auto", "on", "off"):
            raise ValueError("pdl_mode must be auto,on,off")
        if self.maxnreg is not None and (self.maxnreg < 32 or self.maxnreg > 256 or self.maxnreg % 8):
            raise ValueError("maxnreg must be None or 8-aligned in [32,256]")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class C1NoLseSparseDecode:
    """Persistent C=1 runner with no LSE allocation and output-disjoint shards."""

    def __init__(self, problem: DecodeProblem, output: torch.Tensor, *, config: C1NoLseConfig) -> None:
        _validate_problem_and_output(problem, output)
        config.validate()
        total_q, num_heads, _ = problem.q.shape
        group = num_heads // problem.num_kv_heads
        if group != 16 or group % config.gqa_shards:
            raise ValueError("C1 no-LSE kernel only supports acceptance GQA group=16")
        self.problem, self.output, self.config = problem, output, config
        self.shard_size = group // config.gqa_shards
        self.use_fp8 = problem.kv_cache.dtype in _FP8_DTYPES
        (
            self.k_scale_arg, self.v_scale_arg, self.stride_ks_h, self.stride_ks_t,
            self.stride_vs_h, self.stride_vs_t, self.kv_scale_mode,
        ) = (
            _KV_SCALE_ARGS(output, problem.num_kv_heads, problem.k_scale, problem.v_scale)
            if self.use_fp8 else (output, output, 0, 0, 0, 0, 0)
        )
        platform = bool(_BASELINE["current_platform"].is_arch_support_pdl())
        self.use_pdl = platform if config.pdl_mode == "auto" else config.pdl_mode == "on"

    @property
    def metadata(self) -> dict[str, object]:
        return {
            **self.config.as_dict(), "gqa_shard_size": self.shard_size,
            "grid_y": self.problem.num_kv_heads * self.config.gqa_shards,
            "merge_bypassed": True, "lse_workspace_bytes": 0,
            "caller_output_bytes": self.output.numel() * self.output.element_size(),
            "pdl_effective": self.use_pdl,
        }

    @torch.inference_mode()
    def __call__(self) -> torch.Tensor:
        p, q = self.problem, self.problem.q
        total_q, num_heads, head_dim = q.shape
        options: dict[str, Any] = {"num_warps": self.config.num_warps, "num_stages": self.config.num_stages}
        if self.config.maxnreg is not None:
            options["maxnreg"] = self.config.maxnreg
        pdl_launch: dict[str, bool] = {"launch_pdl": True} if self.use_pdl else {}
        _c1_online_softmax_kernel[(total_q, p.num_kv_heads * self.config.gqa_shards)](
            q, p.kv_cache, self.k_scale_arg, self.v_scale_arg, p.topk_idx, self.output,
            p.block_table, p.seq_lens,
            total_q, num_heads // p.num_kv_heads, head_dim, p.topk_idx.shape[-1], p.sm_scale, p.decode_query_len,
            q.stride(0), q.stride(1), q.stride(2),
            p.kv_cache.stride(0), p.kv_cache.stride(1), p.kv_cache.stride(2), p.kv_cache.stride(3),
            self.stride_ks_h, self.stride_ks_t, self.stride_vs_h, self.stride_vs_t,
            p.topk_idx.stride(0), p.topk_idx.stride(1), p.topk_idx.stride(2),
            self.output.stride(0), self.output.stride(1), self.output.stride(2), p.block_table.stride(0),
            BLOCK_SIZE_K=PAGE_SIZE, BLOCK_SIZE_H=self.shard_size, BLOCK_SIZE_D=128,
            GQA_SHARDS=self.config.gqa_shards, USE_FP8=self.use_fp8,
            KV_SCALE_MODE=self.kv_scale_mode, USE_PDL=self.use_pdl,
            **options, **pdl_launch,
        )
        return self.output
