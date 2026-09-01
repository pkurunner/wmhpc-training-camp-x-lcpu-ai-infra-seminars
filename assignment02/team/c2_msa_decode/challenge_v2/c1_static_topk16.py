"""BF16 C=1 decode with a statically unrolled 16-page selected set.

The acceptance ABI fixes ``TOPK=16``.  The ordinary C=1 kernel still carries a
dynamic ``tl.range(0, real_topk)`` loop.  This variant keeps the exact causal
mask and safely masks any inactive page, but emits all 16 page iterations as
compile-time instances.  It is therefore semantically valid even for a short
sequence while allowing Triton to schedule the full selected-page pipeline at
compile time.  Like the no-LSE variant, it has no partial-output or LSE
workspace/store path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import triton
from triton import language as tl

from challenge.prepared_decode import _validate_problem_and_output
from harness.data import DecodeProblem, PAGE_SIZE, TOPK


@triton.jit(do_not_specialize=["decode_query_len"])
def _c1_static_topk16_bf16_kernel(
    q_ptr, kv_cache_ptr, t_ptr, out_ptr, block_table_ptr, seq_lens,
    total_q, gqa_group_size, head_dim, sm_scale, decode_query_len,
    stride_qn, stride_qh, stride_qd,
    stride_kv_blk, stride_kv_h, stride_kv_pos, stride_kv_d,
    stride_th, stride_tn, stride_tk,
    stride_out_n, stride_out_h, stride_out_d, stride_bt_b,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    GQA_SHARDS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_kh_shard = tl.program_id(1)
    pid_kh = pid_kh_shard // GQA_SHARDS
    pid_shard = pid_kh_shard - pid_kh * GQA_SHARDS
    pid_h = pid_kh * gqa_group_size + pid_shard * BLOCK_SIZE_H
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    sm_scale_log2e = sm_scale * 1.4426950409

    seq_len = tl.load(seq_lens + req_id)
    kv_len = tl.maximum(seq_len - decode_query_len + q_offset + 1, 0)
    real_topk = tl.minimum(TOPK, (kv_len + PAGE_SIZE - 1) // PAGE_SIZE)
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    bt_row = block_table_ptr + req_id * stride_bt_b
    off_n = tl.arange(0, PAGE_SIZE)
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

    # TOPK is compile-time 16.  Loads for inactive positions are explicitly
    # masked, so this has the same output as the dynamic loop for short rows.
    for topk_i in tl.static_range(0, TOPK):
        active = topk_i < real_topk
        blk = tl.load(idx_base + topk_i * stride_tk, mask=active, other=0).to(tl.int32)
        page = tl.load(bt_row + blk, mask=active, other=0).to(tl.int64)
        pos = blk * PAGE_SIZE + off_n
        pos_mask = active & (pos < kv_len)
        k = tl.load(
            kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            + off_n[None, :] * stride_kv_pos + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :], other=0.0,
        )
        qk = tl.zeros((BLOCK_SIZE_H, PAGE_SIZE), tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.where(m_i > float("-inf"), tl.exp2(m_i - m_new), 0.0)
        p = tl.where(pos_mask[None, :], tl.exp2(qk - m_new[:, None]), 0.0)
        v = tl.load(
            kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            + off_n[:, None] * stride_kv_pos + (head_dim + off_d[None, :]) * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :], other=0.0,
        )
        acc_o = acc_o * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    out_ptrs = tl.make_block_ptr(
        base=out_ptr + pid_b * stride_out_n + pid_h * stride_out_h,
        shape=(BLOCK_SIZE_H, head_dim), strides=(stride_out_h, stride_out_d),
        offsets=(0, 0), block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D), order=(1, 0),
    )
    tl.store(out_ptrs, (acc_o * inv_l[:, None]).to(out_ptr.dtype.element_ty), boundary_check=(0, 1))


@dataclass(frozen=True)
class StaticTopk16Config:
    gqa_shards: int = 2
    num_warps: int = 2
    num_stages: int = 4

    def validate(self) -> None:
        if self.gqa_shards not in (1, 2, 4):
            raise ValueError("gqa_shards must be one of 1,2,4")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("num_warps must be one of 1,2,4,8")
        if self.num_stages not in (1, 2, 3, 4, 5, 6):
            raise ValueError("num_stages must be one of 1..6")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class StaticTopk16Bf16Decode:
    """Static-16 BF16 runner; no LSE/partial allocations or side effects."""

    def __init__(self, problem: DecodeProblem, output: torch.Tensor, *, config: StaticTopk16Config) -> None:
        _validate_problem_and_output(problem, output)
        config.validate()
        if problem.kv_cache.dtype != torch.bfloat16:
            raise ValueError("static_topk16 candidate is intentionally BF16-only")
        if int(problem.topk_idx.shape[-1]) != TOPK:
            raise ValueError(f"static_topk16 requires fixed TOPK={TOPK}")
        total_q, num_heads, _ = problem.q.shape
        group = num_heads // problem.num_kv_heads
        if group != 16 or group % config.gqa_shards:
            raise ValueError("static_topk16 only supports acceptance GQA group=16")
        self.problem, self.output, self.config = problem, output, config
        self.shard_size = group // config.gqa_shards

    @property
    def metadata(self) -> dict[str, object]:
        return {
            **self.config.as_dict(), "gqa_shard_size": self.shard_size,
            "grid_y": self.problem.num_kv_heads * self.config.gqa_shards,
            "topk_loop": "static_range_16_with_masked_inactive_pages",
            "merge_bypassed": True, "lse_workspace_bytes": 0,
            "caller_output_bytes": self.output.numel() * self.output.element_size(),
            "storage": "bf16", "pdl_effective": False,
        }

    @torch.inference_mode()
    def __call__(self) -> torch.Tensor:
        p, q = self.problem, self.problem.q
        total_q, num_heads, head_dim = q.shape
        _c1_static_topk16_bf16_kernel[(total_q, p.num_kv_heads * self.config.gqa_shards)](
            q, p.kv_cache, p.topk_idx, self.output, p.block_table, p.seq_lens,
            total_q, num_heads // p.num_kv_heads, head_dim, p.sm_scale, p.decode_query_len,
            q.stride(0), q.stride(1), q.stride(2),
            p.kv_cache.stride(0), p.kv_cache.stride(1), p.kv_cache.stride(2), p.kv_cache.stride(3),
            p.topk_idx.stride(0), p.topk_idx.stride(1), p.topk_idx.stride(2),
            self.output.stride(0), self.output.stride(1), self.output.stride(2), p.block_table.stride(0),
            BLOCK_SIZE_H=self.shard_size, BLOCK_SIZE_D=128, GQA_SHARDS=self.config.gqa_shards,
            num_warps=self.config.num_warps, num_stages=self.config.num_stages,
        )
        return self.output
