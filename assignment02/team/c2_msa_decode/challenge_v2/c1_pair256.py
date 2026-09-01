"""BF16 C=1 sparse decode that fuses two selected 128-token pages per step.

``topk_idx`` maps logical block indices and ``block_table`` maps those blocks
to arbitrary physical pages, so the pair is gathered rather than assumed
contiguous.  Two selected pages are represented as one N=256 tile.  This
changes the online-softmax loop from 16 to 8 iterations while retaining the
same per-token causal/tail mask and the C=1 no-LSE output contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import triton
from triton import language as tl

from challenge.prepared_decode import _validate_problem_and_output
from harness.data import DecodeProblem, PAGE_SIZE, TOPK


PAIR_N = 2 * PAGE_SIZE


@triton.jit(do_not_specialize=["decode_query_len"])
def _c1_pair256_bf16_kernel(
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
    off_n = tl.arange(0, PAIR_N)
    in_second_page = off_n >= PAGE_SIZE
    off_in_page = off_n - tl.where(in_second_page, PAGE_SIZE, 0)
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

    # Exactly eight compile-time pair steps.  Each inactive logical page uses
    # a safe zero pointer and is excluded from both score and value paths.
    for pair_i in tl.static_range(0, TOPK // 2):
        topk0 = 2 * pair_i
        topk1 = topk0 + 1
        active0 = topk0 < real_topk
        active1 = topk1 < real_topk
        blk0 = tl.load(idx_base + topk0 * stride_tk, mask=active0, other=0).to(tl.int32)
        blk1 = tl.load(idx_base + topk1 * stride_tk, mask=active1, other=0).to(tl.int32)
        page0 = tl.load(bt_row + blk0, mask=active0, other=0).to(tl.int64)
        page1 = tl.load(bt_row + blk1, mask=active1, other=0).to(tl.int64)
        active = tl.where(in_second_page, active1, active0)
        block = tl.where(in_second_page, blk1, blk0)
        page = tl.where(in_second_page, page1, page0)
        pos = block * PAGE_SIZE + off_in_page
        pos_mask = active & (pos < kv_len)
        k = tl.load(
            kv_cache_ptr + page[None, :] * stride_kv_blk + pid_kh * stride_kv_h
            + off_in_page[None, :] * stride_kv_pos + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :], other=0.0,
        )
        qk = tl.zeros((BLOCK_SIZE_H, PAIR_N), tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.where(m_i > float("-inf"), tl.exp2(m_i - m_new), 0.0)
        p = tl.where(pos_mask[None, :], tl.exp2(qk - m_new[:, None]), 0.0)
        v = tl.load(
            kv_cache_ptr + page[:, None] * stride_kv_blk + pid_kh * stride_kv_h
            + off_in_page[:, None] * stride_kv_pos + (head_dim + off_d[None, :]) * stride_kv_d,
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
class Pair256Config:
    gqa_shards: int = 2
    num_warps: int = 2
    num_stages: int = 2

    def validate(self) -> None:
        if self.gqa_shards not in (1, 2, 4):
            raise ValueError("gqa_shards must be one of 1,2,4")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("num_warps must be one of 1,2,4,8")
        if self.num_stages not in (1, 2, 3, 4):
            raise ValueError("num_stages must be one of 1..4")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class Pair256Bf16Decode:
    """Two-page gathered C=1 kernel with no partial/LSE workspace."""

    def __init__(self, problem: DecodeProblem, output: torch.Tensor, *, config: Pair256Config) -> None:
        _validate_problem_and_output(problem, output)
        config.validate()
        if problem.kv_cache.dtype != torch.bfloat16:
            raise ValueError("pair256 candidate is intentionally BF16-only")
        if int(problem.topk_idx.shape[-1]) != TOPK:
            raise ValueError(f"pair256 requires fixed TOPK={TOPK}")
        total_q, num_heads, _ = problem.q.shape
        group = num_heads // problem.num_kv_heads
        if group != 16 or group % config.gqa_shards:
            raise ValueError("pair256 only supports acceptance GQA group=16")
        self.problem, self.output, self.config = problem, output, config
        self.shard_size = group // config.gqa_shards

    @property
    def metadata(self) -> dict[str, object]:
        return {
            **self.config.as_dict(), "gqa_shard_size": self.shard_size,
            "grid_y": self.problem.num_kv_heads * self.config.gqa_shards,
            "topk_pairing": "eight static gathered 2x128-page N=256 tiles",
            "logical_to_physical": "topk_idx_then_block_table_per_page",
            "merge_bypassed": True, "lse_workspace_bytes": 0,
            "caller_output_bytes": self.output.numel() * self.output.element_size(),
            "storage": "bf16", "pdl_effective": False,
        }

    @torch.inference_mode()
    def __call__(self) -> torch.Tensor:
        p, q = self.problem, self.problem.q
        total_q, num_heads, head_dim = q.shape
        _c1_pair256_bf16_kernel[(total_q, p.num_kv_heads * self.config.gqa_shards)](
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
