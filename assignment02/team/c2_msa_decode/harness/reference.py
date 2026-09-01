"""FP32 reference for the selected, paged KV blocks used by sparse decode."""

from __future__ import annotations

import torch

from .data import DecodeProblem, PAGE_SIZE


def _dequantize_kv(problem: DecodeProblem) -> torch.Tensor:
    """Dequantize exactly according to vLLM's scalar/per-token-head ABI."""
    cache = problem.kv_cache
    if not cache.dtype.is_floating_point or not str(cache.dtype).startswith("torch.float8"):
        return cache.float()
    if problem.k_scale is None or problem.v_scale is None:
        # vLLM permits no scale tensors, which semantically means scale=1.
        return cache.float()
    if problem.k_scale.numel() == 1:
        k_scale = problem.k_scale.float()
        v_scale = problem.v_scale.float()
    else:
        pages, kv_heads, _, _ = cache.shape
        k_scale = problem.k_scale.view(kv_heads, pages, PAGE_SIZE).permute(1, 0, 2)
        v_scale = problem.v_scale.view(kv_heads, pages, PAGE_SIZE).permute(1, 0, 2)
    d = cache.shape[-1] // 2
    result = torch.empty_like(cache, dtype=torch.float32)
    result[..., :d] = cache[..., :d].float() * k_scale[..., None]
    result[..., d:] = cache[..., d:].float() * v_scale[..., None]
    return result


@torch.inference_mode()
def dense_sparse_attention_reference(problem: DecodeProblem) -> torch.Tensor:
    """Compute selected-block causal attention in FP32, then cast to BF16.

    "Dense" refers to the dense matrix multiplication over every token in the
    selected 16 pages.  It deliberately does not invoke Triton, SDPA, or any
    vLLM component, so it remains an independent correctness oracle.
    """
    q = problem.q.float()
    kv = _dequantize_kv(problem)
    batch, max_blocks = problem.block_table.shape
    total_q, num_q_heads, head_dim = q.shape
    gqa_group = num_q_heads // problem.num_kv_heads
    if total_q != batch * problem.decode_query_len:
        raise ValueError("q length must equal batch * decode_query_len")
    out = torch.empty_like(q)
    token_offsets = torch.arange(PAGE_SIZE, device=q.device)

    for request in range(batch):
        seq_len = int(problem.seq_lens[request].item())
        for local_q in range(problem.decode_query_len):
            q_index = request * problem.decode_query_len + local_q
            kv_len = seq_len - problem.decode_query_len + local_q + 1
            for kv_head in range(problem.num_kv_heads):
                logical_blocks = problem.topk_idx[kv_head, q_index].long()
                if logical_blocks.numel() == 0 or int(logical_blocks.max()) >= max_blocks:
                    raise ValueError("topk contains a logical block outside block_table")
                pages = problem.block_table[request].index_select(0, logical_blocks).long()
                selected = kv.index_select(0, pages)[:, kv_head]
                keys = selected[..., :head_dim].reshape(-1, head_dim)
                values = selected[..., head_dim:].reshape(-1, head_dim)
                absolute_pos = (logical_blocks[:, None] * PAGE_SIZE + token_offsets).reshape(-1)
                visible = absolute_pos < kv_len
                q_group = q[q_index, kv_head * gqa_group : (kv_head + 1) * gqa_group]
                scores = (q_group @ keys.T) * problem.sm_scale
                scores.masked_fill_(~visible[None, :], float("-inf"))
                probs = torch.softmax(scores, dim=-1)
                out[q_index, kv_head * gqa_group : (kv_head + 1) * gqa_group] = probs @ values
    return out.to(problem.q.dtype)
