"""Deterministic synthetic inputs for paged MiniMax-M3 sparse decode."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch


PAGE_SIZE = 128
TOPK = 16
HEAD_DIM = 128
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
DecodeDType = Literal["bf16", "fp8-scalar", "fp8-token"]


@dataclass(frozen=True)
class DecodeProblem:
    """One flattened decode batch following vLLM's sparse-attention ABI."""

    q: torch.Tensor
    kv_cache: torch.Tensor
    block_table: torch.Tensor
    topk_idx: torch.Tensor
    seq_lens: torch.Tensor
    num_kv_heads: int
    decode_query_len: int
    sm_scale: float
    k_scale: torch.Tensor | None
    v_scale: torch.Tensor | None
    storage_dtype: str

    @property
    def batch_size(self) -> int:
        return int(self.seq_lens.numel())

    @property
    def num_q_heads(self) -> int:
        return int(self.q.shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.q.shape[2])


def _fp8_dtype() -> torch.dtype:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("This PyTorch build has no float8_e4m3fn support.")
    return dtype


def _token_scales(
    *, num_kv_heads: int, num_pages: int, device: torch.device, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    # Positive scales with a deliberately non-uniform per-token/head pattern;
    # this catches an erroneous logical-block rather than physical-page index.
    shape = (num_kv_heads, num_pages * PAGE_SIZE)
    k_scale = torch.empty(shape, dtype=torch.float32, device=device).uniform_(
        0.125, 0.5, generator=generator
    )
    v_scale = torch.empty(shape, dtype=torch.float32, device=device).uniform_(
        0.125, 0.5, generator=generator
    )
    return k_scale.contiguous(), v_scale.contiguous()


def _quantize_kv(
    kv_bf16: torch.Tensor,
    mode: DecodeDType,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return an FP8 cache plus its exact harness scale ABI, if requested."""
    if mode == "bf16":
        return kv_bf16, None, None

    num_pages, num_kv_heads, _, two_d = kv_bf16.shape
    d = two_d // 2
    fp8 = _fp8_dtype()
    if mode == "fp8-scalar":
        k_scale = torch.tensor(0.25, dtype=torch.float32, device=kv_bf16.device)
        v_scale = torch.tensor(0.5, dtype=torch.float32, device=kv_bf16.device)
        k_divisor = k_scale
        v_divisor = v_scale
    elif mode == "fp8-token":
        k_scale, v_scale = _token_scales(
            num_kv_heads=num_kv_heads,
            num_pages=num_pages,
            device=kv_bf16.device,
            generator=generator,
        )
        k_divisor = k_scale.view(num_kv_heads, num_pages, PAGE_SIZE).permute(1, 0, 2)
        v_divisor = v_scale.view(num_kv_heads, num_pages, PAGE_SIZE).permute(1, 0, 2)
    else:  # pragma: no cover - Literal keeps this defensive branch honest.
        raise ValueError(f"Unknown storage mode: {mode}")

    out = torch.empty_like(kv_bf16, dtype=fp8)
    out[..., :d] = (kv_bf16[..., :d].float() / k_divisor[..., None]).to(fp8)
    out[..., d:] = (kv_bf16[..., d:].float() / v_divisor[..., None]).to(fp8)
    return out, k_scale, v_scale


def make_decode_problem(
    *,
    batch_size: int,
    device: str | torch.device = "cuda",
    storage_dtype: DecodeDType = "bf16",
    seed: int = 20260819,
    decode_query_len: int = 1,
    max_seq_len: int = 4096,
    topk: int = TOPK,
    page_size: int = PAGE_SIZE,
    num_q_heads: int = NUM_Q_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
) -> DecodeProblem:
    """Create random but valid top-k and paged-KV decode inputs.

    The sequence length of each request is independently sampled from
    ``[topk * page_size, max_seq_len]``.  Every selected logical block is
    visible to its query and maps through a random, globally unique page table.
    """
    if batch_size <= 0 or decode_query_len <= 0:
        raise ValueError("batch_size and decode_query_len must be positive")
    if page_size != PAGE_SIZE:
        raise ValueError(f"The vendored baseline fixes page_size={PAGE_SIZE}")
    if topk != TOPK:
        raise ValueError(f"The acceptance shape fixes topk={TOPK}")
    if head_dim != HEAD_DIM or num_q_heads != NUM_Q_HEADS or num_kv_heads != NUM_KV_HEADS:
        raise ValueError("The acceptance shape is Q=64, KV=4, D=128")
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    min_seq_len = topk * page_size + decode_query_len - 1
    if max_seq_len < min_seq_len:
        raise ValueError(
            f"max_seq_len must be >= {min_seq_len} so every top-k block is visible"
        )

    dev = torch.device(device)
    generator = torch.Generator(device=dev)
    generator.manual_seed(seed)
    max_blocks = math.ceil(max_seq_len / page_size)
    num_pages = batch_size * max_blocks
    total_q = batch_size * decode_query_len

    q = (
        torch.randn(
            total_q, num_q_heads, head_dim, dtype=torch.bfloat16, device=dev, generator=generator
        )
        * 0.1
    ).contiguous()
    kv_bf16 = (
        torch.randn(
            num_pages,
            num_kv_heads,
            page_size,
            2 * head_dim,
            dtype=torch.bfloat16,
            device=dev,
            generator=generator,
        )
        * 0.1
    ).contiguous()
    kv_cache, k_scale, v_scale = _quantize_kv(kv_bf16, storage_dtype, generator=generator)

    # Each logical page across the whole batch maps to a different physical page.
    block_table = torch.randperm(num_pages, device=dev, generator=generator).view(
        batch_size, max_blocks
    ).to(torch.int32).contiguous()
    seq_lens = torch.randint(
        min_seq_len,
        max_seq_len + 1,
        (batch_size,),
        dtype=torch.int32,
        device=dev,
        generator=generator,
    ).contiguous()
    topk_idx = torch.empty(
        num_kv_heads, total_q, topk, dtype=torch.int32, device=dev
    )
    for request in range(batch_size):
        for query_offset in range(decode_query_len):
            query_index = request * decode_query_len + query_offset
            query_pos = int(seq_lens[request].item()) - decode_query_len + query_offset
            visible_blocks = (query_pos + 1 + page_size - 1) // page_size
            # Sampling independently per KV head verifies the [KVH, Q, topk] ABI.
            for kv_head in range(num_kv_heads):
                topk_idx[kv_head, query_index] = torch.randperm(
                    visible_blocks, device=dev, generator=generator
                )[:topk].to(torch.int32)

    return DecodeProblem(
        q=q,
        kv_cache=kv_cache,
        block_table=block_table,
        topk_idx=topk_idx.contiguous(),
        seq_lens=seq_lens,
        num_kv_heads=num_kv_heads,
        decode_query_len=decode_query_len,
        sm_scale=head_dim**-0.5,
        k_scale=k_scale,
        v_scale=v_scale,
        storage_dtype=storage_dtype,
    )
