"""ABI wrapper for the isolated value-shard PREFETCH=3, input-stages=3 entry."""

from __future__ import annotations

from typing import Optional

import torch


def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> None:
    try:
        from flash_kda_C import fwd_vshard_p3, get_workspace_size
    except ImportError as exc:
        raise RuntimeError("fresh P1/P2/P3 extension with fwd_vshard_p3 is required") from exc
    if q.ndim != 4 or q.shape[-1] != 128 or any(t.shape != q.shape for t in (k, v, g, out)):
        raise ValueError("prefetch3 requires identical contiguous [B,T,H,128] q/k/v/g/out tensors")
    if beta.shape != q.shape[:-1]:
        raise ValueError("beta must be [B,T,H]")
    batch, tokens, heads, _ = q.shape
    nseq = cu_seqlens.numel() - 1 if cu_seqlens is not None else batch
    workspace = torch.empty(get_workspace_size(batch * tokens, heads, nseq), dtype=torch.uint8, device=q.device)
    fwd_vshard_p3(
        q, k, v, g, beta, float(scale), out, workspace, A_log, dt_bias, float(lower_bound),
        initial_state=initial_state, final_state=final_state, cu_seqlens=cu_seqlens,
    )
