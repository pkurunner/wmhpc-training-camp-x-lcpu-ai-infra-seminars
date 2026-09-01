"""ABI wrapper for the isolated V=16, eight-CTA/head P2S3 entry."""

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
        from flash_kda_C import fwd_vshard8_p2, get_workspace_size
    except ImportError as exc:  # pragma: no cover - target-host diagnostic
        raise RuntimeError(
            "fwd_vshard8_p2 is unavailable; build the isolated one-shot V=16 extension first"
        ) from exc
    if q.ndim != 4 or q.shape[-1] != 128 or any(t.shape != q.shape for t in (k, v, g, out)):
        raise ValueError("vshard8_p2 requires identical [B,T,H,128] q/k/v/g/out tensors")
    if beta.shape != q.shape[:-1]:
        raise ValueError("beta must be [B,T,H]")
    batch, tokens, heads, _ = q.shape
    nseq = cu_seqlens.numel() - 1 if cu_seqlens is not None else batch
    workspace = torch.empty(
        get_workspace_size(batch * tokens, heads, nseq), dtype=torch.uint8, device=q.device
    )
    fwd_vshard8_p2(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        out,
        workspace,
        A_log,
        dt_bias,
        float(lower_bound),
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
    )
