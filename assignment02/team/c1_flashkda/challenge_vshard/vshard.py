"""Python wrapper for the isolated FlashKDA K2 value-shard extension.

The upstream ``flash_kda.fwd`` API intentionally remains untouched.  The
challenge C++ patch adds a distinct ``fwd_vshard`` pybind symbol, which this
wrapper exposes with the same tensor contract as upstream ``fwd``.
"""

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
    """Run the two-CTA-per-head K2 variant.

    The function keeps the upstream ABI.  It deliberately supports only
    `D=128` and two value shards because those are the only shapes proven by
    the CUDA patch and the validation harness.
    """
    try:
        from flash_kda_C import fwd_vshard, get_workspace_size
    except ImportError as exc:  # pragma: no cover - exercised on target host
        raise RuntimeError(
            "fwd_vshard is unavailable. Apply challenge_vshard/"
            "apply_vshard_patch.py to a separate FlashKDA 1ce47ea worktree "
            "and rebuild the extension."
        ) from exc

    if q.ndim != 4 or q.shape[-1] != 128:
        raise ValueError(f"vshard requires q=[B,T,H,128], got {tuple(q.shape)}")
    if any(x.shape != q.shape for x in (k, v, g, out)):
        raise ValueError("q/k/v/g/out must have identical [B,T,H,128] shapes")
    if beta.shape != q.shape[:-1]:
        raise ValueError("beta must have shape [B,T,H]")

    batch, tokens, heads, _ = q.shape
    nseq = cu_seqlens.numel() - 1 if cu_seqlens is not None else batch
    workspace = torch.empty(
        get_workspace_size(batch * tokens, heads, nseq),
        dtype=torch.uint8,
        device=q.device,
    )
    fwd_vshard(
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
