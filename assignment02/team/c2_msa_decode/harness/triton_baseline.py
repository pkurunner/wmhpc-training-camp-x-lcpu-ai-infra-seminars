"""A dependency-free loader for the vendored vLLM Triton decode baseline.

The source of truth remains ``../vllm_msa_ref/sparse_attn.py``.  At import time
we replace only its two vLLM import lines with upstream Triton imports and a
small platform capability shim.  This avoids copying or silently diverging
from the baseline kernel while keeping the harness runnable without vLLM.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import torch

from .data import DecodeProblem


class _PlatformShim:
    """The sole vLLM platform API used by sparse_attn.py's decode wrapper."""

    def is_arch_support_pdl(self) -> bool:
        requested = os.environ.get("MSA_BASELINE_PDL", "auto").lower()
        if requested in {"0", "false", "off"}:
            return False
        if not torch.cuda.is_available():
            return False
        if requested in {"1", "true", "on"}:
            return True
        return torch.cuda.get_device_capability() >= (9, 0)


def _load_vendored_baseline() -> dict[str, Any]:
    try:
        import triton
        from triton import language as tl
    except ModuleNotFoundError as exc:  # Clear message for CPU-only static environments.
        raise RuntimeError(
            "The Triton baseline requires torch + triton. Use the assignment B300 venv."
        ) from exc

    source_path = Path(__file__).resolve().parents[1] / "vllm_msa_ref" / "sparse_attn.py"
    source = source_path.read_text(encoding="utf-8")
    old_platform = "from vllm.platforms import current_platform"
    old_triton = "from vllm.triton_utils import tl, triton"
    if source.count(old_platform) != 1 or source.count(old_triton) != 1:
        raise RuntimeError("Vendored sparse_attn.py import boundary changed; update this thin adapter.")
    adapted = source.replace(old_platform, "current_platform = _CURRENT_PLATFORM").replace(
        old_triton, "# supplied by harness.triton_baseline: tl, triton"
    )
    namespace: dict[str, Any] = {
        "__name__": "harness._adapted_vllm_sparse_attn",
        "__file__": str(source_path),
        "torch": torch,
        "tl": tl,
        "triton": triton,
        "_CURRENT_PLATFORM": _PlatformShim(),
    }
    exec(compile(adapted, str(source_path), "exec"), namespace, namespace)
    return namespace


_BASELINE = _load_vendored_baseline()
minimax_m3_sparse_attn_decode = _BASELINE["minimax_m3_sparse_attn_decode"]


@torch.inference_mode()
def run_triton_baseline_into(
    problem: DecodeProblem, output: torch.Tensor
) -> torch.Tensor:
    """Run the baseline into caller-owned output, matching the vLLM ABI."""
    if problem.q.device.type != "cuda":
        raise ValueError("Triton baseline requires a CUDA tensor/device")
    if output.shape != problem.q.shape:
        raise ValueError("output shape must match q")
    if output.device != problem.q.device or output.dtype != problem.q.dtype:
        raise ValueError("output device and dtype must match q")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if output.data_ptr() == problem.q.data_ptr():
        raise ValueError("output must not alias q")
    minimax_m3_sparse_attn_decode(
        problem.q,
        problem.kv_cache,
        problem.topk_idx,
        problem.block_table,
        problem.seq_lens,
        problem.num_kv_heads,
        problem.sm_scale,
        output,
        decode_query_len=problem.decode_query_len,
        k_scale=problem.k_scale,
        v_scale=problem.v_scale,
    )
    return output


@torch.inference_mode()
def run_triton_baseline(problem: DecodeProblem) -> torch.Tensor:
    """Convenience API that allocates output; benchmarks should use ``into``."""
    return run_triton_baseline_into(problem, torch.empty_like(problem.q))
