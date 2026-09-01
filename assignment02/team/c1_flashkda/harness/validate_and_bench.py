#!/usr/bin/env python3
"""Correctness gate and latency comparison for FlashKDA C1 vshard.

This script treats the installed upstream ``flash_kda.fwd`` as the baseline
and the separately patched ``flash_kda_C.fwd_vshard`` as the candidate.  It
never reports a speedup unless all requested state modes first pass exact
candidate-vs-baseline comparison and (by default) the upstream torch reference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_vshard import vshard  # noqa: E402


@dataclass(frozen=True)
class Inputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    scale: float
    lower_bound: float


def _load_torch_ref(reference_root: Path) -> Callable[..., None]:
    """Import the exact upstream reference without requiring its test package."""
    ref_file = reference_root / "tests" / "torch_ref.py"
    if not ref_file.is_file():
        raise FileNotFoundError(f"missing exact reference: {ref_file}")
    spec = importlib.util.spec_from_file_location("c1_upstream_torch_ref", ref_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {ref_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.torch_ref


def _load_fla_naive(fla_root: Path) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Load the assignment-pinned FLA naive implementation directly."""
    ref_file = fla_root / "naive.py"
    if not ref_file.is_file():
        raise FileNotFoundError(f"missing FLA KDA reference: {ref_file}")
    spec = importlib.util.spec_from_file_location("c1_fla_naive_ref", ref_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {ref_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_recurrent_kda


def make_inputs(tokens: int, heads: int, seed: int) -> Inputs:
    if tokens <= 0 or heads <= 0:
        raise ValueError("tokens and heads must be positive")
    torch.manual_seed(seed)
    device = torch.device("cuda")
    d = 128
    q = F.normalize(torch.randn(1, tokens, heads, d, device=device), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(1, tokens, heads, d, device=device), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(1, tokens, heads, d, dtype=torch.bfloat16, device=device)
    g = torch.randn(1, tokens, heads, d, dtype=torch.bfloat16, device=device)
    beta = torch.randn(1, tokens, heads, dtype=torch.bfloat16, device=device)
    return Inputs(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        a_log=torch.rand(heads, dtype=torch.float32, device=device),
        dt_bias=torch.rand(heads, d, dtype=torch.float32, device=device),
        scale=1.0 / math.sqrt(d),
        lower_bound=-5.0,
    )


def state_tensors(mode: str, heads: int, seed: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if mode == "none":
        return None, None
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    initial = torch.randn(1, heads, 128, 128, dtype=dtype, device="cuda", generator=generator)
    return initial.contiguous(), torch.zeros_like(initial)


def invoke(
    fn: Callable[..., None],
    x: Inputs,
    initial_state: Optional[torch.Tensor],
    final_state: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    out = torch.zeros_like(x.v)
    fn(
        x.q,
        x.k,
        x.v,
        x.g,
        x.beta,
        x.scale,
        out,
        A_log=x.a_log,
        dt_bias=x.dt_bias,
        lower_bound=x.lower_bound,
        initial_state=initial_state,
        final_state=final_state,
    )
    return out, final_state


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def require_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        print(f"PASS exact: {name}")
        return
    raise AssertionError(f"FAIL exact: {name}; max_abs={max_abs(actual, expected):.7g}")


def require_close(name: str, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float) -> None:
    try:
        # FLA's numerical oracle keeps recurrent state in fp32 while FlashKDA
        # intentionally stores the bf16-state mode in bf16.  The gate is about
        # numerical agreement; dtype identity is covered separately by the
        # exact baseline-vs-vshard comparison above.
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, check_dtype=False
        )
    except AssertionError as exc:
        raise AssertionError(
            f"FAIL close: {name}; rtol={rtol}, atol={atol}, max_abs={max_abs(actual, expected):.7g}"
        ) from exc
    print(f"PASS close: {name} (rtol={rtol}, atol={atol}, max_abs={max_abs(actual, expected):.7g})")


def fla_reference(
    naive_recurrent: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    torch_ref_module: object,
    x: Inputs,
    initial_state: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt raw FlashKDA inputs/state ABI to ``fla_kda_ref.naive``.

    Raw FlashKDA stores state as [V,K], whereas FLA uses [K,V].  The gate and
    beta preprocessing below deliberately reuse the upstream exact test's
    tanh-approx helpers, isolating the remaining difference to FLA's recurrent
    math rather than comparing different model inputs.
    """
    q = torch_ref_module.l2_normalize_kernel_match(x.q)
    k = torch_ref_module.l2_normalize_kernel_match(x.k)
    a_exp = torch_ref_module.fp32_ex2_ftz(torch_ref_module.LOG2E * x.a_log).view(1, 1, -1, 1)
    gated = torch_ref_module.sigmoid_ext.sigmoid_tanh_fp32(a_exp * (x.g.float() + x.dt_bias.view(1, 1, *x.dt_bias.shape)))
    g_natural_log = x.lower_bound * gated
    beta_activated = torch_ref_module.sigmoid_ext.sigmoid_tanh_fp32(x.beta.float())
    fla_initial = None if initial_state is None else initial_state.float().transpose(-1, -2).contiguous()
    out, final_state = naive_recurrent(
        q=q,
        k=k,
        v=x.v,
        g=g_natural_log,
        beta=beta_activated,
        scale=x.scale,
        initial_state=fla_initial,
        output_final_state=True,
    )
    # Convert FLA's logical [K,V] result back to the raw FlashKDA [V,K] ABI.
    return out, final_state.transpose(-1, -2).contiguous()


def validate_mode(
    baseline: Callable[..., None],
    torch_ref: Optional[Callable[..., None]],
    torch_ref_module: Optional[object],
    fla_naive: Optional[Callable[..., tuple[torch.Tensor, torch.Tensor]]],
    x: Inputs,
    mode: str,
    state_seed: int,
) -> None:
    initial, final_base = state_tensors(mode, x.q.shape[2], state_seed)
    base_initial = None if initial is None else initial.clone()
    out_base, final_base = invoke(baseline, x, base_initial, final_base)
    torch.cuda.synchronize()

    candidate_initial = None if initial is None else initial.clone()
    _, final_candidate = state_tensors(mode, x.q.shape[2], state_seed + 1)
    out_candidate, final_candidate = invoke(vshard.fwd, x, candidate_initial, final_candidate)
    torch.cuda.synchronize()

    require_exact(f"{mode}/baseline_vs_vshard/output", out_candidate, out_base)
    if final_base is not None and final_candidate is not None:
        require_exact(f"{mode}/baseline_vs_vshard/final_state", final_candidate, final_base)

    if torch_ref is None:
        ref_initial = None
    else:
        ref_initial = None if initial is None else initial.clone()
        ref_final = None if initial is None else torch.zeros_like(initial)
        out_ref = torch.zeros_like(x.v)
        torch_ref(
            x.q,
            x.k,
            x.v,
            x.g,
            x.beta,
            x.scale,
            out_ref,
            A_log=x.a_log,
            dt_bias=x.dt_bias,
            lower_bound=x.lower_bound,
            initial_state=ref_initial,
            final_state=ref_final,
        )
        torch.cuda.synchronize()
        require_exact(f"{mode}/vshard_vs_torch_ref/output", out_candidate, out_ref)
        if final_candidate is not None and ref_final is not None:
            require_exact(f"{mode}/vshard_vs_torch_ref/final_state", final_candidate, ref_final)

    if fla_naive is not None:
        if torch_ref_module is None:
            raise RuntimeError("FLA adapter requires the upstream torch_ref helper module")
        fla_out, fla_final = fla_reference(fla_naive, torch_ref_module, x, initial)
        torch.cuda.synchronize()
        require_close(f"{mode}/vshard_vs_fla_naive/output", out_candidate, fla_out, rtol=2e-2, atol=2e-2)
        if final_candidate is not None:
            require_close(f"{mode}/vshard_vs_fla_naive/final_state", final_candidate, fla_final, rtol=5e-2, atol=5e-2)


def event_ms(fn: Callable[[], None], warmup: int, iters: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for start, end in zip(starts, ends):
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        samples.extend(float(start.elapsed_time(end)) for start, end in zip(starts, ends))
    return samples


def benchmark(
    baseline: Callable[..., None], x: Inputs, mode: str, warmup: int, iters: int, repeats: int
) -> dict[str, float]:
    initial, final_base = state_tensors(mode, x.q.shape[2], 17)
    candidate_initial = None if initial is None else initial.clone()
    _, final_candidate = state_tensors(mode, x.q.shape[2], 19)
    out_base = torch.empty_like(x.v)
    out_candidate = torch.empty_like(x.v)

    def run_base() -> None:
        baseline(
            x.q, x.k, x.v, x.g, x.beta, x.scale, out_base,
            A_log=x.a_log, dt_bias=x.dt_bias, lower_bound=x.lower_bound,
            initial_state=initial, final_state=final_base,
        )

    def run_candidate() -> None:
        vshard.fwd(
            x.q, x.k, x.v, x.g, x.beta, x.scale, out_candidate,
            A_log=x.a_log, dt_bias=x.dt_bias, lower_bound=x.lower_bound,
            initial_state=candidate_initial, final_state=final_candidate,
        )

    baseline_ms = event_ms(run_base, warmup, iters, repeats)
    candidate_ms = event_ms(run_candidate, warmup, iters, repeats)

    def summarize(prefix: str, xs: Iterable[float]) -> dict[str, float]:
        ys = list(xs)
        return {
            f"{prefix}_mean_ms": statistics.fmean(ys),
            f"{prefix}_median_ms": statistics.median(ys),
            f"{prefix}_min_ms": min(ys),
            f"{prefix}_max_ms": max(ys),
        }

    result = summarize("baseline", baseline_ms) | summarize("vshard", candidate_ms)
    result["speedup_median_x"] = result["baseline_median_ms"] / result["vshard_median_ms"]
    print(
        f"BENCH {mode}: baseline median={result['baseline_median_ms']:.4f} ms, "
        f"vshard median={result['vshard_median_ms']:.4f} ms, "
        f"speedup={result['speedup_median_x']:.4f}x"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True, help="pinned FlashKDA 1ce47ea source root")
    parser.add_argument("--fla-root", type=Path, default=Path(__file__).resolve().parents[1] / "fla_kda_ref", help="assignment-pinned fla_kda_ref directory")
    parser.add_argument("--T", type=int, default=256, help="sequence length; official benchmark uses 8192")
    parser.add_argument("--H", type=int, default=2, help="head count; official benchmark uses 96")
    parser.add_argument("--states", choices=("none", "bf16", "fp32", "all"), default="all")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-torch-ref", action="store_true", help="only allowed for timing re-runs after a recorded exact validation")
    parser.add_argument("--skip-fla-ref", action="store_true", help="only allowed for large timing runs after a recorded small-shape FLA reference validation")
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--json", type=Path, help="write machine-readable timing summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required; do not run this harness on a login node")
    if args.T % 16:
        raise ValueError("T must be divisible by CHUNK=16 for the fixed-length exact validation")
    try:
        import flash_kda
        from flash_kda_C import fwd_vshard  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("install patched challenge FlashKDA before running this harness") from exc

    print(f"device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
    print(f"shape=[B=1,T={args.T},H={args.H},K=128,V=128], challenge=2 CTA/head")
    torch_ref_module = None
    torch_ref = None
    if not args.skip_torch_ref or not args.skip_fla_ref:
        ref_file = args.reference_root / "tests" / "torch_ref.py"
        spec = importlib.util.spec_from_file_location("c1_upstream_torch_ref", ref_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {ref_file}")
        torch_ref_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(torch_ref_module)
        torch_ref = None if args.skip_torch_ref else torch_ref_module.torch_ref
    fla_naive = None if args.skip_fla_ref else _load_fla_naive(args.fla_root)
    x = make_inputs(args.T, args.H, args.seed)
    modes = ("none", "bf16", "fp32") if args.states == "all" else (args.states,)
    for index, mode in enumerate(modes):
        validate_mode(flash_kda.fwd, torch_ref, torch_ref_module, fla_naive, x, mode, args.seed + index * 101)

    output: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "shape": {"B": 1, "T": args.T, "H": args.H, "K": 128, "V": 128},
        "states_validated": list(modes),
        "torch_reference": not args.skip_torch_ref,
        "fla_naive_reference": not args.skip_fla_ref,
    }
    if not args.no_bench:
        output["benchmarks"] = {
            mode: benchmark(flash_kda.fwd, x, mode, args.warmup, args.iters, args.repeats)
            for mode in modes
        }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
