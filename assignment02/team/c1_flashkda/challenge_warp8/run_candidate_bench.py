#!/usr/bin/env python3
"""Exact gate and equal-contract CUDA-event benchmark for C1 K2 candidates.

Run this script in a process whose ``PYTHONPATH`` starts with the isolated
FlashKDA clone to test.  ``--variant warp8`` expects ``flash_kda_C.fwd_warp8``;
``--variant vshard`` expects ``flash_kda_C.fwd_vshard``.  Both are measured
against that clone's untouched ``flash_kda.fwd`` in the *same process*, so the
baseline/candidate use the same input tensors, output/state allocation policy,
warmup, event timing boundary, and CUDA device.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F


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


def make_inputs(tokens: int, heads: int, seed: int) -> Inputs:
    torch.manual_seed(seed)
    d = 128
    device = torch.device("cuda")
    q = F.normalize(torch.randn(1, tokens, heads, d, device=device), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(1, tokens, heads, d, device=device), p=2, dim=-1).to(torch.bfloat16)
    return Inputs(
        q=q.contiguous(),
        k=k.contiguous(),
        v=torch.randn(1, tokens, heads, d, dtype=torch.bfloat16, device=device),
        g=torch.randn(1, tokens, heads, d, dtype=torch.bfloat16, device=device),
        beta=torch.randn(1, tokens, heads, dtype=torch.bfloat16, device=device),
        a_log=torch.rand(heads, dtype=torch.float32, device=device),
        dt_bias=torch.rand(heads, d, dtype=torch.float32, device=device),
        scale=1.0 / math.sqrt(d),
        lower_bound=-5.0,
    )


def states(mode: str, heads: int, seed: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if mode == "none":
        return None, None
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    initial = torch.randn(1, heads, 128, 128, dtype=dtype, device="cuda", generator=generator)
    return initial.contiguous(), torch.zeros_like(initial)


def call(fn: Callable[..., None], x: Inputs, out: torch.Tensor, initial: Optional[torch.Tensor], final: Optional[torch.Tensor]) -> None:
    fn(
        x.q, x.k, x.v, x.g, x.beta, x.scale, out,
        A_log=x.a_log, dt_bias=x.dt_bias, lower_bound=x.lower_bound,
        initial_state=initial, final_state=final,
    )


def event_samples(fn: Callable[[], None], warmup: int, iters: int, repeats: int) -> list[float]:
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


def summary(samples: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples": len(samples),
    }


def candidate_function(variant: str) -> Callable[..., None]:
    # Use the public challenge wrappers rather than raw pybind functions:
    # both wrappers allocate the caller-equivalent workspace and preserve the
    # same ABI as `flash_kda.fwd`.  Calling fwd_warp8 directly would omit its
    # mandatory workspace positional argument and make the timing unfair.
    if variant == "warp8":
        from warp8 import fwd

        return fwd
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if variant == "vshard":
        from assignment02.team.c1_flashkda.challenge_vshard.vshard import fwd
    else:
        from assignment02.team.c1_flashkda.challenge_vshard4.vshard4 import fwd

    return fwd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("warp8", "vshard", "vshard4"), required=True)
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--H", type=int, required=True)
    parser.add_argument("--state", choices=("none", "bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-bench", action="store_true", help="run only the exact correctness gate; emit no timing metrics")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--current-best-ms", type=float, help="optional frozen current-best median; records a 10%% target")
    parser.add_argument("--enforce-target", action="store_true", help="exit non-zero when --current-best-ms/1.1 is not beaten")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required; do not run this on a login node")
    if args.T <= 0 or args.H <= 0 or args.T % 16:
        raise ValueError("T and H must be positive and T must be divisible by 16")
    import flash_kda

    candidate = candidate_function(args.variant)
    x = make_inputs(args.T, args.H, args.seed)
    initial, final_base = states(args.state, args.H, args.seed + 11)
    base_initial = None if initial is None else initial.clone()
    candidate_initial = None if initial is None else initial.clone()
    _, final_candidate = states(args.state, args.H, args.seed + 12)
    out_base = torch.zeros_like(x.v)
    out_candidate = torch.zeros_like(x.v)

    call(flash_kda.fwd, x, out_base, base_initial, final_base)
    call(candidate, x, out_candidate, candidate_initial, final_candidate)
    torch.cuda.synchronize()
    output_exact = torch.equal(out_base, out_candidate)
    final_exact = final_base is None or (final_candidate is not None and torch.equal(final_base, final_candidate))
    max_output_abs = float((out_base.float() - out_candidate.float()).abs().max().item())
    max_final_abs = 0.0 if final_base is None else float((final_base.float() - final_candidate.float()).abs().max().item())
    if not (output_exact and final_exact):
        raise AssertionError(
            f"exact gate failed: output={output_exact} max_output_abs={max_output_abs}; "
            f"final={final_exact} max_final_abs={max_final_abs}"
        )
    print(f"PASS exact baseline_vs_{args.variant}: output and final_state")

    # Preallocate out/state exactly as the historical C1 harness does.  The
    # function call (including its same-ABI workspace allocation) is enclosed
    # by one CUDA event pair for both functions.
    if args.no_bench:
        result: dict[str, object] = {
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "variant": args.variant,
            "shape": {"B": 1, "T": args.T, "H": args.H, "K": 128, "V": 128},
            "state": args.state,
            "seed": args.seed,
            "correctness": {
                "baseline_vs_candidate_exact": True,
                "output_max_abs": max_output_abs,
                "final_state_max_abs": max_final_abs,
            },
            "timing": "not run (--no-bench)",
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
        return

    bench_initial, bench_final_base = states(args.state, args.H, args.seed + 31)
    bench_candidate_initial = None if bench_initial is None else bench_initial.clone()
    _, bench_final_candidate = states(args.state, args.H, args.seed + 32)
    bench_out_base = torch.empty_like(x.v)
    bench_out_candidate = torch.empty_like(x.v)

    def run_baseline() -> None:
        call(flash_kda.fwd, x, bench_out_base, bench_initial, bench_final_base)

    def run_candidate() -> None:
        call(candidate, x, bench_out_candidate, bench_candidate_initial, bench_final_candidate)

    base = summary(event_samples(run_baseline, args.warmup, args.iters, args.repeats))
    cand = summary(event_samples(run_candidate, args.warmup, args.iters, args.repeats))
    speedup = base["median_ms"] / cand["median_ms"]
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "variant": args.variant,
        "shape": {"B": 1, "T": args.T, "H": args.H, "K": 128, "V": 128},
        "state": args.state,
        "seed": args.seed,
        "event_contract": "one CUDA-event pair per full Python forward call; same wrapper ABI/workspace policy",
        "correctness": {
            "baseline_vs_candidate_exact": True,
            "output_max_abs": max_output_abs,
            "final_state_max_abs": max_final_abs,
        },
        "baseline": base,
        "candidate": cand,
        "baseline_over_candidate_median_x": speedup,
    }
    target_met = None
    if args.current_best_ms is not None:
        target_ms = args.current_best_ms / 1.1
        target_met = cand["median_ms"] <= target_ms
        result["frozen_current_best_ms"] = args.current_best_ms
        result["strict_10pct_target_ms"] = target_ms
        result["strict_10pct_target_met"] = target_met
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"BENCH {args.variant} H={args.H}: baseline={base['median_ms']:.6f} ms, "
        f"candidate={cand['median_ms']:.6f} ms, speedup={speedup:.6f}x"
    )
    print(f"wrote {args.json}")
    if args.enforce_target and target_met is not True:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
