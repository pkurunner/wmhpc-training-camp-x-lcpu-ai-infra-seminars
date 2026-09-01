#!/usr/bin/env python3
"""Clean correctness and full-call CUDA-event runner for C1 V=32 vshard4.

The shared two-way harness cannot be used for this candidate because it imports
``fwd_vshard``.  This runner instead binds the baseline to ``flash_kda.fwd``
and the candidate to this directory's ``vshard4.fwd``; both events bracket one
full public call, including the respective wrapper's workspace allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_vshard4 import vshard4
from assignment02.team.c1_flashkda.harness import validate_and_bench as common


def _csv(value: str, allowed: set[str] | None = None) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or (allowed is not None and any(item not in allowed for item in result)):
        raise ValueError(f"invalid list {value!r}")
    return result


def _identity() -> dict[str, object]:
    import flash_kda_C

    if not hasattr(flash_kda_C, "fwd_vshard4") or not hasattr(flash_kda_C, "get_workspace_size"):
        raise RuntimeError("loaded extension lacks required fwd_vshard4/get_workspace_size ABI")
    path = Path(flash_kda_C.__file__).resolve()
    return {"extension": str(path), "extension_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _exact(baseline: Callable[..., None], x: common.Inputs, mode: str, seed: int) -> dict[str, object]:
    initial, final_base = common.state_tensors(mode, x.q.shape[2], seed)
    out_base, final_base = common.invoke(baseline, x, None if initial is None else initial.clone(), final_base)
    torch.cuda.synchronize()
    candidate_initial = None if initial is None else initial.clone()
    _, final_candidate = common.state_tensors(mode, x.q.shape[2], seed + 1)
    out_candidate, final_candidate = common.invoke(vshard4.fwd, x, candidate_initial, final_candidate)
    torch.cuda.synchronize()
    common.require_exact(f"{mode}/baseline_vs_vshard4/output", out_candidate, out_base)
    result: dict[str, object] = {
        "baseline_vs_candidate_exact": True,
        "output_max_abs": common.max_abs(out_candidate, out_base),
    }
    if final_base is not None and final_candidate is not None:
        common.require_exact(f"{mode}/baseline_vs_vshard4/final_state", final_candidate, final_base)
        result["final_state_max_abs"] = common.max_abs(final_candidate, final_base)
    return result


def _summary(values: list[float]) -> dict[str, float | int]:
    return {"mean_ms": statistics.fmean(values), "median_ms": statistics.median(values),
            "min_ms": min(values), "max_ms": max(values), "samples": len(values)}


def _event_ab_ba(
    run_base: Callable[[], None], run_candidate: Callable[[], None], warmup: int, iters: int, repeats: int
) -> tuple[list[float], list[float]]:
    """Collect equal per-call event samples with AB/BA order alternation.

    Each event surrounds exactly one public call.  Alternating order for every
    sample prevents a one-sided warm-cache/thermal/order effect from becoming
    a candidate advantage.  ``repeats * iters`` samples are retained per path.
    """
    for index in range(warmup):
        (run_base if index % 2 == 0 else run_candidate)()
        (run_candidate if index % 2 == 0 else run_base)()
    torch.cuda.synchronize()
    base_samples: list[float] = []
    candidate_samples: list[float] = []
    for index in range(repeats * iters):
        base_first = (index % 2) == 0
        first, second = (run_base, run_candidate) if base_first else (run_candidate, run_base)
        first_start, first_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        second_start, second_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        first_start.record(); first(); first_end.record()
        second_start.record(); second(); second_end.record()
        # Synchronize only after both calls are enqueued, so the event boundary
        # remains a single call while the host has no per-sample sync bias.
        torch.cuda.synchronize()
        first_ms, second_ms = float(first_start.elapsed_time(first_end)), float(second_start.elapsed_time(second_end))
        if base_first:
            base_samples.append(first_ms); candidate_samples.append(second_ms)
        else:
            candidate_samples.append(first_ms); base_samples.append(second_ms)
    return base_samples, candidate_samples


def _bench(baseline: Callable[..., None], x: common.Inputs, mode: str, warmup: int, iters: int, repeats: int) -> dict[str, object]:
    initial, final_base = common.state_tensors(mode, x.q.shape[2], 17)
    candidate_initial = None if initial is None else initial.clone()
    _, final_candidate = common.state_tensors(mode, x.q.shape[2], 19)
    out_base, out_candidate = torch.empty_like(x.v), torch.empty_like(x.v)

    def run_base() -> None:
        baseline(x.q, x.k, x.v, x.g, x.beta, x.scale, out_base, A_log=x.a_log, dt_bias=x.dt_bias,
                 lower_bound=x.lower_bound, initial_state=initial, final_state=final_base)

    def run_candidate() -> None:
        vshard4.fwd(x.q, x.k, x.v, x.g, x.beta, x.scale, out_candidate, A_log=x.a_log, dt_bias=x.dt_bias,
                    lower_bound=x.lower_bound, initial_state=candidate_initial, final_state=final_candidate)

    base_values, candidate_values = _event_ab_ba(run_base, run_candidate, warmup, iters, repeats)
    base, candidate = _summary(base_values), _summary(candidate_values)
    return {"baseline": base, "vshard4": candidate,
            "speedup_median_x": float(base["median_ms"]) / float(candidate["median_ms"]),
            "event_contract": "AB/BA alternating one-public-fwd-call/event; includes wrapper workspace allocation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=256)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--states", default="all")
    parser.add_argument("--small-heads", default="", help="exact-only matrix, e.g. 1,2,4")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.T <= 0 or args.H <= 0 or args.T % 16:
        raise RuntimeError("requires CUDA and positive T/H with T divisible by 16")
    states = ("none", "bf16", "fp32") if args.states == "all" else _csv(args.states, {"none", "bf16", "fp32"})
    small_heads = tuple(int(item) for item in _csv(args.small_heads)) if args.small_heads else ()
    if any(head <= 0 for head in small_heads) or (small_heads and not args.no_bench):
        raise ValueError("--small-heads must be positive and exact-only (--no-bench)")

    import flash_kda
    identity = _identity()
    print(f"device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
    print(f"candidate=fwd_vshard4 extension_sha256={identity['extension_sha256']}")
    result: dict[str, object] = {"device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()), "candidate": "fwd_vshard4",
        "extension": identity, "states": list(states), "seed": args.seed,
        "build_target": os.environ.get("C1_BUILD_TARGET", "unspecified")}
    if small_heads:
        result["shape_matrix"] = {"B": 1, "T": args.T, "H": list(small_heads), "K": 128, "V": 128}
        matrix: dict[str, object] = {}
        for h_index, heads in enumerate(small_heads):
            x = common.make_inputs(args.T, heads, args.seed + h_index * 1009)
            matrix[f"H{heads}"] = {mode: _exact(flash_kda.fwd, x, mode, args.seed + h_index * 1009 + i * 101)
                                  for i, mode in enumerate(states)}
        result["correctness_matrix"] = matrix
    else:
        x = common.make_inputs(args.T, args.H, args.seed)
        result["shape"] = {"B": 1, "T": args.T, "H": args.H, "K": 128, "V": 128}
        result["correctness"] = {mode: _exact(flash_kda.fwd, x, mode, args.seed + i * 101)
                                 for i, mode in enumerate(states)}
        if not args.no_bench:
            result["benchmarks"] = {mode: _bench(flash_kda.fwd, x, mode, args.warmup, args.iters, args.repeats)
                                    for mode in states}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
