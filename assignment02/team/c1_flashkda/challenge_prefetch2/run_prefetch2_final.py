#!/usr/bin/env python3
"""Exact P1/P2 gate and same-allocation full-call AB/BA benchmark.

Every timed CUDA event surrounds exactly one public Python ``fwd`` call,
including that wrapper's workspace allocation.  Timing is unreachable until
P1/P2 agree bit-for-bit and both pass the repository's independent-reference
numerical tolerances.
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

from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2  # noqa: E402
from assignment02.team.c1_flashkda.challenge_vshard import vshard  # noqa: E402
from assignment02.team.c1_flashkda.harness import validate_and_bench as common  # noqa: E402


FROZEN_B300_P1_MS = 0.799616
STRICT_B300_TARGET_MS = FROZEN_B300_P1_MS / 1.10
TORCH_REF_OUTPUT_RTOL = 2e-2
TORCH_REF_OUTPUT_ATOL = 2e-2
TORCH_REF_STATE_RTOL = 5e-2
TORCH_REF_STATE_ATOL = 5e-2


def _csv(value: str, allowed: set[str] | None = None) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or (allowed is not None and any(item not in allowed for item in result)):
        raise ValueError(f"invalid list {value!r}")
    return result


def _identity() -> dict[str, object]:
    import flash_kda_C

    required = ("fwd", "fwd_vshard", "fwd_vshard_p2", "get_workspace_size")
    missing = [name for name in required if not hasattr(flash_kda_C, name)]
    if missing:
        raise RuntimeError(f"loaded extension lacks required ABI: {missing}")
    path = Path(flash_kda_C.__file__).resolve()
    return {
        "extension": str(path),
        "extension_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "required_symbols": list(required),
    }


def _exact(
    baseline: Callable[..., None],
    torch_ref: Callable[..., None],
    x: common.Inputs,
    mode: str,
    seed: int,
) -> dict[str, object]:
    initial, _ = common.state_tensors(mode, x.q.shape[2], seed)

    def run(fn: Callable[..., None]) -> tuple[torch.Tensor, torch.Tensor | None]:
        state_in = None if initial is None else initial.clone()
        state_out = None if initial is None else torch.zeros_like(initial)
        result = common.invoke(fn, x, state_in, state_out)
        torch.cuda.synchronize()
        return result

    out_base, final_base = run(baseline)
    out_p1, final_p1 = run(vshard.fwd)
    out_p2, final_p2 = run(prefetch2.fwd)
    out_ref, final_ref = run(torch_ref)

    exact_comparisons = {
        "p1_vs_baseline": (out_p1, out_base, final_p1, final_base),
        "p2_vs_p1": (out_p2, out_p1, final_p2, final_p1),
    }
    result: dict[str, object] = {}
    for label, (actual, expected, actual_state, expected_state) in exact_comparisons.items():
        common.require_exact(f"{mode}/{label}/output", actual, expected)
        item: dict[str, object] = {
            "output_exact": True,
            "output_max_abs": common.max_abs(actual, expected),
        }
        if actual_state is not None and expected_state is not None:
            common.require_exact(f"{mode}/{label}/final_state", actual_state, expected_state)
            item["final_state_exact"] = True
            item["final_state_max_abs"] = common.max_abs(actual_state, expected_state)
        result[label] = item
    for label, actual, actual_state in (
        ("p1_vs_torch_ref", out_p1, final_p1),
        ("p2_vs_torch_ref", out_p2, final_p2),
    ):
        common.require_close(
            f"{mode}/{label}/output", actual, out_ref,
            rtol=TORCH_REF_OUTPUT_RTOL, atol=TORCH_REF_OUTPUT_ATOL,
        )
        item = {
            "output_close": True,
            "output_bitwise_exact": torch.equal(actual, out_ref),
            "output_max_abs": common.max_abs(actual, out_ref),
            "output_tolerance": {"rtol": TORCH_REF_OUTPUT_RTOL, "atol": TORCH_REF_OUTPUT_ATOL},
        }
        if actual_state is not None and final_ref is not None:
            common.require_close(
                f"{mode}/{label}/final_state", actual_state, final_ref,
                rtol=TORCH_REF_STATE_RTOL, atol=TORCH_REF_STATE_ATOL,
            )
            item["final_state_close"] = True
            item["final_state_bitwise_exact"] = torch.equal(actual_state, final_ref)
            item["final_state_max_abs"] = common.max_abs(actual_state, final_ref)
            item["final_state_tolerance"] = {"rtol": TORCH_REF_STATE_RTOL, "atol": TORCH_REF_STATE_ATOL}
        result[label] = item
    return result


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples": len(values),
    }


def _event_ab_ba(
    run_p1: Callable[[], None],
    run_p2: Callable[[], None],
    warmup: int,
    iters: int,
    repeats: int,
) -> tuple[list[float], list[float]]:
    for index in range(warmup):
        first, second = (run_p1, run_p2) if index % 2 == 0 else (run_p2, run_p1)
        first(); second()
    torch.cuda.synchronize()
    p1_samples: list[float] = []
    p2_samples: list[float] = []
    for index in range(iters * repeats):
        p1_first = index % 2 == 0
        first, second = (run_p1, run_p2) if p1_first else (run_p2, run_p1)
        s1, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s2, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s1.record(); first(); e1.record()
        s2.record(); second(); e2.record()
        torch.cuda.synchronize()
        first_ms, second_ms = float(s1.elapsed_time(e1)), float(s2.elapsed_time(e2))
        if p1_first:
            p1_samples.append(first_ms); p2_samples.append(second_ms)
        else:
            p2_samples.append(first_ms); p1_samples.append(second_ms)
    return p1_samples, p2_samples


def _bench(x: common.Inputs, mode: str, warmup: int, iters: int, repeats: int) -> dict[str, object]:
    initial, final_p1 = common.state_tensors(mode, x.q.shape[2], 17)
    p2_initial = None if initial is None else initial.clone()
    _, final_p2 = common.state_tensors(mode, x.q.shape[2], 19)
    out_p1, out_p2 = torch.empty_like(x.v), torch.empty_like(x.v)

    def run_p1() -> None:
        vshard.fwd(
            x.q, x.k, x.v, x.g, x.beta, x.scale, out_p1,
            A_log=x.a_log, dt_bias=x.dt_bias, lower_bound=x.lower_bound,
            initial_state=initial, final_state=final_p1,
        )

    def run_p2() -> None:
        prefetch2.fwd(
            x.q, x.k, x.v, x.g, x.beta, x.scale, out_p2,
            A_log=x.a_log, dt_bias=x.dt_bias, lower_bound=x.lower_bound,
            initial_state=p2_initial, final_state=final_p2,
        )

    p1_values, p2_values = _event_ab_ba(run_p1, run_p2, warmup, iters, repeats)
    p1, p2 = _summary(p1_values), _summary(p2_values)
    p1_median, p2_median = float(p1["median_ms"]), float(p2["median_ms"])
    return {
        "p1_current": p1,
        "p2": p2,
        "p1_over_p2_speedup_median_x": p1_median / p2_median,
        "p2_beats_same_allocation_p1": p2_median < p1_median,
        "frozen_b300_p1_ms": FROZEN_B300_P1_MS,
        "strict_b300_target_ms": STRICT_B300_TARGET_MS,
        "frozen_b300_best_over_p2_x": FROZEN_B300_P1_MS / p2_median,
        "strict_target_numeric_pass": p2_median <= STRICT_B300_TARGET_MS,
        "strict_target_claim_valid_only_on_b300": True,
        "event_contract": "AB/BA alternating; one full public wrapper call per CUDA event; workspace allocation included",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
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

    torch_ref = common._load_torch_ref(args.reference_root)
    identity = _identity()
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "build_target": os.environ.get("C1_BUILD_TARGET", "unspecified"),
        "candidate": "fwd_vshard_p2",
        "comparison": "fwd_vshard (P1 current) vs fwd_vshard_p2 (P2)",
        "extension": identity,
        "states": list(states),
        "seed": args.seed,
    }
    print(f"device={result['device']} capability={result['capability']} build_target={result['build_target']}")
    print(f"extension_sha256={identity['extension_sha256']}")
    if small_heads:
        result["shape_matrix"] = {"B": 1, "T": args.T, "H": list(small_heads), "K": 128, "V": 128}
        matrix: dict[str, object] = {}
        for h_index, heads in enumerate(small_heads):
            x = common.make_inputs(args.T, heads, args.seed + h_index * 1009)
            matrix[f"H{heads}"] = {
                mode: _exact(flash_kda.fwd, torch_ref, x, mode, args.seed + h_index * 1009 + i * 101)
                for i, mode in enumerate(states)
            }
        result["correctness_matrix"] = matrix
        result["exact_gate_pass"] = True
    else:
        x = common.make_inputs(args.T, args.H, args.seed)
        result["shape"] = {"B": 1, "T": args.T, "H": args.H, "K": 128, "V": 128}
        result["correctness"] = {
            mode: _exact(flash_kda.fwd, torch_ref, x, mode, args.seed + i * 101)
            for i, mode in enumerate(states)
        }
        result["exact_gate_pass"] = True
        if not args.no_bench:
            result["benchmark"] = _bench(x, states[0], args.warmup, args.iters, args.repeats)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
