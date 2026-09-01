#!/usr/bin/env python3
"""Exact gate and four-path cyclic benchmark for the V=32 P2S3 candidate.

All wrappers are imported from the same generated extension.  The timed paths
are baseline, current vshard2 P2S3, vshard4 P1, and vshard4 P2S3.  Current
vshard2 P1 remains loaded only as the exactness predecessor of vshard2 P2S3.
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

from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2
from assignment02.team.c1_flashkda.challenge_vshard import vshard
from assignment02.team.c1_flashkda.challenge_vshard4 import vshard4
from assignment02.team.c1_flashkda.challenge_vshard4_prefetch2 import vshard4_prefetch2
from assignment02.team.c1_flashkda.harness import validate_and_bench as common


TORCH_REF_OUTPUT_RTOL = 2e-2
TORCH_REF_OUTPUT_ATOL = 2e-2
TORCH_REF_STATE_RTOL = 5e-2
TORCH_REF_STATE_ATOL = 5e-2


def _csv(value: str, allowed: set[str] | None = None) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or (allowed is not None and any(item not in allowed for item in result)):
        raise ValueError(f"invalid comma-separated list: {value!r}")
    return result


def _identity() -> dict[str, object]:
    import flash_kda_C

    required = ("fwd", "fwd_vshard", "fwd_vshard_p2", "fwd_vshard4", "fwd_vshard4_p2", "get_workspace_size")
    missing = [symbol for symbol in required if not hasattr(flash_kda_C, symbol)]
    if missing:
        raise RuntimeError(f"loaded extension lacks the one-shot comparison ABI: {missing}")
    path = Path(flash_kda_C.__file__).resolve()
    return {
        "extension": str(path),
        "extension_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "required_symbols": list(required),
    }


def _run(
    fn: Callable[..., None], x: common.Inputs, initial: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    state_in = None if initial is None else initial.clone()
    state_out = None if initial is None else torch.zeros_like(initial)
    result = common.invoke(fn, x, state_in, state_out)
    torch.cuda.synchronize()
    return result


def _close_to_ref(
    label: str,
    output: torch.Tensor,
    state: torch.Tensor | None,
    reference_output: torch.Tensor,
    reference_state: torch.Tensor | None,
) -> dict[str, object]:
    common.require_close(label + "/output", output, reference_output, rtol=TORCH_REF_OUTPUT_RTOL, atol=TORCH_REF_OUTPUT_ATOL)
    result: dict[str, object] = {
        "output_close": True,
        "output_bitwise_exact": torch.equal(output, reference_output),
        "output_max_abs": common.max_abs(output, reference_output),
        "output_tolerance": {"rtol": TORCH_REF_OUTPUT_RTOL, "atol": TORCH_REF_OUTPUT_ATOL},
    }
    if state is not None and reference_state is not None:
        common.require_close(label + "/final_state", state, reference_state, rtol=TORCH_REF_STATE_RTOL, atol=TORCH_REF_STATE_ATOL)
        result.update({
            "final_state_close": True,
            "final_state_bitwise_exact": torch.equal(state, reference_state),
            "final_state_max_abs": common.max_abs(state, reference_state),
            "final_state_tolerance": {"rtol": TORCH_REF_STATE_RTOL, "atol": TORCH_REF_STATE_ATOL},
        })
    return result


def _exact(torch_ref: Callable[..., None], x: common.Inputs, mode: str, seed: int) -> dict[str, object]:
    import flash_kda

    initial, _ = common.state_tensors(mode, x.q.shape[2], seed)
    functions: dict[str, Callable[..., None]] = {
        "baseline": flash_kda.fwd,
        "vshard2_p1": vshard.fwd,
        "vshard2_p2": prefetch2.fwd,
        "vshard4_p1": vshard4.fwd,
        "vshard4_p2": vshard4_prefetch2.fwd,
        "torch_ref": torch_ref,
    }
    outputs = {label: _run(fn, x, initial) for label, fn in functions.items()}
    base_out, base_state = outputs["baseline"]
    exact: dict[str, object] = {}
    for label in ("vshard2_p1", "vshard2_p2", "vshard4_p1", "vshard4_p2"):
        out, state = outputs[label]
        common.require_exact(f"{mode}/baseline_vs_{label}/output", out, base_out)
        item: dict[str, object] = {"output_exact": True, "output_max_abs": common.max_abs(out, base_out)}
        if state is not None and base_state is not None:
            common.require_exact(f"{mode}/baseline_vs_{label}/final_state", state, base_state)
            item.update({"final_state_exact": True, "final_state_max_abs": common.max_abs(state, base_state)})
        exact[f"baseline_vs_{label}"] = item
    p1_out, p1_state = outputs["vshard2_p1"]
    p2_out, p2_state = outputs["vshard2_p2"]
    common.require_exact(f"{mode}/vshard2_p1_vs_p2/output", p2_out, p1_out)
    if p1_state is not None and p2_state is not None:
        common.require_exact(f"{mode}/vshard2_p1_vs_p2/final_state", p2_state, p1_state)
    ref_out, ref_state = outputs["torch_ref"]
    tolerances = {
        label: _close_to_ref(f"{mode}/{label}_vs_torch_ref", *outputs[label], ref_out, ref_state)
        for label in ("baseline", "vshard2_p1", "vshard2_p2", "vshard4_p1", "vshard4_p2")
    }
    return {"exact": exact, "torch_ref_tolerance": tolerances}


def _percentile(sorted_values: list[float], percent: float) -> float:
    if not sorted_values:
        raise ValueError("cannot summarize zero samples")
    position = (len(sorted_values) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "samples": len(values), "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(ordered, 50), "p95_ms": _percentile(ordered, 95),
        "p99_ms": _percentile(ordered, 99), "min_ms": ordered[0], "max_ms": ordered[-1],
    }


def _make_timed_calls(x: common.Inputs, mode: str) -> dict[str, Callable[[], None]]:
    import flash_kda

    initial, _ = common.state_tensors(mode, x.q.shape[2], 17)
    variants: dict[str, Callable[..., None]] = {
        "baseline": flash_kda.fwd,
        "vshard2_p2": prefetch2.fwd,
        "vshard4_p1": vshard4.fwd,
        "vshard4_p2": vshard4_prefetch2.fwd,
    }
    calls: dict[str, Callable[[], None]] = {}
    for label, fn in variants.items():
        state_in = None if initial is None else initial.clone()
        state_out = None if initial is None else torch.zeros_like(initial)
        out = torch.empty_like(x.v)

        def call(fn: Callable[..., None] = fn, out: torch.Tensor = out, state_in: torch.Tensor | None = state_in,
                 state_out: torch.Tensor | None = state_out) -> None:
            fn(x.q, x.k, x.v, x.g, x.beta, x.scale, out, A_log=x.a_log, dt_bias=x.dt_bias,
               lower_bound=x.lower_bound, initial_state=state_in, final_state=state_out)

        calls[label] = call
    return calls


def _benchmark_four_paths(x: common.Inputs, mode: str, warmup: int, samples: int) -> dict[str, object]:
    calls = _make_timed_calls(x, mode)
    labels = tuple(calls)
    for round_index in range(warmup):
        order = labels[round_index % len(labels):] + labels[:round_index % len(labels)]
        for label in order:
            calls[label]()
    torch.cuda.synchronize()
    raw: dict[str, list[float]] = {label: [] for label in labels}
    # A complete rotated cycle produces one raw CUDA-event observation for each
    # path; samples=1000 therefore means 1000 retained values *per path*.
    for round_index in range(samples):
        order = labels[round_index % len(labels):] + labels[:round_index % len(labels)]
        for label in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); calls[label](); end.record()
            torch.cuda.synchronize()
            raw[label].append(float(start.elapsed_time(end)))
    summary = {label: _summary(values) for label, values in raw.items()}
    baseline_p50 = float(summary["baseline"]["p50_ms"])
    return {
        "paths": summary,
        "raw_samples_ms": raw,
        "shape_specific_speedup_vs_baseline_p50_x": {
            label: baseline_p50 / float(summary[label]["p50_ms"])
            for label in labels if label != "baseline"
        },
        "event_contract": "four-path cyclic rotation; one complete public-wrapper call per CUDA event; workspace allocation included",
        "sample_contract": f"{samples} raw CUDA-event samples retained for each of {', '.join(labels)}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--T", type=int, default=256)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--states", default="all")
    parser.add_argument("--small-heads", default="", help="exact-only matrix, e.g. 1,2,4")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.T <= 0 or args.H <= 0 or args.T % 16:
        raise RuntimeError("requires CUDA and positive T/H with T divisible by 16")
    if args.warmup < 0 or args.samples <= 0:
        raise ValueError("warmup must be nonnegative and samples must be positive")
    states = ("none", "bf16", "fp32") if args.states == "all" else _csv(args.states, {"none", "bf16", "fp32"})
    small_heads = tuple(int(item) for item in _csv(args.small_heads)) if args.small_heads else ()
    if any(head <= 0 for head in small_heads) or (small_heads and not args.no_bench):
        raise ValueError("--small-heads must be positive and exact-only (--no-bench)")
    torch_ref = common._load_torch_ref(args.reference_root)
    identity = _identity()
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
        "build_target": os.environ.get("C1_BUILD_TARGET", "unspecified"),
        "candidate": "fwd_vshard4_p2", "comparison_paths": ["baseline", "vshard2_p2", "vshard4_p1", "vshard4_p2"],
        "extension": identity, "states": list(states), "seed": args.seed,
    }
    print(f"device={result['device']} capability={result['capability']} build_target={result['build_target']}")
    print(f"extension_sha256={identity['extension_sha256']}")
    if small_heads:
        result["shape_matrix"] = {"B": 1, "T": args.T, "H": list(small_heads), "K": 128, "V": 128}
        result["correctness_matrix"] = {
            f"H{heads}": {mode: _exact(torch_ref, common.make_inputs(args.T, heads, args.seed + index * 1009), mode,
                                         args.seed + index * 1009 + state_index * 101)
                            for state_index, mode in enumerate(states)}
            for index, heads in enumerate(small_heads)
        }
        result["exact_gate_pass"] = True
    else:
        x = common.make_inputs(args.T, args.H, args.seed)
        result["shape"] = {"B": 1, "T": args.T, "H": args.H, "K": 128, "V": 128}
        result["correctness"] = {mode: _exact(torch_ref, x, mode, args.seed + index * 101)
                                 for index, mode in enumerate(states)}
        result["exact_gate_pass"] = True
        if not args.no_bench:
            result["benchmark"] = _benchmark_four_paths(x, states[0], args.warmup, args.samples)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
