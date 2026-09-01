#!/usr/bin/env python3
"""Exactness and public-wrapper timing gate for the non-production P2S4 path.

The four timed paths reside in one freshly generated extension: upstream
baseline, vshard2-P2S3, vshard4-P2S3, and the separate vshard4-P2S4 symbol.
Every CUDA event measures exactly one public Python wrapper call, including
that wrapper's workspace allocation.  This is intentionally not a dispatcher
benchmark and cannot make a candidate production-eligible by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_inputstages4 import inputstages4
from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2
from assignment02.team.c1_flashkda.challenge_vshard4_prefetch2 import vshard4_prefetch2
from assignment02.team.c1_flashkda.harness import validate_and_bench as common


CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
TIMED_VARIANTS = ("baseline", "vshard2_p2s3", "vshard4_p2s3", "vshard4_p2s4")


def _csv(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or any(item not in allowed for item in result) or len(set(result)) != len(result):
        raise ValueError(f"invalid comma-separated list: {value!r}")
    return result


def _identity() -> dict[str, object]:
    import flash_kda_C

    required = (
        "fwd",
        "fwd_vshard",
        "fwd_vshard_p2",
        "fwd_vshard4",
        "fwd_vshard4_p2",
        "fwd_vshard4_p2s4",
        "get_workspace_size",
    )
    missing = [symbol for symbol in required if not hasattr(flash_kda_C, symbol)]
    if missing:
        raise RuntimeError(f"loaded extension lacks the isolated P2S4 ABI: {missing}")
    path = Path(flash_kda_C.__file__).resolve()
    return {
        "extension": str(path),
        "extension_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "required_symbols": list(required),
    }


def _states(contract: str, heads: int, seed: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if contract == "none":
        return None, None
    dtype = torch.bfloat16 if contract == "bf16_both" else torch.float32
    final = torch.zeros((1, heads, 128, 128), dtype=dtype, device="cuda")
    if contract == "fp32_final_only":
        return None, final
    generator = torch.Generator(device="cuda").manual_seed(seed)
    initial = torch.randn(final.shape, dtype=dtype, device="cuda", generator=generator)
    return initial.contiguous(), final


def _clone(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.clone()


def _invoke(
    fn: Callable[..., None],
    x: common.Inputs,
    initial: torch.Tensor | None,
    final: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        initial_state=initial,
        final_state=final,
    )
    torch.cuda.synchronize()
    return out, final


def _compare_exact(
    label: str,
    actual: tuple[torch.Tensor, torch.Tensor | None],
    expected: tuple[torch.Tensor, torch.Tensor | None],
) -> dict[str, object]:
    common.require_exact(label + "/output", actual[0], expected[0])
    result: dict[str, object] = {
        "output_exact": True,
        "output_max_abs": common.max_abs(actual[0], expected[0]),
    }
    if actual[1] is None or expected[1] is None:
        if actual[1] is not None or expected[1] is not None:
            raise AssertionError(f"{label}: final-state presence mismatch")
        result["final_state_present"] = False
    else:
        common.require_exact(label + "/final_state", actual[1], expected[1])
        result.update(
            {
                "final_state_present": True,
                "final_state_exact": True,
                "final_state_max_abs": common.max_abs(actual[1], expected[1]),
            }
        )
    return result


def _exact(x: common.Inputs, contract: str, seed: int) -> dict[str, object]:
    import flash_kda

    initial, final = _states(contract, x.q.shape[2], seed)
    paths: dict[str, Callable[..., None]] = {
        "baseline": flash_kda.fwd,
        "vshard2_p2s3": prefetch2.fwd,
        "vshard4_p2s3": vshard4_prefetch2.fwd,
        "vshard4_p2s4": inputstages4.fwd,
    }
    outputs = {label: _invoke(fn, x, _clone(initial), _clone(final)) for label, fn in paths.items()}
    baseline = outputs["baseline"]
    return {
        label + "_vs_baseline": _compare_exact(label + "_vs_baseline", outputs[label], baseline)
        for label in TIMED_VARIANTS[1:]
    }


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
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(ordered, 50),
        "p95_ms": _percentile(ordered, 95),
        "p99_ms": _percentile(ordered, 99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _timed_calls(x: common.Inputs, contract: str, seed: int) -> dict[str, Callable[[], None]]:
    import flash_kda

    paths: dict[str, Callable[..., None]] = {
        "baseline": flash_kda.fwd,
        "vshard2_p2s3": prefetch2.fwd,
        "vshard4_p2s3": vshard4_prefetch2.fwd,
        "vshard4_p2s4": inputstages4.fwd,
    }
    calls: dict[str, Callable[[], None]] = {}
    for index, (label, fn) in enumerate(paths.items()):
        initial, final = _states(contract, x.q.shape[2], seed + index * 1009)
        out = torch.empty_like(x.v)

        def call(
            fn: Callable[..., None] = fn,
            initial: torch.Tensor | None = initial,
            final: torch.Tensor | None = final,
            out: torch.Tensor = out,
        ) -> None:
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
                initial_state=initial,
                final_state=final,
            )

        calls[label] = call
    return calls


def _benchmark(
    x: common.Inputs, contract: str, seed: int, warmup: int, samples: int
) -> dict[str, object]:
    calls = _timed_calls(x, contract, seed)
    labels = tuple(calls)
    for round_index in range(warmup):
        order = labels[round_index % len(labels) :] + labels[: round_index % len(labels)]
        for label in order:
            calls[label]()
    torch.cuda.synchronize()
    raw: dict[str, list[float]] = {label: [] for label in labels}
    for round_index in range(samples):
        order = labels[round_index % len(labels) :] + labels[: round_index % len(labels)]
        for label in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            calls[label]()
            end.record()
            torch.cuda.synchronize()
            raw[label].append(float(start.elapsed_time(end)))
    paths = {label: _summary(values) for label, values in raw.items()}
    reference = paths["vshard4_p2s3"]
    candidate = paths["vshard4_p2s4"]
    return {
        "paths": paths,
        "raw_samples_ms": raw,
        "candidate_speedup_vs_vshard4_p2s3_x": {
            percentile: float(reference[percentile]) / float(candidate[percentile])
            for percentile in ("p50_ms", "p95_ms", "p99_ms")
        },
        "event_contract": "four-path cyclic rotation; one complete public-wrapper call per CUDA event; workspace allocation included",
        "sample_contract": f"{samples} raw CUDA-event samples retained per path",
    }


def _run_shape(
    tokens: int,
    heads: int,
    contracts: tuple[str, ...],
    seed: int,
    *,
    benchmark: bool,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    x = common.make_inputs(tokens, heads, seed)
    result: dict[str, object] = {
        "shape": {"B": 1, "T": tokens, "H": heads, "K": 128, "V": 128},
        "exact": {
            contract: _exact(x, contract, seed + index * 100_003)
            for index, contract in enumerate(contracts)
        },
        "exact_gate_pass": True,
    }
    if benchmark:
        result["benchmarks"] = {
            contract: _benchmark(x, contract, seed + index * 100_003, warmup, samples)
            for index, contract in enumerate(contracts)
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--H", type=int, default=12)
    parser.add_argument("--contracts", default=",".join(CONTRACTS))
    parser.add_argument("--small-heads", default="", help="exact-only matrix, e.g. 1,2,4")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.T <= 0 or args.H <= 0 or args.T % 16:
        raise RuntimeError("requires CUDA and positive T/H with T divisible by 16")
    if args.warmup < 0 or args.samples <= 0:
        raise ValueError("warmup must be nonnegative and samples must be positive")
    contracts = _csv(args.contracts, CONTRACTS)
    small_heads = tuple(int(value) for value in _csv(args.small_heads, tuple(str(i) for i in range(1, 129)))) if args.small_heads else ()
    if any(head <= 0 for head in small_heads) or (small_heads and not args.no_bench):
        raise ValueError("--small-heads must be positive and exact-only (--no-bench)")
    identity = _identity()
    result: dict[str, object] = {
        "candidate": "fwd_vshard4_p2s4",
        "candidate_status": "non-production; dispatch registration is forbidden",
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "build_target": os.environ.get("C1_BUILD_TARGET", "unspecified"),
        "allocation": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"),
            "hostname": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unspecified"),
        },
        "extension": identity,
        "contracts": list(contracts),
        "seed": args.seed,
    }
    print(
        f"device={result['device']} capability={result['capability']} "
        f"build_target={result['build_target']}"
    )
    print(f"extension_sha256={identity['extension_sha256']}")
    if small_heads:
        result["small_matrix"] = {
            f"H{heads}": _run_shape(
                256,
                heads,
                contracts,
                args.seed + index * 1_000_003,
                benchmark=False,
                warmup=args.warmup,
                samples=args.samples,
            )
            for index, heads in enumerate(small_heads)
        }
        result["small_exact_gate_pass"] = True
    else:
        result["h12"] = _run_shape(
            args.T,
            args.H,
            contracts,
            args.seed,
            benchmark=not args.no_bench,
            warmup=args.warmup,
            samples=args.samples,
        )
        result["h12_exact_gate_pass"] = True
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
