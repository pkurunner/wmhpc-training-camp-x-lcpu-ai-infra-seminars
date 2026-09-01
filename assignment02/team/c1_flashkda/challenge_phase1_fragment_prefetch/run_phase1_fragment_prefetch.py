#!/usr/bin/env python3
"""Exactness plus preregistered H12 full-wrapper timing for Phase-1 prefetch."""

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

from assignment02.team.c1_flashkda.challenge_phase1_fragment_prefetch import phase1_fragment_prefetch
from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2
from assignment02.team.c1_flashkda.challenge_vshard4_prefetch2 import vshard4_prefetch2
from assignment02.team.c1_flashkda.harness import validate_and_bench as common


CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
VARIANTS = ("baseline", "vshard2_p2s3", "vshard4_p2s3", "phase1pf")


def csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if set(result) != set(CONTRACTS) or len(result) != len(CONTRACTS):
        raise ValueError(f"contracts must be exactly {CONTRACTS}")
    return result


def identity() -> dict[str, object]:
    import flash_kda_C

    required = ("fwd", "fwd_vshard", "fwd_vshard_p2", "fwd_vshard4", "fwd_vshard4_p2", "fwd_vshard4_p2_phase1pf", "get_workspace_size")
    missing = [name for name in required if not hasattr(flash_kda_C, name)]
    if missing:
        raise RuntimeError(f"loaded extension lacks Phase-1 candidate ABI: {missing}")
    extension = Path(flash_kda_C.__file__).resolve()
    return {"extension": str(extension), "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(), "required_symbols": list(required)}


def states(contract: str, heads: int, seed: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if contract == "none":
        return None, None
    dtype = torch.bfloat16 if contract == "bf16_both" else torch.float32
    final = torch.zeros((1, heads, 128, 128), dtype=dtype, device="cuda")
    if contract == "fp32_final_only":
        return None, final
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(final.shape, dtype=dtype, device="cuda", generator=generator).contiguous(), final


def clone(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.clone()


def invoke(fn: Callable[..., None], x: common.Inputs, initial: torch.Tensor | None, final: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    out = torch.zeros_like(x.v)
    fn(x.q, x.k, x.v, x.g, x.beta, x.scale, out, A_log=x.a_log, dt_bias=x.dt_bias,
       lower_bound=x.lower_bound, initial_state=initial, final_state=final)
    torch.cuda.synchronize()
    return out, final


def exact_compare(label: str, actual: tuple[torch.Tensor, torch.Tensor | None], expected: tuple[torch.Tensor, torch.Tensor | None]) -> dict[str, object]:
    common.require_exact(label + "/output", actual[0], expected[0])
    result: dict[str, object] = {"output_exact": True, "output_max_abs": common.max_abs(actual[0], expected[0])}
    if actual[1] is None or expected[1] is None:
        if actual[1] is not None or expected[1] is not None:
            raise AssertionError(label + ": final-state presence mismatch")
        result["final_state_present"] = False
    else:
        common.require_exact(label + "/final_state", actual[1], expected[1])
        result.update({"final_state_present": True, "final_state_exact": True, "final_state_max_abs": common.max_abs(actual[1], expected[1])})
    return result


def exact(x: common.Inputs, contract: str, seed: int) -> dict[str, object]:
    import flash_kda

    initial, final = states(contract, x.q.shape[2], seed)
    fns: dict[str, Callable[..., None]] = {
        "baseline": flash_kda.fwd, "vshard2_p2s3": prefetch2.fwd,
        "vshard4_p2s3": vshard4_prefetch2.fwd, "phase1pf": phase1_fragment_prefetch.fwd,
    }
    outputs = {name: invoke(fn, x, clone(initial), clone(final)) for name, fn in fns.items()}
    return {name + "_vs_baseline": exact_compare(name + "_vs_baseline", outputs[name], outputs["baseline"]) for name in VARIANTS[1:]}


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary(values: list[float]) -> dict[str, float | int]:
    return {"samples": len(values), "mean_ms": statistics.fmean(values), "p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95), "p99_ms": percentile(values, 99), "min_ms": min(values), "max_ms": max(values)}


def timed_calls(x: common.Inputs, contract: str, seed: int) -> dict[str, Callable[[], None]]:
    import flash_kda

    fns: dict[str, Callable[..., None]] = {
        "baseline": flash_kda.fwd, "vshard2_p2s3": prefetch2.fwd,
        "vshard4_p2s3": vshard4_prefetch2.fwd, "phase1pf": phase1_fragment_prefetch.fwd,
    }
    calls: dict[str, Callable[[], None]] = {}
    for index, (label, fn) in enumerate(fns.items()):
        initial, final = states(contract, x.q.shape[2], seed + index * 1009)
        out = torch.empty_like(x.v)
        def call(fn: Callable[..., None] = fn, initial: torch.Tensor | None = initial, final: torch.Tensor | None = final, out: torch.Tensor = out) -> None:
            fn(x.q, x.k, x.v, x.g, x.beta, x.scale, out, A_log=x.a_log, dt_bias=x.dt_bias,
               lower_bound=x.lower_bound, initial_state=initial, final_state=final)
        calls[label] = call
    return calls


def benchmark(x: common.Inputs, contract: str, seed: int, warmup: int, samples: int) -> dict[str, object]:
    calls = timed_calls(x, contract, seed)
    labels = tuple(calls)
    for index in range(warmup):
        order = labels[index % len(labels):] + labels[:index % len(labels)]
        for label in order:
            calls[label]()
    torch.cuda.synchronize()
    raw = {label: [] for label in labels}
    for index in range(samples):
        order = labels[index % len(labels):] + labels[:index % len(labels)]
        for label in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); calls[label](); end.record(); torch.cuda.synchronize()
            raw[label].append(float(start.elapsed_time(end)))
    paths = {label: summary(values) for label, values in raw.items()}
    return {
        "paths": paths, "raw_samples_ms": raw,
        "candidate_speedup_vs_vshard4_p2s3_x": {q: float(paths["vshard4_p2s3"][q]) / float(paths["phase1pf"][q]) for q in ("p50_ms", "p95_ms", "p99_ms")},
        "event_contract": "four-path cyclic rotation; one complete public-wrapper call per CUDA event; workspace allocation included",
        "sample_contract": f"{samples} raw CUDA-event samples retained per path",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=8192); parser.add_argument("--H", type=int, default=12)
    parser.add_argument("--contracts", default=",".join(CONTRACTS)); parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--warmup", type=int, default=30); parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=2); parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.T != 8192 or args.H != 12 or args.samples != 1000 or args.repeats < 2 or args.warmup < 0:
        raise RuntimeError("preregistered run requires CUDA, H12/T8192, 1000 samples per path, >=2 repeats")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("preregistered run requires exactly one visible GPU")
    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    capability = tuple(torch.cuda.get_device_capability(device_index))
    properties = torch.cuda.get_device_properties(device_index)
    if "B300" not in device_name.upper() or capability != (10, 3) or properties.multi_processor_count != 148:
        raise RuntimeError(
            "preregistered run requires B300/SM10.3/148 SM, got "
            f"name={device_name!r}, capability={capability}, sm_count={properties.multi_processor_count}"
        )
    contracts, extension = csv(args.contracts), identity()
    x = common.make_inputs(args.T, args.H, args.seed)
    result: dict[str, object] = {
        "candidate": "fwd_vshard4_p2_phase1pf", "candidate_status": "non-production; dispatch registration is forbidden",
        "device": device_name, "capability": list(capability),
        "multiprocessor_count": properties.multi_processor_count,
        "build_target": os.environ.get("C1_BUILD_TARGET", "unspecified"),
        "allocation": {"slurm_job_id": os.environ.get("SLURM_JOB_ID", "none"), "hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unspecified")},
        "extension": extension, "contracts": list(contracts), "seed": args.seed,
        "h12": {"shape": {"B": 1, "T": args.T, "H": args.H, "K": 128, "V": 128},
                "exact": {contract: exact(x, contract, args.seed + index * 100_003) for index, contract in enumerate(contracts)},
                "repeats": [{contract: benchmark(x, contract, args.seed + repeat * 1_000_003 + index * 100_003, args.warmup, args.samples) for index, contract in enumerate(contracts)} for repeat in range(args.repeats)]},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
