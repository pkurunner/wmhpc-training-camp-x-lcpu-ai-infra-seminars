#!/usr/bin/env python3
"""One direct-plugin C2 worker over a pre-generated immutable fixture set."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("c2_v11_worker_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = ("Contract", "_BATCH", "_QUERY_HEADS", "_HEAD_DIM", "_oracle")
    for name in required:
        if not hasattr(module, name):
            raise RuntimeError(f"base harness lacks {name}")
    return module


def visible_gpu_uuid() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if len(entries) == 1 and entries[0].startswith("GPU-"):
        return entries[0]
    if len(entries) == 1 and entries[0].isdigit():
        return subprocess.check_output(
            ["nvidia-smi", "--id", entries[0], "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    props = torch.cuda.get_device_properties(0)
    uuid = getattr(props, "uuid", None)
    if isinstance(uuid, bytes):
        uuid = uuid.decode("ascii")
    if isinstance(uuid, str) and uuid.startswith("GPU-"):
        return uuid
    raise RuntimeError("cannot establish the allocated physical GPU UUID fail-closed")


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def call(op: Any, output: torch.Tensor, inputs: tuple[torch.Tensor, ...], contract: Any):
    return op(output, *inputs, contract.scale, contract.q_scale, contract.k_scale, contract.v_scale)


def profile_one(op: Any, inputs: tuple[torch.Tensor, ...], contract: Any, shape: tuple[int, ...]) -> dict[str, Any]:
    output = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    pointer = output.data_ptr()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        returned = call(op, output, inputs, contract)
        torch.cuda.synchronize()
    events = profiler.key_averages()
    dispatcher = [event for event in events if str(event.key) == "_C::native_c2_msa_decode"]
    kernels = [event for event in events if "native_c2_msa_decode_kernel" in str(event.key)]
    checks = {
        "one_dispatcher_event": len(dispatcher) == 1 and int(getattr(dispatcher[0], "count", 0)) == 1,
        "one_native_cuda_kernel_event": len(kernels) == 1 and int(getattr(kernels[0], "count", 0)) == 1,
        "caller_pointer_unchanged": output.data_ptr() == pointer,
        "return_is_none": returned is None,
    }
    return {
        "pass": all(checks.values()), "checks": checks,
        "dispatcher_events": [{"key": str(event.key), "count": int(getattr(event, "count", 0))} for event in dispatcher],
        "cuda_kernel_events": [{"key": str(event.key), "count": int(getattr(event, "count", 0))} for event in kernels],
    }


def exercise(base: Any, op: Any, inputs: tuple[torch.Tensor, ...], contract: Any, seed: int) -> dict[str, Any]:
    shape = (base._BATCH, base._QUERY_HEADS, base._HEAD_DIM)
    output = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    pointer_before = output.data_ptr()
    returned = call(op, output, inputs, contract)
    torch.cuda.synchronize()
    reference = base._oracle(*inputs, contract, torch.float32)
    reference_fp64 = base._oracle(*inputs, contract, torch.float64)
    actual = output.to(torch.float32)
    actual_fp64 = output.to(torch.float64)
    difference = (actual - reference).abs()
    fp64_difference = (actual_fp64 - reference_fp64).abs()
    denominator = reference.abs().clamp_min(torch.finfo(torch.float32).eps)
    finite = bool(torch.isfinite(actual).all().item())
    allclose = bool(torch.allclose(actual, reference, atol=1.0e-4, rtol=1.0e-3))
    fp64_allclose = bool(torch.allclose(actual_fp64, reference_fp64, atol=1.0e-4, rtol=1.0e-3))
    reference_fp64_fp32_agree = bool(torch.allclose(reference_fp64, reference.to(torch.float64), atol=1.0e-4, rtol=1.0e-3))
    repeat_outputs = [output.clone()]
    repeat_pointers = output.data_ptr() == pointer_before
    repeat_returns = returned is None
    for _ in range(3):
        repeated = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
        repeated_pointer = repeated.data_ptr()
        repeated_return = call(op, repeated, inputs, contract)
        repeat_pointers = repeat_pointers and repeated.data_ptr() == repeated_pointer
        repeat_returns = repeat_returns and repeated_return is None
        repeat_outputs.append(repeated)
    torch.cuda.synchronize()
    bitwise = all(torch.equal(repeat_outputs[0], other) for other in repeat_outputs[1:])
    warmup_output = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    for _ in range(30):
        if call(op, warmup_output, inputs, contract) is not None:
            raise RuntimeError("operator returned a value during warmup")
    torch.cuda.synchronize()
    timing_output = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    timing_pointer = timing_output.data_ptr()
    pairs = []
    for _ in range(200):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        timing_return = call(op, timing_output, inputs, contract)
        end.record()
        if timing_return is not None:
            raise RuntimeError("operator returned a value during timing")
        pairs.append((start, end))
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in pairs]
    profile = profile_one(op, inputs, contract, shape)
    timing_finite = bool(torch.isfinite(timing_output).all().item())
    all_gates = bool(finite and allclose and fp64_allclose and reference_fp64_fp32_agree
                     and output.data_ptr() == pointer_before and returned is None
                     and repeat_pointers and repeat_returns and bitwise
                     and timing_output.data_ptr() == timing_pointer and timing_finite and profile["pass"])
    return {
        "seed": seed, "all_gates_pass": all_gates,
        "correctness": {"oracle_dtype": "float32", "atol": 1.0e-4, "rtol": 1.0e-3,
                        "finite_output": finite, "allclose": allclose,
                        "max_abs": float(difference.max().item()),
                        "mean_abs": float(difference.mean().item()),
                        "max_rel": float((difference / denominator).max().item()),
                        "fp64_allclose": fp64_allclose,
                        "fp64_max_abs": float(fp64_difference.max().item()),
                        "reference_fp64_fp32_agree": reference_fp64_fp32_agree},
        "caller_output": {"pointer_before": pointer_before, "pointer_after": output.data_ptr(),
                          "pointer_unchanged": output.data_ptr() == pointer_before,
                          "return_is_none": returned is None,
                          "timing_pointer_unchanged": timing_output.data_ptr() == timing_pointer,
                          "timing_output_finite": timing_finite},
        "stability": {"repeat_count": 4, "bitwise_repeatable": bitwise,
                      "repeat_pointers_unchanged": repeat_pointers,
                      "repeat_returns_none": repeat_returns},
        "profiling": profile,
        "latency": {"mean_ms": statistics.fmean(samples), "min_ms": min(samples),
                    "p50_ms": percentile(samples, 0.50), "p95_ms": percentile(samples, 0.95),
                    "max_ms": max(samples), "iterations": len(samples),
                    "method": "one CUDA-event pair per direct call; warmup/oracle/profiler excluded"},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
    if torch.cuda.get_device_capability(0) != (10, 3):
        raise RuntimeError(f"B300 capability required, got {torch.cuda.get_device_capability(0)}")
    if not args.library.is_absolute() or not args.base_harness.is_absolute() or not args.fixture.is_absolute():
        raise ValueError("library, base harness, and fixture must be absolute paths")
    if sha256(args.fixture) != args.fixture_sha256:
        raise RuntimeError("fixture file SHA-256 drifted")
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    if fixture.get("schema") != "c2-native-v11-q-fragment-reuse-fixed-fixtures-v1":
        raise RuntimeError("unexpected fixture schema")
    seeds = [17, 23, 42, 2024, 314159, 20260801, 20260815, 20260829]
    if fixture.get("seeds") != seeds or len(fixture.get("rows", [])) != len(seeds):
        raise RuntimeError("fixture seed contract drifted")
    base = load_base(args.base_harness)
    contract = base.Contract(**fixture["contract"])
    torch.ops.load_library(str(args.library))
    if not (hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "native_c2_msa_decode")):
        raise RuntimeError("plugin did not register _C::native_c2_msa_decode")
    if not torch._C._dispatch_has_kernel_for_dispatch_key("_C::native_c2_msa_decode", "CUDA"):
        raise RuntimeError("plugin lacks CUDA dispatch implementation")
    op = torch.ops._C.native_c2_msa_decode
    records = []
    for expected_seed, row in zip(seeds, fixture["rows"], strict=True):
        if row.get("seed") != expected_seed:
            raise RuntimeError("fixture row order drifted")
        inputs = tuple(tensor.to(device="cuda").contiguous() for tensor in row["inputs"])
        records.append(exercise(base, op, inputs, contract, expected_seed) | {"fixture_tensor_sha256": row["tensor_sha256"]})
    gpu_uuid = visible_gpu_uuid()
    return {
        "schema": "c2-native-plugin-v11-q-fragment-reuse-fixed-fixture-worker-v1", "role": args.role,
        "all_gates_pass": all(record["all_gates_pass"] for record in records),
        "operator_library": str(args.library.resolve()), "operator_library_sha256": sha256(args.library),
        "base_harness": str(args.base_harness.resolve()), "base_harness_sha256": sha256(args.base_harness),
        "worker_harness": str(Path(__file__).resolve()), "worker_harness_sha256": sha256(Path(__file__)),
        "fixture": str(args.fixture.resolve()), "fixture_sha256": args.fixture_sha256,
        "fixture_schema": fixture["schema"], "fixture_seeds": seeds,
        "contract": fixture["contract"], "correctness_gate": {"oracle_dtype": "float32", "atol": 1.0e-4, "rtol": 1.0e-3},
        "measurement": {"warmup_per_seed": 30, "iterations_per_seed": 200, "stability_repeats": 4},
        "dispatch": "direct_torch.ops._C.native_c2_msa_decode", "no_monkeypatch": True,
        "device": {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
                   "gpu_uuid": gpu_uuid, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "seeds": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--base-harness", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--role", required=True)
    args = parser.parse_args()
    try:
        payload = run(args)
    except Exception as error:
        payload = {"schema": "c2-native-plugin-v11-q-fragment-reuse-fixed-fixture-worker-v1", "role": args.role,
                   "all_gates_pass": False, "error": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()}
        print(json.dumps(payload, allow_nan=False, sort_keys=True))
        return 1
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0 if payload["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
