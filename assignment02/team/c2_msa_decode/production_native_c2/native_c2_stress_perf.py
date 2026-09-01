#!/usr/bin/env python3
"""Strict multi-seed correctness, stability, and CUDA-event timing for C2.

The target is deliberately the registered AOT operator itself.  In particular,
this program only loads ``_C_stable_libtorch.abi3.so`` with
``torch.ops.load_library`` and invokes ``torch.ops._C.native_c2_msa_decode``;
it neither imports the vLLM MSA dispatch path nor monkeypatches anything.

The existing direct-operator harness owns the fixed production contract and
the independently written oracle.  This file imports only those pure helpers
so its input/oracle semantics cannot silently drift from the first-stage gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import traceback
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch


_DEFAULT_SEEDS = (17, 23, 42, 2024, 314159, 20260801, 20260815, 20260829)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--library", type=Path, required=True,
        help="absolute path to the derived _C_stable_libtorch.abi3.so",
    )
    parser.add_argument(
        "--base-harness", type=Path,
        default=Path(__file__).resolve().with_name("native_c2_operator_bench.py"),
        help="absolute path to native_c2_operator_bench.py",
    )
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in _DEFAULT_SEEDS),
        help="comma-separated, unique integer seeds (at least eight)",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--stability-repeats", type=int, default=4)
    parser.add_argument("--num-physical-pages", type=int, default=64)
    parser.add_argument("--max-logical-pages", type=int, default=32)
    parser.add_argument("--scale", type=float, default=1.0 / math.sqrt(128))
    parser.add_argument("--q-scale", type=float, default=0.25)
    parser.add_argument("--k-scale", type=float, default=0.25)
    parser.add_argument("--v-scale", type=float, default=0.5)
    parser.add_argument("--oracle-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-3)
    return parser.parse_args()


def _load_base_harness(path: Path) -> ModuleType:
    if not path.is_absolute():
        raise ValueError("--base-harness must be an absolute path")
    if not path.is_file():
        raise FileNotFoundError(f"--base-harness does not exist: {path}")
    specification = importlib.util.spec_from_file_location("native_c2_base_harness", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import base harness from {path}")
    module = importlib.util.module_from_spec(specification)
    # ``Contract`` is a dataclass in the base harness; dataclasses consult the
    # defining module through ``sys.modules`` while decorating that class.
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    for name in ("_BATCH", "_QUERY_HEADS", "_HEAD_DIM", "_make_inputs", "_oracle",
                 "_require_registered_operator", "_validate_args"):
        if not hasattr(module, name):
            raise RuntimeError(f"base harness lacks required helper: {name}")
    return module


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("--seeds must be a comma-separated integer list") from error
    if len(values) < 8:
        raise ValueError("--seeds must contain at least eight entries")
    if len(set(values)) != len(values):
        raise ValueError("--seeds entries must be unique")
    return values


def _contract_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        num_physical_pages=args.num_physical_pages,
        max_logical_pages=args.max_logical_pages,
        scale=args.scale,
        q_scale=args.q_scale,
        k_scale=args.k_scale,
        v_scale=args.v_scale,
        atol=args.atol,
        rtol=args.rtol,
    )


def _percentile(samples: list[float], fraction: float) -> float:
    if not samples:
        raise ValueError("cannot calculate a percentile of no samples")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _time_operator(
    op: Any,
    output: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    contract: Any,
    warmup: int,
    iters: int,
) -> list[float]:
    query, kv_cache, topk, block_table, seq_lens = inputs
    for _ in range(warmup):
        returned = op(
            output, query, kv_cache, topk, block_table, seq_lens,
            contract.scale, contract.q_scale, contract.k_scale, contract.v_scale,
        )
        if returned is not None:
            raise RuntimeError("native operator unexpectedly returned a value during warmup")
    torch.cuda.synchronize()
    pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        returned = op(
            output, query, kv_cache, topk, block_table, seq_lens,
            contract.scale, contract.q_scale, contract.k_scale, contract.v_scale,
        )
        end.record()
        if returned is not None:
            raise RuntimeError("native operator unexpectedly returned a value while timing")
        pairs.append((start, end))
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in pairs]


@torch.no_grad()
def _exercise_seed(
    base: ModuleType,
    op: Any,
    contract: Any,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    inputs = base._make_inputs(contract, seed)
    query = inputs[0]
    shape = (base._BATCH, base._QUERY_HEADS, base._HEAD_DIM)
    output = torch.full(shape, float("nan"), device=query.device, dtype=torch.bfloat16)
    output_pointer = output.data_ptr()
    returned = op(
        output, *inputs, contract.scale, contract.q_scale, contract.k_scale, contract.v_scale
    )
    torch.cuda.synchronize()
    pointer_unchanged = output.data_ptr() == output_pointer
    oracle_dtype = torch.float64 if args.oracle_dtype == "float64" else torch.float32
    reference = base._oracle(*inputs, contract, oracle_dtype)
    actual = output.to(oracle_dtype)
    difference = (actual - reference).abs()
    denominator = reference.abs().clamp_min(torch.finfo(oracle_dtype).eps)
    finite = bool(torch.isfinite(actual).all().item())
    allclose = bool(torch.allclose(actual, reference, atol=args.atol, rtol=args.rtol))

    # Determinism is checked before timing.  Every allocation is caller-owned
    # and must retain its address through the direct operator call.
    repeat_outputs = [output.clone()]
    repeat_pointers_unchanged = True
    repeat_returns_none = returned is None
    for _ in range(args.stability_repeats - 1):
        repeated = torch.full(shape, float("nan"), device=query.device, dtype=torch.bfloat16)
        pointer_before = repeated.data_ptr()
        repeat_returned = op(
            repeated, *inputs, contract.scale, contract.q_scale, contract.k_scale, contract.v_scale
        )
        repeat_pointers_unchanged = repeat_pointers_unchanged and repeated.data_ptr() == pointer_before
        repeat_returns_none = repeat_returns_none and repeat_returned is None
        repeat_outputs.append(repeated)
    torch.cuda.synchronize()
    bitwise_repeatable = all(torch.equal(repeat_outputs[0], repeated) for repeated in repeat_outputs[1:])
    all_finite_after_repeats = all(bool(torch.isfinite(repeated).all().item()) for repeated in repeat_outputs)

    timing_output = torch.empty(shape, device=query.device, dtype=torch.bfloat16)
    timing_pointer = timing_output.data_ptr()
    samples_ms = _time_operator(op, timing_output, inputs, contract, args.warmup, args.iters)
    timing_pointer_unchanged = timing_output.data_ptr() == timing_pointer
    timing_finite = bool(torch.isfinite(timing_output).all().item())
    mean_ms = statistics.fmean(samples_ms)
    request_rate = base._BATCH * 1000.0 / mean_ms
    all_gates = bool(
        finite and allclose and pointer_unchanged and returned is None
        and repeat_pointers_unchanged and repeat_returns_none and bitwise_repeatable
        and all_finite_after_repeats and timing_pointer_unchanged and timing_finite
    )
    return {
        "all_gates_pass": all_gates,
        "caller_output": {
            "pointer_before": output_pointer,
            "pointer_after": output.data_ptr(),
            "pointer_unchanged": pointer_unchanged,
            "return_is_none": returned is None,
            "timing_pointer_unchanged": timing_pointer_unchanged,
        },
        "correctness": {
            "allclose": allclose,
            "atol": args.atol,
            "finite_output": finite,
            "max_abs": float(difference.max().item()),
            "max_rel": float((difference / denominator).max().item()),
            "mean_abs": float(difference.mean().item()),
            "oracle_dtype": args.oracle_dtype,
            "rtol": args.rtol,
        },
        "latency": {
            "iterations": len(samples_ms),
            "mean_ms": mean_ms,
            "min_ms": min(samples_ms),
            "p50_ms": _percentile(samples_ms, 0.50),
            "p95_ms": _percentile(samples_ms, 0.95),
            "max_ms": max(samples_ms),
            "method": "one CUDA-event pair per direct operator call; one synchronize after all iterations",
        },
        "seed": seed,
        "stability": {
            "all_finite": all_finite_after_repeats,
            "bitwise_repeatable": bitwise_repeatable,
            "repeat_count": args.stability_repeats,
            "repeat_pointers_unchanged": repeat_pointers_unchanged,
            "repeat_returns_none": repeat_returns_none,
            "timing_output_finite": timing_finite,
        },
        "throughput": {
            "batch_decode_requests_per_s": request_rate,
            "query_tokens_per_s": request_rate,
            "query_head_vectors_per_s": request_rate * base._QUERY_HEADS,
        },
    }


@torch.no_grad()
def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 3):
        raise RuntimeError(f"native C2 requires B300 capability (10, 3), got {capability}")
    if args.warmup < 20:
        raise ValueError("--warmup must be at least 20")
    if args.iters < 100:
        raise ValueError("--iters must be at least 100")
    if args.stability_repeats < 2:
        raise ValueError("--stability-repeats must be at least 2")
    seeds = _parse_seeds(args.seeds)
    base = _load_base_harness(args.base_harness)
    contract = base._validate_args(_contract_args(args))
    operator_library = base._require_registered_operator(args.library)
    op = torch.ops._C.native_c2_msa_decode
    results = [_exercise_seed(base, op, contract, seed, args) for seed in seeds]
    aggregate_samples = [
        result["latency"]["mean_ms"] for result in results
    ]
    return {
        "all_gates_pass": all(bool(result["all_gates_pass"]) for result in results),
        "base_harness": str(args.base_harness),
        "contract": {
            "batch": base._BATCH,
            "head_dim": base._HEAD_DIM,
            "kv_heads": base._KV_HEADS,
            "max_logical_pages": contract.max_logical_pages,
            "num_physical_pages": contract.num_physical_pages,
            "page_size": base._PAGE_SIZE,
            "q_heads": base._QUERY_HEADS,
            "q_scale": contract.q_scale,
            "k_scale": contract.k_scale,
            "v_scale": contract.v_scale,
            "scale": contract.scale,
            "topk": base._TOPK,
        },
        "correctness_gate": {
            "atol": args.atol,
            "oracle_dtype": args.oracle_dtype,
            "rtol": args.rtol,
        },
        "device": {
            "capability": list(capability),
            "name": torch.cuda.get_device_name(),
        },
        "dispatch": "direct_torch.ops._C.native_c2_msa_decode",
        "no_monkeypatch": True,
        "operator_library": operator_library,
        "performance_note": "The FP64/FP32 oracle is used only for correctness and is excluded from all latency and throughput measurements.",
        "measurement": {
            "iterations_per_seed": args.iters,
            "stability_repeats": args.stability_repeats,
            "warmup_per_seed": args.warmup,
        },
        "schema": "c2-native-c2-stress-perf-v1",
        "seed_count": len(seeds),
        "seeds": results,
        "summary_latency_of_seed_means_ms": {
            "mean_ms": statistics.fmean(aggregate_samples),
            "p50_ms": _percentile(aggregate_samples, 0.50),
            "p95_ms": _percentile(aggregate_samples, 0.95),
        },
        "torch": {"cuda": torch.version.cuda, "version": torch.__version__},
    }


def _as_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    try:
        result = _run(_parse_args())
    except Exception as error:
        result = {
            "all_gates_pass": False,
            "error": f"{type(error).__name__}: {error}",
            "schema": "c2-native-c2-stress-perf-v1",
            "traceback": traceback.format_exc(),
        }
        print(_as_json(result))
        return 1
    print(_as_json(result))
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
