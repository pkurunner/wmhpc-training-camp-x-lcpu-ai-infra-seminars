#!/usr/bin/env python3
"""Fail-closed production-lifecycle entry point for the v12 native-C2 wheel.

The immutable staged lifecycle core performs the real checks.  This reviewed
entry point binds that core to the accepted v12 Q-row-padding wheel and makes
the result schema and minimum lifecycle contract explicit: eight concurrent
first loaders, real MiniMaxM3SparseMSAImpl.forward calls, 1,000 steady calls,
two 100-replay CUDA-Graph runs with a static-input mutation, native selection
and rejection boundaries, exact sequence boundaries, dispatcher persistence,
and bounded CUDA memory counters.  A rejected shape is only a native selection
result; this program never claims that a fallback backend executed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from types import ModuleType
from typing import Any


_SCHEMA = "c2-native-c2-v12-production-lifecycle-v1"
_CORE_SCHEMA = "c2-native-c2-v12-lifecycle-core-binding-v1"
_THREAD_COUNT = 8
_MIN_STEADY_ITERATIONS = 1000
_MIN_GRAPH_REPLAYS = 100


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-root", required=True, type=Path)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--base-harness", required=True, type=Path)
    parser.add_argument("--lifecycle-core", required=True, type=Path)
    parser.add_argument("--expected-lifecycle-core-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--steady-iterations", type=int, default=_MIN_STEADY_ITERATIONS)
    parser.add_argument("--graph-replays", type=int, default=_MIN_GRAPH_REPLAYS)
    parser.add_argument("--memory-bound-bytes", type=int, default=8 * 1024 * 1024)
    return parser.parse_args()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _load_core(path: Path, expected_sha: str) -> ModuleType:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise FileNotFoundError("--lifecycle-core must name a staged absolute regular file")
    if path.resolve() == Path(__file__).resolve():
        raise RuntimeError("lifecycle core must be distinct from the v12 entry point")
    if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise ValueError("--expected-lifecycle-core-sha must be lowercase SHA-256")
    actual_sha = _sha256_path(path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"lifecycle core SHA-256 mismatch: {actual_sha}")
    spec = importlib.util.spec_from_file_location("native_c2_v12_lifecycle_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import staged lifecycle core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required_callables = ("_run", "_write_json")
    required_constants = ("_THREAD_COUNT", "_MIN_STEADY_ITERATIONS", "_MIN_GRAPH_REPLAYS")
    missing = [name for name in required_callables if not callable(getattr(module, name, None))]
    missing.extend(name for name in required_constants if not hasattr(module, name))
    if missing:
        raise RuntimeError(f"staged lifecycle core lacks required entry points: {missing}")
    if int(module._THREAD_COUNT) != _THREAD_COUNT:
        raise RuntimeError("staged lifecycle core does not require exactly eight first-load callers")
    if int(module._MIN_STEADY_ITERATIONS) != _MIN_STEADY_ITERATIONS:
        raise RuntimeError("staged lifecycle core does not require 1,000 real steady forwards")
    if int(module._MIN_GRAPH_REPLAYS) != _MIN_GRAPH_REPLAYS:
        raise RuntimeError("staged lifecycle core does not require 100 CUDA Graph replays")
    module._SCHEMA = _SCHEMA
    return module


def _assert_bounded_memory(stage: dict[str, Any], label: str) -> None:
    counters = stage.get("cuda_memory_counters", {})
    samples = stage.get("memory_allocated", {})
    if counters.get("pass") is not True or samples.get("pass") is not True:
        raise RuntimeError(f"v12 lifecycle {label} memory bound did not pass")
    if counters.get("bound_bytes") != 8 * 1024 * 1024:
        raise RuntimeError(f"v12 lifecycle {label} used the wrong memory bound")
    expected_counter_checks = {
        "device_free_drop_bounded",
        "device_total_unchanged",
        "end_allocated_growth_bounded",
        "end_reserved_growth_bounded",
        "peak_allocated_growth_bounded",
        "peak_reserved_growth_bounded",
    }
    if set(counters.get("checks", {})) != expected_counter_checks or not all(counters["checks"].values()):
        raise RuntimeError(f"v12 lifecycle {label} CUDA counter checks are incomplete")
    expected_sample_checks = {
        "within_absolute_bound_from_baseline",
        "within_observed_range_bound",
    }
    if set(samples.get("checks", {})) != expected_sample_checks or not all(samples["checks"].values()):
        raise RuntimeError(f"v12 lifecycle {label} allocation samples are incomplete")


def _required_gate_check(result: dict[str, Any]) -> None:
    required = {
        "fresh_adapter_eight_thread_first_load",
        "real_forward_reaches_dispatcher_and_native_cuda_kernel",
        "steady_1000_real_forward_lifecycle",
        "cuda_graph_capture_replay_and_static_query_update",
        "native_support_predicate_matrix",
        "dynamic_native_rejection_without_fallback_claim",
        "native_forward_sequence_boundaries",
        "dispatcher_surface_persists_through_lifecycle",
    }
    gates = result.get("gates")
    if not isinstance(gates, dict) or set(gates) != required or not all(gates.values()):
        raise RuntimeError("one or more mandatory v12 lifecycle gates did not pass")
    first_load = result.get("first_load", {}).get("concurrent_loader", {})
    if first_load.get("thread_count") != _THREAD_COUNT or first_load.get("load_library_call_count") != 1:
        raise RuntimeError("v12 lifecycle did not prove exactly one loader call across eight threads")
    steady = result.get("steady_lifecycle", {})
    if steady.get("iterations") != _MIN_STEADY_ITERATIONS or steady.get("pass") is not True:
        raise RuntimeError("v12 lifecycle did not complete 1,000 real forwards")
    _assert_bounded_memory(steady, "steady")
    graph = result.get("cuda_graph", {})
    if graph.get("pass") is not True:
        raise RuntimeError("v12 lifecycle CUDA Graph gate did not pass")
    for label in ("original_query", "updated_query"):
        replays = graph.get(label, {}).get("replays", {})
        if replays.get("replays") != _MIN_GRAPH_REPLAYS or replays.get("pass") is not True:
            raise RuntimeError(f"v12 lifecycle did not replay {label} 100 times")
        _assert_bounded_memory(replays, f"CUDA Graph {label}")
    dynamic = result.get("dynamic_rejection_gate", {})
    if dynamic.get("mode") != "selection_only" or not dynamic.get("rows") or not all(row.get("pass") is True for row in dynamic["rows"]):
        raise RuntimeError("v12 lifecycle rejection boundary is incomplete")
    boundaries = result.get("sequence_length_boundaries", {}).get("rows", [])
    if [row.get("sequence_length") for row in boundaries] != [2048, 2049, 4095, 4096]:
        raise RuntimeError("v12 lifecycle did not cover the required sequence-length boundaries")
    if not all(row.get("pass") is True for row in boundaries):
        raise RuntimeError("one or more v12 sequence-length boundaries failed")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steady_iterations != _MIN_STEADY_ITERATIONS:
        raise ValueError("v12 lifecycle requires exactly 1,000 steady forwards")
    if args.graph_replays != _MIN_GRAPH_REPLAYS:
        raise ValueError("v12 lifecycle requires exactly 100 graph replays per static query")
    if args.memory_bound_bytes != 8 * 1024 * 1024:
        raise ValueError("v12 lifecycle requires an exact 8 MiB CUDA-memory bound")
    core = _load_core(args.lifecycle_core, args.expected_lifecycle_core_sha)
    result = core._run(args)
    if not isinstance(result, dict):
        raise RuntimeError("staged lifecycle core returned a non-object result")
    result["schema"] = _SCHEMA
    _required_gate_check(result)
    result["lifecycle_core"] = {
        "schema": _CORE_SCHEMA,
        "staged_path": str(args.lifecycle_core),
        "sha256": args.expected_lifecycle_core_sha,
        "exactly_eight_threads": _THREAD_COUNT,
        "steady_real_forwards": _MIN_STEADY_ITERATIONS,
        "cuda_graph_replays_per_static_query": _MIN_GRAPH_REPLAYS,
        "memory_bound_bytes": args.memory_bound_bytes,
    }
    result["all_gates_pass"] = True
    return result


def main() -> int:
    args = _parse_args()
    for field in ("c2_root", "vllm_root", "base_harness", "lifecycle_core", "output"):
        setattr(args, field, getattr(args, field).resolve())
    try:
        result = _run(args)
    except Exception as error:
        result = {
            "schema": _SCHEMA,
            "all_gates_pass": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
    _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("all_gates_pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
