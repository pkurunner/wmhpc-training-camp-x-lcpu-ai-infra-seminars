#!/usr/bin/env python3
"""Diagnostic-only fp32-both tail-latency experiment for the B300 public FLA route.

This is deliberately not a release gate.  Every GPU process begins from the
frozen two-cell production map, temporarily adds exactly one target cell in
process memory, and proves restoration of that same mutable map object before
it exits.  The three ``stability_main`` processes are the only stability
evidence.  A separate ``non_gating_telemetry`` process may be correlated with
an NVML sidecar by the outer shell, but never promotes a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_seqcount_dispatch import run_seqcount_dispatch as shared  # noqa: E402
from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch, varlen_metadata  # noqa: E402
from assignment02.team.c1_flashkda.challenge_varlen_dispatch import run_varlen_dispatch_confirmation as confirmation  # noqa: E402
from assignment02.team.c1_flashkda.challenge_varlen_dispatch import run_varlen_fla_handoff_candidate as candidate  # noqa: E402


SAMPLES_PER_PATH = 1000
REPEATS = 2
WARMUP_PER_PATH = 100
BLOCK_PAIRS = 100
TARGET_CASE_NAME = "skew_n6_h12_t12288"
TARGET_CONTRACT = "fp32_both"
TARGET_VARIANT = "vshard4_p2"
CLEAN_GPU_GATE_ENV = "C1_VARLEN_FLA_FP32_TAIL_DIAG_CLEAN_GPU"
RUNNER_SHA256_ENV = "C1_VARLEN_FLA_FP32_TAIL_DIAG_RUNNER_SHA256"
PATHS = ("public_registry_c1", "public_registry_pinned")
TAIL_THRESHOLD_MS = 1.20

TARGET_OFFSETS = (0, 1, 2, 3, 4, 5, 12288)
PRODUCTION_MAP = {
    (TARGET_OFFSETS, "none"): "vshard2_p2",
    (TARGET_OFFSETS, "fp32_final_only"): "vshard2_p2",
}
TEMPORARY_TARGET_MAP = {**PRODUCTION_MAP, (TARGET_OFFSETS, TARGET_CONTRACT): TARGET_VARIANT}


def _write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_case() -> shared.Case:
    matches = [case for case in confirmation.CASES if case.name == TARGET_CASE_NAME]
    if len(matches) != 1:
        raise RuntimeError(f"cannot identify exactly one target case: {TARGET_CASE_NAME}")
    case = matches[0]
    if tuple(case.lengths) != (1, 1, 1, 1, 1, 12283) or case.total_tokens != TARGET_OFFSETS[-1]:
        raise RuntimeError("target case layout drift")
    return case


def _target_cell() -> candidate.Cell:
    return candidate.Cell(_target_case(), TARGET_CONTRACT, TARGET_VARIANT)


def _runner_identity() -> dict[str, object]:
    path = Path(__file__).resolve(strict=True)
    actual = _sha(path)
    expected = os.environ.get(RUNNER_SHA256_ENV)
    if expected is None:
        raise RuntimeError(f"{RUNNER_SHA256_ENV} is required for a GPU diagnosis")
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise RuntimeError(f"{RUNNER_SHA256_ENV} must be a lowercase SHA256")
    if actual != expected:
        raise RuntimeError(f"tail-diagnosis runner SHA mismatch: expected {expected}, got {actual}")
    return {
        "path": str(path),
        "sha256": actual,
        "sha256_gate_pass": True,
        "expected_sha256_environment": RUNNER_SHA256_ENV,
    }


def _map_gate(expected: Mapping[tuple[tuple[int, ...], str], str], label: str) -> dict[str, object]:
    evidence = candidate._assert_map_values_and_behavior(expected, label)
    if evidence.get("passed") is not True:
        raise RuntimeError(f"{label}: map behavior evidence failed")
    return evidence


def _install_target_map() -> tuple[object, dict[str, object]]:
    """Mutate exactly the production map object, never a source file or new map."""

    original = auto_dispatch._VARLEN_PUBLIC_VARIANTS
    if not isinstance(original, dict):
        raise RuntimeError("production _VARLEN_PUBLIC_VARIANTS is not a mutable dict")
    before = _map_gate(PRODUCTION_MAP, "production-before-tail-diagnosis")
    original.clear()
    original.update(TEMPORARY_TARGET_MAP)
    if auto_dispatch._VARLEN_PUBLIC_VARIANTS is not original:
        raise RuntimeError("temporary target map replaced the production map object")
    installed = _map_gate(TEMPORARY_TARGET_MAP, "target-temporarily-installed")
    return original, {
        "production_map_object_id": id(original),
        "production_before": before,
        "temporary_target": installed,
        "temporary_entries": len(original),
        "target_only_extra_entry": True,
        "passed": True,
    }


def _restore_target_map(original: object) -> dict[str, object]:
    restored = candidate._restore_production_map(original, PRODUCTION_MAP)
    if restored.get("passed") is not True or restored.get("map_object_id") != id(original):
        raise RuntimeError("production map restoration evidence failed")
    return restored


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sample vector")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: list[float], *, expected_samples: int) -> dict[str, float | int]:
    if len(values) != expected_samples:
        raise ValueError(f"expected {expected_samples} samples, got {len(values)}")
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("CUDA-event samples must be finite positive milliseconds")
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _pair_summary(pairs: list[Mapping[str, object]]) -> dict[str, object]:
    c1_values = [float(pair["c1_ms"]) for pair in pairs]
    pinned_values = [float(pair["pinned_ms"]) for pair in pairs]
    deltas = [c1 - pinned for c1, pinned in zip(c1_values, pinned_values, strict=True)]
    return {
        "paired_samples": len(deltas),
        "delta_definition": "public_registry_c1_ms - public_registry_pinned_ms for the same alternating-order pair",
        "delta_mean_ms": statistics.fmean(deltas),
        "delta_p50_ms": _percentile(deltas, 0.50),
        "delta_p95_ms": _percentile(deltas, 0.95),
        "delta_p99_ms": _percentile(deltas, 0.99),
        "c1_gt_pinned_count": sum(c1 > pinned for c1, pinned in zip(c1_values, pinned_values, strict=True)),
        "c1_gt_threshold_count": sum(value > TAIL_THRESHOLD_MS for value in c1_values),
        "pinned_gt_threshold_count": sum(value > TAIL_THRESHOLD_MS for value in pinned_values),
        "c1_gt_pinned_and_gt_threshold_count": sum(
            c1 > pinned and c1 > TAIL_THRESHOLD_MS
            for c1, pinned in zip(c1_values, pinned_values, strict=True)
        ),
    }


def _relative_candidate_gate(c1: Mapping[str, float | int], pinned: Mapping[str, float | int]) -> dict[str, object]:
    """Report the existing 2% public candidate criterion without authorizing it."""

    winners: dict[str, str] = {}
    margins: dict[str, float | None] = {}
    for percentile_name in ("p50", "p95", "p99"):
        metric = f"{percentile_name}_ms"
        c1_latency = float(c1[metric])
        pinned_latency = float(pinned[metric])
        if c1_latency < pinned_latency:
            winners[percentile_name] = "public_registry_c1"
            margins[percentile_name] = pinned_latency / c1_latency - 1.0
        else:
            winners[percentile_name] = "public_registry_pinned"
            margins[percentile_name] = None
    passed = all(winners[name] == "public_registry_c1" and margins[name] is not None and float(margins[name]) >= 0.02 for name in winners)
    return {
        "definition": "C1 must win P50/P95/P99 against pinned with each margin >=2% in this repeat; diagnostic only",
        "winner_by_percentile": winners,
        "c1_margin_over_pinned_by_percentile": margins,
        "passed": passed,
    }


def _select_path(path: str) -> None:
    if path == "public_registry_c1":
        os.environ["C1_B300_FLASH_KDA"] = "1"
    elif path == "public_registry_pinned":
        os.environ["C1_B300_FLASH_KDA"] = "0"
    else:
        raise ValueError(f"unknown performance path: {path!r}")


def _expected_route(path: str) -> dict[str, int]:
    if path == "public_registry_c1":
        return {"c1": 1, "pinned": 0}
    if path == "public_registry_pinned":
        return {"c1": 0, "pinned": 1}
    raise ValueError(f"unknown performance path: {path!r}")


def _timed_repeat(
    *,
    repeat_index: int,
    x: object,
    cpu: object,
    gpu: object,
    initial: object,
    public_fn: Callable[..., Any],
    counts: dict[str, int],
    target: candidate.Cell,
) -> dict[str, object]:
    """Collect exactly 1,000 paired samples per path with fixed block boundaries."""

    import torch

    if SAMPLES_PER_PATH % 2 or WARMUP_PER_PATH % 2 or SAMPLES_PER_PATH % BLOCK_PAIRS:
        raise RuntimeError("registered schedule requires even samples and integral 100-pair blocks")
    final = True
    paths = PATHS
    reversed_paths = tuple(reversed(paths))

    def order(pair_index: int) -> tuple[str, str]:
        return paths if pair_index % 2 == 0 else reversed_paths

    def invoke(path: str, label: str) -> tuple[object, dict[str, object]]:
        _select_path(path)
        output, spy = candidate._spy_public(
            public_fn,
            x,
            initial,
            final,
            gpu,
            cpu,
            counts,
            path == "public_registry_c1",
            label,
        )
        if spy.get("delta") != _expected_route(path):
            raise AssertionError(f"{label}: route gate drift: {spy}")
        if path == "public_registry_c1":
            decision = auto_dispatch.get_last_decision()
            if decision.get("chosen_variant") != target.expected_variant:
                raise AssertionError(f"{label}: target C1 decision drift: {decision}")
        return output, spy

    immutable_snapshot = candidate._snapshot_input_tensors(x, gpu, cpu, initial)
    with torch.inference_mode():
        warm_before = dict(counts)
        first_c1_decision: dict[str, object] | None = None
        last_c1_decision: dict[str, object] | None = None
        for warm_index in range(WARMUP_PER_PATH):
            for path in order(warm_index):
                output, _ = invoke(path, f"tail/repeat{repeat_index}/warm{warm_index}/{path}")
                if path == "public_registry_c1":
                    last_c1_decision = auto_dispatch.get_last_decision()
                    if first_c1_decision is None:
                        first_c1_decision = last_c1_decision
                del output
        torch.cuda.synchronize()
        warm_after = dict(counts)
        expected_warm = {"c1": WARMUP_PER_PATH, "pinned": WARMUP_PER_PATH}
        warm_delta = {key: warm_after[key] - warm_before[key] for key in expected_warm}
        if warm_delta != expected_warm:
            raise AssertionError(f"tail/repeat{repeat_index}: warm route drift {warm_delta}")
        if (
            first_c1_decision is None
            or last_c1_decision is None
            or first_c1_decision.get("chosen_variant") != target.expected_variant
            or last_c1_decision.get("chosen_variant") != target.expected_variant
            or last_c1_decision.get("canonical_cache_hit") is not True
        ):
            raise AssertionError(f"tail/repeat{repeat_index}: C1 warm decision drift")

        timed_before = dict(counts)
        stream = torch.cuda.current_stream()
        pairs: list[dict[str, object]] = []
        blocks: list[dict[str, object]] = []
        block_started_ns: int | None = None
        for pair_index in range(SAMPLES_PER_PATH):
            if pair_index % BLOCK_PAIRS == 0:
                block_started_ns = time.time_ns()
            measurements: dict[str, float] = {}
            first_path, second_path = order(pair_index)
            for path in (first_path, second_path):
                _select_path(path)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                start.synchronize()
                output, spy = candidate._spy_public(
                    public_fn,
                    x,
                    initial,
                    final,
                    gpu,
                    cpu,
                    counts,
                    path == "public_registry_c1",
                    f"tail/repeat{repeat_index}/pair{pair_index}/{path}",
                )
                end.record(stream)
                end.synchronize()
                if spy.get("delta") != _expected_route(path):
                    raise AssertionError(f"tail/repeat{repeat_index}/pair{pair_index}: route drift {spy}")
                if path == "public_registry_c1":
                    decision = auto_dispatch.get_last_decision()
                    if decision.get("chosen_variant") != target.expected_variant:
                        raise AssertionError(f"tail/repeat{repeat_index}/pair{pair_index}: C1 decision drift")
                measurements[path] = float(start.elapsed_time(end))
                del output, start, end
            pairs.append(
                {
                    "pair_index": pair_index,
                    "block_index": pair_index // BLOCK_PAIRS,
                    "first_path": first_path,
                    "c1_ms": measurements["public_registry_c1"],
                    "pinned_ms": measurements["public_registry_pinned"],
                }
            )
            if (pair_index + 1) % BLOCK_PAIRS == 0:
                if block_started_ns is None:
                    raise AssertionError("block start timestamp missing")
                blocks.append(
                    {
                        "block_index": pair_index // BLOCK_PAIRS,
                        "pair_start": pair_index - BLOCK_PAIRS + 1,
                        "pair_end": pair_index,
                        "pair_count": BLOCK_PAIRS,
                        "epoch_ns_start": block_started_ns,
                        "epoch_ns_end": time.time_ns(),
                    }
                )
                block_started_ns = None
        timed_after = dict(counts)

    expected_timed = {"c1": SAMPLES_PER_PATH, "pinned": SAMPLES_PER_PATH}
    timed_delta = {key: timed_after[key] - timed_before[key] for key in expected_timed}
    if timed_delta != expected_timed:
        raise AssertionError(f"tail/repeat{repeat_index}: timed route drift {timed_delta}")
    if len(pairs) != SAMPLES_PER_PATH or len(blocks) != SAMPLES_PER_PATH // BLOCK_PAIRS:
        raise AssertionError("incomplete tail diagnosis sample or block coverage")
    immutability = candidate._assert_input_immutability(
        f"tail/repeat{repeat_index}/performance", immutable_snapshot, x, gpu, cpu, initial
    )
    c1_values = [float(pair["c1_ms"]) for pair in pairs]
    pinned_values = [float(pair["pinned_ms"]) for pair in pairs]
    first_counts = {
        path: sum(pair["first_path"] == path for pair in pairs)
        for path in PATHS
    }
    if first_counts != {"public_registry_c1": SAMPLES_PER_PATH // 2, "public_registry_pinned": SAMPLES_PER_PATH // 2}:
        raise AssertionError(f"unbalanced alternating schedule: {first_counts}")
    path_summaries = {
        "public_registry_c1": _summary(c1_values, expected_samples=SAMPLES_PER_PATH),
        "public_registry_pinned": _summary(pinned_values, expected_samples=SAMPLES_PER_PATH),
    }
    return {
        "repeat_index": repeat_index,
        "event_contract": "current-stream start event -> start.synchronize -> complete public chunk_kda -> end event -> end.synchronize -> elapsed_time; synchronization is not a sample value",
        "schedule": "even pair index C1->pinned; odd pair index pinned->C1; route environment selection occurs before each event pair",
        "path_order": {
            "even_pair": list(PATHS),
            "odd_pair": list(reversed(PATHS)),
            "first_path_counts": first_counts,
        },
        "warmup_route_spy_delta": warm_delta,
        "first_warm_c1_decision": first_c1_decision,
        "last_warm_c1_decision": last_c1_decision,
        "timed_route_spy_delta": timed_delta,
        "expected_route_delta_per_call": {
            "public_registry_c1": _expected_route("public_registry_c1"),
            "public_registry_pinned": _expected_route("public_registry_pinned"),
        },
        "input_immutability_exact": immutability.get("input_immutability_exact") is True,
        "input_immutability_fields": immutability.get("fields"),
        "pairs": pairs,
        "blocks": blocks,
        "paths": path_summaries,
        "paired": _pair_summary(pairs),
        "relative_candidate_gate": _relative_candidate_gate(
            path_summaries["public_registry_c1"], path_summaries["public_registry_pinned"]
        ),
        "passed": True,
    }


def _initial_result(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "purpose": "diagnostic-only B300 fp32-both tail-latency study; never a production whitelist or promotion authority",
        "diagnostic_only": True,
        "no_release_authority": True,
        "mode": args.mode,
        "process_index": args.process_index,
        "process": {"pid": os.getpid(), "fresh_python_process_required": True},
        "target": {
            "cell": f"{TARGET_CASE_NAME}/{TARGET_CONTRACT}",
            "offsets": list(TARGET_OFFSETS),
            "temporary_variant": TARGET_VARIANT,
            "production_status": "not_whitelisted; temporary process-local diagnostic target only",
        },
        "pre_registered": {
            "main_processes": 3,
            "main_process_indices": [0, 1, 2],
            "repeats_per_process": REPEATS,
            "cuda_event_samples_per_path_per_repeat": SAMPLES_PER_PATH,
            "warmup_per_path_per_repeat": WARMUP_PER_PATH,
            "paired_block_size": BLOCK_PAIRS,
            "tail_threshold_ms": TAIL_THRESHOLD_MS,
            "balanced_order": "even pair C1 first, odd pair pinned first",
            "stability_rule": {
                "scope": "all six repeats from the three stability_main processes only",
                "tail_stable_if_all": [
                    "C1 p99 <= 1.20 ms",
                    "paired (C1-pinned) p99 <= 0 ms",
                    "count(C1 > pinned and C1 > 1.20 ms) == 0",
                ],
                "otherwise": "tail instability or relative tail regression is observed; this remains diagnostic-only and cannot alter the whitelist",
            },
            "relative_candidate_gate": {
                "scope": "all six repeats from the three stability_main processes only",
                "rule": "C1 must win P50/P95/P99 against pinned in every repeat with each margin >=2%",
                "policy": "reported only; it has no release authority in this diagnostic",
            },
            "telemetry_policy": "The separately-run non_gating_telemetry process is explanatory only and is excluded from every stability classification.",
        },
        "identity": {},
        "gates": {
            "production_map_before": {"passed": False},
            "temporary_target_map": {"passed": False},
            "map_restored": {"passed": False},
            "target_correctness": {"passed": False},
            "route": {"passed": False},
            "prepare_spy_restored": {"passed": False},
            "backend_methods_restored": {"passed": False},
            "clean_gpu": {"passed": False},
            "python_nvidia_clean": {"passed": False},
        },
        "map": {"installed": False, "restored": False},
        "correctness": {},
        "performance": {"repeats": [], "complete": False},
        "complete": False,
    }


def _cpu_construction_check() -> dict[str, object]:
    """Torch CPU-only descriptor construction and frozen-map semantics."""

    production = _map_gate(PRODUCTION_MAP, "cpu-construction-production-map")
    offsets = candidate._cpu_offsets(_target_case().lengths)
    if offsets.device.type != "cpu":
        raise AssertionError("CPU construction check produced a non-CPU descriptor")
    values = offsets.tolist()
    if values != list(TARGET_OFFSETS):
        raise AssertionError(f"CPU target offsets drift: {values}")
    return {
        "cpu_offsets": values,
        "dtype": str(offsets.dtype),
        "device": str(offsets.device),
        "production_map": production,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--mode", choices=("stability_main", "non_gating_telemetry"), required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--describe", action="store_true", help="write pre-registered design without importing Torch/FLA")
    parser.add_argument("--cpu-construction-check", action="store_true", help="check CPU descriptor construction only")
    args = parser.parse_args()
    if args.json.suffix.lower() != ".json":
        raise ValueError("--json must use a .json suffix")
    if args.mode == "stability_main" and args.process_index not in (0, 1, 2):
        raise ValueError("stability_main process index must be 0, 1, or 2")
    if args.mode == "non_gating_telemetry" and args.process_index != 3:
        raise ValueError("non_gating_telemetry process index must be 3")
    target = _target_cell()
    result = _initial_result(args)
    _write(args.json, result)
    if args.describe:
        result["describe_only"] = True
        _write(args.json, result)
        print(f"wrote fp32 tail diagnosis plan {args.json}")
        return
    if args.cpu_construction_check:
        result["cpu_construction_check"] = _cpu_construction_check()
        result["cpu_only"] = True
        _write(args.json, result)
        print(f"wrote fp32 tail diagnosis CPU construction check {args.json}")
        return
    if args.reference_root is None:
        raise ValueError("--reference-root is required for GPU diagnosis")
    if (
        os.environ.get(CLEAN_GPU_GATE_ENV) != "1"
        or os.environ.get("C1_B300_FLASH_KDA") != "1"
        or os.environ.get("C1_B300_VARLEN_CPU_DESCRIPTOR") != "1"
        or os.environ.get("FLA_FLASH_KDA") != "1"
    ):
        raise RuntimeError("clean outer shell plus C1 B300/CPU descriptor/FLA pinned opt-ins are required")
    patched_text, fla_text = os.environ.get("PATCHED_ROOT"), os.environ.get("FLA_ROOT")
    if not patched_text or not fla_text:
        raise RuntimeError("PATCHED_ROOT and FLA_ROOT are required")

    result["identity"] = {"tail_runner": _runner_identity()}
    _write(args.json, result)
    pre_torch_clean = candidate._python_clean_gpu_gate()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shared.torch = torch
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import fla_backend
    from fla.ops.kda import chunk_kda

    shared.common = common
    identity = candidate._identity(Path(patched_text), Path(fla_text), args.reference_root)
    identity["tail_runner"] = result["identity"]["tail_runner"]
    identity["runtime_import_identities"] = candidate._runtime_dependency_identities(args, common, fla_backend)
    identity["python_pre_torch_nvidia_smi"] = pre_torch_clean
    result["identity"] = identity
    result["gates"].update(
        {
            "clean_gpu": {"passed": True},
            "python_nvidia_clean": {"passed": True},
            "device": {"passed": True},
            "extension": {"passed": True},
            "fla_pin": {"passed": True},
        }
    )
    torch_ref, helper = confirmation._load_pinned_reference_without_build(common, args.reference_root)
    result["identity"]["pinned_reference_helper"] = helper
    c1, pinned, _registry, registry_snapshot = candidate._registry_backends()
    originals, counts = candidate._install_spies(c1, pinned)
    result["registry"] = {
        "snapshot": registry_snapshot,
        "c1_id": id(c1),
        "pinned_id": id(pinned),
        "spies": "temporary instance-local public route counters",
    }
    map_object: object | None = None
    primary_error: BaseException | None = None
    try:
        map_object, installed = _install_target_map()
        result["map"]["installation"] = installed
        result["map"]["installed"] = True
        result["gates"]["production_map_before"] = installed["production_before"]
        result["gates"]["temporary_target_map"] = installed["temporary_target"]
        _write(args.json, result)
        with torch.inference_mode():
            if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
                raise RuntimeError("tail diagnosis must run under torch.inference_mode")
            correctness_x = shared._make_inputs(target.case, args.seed + args.process_index * 100_003)
            try:
                correctness_cpu = candidate._cpu_offsets(target.case.lengths)
                correctness_gpu = correctness_x.cu_seqlens
                if correctness_gpu is None:
                    raise AssertionError("target input lost GPU offsets")
                varlen_metadata.clear_cache()
                correctness = candidate._positive_cell(
                    target,
                    correctness_x,
                    correctness_cpu,
                    correctness_gpu,
                    originals,
                    counts,
                    chunk_kda,
                    c1,
                    pinned,
                    torch_ref,
                    args.seed + args.process_index * 100_003,
                )
                if correctness.get("passed") is not True:
                    raise AssertionError("target correctness did not pass")
                result["correctness"] = correctness
                result["gates"]["target_correctness"] = {"passed": True}
                no_shadow = candidate._assert_no_prepare_instance_shadow(c1, "after target correctness")
                result["gates"]["prepare_spy_restored"] = no_shadow
                _write(args.json, result)
            finally:
                del correctness_x
                torch.cuda.empty_cache()

            repeats: list[dict[str, object]] = []
            for repeat_index in range(REPEATS):
                x = shared._make_inputs(
                    target.case,
                    args.seed + args.process_index * 100_003 + (repeat_index + 1) * 1009,
                )
                try:
                    cpu = candidate._cpu_offsets(target.case.lengths)
                    gpu = x.cu_seqlens
                    if gpu is None:
                        raise AssertionError("target performance input lost GPU offsets")
                    initial = candidate._initial_state(TARGET_CONTRACT, target.case.sequences)
                    varlen_metadata.clear_cache()
                    repeat = _timed_repeat(
                        repeat_index=repeat_index,
                        x=x,
                        cpu=cpu,
                        gpu=gpu,
                        initial=initial,
                        public_fn=chunk_kda,
                        counts=counts,
                        target=target,
                    )
                    repeats.append(repeat)
                    result["performance"]["repeats"] = repeats
                    _write(args.json, result)
                finally:
                    del x
                    torch.cuda.empty_cache()
            if len(repeats) != REPEATS or not all(repeat.get("passed") is True for repeat in repeats):
                raise AssertionError("incomplete tail diagnosis repeat evidence")
            result["performance"]["complete"] = True
            result["gates"]["route"] = {"passed": True, "per_path_timed_calls": SAMPLES_PER_PATH * REPEATS}
            _write(args.json, result)
    except BaseException as exc:
        primary_error = exc
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        restore_error: BaseException | None = None
        try:
            if map_object is not None:
                restored = _restore_target_map(map_object)
                result["map"]["restoration"] = restored
                result["map"]["restored"] = True
                result["gates"]["map_restored"] = restored
        except BaseException as exc:
            restore_error = exc
            result["map_restoration_failure"] = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            c1.chunk_kda, pinned.chunk_kda = originals["c1"], originals["pinned"]
            backend_restored = c1.chunk_kda is originals["c1"] and pinned.chunk_kda is originals["pinned"]
            result["gates"]["backend_methods_restored"] = {"passed": backend_restored}
            try:
                result["gates"]["prepare_spy_restored_after_restore"] = candidate._assert_no_prepare_instance_shadow(
                    c1, "after tail diagnosis restoration"
                )
            finally:
                _write(args.json, result)
        if restore_error is not None and primary_error is None:
            raise restore_error
    if result["map"].get("restored") is not True:
        raise RuntimeError("production map was not restored")
    result["complete"] = True
    _write(args.json, result)
    print(
        f"wrote diagnostic-only fp32 tail artifact {args.json}; "
        f"mode={args.mode}, process={args.process_index}, repeats={REPEATS}"
    )


if __name__ == "__main__":
    main()
