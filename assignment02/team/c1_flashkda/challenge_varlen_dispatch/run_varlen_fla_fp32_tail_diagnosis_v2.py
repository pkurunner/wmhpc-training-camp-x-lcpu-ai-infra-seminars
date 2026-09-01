#!/usr/bin/env python3
"""Corrected, diagnostic-only FP32-both public-FLA tail experiment.

The v2 timing boundary intentionally mirrors the public candidate benchmark:
route selection, route-counter snapshots, and CUDA event construction are all
outside the measured interval.  Between ``start.record()/synchronize()`` and
``end.record()/synchronize()`` the runner executes only the public
``candidate._call`` invocation.  This program never changes a production
whitelist: the unapproved target is inserted only into the current Python
process's existing mutable map and is restored before process exit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
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


SCHEMA_VERSION = 2
SAMPLES_PER_PATH = 1000
REPEATS = 2
WARMUP_PER_PATH = 100
BLOCK_PAIRS = 100
TARGET_CASE_NAME = "skew_n6_h12_t12288"
TARGET_CONTRACT = "fp32_both"
TARGET_VARIANT = "vshard4_p2"
PATHS = ("public_registry_c1", "public_registry_pinned")
TAIL_THRESHOLD_MS = 1.20
RUNNER_SHA256_ENV = "C1_VARLEN_FLA_FP32_TAIL_DIAG_V2_RUNNER_SHA256"
CLEAN_GPU_GATE_ENV = "C1_VARLEN_FLA_FP32_TAIL_DIAG_V2_CLEAN_GPU"
TARGET_OFFSETS = (0, 1, 2, 3, 4, 5, 12288)
PRODUCTION_MAP = {
    (TARGET_OFFSETS, "none"): "vshard2_p2",
    (TARGET_OFFSETS, "fp32_final_only"): "vshard2_p2",
}
TEMPORARY_TARGET_MAP = {**PRODUCTION_MAP, (TARGET_OFFSETS, TARGET_CONTRACT): TARGET_VARIANT}
GPU_STATE_FIELDS = (
    "index",
    "uuid",
    "pstate",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw",
    "temperature.gpu",
)
TIMING_CONTRACT = (
    "select_path, route-count snapshots, and CUDA Event construction are outside; "
    "after start.record/start.synchronize and before end.record/end.synchronize, "
    "the only invoked operation is candidate._call(public_fn, x, initial, final, gpu, cpu); "
    "route accounting, decision inspection, elapsed_time, and raw-pair recording are outside"
)


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
    if expected is None or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise RuntimeError(f"{RUNNER_SHA256_ENV} must contain a lowercase SHA256")
    if actual != expected:
        raise RuntimeError(f"v2 runner SHA mismatch: expected={expected}, actual={actual}")
    return {"path": str(path), "sha256": actual, "sha256_gate_pass": True, "environment": RUNNER_SHA256_ENV}


def _map_gate(expected: Mapping[tuple[tuple[int, ...], str], str], label: str) -> dict[str, object]:
    evidence = candidate._assert_map_values_and_behavior(expected, label)
    if evidence.get("passed") is not True:
        raise RuntimeError(f"{label}: map evidence failed")
    return evidence


def _install_target_map() -> tuple[object, dict[str, object]]:
    original = auto_dispatch._VARLEN_PUBLIC_VARIANTS
    if not isinstance(original, dict):
        raise RuntimeError("production _VARLEN_PUBLIC_VARIANTS is not a mutable dict")
    production_before = _map_gate(PRODUCTION_MAP, "production-before-v2-diagnosis")
    # Keep recovery local to installation: if a post-mutation proof fails the
    # caller has not yet received ``original`` and therefore cannot restore it.
    try:
        original.clear()
        original.update(TEMPORARY_TARGET_MAP)
        if auto_dispatch._VARLEN_PUBLIC_VARIANTS is not original:
            raise RuntimeError("temporary map replaced, rather than mutated, the production map object")
        temporary = _map_gate(TEMPORARY_TARGET_MAP, "temporary-v2-target-installed")
    except BaseException:
        if auto_dispatch._VARLEN_PUBLIC_VARIANTS is original:
            original.clear()
            original.update(PRODUCTION_MAP)
        raise
    return original, {
        "production_map_object_id": id(original),
        "production_before": production_before,
        "temporary_target": temporary,
        "temporary_entries": len(original),
        "target_only_extra_entry": True,
        "passed": True,
    }


def _restore_target_map(original: object) -> dict[str, object]:
    restored = candidate._restore_production_map(original, PRODUCTION_MAP)
    if restored.get("passed") is not True or restored.get("map_object_id") != id(original):
        raise RuntimeError("same-object production map restoration failed")
    return restored


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1.0 - (position - lower)) + ordered[upper] * (position - lower)


def _summary(values: list[float], expected_samples: int) -> dict[str, float | int]:
    if len(values) != expected_samples or not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("invalid CUDA-event sample vector")
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _paired_summary(pairs: list[Mapping[str, object]]) -> dict[str, object]:
    c1 = [float(pair["c1_ms"]) for pair in pairs]
    pinned = [float(pair["pinned_ms"]) for pair in pairs]
    delta = [left - right for left, right in zip(c1, pinned, strict=True)]
    return {
        "paired_samples": len(delta),
        "delta_definition": "public_registry_c1_ms - public_registry_pinned_ms for same alternating-order pair",
        "delta_mean_ms": statistics.fmean(delta),
        "delta_p50_ms": _percentile(delta, 0.50),
        "delta_p95_ms": _percentile(delta, 0.95),
        "delta_p99_ms": _percentile(delta, 0.99),
        "c1_gt_pinned_count": sum(left > right for left, right in zip(c1, pinned, strict=True)),
        "c1_gt_threshold_count": sum(value > TAIL_THRESHOLD_MS for value in c1),
        "pinned_gt_threshold_count": sum(value > TAIL_THRESHOLD_MS for value in pinned),
        "c1_gt_pinned_and_gt_threshold_count": sum(
            left > right and left > TAIL_THRESHOLD_MS for left, right in zip(c1, pinned, strict=True)
        ),
    }


def _relative_gate(c1: Mapping[str, float | int], pinned: Mapping[str, float | int]) -> dict[str, object]:
    winners: dict[str, str] = {}
    margins: dict[str, float | None] = {}
    for name in ("p50", "p95", "p99"):
        left, right = float(c1[f"{name}_ms"]), float(pinned[f"{name}_ms"])
        if left < right:
            winners[name], margins[name] = PATHS[0], right / left - 1.0
        else:
            winners[name], margins[name] = PATHS[1], None
    passed = all(winners[name] == PATHS[0] and margins[name] is not None and float(margins[name]) >= 0.02 for name in winners)
    return {
        "definition": "C1 must win P50/P95/P99 against pinned with each margin >=2% in this repeat; diagnostic only",
        "winner_by_percentile": winners,
        "c1_margin_over_pinned_by_percentile": margins,
        "passed": passed,
    }


def _select_path(path: str) -> None:
    if path == PATHS[0]:
        os.environ["C1_B300_FLASH_KDA"] = "1"
    elif path == PATHS[1]:
        os.environ["C1_B300_FLASH_KDA"] = "0"
    else:
        raise ValueError(f"unknown route {path!r}")


def _expected_route(path: str) -> dict[str, int]:
    if path == PATHS[0]:
        return {"c1": 1, "pinned": 0}
    if path == PATHS[1]:
        return {"c1": 0, "pinned": 1}
    raise ValueError(path)


def _gpu_state(stage: str) -> dict[str, object]:
    command = ["nvidia-smi", f"--query-gpu={','.join(GPU_STATE_FIELDS)}", "--format=csv,noheader"]
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{stage}: nvidia-smi failed: {proc.stderr.strip()}")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"{stage}: requires exactly one visible GPU, got {lines!r}")
    values = [value.strip() for value in lines[0].split(",")]
    if len(values) != len(GPU_STATE_FIELDS) or not all(values):
        raise RuntimeError(f"{stage}: malformed GPU-state row: {lines[0]!r}")
    return {
        "stage": stage,
        "query_fields": list(GPU_STATE_FIELDS),
        "values": dict(zip(GPU_STATE_FIELDS, values, strict=True)),
        "single_visible_gpu": True,
        "explanatory_only": True,
    }


def _timed_repeat(
    *, repeat_index: int, x: object, cpu: object, gpu: object, initial: object,
    public_fn: Callable[..., Any], counts: dict[str, int], target: candidate.Cell,
) -> dict[str, object]:
    """One corrected candidate-contract repeat; no route spy is inside events."""
    import torch

    if SAMPLES_PER_PATH % 2 or WARMUP_PER_PATH % 2 or SAMPLES_PER_PATH % BLOCK_PAIRS:
        raise RuntimeError("registered schedule is not balanced")
    final = True
    paths, reversed_paths = PATHS, tuple(reversed(PATHS))

    def order(pair_index: int) -> tuple[str, str]:
        return paths if pair_index % 2 == 0 else reversed_paths

    def warm_call(path: str, label: str) -> object:
        _select_path(path)
        before = dict(counts)
        output = candidate._call(public_fn, x, initial, final, gpu, cpu)
        after = dict(counts)
        delta = {key: after[key] - before[key] for key in before}
        if delta != _expected_route(path):
            raise AssertionError(f"{label}: warm route drift: {delta}")
        if path == PATHS[0]:
            decision = auto_dispatch.get_last_decision()
            if decision.get("chosen_variant") != target.expected_variant:
                raise AssertionError(f"{label}: warm C1 route drift: {decision}")
        return output

    immutable_snapshot = candidate._snapshot_input_tensors(x, gpu, cpu, initial)
    with torch.inference_mode():
        warm_before = dict(counts)
        first_c1_decision: dict[str, object] | None = None
        last_c1_decision: dict[str, object] | None = None
        for warm_index in range(WARMUP_PER_PATH):
            for path in order(warm_index):
                output = warm_call(path, f"v2/repeat{repeat_index}/warm{warm_index}/{path}")
                if path == PATHS[0]:
                    last_c1_decision = auto_dispatch.get_last_decision()
                    if first_c1_decision is None:
                        first_c1_decision = last_c1_decision
                del output
        torch.cuda.synchronize()
        warm_after = dict(counts)
        warm_delta = {key: warm_after[key] - warm_before[key] for key in warm_before}
        if warm_delta != {"c1": WARMUP_PER_PATH, "pinned": WARMUP_PER_PATH}:
            raise AssertionError(f"v2/repeat{repeat_index}: warm route counts drift: {warm_delta}")
        if (
            first_c1_decision is None or last_c1_decision is None
            or first_c1_decision.get("chosen_variant") != target.expected_variant
            or last_c1_decision.get("chosen_variant") != target.expected_variant
            or last_c1_decision.get("canonical_cache_hit") is not True
        ):
            raise AssertionError(f"v2/repeat{repeat_index}: C1 warm decision drift")

        timed_before = dict(counts)
        stream = torch.cuda.current_stream()
        pairs: list[dict[str, object]] = []
        blocks: list[dict[str, object]] = []
        block_start_ns: int | None = None
        for pair_index in range(SAMPLES_PER_PATH):
            if pair_index % BLOCK_PAIRS == 0:
                block_start_ns = time.time_ns()
            measured: dict[str, float] = {}
            first_path, second_path = order(pair_index)
            for path in (first_path, second_path):
                # Everything through construction is outside the elapsed interval.
                _select_path(path)
                sample_before = dict(counts)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                start.synchronize()
                # Do not add a spy/wrapper/logging call here: this sole invoke is v2's contract.
                output = candidate._call(public_fn, x, initial, final, gpu, cpu)
                end.record(stream)
                end.synchronize()
                # Route accounting and result handling intentionally begin only here.
                sample_after = dict(counts)
                delta = {key: sample_after[key] - sample_before[key] for key in sample_before}
                if delta != _expected_route(path):
                    raise AssertionError(f"v2/repeat{repeat_index}/pair{pair_index}/{path}: route drift {delta}")
                if path == PATHS[0]:
                    decision = auto_dispatch.get_last_decision()
                    if decision.get("chosen_variant") != target.expected_variant:
                        raise AssertionError(f"v2/repeat{repeat_index}/pair{pair_index}: C1 decision drift")
                measured[path] = float(start.elapsed_time(end))
                del output, start, end
            pairs.append({
                "pair_index": pair_index,
                "block_index": pair_index // BLOCK_PAIRS,
                "first_path": first_path,
                "c1_ms": measured[PATHS[0]],
                "pinned_ms": measured[PATHS[1]],
            })
            if (pair_index + 1) % BLOCK_PAIRS == 0:
                if block_start_ns is None:
                    raise AssertionError("block timestamp missing")
                blocks.append({
                    "block_index": pair_index // BLOCK_PAIRS,
                    "pair_start": pair_index - BLOCK_PAIRS + 1,
                    "pair_end": pair_index,
                    "pair_count": BLOCK_PAIRS,
                    "epoch_ns_start": block_start_ns,
                    "epoch_ns_end": time.time_ns(),
                })
                block_start_ns = None
        timed_after = dict(counts)

    timed_delta = {key: timed_after[key] - timed_before[key] for key in timed_before}
    if timed_delta != {"c1": SAMPLES_PER_PATH, "pinned": SAMPLES_PER_PATH}:
        raise AssertionError(f"v2/repeat{repeat_index}: timed route counts drift: {timed_delta}")
    immutability = candidate._assert_input_immutability(
        f"v2/repeat{repeat_index}/performance", immutable_snapshot, x, gpu, cpu, initial
    )
    c1_values = [float(pair["c1_ms"]) for pair in pairs]
    pinned_values = [float(pair["pinned_ms"]) for pair in pairs]
    first_counts = {path: sum(pair["first_path"] == path for pair in pairs) for path in PATHS}
    if len(pairs) != SAMPLES_PER_PATH or len(blocks) != SAMPLES_PER_PATH // BLOCK_PAIRS:
        raise AssertionError("incomplete timed coverage")
    if first_counts != {PATHS[0]: SAMPLES_PER_PATH // 2, PATHS[1]: SAMPLES_PER_PATH // 2}:
        raise AssertionError(f"unbalanced pair order: {first_counts}")
    paths_summary = {PATHS[0]: _summary(c1_values, SAMPLES_PER_PATH), PATHS[1]: _summary(pinned_values, SAMPLES_PER_PATH)}
    return {
        "repeat_index": repeat_index,
        "timing_contract": TIMING_CONTRACT,
        "schedule": "even pair index C1->pinned; odd pair index pinned->C1; route selection is outside each event pair",
        "path_order": {"even_pair": list(PATHS), "odd_pair": list(reversed_paths), "first_path_counts": first_counts},
        "warmup_route_spy_delta": warm_delta,
        "first_warm_c1_decision": first_c1_decision,
        "last_warm_c1_decision": last_c1_decision,
        "timed_route_spy_delta": timed_delta,
        "expected_route_delta_per_call": {path: _expected_route(path) for path in PATHS},
        "per_sample_route_spy_assertions": {PATHS[0]: SAMPLES_PER_PATH, PATHS[1]: SAMPLES_PER_PATH, "passed": True},
        "input_immutability_exact": immutability.get("input_immutability_exact") is True,
        "input_immutability_fields": immutability.get("fields"),
        "pairs": pairs,
        "blocks": blocks,
        "paths": paths_summary,
        "paired": _paired_summary(pairs),
        "relative_candidate_gate": _relative_gate(paths_summary[PATHS[0]], paths_summary[PATHS[1]]),
        "passed": True,
    }


def _initial_result(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "corrected diagnostic-only B300 FP32-both tail study; never whitelist promotion authority",
        "diagnostic_only": True,
        "no_release_authority": True,
        "whitelist_action": "unchanged",
        "mode": args.mode,
        "process_index": args.process_index,
        "process": {"pid": os.getpid(), "fresh_python_process_required": True},
        "target": {
            "cell": f"{TARGET_CASE_NAME}/{TARGET_CONTRACT}", "offsets": list(TARGET_OFFSETS),
            "temporary_variant": TARGET_VARIANT, "production_status": "not_whitelisted; temporary process-local target only",
        },
        "pre_registered": {
            "main_processes": 3, "main_process_indices": [0, 1, 2], "repeats_per_process": REPEATS,
            "cuda_event_samples_per_path_per_repeat": SAMPLES_PER_PATH, "warmup_per_path_per_repeat": WARMUP_PER_PATH,
            "paired_block_size": BLOCK_PAIRS, "tail_threshold_ms": TAIL_THRESHOLD_MS,
            "balanced_order": "even pair C1 first, odd pair pinned first", "timing_contract": TIMING_CONTRACT,
            "relative_candidate_gate": "all six repeats: C1 wins P50/P95/P99 and each margin >=2%",
            "corrected_tail_rule": "all six repeats: C1 p99 <=1.20ms, paired delta p99 <=0ms, joint(C1>pinned and C1>1.20ms)=0",
            "telemetry_policy": "per-main-process GPU state and separate telemetry sidecar are explanatory only and excluded from every gate",
            "allocation_policy": "second independent allocation is eligible only after all first-allocation gates pass; any relative margin failure stops",
        },
        "identity": {},
        "gates": {name: {"passed": False} for name in (
            "production_map_before", "temporary_target_map", "map_restored", "target_correctness", "route",
            "prepare_spy_restored", "prepare_spy_restored_after_restore", "backend_methods_restored",
            "clean_gpu", "python_nvidia_clean", "device", "extension", "fla_pin",
        )},
        "map": {"installed": False, "restored": False},
        "correctness": {},
        "performance": {"repeats": [], "complete": False},
        "complete": False,
    }


def _cpu_construction_check() -> dict[str, object]:
    production = _map_gate(PRODUCTION_MAP, "v2-cpu-production-map")
    offsets = candidate._cpu_offsets(_target_case().lengths)
    if str(offsets.device) != "cpu" or offsets.tolist() != list(TARGET_OFFSETS):
        raise AssertionError("CPU target descriptor drift")
    return {"cpu_offsets": offsets.tolist(), "dtype": str(offsets.dtype), "device": str(offsets.device), "production_map": production, "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--mode", choices=("stability_main", "non_gating_telemetry"), required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--cpu-construction-check", action="store_true")
    args = parser.parse_args()
    if args.json.suffix.lower() != ".json":
        raise ValueError("--json must use .json suffix")
    if args.mode == "stability_main" and args.process_index not in (0, 1, 2):
        raise ValueError("stability_main index must be 0, 1, or 2")
    if args.mode == "non_gating_telemetry" and args.process_index != 3:
        raise ValueError("non_gating_telemetry index must be 3")
    _target_cell()
    result = _initial_result(args)
    _write(args.json, result)
    if args.describe:
        result["describe_only"] = True
        _write(args.json, result)
        return
    if args.cpu_construction_check:
        result["cpu_construction_check"] = _cpu_construction_check()
        result["cpu_only"] = True
        _write(args.json, result)
        return
    if args.reference_root is None:
        raise ValueError("--reference-root is required for GPU diagnosis")
    if any(os.environ.get(name) != "1" for name in (CLEAN_GPU_GATE_ENV, "C1_B300_FLASH_KDA", "C1_B300_VARLEN_CPU_DESCRIPTOR", "FLA_FLASH_KDA")):
        raise RuntimeError("clean shell and explicit public/CPU-descriptor/FLA opt-ins are required")
    patched_root, fla_root = os.environ.get("PATCHED_ROOT"), os.environ.get("FLA_ROOT")
    if not patched_root or not fla_root:
        raise RuntimeError("PATCHED_ROOT and FLA_ROOT are required")
    result["identity"] = {"v2_runner": _runner_identity(), "candidate_helper": {"path": str(Path(candidate.__file__).resolve()), "sha256": _sha(Path(candidate.__file__).resolve())}}
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
    identity = candidate._identity(Path(patched_root), Path(fla_root), args.reference_root)
    identity["v2_runner"] = result["identity"]["v2_runner"]
    identity["candidate_helper"] = result["identity"]["candidate_helper"]
    identity["runtime_import_identities"] = candidate._runtime_dependency_identities(args, common, fla_backend)
    identity["python_pre_torch_nvidia_smi"] = pre_torch_clean
    result["identity"] = identity
    result["gates"].update({name: {"passed": True} for name in ("clean_gpu", "python_nvidia_clean", "device", "extension", "fla_pin")})
    torch_ref, helper = confirmation._load_pinned_reference_without_build(common, args.reference_root)
    result["identity"]["pinned_reference_helper"] = helper
    c1, pinned, _registry, registry_snapshot = candidate._registry_backends()
    originals, counts = candidate._install_spies(c1, pinned)
    result["registry"] = {"snapshot": registry_snapshot, "c1_id": id(c1), "pinned_id": id(pinned), "spies": "temporary instance-local public route counters"}
    map_object: object | None = None
    primary_error: BaseException | None = None
    try:
        map_object, installed = _install_target_map()
        result["map"]["installation"] = installed
        result["map"]["installed"] = True
        result["gates"]["production_map_before"] = installed["production_before"]
        result["gates"]["temporary_target_map"] = installed["temporary_target"]
        _write(args.json, result)
        target = _target_cell()
        with torch.inference_mode():
            x_correct = shared._make_inputs(target.case, args.seed + args.process_index * 100_003)
            try:
                cpu_correct, gpu_correct = candidate._cpu_offsets(target.case.lengths), x_correct.cu_seqlens
                if gpu_correct is None:
                    raise AssertionError("correctness input lacks GPU offsets")
                varlen_metadata.clear_cache()
                correctness = candidate._positive_cell(target, x_correct, cpu_correct, gpu_correct, originals, counts, chunk_kda, c1, pinned, torch_ref, args.seed)
                if correctness.get("passed") is not True:
                    raise AssertionError("target correctness failed")
                result["correctness"] = correctness
                result["gates"]["target_correctness"] = {"passed": True}
                result["gates"]["prepare_spy_restored"] = candidate._assert_no_prepare_instance_shadow(c1, "after v2 correctness")
                _write(args.json, result)
            finally:
                del x_correct
                torch.cuda.empty_cache()
            # This non-gating query is deliberately before all timed pairs.
            result["performance"]["gpu_state_before_timing"] = _gpu_state("before_timing")
            if args.mode == "stability_main":
                repeats: list[dict[str, object]] = []
                for repeat_index in range(REPEATS):
                    x = shared._make_inputs(target.case, args.seed + args.process_index * 100_003 + (repeat_index + 1) * 1009)
                    try:
                        cpu, gpu = candidate._cpu_offsets(target.case.lengths), x.cu_seqlens
                        if gpu is None:
                            raise AssertionError("performance input lacks GPU offsets")
                        initial = candidate._initial_state(TARGET_CONTRACT, target.case.sequences)
                        varlen_metadata.clear_cache()
                        repeat = _timed_repeat(repeat_index=repeat_index, x=x, cpu=cpu, gpu=gpu, initial=initial, public_fn=chunk_kda, counts=counts, target=target)
                        repeats.append(repeat)
                        result["performance"]["repeats"] = repeats
                        _write(args.json, result)
                    finally:
                        del x
                        torch.cuda.empty_cache()
                if len(repeats) != REPEATS or not all(repeat.get("passed") is True for repeat in repeats):
                    raise AssertionError("incomplete v2 stability repeats")
                result["gates"]["route"] = {"passed": True, "per_path_timed_calls": SAMPLES_PER_PATH * REPEATS}
            else:
                result["performance"]["explanatory_only"] = True
                result["performance"]["timed_pairs_executed"] = 0
                result["gates"]["route"] = {"passed": True, "not_a_main_timing_process": True}
            # This non-gating query is deliberately after all timed pairs.
            result["performance"]["gpu_state_after_timing"] = _gpu_state("after_timing")
            result["performance"]["complete"] = True
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
            result["gates"]["backend_methods_restored"] = {"passed": c1.chunk_kda is originals["c1"] and pinned.chunk_kda is originals["pinned"]}
            try:
                result["gates"]["prepare_spy_restored_after_restore"] = candidate._assert_no_prepare_instance_shadow(c1, "after v2 restoration")
            finally:
                _write(args.json, result)
        if restore_error is not None and primary_error is None:
            raise restore_error
    if result["map"].get("restored") is not True:
        raise RuntimeError("production map was not restored")
    result["complete"] = True
    _write(args.json, result)
    print(f"wrote corrected diagnostic-only v2 artifact {args.json}; mode={args.mode}, process={args.process_index}")


if __name__ == "__main__":
    main()
