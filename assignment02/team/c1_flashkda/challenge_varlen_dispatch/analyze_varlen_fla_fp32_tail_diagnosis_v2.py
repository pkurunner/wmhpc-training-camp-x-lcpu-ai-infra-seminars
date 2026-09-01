#!/usr/bin/env python3
"""Fail-closed stdlib audit for corrected v2 FP32-both tail measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 2
REPEATS = 2
SAMPLES = 1000
BLOCK_PAIRS = 100
TAIL_THRESHOLD_MS = 1.20
PATHS = ("public_registry_c1", "public_registry_pinned")
TARGET_CELL = "skew_n6_h12_t12288/fp32_both"
TARGET_OFFSETS = [0, 1, 2, 3, 4, 5, 12288]
RUNNER_SHA_ENV = "C1_VARLEN_FLA_FP32_TAIL_DIAG_V2_RUNNER_SHA256"
ANALYZER_SHA_ENV = "C1_VARLEN_FLA_FP32_TAIL_DIAG_V2_ANALYZER_SHA256"
CANDIDATE_SHA_ENV = "C1_VARLEN_FLA_FP32_TAIL_DIAG_V2_CANDIDATE_SHA256"
GPU_STATE_FIELDS = (
    "index", "uuid", "pstate", "clocks.current.graphics", "clocks.current.sm",
    "clocks.current.memory", "power.draw", "temperature.gpu",
)
TIMING_CONTRACT = (
    "select_path, route-count snapshots, and CUDA Event construction are outside; "
    "after start.record/start.synchronize and before end.record/end.synchronize, "
    "the only invoked operation is candidate._call(public_fn, x, initial, final, gpu, cpu); "
    "route accounting, decision inspection, elapsed_time, and raw-pair recording are outside"
)
BALANCED_ORDER = "even pair C1 first, odd pair pinned first"
RELATIVE_POLICY = "all six repeats: C1 wins P50/P95/P99 and each margin >=2%"
CORRECTED_TAIL_POLICY = "all six repeats: C1 p99 <=1.20ms, paired delta p99 <=0ms, joint(C1>pinned and C1>1.20ms)=0"
TELEMETRY_POLICY = "per-main-process GPU state and separate telemetry sidecar are explanatory only and excluded from every gate"
ALLOCATION_POLICY = "second independent allocation is eligible only after all first-allocation gates pass; any relative margin failure stops"
SIDECAR_FIELDS = (
    "timestamp", "index", "uuid", "pstate", "clocks.current.graphics", "clocks.current.sm",
    "clocks.current.memory", "power.draw", "utilization.gpu", "utilization.memory", "temperature.gpu",
)


class AuditError(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def mapping(value: object, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def sequence(value: object, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return value


def integer(value: object, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    return int(value)


def integer_list(value: object, label: str) -> list[int]:
    values = sequence(value, label)
    return [integer(item, f"{label}[{index}]") for index, item in enumerate(values)]


def integer_mapping(value: object, expected: Mapping[str, int], label: str) -> dict[str, int]:
    source = mapping(value, label)
    require(set(source) == set(expected), f"{label}: key drift")
    actual = {key: integer(source.get(key), f"{label}.{key}") for key in expected}
    require(actual == dict(expected), f"{label}: integer values drift")
    return actual


def numeric(value: object, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number


def boolean(value: object, label: str) -> bool:
    require(isinstance(value, bool), f"{label} must be an exact bool")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_argument(value: str, label: str) -> str:
    require(len(value) == 64 and all(char in "0123456789abcdef" for char in value), f"{label} must be lowercase SHA256")
    return value


def read_json(path: Path, expected_sha: str, label: str) -> tuple[Mapping[str, Any], str]:
    expected_sha = sha_argument(expected_sha, f"{label}.expected_sha")
    require(path.is_file(), f"{label}: missing file {path}")
    actual = sha(path)
    require(actual == expected_sha, f"{label}: SHA mismatch expected={expected_sha} actual={actual}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label}: invalid JSON: {exc}") from exc
    return mapping(raw, label), actual


def percentile(values: list[float], quantile: float) -> float:
    require(bool(values), "cannot calculate percentile of empty samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1.0 - (position - lower)) + ordered[upper] * (position - lower)


def summary(values: list[float], expected_samples: int) -> dict[str, float | int]:
    require(len(values) == expected_samples, f"sample count mismatch: expected {expected_samples}, got {len(values)}")
    require(all(math.isfinite(value) and value > 0.0 for value in values), "samples must be finite positive milliseconds")
    return {
        "samples": len(values), "mean_ms": statistics.fmean(values), "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95), "p99_ms": percentile(values, 0.99),
        "min_ms": min(values), "max_ms": max(values),
    }


def paired_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    c1 = [numeric(pair["c1_ms"], "pair.c1_ms") for pair in pairs]
    pinned = [numeric(pair["pinned_ms"], "pair.pinned_ms") for pair in pairs]
    delta = [left - right for left, right in zip(c1, pinned, strict=True)]
    return {
        "paired_samples": len(delta),
        "delta_definition": "public_registry_c1_ms - public_registry_pinned_ms for same alternating-order pair",
        "delta_mean_ms": statistics.fmean(delta), "delta_p50_ms": percentile(delta, 0.50),
        "delta_p95_ms": percentile(delta, 0.95), "delta_p99_ms": percentile(delta, 0.99),
        "c1_gt_pinned_count": sum(left > right for left, right in zip(c1, pinned, strict=True)),
        "c1_gt_threshold_count": sum(value > TAIL_THRESHOLD_MS for value in c1),
        "pinned_gt_threshold_count": sum(value > TAIL_THRESHOLD_MS for value in pinned),
        "c1_gt_pinned_and_gt_threshold_count": sum(
            left > right and left > TAIL_THRESHOLD_MS for left, right in zip(c1, pinned, strict=True)
        ),
    }


def relative_gate(c1: Mapping[str, float | int], pinned: Mapping[str, float | int]) -> dict[str, object]:
    winners: dict[str, str] = {}
    margins: dict[str, float | None] = {}
    for name in ("p50", "p95", "p99"):
        left, right = float(c1[f"{name}_ms"]), float(pinned[f"{name}_ms"])
        if left < right:
            winners[name], margins[name] = PATHS[0], right / left - 1.0
        else:
            winners[name], margins[name] = PATHS[1], None
    return {
        "definition": "C1 must win P50/P95/P99 against pinned with each margin >=2% in this repeat; diagnostic only",
        "winner_by_percentile": winners,
        "c1_margin_over_pinned_by_percentile": margins,
        "passed": all(winners[name] == PATHS[0] and margins[name] is not None and float(margins[name]) >= 0.02 for name in winners),
    }


def close(actual: object, expected: float, label: str) -> None:
    measured = numeric(actual, label)
    require(math.isclose(measured, expected, rel_tol=1e-12, abs_tol=1e-12), f"{label} mismatch: runner={measured}, recomputed={expected}")


def require_summary(recorded: object, computed: Mapping[str, float | int], label: str) -> None:
    source = mapping(recorded, label)
    require(set(source) == set(computed), f"{label}: summary key drift")
    for key, expected in computed.items():
        if isinstance(expected, int):
            require(integer(source.get(key), f"{label}.{key}") == expected, f"{label}.{key} mismatch")
        else:
            close(source.get(key), expected, f"{label}.{key}")


def require_paired(recorded: object, computed: Mapping[str, object], label: str) -> None:
    source = mapping(recorded, label)
    require(set(source) == set(computed), f"{label}: paired key drift")
    for key, expected in computed.items():
        actual = source.get(key)
        if isinstance(expected, str):
            require(actual == expected, f"{label}.{key} mismatch")
        elif isinstance(expected, int):
            require(integer(actual, f"{label}.{key}") == expected, f"{label}.{key} mismatch")
        else:
            close(actual, float(expected), f"{label}.{key}")


def gate_true(value: object, label: str) -> None:
    require(boolean(mapping(value, label).get("passed"), f"{label}.passed") is True, f"{label}: gate failed")


def validate_gpu_state(value: object, label: str) -> dict[str, str]:
    state = mapping(value, label)
    require(state.get("query_fields") == list(GPU_STATE_FIELDS), f"{label}: field contract drift")
    require(boolean(state.get("single_visible_gpu"), f"{label}.single_visible_gpu") is True, f"{label}: not one visible GPU")
    require(boolean(state.get("explanatory_only"), f"{label}.explanatory_only") is True, f"{label}: state must be explanatory")
    values = mapping(state.get("values"), f"{label}.values")
    require(set(values) == set(GPU_STATE_FIELDS), f"{label}: GPU-state value field drift")
    normalized: dict[str, str] = {}
    for field in GPU_STATE_FIELDS:
        raw = values.get(field)
        require(isinstance(raw, str) and raw.strip(), f"{label}.{field}: missing state value")
        normalized[field] = raw.strip()
    require(normalized["index"] == "0", f"{label}: unexpected visible-GPU index")
    return normalized


def validate_sidecar(path: Path, expected_sha: str) -> dict[str, object]:
    """Hash and parse the shell-owned explanatory NVML sidecar.

    It is intentionally excluded from all performance gates, but because the
    shell emits it we still bind its bytes and reject malformed multi-GPU rows.
    """
    expected_sha = sha_argument(expected_sha, "sidecar.expected_sha")
    require(path.is_file(), f"sidecar: missing file {path}")
    actual = sha(path)
    require(actual == expected_sha, "sidecar: SHA mismatch")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].startswith("UNAVAILABLE:"):
        return {"path": str(path), "sha256": actual, "available": False, "reason": lines[0] if lines else "no sidecar rows", "excluded_from_gates": True}
    rows = [row for row in csv.reader(lines) if row]
    require(bool(rows) and all(len(row) == len(SIDECAR_FIELDS) for row in rows), "sidecar: column contract drift")
    samples = [{field: row[index].strip() for index, field in enumerate(SIDECAR_FIELDS)} for row in rows]
    require(all(all(sample[field] for field in SIDECAR_FIELDS) for sample in samples), "sidecar: blank value")
    require({sample["index"] for sample in samples} == {"0"}, "sidecar: not a single visible GPU")
    require(len({sample["uuid"] for sample in samples}) == 1, "sidecar: GPU UUID drift")
    return {
        "path": str(path), "sha256": actual, "available": True, "fields": list(SIDECAR_FIELDS),
        "samples": len(samples), "uuid": samples[0]["uuid"], "excluded_from_gates": True,
    }


def validate_identity(data: Mapping[str, Any], label: str) -> None:
    identity = mapping(data.get("identity"), f"{label}.identity")
    expected_runner = sha_argument(os.environ.get(RUNNER_SHA_ENV, ""), RUNNER_SHA_ENV)
    expected_candidate = sha_argument(os.environ.get(CANDIDATE_SHA_ENV, ""), CANDIDATE_SHA_ENV)
    runner = mapping(identity.get("v2_runner"), f"{label}.identity.v2_runner")
    require(runner.get("sha256") == expected_runner and boolean(runner.get("sha256_gate_pass"), f"{label}.runner.pass") is True, f"{label}: runner SHA identity failed")
    candidate = mapping(identity.get("candidate_helper"), f"{label}.identity.candidate_helper")
    require(candidate.get("sha256") == expected_candidate, f"{label}: candidate source identity failed")
    device = mapping(identity.get("device"), f"{label}.identity.device")
    require(boolean(device.get("passed"), f"{label}.device.passed") is True, f"{label}: device gate failed")
    require(integer_list(device.get("capability"), f"{label}.device.capability") == [10, 3] and integer(device.get("multiprocessor_count"), f"{label}.device.sms") == 148, f"{label}: expected B300 SM103a/148SM")
    extension = mapping(identity.get("extension"), f"{label}.identity.extension")
    require(boolean(extension.get("passed"), f"{label}.extension.passed") is True and isinstance(extension.get("sha256"), str), f"{label}: extension identity failed")
    trees = mapping(identity.get("source_trees"), f"{label}.identity.source_trees")
    for tree_name in ("patched", "reference"):
        require(boolean(mapping(trees.get(tree_name), f"{label}.tree.{tree_name}").get("passed"), f"{label}.tree.{tree_name}.passed") is True, f"{label}: {tree_name} source-tree identity failed")
    fla = mapping(identity.get("fla"), f"{label}.identity.fla")
    require(boolean(fla.get("passed"), f"{label}.fla.passed") is True and boolean(fla.get("tracked_status_clean"), f"{label}.fla.clean") is True, f"{label}: FLA source identity failed")
    public_callables = mapping(fla.get("public_callables"), f"{label}.fla.public_callables")
    public_chunk = mapping(public_callables.get("fla.ops.kda.chunk_kda"), f"{label}.fla.public_chunk")
    require(boolean(public_chunk.get("passed"), f"{label}.fla.public_chunk.passed") is True, f"{label}: public chunk callable identity failed")
    runtime = mapping(identity.get("runtime_import_identities"), f"{label}.runtime_identities")
    require(bool(runtime), f"{label}: empty runtime identities")
    for name, raw in runtime.items():
        entry = mapping(raw, f"{label}.runtime.{name}")
        require(boolean(entry.get("sha256_gate_pass"), f"{label}.runtime.{name}.passed") is True, f"{label}: source gate failed for {name}")


def validate_repeat(raw_value: object, label: str, repeat_index: int) -> dict[str, object]:
    repeat = mapping(raw_value, label)
    require(integer(repeat.get("repeat_index"), f"{label}.repeat_index") == repeat_index, f"{label}: repeat index drift")
    require(repeat.get("timing_contract") == TIMING_CONTRACT, f"{label}: corrected timing contract missing")
    require(boolean(repeat.get("passed"), f"{label}.passed") is True, f"{label}: runner marked failed")
    require(boolean(repeat.get("input_immutability_exact"), f"{label}.immutability") is True, f"{label}: input mutation")
    integer_mapping(repeat.get("warmup_route_spy_delta"), {"c1": 100, "pinned": 100}, f"{label}.warm")
    integer_mapping(repeat.get("timed_route_spy_delta"), {"c1": SAMPLES, "pinned": SAMPLES}, f"{label}.timed")
    per_sample = mapping(repeat.get("per_sample_route_spy_assertions"), f"{label}.per_sample")
    require(set(per_sample) == {PATHS[0], PATHS[1], "passed"}, f"{label}: per-sample route assertion keys drift")
    require(integer(per_sample.get(PATHS[0]), f"{label}.per_sample.c1") == SAMPLES and integer(per_sample.get(PATHS[1]), f"{label}.per_sample.pinned") == SAMPLES and boolean(per_sample.get("passed"), f"{label}.per_sample.passed") is True, f"{label}: per-sample route assertion drift")
    order = mapping(repeat.get("path_order"), f"{label}.order")
    require(order.get("even_pair") == list(PATHS) and order.get("odd_pair") == list(reversed(PATHS)), f"{label}: parity order drift")
    integer_mapping(order.get("first_path_counts"), {PATHS[0]: 500, PATHS[1]: 500}, f"{label}.first_counts")
    for decision_name in ("first_warm_c1_decision", "last_warm_c1_decision"):
        decision = mapping(repeat.get(decision_name), f"{label}.{decision_name}")
        require(decision.get("chosen_variant") == "vshard4_p2", f"{label}: warm C1 variant drift")
    require(mapping(repeat.get("last_warm_c1_decision"), f"{label}.last_warm").get("canonical_cache_hit") is True, f"{label}: warm C1 is not cache-hot")

    pairs_raw = sequence(repeat.get("pairs"), f"{label}.pairs")
    require(len(pairs_raw) == SAMPLES, f"{label}: pair count drift")
    pairs: list[Mapping[str, Any]] = []
    for pair_index, raw_pair in enumerate(pairs_raw):
        pair = mapping(raw_pair, f"{label}.pair[{pair_index}]")
        require(integer(pair.get("pair_index"), f"{label}.pair.index") == pair_index, f"{label}: pair index drift")
        require(integer(pair.get("block_index"), f"{label}.pair.block") == pair_index // BLOCK_PAIRS, f"{label}: pair block drift")
        require(pair.get("first_path") == (PATHS[0] if pair_index % 2 == 0 else PATHS[1]), f"{label}: pair alternation drift")
        require(numeric(pair.get("c1_ms"), f"{label}.pair.c1") > 0.0 and numeric(pair.get("pinned_ms"), f"{label}.pair.pinned") > 0.0, f"{label}: invalid sample")
        pairs.append(pair)
    blocks_raw = sequence(repeat.get("blocks"), f"{label}.blocks")
    require(len(blocks_raw) == SAMPLES // BLOCK_PAIRS, f"{label}: block count drift")
    prior_end = -1
    blocks: list[dict[str, object]] = []
    for block_index, raw_block in enumerate(blocks_raw):
        block = mapping(raw_block, f"{label}.block[{block_index}]")
        start, end = integer(block.get("epoch_ns_start"), f"{label}.block.start"), integer(block.get("epoch_ns_end"), f"{label}.block.end")
        require(integer(block.get("block_index"), f"{label}.block.index") == block_index, f"{label}: block index drift")
        require(integer(block.get("pair_start"), f"{label}.block.pair_start") == block_index * BLOCK_PAIRS, f"{label}: block pair start drift")
        require(integer(block.get("pair_end"), f"{label}.block.pair_end") == (block_index + 1) * BLOCK_PAIRS - 1, f"{label}: block pair end drift")
        require(integer(block.get("pair_count"), f"{label}.block.count") == BLOCK_PAIRS and prior_end <= start <= end, f"{label}: invalid epoch block")
        prior_end = end
        block_pairs = pairs[block_index * BLOCK_PAIRS : (block_index + 1) * BLOCK_PAIRS]
        blocks.append({
            "block_index": block_index, "epoch_ns_start": start, "epoch_ns_end": end,
            "duration_ms_wall": (end - start) / 1_000_000.0,
            "c1": summary([numeric(pair["c1_ms"], "block.c1") for pair in block_pairs], BLOCK_PAIRS),
            "pinned": summary([numeric(pair["pinned_ms"], "block.pinned") for pair in block_pairs], BLOCK_PAIRS),
            "paired": paired_summary(block_pairs),
        })
    paths = {PATHS[0]: summary([numeric(pair["c1_ms"], "c1") for pair in pairs], SAMPLES), PATHS[1]: summary([numeric(pair["pinned_ms"], "pinned") for pair in pairs], SAMPLES)}
    recorded_paths = mapping(repeat.get("paths"), f"{label}.paths")
    require(set(recorded_paths) == set(PATHS), f"{label}: path set drift")
    for path in PATHS:
        require_summary(recorded_paths.get(path), paths[path], f"{label}.paths.{path}")
    paired = paired_summary(pairs)
    require_paired(repeat.get("paired"), paired, f"{label}.paired")
    relative = relative_gate(paths[PATHS[0]], paths[PATHS[1]])
    recorded_relative = mapping(repeat.get("relative_candidate_gate"), f"{label}.relative")
    require(recorded_relative.get("definition") == relative["definition"], f"{label}: relative definition drift")
    require(mapping(recorded_relative.get("winner_by_percentile"), f"{label}.relative.winners") == relative["winner_by_percentile"], f"{label}: relative winners drift")
    margins = mapping(recorded_relative.get("c1_margin_over_pinned_by_percentile"), f"{label}.relative.margins")
    for name, expected in mapping(relative["c1_margin_over_pinned_by_percentile"], "computed.margins").items():
        if expected is None:
            require(margins.get(name) is None, f"{label}: null margin drift")
        else:
            close(margins.get(name), float(expected), f"{label}.relative.margin.{name}")
    require(boolean(recorded_relative.get("passed"), f"{label}.relative.passed") is boolean(relative["passed"], "computed.relative.passed"), f"{label}: relative pass drift")
    return {"repeat_index": repeat_index, "pairs": pairs, "blocks": blocks, "paths": paths, "paired": paired, "relative_candidate_gate": relative}


def validate_artifact(data: Mapping[str, Any], label: str, expected_mode: str, expected_index: int) -> dict[str, object]:
    require(integer(data.get("schema_version"), f"{label}.schema") == SCHEMA_VERSION, f"{label}: schema drift")
    require(boolean(data.get("diagnostic_only"), f"{label}.diagnostic") is True and boolean(data.get("no_release_authority"), f"{label}.no_release") is True, f"{label}: diagnostic policy drift")
    require(data.get("whitelist_action") == "unchanged" and data.get("mode") == expected_mode, f"{label}: mode/policy drift")
    require(integer(data.get("process_index"), f"{label}.process_index") == expected_index, f"{label}: process index drift")
    require(boolean(data.get("complete"), f"{label}.complete") is True and "failure" not in data and "map_restoration_failure" not in data, f"{label}: incomplete/failure artifact")
    process = mapping(data.get("process"), f"{label}.process")
    pid = integer(process.get("pid"), f"{label}.pid")
    require(pid > 1 and boolean(process.get("fresh_python_process_required"), f"{label}.fresh") is True, f"{label}: fresh PID assertion missing")
    target = mapping(data.get("target"), f"{label}.target")
    require(target.get("cell") == TARGET_CELL and integer_list(target.get("offsets"), f"{label}.target.offsets") == TARGET_OFFSETS and target.get("temporary_variant") == "vshard4_p2", f"{label}: target drift")
    prereg = mapping(data.get("pre_registered"), f"{label}.prereg")
    require(integer(prereg.get("main_processes"), f"{label}.main_count") == 3 and integer_list(prereg.get("main_process_indices"), f"{label}.main_indices") == [0, 1, 2], f"{label}: main prereg drift")
    require(integer(prereg.get("repeats_per_process"), f"{label}.repeats") == REPEATS and integer(prereg.get("cuda_event_samples_per_path_per_repeat"), f"{label}.samples") == SAMPLES and integer(prereg.get("warmup_per_path_per_repeat"), f"{label}.warmups") == 100 and integer(prereg.get("paired_block_size"), f"{label}.blocks") == BLOCK_PAIRS, f"{label}: schedule prereg drift")
    require(
        numeric(prereg.get("tail_threshold_ms"), f"{label}.threshold") == TAIL_THRESHOLD_MS
        and prereg.get("timing_contract") == TIMING_CONTRACT
        and prereg.get("balanced_order") == BALANCED_ORDER
        and prereg.get("relative_candidate_gate") == RELATIVE_POLICY
        and prereg.get("corrected_tail_rule") == CORRECTED_TAIL_POLICY
        and prereg.get("telemetry_policy") == TELEMETRY_POLICY
        and prereg.get("allocation_policy") == ALLOCATION_POLICY,
        f"{label}: tail/timing/policy prereg drift",
    )
    validate_identity(data, label)
    gates = mapping(data.get("gates"), f"{label}.gates")
    for name in ("production_map_before", "temporary_target_map", "map_restored", "target_correctness", "route", "prepare_spy_restored", "prepare_spy_restored_after_restore", "backend_methods_restored", "clean_gpu", "python_nvidia_clean", "device", "extension", "fla_pin"):
        gate_true(gates.get(name), f"{label}.gate.{name}")
    map_data = mapping(data.get("map"), f"{label}.map")
    require(boolean(map_data.get("installed"), f"{label}.map.installed") is True and boolean(map_data.get("restored"), f"{label}.map.restored") is True, f"{label}: map lifecycle failed")
    install, restore = mapping(map_data.get("installation"), f"{label}.map.install"), mapping(map_data.get("restoration"), f"{label}.map.restore")
    require(integer(install.get("production_map_object_id"), f"{label}.map.install.id") == integer(restore.get("map_object_id"), f"{label}.map.restore.id") and integer(restore.get("entries"), f"{label}.map.restore.entries") == 2, f"{label}: map same-object restore failed")
    correctness = mapping(data.get("correctness"), f"{label}.correctness")
    require(boolean(correctness.get("passed"), f"{label}.correctness.passed") is True and correctness.get("expected_variant") == "vshard4_p2", f"{label}: target correctness failed")
    performance = mapping(data.get("performance"), f"{label}.performance")
    require(boolean(performance.get("complete"), f"{label}.performance.complete") is True, f"{label}: incomplete performance")
    pre_state, post_state = validate_gpu_state(performance.get("gpu_state_before_timing"), f"{label}.gpu.pre"), validate_gpu_state(performance.get("gpu_state_after_timing"), f"{label}.gpu.post")
    require(pre_state["uuid"] == post_state["uuid"], f"{label}: GPU UUID changed within PID")
    if expected_mode == "non_gating_telemetry":
        require(boolean(performance.get("explanatory_only"), f"{label}.telemetry.explanatory") is True and integer(performance.get("timed_pairs_executed"), f"{label}.telemetry.timed_pairs") == 0, f"{label}: telemetry ran main timed pairs")
        return {"pid": pid, "gpu_pre": pre_state, "gpu_post": post_state}
    repeats_raw = sequence(performance.get("repeats"), f"{label}.repeats")
    require(len(repeats_raw) == REPEATS, f"{label}: repeat count drift")
    repeats = [validate_repeat(repeats_raw[index], f"{label}.repeat{index}", index) for index in range(REPEATS)]
    return {"pid": pid, "process_index": expected_index, "repeats": repeats, "gpu_pre": pre_state, "gpu_post": post_state}


def aggregate_main(processes: Sequence[Mapping[str, object]]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    all_pairs: list[Mapping[str, Any]] = []
    parity: dict[str, list[Mapping[str, Any]]] = {PATHS[0]: [], PATHS[1]: []}
    absolute_failures: list[dict[str, object]] = []
    relative_failures: list[dict[str, object]] = []
    per_pid_positive: dict[str, set[int]] = {}
    per_repeat_positive: dict[int, dict[str, set[int]]] = {}
    pair223: list[dict[str, object]] = []
    for process in processes:
        pid, index = integer(process["pid"], "aggregate.pid"), integer(process["process_index"], "aggregate.index")
        positive: set[int] = set()
        for repeat in process["repeats"]:  # type: ignore[index]
            item = mapping(repeat, "aggregate.repeat")
            paths, paired, relative = mapping(item["paths"], "aggregate.paths"), mapping(item["paired"], "aggregate.paired"), mapping(item["relative_candidate_gate"], "aggregate.relative")
            c1_p99 = numeric(mapping(paths[PATHS[0]], "aggregate.c1").get("p99_ms"), "aggregate.c1.p99")
            delta_p99 = numeric(paired.get("delta_p99_ms"), "aggregate.delta.p99")
            joint = integer(paired.get("c1_gt_pinned_and_gt_threshold_count"), "aggregate.joint")
            absolute = {"c1_p99_le_1_20": c1_p99 <= TAIL_THRESHOLD_MS, "paired_delta_p99_le_0": delta_p99 <= 0.0, "joint_tail_excess_zero": joint == 0}
            if not all(absolute.values()):
                absolute_failures.append({"process_index": index, "repeat_index": item["repeat_index"], "criteria": absolute})
            if boolean(relative.get("passed"), "aggregate.relative.passed") is not True:
                relative_failures.append({"process_index": index, "repeat_index": item["repeat_index"], "relative_candidate_gate": relative})
            pairs = [mapping(pair, "aggregate.pair") for pair in item["pairs"]]  # type: ignore[index]
            repeat_positive: set[int] = set()
            for pair in pairs:
                pair_index = integer(pair["pair_index"], "aggregate.pair.index")
                if numeric(pair["c1_ms"], "aggregate.pair.c1") > numeric(pair["pinned_ms"], "aggregate.pair.pinned"):
                    positive.add(pair_index)
                    repeat_positive.add(pair_index)
                if pair_index == 223:
                    pair223.append({"pid": pid, "process_index": index, "repeat_index": item["repeat_index"], "c1_ms": pair["c1_ms"], "pinned_ms": pair["pinned_ms"], "positive_delta": numeric(pair["c1_ms"], "aggregate.223.c1") > numeric(pair["pinned_ms"], "aggregate.223.pinned")})
            all_pairs.extend(pairs)
            for first in PATHS:
                parity[first].extend(pair for pair in pairs if pair["first_path"] == first)
            records.append({"process_index": index, "repeat_index": item["repeat_index"], "c1": paths[PATHS[0]], "pinned": paths[PATHS[1]], "paired": paired, "relative_candidate_gate": relative, "absolute_criteria": absolute, "blocks": item["blocks"]})
            repeat_index = integer(item["repeat_index"], "aggregate.repeat_index")
            per_repeat_positive.setdefault(repeat_index, {})[str(pid)] = repeat_positive
        per_pid_positive[str(pid)] = positive
    require(len(records) == 6, "requires exactly six main repeats")
    c1_all = [numeric(pair["c1_ms"], "all.c1") for pair in all_pairs]
    pinned_all = [numeric(pair["pinned_ms"], "all.pinned") for pair in all_pairs]
    require(set(per_repeat_positive) == {0, 1} and all(len(per_repeat_positive[index]) == 3 for index in (0, 1)), "cross-PID repeat coverage drift")
    shared_positive_by_repeat = {
        str(repeat_index): sorted(set.intersection(*per_repeat_positive[repeat_index].values()))
        for repeat_index in (0, 1)
    }
    require(len(pair223) == 6, "pair223 must have one observation for every main PID/repeat")
    if relative_failures:
        classification = "relative_regression"
    elif absolute_failures:
        classification = "absolute_or_shared_scale_failure"
    else:
        classification = "all_pass"
    return {
        "scope": "three stability_main PIDs only; telemetry excluded from every gate",
        "repeats": records,
        "all_samples": {"c1": summary(c1_all, 6000), "pinned": summary(pinned_all, 6000), "paired": paired_summary(all_pairs)},
        "parity_by_first_path": {first: {"c1": summary([numeric(pair["c1_ms"], "parity.c1") for pair in pairs], 3000), "pinned": summary([numeric(pair["pinned_ms"], "parity.pinned") for pair in pairs], 3000), "paired": paired_summary(pairs)} for first, pairs in parity.items()},
        "relative_candidate_gate": {"passed": not relative_failures, "failed_repeats": relative_failures},
        "corrected_tail_rule": {"passed": not absolute_failures, "failed_repeats": absolute_failures},
        "classification": classification,
        "cross_pid_positive_delta_indices": {
            "per_pid_any_repeat": {pid: sorted(indices) for pid, indices in per_pid_positive.items()},
            "shared_by_all_three_pids_by_repeat": shared_positive_by_repeat,
            "pair223_observations": pair223,
            "interpretation": "each shared set is intersected within the same repeat; repeated indices are explanatory only and never a release signal",
        },
    }


def require_analyzer_identity() -> dict[str, object]:
    path = Path(__file__).resolve(strict=True)
    actual, expected = sha(path), sha_argument(os.environ.get(ANALYZER_SHA_ENV, ""), ANALYZER_SHA_ENV)
    require(actual == expected, "independent analyzer source SHA mismatch")
    return {"path": str(path), "sha256": actual, "sha256_gate_pass": True}


def atomic_write(path: Path, payload: Mapping[str, object]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_json", nargs=3, type=Path)
    parser.add_argument("--expected-main-sha256", nargs=3, required=True)
    parser.add_argument("--telemetry-json", type=Path, required=True)
    parser.add_argument("--expected-telemetry-json-sha256", required=True)
    parser.add_argument("--telemetry-csv", type=Path, required=True)
    parser.add_argument("--expected-telemetry-csv-sha256", required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    require(args.json.suffix.lower() == ".json", "--json must use .json suffix")
    input_paths = [*args.main_json, args.telemetry_json, args.telemetry_csv]
    resolved_inputs = [path.resolve() for path in input_paths]
    require(len(set(resolved_inputs)) == len(resolved_inputs), "all four JSON inputs and sidecar must be distinct")
    require(args.json.resolve() not in set(resolved_inputs), "--json must differ from every input path")
    analyzer = require_analyzer_identity()
    main_inputs = [read_json(path, digest, f"main[{index}]") for index, (path, digest) in enumerate(zip(args.main_json, args.expected_main_sha256, strict=True))]
    main = [validate_artifact(data, f"main[{index}]", "stability_main", index) for index, (data, _digest) in enumerate(main_inputs)]
    pids = [integer(item["pid"], "main.pid") for item in main]
    require(len(set(pids)) == 3, f"main PIDs are not fresh/distinct: {pids}")
    telemetry_data, telemetry_digest = read_json(args.telemetry_json, args.expected_telemetry_json_sha256, "telemetry")
    telemetry = validate_artifact(telemetry_data, "telemetry", "non_gating_telemetry", 3)
    require(integer(telemetry["pid"], "telemetry.pid") not in set(pids), "telemetry PID is not distinct from main PIDs")
    sidecar = validate_sidecar(args.telemetry_csv, args.expected_telemetry_csv_sha256)
    all_gpu_uuids = {
        str(item["gpu_pre"]["uuid"]) for item in main
    } | {
        str(item["gpu_post"]["uuid"]) for item in main
    } | {str(telemetry["gpu_pre"]["uuid"]), str(telemetry["gpu_post"]["uuid"])}
    require(len(all_gpu_uuids) == 1, f"GPU UUID differs across main/telemetry PIDs: {sorted(all_gpu_uuids)}")
    if sidecar.get("available") is True:
        require(sidecar.get("uuid") in all_gpu_uuids, "sidecar UUID differs from runner process GPU UUID")
    aggregate = aggregate_main(main)
    aggregate_relative = mapping(aggregate.get("relative_candidate_gate"), "aggregate.relative_gate")
    aggregate_tail = mapping(aggregate.get("corrected_tail_rule"), "aggregate.corrected_tail_rule")
    relative_passed = boolean(aggregate_relative.get("passed"), "aggregate.relative_gate.passed")
    corrected_tail_passed = boolean(aggregate_tail.get("passed"), "aggregate.corrected_tail_rule.passed")
    classification = aggregate.get("classification")
    require(isinstance(classification, str), "aggregate.classification must be a string")
    eligible = relative_passed and corrected_tail_passed and classification == "all_pass"
    failed_gates: list[str] = []
    if not relative_passed:
        failed_gates.append("relative_candidate_gate")
    if not corrected_tail_passed:
        failed_gates.append("corrected_tail_rule")
    if classification != "all_pass":
        failed_gates.append("classification_not_all_pass")
    second_allocation_decision = {
        "eligible": eligible,
        "criteria": {
            "relative_candidate_gate_passed": relative_passed,
            "corrected_tail_rule_passed": corrected_tail_passed,
            "classification": classification,
            "required_classification": "all_pass",
        },
        "failed_gates": failed_gates,
        "reason": "all first-allocation gates passed" if eligible else "first allocation does not satisfy the pre-registered second-allocation criteria",
        "diagnostic_only": True,
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "independent corrected-v2 diagnostic audit; no promotion authority",
        "diagnostic_only": True,
        "no_release_authority": True,
        "whitelist_action": "unchanged",
        "promotion_decision": "do_not_change",
        "complete": True,
        "analyzer": analyzer,
        "inputs": {"main": [{"path": str(path), "sha256": digest} for path, (_data, digest) in zip(args.main_json, main_inputs, strict=True)], "telemetry": {"path": str(args.telemetry_json), "sha256": telemetry_digest}},
        "main_gpu_state": [{"pid": item["pid"], "before": item["gpu_pre"], "after": item["gpu_post"]} for item in main],
        "telemetry_gpu_state": {"pid": telemetry["pid"], "before": telemetry["gpu_pre"], "after": telemetry["gpu_post"], "excluded_from_gates": True},
        "explanatory_sidecar": sidecar,
        "main_stability": aggregate,
        "second_allocation_decision": second_allocation_decision,
        "allocation_policy": ALLOCATION_POLICY,
    }
    atomic_write(args.json, output)
    print(f"wrote corrected v2 independent audit {args.json}; classification={aggregate['classification']}")


if __name__ == "__main__":
    try:
        main()
    except AuditError as exc:
        print(f"AUDIT_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
