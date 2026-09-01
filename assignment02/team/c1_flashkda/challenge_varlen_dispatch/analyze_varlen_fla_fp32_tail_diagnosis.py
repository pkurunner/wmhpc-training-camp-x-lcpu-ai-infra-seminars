#!/usr/bin/env python3
"""Independent stdlib-only audit for diagnostic fp32-both tail measurements.

The three positional artifacts must be the independent ``stability_main``
processes.  A fourth ``non_gating_telemetry`` artifact and its NVML CSV are
accepted only as explanatory context; the stability calculation never reads
their samples.
"""

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


SCHEMA_VERSION = 1
REPEATS = 2
SAMPLES = 1000
BLOCK_PAIRS = 100
TAIL_THRESHOLD_MS = 1.20
TARGET_CELL = "skew_n6_h12_t12288/fp32_both"
PATHS = ("public_registry_c1", "public_registry_pinned")
RUNNER_SHA_ENV = "C1_VARLEN_FLA_FP32_TAIL_DIAG_RUNNER_SHA256"
ANALYZER_SHA_ENV = "C1_VARLEN_FLA_FP32_TAIL_DIAG_ANALYZER_SHA256"
TELEMETRY_FIELDS = (
    "timestamp",
    "index",
    "uuid",
    "pstate",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
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


def numeric(value: object, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path, expected_sha: str, label: str) -> tuple[Mapping[str, Any], str]:
    require(path.is_file(), f"{label}: missing input {path}")
    require(len(expected_sha) == 64 and all(character in "0123456789abcdef" for character in expected_sha), f"{label}: expected SHA must be lowercase SHA256")
    actual_sha = sha(path)
    require(actual_sha == expected_sha, f"{label}: SHA mismatch expected={expected_sha} actual={actual_sha}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label}: invalid JSON: {exc}") from exc
    return mapping(raw, label), actual_sha


def percentile(values: list[float], quantile: float) -> float:
    require(bool(values), "cannot calculate percentile of empty samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summary(values: list[float], expected_samples: int) -> dict[str, float | int]:
    require(len(values) == expected_samples, f"sample count mismatch: expected {expected_samples}, got {len(values)}")
    require(all(math.isfinite(value) and value > 0.0 for value in values), "samples must be finite positive milliseconds")
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def paired_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    c1 = [numeric(pair["c1_ms"], "pair.c1_ms") for pair in pairs]
    pinned = [numeric(pair["pinned_ms"], "pair.pinned_ms") for pair in pairs]
    deltas = [c1_value - pinned_value for c1_value, pinned_value in zip(c1, pinned, strict=True)]
    return {
        "paired_samples": len(deltas),
        "delta_definition": "public_registry_c1_ms - public_registry_pinned_ms for the same alternating-order pair",
        "delta_mean_ms": statistics.fmean(deltas),
        "delta_p50_ms": percentile(deltas, 0.50),
        "delta_p95_ms": percentile(deltas, 0.95),
        "delta_p99_ms": percentile(deltas, 0.99),
        "c1_gt_pinned_count": sum(c1_value > pinned_value for c1_value, pinned_value in zip(c1, pinned, strict=True)),
        "c1_gt_threshold_count": sum(value > TAIL_THRESHOLD_MS for value in c1),
        "pinned_gt_threshold_count": sum(value > TAIL_THRESHOLD_MS for value in pinned),
        "c1_gt_pinned_and_gt_threshold_count": sum(
            c1_value > pinned_value and c1_value > TAIL_THRESHOLD_MS
            for c1_value, pinned_value in zip(c1, pinned, strict=True)
        ),
    }


def relative_candidate_gate(c1: Mapping[str, float | int], pinned: Mapping[str, float | int]) -> dict[str, object]:
    winners: dict[str, str] = {}
    margins: dict[str, float | None] = {}
    for percentile_name in ("p50", "p95", "p99"):
        metric = f"{percentile_name}_ms"
        c1_latency = float(c1[metric])
        pinned_latency = float(pinned[metric])
        if c1_latency < pinned_latency:
            winners[percentile_name] = PATHS[0]
            margins[percentile_name] = pinned_latency / c1_latency - 1.0
        else:
            winners[percentile_name] = PATHS[1]
            margins[percentile_name] = None
    return {
        "definition": "C1 must win P50/P95/P99 against pinned with each margin >=2% in this repeat; diagnostic only",
        "winner_by_percentile": winners,
        "c1_margin_over_pinned_by_percentile": margins,
        "passed": all(
            winners[name] == PATHS[0] and margins[name] is not None and float(margins[name]) >= 0.02
            for name in winners
        ),
    }


def close(actual: object, expected: float, label: str) -> None:
    measured = numeric(actual, label)
    require(math.isclose(measured, expected, rel_tol=1e-12, abs_tol=1e-12), f"{label}: runner={measured}, recomputed={expected}")


def require_summary(recorded_value: object, recomputed: Mapping[str, float | int], label: str) -> None:
    recorded = mapping(recorded_value, label)
    require(set(recorded) == set(recomputed), f"{label}: summary keys drift")
    for key, expected in recomputed.items():
        if isinstance(expected, int):
            require(integer(recorded.get(key), f"{label}.{key}") == expected, f"{label}.{key} mismatch")
        else:
            close(recorded.get(key), expected, f"{label}.{key}")


def require_paired_summary(recorded_value: object, recomputed: Mapping[str, object], label: str) -> None:
    recorded = mapping(recorded_value, label)
    require(set(recorded) == set(recomputed), f"{label}: paired summary keys drift")
    for key, expected in recomputed.items():
        actual = recorded.get(key)
        if isinstance(expected, str):
            require(actual == expected, f"{label}.{key} mismatch")
        elif isinstance(expected, int):
            require(integer(actual, f"{label}.{key}") == expected, f"{label}.{key} mismatch")
        else:
            close(actual, float(expected), f"{label}.{key}")


def gate_passed(value: object, label: str) -> None:
    gate = mapping(value, label)
    require(gate.get("passed") is True, f"{label}: passed must be true")


def validate_repeat(value: object, label: str, repeat_index: int) -> dict[str, object]:
    repeat = mapping(value, label)
    require(integer(repeat.get("repeat_index"), f"{label}.repeat_index") == repeat_index, f"{label}: index drift")
    require(repeat.get("input_immutability_exact") is True, f"{label}: input immutability missing")
    require(repeat.get("passed") is True, f"{label}: runner marked repeat failed")
    require(mapping(repeat.get("warmup_route_spy_delta"), f"{label}.warm") == {"c1": 100, "pinned": 100}, f"{label}: warm route drift")
    require(mapping(repeat.get("timed_route_spy_delta"), f"{label}.timed") == {"c1": SAMPLES, "pinned": SAMPLES}, f"{label}: timed route drift")
    order = mapping(repeat.get("path_order"), f"{label}.path_order")
    require(order.get("even_pair") == list(PATHS), f"{label}: even order drift")
    require(order.get("odd_pair") == list(reversed(PATHS)), f"{label}: odd order drift")
    require(mapping(order.get("first_path_counts"), f"{label}.first_counts") == {PATHS[0]: SAMPLES // 2, PATHS[1]: SAMPLES // 2}, f"{label}: unbalanced order")
    for decision_label in ("first_warm_c1_decision", "last_warm_c1_decision"):
        decision = mapping(repeat.get(decision_label), f"{label}.{decision_label}")
        require(decision.get("chosen_variant") == "vshard4_p2", f"{label}.{decision_label}: target variant drift")
    final_warm = mapping(repeat.get("last_warm_c1_decision"), f"{label}.last_warm_c1_decision")
    require(final_warm.get("canonical_cache_hit") is True, f"{label}: warm target did not become cache-hot")

    raw_pairs = sequence(repeat.get("pairs"), f"{label}.pairs")
    require(len(raw_pairs) == SAMPLES, f"{label}: expected {SAMPLES} pairs")
    pairs: list[Mapping[str, Any]] = []
    for pair_index, raw_pair in enumerate(raw_pairs):
        pair = mapping(raw_pair, f"{label}.pairs[{pair_index}]")
        require(integer(pair.get("pair_index"), f"{label}.pairs[{pair_index}].pair_index") == pair_index, f"{label}: pair index drift")
        require(integer(pair.get("block_index"), f"{label}.pairs[{pair_index}].block_index") == pair_index // BLOCK_PAIRS, f"{label}: pair block drift")
        expected_first = PATHS[0] if pair_index % 2 == 0 else PATHS[1]
        require(pair.get("first_path") == expected_first, f"{label}: alternating order drift at pair {pair_index}")
        require(numeric(pair.get("c1_ms"), f"{label}.pairs[{pair_index}].c1_ms") > 0.0, f"{label}: nonpositive C1 sample")
        require(numeric(pair.get("pinned_ms"), f"{label}.pairs[{pair_index}].pinned_ms") > 0.0, f"{label}: nonpositive pinned sample")
        pairs.append(pair)

    raw_blocks = sequence(repeat.get("blocks"), f"{label}.blocks")
    require(len(raw_blocks) == SAMPLES // BLOCK_PAIRS, f"{label}: block count drift")
    prior_end = -1
    blocks: list[dict[str, object]] = []
    for block_index, raw_block in enumerate(raw_blocks):
        block = mapping(raw_block, f"{label}.blocks[{block_index}]")
        start_ns = integer(block.get("epoch_ns_start"), f"{label}.blocks[{block_index}].epoch_ns_start")
        end_ns = integer(block.get("epoch_ns_end"), f"{label}.blocks[{block_index}].epoch_ns_end")
        require(integer(block.get("block_index"), f"{label}.blocks[{block_index}].block_index") == block_index, f"{label}: block index drift")
        require(integer(block.get("pair_start"), f"{label}.blocks[{block_index}].pair_start") == block_index * BLOCK_PAIRS, f"{label}: block start drift")
        require(integer(block.get("pair_end"), f"{label}.blocks[{block_index}].pair_end") == (block_index + 1) * BLOCK_PAIRS - 1, f"{label}: block end drift")
        require(integer(block.get("pair_count"), f"{label}.blocks[{block_index}].pair_count") == BLOCK_PAIRS, f"{label}: block pair count drift")
        require(start_ns <= end_ns and start_ns >= prior_end, f"{label}: non-monotonic block epoch boundaries")
        prior_end = end_ns
        block_pairs = pairs[block_index * BLOCK_PAIRS : (block_index + 1) * BLOCK_PAIRS]
        c1 = [numeric(pair["c1_ms"], "block c1") for pair in block_pairs]
        pinned = [numeric(pair["pinned_ms"], "block pinned") for pair in block_pairs]
        blocks.append(
            {
                "block_index": block_index,
                "epoch_ns_start": start_ns,
                "epoch_ns_end": end_ns,
                "duration_ms_wall": (end_ns - start_ns) / 1_000_000.0,
                "c1": summary(c1, BLOCK_PAIRS),
                "pinned": summary(pinned, BLOCK_PAIRS),
                "paired": paired_summary(block_pairs),
            }
        )

    c1_values = [numeric(pair["c1_ms"], "pair c1") for pair in pairs]
    pinned_values = [numeric(pair["pinned_ms"], "pair pinned") for pair in pairs]
    recomputed_paths = {PATHS[0]: summary(c1_values, SAMPLES), PATHS[1]: summary(pinned_values, SAMPLES)}
    recorded_paths = mapping(repeat.get("paths"), f"{label}.paths")
    require(set(recorded_paths) == set(PATHS), f"{label}: path set drift")
    for path in PATHS:
        require_summary(recorded_paths.get(path), recomputed_paths[path], f"{label}.paths.{path}")
    recomputed_paired = paired_summary(pairs)
    require_paired_summary(repeat.get("paired"), recomputed_paired, f"{label}.paired")
    recomputed_relative = relative_candidate_gate(recomputed_paths[PATHS[0]], recomputed_paths[PATHS[1]])
    recorded_relative = mapping(repeat.get("relative_candidate_gate"), f"{label}.relative_candidate_gate")
    require(recorded_relative.get("definition") == recomputed_relative["definition"], f"{label}: relative gate definition drift")
    require(mapping(recorded_relative.get("winner_by_percentile"), f"{label}.relative.winners") == recomputed_relative["winner_by_percentile"], f"{label}: relative gate winners drift")
    recorded_margins = mapping(recorded_relative.get("c1_margin_over_pinned_by_percentile"), f"{label}.relative.margins")
    for name, expected_margin in recomputed_relative["c1_margin_over_pinned_by_percentile"].items():  # type: ignore[index]
        if expected_margin is None:
            require(recorded_margins.get(name) is None, f"{label}: relative margin {name} must be null")
        else:
            close(recorded_margins.get(name), float(expected_margin), f"{label}.relative.margin.{name}")
    require(recorded_relative.get("passed") is recomputed_relative["passed"], f"{label}: relative gate pass drift")
    return {
        "repeat_index": repeat_index,
        "paths": recomputed_paths,
        "paired": recomputed_paired,
        "relative_candidate_gate": recomputed_relative,
        "blocks": blocks,
        "pairs": pairs,
    }


def validate_artifact(data: Mapping[str, Any], label: str, *, expected_mode: str, expected_index: int | None) -> dict[str, object]:
    require(integer(data.get("schema_version"), f"{label}.schema_version") == SCHEMA_VERSION, f"{label}: schema drift")
    require(data.get("diagnostic_only") is True and data.get("no_release_authority") is True, f"{label}: release-authority assertion drift")
    require(data.get("complete") is True, f"{label}: top-level complete must be true")
    require("failure" not in data and "map_restoration_failure" not in data, f"{label}: artifact contains recorded failure state")
    require(data.get("mode") == expected_mode, f"{label}: mode drift")
    if expected_index is not None:
        require(integer(data.get("process_index"), f"{label}.process_index") == expected_index, f"{label}: process index drift")
    process = mapping(data.get("process"), f"{label}.process")
    pid = integer(process.get("pid"), f"{label}.process.pid")
    require(pid > 1 and process.get("fresh_python_process_required") is True, f"{label}: PID/fresh process gate drift")
    target = mapping(data.get("target"), f"{label}.target")
    require(target.get("cell") == TARGET_CELL and target.get("temporary_variant") == "vshard4_p2", f"{label}: target drift")
    require(target.get("production_status") == "not_whitelisted; temporary process-local diagnostic target only", f"{label}: production status drift")
    prereg = mapping(data.get("pre_registered"), f"{label}.pre_registered")
    require(integer(prereg.get("main_processes"), f"{label}.main_processes") == 3, f"{label}: main process count drift")
    require(prereg.get("main_process_indices") == [0, 1, 2], f"{label}: main index registration drift")
    require(integer(prereg.get("repeats_per_process"), f"{label}.repeats") == REPEATS, f"{label}: repeat registration drift")
    require(integer(prereg.get("cuda_event_samples_per_path_per_repeat"), f"{label}.samples") == SAMPLES, f"{label}: sample registration drift")
    require(integer(prereg.get("paired_block_size"), f"{label}.block_size") == BLOCK_PAIRS, f"{label}: block registration drift")
    require(numeric(prereg.get("tail_threshold_ms"), f"{label}.threshold") == TAIL_THRESHOLD_MS, f"{label}: threshold drift")
    require(prereg.get("telemetry_policy") == "The separately-run non_gating_telemetry process is explanatory only and is excluded from every stability classification.", f"{label}: telemetry policy drift")
    relative_prereg = mapping(prereg.get("relative_candidate_gate"), f"{label}.relative_candidate_gate")
    require(relative_prereg.get("scope") == "all six repeats from the three stability_main processes only", f"{label}: relative scope drift")
    require(relative_prereg.get("rule") == "C1 must win P50/P95/P99 against pinned in every repeat with each margin >=2%", f"{label}: relative rule drift")
    require(relative_prereg.get("policy") == "reported only; it has no release authority in this diagnostic", f"{label}: relative policy drift")
    identity = mapping(data.get("identity"), f"{label}.identity")
    runner = mapping(identity.get("tail_runner"), f"{label}.identity.tail_runner")
    expected_runner_sha = os.environ.get(RUNNER_SHA_ENV)
    require(expected_runner_sha is not None, f"{RUNNER_SHA_ENV} must be set for independent audit")
    require(runner.get("sha256") == expected_runner_sha and runner.get("sha256_gate_pass") is True, f"{label}: runner source gate drift")
    gates = mapping(data.get("gates"), f"{label}.gates")
    for gate_name in (
        "production_map_before",
        "temporary_target_map",
        "map_restored",
        "target_correctness",
        "route",
        "prepare_spy_restored",
        "backend_methods_restored",
        "clean_gpu",
        "python_nvidia_clean",
        "prepare_spy_restored_after_restore",
        "device",
        "extension",
        "fla_pin",
    ):
        gate_passed(gates.get(gate_name), f"{label}.gates.{gate_name}")
    map_data = mapping(data.get("map"), f"{label}.map")
    require(map_data.get("installed") is True and map_data.get("restored") is True, f"{label}: map lifecycle drift")
    installation = mapping(map_data.get("installation"), f"{label}.map.installation")
    restoration = mapping(map_data.get("restoration"), f"{label}.map.restoration")
    installation_object_id = integer(
        installation.get("production_map_object_id"), f"{label}.map.installation.production_map_object_id"
    )
    restoration_object_id = integer(
        restoration.get("map_object_id"), f"{label}.map.restoration.map_object_id"
    )
    require(installation_object_id > 1 and restoration_object_id > 1, f"{label}: invalid map object ID")
    require(installation_object_id == restoration_object_id, f"{label}: map object identity changed")
    require(integer(restoration.get("entries"), f"{label}.map.restoration.entries") == 2, f"{label}: production entry count after restore")
    correctness = mapping(data.get("correctness"), f"{label}.correctness")
    require(correctness.get("passed") is True and correctness.get("expected_variant") == "vshard4_p2", f"{label}: target correctness gate drift")
    performance = mapping(data.get("performance"), f"{label}.performance")
    require(performance.get("complete") is True, f"{label}: incomplete performance")
    raw_repeats = sequence(performance.get("repeats"), f"{label}.performance.repeats")
    require(len(raw_repeats) == REPEATS, f"{label}: repeat count drift")
    repeats = [validate_repeat(raw_repeats[index], f"{label}.repeat{index}", index) for index in range(REPEATS)]
    return {"pid": pid, "process_index": integer(data.get("process_index"), f"{label}.process_index"), "repeats": repeats}


def aggregate_main(processes: Sequence[Mapping[str, object]]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    c1_all: list[float] = []
    pinned_all: list[float] = []
    all_pairs: list[Mapping[str, Any]] = []
    parity: dict[str, list[Mapping[str, Any]]] = {PATHS[0]: [], PATHS[1]: []}
    high_blocks: list[dict[str, object]] = []
    rule_failures: list[dict[str, object]] = []
    for process in processes:
        process_index = integer(process["process_index"], "aggregate.process_index")
        for repeat in process["repeats"]:  # type: ignore[index]
            repeat_data = mapping(repeat, "aggregate.repeat")
            pairs = [mapping(pair, "aggregate.pair") for pair in repeat_data["pairs"]]  # type: ignore[index]
            path_summaries = mapping(repeat_data["paths"], "aggregate.paths")
            paired = mapping(repeat_data["paired"], "aggregate.paired")
            c1_p99 = numeric(mapping(path_summaries[PATHS[0]], "aggregate.c1").get("p99_ms"), "aggregate.c1.p99")
            paired_p99 = numeric(paired.get("delta_p99_ms"), "aggregate.paired.p99")
            joint_count = integer(paired.get("c1_gt_pinned_and_gt_threshold_count"), "aggregate.paired.joint_count")
            criteria = {
                "c1_p99_le_1_20": c1_p99 <= TAIL_THRESHOLD_MS,
                "paired_p99_le_0": paired_p99 <= 0.0,
                "joint_tail_excess_count_zero": joint_count == 0,
            }
            if not all(criteria.values()):
                rule_failures.append({"process_index": process_index, "repeat_index": repeat_data["repeat_index"], "criteria": criteria})
            records.append(
                {
                    "process_index": process_index,
                    "repeat_index": repeat_data["repeat_index"],
                    "c1": path_summaries[PATHS[0]],
                    "pinned": path_summaries[PATHS[1]],
                    "paired": paired,
                    "criteria": criteria,
                    "relative_candidate_gate": repeat_data["relative_candidate_gate"],
                }
            )
            c1_all.extend(numeric(pair["c1_ms"], "aggregate.c1") for pair in pairs)
            pinned_all.extend(numeric(pair["pinned_ms"], "aggregate.pinned") for pair in pairs)
            all_pairs.extend(pairs)
            for path in PATHS:
                parity[path].extend(pair for pair in pairs if pair["first_path"] == path)
            for block in repeat_data["blocks"]:  # type: ignore[index]
                block_data = mapping(block, "aggregate.block")
                block_pair = mapping(block_data["paired"], "aggregate.block.paired")
                if integer(block_pair["c1_gt_pinned_and_gt_threshold_count"], "aggregate.block.joint") > 0:
                    high_blocks.append(
                        {
                            "process_index": process_index,
                            "repeat_index": repeat_data["repeat_index"],
                            "block_index": block_data["block_index"],
                            "epoch_ns_start": block_data["epoch_ns_start"],
                            "epoch_ns_end": block_data["epoch_ns_end"],
                            "paired": block_pair,
                        }
                    )
    require(len(records) == 6, "main aggregation requires exactly six repeats")
    global_paired = paired_summary(all_pairs)
    parity_output = {
        first_path: {
            "c1": summary([numeric(pair["c1_ms"], "parity c1") for pair in pairs], 3000),
            "pinned": summary([numeric(pair["pinned_ms"], "parity pinned") for pair in pairs], 3000),
            "paired": paired_summary(pairs),
        }
        for first_path, pairs in parity.items()
    }
    stable = not rule_failures
    if stable:
        classification = "tail_stable_under_pre_registered_rule"
    elif any(
        not bool(failure["criteria"]["c1_p99_le_1_20"])
        or not bool(failure["criteria"]["joint_tail_excess_count_zero"])
        for failure in rule_failures
    ):
        classification = "c1_tail_excess_observed"
    else:
        classification = "relative_tail_regression_observed"
    return {
        "scope": "only the three stability_main processes; telemetry process excluded",
        "processes": len(processes),
        "repeats": records,
        "all_samples": {
            "c1": summary(c1_all, 6000),
            "pinned": summary(pinned_all, 6000),
            "paired": global_paired,
        },
        "parity_by_first_path": parity_output,
        "tail_excess_blocks": high_blocks,
        "pre_registered_stability_rule": {
            "tail_stable": stable,
            "failed_repeats": rule_failures,
            "classification": classification,
        },
        "relative_candidate_gate": {
            "scope": "only the six main repeats; diagnostic only and never a whitelist authority",
            "passed": all(bool(mapping(record["relative_candidate_gate"], "aggregate.relative").get("passed")) for record in records),
            "failed_repeats": [
                {"process_index": record["process_index"], "repeat_index": record["repeat_index"]}
                for record in records
                if mapping(record["relative_candidate_gate"], "aggregate.relative").get("passed") is not True
            ],
        },
    }


def parse_number(text: str) -> float | None:
    value = text.strip()
    if value.upper() in {"N/A", "NOT SUPPORTED", "UNKNOWN", ""}:
        return None
    token = value.split()[0]
    try:
        parsed = float(token)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_telemetry(path: Path, expected_sha: str) -> dict[str, object]:
    require(path.is_file(), f"missing telemetry CSV {path}")
    actual_sha = sha(path)
    require(actual_sha == expected_sha, "telemetry CSV SHA mismatch")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].startswith("UNAVAILABLE:"):
        return {
            "available": False,
            "csv_path": str(path),
            "csv_sha256": actual_sha,
            "reason": lines[0] if lines else "telemetry sidecar emitted no rows",
            "excluded_from_stability": True,
        }
    reader = csv.reader(lines)
    rows = [row for row in reader if row]
    # The sidecar intentionally uses no header.  A driver could nevertheless
    # emit a banner/header; reject it as unavailable rather than corrupting the
    # main stability conclusion.
    if not rows or any(len(row) != len(TELEMETRY_FIELDS) for row in rows):
        return {
            "available": False,
            "csv_path": str(path),
            "csv_sha256": actual_sha,
            "reason": "unexpected NVML CSV column count",
            "excluded_from_stability": True,
        }
    samples = [{field: row[index].strip() for index, field in enumerate(TELEMETRY_FIELDS)} for row in rows]
    numeric_fields = (
        "clocks.current.graphics",
        "clocks.current.sm",
        "clocks.current.memory",
        "power.draw",
        "utilization.gpu",
        "utilization.memory",
        "temperature.gpu",
    )
    summaries: dict[str, object] = {}
    unavailable: list[str] = []
    for field in numeric_fields:
        values = [value for value in (parse_number(sample[field]) for sample in samples) if value is not None]
        if not values:
            unavailable.append(field)
        else:
            summaries[field] = {
                "samples": len(values),
                "min": min(values),
                "mean": statistics.fmean(values),
                "max": max(values),
            }
    return {
        "available": True,
        "csv_path": str(path),
        "csv_sha256": actual_sha,
        "fields": list(TELEMETRY_FIELDS),
        "samples": len(samples),
        "first_timestamp": samples[0]["timestamp"],
        "last_timestamp": samples[-1]["timestamp"],
        "numeric_summaries": summaries,
        "unavailable_numeric_fields": unavailable,
        "excluded_from_stability": True,
        "interpretation": "single non-gating process sidecar; descriptive only, not time-aligned tightly enough to establish causality",
    }


def require_analyzer_identity() -> dict[str, object]:
    path = Path(__file__).resolve(strict=True)
    expected = os.environ.get(ANALYZER_SHA_ENV)
    actual = sha(path)
    require(expected is not None, f"{ANALYZER_SHA_ENV} must be set")
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
    parser.add_argument("main_json", nargs=3, type=Path, help="three stability_main JSON artifacts")
    parser.add_argument("--expected-main-sha256", nargs=3, required=True)
    parser.add_argument("--telemetry-json", type=Path, required=True)
    parser.add_argument("--expected-telemetry-json-sha256", required=True)
    parser.add_argument("--telemetry-csv", type=Path, required=True)
    parser.add_argument("--expected-telemetry-csv-sha256", required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    require(args.json.suffix.lower() == ".json", "--json must use .json suffix")
    analyzer_identity = require_analyzer_identity()
    main_inputs = [
        read_json(path, expected_sha, f"main[{index}]")
        for index, (path, expected_sha) in enumerate(zip(args.main_json, args.expected_main_sha256, strict=True))
    ]
    validated_main = [
        validate_artifact(data, f"main[{index}]", expected_mode="stability_main", expected_index=index)
        for index, (data, _digest) in enumerate(main_inputs)
    ]
    pids = [integer(item["pid"], "main PID") for item in validated_main]
    require(len(set(pids)) == 3, f"main processes are not fresh/distinct: {pids}")
    telemetry_data, telemetry_json_sha = read_json(
        args.telemetry_json, args.expected_telemetry_json_sha256, "non_gating_telemetry"
    )
    validated_telemetry = validate_artifact(
        telemetry_data, "non_gating_telemetry", expected_mode="non_gating_telemetry", expected_index=3
    )
    require(integer(validated_telemetry["pid"], "telemetry PID") not in set(pids), "telemetry process PID must differ from all main processes")
    telemetry = parse_telemetry(args.telemetry_csv, args.expected_telemetry_csv_sha256)
    main_result = aggregate_main(validated_main)
    result = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "independent diagnostic-only fp32-both tail audit; no whitelist/promotion authority",
        "diagnostic_only": True,
        "no_release_authority": True,
        "analyzer": analyzer_identity,
        "main_inputs": [
            {"path": str(path), "sha256": digest, "process_index": index}
            for index, (path, (_data, digest)) in enumerate(zip(args.main_json, main_inputs, strict=True))
        ],
        "non_gating_telemetry_input": {
            "json_path": str(args.telemetry_json),
            "json_sha256": telemetry_json_sha,
            "process_index": 3,
            "telemetry": telemetry,
        },
        "main_stability": main_result,
        "promotion_decision": {
            "action": "do_not_change_production_whitelist",
            "reason": "This experiment is pre-registered diagnostic evidence only; telemetry is explicitly excluded from stability and promotion logic.",
        },
        "complete": True,
    }
    atomic_write(args.json, result)
    print(
        f"wrote independent fp32 tail audit {args.json}; "
        f"classification={main_result['pre_registered_stability_rule']['classification']}"
    )


if __name__ == "__main__":
    main()
