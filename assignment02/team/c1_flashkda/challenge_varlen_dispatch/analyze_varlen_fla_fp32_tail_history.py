#!/usr/bin/env python3
"""Recompute the r1/r3 fp32-both tail history from raw samples only.

This is a stdlib-only, diagnostic-only artifact.  It deliberately does not
read runner summaries, audits, or logs: every reported statistic is derived
from the two pinned raw JSON inputs below.  The input paths and SHA-256
digests are fixed so that a changed source fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
TARGET_CELL = "skew_n6_h12_t12288/fp32_both"
PATHS = ("public_registry_c1", "public_registry_pinned")
REPEATS = 2
SAMPLES = 1000
BLOCK_SIZE = 100
TAIL_THRESHOLD_MS = 1.20
INPUTS = (
    (
        "c1_varlen_fla_handoff_candidate_b300_sm103a_r1.json",
        "c302de7cbb72db7d8ff4c60de157a8ff98172b94e42a07b19706fca05562491d",
    ),
    (
        "c1_varlen_fla_handoff_candidate_b300_sm103a_r3.json",
        "2d50d219c5eb33cde726331cfc7ee613c2a7959918f67675e344e29fead23158",
    ),
)


class AuditError(AssertionError):
    """Raised for any input/schema drift."""


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


def number(value: object, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: Sequence[float], quantile: float) -> float:
    require(bool(values), "percentile requires non-empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summary(values: Sequence[float], label: str, *, positive: bool) -> dict[str, float | int]:
    require(values, f"{label} must be non-empty")
    for index, value in enumerate(values):
        number(value, f"{label}[{index}]")
        if positive:
            require(value > 0.0, f"{label}[{index}] must be positive")
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
    }


def threshold_hits(values: Sequence[bool]) -> dict[str, object]:
    indices = [index for index, hit in enumerate(values) if hit]
    runs: list[list[int]] = []
    for index in indices:
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index, index])
        else:
            runs[-1][-1] = index
    return {
        "count": len(indices),
        "min_index": min(indices) if indices else None,
        "max_index": max(indices) if indices else None,
        "consecutive_runs": runs,
    }


def paired_summary(c1: Sequence[float], pinned: Sequence[float], label: str) -> dict[str, object]:
    require(len(c1) == len(pinned), f"{label}: paired lengths differ")
    delta = [c1_value - pinned_value for c1_value, pinned_value in zip(c1, pinned)]
    return {
        "delta_definition": "d[i] = public_registry_c1[i] - public_registry_pinned[i]",
        "delta": summary(delta, f"{label}.delta", positive=False),
        "threshold_hits": {
            "c1_gt_pinned": threshold_hits([c1_value > pinned_value for c1_value, pinned_value in zip(c1, pinned)]),
            "c1_gt_1_20_ms": threshold_hits([value > TAIL_THRESHOLD_MS for value in c1]),
            "pinned_gt_1_20_ms": threshold_hits([value > TAIL_THRESHOLD_MS for value in pinned]),
        },
    }


def extract_raw(data: Mapping[str, Any], label: str) -> tuple[list[float], list[float]]:
    performance = mapping(data.get("performance"), f"{label}.performance")
    cells = mapping(performance.get("cells"), f"{label}.performance.cells")
    require(set(cells) >= {TARGET_CELL}, f"{label}: target cell missing")
    cell = mapping(cells.get(TARGET_CELL), f"{label}.{TARGET_CELL}")
    repeats = sequence(cell.get("repeats"), f"{label}.{TARGET_CELL}.repeats")
    require(len(repeats) == REPEATS, f"{label}: expected exactly {REPEATS} repeats")
    all_c1: list[float] = []
    all_pinned: list[float] = []
    for repeat_index, raw_repeat in enumerate(repeats):
        repeat = mapping(raw_repeat, f"{label}.repeat{repeat_index}")
        require(integer(repeat.get("repeat_index"), f"{label}.repeat{repeat_index}.repeat_index") == repeat_index, f"{label}: repeat index drift")
        raw = mapping(repeat.get("raw_samples_ms"), f"{label}.repeat{repeat_index}.raw_samples_ms")
        require(set(raw) == set(PATHS), f"{label}.repeat{repeat_index}: raw path set drift")
        c1 = sequence(raw.get(PATHS[0]), f"{label}.repeat{repeat_index}.c1")
        pinned = sequence(raw.get(PATHS[1]), f"{label}.repeat{repeat_index}.pinned")
        require(len(c1) == SAMPLES and len(pinned) == SAMPLES, f"{label}.repeat{repeat_index}: expected {SAMPLES} samples per path")
        for index, value in enumerate(c1):
            parsed = number(value, f"{label}.repeat{repeat_index}.c1[{index}]")
            require(parsed > 0.0, f"{label}.repeat{repeat_index}.c1[{index}] must be positive")
            all_c1.append(parsed)
        for index, value in enumerate(pinned):
            parsed = number(value, f"{label}.repeat{repeat_index}.pinned[{index}]")
            require(parsed > 0.0, f"{label}.repeat{repeat_index}.pinned[{index}] must be positive")
            all_pinned.append(parsed)
    return all_c1, all_pinned


def analyze_repeat(c1: Sequence[float], pinned: Sequence[float], repeat_index: int) -> dict[str, object]:
    require(len(c1) == SAMPLES and len(pinned) == SAMPLES, "repeat sample count drift")
    delta = [c1_value - pinned_value for c1_value, pinned_value in zip(c1, pinned)]
    parity: dict[str, object] = {}
    for parity_name, indices in (("even", range(0, SAMPLES, 2)), ("odd", range(1, SAMPLES, 2))):
        selected = list(indices)
        parity_c1 = [c1[index] for index in selected]
        parity_pinned = [pinned[index] for index in selected]
        parity_delta = [delta[index] for index in selected]
        parity[parity_name] = {
            "sample_indices": {"first": selected[0], "last": selected[-1], "step": 2},
            "first_path": PATHS[0] if parity_name == "even" else PATHS[1],
            "c1": summary(parity_c1, f"repeat{repeat_index}.{parity_name}.c1", positive=True),
            "pinned": summary(parity_pinned, f"repeat{repeat_index}.{parity_name}.pinned", positive=True),
            "paired": paired_summary(parity_c1, parity_pinned, f"repeat{repeat_index}.{parity_name}"),
        }

    blocks: list[dict[str, object]] = []
    for block_index, start in enumerate(range(0, SAMPLES, BLOCK_SIZE)):
        end = start + BLOCK_SIZE
        block_c1 = list(c1[start:end])
        block_pinned = list(pinned[start:end])
        blocks.append(
            {
                "block_index": block_index,
                "sample_index_start": start,
                "sample_index_end": end - 1,
                "c1": summary(block_c1, f"repeat{repeat_index}.block{block_index}.c1", positive=True),
                "pinned": summary(block_pinned, f"repeat{repeat_index}.block{block_index}.pinned", positive=True),
                "paired": paired_summary(block_c1, block_pinned, f"repeat{repeat_index}.block{block_index}"),
            }
        )
    return {
        "repeat_index": repeat_index,
        "c1": summary(c1, f"repeat{repeat_index}.c1", positive=True),
        "pinned": summary(pinned, f"repeat{repeat_index}.pinned", positive=True),
        "paired": paired_summary(c1, pinned, f"repeat{repeat_index}"),
        "first_path_parity": parity,
        "blocks": blocks,
    }


def load_input(base: Path, filename: str, expected_sha: str, label: str) -> tuple[Mapping[str, Any], str]:
    path = base / filename
    require(path.is_file(), f"{label}: missing input {path}")
    actual_sha = sha256(path)
    require(actual_sha == expected_sha, f"{label}: SHA mismatch expected={expected_sha} actual={actual_sha}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label}: invalid JSON: {exc}") from exc
    return mapping(data, label), actual_sha


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(__file__).resolve().parent / "results" / "c1_varlen_fla_fp32_tail_history_r1_r3.json"
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    results_dir = Path(__file__).resolve().parent / "results"
    input_records: list[dict[str, object]] = []
    repeats: list[dict[str, object]] = []
    for input_index, (filename, expected_sha) in enumerate(INPUTS):
        data, actual_sha = load_input(results_dir, filename, expected_sha, f"input{input_index}")
        c1, pinned = extract_raw(data, f"input{input_index}")
        for repeat_index in range(REPEATS):
            start = repeat_index * SAMPLES
            repeats.append(analyze_repeat(c1[start : start + SAMPLES], pinned[start : start + SAMPLES], repeat_index))
        input_records.append({"filename": filename, "sha256": actual_sha, "target_cell": TARGET_CELL})

    # The source files contain two repeats each; the artifact labels them by
    # input rather than silently presenting a four-repeat pooled history.
    grouped_repeats = []
    for input_index in range(len(INPUTS)):
        grouped_repeats.append(
            {
                "input_index": input_index,
                "input_filename": INPUTS[input_index][0],
                "repeats": repeats[input_index * REPEATS : (input_index + 1) * REPEATS],
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "offline fp32-both tail history recomputed from raw_samples_ms; diagnostic discussion support only",
        "diagnostic_only": True,
        "no_release_authority": True,
        "target": {
            "cell": TARGET_CELL,
            "paths": list(PATHS),
            "threshold_ms": TAIL_THRESHOLD_MS,
            "sample_count_per_path_per_repeat": SAMPLES,
            "block_size_samples": BLOCK_SIZE,
            "percentile_definition": "linear interpolation at (n-1)*q, q in {0.50, 0.95, 0.99}",
        },
        "inputs": input_records,
        "repeats_by_input": grouped_repeats,
        "pooling_boundary": "No cross-input pooling is reported; r1 and r3 remain separate histories, each with exactly two repeats.",
        "what_this_can_confirm": [
            "The same-index C1/pinned distribution, paired delta, parity, block, and threshold-hit statistics for the two fixed raw artifacts.",
            "Whether 1.20 ms threshold hits cluster in sample-index runs within each stored repeat.",
        ],
        "what_this_cannot_confirm": [
            "A cause of the tail (for example clocks, scheduler state, or thermal behavior), because no time-aligned telemetry is used.",
            "A new whitelist or production decision; this artifact has no release authority.",
            "Generalization beyond this target cell, two artifacts, two repeats, and their recorded sample order.",
        ],
        "complete": True,
    }
    write_json(args.output, payload)
    print(f"wrote {args.output} ({sha256(args.output)})")


if __name__ == "__main__":
    try:
        main()
    except (AuditError, OSError) as exc:
        raise SystemExit(f"AUDIT_ERROR: {exc}") from exc
