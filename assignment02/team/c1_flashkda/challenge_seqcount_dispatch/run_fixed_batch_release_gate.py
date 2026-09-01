#!/usr/bin/env python3
"""Qualify exact fixed-batch FlashKDA cells for public-FLA integration review.

This is a release gate, not a new discovery study and not a dispatcher
mutation.  The earlier 12-cell confirmation correctly failed as a whole when
the B=8/``none`` P99 runner-up margin was 1.8067%.  That non-promotion result
is retained: B=8, B=4 FP32-both, and every other newly considered B>1 fixed
batch cell remain baseline.  Existing B=1 dispatcher policy is unchanged.

Only the eleven separately pre-registered cells below are eligible for a new,
independent decision.  Each cell carries its own historical and new-run gates;
there is intentionally no all-or-nothing aggregate promotion rule.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_seqcount_dispatch import (  # noqa: E402
    run_seqcount_dispatch as shared,
)


DIM = 128
WARMUP = 100
SAMPLES_PER_REPEAT = 1000
REPEATS = 2
MIN_WINNER_MARGIN = 0.02
PERCENTILES = (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
VARIANTS = ("baseline", "vshard2_p2", "vshard4_p2")
RAW_CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
CLEAN_GPU_GATE_ENV = "C1_FIXED_BATCH_RELEASE_GATE_CLEAN_GPU"

DISCOVERY_SHA256 = "46cd27f2fbdcaeeb61011c49c6175a0c05d15d4365bfda800cf52040dbe414f7"
CONFIRMATION_SHA256 = "b7084ecf73461ba0e590b7db74719af3ba83fd98f1f174103cc451515dfb9795"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_DISCOVERY_JSON = RESULTS_DIR / "c1_seqcount_dispatch_b300_sm103a_r2.json"
DEFAULT_CONFIRMATION_JSON = RESULTS_DIR / "c1_fixed_batch_confirmation_b300_sm103a_r1.json"


@dataclass(frozen=True)
class ReleaseCell:
    """One exact B/H/T/state cell that may independently leave baseline."""

    batch: int
    contract: str
    expected_winner: str
    discovery_case: str
    confirmation_case: str

    @property
    def key(self) -> str:
        return f"b{self.batch}_h12_t2048/{self.contract}"


def _cells_for_batch(
    batch: int, expected_winner: str, discovery_case: str
) -> tuple[ReleaseCell, ...]:
    confirmation_case = f"b{batch}_h12_t2048"
    return tuple(
        ReleaseCell(batch, contract, expected_winner, discovery_case, confirmation_case)
        for contract in ("none", "fp32_final_only", "fp32_both")
    )


RELEASE_CELLS = (
    *_cells_for_batch(2, "vshard4_p2", "m024_n02_h12_fixed"),
    *_cells_for_batch(3, "vshard4_p2", "m036_n03_h12_fixed"),
    ReleaseCell(4, "none", "vshard2_p2", "m048_n04_h12_fixed", "b4_h12_t2048"),
    ReleaseCell(4, "fp32_final_only", "vshard2_p2", "m048_n04_h12_fixed", "b4_h12_t2048"),
    *_cells_for_batch(6, "vshard2_p2", "m072_n06_h12_fixed"),
)
FALLBACK_POLICY = {
    "b4_h12_t2048/fp32_both": "baseline",
    "b8_h12_t2048/none": "baseline",
    "b8_h12_t2048/fp32_final_only": "baseline",
    "b8_h12_t2048/fp32_both": "baseline",
    "unlisted_new_b_gt1_fixed_batch_cells": "baseline",
}
EXISTING_POLICY_BOUNDARY = "all existing B=1 and non-fixed dispatcher entries remain unchanged"


def _case(batch: int) -> shared.Case:
    return shared.Case(
        name=f"b{batch}_h12_t2048",
        form="fixed",
        sequences=batch,
        heads=12,
        lengths=(2048,) * batch,
        family="fixed_batch_release_gate",
    )


CASES = tuple(_case(batch) for batch in (2, 3, 4, 6))
CASES_BY_BATCH = {case.sequences: case for case in CASES}


def _write(path: Path, result: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pre_registration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "shape": {"H": 12, "T": 2048, "K": DIM, "V": DIM},
        "new_job_seed": args.seed,
        "repeats": REPEATS,
        "samples_per_repeat": SAMPLES_PER_REPEAT,
        "warmup_per_path": WARMUP,
        "percentiles": [name for name, _ in PERCENTILES],
        "minimum_runner_up_margin": MIN_WINNER_MARGIN,
        "release_cells": [
            {
                "key": cell.key,
                "expected_winner": cell.expected_winner,
                "discovery_case": cell.discovery_case,
                "confirmation_case": cell.confirmation_case,
            }
            for cell in RELEASE_CELLS
        ],
        "fallback_policy": FALLBACK_POLICY,
        "existing_policy_boundary": EXISTING_POLICY_BOUNDARY,
        "mapping_precedence": "an exact eligible cell key takes precedence over the new-B>1 default fallback",
        "history": {
            "discovery_json": {"path": str(args.discovery_json), "sha256": DISCOVERY_SHA256},
            "confirmation_json": {"path": str(args.confirmation_json), "sha256": CONFIRMATION_SHA256},
        },
        "per_cell_release_rule": (
            "A cell is eligible for public-FLA integration review only when its discovery single "
            "repeat, both confirmation repeats, "
            "and both new-job repeats independently select the pre-registered winner at P50/P95/P99 "
            "with a runner-up margin of at least 2%.  A failure affects that cell only."
        ),
    }


def _initial_result(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "purpose": "per-cell fixed-batch raw-evidence gate; no dispatcher mutation or public-FLA claim",
        "preregistration": _pre_registration(args),
        "cases": [shared._case_dict(case) for case in CASES],
        "raw_abi_contracts": list(RAW_CONTRACTS),
        "history_evidence": {},
        "raw_abi_correctness": {},
        "new_measurements": {},
        "cell_status": {
            cell.key: {
                "status": "baseline",
                "expected_winner": cell.expected_winner,
                "reason": "not_run",
            }
            for cell in RELEASE_CELLS
        },
        "fallback_policy": FALLBACK_POLICY,
        "existing_policy_boundary": EXISTING_POLICY_BOUNDARY,
        "public_fla_integration_candidate_mapping": {},
        "fallback_mapping": dict(FALLBACK_POLICY),
        "gates": {
            "scope_count": {"required": 11, "actual": len(RELEASE_CELLS), "passed": len(RELEASE_CELLS) == 11},
            "clean_gpu_shell_gate": {"required": True, "passed": False},
            "device_gate": {"required": "B300, capability 10.3, 148 SM", "passed": False},
            "audited_extension_sha256_gate": {"required": shared.AUDITED_EXTENSION_SHA256, "passed": False},
            "historical_identity_gate": {"passed": False},
            "current_raw_abi_exact_gate": {"passed": False},
        },
        "complete": False,
    }


def _as_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _as_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def _load_json_with_fixed_hash(path: Path, expected_sha256: str, label: str) -> tuple[Mapping[str, Any], dict[str, object]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read {label} evidence {path}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} is not valid JSON despite matching its expected identity") from exc
    return _as_mapping(parsed, label), {"path": str(path), "sha256": actual, "sha256_gate_pass": True}


def _require_b300_identity(value: Mapping[str, Any], label: str) -> dict[str, object]:
    """Validate the hardware and extension identifiers recorded in pinned history."""
    device = _as_mapping(value.get("device"), f"{label}.device")
    extension = _as_mapping(value.get("extension"), f"{label}.extension")
    name = str(device.get("name", ""))
    capability = device.get("capability")
    sm_count = device.get("multiprocessor_count")
    extension_sha = str(extension.get("sha256", ""))
    if (
        "B300" not in name.upper()
        or capability != [10, 3]
        or sm_count != 148
        or extension_sha != shared.AUDITED_EXTENSION_SHA256
    ):
        raise RuntimeError(
            f"{label} identity is not the audited B300 artifact: "
            f"name={name!r}, capability={capability!r}, SMs={sm_count!r}, extension={extension_sha!r}"
        )
    return {
        "name": name,
        "capability": capability,
        "multiprocessor_count": sm_count,
        "extension_sha256": extension_sha,
        "identity_gate_pass": True,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _samples(raw: object, label: str) -> list[float]:
    values = _as_list(raw, label)
    if len(values) != SAMPLES_PER_REPEAT:
        raise RuntimeError(f"{label} must contain exactly {SAMPLES_PER_REPEAT} samples, got {len(values)}")
    numeric = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
        raise RuntimeError(f"{label} contains a non-finite or non-positive CUDA-event latency")
    return numeric


def _recompute_latency_evidence(benchmark: Mapping[str, Any], label: str) -> dict[str, object]:
    """Recompute all winner/margin facts from raw samples, never saved booleans."""
    raw = _as_mapping(benchmark.get("raw_samples_ms"), f"{label}.raw_samples_ms")
    if set(raw) != set(VARIANTS):
        raise RuntimeError(f"{label} must contain exactly raw paths {VARIANTS}, got {sorted(raw)}")
    summaries: dict[str, dict[str, float | int]] = {}
    for variant in VARIANTS:
        samples = _samples(raw[variant], f"{label}.raw_samples_ms.{variant}")
        summaries[variant] = {
            "samples": len(samples),
            "mean_ms": statistics.fmean(samples),
            "p50_ms": _percentile(samples, 0.50),
            "p95_ms": _percentile(samples, 0.95),
            "p99_ms": _percentile(samples, 0.99),
        }
    winner_by_percentile: dict[str, str] = {}
    runner_up_margin: dict[str, float] = {}
    for percentile, _ in PERCENTILES:
        metric = f"{percentile}_ms"
        ranked = sorted((float(summaries[variant][metric]), variant) for variant in VARIANTS)
        if ranked[0][0] <= 0.0:
            raise RuntimeError(f"{label}.{percentile} has a non-positive winner latency")
        winner_by_percentile[percentile] = ranked[0][1]
        runner_up_margin[percentile] = ranked[1][0] / ranked[0][0] - 1.0
    return {
        "recomputed_from_raw_samples": True,
        "summaries": summaries,
        "winner_by_percentile": winner_by_percentile,
        "runner_up_margin_by_percentile": runner_up_margin,
    }


def _assess_expected(benchmark: Mapping[str, Any], expected_winner: str, label: str) -> dict[str, object]:
    evidence = _recompute_latency_evidence(benchmark, label)
    winners = _as_mapping(evidence["winner_by_percentile"], f"{label}.winners")
    margins = _as_mapping(evidence["runner_up_margin_by_percentile"], f"{label}.margins")
    expected_at_all_percentiles = all(winners[percentile] == expected_winner for percentile, _ in PERCENTILES)
    margin_gate_pass = all(float(margins[percentile]) >= MIN_WINNER_MARGIN for percentile, _ in PERCENTILES)
    return {
        **evidence,
        "expected_winner": expected_winner,
        "expected_winner_at_all_percentiles": expected_at_all_percentiles,
        "minimum_required_margin": MIN_WINNER_MARGIN,
        "margin_gate_pass": margin_gate_pass,
        "gate_pass": expected_at_all_percentiles and margin_gate_pass,
    }


def _validate_exact_record(value: object, label: str) -> None:
    record = _as_mapping(value, label)
    if record.get("output_exact") is not True or float(record.get("output_max_abs", float("nan"))) != 0.0:
        raise RuntimeError(f"exactness error in {label}.output")
    present = record.get("final_state_present")
    if present is True:
        if record.get("final_state_exact") is not True or float(record.get("final_state_max_abs", float("nan"))) != 0.0:
            raise RuntimeError(f"exactness error in {label}.final_state")
    elif present is not False:
        raise RuntimeError(f"exactness record malformed in {label}.final_state_present")


def _validate_historical_raw_abi(
    correctness: Mapping[str, Any], case_name: str, label: str
) -> dict[str, object]:
    case_data = _as_mapping(correctness.get(case_name), f"{label}.{case_name}")
    for contract in RAW_CONTRACTS:
        contract_data = _as_mapping(case_data.get(contract), f"{label}.{case_name}.{contract}")
        for variant in ("vshard2_p2", "vshard4_p2"):
            _validate_exact_record(
                contract_data.get(variant), f"{label}.{case_name}.{contract}.{variant}"
            )
    return {"case": case_name, "contracts": list(RAW_CONTRACTS), "exact_gate_pass": True}


def _load_historical_evidence(args: argparse.Namespace) -> dict[str, object]:
    discovery, discovery_identity = _load_json_with_fixed_hash(
        args.discovery_json, DISCOVERY_SHA256, "discovery"
    )
    confirmation, confirmation_identity = _load_json_with_fixed_hash(
        args.confirmation_json, CONFIRMATION_SHA256, "confirmation"
    )
    if discovery.get("complete") is not True or confirmation.get("complete") is not True:
        raise RuntimeError("historical evidence is incomplete")
    discovery_device = _require_b300_identity(discovery, "discovery")
    confirmation_identity_block = _as_mapping(confirmation.get("identity"), "confirmation.identity")
    confirmation_device = _require_b300_identity(confirmation_identity_block, "confirmation")
    discovery_benchmark = _as_mapping(discovery.get("benchmark"), "discovery.benchmark")
    discovery_correctness = _as_mapping(discovery.get("correctness"), "discovery.correctness")
    confirmation_benchmark = _as_mapping(
        confirmation.get("raw_wrapper_public_contract_benchmarks"),
        "confirmation.raw_wrapper_public_contract_benchmarks",
    )
    confirmation_correctness = _as_mapping(
        confirmation.get("raw_abi_correctness"), "confirmation.raw_abi_correctness"
    )

    exact_by_batch: dict[str, object] = {}
    for batch, discovery_case, confirmation_case in (
        (2, "m024_n02_h12_fixed", "b2_h12_t2048"),
        (3, "m036_n03_h12_fixed", "b3_h12_t2048"),
        (4, "m048_n04_h12_fixed", "b4_h12_t2048"),
        (6, "m072_n06_h12_fixed", "b6_h12_t2048"),
    ):
        exact_by_batch[f"b{batch}"] = {
            "discovery": _validate_historical_raw_abi(
                discovery_correctness, discovery_case, "discovery.correctness"
            ),
            "confirmation": _validate_historical_raw_abi(
                confirmation_correctness, confirmation_case, "confirmation.raw_abi_correctness"
            ),
        }

    cell_evidence: dict[str, object] = {}
    for cell in RELEASE_CELLS:
        discovery_case = _as_mapping(
            discovery_benchmark.get(cell.discovery_case), f"discovery.benchmark.{cell.discovery_case}"
        )
        discovery_cell = _assess_expected(
            _as_mapping(discovery_case.get(cell.contract), f"discovery cell {cell.key}"),
            cell.expected_winner,
            f"discovery/{cell.key}",
        )
        confirmation_case = _as_mapping(
            confirmation_benchmark.get(cell.confirmation_case),
            f"confirmation.benchmark.{cell.confirmation_case}",
        )
        confirmation_cell = _as_mapping(
            confirmation_case.get(cell.contract), f"confirmation cell {cell.key}"
        )
        repeats = _as_list(confirmation_cell.get("repeats"), f"confirmation repeats {cell.key}")
        if len(repeats) != REPEATS:
            raise RuntimeError(f"confirmation repeats for {cell.key} must be exactly {REPEATS}")
        confirmation_repeats = [
            _assess_expected(
                _as_mapping(_as_mapping(repeat, f"confirmation repeat {cell.key}").get("benchmark"),
                            f"confirmation benchmark {cell.key}"),
                cell.expected_winner,
                f"confirmation/{cell.key}/repeat{index}",
            )
            for index, repeat in enumerate(repeats)
        ]
        history_gate_pass = bool(discovery_cell["gate_pass"]) and all(
            bool(repeat["gate_pass"]) for repeat in confirmation_repeats
        )
        cell_evidence[cell.key] = {
            "expected_winner": cell.expected_winner,
            "discovery_single_repeat": discovery_cell,
            "confirmation_two_repeats": confirmation_repeats,
            "historical_gate_pass": history_gate_pass,
        }
    return {
        "discovery": {"artifact": discovery_identity, "identity": discovery_device},
        "confirmation": {"artifact": confirmation_identity, "identity": confirmation_device},
        "raw_abi_exactness": exact_by_batch,
        "cells": cell_evidence,
        "historical_identity_gate_pass": True,
    }


def _clone(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.clone()


def _raw_abi_exactness(
    functions: Mapping[str, Callable[..., None]],
    case: shared.Case,
    x: object,
    contract: str,
    seed: int,
) -> dict[str, object]:
    initial, final = shared._states(contract, case, seed)
    outputs = {
        label: shared._invoke(fn, x, _clone(initial), _clone(final))
        for label, fn in functions.items()
    }
    baseline = outputs["baseline"]
    return {
        label: shared._compare(
            f"fixed_batch_release_gate/{case.name}/{contract}/{label}_vs_baseline",
            outputs[label],
            baseline,
        )
        for label in ("vshard2_p2", "vshard4_p2")
    }


def _check_args(args: argparse.Namespace) -> None:
    if args.samples != SAMPLES_PER_REPEAT or args.repeats != REPEATS or args.warmup != WARMUP:
        raise ValueError(
            "this pre-registered release gate fixes --samples=1000, --repeats=2, and --warmup=100"
        )
    if len(RELEASE_CELLS) != 11:
        raise RuntimeError(f"release scope corruption: expected 11 cells, got {len(RELEASE_CELLS)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_REPEAT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--discovery-json", type=Path, default=DEFAULT_DISCOVERY_JSON)
    parser.add_argument("--confirmation-json", type=Path, default=DEFAULT_CONFIRMATION_JSON)
    parser.add_argument("--describe", action="store_true", help="write the fixed 11-cell release scope without CUDA")
    args = parser.parse_args()
    _check_args(args)
    result = _initial_result(args)
    if args.describe:
        result["describe_only"] = True
        _write(args.json, result)
        print(f"wrote fixed-batch release-gate preregistration {args.json}")
        return
    if os.environ.get(CLEAN_GPU_GATE_ENV) != "1":
        raise RuntimeError(
            "refusing a direct GPU run: use run_clean_fixed_batch_release_gate_audit.sh so "
            f"{CLEAN_GPU_GATE_ENV}=1 is set only after its PRE clean-GPU check"
        )

    history = _load_historical_evidence(args)
    result["history_evidence"] = history
    result["gates"]["historical_identity_gate"]["passed"] = True  # type: ignore[index]
    _write(args.json, result)

    import torch
    from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2
    from assignment02.team.c1_flashkda.challenge_vshard4_prefetch2 import vshard4_prefetch2
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shared.torch = torch
    shared.common = common
    import flash_kda

    functions: dict[str, Callable[..., None]] = {
        "baseline": flash_kda.fwd,
        "vshard2_p2": prefetch2.fwd,
        "vshard4_p2": vshard4_prefetch2.fwd,
    }
    result["current_identity"] = {
        "device": shared._device_identity(),
        "extension": shared._identity(),
        "clean_gpu_shell_gate": {"environment_name": CLEAN_GPU_GATE_ENV, "passed": True},
    }
    result["gates"]["clean_gpu_shell_gate"]["passed"] = True  # type: ignore[index]
    result["gates"]["device_gate"]["passed"] = True  # type: ignore[index]
    result["gates"]["audited_extension_sha256_gate"]["passed"] = True  # type: ignore[index]
    _write(args.json, result)

    for case_index, case in enumerate(CASES):
        print(f"raw ABI exactness {case.name}: contracts={RAW_CONTRACTS}")
        x = shared._make_inputs(case, args.seed + case_index * 10_007)
        result["raw_abi_correctness"][case.name] = {}  # type: ignore[index]
        for contract_index, contract in enumerate(RAW_CONTRACTS):
            result["raw_abi_correctness"][case.name][contract] = _raw_abi_exactness(  # type: ignore[index]
                functions,
                case,
                x,
                contract,
                args.seed + case_index * 10_007 + contract_index * 101,
            )
            _write(args.json, result)
        del x
        torch.cuda.empty_cache()
    result["gates"]["current_raw_abi_exact_gate"]["passed"] = True  # type: ignore[index]
    _write(args.json, result)

    historical_cells = _as_mapping(history["cells"], "history cells")
    for cell_index, cell in enumerate(RELEASE_CELLS):
        print(
            f"release benchmark {cell.key}: expected={cell.expected_winner}, "
            f"repeats={REPEATS}, samples={SAMPLES_PER_REPEAT}"
        )
        case = CASES_BY_BATCH[cell.batch]
        repeats: list[dict[str, object]] = []
        result["new_measurements"][cell.key] = {  # type: ignore[index]
            "expected_winner": cell.expected_winner,
            "repeats": repeats,
        }
        for repeat_index in range(REPEATS):
            repeat_seed = args.seed + cell_index * 100_003 + repeat_index * 1_009
            x = shared._make_inputs(case, repeat_seed)
            benchmark = shared._benchmark(
                functions, case, x, cell.contract, repeat_seed + 101, args.warmup, args.samples
            )
            repeats.append(
                {
                    "repeat_index": repeat_index,
                    "input_seed": repeat_seed,
                    "state_seed": repeat_seed + 101,
                    "benchmark": benchmark,
                    "recomputed_gate": _assess_expected(
                        _as_mapping(benchmark, f"new benchmark {cell.key}"),
                        cell.expected_winner,
                        f"new/{cell.key}/repeat{repeat_index}",
                    ),
                }
            )
            del x
            torch.cuda.empty_cache()
            _write(args.json, result)

        historical = _as_mapping(historical_cells.get(cell.key), f"history cell {cell.key}")
        history_pass = bool(historical.get("historical_gate_pass"))
        new_pass = all(
            bool(_as_mapping(repeat, f"new repeat {cell.key}")["recomputed_gate"]["gate_pass"])
            for repeat in repeats
        )
        if history_pass and new_pass:
            status = "eligible_for_public_fla_integration_review"
            reason = "all historical and two new raw-wrapper repeat gates passed"
            result["public_fla_integration_candidate_mapping"][cell.key] = cell.expected_winner  # type: ignore[index]
        elif not history_pass:
            status = "baseline"
            reason = "historical discovery or confirmation evidence did not meet this cell's gate"
        else:
            status = "baseline"
            reason = "at least one new repeat did not meet this cell's gate"
        result["cell_status"][cell.key] = {  # type: ignore[index]
            "status": status,
            "expected_winner": cell.expected_winner,
            "historical_gate_pass": history_pass,
            "new_gate_pass": new_pass,
            "reason": reason,
        }
        if status == "baseline":
            result["fallback_mapping"][cell.key] = "baseline"  # type: ignore[index]
        _write(args.json, result)

    result["complete"] = True
    _write(args.json, result)
    eligible = _as_mapping(
        result["public_fla_integration_candidate_mapping"], "public-FLA integration candidate mapping"
    )
    fallback = _as_mapping(result["fallback_mapping"], "fallback mapping")
    print(f"wrote {args.json}; public_fla_integration_candidates={dict(eligible)}; fallback={dict(fallback)}")


if __name__ == "__main__":
    main()
