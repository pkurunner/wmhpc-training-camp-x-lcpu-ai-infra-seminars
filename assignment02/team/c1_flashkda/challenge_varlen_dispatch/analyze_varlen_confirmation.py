#!/usr/bin/env python3
"""Independently audit a packed-varlen confirmation JSON from raw samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


EXPECTED_SEED = 20260829
EXPECTED_SAMPLES = 1000
EXPECTED_REPEATS = 2
MIN_MARGIN = 0.02
PERCENTILES = {"p50": 0.50, "p95": 0.95, "p99": 0.99}
VARIANTS = ("baseline", "vshard2_p2", "vshard4_p2")
EXPECTED_HELPER_SHA256 = "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
EXPECTED_SO_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
PROMOTION_CELLS = {
    "equal_n2_h12_t2048": {
        "none": "vshard4_p2",
        "fp32_final_only": "vshard4_p2",
        "fp32_both": "vshard4_p2",
    },
    "equal_n4_h12_t2048": {
        "none": "vshard2_p2",
        "fp32_final_only": "vshard2_p2",
        "fp32_both": "vshard4_p2",
    },
    "mixed_n6_h12_t8192": {
        "none": "vshard2_p2",
        "fp32_final_only": "vshard2_p2",
    },
    "skew_n6_h12_t12288": {
        "none": "vshard2_p2",
        "fp32_final_only": "vshard2_p2",
        "fp32_both": "vshard4_p2",
    },
}
RECORD_ONLY = ("mixed_n6_h12_t8192", "fp32_both")
PUBLIC_CONTRACTS = ("none", "fp32_final_only", "fp32_both")
RAW_CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def percentile(values: list[float], quantile: float) -> float:
    require(len(values) == EXPECTED_SAMPLES, f"expected {EXPECTED_SAMPLES} samples")
    require(all(math.isfinite(value) and value > 0.0 for value in values), "invalid sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def audit_exactness(node: object, path: str = "root") -> int:
    """Require every stored `*_exact` flag recursively; return flag count."""
    count = 0
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}/{key}"
            if key.endswith("_exact"):
                require(value is True, f"false exactness flag: {child}")
                count += 1
            count += audit_exactness(value, child)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            count += audit_exactness(value, f"{path}/{index}")
    return count


def audit(path: Path, expected_sha256: str | None) -> dict[str, object]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        require(digest == expected_sha256, f"artifact SHA mismatch: {digest}")
    data = json.loads(payload)

    require(data["complete"] is True, "artifact is incomplete")
    require(data["seed"] == EXPECTED_SEED, "seed drift")
    prereg = data["preregistration"]
    require(prereg["fixed_seed"] == EXPECTED_SEED, "preregistered seed drift")
    require(prereg["promotion_cell_count"] == 11, "promotion scope drift")
    require(prereg["repeats_per_public_cell"] == EXPECTED_REPEATS, "repeat-count drift")
    require(prereg["cyclic_cuda_event_samples_per_repeat"] == EXPECTED_SAMPLES, "sample drift")
    require(math.isclose(prereg["minimum_expected_winner_margin_over_runner_up"], MIN_MARGIN), "margin drift")
    require(prereg["promotion_cells"] == PROMOTION_CELLS, "promotion mapping drift")

    identity = data["identity"]
    device = identity["device"]
    require("B300" in device["name"].upper(), "not B300")
    require(device["capability"] == [10, 3] and device["multiprocessor_count"] == 148, "device drift")
    require(identity["extension"]["sha256"] == EXPECTED_SO_SHA256, "extension drift")
    helper = identity["pinned_torch_reference_helper"]
    require(helper["sha256"] == EXPECTED_HELPER_SHA256 and helper["no_build"] is True, "helper drift")
    require(helper["intercepted_names"] == ["sigmoid_ext"], "helper interception drift")

    gates = data["gates"]
    for gate_name in (
        "clean_gpu_shell_gate",
        "device_gate",
        "audited_extension_sha256_gate",
        "raw_abi_exact_gate",
        "pinned_torch_reference_gate",
        "confirmation_gate",
    ):
        require(gates[gate_name]["passed"] is True, f"gate failed: {gate_name}")

    correctness = data["raw_abi_correctness"]
    require(set(correctness) == set(PROMOTION_CELLS), "correctness case drift")
    for case_name, contracts in correctness.items():
        require(set(contracts) == set(RAW_CONTRACTS), f"{case_name}: raw contract drift")
        for contract, result in contracts.items():
            require(result["input_immutability_exact"] is True, f"{case_name}/{contract}: mutation")
            require("vshard2_p2" in result and "vshard4_p2" in result, "candidate missing")
            if contract in PUBLIC_CONTRACTS:
                require("baseline_vs_pinned_torch_ref" in result, "Torch reference missing")
    exact_flag_count = audit_exactness(correctness, "raw_abi_correctness")

    benchmarks = data["raw_wrapper_public_contract_benchmarks"]
    require(set(benchmarks) == set(PROMOTION_CELLS), "benchmark case drift")
    promotion_rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    global_min_margin = float("inf")
    for case_name, contracts in benchmarks.items():
        require(set(contracts) == set(PUBLIC_CONTRACTS), f"{case_name}: public contract drift")
        for contract, cell in contracts.items():
            expected = PROMOTION_CELLS.get(case_name, {}).get(contract)
            is_record_only = (case_name, contract) == RECORD_ONLY
            require((expected is None) == is_record_only, f"{case_name}/{contract}: scope drift")
            repeats = cell["repeats"]
            require(len(repeats) == EXPECTED_REPEATS, f"{case_name}/{contract}: repeat drift")
            repeat_rows = []
            for repeat_index, repeat in enumerate(repeats):
                require(repeat["repeat_index"] == repeat_index, "repeat index drift")
                require(repeat["input_immutability_exact"] is True, "benchmark input mutation")
                raw = repeat["benchmark"]["raw_samples_ms"]
                require(set(raw) == set(VARIANTS), "variant drift")
                summaries = {
                    variant: {
                        name: percentile([float(v) for v in raw[variant]], quantile)
                        for name, quantile in PERCENTILES.items()
                    }
                    for variant in VARIANTS
                }
                winners: dict[str, str] = {}
                margins: dict[str, float] = {}
                for name in PERCENTILES:
                    ranked = sorted((values[name], variant) for variant, values in summaries.items())
                    winners[name] = ranked[0][1]
                    margins[name] = ranked[1][0] / ranked[0][0] - 1.0
                recorded = repeat["independent_summary"]
                require(recorded["winner_by_percentile"] == winners, "recorded winner mismatch")
                for name, margin in margins.items():
                    require(
                        math.isclose(recorded["winner_margin_over_runner_up"][name], margin, rel_tol=0.0, abs_tol=1e-12),
                        f"recorded margin mismatch: {case_name}/{contract}/{repeat_index}/{name}",
                    )
                if expected is not None:
                    require(all(winner == expected for winner in winners.values()), "winner gate failed")
                    require(all(margin >= MIN_MARGIN for margin in margins.values()), "margin gate failed")
                    global_min_margin = min(global_min_margin, *margins.values())
                repeat_rows.append(
                    {
                        "repeat": repeat_index,
                        "winner_by_percentile": winners,
                        "margin_by_percentile": margins,
                        "minimum_margin": min(margins.values()),
                    }
                )
            row = {"case": case_name, "contract": contract, "expected": expected, "repeats": repeat_rows}
            (record_rows if is_record_only else promotion_rows).append(row)

    assessment = data["confirmation_assessment"]
    require(assessment["scope_cell_count"] == 11, "assessment scope drift")
    require(assessment["confirmation_gate_pass"] is True, "assessment failed")
    require(
        assessment["decision"] == "eligible_for_separate_packed_varlen_dispatch_integration_review",
        "unexpected decision",
    )
    return {
        "artifact": str(path.resolve()),
        "artifact_sha256": digest,
        "independent_audit_pass": True,
        "exact_flag_count": exact_flag_count,
        "promotion_cell_count": len(promotion_rows),
        "record_only_cell_count": len(record_rows),
        "global_minimum_promotion_margin": global_min_margin,
        "promotion_cells": promotion_rows,
        "record_only_cells": record_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    summary = audit(args.artifact, args.expected_sha256)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
