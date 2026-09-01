#!/usr/bin/env python3
"""Apply the Phase-1 candidate's preregistered first-allocation gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
PERCENTILES = ("p50_ms", "p95_ms", "p99_ms")
MIN_SPEEDUP_X = 1.02


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def recompute_summary(raw: object, label: str) -> dict[str, float | int]:
    expect(isinstance(raw, list) and len(raw) == 1000, f"{label}: requires exactly 1000 raw samples")
    values = [float(value) for value in raw]
    expect(all(math.isfinite(value) and value > 0.0 for value in values), f"{label}: raw samples must be finite and positive")
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def require_same_summary(recorded: object, recomputed: dict[str, float | int], label: str) -> None:
    expect(isinstance(recorded, dict) and set(recorded) == set(recomputed), f"{label}: summary fields changed")
    expect(recorded.get("samples") == recomputed["samples"], f"{label}: sample count summary mismatch")
    for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"):
        value = recorded.get(key)
        expect(
            isinstance(value, (int, float))
            and math.isclose(float(value), float(recomputed[key]), rel_tol=1e-12, abs_tol=1e-12),
            f"{label}: {key} was not recomputed from raw samples",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptxas", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--require-eligible", action="store_true")
    args = parser.parse_args()
    ptxas, result = load(args.ptxas), load(args.result)
    formal = ptxas.get("phase1pf_formal_bf16_fixed_both_state")
    expect(isinstance(formal, dict), "missing formal BF16 fixed-state candidate ptxas record")
    zero_spill = formal.get("spill_store_bytes") == 0 and formal.get("spill_load_bytes") == 0
    shared = bool(ptxas.get("resource_evidence", {}).get("candidate_shared_memory_evidence"))
    expect(result.get("candidate") == "fwd_vshard4_p2_phase1pf", "wrong candidate result")
    device_name = result.get("device")
    expect(isinstance(device_name, str) and "B300" in device_name.upper(), "result was not measured on B300")
    expect(result.get("capability") == [10, 3], "result was not measured on SM10.3")
    expect(result.get("multiprocessor_count") == 148, "result was not measured on a 148-SM B300")
    h12 = result.get("h12")
    expect(isinstance(h12, dict) and h12.get("shape") == {"B": 1, "T": 8192, "H": 12, "K": 128, "V": 128}, "wrong H12 shape")
    exact = h12.get("exact")
    expect(isinstance(exact, dict) and set(exact) == set(CONTRACTS), "missing all four exact contracts")
    for contract in CONTRACTS:
        exact_contract = exact[contract]
        expected_exact_keys = {"vshard2_p2s3_vs_baseline", "vshard4_p2s3_vs_baseline", "phase1pf_vs_baseline"}
        expect(isinstance(exact_contract, dict) and set(exact_contract) == expected_exact_keys, f"{contract}: exact variant set changed")
        for variant_key in expected_exact_keys:
            comparison = exact_contract.get(variant_key)
            expect(isinstance(comparison, dict) and comparison.get("output_exact") is True, f"{contract}/{variant_key}: output not exact")
            if contract == "none":
                expect(comparison.get("final_state_present") is False, f"{contract}/{variant_key}: unexpected final state")
            else:
                expect(comparison.get("final_state_exact") is True, f"{contract}/{variant_key}: final state not exact")
    repeats = h12.get("repeats")
    expect(isinstance(repeats, list) and len(repeats) >= 2, "requires at least two timing repeats")
    values: dict[str, list[float]] = {q: [] for q in PERCENTILES}
    for repeat_index, repeat in enumerate(repeats):
        expect(isinstance(repeat, dict) and set(repeat) == set(CONTRACTS), f"repeat {repeat_index}: missing contracts")
        for contract in CONTRACTS:
            bench = repeat[contract]
            expect(bench.get("event_contract") == "four-path cyclic rotation; one complete public-wrapper call per CUDA event; workspace allocation included", f"repeat {repeat_index}/{contract}: event contract changed")
            paths = bench.get("paths")
            expect(isinstance(paths, dict) and set(paths) == {"baseline", "vshard2_p2s3", "vshard4_p2s3", "phase1pf"}, f"repeat {repeat_index}/{contract}: wrong paths")
            raw_paths = bench.get("raw_samples_ms")
            expect(isinstance(raw_paths, dict) and set(raw_paths) == set(paths), f"repeat {repeat_index}/{contract}: wrong raw path set")
            recomputed_paths: dict[str, dict[str, float | int]] = {}
            for name, path in paths.items():
                recomputed_paths[name] = recompute_summary(raw_paths[name], f"repeat {repeat_index}/{contract}/{name}")
                require_same_summary(path, recomputed_paths[name], f"repeat {repeat_index}/{contract}/{name}")
            speedups = bench.get("candidate_speedup_vs_vshard4_p2s3_x")
            expect(isinstance(speedups, dict), f"repeat {repeat_index}/{contract}: missing speedups")
            for q in PERCENTILES:
                value = speedups.get(q)
                recomputed_speedup = float(recomputed_paths["vshard4_p2s3"][q]) / float(recomputed_paths["phase1pf"][q])
                expect(
                    isinstance(value, (int, float))
                    and math.isclose(float(value), recomputed_speedup, rel_tol=1e-12, abs_tol=1e-12),
                    f"repeat {repeat_index}/{contract}/{q}: speedup was not recomputed from raw samples",
                )
                values[q].append(recomputed_speedup)
    minimums = {q: min(value) for q, value in values.items()}
    performance_pass = all(value >= MIN_SPEEDUP_X for value in minimums.values())
    eligible = zero_spill and shared and performance_pass
    summary = {
        "candidate": "fwd_vshard4_p2_phase1pf", "candidate_status": "non-production; no dispatch/current files changed",
        "pre_registered_gate": {"minimum_speedup_x": MIN_SPEEDUP_X, "percentiles": list(PERCENTILES), "contracts": list(CONTRACTS), "repeats": len(repeats), "samples_per_path_per_repeat": 1000, "formal_bf16_fixed_zero_spill": True, "shared_memory_evidence": True, "all_contracts_exact": True},
        "minimum_speedup_over_all_repeat_contract_cells_x": minimums,
        "device_gate": {"name": device_name, "capability": [10, 3], "multiprocessor_count": 148, "passed": True},
        "zero_spill": zero_spill, "shared_memory_evidence": shared,
        "performance_pass": performance_pass, "eligible_for_independent_confirmation": eligible,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_eligible and not eligible:
        raise RuntimeError("STOP: candidate failed first-allocation preregistered gate")


if __name__ == "__main__":
    main()
