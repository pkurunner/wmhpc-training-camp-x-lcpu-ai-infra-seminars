#!/usr/bin/env python3
"""Apply the pre-registered P2S4 gates without registering any dispatch route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
PERCENTILES = ("p50_ms", "p95_ms", "p99_ms")
MIN_SPEEDUP_X = 1.02


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_exact(item: dict[str, Any], context: str) -> None:
    exact = item.get("exact")
    _expect(isinstance(exact, dict), f"{context}: missing exact object")
    _expect(set(exact) == set(CONTRACTS), f"{context}: exact contracts must be exactly {CONTRACTS}")
    for contract in CONTRACTS:
        candidate = exact[contract].get("vshard4_p2s4_vs_baseline")
        _expect(isinstance(candidate, dict), f"{context}/{contract}: missing P2S4 exact comparison")
        _expect(candidate.get("output_exact") is True, f"{context}/{contract}: candidate output is not exact")
        if contract == "none":
            _expect(
                candidate.get("final_state_present") is False,
                f"{context}/{contract}: unexpected final-state contract",
            )
        else:
            _expect(
                candidate.get("final_state_exact") is True,
                f"{context}/{contract}: candidate final state is not exact",
            )


def _validate_small(result: dict[str, Any]) -> None:
    matrix = result.get("small_matrix")
    _expect(isinstance(matrix, dict), "small result: missing small_matrix")
    _expect(set(matrix) == {"H1", "H2", "H4"}, "small result: expected exactly H1/H2/H4")
    for name in ("H1", "H2", "H4"):
        item = matrix[name]
        _expect(isinstance(item, dict), f"small result/{name}: malformed entry")
        shape = item.get("shape")
        _expect(
            shape == {"B": 1, "T": 256, "H": int(name[1:]), "K": 128, "V": 128},
            f"small result/{name}: unexpected shape {shape!r}",
        )
        _validate_exact(item, "small result/" + name)


def _validate_ptxas(result: dict[str, Any]) -> dict[str, Any]:
    formal = result.get("p2s4_formal_bf16_fixed_both_state")
    _expect(isinstance(formal, dict), "ptxas: missing formal P2S4 BF16 fixed-state record")
    _expect(formal.get("variant") == "vshard4_p2s4", "ptxas: formal record is not P2S4")
    _expect(formal.get("has_state_in") is True and formal.get("has_state_out") is True, "ptxas: formal record is not both-state")
    _expect(formal.get("state_fp32") is False and formal.get("is_varlen") is False, "ptxas: formal record is not fixed BF16")
    _expect(formal.get("spill_store_bytes") == 0 and formal.get("spill_load_bytes") == 0, "ptxas: P2S4 BF16 fixed-state spills")
    resource = result.get("resource_evidence")
    _expect(isinstance(resource, dict), "ptxas: missing CUBIN resource evidence")
    records = resource.get("candidate_shared_records")
    _expect(isinstance(records, list) and records, "ptxas: missing P2S4 SHARED/SMEM resource record")
    _expect(
        all(
            isinstance(record, dict)
            and isinstance(record.get("shared_memory_bytes"), int)
            and record["shared_memory_bytes"] >= 0
            for record in records
        ),
        "ptxas: malformed shared-memory byte count",
    )
    return {"formal": formal, "shared_records": records}


def _validate_h12(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    _expect(result.get("candidate") == "fwd_vshard4_p2s4", f"{path}: unexpected candidate")
    _expect(result.get("candidate_status") == "non-production; dispatch registration is forbidden", f"{path}: candidate status changed")
    h12 = result.get("h12")
    _expect(isinstance(h12, dict), f"{path}: missing h12 object")
    _expect(h12.get("shape") == {"B": 1, "T": 8192, "H": 12, "K": 128, "V": 128}, f"{path}: unexpected H12 shape")
    _validate_exact(h12, str(path))
    benchmarks = h12.get("benchmarks")
    _expect(isinstance(benchmarks, dict), f"{path}: missing H12 public-wrapper benchmarks")
    _expect(set(benchmarks) == set(CONTRACTS), f"{path}: benchmark contracts must be exactly {CONTRACTS}")
    speedups: dict[str, dict[str, float]] = {}
    for contract in CONTRACTS:
        benchmark = benchmarks[contract]
        _expect(isinstance(benchmark, dict), f"{path}/{contract}: malformed benchmark")
        _expect(
            benchmark.get("event_contract")
            == "four-path cyclic rotation; one complete public-wrapper call per CUDA event; workspace allocation included",
            f"{path}/{contract}: benchmark contract changed",
        )
        paths = benchmark.get("paths")
        _expect(isinstance(paths, dict), f"{path}/{contract}: missing paths")
        _expect(set(paths) == {"baseline", "vshard2_p2s3", "vshard4_p2s3", "vshard4_p2s4"}, f"{path}/{contract}: wrong paths")
        for label, data in paths.items():
            _expect(isinstance(data, dict), f"{path}/{contract}/{label}: malformed summary")
            _expect(data.get("samples") == 1000, f"{path}/{contract}/{label}: must retain exactly 1000 samples")
        values = benchmark.get("candidate_speedup_vs_vshard4_p2s3_x")
        _expect(isinstance(values, dict), f"{path}/{contract}: missing P2S3 speedups")
        speedups[contract] = {}
        for percentile in PERCENTILES:
            value = values.get(percentile)
            _expect(isinstance(value, (int, float)) and value > 0, f"{path}/{contract}/{percentile}: invalid speedup")
            speedups[contract][percentile] = float(value)
    allocation = result.get("allocation")
    _expect(isinstance(allocation, dict), f"{path}: missing allocation identity")
    job = allocation.get("slurm_job_id")
    _expect(isinstance(job, str) and job and job != "none", f"{path}: no Slurm allocation identity")
    return {"path": str(path.resolve()), "job_id": job, "speedups": speedups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptxas", type=Path, required=True)
    parser.add_argument("--small-result", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True, help="one H12 audit JSON per clean allocation")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--require-publication", action="store_true", help="fail unless exactly two independent allocations meet every gate")
    args = parser.parse_args()
    ptxas = _validate_ptxas(_load(args.ptxas))
    _validate_small(_load(args.small_result))
    allocations = [_validate_h12(path, _load(path)) for path in args.result]
    minimums: dict[str, float] = {
        percentile: min(
            allocation["speedups"][contract][percentile]
            for allocation in allocations
            for contract in CONTRACTS
        )
        for percentile in PERCENTILES
    }
    performance_pass = all(value >= MIN_SPEEDUP_X for value in minimums.values())
    distinct_jobs = len({allocation["job_id"] for allocation in allocations}) == len(allocations)
    publication_eligible = len(allocations) == 2 and distinct_jobs and performance_pass
    summary = {
        "candidate": "fwd_vshard4_p2s4",
        "candidate_status": "non-production; no dispatch/current files were changed",
        "pre_registered_gate": {
            "minimum_speedup_x": MIN_SPEEDUP_X,
            "percentiles": list(PERCENTILES),
            "contracts": list(CONTRACTS),
            "required_clean_allocations": 2,
            "ptxas_bf16_fixed_zero_spill": True,
            "shared_memory_evidence": True,
            "small_h1_h2_h4_all_contracts_exact": True,
            "h12_all_contracts_exact": True,
        },
        "ptxas": ptxas,
        "allocations": allocations,
        "minimum_speedup_over_all_cells_x": minimums,
        "performance_pass": performance_pass,
        "distinct_slurm_jobs": distinct_jobs,
        "publication_eligible": publication_eligible,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.require_publication and not publication_eligible:
        raise RuntimeError("STOP: P2S4 cannot be published; two independent allocations and every 2% P50/P95/P99 cell are required")


if __name__ == "__main__":
    main()
