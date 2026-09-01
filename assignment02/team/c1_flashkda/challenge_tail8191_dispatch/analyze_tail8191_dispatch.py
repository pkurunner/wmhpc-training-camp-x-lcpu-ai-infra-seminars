#!/usr/bin/env python3
"""Stdlib-only, fail-closed auditor for the tail-8191 public-route protocol.

It recomputes every P50/P95/P99 decision from raw CUDA-event samples.  A
performance miss is a valid negative result; malformed identity/evidence is an
audit error.  ``--chain`` is the only mode that can label the two allocation
artifacts ``eligible_for_public_freeze``; it still never modifies a source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


CONTRACTS = ("none", "fp32_final_only")
PERCENTILES = ("p50", "p95", "p99")
PATHS = ("pinned_public", "c1_test_route_public")
SAMPLES = 1000
REPEATS = 2
MIN_MARGIN = 0.02
EXPECTED_EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"


class AuditError(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def mapping(value: object, label: str) -> Mapping[str, Any]:
    require(type(value) is dict, f"{label} must be a JSON object")
    return value  # type: ignore[return-value]


def array(value: object, label: str) -> list[Any]:
    require(type(value) is list, f"{label} must be a JSON array")
    return value  # type: ignore[return-value]


def exact_bool(value: object, label: str) -> bool:
    require(type(value) is bool, f"{label} must be an exact bool")
    return value  # type: ignore[return-value]


def integer(value: object, label: str) -> int:
    require(type(value) is int, f"{label} must be an integer (bool rejected)")
    return value  # type: ignore[return-value]


def number(value: object, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric (bool rejected)")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_text(value: object, label: str) -> str:
    require(type(value) is str and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), f"{label} must be lowercase SHA256")
    return value  # type: ignore[return-value]


def atomic_write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path, expected: str, label: str) -> tuple[Mapping[str, Any], str]:
    sha_text(expected, f"{label}.expected_sha256")
    actual = sha(path)
    require(actual == expected, f"{label}: artifact SHA mismatch")
    try:
        return mapping(json.loads(path.read_text(encoding="utf-8")), label), actual
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label}: invalid JSON") from exc


def percentile(values: Sequence[float], q: float) -> float:
    require(len(values) == SAMPLES, "raw sample count drift")
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lo = int(point)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def validate_samples(value: object, label: str) -> list[float]:
    values = [number(item, f"{label}[{index}]") for index, item in enumerate(array(value, label))]
    require(len(values) == SAMPLES and all(item > 0.0 for item in values), f"{label}: requires {SAMPLES} positive samples")
    return values


def validate_identity(value: object, expected_runner_sha: str, label: str) -> Mapping[str, Any]:
    identity = mapping(value, label)
    runner = mapping(identity.get("runner"), f"{label}.runner")
    require(runner.get("sha256") == expected_runner_sha and exact_bool(runner.get("gate_pass"), f"{label}.runner.gate_pass"), f"{label}: runner identity mismatch")
    device = mapping(identity.get("device"), f"{label}.device")
    require(device.get("capability") == [10, 3] and integer(device.get("multiprocessor_count"), f"{label}.device.sm") == 148 and "B300" in str(device.get("name", "")).upper() and exact_bool(device.get("gate_pass"), f"{label}.device.gate"), f"{label}: B300 identity mismatch")
    uuid = identity.get("gpu_uuid")
    require(type(uuid) is str and uuid, f"{label}: GPU UUID missing")
    extension = mapping(identity.get("extension"), f"{label}.extension")
    require(extension.get("sha256") == EXPECTED_EXTENSION_SHA256 and exact_bool(extension.get("gate_pass"), f"{label}.extension.gate"), f"{label}: extension identity mismatch")
    commits = mapping(identity.get("commits"), f"{label}.commits")
    require(set(commits) == {"patched", "reference", "fla"}, f"{label}: commit identity drift")
    sources = mapping(identity.get("c1_sources"), f"{label}.c1_sources")
    require(set(sources) == {"auto_dispatch", "fla_backend", "harness"}, f"{label}: C1 source set drift")
    for source_name, source in sources.items():
        sha_text(mapping(source, f"{label}.c1_sources.{source_name}").get("sha256"), f"{label}.c1_sources.{source_name}.sha")
    reference = mapping(identity.get("reference_torch_ref"), f"{label}.reference")
    sha_text(reference.get("sha256"), f"{label}.reference.sha")
    return identity


def validate_raw_correctness(value: object, label: str) -> None:
    records = mapping(value, label)
    require(set(records) == set(CONTRACTS), f"{label}: raw contract set drift")
    for contract in CONTRACTS:
        record = mapping(records[contract], f"{label}.{contract}")
        require(exact_bool(record.get("passed"), f"{label}.{contract}.passed"), f"{label}.{contract}: raw path failed")
        for comparison_name in ("baseline_vs_pinned_torch_reference", "vshard4_vs_pinned_torch_reference", "vshard4_vs_baseline"):
            comparison = mapping(record.get(comparison_name), f"{label}.{contract}.{comparison_name}")
            require(exact_bool(comparison.get("output_exact"), f"{label}.{contract}.{comparison_name}.output"), f"{label}.{contract}: output exactness failed")
            if contract == "none":
                require(comparison.get("final_state_present") is False, f"{label}.{contract}: unexpected final state")
            else:
                require(exact_bool(comparison.get("final_state_exact"), f"{label}.{contract}.{comparison_name}.final"), f"{label}.{contract}: final exactness failed")
        immutable = mapping(record.get("immutability"), f"{label}.{contract}.immutability")
        require(set(immutable) == {"reference", "baseline", "vshard4"}, f"{label}.{contract}: immutability path set drift")
        for name, evidence in immutable.items():
            detail = mapping(evidence, f"{label}.{contract}.immutability.{name}")
            require(exact_bool(detail.get("input_immutability_exact"), f"{label}.{contract}.{name}.input") and exact_bool(detail.get("initial_state_immutability_exact"), f"{label}.{contract}.{name}.initial"), f"{label}.{contract}: mutation evidence failed")


def validate_negative_controls(value: object, label: str) -> None:
    controls = mapping(value, label)
    require(exact_bool(controls.get("production_source_unmodified"), f"{label}.unmodified") and exact_bool(controls.get("passed"), f"{label}.passed"), f"{label}: source/matrix control failure")
    contracts = mapping(controls.get("negative_contracts"), f"{label}.negative_contracts")
    require(set(contracts) == {"bf16_both", "fp32_both"}, f"{label}: negative contracts drift")
    for contract, record in contracts.items():
        control = mapping(record, f"{label}.{contract}")
        require(control.get("requested_variant") == "baseline" and control.get("chosen_variant") == "baseline" and exact_bool(control.get("passed"), f"{label}.{contract}.passed"), f"{label}: T8191 {contract} is not pre-launch baseline")


def validate_repeat(value: object, process_index: int, repeat_index: int, label: str) -> dict[str, object]:
    repeat = mapping(value, label)
    require(integer(repeat.get("process_index"), f"{label}.process") == process_index and integer(repeat.get("repeat_index"), f"{label}.repeat") == repeat_index, f"{label}: process/repeat identity drift")
    require(exact_bool(repeat.get("input_immutability_exact"), f"{label}.inputs") and exact_bool(repeat.get("initial_state_immutability_exact"), f"{label}.initial"), f"{label}: mutation gate failed")
    precheck = mapping(repeat.get("public_precheck"), f"{label}.precheck")
    c1_proof = mapping(precheck.get("c1_test_route"), f"{label}.c1_proof")
    pinned_proof = mapping(precheck.get("pinned"), f"{label}.pinned_proof")
    exact = mapping(precheck.get("exact"), f"{label}.precheck_exact")
    require(integer(c1_proof.get("c1_spy_delta"), f"{label}.c1.delta") == 1 and integer(c1_proof.get("pinned_spy_delta"), f"{label}.c1.pinned_delta") == 0 and exact_bool(c1_proof.get("passed"), f"{label}.c1.passed"), f"{label}: C1 public-route proof failed")
    decision = mapping(c1_proof.get("decision"), f"{label}.decision")
    require(decision.get("chosen_variant") == "vshard4_p2" and decision.get("test_only_route") is True and decision.get("production_source_mutated") is False, f"{label}: test-only route decision drift")
    require(integer(pinned_proof.get("c1_spy_delta"), f"{label}.pinned.c1_delta") == 0 and integer(pinned_proof.get("pinned_spy_delta"), f"{label}.pinned.delta") == 1 and exact_bool(pinned_proof.get("passed"), f"{label}.pinned.passed"), f"{label}: pinned public-route proof failed")
    require(exact_bool(exact.get("output_exact"), f"{label}.precheck.output"), f"{label}: public output mismatch")
    paths = mapping(repeat.get("paths"), f"{label}.paths")
    raw = mapping(repeat.get("raw_samples_ms"), f"{label}.raw")
    require(set(paths) == set(PATHS) and set(raw) == set(PATHS), f"{label}: public path set drift")
    summaries: dict[str, dict[str, float]] = {}
    for path in PATHS:
        samples = validate_samples(raw[path], f"{label}.raw.{path}")
        observed = mapping(paths[path], f"{label}.paths.{path}")
        recomputed = {"mean_ms": statistics.fmean(samples), "p50_ms": percentile(samples, 0.50), "p95_ms": percentile(samples, 0.95), "p99_ms": percentile(samples, 0.99)}
        require(integer(observed.get("samples"), f"{label}.{path}.samples") == SAMPLES, f"{label}: summary sample drift")
        for metric, metric_value in recomputed.items():
            require(close(number(observed.get(metric), f"{label}.{path}.{metric}"), metric_value), f"{label}: {path} {metric} summary mismatch")
        summaries[path] = recomputed
    margins = mapping(repeat.get("c1_margin_over_pinned_by_percentile"), f"{label}.margins")
    recomputed_margins = {name: summaries["pinned_public"][f"{name}_ms"] / summaries["c1_test_route_public"][f"{name}_ms"] - 1.0 for name in PERCENTILES}
    for name, margin in recomputed_margins.items():
        require(close(number(margins.get(name), f"{label}.margin.{name}"), margin), f"{label}: margin recomputation mismatch")
    return {"margins": recomputed_margins, "repeat_gate_pass": all(value >= MIN_MARGIN for value in recomputed_margins.values())}


def validate_main(record: Mapping[str, Any], expected_runner_sha: str, allocation: str, process_index: int, label: str) -> dict[str, object]:
    require(integer(record.get("schema_version"), f"{label}.schema") == 1 and record.get("allocation_id") == allocation and integer(record.get("pid"), f"{label}.pid") > 0, f"{label}: top-level identity drift")
    job_id = record.get("slurm_job_id")
    require(type(job_id) is str and job_id.isdecimal() and int(job_id) > 0, f"{label}: Slurm job identity missing")
    require(exact_bool(record.get("complete"), f"{label}.complete"), f"{label}: incomplete runner artifact")
    identity = validate_identity(record.get("identity"), expected_runner_sha, f"{label}.identity")
    validate_raw_correctness(record.get("raw_abi_correctness"), f"{label}.raw")
    validate_negative_controls(record.get("negative_controls"), f"{label}.negative")
    benchmarks = mapping(record.get("public_benchmarks"), f"{label}.benchmarks")
    require(set(benchmarks) == set(CONTRACTS), f"{label}: benchmark contract set drift")
    assessment: dict[str, object] = {}
    for contract in CONTRACTS:
        repeats = array(benchmarks[contract], f"{label}.{contract}")
        require(len(repeats) == REPEATS, f"{label}.{contract}: repeat count drift")
        evidence = [validate_repeat(value, process_index, repeat_index, f"{label}.{contract}.repeat{repeat_index}") for repeat_index, value in enumerate(repeats)]
        assessment[contract] = {"repeats": evidence, "contract_gate_pass": all(bool(item["repeat_gate_pass"]) for item in evidence)}
    return {"pid": record["pid"], "slurm_job_id": job_id, "gpu_uuid": identity["gpu_uuid"], "extension_sha256": mapping(identity["extension"], f"{label}.extension")["sha256"], "reference_sha256": mapping(identity["reference_torch_ref"], f"{label}.reference")["sha256"], "contracts": assessment}


def allocation_audit(args: argparse.Namespace) -> None:
    actual_analyzer_sha = sha(Path(__file__).resolve(strict=True))
    require(actual_analyzer_sha == args.expected_analyzer_sha256, "analyzer source SHA mismatch")
    data = [read_json(path, expected, f"main{index}") for index, (path, expected) in enumerate(zip(args.main_json, args.expected_main_sha256, strict=True))]
    validated = [validate_main(record, args.expected_runner_sha256, args.allocation, index, f"main{index}") for index, (record, _digest) in enumerate(data)]
    pids = [integer(item["pid"], f"main{index}.pid") for index, item in enumerate(validated)]
    require(len(set(pids)) == 2, f"{args.allocation}: two fresh PIDs required, got {pids}")
    jobs = {str(item["slurm_job_id"]) for item in validated}
    uuids = {str(item["gpu_uuid"]) for item in validated}
    extensions = {str(item["extension_sha256"]) for item in validated}
    references = {str(item["reference_sha256"]) for item in validated}
    require(len(jobs) == len(uuids) == len(extensions) == len(references) == 1 and next(iter(extensions)) == EXPECTED_EXTENSION_SHA256, f"{args.allocation}: process source/GPU identity drift")
    contracts: dict[str, object] = {}
    eligible = True
    for contract in CONTRACTS:
        all_repeats: list[object] = []
        for item in validated:
            all_repeats.extend(mapping(item["contracts"], "contracts")[contract]["repeats"])  # type: ignore[index]
        gate = all(bool(mapping(item, "repeat")["repeat_gate_pass"]) for item in all_repeats)
        contracts[contract] = {"all_four_repeats": all_repeats, "contract_gate_pass": gate}
        eligible = eligible and gate
    decision = "eligible_for_A2_only" if args.allocation == "A1" and eligible else ("eligible_for_cross_allocation_freeze_review" if eligible else "STOP_keep_production_baseline")
    payload: dict[str, object] = {
        "schema_version": 1, "purpose": "one clean allocation audit; never production mutation", "allocation_id": args.allocation,
        "source_identity": {"analyzer_sha256": actual_analyzer_sha, "runner_sha256": args.expected_runner_sha256},
        "artifacts": [{"path": str(path), "sha256": digest} for path, (_record, digest) in zip(args.main_json, data, strict=True)],
        "identity_consistency": {"distinct_pids": pids, "slurm_job_id": next(iter(jobs)), "gpu_uuid": next(iter(uuids)), "extension_sha256": next(iter(extensions)), "reference_torch_ref_sha256": next(iter(references)), "passed": True},
        "contract_assessment": contracts,
        "allocation_gate": {"eligible": eligible, "decision": decision, "meaning": "A1/A2 both must pass the separate chain before any public-freeze eligibility label"}, "complete": True,
    }
    atomic_write(args.json, payload)
    print(f"wrote {args.allocation} tail8191 audit {args.json}; {decision}")


def validate_allocation_audit(record: Mapping[str, Any], expected_allocation: str, expected_analyzer_sha: str, label: str) -> Mapping[str, Any]:
    require(integer(record.get("schema_version"), f"{label}.schema") == 1 and record.get("allocation_id") == expected_allocation and exact_bool(record.get("complete"), f"{label}.complete"), f"{label}: allocation audit identity drift")
    source = mapping(record.get("source_identity"), f"{label}.source")
    require(source.get("analyzer_sha256") == expected_analyzer_sha, f"{label}: analyzer SHA drift")
    consistency = mapping(record.get("identity_consistency"), f"{label}.consistency")
    require(exact_bool(consistency.get("passed"), f"{label}.passed") and consistency.get("extension_sha256") == EXPECTED_EXTENSION_SHA256, f"{label}: source identity failure")
    contracts = mapping(record.get("contract_assessment"), f"{label}.contracts")
    require(set(contracts) == set(CONTRACTS), f"{label}: contract set drift")
    for contract in CONTRACTS:
        assessment = mapping(contracts[contract], f"{label}.{contract}")
        repeats = array(assessment.get("all_four_repeats"), f"{label}.{contract}.repeats")
        require(len(repeats) == 4 and exact_bool(assessment.get("contract_gate_pass"), f"{label}.{contract}.gate"), f"{label}: allocation contract gate failed")
    gate = mapping(record.get("allocation_gate"), f"{label}.gate")
    require(exact_bool(gate.get("eligible"), f"{label}.eligible"), f"{label}: allocation did not pass")
    return consistency


def chain_audits(args: argparse.Namespace) -> None:
    actual_analyzer_sha = sha(Path(__file__).resolve(strict=True))
    require(actual_analyzer_sha == args.expected_analyzer_sha256, "analyzer source SHA mismatch")
    a1, a1_sha = read_json(args.a1_audit, args.expected_a1_sha256, "A1")
    a2, a2_sha = read_json(args.a2_audit, args.expected_a2_sha256, "A2")
    identity_a1 = validate_allocation_audit(a1, "A1", actual_analyzer_sha, "A1")
    identity_a2 = validate_allocation_audit(a2, "A2", actual_analyzer_sha, "A2")
    require(identity_a1.get("slurm_job_id") != identity_a2.get("slurm_job_id"), "A1/A2 must be distinct Slurm allocations")
    payload: dict[str, object] = {
        "schema_version": 1, "purpose": "two-clean-allocation tail8191 public-freeze eligibility chain; no automatic source change",
        "source_identity": {"analyzer_sha256": actual_analyzer_sha},
        "allocations": {"A1": {"path": str(args.a1_audit), "sha256": a1_sha, "slurm_job_id": identity_a1["slurm_job_id"], "gpu_uuid": identity_a1["gpu_uuid"]}, "A2": {"path": str(args.a2_audit), "sha256": a2_sha, "slurm_job_id": identity_a2["slurm_job_id"], "gpu_uuid": identity_a2["gpu_uuid"]}},
        "eligible_for_public_freeze": True, "automatic_production_mutation": False,
        "next_step": "manual integration/review may consider only the exact (T=8191, none/fp32_final_only) table; bf16_both and fp32_both remain baseline negative controls", "complete": True,
    }
    atomic_write(args.json, payload)
    print(f"wrote tail8191 two-allocation chain {args.json}; eligible_for_public_freeze=true")


def self_test() -> None:
    require(close(percentile([float(index) for index in range(SAMPLES)], 0.50), 499.5), "percentile self-test failed")
    require(close(100.0 / 80.0 - 1.0, 0.25), "margin self-test failed")
    try:
        validate_samples([1.0], "self-test")
    except AuditError:
        pass
    else:
        raise AuditError("sample-count rejection self-test failed")
    print("analyzer self-test PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--chain", action="store_true")
    parser.add_argument("--allocation", choices=("A1", "A2"))
    parser.add_argument("--main-json", type=Path, nargs=2)
    parser.add_argument("--expected-main-sha256", nargs=2)
    parser.add_argument("--expected-runner-sha256")
    parser.add_argument("--expected-analyzer-sha256")
    parser.add_argument("--a1-audit", type=Path)
    parser.add_argument("--a2-audit", type=Path)
    parser.add_argument("--expected-a1-sha256")
    parser.add_argument("--expected-a2-sha256")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    require(args.expected_analyzer_sha256 is not None and args.json is not None, "analyzer SHA and --json are required")
    sha_text(args.expected_analyzer_sha256, "expected analyzer SHA")
    if args.chain:
        require(all(value is not None for value in (args.a1_audit, args.a2_audit, args.expected_a1_sha256, args.expected_a2_sha256)), "--chain requires both audited allocation artifacts and SHA256 values")
        chain_audits(args); return
    require(args.allocation is not None and args.main_json is not None and args.expected_main_sha256 is not None and args.expected_runner_sha256 is not None, "allocation mode requires --allocation, two --main-json, two expected SHA, and runner SHA")
    for expected in args.expected_main_sha256:
        sha_text(expected, "expected main SHA")
    sha_text(args.expected_runner_sha256, "expected runner SHA")
    allocation_audit(args)


if __name__ == "__main__":
    try:
        main()
    except AuditError as exc:
        print(f"AUDIT_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
