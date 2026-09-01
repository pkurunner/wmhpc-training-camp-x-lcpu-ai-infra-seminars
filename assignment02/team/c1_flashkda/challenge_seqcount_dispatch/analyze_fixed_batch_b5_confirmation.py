#!/usr/bin/env python3
"""Fail-closed cross-allocation confirmation chain for fixed-batch B=5.

This stdlib-only tool consumes two already independent measurement audits: the
frozen B=5 discovery result and one fresh allocation run of exactly the same
measurement engine.  It never authorizes a dispatcher change.  A genuine
performance miss, including a change from ``vshard2_p2`` to another valid
candidate, is a complete result with ``eligible_for_public_integration_review:
false`` and exit code zero.  Identity, schema, type, or provenance drift is an
audit failure and exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
CONTRACTS = ("none", "fp32_final_only", "fp32_both")
PERCENTILES = ("p50", "p95", "p99")
VARIANTS = ("baseline", "vshard2_p2", "vshard4_p2")
TARGET_WINNER = "vshard2_p2"
MIN_WINNER_MARGIN = 0.02


class AuditError(AssertionError):
    """Evidence cannot support the chain when this exception is raised."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def mapping(value: object, label: str) -> Mapping[str, Any]:
    require(type(value) is dict, f"{label} must be a JSON object")
    return value  # type: ignore[return-value]


def sequence(value: object, label: str) -> list[Any]:
    require(type(value) is list, f"{label} must be a JSON array")
    return value  # type: ignore[return-value]


def exact_bool(value: object, label: str) -> bool:
    require(type(value) is bool, f"{label} must be an exact bool")
    return value  # type: ignore[return-value]


def integer(value: object, label: str) -> int:
    require(type(value) is int, f"{label} must be an integer (bool is rejected)")
    return value  # type: ignore[return-value]


def numeric(value: object, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric (bool is rejected)")
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number


def lowercase_sha(value: object, label: str) -> str:
    require(type(value) is str, f"{label} must be a SHA string")
    require(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA256",
    )
    return value  # type: ignore[return-value]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path, expected_sha: str, label: str) -> tuple[Mapping[str, Any], str]:
    lowercase_sha(expected_sha, f"{label}.expected_sha256")
    actual_sha = sha(path)
    require(actual_sha == expected_sha, f"{label}: SHA mismatch expected={expected_sha} actual={actual_sha}")
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label}: invalid JSON") from exc
    return mapping(decoded, label), actual_sha


def positive_job_id(text: str, label: str) -> int:
    require(text.isdecimal(), f"{label} must be a decimal Slurm job ID")
    value = int(text)
    require(value > 0, f"{label} must be positive")
    return value


def validate_contract(record_value: object, label: str) -> dict[str, object]:
    """Validate the discovery analyzer's public per-contract assessment."""

    record = mapping(record_value, label)
    expected_keys = {
        "repeats",
        "common_winner_all_four_repeats_and_percentiles",
        "same_winner_all_four_repeats_and_percentiles",
        "margin_at_least_2_percent_every_repeat_and_percentile",
        "optimized_winner_not_baseline",
        "eligible_for_confirmation",
    }
    require(set(record) == expected_keys, f"{label}: contract-assessment keys drift")
    repeats = sequence(record.get("repeats"), f"{label}.repeats")
    require(len(repeats) == 4, f"{label}: exactly four repeats are required")
    expected_repeat_indices = {(process_index, repeat_index) for process_index in (0, 1) for repeat_index in (0, 1)}
    seen_indices: set[tuple[int, int]] = set()
    flattened_winners: list[str] = []
    flattened_margins: list[float] = []
    public_repeats: list[dict[str, object]] = []
    for index, repeat_value in enumerate(repeats):
        repeat = mapping(repeat_value, f"{label}.repeats[{index}]")
        require(
            set(repeat) == {
                "process_index",
                "repeat_index",
                "winner_by_percentile",
                "winner_margin_over_runner_up",
            },
            f"{label}.repeats[{index}]: keys drift",
        )
        process_index = integer(repeat.get("process_index"), f"{label}.repeats[{index}].process_index")
        repeat_index = integer(repeat.get("repeat_index"), f"{label}.repeats[{index}].repeat_index")
        seen_indices.add((process_index, repeat_index))
        winners = mapping(repeat.get("winner_by_percentile"), f"{label}.repeats[{index}].winners")
        margins = mapping(repeat.get("winner_margin_over_runner_up"), f"{label}.repeats[{index}].margins")
        require(set(winners) == set(PERCENTILES), f"{label}.repeats[{index}]: winner percentiles drift")
        require(set(margins) == set(PERCENTILES), f"{label}.repeats[{index}]: margin percentiles drift")
        winner_values: dict[str, str] = {}
        margin_values: dict[str, float] = {}
        for percentile in PERCENTILES:
            winner = winners.get(percentile)
            require(type(winner) is str and winner in VARIANTS, f"{label}/{percentile}: invalid winner")
            margin = numeric(margins.get(percentile), f"{label}/{percentile}: invalid margin")
            winner_values[percentile] = winner
            margin_values[percentile] = margin
            flattened_winners.append(winner)
            flattened_margins.append(margin)
        public_repeats.append(
            {
                "process_index": process_index,
                "repeat_index": repeat_index,
                "winner_by_percentile": winner_values,
                "winner_margin_over_runner_up": margin_values,
            }
        )
    require(seen_indices == expected_repeat_indices, f"{label}: process/repeat coverage drift")
    computed_same_winner = len(set(flattened_winners)) == 1
    computed_common_winner: str | None = flattened_winners[0] if computed_same_winner else None
    computed_margin_gate = all(value >= MIN_WINNER_MARGIN for value in flattened_margins)
    computed_optimized = computed_common_winner in ("vshard2_p2", "vshard4_p2")
    computed_eligible = computed_same_winner and computed_margin_gate and computed_optimized
    common_winner = record.get("common_winner_all_four_repeats_and_percentiles")
    require(
        common_winner == computed_common_winner,
        f"{label}.common_winner does not match the repeats",
    )
    require(
        exact_bool(record.get("same_winner_all_four_repeats_and_percentiles"), f"{label}.same_winner")
        is computed_same_winner,
        f"{label}.same_winner does not match the repeats",
    )
    require(
        exact_bool(record.get("margin_at_least_2_percent_every_repeat_and_percentile"), f"{label}.margin_gate")
        is computed_margin_gate,
        f"{label}.margin_gate does not match the repeats",
    )
    require(
        exact_bool(record.get("optimized_winner_not_baseline"), f"{label}.optimized") is computed_optimized,
        f"{label}.optimized does not match the repeats",
    )
    require(
        exact_bool(record.get("eligible_for_confirmation"), f"{label}.eligible") is computed_eligible,
        f"{label}.eligible does not match the repeats",
    )
    return {
        "repeats": public_repeats,
        "common_winner": computed_common_winner,
        "same_winner": computed_same_winner,
        "margin_gate": computed_margin_gate,
        "optimized_winner": computed_optimized,
        "eligible_for_confirmation": computed_eligible,
    }


def validate_measurement_audit(
    data: Mapping[str, Any],
    label: str,
    expected_runner_sha: str,
    expected_discovery_analyzer_sha: str,
) -> dict[str, object]:
    """Validate the stable part of ``analyze_fixed_batch_b5_discovery`` output."""

    require(integer(data.get("schema_version"), f"{label}.schema_version") == SCHEMA_VERSION, f"{label}: schema drift")
    for flag in ("diagnostic_only", "discovery_only", "no_release_authority", "no_production_mapping", "complete"):
        require(exact_bool(data.get(flag), f"{label}.{flag}") is True, f"{label}.{flag} must be true")
    source_identity = mapping(data.get("source_identity"), f"{label}.source_identity")
    require(set(source_identity) == {"analyzer", "runner_expected_sha256"}, f"{label}: source identity keys drift")
    analyzer = mapping(source_identity.get("analyzer"), f"{label}.source_identity.analyzer")
    require(set(analyzer) == {"path", "sha256", "sha256_gate_pass"}, f"{label}: analyzer identity keys drift")
    require(type(analyzer.get("path")) is str and bool(str(analyzer["path"]).strip()), f"{label}: analyzer path missing")
    require(analyzer.get("sha256") == expected_discovery_analyzer_sha, f"{label}: discovery analyzer SHA drift")
    require(exact_bool(analyzer.get("sha256_gate_pass"), f"{label}.analyzer.sha256_gate_pass") is True, f"{label}: analyzer gate failed")
    require(source_identity.get("runner_expected_sha256") == expected_runner_sha, f"{label}: runner SHA drift")

    artifacts = sequence(data.get("artifacts"), f"{label}.artifacts")
    require(len(artifacts) == 2, f"{label}: exactly two main artifacts are required")
    artifact_shas: list[str] = []
    normalized_artifacts: list[dict[str, str]] = []
    for index, artifact_value in enumerate(artifacts):
        artifact = mapping(artifact_value, f"{label}.artifacts[{index}]")
        require(set(artifact) == {"path", "sha256", "sha256_gate_pass"}, f"{label}.artifacts[{index}]: keys drift")
        path = artifact.get("path")
        require(type(path) is str and bool(path.strip()), f"{label}.artifacts[{index}].path missing")
        artifact_sha = lowercase_sha(artifact.get("sha256"), f"{label}.artifacts[{index}].sha256")
        require(exact_bool(artifact.get("sha256_gate_pass"), f"{label}.artifacts[{index}].sha256_gate_pass") is True, f"{label}.artifacts[{index}]: SHA gate failed")
        artifact_shas.append(artifact_sha)
        normalized_artifacts.append({"path": path, "sha256": artifact_sha})
    require(len(set(artifact_shas)) == 2, f"{label}: main artifacts cannot share a SHA")

    identity = mapping(data.get("identity_consistency"), f"{label}.identity_consistency")
    expected_identity_keys = {
        "distinct_fresh_pids",
        "same_gpu_uuid",
        "same_extension_so",
        "same_pinned_torch_reference_sha256",
        "passed",
    }
    require(set(identity) == expected_identity_keys, f"{label}: identity-consistency keys drift")
    pids = sequence(identity.get("distinct_fresh_pids"), f"{label}.identity_consistency.pids")
    require(len(pids) == 2, f"{label}: two fresh PIDs required")
    pid_values = [integer(pid, f"{label}.identity_consistency.pids[{index}]") for index, pid in enumerate(pids)]
    require(all(pid > 1 for pid in pid_values) and len(set(pid_values)) == 2, f"{label}: fresh PID proof failed")
    for key in ("same_gpu_uuid", "same_extension_so", "same_pinned_torch_reference_sha256"):
        require(type(identity.get(key)) is str and bool(str(identity[key]).strip()), f"{label}.{key} missing")
    require(exact_bool(identity.get("passed"), f"{label}.identity_consistency.passed") is True, f"{label}: identity-consistency failed")

    assessments = mapping(data.get("contract_assessment"), f"{label}.contract_assessment")
    require(set(assessments) == set(CONTRACTS), f"{label}: contract keys drift")
    contracts = {contract: validate_contract(assessments.get(contract), f"{label}.contract_assessment.{contract}") for contract in CONTRACTS}

    decision = mapping(data.get("second_allocation_decision"), f"{label}.second_allocation_decision")
    expected_decision_keys = {"eligible", "decision", "meaning", "automatic_second_allocation_submitted"}
    require(set(decision) == expected_decision_keys, f"{label}: decision keys drift")
    expected_eligible = all(contracts[contract]["eligible_for_confirmation"] is True for contract in CONTRACTS)
    observed_eligible = exact_bool(decision.get("eligible"), f"{label}.second_allocation_decision.eligible")
    require(observed_eligible is expected_eligible, f"{label}: decision eligibility does not match contracts")
    expected_decision = "eligible_for_independent_confirmation" if expected_eligible else "not_eligible_for_independent_confirmation"
    require(decision.get("decision") == expected_decision, f"{label}: decision string drift")
    require(
        decision.get("meaning") == "eligible only for an independently authorized confirmation allocation; never a production mapping",
        f"{label}: decision meaning drift",
    )
    require(
        exact_bool(decision.get("automatic_second_allocation_submitted"), f"{label}.automatic_second_allocation_submitted") is False,
        f"{label}: automatic allocation is forbidden",
    )
    return {
        "artifacts": normalized_artifacts,
        "artifact_sha_set": set(artifact_shas),
        "gpu_uuid": str(identity["same_gpu_uuid"]),
        "contracts": contracts,
        "decision_eligible": observed_eligible,
    }


def assessment_for_target(contracts: Mapping[str, Mapping[str, object]]) -> tuple[bool, dict[str, dict[str, object]]]:
    """Make switching candidates a valid negative, never a positive promotion."""

    all_pass = True
    details: dict[str, dict[str, object]] = {}
    for contract in CONTRACTS:
        record = contracts[contract]
        candidate_matches = record["common_winner"] == TARGET_WINNER
        same_winner = record["same_winner"] is True
        margin_gate = record["margin_gate"] is True
        optimized = record["optimized_winner"] is True
        discovery_eligible = record["eligible_for_confirmation"] is True
        passed = candidate_matches and same_winner and margin_gate and optimized and discovery_eligible
        details[contract] = {
            "common_winner": record["common_winner"],
            "candidate_matches_vshard2_p2": candidate_matches,
            "same_winner_all_four_repeats_and_percentiles": same_winner,
            "margin_at_least_2_percent_every_repeat_and_percentile": margin_gate,
            "optimized_winner_not_baseline": optimized,
            "eligible_for_confirmation": discovery_eligible,
            "contract_pass": passed,
        }
        all_pass = all_pass and passed
    return all_pass, details


def validate_current_seed_provenance(
    artifacts: Sequence[Mapping[str, str]],
    expected_seed: int,
) -> list[dict[str, object]]:
    """Bind the fresh audit to the frozen runner's ``--seed`` formula.

    The discovery analyzer intentionally summarizes latency and so does not
    preserve a top-level seed.  The current audit's pinned raw artifacts do,
    however.  Reading them here prevents a result from another seed from being
    presented as the required ``20260830`` independent allocation.
    """

    require(expected_seed > 0, "--expected-current-seed must be positive")
    require(len(artifacts) == 2, "seed provenance needs exactly two artifacts")
    evidence: list[dict[str, object]] = []
    for expected_process_index, artifact in enumerate(artifacts):
        path = Path(artifact["path"])
        raw, actual_sha = read_json(path, artifact["sha256"], f"current_raw_main{expected_process_index}")
        require(
            integer(raw.get("schema_version"), f"current_raw_main{expected_process_index}.schema_version") == SCHEMA_VERSION,
            f"current_raw_main{expected_process_index}: schema drift",
        )
        for flag in ("diagnostic_only", "discovery_only", "no_release_authority", "no_production_mapping", "complete"):
            require(
                exact_bool(raw.get(flag), f"current_raw_main{expected_process_index}.{flag}") is True,
                f"current_raw_main{expected_process_index}.{flag} must be true",
            )
        process = mapping(raw.get("process"), f"current_raw_main{expected_process_index}.process")
        require(
            integer(process.get("process_index"), f"current_raw_main{expected_process_index}.process_index")
            == expected_process_index,
            f"current_raw_main{expected_process_index}: process index drift",
        )
        performance = mapping(raw.get("performance"), f"current_raw_main{expected_process_index}.performance")
        require(set(performance) == set(CONTRACTS), f"current_raw_main{expected_process_index}: performance contracts drift")
        for contract_index, contract in enumerate(CONTRACTS):
            contract_record = mapping(
                performance.get(contract), f"current_raw_main{expected_process_index}.performance.{contract}"
            )
            repeats = sequence(
                contract_record.get("repeats"),
                f"current_raw_main{expected_process_index}.performance.{contract}.repeats",
            )
            require(len(repeats) == 2, f"current_raw_main{expected_process_index}/{contract}: repeat count drift")
            for repeat_index, repeat_value in enumerate(repeats):
                repeat = mapping(
                    repeat_value,
                    f"current_raw_main{expected_process_index}.performance.{contract}.repeats[{repeat_index}]",
                )
                require(
                    integer(repeat.get("process_index"), f"current_raw_main{expected_process_index}/{contract}/{repeat_index}.process_index")
                    == expected_process_index,
                    f"current_raw_main{expected_process_index}/{contract}/{repeat_index}: process index drift",
                )
                require(
                    integer(repeat.get("repeat_index"), f"current_raw_main{expected_process_index}/{contract}/{repeat_index}.repeat_index")
                    == repeat_index,
                    f"current_raw_main{expected_process_index}/{contract}/{repeat_index}: repeat index drift",
                )
                computed_input_seed = (
                    expected_seed
                    + expected_process_index * 1_000_003
                    + contract_index * 10_007
                    + repeat_index * 1_009
                )
                require(
                    integer(repeat.get("input_seed"), f"current_raw_main{expected_process_index}/{contract}/{repeat_index}.input_seed")
                    == computed_input_seed,
                    f"current_raw_main{expected_process_index}/{contract}/{repeat_index}: input seed drift",
                )
                require(
                    integer(repeat.get("state_seed"), f"current_raw_main{expected_process_index}/{contract}/{repeat_index}.state_seed")
                    == computed_input_seed + 101,
                    f"current_raw_main{expected_process_index}/{contract}/{repeat_index}: state seed drift",
                )
        evidence.append(
            {
                "path": str(path),
                "sha256": actual_sha,
                "process_index": expected_process_index,
                "seed_formula_gate_pass": True,
            }
        )
    return evidence


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
    parser.add_argument("history_audit", type=Path, help="frozen B=5 discovery measurement audit")
    parser.add_argument("measurement_audit", type=Path, help="fresh allocation measurement audit")
    parser.add_argument("--expected-history-audit-sha256", required=True)
    parser.add_argument("--expected-measurement-audit-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-discovery-analyzer-sha256", required=True)
    parser.add_argument("--expected-confirmation-analyzer-sha256", required=True)
    parser.add_argument("--expected-current-seed", type=int, required=True)
    parser.add_argument("--history-slurm-log", type=Path, required=True)
    parser.add_argument("--expected-history-slurm-log-sha256", required=True)
    parser.add_argument("--history-slurm-job-id", required=True)
    parser.add_argument("--current-slurm-log", type=Path, required=True)
    parser.add_argument("--current-slurm-job-id", required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    for argument_name in (
        "expected_history_audit_sha256",
        "expected_measurement_audit_sha256",
        "expected_runner_sha256",
        "expected_discovery_analyzer_sha256",
        "expected_confirmation_analyzer_sha256",
        "expected_history_slurm_log_sha256",
    ):
        lowercase_sha(getattr(args, argument_name), f"--{argument_name.replace('_', '-')}")
    actual_analyzer_sha = sha(Path(__file__).resolve(strict=True))
    require(
        actual_analyzer_sha == args.expected_confirmation_analyzer_sha256,
        "confirmation analyzer source SHA mismatch",
    )
    history_job_id = positive_job_id(args.history_slurm_job_id, "--history-slurm-job-id")
    current_job_id = positive_job_id(args.current_slurm_job_id, "--current-slurm-job-id")
    history_audit, history_audit_sha = read_json(
        args.history_audit, args.expected_history_audit_sha256, "history_audit"
    )
    current_audit, current_audit_sha = read_json(
        args.measurement_audit, args.expected_measurement_audit_sha256, "measurement_audit"
    )
    history_log_sha = sha(args.history_slurm_log)
    require(
        history_log_sha == args.expected_history_slurm_log_sha256,
        "history Slurm log SHA mismatch",
    )
    require(args.current_slurm_log.is_file(), "current Slurm log is missing")
    current_log_sha_observed = sha(args.current_slurm_log)
    require(f"job{history_job_id}" in args.history_slurm_log.name, "history log path does not encode its Slurm job ID")
    require(f"job{current_job_id}" in args.current_slurm_log.name, "current log path does not encode its Slurm job ID")

    history = validate_measurement_audit(
        history_audit, "history_audit", args.expected_runner_sha256, args.expected_discovery_analyzer_sha256
    )
    current = validate_measurement_audit(
        current_audit, "measurement_audit", args.expected_runner_sha256, args.expected_discovery_analyzer_sha256
    )
    current_seed_provenance = validate_current_seed_provenance(
        current["artifacts"], args.expected_current_seed  # type: ignore[arg-type]
    )
    history_pass, history_contracts = assessment_for_target(history["contracts"])  # type: ignore[arg-type]
    current_pass, current_contracts = assessment_for_target(current["contracts"])  # type: ignore[arg-type]
    artifact_sets_disjoint = not bool(history["artifact_sha_set"] & current["artifact_sha_set"])  # type: ignore[operator]
    distinct_job_ids = history_job_id != current_job_id
    distinct_log_paths = args.history_slurm_log.resolve() != args.current_slurm_log.resolve()
    allocation_evidence_pass = artifact_sets_disjoint and distinct_job_ids and distinct_log_paths
    eligible = history_pass and current_pass and allocation_evidence_pass
    decision = "eligible_for_public_integration_review" if eligible else "not_eligible_for_public_integration_review"

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "independent B=5 cross-allocation confirmation chain; no dispatcher, map, or release authority",
        "diagnostic_only": True,
        "confirmation_only": True,
        "no_release_authority": True,
        "no_production_mapping": True,
        "source_identity": {
            "confirmation_analyzer": {
                "path": str(Path(__file__).resolve()),
                "sha256": actual_analyzer_sha,
                "sha256_gate_pass": True,
            },
            "measurement_runner_expected_sha256": args.expected_runner_sha256,
            "measurement_discovery_analyzer_expected_sha256": args.expected_discovery_analyzer_sha256,
        },
        "input_audits": {
            "historical_discovery": {"path": str(args.history_audit), "sha256": history_audit_sha, "sha256_gate_pass": True},
            "current_measurement": {"path": str(args.measurement_audit), "sha256": current_audit_sha, "sha256_gate_pass": True},
        },
        "historical_assessment": {
            "expected_candidate": TARGET_WINNER,
            "contracts": history_contracts,
            "all_three_contracts_pass": history_pass,
        },
        "current_assessment": {
            "expected_candidate": TARGET_WINNER,
            "contracts": current_contracts,
            "all_three_contracts_pass": current_pass,
        },
        "current_measurement_seed_provenance": {
            "expected_seed": args.expected_current_seed,
            "raw_main_artifacts": current_seed_provenance,
            "passed": True,
        },
        "independent_allocation_evidence": {
            "historical_main_artifact_sha256": sorted(history["artifact_sha_set"]),  # type: ignore[arg-type]
            "current_main_artifact_sha256": sorted(current["artifact_sha_set"]),  # type: ignore[arg-type]
            "artifact_sha_sets_disjoint": artifact_sets_disjoint,
            "historical_slurm": {
                "job_id": history_job_id,
                "log_path": str(args.history_slurm_log),
                "log_sha256": history_log_sha,
                "log_sha256_gate_pass": True,
            },
            "current_slurm": {
                "job_id": current_job_id,
                "log_path": str(args.current_slurm_log),
                "log_sha256_observed_before_confirmation_analysis": current_log_sha_observed,
            },
            "distinct_slurm_job_ids": distinct_job_ids,
            "distinct_slurm_log_paths": distinct_log_paths,
            "historical_gpu_uuid": history["gpu_uuid"],
            "current_gpu_uuid": current["gpu_uuid"],
            "same_gpu_uuid_across_allocations": history["gpu_uuid"] == current["gpu_uuid"],
            "independence_evidence_pass": allocation_evidence_pass,
        },
        "public_integration_decision": {
            "eligible_for_public_integration_review": eligible,
            "decision": decision,
            "meaning": "eligible only for human public-integration review; never a dispatcher edit, production mapping, or release",
            "automatic_public_integration": False,
        },
        "complete": True,
    }
    atomic_write(args.json, payload)
    print(f"wrote B=5 confirmation chain {args.json}; {decision}")


if __name__ == "__main__":
    try:
        main()
    except AuditError as exc:
        print(f"AUDIT_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
