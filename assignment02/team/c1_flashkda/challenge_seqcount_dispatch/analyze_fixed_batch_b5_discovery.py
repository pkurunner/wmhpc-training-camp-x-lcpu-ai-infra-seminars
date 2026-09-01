#!/usr/bin/env python3
"""Independent stdlib-only audit for B=5 fixed-batch discovery artifacts.

The analyzer is intentionally independent from CUDA, Torch, and the runner's
summary fields.  It fail-closes on identity, type, sample-count, raw-data, or
correctness drift; however, a genuine latency *negative* is a complete valid
discovery result and exits successfully with ``eligible: false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
BATCH = 5
HEADS = 12
TOKENS = 2048
DIM = 128
RAW_CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
PERFORMANCE_CONTRACTS = ("none", "fp32_final_only", "fp32_both")
VARIANTS = ("baseline", "vshard2_p2", "vshard4_p2")
PERCENTILES = ("p50", "p95", "p99")
WARMUP_PER_PATH = 100
SAMPLES_PER_PATH = 1000
REPEATS_PER_PROCESS = 2
MIN_WINNER_MARGIN = 0.02
EXPECTED_PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_REFERENCE_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
EXPECTED_EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_HELPER_SHA256 = "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
EXPECTED_FLASH_KDA_PYTHON_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
EXPECTED_PINNED_LOADER_SHA256 = "9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b"


class AuditError(AssertionError):
    """A malformed, unpinned, or incomplete artifact must fail the audit."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def mapping(value: object, label: str) -> Mapping[str, Any]:
    require(type(value) is dict, f"{label} must be a JSON object")
    return value  # type: ignore[return-value]


def sequence(value: object, label: str) -> list[Any]:
    require(type(value) is list, f"{label} must be a JSON array")
    return value  # type: ignore[return-value]


def integer(value: object, label: str) -> int:
    require(type(value) is int, f"{label} must be an integer (bool is rejected)")
    return value  # type: ignore[return-value]


def numeric(value: object, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric (bool is rejected)")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


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
    lowercase_sha(expected_sha, f"{label}.expected_sha")
    actual_sha = sha(path)
    require(actual_sha == expected_sha, f"{label}: SHA mismatch expected={expected_sha} actual={actual_sha}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label}: invalid JSON") from exc
    return mapping(parsed, label), actual_sha


def percentile(values: list[float], quantile: float) -> float:
    require(bool(values), "cannot calculate percentile of no samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(values: list[float]) -> dict[str, float | int]:
    require(len(values) == SAMPLES_PER_PATH, f"sample count must be {SAMPLES_PER_PATH}")
    require(all(value > 0.0 and math.isfinite(value) for value in values), "samples must be finite positive ms")
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def close(actual: object, expected: float, label: str) -> None:
    measured = numeric(actual, label)
    require(
        math.isclose(measured, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{label}: recorded={measured}, recomputed={expected}",
    )


def require_summary(recorded_value: object, expected: Mapping[str, float | int], label: str) -> None:
    recorded = mapping(recorded_value, label)
    require(set(recorded) == set(expected), f"{label}: summary keys drift")
    for key, expected_value in expected.items():
        actual = recorded.get(key)
        if type(expected_value) is int:
            require(integer(actual, f"{label}.{key}") == expected_value, f"{label}.{key} mismatch")
        else:
            close(actual, float(expected_value), f"{label}.{key}")


def winner_and_margins(paths: Mapping[str, Mapping[str, float | int]]) -> dict[str, object]:
    winners: dict[str, str] = {}
    margins: dict[str, float] = {}
    for percentile_name in PERCENTILES:
        metric = f"{percentile_name}_ms"
        ranked = sorted(
            ((float(paths[variant][metric]), variant) for variant in VARIANTS),
            key=lambda item: item[0],
        )
        winner_value, winner = ranked[0]
        runner_up_value, _ = ranked[1]
        winners[percentile_name] = winner
        margins[percentile_name] = runner_up_value / winner_value - 1.0
    single_winner = len(set(winners.values())) == 1
    margin_gate = all(value >= MIN_WINNER_MARGIN for value in margins.values())
    return {
        "winner_by_percentile": winners,
        "winner_margin_over_runner_up": margins,
        "single_winner_all_percentiles": single_winner,
        "margin_gate_pass": margin_gate,
        "repeat_gate_pass": single_winner and margin_gate,
    }


def require_passed_gate(value: object, label: str) -> None:
    gate = mapping(value, label)
    require(gate.get("passed") is True, f"{label}.passed must be true")


def validate_immutability(value: object, label: str) -> None:
    evidence = mapping(value, label)
    require(evidence.get("input_immutability_exact") is True, f"{label}: input immutable gate missing")
    require(evidence.get("initial_state_immutability_exact") is True, f"{label}: initial immutable gate missing")


def validate_comparison(value: object, label: str, *, final_present: bool) -> None:
    comparison = mapping(value, label)
    require(comparison.get("output_exact") is True, f"{label}.output_exact must be true")
    numeric(comparison.get("output_max_abs"), f"{label}.output_max_abs")
    require(comparison.get("final_state_present") is final_present, f"{label}.final_state_present drift")
    if final_present:
        require(comparison.get("final_state_exact") is True, f"{label}.final_state_exact must be true")
        numeric(comparison.get("final_state_max_abs"), f"{label}.final_state_max_abs")


def validate_identity(
    data: Mapping[str, Any],
    label: str,
    expected_runner_sha: str,
) -> dict[str, str]:
    identity = mapping(data.get("identity"), f"{label}.identity")
    runner = mapping(identity.get("runner"), f"{label}.identity.runner")
    require(runner.get("sha256") == expected_runner_sha, f"{label}: runner source SHA drift")
    require(runner.get("sha256_gate_pass") is True, f"{label}: runner SHA gate missing")
    device = mapping(identity.get("device"), f"{label}.identity.device")
    require(type(device.get("name")) is str and "B300" in str(device["name"]).upper(), f"{label}: B300 name drift")
    require(device.get("capability") == [10, 3], f"{label}: capability drift")
    require(integer(device.get("multiprocessor_count"), f"{label}.SM") == 148, f"{label}: SM count drift")
    require(device.get("b300_gate_pass") is True, f"{label}: B300 gate missing")
    uuid = identity.get("gpu_uuid")
    require(type(uuid) is str and bool(uuid.strip()), f"{label}: missing GPU UUID")
    extension = mapping(identity.get("extension"), f"{label}.identity.extension")
    extension_path = extension.get("path")
    require(type(extension_path) is str and extension_path.endswith(".so"), f"{label}: extension path drift")
    require(extension.get("sha256") == EXPECTED_EXTENSION_SHA256, f"{label}: extension SHA drift")
    require(extension.get("sha256_gate_pass") is True, f"{label}: extension gate missing")
    required_symbols = sequence(extension.get("required_symbols"), f"{label}.extension.required_symbols")
    require(required_symbols == ["fwd", "fwd_vshard_p2", "fwd_vshard4_p2", "get_workspace_size"], f"{label}: extension ABI drift")
    flash_kda_python = mapping(identity.get("flash_kda_python"), f"{label}.identity.flash_kda_python")
    require(
        type(flash_kda_python.get("path")) is str
        and str(flash_kda_python["path"]).endswith("/flash_kda/__init__.py"),
        f"{label}: loaded flash_kda Python path drift",
    )
    require(
        flash_kda_python.get("sha256") == EXPECTED_FLASH_KDA_PYTHON_SHA256
        and flash_kda_python.get("sha256_gate_pass") is True,
        f"{label}: loaded flash_kda Python SHA gate failed",
    )
    pinned_loader = mapping(identity.get("pinned_reference_loader"), f"{label}.identity.pinned_loader")
    require(
        type(pinned_loader.get("path")) is str
        and str(pinned_loader["path"]).endswith(
            "/challenge_varlen_dispatch/run_varlen_dispatch_confirmation.py"
        ),
        f"{label}: pinned-reference loader path drift",
    )
    require(
        pinned_loader.get("sha256") == EXPECTED_PINNED_LOADER_SHA256
        and pinned_loader.get("sha256_gate_pass") is True,
        f"{label}: pinned-reference loader SHA gate failed",
    )
    commits = mapping(identity.get("commits"), f"{label}.identity.commits")
    expected_commits = {
        "patched": EXPECTED_PATCHED_COMMIT,
        "reference": EXPECTED_REFERENCE_COMMIT,
        "fla": EXPECTED_FLA_COMMIT,
    }
    require(dict(commits) == expected_commits, f"{label}: pinned commits drift")
    helper = mapping(identity.get("pinned_reference_helper"), f"{label}.identity.helper")
    require(helper.get("sha256") == EXPECTED_HELPER_SHA256, f"{label}: helper SHA drift")
    require(helper.get("sha256_gate_pass") is True, f"{label}: helper SHA gate missing")
    require(helper.get("no_build") is True, f"{label}: helper no-build proof missing")
    require(
        helper.get("load_contract")
        == "direct cached binary; exactly one pinned load_inline('sigmoid_ext') intercepted",
        f"{label}: helper load contract drift",
    )
    require(helper.get("intercepted_names") == ["sigmoid_ext"], f"{label}: helper interception drift")
    reference = mapping(identity.get("reference_torch_ref"), f"{label}.identity.reference")
    lowercase_sha(reference.get("sha256"), f"{label}.reference.sha256")
    require(type(reference.get("path")) is str and str(reference["path"]).endswith("tests/torch_ref.py"), f"{label}: reference path drift")
    return {
        "gpu_uuid": uuid,
        "extension_path": extension_path,
        "reference_torch_ref_sha256": str(reference["sha256"]),
    }


def validate_correctness(data: Mapping[str, Any], label: str) -> None:
    correctness = mapping(data.get("correctness"), f"{label}.correctness")
    require(set(correctness) == set(RAW_CONTRACTS), f"{label}: raw correctness contract set drift")
    for contract in RAW_CONTRACTS:
        record = mapping(correctness.get(contract), f"{label}.correctness.{contract}")
        require(record.get("contract") == contract, f"{label}/{contract}: contract label drift")
        require(record.get("passed") is True, f"{label}/{contract}: runner marked correctness failed")
        final_present = contract != "none"
        direct = mapping(record.get("direct_wrapper_exactness"), f"{label}/{contract}.direct")
        require(set(direct) == {"baseline_vs_vshard2_p2", "baseline_vs_vshard4_p2"}, f"{label}/{contract}: direct set drift")
        for comparison_name in ("baseline_vs_vshard2_p2", "baseline_vs_vshard4_p2"):
            validate_comparison(direct.get(comparison_name), f"{label}/{contract}.{comparison_name}", final_present=final_present)
        invocations = mapping(record.get("invocation_immutability"), f"{label}/{contract}.immutability")
        require(set(invocations) == set(VARIANTS), f"{label}/{contract}: invocation variant set drift")
        for variant in VARIANTS:
            validate_immutability(invocations.get(variant), f"{label}/{contract}.{variant}.immutability")
        reference = record.get("pinned_torch_reference_exactness")
        if contract in PERFORMANCE_CONTRACTS:
            reference_record = mapping(reference, f"{label}/{contract}.reference")
            validate_comparison(
                reference_record.get("baseline_vs_pinned_torch_reference"),
                f"{label}/{contract}.reference_exact",
                final_present=final_present,
            )
            validate_immutability(reference_record.get("invocation_immutability"), f"{label}/{contract}.reference_immutability")
        else:
            require(reference is None, f"{label}/{contract}: raw-only contract must not claim reference evidence")


def validate_repeat(value: object, label: str, process_index: int, repeat_index: int) -> dict[str, object]:
    repeat = mapping(value, label)
    require(integer(repeat.get("process_index"), f"{label}.process_index") == process_index, f"{label}: process index drift")
    require(integer(repeat.get("repeat_index"), f"{label}.repeat_index") == repeat_index, f"{label}: repeat index drift")
    require(repeat.get("passed") is True, f"{label}: runner marked repeat failed")
    require(repeat.get("input_immutability_exact") is True, f"{label}: input immutability missing")
    require(repeat.get("initial_state_immutability_exact") is True, f"{label}: initial immutability missing")
    by_variant = mapping(repeat.get("initial_state_immutability_by_variant"), f"{label}.initial_by_variant")
    require(by_variant == {variant: True for variant in VARIANTS}, f"{label}: initial per-variant drift")
    require(repeat.get("warmup_calls_per_path") == {variant: WARMUP_PER_PATH for variant in VARIANTS}, f"{label}: warmup count drift")
    require(repeat.get("timed_calls_per_path") == {variant: SAMPLES_PER_PATH for variant in VARIANTS}, f"{label}: timed count drift")
    order = mapping(repeat.get("path_order"), f"{label}.path_order")
    require(order.get("variants") == list(VARIANTS), f"{label}: variant order drift")
    require(order.get("offset_rule") == "sample_or_warmup_index modulo three rotates the first path", f"{label}: cyclic rule drift")
    require(order.get("timed_first_path_counts") == {"baseline": 334, "vshard2_p2": 333, "vshard4_p2": 333}, f"{label}: cyclic count drift")
    raw = mapping(repeat.get("raw_samples_ms"), f"{label}.raw_samples_ms")
    require(set(raw) == set(VARIANTS), f"{label}: raw path set drift")
    paths: dict[str, dict[str, float | int]] = {}
    for variant in VARIANTS:
        raw_values = sequence(raw.get(variant), f"{label}.raw.{variant}")
        values = [numeric(sample, f"{label}.raw.{variant}[{index}]") for index, sample in enumerate(raw_values)]
        paths[variant] = summary(values)
    recorded_paths = mapping(repeat.get("paths"), f"{label}.paths")
    require(set(recorded_paths) == set(VARIANTS), f"{label}: path summary set drift")
    for variant in VARIANTS:
        require_summary(recorded_paths.get(variant), paths[variant], f"{label}.paths.{variant}")
    recomputed = winner_and_margins(paths)
    require(mapping(repeat.get("winner_by_percentile"), f"{label}.winners") == recomputed["winner_by_percentile"], f"{label}: winner drift")
    recorded_margins = mapping(repeat.get("winner_margin_over_runner_up"), f"{label}.margins")
    for percentile_name, margin in recomputed["winner_margin_over_runner_up"].items():  # type: ignore[index]
        close(recorded_margins.get(percentile_name), float(margin), f"{label}.margin.{percentile_name}")
    for key in ("single_winner_all_percentiles", "margin_gate_pass", "repeat_gate_pass"):
        require(repeat.get(key) is recomputed[key], f"{label}.{key}: gate drift")
    return {
        "process_index": process_index,
        "repeat_index": repeat_index,
        "paths": paths,
        "winner_by_percentile": recomputed["winner_by_percentile"],
        "winner_margin_over_runner_up": recomputed["winner_margin_over_runner_up"],
        "single_winner_all_percentiles": recomputed["single_winner_all_percentiles"],
        "margin_gate_pass": recomputed["margin_gate_pass"],
        "repeat_gate_pass": recomputed["repeat_gate_pass"],
    }


def validate_artifact(
    data: Mapping[str, Any],
    label: str,
    expected_runner_sha: str,
    expected_process_index: int,
) -> dict[str, object]:
    require(integer(data.get("schema_version"), f"{label}.schema_version") == SCHEMA_VERSION, f"{label}: schema drift")
    require(data.get("diagnostic_only") is True, f"{label}: diagnostic_only must be true")
    require(data.get("discovery_only") is True, f"{label}: discovery_only must be true")
    require(data.get("no_release_authority") is True, f"{label}: release authority must be false")
    require(data.get("no_production_mapping") is True, f"{label}: production mapping flag drift")
    require(data.get("complete") is True, f"{label}: artifact is incomplete")
    require("failure" not in data, f"{label}: artifact records a failure")
    target = mapping(data.get("target"), f"{label}.target")
    require(
        target == {
            "B": BATCH,
            "H": HEADS,
            "T": TOKENS,
            "K": DIM,
            "V": DIM,
            "form": "fixed",
            "case_name": "b5_h12_t2048",
            "lengths": [TOKENS] * BATCH,
        },
        f"{label}: target shape drift",
    )
    process = mapping(data.get("process"), f"{label}.process")
    pid = integer(process.get("pid"), f"{label}.pid")
    require(pid > 1, f"{label}: invalid PID")
    require(integer(process.get("process_index"), f"{label}.process_index") == expected_process_index, f"{label}: process index drift")
    require(process.get("fresh_python_process_required") is True, f"{label}: fresh process gate missing")
    prereg = mapping(data.get("pre_registered"), f"{label}.pre_registered")
    require(prereg.get("raw_abi_contracts") == list(RAW_CONTRACTS), f"{label}: raw contract drift")
    require(prereg.get("pinned_torch_reference_contracts") == list(PERFORMANCE_CONTRACTS), f"{label}: reference contract drift")
    require(prereg.get("performance_contracts") == list(PERFORMANCE_CONTRACTS), f"{label}: performance contract drift")
    require(prereg.get("variants") == list(VARIANTS), f"{label}: variant drift")
    require(integer(prereg.get("fresh_main_processes"), f"{label}.fresh_main_processes") == 2, f"{label}: process count drift")
    require(prereg.get("required_process_indices") == [0, 1], f"{label}: process index registration drift")
    require(integer(prereg.get("repeats_per_process"), f"{label}.repeats") == REPEATS_PER_PROCESS, f"{label}: repeat registration drift")
    require(integer(prereg.get("warmup_per_path_per_repeat"), f"{label}.warmup") == WARMUP_PER_PATH, f"{label}: warmup registration drift")
    require(integer(prereg.get("cuda_event_samples_per_path_per_repeat"), f"{label}.samples") == SAMPLES_PER_PATH, f"{label}: sample registration drift")
    require(prereg.get("required_percentiles") == list(PERCENTILES), f"{label}: percentile registration drift")
    close(prereg.get("minimum_runner_up_margin"), MIN_WINNER_MARGIN, f"{label}.minimum_margin")
    require(
        prereg.get("optimized_winner_required") == ["vshard2_p2", "vshard4_p2"],
        f"{label}: optimized-winner registration drift",
    )
    identity = validate_identity(data, label, expected_runner_sha)
    gates = mapping(data.get("gates"), f"{label}.gates")
    for gate_name in ("clean_gpu_shell", "device", "extension", "pinned_commits", "runner_source", "python_sources", "pinned_reference_helper", "raw_abi_exact", "pinned_torch_reference_exact"):
        require_passed_gate(gates.get(gate_name), f"{label}.gates.{gate_name}")
    direct_only = mapping(gates.get("no_dispatcher_or_map_mutation"), f"{label}.gates.direct_only")
    require(direct_only.get("passed") is True, f"{label}: no-mutation gate failed")
    require(direct_only.get("method") == "direct raw ABI wrapper calls only; auto_dispatch is not imported", f"{label}: direct-only declaration drift")
    validate_correctness(data, label)
    performance = mapping(data.get("performance"), f"{label}.performance")
    require(set(performance) == set(PERFORMANCE_CONTRACTS), f"{label}: performance contracts drift")
    per_contract: dict[str, list[dict[str, object]]] = {}
    for contract in PERFORMANCE_CONTRACTS:
        record = mapping(performance.get(contract), f"{label}.performance.{contract}")
        repeats = sequence(record.get("repeats"), f"{label}.performance.{contract}.repeats")
        require(len(repeats) == REPEATS_PER_PROCESS, f"{label}/{contract}: repeat count drift")
        per_contract[contract] = [
            validate_repeat(repeats[index], f"{label}/{contract}.repeat{index}", expected_process_index, index)
            for index in range(REPEATS_PER_PROCESS)
        ]
    return {
        "pid": pid,
        "identity": identity,
        "per_contract": per_contract,
    }


def assess_contract(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    require(len(records) == 4, "each contract must have exactly four repeats")
    flattened_winners: list[str] = []
    all_margins: list[float] = []
    repeats: list[dict[str, object]] = []
    for record in records:
        winners = mapping(record["winner_by_percentile"], "assessment.winners")
        margins = mapping(record["winner_margin_over_runner_up"], "assessment.margins")
        require(set(winners) == set(PERCENTILES), "assessment winner percentile set drift")
        require(set(margins) == set(PERCENTILES), "assessment margin percentile set drift")
        flattened_winners.extend(str(winners[name]) for name in PERCENTILES)
        all_margins.extend(numeric(margins[name], f"assessment.margin.{name}") for name in PERCENTILES)
        repeats.append(
            {
                "process_index": integer(record["process_index"], "assessment.process_index"),
                "repeat_index": integer(record["repeat_index"], "assessment.repeat_index"),
                "winner_by_percentile": dict(winners),
                "winner_margin_over_runner_up": dict(margins),
            }
        )
    same_winner = len(set(flattened_winners)) == 1
    common_winner = flattened_winners[0] if same_winner else None
    margins_pass = all(value >= MIN_WINNER_MARGIN for value in all_margins)
    optimized_winner = common_winner in ("vshard2_p2", "vshard4_p2")
    eligible = same_winner and margins_pass and optimized_winner
    return {
        "repeats": repeats,
        "common_winner_all_four_repeats_and_percentiles": common_winner,
        "same_winner_all_four_repeats_and_percentiles": same_winner,
        "margin_at_least_2_percent_every_repeat_and_percentile": margins_pass,
        "optimized_winner_not_baseline": optimized_winner,
        "eligible_for_confirmation": eligible,
    }


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
    parser.add_argument("main_json", nargs=2, type=Path, help="the two fresh B=5 main-process artifacts")
    parser.add_argument("--expected-main-sha256", nargs=2, required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-analyzer-sha256", required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    lowercase_sha(args.expected_runner_sha256, "--expected-runner-sha256")
    lowercase_sha(args.expected_analyzer_sha256, "--expected-analyzer-sha256")
    actual_analyzer_sha = sha(Path(__file__).resolve(strict=True))
    require(
        actual_analyzer_sha == args.expected_analyzer_sha256,
        f"analyzer source SHA mismatch expected={args.expected_analyzer_sha256} actual={actual_analyzer_sha}",
    )
    artifacts = [
        read_json(path, expected_sha, f"main{index}")
        for index, (path, expected_sha) in enumerate(zip(args.main_json, args.expected_main_sha256, strict=True))
    ]
    validated = [
        validate_artifact(data, f"main{index}", args.expected_runner_sha256, index)
        for index, (data, _actual_sha) in enumerate(artifacts)
    ]
    pids = [integer(record["pid"], f"validated[{index}].pid") for index, record in enumerate(validated)]
    require(len(set(pids)) == 2, f"two fresh main artifacts require distinct PIDs, got {pids}")
    identities = [mapping(record["identity"], f"validated[{index}].identity") for index, record in enumerate(validated)]
    uuids = {str(identity["gpu_uuid"]) for identity in identities}
    extension_paths = {str(identity["extension_path"]) for identity in identities}
    reference_shas = {str(identity["reference_torch_ref_sha256"]) for identity in identities}
    require(len(uuids) == 1, f"both processes must use one GPU UUID, got {sorted(uuids)}")
    require(len(extension_paths) == 1, f"both processes must use one extension SO, got {sorted(extension_paths)}")
    require(len(reference_shas) == 1, "pinned Torch reference source drift across processes")
    contracts: dict[str, dict[str, object]] = {}
    all_eligible = True
    for contract in PERFORMANCE_CONTRACTS:
        all_records: list[Mapping[str, object]] = []
        for validated_record in validated:
            per_contract = mapping(validated_record["per_contract"], "validated.per_contract")
            all_records.extend(per_contract[contract])  # type: ignore[arg-type]
        assessment = assess_contract(all_records)
        contracts[contract] = assessment
        all_eligible = all_eligible and assessment["eligible_for_confirmation"] is True
    decision = (
        "eligible_for_independent_confirmation"
        if all_eligible
        else "not_eligible_for_independent_confirmation"
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "purpose": "independent B=5 discovery audit; no production mapping or release authority",
        "diagnostic_only": True,
        "discovery_only": True,
        "no_release_authority": True,
        "no_production_mapping": True,
        "source_identity": {
            "analyzer": {
                "path": str(Path(__file__).resolve()),
                "sha256": actual_analyzer_sha,
                "sha256_gate_pass": True,
            },
            "runner_expected_sha256": args.expected_runner_sha256,
        },
        "artifacts": [
            {"path": str(path), "sha256": actual_sha, "sha256_gate_pass": True}
            for path, (_data, actual_sha) in zip(args.main_json, artifacts, strict=True)
        ],
        "identity_consistency": {
            "distinct_fresh_pids": pids,
            "same_gpu_uuid": next(iter(uuids)),
            "same_extension_so": next(iter(extension_paths)),
            "same_pinned_torch_reference_sha256": next(iter(reference_shas)),
            "passed": True,
        },
        "contract_assessment": contracts,
        "second_allocation_decision": {
            "eligible": all_eligible,
            "decision": decision,
            "meaning": "eligible only for an independently authorized confirmation allocation; never a production mapping",
            "automatic_second_allocation_submitted": False,
        },
        "complete": True,
    }
    atomic_write(args.json, payload)
    print(f"wrote independent B=5 discovery audit {args.json}; {decision}")


if __name__ == "__main__":
    try:
        main()
    except AuditError as exc:
        print(f"AUDIT_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
