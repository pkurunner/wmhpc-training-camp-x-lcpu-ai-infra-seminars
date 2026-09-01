#!/usr/bin/env python3
"""Stdlib-only, fail-closed audit for fresh B=7 none public-route evidence.

The analyzer never trusts a runner summary, a top-level eligibility bit, or a
previous allocation audit.  It reads every raw main artifact again, validates
the exact schema/types, recomputes all percentile gates, and re-hashes the
complete source/protocol identity recorded by the runner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


RAW_CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
PATHS = ("pinned_public", "c1_test_route_public")
PERCENTILES = ("p50", "p95", "p99")
SAMPLES, REPEATS, WARMUP, MIN_MARGIN = 1000, 2, 100, 0.02
SCHEMA_VERSION = 3
EXPECTED_SO = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_AUTO_DISPATCH = "9cdd460058254016af58723875bdf99ebe74f8e016a4c6027eb7fb38c8e9a88c"
EXPECTED_FLA_BACKEND = "206e448abcd3d64826f87a20e7d57c790fef6adacd91e26edcb10a3711b9b656"
EXPECTED_HARNESS = "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"
EXPECTED_FLASH_KDA_PYTHON = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
PINNED_REFERENCE_HELPER_PATH = "/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so"
PINNED_REFERENCE_HELPER_SHA256 = "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
PINNED_REFERENCE_HELPER_LOAD_CONTRACT = "direct cached binary; exactly one pinned load_inline('sigmoid_ext') intercepted"
EXPECTED_PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_REFERENCE_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
FLA_FILES = {
    "fla/__init__.py": "b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d",
    "fla/ops/backends/__init__.py": "a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635",
    "fla/ops/kda/__init__.py": "24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb",
    "fla/ops/kda/backends/__init__.py": "86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797",
    "fla/ops/kda/backends/flash_kda.py": "0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2",
    "fla/ops/kda/chunk.py": "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8",
}
PATCHED_DIRTY_FILES = {
    "csrc/flash_kda.cpp": "38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4",
    "csrc/fwd.h": "613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083",
    "csrc/smxx/fwd_launch.cu": "a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928",
}
PATCHED_DIRTY_STATUS = " M"
TEST_ROUTE_REASON = "test_only_b7_h12_t2048_none_exact_route"
NEGATIVE_REASON = "fixed_batch_shape_not_whitelisted"
TIMED_EVENT_CONTRACT = "CUDA current-stream: prepared environment/context/kwargs/counters/events and start.record+start.synchronize before interval; interval exactly one public chunk_kda -> end.record; host-only audit then end.synchronize"
TIMED_SCHEDULE = "two-path cyclic; 100 warmups/path; 1000 CUDA-event samples/path"
INPUT_FIELDS = ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "scale", "lower_bound")


class AuditError(AssertionError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise AuditError(message)


def obj(value: object, label: str) -> Mapping[str, Any]:
    require(type(value) is dict, f"{label} must be an object")
    return value  # type: ignore[return-value]


def arr(value: object, label: str) -> list[Any]:
    require(type(value) is list, f"{label} must be an array")
    return value  # type: ignore[return-value]


def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    require(actual == expected, f"{label}: keys drift; missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}")


def boolean(value: object, label: str) -> bool:
    require(type(value) is bool, f"{label} must be exact bool")
    return value  # type: ignore[return-value]


def integer(value: object, label: str) -> int:
    require(type(value) is int, f"{label} must be exact integer")
    return value  # type: ignore[return-value]


def floating(value: object, label: str) -> float:
    require(type(value) is float, f"{label} must be exact float")
    require(math.isfinite(value), f"{label} must be finite")
    return value  # type: ignore[return-value]


def text(value: object, label: str) -> str:
    require(type(value) is str, f"{label} must be exact string")
    return value  # type: ignore[return-value]


def sha_text(value: object, label: str) -> str:
    value = text(value, label)
    require(len(value) == 64 and all(character in "0123456789abcdef" for character in value), f"{label} must be lowercase SHA256")
    return value


def positive_decimal_job(value: object, label: str) -> str:
    job = text(value, label)
    require(job.isascii() and job.isdecimal() and job[:1] != "0" and int(job) > 0, f"{label} must be a positive canonical-decimal Slurm job ID")
    return job


def require_distinct_jobs(a1_job: object, current_a2_job: object, label: str) -> None:
    require(positive_decimal_job(a1_job, f"{label}.a1") != positive_decimal_job(current_a2_job, f"{label}.current_a2"), f"{label}: A2 must use a distinct Slurm job")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuditError("JSON identity cannot be canonically encoded") from exc
    return hashlib.sha256(payload).hexdigest()


def strict_json_equal(left: object, right: object) -> bool:
    """Compare JSON values recursively without Python's bool/int coercion."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        if not all(type(key) is str for key in left) or not all(type(key) is str for key in right):
            return False
        return set(left) == set(right) and all(strict_json_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(strict_json_equal(a, b) for a, b in zip(left, right, strict=True))
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and left == right
    return type(left) in (str, int, bool, type(None)) and left == right


def reject_nonfinite_json_constant(constant: str) -> object:
    raise ValueError(f"non-finite JSON constant: {constant}")


def normalized_path(value: object, label: str) -> str:
    return text(value, label).replace("\\", "/")


def require_identity_path(record: Mapping[str, Any], expected: Path, label: str) -> None:
    require(normalized_path(record["path"], f"{label}.path") == str(expected).replace("\\", "/"), f"{label}: recorded identity path drift")


def write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read(path: Path, expected: str, label: str) -> tuple[Mapping[str, Any], str]:
    sha_text(expected, f"{label}.expected_sha")
    try:
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        value = obj(json.loads(payload.decode("utf-8"), parse_constant=reject_nonfinite_json_constant), label)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label}: unreadable or invalid JSON") from exc
    require(actual == expected, f"{label}: artifact SHA mismatch")
    return value, actual


def percentile(values: Sequence[float], q: float) -> float:
    require(len(values) == SAMPLES, "raw sample count drift")
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lo = int(point)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def samples(value: object, label: str) -> list[float]:
    answer = [floating(item, f"{label}[{index}]") for index, item in enumerate(arr(value, label))]
    require(len(answer) == SAMPLES and all(item > 0.0 for item in answer), f"{label}: requires {SAMPLES} positive float values")
    return answer


def validate_file_identity(value: object, expected_sha: str | None, label: str, revalidate_files: bool) -> Mapping[str, Any]:
    record = obj(value, label)
    exact_keys(record, {"path", "sha256"}, label)
    path_text = text(record["path"], f"{label}.path")
    recorded_sha = sha_text(record["sha256"], f"{label}.sha256")
    if expected_sha is not None:
        require(recorded_sha == expected_sha, f"{label}: pinned SHA drift")
    if revalidate_files:
        try:
            path = Path(path_text).resolve(strict=True)
            require(path.is_file(), f"{label}: identity target is not a regular file")
            require(sha(path) == recorded_sha, f"{label}: current file SHA differs from recorded identity")
        except OSError as exc:
            raise AuditError(f"{label}: identity target unavailable") from exc
    return record


def validate_commit(value: object, expected_head: str, label: str, revalidate_files: bool) -> Mapping[str, Any]:
    record = obj(value, label)
    exact_keys(record, {"root", "head"}, label)
    root = text(record["root"], f"{label}.root")
    head = text(record["head"], f"{label}.head")
    require(head == expected_head, f"{label}: pinned commit drift")
    if revalidate_files:
        try:
            current = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise AuditError(f"{label}: cannot revalidate Git commit") from exc
        require(current == head, f"{label}: current Git head differs from recorded identity")
    return record


def validate_patched_dirty_overlay(value: object, label: str, revalidate_files: bool) -> Mapping[str, Any]:
    overlay = obj(value, label)
    exact_keys(overlay, {"root", "git_status_porcelain_v1", "files"}, label)
    root = text(overlay["root"], f"{label}.root")
    statuses = obj(overlay["git_status_porcelain_v1"], f"{label}.git_status_porcelain_v1")
    exact_keys(statuses, set(PATCHED_DIRTY_FILES), f"{label}.git_status_porcelain_v1")
    for relative in PATCHED_DIRTY_FILES:
        require(statuses[relative] == PATCHED_DIRTY_STATUS, f"{label}: {relative} tracked status drift")
    files = obj(overlay["files"], f"{label}.files")
    exact_keys(files, set(PATCHED_DIRTY_FILES), f"{label}.files")
    for relative, expected_sha in PATCHED_DIRTY_FILES.items():
        file_record = validate_file_identity(files[relative], expected_sha, f"{label}.files.{relative}", revalidate_files)
        require(text(file_record["path"], f"{label}.files.{relative}.path").replace("\\", "/") == str(Path(root) / relative).replace("\\", "/"), f"{label}: {relative} identity path escapes patched root")
    if revalidate_files:
        try:
            actual_lines = [line for line in subprocess.run(["git", "-C", root, "status", "--porcelain=v1", "--untracked-files=no"], check=True, text=True, capture_output=True).stdout.splitlines() if line]
        except (OSError, subprocess.CalledProcessError) as exc:
            raise AuditError(f"{label}: cannot revalidate patched tracked dirty set") from exc
        expected_lines = {f"{PATCHED_DIRTY_STATUS} {relative}" for relative in PATCHED_DIRTY_FILES}
        require(len(actual_lines) == len(expected_lines) and set(actual_lines) == expected_lines, f"{label}: patched tracked dirty set drift")
    return overlay


def validate_protocol_identity(value: object, runner_sha: str, analyzer_sha: str, shell_sha: str, label: str, revalidate_files: bool) -> Mapping[str, Any]:
    protocol = obj(value, label)
    exact_keys(protocol, {"runner", "analyzer", "protocol_shell", "extension", "flash_kda_python", "auto_dispatch", "fla_backend", "harness", "reference_torch_ref", "pinned_reference_helper", "commits", "patched_dirty_overlay", "fla_source_map"}, label)
    validate_file_identity(protocol["runner"], runner_sha, f"{label}.runner", revalidate_files)
    validate_file_identity(protocol["analyzer"], analyzer_sha, f"{label}.analyzer", revalidate_files)
    validate_file_identity(protocol["protocol_shell"], shell_sha, f"{label}.protocol_shell", revalidate_files)
    validate_file_identity(protocol["extension"], EXPECTED_SO, f"{label}.extension", revalidate_files)
    validate_file_identity(protocol["flash_kda_python"], EXPECTED_FLASH_KDA_PYTHON, f"{label}.flash_kda_python", revalidate_files)
    validate_file_identity(protocol["auto_dispatch"], EXPECTED_AUTO_DISPATCH, f"{label}.auto_dispatch", revalidate_files)
    validate_file_identity(protocol["fla_backend"], EXPECTED_FLA_BACKEND, f"{label}.fla_backend", revalidate_files)
    validate_file_identity(protocol["harness"], EXPECTED_HARNESS, f"{label}.harness", revalidate_files)
    validate_file_identity(protocol["reference_torch_ref"], None, f"{label}.reference_torch_ref", revalidate_files)
    helper = validate_file_identity(protocol["pinned_reference_helper"], PINNED_REFERENCE_HELPER_SHA256, f"{label}.pinned_reference_helper", revalidate_files)
    require(normalized_path(helper["path"], f"{label}.pinned_reference_helper.path") == PINNED_REFERENCE_HELPER_PATH, f"{label}: pinned reference helper path drift")
    commits = obj(protocol["commits"], f"{label}.commits")
    exact_keys(commits, {"patched", "reference", "fla"}, f"{label}.commits")
    patched_commit = validate_commit(commits["patched"], EXPECTED_PATCHED_COMMIT, f"{label}.commits.patched", revalidate_files)
    reference_commit = validate_commit(commits["reference"], EXPECTED_REFERENCE_COMMIT, f"{label}.commits.reference", revalidate_files)
    fla_commit = validate_commit(commits["fla"], EXPECTED_FLA_COMMIT, f"{label}.commits.fla", revalidate_files)
    overlay = validate_patched_dirty_overlay(protocol["patched_dirty_overlay"], f"{label}.patched_dirty_overlay", revalidate_files)
    require(overlay["root"] == patched_commit["root"], f"{label}: patched overlay root differs from pinned patched commit root")
    patched_root = Path(text(patched_commit["root"], f"{label}.commits.patched.root"))
    reference_root = Path(text(reference_commit["root"], f"{label}.commits.reference.root"))
    fla_root = Path(text(fla_commit["root"], f"{label}.commits.fla.root"))
    extension_path = Path(text(obj(protocol["extension"], f"{label}.extension")["path"], f"{label}.extension.path"))
    require(str(extension_path.parent).replace("\\", "/") == str(patched_root).replace("\\", "/"), f"{label}: extension identity escapes patched root")
    require_identity_path(obj(protocol["flash_kda_python"], f"{label}.flash_kda_python"), patched_root / "flash_kda" / "__init__.py", f"{label}.flash_kda_python")
    require_identity_path(obj(protocol["reference_torch_ref"], f"{label}.reference_torch_ref"), reference_root / "tests" / "torch_ref.py", f"{label}.reference_torch_ref")
    source_map = obj(protocol["fla_source_map"], f"{label}.fla_source_map")
    exact_keys(source_map, set(FLA_FILES), f"{label}.fla_source_map")
    for relative, expected_sha in FLA_FILES.items():
        record = validate_file_identity(source_map[relative], expected_sha, f"{label}.fla_source_map.{relative}", revalidate_files)
        require_identity_path(record, fla_root / relative, f"{label}.fla_source_map.{relative}")
    return protocol


def validate_runtime_identity(value: object, helper: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    runtime = obj(value, label)
    exact_keys(runtime, {"device", "gpu_uuid", "pinned_reference_helper_load"}, label)
    device = obj(runtime["device"], f"{label}.device")
    exact_keys(device, {"name", "capability", "multiprocessor_count", "gate_pass"}, f"{label}.device")
    require("B300" in text(device["name"], f"{label}.device.name").upper(), f"{label}: not B300")
    capability = arr(device["capability"], f"{label}.device.capability")
    require(len(capability) == 2 and integer(capability[0], f"{label}.device.capability[0]") == 10 and integer(capability[1], f"{label}.device.capability[1]") == 3, f"{label}: capability drift")
    require(integer(device["multiprocessor_count"], f"{label}.device.sm") == 148 and boolean(device["gate_pass"], f"{label}.device.gate"), f"{label}: B300 device gate failed")
    require(bool(text(runtime["gpu_uuid"], f"{label}.gpu_uuid")), f"{label}: GPU UUID absent")
    proof = obj(runtime["pinned_reference_helper_load"], f"{label}.pinned_reference_helper_load")
    exact_keys(proof, {"path", "sha256", "load_contract", "intercepted_names", "no_build"}, f"{label}.pinned_reference_helper_load")
    require(normalized_path(proof["path"], f"{label}.pinned_reference_helper_load.path") == PINNED_REFERENCE_HELPER_PATH, f"{label}: runtime helper path drift")
    require(sha_text(proof["sha256"], f"{label}.pinned_reference_helper_load.sha256") == PINNED_REFERENCE_HELPER_SHA256, f"{label}: runtime helper SHA drift")
    require(text(proof["load_contract"], f"{label}.pinned_reference_helper_load.load_contract") == PINNED_REFERENCE_HELPER_LOAD_CONTRACT, f"{label}: runtime helper load contract drift")
    intercepted = arr(proof["intercepted_names"], f"{label}.pinned_reference_helper_load.intercepted_names")
    require(len(intercepted) == 1 and text(intercepted[0], f"{label}.pinned_reference_helper_load.intercepted_names[0]") == "sigmoid_ext", f"{label}: helper must intercept exactly one sigmoid_ext load_inline")
    require(boolean(proof["no_build"], f"{label}.pinned_reference_helper_load.no_build") is True, f"{label}: helper loader must prove no-build")
    require(
        strict_json_equal(
            {"path": proof["path"], "sha256": proof["sha256"]},
            {"path": helper["path"], "sha256": helper["sha256"]},
        ),
        f"{label}: runtime helper proof is not bound to protocol helper identity",
    )
    return runtime


def validate_identity(value: object, runner_sha: str, analyzer_sha: str, shell_sha: str, label: str, revalidate_files: bool) -> Mapping[str, Any]:
    identity = obj(value, label)
    exact_keys(identity, {"protocol", "runtime"}, label)
    protocol = validate_protocol_identity(identity["protocol"], runner_sha, analyzer_sha, shell_sha, f"{label}.protocol", revalidate_files)
    runtime = validate_runtime_identity(identity["runtime"], obj(protocol["pinned_reference_helper"], f"{label}.protocol.pinned_reference_helper"), f"{label}.runtime")
    return {"protocol": protocol, "runtime": runtime}


def validate_immutable(value: object, label: str) -> None:
    immutable = obj(value, label)
    exact_keys(immutable, {"input_immutability_exact", "input_immutability_fields", "initial_state_immutability_exact"}, label)
    require(boolean(immutable["input_immutability_exact"], f"{label}.input_exact") and boolean(immutable["initial_state_immutability_exact"], f"{label}.initial_exact"), f"{label}: mutability gate failed")
    fields = obj(immutable["input_immutability_fields"], f"{label}.fields")
    exact_keys(fields, set(INPUT_FIELDS), f"{label}.fields")
    for field in INPUT_FIELDS:
        require(boolean(fields[field], f"{label}.fields.{field}"), f"{label}: {field} changed")


def validate_comparison(value: object, contract: str, label: str) -> None:
    comparison = obj(value, label)
    expected = {"output_exact", "output_max_abs", "final_state_present"}
    if contract != "none":
        expected |= {"final_state_exact", "final_state_max_abs"}
    exact_keys(comparison, expected, label)
    require(boolean(comparison["output_exact"], f"{label}.output_exact"), f"{label}: output mismatch")
    require(floating(comparison["output_max_abs"], f"{label}.output_max_abs") >= 0.0, f"{label}: negative output max")
    if contract == "none":
        require(comparison["final_state_present"] is False, f"{label}: none contract unexpectedly has a final state")
    else:
        require(comparison["final_state_present"] is True and boolean(comparison["final_state_exact"], f"{label}.final_state_exact"), f"{label}: non-none state missing/mismatched")
        require(floating(comparison["final_state_max_abs"], f"{label}.final_state_max_abs") >= 0.0, f"{label}: negative state max")


def validate_raw(value: object, label: str) -> None:
    raw = obj(value, label)
    exact_keys(raw, set(RAW_CONTRACTS), label)
    compare_names = ("baseline_vs_pinned_torch_reference", "vshard2_vs_pinned_torch_reference", "vshard2_vs_baseline")
    for contract in RAW_CONTRACTS:
        record = obj(raw[contract], f"{label}.{contract}")
        exact_keys(record, {*compare_names, "immutability", "passed"}, f"{label}.{contract}")
        require(boolean(record["passed"], f"{label}.{contract}.passed"), f"{label}.{contract}: runner raw failure")
        for compare_name in compare_names:
            validate_comparison(record[compare_name], contract, f"{label}.{contract}.{compare_name}")
        immutable = obj(record["immutability"], f"{label}.{contract}.immutability")
        exact_keys(immutable, {"reference", "baseline", "vshard2_p2"}, f"{label}.{contract}.immutability")
        for path in ("reference", "baseline", "vshard2_p2"):
            validate_immutable(immutable[path], f"{label}.{contract}.immutability.{path}")


def validate_negatives(value: object, label: str) -> None:
    negatives = obj(value, label)
    exact_keys(negatives, {"production_source_unmodified", "controls", "passed"}, label)
    require(boolean(negatives["production_source_unmodified"], f"{label}.production_source_unmodified") and boolean(negatives["passed"], f"{label}.passed"), f"{label}: negative top gate failed")
    controls = obj(negatives["controls"], f"{label}.controls")
    expected = {f"b{batch}/{contract}" for batch in (7, 8) for contract in RAW_CONTRACTS}
    exact_keys(controls, expected, f"{label}.controls")
    for key in sorted(expected):
        record = obj(controls[key], f"{label}.controls.{key}")
        exact_keys(record, {"requested_variant", "chosen_variant", "reason", "passed"}, f"{label}.controls.{key}")
        require(record["requested_variant"] == "baseline" and record["chosen_variant"] == "baseline" and record["reason"] == NEGATIVE_REASON and boolean(record["passed"], f"{label}.controls.{key}.passed"), f"{label}: {key} escaped exact baseline negative control")


def validate_c1_decision(value: object, label: str) -> None:
    decision = obj(value, label)
    exact_keys(decision, {"requested_variant", "chosen_variant", "reason", "extension_sha256", "test_only_route", "production_source_mutated"}, label)
    require(decision["requested_variant"] == "vshard2_p2" and decision["chosen_variant"] == "vshard2_p2" and decision["reason"] == TEST_ROUTE_REASON and decision["extension_sha256"] == EXPECTED_SO and decision["test_only_route"] is True and decision["production_source_mutated"] is False, f"{label}: exact test-only decision drift")


def validate_public_proof(value: object, path: str, label: str) -> None:
    proof = obj(value, label)
    exact_keys(proof, {"c1_spy_delta", "pinned_spy_delta", "decision", "passed"}, label)
    require(boolean(proof["passed"], f"{label}.passed"), f"{label}: route proof marked failed")
    if path == "pinned_public":
        require(integer(proof["c1_spy_delta"], f"{label}.c1") == 0 and integer(proof["pinned_spy_delta"], f"{label}.pinned") == 1 and proof["decision"] is None, f"{label}: pinned path proof drift")
    else:
        require(integer(proof["c1_spy_delta"], f"{label}.c1") == 1 and integer(proof["pinned_spy_delta"], f"{label}.pinned") == 0, f"{label}: C1 spy proof drift")
        validate_c1_decision(proof["decision"], f"{label}.decision")


def validate_count_map(value: object, expected: Mapping[str, int], label: str) -> None:
    counts = obj(value, label)
    exact_keys(counts, set(expected), label)
    for path, answer in expected.items():
        require(integer(counts[path], f"{label}.{path}") == answer, f"{label}: {path} count drift")


def validate_timed_route_checks(value: object, label: str) -> None:
    checks = obj(value, label)
    exact_keys(checks, set(PATHS), label)
    expected = {
        "pinned_public": {"calls": SAMPLES, "c1_spy_delta_total": 0, "pinned_spy_delta_total": SAMPLES, "decision_checks": 0},
        "c1_test_route_public": {"calls": SAMPLES, "c1_spy_delta_total": SAMPLES, "pinned_spy_delta_total": 0, "decision_checks": SAMPLES},
    }
    for path in PATHS:
        check = obj(checks[path], f"{label}.{path}")
        exact_keys(check, set(expected[path]), f"{label}.{path}")
        for name, answer in expected[path].items():
            require(integer(check[name], f"{label}.{path}.{name}") == answer, f"{label}.{path}: {name} drift")


def validate_post_restore(value: object, label: str) -> None:
    proof = obj(value, label)
    exact_keys(proof, {"test_route_dispatcher_restored", "dispatcher_identity_matches_production", "c1_backend_spy_restored", "pinned_backend_spy_restored", "passed"}, label)
    for name in ("test_route_dispatcher_restored", "dispatcher_identity_matches_production", "c1_backend_spy_restored", "pinned_backend_spy_restored", "passed"):
        require(boolean(proof[name], f"{label}.{name}"), f"{label}: {name} must be true")


def validate_repeat(value: object, process: int, repeat: int, label: str) -> dict[str, object]:
    record = obj(value, label)
    exact_keys(record, {"process_index", "repeat_index", "event_contract", "schedule", "first_path_counts", "warmup_public_call_counts", "timed_public_call_counts", "timed_route_checks", "timed_route_checks_without_gpu_sync", "public_precheck", "input_immutability_exact", "input_immutability_fields", "initial_state_immutability_exact", "raw_samples_ms", "paths", "c1_margin_over_pinned_by_percentile", "winner_by_percentile", "repeat_gate_pass", "passed"}, label)
    require(integer(record["process_index"], f"{label}.process_index") == process and integer(record["repeat_index"], f"{label}.repeat_index") == repeat, f"{label}: process/repeat drift")
    require(record["event_contract"] == TIMED_EVENT_CONTRACT and record["schedule"] == TIMED_SCHEDULE, f"{label}: event/schedule contract drift")
    validate_count_map(record["first_path_counts"], {"pinned_public": SAMPLES // 2, "c1_test_route_public": SAMPLES // 2}, f"{label}.first_path_counts")
    validate_count_map(record["warmup_public_call_counts"], {"pinned_public": WARMUP, "c1_test_route_public": WARMUP}, f"{label}.warmup_public_call_counts")
    validate_count_map(record["timed_public_call_counts"], {"pinned_public": SAMPLES, "c1_test_route_public": SAMPLES}, f"{label}.timed_public_call_counts")
    require(record["timed_route_checks_without_gpu_sync"] is True, f"{label}: timed route checks claim a GPU sync")
    validate_timed_route_checks(record["timed_route_checks"], f"{label}.timed_route_checks")
    precheck = obj(record["public_precheck"], f"{label}.public_precheck")
    exact_keys(precheck, {"pinned", "c1_test_route", "exact"}, f"{label}.public_precheck")
    validate_public_proof(precheck["pinned"], "pinned_public", f"{label}.public_precheck.pinned")
    validate_public_proof(precheck["c1_test_route"], "c1_test_route_public", f"{label}.public_precheck.c1_test_route")
    validate_comparison(precheck["exact"], "none", f"{label}.public_precheck.exact")
    validate_immutable({"input_immutability_exact": record["input_immutability_exact"], "input_immutability_fields": record["input_immutability_fields"], "initial_state_immutability_exact": record["initial_state_immutability_exact"]}, f"{label}.performance_immutability")
    paths = obj(record["paths"], f"{label}.paths")
    raw = obj(record["raw_samples_ms"], f"{label}.raw_samples_ms")
    exact_keys(paths, set(PATHS), f"{label}.paths")
    exact_keys(raw, set(PATHS), f"{label}.raw_samples_ms")
    measures: dict[str, dict[str, float]] = {}
    for path in PATHS:
        raw_values = samples(raw[path], f"{label}.raw_samples_ms.{path}")
        observed = obj(paths[path], f"{label}.paths.{path}")
        exact_keys(observed, {"samples", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"}, f"{label}.paths.{path}")
        expected_summary = {
            "mean_ms": statistics.fmean(raw_values),
            "p50_ms": percentile(raw_values, .50),
            "p95_ms": percentile(raw_values, .95),
            "p99_ms": percentile(raw_values, .99),
            "min_ms": min(raw_values),
            "max_ms": max(raw_values),
        }
        require(integer(observed["samples"], f"{label}.paths.{path}.samples") == SAMPLES, f"{label}: {path} sample count drift")
        for metric, answer in expected_summary.items():
            require(close(floating(observed[metric], f"{label}.paths.{path}.{metric}"), answer), f"{label}: {path} {metric} summary mismatch")
        measures[path] = expected_summary
    reported_margins = obj(record["c1_margin_over_pinned_by_percentile"], f"{label}.margins")
    exact_keys(reported_margins, set(PERCENTILES), f"{label}.margins")
    margins = {name: measures["pinned_public"][f"{name}_ms"] / measures["c1_test_route_public"][f"{name}_ms"] - 1.0 for name in PERCENTILES}
    winners = obj(record["winner_by_percentile"], f"{label}.winner_by_percentile")
    exact_keys(winners, set(PERCENTILES), f"{label}.winner_by_percentile")
    for name, answer in margins.items():
        require(close(floating(reported_margins[name], f"{label}.margins.{name}"), answer), f"{label}: {name} margin mismatch")
        require(winners[name] == ("c1_test_route_public" if answer > 0.0 else "pinned_public"), f"{label}: {name} winner mismatch")
    gate = all(answer >= MIN_MARGIN for answer in margins.values())
    require(record["repeat_gate_pass"] is gate and boolean(record["passed"], f"{label}.passed"), f"{label}: repeat gate/pass drift")
    return {"process_index": process, "repeat_index": repeat, "margins": margins, "repeat_gate_pass": gate}


def validate_main(value: Mapping[str, Any], allocation: str, process: int, runner_sha: str, analyzer_sha: str, shell_sha: str, label: str, revalidate_files: bool = True) -> dict[str, object]:
    exact_keys(value, {"schema_version", "artifact_kind", "allocation_id", "process_index", "pid", "slurm_job_id", "shape", "public_contract", "raw_abi_contracts", "identity", "artifact_content_identity", "raw_abi_correctness", "negative_controls", "public_benchmarks", "post_restore_proof", "complete"}, label)
    require(integer(value["schema_version"], f"{label}.schema_version") == SCHEMA_VERSION and value["artifact_kind"] == "fresh_b7_none_vshard2_p2_main" and value["allocation_id"] == allocation and integer(value["process_index"], f"{label}.process_index") == process and integer(value["pid"], f"{label}.pid") > 0 and boolean(value["complete"], f"{label}.complete"), f"{label}: main top-level drift")
    job = positive_decimal_job(value["slurm_job_id"], f"{label}.slurm_job_id")
    shape = obj(value["shape"], f"{label}.shape")
    exact_keys(shape, {"B", "H", "T", "K", "V"}, f"{label}.shape")
    require([integer(shape[name], f"{label}.shape.{name}") for name in ("B", "H", "T", "K", "V")] == [7, 12, 2048, 128, 128], f"{label}: shape drift")
    require(value["public_contract"] == "none" and arr(value["raw_abi_contracts"], f"{label}.raw_abi_contracts") == list(RAW_CONTRACTS), f"{label}: contract drift")
    identity = validate_identity(value["identity"], runner_sha, analyzer_sha, shell_sha, f"{label}.identity", revalidate_files)
    content = obj(value["artifact_content_identity"], f"{label}.artifact_content_identity")
    exact_keys(content, {"allocation_id", "process_index", "protocol_identity_sha256", "runtime_identity"}, f"{label}.artifact_content_identity")
    require(content["allocation_id"] == allocation and integer(content["process_index"], f"{label}.artifact_content_identity.process_index") == process and sha_text(content["protocol_identity_sha256"], f"{label}.artifact_content_identity.protocol_identity_sha256") == canonical_sha(identity["protocol"]) and strict_json_equal(obj(content["runtime_identity"], f"{label}.artifact_content_identity.runtime_identity"), identity["runtime"]), f"{label}: content identity drift")
    validate_raw(value["raw_abi_correctness"], f"{label}.raw_abi_correctness")
    validate_negatives(value["negative_controls"], f"{label}.negative_controls")
    repeats = arr(value["public_benchmarks"], f"{label}.public_benchmarks")
    require(len(repeats) == REPEATS, f"{label}: public repeat count drift")
    assessment = [validate_repeat(item, process, index, f"{label}.repeat{index}") for index, item in enumerate(repeats)]
    validate_post_restore(value["post_restore_proof"], f"{label}.post_restore_proof")
    return {"pid": value["pid"], "slurm_job_id": job, "protocol_identity": identity["protocol"], "runtime_identity": identity["runtime"], "content_identity": content, "repeats": assessment}


def expected_allocation_decision(allocation: str, eligible: bool) -> str:
    if not eligible:
        return "STOP_keep_production_baseline"
    return "eligible_for_A2_only" if allocation == "A1" else "eligible_for_cross_allocation_freeze_review"


def full_source_identity(value: Mapping[str, Any], label: str) -> dict[str, object]:
    """Return the whole source/protocol identity that A2 must inherit from A1.

    ``source_identity`` alone carries the auditor and externally supplied
    digest arguments, while ``protocol_identity`` contains the full physical
    runner/SO/worktree/FLA ledger.  Persist both to make the A1 binding useful
    after the pre-launch shell process has exited.
    """

    source = obj(value["source_identity"], f"{label}.source_identity")
    protocol = obj(value["protocol_identity"], f"{label}.protocol_identity")
    runtime = obj(value["runtime_identity"], f"{label}.runtime_identity")
    return {
        "allocation_source_identity": dict(source),
        "protocol_identity": dict(protocol),
        "pinned_reference_helper_load": dict(obj(runtime["pinned_reference_helper_load"], f"{label}.runtime_identity.pinned_reference_helper_load")),
    }


def reopen_a1_prerequisite(
    audit_path: Path,
    expected_audit_sha256: str,
    current_a2_job: object,
    runner_sha: str,
    analyzer_sha: str,
    shell_sha: str,
    label: str,
    *,
    revalidate_files: bool = True,
) -> dict[str, object]:
    """Reopen the exact A1 audit and all of its signed main evidence.

    This is deliberately shared by the precondition, A2 allocation writer,
    allocation validator, and final chain.  A caller cannot turn a prior
    precondition into an unaudited A2 binding merely by changing shell
    variables between those two phases.
    """

    expected_audit_sha256 = sha_text(expected_audit_sha256, f"{label}.expected_a1_sha256")
    current = positive_decimal_job(current_a2_job, f"{label}.current_a2_job")
    try:
        resolved = audit_path.resolve(strict=True)
    except OSError as exc:
        raise AuditError(f"{label}: A1 audit path is unavailable") from exc
    audit, audit_sha = read(resolved, expected_audit_sha256, f"{label}.A1_audit")
    checked = validate_allocation(
        audit,
        "A1",
        runner_sha,
        analyzer_sha,
        shell_sha,
        f"{label}.A1_audit",
        revalidate_files=revalidate_files,
    )
    require(checked["eligible"] is True, f"{label}: A1 prerequisite requires exact eligible=true")
    require_distinct_jobs(checked["slurm_job_id"], current, f"{label}.A1_job")
    return {
        "path": str(resolved),
        "sha256": audit_sha,
        "slurm_job_id": checked["slurm_job_id"],
        "full_source_identity": full_source_identity(audit, f"{label}.A1_audit"),
    }


def allocation_mode(args: argparse.Namespace) -> None:
    analyzer_path = Path(__file__).resolve(strict=True)
    analyzer_sha = sha(analyzer_path)
    require(analyzer_sha == args.expected_analyzer_sha256, "analyzer SHA mismatch")
    records = [read(path, expected, f"main{index}") for index, (path, expected) in enumerate(zip(args.main_json, args.expected_main_sha256, strict=True))]
    checked = [validate_main(record, args.allocation, index, args.expected_runner_sha256, analyzer_sha, args.expected_protocol_shell_sha256, f"main{index}") for index, (record, _digest) in enumerate(records)]
    pids = [integer(item["pid"], f"main{index}.pid") for index, item in enumerate(checked)]
    jobs = {positive_decimal_job(item["slurm_job_id"], f"main{index}.job") for index, item in enumerate(checked)}
    protocols = [obj(item["protocol_identity"], f"main{index}.protocol") for index, item in enumerate(checked)]
    runtimes = [obj(item["runtime_identity"], f"main{index}.runtime") for index, item in enumerate(checked)]
    require(len(set(pids)) == 2 and len(jobs) == 1 and len({canonical_sha(protocol) for protocol in protocols}) == 1 and len({canonical_sha(runtime) for runtime in runtimes}) == 1, f"{args.allocation}: fresh PID/job/protocol/runtime identity gate failed")
    all_repeats: list[Mapping[str, object]] = []
    for item in checked:
        all_repeats.extend(item["repeats"])  # type: ignore[arg-type]
    eligible = len(all_repeats) == 4 and all(boolean(item["repeat_gate_pass"], "repeat_gate_pass") for item in all_repeats)
    decision = expected_allocation_decision(args.allocation, eligible)
    protocol = protocols[0]
    runtime = runtimes[0]
    source_identity: dict[str, object] = {
        "analyzer": {"path": str(analyzer_path), "sha256": analyzer_sha},
        "expected_runner_sha256": args.expected_runner_sha256,
        "expected_protocol_shell_sha256": args.expected_protocol_shell_sha256,
    }
    current_source = full_source_identity(
        {
            "source_identity": source_identity,
            "protocol_identity": protocol,
            "runtime_identity": {"pinned_reference_helper_load": runtime["pinned_reference_helper_load"]},
        },
        f"{args.allocation}.current_source",
    )
    a1_prerequisite: object = None
    if args.allocation == "A1":
        require(
            args.a1_audit is None and args.expected_a1_sha256 is None and args.current_slurm_job_id is None,
            "A1 allocation rejects A1 prerequisite arguments",
        )
    else:
        require(
            args.a1_audit is not None and args.expected_a1_sha256 is not None and args.current_slurm_job_id is not None,
            "A2 allocation requires A1 audit path/SHA and current Slurm job",
        )
        current_job = positive_decimal_job(args.current_slurm_job_id, "A2 allocation current Slurm job")
        allocation_job = next(iter(jobs))
        require(current_job == allocation_job, "A2 allocation current Slurm job differs from its raw main artifacts")
        a1_prerequisite = reopen_a1_prerequisite(
            args.a1_audit,
            args.expected_a1_sha256,
            current_job,
            args.expected_runner_sha256,
            analyzer_sha,
            args.expected_protocol_shell_sha256,
            "A2 allocation",
        )
        prerequisite_source = obj(
            obj(a1_prerequisite, "A2 allocation prerequisite")["full_source_identity"],
            "A2 allocation prerequisite.full_source_identity",
        )
        require(
            canonical_sha(prerequisite_source) == canonical_sha(current_source),
            "A2 allocation source/protocol identity differs from its exact A1 prerequisite",
        )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fresh_b7_none_vshard2_p2_allocation_audit",
        "purpose": "fresh B7 none allocation audit; historic discovery excluded",
        "allocation_id": args.allocation,
        "source_identity": source_identity,
        "protocol_identity": protocol,
        "runtime_identity": {
            "device": runtime["device"],
            "gpu_uuid": runtime["gpu_uuid"],
            "pinned_reference_helper_load": runtime["pinned_reference_helper_load"],
            "slurm_job_id": next(iter(jobs)),
            "pids": pids,
        },
        "main_artifacts": [
            {"process_index": index, "path": str(path.resolve(strict=True)), "sha256": digest, "content_identity": checked[index]["content_identity"]}
            for index, (path, (_record, digest)) in enumerate(zip(args.main_json, records, strict=True))
        ],
        "repeat_assessment": all_repeats,
        "a1_prerequisite": a1_prerequisite,
        "allocation_gate": {"eligible": eligible, "decision": decision},
        "complete": True,
    }
    write(args.json, payload)
    print(f"wrote {args.allocation} B7-none audit {args.json}; {decision}")


def validate_allocation(value: Mapping[str, Any], allocation: str, runner_sha: str, analyzer_sha: str, shell_sha: str, label: str, revalidate_files: bool = True) -> dict[str, object]:
    exact_keys(value, {"schema_version", "kind", "purpose", "allocation_id", "source_identity", "protocol_identity", "runtime_identity", "main_artifacts", "repeat_assessment", "a1_prerequisite", "allocation_gate", "complete"}, label)
    require(integer(value["schema_version"], f"{label}.schema_version") == SCHEMA_VERSION and value["kind"] == "fresh_b7_none_vshard2_p2_allocation_audit" and value["purpose"] == "fresh B7 none allocation audit; historic discovery excluded" and value["allocation_id"] == allocation and boolean(value["complete"], f"{label}.complete"), f"{label}: allocation top-level drift")
    source = obj(value["source_identity"], f"{label}.source_identity")
    exact_keys(source, {"analyzer", "expected_runner_sha256", "expected_protocol_shell_sha256"}, f"{label}.source_identity")
    validate_file_identity(source["analyzer"], analyzer_sha, f"{label}.source_identity.analyzer", revalidate_files)
    require(source["expected_runner_sha256"] == runner_sha and source["expected_protocol_shell_sha256"] == shell_sha, f"{label}: external source identity drift")
    protocol = validate_protocol_identity(value["protocol_identity"], runner_sha, analyzer_sha, shell_sha, f"{label}.protocol_identity", revalidate_files)
    runtime = obj(value["runtime_identity"], f"{label}.runtime_identity")
    exact_keys(runtime, {"device", "gpu_uuid", "pinned_reference_helper_load", "slurm_job_id", "pids"}, f"{label}.runtime_identity")
    validate_runtime_identity(
        {
            "device": runtime["device"],
            "gpu_uuid": runtime["gpu_uuid"],
            "pinned_reference_helper_load": runtime["pinned_reference_helper_load"],
        },
        obj(protocol["pinned_reference_helper"], f"{label}.protocol_identity.pinned_reference_helper"),
        f"{label}.runtime_identity",
    )
    job = positive_decimal_job(runtime["slurm_job_id"], f"{label}.runtime_identity.slurm_job_id")
    pids = [integer(item, f"{label}.runtime_identity.pids[{index}]") for index, item in enumerate(arr(runtime["pids"], f"{label}.runtime_identity.pids"))]
    require(len(pids) == 2 and len(set(pids)) == 2 and all(item > 0 for item in pids), f"{label}: fresh PID evidence drift")
    artifacts = arr(value["main_artifacts"], f"{label}.main_artifacts")
    require(len(artifacts) == 2, f"{label}: must retain exactly two main artifacts")
    recomputed: list[dict[str, object]] = []
    for index, artifact_value in enumerate(artifacts):
        artifact = obj(artifact_value, f"{label}.main_artifacts[{index}]")
        exact_keys(artifact, {"process_index", "path", "sha256", "content_identity"}, f"{label}.main_artifacts[{index}]")
        require(integer(artifact["process_index"], f"{label}.main_artifacts[{index}].process_index") == index, f"{label}: main process index drift")
        artifact_path = Path(text(artifact["path"], f"{label}.main_artifacts[{index}].path"))
        artifact_sha = sha_text(artifact["sha256"], f"{label}.main_artifacts[{index}].sha256")
        main, _actual_sha = read(artifact_path, artifact_sha, f"{label}.main_artifacts[{index}]")
        checked = validate_main(main, allocation, index, runner_sha, analyzer_sha, shell_sha, f"{label}.main_artifacts[{index}].content", revalidate_files)
        content_identity = obj(artifact["content_identity"], f"{label}.main_artifacts[{index}].content_identity")
        require(strict_json_equal(content_identity, checked["content_identity"]), f"{label}: main SHA/content identity mismatch")
        require(canonical_sha(obj(checked["protocol_identity"], "protocol")) == canonical_sha(protocol), f"{label}: main protocol identity differs from allocation identity")
        expected_runtime = {
            "device": runtime["device"],
            "gpu_uuid": runtime["gpu_uuid"],
            "pinned_reference_helper_load": runtime["pinned_reference_helper_load"],
        }
        require(strict_json_equal(obj(checked["runtime_identity"], "runtime"), expected_runtime) and integer(checked["pid"], "pid") == pids[index] and positive_decimal_job(checked["slurm_job_id"], "main slurm job") == job, f"{label}: main runtime identity differs from allocation identity")
        recomputed.append(checked)
    expected_repeats: list[Mapping[str, object]] = []
    for checked in recomputed:
        expected_repeats.extend(checked["repeats"])  # type: ignore[arg-type]
    reported_repeats = arr(value["repeat_assessment"], f"{label}.repeat_assessment")
    require(strict_json_equal(reported_repeats, expected_repeats), f"{label}: recomputed repeat assessment differs from audit")
    a1_prerequisite: object = value["a1_prerequisite"]
    if allocation == "A1":
        require(a1_prerequisite is None, f"{label}: A1 must not contain an A1 prerequisite")
    else:
        prerequisite = obj(a1_prerequisite, f"{label}.a1_prerequisite")
        exact_keys(prerequisite, {"path", "sha256", "slurm_job_id", "full_source_identity"}, f"{label}.a1_prerequisite")
        raw_path = Path(text(prerequisite["path"], f"{label}.a1_prerequisite.path"))
        require(raw_path.is_absolute(), f"{label}: A1 prerequisite path must be absolute")
        try:
            resolved_path = raw_path.resolve(strict=True)
        except OSError as exc:
            raise AuditError(f"{label}: A1 prerequisite path is unavailable") from exc
        require(str(raw_path) == str(resolved_path), f"{label}: A1 prerequisite path must be resolved/canonical")
        prerequisite_sha = sha_text(prerequisite["sha256"], f"{label}.a1_prerequisite.sha256")
        prerequisite_job = positive_decimal_job(prerequisite["slurm_job_id"], f"{label}.a1_prerequisite.slurm_job_id")
        require_distinct_jobs(prerequisite_job, job, f"{label}.a1_prerequisite")
        recorded_full_source = obj(prerequisite["full_source_identity"], f"{label}.a1_prerequisite.full_source_identity")
        exact_keys(recorded_full_source, {"allocation_source_identity", "protocol_identity", "pinned_reference_helper_load"}, f"{label}.a1_prerequisite.full_source_identity")
        reopened = reopen_a1_prerequisite(
            resolved_path,
            prerequisite_sha,
            job,
            runner_sha,
            analyzer_sha,
            shell_sha,
            f"{label}.a1_prerequisite",
            revalidate_files=revalidate_files,
        )
        require(
            prerequisite["path"] == reopened["path"]
            and prerequisite["sha256"] == reopened["sha256"]
            and prerequisite["slurm_job_id"] == reopened["slurm_job_id"]
            and canonical_sha(recorded_full_source) == canonical_sha(obj(reopened["full_source_identity"], f"{label}.reopened_full_source")),
            f"{label}: A2 prerequisite does not exactly bind the reopened A1 audit",
        )
        require(
            canonical_sha(recorded_full_source) == canonical_sha(full_source_identity(value, label)),
            f"{label}: A2 source/protocol identity differs from its bound A1 prerequisite",
        )
    gate = obj(value["allocation_gate"], f"{label}.allocation_gate")
    exact_keys(gate, {"eligible", "decision"}, f"{label}.allocation_gate")
    eligible = len(expected_repeats) == 4 and all(boolean(item["repeat_gate_pass"], "repeat_gate_pass") for item in expected_repeats)
    require(gate["eligible"] is eligible and gate["decision"] == expected_allocation_decision(allocation, eligible), f"{label}: allocation eligibility/decision is forged or stale")
    return {"protocol_identity": protocol, "runtime_identity": runtime, "main_artifacts": artifacts, "a1_prerequisite": a1_prerequisite, "eligible": eligible, "slurm_job_id": job}


def precondition_a1_mode(args: argparse.Namespace) -> None:
    analyzer_sha = sha(Path(__file__).resolve(strict=True))
    require(analyzer_sha == args.expected_analyzer_sha256, "analyzer SHA mismatch")
    reopen_a1_prerequisite(
        args.a1_audit,
        args.expected_a1_sha256,
        args.current_slurm_job_id,
        args.expected_runner_sha256,
        analyzer_sha,
        args.expected_protocol_shell_sha256,
        "A1 precondition",
    )
    print("A1_PRECONDITION_PASS: A1 exact eligible audit, artifact SHA/content identity, complete source/protocol identity, and distinct current A2 Slurm job revalidated")


def chain_mode(args: argparse.Namespace) -> None:
    analyzer_path = Path(__file__).resolve(strict=True)
    analyzer_sha = sha(analyzer_path)
    require(analyzer_sha == args.expected_analyzer_sha256, "analyzer SHA mismatch")
    a1, a1_sha = read(args.a1_audit, args.expected_a1_sha256, "A1")
    a2, a2_sha = read(args.a2_audit, args.expected_a2_sha256, "A2")
    first = validate_allocation(a1, "A1", args.expected_runner_sha256, analyzer_sha, args.expected_protocol_shell_sha256, "A1")
    second = validate_allocation(a2, "A2", args.expected_runner_sha256, analyzer_sha, args.expected_protocol_shell_sha256, "A2")
    require(first["eligible"] is True and second["eligible"] is True, "both allocations must be eligible")
    require_distinct_jobs(first["slurm_job_id"], second["slurm_job_id"], "A1/A2 jobs")
    first_protocol = obj(first["protocol_identity"], "A1 protocol")
    second_protocol = obj(second["protocol_identity"], "A2 protocol")
    require(canonical_sha(first_protocol) == canonical_sha(second_protocol), "A1/A2 complete source/protocol identity differs")
    first_full_source = full_source_identity(a1, "A1")
    second_full_source = full_source_identity(a2, "A2")
    require(canonical_sha(first_full_source) == canonical_sha(second_full_source), "A1/A2 full source identity differs")
    expected_a1_binding = {
        "path": str(args.a1_audit.resolve(strict=True)),
        "sha256": a1_sha,
        "slurm_job_id": first["slurm_job_id"],
        "full_source_identity": first_full_source,
    }
    actual_a1_binding = obj(second["a1_prerequisite"], "A2.a1_prerequisite")
    require(
        actual_a1_binding.get("path") == expected_a1_binding["path"]
        and actual_a1_binding.get("sha256") == expected_a1_binding["sha256"]
        and actual_a1_binding.get("slurm_job_id") == expected_a1_binding["slurm_job_id"]
        and canonical_sha(obj(actual_a1_binding.get("full_source_identity"), "A2.a1_prerequisite.full_source_identity")) == canonical_sha(first_full_source),
        "A2 allocation is not bound to the exact A1 audit/path/SHA/job/source identity supplied to chain",
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fresh_b7_none_vshard2_p2_cross_allocation_chain",
        "purpose": "fresh B7 none two-allocation public-freeze eligibility; no automatic production change",
        "source_identity": {
            "analyzer": {"path": str(analyzer_path), "sha256": analyzer_sha},
            "expected_runner_sha256": args.expected_runner_sha256,
            "expected_protocol_shell_sha256": args.expected_protocol_shell_sha256,
        },
        "protocol_identity": first_protocol,
        "allocations": {
            "A1": {"path": str(args.a1_audit.resolve(strict=True)), "sha256": a1_sha, "slurm_job_id": first["slurm_job_id"], "gpu_uuid": obj(first["runtime_identity"], "A1 runtime")["gpu_uuid"], "pinned_reference_helper_load": obj(first["runtime_identity"], "A1 runtime")["pinned_reference_helper_load"], "main_artifacts": first["main_artifacts"]},
            "A2": {"path": str(args.a2_audit.resolve(strict=True)), "sha256": a2_sha, "slurm_job_id": second["slurm_job_id"], "gpu_uuid": obj(second["runtime_identity"], "A2 runtime")["gpu_uuid"], "pinned_reference_helper_load": obj(second["runtime_identity"], "A2 runtime")["pinned_reference_helper_load"], "main_artifacts": second["main_artifacts"], "a1_prerequisite": actual_a1_binding},
        },
        "eligible_for_public_freeze": True,
        "automatic_production_mutation": False,
        "next_step": "manual review may consider only B7,H12,T2048,none -> vshard2-P2; B7 other states and B8 remain baseline",
        "complete": True,
    }
    write(args.json, payload)
    print(f"wrote B7-none A1/A2 chain {args.json}; eligible_for_public_freeze=true")


def synthetic_protocol(runner_sha: str, analyzer_sha: str, shell_sha: str) -> dict[str, object]:
    file_record = lambda path, digest: {"path": path, "sha256": digest}
    return {
        "runner": file_record("/synthetic/runner.py", runner_sha),
        "analyzer": file_record("/synthetic/analyzer.py", analyzer_sha),
        "protocol_shell": file_record("/synthetic/protocol.sh", shell_sha),
        "extension": file_record("/synthetic/patched/flash_kda_C.so", EXPECTED_SO),
        "flash_kda_python": file_record("/synthetic/patched/flash_kda/__init__.py", EXPECTED_FLASH_KDA_PYTHON),
        "auto_dispatch": file_record("/synthetic/auto_dispatch.py", EXPECTED_AUTO_DISPATCH),
        "fla_backend": file_record("/synthetic/fla_backend.py", EXPECTED_FLA_BACKEND),
        "harness": file_record("/synthetic/harness.py", EXPECTED_HARNESS),
        "reference_torch_ref": file_record("/synthetic/reference/tests/torch_ref.py", "d" * 64),
        "pinned_reference_helper": file_record(PINNED_REFERENCE_HELPER_PATH, PINNED_REFERENCE_HELPER_SHA256),
        "commits": {
            "patched": {"root": "/synthetic/patched", "head": EXPECTED_PATCHED_COMMIT},
            "reference": {"root": "/synthetic/reference", "head": EXPECTED_REFERENCE_COMMIT},
            "fla": {"root": "/synthetic/fla", "head": EXPECTED_FLA_COMMIT},
        },
        "patched_dirty_overlay": {
            "root": "/synthetic/patched",
            "git_status_porcelain_v1": {relative: PATCHED_DIRTY_STATUS for relative in PATCHED_DIRTY_FILES},
            "files": {relative: file_record(f"/synthetic/patched/{relative}", digest) for relative, digest in PATCHED_DIRTY_FILES.items()},
        },
        "fla_source_map": {relative: file_record(f"/synthetic/fla/{relative}", digest) for relative, digest in FLA_FILES.items()},
    }


def synthetic_main() -> tuple[dict[str, object], str, str, str]:
    runner_sha, analyzer_sha, shell_sha = "a" * 64, "b" * 64, "c" * 64
    protocol = synthetic_protocol(runner_sha, analyzer_sha, shell_sha)
    runtime = {
        "device": {"name": "NVIDIA B300", "capability": [10, 3], "multiprocessor_count": 148, "gate_pass": True},
        "gpu_uuid": "GPU-synthetic",
        "pinned_reference_helper_load": {
            "path": PINNED_REFERENCE_HELPER_PATH,
            "sha256": PINNED_REFERENCE_HELPER_SHA256,
            "load_contract": PINNED_REFERENCE_HELPER_LOAD_CONTRACT,
            "intercepted_names": ["sigmoid_ext"],
            "no_build": True,
        },
    }
    immutable = {"input_immutability_exact": True, "input_immutability_fields": {field: True for field in INPUT_FIELDS}, "initial_state_immutability_exact": True}
    raw: dict[str, object] = {}
    for contract in RAW_CONTRACTS:
        comparison: dict[str, object] = {"output_exact": True, "output_max_abs": 0.0, "final_state_present": contract != "none"}
        if contract != "none":
            comparison.update({"final_state_exact": True, "final_state_max_abs": 0.0})
        raw[contract] = {
            "baseline_vs_pinned_torch_reference": dict(comparison),
            "vshard2_vs_pinned_torch_reference": dict(comparison),
            "vshard2_vs_baseline": dict(comparison),
            "immutability": {"reference": copy.deepcopy(immutable), "baseline": copy.deepcopy(immutable), "vshard2_p2": copy.deepcopy(immutable)},
            "passed": True,
        }
    controls = {f"b{batch}/{contract}": {"requested_variant": "baseline", "chosen_variant": "baseline", "reason": NEGATIVE_REASON, "passed": True} for batch in (7, 8) for contract in RAW_CONTRACTS}
    proof_pinned = {"c1_spy_delta": 0, "pinned_spy_delta": 1, "decision": None, "passed": True}
    proof_c1 = {"c1_spy_delta": 1, "pinned_spy_delta": 0, "decision": {"requested_variant": "vshard2_p2", "chosen_variant": "vshard2_p2", "reason": TEST_ROUTE_REASON, "extension_sha256": EXPECTED_SO, "test_only_route": True, "production_source_mutated": False}, "passed": True}
    pinned_values, c1_values = [1.0] * SAMPLES, [0.98] * SAMPLES
    def summary_for(values: list[float]) -> dict[str, object]:
        return {"samples": SAMPLES, "mean_ms": statistics.fmean(values), "p50_ms": percentile(values, .50), "p95_ms": percentile(values, .95), "p99_ms": percentile(values, .99), "min_ms": min(values), "max_ms": max(values)}
    margin = 1.0 / .98 - 1.0
    repeat = {
        "process_index": 0,
        "repeat_index": 0,
        "event_contract": TIMED_EVENT_CONTRACT,
        "schedule": TIMED_SCHEDULE,
        "first_path_counts": {"pinned_public": 500, "c1_test_route_public": 500},
        "warmup_public_call_counts": {"pinned_public": WARMUP, "c1_test_route_public": WARMUP},
        "timed_public_call_counts": {"pinned_public": SAMPLES, "c1_test_route_public": SAMPLES},
        "timed_route_checks": {"pinned_public": {"calls": SAMPLES, "c1_spy_delta_total": 0, "pinned_spy_delta_total": SAMPLES, "decision_checks": 0}, "c1_test_route_public": {"calls": SAMPLES, "c1_spy_delta_total": SAMPLES, "pinned_spy_delta_total": 0, "decision_checks": SAMPLES}},
        "timed_route_checks_without_gpu_sync": True,
        "public_precheck": {"pinned": proof_pinned, "c1_test_route": proof_c1, "exact": {"output_exact": True, "output_max_abs": 0.0, "final_state_present": False}},
        **copy.deepcopy(immutable),
        "raw_samples_ms": {"pinned_public": pinned_values, "c1_test_route_public": c1_values},
        "paths": {"pinned_public": summary_for(pinned_values), "c1_test_route_public": summary_for(c1_values)},
        "c1_margin_over_pinned_by_percentile": {name: margin for name in PERCENTILES},
        "winner_by_percentile": {name: "c1_test_route_public" for name in PERCENTILES},
        "repeat_gate_pass": True,
        "passed": True,
    }
    main = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "fresh_b7_none_vshard2_p2_main",
        "allocation_id": "A1",
        "process_index": 0,
        "pid": 123,
        "slurm_job_id": "1",
        "shape": {"B": 7, "H": 12, "T": 2048, "K": 128, "V": 128},
        "public_contract": "none",
        "raw_abi_contracts": list(RAW_CONTRACTS),
        "identity": {"protocol": protocol, "runtime": runtime},
        "artifact_content_identity": {"allocation_id": "A1", "process_index": 0, "protocol_identity_sha256": canonical_sha(protocol), "runtime_identity": runtime},
        "raw_abi_correctness": raw,
        "negative_controls": {"production_source_unmodified": True, "controls": controls, "passed": True},
        "public_benchmarks": [repeat, {**copy.deepcopy(repeat), "repeat_index": 1}],
        "post_restore_proof": {"test_route_dispatcher_restored": True, "dispatcher_identity_matches_production": True, "c1_backend_spy_restored": True, "pinned_backend_spy_restored": True, "passed": True},
        "complete": True,
    }
    return main, runner_sha, analyzer_sha, shell_sha


def synthetic_allocation(
    directory: Path,
    allocation: str,
    job: str,
    runner_sha: str,
    analyzer_sha: str,
    shell_sha: str,
    a1_prerequisite: object,
) -> dict[str, object]:
    """Build signed-on-disk synthetic mains for the A1/A2 binding self-test."""

    mains: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    assessments: list[Mapping[str, object]] = []
    pids = (123 if allocation == "A1" else 223, 124 if allocation == "A1" else 224)
    for process, pid in enumerate(pids):
        main, _runner, _analyzer, _shell = synthetic_main()
        main = copy.deepcopy(main)
        main["allocation_id"] = allocation
        main["process_index"] = process
        main["pid"] = pid
        main["slurm_job_id"] = job
        content = obj(main["artifact_content_identity"], f"synthetic.{allocation}.content")
        content["allocation_id"] = allocation
        content["process_index"] = process
        for repeat in arr(main["public_benchmarks"], f"synthetic.{allocation}.repeats"):
            obj(repeat, f"synthetic.{allocation}.repeat")["process_index"] = process
        path = directory / f"{allocation}.main{process}.json"
        write(path, main)
        checked = validate_main(
            main,
            allocation,
            process,
            runner_sha,
            analyzer_sha,
            shell_sha,
            f"synthetic.{allocation}.main{process}",
            revalidate_files=False,
        )
        mains.append(main)
        artifacts.append(
            {
                "process_index": process,
                "path": str(path.resolve(strict=True)),
                "sha256": sha(path),
                "content_identity": checked["content_identity"],
            }
        )
        assessments.extend(checked["repeats"])  # type: ignore[arg-type]
    protocol = obj(mains[0]["identity"], f"synthetic.{allocation}.identity")["protocol"]
    runtime = obj(mains[0]["identity"], f"synthetic.{allocation}.runtime")["runtime"]
    eligible = True
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "fresh_b7_none_vshard2_p2_allocation_audit",
        "purpose": "fresh B7 none allocation audit; historic discovery excluded",
        "allocation_id": allocation,
        "source_identity": {
            "analyzer": {"path": "/synthetic/analyzer.py", "sha256": analyzer_sha},
            "expected_runner_sha256": runner_sha,
            "expected_protocol_shell_sha256": shell_sha,
        },
        "protocol_identity": protocol,
        "runtime_identity": {
            "device": obj(runtime, f"synthetic.{allocation}.runtime")["device"],
            "gpu_uuid": obj(runtime, f"synthetic.{allocation}.runtime")["gpu_uuid"],
            "pinned_reference_helper_load": obj(runtime, f"synthetic.{allocation}.runtime")["pinned_reference_helper_load"],
            "slurm_job_id": job,
            "pids": list(pids),
        },
        "main_artifacts": artifacts,
        "repeat_assessment": assessments,
        "a1_prerequisite": a1_prerequisite,
        "allocation_gate": {"eligible": eligible, "decision": expected_allocation_decision(allocation, eligible)},
        "complete": True,
    }


def self_test() -> None:
    require(close(percentile([float(index) for index in range(SAMPLES)], .50), 499.5), "percentile self-test")
    require(not strict_json_equal({"value": False}, {"value": 0}), "strict JSON equality accepted bool-for-int")
    require(not strict_json_equal({"value": 1}, {"value": 1.0}), "strict JSON equality accepted int-for-float")
    with tempfile.TemporaryDirectory(prefix="b7_none_single_read_") as temporary:
        path = Path(temporary) / "artifact.json"
        trusted, forged = b'{"marker":"trusted"}', b'{"marker":"forged"}'
        path.write_bytes(trusted)
        expected = hashlib.sha256(trusted).hexdigest()
        original_read_bytes = Path.read_bytes
        def replace_after_read(self: Path) -> bytes:
            payload = original_read_bytes(self)
            if self == path:
                path.write_bytes(forged)
            return payload
        try:
            Path.read_bytes = replace_after_read  # type: ignore[method-assign]
            record, actual = read(path, expected, "self-test.single_read")
            require(record["marker"] == "trusted" and actual == expected, "read parsed bytes different from its authenticated SHA payload")
        finally:
            Path.read_bytes = original_read_bytes  # type: ignore[method-assign]
    main, runner_sha, analyzer_sha, shell_sha = synthetic_main()
    validate_main(main, "A1", 0, runner_sha, analyzer_sha, shell_sha, "self-test.valid", revalidate_files=False)
    synthetic_identity = obj(main["identity"], "self-test.identity")
    synthetic_current_source = full_source_identity(
        {
            "source_identity": {
                "analyzer": {"path": "/synthetic/analyzer.py", "sha256": analyzer_sha},
                "expected_runner_sha256": runner_sha,
                "expected_protocol_shell_sha256": shell_sha,
            },
            "protocol_identity": synthetic_identity["protocol"],
            "runtime_identity": {"pinned_reference_helper_load": obj(synthetic_identity["runtime"], "self-test.runtime")["pinned_reference_helper_load"]},
        },
        "self-test.current_source",
    )
    require(
        strict_json_equal(
            obj(synthetic_current_source["pinned_reference_helper_load"], "self-test.current_source.helper"),
            obj(obj(synthetic_identity["runtime"], "self-test.runtime")["pinned_reference_helper_load"], "self-test.runtime.helper"),
        ),
        "allocation-mode full source omitted the helper no-build proof",
    )
    def forge_content_runtime_bool_as_int(value: Mapping[str, Any]) -> None:
        content = obj(value["artifact_content_identity"], "self-test.content")
        runtime = copy.deepcopy(obj(content["runtime_identity"], "self-test.content.runtime"))
        obj(runtime["device"], "self-test.content.runtime.device")["gate_pass"] = 1
        content["runtime_identity"] = runtime
    for name, mutation in (
        ("schema2_legacy", lambda value: value.__setitem__("schema_version", 2)),
        ("event_contract", lambda value: value["public_benchmarks"][0].__setitem__("event_contract", "forged")),
        ("c1_reason", lambda value: value["public_benchmarks"][0]["public_precheck"]["c1_test_route"]["decision"].__setitem__("reason", "forged")),
        ("non_none_final_state", lambda value: value["raw_abi_correctness"]["fp32_both"]["vshard2_vs_baseline"].pop("final_state_present")),
        ("patched_dirty_overlay", lambda value: value["identity"]["protocol"]["patched_dirty_overlay"]["git_status_porcelain_v1"].__setitem__("csrc/fwd.h", "M ")),
        ("patched_dirty_path", lambda value: value["identity"]["protocol"]["patched_dirty_overlay"]["files"]["csrc/fwd.h"].__setitem__("path", "/forged/fwd.h")),
        ("helper_protocol_path", lambda value: value["identity"]["protocol"]["pinned_reference_helper"].__setitem__("path", "/forged/sigmoid_ext.so")),
        ("helper_protocol_sha", lambda value: value["identity"]["protocol"]["pinned_reference_helper"].__setitem__("sha256", "0" * 64)),
        ("helper_runtime_path", lambda value: value["identity"]["runtime"]["pinned_reference_helper_load"].__setitem__("path", "/forged/sigmoid_ext.so")),
        ("helper_runtime_sha", lambda value: value["identity"]["runtime"]["pinned_reference_helper_load"].__setitem__("sha256", "0" * 64)),
        ("helper_runtime_no_build", lambda value: value["identity"]["runtime"]["pinned_reference_helper_load"].__setitem__("no_build", False)),
        ("helper_runtime_intercepts", lambda value: value["identity"]["runtime"]["pinned_reference_helper_load"].__setitem__("intercepted_names", ["sigmoid_ext", "sigmoid_ext"])),
        ("post_restore_proof", lambda value: value["post_restore_proof"].__setitem__("pinned_backend_spy_restored", False)),
        ("content_runtime_bool_as_int", forge_content_runtime_bool_as_int),
    ):
        forged = copy.deepcopy(main)
        mutation(forged)
        try:
            validate_main(forged, "A1", 0, runner_sha, analyzer_sha, shell_sha, f"self-test.{name}", revalidate_files=False)
        except AuditError:
            continue
        raise AuditError(f"forged {name} artifact was accepted")
    for forged_job in ("0", "01", "１"):
        try:
            positive_decimal_job(forged_job, "self-test current job")
        except AuditError:
            continue
        raise AuditError("noncanonical current A2 job was accepted")
    try:
        require_distinct_jobs("1", "1", "self-test same-job")
    except AuditError:
        pass
    else:
        raise AuditError("same A1/A2 Slurm job was accepted")
    try:
        require_distinct_jobs("1", "１", "self-test Unicode same-job")
    except AuditError:
        pass
    else:
        raise AuditError("Unicode alias of the same Slurm job was accepted")
    with tempfile.TemporaryDirectory(prefix="b7_none_a1_a2_") as temporary:
        directory = Path(temporary)
        alias = directory / "alias"
        alias.mkdir()
        a1 = synthetic_allocation(directory, "A1", "1", runner_sha, analyzer_sha, shell_sha, None)
        a1_path = directory / "A1.allocation_audit.json"
        write(a1_path, a1)
        a1_sha = sha(a1_path)
        validate_allocation(a1, "A1", runner_sha, analyzer_sha, shell_sha, "self-test.A1", revalidate_files=False)
        a1_binding = {
            "path": str(a1_path.resolve(strict=True)),
            "sha256": a1_sha,
            "slurm_job_id": "1",
            "full_source_identity": full_source_identity(a1, "self-test.A1"),
        }
        a2 = synthetic_allocation(directory, "A2", "2", runner_sha, analyzer_sha, shell_sha, a1_binding)
        validate_allocation(a2, "A2", runner_sha, analyzer_sha, shell_sha, "self-test.A2", revalidate_files=False)
        for name, mutation in (
            ("a1_binding_sha", lambda value: obj(value["a1_prerequisite"], "forged binding").__setitem__("sha256", "0" * 64)),
            ("a1_binding_same_job", lambda value: obj(value["a1_prerequisite"], "forged binding").__setitem__("slurm_job_id", "2")),
            ("a1_binding_noncanonical_path", lambda value: obj(value["a1_prerequisite"], "forged binding").__setitem__("path", str(alias / ".." / a1_path.name))),
            ("a2_current_source_mismatch", lambda value: obj(obj(value["source_identity"], "forged source")["analyzer"], "forged analyzer").__setitem__("path", "/synthetic/other_analyzer.py")),
            ("artifact_content_bool_as_int", lambda value: obj(obj(arr(value["main_artifacts"], "forged artifacts")[0], "forged artifact")["content_identity"], "forged content")["runtime_identity"].__setitem__("device", {"name": "NVIDIA B300", "capability": [10, 3], "multiprocessor_count": 148, "gate_pass": 1})),
            ("repeat_assessment_bool_as_int", lambda value: obj(arr(value["repeat_assessment"], "forged repeats")[0], "forged repeat").__setitem__("repeat_gate_pass", 1)),
        ):
            forged = copy.deepcopy(a2)
            mutation(forged)
            try:
                validate_allocation(forged, "A2", runner_sha, analyzer_sha, shell_sha, f"self-test.{name}", revalidate_files=False)
            except AuditError:
                continue
            raise AuditError(f"forged {name} A2 prerequisite was accepted")
    print("analyzer self-test PASS (single-read SHA/parse, strict JSON types, Unicode jobs, and forged A1/A2 bindings rejected)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--precondition-a1", action="store_true")
    parser.add_argument("--chain", action="store_true")
    parser.add_argument("--allocation", choices=("A1", "A2"))
    parser.add_argument("--main-json", nargs=2, type=Path)
    parser.add_argument("--expected-main-sha256", nargs=2)
    parser.add_argument("--expected-runner-sha256")
    parser.add_argument("--expected-analyzer-sha256")
    parser.add_argument("--expected-protocol-shell-sha256")
    parser.add_argument("--a1-audit", type=Path)
    parser.add_argument("--a2-audit", type=Path)
    parser.add_argument("--expected-a1-sha256")
    parser.add_argument("--expected-a2-sha256")
    parser.add_argument("--current-slurm-job-id")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.self_test:
        require(not args.precondition_a1 and not args.chain and args.allocation is None, "--self-test cannot combine with audit modes")
        self_test()
        return
    require(sum(bool(value) for value in (args.precondition_a1, args.chain, args.allocation is not None)) == 1, "select exactly one of --precondition-a1, --chain, or --allocation")
    require(args.expected_runner_sha256 is not None and args.expected_analyzer_sha256 is not None and args.expected_protocol_shell_sha256 is not None, "runner/analyzer/protocol-shell expected SHA arguments are required")
    sha_text(args.expected_runner_sha256, "runner SHA")
    sha_text(args.expected_analyzer_sha256, "analyzer SHA")
    sha_text(args.expected_protocol_shell_sha256, "protocol-shell SHA")
    if args.precondition_a1:
        require(args.a1_audit is not None and args.expected_a1_sha256 is not None and args.current_slurm_job_id is not None, "A1 precondition requires --a1-audit, --expected-a1-sha256, and --current-slurm-job-id")
        sha_text(args.expected_a1_sha256, "A1 audit SHA")
        precondition_a1_mode(args)
        return
    require(args.json is not None, "--json is required for allocation/chain modes")
    if args.chain:
        require(all(value is not None for value in (args.a1_audit, args.a2_audit, args.expected_a1_sha256, args.expected_a2_sha256)), "chain requires two audits and their SHAs")
        sha_text(args.expected_a1_sha256, "A1 audit SHA")
        sha_text(args.expected_a2_sha256, "A2 audit SHA")
        chain_mode(args)
        return
    require(args.main_json is not None and args.expected_main_sha256 is not None, "allocation mode requires two mains and their SHAs")
    for value in args.expected_main_sha256:
        sha_text(value, "main SHA")
    if args.allocation == "A1":
        require(
            args.a1_audit is None and args.expected_a1_sha256 is None and args.current_slurm_job_id is None,
            "A1 allocation rejects A1 prerequisite arguments",
        )
    else:
        require(
            args.a1_audit is not None and args.expected_a1_sha256 is not None and args.current_slurm_job_id is not None,
            "A2 allocation requires --a1-audit, --expected-a1-sha256, and --current-slurm-job-id",
        )
        sha_text(args.expected_a1_sha256, "A1 audit SHA")
        positive_decimal_job(args.current_slurm_job_id, "A2 current Slurm job")
    allocation_mode(args)


if __name__ == "__main__":
    try:
        main()
    except AuditError as exc:
        print(f"AUDIT_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
