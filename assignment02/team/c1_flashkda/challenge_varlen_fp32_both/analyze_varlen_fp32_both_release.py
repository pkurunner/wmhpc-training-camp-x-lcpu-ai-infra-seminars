#!/usr/bin/env python3
"""Stdlib-only fail-closed analyzer for the relative-only A1/A2 protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import subprocess
from typing import Any, Mapping


SCHEMA, TARGET, OFFSETS = 3, "skew_n6_h12_t12288/fp32_both", [0, 1, 2, 3, 4, 5, 12288]
PATHS, REPEATS, SAMPLES, WARMUP, MIN_SPEEDUP = ("public_registry_c1", "public_registry_pinned"), 2, 1000, 100, 1.02
ANALYZER_SHA_ENV = "C1_VARLEN_FP32_BOTH_ANALYZER_SHA256"
RUNNER_SHA_ENV = "C1_VARLEN_FP32_BOTH_RUNNER_SHA256"
PROTOCOL_SHELL_SHA_ENV = "EXPECTED_PROTOCOL_SHELL_SHA256"
PROTOCOL_SHELL_PATH_ENV = "C1_VARLEN_FP32_BOTH_PROTOCOL_SHELL_PATH"
TELEMETRY_FIELDS = ("timestamp_ns", "index", "uuid", "pstate", "clocks.current.sm", "clocks.current.memory", "power.draw", "temperature.gpu", "power.limit")
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
POSITIVE_DECIMAL = re.compile(r"[1-9][0-9]*\Z")
CANDIDATE_HELPER_SHA256 = "e07481e7e66551f1f9eaafa7f1fb9db9ef193f1694ff831ee0a07802c8ffbe14"
PRODUCTION_WRAPPER_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
AUDITED_EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
PATCHED_EXACT_DIRTY_FILES = {
    "csrc/flash_kda.cpp": "38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4",
    "csrc/fwd.h": "613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083",
    "csrc/smxx/fwd_launch.cu": "a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928",
}
RUNTIME_IMPORT_IDENTITY_KEYS = frozenset((
    "auto_dispatch", "fla_backend", "varlen_metadata", "confirmation_runner",
    "shared_seqcount_runner", "prefetch2", "vshard4_prefetch2", "harness",
    "pinned_torch_ref", "pinned_reference_helper",
))
RUNTIME_LEDGER_AUTHORITY = "owned_runner_current_runtime_import_ledger"
RUNTIME_IMPORT_SHA256 = {
    "auto_dispatch": "9cdd460058254016af58723875bdf99ebe74f8e016a4c6027eb7fb38c8e9a88c",
    "fla_backend": "206e448abcd3d64826f87a20e7d57c790fef6adacd91e26edcb10a3711b9b656",
    "varlen_metadata": "f89a97ba284c7a24a3df54efca7bb60eb70f9cbda659200bd6cb3e8dfdaf4ccd",
    "confirmation_runner": "9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b",
    "shared_seqcount_runner": "4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f",
    "prefetch2": "752126488487ac317a7ee167b660b0895562e4877aedbc4ce2a599c1f59a10d0",
    "vshard4_prefetch2": "445c90815919fb8c1db3bc79289d10029ea77b1d9accb393c326c120ba1f8385",
    "harness": "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52",
    "pinned_torch_ref": "bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5",
    "pinned_reference_helper": "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f",
}
FLA_FILE_SHA256 = {
    "fla/__init__.py": "b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d",
    "fla/ops/backends/__init__.py": "a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635",
    "fla/ops/kda/__init__.py": "24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb",
    "fla/ops/kda/backends/__init__.py": "86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797",
    "fla/ops/kda/backends/flash_kda.py": "0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2",
    "fla/ops/kda/chunk.py": "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8",
}
TELEMETRY_COVERAGE_TOLERANCE_NS = 1_000_000_000
TELEMETRY_MAX_ADJACENT_GAP_NS = 1_000_000_000
TELEMETRY_SM_CLOCK_POLICY = {
    "all_positive": True,
    "minimum_median_mhz": 1000.0,
    "near_median_ratio": 0.95,
    "minimum_near_median_fraction": 0.80,
}


class AuditError(AssertionError):
    pass


def require(value: bool, message: str) -> None:
    if not value: raise AuditError(message)


def mapping(value: object, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), label + " must be an object"); return value  # type: ignore[return-value]


def integer(value: object, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), label + " must be integer"); return int(value)


def boolean(value: object, label: str) -> bool:
    require(isinstance(value, bool), label + " must be bool"); return value


def numeric(value: object, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), label + " must be number")
    parsed = float(value); require(math.isfinite(parsed), label + " must be finite"); return parsed


def exact_int_mapping(value: object, expected: Mapping[str, int], label: str) -> dict[str, int]:
    actual = mapping(value, label)
    require(set(actual) == set(expected), label + ": key set drift")
    for key, target in expected.items():
        require(type(actual.get(key)) is int and actual.get(key) == target, label + "." + key + ": exact integer value/type drift")
    return {key: int(actual[key]) for key in expected}


def exact_offsets(value: object, label: str) -> list[int]:
    """Require the JSON representation of the one certified skew layout."""

    require(type(value) is list and len(value) == len(OFFSETS) and all(type(item) is int for item in value) and value == OFFSETS, label + ": exact offset list/type drift")
    return list(OFFSETS)


def strict_json_equal(actual: object, expected: object, label: str) -> None:
    """Compare persisted JSON evidence without Python's bool/int coercion."""

    if type(actual) is dict or type(expected) is dict:
        require(type(actual) is dict and type(expected) is dict, label + ": object type drift")
        actual_dict, expected_dict = actual, expected  # type: ignore[assignment]
        require(all(type(key) is str for key in actual_dict) and all(type(key) is str for key in expected_dict) and set(actual_dict) == set(expected_dict), label + ": object key-set/type drift")
        for key in actual_dict:
            strict_json_equal(actual_dict[key], expected_dict[key], label + "." + key)
        return
    if type(actual) is list or type(expected) is list:
        require(type(actual) is list and type(expected) is list and len(actual) == len(expected), label + ": array type/length drift")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            strict_json_equal(actual_item, expected_item, label + f"[{index}]")
        return
    require(type(actual) is type(expected) and actual == expected, label + ": scalar type/value drift")


def required_runner_identity_schema(value: object, label: str) -> Mapping[str, Any]:
    identity = mapping(value, label)
    for name in ("runner", "protocol_shell", "candidate_helper", "production_wrapper", "patched_tracked_identity"):
        mapping(identity.get(name), label + "." + name)
    return identity


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_sha(value: str, label: str) -> str:
    require(len(value) == 64 and all(char in "0123456789abcdef" for char in value), label + " must be SHA256"); return value


def read_hashed_bytes(path: Path, expected: str, label: str) -> tuple[bytes, str]:
    """Read once, then hash and parse the exact same immutable payload."""

    expected = valid_sha(expected, label + ".sha")
    require(path.is_file(), label + " missing")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AuditError(label + ": read failed") from exc
    require(hashlib.sha256(payload).hexdigest() == expected, label + " SHA mismatch")
    return payload, expected


def positive_job_id(value: object, label: str) -> str:
    require(isinstance(value, str) and POSITIVE_DECIMAL.fullmatch(value) is not None, label + " must be a strictly positive decimal Slurm ID")
    return value


def target_identity(value: object, label: str) -> dict[str, object]:
    target = mapping(value, label)
    exact_offsets(target.get("offsets"), label + ".offsets")
    require(target.get("cell") == TARGET and target.get("variant") == "vshard4_p2", label + ": target drift")
    return {"cell": TARGET, "offsets": list(OFFSETS), "variant": "vshard4_p2"}


def env_path(name: str) -> Path:
    value = os.environ.get(name, "")
    require(bool(value), name + " is required for current identity revalidation")
    try:
        return Path(value).resolve(strict=True)
    except OSError as exc:
        raise AuditError(name + " does not resolve") from exc


def current_ledger_specs() -> dict[str, dict[str, object]]:
    """The analyzer is authoritative: exact key, root, path, and SHA values."""

    a02 = env_path("A02_ROOT")
    reference = env_path("REFERENCE_ROOT")
    helper = env_path("C1_PINNED_REFERENCE_HELPER_PATH")
    owned = a02 / "team/c1_flashkda"
    specs = {
        "auto_dispatch": (owned, owned / "challenge_tp8_dispatch/auto_dispatch.py"),
        "fla_backend": (owned, owned / "challenge_tp8_dispatch/fla_backend.py"),
        "varlen_metadata": (owned, owned / "challenge_tp8_dispatch/varlen_metadata.py"),
        "confirmation_runner": (owned, owned / "challenge_varlen_dispatch/run_varlen_dispatch_confirmation.py"),
        "shared_seqcount_runner": (owned, owned / "challenge_seqcount_dispatch/run_seqcount_dispatch.py"),
        "prefetch2": (owned, owned / "challenge_prefetch2/prefetch2.py"),
        "vshard4_prefetch2": (owned, owned / "challenge_vshard4_prefetch2/vshard4_prefetch2.py"),
        "harness": (owned, owned / "harness/validate_and_bench.py"),
        "pinned_torch_ref": (reference, reference / "tests/torch_ref.py"),
        "pinned_reference_helper": (helper.parent, helper),
    }
    require(set(specs) == RUNTIME_IMPORT_IDENTITY_KEYS == set(RUNTIME_IMPORT_SHA256), "authoritative runtime ledger key drift")
    output: dict[str, dict[str, object]] = {}
    for name, (root, path) in specs.items():
        try:
            resolved_root, resolved_path = root.resolve(strict=True), path.resolve(strict=True)
        except OSError as exc:
            raise AuditError(name + ": authoritative ledger path missing") from exc
        require(sha(resolved_path) == RUNTIME_IMPORT_SHA256[name], name + ": current authoritative ledger SHA drift")
        output[name] = {"expected_root": str(resolved_root), "expected_path": str(resolved_path), "sha256": RUNTIME_IMPORT_SHA256[name]}
    return output


def current_protocol_shell_identity() -> dict[str, object]:
    expected = valid_sha(os.environ.get(PROTOCOL_SHELL_SHA_ENV, ""), PROTOCOL_SHELL_SHA_ENV)
    shell = env_path(PROTOCOL_SHELL_PATH_ENV)
    a02 = env_path("A02_ROOT")
    expected_path = (a02 / "team/c1_flashkda/challenge_varlen_fp32_both/run_clean_varlen_fp32_both_release.sh").resolve(strict=True)
    require(shell == expected_path and sha(shell) == expected, "current protocol shell identity drift")
    return {"path": str(shell), "expected_path": str(expected_path), "expected_root": str(a02), "sha256": expected, "sha256_gate_pass": True}


def current_wrapper_identity() -> dict[str, object]:
    patched = env_path("PATCHED_ROOT")
    wrapper = (patched / "flash_kda/__init__.py").resolve(strict=True)
    require(sha(wrapper) == PRODUCTION_WRAPPER_SHA256, "current patched production wrapper SHA drift")
    return {"path": str(wrapper), "expected_path": str(wrapper), "expected_root": str(patched), "sha256": PRODUCTION_WRAPPER_SHA256, "sha256_gate_pass": True}


def current_runner_identity() -> dict[str, object]:
    expected = valid_sha(os.environ.get(RUNNER_SHA_ENV, ""), RUNNER_SHA_ENV)
    a02 = env_path("A02_ROOT")
    path = (a02 / "team/c1_flashkda/challenge_varlen_fp32_both/run_varlen_fp32_both_release.py").resolve(strict=True)
    require(sha(path) == expected, "current runner SHA drift")
    return {"path": str(path), "expected_path": str(path), "expected_root": str(a02), "sha256": expected, "sha256_gate_pass": True}


def git_head_and_clean(root: Path, label: str, expected_commit: str) -> None:
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuditError(label + ": cannot resolve git source identity") from exc
    require(head == expected_commit and not status, label + ": current commit/tracked-tree drift")


def current_patched_tracked_identity() -> dict[str, object]:
    patched = env_path("PATCHED_ROOT")
    try:
        head = subprocess.run(["git", "-C", str(patched), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        lines = subprocess.run(["git", "-C", str(patched), "status", "--porcelain=v1", "--untracked-files=no"], check=True, capture_output=True, text=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuditError("patched: cannot resolve exact dirty-tree identity") from exc
    records: dict[str, dict[str, object]] = {}
    for line in lines:
        require(len(line) >= 4 and line[:2] == " M" and line[2] == " ", "patched: status is not an exact permitted unstaged modification")
        relative = line[3:]
        require(relative in PATCHED_EXACT_DIRTY_FILES and relative not in records, "patched: dirty path outside exact permitted set")
        path = (patched / relative).resolve(strict=True)
        records[relative] = {"status": " M", "path": str(path), "sha256": sha(path)}
    require(head == PATCHED_COMMIT and set(records) == set(PATCHED_EXACT_DIRTY_FILES), "patched: commit/dirty-set drift")
    for relative, digest in PATCHED_EXACT_DIRTY_FILES.items():
        require(records[relative]["sha256"] == digest, "patched: exact dirty-file SHA drift: " + relative)
    return {"root": str(patched), "head": PATCHED_COMMIT, "dirty_files": records, "gate_pass": True}


def current_source_identity() -> dict[str, object]:
    """Re-read source and SO bytes; raw runner claims alone never authorize freeze."""

    patched, reference, fla = env_path("PATCHED_ROOT"), env_path("REFERENCE_ROOT"), env_path("FLA_ROOT")
    current_patched_tracked_identity()
    git_head_and_clean(reference, "reference", PATCHED_COMMIT)
    git_head_and_clean(fla, "FLA", FLA_COMMIT)
    extensions = sorted(patched.glob("flash_kda_C*.so"))
    require(len(extensions) == 1, "current extension path ambiguity")
    extension = extensions[0].resolve(strict=True)
    require(sha(extension) == AUDITED_EXTENSION_SHA256, "current extension SO SHA drift")
    for relative, digest in FLA_FILE_SHA256.items():
        path = (fla / relative).resolve(strict=True)
        require(sha(path) == digest, "current FLA source SHA drift: " + relative)
    return {"patched_root": str(patched), "reference_root": str(reference), "fla_root": str(fla), "extension_path": str(extension), "extension_sha256": AUDITED_EXTENSION_SHA256}


def runtime_import_identities(value: object, label: str) -> dict[str, dict[str, object]]:
    identities = mapping(value, label)
    require(set(identities) == RUNTIME_IMPORT_IDENTITY_KEYS, label + ": runtime import identity key set drift")
    expected = current_ledger_specs()
    verified: dict[str, dict[str, object]] = {}
    for name in sorted(RUNTIME_IMPORT_IDENTITY_KEYS):
        item = mapping(identities.get(name), label + "." + name)
        path, root, expected_path, digest = item.get("path"), item.get("expected_root"), item.get("expected_path"), item.get("sha256")
        require(path == expected[name]["expected_path"] and expected_path == expected[name]["expected_path"] and root == expected[name]["expected_root"], label + "." + name + ": resolved path/root drift")
        require(digest == expected[name]["sha256"], label + "." + name + ": authoritative SHA drift")
        verified[name] = {
            "path": path,
            "expected_path": expected_path,
            "expected_root": root,
            "sha256": valid_sha(str(digest), label + "." + name + ".sha256"),
            "sha256_gate_pass": boolean(item.get("sha256_gate_pass"), label + "." + name + ".sha256_gate_pass"),
        }
        require(verified[name]["sha256_gate_pass"] is True, label + "." + name + ": SHA gate failed")
    return verified


def read_json(path: Path, expected: str, label: str) -> tuple[Mapping[str, Any], str]:
    payload, expected = read_hashed_bytes(path, expected, label)
    try: data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise AuditError(label + ": invalid JSON") from exc
    return mapping(data, label), expected


def analyzer_identity() -> dict[str, object]:
    path = Path(__file__).resolve(strict=True)
    expected = valid_sha(os.environ.get(ANALYZER_SHA_ENV, ""), ANALYZER_SHA_ENV)
    actual = sha(path)
    require(actual == expected, "analyzer source SHA failed")
    return {"path": str(path), "sha256": actual, "sha256_gate_pass": True}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values); position = (len(ordered) - 1) * q; low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] * (1 - position + low) + ordered[high] * (position - low)


def recompute(values: list[float]) -> dict[str, float | int]:
    require(len(values) == SAMPLES and all(math.isfinite(x) and x > 0 for x in values), "invalid raw sample vector")
    return {"samples": len(values), "mean_ms": statistics.fmean(values), "p50_ms": percentile(values, .5), "p95_ms": percentile(values, .95), "p99_ms": percentile(values, .99), "min_ms": min(values), "max_ms": max(values)}


def close(actual: object, expected: float, label: str) -> None:
    require(math.isclose(numeric(actual, label), expected, rel_tol=1e-12, abs_tol=1e-12), label + " recomputation drift")


def parse_number(value: str, label: str) -> float:
    match = NUMBER.search(value.replace(",", "")); require(match is not None, label + " lacks number")
    parsed = float(match.group()); require(math.isfinite(parsed), label + " nonfinite"); return parsed


def telemetry_window(start: object, end: object, label: str) -> dict[str, int]:
    begin, finish = integer(start, label + ".main0_start_ns"), integer(end, label + ".main1_end_ns")
    require(begin > 0 and finish > begin, label + ": invalid main timing window")
    return {"main0_start_ns": begin, "main1_end_ns": finish}


def telemetry(path: Path, expected_sha: str, expected_uuid: str, window: Mapping[str, Any]) -> dict[str, object]:
    payload, expected_sha = read_hashed_bytes(path, expected_sha, "telemetry")
    try:
        return parse_telemetry(path, payload, expected_sha, expected_uuid, window)
    except (AuditError, UnicodeDecodeError, ValueError) as exc:
        # NVML trouble invalidates this allocation but must not be represented
        # as a relative performance regression.
        return {"path": str(path.resolve()), "sha256": expected_sha, "uuid": expected_uuid, "window": dict(window), "checks": {}, "allocation_telemetry_valid": False, "invalid_reason": str(exc)}


def validate_telemetry_stamps(stamps: list[int], window: Mapping[str, Any], label: str) -> int:
    required_window = telemetry_window(window.get("main0_start_ns"), window.get("main1_end_ns"), label + ".window")
    require(len(stamps) >= 2 and all(left < right for left, right in zip(stamps, stamps[1:])), label + ": timestamps are not fresh/strictly increasing")
    require(stamps[0] <= required_window["main0_start_ns"] + TELEMETRY_COVERAGE_TOLERANCE_NS and stamps[-1] >= required_window["main1_end_ns"] - TELEMETRY_COVERAGE_TOLERANCE_NS, label + ": timestamps do not cover main0-to-main1 window")
    max_gap = max(right - left for left, right in zip(stamps, stamps[1:]))
    require(max_gap <= TELEMETRY_MAX_ADJACENT_GAP_NS, label + ": adjacent collector timestamp gap exceeds preregistered tolerance")
    return max_gap


def validate_telemetry_window_sidecar(value: object, window: Mapping[str, Any], label: str) -> dict[str, object]:
    sidecar = mapping(value, label)
    exact = telemetry_window(window.get("main0_start_ns"), window.get("main1_end_ns"), label + ".window")
    require(set(sidecar) == {"main0_start_ns", "main1_end_ns", "telemetry_pid_was_alive"}, label + ": sidecar key set drift")
    require(integer(sidecar.get("main0_start_ns"), label + ".main0_start_ns") == exact["main0_start_ns"] and integer(sidecar.get("main1_end_ns"), label + ".main1_end_ns") == exact["main1_end_ns"] and boolean(sidecar.get("telemetry_pid_was_alive"), label + ".telemetry_pid_was_alive"), label + ": sidecar value/type drift")
    return {**exact, "telemetry_pid_was_alive": True}


def sm_clock_quality(sm: list[float]) -> tuple[dict[str, bool], float]:
    """Validate active-clock quality without rejecting bounded startup idle samples."""

    require(sm and all(math.isfinite(clock) for clock in sm), "telemetry SM clocks missing/non-finite")
    median = statistics.median(sm)
    near_median = sum(clock >= TELEMETRY_SM_CLOCK_POLICY["near_median_ratio"] * median for clock in sm) / len(sm)
    checks = {
        "sm_clock_all_positive": min(sm) > 0,
        "sm_clock_median_at_least_1000mhz": median >= TELEMETRY_SM_CLOCK_POLICY["minimum_median_mhz"],
        "sm_clock_at_least_80pct_within_95pct_median": near_median >= TELEMETRY_SM_CLOCK_POLICY["minimum_near_median_fraction"],
    }
    return checks, near_median


def parse_telemetry(path: Path, payload: bytes, expected_sha: str, expected_uuid: str, window: Mapping[str, Any]) -> dict[str, object]:
    rows = [row for row in csv.reader(payload.decode("utf-8").splitlines()) if row]
    required_window = telemetry_window(window.get("main0_start_ns"), window.get("main1_end_ns"), "telemetry.window")
    duration_ns = required_window["main1_end_ns"] - required_window["main0_start_ns"]
    # The collection loop is preregistered at 200ms.  A one-second startup / stop
    # tolerance permits scheduling jitter, but a five-row sidecar can never cover
    # this protocol regardless of how short an apparent main call claims to be.
    min_samples = max(12, duration_ns // 1_000_000_000 + 1)
    require(len(rows) >= min_samples and all(len(row) == len(TELEMETRY_FIELDS) for row in rows), "telemetry row/schema/coverage sample count insufficient")
    items = [{field: row[index].strip() for index, field in enumerate(TELEMETRY_FIELDS)} for row in rows]
    require(all(all(item[field] for field in TELEMETRY_FIELDS) for item in items), "telemetry blank field")
    require({item["index"] for item in items} == {"0"} and {item["uuid"] for item in items} == {expected_uuid}, "telemetry GPU identity drift")
    stamps = [integer(int(item["timestamp_ns"]), "telemetry.timestamp") for item in items]
    max_gap_ns = validate_telemetry_stamps(stamps, required_window, "telemetry")
    sm = [parse_number(item["clocks.current.sm"], "sm") for item in items]; mem = [parse_number(item["clocks.current.memory"], "mem") for item in items]
    power = [parse_number(item["power.draw"], "power") for item in items]; limit = [parse_number(item["power.limit"], "limit") for item in items]; temp = [parse_number(item["temperature.gpu"], "temp") for item in items]
    p0 = all(item["pstate"].upper() == "P0" for item in items)
    sm_checks, sm_near_median_fraction = sm_clock_quality(sm)
    checks = {"all_pstate_p0": p0, **sm_checks, "memory_clock_positive_and_within_95pct_median": min(mem) > 0 and min(mem) >= .95 * statistics.median(mem), "temperature_below_85c": 0 < min(temp) and max(temp) < 85, "power_positive_and_not_over_limit": min(power) > 0 and all(draw <= cap * 1.01 for draw, cap in zip(power, limit, strict=True)), "timestamp_coverage_and_max_gap": True}
    return {"path": str(path.resolve()), "sha256": expected_sha, "samples": len(items), "uuid": expected_uuid, "window": required_window, "checks": checks, "allocation_telemetry_valid": all(checks.values()), "summary": {"max_adjacent_gap_ns": max_gap_ns, "sm_clock_mhz_min_median_max": [min(sm), statistics.median(sm), max(sm)], "sm_clock_fraction_within_95pct_median": sm_near_median_fraction, "sm_clock_minimum_fraction_required": TELEMETRY_SM_CLOCK_POLICY["minimum_near_median_fraction"], "memory_clock_mhz_min_median_max": [min(mem), statistics.median(mem), max(mem)], "temperature_c_min_max": [min(temp), max(temp)], "power_w_min_max": [min(power), max(power)], "power_limit_w_min_max": [min(limit), max(limit)]}}


def validate_runner(data: Mapping[str, Any], label: str, expected_alloc: str, expected_index: int, expected_runner_sha: str) -> dict[str, object]:
    require(integer(data.get("schema_version"), label + ".schema") == SCHEMA and data.get("allocation_id") == expected_alloc and integer(data.get("process_index"), label + ".process_index") == expected_index, label + ": allocation/schema/process-index drift")
    require(boolean(data.get("complete"), label + ".complete") and "failure" not in data and "map_restoration_failure" not in data, label + ": incomplete/failure")
    process, allocation, target, prereg = mapping(data.get("process"), label + ".process"), mapping(data.get("allocation"), label + ".allocation"), mapping(data.get("target"), label + ".target"), mapping(data.get("pre_registered"), label + ".prereg")
    pid, job = integer(process.get("pid"), label + ".pid"), allocation.get("slurm_job_id")
    require(pid > 1 and boolean(process.get("fresh_python_process_required"), label + ".fresh"), label + ": fresh PID failed")
    job = positive_job_id(job, label + ".slurm_job_id")
    validated_target = target_identity(target, label + ".target")
    require(integer(prereg.get("repeats"), label + ".repeats") == REPEATS and integer(prereg.get("samples_per_path_per_repeat"), label + ".samples") == SAMPLES and integer(prereg.get("warmup_per_path_per_repeat"), label + ".warmup") == WARMUP and numeric(prereg.get("minimum_relative_speedup_x"), label + ".margin") == MIN_SPEEDUP, label + ": prereg drift")
    strict_json_equal(prereg.get("telemetry_sm_clock_policy"), TELEMETRY_SM_CLOCK_POLICY, label + ".telemetry_sm_clock_policy")
    require("relative-only" in str(prereg.get("policy")), label + ": relative-only policy missing")
    identity = required_runner_identity_schema(data.get("identity"), label + ".identity"); runner = mapping(identity.get("runner"), label + ".runner")
    current_runner = current_runner_identity()
    strict_json_equal(runner, current_runner, label + ".runner")
    require(runner.get("sha256") == expected_runner_sha, label + ": runner SHA")
    protocol_shell = mapping(identity.get("protocol_shell"), label + ".protocol_shell")
    strict_json_equal(protocol_shell, current_protocol_shell_identity(), label + ".protocol_shell")
    helper = mapping(identity.get("candidate_helper"), label + ".candidate_helper")
    expected_helper = {
        "path": str((env_path("A02_ROOT") / "team/c1_flashkda/challenge_varlen_dispatch/run_varlen_fla_handoff_candidate.py").resolve(strict=True)),
        "expected_path": str((env_path("A02_ROOT") / "team/c1_flashkda/challenge_varlen_dispatch/run_varlen_fla_handoff_candidate.py").resolve(strict=True)),
        "expected_root": str(env_path("A02_ROOT")),
        "sha256": CANDIDATE_HELPER_SHA256,
        "sha256_gate_pass": True,
    }
    strict_json_equal(helper, expected_helper, label + ".candidate_helper")
    require(sha(Path(str(expected_helper["path"]))) == CANDIDATE_HELPER_SHA256, label + ": candidate helper SHA")
    wrapper = mapping(identity.get("production_wrapper"), label + ".production_wrapper")
    strict_json_equal(wrapper, current_wrapper_identity(), label + ".production_wrapper")
    patched_tracked = mapping(identity.get("patched_tracked_identity"), label + ".patched_tracked_identity")
    strict_json_equal(patched_tracked, current_patched_tracked_identity(), label + ".patched_tracked_identity")
    runtime_identities = runtime_import_identities(identity.get("runtime_import_identities"), label + ".runtime_import_identities")
    ledger_authority = mapping(identity.get("runtime_import_ledger_authority"), label + ".runtime_import_ledger_authority")
    require(ledger_authority.get("provider") == RUNTIME_LEDGER_AUTHORITY and ledger_authority.get("candidate_helper_embedded_ledger_trusted") is False and ledger_authority.get("expected_keys") == sorted(RUNTIME_IMPORT_IDENTITY_KEYS) and isinstance(ledger_authority.get("reason"), str) and ledger_authority.get("reason"), label + ": runtime ledger authority drift")
    device, extension, fla = mapping(identity.get("device"), label + ".device"), mapping(identity.get("extension"), label + ".extension"), mapping(identity.get("fla"), label + ".fla")
    require(boolean(device.get("passed"), label + ".device.pass") and device.get("capability") == [10, 3] and integer(device.get("multiprocessor_count"), label + ".sms") == 148 and boolean(extension.get("passed"), label + ".so.pass") and isinstance(extension.get("sha256"), str) and valid_sha(str(extension.get("sha256")), label + ".so.sha256") == extension.get("sha256") and boolean(fla.get("passed"), label + ".fla.pass"), label + ": source/SO/GPU identity failed")
    current_source = current_source_identity()
    require(extension.get("path") == current_source["extension_path"] and extension.get("sha256") == current_source["extension_sha256"], label + ": current extension path/SHA drift")
    flash_python = mapping(identity.get("flash_kda_python"), label + ".flash_kda_python")
    current_wrapper = current_wrapper_identity()
    require(flash_python.get("path") == current_wrapper["path"] and flash_python.get("sha256") == current_wrapper["sha256"], label + ": current production wrapper source drift")
    source_trees = mapping(identity.get("source_trees"), label + ".source_trees")
    patched_tree, reference_tree = mapping(source_trees.get("patched"), label + ".source_trees.patched"), mapping(source_trees.get("reference"), label + ".source_trees.reference")
    require(patched_tree.get("root") == current_source["patched_root"] and patched_tree.get("commit") == PATCHED_COMMIT and boolean(patched_tree.get("passed"), label + ".patched.pass") and reference_tree.get("root") == current_source["reference_root"] and reference_tree.get("commit") == PATCHED_COMMIT and boolean(reference_tree.get("tracked_status_clean"), label + ".reference.clean") and boolean(reference_tree.get("passed"), label + ".reference.pass"), label + ": current source-tree identity drift")
    strict_json_equal(mapping(fla.get("files"), label + ".fla.files"), FLA_FILE_SHA256, label + ".fla.files")
    require(fla.get("root") == current_source["fla_root"] and fla.get("commit") == FLA_COMMIT and boolean(fla.get("tracked_status_clean"), label + ".fla.clean"), label + ": current FLA source identity drift")
    public = mapping(mapping(fla.get("public_callables"), label + ".public").get("fla.ops.kda.chunk_kda"), label + ".chunk")
    require(boolean(public.get("passed"), label + ".public.chunk"), label + ": public chunk identity failed")
    gates = mapping(data.get("gates"), label + ".gates")
    for name in ("target_exact_public_route", "fallback", "descriptor_freshness", "route", "identity", "prepare_spy_restored", "backend_spies_restored", "prepare_spy_restored_after"):
        require(boolean(mapping(gates.get(name), label + ".gate." + name).get("passed"), label + ".gate." + name), label + ": gate failed: " + name)
    correctness, fallback = mapping(data.get("correctness"), label + ".exact"), mapping(data.get("fallback"), label + ".fallback")
    require(boolean(correctness.get("passed"), label + ".exact.pass") and boolean(fallback.get("passed"), label + ".fallback.pass"), label + ": exact/fallback failed")
    descriptors = data.get("descriptor_freshness")
    require(isinstance(descriptors, list) and len(descriptors) == REPEATS, label + ": descriptor repeat count")
    descriptor_ids: set[int] = set()
    expected_empty_cache = {"entries": 0, "hits": 0, "misses": 0, "capture_miss_rejections": 0, "capture_hit_rejections": 0}
    for index, raw_descriptor in enumerate(descriptors):
        descriptor = mapping(raw_descriptor, label + f".descriptor{index}")
        descriptor_id = integer(descriptor.get("descriptor_object_id"), label + ".descriptor.id")
        descriptor_ids.add(descriptor_id)
        require(integer(descriptor.get("repeat_index"), label + ".descriptor.repeat_index") == index, label + ": descriptor repeat index drift")
        require(descriptor.get("probe") == "one timing-external public fla.ops.kda.chunk_kda C1 call" and integer(descriptor.get("public_chunk_kda_call_count"), label + ".descriptor.public_calls") == 1 and integer(descriptor.get("issue_descriptor_call_count"), label + ".descriptor.issue_calls") == 1, label + ": descriptor probe did not use exactly one real public call/issue")
        require(exact_int_mapping(descriptor.get("route_spy_delta"), {"c1": 1, "pinned": 0}, label + ".descriptor.route") == {"c1": 1, "pinned": 0} and descriptor.get("chosen_variant") == "vshard4_p2", label + ": descriptor probe C1 route drift")
        decision = mapping(descriptor.get("decision"), label + ".descriptor.decision")
        require(decision.get("chosen_variant") == "vshard4_p2" and exact_offsets(decision.get("certified_varlen_offsets"), label + ".descriptor.decision.certified_varlen_offsets") == OFFSETS and decision.get("canonical_cache_hit") is False, label + ": descriptor public decision/canonical offsets drift")
        require(integer(descriptor.get("cpu_offsets_object_id"), label + ".descriptor.cpu_id") > 0 and boolean(descriptor.get("descriptor_cpu_tensor_identity"), label + ".descriptor.cpu_identity") and exact_offsets(descriptor.get("offsets"), label + ".descriptor.offsets") == OFFSETS and boolean(descriptor.get("fresh_against_prior"), label + ".descriptor.fresh") and boolean(descriptor.get("issue_descriptor_spy_restored"), label + ".descriptor.spy_restored") and boolean(descriptor.get("cache_cleared_before_probe"), label + ".descriptor.cache_before") and boolean(descriptor.get("cache_cleared_after_probe_before_timing"), label + ".descriptor.cache_after"), label + ": descriptor authority/freshness/restoration failed")
        require(exact_int_mapping(descriptor.get("cache_after_probe_clear"), expected_empty_cache, label + ".descriptor.cache_state") == expected_empty_cache and boolean(descriptor.get("input_immutability_exact"), label + ".descriptor.immutability") and boolean(descriptor.get("passed"), label + ".descriptor.pass"), label + ": descriptor cache/immutability/pass failed")
        final = mapping(descriptor.get("final_state_contract"), label + ".descriptor.final")
        output = mapping(descriptor.get("output_contract"), label + ".descriptor.output")
        require(final.get("present") is True and final.get("dtype") == "torch.float32" and final.get("shape") == [6, 12, 128, 128] and final.get("contiguous") is True and output.get("dtype") == "torch.bfloat16" and output.get("contiguous") is True, label + ": descriptor probe output/final ABI drift")
    require(len(descriptor_ids) == REPEATS, label + ": descriptor objects were reused across repeats")
    performance = mapping(data.get("performance"), label + ".performance"); pre, post = mapping(performance.get("gpu_state_before_timing"), label + ".gpu.pre"), mapping(performance.get("gpu_state_after_timing"), label + ".gpu.post")
    pre_values, post_values = mapping(pre.get("values"), label + ".gpu.pre.values"), mapping(post.get("values"), label + ".gpu.post.values")
    require(pre_values.get("uuid") == post_values.get("uuid") and pre_values.get("pstate", "").upper() == "P0" and post_values.get("pstate", "").upper() == "P0", label + ": per-PID P0/GPU state invalid")
    repeats = performance.get("repeats"); require(isinstance(repeats, list) and len(repeats) == REPEATS, label + ": repeat count")
    minimums: dict[str, float] = {q: float("inf") for q in ("p50_ms", "p95_ms", "p99_ms")}
    for index, raw in enumerate(repeats):
        item = mapping(raw, f"{label}.repeat{index}"); require(integer(item.get("repeat_index"), label + ".repeat.index") == index and item.get("timing_contract") is not None and boolean(item.get("passed"), label + ".repeat.pass") and boolean(item.get("input_immutability_exact"), label + ".immutability"), label + ": repeat evidence")
        require(exact_int_mapping(item.get("warmup_route_spy_delta"), {"c1": WARMUP, "pinned": WARMUP}, label + ".warm") == {"c1": WARMUP, "pinned": WARMUP} and exact_int_mapping(item.get("timed_route_spy_delta"), {"c1": SAMPLES, "pinned": SAMPLES}, label + ".timed") == {"c1": SAMPLES, "pinned": SAMPLES}, label + ": route counts")
        raw_samples, paths = mapping(item.get("raw_samples_ms"), label + ".raw"), mapping(item.get("paths"), label + ".paths")
        computed: dict[str, Mapping[str, float | int]] = {}
        for path in PATHS:
            values = raw_samples.get(path); require(isinstance(values, list), label + ": raw path missing")
            computed[path] = recompute([numeric(value, label + ".sample") for value in values])
            stored = mapping(paths.get(path), label + ".path")
            for key, value in computed[path].items():
                if key == "samples": require(integer(stored.get(key), label + ".samples") == value, label + ": sample count recompute")
                else: close(stored.get(key), float(value), label + ".summary")
        speedups = {q: float(computed[PATHS[1]][q]) / float(computed[PATHS[0]][q]) for q in minimums}
        stored_speedups = mapping(item.get("speedup_c1_over_pinned_x"), label + ".speedups")
        for q, value in speedups.items(): close(stored_speedups.get(q), value, label + ".speedup"); minimums[q] = min(minimums[q], value)
        require(boolean(mapping(item.get("relative_gate"), label + ".gate").get("passed"), label + ".relative.pass") is all(value >= MIN_SPEEDUP for value in speedups.values()), label + ": relative-gate recompute")
    return {"pid": pid, "job_id": job, "gpu_uuid": str(pre_values["uuid"]), "extension_sha256": str(extension["sha256"]), "target": validated_target, "runner_sha256": expected_runner_sha, "candidate_helper_sha256": CANDIDATE_HELPER_SHA256, "protocol_shell": dict(protocol_shell), "production_wrapper": dict(wrapper), "runtime_import_ledger_authority": RUNTIME_LEDGER_AUTHORITY, "runtime_import_identities": runtime_identities, "minimum_speedups_x": minimums, "relative_performance_pass": all(value >= MIN_SPEEDUP for value in minimums.values())}


def build_status(records: list[dict[str, object]], telem: Mapping[str, Any]) -> dict[str, object]:
    performance = all(boolean(record["relative_performance_pass"], "record.relative_performance_pass") for record in records)
    telemetry_valid = boolean(telem.get("allocation_telemetry_valid"), "telemetry.allocation_telemetry_valid")
    allocation_pass = telemetry_valid and performance
    return {
        "allocation_valid": telemetry_valid,
        "relative_performance_pass": performance,
        "allocation_pass": allocation_pass,
        "classification": "telemetry_invalid_not_performance_failure" if not telemetry_valid else "relative_performance_failure" if not performance else "pass",
    }


def protocol_identity(analyzer: Mapping[str, Any], runner: Mapping[str, Any], shell: Mapping[str, Any]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA,
        "target": {"cell": TARGET, "offsets": list(OFFSETS), "variant": "vshard4_p2"},
        "runner": dict(runner),
        "analyzer": dict(analyzer),
        "protocol_shell": dict(shell),
        "candidate_helper_sha256": CANDIDATE_HELPER_SHA256,
        "runtime_import_ledger_authority": RUNTIME_LEDGER_AUTHORITY,
        "runtime_import_identity_keys": sorted(RUNTIME_IMPORT_IDENTITY_KEYS),
    }


def source_identity(records: list[dict[str, object]], wrapper: Mapping[str, Any]) -> dict[str, object]:
    return {
        "candidate_helper_sha256": CANDIDATE_HELPER_SHA256,
        "production_wrapper": dict(wrapper),
        "runtime_import_ledger_authority": RUNTIME_LEDGER_AUTHORITY,
        "runtime_import_identities_by_pid": [record["runtime_import_identities"] for record in records],
    }


def recompute_allocation(*, allocation_id: str, runner_paths: list[Path], runner_digests: list[str], cli_runner_sha: str, telemetry_path: Path, telemetry_digest: str, window_sidecar_path: Path, window_sidecar_digest: str, window: Mapping[str, Any]) -> dict[str, object]:
    """The only authority for performance, telemetry, and all allocation flags."""

    analyzer = analyzer_identity()
    current_runner = current_runner_identity()
    shell = current_protocol_shell_identity()
    wrapper = current_wrapper_identity()
    expected_runner = valid_sha(cli_runner_sha, "cli.expected_runner_sha256")
    require(expected_runner == current_runner["sha256"], "CLI/current runner SHA drift")
    require(len(runner_paths) == len(runner_digests) == 2 and len({str(path.resolve()) for path in runner_paths}) == 2, "raw runner artifact count/path drift")
    artifacts: list[dict[str, object]] = []
    inputs: list[Mapping[str, Any]] = []
    for index, (path, digest) in enumerate(zip(runner_paths, runner_digests, strict=True)):
        digest = valid_sha(digest, f"runner{index}.cli_expected_sha256")
        data, actual_digest = read_json(path, digest, f"runner{index}")
        require(actual_digest == digest, f"runner{index}: raw artifact SHA mismatch")
        artifacts.append({"process_index": index, "path": str(path.resolve()), "sha256": actual_digest, "cli_expected_sha256": digest})
        inputs.append(data)
    records = [validate_runner(inputs[index], f"runner{index}", allocation_id, index, expected_runner) for index in range(2)]
    require(len({record["pid"] for record in records}) == 2 and len({record["job_id"] for record in records}) == 1 and len({record["gpu_uuid"] for record in records}) == 1 and len({record["extension_sha256"] for record in records}) == 1, "fresh PID/allocation/GPU/SO identity drift")
    for index, record in enumerate(records):
        strict_json_equal(record["target"], {"cell": TARGET, "offsets": OFFSETS, "variant": "vshard4_p2"}, f"runner{index}.target")
        strict_json_equal(record["protocol_shell"], shell, f"runner{index}.protocol_shell")
        strict_json_equal(record["production_wrapper"], wrapper, f"runner{index}.production_wrapper")
    require(all(record["runner_sha256"] == expected_runner and record["candidate_helper_sha256"] == CANDIDATE_HELPER_SHA256 and record["runtime_import_ledger_authority"] == RUNTIME_LEDGER_AUTHORITY for record in records), "runner/helper/runtime-ledger authority drift")
    telem_digest = valid_sha(telemetry_digest, "telemetry.cli_expected_sha256")
    checked_window = telemetry_window(window.get("main0_start_ns"), window.get("main1_end_ns"), "telemetry.window")
    window_sidecar, actual_window_digest = read_json(window_sidecar_path, window_sidecar_digest, "telemetry_window_sidecar")
    validate_telemetry_window_sidecar(window_sidecar, checked_window, "telemetry_window_sidecar")
    telem = telemetry(telemetry_path, telem_digest, str(records[0]["gpu_uuid"]), checked_window)
    allocation_identity = {"slurm_job_id": records[0]["job_id"], "gpu_uuid": records[0]["gpu_uuid"], "extension_sha256": records[0]["extension_sha256"], "fresh_pids": [record["pid"] for record in records]}
    return {
        "allocation_identity": allocation_identity,
        "protocol_identity": protocol_identity(analyzer, current_runner, shell),
        "source_identity": source_identity(records, wrapper),
        "runner_records": records,
        "input_artifacts": {"raw_runner_json": artifacts, "telemetry_csv": {"path": str(telemetry_path.resolve()), "sha256": telem_digest, "cli_expected_sha256": telem_digest, "window": checked_window}, "telemetry_window_sidecar": {"path": str(window_sidecar_path.resolve()), "sha256": actual_window_digest, "cli_expected_sha256": valid_sha(window_sidecar_digest, "telemetry_window_sidecar.cli_expected_sha256")}},
        "telemetry": telem,
        "status": build_status(records, telem),
        "analyzer": analyzer,
    }


def allocation(args: argparse.Namespace) -> None:
    computed = recompute_allocation(
        allocation_id=args.allocation_id, runner_paths=list(args.runner_json), runner_digests=list(args.expected_runner_sha256s),
        cli_runner_sha=args.expected_runner_sha256, telemetry_path=args.telemetry_csv, telemetry_digest=args.expected_telemetry_sha256, window_sidecar_path=args.telemetry_window_sidecar, window_sidecar_digest=args.expected_telemetry_window_sidecar_sha256,
        window={"main0_start_ns": args.telemetry_window_start_ns, "main1_end_ns": args.telemetry_window_end_ns},
    )
    a1_prerequisite: object = None
    if args.allocation_id == "A2":
        require(args.a1_allocation_manifest is not None and args.expected_a1_allocation_manifest_sha256 is not None and args.current_slurm_job_id is not None, "A2 allocation requires fixed A1 manifest path/SHA and current Slurm job")
        current_job = positive_job_id(args.current_slurm_job_id, "A2 current Slurm job")
        require(current_job == mapping(computed["allocation_identity"], "A2.identity").get("slurm_job_id"), "A2 CLI/current raw Slurm job drift")
        try:
            a1_manifest = args.a1_allocation_manifest.resolve(strict=True)
        except OSError as exc:
            raise AuditError("A2 fixed A1 manifest path does not resolve") from exc
        a1_prerequisite = reopen_a1_prerequisite(a1_manifest, args.expected_a1_allocation_manifest_sha256, current_job, "A2 allocation prerequisite")
    else:
        require(args.a1_allocation_manifest is None and args.expected_a1_allocation_manifest_sha256 is None and args.current_slurm_job_id is None, "A1 allocation must not accept an A1 prerequisite binding")
    output = {"schema_version": SCHEMA, "kind": "one_allocation_relative_only", "allocation_id": args.allocation_id, "a1_prerequisite": a1_prerequisite, **{key: value for key, value in computed.items() if key != "status"}, **mapping(computed["status"], "computed.status")}
    args.json.parent.mkdir(parents=True, exist_ok=True); args.json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(f"wrote {args.json}; classification={output['classification']}")
    if args.require_pass and not boolean(output["allocation_pass"], "allocation_pass"):
        print("ALLOCATION_REQUIRE_PASS_FAILED", file=sys.stderr)
        raise SystemExit(3)


def require_a1_prerequisite_identity(allocation_id: object, a1_job: object, current_job: object) -> None:
    require(allocation_id == "A1", "A2 prerequisite must be exact allocation A1")
    require(positive_job_id(a1_job, "A1 prerequisite job") != positive_job_id(current_job, "current Slurm job"), "A1 prerequisite is not independent of current A2 Slurm job")


def require_exact_a1_binding(value: object, expected: Mapping[str, object], label: str) -> None:
    binding = mapping(value, label)
    require(set(binding) == {"path", "sha256", "slurm_job_id", "source_identity"}, label + ": A2 binding key set drift")
    strict_json_equal(binding, expected, label)


def reopen_a1_prerequisite(path: Path, expected_sha: str, current_job: object, label: str) -> dict[str, object]:
    """Recompute fixed A1 evidence, never trusting its stored eligibility flags."""

    data, actual_sha = read_json(path, expected_sha, label + ".manifest")
    computed = revalidate_allocation_manifest(data, label, "A1")
    identity = mapping(computed["allocation_identity"], label + ".identity")
    require_a1_prerequisite_identity("A1", identity.get("slurm_job_id"), current_job)
    status = mapping(computed["status"], label + ".status")
    require(boolean(status["allocation_valid"], label + ".valid") and boolean(status["relative_performance_pass"], label + ".performance") and boolean(status["allocation_pass"], label + ".pass"), label + ": A1 prerequisite raw recomputation is not eligible")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": actual_sha,
        "slurm_job_id": positive_job_id(identity.get("slurm_job_id"), label + ".slurm_job_id"),
        "source_identity": dict(mapping(computed["source_identity"], label + ".source_identity")),
    }


def revalidate_allocation_manifest(data: Mapping[str, Any], label: str, allocation_id: str) -> dict[str, object]:
    expected_manifest_keys = {"schema_version", "kind", "allocation_id", "a1_prerequisite", "allocation_identity", "protocol_identity", "source_identity", "runner_records", "input_artifacts", "telemetry", "analyzer", "allocation_valid", "relative_performance_pass", "allocation_pass", "classification"}
    require(set(data) == expected_manifest_keys and integer(data.get("schema_version"), label + ".schema") == SCHEMA and data.get("kind") == "one_allocation_relative_only" and data.get("allocation_id") == allocation_id, label + ": allocation manifest identity drift")
    analyzer = analyzer_identity()
    strict_json_equal(mapping(data.get("analyzer"), label + ".analyzer"), analyzer, label + ".analyzer")
    inputs = mapping(data.get("input_artifacts"), label + ".input_artifacts")
    raw = inputs.get("raw_runner_json")
    require(isinstance(raw, list) and len(raw) == 2, label + ": raw runner artifact list drift")
    runner_paths: list[Path] = []
    runner_digests: list[str] = []
    for index, item_value in enumerate(raw):
        item = mapping(item_value, label + f".raw{index}")
        require(integer(item.get("process_index"), label + ".raw.process_index") == index, label + ": raw runner process index drift")
        path, digest, cli = item.get("path"), item.get("sha256"), item.get("cli_expected_sha256")
        require(isinstance(path, str) and path and isinstance(digest, str) and isinstance(cli, str) and digest == cli, label + ": raw runner artifact field/type drift")
        runner_paths.append(Path(path)); runner_digests.append(valid_sha(digest, label + ".raw.sha"))
    telemetry_input = mapping(inputs.get("telemetry_csv"), label + ".telemetry_input")
    tpath, tdigest, tcli = telemetry_input.get("path"), telemetry_input.get("sha256"), telemetry_input.get("cli_expected_sha256")
    require(isinstance(tpath, str) and tpath and isinstance(tdigest, str) and isinstance(tcli, str) and tdigest == tcli, label + ": telemetry artifact field/type drift")
    window = mapping(telemetry_input.get("window"), label + ".telemetry_window")
    window_sidecar = mapping(inputs.get("telemetry_window_sidecar"), label + ".telemetry_window_sidecar")
    wpath, wdigest, wcli = window_sidecar.get("path"), window_sidecar.get("sha256"), window_sidecar.get("cli_expected_sha256")
    require(isinstance(wpath, str) and wpath and isinstance(wdigest, str) and isinstance(wcli, str) and wdigest == wcli, label + ": telemetry window sidecar field/type drift")
    protocol = mapping(data.get("protocol_identity"), label + ".protocol")
    runner = mapping(protocol.get("runner"), label + ".protocol.runner")
    computed = recompute_allocation(
        allocation_id=allocation_id, runner_paths=runner_paths, runner_digests=runner_digests,
        cli_runner_sha=str(runner.get("sha256", "")), telemetry_path=Path(tpath), telemetry_digest=valid_sha(tdigest, label + ".telemetry.sha"), window_sidecar_path=Path(wpath), window_sidecar_digest=valid_sha(wdigest, label + ".window.sha"), window=window,
    )
    # Any stored status is evidence only.  Exact equality with a fresh raw
    # recomputation makes forged top-level flags, summary values, or telemetry
    # classifications fail closed.
    for key in ("allocation_identity", "protocol_identity", "source_identity", "runner_records", "input_artifacts", "telemetry", "analyzer"):
        strict_json_equal(data.get(key), computed[key], label + ".stored." + key)
    status = mapping(computed["status"], label + ".computed.status")
    for key in ("allocation_valid", "relative_performance_pass", "allocation_pass"):
        require(boolean(data.get(key), label + "." + key) is status[key], label + ": forged/stale top-level " + key)
    require(isinstance(data.get("classification"), str) and data.get("classification") == status["classification"], label + ": forged/stale top-level classification")
    if allocation_id == "A1":
        require(data.get("a1_prerequisite") is None, label + ": A1 must not carry an A1 prerequisite")
    else:
        current_job = mapping(computed["allocation_identity"], label + ".A2.identity").get("slurm_job_id")
        binding = mapping(data.get("a1_prerequisite"), label + ".A2.a1_prerequisite")
        require(set(binding) == {"path", "sha256", "slurm_job_id", "source_identity"}, label + ": A2 prerequisite key set drift")
        path, digest = binding.get("path"), binding.get("sha256")
        require(isinstance(path, str) and bool(path) and isinstance(digest, str), label + ": A2 prerequisite path/SHA type drift")
        actual_binding = reopen_a1_prerequisite(Path(path), valid_sha(digest, label + ".A2.a1_prerequisite.sha256"), current_job, label + ".A2.a1_prerequisite")
        require_exact_a1_binding(binding, actual_binding, label + ".A2.a1_prerequisite")
    return computed


def verify_allocation(args: argparse.Namespace) -> None:
    if args.require_independent_current_job:
        require(args.expected_allocation_id == "A1" and args.current_slurm_job_id is not None, "independent-current-job verification is only valid for fixed A1")
        reopen_a1_prerequisite(args.allocation_manifest, args.expected_allocation_sha256, args.current_slurm_job_id, "prerequisite")
    else:
        data, _ = read_json(args.allocation_manifest, args.expected_allocation_sha256, "prerequisite")
        computed = revalidate_allocation_manifest(data, "prerequisite", args.expected_allocation_id)
        status = mapping(computed["status"], "prerequisite.status")
        require(boolean(status["allocation_valid"], "prerequisite.valid") and boolean(status["relative_performance_pass"], "prerequisite.performance") and boolean(status["allocation_pass"], "prerequisite.pass"), "allocation prerequisite raw recomputation is not eligible")
    print("ALLOCATION_REVALIDATED=" + str(args.allocation_manifest.resolve()))


def freeze(args: argparse.Namespace) -> None:
    analyzer = analyzer_identity()
    left, left_sha = read_json(args.allocation_a, args.expected_allocation_a_sha256, "A1")
    right, _right_sha = read_json(args.allocation_b, args.expected_allocation_b_sha256, "A2")
    left_computed = revalidate_allocation_manifest(left, "A1", "A1")
    right_computed = revalidate_allocation_manifest(right, "A2", "A2")
    left_identity, right_identity = mapping(left_computed["allocation_identity"], "A1.identity"), mapping(right_computed["allocation_identity"], "A2.identity")
    require(left_identity.get("slurm_job_id") != right_identity.get("slurm_job_id"), "A1/A2 are not independent Slurm allocations")
    require(left_identity.get("extension_sha256") == right_identity.get("extension_sha256"), "A1/A2 extension SO identity drift")
    strict_json_equal(left_computed["protocol_identity"], right_computed["protocol_identity"], "freeze.protocol_identity")
    strict_json_equal(left_computed["source_identity"], right_computed["source_identity"], "freeze.source_identity")
    expected_a1_binding = reopen_a1_prerequisite(args.allocation_a, left_sha, right_identity.get("slurm_job_id"), "freeze.A2_exact_A1_binding")
    require_exact_a1_binding(right.get("a1_prerequisite"), expected_a1_binding, "freeze.A2.a1_prerequisite")
    strict_json_equal(expected_a1_binding["source_identity"], left_computed["source_identity"], "freeze.A1_binding_source_identity")
    left_status, right_status = mapping(left_computed["status"], "A1.status"), mapping(right_computed["status"], "A2.status")
    eligible = boolean(left_status["allocation_pass"], "A1.pass") and boolean(right_status["allocation_pass"], "A2.pass")
    output = {"schema_version": SCHEMA, "kind": "public_freeze_decision", "production_action": "unchanged", "A1_A2_independent_job_ids": [left_identity.get("slurm_job_id"), right_identity.get("slurm_job_id")], "A2_exact_A1_prerequisite": expected_a1_binding, "eligible_for_public_freeze": eligible, "criteria": {"A1_valid": left_status["allocation_valid"], "A2_valid": right_status["allocation_valid"], "A1_relative_performance_pass": left_status["relative_performance_pass"], "A2_relative_performance_pass": right_status["relative_performance_pass"]}, "protocol_identity": left_computed["protocol_identity"], "source_identity": left_computed["source_identity"], "policy": "relative-only; telemetry invalidates an allocation rather than constituting a performance failure", "analyzer": analyzer}
    args.json.parent.mkdir(parents=True, exist_ok=True); args.json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(f"wrote {args.json}; eligible_for_public_freeze={eligible}")
    if args.require_eligible and not eligible:
        print("FREEZE_REQUIRE_ELIGIBLE_FAILED", file=sys.stderr)
        raise SystemExit(3)


def expect_rejected(name: str, thunk: object) -> None:
    try:
        thunk()  # type: ignore[operator]
    except AuditError:
        return
    raise AssertionError("self-test accepted adversarial case: " + name)


def self_test(_args: argparse.Namespace) -> None:
    """No-GPU adversarial tests for the release gates' trust boundaries."""

    expect_rejected("bool-offset", lambda: target_identity({"cell": TARGET, "offsets": [0, True, 2, 3, 4, 5, 12288], "variant": "vshard4_p2"}, "self.bool"))
    expect_rejected("descriptor-false-offset", lambda: exact_offsets([False, 1, 2, 3, 4, 5, 12288], "self.descriptor_false"))
    expect_rejected("decision-true-certified-offset", lambda: exact_offsets([0, True, 2, 3, 4, 5, 12288], "self.decision_true"))
    identity_schema = {name: {} for name in ("runner", "protocol_shell", "candidate_helper", "production_wrapper", "patched_tracked_identity")}
    required_runner_identity_schema(identity_schema, "self.identity")
    expect_rejected("identity-missing-protocol-shell", lambda: required_runner_identity_schema({name: {} for name in ("runner", "candidate_helper", "production_wrapper", "patched_tracked_identity")}, "self.identity_missing"))
    low_records = [{"relative_performance_pass": False}, {"relative_performance_pass": True}]
    valid_telem = {"allocation_telemetry_valid": True}
    status = build_status(low_records, valid_telem)
    strict_json_equal(status, {"allocation_valid": True, "relative_performance_pass": False, "allocation_pass": False, "classification": "relative_performance_failure"}, "self.low_speed_status")
    # Top-level flags are deliberately not inputs to build_status; a forged
    # success would differ from the value revalidate_allocation_manifest checks.
    forged = {"allocation_valid": True, "relative_performance_pass": True, "allocation_pass": True, "classification": "pass"}
    expect_rejected("forged-top-level-success", lambda: strict_json_equal(forged, status, "self.top_level_status"))
    invalid_status = build_status([{"relative_performance_pass": True}, {"relative_performance_pass": True}], {"allocation_telemetry_valid": False})
    require(invalid_status["classification"] == "telemetry_invalid_not_performance_failure" and invalid_status["allocation_pass"] is False, "self-test telemetry invalid classification")
    expect_rejected("A2-as-A1-prerequisite", lambda: require_a1_prerequisite_identity("A2", "12", "13"))
    expect_rejected("same-A1-A2-job", lambda: require_a1_prerequisite_identity("A1", "12", "12"))
    fixed_binding = {"path": "/fixed/A1.json", "sha256": "0" * 64, "slurm_job_id": "12", "source_identity": {"source": "fixed", "sha256_gate_pass": True, "nested": [1, True]}}
    expect_rejected("forged-A2-A1-binding", lambda: require_exact_a1_binding({**fixed_binding, "sha256": "1" * 64}, fixed_binding, "self.binding"))
    expect_rejected("nested-bool-as-int-binding", lambda: require_exact_a1_binding({**fixed_binding, "source_identity": {"source": "fixed", "sha256_gate_pass": 1, "nested": [1, True]}}, fixed_binding, "self.nested_bool"))
    expect_rejected("nested-int-as-bool-binding", lambda: require_exact_a1_binding({**fixed_binding, "source_identity": {"source": "fixed", "sha256_gate_pass": True, "nested": [1, 1]}}, fixed_binding, "self.nested_int"))
    expect_rejected("nested-list-bool-int-attack", lambda: strict_json_equal({"items": [[True]]}, {"items": [[1]]}, "self.nested_list"))
    expect_rejected("nested-float-int-attack", lambda: strict_json_equal({"items": [[1]]}, {"items": [[1.0]]}, "self.nested_float"))
    telemetry_test_window = {"main0_start_ns": 1_000_000_000, "main1_end_ns": 21_000_000_000}
    expect_rejected("telemetry-sparse-middle-gap", lambda: validate_telemetry_stamps([1_000_000_000, 1_200_000_000, 20_980_000_000, 21_000_000_000], telemetry_test_window, "self.telemetry_gap"))
    expect_rejected("telemetry-sidecar-bool-forgery", lambda: validate_telemetry_window_sidecar({"main0_start_ns": 1_000_000_000, "main1_end_ns": 21_000_000_000, "telemetry_pid_was_alive": 1}, telemetry_test_window, "self.telemetry_sidecar"))
    class OneReadPath:
        def __init__(self, payload: bytes) -> None:
            self.payload, self.read_count = payload, 0
        def is_file(self) -> bool: return True
        def read_bytes(self) -> bytes:
            self.read_count += 1
            require(self.read_count == 1, "self-test hashed payload read more than once")
            return self.payload
    one_read = OneReadPath(b'{"trusted":true}')
    trusted, _ = read_json(one_read, hashlib.sha256(one_read.payload).hexdigest(), "self.one_read")  # type: ignore[arg-type]
    require(one_read.read_count == 1 and boolean(trusted.get("trusted"), "self.one_read.trusted"), "self-test JSON hash/parse payload mismatch")
    sm_79_checks, sm_79_fraction = sm_clock_quality([2000.0] * 79 + [120.0] * 21)
    require(sm_79_fraction == 0.79 and not all(sm_79_checks.values()), "self-test 79% SM clock fraction must fail")
    sm_80_checks, sm_80_fraction = sm_clock_quality([2000.0] * 80 + [120.0] * 20)
    require(sm_80_fraction == 0.80 and all(sm_80_checks.values()), "self-test 80% SM clock fraction must pass")
    sm_low_checks, _ = sm_clock_quality([120.0] * 100)
    require(sm_low_checks["sm_clock_all_positive"] and not sm_low_checks["sm_clock_median_at_least_1000mhz"], "self-test low median SM clock must fail")
    expected = {"path": "/expected/a.py", "expected_path": "/expected/a.py", "expected_root": "/expected", "sha256": "0" * 64, "sha256_gate_pass": True}
    drifted_path = {**expected, "path": "/other/a.py"}
    drifted_wrapper = {**expected, "sha256": "1" * 64}
    expect_rejected("ledger-path-drift", lambda: strict_json_equal(drifted_path, expected, "self.ledger_path"))
    expect_rejected("wrapper-sha-drift", lambda: strict_json_equal(drifted_wrapper, expected, "self.wrapper"))
    print("SELF_TEST_PASS=bool_offset,descriptor_false_offset,decision_true_offset,identity_schema,top_flag,raw_low_speed,telemetry_invalid,telemetry_gap,telemetry_sidecar_type,single_read_hash_parse,sm_clock_79_fail,sm_clock_80_pass,sm_clock_low_median_fail,A2_prerequisite,A2_binding,nested_bool_int,nested_list,nested_float_int,ledger_path,wrapper")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    allocation_parser = sub.add_parser("allocation"); allocation_parser.add_argument("--allocation-id", choices=("A1", "A2"), required=True); allocation_parser.add_argument("--runner-json", nargs=2, type=Path, required=True); allocation_parser.add_argument("--expected-runner-sha256s", nargs=2, required=True); allocation_parser.add_argument("--expected-runner-sha256", required=True); allocation_parser.add_argument("--telemetry-csv", type=Path, required=True); allocation_parser.add_argument("--expected-telemetry-sha256", required=True); allocation_parser.add_argument("--telemetry-window-sidecar", type=Path, required=True); allocation_parser.add_argument("--expected-telemetry-window-sidecar-sha256", required=True); allocation_parser.add_argument("--telemetry-window-start-ns", type=int, required=True); allocation_parser.add_argument("--telemetry-window-end-ns", type=int, required=True); allocation_parser.add_argument("--a1-allocation-manifest", type=Path); allocation_parser.add_argument("--expected-a1-allocation-manifest-sha256"); allocation_parser.add_argument("--current-slurm-job-id"); allocation_parser.add_argument("--json", type=Path, required=True); allocation_parser.add_argument("--require-pass", action="store_true", help="write manifest, then exit nonzero unless raw recomputation passes"); allocation_parser.set_defaults(func=allocation)
    verify_parser = sub.add_parser("verify-allocation"); verify_parser.add_argument("--allocation-manifest", type=Path, required=True); verify_parser.add_argument("--expected-allocation-sha256", required=True); verify_parser.add_argument("--expected-allocation-id", choices=("A1", "A2"), required=True); verify_parser.add_argument("--current-slurm-job-id", required=True); verify_parser.add_argument("--require-independent-current-job", action="store_true"); verify_parser.set_defaults(func=verify_allocation)
    freeze_parser = sub.add_parser("freeze"); freeze_parser.add_argument("--allocation-a", type=Path, required=True); freeze_parser.add_argument("--expected-allocation-a-sha256", required=True); freeze_parser.add_argument("--allocation-b", type=Path, required=True); freeze_parser.add_argument("--expected-allocation-b-sha256", required=True); freeze_parser.add_argument("--json", type=Path, required=True); freeze_parser.add_argument("--require-eligible", action="store_true", help="write decision, then exit nonzero unless eligible"); freeze_parser.set_defaults(func=freeze)
    self_parser = sub.add_parser("self-test"); self_parser.set_defaults(func=self_test)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    try: main()
    except AuditError as exc: print("AUDIT_FAIL: " + str(exc), file=sys.stderr); raise SystemExit(2)
