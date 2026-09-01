#!/usr/bin/env python3
"""Strict stdlib-only auditor for the v5 public-registry cross-map regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Mapping


SCHEMA_VERSION = 4
SAMPLES = 100
WARMUP = 12
PERCENTILES = ("p50", "p95", "p99")
AUTO_SHA = "9633179c6a5e8ce61b1857a102f81074c6bdfdc63163e13980f40c24d6583c29"
BACKEND_SHA = "152ac4ba897c1887963553d592456b7f988c89105e6f6cb507a44757058d5fd1"
EXTENSION_SHA = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
HELPER_PATH = "/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so"
HELPER_SHA = "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
SUPPORTING_HELPER_SHA256 = {
    "varlen_helper": "e07481e7e66551f1f9eaafa7f1fb9db9ef193f1694ff831ee0a07802c8ffbe14",
    "tail_helper": "f4144f5fbdd61396ff907c6290b767b5570e04d19087f8332f9db10e56e7b1dc",
    "shared_seqcount": "4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f",
    "confirmation": "9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b",
    "harness": "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52",
    "varlen_metadata": "f89a97ba284c7a24a3df54efca7bb60eb70f9cbda659200bd6cb3e8dfdaf4ccd",
    "reference_torch_ref": "bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5",
}
FLA_SOURCE_SHA256 = {
    "fla/__init__.py": "b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d",
    "fla/ops/backends/__init__.py": "a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635",
    "fla/ops/kda/__init__.py": "24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb",
    "fla/ops/kda/backends/__init__.py": "86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797",
    "fla/ops/kda/backends/flash_kda.py": "0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2",
    "fla/ops/kda/chunk.py": "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8",
}
PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
PATCHED_DIRTY = {
    "csrc/flash_kda.cpp": "38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4",
    "csrc/fwd.h": "613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083",
    "csrc/smxx/fwd_launch.cu": "a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928",
}
POSITIVES = {
    "fixed_b2_h12_t2048_fp32_both": ("vshard4_p2", "fixed_batch_b2_h12_t2048_fp32_both_whitelist_hit", True),
    "fixed_b5_h12_t2048_fp32_both": ("vshard2_p2", "fixed_batch_b5_h12_t2048_fp32_both_whitelist_hit", True),
    "fixed_b1_h12_t8191_none": ("vshard4_p2", "fixed_single_batch_b1_h12_t8191_none_whitelist_hit", False),
    "varlen_skew_n6_h12_t12288_fp32_both": ("vshard4_p2", "varlen_skew_n6_h12_t12288_fp32_both_whitelist_hit", True),
}
NEGATIVES = {
    "b7_none": "fixed_batch_shape_not_whitelisted",
    "t8191_fp32_both": "state_contract_fp32_both_h12_length_not_whitelisted",
    "adjacent_offsets_fp32_both": "varlen_offsets_not_whitelisted",
}
CELL_SEQUENCES = {
    "fixed_b2_h12_t2048_fp32_both": 2,
    "fixed_b5_h12_t2048_fp32_both": 5,
    "fixed_b1_h12_t8191_none": 1,
    "varlen_skew_n6_h12_t12288_fp32_both": 6,
    "b7_none": 7,
    "t8191_fp32_both": 1,
    "adjacent_offsets_fp32_both": 6,
}
VARLEN_OFFSETS = {
    "varlen_skew_n6_h12_t12288_fp32_both": [0, 1, 2, 3, 4, 5, 12288],
    "adjacent_offsets_fp32_both": [0, 1, 2, 3, 4, 6, 12288],
}
CELL_OUTPUT_SHAPES = {
    "fixed_b2_h12_t2048_fp32_both": [2, 2048, 12, 128],
    "fixed_b5_h12_t2048_fp32_both": [5, 2048, 12, 128],
    "fixed_b1_h12_t8191_none": [1, 8191, 12, 128],
    "varlen_skew_n6_h12_t12288_fp32_both": [1, 12288, 12, 128],
    "b7_none": [7, 2048, 12, 128],
    "t8191_fp32_both": [1, 8191, 12, 128],
    "adjacent_offsets_fp32_both": [1, 12288, 12, 128],
}
VARLEN_CACHE_FIELDS = {
    "entries", "hits", "misses", "capture_miss_rejections", "capture_hit_rejections",
}
LOADED_RUNTIME_MODULES = {
    "auto_dispatch": "assignment02.team.c1_flashkda.challenge_tp8_dispatch.auto_dispatch",
    "fla_backend": "assignment02.team.c1_flashkda.challenge_tp8_dispatch.fla_backend",
    "varlen_metadata": "assignment02.team.c1_flashkda.challenge_tp8_dispatch.varlen_metadata",
    "shared_seqcount": "assignment02.team.c1_flashkda.challenge_seqcount_dispatch.run_seqcount_dispatch",
    "confirmation": "assignment02.team.c1_flashkda.challenge_varlen_dispatch.run_varlen_dispatch_confirmation",
    "varlen_helper": "assignment02.team.c1_flashkda.challenge_varlen_dispatch.run_varlen_fla_handoff_candidate",
    "tail_helper": "assignment02.team.c1_flashkda.challenge_tail8191_production_freeze.run_tail8191_production_freeze",
    "harness": "assignment02.team.c1_flashkda.harness.validate_and_bench",
    "fla_ops_kda": "fla.ops.kda",
}
BOOTSTRAP_STAGES = [
    "source_ledger_pre_torch",
    "pre_torch_clean_gpu",
    "heavy_runtime_import",
    "loaded_module_identity",
    "shared_make_inputs_hydration",
    "canonical_map_pre",
]
SHARED_SEQCOUNT_LABEL = "shared_seqcount"
SHARED_MAKE_INPUTS = "_make_inputs"
SHARED_BINDING_RECORD_FIELDS = {
    "phase", "module_label", "module", "module_path", "module_object_id",
    "function", "function_module", "function_object_id",
    "function_globals_is_module_dict", "torch_global_present", "torch_object_id",
    "sys_modules_torch_object_id", "torch_is_sys_modules_canonical", "passed",
}
MAIN_SCHEMA_FIELDS = {
    "schema_version", "purpose", "allocation_id", "process_index", "allocation", "process",
    "positive_cells", "public_registry_negative_controls", "source_pre_torch", "bootstrap",
    "source_pre", "source_post", "map_pre", "map_post", "map_readonly", "identity",
    "performance", "complete",
}


class AuditError(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def obj(value: object, label: str) -> Mapping[str, Any]:
    require(type(value) is dict, f"{label} must be a JSON object")
    return value  # type: ignore[return-value]


def arr(value: object, label: str) -> list[Any]:
    require(type(value) is list, f"{label} must be a JSON array")
    return value  # type: ignore[return-value]


def boolean(value: object, label: str) -> bool:
    require(type(value) is bool, f"{label} must be an exact bool")
    return value  # type: ignore[return-value]


def integer(value: object, label: str) -> int:
    require(type(value) is int, f"{label} must be an integer (bool rejected)")
    return value  # type: ignore[return-value]


def string(value: object, label: str) -> str:
    require(type(value) is str and bool(value), f"{label} must be a nonempty string")
    return value  # type: ignore[return-value]


def digest(value: object, label: str) -> str:
    text = string(value, label)
    require(len(text) == 64 and all(char in "0123456789abcdef" for char in text), f"{label} must be lowercase SHA256")
    return text


def finite(value: object, label: str) -> float:
    require(type(value) in (int, float) and type(value) is not bool, f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def floating(value: object, label: str) -> float:
    require(type(value) is float and math.isfinite(value), f"{label} must be a finite JSON float (int/bool rejected)")
    return value  # type: ignore[return-value]


def strict_equal(left: object, right: object, label: str) -> None:
    require(type(left) is type(right), f"{label}: JSON type drift")
    if type(left) is dict:
        require(set(left) == set(right), f"{label}: object keys drift")
        for key in left:
            strict_equal(left[key], right[key], f"{label}.{key}")
    elif type(left) is list:
        require(len(left) == len(right), f"{label}: array length drift")
        for index, (a, b) in enumerate(zip(left, right)):
            strict_equal(a, b, f"{label}[{index}]")
    else:
        require(left == right, f"{label}: value drift")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_once(path: Path, expected_sha: str, label: str) -> tuple[Mapping[str, Any], dict[str, str]]:
    resolved = path.resolve(strict=True)
    data = resolved.read_bytes()
    actual = _sha_bytes(data)
    require(actual == digest(expected_sha, label + ".expected_sha"), f"{label}: SHA mismatch")
    try:
        decoded = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label}: invalid UTF-8 JSON") from exc
    return obj(decoded, label), {"path": str(resolved), "sha256": actual}


def write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values); point = (len(ordered) - 1) * q
    low, high = int(point), min(int(point) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (point - low)


def summary(values: list[float]) -> dict[str, float | int]:
    return {"samples": len(values), "mean_ms": statistics.fmean(values), "p50_ms": percentile(values, .5), "p95_ms": percentile(values, .95), "p99_ms": percentile(values, .99)}


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _decision_object(value: object, label: str) -> Mapping[str, Any]:
    decision = obj(value, label)
    expected = {"requested_variant", "chosen_variant", "reason", "extension_sha256", "varlen_cpu_authoritative", "certified_varlen_offsets", "canonical_cache_hit"}
    require(set(decision) == expected, f"{label}: decision keys drift")
    require("test_only_route" not in decision, f"{label}: test-only marker forbidden")
    return decision


def validate_positive_decision(value: object, variant: str, reason: str, *, offsets: list[int] | None, cache_hit: bool | None, label: str) -> None:
    decision = _decision_object(value, label)
    require(decision["requested_variant"] == variant and decision["chosen_variant"] == variant and decision["reason"] == reason, f"{label}: variant/reason drift")
    require(digest(decision["extension_sha256"], label + ".extension") == EXTENSION_SHA, f"{label}: extension drift")
    if offsets is None:
        require(boolean(decision["varlen_cpu_authoritative"], label + ".varlen_cpu_authoritative") is False and decision["certified_varlen_offsets"] is None and decision["canonical_cache_hit"] is None, f"{label}: fixed decision provenance drift")
    else:
        require(boolean(decision["varlen_cpu_authoritative"], label + ".varlen_cpu_authoritative") is True and decision["certified_varlen_offsets"] == offsets and (cache_hit is None or boolean(decision["canonical_cache_hit"], label + ".canonical_cache_hit") is cache_hit), f"{label}: varlen decision provenance drift")


def validate_baseline_decision(value: object, reason: str, label: str) -> None:
    decision = _decision_object(value, label)
    require(
        decision["requested_variant"] == "baseline"
        and decision["chosen_variant"] == "baseline"
        and decision["reason"] == reason
        and decision["extension_sha256"] is None
        and boolean(decision["varlen_cpu_authoritative"], label + ".varlen_cpu_authoritative") is False
        and decision["certified_varlen_offsets"] is None
        and decision["canonical_cache_hit"] is None,
        f"{label}: baseline-null decision provenance drift",
    )


def _validate_output_abi(value: object, output_shape: list[int], label: str) -> None:
    abi = obj(value, label)
    shape = arr(abi.get("shape"), label + ".shape")
    require(set(abi) == {"shape", "dtype", "contiguous"} and shape == output_shape and all(type(item) is int and item >= 1 for item in shape) and abi.get("dtype") == "torch.bfloat16" and boolean(abi.get("contiguous"), label + ".contiguous") is True, f"{label}: output ABI/cell shape drift")


def _validate_final_abi(value: object, *, final: bool, sequences: int, label: str) -> None:
    abi = obj(value, label)
    if not final:
        require(abi == {"present": False}, f"{label}: none contract must have no final state")
        return
    require(abi == {"present": True, "dtype": "torch.float32", "shape": [sequences, 12, 128, 128], "contiguous": True}, f"{label}: FP32 final-state ABI drift")


def validate_exact(value: object, final: bool, sequences: int, output_shape: list[int], label: str) -> None:
    record = obj(value, label)
    expected_keys = {"output_exact", "output_max_abs", "actual_output", "expected_output", "actual_final", "expected_final"}
    if final:
        expected_keys |= {"final_exact", "final_max_abs"}
    require(set(record) == expected_keys, f"{label}: exact ABI/equality field scope drift")
    require(boolean(record.get("output_exact"), label + ".output_exact") is True, f"{label}: output mismatch")
    require(finite(record.get("output_max_abs"), label + ".output_max_abs") == 0.0, f"{label}: nonzero output maximum")
    _validate_output_abi(record.get("actual_output"), output_shape, label + ".actual_output")
    _validate_output_abi(record.get("expected_output"), output_shape, label + ".expected_output")
    strict_equal(record["actual_output"], record["expected_output"], label + ".output ABI equality")
    _validate_final_abi(record.get("actual_final"), final=final, sequences=sequences, label=label + ".actual_final")
    _validate_final_abi(record.get("expected_final"), final=final, sequences=sequences, label=label + ".expected_final")
    strict_equal(record["actual_final"], record["expected_final"], label + ".final ABI equality")
    if final:
        require(boolean(record.get("final_exact"), label + ".final_exact") is True, f"{label}: final mismatch")
        require(finite(record.get("final_max_abs"), label + ".final_max_abs") == 0.0, f"{label}: nonzero final maximum")


def validate_spy(value: object, expected: dict[str, int], label: str) -> None:
    record = obj(value, label)
    require(set(record) == {"before", "after", "delta", "passed"}, f"{label}: registry spy schema drift")
    before = obj(record.get("before"), label + ".before")
    after = obj(record.get("after"), label + ".after")
    delta = obj(record.get("delta"), label + ".delta")
    require(set(before) == set(after) == set(delta) == set(expected), f"{label}: registry route counter scope drift")
    for key, expected_value in expected.items():
        before_value = integer(before.get(key), label + ".before." + key)
        after_value = integer(after.get(key), label + ".after." + key)
        delta_value = integer(delta.get(key), label + ".delta." + key)
        require(before_value >= 0 and after_value >= before_value and delta_value == after_value - before_value == expected_value, f"{label}: registry route counter/delta drift")
    require(boolean(record.get("passed"), label + ".passed") is True, f"{label}: spy did not pass")


def validate_physical_ledger_entry(value: object, expected: str, label: str) -> None:
    entry = obj(value, label)
    require(set(entry) == {"path", "sha256"}, f"{label}: identity keys drift")
    path = Path(string(entry["path"], label + ".path")).resolve(strict=True)
    observed = digest(entry["sha256"], label + ".sha256")
    require(observed == expected and _sha_bytes(path.read_bytes()) == expected, f"{label}: physical SHA drift")


def validate_source_ledger(value: object, label: str, runner_sha: str, analyzer_sha: str, shell_sha: str) -> dict[str, object]:
    source = obj(value, label)
    require(set(source) == {"auto_dispatch", "fla_backend", "runner", "analyzer", "protocol_shell", "supporting_helpers", "fla_sources", "passed"}, f"{label}: source ledger keys drift")
    for key, expected in (("auto_dispatch", AUTO_SHA), ("fla_backend", BACKEND_SHA), ("runner", runner_sha), ("analyzer", analyzer_sha), ("protocol_shell", shell_sha)):
        validate_physical_ledger_entry(source[key], expected, label + "." + key)
    helpers = obj(source["supporting_helpers"], label + ".supporting_helpers")
    require(set(helpers) == set(SUPPORTING_HELPER_SHA256), f"{label}: supporting helper scope drift")
    for key, expected in SUPPORTING_HELPER_SHA256.items():
        validate_physical_ledger_entry(helpers[key], expected, label + ".supporting_helpers." + key)
    fla_sources = obj(source["fla_sources"], label + ".fla_sources")
    require(set(fla_sources) == set(FLA_SOURCE_SHA256), f"{label}: FLA source scope drift")
    for relative, expected in FLA_SOURCE_SHA256.items():
        validate_physical_ledger_entry(fla_sources[relative], expected, label + ".fla_sources." + relative)
    require(boolean(source["passed"], label + ".passed") is True, f"{label}: source gate failed")
    return dict(source)


def _expected_loaded_module_paths(ledger: Mapping[str, Any], label: str) -> dict[str, Path]:
    helpers = obj(ledger.get("supporting_helpers"), label + ".supporting_helpers")
    fla_sources = obj(ledger.get("fla_sources"), label + ".fla_sources")
    return {
        "auto_dispatch": Path(string(obj(ledger.get("auto_dispatch"), label + ".auto_dispatch").get("path"), label + ".auto_dispatch.path")),
        "fla_backend": Path(string(obj(ledger.get("fla_backend"), label + ".fla_backend").get("path"), label + ".fla_backend.path")),
        "varlen_metadata": Path(string(obj(helpers.get("varlen_metadata"), label + ".varlen_metadata").get("path"), label + ".varlen_metadata.path")),
        "shared_seqcount": Path(string(obj(helpers.get("shared_seqcount"), label + ".shared_seqcount").get("path"), label + ".shared_seqcount.path")),
        "confirmation": Path(string(obj(helpers.get("confirmation"), label + ".confirmation").get("path"), label + ".confirmation.path")),
        "varlen_helper": Path(string(obj(helpers.get("varlen_helper"), label + ".varlen_helper").get("path"), label + ".varlen_helper.path")),
        "tail_helper": Path(string(obj(helpers.get("tail_helper"), label + ".tail_helper").get("path"), label + ".tail_helper.path")),
        "harness": Path(string(obj(helpers.get("harness"), label + ".harness").get("path"), label + ".harness.path")),
        "fla_ops_kda": Path(string(obj(fla_sources.get("fla/ops/kda/__init__.py"), label + ".fla_ops_kda").get("path"), label + ".fla_ops_kda.path")),
    }


def validate_loaded_module_identity(value: object, ledger: Mapping[str, Any], label: str) -> dict[str, object]:
    identity = obj(value, label)
    require(set(identity) == {"modules", "passed"} and boolean(identity.get("passed"), label + ".passed") is True, f"{label}: loaded-module identity scope drift")
    modules = obj(identity.get("modules"), label + ".modules")
    expected_paths = _expected_loaded_module_paths(ledger, label + ".expected")
    require(set(modules) == set(LOADED_RUNTIME_MODULES) == set(expected_paths), f"{label}: loaded-module scope drift")
    for name, module_name in LOADED_RUNTIME_MODULES.items():
        record = obj(modules.get(name), label + "." + name)
        expected_path = expected_paths[name].resolve(strict=True)
        actual_path = Path(string(record.get("path"), label + "." + name + ".path")).resolve(strict=True)
        require(set(record) == {"module", "path"} and record.get("module") == module_name and actual_path == expected_path, f"{label}: canonical loaded-module binding drift for {name}")
    return dict(identity)


def validate_runtime_source(value: object, label: str, runner_sha: str, analyzer_sha: str, shell_sha: str) -> dict[str, object]:
    source = obj(value, label)
    require(set(source) == {"ledger", "loaded_modules", "passed"} and boolean(source.get("passed"), label + ".passed") is True, f"{label}: runtime source wrapper scope drift")
    ledger = validate_source_ledger(source.get("ledger"), label + ".ledger", runner_sha, analyzer_sha, shell_sha)
    loaded = validate_loaded_module_identity(source.get("loaded_modules"), ledger, label + ".loaded_modules")
    return {"ledger": ledger, "loaded_modules": loaded, "passed": True}


def validate_bootstrap(value: object, label: str) -> None:
    record = obj(value, label)
    require(
        set(record) == {"source_ledger_mode", "stages", "heavy_runtime_import_after_clean_gate", "passed"}
        and record.get("source_ledger_mode") == "canonical_path_sha256_without_module_import"
        and arr(record.get("stages"), label + ".stages") == BOOTSTRAP_STAGES
        and boolean(record.get("heavy_runtime_import_after_clean_gate"), label + ".heavy_runtime_import_after_clean_gate") is True
        and boolean(record.get("passed"), label + ".passed") is True,
        f"{label}: pre-Torch source/clean/import bootstrap contract drift",
    )


def validate_map(value: object, label: str, *, raw_pid_snapshot: bool) -> dict[str, object]:
    record = obj(value, label)
    expected_keys = {"entries", "object_ids", "digest"} if raw_pid_snapshot else {"entries", "digest"}
    require(set(record) == expected_keys, f"{label}: map evidence keys drift")
    entries = obj(record["entries"], label + ".entries")
    ids = obj(record["object_ids"], label + ".object_ids") if raw_pid_snapshot else None
    require(set(entries) == {"fixed_batch", "fixed_single_batch", "varlen"} and (ids is None or set(ids) == set(entries)), f"{label}: map names drift")
    for name in entries:
        if ids is not None:
            require(integer(ids[name], label + ".object_ids." + name) > 0, f"{label}: map object identity invalid")
        rows = arr(entries[name], label + ".entries." + name)
        for index, row_value in enumerate(rows):
            row = obj(row_value, label + f".entries.{name}[{index}]")
            if name == "varlen":
                offsets = arr(row.get("offsets"), label + f".entries.{name}[{index}].offsets")
                require(set(row) == {"offsets", "contract", "variant"} and len(offsets) >= 2 and all(type(offset) is int for offset in offsets) and type(row.get("contract")) is str and type(row.get("variant")) is str, f"{label}: varlen map row type drift")
            else:
                require(set(row) == {"first", "contract", "variant"} and type(row.get("first")) is int and type(row.get("contract")) is str and type(row.get("variant")) is str, f"{label}: fixed map row type drift")
        require(rows == sorted(rows, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)), f"{label}: non-canonical map row ordering")
    required = {
        ("fixed_batch", "first", 2, "fp32_both"): "vshard4_p2",
        ("fixed_batch", "first", 5, "fp32_both"): "vshard2_p2",
        ("fixed_single_batch", "first", 8191, "none"): "vshard4_p2",
        ("varlen", "offsets", (0, 1, 2, 3, 4, 5, 12288), "fp32_both"): "vshard4_p2",
    }
    for (map_name, field, first, contract), variant in required.items():
        expected_first = list(first) if field == "offsets" else first
        matches = [row for row in arr(entries[map_name], label + ".entries." + map_name) if type(row) is dict and row.get(field) == expected_first and row.get("contract") == contract and row.get("variant") == variant]
        require(len(matches) == 1, f"{label}: required public map cell missing/duplicated")
    observed_digest = digest(record["digest"], label + ".digest")
    expected_digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    require(observed_digest == expected_digest, f"{label}: canonical map digest drift")
    return {"entries": entries, "digest": observed_digest}


def _require_content_map_binding(raw_content: object, audit_content: object, label: str) -> None:
    """Bind a portable raw-PID map view to an audit/chain map without IDs."""
    strict_equal(raw_content, audit_content, label)


def _validate_tree(value: object, label: str, *, commit: str, dirty: Mapping[str, str]) -> None:
    record = obj(value, label)
    require(set(record) == {"root", "commit", "tracked_status", "tracked_dirty_sha256", "passed"}, f"{label}: tree ledger keys drift")
    root = Path(string(record["root"], label + ".root")).resolve(strict=True)
    require(record["commit"] == commit, f"{label}: commit drift")
    expected_status = [f" M {relative}" for relative in dirty]
    require(arr(record["tracked_status"], label + ".tracked_status") == expected_status, f"{label}: tracked status drift")
    try:
        actual_commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        actual_status = subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"], check=True, capture_output=True, text=True).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AuditError(f"{label}: cannot physically reopen Git tree") from exc
    require(actual_commit == commit and actual_status == expected_status, f"{label}: physical Git tree drift")
    files = obj(record["tracked_dirty_sha256"], label + ".tracked_dirty_sha256")
    require(set(files) == set(dirty), f"{label}: dirty file scope drift")
    for relative, expected in dirty.items():
        require(digest(files[relative], label + ".dirty." + relative) == expected and _sha_bytes((root / relative).read_bytes()) == expected, f"{label}: dirty file SHA drift")
    require(boolean(record["passed"], label + ".passed") is True, f"{label}: tree gate failed")


def _require_identity_sections(identity: Mapping[str, Any], label: str) -> None:
    required = {"pre_torch_clean_gpu", "runtime", "patched_dirty_set", "reference_clean", "fla_clean", "pinned_reference_helper", "helper_runtime_globals", "registry", "registry_spies_restored"}
    require(set(identity) == required, f"{label}: dirty/tree/helper/registry identity scope drift")


def _require_runtime_fla_ledger(runtime: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    return obj(runtime.get("fla"), label + ".fla")


def _validate_shared_make_inputs_binding(
    value: object,
    label: str,
    *,
    phase: str,
    source_runtime: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate one typed record for the sole explicitly hydrated global.

    The record is deliberately narrow: it does *not* claim a static proof for
    other helper execution paths.  Its source/module/function identity is
    tied to the independently reopened runtime ledger, and object identities
    are compared only inside this one raw PID.
    """
    record = obj(value, label)
    require(set(record) == SHARED_BINDING_RECORD_FIELDS, f"{label}: shared binding field scope drift")
    require(record.get("phase") == phase, f"{label}: shared binding phase drift")
    require(
        record.get("module_label") == SHARED_SEQCOUNT_LABEL
        and record.get("module") == LOADED_RUNTIME_MODULES[SHARED_SEQCOUNT_LABEL]
        and record.get("function") == SHARED_MAKE_INPUTS
        and record.get("function_module") == LOADED_RUNTIME_MODULES[SHARED_SEQCOUNT_LABEL]
        and integer(record.get("module_object_id"), label + ".module_object_id") > 0
        and integer(record.get("function_object_id"), label + ".function_object_id") > 0
        and boolean(record.get("function_globals_is_module_dict"), label + ".function_globals_is_module_dict") is True
        and integer(record.get("sys_modules_torch_object_id"), label + ".sys_modules_torch_object_id") > 0
        and boolean(record.get("passed"), label + ".passed") is True,
        f"{label}: shared canonical module/function identity drift",
    )
    bound = phase != "pre_hydration"
    require(
        boolean(record.get("torch_global_present"), label + ".torch_global_present") is bound
        and boolean(record.get("torch_is_sys_modules_canonical"), label + ".torch_is_sys_modules_canonical") is bound,
        f"{label}: shared torch presence/canonicality phase drift",
    )
    if bound:
        require(
            integer(record.get("torch_object_id"), label + ".torch_object_id") > 0
            and record.get("torch_object_id") == record.get("sys_modules_torch_object_id"),
            f"{label}: shared torch object is not the recorded canonical sys.modules object",
        )
    else:
        require(record.get("torch_object_id") is None, f"{label}: pre-hydration shared torch must be absent")
    loaded = obj(source_runtime.get("loaded_modules"), label + ".source_runtime.loaded_modules")
    loaded_rows = obj(loaded.get("modules"), label + ".source_runtime.loaded_modules.modules")
    loaded_shared = obj(loaded_rows.get(SHARED_SEQCOUNT_LABEL), label + ".source_runtime.shared_seqcount")
    expected_path = Path(string(loaded_shared.get("path"), label + ".source_runtime.shared_seqcount.path")).resolve(strict=True)
    actual_path = Path(string(record.get("module_path"), label + ".module_path")).resolve(strict=True)
    require(actual_path == expected_path, f"{label}: shared module path detached from canonical loaded module")
    ledger = obj(source_runtime.get("ledger"), label + ".source_runtime.ledger")
    helpers = obj(ledger.get("supporting_helpers"), label + ".source_runtime.supporting_helpers")
    ledger_shared = obj(helpers.get(SHARED_SEQCOUNT_LABEL), label + ".source_runtime.ledger.shared_seqcount")
    require(actual_path == Path(string(ledger_shared.get("path"), label + ".source_runtime.ledger.shared_seqcount.path")).resolve(strict=True), f"{label}: shared module path detached from source ledger")
    return record


def validate_helper_runtime_globals(value: object, label: str, source_runtime: Mapping[str, Any]) -> None:
    evidence = obj(value, label)
    require(
        set(evidence) == {"pre_hydration", "bound_pre_workload", "post_workload", "passed"}
        and boolean(evidence.get("passed"), label + ".passed") is True,
        f"{label}: shared-hydration evidence scope drift",
    )
    pre = _validate_shared_make_inputs_binding(
        evidence.get("pre_hydration"), label + ".pre_hydration", phase="pre_hydration", source_runtime=source_runtime,
    )
    bound = _validate_shared_make_inputs_binding(
        evidence.get("bound_pre_workload"), label + ".bound_pre_workload", phase="bound_pre_workload", source_runtime=source_runtime,
    )
    post = _validate_shared_make_inputs_binding(
        evidence.get("post_workload"), label + ".post_workload", phase="post_workload", source_runtime=source_runtime,
    )
    for field in ("module_label", "module", "module_path", "module_object_id", "function", "function_module", "function_object_id", "function_globals_is_module_dict", "sys_modules_torch_object_id"):
        strict_equal(pre.get(field), bound.get(field), label + ".pre_to_bound." + field)
        strict_equal(bound.get(field), post.get(field), label + ".bound_to_post." + field)
    strict_equal(bound.get("torch_object_id"), post.get("torch_object_id"), label + ".bound_to_post.torch_object_id")


def validate_identity(value: object, label: str, source_runtime: Mapping[str, Any]) -> str:
    identity = obj(value, label)
    _require_identity_sections(identity, label)
    clean = obj(identity.get("pre_torch_clean_gpu"), label + ".pre_torch_clean_gpu")
    require(
        set(clean) == {"index", "uuid", "name", "compute_capability", "memory_used_mib", "compute_apps", "torch_modules_before_gate", "heavy_modules_before_gate", "passed"}
        and type(clean.get("index")) is str
        and type(clean.get("name")) is str
        and type(clean.get("compute_capability")) is str
        and boolean(clean.get("passed"), label + ".clean.passed") is True
        and integer(clean.get("memory_used_mib"), label + ".clean.memory") == 0
        and arr(clean.get("compute_apps"), label + ".clean.compute_apps") == []
        and arr(clean.get("torch_modules_before_gate"), label + ".clean.torch_modules_before_gate") == []
        and arr(clean.get("heavy_modules_before_gate"), label + ".clean.heavy_modules_before_gate") == [],
        f"{label}: earliest pre-Torch clean GPU/import gate failed",
    )
    uuid = string(clean.get("uuid"), label + ".clean.uuid")
    runtime = obj(identity.get("runtime"), label + ".runtime")
    device = obj(runtime.get("device"), label + ".runtime.device")
    require("B300" in string(device.get("name"), label + ".runtime.device.name").upper(), f"{label}: not B300")
    require(arr(device.get("capability"), label + ".runtime.device.capability") == [10, 3], f"{label}: not SM103")
    require(integer(device.get("multiprocessor_count"), label + ".runtime.device.sms") == 148, f"{label}: not 148SM")
    extension = obj(runtime.get("extension"), label + ".runtime.extension")
    extension_path = Path(string(extension.get("path"), label + ".extension.path")).resolve(strict=True)
    require(digest(extension.get("sha256"), label + ".extension.sha") == EXTENSION_SHA and _sha_bytes(extension_path.read_bytes()) == EXTENSION_SHA, f"{label}: extension SHA drift")
    source_trees = obj(runtime.get("source_trees"), label + ".runtime.source_trees")
    require(obj(source_trees.get("patched"), label + ".runtime.patched").get("commit") == PATCHED_COMMIT, f"{label}: runtime patched commit drift")
    require(obj(source_trees.get("reference"), label + ".runtime.reference").get("commit") == PATCHED_COMMIT, f"{label}: runtime reference commit drift")
    fla = _require_runtime_fla_ledger(runtime, label + ".runtime")
    require(fla.get("commit") == FLA_COMMIT and boolean(fla.get("tracked_status_clean"), label + ".runtime.fla.clean") is True, f"{label}: FLA runtime pin/clean drift")
    fla_root = Path(string(fla.get("root"), label + ".runtime.fla.root")).resolve(strict=True)
    files = obj(fla.get("files"), label + ".runtime.fla.files")
    require(set(files) == set(FLA_SOURCE_SHA256), f"{label}: runtime FLA file scope drift")
    for relative, expected in FLA_SOURCE_SHA256.items():
        require(digest(files[relative], label + ".runtime.fla." + relative) == expected and _sha_bytes((fla_root / relative).read_bytes()) == expected, f"{label}: runtime FLA file SHA drift")
    modules = obj(fla.get("loaded_modules"), label + ".runtime.fla.modules")
    expected_modules = {"fla", "fla.ops.backends", "fla.ops.kda", "fla.ops.kda.backends", "fla.ops.kda.backends.flash_kda", "fla.ops.kda.chunk"}
    require(set(modules) == expected_modules and all(type(modules[key]) is str and modules[key] for key in expected_modules), f"{label}: loaded FLA module identity drift")
    expected_module_paths = {
        "fla": "fla/__init__.py", "fla.ops.backends": "fla/ops/backends/__init__.py", "fla.ops.kda": "fla/ops/kda/__init__.py",
        "fla.ops.kda.backends": "fla/ops/kda/backends/__init__.py", "fla.ops.kda.backends.flash_kda": "fla/ops/kda/backends/flash_kda.py", "fla.ops.kda.chunk": "fla/ops/kda/chunk.py",
    }
    for module, relative in expected_module_paths.items():
        require(Path(modules[module]).resolve(strict=True) == (fla_root / relative).resolve(strict=True), f"{label}: loaded FLA module path drift: {module}")
    public_callables = obj(fla.get("public_callables"), label + ".runtime.fla.public_callables")
    public_chunk = obj(public_callables.get("fla.ops.kda.chunk_kda"), label + ".runtime.fla.public_chunk")
    require(
        boolean(public_chunk.get("implementation_identity_match"), label + ".runtime.fla.public_chunk.identity") is True
        and public_chunk.get("module") == "fla.ops.kda.chunk"
        and public_chunk.get("qualname") == "chunk_kda"
        and Path(string(public_chunk.get("source_path"), label + ".runtime.fla.public_chunk.path")).resolve(strict=True) == (fla_root / "fla/ops/kda/chunk.py").resolve(strict=True)
        and boolean(public_chunk.get("passed"), label + ".runtime.fla.public_chunk.passed") is True,
        f"{label}: public FLA callable identity drift",
    )
    _validate_tree(identity.get("patched_dirty_set"), label + ".patched_dirty_set", commit=PATCHED_COMMIT, dirty=PATCHED_DIRTY)
    _validate_tree(identity.get("reference_clean"), label + ".reference_clean", commit=PATCHED_COMMIT, dirty={})
    _validate_tree(identity.get("fla_clean"), label + ".fla_clean", commit=FLA_COMMIT, dirty={})
    helper = obj(identity.get("pinned_reference_helper"), label + ".helper")
    helper_id, proof = obj(helper.get("identity"), label + ".helper.identity"), obj(helper.get("load_proof"), label + ".helper.proof")
    require(helper_id == {"path": HELPER_PATH, "sha256": HELPER_SHA} and _sha_bytes(Path(HELPER_PATH).resolve(strict=True).read_bytes()) == HELPER_SHA, f"{label}: helper identity drift")
    require(proof.get("path") == HELPER_PATH and proof.get("sha256") == HELPER_SHA and proof.get("intercepted_names") == ["sigmoid_ext"] and boolean(proof.get("no_build"), label + ".helper.no_build") is True, f"{label}: helper no-build proof drift")
    validate_helper_runtime_globals(identity.get("helper_runtime_globals"), label + ".helper_runtime_globals", source_runtime)
    registry = obj(identity.get("registry"), label + ".registry")
    require(registry.get("public_callable") == "fla.ops.kda.chunk_kda" and boolean(registry.get("test_only_route_installed"), label + ".registry.test_only") is False, f"{label}: registry provenance drift")
    snapshot = arr(registry.get("snapshot"), label + ".registry.snapshot")
    backend_rows = [obj(row, label + ".registry.snapshot[]") for row in snapshot]
    c1_rows = [index for index, row in enumerate(backend_rows) if row.get("backend_type") == "c1_b300_flash_kda"]
    pinned_rows = [index for index, row in enumerate(backend_rows) if row.get("backend_type") == "flash_kda"]
    require(len(c1_rows) == len(pinned_rows) == 1 and c1_rows[0] < pinned_rows[0], f"{label}: public FLA backend order drift")
    for row in backend_rows:
        require(set(row) == {"backend_type", "priority", "id"} and type(row["backend_type"]) is str and type(row["priority"]) is int and integer(row["id"], label + ".registry.id") > 0, f"{label}: registry backend snapshot type drift")
    restored = obj(identity.get("registry_spies_restored"), label + ".restored")
    require(all(boolean(restored.get(key), label + ".restored." + key) is True for key in ("c1", "pinned", "passed")), f"{label}: spies not restored")
    return uuid


def _immutable_fields(*, varlen: bool, final: bool) -> list[str]:
    fields = {"q", "k", "v", "g", "beta", "A_log", "dt_bias"}
    if varlen:
        fields |= {"cu_seqlens", "cu_seqlens_cpu"}
    if final:
        fields.add("initial_state")
    return sorted(fields)


def _validate_immutability_record(value: object, expected_fields: list[str], label: str) -> None:
    facts = obj(value, label)
    require(
        set(facts) == {"input_immutability_exact", "fields"}
        and boolean(facts.get("input_immutability_exact"), label + ".exact") is True
        and arr(facts.get("fields"), label + ".fields") == expected_fields,
        f"{label}: input immutable field set drift",
    )


def validate_immutability(value: object, label: str, expected_paths: set[str], *, varlen: bool, final: bool) -> None:
    records = obj(value, label)
    require(bool(records), f"{label}: no input immutability evidence")
    require(set(records) == expected_paths, f"{label}: immutable call-path scope drift")
    fields = _immutable_fields(varlen=varlen, final=final)
    for name, record in records.items():
        _validate_immutability_record(record, fields, label + "." + name)


def validate_torch_reference_immutability(value: object, *, varlen: bool, final: bool, label: str) -> None:
    _validate_immutability_record(value, _immutable_fields(varlen=varlen, final=final), label)


def validate_fixed_handoff(value: object, label: str) -> None:
    facts = obj(value, label)
    require(facts == {"applicable": False, "reason": "fixed_batch_has_no_CPU_varlen_descriptor", "passed": True}, f"{label}: fixed handoff/cache contract drift")


def validate_varlen_positive_handoff(value: object, label: str) -> None:
    facts = obj(value, label)
    require(
        set(facts) == {"public_prepare", "cache_stats", "direct_canonical_cache_hit", "public_canonical_cache_hit", "direct_miss_to_public_hit", "passed"}
        and boolean(facts["direct_canonical_cache_hit"], label + ".direct_cache_hit") is False
        and boolean(facts["public_canonical_cache_hit"], label + ".public_cache_hit") is True
        and boolean(facts["direct_miss_to_public_hit"], label + ".miss_to_hit") is True
        and boolean(facts["passed"], label + ".passed") is True,
        f"{label}: varlen handoff/cache keys or miss-to-hit provenance drift",
    )
    prepare = obj(facts["public_prepare"], label + ".public_prepare")
    c1, pinned = obj(prepare.get("c1"), label + ".prepare.c1"), obj(prepare.get("pinned"), label + ".prepare.pinned")
    require(
        set(prepare) == {"c1", "pinned"}
        and set(c1) == {"prepare_delta", "prepare_calls_total", "passed"}
        and integer(c1.get("prepare_delta"), label + ".prepare.c1.delta") == 1
        and integer(c1.get("prepare_calls_total"), label + ".prepare.c1.calls") == 1
        and boolean(c1.get("passed"), label + ".prepare.c1.passed") is True
        and set(pinned) == {"prepare_delta", "prepare_calls_total", "c1_immutability", "pinned_immutability", "passed"}
        and integer(pinned.get("prepare_delta"), label + ".prepare.pinned.delta") == 0
        and integer(pinned.get("prepare_calls_total"), label + ".prepare.pinned.calls") == 1
        and boolean(pinned.get("passed"), label + ".prepare.pinned.passed") is True,
        f"{label}: varlen public-prepare nested schema/value drift",
    )
    fields = _immutable_fields(varlen=True, final=True)
    _validate_immutability_record(pinned.get("c1_immutability"), fields, label + ".prepare.pinned.c1_immutability")
    _validate_immutability_record(pinned.get("pinned_immutability"), fields, label + ".prepare.pinned.pinned_immutability")
    _validate_varlen_cache_stats(facts["cache_stats"], label + ".cache_stats", mode="positive")


def validate_accepting_verifiers(value: object, label: str) -> None:
    verifiers = obj(value, label)
    require(set(verifiers) == {"c1", "pinned"}, f"{label}: verifier backend scope drift")
    for name in ("c1", "pinned"):
        record = obj(verifiers[name], label + "." + name)
        require(set(record) == {"accepted", "reason"} and boolean(record.get("accepted"), label + "." + name + ".accepted") is True and (record.get("reason") is None or type(record.get("reason")) is str), f"{label}: accepting verifier schema drift")


def _validate_varlen_cache_stats(value: object, label: str, *, mode: str) -> None:
    cache = obj(value, label)
    require(set(cache) == VARLEN_CACHE_FIELDS, f"{label}: varlen cache schema drift")
    for key, count in cache.items():
        observed = integer(count, label + "." + key)
        require(observed >= 0, f"{label}: negative cache count")
    if mode == "zero":
        require(all(cache[key] == 0 for key in VARLEN_CACHE_FIELDS), f"{label}: cache must be exactly clear")
    elif mode == "positive":
        require(cache == {"entries": 1, "hits": 1, "misses": 1, "capture_miss_rejections": 0, "capture_hit_rejections": 0}, f"{label}: varlen cache miss→hit accounting drift")
    elif mode != "positive":
        raise AuditError(f"{label}: unknown cache validation mode")


def _validate_accelerated_baseline_evidence(record: Mapping[str, Any], label: str) -> None:
    entry = obj(record.get("c1_backend_entry"), label + ".c1_backend_entry")
    require(entry == {"direct": True, "public": True, "passed": True}, f"{label}: baseline did not enter C1 backend through both required paths")
    require(arr(record.get("accelerated_variants_forbidden"), label + ".accelerated_variants_forbidden") == ["vshard2_p2", "vshard4_p2"], f"{label}: accelerated forbid set drift")


def _validate_first_path_counts(value: object, label: str) -> None:
    counts = obj(value, label)
    require(
        set(counts) == {"pinned_public", "c1_public"}
        and integer(counts.get("pinned_public"), label + ".pinned_public") == SAMPLES // 2
        and integer(counts.get("c1_public"), label + ".c1_public") == SAMPLES // 2,
        f"{label}: first-path exact integer counts drift",
    )


def validate_controls(value: object, label: str) -> None:
    controls = obj(value, label)
    require(set(controls) == set(NEGATIVES), f"{label}: negative control scope drift")
    for name, reason in NEGATIVES.items():
        record = obj(controls[name], label + "." + name)
        expected_control_keys = (
            {"expected_variant", "expected_reason", "verifier", "direct_pinned_vs_reference", "direct_c1_vs_pinned", "public_c1_vs_pinned", "public_pinned_vs_reference", "direct_c1_vs_reference", "public_c1_vs_reference", "direct_decision", "public_decision", "public_c1_spy", "public_pinned_spy", "torch_reference_immutability", "c1_backend_entry", "accelerated_variants_forbidden", "no_accelerated_variant_selected_or_launched", "input_immutability", "handoff_cache", "passed"}
            if name != "adjacent_offsets_fp32_both"
            else {"expected_variant", "expected_reason", "c1_verifier", "public_c1_spy", "public_pinned_spy", "direct_pinned_vs_reference", "public_c1_fallback_vs_pinned", "public_c1_fallback_vs_reference", "public_pinned_vs_reference", "input_immutability", "torch_reference_immutability", "handoff_cache", "no_accelerated_variant_selected_or_launched", "passed"}
        )
        require(set(record) == expected_control_keys, f"{label}.{name}: negative control schema field scope drift")
        require(record.get("expected_variant") == "baseline" and record.get("expected_reason") == reason and boolean(record.get("no_accelerated_variant_selected_or_launched"), label + "." + name + ".no_accel") is True and boolean(record.get("passed"), label + "." + name + ".passed") is True, f"{label}.{name}: negative metadata drift")
        exact_keys = [key for key in record if "_vs_" in key]
        expected_exact = (
            {"direct_pinned_vs_reference", "direct_c1_vs_pinned", "public_c1_vs_pinned", "public_pinned_vs_reference", "direct_c1_vs_reference", "public_c1_vs_reference"}
            if name != "adjacent_offsets_fp32_both"
            else {"direct_pinned_vs_reference", "public_c1_fallback_vs_pinned", "public_c1_fallback_vs_reference", "public_pinned_vs_reference"}
        )
        require(set(exact_keys) == expected_exact, f"{label}.{name}: public correctness comparison scope drift")
        final = name != "b7_none"
        sequences = CELL_SEQUENCES[name]
        for key in exact_keys:
            validate_exact(record[key], final, sequences, CELL_OUTPUT_SHAPES[name], label + "." + name + "." + key)
        if name == "adjacent_offsets_fp32_both":
            verifier = obj(record.get("c1_verifier"), label + "." + name + ".c1_verifier")
            require(
                set(verifier) == {"call_count", "q_tensor_identity", "gpu_offsets_tensor_identity", "cpu_offsets_tensor_identity", "accepted", "reason", "issuer_call_count", "issuer_cpu_offsets_tensor_identity", "certified_offsets", "issuer_spy_restored", "verifier_spy_restored", "passed"}
                and integer(verifier.get("call_count"), label + "." + name + ".verifier.calls") == 1
                and integer(verifier.get("issuer_call_count"), label + "." + name + ".verifier.issuer_calls") == 1
                and all(boolean(verifier.get(field), label + "." + name + ".verifier." + field) is True for field in ("q_tensor_identity", "gpu_offsets_tensor_identity", "cpu_offsets_tensor_identity", "issuer_cpu_offsets_tensor_identity", "issuer_spy_restored", "verifier_spy_restored", "passed"))
                and boolean(verifier.get("accepted"), label + "." + name + ".accepted") is False
                and verifier.get("reason") == "C1 packed-varlen preflight rejected: varlen_offsets_not_whitelisted"
                and verifier.get("certified_offsets") == [0, 1, 2, 3, 4, 6, 12288],
                f"{label}.{name}: exact verifier rejection/current-offset evidence drift",
            )
            validate_spy(record.get("public_c1_spy"), {"c1": 0, "pinned": 1}, label + "." + name + ".c1_spy")
            validate_spy(record.get("public_pinned_spy"), {"c1": 0, "pinned": 1}, label + "." + name + ".pinned_spy")
            validate_immutability(record.get("input_immutability"), label + "." + name + ".immutability", {"direct_pinned", "public_c1_fallback", "public_pinned"}, varlen=True, final=True)
            validate_torch_reference_immutability(record.get("torch_reference_immutability"), varlen=True, final=True, label=label + "." + name + ".torch_reference")
            handoff = obj(record.get("handoff_cache"), label + "." + name + ".handoff")
            require(
                set(handoff) == {"clear_handoff_api", "metadata_clear_api", "cache_before_clear", "cache_after_clear", "handoff_empty_after_clear", "handoff_empty_after_public", "cache_after_cleanup", "issuer", "verifier", "last_decision_read", "spies_restored", "passed"}
                and handoff.get("clear_handoff_api") == "C1B300FlashKDABackend._clear_varlen_handoff"
                and handoff.get("metadata_clear_api") == "varlen_metadata.clear_cache"
                and all(boolean(handoff.get(field), label + "." + name + ".handoff." + field) is True for field in ("handoff_empty_after_clear", "handoff_empty_after_public", "spies_restored", "passed"))
                and handoff.get("last_decision_read") is False,
                f"{label}.{name}: stale-decision/handoff restoration drift",
            )
            _validate_varlen_cache_stats(handoff.get("cache_before_clear"), label + "." + name + ".cache_before_clear", mode="zero")
            _validate_varlen_cache_stats(handoff.get("cache_after_clear"), label + "." + name + ".cache_after_clear", mode="zero")
            _validate_varlen_cache_stats(handoff.get("cache_after_cleanup"), label + "." + name + ".cache_after_cleanup", mode="zero")
            issuer = obj(handoff.get("issuer"), label + "." + name + ".issuer")
            require(issuer == {"cpu_tensor_identity": True, "certified_offsets": [0, 1, 2, 3, 4, 6, 12288]} and handoff.get("verifier") == verifier, f"{label}.{name}: issuer/verifier current-offset proof drift")
        else:
            validate_accepting_verifiers(record.get("verifier"), label + "." + name + ".verifier")
            validate_baseline_decision(record.get("direct_decision"), reason, label + "." + name + ".direct")
            validate_baseline_decision(record.get("public_decision"), reason, label + "." + name + ".public")
            validate_spy(record.get("public_c1_spy"), {"c1": 1, "pinned": 0}, label + "." + name + ".c1_spy")
            validate_spy(record.get("public_pinned_spy"), {"c1": 0, "pinned": 1}, label + "." + name + ".pinned_spy")
            _validate_accelerated_baseline_evidence(record, label + "." + name)
            validate_immutability(record.get("input_immutability"), label + "." + name + ".immutability", {"direct_pinned", "direct_c1", "public_c1", "public_pinned"}, varlen=False, final=final)
            validate_torch_reference_immutability(record.get("torch_reference_immutability"), varlen=False, final=final, label=label + "." + name + ".torch_reference")
            validate_fixed_handoff(record.get("handoff_cache"), label + "." + name + ".handoff")


def validate_performance(value: object, label: str) -> None:
    rounds = arr(value, label)
    require(len(rounds) == 1, f"{label}: exactly one pre-registered round required")
    round0 = obj(rounds[0], label + "[0]")
    require(
        set(round0) == {"round_index", "event_contract", "warmup", "samples_per_path", "first_path_counts", "raw_samples_ms", "paths", "pinned_over_c1_by_percentile", "regression_gate", "passed"}
        and integer(round0.get("round_index"), label + ".round_index") == 0
        and round0.get("event_contract") == "one uninstrumented real FLA public call per CUDA-event sample; spy and source checks excluded"
        and integer(round0.get("warmup"), label + ".warmup") == WARMUP
        and integer(round0.get("samples_per_path"), label + ".samples") == SAMPLES
        and round0.get("regression_gate") == "pinned/C1 > 1 at P50/P95/P99 (non-release gate)",
        f"{label}: timing metadata/round contract drift",
    )
    _validate_first_path_counts(round0.get("first_path_counts"), label + ".first_path_counts")
    raw, paths, ratios = obj(round0.get("raw_samples_ms"), label + ".raw"), obj(round0.get("paths"), label + ".paths"), obj(round0.get("pinned_over_c1_by_percentile"), label + ".ratios")
    require(set(raw) == {"pinned_public", "c1_public"} and set(paths) == set(raw), f"{label}: timing paths drift")
    recomputed: dict[str, dict[str, float | int]] = {}
    for path in raw:
        samples = [floating(item, label + ".raw." + path) for item in arr(raw[path], label + ".raw." + path)]
        require(len(samples) == SAMPLES and all(item > 0 for item in samples), f"{label}: invalid raw samples")
        recomputed[path] = summary(samples)
        observed = obj(paths[path], label + ".paths." + path)
        require(set(observed) == {"samples", "mean_ms", "p50_ms", "p95_ms", "p99_ms"}, f"{label}: timing summary field scope drift")
        for key, expected in recomputed[path].items():
            require((integer(observed.get(key), label + "." + path + "." + key) == expected) if key == "samples" else close(floating(observed.get(key), label + "." + path + "." + key), float(expected)), f"{label}: summary drift for {path}/{key}")
    for percentile_name in PERCENTILES:
        expected = float(recomputed["pinned_public"][percentile_name + "_ms"]) / float(recomputed["c1_public"][percentile_name + "_ms"])
        require(close(floating(ratios.get(percentile_name), label + ".ratio." + percentile_name), expected) and expected > 1.0, f"{label}: non-release regression gate failed at {percentile_name}")
    require(set(ratios) == set(PERCENTILES), f"{label}: percentile scope drift")
    require(boolean(round0.get("passed"), label + ".passed") is True, f"{label}: timing failed")


def _validate_main_envelope(raw: Mapping[str, Any], label: str) -> None:
    require(
        set(raw) == MAIN_SCHEMA_FIELDS
        and raw.get("purpose") == "v5 public-registry cross-map read-only regression; no production source or map mutation"
        and integer(raw.get("schema_version"), label + ".schema") == SCHEMA_VERSION,
        f"{label}: v4 raw main schema/purpose/field scope drift",
    )


def validate_main(raw: Mapping[str, Any], *, allocation: str, process: int, runner_sha: str, analyzer_sha: str, shell_sha: str, label: str) -> dict[str, object]:
    _validate_main_envelope(raw, label)
    require(raw.get("allocation_id") == allocation and integer(raw.get("process_index"), label + ".process") == process, f"{label}: allocation/process drift")
    alloc = obj(raw.get("allocation"), label + ".allocation")
    job = string(alloc.get("slurm_job_id"), label + ".job")
    require(job.isdecimal() and int(job) > 0, f"{label}: invalid Slurm job")
    process_record = obj(raw.get("process"), label + ".process_record")
    pid = integer(process_record.get("pid"), label + ".pid")
    require(pid > 0 and boolean(process_record.get("fresh_python_process_required"), label + ".fresh_python_process_required") is True, f"{label}: fresh PID proof drift")
    source_pre_torch = validate_source_ledger(raw.get("source_pre_torch"), label + ".source_pre_torch", runner_sha, analyzer_sha, shell_sha)
    validate_bootstrap(raw.get("bootstrap"), label + ".bootstrap")
    source_pre = validate_runtime_source(raw.get("source_pre"), label + ".source_pre", runner_sha, analyzer_sha, shell_sha)
    source_post = validate_runtime_source(raw.get("source_post"), label + ".source_post", runner_sha, analyzer_sha, shell_sha)
    strict_equal(source_pre_torch, source_pre["ledger"], label + ".pre-Torch source to loaded source binding")
    strict_equal(source_pre, source_post, label + ".source read-only")
    map_pre, map_post = validate_map(raw.get("map_pre"), label + ".map_pre", raw_pid_snapshot=True), validate_map(raw.get("map_post"), label + ".map_post", raw_pid_snapshot=True)
    strict_equal(map_pre, map_post, label + ".map read-only")
    raw_ids_pre = obj(raw.get("map_pre"), label + ".map_pre")["object_ids"]
    raw_ids_post = obj(raw.get("map_post"), label + ".map_post")["object_ids"]
    strict_equal(raw_ids_pre, raw_ids_post, label + ".map object identity within PID")
    readonly = obj(raw.get("map_readonly"), label + ".map_readonly")
    require(readonly == {"content_unchanged": True, "object_ids_unchanged_within_raw_pid": True, "passed": True}, f"{label}: raw map readonly declaration drift")
    uuid = validate_identity(raw.get("identity"), label + ".identity", source_pre)
    validate_controls(raw.get("public_registry_negative_controls"), label + ".controls")
    positives = obj(raw.get("positive_cells"), label + ".positives")
    performance = obj(raw.get("performance"), label + ".performance")
    require(set(positives) == set(POSITIVES) and set(performance) == set(POSITIVES), f"{label}: positive/performance scope drift")
    for name, (variant, reason, final) in POSITIVES.items():
        record = obj(positives[name], label + "." + name)
        expected_positive_keys = (
            {"expected_variant", "expected_reason", "verifier", "pinned_direct_vs_reference", "direct_c1_vs_pinned", "public_c1_vs_pinned", "public_pinned_vs_reference", "direct_c1_vs_reference", "public_c1_vs_reference", "public_c1_spy", "public_pinned_spy", "direct_decision", "public_decision", "torch_reference_immutability", "input_immutability", "handoff_cache", "passed"}
            if name not in VARLEN_OFFSETS
            else {"expected_variant", "expected_reason", "verifier", "pinned_vs_torch_ref", "direct_c1_vs_pinned", "public_vs_pinned", "public_pinned_vs_torch_ref", "direct_c1_vs_torch_ref", "public_vs_torch_ref", "public_c1_spy", "public_pinned_spy", "direct_decision", "public_decision", "input_immutability_by_path", "torch_reference_immutability", "handoff_cache", "passed"}
        )
        require(set(record) == expected_positive_keys, f"{label}.{name}: positive schema field scope drift")
        require(record.get("expected_variant") == variant and record.get("expected_reason", reason) == reason, f"{label}.{name}: expected cell drift")
        offsets = VARLEN_OFFSETS.get(name)
        validate_positive_decision(record.get("direct_decision"), variant, reason, offsets=offsets, cache_hit=False if offsets is not None else None, label=label + "." + name + ".direct")
        validate_positive_decision(record.get("public_decision"), variant, reason, offsets=offsets, cache_hit=True if offsets is not None else None, label=label + "." + name + ".public")
        validate_spy(record.get("public_c1_spy"), {"c1": 1, "pinned": 0}, label + "." + name + ".c1_spy")
        validate_spy(record.get("public_pinned_spy"), {"c1": 0, "pinned": 1}, label + "." + name + ".pinned_spy")
        validate_accepting_verifiers(record.get("verifier"), label + "." + name + ".verifier")
        exact_keys = [key for key in record if "_vs_" in key]
        expected_exact = (
            {"pinned_vs_torch_ref", "direct_c1_vs_pinned", "public_vs_pinned", "public_pinned_vs_torch_ref", "direct_c1_vs_torch_ref", "public_vs_torch_ref"}
            if offsets is not None
            else {"pinned_direct_vs_reference", "direct_c1_vs_pinned", "public_c1_vs_pinned", "public_pinned_vs_reference", "direct_c1_vs_reference", "public_c1_vs_reference"}
        )
        require(set(exact_keys) == expected_exact, f"{label}.{name}: fixed/varlen exact comparison ABI scope drift")
        for key in exact_keys:
            validate_exact(record[key], final, CELL_SEQUENCES[name], CELL_OUTPUT_SHAPES[name], label + "." + name + "." + key)
        if name.startswith("varlen_"):
            validate_immutability(record.get("input_immutability_by_path"), label + "." + name + ".immutability", {"pinned", "direct_c1", "public_c1", "public_pinned"}, varlen=True, final=final)
            validate_torch_reference_immutability(record.get("torch_reference_immutability"), varlen=True, final=final, label=label + "." + name + ".torch_reference")
            validate_varlen_positive_handoff(record.get("handoff_cache"), label + "." + name + ".handoff")
        else:
            validate_immutability(record.get("input_immutability"), label + "." + name + ".immutability", {"direct_pinned", "direct_c1", "public_c1", "public_pinned"}, varlen=False, final=final)
            validate_torch_reference_immutability(record.get("torch_reference_immutability"), varlen=False, final=final, label=label + "." + name + ".torch_reference")
            validate_fixed_handoff(record.get("handoff_cache"), label + "." + name + ".handoff")
        require(boolean(record.get("passed"), label + "." + name + ".passed") is True, f"{label}.{name}: correctness did not pass")
        validate_performance(performance[name], label + ".performance." + name)
    require(boolean(raw.get("complete"), label + ".complete") is True, f"{label}: incomplete")
    return {"allocation": allocation, "job": job, "process": process, "pid": pid, "uuid": uuid, "source": source_pre, "map": map_pre}


def _require_distinct_pids(pids: set[int], label: str) -> None:
    require(len(pids) == 2 and all(type(pid) is int and pid > 0 for pid in pids), f"{label}: allocation must contain two distinct fresh PIDs")


def _validate_a1_binding(binding: object, a1_link: Mapping[str, str], a1_job: str, label: str) -> None:
    record = obj(binding, label)
    strict_equal(record.get("a1_audit"), a1_link, label + ".exact A1 path/SHA")
    require(boolean(record.get("different_slurm_job"), label + ".different_slurm_job") is True, f"{label}: different-job declaration drift")
    require(record.get("a1_job") == a1_job, f"{label}: A1 job binding drift")


def _reopen_a2_prerequisite(
    a1_audit: Path | None,
    expected_a1_audit_sha256: str | None,
    current_job: str,
    runner_sha: str,
    analyzer_sha: str,
    shell_sha: str,
) -> dict[str, object]:
    """Physically reopen and bind A1 before accepting an A2 allocation.

    This always invokes the strict allocation validator before accessing the
    A1 allocation object; no production caller can opt out of that gate.
    """
    require(a1_audit is not None and expected_a1_audit_sha256 is not None, "A2 requires externally frozen A1 audit path/SHA")
    a1, a1_link = read_once(a1_audit, expected_a1_audit_sha256, "A1 prerequisite")
    validate_allocation_record(a1, "A1", runner_sha, analyzer_sha, shell_sha, "A1 prerequisite")
    a1_allocation = obj(a1.get("allocation"), "A1 prerequisite.allocation")
    a1_job = string(a1_allocation.get("slurm_job_id"), "A1 prerequisite.job")
    require(a1_job != current_job, "A2 must use a distinct Slurm job")
    return {"a1_audit": a1_link, "a1_job": a1_job, "different_slurm_job": True}


def allocation(args: argparse.Namespace) -> None:
    raw0, link0 = read_once(args.main0, args.main0_sha256, "main0")
    raw1, link1 = read_once(args.main1, args.main1_sha256, "main1")
    current_job = string(args.current_slurm_job_id, "current_slurm_job_id")
    require(current_job.isdecimal() and int(current_job) > 0, "current Slurm job invalid")
    records = [validate_main(raw0, allocation=args.allocation_id, process=0, runner_sha=args.expected_runner_sha256, analyzer_sha=args.expected_analyzer_sha256, shell_sha=args.expected_protocol_shell_sha256, label="main0"), validate_main(raw1, allocation=args.allocation_id, process=1, runner_sha=args.expected_runner_sha256, analyzer_sha=args.expected_analyzer_sha256, shell_sha=args.expected_protocol_shell_sha256, label="main1")]
    require(all(record["job"] == current_job for record in records), "raw main Slurm jobs do not match current allocation")
    _require_distinct_pids({record["pid"] for record in records}, "allocation")
    strict_equal(records[0]["source"], records[1]["source"], "allocation source identity")
    _require_content_map_binding(records[0]["map"], records[1]["map"], "allocation raw content-map identity")
    binding: dict[str, object] | None = None
    if args.allocation_id == "A2":
        binding = _reopen_a2_prerequisite(
            args.a1_audit,
            args.expected_a1_audit_sha256,
            current_job,
            args.expected_runner_sha256,
            args.expected_analyzer_sha256,
            args.expected_protocol_shell_sha256,
        )
    elif args.a1_audit is not None or args.expected_a1_audit_sha256 is not None:
        raise AuditError("A1 must not receive A1 prerequisite arguments")
    require(records[0]["uuid"] == records[1]["uuid"], "allocation raw GPU UUID drift")
    output = {"schema_version": SCHEMA_VERSION, "purpose": "v5 cross-map allocation audit", "allocation": {"id": args.allocation_id, "slurm_job_id": current_job, "gpu_uuid": records[0]["uuid"]}, "raw_inputs": [link0, link1], "source": records[0]["source"], "content_map": records[0]["map"], "a1_binding": binding, "allocation_passed": True}
    write(args.json, output); print(f"wrote {args.json}")


def validate_allocation_record(record: Mapping[str, Any], expected_allocation: str, runner_sha: str, analyzer_sha: str, shell_sha: str, label: str) -> None:
    require(integer(record.get("schema_version"), label + ".schema") == SCHEMA_VERSION and record.get("purpose") == "v5 cross-map allocation audit", f"{label}: allocation schema/purpose drift")
    allocation_record = obj(record.get("allocation"), label + ".allocation")
    require(allocation_record.get("id") == expected_allocation, f"{label}: allocation ID drift")
    job = string(allocation_record.get("slurm_job_id"), label + ".job")
    require(job.isdecimal() and int(job) > 0, f"{label}: allocation job drift")
    uuid = string(allocation_record.get("gpu_uuid"), label + ".gpu_uuid")
    source = validate_runtime_source(record.get("source"), label + ".source", runner_sha, analyzer_sha, shell_sha)
    content_map = validate_map(record.get("content_map"), label + ".content_map", raw_pid_snapshot=False)
    links = arr(record.get("raw_inputs"), label + ".raw_inputs")
    require(len(links) == 2, f"{label}: must reference two fresh PIDs")
    seen: set[int] = set(); pids: set[int] = set()
    for index, link in enumerate(links):
        link_obj = obj(link, label + f".raw[{index}]")
        raw, _ = read_once(Path(string(link_obj.get("path"), label + ".raw.path")), digest(link_obj.get("sha256"), label + ".raw.sha"), label + f".raw[{index}]")
        actual = validate_main(raw, allocation=expected_allocation, process=index, runner_sha=runner_sha, analyzer_sha=analyzer_sha, shell_sha=shell_sha, label=label + f".reopened[{index}]")
        require(actual["job"] == job, f"{label}: raw job drift")
        require(actual["uuid"] == uuid, f"{label}: raw GPU UUID drift")
        seen.add(actual["process"])
        pids.add(actual["pid"])
        strict_equal(actual["source"], source, label + ".reopened source")
        _require_content_map_binding(actual["map"], content_map, label + ".reopened content-map binding")
    require(seen == {0, 1}, f"{label}: process set drift")
    _require_distinct_pids(pids, label + ".reopened")
    require(boolean(record.get("allocation_passed"), label + ".passed") is True, f"{label}: allocation not passed")


def freeze(args: argparse.Namespace) -> None:
    a1, a1_link = read_once(args.a1_audit, args.expected_a1_sha256, "A1 audit")
    a2, a2_link = read_once(args.a2_audit, args.expected_a2_sha256, "A2 audit")
    validate_allocation_record(a1, "A1", args.expected_runner_sha256, args.expected_analyzer_sha256, args.expected_protocol_shell_sha256, "A1")
    validate_allocation_record(a2, "A2", args.expected_runner_sha256, args.expected_analyzer_sha256, args.expected_protocol_shell_sha256, "A2")
    a1_allocation = obj(a1.get("allocation"), "A1.allocation")
    a2_allocation = obj(a2.get("allocation"), "A2.allocation")
    a1_job = string(a1_allocation.get("slurm_job_id"), "A1.job")
    a2_job = string(a2_allocation.get("slurm_job_id"), "A2.job")
    require(a1_job != a2_job, "A1/A2 must have distinct Slurm jobs")
    _validate_a1_binding(a2.get("a1_binding"), a1_link, a1_job, "A2.a1_binding")
    strict_equal(a1.get("source"), a2.get("source"), "A1/A2 source")
    _require_content_map_binding(a1.get("content_map"), a2.get("content_map"), "A1/A2 content-map identity")
    a1_uuid = string(a1_allocation.get("gpu_uuid"), "A1.gpu_uuid")
    a2_uuid = string(a2_allocation.get("gpu_uuid"), "A2.gpu_uuid")
    output = {"schema_version": SCHEMA_VERSION, "purpose": "v5 cross-map A1/A2 freeze", "a1_audit": a1_link, "a2_audit": a2_link, "source": a1["source"], "content_map": a1["content_map"], "jobs": {"A1": a1_job, "A2": a2_job}, "gpu_uuids": {"A1": a1_uuid, "A2": a2_uuid}, "production_freeze_passed": True}
    write(args.json, output); print(f"wrote {args.json}")


def _self_test_a2_allocation_prerequisite_reopen() -> None:
    """Exercise the production A2 prerequisite path with a physical JSON reopen.

    Full allocation raw artifacts intentionally need a B300 and cannot be made
    in a planning-machine self-test.  This test nonetheless uses the real
    ``read_once`` SHA/parse/reopen operation, the actual A2 binding builder,
    and its distinct-job comparison.  The raw-artifact validators are
    temporarily witnessed only inside this self-test, so minimal temporary
    JSON can reach the same production allocation branch deterministically.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="v5-crossmap-a2-selftest-") as temporary:
        a1_path = Path(temporary) / "a1-audit.json"
        # Traverse the production ``allocation(A2)`` branch itself.  The
        # raw-artifact validators are witnessed because a planning machine
        # cannot construct B300-complete raws, but all three input files are
        # physically SHA-reopened by the real ``read_once`` and the real A2
        # prerequisite/binding/distinct-job/output code is exercised.
        main0_path = Path(temporary) / "main0.json"
        main1_path = Path(temporary) / "main1.json"
        output_path = Path(temporary) / "a2-audit.json"
        write(main0_path, {"kind": "raw0"})
        write(main1_path, {"kind": "raw1"})
        write(a1_path, {"kind": "a1", "allocation": {"slurm_job_id": "7001"}})
        observed_main: list[tuple[str, int]] = []
        observed_a1: list[str] = []

        def validate_main_witness(
            record: Mapping[str, Any], *, allocation: str, process: int,
            runner_sha: str, analyzer_sha: str, shell_sha: str, label: str,
        ) -> dict[str, object]:
            if record.get("kind") != f"raw{process}" or allocation != "A2" or label != f"main{process}":
                raise AssertionError("A2 allocation used a drifted raw reopen/validation contract")
            if (runner_sha, analyzer_sha, shell_sha) != ("0" * 64, "1" * 64, "2" * 64):
                raise AssertionError("A2 allocation did not forward protocol SHA arguments")
            observed_main.append((label, process))
            return {
                "allocation": allocation,
                "job": "7002",
                "process": process,
                "pid": 8100 + process,
                "uuid": "GPU-selftest",
                "source": {"same": True},
                "map": {"entries": {}, "digest": "0" * 64},
            }

        def validate_allocation_witness(
            record: Mapping[str, Any], expected_allocation: str, runner_sha: str,
            analyzer_sha: str, shell_sha: str, label: str,
        ) -> None:
            if (
                record.get("kind") != "a1"
                or expected_allocation != "A1"
                or (runner_sha, analyzer_sha, shell_sha) != ("0" * 64, "1" * 64, "2" * 64)
                or label != "A1 prerequisite"
            ):
                raise AssertionError("A2 allocation used a drifted A1 prerequisite validator contract")
            observed_a1.append(label)

        def allocation_args(a1_sha: str, json_path: Path) -> argparse.Namespace:
            return argparse.Namespace(
                main0=main0_path,
                main1=main1_path,
                main0_sha256=_sha_bytes(main0_path.read_bytes()),
                main1_sha256=_sha_bytes(main1_path.read_bytes()),
                current_slurm_job_id="7002",
                allocation_id="A2",
                expected_runner_sha256="0" * 64,
                expected_analyzer_sha256="1" * 64,
                expected_protocol_shell_sha256="2" * 64,
                a1_audit=a1_path,
                expected_a1_audit_sha256=a1_sha,
                json=json_path,
            )

        original_validate_main = globals()["validate_main"]
        original_validate_allocation_record = globals()["validate_allocation_record"]
        try:
            globals()["validate_main"] = validate_main_witness
            globals()["validate_allocation_record"] = validate_allocation_witness
            try:
                allocation(allocation_args(_sha_bytes(a1_path.read_bytes()), output_path))
            except TypeError as exc:
                raise AssertionError("allocation(A2) leaked a typed-helper TypeError") from exc
            if observed_main != [("main0", 0), ("main1", 1)] or observed_a1 != ["A1 prerequisite"]:
                raise AssertionError("allocation(A2) did not reopen every required raw/A1 artifact exactly once")
            audit = obj(json.loads(output_path.read_text(encoding="utf-8")), "selftest.a2_audit")
            _validate_a1_binding(audit.get("a1_binding"), {"path": str(a1_path.resolve()), "sha256": _sha_bytes(a1_path.read_bytes())}, "7001", "selftest.a2_allocation_binding")
            require(audit.get("allocation") == {"id": "A2", "slurm_job_id": "7002", "gpu_uuid": "GPU-selftest"}, "selftest A2 allocation output drift")

            same_job_output = Path(temporary) / "same-job-a2-audit.json"
            write(a1_path, {"kind": "a1", "allocation": {"slurm_job_id": "7002"}})
            observed_main.clear(); observed_a1.clear()
            try:
                allocation(allocation_args(_sha_bytes(a1_path.read_bytes()), same_job_output))
            except AuditError:
                pass
            except TypeError as exc:
                raise AssertionError("allocation(A2) leaked TypeError for same-job A1 prerequisite") from exc
            else:
                raise AssertionError("allocation(A2) accepted the same A1/A2 Slurm job")
            if observed_main != [("main0", 0), ("main1", 1)] or observed_a1 != ["A1 prerequisite"] or same_job_output.exists():
                raise AssertionError("allocation(A2) did not fail closed after its reopened same-job A1 prerequisite")

            for allocation_value, description in (
                ([], "non-object A1 allocation"),
                ({"slurm_job_id": 7001}, "non-string A1 Slurm job"),
            ):
                malformed_output = Path(temporary) / ("malformed-" + description.replace(" ", "-") + ".json")
                write(a1_path, {"kind": "a1", "allocation": allocation_value})
                observed_main.clear(); observed_a1.clear()
                try:
                    allocation(allocation_args(_sha_bytes(a1_path.read_bytes()), malformed_output))
                except AuditError:
                    pass
                except TypeError as exc:
                    raise AssertionError(f"allocation(A2) leaked TypeError for {description}") from exc
                else:
                    raise AssertionError(f"allocation(A2) accepted forged {description}")
                if observed_main != [("main0", 0), ("main1", 1)] or observed_a1 != ["A1 prerequisite"] or malformed_output.exists():
                    raise AssertionError(f"allocation(A2) did not fail closed after reopened {description}")
        finally:
            globals()["validate_main"] = original_validate_main
            globals()["validate_allocation_record"] = original_validate_allocation_record


def self_test() -> None:
    if SCHEMA_VERSION != 4:
        raise AssertionError("v1/v2/v3 partial B300 artifacts must remain non-reopenable")
    for old_version in (1, 2, 3):
        old_schema = {key: {} for key in MAIN_SCHEMA_FIELDS}
        old_schema.update({"schema_version": old_version, "purpose": "v5 public-registry cross-map read-only regression; no production source or map mutation"})
        try:
            _validate_main_envelope(old_schema, f"old-v{old_version}-partial")
        except AuditError:
            pass
        else:
            raise AssertionError(f"analyzer accepted an old v{old_version}/partial B300 artifact")
    try:
        validate_bootstrap({"source_ledger_mode": "canonical_path_sha256_without_module_import", "stages": BOOTSTRAP_STAGES[:-1], "heavy_runtime_import_after_clean_gate": True, "passed": True}, "forged-bootstrap")
    except AuditError:
        pass
    else:
        raise AssertionError("analyzer accepted a bootstrap missing canonical-map-after-import stage")
    try:
        validate_helper_runtime_globals({}, "missing-helper-runtime-globals", {})
    except AuditError:
        pass
    else:
        raise AssertionError("analyzer accepted missing helper runtime-global evidence")
    _self_test_a2_allocation_prerequisite_reopen()
    forged_shared_binding = {
        "phase": "pre_hydration",
        "module_label": SHARED_SEQCOUNT_LABEL,
        "module": LOADED_RUNTIME_MODULES[SHARED_SEQCOUNT_LABEL],
        "module_path": "/detached/shared.py",
        "module_object_id": 1,
        "function": SHARED_MAKE_INPUTS,
        "function_module": LOADED_RUNTIME_MODULES[SHARED_SEQCOUNT_LABEL],
        "function_object_id": 2,
        "function_globals_is_module_dict": True,
        "torch_global_present": False,
        "torch_object_id": None,
        "sys_modules_torch_object_id": 3,
        "torch_is_sys_modules_canonical": False,
        "passed": True,
    }
    for field, forged_value in (
        ("function_globals_is_module_dict", False),
        ("module", "forged.shared"),
        ("function_module", "forged.shared"),
    ):
        try:
            _validate_shared_make_inputs_binding(
                {**forged_shared_binding, field: forged_value},
                "forged-shared-binding." + field,
                phase="pre_hydration",
                source_runtime={},
            )
        except AuditError:
            pass
        else:
            raise AssertionError("analyzer accepted a detached/noncanonical shared hydration record")
    try:
        _validate_shared_make_inputs_binding(
            {**forged_shared_binding, "invented": True},
            "forged-shared-binding.invented-field",
            phase="pre_hydration",
            source_runtime={},
        )
    except AuditError:
        pass
    else:
        raise AssertionError("analyzer accepted an invented shared hydration field")
    try:
        _validate_shared_make_inputs_binding(
            {**forged_shared_binding, "torch_global_present": True},
            "forged-shared-binding.pre-torch",
            phase="pre_hydration",
            source_runtime={},
        )
    except AuditError:
        pass
    else:
        raise AssertionError("analyzer accepted a pre-hydration torch global")
    try:
        _validate_shared_make_inputs_binding(
            {
                **forged_shared_binding,
                "phase": "bound_pre_workload",
                "torch_global_present": True,
                "torch_object_id": 4,
                "torch_is_sys_modules_canonical": True,
            },
            "forged-shared-binding.noncanonical-torch",
            phase="bound_pre_workload",
            source_runtime={},
        )
    except AuditError:
        pass
    else:
        raise AssertionError("analyzer accepted a noncanonical shared torch-object record")
    try:
        integer(True, "forged")
    except AuditError:
        pass
    else:
        raise AssertionError("self test accepted bool as int")
    baseline = {
        "requested_variant": "baseline", "chosen_variant": "baseline", "reason": "fixed_batch_shape_not_whitelisted",
        "extension_sha256": None, "varlen_cpu_authoritative": False,
        "certified_varlen_offsets": None, "canonical_cache_hit": None,
    }
    validate_baseline_decision(baseline, "fixed_batch_shape_not_whitelisted", "selftest-baseline")
    try:
        validate_baseline_decision({**baseline, "extension_sha256": EXTENSION_SHA}, "fixed_batch_shape_not_whitelisted", "selftest-baseline-forged")
    except AuditError:
        pass
    else:
        raise AssertionError("self test accepted baseline non-null extension provenance")
    try:
        strict_equal(1, 1.0, "forged")
    except AuditError:
        pass
    else:
        raise AssertionError("self test accepted JSON type substitution")
    try:
        digest("A" * 64, "forged")
    except AuditError:
        pass
    else:
        raise AssertionError("self test accepted uppercase SHA")
    try:
        _validate_output_abi({"shape": [1, 2048, 12, 128], "dtype": "torch.bfloat16", "contiguous": True}, CELL_OUTPUT_SHAPES["fixed_b2_h12_t2048_fp32_both"], "wrong-cell-shape")
    except AuditError:
        pass
    else:
        raise AssertionError("self test accepted an output ABI from the wrong cell")
    forged_spy = {
        "before": {"c1": 3, "pinned": 4},
        "after": {"c1": 4, "pinned": 4},
        "delta": {"c1": 0, "pinned": 0},
        "passed": True,
    }
    try:
        validate_spy(forged_spy, {"c1": 1, "pinned": 0}, "forged-spy")
    except AuditError:
        pass
    else:
        raise AssertionError("self test accepted a forged route-spy delta")
    try:
        _validate_varlen_cache_stats({"entries": 0, "hits": 0, "misses": 0, "capture_miss_rejections": 0}, "missing-cache-field", mode="zero")
    except AuditError:
        pass
    else:
        raise AssertionError("self test accepted an incomplete varlen cache schema")
    try:
        validate_varlen_positive_handoff({"public_prepare": {}, "cache_stats": {}, "direct_canonical_cache_hit": False, "public_canonical_cache_hit": True, "direct_miss_to_public_hit": True, "passed": True, "invented": True}, "invented-handoff-field")
    except AuditError:
        pass
    else:
        raise AssertionError("self test accepted an invented varlen handoff field")
    for forged_counts in (
        {"pinned_public": True, "c1_public": SAMPLES // 2},
        {"pinned_public": float(SAMPLES // 2), "c1_public": SAMPLES // 2},
    ):
        try:
            _validate_first_path_counts(forged_counts, "forged-first-path-counts")
        except AuditError:
            pass
        else:
            raise AssertionError("self test accepted non-integer first-path counts")
    for forged_latency in (1, True):
        try:
            floating(forged_latency, "forged-latency")
        except AuditError:
            pass
        else:
            raise AssertionError("self test accepted a non-float latency/ratio")
    for probe in ({}, {"entries": {"fixed_batch": [], "fixed_single_batch": [], "varlen": []}, "digest": "0" * 64}):
        try:
            validate_map(probe, "forged-map", raw_pid_snapshot=False)
        except AuditError:
            pass
        else:
            raise AssertionError("self test accepted a detached/incomplete content map")
    try:
        _require_content_map_binding({"entries": {"fixed_batch": [], "fixed_single_batch": [], "varlen": []}, "digest": "0" * 64}, {"entries": {"fixed_batch": [], "fixed_single_batch": [], "varlen": []}, "digest": "1" * 64}, "forged-raw-map-binding")
    except AuditError:
        pass
    else:
        raise AssertionError("self test accepted a raw content-map detached from its audit map")
    try:
        _require_distinct_pids({17}, "forged-pids")
    except AuditError:
        pass
    else:
        raise AssertionError("self test accepted duplicate fresh PID evidence")
    try:
        _validate_a1_binding({"a1_audit": {"path": "/a1", "sha256": "0" * 64}, "a1_job": "999", "different_slurm_job": True}, {"path": "/a1", "sha256": "0" * 64}, "123", "forged-binding")
    except AuditError:
        pass
    else:
        raise AssertionError("self test accepted a lied A1 job binding")
    try:
        _validate_tree({}, "missing-dirty", commit=PATCHED_COMMIT, dirty=PATCHED_DIRTY)
    except AuditError:
        pass
    else:
        raise AssertionError("self test accepted missing patched dirty-set ledger")
    try:
        _require_runtime_fla_ledger({}, "missing-fla")
    except AuditError:
        pass
    else:
        raise AssertionError("self test accepted missing FLA public ledger")
    print("ANALYZER_SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--expected-runner-sha256", required=True)
    common.add_argument("--expected-analyzer-sha256", required=True)
    common.add_argument("--expected-protocol-shell-sha256", required=True)
    p_alloc = sub.add_parser("allocation", parents=(common,))
    p_alloc.add_argument("--allocation-id", choices=("A1", "A2"), required=True); p_alloc.add_argument("--current-slurm-job-id", required=True); p_alloc.add_argument("--main0", type=Path, required=True); p_alloc.add_argument("--main1", type=Path, required=True); p_alloc.add_argument("--main0-sha256", required=True); p_alloc.add_argument("--main1-sha256", required=True); p_alloc.add_argument("--a1-audit", type=Path); p_alloc.add_argument("--expected-a1-audit-sha256"); p_alloc.add_argument("--json", type=Path, required=True)
    p_freeze = sub.add_parser("freeze", parents=(common,))
    p_freeze.add_argument("--a1-audit", type=Path, required=True); p_freeze.add_argument("--a2-audit", type=Path, required=True); p_freeze.add_argument("--expected-a1-sha256", required=True); p_freeze.add_argument("--expected-a2-sha256", required=True); p_freeze.add_argument("--json", type=Path, required=True)
    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "self-test": self_test()
    elif args.command == "allocation": allocation(args)
    else: freeze(args)


if __name__ == "__main__":
    main()
