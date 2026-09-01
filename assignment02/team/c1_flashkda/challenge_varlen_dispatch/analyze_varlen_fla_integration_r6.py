#!/usr/bin/env python3
"""Fail-closed independent audit of the r6 public-FLA production freeze.

This program is deliberately stdlib-only.  It recalculates performance facts
from the two raw 1,000-sample vectors per cell and treats all runner-produced
summaries and release lists only as claims to cross-check, never as evidence.
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
from typing import Any, Mapping


SCHEMA_VERSION = 4
PERFORMANCE_SEED = 20260901
WARMUP = 100
SAMPLES = 1000
REPEATS = 2
MIN_MARGIN = 0.02
PERCENTILES = (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
PATHS = ("public_registry_c1", "public_registry_pinned")
AUDITED_EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
HELPER_SHA256 = "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
PATCHED_FLASH_KDA_INIT_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
INTEGRATION_RUNNER_SHA256 = "71a016307d385d846dfc9e58fefeb041446616a08d3ee36d73f2ac2d3d5ac058"
INTEGRATION_RUNNER_SHA256_ENV = "C1_VARLEN_FLA_INTEGRATION_R6_RUNNER_SHA256"
PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
FLA_FILE_SHA256 = {
    "fla/__init__.py": "b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d",
    "fla/ops/backends/__init__.py": "a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635",
    "fla/ops/kda/__init__.py": "24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb",
    "fla/ops/kda/backends/__init__.py": "86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797",
    "fla/ops/kda/backends/flash_kda.py": "0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2",
    "fla/ops/kda/chunk.py": "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8",
}
FLA_LOADED_MODULES = {
    "fla": "fla/__init__.py",
    "fla.ops.backends": "fla/ops/backends/__init__.py",
    "fla.ops.kda": "fla/ops/kda/__init__.py",
    "fla.ops.kda.backends": "fla/ops/kda/backends/__init__.py",
    "fla.ops.kda.backends.flash_kda": "fla/ops/kda/backends/flash_kda.py",
    "fla.ops.kda.chunk": "fla/ops/kda/chunk.py",
}
INTEGRATION_RUNTIME_IDENTITIES = {
    "auto_dispatch": (
        "2b817adb7d21d1f223e8df4616eeccd74e34a5b1944492211f0f0254147ba883",
        "assignment02/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py",
    ),
    "fla_backend": (
        "8555995c04ecd666a580ddee02eae1d34820ef1a601cbad5d10f9c6b8505974b",
        "assignment02/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py",
    ),
    "varlen_metadata": (
        "f89a97ba284c7a24a3df54efca7bb60eb70f9cbda659200bd6cb3e8dfdaf4ccd",
        "assignment02/team/c1_flashkda/challenge_tp8_dispatch/varlen_metadata.py",
    ),
    "confirmation_runner": (
        "9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b",
        "assignment02/team/c1_flashkda/challenge_varlen_dispatch/run_varlen_dispatch_confirmation.py",
    ),
    "shared_seqcount_runner": (
        "4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f",
        "assignment02/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py",
    ),
    "prefetch2": (
        "752126488487ac317a7ee167b660b0895562e4877aedbc4ce2a599c1f59a10d0",
        "assignment02/team/c1_flashkda/challenge_prefetch2/prefetch2.py",
    ),
    "vshard4_prefetch2": (
        "445c90815919fb8c1db3bc79289d10029ea77b1d9accb393c326c120ba1f8385",
        "assignment02/team/c1_flashkda/challenge_vshard4_prefetch2/vshard4_prefetch2.py",
    ),
    "harness": (
        "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52",
        "assignment02/team/c1_flashkda/harness/validate_and_bench.py",
    ),
    "pinned_torch_ref": ("bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5", "tests/torch_ref.py"),
    "pinned_reference_helper": ("8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f", "sigmoid_ext/sigmoid_ext.so"),
}
EVENT_CONTRACT = (
    "current-stream start event -> immediate start.synchronize -> public FLA chunk_kda -> "
    "end event -> immediate end.synchronize -> elapsed_time; both synchronizations are excluded from the sample value"
)
PATH_ORDER = {
    "even_sample": ["public_registry_c1", "public_registry_pinned"],
    "odd_sample": ["public_registry_pinned", "public_registry_c1"],
    "timed_first_path_counts": {"public_registry_c1": SAMPLES // 2, "public_registry_pinned": SAMPLES // 2},
    "warmup_first_path_counts": {"public_registry_c1": WARMUP // 2, "public_registry_pinned": WARMUP // 2},
}
PATH_ENVIRONMENT = {
    "public_registry_c1": {"C1_B300_FLASH_KDA": "1", "FLA_FLASH_KDA": "1"},
    "public_registry_pinned": {"C1_B300_FLASH_KDA": "0", "FLA_FLASH_KDA": "1"},
}
EXPECTED_ROUTE_DELTA_PER_CALL = {
    "public_registry_c1": {"c1": 1, "pinned": 0},
    "public_registry_pinned": {"c1": 0, "pinned": 1},
}

RAW_RELEASE_FAILED_CELL = "equal_n4_h12_t2048/fp32_both"
RECORD_ONLY = "mixed_n6_h12_t8192/fp32_both"
PUBLIC_CONTRACTS = ("none", "fp32_final_only", "fp32_both")
CASE_LAYOUTS = {
    "equal_n2_h12_t2048": "equal_n2_h12_t4096",
    "equal_n4_h12_t2048": "equal_n4_h12_t8192",
    "mixed_n6_h12_t8192": "mixed_n6_h12_t8192",
    "skew_n6_h12_t12288": "skew_n6_h12_t12288",
}
CASE_SEQUENCES = {
    "equal_n2_h12_t2048": 2,
    "equal_n4_h12_t2048": 4,
    "mixed_n6_h12_t8192": 6,
    "skew_n6_h12_t12288": 6,
}
# r6 production freeze: only the public performance-release intersection may
# take C1.  Every other public contract is an explicit pinned-only fallback.
POSITIVE: dict[str, str] = {
    "skew_n6_h12_t12288/none": "vshard2_p2",
    "skew_n6_h12_t12288/fp32_final_only": "vshard2_p2",
}
PRODUCTION_MAP_ENTRIES = [
    {"offsets": [0, 1, 2, 3, 4, 5, 12288], "contract": "none", "variant": "vshard2_p2"},
    {"offsets": [0, 1, 2, 3, 4, 5, 12288], "contract": "fp32_final_only", "variant": "vshard2_p2"},
]
ALL_PUBLIC_CELLS = tuple(
    f"{case}/{contract}"
    for case in CASE_LAYOUTS
    for contract in PUBLIC_CONTRACTS
)
POLICY_FALLBACK_CELLS: dict[str, dict[str, str]] = {
    key: {
        "classification": (
            "raw_release_failed" if key == RAW_RELEASE_FAILED_CELL
            else "record_only" if key == RECORD_ONLY
            else "public_release_failed"
        ),
        "action": "pinned_baseline_only",
    }
    for key in ALL_PUBLIC_CELLS
    if key not in POSITIVE
}
POLICY_FALLBACK_FINAL_CONTRACTS = {
    key: (CASE_SEQUENCES[key.split("/", 1)[0]], key.split("/", 1)[1] != "none")
    for key in POLICY_FALLBACK_CELLS
}
POLICY_FALLBACK_C1_REASONS = {
    key: (
        "C1 packed-varlen preflight rejected: "
        f"varlen_{CASE_LAYOUTS[key.split('/', 1)[0]]}_{key.split('/', 1)[1]}_not_whitelisted"
    )
    for key in POLICY_FALLBACK_CELLS
}
GENERAL_NEGATIVE_C1_REASONS = {
    "same_n_total_different_split": "C1 packed-varlen preflight rejected: varlen_offsets_not_whitelisted",
    "varlen_env_unset": (
        "C1 packed-varlen preflight rejected: "
        "set C1_B300_VARLEN_CPU_DESCRIPTOR=1 to opt into CPU-authoritative packed varlen"
    ),
    "cpu_missing": "C1 packed-varlen preflight rejected: cpu_descriptor_must_be_cpu",
    "cpu_malformed": "C1 packed-varlen preflight rejected: cpu_descriptor_offsets_must_be_strictly_increasing",
}
EXPECTED_C1_REJECTION_REASONS = {
    **POLICY_FALLBACK_C1_REASONS,
    **GENERAL_NEGATIVE_C1_REASONS,
}
OFFSETS_BY_CASE = {
    "equal_n2_h12_t2048": [0, 2048, 4096],
    "equal_n4_h12_t2048": [0, 2048, 4096, 6144, 8192],
    "mixed_n6_h12_t8192": [0, 17, 528, 1552, 2852, 4901, 8192],
    "skew_n6_h12_t12288": [0, 1, 2, 3, 4, 5, 12288],
}
COMMON_INPUT_FIELDS = {
    "q", "k", "v", "g", "beta", "A_log", "dt_bias", "cu_seqlens", "cu_seqlens_cpu",
}
NEGATIVE_SIMPLE = tuple(POLICY_FALLBACK_CELLS) + (
    "same_n_total_different_split",
    "varlen_env_unset",
    "cpu_missing",
    "cpu_malformed",
)
SKEW_NONE_REASON = "varlen_skew_n6_h12_t12288_none_whitelist_hit"
RECORDED_DECISION_FIELDS = {
    "requested_variant",
    "chosen_variant",
    "reason",
    "extension_sha256",
    "varlen_cpu_authoritative",
    "certified_varlen_offsets",
    "canonical_cache_hit",
}


class AuditError(AssertionError):
    """The JSON lacks enough unambiguous evidence to accept the experiment."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def mapping(value: object, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be a JSON object")
    return value


def sequence(value: object, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a JSON array")
    return value


def require_production_map_evidence(value: object, label: str) -> None:
    """Validate the frozen production map without Python bool/int/float equality aliases."""

    evidence = mapping(value, label)
    require(
        set(evidence)
        == {
            "passed",
            "exact_entries",
            "entry_count",
            "checked_before_cuda_initialization",
            "runner_mutates_production_map",
        },
        f"{label}: field-set drift",
    )
    require(evidence.get("passed") is True, f"{label}: pass marker drift")
    require(
        type(evidence.get("entry_count")) is int and evidence.get("entry_count") == 2,
        f"{label}: entry count must be the exact JSON integer 2",
    )
    require(
        evidence.get("checked_before_cuda_initialization") is True
        and evidence.get("runner_mutates_production_map") is False,
        f"{label}: lifecycle markers drift",
    )
    entries = sequence(evidence.get("exact_entries"), f"{label}.exact_entries")
    require(len(entries) == len(PRODUCTION_MAP_ENTRIES), f"{label}: entry scope drift")
    for index, (entry_value, expected) in enumerate(zip(entries, PRODUCTION_MAP_ENTRIES)):
        entry = mapping(entry_value, f"{label}.exact_entries[{index}]")
        require(
            set(entry) == {"offsets", "contract", "variant"},
            f"{label}.exact_entries[{index}]: field-set drift",
        )
        offsets = sequence(entry.get("offsets"), f"{label}.exact_entries[{index}].offsets")
        require(
            all(type(offset) is int for offset in offsets),
            f"{label}.exact_entries[{index}]: offsets must be exact JSON integers",
        )
        require(
            type(entry.get("contract")) is str and type(entry.get("variant")) is str,
            f"{label}.exact_entries[{index}]: contract/variant type drift",
        )
        require(dict(entry) == expected, f"{label}.exact_entries[{index}]: value drift")


def numeric(value: object, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def close(actual: object, expected: float, label: str) -> None:
    require(math.isclose(numeric(actual, label), expected, rel_tol=0.0, abs_tol=1e-12), f"{label} disagrees with raw recomputation")


def read_artifact(path: Path, expected_sha256: str) -> tuple[Mapping[str, Any], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AuditError(f"cannot read integration artifact: {path}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == expected_sha256, f"integration artifact SHA256 mismatch: {digest}")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AuditError("integration artifact is invalid JSON") from exc
    return mapping(parsed, "artifact"), digest


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def recompute_repeat(repeat_value: object, key: str, repeat_index: int) -> dict[str, object]:
    repeat = mapping(repeat_value, f"performance.{key}.repeat{repeat_index}")
    require(repeat.get("repeat_index") == repeat_index, f"performance {key}: repeat-index drift")
    raw = mapping(repeat.get("raw_samples_ms"), f"performance.{key}.repeat{repeat_index}.raw_samples_ms")
    require(set(raw) == set(PATHS), f"performance {key}: raw path scope drift")
    summaries: dict[str, dict[str, object]] = {}
    for path in PATHS:
        samples = sequence(raw[path], f"performance.{key}.repeat{repeat_index}.{path}")
        require(len(samples) == SAMPLES, f"performance {key}/{path}: expected exactly {SAMPLES} raw samples")
        values = [numeric(value, f"performance.{key}.{path}[{index}]") for index, value in enumerate(samples)]
        require(all(value > 0.0 for value in values), f"performance {key}/{path}: nonpositive raw sample")
        summaries[path] = {
            "samples": SAMPLES,
            "mean_ms": statistics.fmean(values),
            "p50_ms": percentile(values, 0.50),
            "p95_ms": percentile(values, 0.95),
            "p99_ms": percentile(values, 0.99),
        }
    winners: dict[str, str] = {}
    margins: dict[str, float | None] = {}
    for name, _ in PERCENTILES:
        ranked = sorted((float(summaries[path][f"{name}_ms"]), path) for path in PATHS)
        winner = ranked[0][1]
        winners[name] = winner
        margins[name] = ranked[1][0] / ranked[0][0] - 1.0 if winner == "public_registry_c1" else None
    winner_pass = all(winners[name] == "public_registry_c1" for name, _ in PERCENTILES)
    margin_pass = all(margins[name] is not None and float(margins[name]) >= MIN_MARGIN for name, _ in PERCENTILES)
    return {
        "recomputed_from_raw_samples": True,
        "summaries": summaries,
        "winner_by_percentile": winners,
        "c1_margin_over_pinned_by_percentile": margins,
        "winner_gate_pass": winner_pass,
        "margin_gate_pass": margin_pass,
        "repeat_gate_pass": winner_pass and margin_pass,
    }


def require_spy(value: object, expected: Mapping[str, int], label: str) -> None:
    spy = mapping(value, label)
    require(spy.get("passed") is True, f"{label}: spy did not pass")
    before, after, delta = mapping(spy.get("before"), f"{label}.before"), mapping(spy.get("after"), f"{label}.after"), mapping(spy.get("delta"), f"{label}.delta")
    require(dict(delta) == dict(expected), f"{label}: route delta drift")
    for name, count in expected.items():
        before_count = before.get(name)
        after_count = after.get(name)
        require(isinstance(before_count, int) and not isinstance(before_count, bool), f"{label}.before.{name} invalid")
        require(isinstance(after_count, int) and not isinstance(after_count, bool), f"{label}.after.{name} invalid")
        require(after_count - before_count == count, f"{label}.{name}: before/after counter drift")


def require_immutability(value: object, label: str, expected_fields: set[str]) -> list[str]:
    evidence = mapping(value, label)
    require(evidence.get("input_immutability_exact") is True, f"{label}: immutable-input check failed")
    fields = sequence(evidence.get("fields"), f"{label}.fields")
    require(fields and all(isinstance(field, str) and field for field in fields), f"{label}: immutable-input field evidence is empty or malformed")
    require(fields == sorted(expected_fields), f"{label}: immutable-input coverage drift")
    return [str(field) for field in fields]


def require_final_contract(value: object, label: str, *, sequences: int, final_required: bool) -> None:
    final = mapping(value, label)
    if not final_required:
        require(set(final) == {"present"} and final.get("present") is False, f"{label}: unexpected final-state fields")
        return
    expected_shape = [sequences, 12, 128, 128]
    require(
        set(final) == {"present", "dtype", "shape", "contiguous"}
        and final.get("present") is True
        and final.get("dtype") == "torch.float32"
        and final.get("shape") == expected_shape
        and final.get("contiguous") is True,
        f"{label}: final-state contract/shape drift",
    )


def require_output_contract(value: object, label: str, output_shape: list[int]) -> None:
    contract = mapping(value, label)
    require(
        set(contract) == {"shape", "dtype", "contiguous"}
        and contract.get("shape") == output_shape
        and contract.get("dtype") == "torch.bfloat16"
        and contract.get("contiguous") is True,
        f"{label}: output tensor contract/shape drift",
    )


def require_exact(
    value: object,
    label: str,
    *,
    sequences: int,
    total_tokens: int,
    output_batch: int,
    final_required: bool,
) -> None:
    exact = mapping(value, label)
    expected_fields = {
        "output_exact", "output_max_abs", "actual_output", "expected_output", "actual_final", "expected_final",
    }
    if final_required:
        expected_fields |= {"final_exact", "final_max_abs"}
    require(set(exact) == expected_fields, f"{label}: exact comparison field-set drift")
    require(exact.get("output_exact") is True, f"{label}: output exactness failed")
    require(numeric(exact.get("output_max_abs"), f"{label}.output_max_abs") == 0.0, f"{label}: nonzero output delta")
    output_shape = [output_batch, total_tokens, 12, 128]
    require_output_contract(exact.get("actual_output"), f"{label}.actual_output", output_shape)
    require_output_contract(exact.get("expected_output"), f"{label}.expected_output", output_shape)
    actual_final = exact.get("actual_final")
    expected_final = exact.get("expected_final")
    require_final_contract(actual_final, f"{label}.actual_final", sequences=sequences, final_required=final_required)
    require_final_contract(expected_final, f"{label}.expected_final", sequences=sequences, final_required=final_required)
    if final_required:
        require(exact.get("final_exact") is True, f"{label}: final exactness failed")
        require(numeric(exact.get("final_max_abs"), f"{label}.final_max_abs") == 0.0, f"{label}: nonzero final delta")


def require_identity_and_gates(data: Mapping[str, Any]) -> None:
    require(
        len(POSITIVE) == 2
        and len(POLICY_FALLBACK_CELLS) == 10
        and set(POSITIVE).isdisjoint(POLICY_FALLBACK_CELLS)
        and set(POSITIVE).union(POLICY_FALLBACK_CELLS) == set(ALL_PUBLIC_CELLS),
        "r6 public matrix constant drift",
    )
    classifications = [entry["classification"] for entry in POLICY_FALLBACK_CELLS.values()]
    require(
        classifications.count("public_release_failed") == 8
        and classifications.count("raw_release_failed") == 1
        and classifications.count("record_only") == 1,
        "r6 fallback classification scope drift",
    )
    require(
        type(data.get("schema_version")) is int and data.get("schema_version") == SCHEMA_VERSION,
        "integration schema version drift",
    )
    require(data.get("complete") is True, "integration artifact incomplete")
    gates = mapping(data.get("gates"), "gates")
    required = ("scope", "production_map", "clean_gpu", "device", "extension", "fla_pin", "inference_mode", "registry_spy_identity", "prepare_spy_restored", "python_nvidia_clean")
    require(set(gates) == set(required), "integration gate scope drift")
    for name in required:
        require(mapping(gates.get(name), f"gates.{name}").get("passed") is True, f"required gate failed: {name}")
    scope = mapping(gates.get("scope"), "gates.scope")
    require(
        set(scope)
        == {
            "positive_cells",
            "policy_fallback_cells",
            "required_positive_cells",
            "required_policy_fallback_cells",
            "passed",
        }
        and type(scope.get("positive_cells")) is int
        and scope.get("positive_cells") == 2
        and type(scope.get("policy_fallback_cells")) is int
        and scope.get("policy_fallback_cells") == 10
        and type(scope.get("required_positive_cells")) is int
        and scope.get("required_positive_cells") == 2
        and type(scope.get("required_policy_fallback_cells")) is int
        and scope.get("required_policy_fallback_cells") == 10
        and scope.get("passed") is True,
        "r6 public scope gate drift",
    )
    require_production_map_evidence(
        data.get("production_map_before_gpu"), "production_map_before_gpu"
    )
    require_production_map_evidence(gates.get("production_map"), "gates.production_map")
    prepare_restore = mapping(gates.get("prepare_spy_restored"), "gates.prepare_spy_restored")
    require(
        set(prepare_restore) == {"instance_shadow", "descriptor_binding", "passed"}
        and prepare_restore.get("instance_shadow") is False
        and prepare_restore.get("descriptor_binding") is True
        and prepare_restore.get("passed") is True,
        "r6 prepare-spy restoration gate drift",
    )
    inference_mode = mapping(gates.get("inference_mode"), "gates.inference_mode")
    require(
        set(inference_mode) == {"scope", "grad_enabled", "inference_mode_enabled", "passed"}
        and inference_mode.get("scope")
        == "single main-thread torch.inference_mode covers positive, policy fallback, cache/capture, hot-sync, fixed control, and performance"
        and inference_mode.get("grad_enabled") is False
        and inference_mode.get("inference_mode_enabled") is True,
        "inference-mode execution gate drift",
    )
    policy_fallbacks = mapping(data.get("policy_fallback_cells"), "policy_fallback_cells")
    require(dict(policy_fallbacks) == POLICY_FALLBACK_CELLS, "policy fallback preregistration drift")
    record_only = mapping(data.get("record_only_cell"), "record_only_cell")
    require(
        record_only == {"cell": RECORD_ONLY, "action": "pinned_baseline_only"},
        "record-only preregistration drift",
    )
    identity = mapping(data.get("identity"), "identity")
    device = mapping(identity.get("device"), "identity.device")
    require(type(device.get("name")) is str and "B300" in device["name"].upper(), "not B300")
    capability = sequence(device.get("capability"), "identity.device.capability")
    require(
        len(capability) == 2
        and all(type(component) is int for component in capability)
        and capability == [10, 3]
        and type(device.get("multiprocessor_count")) is int
        and device.get("multiprocessor_count") == 148,
        "B300 identity drift",
    )
    extension = mapping(identity.get("extension"), "identity.extension")
    require(extension.get("sha256") == AUDITED_EXTENSION_SHA256 and extension.get("passed") is True, "extension identity drift")
    flash_kda_python = mapping(identity.get("flash_kda_python"), "identity.flash_kda_python")
    require(flash_kda_python.get("sha256") == PATCHED_FLASH_KDA_INIT_SHA256, "flash_kda Python SHA drift")
    flash_kda_path = flash_kda_python.get("path")
    require(
        isinstance(flash_kda_path, str)
        and flash_kda_path.replace("\\", "/").rstrip("/").endswith("flash_kda/__init__.py"),
        "flash_kda Python path drift",
    )
    integration_runner = mapping(identity.get("integration_runner"), "identity.integration_runner")
    require(
        set(integration_runner) == {"path", "sha256", "sha256_gate_pass", "expected_sha256_environment"},
        "integration-runner identity field-set drift",
    )
    require(
        integration_runner.get("sha256_gate_pass") is True
        and integration_runner.get("sha256") == INTEGRATION_RUNNER_SHA256
        and integration_runner.get("expected_sha256_environment") == INTEGRATION_RUNNER_SHA256_ENV,
        "integration runner SHA/environment drift",
    )
    runner_path = integration_runner.get("path")
    require(
        isinstance(runner_path, str)
        and runner_path.replace("\\", "/").rstrip("/").endswith(
            "assignment02/team/c1_flashkda/challenge_varlen_dispatch/run_varlen_fla_integration_r6.py"
        ),
        "integration runner path drift",
    )
    runtime = mapping(identity.get("runtime_import_identities"), "identity.runtime_import_identities")
    require(set(runtime) == set(INTEGRATION_RUNTIME_IDENTITIES), "integration runtime identity key-set drift")
    for name, (expected_sha256, expected_suffix) in INTEGRATION_RUNTIME_IDENTITIES.items():
        entry = mapping(runtime.get(name), f"identity.runtime_import_identities.{name}")
        require(
            set(entry) == {"path", "sha256", "sha256_gate_pass"}
            and entry.get("sha256") == expected_sha256
            and entry.get("sha256_gate_pass") is True,
            f"integration runtime identity SHA/field drift: {name}",
        )
        item_path = entry.get("path")
        require(
            isinstance(item_path, str) and item_path.replace("\\", "/").rstrip("/").endswith(expected_suffix),
            f"integration runtime identity path drift: {name}",
        )
    helper = mapping(identity.get("pinned_reference_helper"), "identity.pinned_reference_helper")
    require(helper.get("sha256") == HELPER_SHA256 and helper.get("no_build") is True, "pinned helper identity drift")
    fla = mapping(identity.get("fla"), "identity.fla")
    require(
        set(fla) == {"root", "commit", "tracked_status_clean", "files", "loaded_modules", "public_callables", "passed"},
        "FLA identity field-set drift",
    )
    require(
        fla.get("commit") == FLA_COMMIT and fla.get("passed") is True and fla.get("tracked_status_clean") is True,
        "FLA commit/tracked-status identity drift",
    )
    require(mapping(fla.get("files"), "identity.fla.files") == FLA_FILE_SHA256, "FLA source identity drift")
    fla_root = fla.get("root")
    require(
        isinstance(fla_root, str) and fla_root and (Path(fla_root).is_absolute() or fla_root.startswith("/")),
        "FLA root is missing or non-absolute",
    )
    normalized_root = fla_root.replace("\\", "/").rstrip("/")
    loaded_modules = mapping(fla.get("loaded_modules"), "identity.fla.loaded_modules")
    require(set(loaded_modules) == set(FLA_LOADED_MODULES), "FLA loaded-module key-set drift")
    for module, relative in FLA_LOADED_MODULES.items():
        expected_path = f"{normalized_root}/{relative}"
        require(loaded_modules.get(module) == expected_path, f"FLA loaded-module path drift: {module}")
    public_callables = mapping(fla.get("public_callables"), "identity.fla.public_callables")
    require(set(public_callables) == {"fla.ops.kda.chunk_kda"}, "FLA public-callable scope drift")
    chunk_kda = mapping(public_callables.get("fla.ops.kda.chunk_kda"), "identity.fla.public_callables.chunk_kda")
    source_path = chunk_kda.get("source_path")
    require(
        set(chunk_kda) == {"implementation_identity_match", "module", "qualname", "source_path", "passed"}
        and chunk_kda.get("implementation_identity_match") is True
        and chunk_kda.get("module") == "fla.ops.kda.chunk"
        and chunk_kda.get("qualname") == "chunk_kda"
        and isinstance(source_path, str)
        and (Path(source_path).is_absolute() or source_path.startswith("/"))
        and source_path.replace("\\", "/").rstrip("/") == f"{normalized_root}/fla/ops/kda/chunk.py"
        and chunk_kda.get("passed") is True,
        "FLA public chunk_kda identity drift",
    )
    source_trees = mapping(identity.get("source_trees"), "identity.source_trees")
    require(set(source_trees) == {"patched", "reference"}, "source-tree identity key-set drift")
    patched = mapping(source_trees.get("patched"), "identity.source_trees.patched")
    reference = mapping(source_trees.get("reference"), "identity.source_trees.reference")
    require(
        set(patched) == {"root", "commit", "passed"}
        and isinstance(patched.get("root"), str)
        and patched.get("commit") == PATCHED_COMMIT
        and patched.get("passed") is True,
        "patched source-tree identity drift",
    )
    require(
        set(reference) == {"root", "commit", "tracked_status_clean", "passed"}
        and isinstance(reference.get("root"), str)
        and reference.get("commit") == PATCHED_COMMIT
        and reference.get("tracked_status_clean") is True
        and reference.get("passed") is True,
        "reference source-tree identity drift",
    )
    clean = mapping(identity.get("python_pre_torch_nvidia_smi"), "identity.python_pre_torch_nvidia_smi")
    require(
        clean.get("passed") is True
        and type(clean.get("memory_used_mib")) is int
        and clean.get("memory_used_mib") == 0,
        "python clean-GPU identity drift",
    )
    registry = mapping(data.get("registry"), "registry")
    require(
        type(registry.get("c1_id")) is int
        and registry.get("c1_id") > 0
        and type(registry.get("pinned_id")) is int
        and registry.get("pinned_id") > 0
        and registry["c1_id"] != registry["pinned_id"],
        "registry identity drift",
    )
    snapshot = sequence(registry.get("snapshot"), "registry.snapshot")
    types = [mapping(item, f"registry.snapshot[{index}]").get("backend_type") for index, item in enumerate(snapshot)]
    require(types.count("c1_b300_flash_kda") == 1 and types.count("flash_kda") == 1, "registry backend scope drift")
    require(types.index("c1_b300_flash_kda") < types.index("flash_kda"), "C1 registry order drift")


def require_positive(data: Mapping[str, Any]) -> dict[str, object]:
    preregistered = sequence(data.get("positive_cells"), "positive_cells")
    require(len(preregistered) == 2, "positive preregistration count drift")
    preregistered_map = {mapping(item, f"positive_cells[{index}]").get("cell"): mapping(item, f"positive_cells[{index}]").get("expected_variant") for index, item in enumerate(preregistered)}
    require(preregistered_map == POSITIVE, "positive preregistration mapping drift")
    positives = mapping(data.get("positive_results"), "positive_results")
    require(set(positives) == set(POSITIVE), "positive result scope drift")
    audit: dict[str, object] = {}
    comparisons = ("pinned_vs_torch_ref", "direct_c1_vs_pinned", "public_vs_pinned", "public_pinned_vs_torch_ref", "direct_c1_vs_torch_ref", "public_vs_torch_ref")
    for key, expected_variant in POSITIVE.items():
        case, contract = key.split("/", 1)
        expected_offsets = OFFSETS_BY_CASE[case]
        expected_fields = set(COMMON_INPUT_FIELDS)
        if contract == "fp32_both":
            expected_fields.add("initial_state")
        final_required = contract != "none"
        entry = mapping(positives.get(key), f"positive_results.{key}")
        require(entry.get("passed") is True and entry.get("expected_variant") == expected_variant, f"positive {key}: completion/variant drift")
        # These fields are deliberately mandatory: without post-call input
        # snapshots for all public and direct routes, an offline JSON auditor cannot
        # claim public-path input immutability.
        require(entry.get("input_immutability_exact") is True, f"positive {key}: immutable-input evidence missing or failed")
        aggregate_fields = sequence(entry.get("input_immutability_fields"), f"positive.{key}.input_immutability_fields")
        paths = mapping(entry.get("input_immutability_by_path"), f"positive.{key}.input_immutability_by_path")
        require(set(paths) == {"pinned", "direct_c1", "public_c1", "public_pinned"}, f"positive {key}: immutable-input route scope drift")
        observed_fields: set[str] = set()
        for path, evidence in paths.items():
            observed_fields.update(require_immutability(evidence, f"positive.{key}.immutability.{path}", expected_fields))
        require(aggregate_fields == sorted(expected_fields) and observed_fields == expected_fields, f"positive {key}: aggregate immutable-input fields drift")
        verifier = mapping(entry.get("verifier"), f"positive.{key}.verifier")
        require(mapping(verifier.get("c1"), f"positive.{key}.c1").get("passed") is True, f"positive {key}: C1 verifier failed")
        require(mapping(verifier.get("pinned"), f"positive.{key}.pinned").get("passed") is True, f"positive {key}: pinned verifier failed")
        direct = mapping(entry.get("direct_decision"), f"positive.{key}.direct_decision")
        public = mapping(entry.get("public_decision"), f"positive.{key}.public_decision")
        require(direct.get("chosen_variant") == expected_variant and public.get("chosen_variant") == expected_variant, f"positive {key}: chosen variant drift")
        require(direct.get("canonical_cache_hit") is False and public.get("canonical_cache_hit") is True, f"positive {key}: cold/hot cache evidence drift")
        require(direct.get("certified_varlen_offsets") == expected_offsets, f"positive {key}: direct CPU offsets drift")
        require(public.get("certified_varlen_offsets") == expected_offsets, f"positive {key}: public CPU offsets drift")
        sequences = len(expected_offsets) - 1
        for comparison in comparisons:
            require_exact(
                entry.get(comparison), f"positive.{key}.{comparison}",
                sequences=sequences, total_tokens=expected_offsets[-1], output_batch=1,
                final_required=final_required,
            )
        require_spy(entry.get("public_c1_spy"), {"c1": 1, "pinned": 0}, f"positive.{key}.public_c1_spy")
        require_spy(entry.get("public_pinned_spy"), {"c1": 0, "pinned": 1}, f"positive.{key}.public_pinned_spy")
        handoff = mapping(entry.get("public_handoff_prepare"), f"positive.{key}.public_handoff_prepare")
        require(set(handoff) == {"c1", "pinned"}, f"positive {key}: handoff spy path scope drift")
        c1_handoff = mapping(handoff.get("c1"), f"positive.{key}.handoff.c1")
        pinned_handoff = mapping(handoff.get("pinned"), f"positive.{key}.handoff.pinned")
        require(
            set(c1_handoff) == {"prepare_delta", "prepare_calls_total", "passed"}
            and type(c1_handoff.get("prepare_delta")) is int
            and c1_handoff.get("prepare_delta") == 1
            and type(c1_handoff.get("prepare_calls_total")) is int
            and c1_handoff.get("prepare_calls_total") == 1
            and c1_handoff.get("passed") is True,
            f"positive {key}: public C1 must prepare exactly once",
        )
        require(
            set(pinned_handoff) == {"prepare_delta", "prepare_calls_total", "c1_immutability", "pinned_immutability", "passed"}
            and type(pinned_handoff.get("prepare_delta")) is int
            and pinned_handoff.get("prepare_delta") == 0
            and type(pinned_handoff.get("prepare_calls_total")) is int
            and pinned_handoff.get("prepare_calls_total") == 1
            and pinned_handoff.get("passed") is True,
            f"positive {key}: pinned public path unexpectedly prepared C1 handoff",
        )
        audit[key] = {"expected_variant": expected_variant, "correctness": "exact", "route": "public_registry_c1", "immutability": True}
    return audit


def require_negative_simple(entry_value: object, label: str) -> None:
    entry = mapping(entry_value, f"negative_results.{label}")
    policy_fallback = label in POLICY_FALLBACK_C1_REASONS
    expected_entry_fields = {"c1_verifier", "pinned_verifier", "public_pinned_spy", "final", "passed"}
    if policy_fallback:
        expected_entry_fields |= {
            "direct_pinned_vs_torch_ref",
            "public_vs_direct_pinned",
            "public_vs_torch_ref",
            "input_immutability_exact",
            "input_immutability_fields",
            "input_immutability_by_path",
        }
    require(
        set(entry) == expected_entry_fields,
        f"negative {label}: field-set drift",
    )
    require(entry.get("passed") is True, f"negative {label}: failed")
    c1 = mapping(entry.get("c1_verifier"), f"negative.{label}.c1")
    pinned = mapping(entry.get("pinned_verifier"), f"negative.{label}.pinned")
    require(c1.get("passed") is False and pinned.get("passed") is True, f"negative {label}: verifier fallback drift")
    expected_rejection_reason = EXPECTED_C1_REJECTION_REASONS.get(label)
    if expected_rejection_reason is not None:
        require(
            set(c1) == {"passed", "reason"}
            and c1.get("reason") == expected_rejection_reason,
            f"negative {label}: C1 rejection reason drift",
        )
        require(
            set(pinned) == {"passed", "reason"} and pinned.get("reason") is None,
            f"negative {label}: pinned verifier reason drift",
        )
    require_spy(entry.get("public_pinned_spy"), {"c1": 0, "pinned": 1}, f"negative.{label}.public_pinned_spy")
    sequences, final_required = POLICY_FALLBACK_FINAL_CONTRACTS.get(label, (6, False))
    require_final_contract(entry.get("final"), f"negative.{label}.final", sequences=sequences, final_required=final_required)
    if policy_fallback:
        case, contract = label.split("/", 1)
        expected_offsets = OFFSETS_BY_CASE[case]
        expected_fields = set(COMMON_INPUT_FIELDS)
        if contract == "fp32_both":
            expected_fields.add("initial_state")
        require(entry.get("input_immutability_exact") is True, f"negative {label}: immutable-input evidence missing or failed")
        aggregate_fields = sequence(entry.get("input_immutability_fields"), f"negative.{label}.input_immutability_fields")
        paths = mapping(entry.get("input_immutability_by_path"), f"negative.{label}.input_immutability_by_path")
        require(set(paths) == {"torch_ref", "direct_pinned", "public_pinned"}, f"negative {label}: immutable-input route scope drift")
        observed_fields: set[str] = set()
        for path, evidence in paths.items():
            observed_fields.update(require_immutability(evidence, f"negative.{label}.immutability.{path}", expected_fields))
        require(aggregate_fields == sorted(expected_fields) and observed_fields == expected_fields, f"negative {label}: aggregate immutable-input fields drift")
        for comparison in ("direct_pinned_vs_torch_ref", "public_vs_direct_pinned", "public_vs_torch_ref"):
            require_exact(
                entry.get(comparison), f"negative.{label}.{comparison}",
                sequences=sequences,
                total_tokens=expected_offsets[-1],
                output_batch=1,
                final_required=final_required,
            )


def require_skew_none_decision(value: object, label: str, *, cache_hit: bool) -> None:
    """Validate every field emitted by auto_dispatch._record for this probe.

    The CPU-authoritative negative must not be reduced to an output-only
    check: it is also evidence that the public route selected the r6-frozen
    skew/none v2 path from the CPU offsets, rather than the caller's GPU data.
    """
    decision = mapping(value, label)
    offsets = OFFSETS_BY_CASE["skew_n6_h12_t12288"]
    require(set(decision) == RECORDED_DECISION_FIELDS, f"{label}: recorded-decision field-set drift")
    require(
        decision.get("requested_variant") == "vshard2_p2"
        and decision.get("chosen_variant") == "vshard2_p2"
        and decision.get("reason") == SKEW_NONE_REASON
        and decision.get("extension_sha256") == AUDITED_EXTENSION_SHA256
        and decision.get("varlen_cpu_authoritative") is True
        and decision.get("certified_varlen_offsets") == offsets
        and decision.get("canonical_cache_hit") is cache_hit,
        f"{label}: skew/none CPU-authoritative v2 decision drift",
    )


def require_negative_and_cache(data: Mapping[str, Any]) -> None:
    negatives = mapping(data.get("negative_results"), "negative_results")
    required = set(NEGATIVE_SIMPLE) | {"gpu_structural_mismatch_preflight", "allow_neg_eigval_semantic_fallback", "fixed_representative"}
    require(set(negatives) == required, "negative result scope drift")
    for label in NEGATIVE_SIMPLE:
        require_negative_simple(negatives.get(label), label)
    mismatch = mapping(negatives.get("gpu_structural_mismatch_preflight"), "negative.gpu_structural_mismatch_preflight")
    require(mismatch.get("passed") is True, "GPU mismatch preflight failed")
    require(mapping(mismatch.get("c1_verifier_malformed_gpu"), "mismatch.c1").get("passed") is False, "malformed GPU accepted")
    require(mapping(mismatch.get("pinned_verifier_valid_gpu"), "mismatch.pinned").get("passed") is True, "valid pinned fallback rejected")
    takeover = mapping(mismatch.get("pinned_direct_takeover_spy"), "mismatch.takeover")
    require(takeover.get("passed") is True and takeover.get("delta") == {"c1": 0, "pinned": 1}, "mismatch pinned takeover route drift")
    require_final_contract(mismatch.get("final"), "mismatch.final", sequences=6, final_required=False)
    allow = mapping(negatives.get("allow_neg_eigval_semantic_fallback"), "negative.allow_neg_eigval")
    require(allow.get("passed") is True and allow.get("allow_neg_eigval") is True, "allow-neg semantic gate failed")
    require(mapping(allow.get("c1_verifier"), "allow-neg.c1").get("passed") is True, "allow-neg was not captured by C1")
    require(mapping(allow.get("pinned_verifier"), "allow-neg.pinned").get("passed") is True, "allow-neg pinned baseline unavailable")
    for label in ("keyword_public_call", "positional_public_call"):
        call = mapping(allow.get(label), f"allow-neg.{label}")
        require(call.get("passed") is True, f"allow-neg {label} failed")
        require(call.get("registry_backend_delta") == {"c1": 1, "pinned": 0}, f"allow-neg {label}: registry route drift")
        require(call.get("kernel_launch_delta") == {"c1_raw_dispatch": 0, "pinned_raw_flash_kda": 0}, f"allow-neg {label}: raw launch occurred")
    fixed = mapping(negatives.get("fixed_representative"), "negative.fixed_representative")
    require(fixed.get("passed") is True and fixed.get("expected_variant") == "vshard4_p2", "fixed path regression")
    require_exact(
        fixed.get("direct_vs_pinned"), "fixed.direct_vs_pinned",
        sequences=2, total_tokens=2048, output_batch=2, final_required=False,
    )
    require_exact(
        fixed.get("public_vs_pinned"), "fixed.public_vs_pinned",
        sequences=2, total_tokens=2048, output_batch=2, final_required=False,
    )
    require_spy(fixed.get("public_spy"), {"c1": 1, "pinned": 0}, "fixed.public_spy")

    cache = mapping(data.get("cache_observations"), "cache_observations")
    require(set(cache) == {"cpu_canonical_gpu_values_ignored", "concurrency_and_capture", "hot_sync"}, "cache observation scope drift")
    semantic = mapping(cache.get("cpu_canonical_gpu_values_ignored"), "cache.cpu_canonical_gpu_values_ignored")
    offsets = OFFSETS_BY_CASE["skew_n6_h12_t12288"]
    require(
        set(semantic)
        == {
            "caller_gpu_values_equal_cpu",
            "caller_gpu_numel",
            "cpu_canonical_offsets",
            "direct_vs_cpu_canonical_torch_ref",
            "public_vs_cpu_canonical_torch_ref",
            "public_spy",
            "direct_decision",
            "public_decision",
            "passed",
        },
        "CPU-authoritative semantic field-set drift",
    )
    require(
        semantic.get("passed") is True
        and semantic.get("caller_gpu_values_equal_cpu") is False
        and semantic.get("caller_gpu_numel") == len(offsets)
        and semantic.get("cpu_canonical_offsets") == offsets,
        "CPU-authoritative semantic descriptor evidence drift",
    )
    require_exact(
        semantic.get("direct_vs_cpu_canonical_torch_ref"), "cache.semantic.direct",
        sequences=6, total_tokens=offsets[-1], output_batch=1, final_required=False,
    )
    require_exact(
        semantic.get("public_vs_cpu_canonical_torch_ref"), "cache.semantic.public",
        sequences=6, total_tokens=offsets[-1], output_batch=1, final_required=False,
    )
    require_spy(semantic.get("public_spy"), {"c1": 1, "pinned": 0}, "cache.semantic.public_spy")
    require_skew_none_decision(semantic.get("direct_decision"), "cache.semantic.direct_decision", cache_hit=False)
    require_skew_none_decision(semantic.get("public_decision"), "cache.semantic.public_decision", cache_hit=True)
    concurrency_and_capture = mapping(cache.get("concurrency_and_capture"), "cache.concurrency_and_capture")
    require(set(concurrency_and_capture) == {"two_stream_same_tuple", "capture"}, "concurrency/capture scope drift")
    two_stream = mapping(concurrency_and_capture.get("two_stream_same_tuple"), "cache.two_stream_same_tuple")
    require(two_stream.get("passed") is True and two_stream.get("first_cache_hit") is False and two_stream.get("second_cache_hit") is True, "two-stream cache route drift")
    stats = mapping(two_stream.get("stats"), "cache.two_stream.stats")
    require(stats.get("misses") == 1 and isinstance(stats.get("hits"), int) and int(stats["hits"]) >= 1, "two-stream cache counters drift")
    capture = mapping(concurrency_and_capture.get("capture"), "cache.capture")
    for label, error_name, counter in (("cold_miss", "CaptureCacheMissError", "capture_miss_rejections"), ("hot_hit", "CaptureCacheHitError", "capture_hit_rejections")):
        observation = mapping(capture.get(label), f"cache.capture.{label}")
        require(observation.get("passed") is True and observation.get("rejected") is True and observation.get("error_type") == error_name, f"capture {label} did not fail closed")
        require(observation.get("c1_pinned_backend_delta") == {"c1": 0, "pinned": 0}, f"capture {label}: backend entered")
        require(observation.get("c1_pinned_kernel_launch_delta") == {"c1_raw_dispatch": 0, "pinned_raw_flash_kda": 0}, f"capture {label}: raw launch")
        require(mapping(observation.get("stats"), f"capture.{label}.stats").get(counter) == 1, f"capture {label}: counter drift")
    clear = mapping(capture.get("clear_after_cold"), "cache.capture.clear_after_cold")
    require(clear.get("entries") == 0, "capture cache clear drift")
    warm = mapping(capture.get("warm_before_hot"), "cache.capture.warm_before_hot")
    require(warm.get("cache_hit") is False, "capture warm setup was unexpectedly a hit")
    hot_sync = mapping(cache.get("hot_sync"), "cache.hot_sync")
    require(hot_sync.get("passed") is True, "hot-sync evidence failed")
    returned, threshold = numeric(hot_sync.get("public_return_wall_s"), "hot_sync.return"), numeric(hot_sync.get("return_threshold_s"), "hot_sync.threshold")
    explicit, floor = numeric(hot_sync.get("explicit_sync_wall_s"), "hot_sync.sync"), numeric(hot_sync.get("sync_floor_s"), "hot_sync.floor")
    require(returned <= threshold and explicit >= floor, "hot-sync timing inequality failed")


def cross_check_runner_repeat(repeat_value: object, independent: Mapping[str, Any], label: str) -> None:
    repeat = mapping(repeat_value, label)
    stored_paths = mapping(repeat.get("paths"), f"{label}.paths")
    independent_paths = mapping(independent["summaries"], f"{label}.raw_paths")
    require(set(stored_paths) == set(PATHS), f"{label}: runner path-summary scope drift")
    for path in PATHS:
        stored = mapping(stored_paths.get(path), f"{label}.paths.{path}")
        raw = mapping(independent_paths[path], f"{label}.raw_paths.{path}")
        require(stored.get("samples") == raw["samples"], f"{label}.{path}.samples drift")
        for metric in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
            close(stored.get(metric), float(raw[metric]), f"{label}.{path}.{metric}")
    require(repeat.get("winner_by_percentile") == independent["winner_by_percentile"], f"{label}: runner winner drift")
    stored_margin = mapping(repeat.get("c1_margin_over_pinned_by_percentile"), f"{label}.margins")
    independent_margin = mapping(independent["c1_margin_over_pinned_by_percentile"], f"{label}.raw_margins")
    for name, _ in PERCENTILES:
        expected = independent_margin[name]
        actual = stored_margin.get(name)
        if expected is None:
            require(actual is None, f"{label}.{name}: winner-margin nullability drift")
        else:
            close(actual, float(expected), f"{label}.{name}")
    for name in ("winner_gate_pass", "margin_gate_pass", "repeat_gate_pass"):
        require(repeat.get(name) is independent[name], f"{label}.{name} drift")


def require_performance(data: Mapping[str, Any]) -> tuple[dict[str, object], list[str], list[str]]:
    performance = mapping(data.get("performance"), "performance")
    require(performance.get("complete") is True, "performance incomplete")
    prereg = mapping(performance.get("pre_registered"), "performance.pre_registered")
    require(prereg.get("cells") == list(POSITIVE), "performance cell preregistration drift")
    require(prereg.get("fixed_measurement_seed") == PERFORMANCE_SEED, "performance seed drift")
    require(prereg.get("paths") == list(PATHS), "performance path scope drift")
    require(prereg.get("repeats") == REPEATS and prereg.get("warmup_per_path_per_repeat") == WARMUP and prereg.get("cyclic_cuda_event_samples_per_path_per_repeat") == SAMPLES, "performance sampling contract drift")
    require(prereg.get("percentiles") == [name for name, _ in PERCENTILES], "performance percentile contract drift")
    close(prereg.get("minimum_c1_margin_over_pinned"), MIN_MARGIN, "performance minimum margin")
    require(prereg.get("prepare_spy_in_timed_region") is False, "r6 performance prepare-spy scope drift")
    expected_timed_calls = len(POSITIVE) * REPEATS * SAMPLES * len(PATHS)
    expected_warmup_calls = len(POSITIVE) * REPEATS * WARMUP * len(PATHS)
    expected_syncs = expected_timed_calls * 2
    require(
        prereg.get("timed_public_calls") == expected_timed_calls
        and prereg.get("warmup_public_calls") == expected_warmup_calls
        and prereg.get("timed_event_synchronizations") == expected_syncs,
        "r6 performance call/synchronization scope drift",
    )
    close(
        prereg.get("one_hour_mean_call_budget_ms_including_warmup"),
        3600_000 / (len(POSITIVE) * REPEATS * (SAMPLES + WARMUP) * len(PATHS)),
        "r6 performance one-hour budget",
    )
    cells = mapping(performance.get("cells"), "performance.cells")
    cold = mapping(performance.get("cold_miss_observations"), "performance.cold_miss_observations")
    require(set(cells) == set(POSITIVE) and set(cold) == set(POSITIVE), "performance/cold cell scope drift")
    audit: dict[str, object] = {}
    released: list[str] = []
    failed: list[str] = []
    for key, expected_variant in POSITIVE.items():
        _, contract = key.split("/", 1)
        expected_fields = set(COMMON_INPUT_FIELDS)
        if contract == "fp32_both":
            expected_fields.add("initial_state")
        observation = mapping(cold.get(key), f"performance.cold.{key}")
        require(observation.get("passed") is True and observation.get("not_part_of_performance_gate") is True, f"cold observation failed: {key}")
        require(observation.get("route_spy_delta") == {"c1": 1, "pinned": 0}, f"cold route drift: {key}")
        require_immutability(observation.get("input_immutability"), f"performance.cold.{key}.immutability", expected_fields)
        decision = mapping(observation.get("decision"), f"cold.{key}.decision")
        require(decision.get("chosen_variant") == expected_variant and decision.get("canonical_cache_hit") is False, f"cold cache decision drift: {key}")
        cell = mapping(cells.get(key), f"performance.cells.{key}")
        require(cell.get("expected_winner") == "public_registry_c1", f"performance expected winner drift: {key}")
        require(cell.get("input_immutability_exact") is True, f"performance immutable-input aggregate failed: {key}")
        aggregate_fields = sequence(cell.get("input_immutability_fields"), f"performance.{key}.input_immutability_fields")
        repeats = sequence(cell.get("repeats"), f"performance.cells.{key}.repeats")
        require(len(repeats) == REPEATS, f"performance repeat count drift: {key}")
        rows = [recompute_repeat(repeats[index], key, index) for index in range(REPEATS)]
        repeat_fields: set[str] = set()
        for index, row in enumerate(rows):
            cross_check_runner_repeat(repeats[index], row, f"performance.{key}.repeat{index}")
            repeat = mapping(repeats[index], f"performance.{key}.repeat{index}")
            require(repeat.get("event_contract") == EVENT_CONTRACT, f"performance event contract drift: {key}/repeat{index}")
            require(repeat.get("path_order") == PATH_ORDER, f"performance alternating path-order drift: {key}/repeat{index}")
            require(repeat.get("path_environment") == PATH_ENVIRONMENT, f"performance path-environment drift: {key}/repeat{index}")
            require(
                repeat.get("expected_route_delta_per_call") == EXPECTED_ROUTE_DELTA_PER_CALL,
                f"performance expected per-call route map drift: {key}/repeat{index}",
            )
            require(repeat.get("input_immutability_exact") is True, f"performance immutable-input evidence failed: {key}/repeat{index}")
            fields = sequence(repeat.get("input_immutability_fields"), f"performance.{key}.repeat{index}.input_immutability_fields")
            require(fields == sorted(expected_fields), f"performance immutable-input coverage drift: {key}/repeat{index}")
            repeat_fields.update(str(field) for field in fields)
            require(repeat.get("warmup_route_spy_delta") == {"c1": WARMUP, "pinned": WARMUP}, f"performance warmup route drift: {key}")
            require(repeat.get("timed_route_spy_delta") == {"c1": SAMPLES, "pinned": SAMPLES}, f"performance timed route drift: {key}")
            require(
                repeat.get("per_sample_route_spy_assertions") == {
                    "public_registry_c1": SAMPLES,
                    "public_registry_pinned": SAMPLES,
                    "passed": True,
                },
                f"performance per-sample route evidence drift: {key}",
            )
        require(aggregate_fields == sorted(expected_fields) and repeat_fields == expected_fields, f"performance immutable-input aggregate fields drift: {key}")
        passing = all(bool(row["repeat_gate_pass"]) for row in rows)
        require(cell.get("performance_release") is passing, f"performance release bool drift: {key}")
        if passing:
            released.append(key)
        else:
            failed.append(key)
        audit[key] = {"expected_variant": expected_variant, "repeats": rows, "performance_release": passing}
    require(performance.get("performance_release_cells") == released, "nested performance release list drift")
    require(performance.get("failed_cells") == failed, "nested performance failed list drift")
    require(data.get("performance_release_cells") == released, "top-level performance release list drift")
    require(data.get("performance_failed_cells") == failed, "top-level performance failed list drift")
    require(released == list(POSITIVE) and failed == [], "r6 production freeze requires exactly both released cells and no failures")
    return audit, released, failed


def audit(path: Path, expected_sha256: str) -> dict[str, object]:
    data, digest = read_artifact(path, expected_sha256)
    require_identity_and_gates(data)
    positive = require_positive(data)
    require_negative_and_cache(data)
    performance, released, failed = require_performance(data)
    return {
        "audit_schema_version": 1,
        "artifact": str(path.resolve()),
        "artifact_sha256": digest,
        "independent_audit_pass": True,
        "positive_correctness_route_and_immutability": positive,
        "negative_capture_multistream_hot_sync_and_fixed_regression": "passed",
        "performance_cells": performance,
        "independently_performance_release_cells": released,
        "independently_performance_failed_cells": failed,
    }


def atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_audit_output(path: Path, inputs: tuple[Path, ...]) -> None:
    """Reject overwrite-shaped audit output before any write is attempted."""
    require(path.suffix == ".json", "audit output must use a .json suffix")
    target = path.resolve()
    require(all(target != item.resolve() for item in inputs), "audit output resolves to an input artifact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="completed public-FLA integration JSON")
    parser.add_argument("--expected-sha256", required=True, help="exact SHA256 of the integration JSON")
    parser.add_argument("--json", type=Path, required=True, help="atomically written independent audit JSON")
    args = parser.parse_args()
    try:
        validate_audit_output(args.json, (args.artifact,))
        result = audit(args.artifact, args.expected_sha256)
    except (AuditError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    atomic_write(args.json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
