#!/usr/bin/env python3
"""Fail-closed, raw-sample-only audit for a packed-varlen release artifact.

The release result contains only *derived* historical evidence.  Therefore
this program requires the three frozen historical JSON artifacts as inputs
(or reads the exact paths recorded in the release preregistration).  It never
uses a runner winner, percentile, margin, or release-list field to decide the
audit outcome.
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


SCHEMA_VERSION = 1
RELEASE_SEED = 20260830
REPEATS = 2
NEW_SAMPLES = 1000
SEQCOUNT_SAMPLES = 1000
MIXED_SAMPLES = 300
MIN_MARGIN = 0.02
PERCENTILES = (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
VARIANTS = ("baseline", "vshard2_p2", "vshard4_p2")
RAW_CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
PUBLIC_CONTRACTS = ("none", "fp32_final_only", "fp32_both")

AUDITED_EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
HELPER_SHA256 = "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
SEQCOUNT_SHA256 = "46cd27f2fbdcaeeb61011c49c6175a0c05d15d4365bfda800cf52040dbe414f7"
MIXED_SHA256 = "b2dae8d42f43c3e42c44ca20fdc2c8443ec8b6b1b1ff2b81aff74be5b877fcd3"
CONFIRMATION_SHA256 = "447d7f49a624fa5b92adc431b350450f99d53f5b20f3a07a1bf4d2f76a64e51c"
CONFIRMATION_RUNNER_SHA256 = "9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b"
RELEASE_RUNNER_SHA256 = "1e8ff86e79683dd3b1266abe2013e7cec8c95b6b099d4c315ecb79419b2d2a42"
RELEASE_RUNNER_SHA256_ENV = "C1_VARLEN_DISPATCH_RELEASE_RUNNER_SHA256"
PINNED_TORCH_REF_SHA256 = "bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5"
PATCHED_FLASH_KDA_INIT_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
SHARED_SEQCOUNT_RUNNER_SHA256 = "4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f"
VARLEN_TAIL_RUNNER_SHA256 = "ff771c0b2f1b66f3062bc310c14634bf23830f706aec39f1b8ff03ff8b567621"
PREFETCH2_SHA256 = "752126488487ac317a7ee167b660b0895562e4877aedbc4ce2a599c1f59a10d0"
VSHARD4_PREFETCH2_SHA256 = "445c90815919fb8c1db3bc79289d10029ea77b1d9accb393c326c120ba1f8385"
HARNESS_SHA256 = "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"

# The release runner constructs exactly this identity ledger before importing
# or exercising the kernels.  Do not accept a subset or an additional,
# unreviewed identity entry: both are evidence-schema drift.
RUNTIME_IMPORT_IDENTITIES = {
    "confirmation_runner": (
        CONFIRMATION_RUNNER_SHA256,
        "assignment02/team/c1_flashkda/challenge_varlen_dispatch/run_varlen_dispatch_confirmation.py",
    ),
    "shared_seqcount_runner": (
        SHARED_SEQCOUNT_RUNNER_SHA256,
        "assignment02/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py",
    ),
    "varlen_tail_runner": (
        VARLEN_TAIL_RUNNER_SHA256,
        "assignment02/team/c1_flashkda/challenge_varlen_tail/run_varlen_tail.py",
    ),
    "prefetch2": (
        PREFETCH2_SHA256,
        "assignment02/team/c1_flashkda/challenge_prefetch2/prefetch2.py",
    ),
    "vshard4_prefetch2": (
        VSHARD4_PREFETCH2_SHA256,
        "assignment02/team/c1_flashkda/challenge_vshard4_prefetch2/vshard4_prefetch2.py",
    ),
    "harness": (
        HARNESS_SHA256,
        "assignment02/team/c1_flashkda/harness/validate_and_bench.py",
    ),
    "pinned_torch_ref": (PINNED_TORCH_REF_SHA256, "tests/torch_ref.py"),
    "patched_flash_kda_init": (PATCHED_FLASH_KDA_INIT_SHA256, "flash_kda/__init__.py"),
    "pinned_reference_helper": (HELPER_SHA256, "sigmoid_ext/sigmoid_ext.so"),
}

PROMOTION_CELLS: dict[str, dict[str, str]] = {
    "equal_n2_h12_t2048": {
        "none": "vshard4_p2", "fp32_final_only": "vshard4_p2", "fp32_both": "vshard4_p2",
    },
    "equal_n4_h12_t2048": {
        "none": "vshard2_p2", "fp32_final_only": "vshard2_p2", "fp32_both": "vshard4_p2",
    },
    "mixed_n6_h12_t8192": {"none": "vshard2_p2", "fp32_final_only": "vshard2_p2"},
    "skew_n6_h12_t12288": {
        "none": "vshard2_p2", "fp32_final_only": "vshard2_p2", "fp32_both": "vshard4_p2",
    },
}
CASE_LENGTHS = {
    "equal_n2_h12_t2048": [2048, 2048],
    "equal_n4_h12_t2048": [2048, 2048, 2048, 2048],
    "mixed_n6_h12_t8192": [17, 511, 1024, 1300, 2049, 3291],
    "skew_n6_h12_t12288": [1, 1, 1, 1, 1, 12283],
}
RECORD_ONLY_KEY = "mixed_n6_h12_t8192/fp32_both"
DISCOVERY_SOURCE = {
    "equal_n2_h12_t2048": ("seqcount", "m024_n02_h12_balanced_varlen"),
    "equal_n4_h12_t2048": ("seqcount", "m048_n04_h12_balanced_varlen"),
    "mixed_n6_h12_t8192": ("mixed", "varlen_mixed_t8192"),
    "skew_n6_h12_t12288": ("seqcount", "m072_n06_h12_skewed_varlen"),
}


class AuditError(AssertionError):
    """An artifact cannot safely support the advertised release result."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def mapping(value: object, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be a JSON object")
    return value


def sequence(value: object, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a JSON array")
    return value


def exact_bool(value: object, label: str) -> bool:
    require(isinstance(value, bool), f"{label} must be boolean")
    return value


def finite_number(value: object, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    converted = float(value)
    require(math.isfinite(converted), f"{label} must be finite")
    return converted


def read_json(path: Path, expected_sha256: str, label: str) -> tuple[Mapping[str, Any], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AuditError(f"cannot read {label}: {path}") from exc
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == expected_sha256, f"{label} SHA256 mismatch: {digest}")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AuditError(f"{label} is invalid JSON") from exc
    return mapping(parsed, label), digest


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def recompute_benchmark(benchmark: object, expected: str | None, label: str, samples: int) -> dict[str, object]:
    raw = mapping(mapping(benchmark, label).get("raw_samples_ms"), f"{label}.raw_samples_ms")
    require(set(raw) == set(VARIANTS), f"{label}: raw paths must be exactly {VARIANTS}")
    summaries: dict[str, dict[str, object]] = {}
    for variant in VARIANTS:
        raw_values = sequence(raw[variant], f"{label}.{variant}")
        require(len(raw_values) == samples, f"{label}.{variant}: expected exactly {samples} raw samples")
        values = [finite_number(value, f"{label}.{variant}[{index}]") for index, value in enumerate(raw_values)]
        require(all(value > 0.0 for value in values), f"{label}.{variant}: samples must be positive")
        summaries[variant] = {
            "samples": samples,
            "mean_ms": statistics.fmean(values),
            "p50_ms": percentile(values, 0.50),
            "p95_ms": percentile(values, 0.95),
            "p99_ms": percentile(values, 0.99),
        }
    winners: dict[str, str] = {}
    margins: dict[str, float] = {}
    for name, _ in PERCENTILES:
        ranked = sorted((float(summaries[variant][f"{name}_ms"]), variant) for variant in VARIANTS)
        winners[name] = ranked[0][1]
        margins[name] = ranked[1][0] / ranked[0][0] - 1.0
    winner_pass = expected is not None and all(winners[name] == expected for name, _ in PERCENTILES)
    margin_pass = expected is not None and all(margins[name] >= MIN_MARGIN for name, _ in PERCENTILES)
    return {
        "recomputed_from_raw_samples": True,
        "expected_raw_sample_count": samples,
        "expected_winner": expected,
        "summaries": summaries,
        "winner_by_percentile": winners,
        "runner_up_margin_by_percentile": margins,
        "expected_winner_at_all_percentiles": winner_pass,
        "margin_gate_pass": margin_pass,
        "gate_pass": winner_pass and margin_pass,
    }


def close_float(actual: object, expected: float, label: str) -> None:
    require(math.isclose(finite_number(actual, label), expected, rel_tol=0.0, abs_tol=1e-12), f"{label}: runner summary disagrees with raw recomputation")


def verify_runner_gate(record: object, independent: Mapping[str, Any], label: str) -> None:
    """Cross-check stored derived evidence, but never derive a decision from it."""
    stored = mapping(record, label)
    require(stored.get("expected_winner") == independent["expected_winner"], f"{label}.expected_winner drift")
    require(stored.get("winner_by_percentile") == independent["winner_by_percentile"], f"{label}.winner_by_percentile drift")
    stored_summaries = mapping(stored.get("summaries"), f"{label}.summaries")
    independent_summaries = mapping(independent["summaries"], f"{label}.independent_summaries")
    require(set(stored_summaries) == set(VARIANTS), f"{label}.summary variant scope drift")
    for variant in VARIANTS:
        stored_summary = mapping(stored_summaries.get(variant), f"{label}.summaries.{variant}")
        independent_summary = mapping(independent_summaries[variant], f"{label}.independent_summaries.{variant}")
        require(stored_summary.get("samples") == independent_summary["samples"], f"{label}.{variant}.samples drift")
        for metric in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
            close_float(stored_summary.get(metric), float(independent_summary[metric]), f"{label}.{variant}.{metric}")
    stored_margin = mapping(stored.get("runner_up_margin_by_percentile"), f"{label}.runner_up_margin_by_percentile")
    independent_margin = mapping(independent["runner_up_margin_by_percentile"], f"{label}.independent_margin")
    for name, _ in PERCENTILES:
        close_float(stored_margin.get(name), float(independent_margin[name]), f"{label}.margin.{name}")
    require(stored.get("gate_pass") is independent["gate_pass"], f"{label}.gate_pass drift")


def require_identity(identity_value: object, label: str) -> None:
    identity = mapping(identity_value, label)
    device = mapping(identity.get("device"), f"{label}.device")
    require("B300" in str(device.get("name", "")).upper(), f"{label}: not B300")
    require(device.get("capability") == [10, 3], f"{label}: capability drift")
    require(device.get("multiprocessor_count") == 148, f"{label}: SM count drift")
    extension = mapping(identity.get("extension"), f"{label}.extension")
    require(extension.get("sha256") == AUDITED_EXTENSION_SHA256, f"{label}: extension SHA drift")


def require_exact_record(value: object, label: str, *, final_state_expected: bool) -> None:
    record = mapping(value, label)
    expected_fields = (
        {"output_exact", "output_max_abs", "final_state_present"}
        if not final_state_expected
        else {"output_exact", "output_max_abs", "final_state_present", "final_state_exact", "final_state_max_abs"}
    )
    require(set(record) == expected_fields, f"{label}: exactness field-set drift")
    require(not any(str(key).startswith("final_eig") for key in record), f"{label}: unsupported final_eig field present")
    require(record.get("output_exact") is True, f"{label}: output exactness failed")
    require(finite_number(record.get("output_max_abs"), f"{label}.output_max_abs") == 0.0, f"{label}: nonzero output delta")
    if final_state_expected:
        require(record.get("final_state_present") is True, f"{label}: final state unexpectedly absent")
        require(record.get("final_state_exact") is True, f"{label}: final exactness failed")
        require(finite_number(record.get("final_state_max_abs"), f"{label}.final_state_max_abs") == 0.0, f"{label}: nonzero final delta")
    else:
        require(record.get("final_state_present") is False, f"{label}: final state unexpectedly present")


def require_no_final_eig(value: object, label: str) -> None:
    """The pinned raw ABI has no final-eigenvalue output; reject invented fields."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            require(not str(key).startswith("final_eig"), f"{label}: unsupported final_eig field {key!r}")
            require_no_final_eig(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            require_no_final_eig(child, f"{label}[{index}]")


def require_tensor_contract(value: object, case: str, contract: str, label: str) -> None:
    evidence = mapping(value, label)
    expected_fields = {
        "case", "contract", "lengths", "canonical_offsets", "observed_gpu_offsets", "sequence_count",
        "cu_seqlens_shape", "cu_seqlens_dtype", "cu_seqlens_contiguous", "expected_output_shape",
        "expected_output_dtype", "expected_final_state_present", "expected_final_state_shape",
        "expected_final_state_dtype", "implementations", "passed",
    }
    require(set(evidence) == expected_fields, f"{label}: tensor-contract field-set drift")
    lengths = CASE_LENGTHS[case]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    sequences, total = len(lengths), offsets[-1]
    require(
        evidence.get("case") == case
        and evidence.get("contract") == contract
        and evidence.get("lengths") == lengths
        and evidence.get("canonical_offsets") == offsets
        and evidence.get("observed_gpu_offsets") == offsets
        and evidence.get("sequence_count") == sequences
        and evidence.get("cu_seqlens_shape") == [sequences + 1]
        and evidence.get("cu_seqlens_dtype") == "torch.int64"
        and evidence.get("cu_seqlens_contiguous") is True
        and evidence.get("expected_output_shape") == [1, total, 12, 128]
        and evidence.get("expected_output_dtype") == "torch.bfloat16"
        and evidence.get("passed") is True,
        f"{label}: canonical offset/output tensor contract drift",
    )
    final_required = contract != "none"
    final_dtype = "torch.bfloat16" if contract == "bf16_both" else "torch.float32"
    require(
        evidence.get("expected_final_state_present") is final_required
        and evidence.get("expected_final_state_shape") == ([sequences, 12, 128, 128] if final_required else None)
        and evidence.get("expected_final_state_dtype") == (final_dtype if final_required else None),
        f"{label}: expected final-state contract drift",
    )
    implementations = mapping(evidence.get("implementations"), f"{label}.implementations")
    expected_implementations = {"baseline", "vshard2_p2", "vshard4_p2"}
    if contract in PUBLIC_CONTRACTS:
        expected_implementations.add("pinned_torch_ref")
    require(set(implementations) == expected_implementations, f"{label}: implementation scope drift")
    for implementation, implementation_value in implementations.items():
        actual = mapping(implementation_value, f"{label}.implementations.{implementation}")
        expected_actual_fields = {"output_shape", "output_dtype", "output_contiguous", "final_state_present", "passed"}
        if final_required:
            expected_actual_fields |= {"final_state_shape", "final_state_dtype", "final_state_contiguous"}
        require(set(actual) == expected_actual_fields, f"{label}.{implementation}: implementation tensor field-set drift")
        require(
            actual.get("output_shape") == [1, total, 12, 128]
            and actual.get("output_dtype") == "torch.bfloat16"
            and actual.get("output_contiguous") is True
            and actual.get("final_state_present") is final_required
            and actual.get("passed") is True,
            f"{label}.{implementation}: output tensor contract drift",
        )
        if final_required:
            require(
                actual.get("final_state_shape") == [sequences, 12, 128, 128]
                and actual.get("final_state_dtype") == final_dtype
                and actual.get("final_state_contiguous") is True,
                f"{label}.{implementation}: final tensor contract drift",
            )
    require_no_final_eig(evidence, label)


def require_raw_exactness(
    raw_value: object,
    label: str,
    *,
    required_cases: set[str] | None = None,
    exact_scope: bool = True,
    immutability_required: bool = True,
    pinned_reference_required: bool = True,
    tensor_contract_required: bool = True,
) -> None:
    raw = mapping(raw_value, label)
    cases = set(PROMOTION_CELLS) if required_cases is None else required_cases
    if exact_scope:
        require(set(raw) == cases, f"{label}: raw correctness scope drift")
    else:
        require(cases.issubset(raw), f"{label}: required raw correctness case is missing")
    for case in cases:
        contracts = mapping(raw.get(case), f"{label}.{case}")
        require(set(contracts) == set(RAW_CONTRACTS), f"{label}.{case}: raw contract scope drift")
        for contract in RAW_CONTRACTS:
            entry = mapping(contracts.get(contract), f"{label}.{case}.{contract}")
            expected_entry_fields = {"vshard2_p2", "vshard4_p2", "tensor_contract"}
            if immutability_required:
                expected_entry_fields.add("input_immutability_exact")
            if contract in PUBLIC_CONTRACTS and pinned_reference_required:
                expected_entry_fields.add("baseline_vs_pinned_torch_ref")
            if tensor_contract_required:
                require(set(entry) == expected_entry_fields, f"{label}.{case}.{contract}: raw-record field-set drift")
            if immutability_required:
                require(entry.get("input_immutability_exact") is True, f"{label}.{case}.{contract}: input mutation")
            expected_final = contract != "none"
            for variant in ("vshard2_p2", "vshard4_p2"):
                require_exact_record(
                    entry.get(variant), f"{label}.{case}.{contract}.{variant}", final_state_expected=expected_final
                )
            if tensor_contract_required:
                require_tensor_contract(
                    entry.get("tensor_contract"), case, contract, f"{label}.{case}.{contract}.tensor_contract"
                )
            if contract in PUBLIC_CONTRACTS and pinned_reference_required:
                require_exact_record(
                    entry.get("baseline_vs_pinned_torch_ref"),
                    f"{label}.{case}.{contract}.baseline_vs_pinned_torch_ref",
                    final_state_expected=expected_final,
                )
            require_no_final_eig(entry, f"{label}.{case}.{contract}")


def locate_frozen(explicit: Path | None, prereg: Mapping[str, Any], name: str) -> Path:
    if explicit is not None:
        return explicit
    frozen = mapping(prereg.get("frozen_artifacts"), "preregistration.frozen_artifacts")
    entry = mapping(frozen.get(name), f"preregistration.frozen_artifacts.{name}")
    path_value = entry.get("path")
    require(isinstance(path_value, str) and path_value, f"frozen {name} path is missing")
    return Path(path_value)


def verify_release_preregistration(data: Mapping[str, Any]) -> Mapping[str, Any]:
    require(data.get("schema_version") == SCHEMA_VERSION, "release schema version drift")
    require(data.get("complete") is True, "release artifact is incomplete")
    prereg = mapping(data.get("preregistration"), "preregistration")
    require(prereg.get("release_seed") == RELEASE_SEED, "release seed drift")
    require(prereg.get("warmup_per_path") == 100, "warmup drift")
    require(prereg.get("samples_per_repeat") == NEW_SAMPLES, "new sample count drift")
    require(prereg.get("repeats") == REPEATS, "new repeat count drift")
    require(prereg.get("percentiles") == [name for name, _ in PERCENTILES], "percentile contract drift")
    close_float(prereg.get("minimum_runner_up_margin"), MIN_MARGIN, "preregistration.minimum_runner_up_margin")
    require(prereg.get("eligible_cells") == PROMOTION_CELLS, "eligible mapping drift")
    require(prereg.get("eligible_cell_count") == 11, "eligible scope count drift")
    record = mapping(prereg.get("record_only"), "preregistration.record_only")
    require(record.get("case") == "mixed_n6_h12_t8192" and record.get("contract") == "fp32_both", "record-only scope drift")
    frozen = mapping(prereg.get("frozen_artifacts"), "preregistration.frozen_artifacts")
    expected = {"seqcount_discovery": SEQCOUNT_SHA256, "mixed_discovery": MIXED_SHA256, "confirmation": CONFIRMATION_SHA256}
    require(set(frozen) == set(expected), "frozen artifact scope drift")
    for name, digest in expected.items():
        require(mapping(frozen[name], f"frozen.{name}").get("sha256") == digest, f"frozen {name} SHA declaration drift")
    require(prereg.get("confirmation_runner_sha256") == CONFIRMATION_RUNNER_SHA256, "confirmation runner identity drift")
    return prereg


def verify_release_identity_and_gates(data: Mapping[str, Any]) -> None:
    gates = mapping(data.get("gates"), "gates")
    required = (
        "scope_count", "confirmation_runner_identity", "historical_artifacts", "clean_gpu_shell_gate",
        "clean_gpu_runner_preflight_gate", "device_gate", "audited_extension_sha256_gate",
        "current_raw_abi_exact_gate", "pinned_torch_reference_gate", "runtime_import_identity_gate",
    )
    require(set(gates) == set(required), "release gate scope drift")
    for name in required:
        require(mapping(gates[name], f"gates.{name}").get("passed") is True, f"required gate failed: {name}")
    scope = mapping(gates["scope_count"], "gates.scope_count")
    require(scope.get("required") == 11 and scope.get("actual") == 11, "scope-count gate drift")
    identity = mapping(data.get("identity"), "identity")
    release_runner = mapping(identity.get("release_runner"), "identity.release_runner")
    require(
        set(release_runner) == {"path", "sha256", "sha256_gate_pass", "expected_sha256_environment"},
        "release-runner identity field-set drift",
    )
    require(
        release_runner.get("sha256") == RELEASE_RUNNER_SHA256
        and release_runner.get("sha256_gate_pass") is True
        and release_runner.get("expected_sha256_environment") == RELEASE_RUNNER_SHA256_ENV,
        "release-runner SHA/environment drift",
    )
    release_runner_path = release_runner.get("path")
    require(
        isinstance(release_runner_path, str)
        and release_runner_path.replace("\\", "/").rstrip("/").endswith(
            "assignment02/team/c1_flashkda/challenge_varlen_dispatch/run_varlen_dispatch_release.py"
        ),
        "release-runner path drift",
    )
    require_identity(identity, "identity")
    shell = mapping(identity.get("clean_gpu_shell_gate"), "identity.clean_gpu_shell_gate")
    runner = mapping(identity.get("clean_gpu_runner_preflight_gate"), "identity.clean_gpu_runner_preflight_gate")
    require(shell.get("passed") is True and runner.get("passed") is True, "clean GPU identity evidence failed")
    helper = mapping(identity.get("pinned_torch_reference_helper"), "identity.pinned_torch_reference_helper")
    require(helper.get("sha256") == HELPER_SHA256 and helper.get("no_build") is True, "pinned reference helper drift")
    runtime = mapping(identity.get("runtime_import_identities"), "identity.runtime_import_identities")
    require(set(runtime) == set(RUNTIME_IMPORT_IDENTITIES), "runtime import identity key-set drift")
    for name, (expected_sha256, expected_suffix) in RUNTIME_IMPORT_IDENTITIES.items():
        entry = mapping(runtime.get(name), f"identity.runtime_import_identities.{name}")
        require(entry.get("sha256_gate_pass") is True, f"runtime identity gate failed: {name}")
        require(entry.get("sha256") == expected_sha256, f"runtime identity SHA drift: {name}")
        path = entry.get("path")
        require(isinstance(path, str) and path, f"runtime identity path missing: {name}")
        normalized_path = path.replace("\\", "/").rstrip("/")
        require(normalized_path.endswith(expected_suffix), f"runtime identity path drift: {name}")


def require_historical_identity(data: Mapping[str, Any], label: str) -> None:
    identity = mapping(data.get("identity"), f"{label}.identity") if "identity" in data else data
    # Discovery artifacts may place the device fields one level below identity.
    device_value = identity.get("device")
    device = mapping(device_value, f"{label}.device") if isinstance(device_value, Mapping) else None
    name = str(device.get("name", "")) if device is not None else str(data.get("device", ""))
    capability = device.get("capability") if device is not None else data.get("capability")
    sms = device.get("multiprocessor_count") if device is not None else data.get("multiprocessor_count")
    extension = mapping(identity.get("extension", data.get("extension")), f"{label}.extension")
    require("B300" in name.upper() and capability == [10, 3] and sms == 148, f"{label}: B300 identity drift")
    require(extension.get("sha256") == AUDITED_EXTENSION_SHA256, f"{label}: extension identity drift")


def confirmation_benchmark(confirmation: Mapping[str, Any], case: str, contract: str, repeat: int) -> object:
    benchmarks = mapping(confirmation.get("raw_wrapper_public_contract_benchmarks"), "confirmation.benchmarks")
    cell = mapping(mapping(benchmarks.get(case), f"confirmation.benchmarks.{case}").get(contract), f"confirmation.{case}.{contract}")
    repeats = sequence(cell.get("repeats"), f"confirmation.{case}.{contract}.repeats")
    require(len(repeats) == REPEATS, f"confirmation {case}/{contract}: repeat count drift")
    item = mapping(repeats[repeat], f"confirmation.{case}.{contract}.repeat{repeat}")
    require(item.get("repeat_index") == repeat, f"confirmation {case}/{contract}: repeat index drift")
    require(item.get("input_immutability_exact") is True, f"confirmation {case}/{contract}: input mutation")
    return item.get("benchmark")


def validate_confirmation_artifact(data: Mapping[str, Any]) -> None:
    require(data.get("complete") is True, "confirmation artifact incomplete")
    prereg = mapping(data.get("preregistration"), "confirmation.preregistration")
    require(prereg.get("promotion_cells") == PROMOTION_CELLS and prereg.get("promotion_cell_count") == 11, "confirmation scope drift")
    require(prereg.get("repeats_per_public_cell") == REPEATS and prereg.get("cyclic_cuda_event_samples_per_repeat") == NEW_SAMPLES, "confirmation timing scope drift")
    gates = mapping(data.get("gates"), "confirmation.gates")
    for name in ("clean_gpu_shell_gate", "device_gate", "audited_extension_sha256_gate", "raw_abi_exact_gate", "pinned_torch_reference_gate", "confirmation_gate"):
        require(mapping(gates.get(name), f"confirmation.gates.{name}").get("passed") is True, f"confirmation gate failed: {name}")
    require_historical_identity(data, "confirmation")
    helper = mapping(mapping(data.get("identity"), "confirmation.identity").get("pinned_torch_reference_helper"), "confirmation.helper")
    require(helper.get("sha256") == HELPER_SHA256 and helper.get("no_build") is True, "confirmation helper drift")
    require_raw_exactness(
        data.get("raw_abi_correctness"), "confirmation.raw_abi_correctness", tensor_contract_required=False
    )


def discovery_benchmark(data: Mapping[str, Any], source_case: str, contract: str, label: str) -> object:
    benchmarks = mapping(data.get("benchmark"), f"{label}.benchmark")
    cell = mapping(benchmarks.get(source_case), f"{label}.benchmark.{source_case}")
    return cell.get(contract)


def validate_discovery_artifact(data: Mapping[str, Any], label: str, *, complete_required: bool, required_cases: set[str]) -> None:
    if complete_required:
        require(data.get("complete") is True, f"{label}: incomplete")
    require(data.get("exact_gate_pass") is True, f"{label}: exact gate failed")
    require_historical_identity(data, label)
    # Discovery runners legitimately contain their own case names and may
    # include unrelated exploratory cells.  Their *relevant* source cases
    # still need every raw ABI contract and bitwise result.
    require_raw_exactness(
        data.get("correctness"), f"{label}.correctness", required_cases=required_cases,
        exact_scope=False, immutability_required=False, pinned_reference_required=False, tensor_contract_required=False,
    )


def audit(path: Path, expected_sha256: str, seqcount_path: Path | None, mixed_path: Path | None, confirmation_path: Path | None) -> dict[str, object]:
    release, release_digest = read_json(path, expected_sha256, "release artifact")
    prereg = verify_release_preregistration(release)
    verify_release_identity_and_gates(release)
    require_raw_exactness(release.get("raw_abi_correctness"), "release.raw_abi_correctness")

    seqcount, seq_digest = read_json(locate_frozen(seqcount_path, prereg, "seqcount_discovery"), SEQCOUNT_SHA256, "seqcount discovery")
    mixed, mixed_digest = read_json(locate_frozen(mixed_path, prereg, "mixed_discovery"), MIXED_SHA256, "mixed discovery")
    confirmation, confirmation_digest = read_json(locate_frozen(confirmation_path, prereg, "confirmation"), CONFIRMATION_SHA256, "confirmation")
    validate_discovery_artifact(
        seqcount, "seqcount discovery", complete_required=True,
        required_cases={case for source, case in DISCOVERY_SOURCE.values() if source == "seqcount"},
    )
    validate_discovery_artifact(
        mixed, "mixed discovery", complete_required=False,
        required_cases={case for source, case in DISCOVERY_SOURCE.values() if source == "mixed"},
    )
    validate_confirmation_artifact(confirmation)
    discoveries = {"seqcount": seqcount, "mixed": mixed}

    new_measurements = mapping(release.get("new_measurements"), "release.new_measurements")
    require(set(new_measurements) == set(PROMOTION_CELLS), "new measurement case scope drift")
    history_rows: dict[str, object] = {}
    expected_released: dict[str, str] = {}
    expected_fallback: dict[str, str] = {}
    stored_history = mapping(release.get("history_evidence"), "release.history_evidence")
    stored_history_cells = mapping(stored_history.get("cells"), "release.history_evidence.cells")
    require(set(stored_history_cells) == {f"{case}/{contract}" for case, contracts in PROMOTION_CELLS.items() for contract in contracts}, "stored history cell scope drift")

    for case, expected_contracts in PROMOTION_CELLS.items():
        new_contracts = mapping(new_measurements.get(case), f"new_measurements.{case}")
        require(set(new_contracts) == set(PUBLIC_CONTRACTS), f"new measurement contract scope drift: {case}")
        source_name, source_case = DISCOVERY_SOURCE[case]
        source = discoveries[source_name]
        source_samples = SEQCOUNT_SAMPLES if source_name == "seqcount" else MIXED_SAMPLES
        for contract in PUBLIC_CONTRACTS:
            key = f"{case}/{contract}"
            expected = expected_contracts.get(contract)
            cell = mapping(new_contracts.get(contract), f"new_measurements.{key}")
            require(cell.get("expected_winner") == expected, f"new_measurements.{key}: expected winner drift")
            require(cell.get("record_only") is (expected is None), f"new_measurements.{key}: record-only drift")
            repeats = sequence(cell.get("repeats"), f"new_measurements.{key}.repeats")
            require(len(repeats) == REPEATS, f"new_measurements.{key}: repeat count drift")
            new_rows: list[dict[str, object]] = []
            for index, item_value in enumerate(repeats):
                item = mapping(item_value, f"new_measurements.{key}.repeat{index}")
                require(item.get("repeat_index") == index, f"new_measurements.{key}: repeat-index drift")
                require(item.get("input_immutability_exact") is True, f"new_measurements.{key}: input mutation")
                independent = recompute_benchmark(item.get("benchmark"), expected, f"new/{key}/repeat{index}", NEW_SAMPLES)
                if expected is not None:
                    verify_runner_gate(item.get("recomputed_gate"), independent, f"new/{key}/repeat{index}.recomputed_gate")
                new_rows.append(independent)
            if expected is None:
                require(key == RECORD_ONLY_KEY, f"unexpected unpromotable cell: {key}")
                continue

            discovery = recompute_benchmark(discovery_benchmark(source, source_case, contract, f"{source_name}/{key}"), expected, f"discovery/{key}", source_samples)
            confirmation_rows = [
                recompute_benchmark(confirmation_benchmark(confirmation, case, contract, index), expected, f"confirmation/{key}/repeat{index}", NEW_SAMPLES)
                for index in range(REPEATS)
            ]
            history = mapping(stored_history_cells.get(key), f"history_evidence.cells.{key}")
            require(history.get("expected_winner") == expected and history.get("discovery_source") == source_name, f"history evidence identity drift: {key}")
            verify_runner_gate(history.get("discovery_single_repeat"), discovery, f"history.{key}.discovery")
            stored_confirmations = sequence(history.get("confirmation_two_repeats"), f"history.{key}.confirmation")
            require(len(stored_confirmations) == REPEATS, f"history {key}: confirmation repeat scope drift")
            for index, independent in enumerate(confirmation_rows):
                verify_runner_gate(stored_confirmations[index], independent, f"history.{key}.confirmation{index}")
            historical_pass = bool(discovery["gate_pass"]) and all(bool(row["gate_pass"]) for row in confirmation_rows)
            new_pass = all(bool(row["gate_pass"]) for row in new_rows)
            expected_release = historical_pass and new_pass
            if expected_release:
                expected_released[key] = expected
            else:
                expected_fallback[key] = "baseline"
            status = mapping(mapping(release.get("cell_status"), "release.cell_status").get(key), f"release.cell_status.{key}")
            require(status.get("status") == ("released" if expected_release else "rejected"), f"release status drift: {key}")
            require(status.get("historical_gate_pass") is historical_pass and status.get("new_gate_pass") is new_pass, f"release gate status drift: {key}")
            history_rows[key] = {
                "expected_winner": expected,
                "discovery": discovery,
                "confirmation_repeats": confirmation_rows,
                "new_repeats": new_rows,
                "historical_gate_pass": historical_pass,
                "new_gate_pass": new_pass,
                "release": expected_release,
            }

    record_only = mapping(release.get("record_only"), "release.record_only")
    require(set(record_only) == {RECORD_ONLY_KEY}, "release record-only scope drift")
    record = mapping(record_only.get(RECORD_ONLY_KEY), f"release.record_only.{RECORD_ONLY_KEY}")
    require(record.get("status") == "baseline", "record-only was promoted")
    record_repeats = sequence(record.get("repeats"), "record_only.repeats")
    require(len(record_repeats) == REPEATS, "record-only new repeat count drift")
    record_new = []
    for index, item_value in enumerate(record_repeats):
        item = mapping(item_value, f"record_only.repeat{index}")
        require(item.get("repeat_index") == index and item.get("input_immutability_exact") is True, "record-only repeat drift or mutation")
        record_new.append(recompute_benchmark(item.get("benchmark"), None, f"record_only/new/repeat{index}", NEW_SAMPLES))
    record_confirm = [
        recompute_benchmark(confirmation_benchmark(confirmation, "mixed_n6_h12_t8192", "fp32_both", index), None, f"record_only/confirmation/repeat{index}", NEW_SAMPLES)
        for index in range(REPEATS)
    ]
    # The frozen mixed-tail runner pre-registered only none/fp32-final-only
    # benchmarks.  FP32-both is deliberately record-only and has no frozen
    # discovery timing vector, so inventing one here would weaken rather than
    # strengthen the audit.  It is still independently recomputed for both
    # confirmation and new-release observations below.

    released = mapping(release.get("released_mapping"), "release.released_mapping")
    fallback = mapping(release.get("fallback_mapping"), "release.fallback_mapping")
    require(dict(released) == expected_released, "released mapping disagrees with independent raw recomputation")
    require(dict(fallback) == expected_fallback, "fallback mapping disagrees with independent raw recomputation")
    require(set(released) | set(fallback) == set(history_rows) and not (set(released) & set(fallback)), "release/fallback mapping scope drift")

    return {
        "audit_schema_version": 1,
        "artifact": str(path.resolve()),
        "artifact_sha256": release_digest,
        "frozen_artifacts": {
            "seqcount_discovery": {"sha256": seq_digest},
            "mixed_discovery": {"sha256": mixed_digest},
            "confirmation": {"sha256": confirmation_digest},
        },
        "independent_audit_pass": True,
        "promotion_cell_count": len(history_rows),
        "record_only_cell": {
            "key": RECORD_ONLY_KEY,
            "frozen_discovery": "not_pre_registered_for_record_only_cell",
            "confirmation_repeats": record_confirm,
            "new_repeats": record_new,
            "action": "baseline_only",
        },
        "cells": history_rows,
        "independently_released_mapping": expected_released,
        "independently_fallback_mapping": expected_fallback,
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
    parser.add_argument("artifact", type=Path, help="completed release JSON")
    parser.add_argument("--expected-sha256", required=True, help="exact SHA256 of the release JSON")
    parser.add_argument("--seqcount-discovery", type=Path, help="frozen seqcount raw JSON; required if recorded remote path is unavailable")
    parser.add_argument("--mixed-discovery", type=Path, help="frozen mixed raw JSON; required if recorded remote path is unavailable")
    parser.add_argument("--confirmation", type=Path, help="frozen confirmation raw JSON; required if recorded remote path is unavailable")
    parser.add_argument("--json", type=Path, required=True, help="atomically written independent audit JSON")
    args = parser.parse_args()
    try:
        # Read only the small preregistration envelope first so collision
        # checks cover both the supplied staging copies and their recorded
        # frozen source paths.  Full content validation remains in audit().
        release, _ = read_json(args.artifact, args.expected_sha256, "release artifact")
        prereg = verify_release_preregistration(release)
        recorded_frozen = tuple(
            locate_frozen(None, prereg, name)
            for name in ("seqcount_discovery", "mixed_discovery", "confirmation")
        )
        explicit_frozen = tuple(
            item for item in (args.seqcount_discovery, args.mixed_discovery, args.confirmation) if item is not None
        )
        validate_audit_output(args.json, (args.artifact, *recorded_frozen, *explicit_frozen))
        result = audit(args.artifact, args.expected_sha256, args.seqcount_discovery, args.mixed_discovery, args.confirmation)
    except (AuditError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    atomic_write(args.json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
