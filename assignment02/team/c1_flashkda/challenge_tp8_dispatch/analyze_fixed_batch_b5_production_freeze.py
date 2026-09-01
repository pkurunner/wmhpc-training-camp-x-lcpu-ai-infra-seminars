#!/usr/bin/env python3
"""Fail-closed two-allocation freeze gate for public fixed-batch B=5.

This program is deliberately stdlib-only.  It consumes the historical public
FLA integration result and a fresh result from the *same frozen public runner*
and accepts neither loose truthiness nor partial records.  A successful output
means that both allocations proved the complete 18-cell public mapping exactly;
it is not a launcher and never changes source, a registry, or a dispatcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


SCHEMA_VERSION = 1
HISTORY_SEED = 20260829
CURRENT_SEED = 20260831
DIM = 128
HEADS = 12
TOKENS = 2048
BATCHES = (2, 3, 4, 5, 6, 8)
CONTRACTS = ("none", "fp32_final_only", "fp32_both")
EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
FLASH_KDA_PYTHON_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
FLA_FILE_SHA256 = {
    "fla/__init__.py": "b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d",
    "fla/ops/backends/__init__.py": "a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635",
    "fla/ops/kda/__init__.py": "24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb",
    "fla/ops/kda/backends/__init__.py": "86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797",
    "fla/ops/kda/backends/flash_kda.py": "0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2",
    "fla/ops/kda/chunk.py": "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8",
}
PURPOSE = "real pinned-FLA/public-registry fixed-batch correctness integration audit"
PERFORMANCE_OBSERVATION = "not_run; this is a correctness/registry integration gate, not a release latency gate"


class AuditError(AssertionError):
    """An evidence or semantic gate failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def object_of(value: object, label: str) -> Mapping[str, Any]:
    require(type(value) is dict, f"{label} must be a JSON object")
    return value  # type: ignore[return-value]


def array_of(value: object, label: str) -> list[Any]:
    require(type(value) is list, f"{label} must be a JSON array")
    return value  # type: ignore[return-value]


def exact_bool(value: object, label: str) -> bool:
    require(type(value) is bool, f"{label} must be an exact bool")
    return value  # type: ignore[return-value]


def exact_int(value: object, label: str) -> int:
    require(type(value) is int, f"{label} must be an integer (bool is rejected)")
    return value  # type: ignore[return-value]


def finite_number(value: object, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric (bool is rejected)")
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number


def exact_zero_float(value: object, label: str) -> None:
    """The runner serializes exact-comparison maxima as JSON floats, not ints."""

    require(type(value) is float, f"{label} must be a JSON float")
    require(math.isfinite(value) and value == 0.0, f"{label} must be exactly 0.0")


def nonempty_string(value: object, label: str) -> str:
    require(type(value) is str and bool(value.strip()), f"{label} must be a non-empty string")
    return value  # type: ignore[return-value]


def sha256_string(value: object, label: str) -> str:
    text = nonempty_string(value, label)
    require(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text),
        f"{label} must be a lowercase SHA256",
    )
    return text


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path, expected_sha: str, label: str) -> tuple[Mapping[str, Any], str]:
    require(path.is_file(), f"{label} is missing: {path}")
    actual_sha = sha256(path)
    require(actual_sha == expected_sha, f"{label} SHA mismatch: expected={expected_sha} actual={actual_sha}")
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label} is not valid JSON") from exc
    return object_of(decoded, label), actual_sha


def expected_variant(batch: int, contract: str) -> str:
    if batch in (2, 3):
        return "vshard4_p2"
    if batch == 5:
        return "vshard2_p2"
    if batch in (4, 6) and contract in ("none", "fp32_final_only"):
        return "vshard2_p2"
    return "baseline"


def cell_key(batch: int, contract: str) -> str:
    return f"b{batch}_h12_t2048/{contract}"


CELL_KEYS = tuple(cell_key(batch, contract) for batch in BATCHES for contract in CONTRACTS)
POSITIVE_KEYS = tuple(key for key in CELL_KEYS if expected_variant(int(key[1:key.index("_h")]), key.split("/", 1)[1]) != "baseline")
NEGATIVE_KEYS = tuple(key for key in CELL_KEYS if key not in POSITIVE_KEYS)


def expected_reason(batch: int, contract: str, variant: str) -> str:
    if variant != "baseline":
        return f"fixed_batch_b{batch}_h12_t2048_{contract}_whitelist_hit"
    if batch in (4, 6):
        return f"fixed_batch_b{batch}_{contract}_not_whitelisted"
    return "fixed_batch_shape_not_whitelisted"


def expected_matrix() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for batch in BATCHES:
        for contract in CONTRACTS:
            variant = expected_variant(batch, contract)
            result.append(
                {
                    "cell": cell_key(batch, contract),
                    "expected_variant": variant,
                    "release_role": "negative" if variant == "baseline" else "positive",
                }
            )
    return result


def exact_fp32_state_shape(value: object, batch: int, label: str) -> None:
    """Require the serialized ``[B, 12, 128, 128]`` shape without coercion.

    JSON's ``true == 1`` and ``5.0 == 5`` comparison rules make a direct
    list equality check unsound for an audit.  Every dimension is therefore
    checked with :func:`exact_int`, which rejects booleans and floats.
    """

    shape = array_of(value, label)
    expected = (batch, HEADS, DIM, DIM)
    require(len(shape) == len(expected), f"{label} length drift")
    for index, expected_value in enumerate(expected):
        require(
            exact_int(shape[index], f"{label}[{index}]") == expected_value,
            f"{label}[{index}] drift",
        )


def require_exact_keys(record: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(set(record) == expected, f"{label} keys drift: expected={sorted(expected)} actual={sorted(record)}")


def validate_identity(value: object, label: str) -> dict[str, object]:
    identity = object_of(value, label)
    require_exact_keys(identity, {"device", "extension", "flash_kda_python", "fla"}, label)

    device = object_of(identity.get("device"), f"{label}.device")
    require_exact_keys(device, {"name", "capability", "multiprocessor_count", "gate_pass"}, f"{label}.device")
    require("B300" in nonempty_string(device.get("name"), f"{label}.device.name").upper(), f"{label}.device is not B300")
    capability = array_of(device.get("capability"), f"{label}.device.capability")
    require(len(capability) == 2, f"{label}.device capability length drift")
    require(exact_int(capability[0], f"{label}.device.capability[0]") == 10, f"{label}.device major capability drift")
    require(exact_int(capability[1], f"{label}.device.capability[1]") == 3, f"{label}.device minor capability drift")
    require(exact_int(device.get("multiprocessor_count"), f"{label}.device.sm_count") == 148, f"{label}.device SM count drift")
    require(exact_bool(device.get("gate_pass"), f"{label}.device.gate_pass") is True, f"{label}.device gate failed")

    extension = object_of(identity.get("extension"), f"{label}.extension")
    require_exact_keys(extension, {"path", "sha256", "required_symbols", "gate_pass"}, f"{label}.extension")
    require(nonempty_string(extension.get("path"), f"{label}.extension.path").endswith(".so"), f"{label}.extension path is not a shared object")
    require(sha256_string(extension.get("sha256"), f"{label}.extension.sha256") == EXTENSION_SHA256, f"{label}.extension SHA drift")
    require(array_of(extension.get("required_symbols"), f"{label}.extension.required_symbols") == ["fwd", "fwd_vshard_p2", "fwd_vshard4_p2", "get_workspace_size"], f"{label}.extension symbols drift")
    require(exact_bool(extension.get("gate_pass"), f"{label}.extension.gate_pass") is True, f"{label}.extension gate failed")

    package = object_of(identity.get("flash_kda_python"), f"{label}.flash_kda_python")
    require_exact_keys(package, {"path", "sha256", "gate_pass"}, f"{label}.flash_kda_python")
    require(nonempty_string(package.get("path"), f"{label}.flash_kda_python.path").endswith("/flash_kda/__init__.py"), f"{label}.flash_kda_python path drift")
    require(sha256_string(package.get("sha256"), f"{label}.flash_kda_python.sha256") == FLASH_KDA_PYTHON_SHA256, f"{label}.flash_kda_python SHA drift")
    require(exact_bool(package.get("gate_pass"), f"{label}.flash_kda_python.gate_pass") is True, f"{label}.flash_kda_python gate failed")

    fla = object_of(identity.get("fla"), f"{label}.fla")
    require_exact_keys(fla, {"root", "commit", "files", "loaded_modules", "gate_pass"}, f"{label}.fla")
    root = nonempty_string(fla.get("root"), f"{label}.fla.root").rstrip("/")
    require(fla.get("commit") == FLA_COMMIT, f"{label}.fla commit drift")
    files = object_of(fla.get("files"), f"{label}.fla.files")
    require(dict(files) == FLA_FILE_SHA256, f"{label}.fla files drift")
    modules = object_of(fla.get("loaded_modules"), f"{label}.fla.loaded_modules")
    expected_module_files = {
        "fla": "fla/__init__.py",
        "fla.ops.backends": "fla/ops/backends/__init__.py",
        "fla.ops.kda": "fla/ops/kda/__init__.py",
        "fla.ops.kda.backends": "fla/ops/kda/backends/__init__.py",
        "fla.ops.kda.backends.flash_kda": "fla/ops/kda/backends/flash_kda.py",
        "fla.ops.kda.chunk": "fla/ops/kda/chunk.py",
    }
    require(set(modules) == set(expected_module_files), f"{label}.fla loaded-module keys drift")
    for module, relative in expected_module_files.items():
        require(modules.get(module) == f"{root}/{relative}", f"{label}.fla module path drift: {module}")
    require(exact_bool(fla.get("gate_pass"), f"{label}.fla.gate_pass") is True, f"{label}.fla gate failed")
    # Content identities above are individually pinned.  Preserve the loaded
    # paths as well so the two allocations cannot silently change import mode
    # while retaining identical file bytes.
    return {
        "device_name": device["name"],
        "extension_path": extension["path"],
        "flash_kda_python_path": package["path"],
        "fla_root": root,
    }


def validate_registry(value: object, label: str) -> None:
    registry = object_of(value, label)
    require_exact_keys(registry, {"backend_order", "registration_idempotent", "custom_before_pinned"}, label)
    expected = [
        {"backend_type": "triton_ascend", "priority": 0, "available": False, "enabled": True},
        {"backend_type": "c1_b300_flash_kda", "priority": 2, "available": True, "enabled": True},
        {"backend_type": "flash_kda", "priority": 3, "available": True, "enabled": True},
        {"backend_type": "tilelang", "priority": 5, "available": True, "enabled": True},
    ]
    records = array_of(registry.get("backend_order"), f"{label}.backend_order")
    require(len(records) == len(expected), f"{label}.backend_order length drift")
    for index, expected_record in enumerate(expected):
        record = object_of(records[index], f"{label}.backend_order[{index}]")
        require_exact_keys(record, {"backend_type", "priority", "available", "enabled"}, f"{label}.backend_order[{index}]")
        require(record.get("backend_type") == expected_record["backend_type"], f"{label}.backend_order[{index}].backend_type drift")
        require(exact_int(record.get("priority"), f"{label}.backend_order[{index}].priority") == expected_record["priority"], f"{label}.backend_order[{index}].priority drift")
        require(exact_bool(record.get("available"), f"{label}.backend_order[{index}].available") is expected_record["available"], f"{label}.backend_order[{index}].available drift")
        require(exact_bool(record.get("enabled"), f"{label}.backend_order[{index}].enabled") is expected_record["enabled"], f"{label}.backend_order[{index}].enabled drift")
    require(exact_bool(registry.get("registration_idempotent"), f"{label}.registration_idempotent") is True, f"{label}.registration_idempotent failed")
    require(exact_bool(registry.get("custom_before_pinned"), f"{label}.custom_before_pinned") is True, f"{label}.custom_before_pinned failed")


def validate_gates(value: object, label: str) -> None:
    gates = object_of(value, label)
    require_exact_keys(gates, {"scope", "clean_gpu_shell_gate", "device", "extension", "fla_pin", "registry"}, label)
    scope = object_of(gates.get("scope"), f"{label}.scope")
    require_exact_keys(scope, {"required_cells", "actual_cells", "passed"}, f"{label}.scope")
    require(exact_int(scope.get("required_cells"), f"{label}.scope.required_cells") == 18, f"{label}.scope required count drift")
    require(exact_int(scope.get("actual_cells"), f"{label}.scope.actual_cells") == 18, f"{label}.scope actual count drift")
    require(exact_bool(scope.get("passed"), f"{label}.scope.passed") is True, f"{label}.scope failed")

    clean = object_of(gates.get("clean_gpu_shell_gate"), f"{label}.clean_gpu_shell_gate")
    require_exact_keys(clean, {"required", "passed"}, f"{label}.clean_gpu_shell_gate")
    require(exact_bool(clean.get("required"), f"{label}.clean_gpu_shell_gate.required") is True, f"{label}.clean GPU not required")
    require(exact_bool(clean.get("passed"), f"{label}.clean_gpu_shell_gate.passed") is True, f"{label}.clean GPU gate failed")

    device = object_of(gates.get("device"), f"{label}.device")
    require_exact_keys(device, {"required", "passed"}, f"{label}.device")
    require(device.get("required") == "B300, capability 10.3, 148 SM", f"{label}.device requirement drift")
    require(exact_bool(device.get("passed"), f"{label}.device.passed") is True, f"{label}.device gate failed")

    extension = object_of(gates.get("extension"), f"{label}.extension")
    require_exact_keys(extension, {"required_sha256", "passed"}, f"{label}.extension")
    require(extension.get("required_sha256") == EXTENSION_SHA256, f"{label}.extension SHA requirement drift")
    require(exact_bool(extension.get("passed"), f"{label}.extension.passed") is True, f"{label}.extension gate failed")

    fla = object_of(gates.get("fla_pin"), f"{label}.fla_pin")
    require_exact_keys(fla, {"commit", "file_hashes", "passed"}, f"{label}.fla_pin")
    require(fla.get("commit") == FLA_COMMIT, f"{label}.fla_pin commit drift")
    require(dict(object_of(fla.get("file_hashes"), f"{label}.fla_pin.file_hashes")) == FLA_FILE_SHA256, f"{label}.fla_pin hashes drift")
    require(exact_bool(fla.get("passed"), f"{label}.fla_pin.passed") is True, f"{label}.fla pin gate failed")

    registry = object_of(gates.get("registry"), f"{label}.registry")
    require_exact_keys(registry, {"passed"}, f"{label}.registry")
    require(exact_bool(registry.get("passed"), f"{label}.registry.passed") is True, f"{label}.registry gate failed")


def validate_final_contract(value: object, batch: int, expected_present: bool, label: str) -> None:
    contract = object_of(value, label)
    if not expected_present:
        require_exact_keys(contract, {"present"}, label)
        require(exact_bool(contract.get("present"), f"{label}.present") is False, f"{label}.present must be false")
        return
    require_exact_keys(contract, {"present", "dtype", "shape", "contiguous"}, label)
    require(exact_bool(contract.get("present"), f"{label}.present") is True, f"{label}.present must be true")
    require(contract.get("dtype") == "torch.float32", f"{label}.dtype drift")
    exact_fp32_state_shape(contract.get("shape"), batch, f"{label}.shape")
    require(exact_bool(contract.get("contiguous"), f"{label}.contiguous") is True, f"{label}.contiguous must be true")


def validate_exact_comparison(value: object, batch: int, output_final: bool, label: str) -> None:
    comparison = object_of(value, label)
    expected_keys = {"output_exact", "output_max_abs", "actual_final", "pinned_final"}
    if output_final:
        expected_keys.update({"final_state_exact", "final_state_max_abs"})
    require_exact_keys(comparison, expected_keys, label)
    require(exact_bool(comparison.get("output_exact"), f"{label}.output_exact") is True, f"{label}.output is not bit-exact")
    exact_zero_float(comparison.get("output_max_abs"), f"{label}.output_max_abs")
    validate_final_contract(comparison.get("actual_final"), batch, output_final, f"{label}.actual_final")
    validate_final_contract(comparison.get("pinned_final"), batch, output_final, f"{label}.pinned_final")
    if output_final:
        require(exact_bool(comparison.get("final_state_exact"), f"{label}.final_state_exact") is True, f"{label}.final state is not bit-exact")
        exact_zero_float(comparison.get("final_state_max_abs"), f"{label}.final_state_max_abs")


def validate_cell(value: object, batch: int, contract: str, label: str) -> str:
    cell = object_of(value, label)
    require_exact_keys(
        cell,
        {
            "expected_variant", "verifiers", "initial_state", "expected_final_state",
            "direct_custom_vs_pinned", "public_registry_vs_pinned", "public_custom_backend_spy",
            "public_decision", "cell_gate_pass",
        },
        label,
    )
    variant = expected_variant(batch, contract)
    output_final = contract != "none"
    require(cell.get("expected_variant") == variant, f"{label}.expected_variant drift")

    verifiers = object_of(cell.get("verifiers"), f"{label}.verifiers")
    require_exact_keys(verifiers, {"pinned_flash_kda", "custom_backend"}, f"{label}.verifiers")
    for name in ("pinned_flash_kda", "custom_backend"):
        verifier = object_of(verifiers.get(name), f"{label}.verifiers.{name}")
        require_exact_keys(verifier, {"passed", "reason"}, f"{label}.verifiers.{name}")
        require(exact_bool(verifier.get("passed"), f"{label}.verifiers.{name}.passed") is True, f"{label}.{name} verifier failed")
        require(verifier.get("reason") is None, f"{label}.{name} verifier reason drift")

    initial = object_of(cell.get("initial_state"), f"{label}.initial_state")
    require_exact_keys(initial, {"present", "dtype", "shape", "construction"}, f"{label}.initial_state")
    if contract == "fp32_both":
        require(exact_bool(initial.get("present"), f"{label}.initial_state.present") is True, f"{label}.initial state missing")
        require(initial.get("dtype") == "torch.float32", f"{label}.initial state dtype drift")
        exact_fp32_state_shape(initial.get("shape"), batch, f"{label}.initial_state.shape")
        require(initial.get("construction") == "deterministic non-symmetric contiguous affine FP32 values", f"{label}.initial construction drift")
    else:
        require(exact_bool(initial.get("present"), f"{label}.initial_state.present") is False, f"{label}.initial state unexpectedly present")
        require(initial.get("dtype") is None and initial.get("shape") is None and initial.get("construction") is None, f"{label}.initial absent contract drift")

    expected_final = object_of(cell.get("expected_final_state"), f"{label}.expected_final_state")
    require_exact_keys(expected_final, {"present", "dtype", "shape"}, f"{label}.expected_final_state")
    require(exact_bool(expected_final.get("present"), f"{label}.expected_final_state.present") is output_final, f"{label}.expected final presence drift")
    if output_final:
        require(expected_final.get("dtype") == "torch.float32", f"{label}.expected final dtype drift")
        exact_fp32_state_shape(expected_final.get("shape"), batch, f"{label}.expected_final_state.shape")
    else:
        require(expected_final.get("dtype") is None and expected_final.get("shape") is None, f"{label}.unexpected final metadata")

    validate_exact_comparison(cell.get("direct_custom_vs_pinned"), batch, output_final, f"{label}.direct_custom_vs_pinned")
    validate_exact_comparison(cell.get("public_registry_vs_pinned"), batch, output_final, f"{label}.public_registry_vs_pinned")

    spy = object_of(cell.get("public_custom_backend_spy"), f"{label}.public_custom_backend_spy")
    require_exact_keys(spy, {"before", "after", "delta", "passed"}, f"{label}.public_custom_backend_spy")
    before = exact_int(spy.get("before"), f"{label}.spy.before")
    after = exact_int(spy.get("after"), f"{label}.spy.after")
    require(before >= 0 and after == before + 1, f"{label}.spy must increase exactly once")
    require(exact_int(spy.get("delta"), f"{label}.spy.delta") == 1, f"{label}.spy delta must be one")
    require(exact_bool(spy.get("passed"), f"{label}.spy.passed") is True, f"{label}.spy failed")

    decision = object_of(cell.get("public_decision"), f"{label}.public_decision")
    require_exact_keys(
        decision,
        {"requested_variant", "chosen_variant", "reason", "extension_sha256", "varlen_cpu_authoritative", "certified_varlen_offsets", "canonical_cache_hit"},
        f"{label}.public_decision",
    )
    require(decision.get("requested_variant") == variant, f"{label}.requested variant drift")
    require(decision.get("chosen_variant") == variant, f"{label}.chosen variant drift")
    require(decision.get("reason") == expected_reason(batch, contract, variant), f"{label}.decision reason drift")
    if variant == "baseline":
        require(decision.get("extension_sha256") is None, f"{label}.baseline must not load extension")
    else:
        require(decision.get("extension_sha256") == EXTENSION_SHA256, f"{label}.custom decision extension drift")
    require(exact_bool(decision.get("varlen_cpu_authoritative"), f"{label}.decision.varlen_cpu_authoritative") is False, f"{label}.fixed batch was marked varlen")
    require(decision.get("certified_varlen_offsets") is None and decision.get("canonical_cache_hit") is None, f"{label}.fixed batch contains varlen provenance")
    require(exact_bool(cell.get("cell_gate_pass"), f"{label}.cell_gate_pass") is True, f"{label}.cell gate failed")
    return variant


def validate_result(data: Mapping[str, Any], expected_seed: int, label: str) -> dict[str, object]:
    require_exact_keys(
        data,
        {
            "schema_version", "purpose", "shape", "seed", "matrix", "positive_cells", "negative_cells",
            "performance_observation", "identity", "registry", "cells", "gates", "complete",
        },
        label,
    )
    require(exact_int(data.get("schema_version"), f"{label}.schema_version") == SCHEMA_VERSION, f"{label}.schema drift")
    require(data.get("purpose") == PURPOSE, f"{label}.purpose drift")
    shape = object_of(data.get("shape"), f"{label}.shape")
    require_exact_keys(shape, {"H", "T", "K", "V"}, f"{label}.shape")
    require(exact_int(shape.get("H"), f"{label}.shape.H") == HEADS, f"{label}.shape.H drift")
    require(exact_int(shape.get("T"), f"{label}.shape.T") == TOKENS, f"{label}.shape.T drift")
    require(exact_int(shape.get("K"), f"{label}.shape.K") == DIM, f"{label}.shape.K drift")
    require(exact_int(shape.get("V"), f"{label}.shape.V") == DIM, f"{label}.shape.V drift")
    require(exact_int(data.get("seed"), f"{label}.seed") == expected_seed, f"{label}.seed drift")
    require(array_of(data.get("matrix"), f"{label}.matrix") == expected_matrix(), f"{label}.matrix drift")
    require(array_of(data.get("positive_cells"), f"{label}.positive_cells") == list(POSITIVE_KEYS), f"{label}.positive cells drift")
    require(array_of(data.get("negative_cells"), f"{label}.negative_cells") == list(NEGATIVE_KEYS), f"{label}.negative cells drift")
    require(data.get("performance_observation") == PERFORMANCE_OBSERVATION, f"{label}.performance observation drift")
    identity_summary = validate_identity(data.get("identity"), f"{label}.identity")
    validate_registry(data.get("registry"), f"{label}.registry")
    validate_gates(data.get("gates"), f"{label}.gates")
    cells = object_of(data.get("cells"), f"{label}.cells")
    require(set(cells) == set(CELL_KEYS), f"{label}.cell key set drift")
    observed_variants: dict[str, str] = {}
    for batch in BATCHES:
        for contract in CONTRACTS:
            key = cell_key(batch, contract)
            observed_variants[key] = validate_cell(cells.get(key), batch, contract, f"{label}.cells[{key}]")
    require(sum(variant != "baseline" for variant in observed_variants.values()) == 13, f"{label}.positive count drift")
    require(sum(variant == "baseline" for variant in observed_variants.values()) == 5, f"{label}.negative count drift")
    require(all(observed_variants[key] == "baseline" for key in NEGATIVE_KEYS), f"{label}.negative cells did not use baseline")
    require(exact_bool(data.get("complete"), f"{label}.complete") is True, f"{label}.complete must be true")
    return {"identity": identity_summary, "variants": observed_variants}


def positive_job_id(value: object, label: str) -> int:
    text = nonempty_string(value, label)
    require(text.isdecimal() and int(text) > 0, f"{label} must be a positive decimal Slurm job ID")
    return int(text)


def validate_log(path: Path, expected_sha: str | None, job_id: int, source_hashes: Mapping[str, str], label: str) -> tuple[str, str]:
    require(path.is_file(), f"{label} is missing: {path}")
    digest = sha256(path)
    if expected_sha is not None:
        require(digest == expected_sha, f"{label} SHA mismatch")
    text = path.read_text(encoding="utf-8")
    require(f"job{job_id}" in path.name, f"{label} filename does not bind Slurm job ID")
    require(f"SLURM_JOB_ID={job_id}" in text or label == "history_log", f"{label} does not record its Slurm job ID")
    for source, expected in source_hashes.items():
        require(expected in text and source in text, f"{label} lacks frozen source evidence for {source}")
    require("18/18 public FLA cells passed" in text, f"{label} lacks successful 18-cell runner record")
    if label == "history_log":
        require("FINAL_RC=0" in text, f"{label} did not complete successfully")
    return str(path.resolve()), digest


def atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_json", type=Path)
    parser.add_argument("current_json", type=Path)
    parser.add_argument("--expected-history-json-sha256", required=True)
    parser.add_argument("--expected-current-json-sha256", required=True)
    parser.add_argument("--expected-auto-dispatch-sha256", required=True)
    parser.add_argument("--expected-fla-backend-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--expected-analyzer-sha256", required=True)
    parser.add_argument("--history-slurm-log", type=Path, required=True)
    parser.add_argument("--expected-history-slurm-log-sha256", required=True)
    parser.add_argument("--history-slurm-job-id", required=True)
    parser.add_argument("--current-slurm-log", type=Path, required=True)
    parser.add_argument("--current-slurm-job-id", required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    source_hashes = {
        "auto_dispatch.py": sha256_string(args.expected_auto_dispatch_sha256, "--expected-auto-dispatch-sha256"),
        "fla_backend.py": sha256_string(args.expected_fla_backend_sha256, "--expected-fla-backend-sha256"),
        "run_fixed_batch_fla_integration.py": sha256_string(args.expected_runner_sha256, "--expected-runner-sha256"),
        "test_auto_dispatch_policy.py": sha256_string(args.expected_policy_sha256, "--expected-policy-sha256"),
    }
    analyzer_sha = sha256_string(args.expected_analyzer_sha256, "--expected-analyzer-sha256")
    require(sha256(Path(__file__).resolve(strict=True)) == analyzer_sha, "analyzer source SHA mismatch")
    history_json_sha = sha256_string(args.expected_history_json_sha256, "--expected-history-json-sha256")
    current_json_sha = sha256_string(args.expected_current_json_sha256, "--expected-current-json-sha256")
    history_log_sha = sha256_string(args.expected_history_slurm_log_sha256, "--expected-history-slurm-log-sha256")
    history_job_id = positive_job_id(args.history_slurm_job_id, "--history-slurm-job-id")
    current_job_id = positive_job_id(args.current_slurm_job_id, "--current-slurm-job-id")
    require(history_job_id != current_job_id, "fresh allocation reused the historical Slurm job ID")
    require(args.history_json.resolve() != args.current_json.resolve(), "history and current JSON paths must differ")
    require(args.history_slurm_log.resolve() != args.current_slurm_log.resolve(), "history and current Slurm log paths must differ")

    history_json, actual_history_json_sha = read_json(args.history_json, history_json_sha, "history_json")
    current_json, actual_current_json_sha = read_json(args.current_json, current_json_sha, "current_json")
    require(actual_history_json_sha != actual_current_json_sha, "history and current JSON SHA256 must differ")
    history_summary = validate_result(history_json, HISTORY_SEED, "history_json")
    current_summary = validate_result(current_json, CURRENT_SEED, "current_json")
    require(history_summary["identity"] == current_summary["identity"], "identity/mode drift across allocations")
    require(history_summary["variants"] == current_summary["variants"], "public mapping drift across allocations")

    history_log_path, actual_history_log_sha = validate_log(
        args.history_slurm_log, history_log_sha, history_job_id, source_hashes, "history_log"
    )
    current_log_path, current_log_sha = validate_log(
        args.current_slurm_log, None, current_job_id, source_hashes, "current_log"
    )

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "two-allocation frozen production public-FLA integration gate for fixed B=5",
        "production_freeze_only": True,
        "no_dispatcher_mutation": True,
        "no_automatic_submission": True,
        "source_identity": {
            "analyzer": {"path": str(Path(__file__).resolve()), "sha256": analyzer_sha, "sha256_gate_pass": True},
            "frozen_production_sources": source_hashes,
        },
        "historical_public_result": {"path": str(args.history_json), "sha256": actual_history_json_sha, "sha256_gate_pass": True, "seed": HISTORY_SEED},
        "fresh_production_result": {"path": str(args.current_json), "sha256": actual_current_json_sha, "sha256_gate_pass": True, "seed": CURRENT_SEED},
        "cross_allocation": {
            "json_sha256_different": True,
            "history_job_id": history_job_id,
            "current_job_id": current_job_id,
            "job_ids_different": True,
            "history_log_path": history_log_path,
            "current_log_path": current_log_path,
            "log_paths_different": True,
            "history_log_sha256": actual_history_log_sha,
            "history_log_sha256_gate_pass": True,
            "current_log_sha256_observed_before_analysis": current_log_sha,
            "identity_equal": True,
            "mapping_equal": True,
        },
        "matrix": expected_matrix(),
        "production_freeze_passed": True,
        "complete": True,
    }
    atomic_write(args.json, payload)
    print(f"wrote production freeze gate {args.json}; production_freeze_passed=true")


if __name__ == "__main__":
    try:
        main()
    except AuditError as exc:
        print(f"AUDIT_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
