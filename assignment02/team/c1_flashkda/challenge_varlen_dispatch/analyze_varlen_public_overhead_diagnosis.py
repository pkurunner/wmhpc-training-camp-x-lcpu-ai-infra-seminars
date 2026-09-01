#!/usr/bin/env python3
"""Fail-closed, stdlib-only audit of the public-overhead diagnostic.

The runner's differential summary is deliberately not trusted.  This tool
recomputes all path summaries and the paired differential directly from the
same-index raw samples before writing an audit artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping


SCHEMA_VERSION = 1
RUNNER_SHA256 = "651ff9af72ddd423d094b018ab7b29438a4283cb2cc50f39254874b8fd84e866"
INTEGRATION_RUNNER_SHA256 = "5db71f29335220496ca9540924e17c5f160b0bc8237060921cffaecb708f22bb"
EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
FLASH_KDA_INIT_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
HELPER_SHA256 = "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
RUNTIME_IDENTITIES = {
    "auto_dispatch": ("2b817adb7d21d1f223e8df4616eeccd74e34a5b1944492211f0f0254147ba883", "auto_dispatch.py"),
    "fla_backend": ("6321b1a75713560d25fd92bb94e8e4e15401d206269a8fc10ca5b8ab4433174f", "fla_backend.py"),
    "varlen_metadata": ("16c01cfc2a8aeee4d80362435053009c3b6397ab09e01d390ac14a38a29b822d", "varlen_metadata.py"),
    "confirmation_runner": ("9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b", "run_varlen_dispatch_confirmation.py"),
    "shared_seqcount_runner": ("4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f", "run_seqcount_dispatch.py"),
    "prefetch2": ("752126488487ac317a7ee167b660b0895562e4877aedbc4ce2a599c1f59a10d0", "prefetch2.py"),
    "vshard4_prefetch2": ("445c90815919fb8c1db3bc79289d10029ea77b1d9accb393c326c120ba1f8385", "vshard4_prefetch2.py"),
    "harness": ("5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52", "validate_and_bench.py"),
    "pinned_torch_ref": ("bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5", "tests/torch_ref.py"),
    "pinned_reference_helper": (HELPER_SHA256, "sigmoid_ext/sigmoid_ext.so"),
}
FLA_FILES = {
    "fla/__init__.py": "b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d",
    "fla/ops/backends/__init__.py": "a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635",
    "fla/ops/kda/__init__.py": "24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb",
    "fla/ops/kda/backends/__init__.py": "86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797",
    "fla/ops/kda/backends/flash_kda.py": "0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2",
    "fla/ops/kda/chunk.py": "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8",
}

PATHS = ("public_c1", "direct_c1", "public_pinned", "direct_pinned")
CELLS = {
    "equal_n2_h12_t2048/none": ("vshard4_p2", [0, 2048, 4096]),
    "mixed_n6_h12_t8192/none": ("vshard2_p2", [0, 17, 528, 1552, 2852, 4901, 8192]),
    "skew_n6_h12_t12288/none": ("vshard2_p2", [0, 1, 2, 3, 4, 5, 12288]),
}
PRODUCTION_MAP = [
    {"offsets": [0, 1, 2, 3, 4, 5, 12288], "contract": "fp32_final_only", "variant": "vshard2_p2"},
    {"offsets": [0, 1, 2, 3, 4, 5, 12288], "contract": "none", "variant": "vshard2_p2"},
]
TEMPORARY_MAP = [
    {"offsets": [0, 1, 2, 3, 4, 5, 12288], "contract": "fp32_both", "variant": "vshard4_p2"},
    {"offsets": [0, 1, 2, 3, 4, 5, 12288], "contract": "fp32_final_only", "variant": "vshard2_p2"},
    {"offsets": [0, 1, 2, 3, 4, 5, 12288], "contract": "none", "variant": "vshard2_p2"},
    {"offsets": [0, 17, 528, 1552, 2852, 4901, 8192], "contract": "fp32_final_only", "variant": "vshard2_p2"},
    {"offsets": [0, 17, 528, 1552, 2852, 4901, 8192], "contract": "none", "variant": "vshard2_p2"},
    {"offsets": [0, 2048, 4096], "contract": "fp32_both", "variant": "vshard4_p2"},
    {"offsets": [0, 2048, 4096], "contract": "fp32_final_only", "variant": "vshard4_p2"},
    {"offsets": [0, 2048, 4096], "contract": "none", "variant": "vshard4_p2"},
    {"offsets": [0, 2048, 4096, 6144, 8192], "contract": "fp32_final_only", "variant": "vshard2_p2"},
    {"offsets": [0, 2048, 4096, 6144, 8192], "contract": "none", "variant": "vshard2_p2"},
]


class AuditError(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def obj(value: object, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def arr(value: object, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def num(value: object, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    value = float(value)
    require(math.isfinite(value), f"{label} must be finite")
    return value


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def stats(values: list[float]) -> dict[str, float | int]:
    require(values, "cannot summarize an empty vector")
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
        "negative_count": sum(value < 0.0 for value in values),
    }


def close(actual: object, expected: float, label: str) -> None:
    require(math.isclose(num(actual, label), expected, rel_tol=0.0, abs_tol=1e-12), f"{label} disagrees with raw recomputation")


def suffix_path(value: object, suffix: str, label: str) -> None:
    require(isinstance(value, str) and value.replace("\\", "/").rstrip("/").endswith(suffix), f"{label} path drift")


def validate_identity(data: Mapping[str, Any]) -> None:
    identity = obj(data.get("identity"), "identity")
    extension = obj(identity.get("extension"), "identity.extension")
    require(extension.get("sha256") == EXTENSION_SHA256 and extension.get("passed") is True, "extension identity drift")
    python = obj(identity.get("flash_kda_python"), "identity.flash_kda_python")
    require(python.get("sha256") == FLASH_KDA_INIT_SHA256, "flash_kda Python SHA drift")
    suffix_path(python.get("path"), "flash_kda/__init__.py", "flash_kda Python")
    trees = obj(identity.get("source_trees"), "identity.source_trees")
    patched = obj(trees.get("patched"), "identity.source_trees.patched")
    reference = obj(trees.get("reference"), "identity.source_trees.reference")
    require(patched.get("commit") == PATCHED_COMMIT and patched.get("passed") is True, "patched source commit drift")
    require(reference.get("commit") == PATCHED_COMMIT and reference.get("tracked_status_clean") is True and reference.get("passed") is True, "reference source commit drift")
    fla = obj(identity.get("fla"), "identity.fla")
    require(fla.get("commit") == FLA_COMMIT and fla.get("tracked_status_clean") is True and fla.get("passed") is True, "FLA source commit drift")
    fla_files = obj(fla.get("files"), "identity.fla.files")
    require(dict(fla_files) == FLA_FILES, "FLA source file hash drift")
    runner = obj(identity.get("diagnostic_runner"), "identity.diagnostic_runner")
    require(runner.get("sha256") == RUNNER_SHA256 and runner.get("sha256_gate_pass") is True, "diagnostic runner SHA drift")
    suffix_path(runner.get("path"), "run_varlen_public_overhead_diagnosis.py", "diagnostic runner")
    current = obj(runner.get("current_integration_runner"), "identity.diagnostic_runner.current_integration_runner")
    require(current.get("sha256") == INTEGRATION_RUNNER_SHA256 and current.get("passed") is True, "integration runner SHA drift")
    suffix_path(current.get("path"), "run_varlen_fla_integration.py", "integration runner")
    runtime = obj(identity.get("runtime_import_identities"), "identity.runtime_import_identities")
    require(set(runtime) == set(RUNTIME_IDENTITIES), "runtime identity key drift")
    for name, (expected_sha, expected_suffix) in RUNTIME_IDENTITIES.items():
        entry = obj(runtime.get(name), f"runtime identity {name}")
        require(entry.get("sha256") == expected_sha and entry.get("sha256_gate_pass") is True, f"runtime identity SHA drift: {name}")
        suffix_path(entry.get("path"), expected_suffix, f"runtime identity {name}")
    helper = obj(identity.get("pinned_reference_helper"), "identity.pinned_reference_helper")
    require(helper.get("sha256") == HELPER_SHA256 and helper.get("no_build") is True, "pinned helper identity drift")
    device = obj(identity.get("device"), "identity.device")
    require("B300" in str(device.get("name", "")).upper() and device.get("capability") == [10, 3] and device.get("multiprocessor_count") == 148 and device.get("passed") is True, "B300 identity drift")


def validate_maps(data: Mapping[str, Any]) -> None:
    maps = obj(data.get("maps"), "maps")
    require(maps.get("production_r5_before_temporary_install") == PRODUCTION_MAP, "production map is not the frozen two-cell r5 map")
    require(maps.get("temporary_r4_diagnostic") == TEMPORARY_MAP, "temporary map is not the frozen ten-cell r4 map")
    require(maps.get("temporary_installation") == {"attempted": True, "passed": True}, "temporary installation evidence drift")
    require(maps.get("finally_restored_r5") == {"attempted": True, "passed": True}, "r5 restoration evidence drift")


def validate_gates(data: Mapping[str, Any]) -> None:
    gates = obj(data.get("gates"), "gates")
    require(set(gates) == {"production_map_exact_two", "clean_gpu", "device", "extension", "fla_pin", "runtime_dependencies", "inference_mode", "temporary_map_restored"}, "gate scope drift")
    for name, value in gates.items():
        gate = obj(value, f"gate {name}")
        require(gate.get("passed") is True, f"gate failed: {name}")
    inference = obj(gates.get("inference_mode"), "inference mode")
    require(inference.get("grad_enabled") is False and inference.get("inference_mode_enabled") is True, "inference-mode evidence drift")


def validate_spies(correctness: Mapping[str, Any], cell: str) -> None:
    probe = obj(correctness.get("non_timed_prepare_varlen_probe"), f"{cell}.prepare probe")
    expected = {
        "public_c1": ({"c1": 1, "pinned": 0}, {"prepare_varlen": 2}),
        "direct_c1": ({"c1": 0, "pinned": 0}, {"prepare_varlen": 1}),
        "public_pinned": ({"c1": 0, "pinned": 1}, {"prepare_varlen": 0}),
        "direct_pinned": ({"c1": 0, "pinned": 0}, {"prepare_varlen": 0}),
    }
    require(set(probe) == set(expected), f"{cell}: spy path scope drift")
    for path, (backend_expected, prepare_expected) in expected.items():
        entry = obj(probe.get(path), f"{cell}.{path} probe")
        require(obj(entry.get("backend_chunk_spy_delta"), "backend spy") == backend_expected, f"{cell}.{path}: backend route drift")
        require(obj(entry.get("prepare_varlen_spy_delta"), "prepare spy") == prepare_expected, f"{cell}.{path}: duplicate preflight count drift")
    route = obj(correctness.get("public_route_proof"), f"{cell}.public_route_proof")
    require(route.get("passed") is True and route.get("public_c1") == {"c1": 1, "pinned": 0} and route.get("public_pinned") == {"c1": 0, "pinned": 1}, f"{cell}: public route proof drift")


def validate_cell(data: Mapping[str, Any], cell: str, expected_variant: str, offsets: list[int]) -> list[dict[str, Any]]:
    cells = obj(data.get("cells"), "cells")
    entry = obj(cells.get(cell), f"cell {cell}")
    require(entry.get("expected_diagnostic_variant") == expected_variant and entry.get("contract") == "none" and entry.get("passed") is True, f"{cell}: cell contract/variant drift")
    correctness = obj(entry.get("correctness_and_route"), f"{cell}.correctness_and_route")
    require(correctness.get("passed") is True, f"{cell}: correctness failed")
    verifier = obj(correctness.get("verifier"), f"{cell}.verifier")
    require(obj(verifier.get("c1"), "c1 verifier").get("passed") is True and obj(verifier.get("pinned"), "pinned verifier").get("passed") is True, f"{cell}: verifier failed")
    exact = obj(correctness.get("four_path_bitwise_exact"), f"{cell}.four_path_bitwise_exact")
    require(set(exact) == {"direct_c1_vs_direct_pinned", "public_c1_vs_direct_c1", "public_pinned_vs_direct_pinned", "public_c1_vs_public_pinned"}, f"{cell}: exact comparison scope drift")
    for name, value in exact.items():
        comparison = obj(value, f"{cell}.{name}")
        require(comparison.get("output_exact") is True and num(comparison.get("output_max_abs"), f"{cell}.{name}.output_max_abs") == 0.0, f"{cell}.{name}: non-exact output")
        for side in ("actual_output", "expected_output"):
            tensor = obj(comparison.get(side), f"{cell}.{name}.{side}")
            require(tensor.get("shape") == [1, offsets[-1], 12, 128] and tensor.get("dtype") == "torch.bfloat16" and tensor.get("contiguous") is True, f"{cell}.{name}: output contract drift")
    immutability = obj(correctness.get("input_immutability"), f"{cell}.input_immutability")
    require(immutability.get("input_immutability_exact") is True and arr(immutability.get("fields"), "immutable fields"), f"{cell}: input immutability failed")
    validate_spies(correctness, cell)
    decisions = obj(correctness.get("c1_dispatch_decisions"), f"{cell}.decisions")
    require(decisions.get("passed") is True and decisions.get("expected_variant") == expected_variant, f"{cell}: dispatch decision failed")
    for path in ("public_c1", "direct_c1"):
        decision = obj(decisions.get(path), f"{cell}.{path}.decision")
        require(decision.get("chosen_variant") == expected_variant and decision.get("certified_varlen_offsets") == offsets, f"{cell}.{path}: route/offset drift")
    repeats = arr(entry.get("repeats"), f"{cell}.repeats")
    require(len(repeats) == 2, f"{cell}: expected two repeats")
    outputs: list[dict[str, Any]] = []
    for repeat_index, repeat_value in enumerate(repeats):
        repeat = obj(repeat_value, f"{cell}.repeat{repeat_index}")
        require(repeat.get("repeat_index") == repeat_index and repeat.get("passed") is True, f"{cell}: repeat {repeat_index} incomplete")
        order = obj(repeat.get("path_order"), f"{cell}.repeat{repeat_index}.path_order")
        require(order.get("cycle") == list(PATHS) and order.get("warmup_first_path_counts") == {path: 25 for path in PATHS} and order.get("timed_first_path_counts") == {path: 250 for path in PATHS} and order.get("passed") is True, f"{cell}: repeat {repeat_index} is not balanced")
        raw = obj(repeat.get("raw_samples_ms"), f"{cell}.repeat{repeat_index}.raw_samples_ms")
        require(set(raw) == set(PATHS), f"{cell}: raw path scope drift")
        vectors: dict[str, list[float]] = {}
        for path in PATHS:
            values = [num(value, f"{cell}.repeat{repeat_index}.{path}[{i}]") for i, value in enumerate(arr(raw.get(path), f"{cell}.{path}.raw"))]
            require(len(values) == 1000 and all(value > 0.0 for value in values), f"{cell}.{path}: expected 1000 positive finite samples")
            vectors[path] = values
        summaries = obj(repeat.get("paths"), f"{cell}.repeat{repeat_index}.paths")
        require(set(summaries) == set(PATHS), f"{cell}: runner summary path scope drift")
        recomputed: dict[str, dict[str, float | int]] = {}
        for path in PATHS:
            expected = stats(vectors[path])
            observed = obj(summaries.get(path), f"{cell}.{path}.summary")
            require(set(observed) == {"samples", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms"}, f"{cell}.{path}: summary field drift")
            for key, value in expected.items():
                if key == "negative_count":
                    continue
                close(observed.get(key), float(value), f"{cell}.{path}.{key}")
            recomputed[path] = expected
        differential = [(vectors["public_c1"][i] - vectors["direct_c1"][i]) - (vectors["public_pinned"][i] - vectors["direct_pinned"][i]) for i in range(1000)]
        c1_public_overhead = [vectors["public_c1"][i] - vectors["direct_c1"][i] for i in range(1000)]
        pinned_public_overhead = [vectors["public_pinned"][i] - vectors["direct_pinned"][i] for i in range(1000)]
        direct_advantage = [vectors["direct_pinned"][i] - vectors["direct_c1"][i] for i in range(1000)]
        diff_stats = stats(differential)
        outputs.append({
            "repeat_index": repeat_index,
            "independent_differential_ms": diff_stats,
            "public_overhead_c1_ms": stats(c1_public_overhead),
            "public_overhead_pinned_ms": stats(pinned_public_overhead),
            "direct_c1_advantage_ms": stats(direct_advantage),
            "estimated_public_c1_after_removing_paired_mean_diff_ms": statistics.fmean(vectors["public_c1"]) - float(diff_stats["mean_ms"]),
        })
    duplicate_consistent = all(item["independent_differential_ms"]["mean_ms"] > 0.0 and item["independent_differential_ms"]["p50_ms"] > 0.0 for item in outputs)
    return [{"cell": cell, "repeat_summaries": outputs, "duplicate_preflight_consistent": duplicate_consistent}]


def audit(input_path: Path) -> dict[str, Any]:
    payload = input_path.read_bytes()
    input_sha = hashlib.sha256(payload).hexdigest()
    data = json.loads(payload)
    require(isinstance(data, Mapping), "input must be a JSON object")
    expected_top = {"schema_version", "purpose", "diagnostic_only", "no_release_authority", "no_policy_mutation", "seed", "representative_cells", "maps", "measurement", "identity", "gates", "cells", "complete", "registry"}
    require(set(data) == expected_top, "top-level schema scope drift")
    require(data.get("schema_version") == SCHEMA_VERSION and data.get("diagnostic_only") is True and data.get("no_release_authority") is True and data.get("complete") is True, "diagnostic-only completion contract failed")
    require(data.get("no_policy_mutation") == "auto_dispatch.py is never written; the r4 map is process-local and finally-restored", "policy-mutation contract drift")
    validate_maps(data)
    validate_identity(data)
    validate_gates(data)
    registry = obj(data.get("registry"), "registry")
    require(isinstance(registry.get("c1_id"), int) and isinstance(registry.get("pinned_id"), int) and registry.get("c1_id") != registry.get("pinned_id") and registry.get("timing_has_no_spies") is True, "registry identity drift")
    measurement = obj(data.get("measurement"), "measurement")
    require(measurement.get("paths") == list(PATHS) and measurement.get("repeats") == 2 and measurement.get("warmup_per_path_per_repeat") == 100 and measurement.get("samples_per_path_per_repeat") == 1000 and measurement.get("percentiles") == ["p50", "p95", "p99"], "measurement contract drift")
    cells = obj(data.get("cells"), "cells")
    require(set(cells) == set(CELLS) and len(cells) == 3, "representative cell scope drift")
    reports: list[dict[str, Any]] = []
    for cell, (variant, offsets) in CELLS.items():
        reports.extend(validate_cell(data, cell, variant, offsets))
    return {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": input_sha,
        "input": {"path": str(input_path), "sha256": input_sha},
        "audit_pass": True,
        "summary": {
            "diagnostic_only": True,
            "cells": reports,
            "all_cells_duplicate_preflight_consistent": all(item["duplicate_preflight_consistent"] for item in reports),
            "differential_formula": "(public_c1 - direct_c1) - (public_pinned - direct_pinned)",
            "differential_ms_source": "independently recomputed from same-index raw_samples_ms; runner differential_ms is ignored",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(args.input.resolve() != args.output.resolve(), "input and output must differ")
        result = audit(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"audit_pass=true input_sha256={result['input']['sha256']} output={args.output}")
        return 0
    except (AuditError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"audit_pass=false: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
