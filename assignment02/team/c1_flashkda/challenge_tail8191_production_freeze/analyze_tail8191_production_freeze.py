#!/usr/bin/env python3
"""Stdlib-only auditor for real-production T=8191 public-route freeze data.

It intentionally does not trust runner summaries: every percentile and margin
is recomputed from the 1000 raw CUDA-event values.  The chain mode reopens both
main artifacts and rechecks current source/extension identities before it can
write ``production_freeze_passed=true``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


CONTRACTS = ("none", "fp32_final_only")
PATHS = ("pinned_public", "c1_production_public")
PERCENTILES = ("p50", "p95", "p99")
SAMPLES, REPEATS, MIN_MARGIN = 1000, 2, 0.02
SCHEMA_VERSION = 4
EXPECTED_EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_AUTO_DISPATCH_SHA256 = "9633179c6a5e8ce61b1857a102f81074c6bdfdc63163e13980f40c24d6583c29"
EXPECTED_FLA_BACKEND_SHA256 = "152ac4ba897c1887963553d592456b7f988c89105e6f6cb507a44757058d5fd1"
EXPECTED_PINNED_REFERENCE_HELPER_PATH = "/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so"
EXPECTED_PINNED_REFERENCE_HELPER_SHA256 = "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
PINNED_REFERENCE_HELPER_LOAD_CONTRACT = "direct cached binary; exactly one pinned load_inline('sigmoid_ext') intercepted"
EXPECTED_PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_REFERENCE_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
EXPECTED_STATIC_LEDGER_SHA256 = {
    "auto_dispatch": EXPECTED_AUTO_DISPATCH_SHA256,
    "fla_backend": EXPECTED_FLA_BACKEND_SHA256,
    "harness": "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52",
    "extension": EXPECTED_EXTENSION_SHA256,
    "flash_kda_python": "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84",
    "reference_torch_ref": "bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5",
    "pinned_reference_helper": EXPECTED_PINNED_REFERENCE_HELPER_SHA256,
    "fla:fla/__init__.py": "b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d",
    "fla:fla/ops/backends/__init__.py": "a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635",
    "fla:fla/ops/kda/__init__.py": "24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb",
    "fla:fla/ops/kda/backends/__init__.py": "86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797",
    "fla:fla/ops/kda/backends/flash_kda.py": "0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2",
    "fla:fla/ops/kda/chunk.py": "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8",
    "patched:csrc/flash_kda.cpp": "38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4",
    "patched:csrc/fwd.h": "613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083",
    "patched:csrc/smxx/fwd_launch.cu": "a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928",
}
EXPECTED_PATCHED_TRACKED_STATUS = [" M csrc/flash_kda.cpp", " M csrc/fwd.h", " M csrc/smxx/fwd_launch.cu"]
EXPECTED_REASON = {"none": "fixed_single_batch_b1_h12_t8191_none_whitelist_hit", "fp32_final_only": "fixed_single_batch_b1_h12_t8191_fp32_final_only_whitelist_hit"}
EXPECTED_EVENT_CONTRACT = "unmodified registry/context/kwargs/events prepared; start.record+start.synchronize; exactly one real uninstrumented public chunk_kda call; immediate end.record; route checks and end.synchronize excluded"


class AuditError(AssertionError): pass
def require(value: bool, message: str) -> None:
    if not value: raise AuditError(message)
def mapping(value: object, label: str) -> Mapping[str, Any]:
    require(type(value) is dict, f"{label}: expected object"); return value  # type: ignore[return-value]
def array(value: object, label: str) -> list[Any]:
    require(type(value) is list, f"{label}: expected array"); return value  # type: ignore[return-value]
def boolean(value: object, label: str) -> bool:
    require(type(value) is bool, f"{label}: expected exact bool"); return value  # type: ignore[return-value]
def integer(value: object, label: str) -> int:
    require(type(value) is int, f"{label}: expected integer"); return value  # type: ignore[return-value]
def string(value: object, label: str) -> str:
    require(type(value) is str and bool(value), f"{label}: expected nonempty string"); return value  # type: ignore[return-value]
def number(value: object, label: str) -> float:
    require(type(value) in (int, float) and type(value) is not bool, f"{label}: expected number"); result = float(value); require(math.isfinite(result), f"{label}: non-finite number"); return result
def exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label}: key set drift: {set(value) ^ expected}")
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def sha_text(value: object, label: str) -> str:
    require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), f"{label}: invalid SHA256"); return value  # type: ignore[return-value]
def positive_decimal_job(value: object, label: str) -> str:
    job = string(value, label)
    require(job.isascii() and job.isdecimal() and job[:1] != "0" and int(job) > 0, f"{label}: expected a positive canonical-decimal Slurm job ID")
    return job
def canonical_sha(value: object, label: str) -> str:
    try:
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label}: cannot canonicalize JSON identity") from exc
def close(a: float, b: float) -> bool: return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
def git_output(root: Path, *args: str) -> str:
    try: return subprocess.run(["git", "-C", str(root), *args], check=True, text=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc: raise AuditError(f"git identity query failed for {root}") from exc


def write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.tmp")
    try: tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); tmp.replace(path)
    finally:
        if tmp.exists(): tmp.unlink()


def read(path: Path, expected: str, label: str) -> tuple[Mapping[str, Any], str]:
    sha_text(expected, f"{label}.expected_sha"); require(path.is_file(), f"{label}: artifact missing")
    try:
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        require(actual == expected, f"{label}: artifact SHA mismatch")
        return mapping(json.loads(payload.decode("utf-8")), label), actual
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label}: invalid JSON") from exc


def percentile(values: Sequence[float], q: float) -> float:
    require(len(values) == SAMPLES, "sample count drift"); ordered = sorted(values); point = (len(ordered) - 1) * q; lo, hi = int(point), min(int(point) + 1, len(ordered) - 1); return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)
def samples(value: object, label: str) -> list[float]:
    result = [number(item, f"{label}[{index}]") for index, item in enumerate(array(value, label))]
    require(len(result) == SAMPLES and all(item > 0 for item in result), f"{label}: requires 1000 positive samples"); return result
def recompute(values: list[float]) -> dict[str, float]:
    return {"mean_ms": statistics.fmean(values), "p50_ms": percentile(values, .5), "p95_ms": percentile(values, .95), "p99_ms": percentile(values, .99)}
def validate_summary(values: list[float], observed: object, label: str) -> dict[str, float]:
    result = recompute(values); view = mapping(observed, label); exact_keys(view, {"samples", "mean_ms", "p50_ms", "p95_ms", "p99_ms"}, label); require(integer(view.get("samples"), f"{label}.samples") == SAMPLES, f"{label}: samples summary drift")
    for name, value in result.items(): require(close(number(view.get(name), f"{label}.{name}"), value), f"{label}: forged or stale {name}")
    return result


def _physical_sha(record: Mapping[str, Any], expected_path: Path, expected_sha: str, label: str) -> str:
    exact_keys(record, {"path", "sha256", "gate_pass"}, label)
    path = Path(string(record.get("path"), f"{label}.path"))
    require(path == expected_path, f"{label}: authoritative path drift: {path} != {expected_path}")
    digest = sha_text(record.get("sha256"), f"{label}.sha256")
    require(digest == expected_sha and boolean(record.get("gate_pass"), f"{label}.gate_pass"), f"{label}: pinned SHA/gate drift")
    require(path.is_file() and sha(path) == digest, f"{label}: current on-disk source drift")
    return digest


def validate_schema(value: object, label: str) -> None:
    require(integer(value, label) == SCHEMA_VERSION, f"{label}: schema-{SCHEMA_VERSION} evidence required")


def validate_pinned_reference_helper_load(
    value: object,
    helper_source: Mapping[str, Any],
    label: str,
) -> dict[str, object]:
    """Validate the no-JIT helper proof independently of performance data."""

    exact_keys(helper_source, {"path", "sha256", "gate_pass"}, f"{label}.source")
    source_path = string(helper_source.get("path"), f"{label}.source.path")
    source_sha = sha_text(helper_source.get("sha256"), f"{label}.source.sha256")
    require(
        source_path == EXPECTED_PINNED_REFERENCE_HELPER_PATH
        and source_sha == EXPECTED_PINNED_REFERENCE_HELPER_SHA256
        and boolean(helper_source.get("gate_pass"), f"{label}.source.gate_pass"),
        f"{label}: pinned helper source identity drift",
    )
    proof = mapping(value, label)
    exact_keys(proof, {"path", "sha256", "load_contract", "intercepted_names", "no_build"}, label)
    names = array(proof.get("intercepted_names"), f"{label}.intercepted_names")
    require(
        proof.get("path") == source_path
        and proof.get("sha256") == source_sha
        and proof.get("load_contract") == PINNED_REFERENCE_HELPER_LOAD_CONTRACT
        and names == ["sigmoid_ext"]
        and all(type(name) is str for name in names)
        and boolean(proof.get("no_build"), f"{label}.no_build"),
        f"{label}: helper must be a single intercepted sigmoid_ext no-build load",
    )
    return dict(proof)


def validate_identity(value: object, expected_runner: str, expected_analyzer: str, expected_shell: str, label: str) -> Mapping[str, Any]:
    identity = mapping(value, label)
    exact_keys(identity, {"roots", "source_ledger", "patched_tracked_status", "device", "gpu_uuid", "pinned_reference_helper_load", "commits"}, label)
    roots_view = mapping(identity.get("roots"), f"{label}.roots")
    exact_keys(roots_view, {"repo", "patched", "reference", "fla"}, f"{label}.roots")
    roots = {name: Path(string(roots_view.get(name), f"{label}.roots.{name}")) for name in roots_view}
    require(all(path.is_absolute() and path.is_dir() and path.resolve(strict=True) == path for path in roots.values()), f"{label}: roots must be existing canonical directories")
    owned = roots["repo"] / "assignment02/team/c1_flashkda/challenge_tail8191_production_freeze"
    auto_owned = roots["repo"] / "assignment02/team/c1_flashkda/challenge_tp8_dispatch"
    expected_paths = {
        "protocol_shell": owned / "run_clean_tail8191_production_freeze.sh",
        "runner": owned / "run_tail8191_production_freeze.py",
        "analyzer": owned / "analyze_tail8191_production_freeze.py",
        "auto_dispatch": auto_owned / "auto_dispatch.py",
        "fla_backend": auto_owned / "fla_backend.py",
        "harness": roots["repo"] / "assignment02/team/c1_flashkda/harness/validate_and_bench.py",
        "flash_kda_python": roots["patched"] / "flash_kda/__init__.py",
        "reference_torch_ref": roots["reference"] / "tests/torch_ref.py",
        "pinned_reference_helper": Path(EXPECTED_PINNED_REFERENCE_HELPER_PATH),
    }
    for relative in (key.removeprefix("fla:") for key in EXPECTED_STATIC_LEDGER_SHA256 if key.startswith("fla:")):
        expected_paths[f"fla:{relative}"] = roots["fla"] / relative
    for relative in (key.removeprefix("patched:") for key in EXPECTED_STATIC_LEDGER_SHA256 if key.startswith("patched:")):
        expected_paths[f"patched:{relative}"] = roots["patched"] / relative
    ledger = mapping(identity.get("source_ledger"), f"{label}.source_ledger")
    expected_keys = {"protocol_shell", "runner", "analyzer", *EXPECTED_STATIC_LEDGER_SHA256.keys()}
    exact_keys(ledger, expected_keys, f"{label}.source_ledger")
    extension = mapping(ledger.get("extension"), f"{label}.source_ledger.extension")
    extension_path = Path(string(extension.get("path"), f"{label}.source_ledger.extension.path"))
    require(extension_path.parent == roots["patched"] and extension_path.name.startswith("flash_kda_C.cpython-") and extension_path.name.endswith("-linux-gnu.so"), f"{label}: extension authoritative path drift")
    expected_paths["extension"] = extension_path
    expected_shas = {"protocol_shell": expected_shell, "runner": expected_runner, "analyzer": expected_analyzer, **EXPECTED_STATIC_LEDGER_SHA256}
    for name in sorted(expected_keys):
        _physical_sha(mapping(ledger[name], f"{label}.source_ledger.{name}"), expected_paths[name], expected_shas[name], f"{label}.source_ledger.{name}")
    validate_pinned_reference_helper_load(
        identity.get("pinned_reference_helper_load"),
        mapping(ledger["pinned_reference_helper"], f"{label}.source_ledger.pinned_reference_helper"),
        f"{label}.pinned_reference_helper_load",
    )
    status = array(identity.get("patched_tracked_status"), f"{label}.patched_tracked_status"); require(all(type(item) is str for item in status) and status == EXPECTED_PATCHED_TRACKED_STATUS, f"{label}: patched tracked status drift")
    device = mapping(identity.get("device"), f"{label}.device")
    exact_keys(device, {"name", "capability", "multiprocessor_count", "gate_pass"}, f"{label}.device")
    capability = [integer(item, f"{label}.device.capability[{index}]") for index, item in enumerate(array(device.get("capability"), f"{label}.device.capability"))]
    require("B300" in str(device.get("name", "")).upper() and capability == [10, 3] and integer(device.get("multiprocessor_count"), f"{label}.sm") == 148 and boolean(device.get("gate_pass"), f"{label}.device.gate"), f"{label}: not an exact B300 SM103a 148-SM identity")
    require(type(identity.get("gpu_uuid")) is str and bool(identity["gpu_uuid"]), f"{label}: GPU UUID missing")
    commits = mapping(identity.get("commits"), f"{label}.commits")
    require(dict(commits) == {"patched": EXPECTED_PATCHED_COMMIT, "reference": EXPECTED_REFERENCE_COMMIT, "fla": EXPECTED_FLA_COMMIT}, f"{label}: exact commit ledger drift")
    for name, expected in (("patched", EXPECTED_PATCHED_COMMIT), ("reference", EXPECTED_REFERENCE_COMMIT), ("fla", EXPECTED_FLA_COMMIT)):
        require(git_output(roots[name], "rev-parse", "HEAD").strip() == expected, f"{label}: current {name} commit drift")
    current_patched_status = git_output(roots["patched"], "status", "--porcelain=v1", "--untracked-files=no").splitlines()
    require(current_patched_status == EXPECTED_PATCHED_TRACKED_STATUS, f"{label}: current patched tracked status drift")
    require(not git_output(roots["reference"], "status", "--porcelain=v1", "--untracked-files=no").splitlines() and not git_output(roots["fla"], "status", "--porcelain=v1", "--untracked-files=no").splitlines(), f"{label}: current reference/FLA tracked tree dirty")
    return identity


def validate_raw(value: object, label: str) -> None:
    rows = mapping(value, label); require(set(rows) == set(CONTRACTS), f"{label}: raw contract set drift")
    for contract in CONTRACTS:
        row = mapping(rows[contract], f"{label}.{contract}"); exact_keys(row, {"baseline_vs_pinned_torch_reference", "vshard4_vs_pinned_torch_reference", "vshard4_vs_baseline", "immutability", "passed"}, f"{label}.{contract}"); require(boolean(row.get("passed"), f"{label}.{contract}.passed"), f"{label}.{contract}: raw gate failed")
        for relation in ("baseline_vs_pinned_torch_reference", "vshard4_vs_pinned_torch_reference", "vshard4_vs_baseline"):
            evidence = mapping(row.get(relation), f"{label}.{contract}.{relation}")
            expected_evidence = {"output_exact", "output_max_abs", "final_state_present"} if contract == "none" else {"output_exact", "output_max_abs", "final_state_present", "final_state_exact", "final_state_max_abs"}
            exact_keys(evidence, expected_evidence, f"{label}.{contract}.{relation}")
            require(boolean(evidence.get("output_exact"), f"{label}.{contract}.{relation}.out") and number(evidence.get("output_max_abs"), f"{label}.{contract}.{relation}.max") == 0.0, f"{label}.{contract}: raw output not exact")
            if contract == "none": require(boolean(evidence.get("final_state_present"), f"{label}.{contract}.{relation}.present") is False, f"{label}.{contract}: unexpected raw final")
            else: require(boolean(evidence.get("final_state_present"), f"{label}.{contract}.{relation}.present") is True and boolean(evidence.get("final_state_exact"), f"{label}.{contract}.{relation}.final") and number(evidence.get("final_state_max_abs"), f"{label}.{contract}.{relation}.finalmax") == 0.0, f"{label}.{contract}: raw final not exact")
        immutable = mapping(row.get("immutability"), f"{label}.{contract}.immutability"); require(set(immutable) == {"reference", "baseline", "vshard4"}, f"{label}.{contract}: immutability coverage drift")
        for evidence in immutable.values():
            detail = mapping(evidence, f"{label}.{contract}.immutable"); exact_keys(detail, {"input_immutability_exact", "input_immutability_fields", "initial_state_immutability_exact"}, f"{label}.{contract}.immutable"); fields = mapping(detail.get("input_immutability_fields"), f"{label}.{contract}.immutable.fields"); require(set(fields) == {"q", "k", "v", "g", "beta", "A_log", "dt_bias", "scale", "lower_bound"} and all(boolean(item, f"{label}.{contract}.immutable.field") for item in fields.values()) and boolean(detail.get("input_immutability_exact"), f"{label}.input") and boolean(detail.get("initial_state_immutability_exact"), f"{label}.initial"), f"{label}.{contract}: mutation evidence failed")


def validate_controls(value: object, label: str) -> None:
    rows = mapping(value, label); exact_keys(rows, {"negative_contracts", "neighborhoods", "passed"}, label); require(boolean(rows.get("passed"), f"{label}.passed"), f"{label}: negative control top gate failed")
    negatives = mapping(rows.get("negative_contracts"), f"{label}.negative_contracts"); require(set(negatives) == {"fp32_both", "bf16_both"}, f"{label}: negative state set drift")
    negative_reasons = {"fp32_both": "state_contract_fp32_both_h12_length_not_whitelisted", "bf16_both": "state_contract_bf16_both_length_head_not_whitelisted"}
    for name, row in negatives.items():
        detail = mapping(row, f"{label}.{name}"); exact_keys(detail, {"requested_variant", "chosen_variant", "reason", "scale", "lower_bound", "passed"}, f"{label}.{name}"); require(detail.get("requested_variant") == "baseline" and detail.get("chosen_variant") == "baseline" and detail.get("reason") == negative_reasons[name] and close(number(detail.get("scale"), f"{label}.{name}.scale"), 1.0 / math.sqrt(128)) and number(detail.get("lower_bound"), f"{label}.{name}.lower_bound") == -5.0 and boolean(detail.get("passed"), f"{label}.{name}.pass"), f"{label}.{name}: must stay exact baseline contract")
    neighborhood = mapping(rows.get("neighborhoods"), f"{label}.neighborhoods"); require(boolean(neighborhood.get("passed"), f"{label}.neighborhoods.pass"), f"{label}: neighborhood gate failed")
    exact_keys(neighborhood, {"selector_neighborhoods", "passed"}, f"{label}.neighborhoods")
    cases = mapping(neighborhood.get("selector_neighborhoods"), f"{label}.selector_neighborhoods"); require(set(cases) == {"h11", "t8190", "b2"}, f"{label}: neighborhood case drift")
    expected_shapes = {"h11": (1, 8191, 11), "t8190": (1, 8190, 12), "b2": (2, 8191, 12)}
    expected_reasons = {
        "h11": {"none": "state_contract_none_only_h12_whitelisted", "fp32_final_only": "state_contract_fla_fp32_final_only_only_h12_whitelisted"},
        "t8190": {"none": "state_contract_none_h12_length_not_whitelisted", "fp32_final_only": "state_contract_fla_fp32_final_only_h12_length_not_whitelisted"},
        "b2": {"none": "fixed_batch_shape_not_whitelisted", "fp32_final_only": "fixed_batch_shape_not_whitelisted"},
    }
    for name, case in cases.items():
        detail = mapping(case, f"{label}.neighborhood.{name}"); exact_keys(detail, {"shape", "scale", "lower_bound", "contracts", "passed"}, f"{label}.neighborhood.{name}")
        shape = mapping(detail.get("shape"), f"{label}.{name}.shape"); exact_keys(shape, {"B", "T", "H"}, f"{label}.{name}.shape")
        observed_shape = tuple(integer(shape.get(axis), f"{label}.{name}.shape.{axis}") for axis in ("B", "T", "H")); require(observed_shape == expected_shapes[name], f"{label}.{name}: shape drift")
        require(close(number(detail.get("scale"), f"{label}.{name}.scale"), 1.0 / math.sqrt(128)) and number(detail.get("lower_bound"), f"{label}.{name}.lower_bound") == -5.0 and boolean(detail.get("passed"), f"{label}.{name}.passed"), f"{label}.{name}: scalar/control drift")
        contracts = mapping(detail.get("contracts"), f"{label}.{name}.contracts"); require(set(contracts) == set(CONTRACTS), f"{label}.{name}: neighborhood contract coverage drift")
        for contract in CONTRACTS:
            route = mapping(contracts[contract], f"{label}.{name}.{contract}"); exact_keys(route, {"requested_variant", "chosen_variant", "reason", "passed"}, f"{label}.{name}.{contract}"); require(route.get("requested_variant") == "baseline" and route.get("chosen_variant") == "baseline" and route.get("reason") == expected_reasons[name][contract] and boolean(route.get("passed"), f"{label}.{name}.{contract}.passed"), f"{label}.{name}/{contract}: neighborhood is not exact baseline")


def validate_c1_decision(value: object, contract: str, label: str) -> None:
    decision = mapping(value, label)
    exact_keys(decision, {"requested_variant", "chosen_variant", "reason", "extension_sha256", "varlen_cpu_authoritative", "certified_varlen_offsets", "canonical_cache_hit"}, label)
    require(
        decision.get("requested_variant") == "vshard4_p2"
        and decision.get("chosen_variant") == "vshard4_p2"
        and decision.get("reason") == EXPECTED_REASON[contract]
        and decision.get("extension_sha256") == EXPECTED_EXTENSION_SHA256
        and boolean(decision.get("varlen_cpu_authoritative"), f"{label}.varlen_cpu_authoritative") is False
        and decision.get("certified_varlen_offsets") is None
        and decision.get("canonical_cache_hit") is None
        and "test_only_route" not in decision,
        f"{label}: not the exact production vshard4 decision",
    )


def validate_repeat(value: object, contract: str, process: int, repeat: int, label: str) -> dict[str, object]:
    row = mapping(value, label)
    exact_keys(row, {"process_index", "repeat_index", "event_contract", "schedule", "first_path_counts", "timed_route_proof_counts", "timed_call_audit_checks_outside_event", "timed_registry_spy_present", "public_precheck", "input_immutability_exact", "input_immutability_fields", "initial_state_immutability_exact", "raw_samples_ms", "paths", "c1_margin_over_pinned_by_percentile", "repeat_gate_pass", "passed"}, label)
    require(integer(row.get("process_index"), f"{label}.process") == process and integer(row.get("repeat_index"), f"{label}.repeat") == repeat and boolean(row.get("passed"), f"{label}.passed"), f"{label}: repeat identity drift")
    require(row.get("event_contract") == EXPECTED_EVENT_CONTRACT and row.get("schedule") == "alternating two-path; 100 warmups, one post-warmup synchronize, 1000 samples per path" and boolean(row.get("timed_call_audit_checks_outside_event"), f"{label}.audit_outside_event") and boolean(row.get("timed_registry_spy_present"), f"{label}.timed_registry_spy_present") is False, f"{label}: event/schedule/instrumentation contract drift")
    require(boolean(row.get("input_immutability_exact"), f"{label}.input") and boolean(row.get("initial_state_immutability_exact"), f"{label}.initial"), f"{label}: performance mutation")
    fields = mapping(row.get("input_immutability_fields"), f"{label}.immutability_fields"); require(set(fields) == {"q", "k", "v", "g", "beta", "A_log", "dt_bias", "scale", "lower_bound"} and all(boolean(value, f"{label}.immutability_field") for value in fields.values()), f"{label}: incomplete immutability proof")
    first = mapping(row.get("first_path_counts"), f"{label}.first"); exact_keys(first, set(PATHS), f"{label}.first"); require(integer(first.get("pinned_public"), f"{label}.first.pinned_public") == 500 and integer(first.get("c1_production_public"), f"{label}.first.c1_production_public") == 500, f"{label}: alternating order drift")
    verified = mapping(row.get("timed_route_proof_counts"), f"{label}.timed_route_proof_counts"); require(set(verified) == set(PATHS) and all(integer(verified.get(path), f"{label}.verified.{path}") == SAMPLES for path in PATHS), f"{label}: timed route proof coverage drift")
    precheck = mapping(row.get("public_precheck"), f"{label}.precheck"); exact_keys(precheck, {"pinned", "c1_production", "exact", "registry_spy_restored_before_timing"}, f"{label}.precheck"); require(boolean(precheck.get("registry_spy_restored_before_timing"), f"{label}.precheck.registry_restored"), f"{label}: registry spy not restored before timing"); c1 = mapping(precheck.get("c1_production"), f"{label}.c1"); pinned = mapping(precheck.get("pinned"), f"{label}.pinned")
    exact_keys(c1, {"c1_spy_delta", "pinned_spy_delta", "decision", "passed"}, f"{label}.c1")
    exact_keys(pinned, {"c1_spy_delta", "pinned_spy_delta", "passed"}, f"{label}.pinned")
    require(integer(c1.get("c1_spy_delta"), f"{label}.c1.delta") == 1 and integer(c1.get("pinned_spy_delta"), f"{label}.pinned.delta") == 0 and boolean(c1.get("passed"), f"{label}.c1.pass"), f"{label}: C1 registry proof failed")
    validate_c1_decision(c1.get("decision"), contract, f"{label}.decision")
    require(integer(pinned.get("c1_spy_delta"), f"{label}.pinned.c1") == 0 and integer(pinned.get("pinned_spy_delta"), f"{label}.pinned.pinned") == 1 and boolean(pinned.get("passed"), f"{label}.pinned.pass"), f"{label}: pinned registry proof failed")
    exact = mapping(precheck.get("exact"), f"{label}.exact")
    exact_keys(exact, {"output_exact", "output_max_abs", "final_state_present"} if contract == "none" else {"output_exact", "output_max_abs", "final_state_present", "final_state_exact", "final_state_max_abs"}, f"{label}.exact")
    require(boolean(exact.get("output_exact"), f"{label}.exact.output") and number(exact.get("output_max_abs"), f"{label}.exact.output_max_abs") == 0.0, f"{label}: public output not exact")
    if contract == "none":
        require(boolean(exact.get("final_state_present"), f"{label}.exact.final_state_present") is False, f"{label}: unexpected public final state")
    else:
        require(boolean(exact.get("final_state_present"), f"{label}.exact.final_state_present") and boolean(exact.get("final_state_exact"), f"{label}.exact.final_state_exact") and number(exact.get("final_state_max_abs"), f"{label}.exact.final_state_max_abs") == 0.0, f"{label}: public FP32 final state not exact")
    raw, paths = mapping(row.get("raw_samples_ms"), f"{label}.raw"), mapping(row.get("paths"), f"{label}.paths"); require(set(raw) == set(PATHS) and set(paths) == set(PATHS), f"{label}: path set drift")
    summary = {path: validate_summary(samples(raw[path], f"{label}.raw.{path}"), paths[path], f"{label}.paths.{path}") for path in PATHS}
    expected = {p: summary["pinned_public"][f"{p}_ms"] / summary["c1_production_public"][f"{p}_ms"] - 1.0 for p in PERCENTILES}; reported = mapping(row.get("c1_margin_over_pinned_by_percentile"), f"{label}.margins"); require(set(reported) == set(PERCENTILES), f"{label}: margin key drift")
    for name, margin in expected.items(): require(close(number(reported.get(name), f"{label}.margin.{name}"), margin), f"{label}: forged margin {name}")
    passed = all(item >= MIN_MARGIN for item in expected.values())
    require(boolean(row.get("repeat_gate_pass"), f"{label}.repeat_gate_pass") is passed, f"{label}: stale repeat gate")
    return {"margins": expected, "repeat_gate_pass": passed}


def validate_main(record: Mapping[str, Any], expected_runner: str, expected_analyzer: str, expected_shell: str, allocation: str, process: int, label: str) -> dict[str, object]:
    exact_keys(record, {"schema_version", "purpose", "describe_only", "allocation_id", "process_index", "shape", "scale", "lower_bound", "contracts", "public_paths", "fresh_pids_per_allocation", "repeats_per_pid", "samples_per_path_repeat", "required_percentiles", "minimum_c1_margin", "test_only_route_installed", "pinned_reference_helper", "pid", "slurm_job_id", "identity", "artifact_content_identity", "raw_abi_correctness", "public_benchmarks", "negative_controls", "registry_spy_restored", "registry_spy_restore_proof", "complete"}, label)
    validate_schema(record.get("schema_version"), f"{label}.schema")
    require(record.get("purpose") == "two-allocation real C1 production public-route T8191 freeze" and boolean(record.get("describe_only"), f"{label}.describe_only") is False and record.get("allocation_id") == allocation and integer(record.get("process_index"), f"{label}.process_index") == process and integer(record.get("pid"), f"{label}.pid") > 0 and boolean(record.get("complete"), f"{label}.complete"), f"{label}: top-level identity drift")
    shape = mapping(record.get("shape"), f"{label}.shape"); exact_keys(shape, {"B", "H", "T", "K", "V"}, f"{label}.shape"); require(tuple(integer(shape.get(axis), f"{label}.shape.{axis}") for axis in ("B", "H", "T", "K", "V")) == (1, 12, 8191, 128, 128), f"{label}: shape drift")
    require(close(number(record.get("scale"), f"{label}.scale"), 1.0 / math.sqrt(128)) and number(record.get("lower_bound"), f"{label}.lower_bound") == -5.0, f"{label}: scalar contract drift")
    require(integer(record.get("fresh_pids_per_allocation"), f"{label}.fresh_pids") == 2 and integer(record.get("repeats_per_pid"), f"{label}.repeats") == REPEATS and integer(record.get("samples_per_path_repeat"), f"{label}.samples") == SAMPLES and record.get("required_percentiles") == list(PERCENTILES) and close(number(record.get("minimum_c1_margin"), f"{label}.margin"), MIN_MARGIN), f"{label}: preregistration drift")
    require(record.get("test_only_route_installed") is False and record.get("contracts") == list(CONTRACTS) and record.get("public_paths") == list(PATHS) and boolean(record.get("registry_spy_restored"), f"{label}.registry_spy_restored"), f"{label}: protocol scope/spy restoration drift")
    restore = mapping(record.get("registry_spy_restore_proof"), f"{label}.registry_spy_restore_proof"); exact_keys(restore, {"restored", "c1_instance_slot_restored", "pinned_instance_slot_restored"}, f"{label}.registry_spy_restore_proof"); require(all(boolean(restore.get(name), f"{label}.registry_spy_restore_proof.{name}") for name in restore), f"{label}: registry instance-slot provenance was not restored")
    job = positive_decimal_job(record.get("slurm_job_id"), f"{label}.slurm_job_id")
    identity = validate_identity(record.get("identity"), expected_runner, expected_analyzer, expected_shell, f"{label}.identity")
    helper_source = mapping(mapping(identity.get("source_ledger"), f"{label}.identity.source_ledger").get("pinned_reference_helper"), f"{label}.identity.source_ledger.pinned_reference_helper")
    validate_pinned_reference_helper_load(record.get("pinned_reference_helper"), helper_source, f"{label}.pinned_reference_helper")
    validate_raw(record.get("raw_abi_correctness"), f"{label}.raw"); validate_controls(record.get("negative_controls"), f"{label}.controls")
    content_identity = mapping(record.get("artifact_content_identity"), f"{label}.artifact_content_identity")
    exact_keys(content_identity, {"allocation_id", "process_index", "pid", "slurm_job_id", "gpu_uuid", "identity_sha256", "pinned_reference_helper_load"}, f"{label}.artifact_content_identity")
    content_helper_load = validate_pinned_reference_helper_load(content_identity.get("pinned_reference_helper_load"), helper_source, f"{label}.artifact_content_identity.pinned_reference_helper_load")
    require(
        content_identity.get("allocation_id") == allocation
        and integer(content_identity.get("process_index"), f"{label}.artifact_content_identity.process_index") == process
        and integer(content_identity.get("pid"), f"{label}.artifact_content_identity.pid") == integer(record.get("pid"), f"{label}.pid")
        and positive_decimal_job(content_identity.get("slurm_job_id"), f"{label}.artifact_content_identity.slurm_job_id") == job
        and string(content_identity.get("gpu_uuid"), f"{label}.artifact_content_identity.gpu_uuid") == string(identity.get("gpu_uuid"), f"{label}.identity.gpu_uuid")
        and sha_text(content_identity.get("identity_sha256"), f"{label}.artifact_content_identity.identity_sha256") == canonical_sha(identity, f"{label}.identity")
        and canonical_sha(content_helper_load, f"{label}.artifact_content_identity.pinned_reference_helper_load") == canonical_sha(mapping(identity.get("pinned_reference_helper_load"), f"{label}.identity.pinned_reference_helper_load"), f"{label}.identity.pinned_reference_helper_load"),
        f"{label}: artifact content identity does not bind the raw main process/job/GPU/source identity",
    )
    bench = mapping(record.get("public_benchmarks"), f"{label}.bench"); require(set(bench) == set(CONTRACTS), f"{label}: benchmark contract drift"); contracts: dict[str, object] = {}
    for contract in CONTRACTS:
        repeats = array(bench[contract], f"{label}.{contract}"); require(len(repeats) == REPEATS, f"{label}.{contract}: repeat count drift"); evidence = [validate_repeat(row, contract, process, index, f"{label}.{contract}.repeat{index}") for index, row in enumerate(repeats)]; contracts[contract] = {"repeats": evidence, "contract_gate_pass": all(boolean(item["repeat_gate_pass"], f"{label}.{contract}.repeat_gate_pass") for item in evidence)}
    return {"process_index": process, "pid": record["pid"], "slurm_job_id": job, "identity": identity, "content_identity": content_identity, "contracts": contracts}


def source_signature(identity: Mapping[str, Any]) -> dict[str, object]:
    return {
        "roots": dict(mapping(identity.get("roots"), "identity.roots")),
        "source_ledger": dict(mapping(identity.get("source_ledger"), "identity.source_ledger")),
        "patched_tracked_status": list(array(identity.get("patched_tracked_status"), "identity.patched_tracked_status")),
        "pinned_reference_helper_load": dict(mapping(identity.get("pinned_reference_helper_load"), "identity.pinned_reference_helper_load")),
        "commits": dict(mapping(identity.get("commits"), "identity.commits")),
    }


def canonical_existing_file(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AuditError(f"{label}: path is unavailable") from exc
    require(path.is_absolute() and path == resolved and resolved.is_file(), f"{label}: path must be an existing canonical absolute file")
    return resolved


def validate_allocation_prerequisite_arguments(allocation_id: object, a1_audit: object, expected_a1_sha256: object, current_slurm_job_id: object) -> None:
    require(allocation_id in ("A1", "A2"), "allocation must be A1 or A2")
    if allocation_id == "A1":
        require(a1_audit is None and expected_a1_sha256 is None and current_slurm_job_id is None, "A1 allocation must reject A1-prerequisite and current-A2-job arguments")
    else:
        require(a1_audit is not None and expected_a1_sha256 is not None and current_slurm_job_id is not None, "A2 allocation requires A1 audit path/SHA and its current Slurm job")
        sha_text(expected_a1_sha256, "A2.expected_a1_sha256")
        positive_decimal_job(current_slurm_job_id, "A2.current_slurm_job_id")


def validate_allocation_identity(value: object, label: str) -> dict[str, object]:
    identity = mapping(value, label)
    exact_keys(identity, {"distinct_pids", "slurm_job_id", "gpu_uuid", "passed"}, label)
    pids = [integer(pid, f"{label}.distinct_pids[{index}]") for index, pid in enumerate(array(identity.get("distinct_pids"), f"{label}.distinct_pids"))]
    require(len(pids) == 2 and len(set(pids)) == 2 and all(pid > 0 for pid in pids), f"{label}: two ordered positive fresh PIDs required")
    job = positive_decimal_job(identity.get("slurm_job_id"), f"{label}.slurm_job_id")
    gpu_uuid = string(identity.get("gpu_uuid"), f"{label}.gpu_uuid")
    require(boolean(identity.get("passed"), f"{label}.passed"), f"{label}: identity gate failed")
    return {"pids": pids, "slurm_job_id": job, "gpu_uuid": gpu_uuid}


def reopen_main_artifacts(
    artifacts_value: object,
    allocation: str,
    allocation_identity: Mapping[str, object],
    allocation_source: Mapping[str, Any],
    expected_runner: str,
    expected_analyzer: str,
    expected_shell: str,
    label: str,
) -> list[dict[str, object]]:
    """Reopen raw evidence and bind every signed main to its allocation audit."""
    artifacts = array(artifacts_value, f"{label}.artifacts")
    require(len(artifacts) == 2, f"{label}: exactly two raw main artifacts required")
    pids = [integer(pid, f"{label}.identity.pids[{index}]") for index, pid in enumerate(array(allocation_identity.get("pids"), f"{label}.identity.pids"))]
    require(len(pids) == 2 and len(set(pids)) == 2 and all(pid > 0 for pid in pids), f"{label}: invalid allocation PID identity")
    job = positive_decimal_job(allocation_identity.get("slurm_job_id"), f"{label}.identity.slurm_job_id")
    gpu_uuid = string(allocation_identity.get("gpu_uuid"), f"{label}.identity.gpu_uuid")
    source_digest = canonical_sha(allocation_source, f"{label}.source_identity")
    seen_paths: set[str] = set()
    rows: list[dict[str, object]] = []
    for index, artifact_value in enumerate(artifacts):
        artifact = mapping(artifact_value, f"{label}.artifact{index}")
        exact_keys(artifact, {"process_index", "path", "sha256", "content_identity"}, f"{label}.artifact{index}")
        require(integer(artifact.get("process_index"), f"{label}.artifact{index}.process_index") == index, f"{label}: artifact process index drift")
        artifact_path = canonical_existing_file(Path(string(artifact.get("path"), f"{label}.artifact{index}.path")), f"{label}.artifact{index}.path")
        require(str(artifact_path) not in seen_paths, f"{label}: raw main artifact paths must be distinct")
        seen_paths.add(str(artifact_path))
        artifact_sha = sha_text(artifact.get("sha256"), f"{label}.artifact{index}.sha256")
        main, actual_sha = read(artifact_path, artifact_sha, f"{label}.main{index}")
        require(actual_sha == artifact_sha, f"{label}.main{index}: artifact SHA reread drift")
        row = validate_main(main, expected_runner, expected_analyzer, expected_shell, allocation, index, f"{label}.main{index}")
        recorded_content = mapping(artifact.get("content_identity"), f"{label}.artifact{index}.content_identity")
        require(
            canonical_sha(recorded_content, f"{label}.artifact{index}.content_identity")
            == canonical_sha(row["content_identity"], f"{label}.main{index}.content_identity"),
            f"{label}.main{index}: signed content identity differs from raw main",
        )
        row_identity = mapping(row["identity"], f"{label}.main{index}.identity")
        require(
            integer(row["process_index"], f"{label}.main{index}.process_index") == index
            and integer(row["pid"], f"{label}.main{index}.pid") == pids[index]
            and positive_decimal_job(row["slurm_job_id"], f"{label}.main{index}.slurm_job_id") == job
            and string(row_identity.get("gpu_uuid"), f"{label}.main{index}.gpu_uuid") == gpu_uuid
            and canonical_sha(source_signature(row_identity), f"{label}.main{index}.source_identity") == source_digest,
            f"{label}.main{index}: process/PID/job/GPU/source identity is not exactly bound to its allocation audit",
        )
        rows.append(row)
    return rows


def allocation(args: argparse.Namespace) -> bool:
    actual = sha(Path(__file__).resolve(strict=True)); require(actual == args.expected_analyzer_sha256, "analyzer source SHA mismatch")
    validate_allocation_prerequisite_arguments(args.allocation, args.a1_audit, args.expected_a1_sha256, args.current_slurm_job_id)
    data = [read(path, digest, f"main{index}") for index, (path, digest) in enumerate(zip(args.main_json, args.expected_main_sha256, strict=True))]
    rows = [validate_main(record, args.expected_runner_sha256, actual, args.expected_protocol_shell_sha256, args.allocation, index, f"main{index}") for index, (record, _digest) in enumerate(data)]
    pids = [integer(row["pid"], f"pid{index}") for index, row in enumerate(rows)]; require(len(set(pids)) == 2, f"{args.allocation}: two fresh PIDs required")
    jobs = {positive_decimal_job(row["slurm_job_id"], f"main{index}.slurm_job_id") for index, row in enumerate(rows)}
    uuids = {string(mapping(row["identity"], f"identity{index}").get("gpu_uuid"), f"identity{index}.gpu_uuid") for index, row in enumerate(rows)}
    require(len(jobs) == len(uuids) == 1, f"{args.allocation}: job/GPU identity drift")
    signatures = [source_signature(mapping(row["identity"], f"identity{index}")) for index, row in enumerate(rows)]
    require(canonical_sha(signatures[0], f"{args.allocation}.source0") == canonical_sha(signatures[1], f"{args.allocation}.source1"), f"{args.allocation}: source identity differs across fresh PIDs")
    a1_prerequisite: object = None
    if args.allocation == "A1":
        require(args.a1_audit is None and args.expected_a1_sha256 is None and args.current_slurm_job_id is None, "A1 allocation must reject A1-prerequisite and current-A2-job arguments")
    else:
        require(args.a1_audit is not None and args.expected_a1_sha256 is not None, "A2 allocation audit requires the fixed A1 audit path and SHA")
        require(positive_decimal_job(args.current_slurm_job_id, "A2.current_slurm_job_id") == next(iter(jobs)), "A2 current Slurm job differs from its raw main artifacts")
        a1_prerequisite = reopen_a1_prerequisite(args.a1_audit, args.expected_a1_sha256, next(iter(jobs)), args.expected_runner_sha256, actual, args.expected_protocol_shell_sha256)
        require(canonical_sha(mapping(mapping(a1_prerequisite, "A2.a1_prerequisite").get("source_identity"), "A2.a1_prerequisite.source_identity"), "A2.a1_prerequisite.source_identity") == canonical_sha(signatures[0], "A2.source_identity"), "A2 source identity differs from fixed A1")
    assessments: dict[str, object] = {}; eligible = True
    for contract in CONTRACTS:
        repeats: list[object] = []
        for row in rows:
            contract_evidence = mapping(mapping(row["contracts"], "contracts").get(contract), f"{contract}.contracts")
            repeats.extend(array(contract_evidence.get("repeats"), f"{contract}.repeats"))
        passed = all(boolean(mapping(item, "repeat").get("repeat_gate_pass"), "repeat.repeat_gate_pass") for item in repeats)
        assessments[contract] = {"all_four_repeats": repeats, "contract_gate_pass": passed}
        eligible = eligible and passed
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "one clean production T8191 allocation; no source mutation",
        "allocation_id": args.allocation,
        "source_identity": signatures[0],
        "a1_prerequisite": a1_prerequisite,
        "artifacts": [
            {"process_index": index, "path": str(path.resolve(strict=True)), "sha256": digest, "content_identity": dict(mapping(rows[index]["content_identity"], f"main{index}.content_identity"))}
            for index, (path, (_record, digest)) in enumerate(zip(args.main_json, data, strict=True))
        ],
        "identity_consistency": {"distinct_pids": pids, "slurm_job_id": next(iter(jobs)), "gpu_uuid": next(iter(uuids)), "passed": True},
        "contract_assessment": assessments,
        "allocation_gate": {"eligible": eligible, "decision": "eligible_for_A2_only" if args.allocation == "A1" and eligible else ("eligible_for_cross_allocation_freeze_review" if eligible else "STOP_keep_production_baseline")},
        "complete": True,
    }
    write(args.json, payload); print(f"wrote {args.allocation} production-freeze allocation audit {args.json}; eligible={eligible}"); return eligible


def validate_allocation(record: Mapping[str, Any], expected_allocation: str, expected_runner: str, expected_analyzer: str, expected_shell: str, label: str) -> Mapping[str, Any]:
    exact_keys(record, {"schema_version", "purpose", "allocation_id", "source_identity", "a1_prerequisite", "artifacts", "identity_consistency", "contract_assessment", "allocation_gate", "complete"}, label)
    validate_schema(record.get("schema_version"), f"{label}.schema")
    require(record.get("purpose") == "one clean production T8191 allocation; no source mutation" and record.get("allocation_id") == expected_allocation and boolean(record.get("complete"), f"{label}.complete"), f"{label}: allocation identity drift")
    source = mapping(record.get("source_identity"), f"{label}.source"); exact_keys(source, {"roots", "source_ledger", "patched_tracked_status", "pinned_reference_helper_load", "commits"}, f"{label}.source")
    ledger = mapping(source.get("source_ledger"), f"{label}.source.ledger")
    expected_ledger_keys = {"protocol_shell", "runner", "analyzer", *EXPECTED_STATIC_LEDGER_SHA256.keys()}; exact_keys(ledger, expected_ledger_keys, f"{label}.source.ledger")
    required_digests = {"protocol_shell": expected_shell, "runner": expected_runner, "analyzer": expected_analyzer, **EXPECTED_STATIC_LEDGER_SHA256}
    for name, digest in required_digests.items():
        entry = mapping(ledger[name], f"{label}.source.ledger.{name}"); exact_keys(entry, {"path", "sha256", "gate_pass"}, f"{label}.source.ledger.{name}"); require(entry.get("sha256") == digest and boolean(entry.get("gate_pass"), f"{label}.source.ledger.{name}.gate"), f"{label}: source digest drift for {name}")
    validate_pinned_reference_helper_load(
        source.get("pinned_reference_helper_load"),
        mapping(ledger["pinned_reference_helper"], f"{label}.source.ledger.pinned_reference_helper"),
        f"{label}.source.pinned_reference_helper_load",
    )
    require(dict(mapping(source.get("commits"), f"{label}.source.commits")) == {"patched": EXPECTED_PATCHED_COMMIT, "reference": EXPECTED_REFERENCE_COMMIT, "fla": EXPECTED_FLA_COMMIT}, f"{label}: commit signature drift")
    status = array(source.get("patched_tracked_status"), f"{label}.source.patched_tracked_status"); require(all(type(item) is str for item in status) and status == EXPECTED_PATCHED_TRACKED_STATUS, f"{label}: patched tracked status signature drift")
    identity = validate_allocation_identity(record.get("identity_consistency"), f"{label}.identity")
    pids = [integer(item, f"{label}.identity.pids[{index}]") for index, item in enumerate(array(identity["pids"], f"{label}.identity.pids"))]
    job = positive_decimal_job(identity["slurm_job_id"], f"{label}.identity.slurm_job_id")
    artifacts = array(record.get("artifacts"), f"{label}.artifacts")
    rows = reopen_main_artifacts(artifacts, expected_allocation, identity, source, expected_runner, expected_analyzer, expected_shell, label)
    if expected_allocation == "A1":
        require(record.get("a1_prerequisite") is None, f"{label}: A1 must not claim an A1 prerequisite")
    else:
        prerequisite = mapping(record.get("a1_prerequisite"), f"{label}.a1_prerequisite")
        exact_keys(prerequisite, {"path", "sha256", "slurm_job_id", "source_identity"}, f"{label}.a1_prerequisite")
        prerequisite_path = canonical_existing_file(Path(string(prerequisite.get("path"), f"{label}.a1.path")), f"{label}.a1.path")
        prerequisite_sha = sha_text(prerequisite.get("sha256"), f"{label}.a1.sha")
        prerequisite_job = positive_decimal_job(prerequisite.get("slurm_job_id"), f"{label}.a1.job")
        prerequisite_source = mapping(prerequisite.get("source_identity"), f"{label}.a1.source")
        require(prerequisite_job != job and canonical_sha(prerequisite_source, f"{label}.a1.source") == canonical_sha(source, f"{label}.source"), f"{label}: invalid or same-job A1 prerequisite binding")
        reopened = reopen_a1_prerequisite(prerequisite_path, prerequisite_sha, job, expected_runner, expected_analyzer, expected_shell)
        require(
            str(prerequisite_path) == reopened["path"]
            and prerequisite_sha == reopened["sha256"]
            and prerequisite_job == reopened["slurm_job_id"]
            and canonical_sha(prerequisite_source, f"{label}.a1.source") == canonical_sha(reopened["source_identity"], f"{label}.a1.reopened_source"),
            f"{label}: A2 prerequisite does not exactly bind its reopened A1 audit",
        )
    contracts = mapping(record.get("contract_assessment"), f"{label}.contracts")
    expected_assessment: dict[str, object] = {}
    eligible = True
    for contract in CONTRACTS:
        repeats: list[object] = []
        for row in rows:
            evidence = mapping(mapping(row["contracts"], f"{label}.main.contracts").get(contract), f"{label}.main.{contract}")
            repeats.extend(array(evidence.get("repeats"), f"{label}.main.{contract}.repeats"))
        passed = all(boolean(mapping(repeat, f"{label}.{contract}.repeat").get("repeat_gate_pass"), f"{label}.{contract}.repeat_gate_pass") for repeat in repeats)
        expected_assessment[contract] = {"all_four_repeats": repeats, "contract_gate_pass": passed}
        eligible = eligible and passed
    require(canonical_sha(contracts, f"{label}.contract_assessment") == canonical_sha(expected_assessment, f"{label}.recomputed_contract_assessment"), f"{label}: allocation contract assessment is not the exact raw-main recomputation")
    gate = mapping(record.get("allocation_gate"), f"{label}.gate"); exact_keys(gate, {"eligible", "decision"}, f"{label}.gate")
    require(boolean(gate.get("eligible"), f"{label}.eligible") is eligible and eligible, f"{label}: allocation not exactly eligible from reopened raw evidence")
    expected_decision = "eligible_for_A2_only" if expected_allocation == "A1" else "eligible_for_cross_allocation_freeze_review"
    require(gate.get("decision") == expected_decision, f"{label}: allocation decision drift")
    return {"pids": pids, "slurm_job_id": job, "gpu_uuid": identity["gpu_uuid"], "source_identity": source, "artifacts": artifacts, "rows": rows, "eligible": eligible}


def reopen_a1_prerequisite(audit_path: Path, expected_audit_sha256: str, current_job: str, expected_runner: str, expected_analyzer: str, expected_shell: str) -> dict[str, object]:
    resolved_audit_path = canonical_existing_file(audit_path, "allocation")
    audit, audit_sha = read(resolved_audit_path, expected_audit_sha256, "allocation")
    require(audit.get("allocation_id") == "A1", "A2 prerequisite must be a fixed eligible A1 audit")
    identity = validate_allocation(audit, "A1", expected_runner, expected_analyzer, expected_shell, "allocation")
    current = positive_decimal_job(current_job, "current_slurm_job_id")
    require(current != identity["slurm_job_id"], "A2 must run in a distinct positive Slurm job from A1")
    return {"path": str(resolved_audit_path), "sha256": audit_sha, "slurm_job_id": identity["slurm_job_id"], "source_identity": dict(mapping(identity["source_identity"], "allocation.source"))}


def verify_allocation(args: argparse.Namespace) -> bool:
    """Reopen a completed A1 before a later shell is allowed to launch A2."""
    actual = sha(Path(__file__).resolve(strict=True)); require(actual == args.expected_analyzer_sha256, "analyzer source SHA mismatch")
    reopen_a1_prerequisite(args.audit, args.expected_audit_sha256, args.current_slurm_job_id, args.expected_runner_sha256, actual, args.expected_protocol_shell_sha256)
    print("verified fixed A1 as eligible prerequisite in a distinct Slurm job")
    return True


def chain(args: argparse.Namespace) -> bool:
    actual = sha(Path(__file__).resolve(strict=True)); require(actual == args.expected_analyzer_sha256, "analyzer source SHA mismatch")
    a1_path = canonical_existing_file(args.a1_audit, "A1")
    a2_path = canonical_existing_file(args.a2_audit, "A2")
    a1, a1_sha = read(a1_path, args.expected_a1_sha256, "A1"); a2, a2_sha = read(a2_path, args.expected_a2_sha256, "A2")
    i1 = validate_allocation(a1, "A1", args.expected_runner_sha256, actual, args.expected_protocol_shell_sha256, "A1")
    i2 = validate_allocation(a2, "A2", args.expected_runner_sha256, actual, args.expected_protocol_shell_sha256, "A2")
    require(positive_decimal_job(i1["slurm_job_id"], "A1.job") != positive_decimal_job(i2["slurm_job_id"], "A2.job"), "A1/A2 must use distinct Slurm jobs")
    require(canonical_sha(mapping(i1["source_identity"], "A1.source"), "A1.source") == canonical_sha(mapping(i2["source_identity"], "A2.source"), "A2.source"), "A1/A2 full source identities differ")
    prerequisite = mapping(a2.get("a1_prerequisite"), "A2.a1_prerequisite")
    require(prerequisite.get("path") == str(a1_path) and prerequisite.get("sha256") == a1_sha and prerequisite.get("slurm_job_id") == i1["slurm_job_id"] and canonical_sha(mapping(prerequisite.get("source_identity"), "A2.a1.source"), "A2.a1.source") == canonical_sha(mapping(i1["source_identity"], "A1.source"), "A1.source"), "A2 is not bound to the supplied exact A1 audit")
    # Reopen every signed main artifact so the chain validates raw evidence and
    # current on-disk production identities a second time.
    all_artifact_paths: list[str] = []
    for allocation_name, checked in (("A1", i1), ("A2", i2)):
        artifacts = array(checked["artifacts"], f"{allocation_name}.artifacts")
        reopen_main_artifacts(
            artifacts,
            allocation_name,
            {"pids": checked["pids"], "slurm_job_id": checked["slurm_job_id"], "gpu_uuid": checked["gpu_uuid"]},
            mapping(checked["source_identity"], f"{allocation_name}.source"),
            args.expected_runner_sha256,
            actual,
            args.expected_protocol_shell_sha256,
            f"{allocation_name}.second_reopen",
        )
        all_artifact_paths.extend(str(canonical_existing_file(Path(string(mapping(item, f"{allocation_name}.artifact").get("path"), f"{allocation_name}.artifact.path")), f"{allocation_name}.artifact.path")) for item in artifacts)
    require(len(set(all_artifact_paths)) == 4, "A1/A2 require four distinct raw main artifact paths")
    payload: dict[str, object] = {"schema_version": SCHEMA_VERSION, "purpose": "two clean real-production T8191 public-route freeze allocations", "source_identity": dict(mapping(a1.get("source_identity"), "A1.source")), "allocations": {"A1": {"path": str(a1_path), "sha256": a1_sha, "slurm_job_id": i1["slurm_job_id"]}, "A2": {"path": str(a2_path), "sha256": a2_sha, "slurm_job_id": i2["slurm_job_id"]}}, "production_freeze_passed": True, "test_only_history_is_not_this_evidence": True, "complete": True}
    write(args.json, payload); print(f"wrote production-freeze chain {args.json}; production_freeze_passed=true"); return True


def self_test() -> None:
    raw = [1.0 + index / 1000 for index in range(SAMPLES)]; checked = recompute(raw); observed = {"samples": SAMPLES, **checked}; validate_summary(raw, observed, "synthetic")
    observed["p95_ms"] = observed["p95_ms"] + 0.01
    try: validate_summary(raw, observed, "synthetic_forged")
    except AuditError: pass
    else: raise AuditError("synthetic forged-summary rejection failed")
    try: number(True, "synthetic_bool_number")
    except AuditError: pass
    else: raise AuditError("bool-as-number rejection failed")
    require(canonical_sha({"value": True}, "synthetic_true") != canonical_sha({"value": 1}, "synthetic_one"), "canonical JSON identity lost bool/int type distinction")
    for forged_job in ("0", "01", "１"):
        try: positive_decimal_job(forged_job, "synthetic_job")
        except AuditError: pass
        else: raise AuditError("noncanonical Slurm job was accepted")
    try: validate_allocation_prerequisite_arguments("A1", Path("/tmp/fake-a1.json"), "0" * 64, None)
    except AuditError: pass
    else: raise AuditError("A1 silently accepted A1-prerequisite arguments")
    helper_source = {"path": EXPECTED_PINNED_REFERENCE_HELPER_PATH, "sha256": EXPECTED_PINNED_REFERENCE_HELPER_SHA256, "gate_pass": True}
    helper_proof = {"path": EXPECTED_PINNED_REFERENCE_HELPER_PATH, "sha256": EXPECTED_PINNED_REFERENCE_HELPER_SHA256, "load_contract": PINNED_REFERENCE_HELPER_LOAD_CONTRACT, "intercepted_names": ["sigmoid_ext"], "no_build": True}
    validate_pinned_reference_helper_load(helper_proof, helper_source, "synthetic_helper")
    forged_helper = dict(helper_proof); forged_helper["no_build"] = False
    try: validate_pinned_reference_helper_load(forged_helper, helper_source, "synthetic_forged_helper")
    except AuditError: pass
    else: raise AuditError("forged helper no-build proof was accepted")
    try: validate_schema(3, "synthetic_schema3")
    except AuditError: pass
    else: raise AuditError("schema-3 raw evidence was accepted after helper protocol migration")
    # Exercise the audit-to-main binding helper without a GPU or production tree:
    # JSON's False == 0 must *not* make a forged signed content identity pass.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = {"fixed_source": "synthetic"}
        artifacts: list[dict[str, object]] = []
        for index in range(2):
            path = root / f"main{index}.json"
            payload = {"content_identity": {"token": index}}
            path.write_text(json.dumps(payload), encoding="utf-8")
            artifacts.append({"process_index": index, "path": str(path.resolve(strict=True)), "sha256": sha(path), "content_identity": {"token": index}})
        original_validate_main, original_source_signature = validate_main, source_signature
        def synthetic_validate_main(record: Mapping[str, Any], _runner: str, _analyzer: str, _shell: str, _allocation: str, process: int, _label: str) -> dict[str, object]:
            return {"process_index": process, "pid": 101 + process, "slurm_job_id": "123", "identity": {"gpu_uuid": "GPU-synthetic"}, "content_identity": mapping(record.get("content_identity"), "synthetic.content_identity"), "contracts": {}}
        def synthetic_source_signature(_identity: Mapping[str, Any]) -> dict[str, object]:
            return dict(source)
        try:
            globals()["validate_main"] = synthetic_validate_main
            globals()["source_signature"] = synthetic_source_signature
            audit_identity = {"pids": [101, 102], "slurm_job_id": "123", "gpu_uuid": "GPU-synthetic"}
            require(len(reopen_main_artifacts(artifacts, "A1", audit_identity, source, "runner", "analyzer", "shell", "synthetic_binding")) == 2, "synthetic raw-main binding failed")
            forged = [dict(item) for item in artifacts]
            forged[0]["content_identity"] = {"token": False}
            try: reopen_main_artifacts(forged, "A1", audit_identity, source, "runner", "analyzer", "shell", "synthetic_forged_content")
            except AuditError: pass
            else: raise AuditError("bool-for-int forged raw-main content identity was accepted")
        finally:
            globals()["validate_main"] = original_validate_main
            globals()["source_signature"] = original_source_signature
    decision: dict[str, object] = {"requested_variant": "vshard4_p2", "chosen_variant": "vshard4_p2", "reason": EXPECTED_REASON["none"], "extension_sha256": EXPECTED_EXTENSION_SHA256, "varlen_cpu_authoritative": False, "certified_varlen_offsets": None, "canonical_cache_hit": None}
    validate_c1_decision(decision, "none", "synthetic_decision")
    decision["test_only_route"] = False
    try: validate_c1_decision(decision, "none", "synthetic_test_only_key")
    except AuditError: pass
    else: raise AuditError("test-only decision-key rejection failed")
    print("analyzer self-test PASS: raw recompute, strict identity types, helper/schema-3 rejection, A1 prerequisite rejection, and production decision-key rejection")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--chain", action="store_true"); parser.add_argument("--verify-allocation", action="store_true"); parser.add_argument("--allocation", choices=("A1", "A2")); parser.add_argument("--main-json", type=Path, nargs=2); parser.add_argument("--expected-main-sha256", nargs=2); parser.add_argument("--expected-runner-sha256"); parser.add_argument("--expected-analyzer-sha256"); parser.add_argument("--expected-protocol-shell-sha256"); parser.add_argument("--current-slurm-job-id"); parser.add_argument("--a1-audit", type=Path); parser.add_argument("--a2-audit", type=Path); parser.add_argument("--expected-a1-sha256"); parser.add_argument("--expected-a2-sha256"); parser.add_argument("--audit", type=Path); parser.add_argument("--expected-audit-sha256"); parser.add_argument("--json", type=Path); parser.add_argument("--require-pass", action="store_true"); args = parser.parse_args()
    if args.self_test: self_test(); return
    require(args.expected_analyzer_sha256 is not None and args.expected_runner_sha256 is not None and args.expected_protocol_shell_sha256 is not None, "expected analyzer/runner/protocol-shell SHA required"); sha_text(args.expected_analyzer_sha256, "expected analyzer SHA"); sha_text(args.expected_runner_sha256, "expected runner SHA"); sha_text(args.expected_protocol_shell_sha256, "expected protocol shell SHA")
    if args.verify_allocation:
        require(args.audit is not None and args.expected_audit_sha256 is not None and args.current_slurm_job_id is not None, "verify-allocation requires audit path/SHA and current Slurm job"); passed = verify_allocation(args)
    elif args.chain:
        require(args.json is not None, "chain requires --json")
        require(all(value is not None for value in (args.a1_audit, args.a2_audit, args.expected_a1_sha256, args.expected_a2_sha256)), "chain requires A1/A2 audit path and SHA"); passed = chain(args)
    else:
        require(args.json is not None, "allocation requires --json")
        require(args.allocation is not None and args.main_json is not None and args.expected_main_sha256 is not None, "allocation requires two main artifacts and hashes"); [sha_text(value, "main SHA") for value in args.expected_main_sha256]; passed = allocation(args)
    if args.require_pass and not passed: raise AuditError("preregistered production-freeze performance gate failed")


if __name__ == "__main__":
    try: main()
    except AuditError as exc: print(f"AUDIT_FAIL: {exc}", file=sys.stderr); raise SystemExit(2) from exc
