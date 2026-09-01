#!/usr/bin/env python3
"""Stdlib-only, fail-closed chain analyzer for the production skew freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


SCHEMA = 1
TARGET_OFFSETS = [0, 1, 2, 3, 4, 5, 12288]
TARGET_VARIANT = "vshard4_p2"
TARGET_REASON = "varlen_skew_n6_h12_t12288_fp32_both_whitelist_hit"
FALLBACK_OFFSETS = [0, 1, 2, 3, 4, 6, 12288]
FALLBACK_REASON = "varlen_offsets_not_whitelisted"
RUNNER_SHA_ENV = "C1_SKEW_PRODUCTION_FREEZE_RUNNER_SHA256"
ANALYZER_SHA_ENV = "C1_SKEW_PRODUCTION_FREEZE_ANALYZER_SHA256"
SHELL_SHA_ENV = "EXPECTED_PROTOCOL_SHELL_SHA256"
SHELL_PATH_ENV = "C1_SKEW_PRODUCTION_FREEZE_SHELL_PATH"
AUTO_SHA_ENV = "C1_SKEW_PRODUCTION_AUTO_DISPATCH_SHA256"
FLA_BACKEND_SHA_ENV = "C1_SKEW_PRODUCTION_FLA_BACKEND_SHA256"
POSITIVE_JOB = re.compile(r"[1-9][0-9]*\Z")
PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
PRODUCTION_WRAPPER_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
AUDITED_EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
FLA_FILE_SHA256 = {
    "fla/__init__.py": "b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d",
    "fla/ops/backends/__init__.py": "a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635",
    "fla/ops/kda/__init__.py": "24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb",
    "fla/ops/kda/backends/__init__.py": "86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797",
    "fla/ops/kda/backends/flash_kda.py": "0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2",
    "fla/ops/kda/chunk.py": "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8",
}
PATCHED_DIRTY_SHA256 = {
    "csrc/flash_kda.cpp": "38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4",
    "csrc/fwd.h": "613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083",
    "csrc/smxx/fwd_launch.cu": "a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928",
}
PINNED_LOAD_CONTRACT = "direct cached binary; exactly one pinned load_inline('sigmoid_ext') intercepted"
STATIC_LEDGER = {
    "varlen_metadata": "f89a97ba284c7a24a3df54efca7bb60eb70f9cbda659200bd6cb3e8dfdaf4ccd",
    "confirmation_runner": "9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b",
    "shared_seqcount_runner": "4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f",
    "candidate_helper": "e07481e7e66551f1f9eaafa7f1fb9db9ef193f1694ff831ee0a07802c8ffbe14",
    "prefetch2": "752126488487ac317a7ee167b660b0895562e4877aedbc4ce2a599c1f59a10d0",
    "vshard4_prefetch2": "445c90815919fb8c1db3bc79289d10029ea77b1d9accb393c326c120ba1f8385",
    "harness": "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52",
    "pinned_torch_ref": "bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5",
    "pinned_reference_helper": "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f",
}


class AuditError(AssertionError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise AuditError(message)


def obj(value: object, label: str) -> Mapping[str, Any]:
    require(type(value) is dict, label + " must be a JSON object")
    return value  # type: ignore[return-value]


def boolean(value: object, label: str) -> bool:
    require(type(value) is bool, label + " must be bool")
    return value


def integer(value: object, label: str) -> int:
    require(type(value) is int, label + " must be int")
    return value


def string(value: object, label: str) -> str:
    require(type(value) is str, label + " must be string")
    return value


def sha256(value: object, label: str) -> str:
    text = string(value, label)
    require(len(text) == 64 and all(character in "0123456789abcdef" for character in text), label + " must be lowercase SHA-256")
    return text


def strict_equal(actual: object, expected: object, label: str) -> None:
    """Recursive JSON equality that does not coerce True == 1 or 1 == 1.0."""

    if type(actual) is dict or type(expected) is dict:
        require(type(actual) is dict and type(expected) is dict, label + ": object type drift")
        require(set(actual) == set(expected) and all(type(key) is str for key in actual) and all(type(key) is str for key in expected), label + ": object key drift")
        for key in actual:
            strict_equal(actual[key], expected[key], label + "." + key)
        return
    if type(actual) is list or type(expected) is list:
        require(type(actual) is list and type(expected) is list and len(actual) == len(expected), label + ": list type/length drift")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            strict_equal(left, right, label + f"[{index}]")
        return
    require(type(actual) is type(expected) and actual == expected, label + ": scalar type/value drift")


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_json(payload: bytes, label: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError("non-finite JSON constant: " + value)
    try:
        return json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AuditError(label + ": invalid strict JSON") from exc


def read_json_once(path: Path, expected_sha: str, label: str) -> tuple[Mapping[str, Any], dict[str, str]]:
    """Hash and parse exactly one immutable read; never reopen between them."""

    expected = sha256(expected_sha, label + ".expected_sha")
    try:
        resolved = path.resolve(strict=True)
        payload = resolved.read_bytes()
    except OSError as exc:
        raise AuditError(label + ": cannot read artifact") from exc
    actual = _sha_bytes(payload)
    require(actual == expected, label + ": SHA mismatch")
    value = obj(_parse_json(payload, label), label)
    return value, {"path": str(resolved), "sha256": actual}


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def env_sha(name: str) -> str:
    return sha256(os.environ.get(name, ""), name)


def env_path(name: str) -> Path:
    value = os.environ.get(name, "")
    require(bool(value), name + " is required")
    try:
        return Path(value).resolve(strict=True)
    except OSError as exc:
        raise AuditError(name + " path does not resolve") from exc


def current_identity() -> dict[str, object]:
    """Revalidate the same source ledger that a raw runner records."""

    a02 = env_path("A02_ROOT")
    reference = env_path("REFERENCE_ROOT")
    helper = env_path("C1_PINNED_REFERENCE_HELPER_PATH")
    owned = (a02 / "team/c1_flashkda").resolve(strict=True)
    runner = owned / "challenge_varlen_fp32_both_production_freeze/run_varlen_fp32_both_production_freeze.py"
    analyzer = owned / "challenge_varlen_fp32_both_production_freeze/analyze_varlen_fp32_both_production_freeze.py"
    shell = owned / "challenge_varlen_fp32_both_production_freeze/run_clean_varlen_fp32_both_production_freeze.sh"
    expected_ledger = {
        "auto_dispatch": (owned / "challenge_tp8_dispatch/auto_dispatch.py", env_sha(AUTO_SHA_ENV)),
        "fla_backend": (owned / "challenge_tp8_dispatch/fla_backend.py", env_sha(FLA_BACKEND_SHA_ENV)),
        "varlen_metadata": (owned / "challenge_tp8_dispatch/varlen_metadata.py", STATIC_LEDGER["varlen_metadata"]),
        "confirmation_runner": (owned / "challenge_varlen_dispatch/run_varlen_dispatch_confirmation.py", STATIC_LEDGER["confirmation_runner"]),
        "shared_seqcount_runner": (owned / "challenge_seqcount_dispatch/run_seqcount_dispatch.py", STATIC_LEDGER["shared_seqcount_runner"]),
        "candidate_helper": (owned / "challenge_varlen_dispatch/run_varlen_fla_handoff_candidate.py", STATIC_LEDGER["candidate_helper"]),
        "prefetch2": (owned / "challenge_prefetch2/prefetch2.py", STATIC_LEDGER["prefetch2"]),
        "vshard4_prefetch2": (owned / "challenge_vshard4_prefetch2/vshard4_prefetch2.py", STATIC_LEDGER["vshard4_prefetch2"]),
        "harness": (owned / "harness/validate_and_bench.py", STATIC_LEDGER["harness"]),
        "pinned_torch_ref": (reference / "tests/torch_ref.py", STATIC_LEDGER["pinned_torch_ref"]),
        "pinned_reference_helper": (helper, STATIC_LEDGER["pinned_reference_helper"]),
    }
    ledger: dict[str, object] = {}
    for name, (path, digest) in expected_ledger.items():
        resolved = path.resolve(strict=True)
        require(_sha_bytes(resolved.read_bytes()) == digest, name + ": authoritative source SHA drift")
        ledger[name] = {"path": str(resolved), "expected_path": str(resolved), "sha256": digest, "sha256_gate_pass": True}
    runner_resolved = runner.resolve(strict=True)
    analyzer_expected = analyzer.resolve(strict=True)
    analyzer_actual = Path(__file__).resolve(strict=True)
    require(analyzer_actual == analyzer_expected, "analyzer must execute the canonical expected file, not a copied/spooled artifact")
    shell_expected = shell.resolve(strict=True)
    shell_actual = env_path(SHELL_PATH_ENV)
    require(shell_actual == shell_expected, "protocol shell must be the canonical expected file, not a copied/spooled artifact")
    runner_sha, analyzer_sha, shell_sha = env_sha(RUNNER_SHA_ENV), env_sha(ANALYZER_SHA_ENV), env_sha(SHELL_SHA_ENV)
    require(_sha_bytes(runner_resolved.read_bytes()) == runner_sha, "runner authoritative source SHA drift")
    # Keep the read and hash of this actually executed analyzer file together.
    analyzer_payload = analyzer_actual.read_bytes()
    require(_sha_bytes(analyzer_payload) == analyzer_sha, "analyzer authoritative source SHA drift")
    shell_payload = shell_actual.read_bytes()
    require(_sha_bytes(shell_payload) == shell_sha, "shell authoritative source SHA drift")
    return {
        "runner": {"path": str(runner_resolved), "expected_path": str(runner_resolved), "sha256": runner_sha, "sha256_gate_pass": True},
        "analyzer": {"path": str(analyzer_actual), "expected_path": str(analyzer_expected), "sha256": analyzer_sha, "sha256_gate_pass": True},
        "shell": {"path": str(shell_actual), "expected_path": str(shell_expected), "sha256": shell_sha, "sha256_gate_pass": True},
        "ledger": ledger,
        "production_hashes": {"auto_dispatch_env": AUTO_SHA_ENV, "auto_dispatch_sha256": env_sha(AUTO_SHA_ENV), "fla_backend_env": FLA_BACKEND_SHA_ENV, "fla_backend_sha256": env_sha(FLA_BACKEND_SHA_ENV), "passed": True},
    }


def exact_offsets(value: object, expected: list[int], label: str) -> list[int]:
    require(type(value) is list and value == expected and all(type(item) is int for item in value), label + ": exact offset/type drift")
    return list(expected)


def exact_delta(value: object, expected: Mapping[str, int], label: str) -> None:
    received = obj(value, label)
    require(set(received) == set(expected), label + ": delta key drift")
    for key, amount in expected.items():
        require(type(received.get(key)) is int and received.get(key) == amount, label + "." + key + ": delta type/value drift")


def zero_cache_stats(value: object, label: str) -> None:
    stats = obj(value, label)
    require(bool(stats), label + ": cache statistics are empty")
    for name, count in stats.items():
        require(type(name) is str and integer(count, label + "." + name) == 0, label + ": cache was not fully cleared")


def validate_control_evidence(
    value: object,
    label: str,
    expected_offsets: list[int],
    expected_accepted: bool,
    expected_reason: str | None,
) -> None:
    evidence = obj(value, label)
    require(set(evidence) == {"isolation", "verifier"}, label + ": control evidence key set drift")
    isolation = obj(evidence.get("isolation"), label + ".isolation")
    require(
        set(isolation) == {
            "clear_handoff_api", "metadata_clear_api", "cache_before_clear", "cache_after_clear",
            "handoff_empty_after_clear", "handoff_empty_after_public", "cache_after_cleanup", "passed",
        },
        label + ".isolation: key set drift",
    )
    require(
        isolation.get("clear_handoff_api") == "C1B300FlashKDABackend._clear_varlen_handoff"
        and isolation.get("metadata_clear_api") == "varlen_metadata.clear_cache",
        label + ".isolation: clear API drift",
    )
    before = obj(isolation.get("cache_before_clear"), label + ".isolation.before")
    require(bool(before), label + ".isolation.before: cache statistics are empty")
    for name, count in before.items():
        require(type(name) is str and integer(count, label + ".isolation.before." + name) >= 0, label + ".isolation.before: invalid statistic")
    zero_cache_stats(isolation.get("cache_after_clear"), label + ".isolation.after_clear")
    zero_cache_stats(isolation.get("cache_after_cleanup"), label + ".isolation.after_cleanup")
    require(
        boolean(isolation.get("handoff_empty_after_clear"), label + ".isolation.empty_after_clear")
        and boolean(isolation.get("handoff_empty_after_public"), label + ".isolation.empty_after_public")
        and boolean(isolation.get("passed"), label + ".isolation.passed"),
        label + ".isolation: handoff/cache isolation failed",
    )
    verifier = obj(evidence.get("verifier"), label + ".verifier")
    require(
        set(verifier) == {
            "call_count", "q_tensor_identity", "gpu_offsets_tensor_identity", "cpu_offsets_tensor_identity",
            "accepted", "reason", "issuer_call_count", "issuer_cpu_offsets_tensor_identity",
            "certified_offsets", "issuer_spy_restored", "verifier_spy_restored", "descriptor_object_id", "passed",
        },
        label + ".verifier: key set drift",
    )
    require(integer(verifier.get("call_count"), label + ".verifier.call_count") == 1, label + ".verifier: call count drift")
    require(integer(verifier.get("issuer_call_count"), label + ".verifier.issuer_call_count") == 1, label + ".verifier: issuer count drift")
    require(integer(verifier.get("descriptor_object_id"), label + ".verifier.descriptor_object_id") > 0, label + ".verifier: descriptor identity drift")
    for key in (
        "q_tensor_identity", "gpu_offsets_tensor_identity", "cpu_offsets_tensor_identity",
        "issuer_cpu_offsets_tensor_identity", "issuer_spy_restored", "verifier_spy_restored", "passed",
    ):
        require(boolean(verifier.get(key), label + ".verifier." + key), label + ".verifier: " + key + " failed")
    require(boolean(verifier.get("accepted"), label + ".verifier.accepted") is expected_accepted, label + ".verifier: acceptance drift")
    if expected_reason is None:
        require(verifier.get("reason") is None, label + ".verifier: unexpected verifier reason")
    else:
        require(string(verifier.get("reason"), label + ".verifier.reason") == expected_reason, label + ".verifier: reason drift")
    exact_offsets(verifier.get("certified_offsets"), expected_offsets, label + ".verifier.certified_offsets")


def validate_pinned_reference_load(value: object, ledger: Mapping[str, Any], label: str) -> None:
    helper = obj(ledger.get("pinned_reference_helper"), label + ".ledger_helper")
    expected = {
        "path": string(helper.get("path"), label + ".ledger_helper.path"),
        "sha256": sha256(helper.get("sha256"), label + ".ledger_helper.sha"),
        "load_contract": PINNED_LOAD_CONTRACT,
        "intercepted_names": ["sigmoid_ext"],
        "no_build": True,
    }
    strict_equal(value, expected, label)


def require_output_contract(value: object, label: str) -> None:
    expected = {"shape": [1, 12288, 12, 128], "dtype": "torch.bfloat16", "contiguous": True}
    strict_equal(value, expected, label)


def require_final_contract(value: object, label: str) -> None:
    expected = {"present": True, "dtype": "torch.float32", "shape": [6, 12, 128, 128], "contiguous": True}
    strict_equal(value, expected, label)


def require_exact_pair(value: object, label: str) -> None:
    evidence = obj(value, label)
    require(boolean(evidence.get("output_exact"), label + ".output_exact"), label + ": output not exact")
    require(boolean(evidence.get("final_exact"), label + ".final_exact"), label + ": final FP32 state not exact")
    require_output_contract(evidence.get("actual_output"), label + ".actual_output")
    require_output_contract(evidence.get("expected_output"), label + ".expected_output")
    require_final_contract(evidence.get("actual_final"), label + ".actual_final")
    require_final_contract(evidence.get("expected_final"), label + ".expected_final")


def canonical_map_entries(value: object, label: str) -> tuple[list[dict[str, object]], str]:
    require(type(value) is list, label + ": entries must be a list")
    typed: list[dict[str, object]] = []
    for index, item in enumerate(value):
        entry = obj(item, label + f"[{index}]")
        require(set(entry) == {"offsets", "contract", "variant"}, label + f"[{index}]: entry key-set drift")
        offsets = entry.get("offsets")
        require(type(offsets) is list and len(offsets) >= 2 and all(type(offset) is int for offset in offsets), label + f"[{index}]: offset type drift")
        require(offsets[0] == 0 and all(left < right for left, right in zip(offsets, offsets[1:], strict=False)), label + f"[{index}]: offsets not canonical strict-increasing")
        contract, variant = string(entry.get("contract"), label + f"[{index}].contract"), string(entry.get("variant"), label + f"[{index}].variant")
        typed.append({"offsets": list(offsets), "contract": contract, "variant": variant})
    canonical = sorted(typed, key=lambda item: (tuple(item["offsets"]), item["contract"]))
    strict_equal(value, canonical, label + ".canonical_order")
    payload = json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return canonical, _sha_bytes(payload)


def validate_production_map(value: object, label: str) -> str:
    evidence = obj(value, label)
    require(boolean(evidence.get("checked_before_cuda_initialization"), label + ".pre_cuda"), label + ": map gate did not precede CUDA")
    require(evidence.get("runner_mutates_production_map") is False, label + ": runner claims map mutation")
    integer(evidence.get("map_object_id"), label + ".object_id")
    pre_entries, pre_digest = canonical_map_entries(evidence.get("all_entries"), label + ".pre_entries")
    require(sha256(evidence.get("all_entries_sha256"), label + ".pre_sha") == pre_digest, label + ": pre canonical digest mismatch")
    post_entries, post_digest = canonical_map_entries(evidence.get("post_all_entries"), label + ".post_entries")
    require(sha256(evidence.get("post_all_entries_sha256"), label + ".post_sha") == post_digest, label + ": post canonical digest mismatch")
    strict_equal(pre_entries, post_entries, label + ".pre_post_entries")
    require(pre_digest == post_digest, label + ": pre/post digest drift")
    require(boolean(evidence.get("same_map_object_after_workload"), label + ".same_object"), label + ": map object changed")
    require(boolean(evidence.get("canonical_entries_unchanged"), label + ".canonical_entries_unchanged"), label + ": canonical entries changed")
    require(boolean(evidence.get("canonical_digest_unchanged"), label + ".canonical_digest_unchanged"), label + ": canonical digest changed")
    require(boolean(evidence.get("passed_after_workload"), label + ".passed_after_workload"), label + ": post map gate failed")
    require(boolean(evidence.get("passed"), label + ".passed"), label + ": pre map gate failed")
    expected_required = [
        {"offsets": list(TARGET_OFFSETS), "contract": "none", "variant": "vshard2_p2"},
        {"offsets": list(TARGET_OFFSETS), "contract": "fp32_final_only", "variant": "vshard2_p2"},
        {"offsets": list(TARGET_OFFSETS), "contract": "fp32_both", "variant": TARGET_VARIANT},
    ]
    strict_equal(evidence.get("required_target_entries"), expected_required, label + ".required_target_entries")
    found = {(tuple(item["offsets"]), item["contract"]): item["variant"] for item in pre_entries}
    for item in expected_required:
        require(found.get((tuple(item["offsets"]), item["contract"])) == item["variant"], label + ": required production entry drift")
    return pre_digest


def validate_patched_identity(value: object, label: str) -> None:
    patched = obj(value, label)
    root = env_path("PATCHED_ROOT")
    require(patched.get("root") == str(root) and patched.get("commit") == PATCHED_COMMIT and boolean(patched.get("passed"), label + ".passed"), label + ": patched root/commit drift")
    dirty = obj(patched.get("dirty_files"), label + ".dirty_files")
    require(set(dirty) == set(PATCHED_DIRTY_SHA256), label + ": dirty-file key set drift")
    for relative, digest in PATCHED_DIRTY_SHA256.items():
        expected_path = str((root / relative).resolve(strict=True))
        strict_equal(dirty.get(relative), {"path": expected_path, "expected_path": expected_path, "sha256": digest, "sha256_gate_pass": True}, label + ".dirty_files." + relative)
    wrapper_path = str((root / "flash_kda/__init__.py").resolve(strict=True))
    strict_equal(patched.get("production_wrapper"), {"path": wrapper_path, "expected_path": wrapper_path, "sha256": PRODUCTION_WRAPPER_SHA256, "sha256_gate_pass": True}, label + ".production_wrapper")


def validate_b300_and_fla(value: object, label: str) -> None:
    evidence = obj(value, label)
    device = obj(evidence.get("device"), label + ".device")
    require("B300" in string(device.get("name"), label + ".device.name").upper() and device.get("capability") == [10, 3] and integer(device.get("multiprocessor_count"), label + ".device.sm") == 148 and boolean(device.get("passed"), label + ".device.passed"), label + ": B300 SM103/148 drift")
    extension = obj(evidence.get("extension"), label + ".extension")
    require(extension.get("sha256") == AUDITED_EXTENSION_SHA256 and boolean(extension.get("passed"), label + ".extension.passed"), label + ": audited extension drift")
    patched_root, reference_root, fla_root = env_path("PATCHED_ROOT"), env_path("REFERENCE_ROOT"), env_path("FLA_ROOT")
    strict_equal(evidence.get("flash_kda_python"), {"path": str((patched_root / "flash_kda/__init__.py").resolve(strict=True)), "sha256": PRODUCTION_WRAPPER_SHA256}, label + ".flash_kda_python")
    trees = obj(evidence.get("source_trees"), label + ".source_trees")
    strict_equal(trees.get("patched"), {"root": str(patched_root), "commit": PATCHED_COMMIT, "passed": True}, label + ".source_trees.patched")
    strict_equal(trees.get("reference"), {"root": str(reference_root), "commit": PATCHED_COMMIT, "tracked_status_clean": True, "passed": True}, label + ".source_trees.reference")
    fla = obj(evidence.get("fla"), label + ".fla")
    require(fla.get("root") == str(fla_root) and fla.get("commit") == FLA_COMMIT and boolean(fla.get("tracked_status_clean"), label + ".fla.clean") and boolean(fla.get("passed"), label + ".fla.passed"), label + ": FLA root/commit/clean drift")
    strict_equal(fla.get("files"), FLA_FILE_SHA256, label + ".fla.files")
    expected_modules = {
        "fla": str((fla_root / "fla/__init__.py").resolve(strict=True)),
        "fla.ops.backends": str((fla_root / "fla/ops/backends/__init__.py").resolve(strict=True)),
        "fla.ops.kda": str((fla_root / "fla/ops/kda/__init__.py").resolve(strict=True)),
        "fla.ops.kda.backends": str((fla_root / "fla/ops/kda/backends/__init__.py").resolve(strict=True)),
        "fla.ops.kda.backends.flash_kda": str((fla_root / "fla/ops/kda/backends/flash_kda.py").resolve(strict=True)),
        "fla.ops.kda.chunk": str((fla_root / "fla/ops/kda/chunk.py").resolve(strict=True)),
    }
    strict_equal(fla.get("loaded_modules"), expected_modules, label + ".fla.loaded_modules")
    expected_public = {"fla.ops.kda.chunk_kda": {"implementation_identity_match": True, "module": "fla.ops.kda.chunk", "qualname": "chunk_kda", "source_path": expected_modules["fla.ops.kda.chunk"], "passed": True}}
    strict_equal(fla.get("public_callables"), expected_public, label + ".fla.public_callables")


def validate_registry(value: object, label: str) -> None:
    registry = obj(value, label)
    require(registry.get("public_callable") == "fla.ops.kda.chunk_kda", label + ": registry callable drift")
    require(registry.get("instrumentation") == "instance-local C1/pinned backend counters only; dispatcher unmodified", label + ": registry instrumentation drift")
    snapshot = registry.get("snapshot")
    require(type(snapshot) is list and snapshot, label + ": registry snapshot missing")
    c1_count = pinned_count = 0
    for index, item in enumerate(snapshot):
        backend = obj(item, label + f".snapshot[{index}]")
        require(set(backend) == {"backend_type", "priority", "id"}, label + f".snapshot[{index}]: key set drift")
        backend_type = string(backend.get("backend_type"), label + f".snapshot[{index}].type")
        integer(backend.get("priority"), label + f".snapshot[{index}].priority")
        integer(backend.get("id"), label + f".snapshot[{index}].id")
        c1_count += backend_type == "c1_b300_flash_kda"
        pinned_count += backend_type == "flash_kda"
    require(c1_count == 1 and pinned_count == 1, label + ": registry must contain exactly one C1 and one pinned backend")


def validate_raw(raw: Mapping[str, Any], *, expected_allocation: str | None, expected_process: int | None, current: Mapping[str, Any]) -> dict[str, object]:
    require(integer(raw.get("schema_version"), "raw.schema_version") == SCHEMA, "raw schema mismatch")
    require(string(raw.get("purpose"), "raw.purpose") == "production packed-varlen skew FP32-both functional and registry-route freeze; no production policy mutation", "raw purpose drift")
    allocation = string(raw.get("allocation_id"), "raw.allocation_id")
    require(allocation in ("A1", "A2"), "raw allocation drift")
    if expected_allocation is not None:
        require(allocation == expected_allocation, "raw allocation expected binding drift")
    process = integer(raw.get("process_index"), "raw.process_index")
    require(process in (0, 1), "raw process index drift")
    if expected_process is not None:
        require(process == expected_process, "raw process expected binding drift")
    job = string(obj(raw.get("allocation"), "raw.allocation").get("slurm_job_id"), "raw.job")
    require(POSITIVE_JOB.fullmatch(job) is not None, "raw job id must be positive ASCII decimal")
    target = obj(raw.get("target"), "raw.target")
    exact_offsets(target.get("offsets"), TARGET_OFFSETS, "raw.target.offsets")
    require(target.get("contract") == "fp32_both" and target.get("variant") == TARGET_VARIANT and target.get("reason") == TARGET_REASON, "raw target contract/route drift")
    identity = obj(raw.get("identity"), "raw.identity")
    strict_equal(identity.get("runner"), current["runner"], "raw.runner identity")
    shell_identity = dict(current["shell"])
    strict_equal(identity.get("protocol_shell"), shell_identity, "raw.shell identity")
    strict_equal(identity.get("runtime_import_ledger"), current["ledger"], "raw runtime ledger")
    strict_equal(identity.get("production_source_hashes_external"), current["production_hashes"], "raw external production hashes")
    validate_patched_identity(identity.get("patched"), "raw.patched")
    validate_b300_and_fla(identity.get("b300_and_fla"), "raw.b300_and_fla")
    validate_pinned_reference_load(identity.get("pinned_reference_load"), obj(current.get("ledger"), "current.ledger"), "raw.pinned_reference_load")
    map_digest = validate_production_map(raw.get("production_map"), "raw.production_map")
    validate_registry(raw.get("registry"), "raw.registry")
    negative = obj(raw.get("prelaunch_negative_control"), "raw.prelaunch_negative_control")
    expected_negative_reason = TARGET_REASON + "; fwd_vshard4_p2_missing_prelaunch_fallback_to_baseline"
    require(negative.get("scope") == "synthetic loader inventory only; no map mutation and no auto_dispatch.fwd replacement" and negative.get("requested_variant") == TARGET_VARIANT and negative.get("available_symbols") == ["fwd_vshard_p2"] and negative.get("chosen_variant") == "baseline" and negative.get("reason") == expected_negative_reason and negative.get("extension_is_none") is True and negative.get("loader_restored") is True and boolean(negative.get("passed"), "raw.prelaunch_negative_control.passed"), "raw v4-symbol-missing negative control drift")
    correctness = obj(raw.get("correctness"), "raw.correctness")
    for key in ("pinned_vs_torch_ref", "direct_c1_vs_pinned", "public_vs_pinned", "public_pinned_vs_torch_ref", "direct_c1_vs_torch_ref", "public_vs_torch_ref"):
        require_exact_pair(correctness.get(key), "raw.correctness." + key)
    decision = obj(correctness.get("public_decision"), "raw.public_decision")
    require(decision.get("chosen_variant") == TARGET_VARIANT and decision.get("reason") == TARGET_REASON, "raw public production decision drift")
    exact_delta(obj(correctness.get("public_c1_spy"), "raw.public_c1_spy").get("delta"), {"c1": 1, "pinned": 0}, "raw public C1 spy")
    prepare = obj(correctness.get("public_handoff_prepare"), "raw.prepare")
    require(integer(obj(prepare.get("c1"), "raw.prepare.c1").get("prepare_delta"), "raw.prepare.c1.delta") == 1 and integer(obj(prepare.get("pinned"), "raw.prepare.pinned").get("prepare_delta"), "raw.prepare.pinned.delta") == 0, "raw C1/pinned prepare handoff drift")
    require(boolean(correctness.get("input_immutability_exact"), "raw.correctness.immutability"), "raw target input/initial immutability failed")
    descriptor = obj(raw.get("descriptor_handoff"), "raw.descriptor")
    require(descriptor.get("real_public_registry_call") == "fla.ops.kda.chunk_kda" and integer(descriptor.get("public_call_count"), "raw.descriptor.calls") == 1 and integer(descriptor.get("issue_descriptor_call_count"), "raw.descriptor.issue") == 1, "raw CPU descriptor public handoff drift")
    exact_delta(descriptor.get("route_spy_delta"), {"c1": 1, "pinned": 0}, "raw descriptor C1 spy")
    exact_offsets(descriptor.get("offsets"), TARGET_OFFSETS, "raw descriptor offsets")
    exact_offsets(descriptor.get("certified_offsets"), TARGET_OFFSETS, "raw descriptor certified")
    require(descriptor.get("chosen_variant") == TARGET_VARIANT and descriptor.get("reason") == TARGET_REASON and descriptor.get("descriptor_cpu_tensor_identity") is True and boolean(descriptor.get("input_immutability_exact"), "raw.descriptor.immutability") and boolean(descriptor.get("issue_descriptor_spy_restored"), "raw.descriptor.restore"), "raw descriptor evidence drift")
    require_output_contract(descriptor.get("output_contract"), "raw.descriptor.output_contract")
    require_final_contract(descriptor.get("final_state_contract"), "raw.descriptor.final_state_contract")
    controls = obj(raw.get("controls"), "raw.controls")
    state = obj(controls.get("adjacent_state_control"), "raw.controls.state")
    require(state.get("contract") == "fp32_final_only" and state.get("expected_variant") == "vshard2_p2", "raw adjacent state control drift")
    exact_delta(state.get("route_spy_delta"), {"c1": 1, "pinned": 0}, "raw adjacent state route")
    state_decision = obj(state.get("decision"), "raw.state decision")
    require(state_decision.get("chosen_variant") == "vshard2_p2" and state_decision.get("reason") == "varlen_skew_n6_h12_t12288_fp32_final_only_whitelist_hit", "raw adjacent state decision drift")
    validate_control_evidence({"isolation": state.get("handoff_and_cache_isolation"), "verifier": state.get("public_verifier")}, "raw.state fresh verifier", TARGET_OFFSETS, True, None)
    require_output_contract(state.get("output_contract"), "raw adjacent state output_contract")
    require_final_contract(state.get("final_state_contract"), "raw adjacent state final_state_contract")
    offset = obj(controls.get("adjacent_offset_fallback"), "raw.controls.offset")
    exact_offsets(offset.get("offsets"), FALLBACK_OFFSETS, "raw fallback offsets")
    exact_delta(offset.get("route_spy_delta"), {"c1": 0, "pinned": 1}, "raw offset fallback route")
    validate_control_evidence({"isolation": offset.get("handoff_and_cache_isolation"), "verifier": offset.get("public_verifier")}, "raw offset fresh verifier", FALLBACK_OFFSETS, False, "C1 packed-varlen preflight rejected: " + FALLBACK_REASON)
    require_exact_pair(offset.get("public_vs_pinned"), "raw offset public/pinned")
    gates = obj(raw.get("gates"), "raw.gates")
    for key in ("pre_cuda_source_and_map", "v4_symbol_missing_prelaunch_baseline", "target_public_route", "exact_output_and_fp32_final", "input_and_initial_immutability", "cpu_descriptor_handoff", "adjacent_state_and_offset_controls", "backend_spies_restored", "prepare_descriptor_binding_restored"):
        require(boolean(obj(gates.get(key), "raw.gates." + key).get("passed"), "raw.gates." + key + ".passed"), "raw gate failed: " + key)
    clean = obj(gates.get("python_pre_torch_clean"), "raw.clean")
    require(integer(clean.get("memory_used_mib"), "raw.clean.memory") == 0 and type(clean.get("uuid")) is str and bool(clean.get("uuid")) and clean.get("compute_apps") == [] and boolean(clean.get("passed"), "raw.clean.passed"), "raw single-GPU UUID/clean gate drift")
    require(boolean(raw.get("complete"), "raw.complete"), "raw incomplete")
    process_evidence = obj(raw.get("process"), "raw.process")
    pid = integer(process_evidence.get("pid"), "raw.pid")
    require(pid > 0 and boolean(process_evidence.get("fresh_python_process_required"), "raw.fresh_python_process_required"), "raw fresh-PID evidence drift")
    return {"allocation_id": allocation, "process_index": process, "slurm_job_id": job, "pid": pid, "gpu_uuid": string(clean.get("uuid"), "raw.uuid"), "production_map_sha256": map_digest, "passed": True}


def build_allocation(*, allocation_id: str, paths: list[Path], expected_shas: list[str], current: Mapping[str, Any], a1_path: Path | None = None, a1_sha: str | None = None, a1_loaded: tuple[Mapping[str, Any], Mapping[str, str]] | None = None, current_job: str | None = None) -> dict[str, object]:
    require(allocation_id in ("A1", "A2") and len(paths) == len(expected_shas) == 2, "allocation input cardinality drift")
    raw_records = []
    pids, jobs, uuids, map_digests = set(), set(), set(), set()
    for process_index, (path, digest) in enumerate(zip(paths, expected_shas, strict=True)):
        raw, link = read_json_once(path, digest, f"raw[{process_index}]")
        proof = validate_raw(raw, expected_allocation=allocation_id, expected_process=process_index, current=current)
        raw_records.append({**link, "proof": proof})
        pids.add(proof["pid"]); jobs.add(proof["slurm_job_id"]); uuids.add(proof["gpu_uuid"]); map_digests.add(proof["production_map_sha256"])
    require(len(pids) == 2 and len(jobs) == 1 and len(uuids) == 1 and len(map_digests) == 1, "allocation requires two fresh PIDs, one Slurm job, one GPU UUID, and one canonical production-map digest")
    job = next(iter(jobs))
    if current_job is not None:
        require(job == current_job, "allocation current Slurm job binding drift")
    result: dict[str, object] = {"schema_version": SCHEMA, "purpose": "production skew FP32-both allocation chain evidence", "allocation_id": allocation_id, "slurm_job_id": job, "gpu_uuid": next(iter(uuids)), "production_map_sha256": next(iter(map_digests)), "raw_runner_artifacts": raw_records, "current_identity": current, "allocation_valid": True, "complete": True}
    if allocation_id == "A2":
        if a1_loaded is None:
            require(a1_path is not None and a1_sha is not None, "A2 requires A1 manifest binding")
            a1, a1_link = read_json_once(a1_path, a1_sha, "A2 A1 manifest")
        else:
            a1, a1_link = a1_loaded
        require(a1.get("allocation_id") == "A1" and boolean(a1.get("allocation_valid"), "A2 A1 allocation_valid") and string(a1.get("slurm_job_id"), "A2 A1 job") != job, "A2 A1 binding must be valid and a different ASCII Slurm job")
        result["a1_binding"] = {**a1_link, "slurm_job_id": a1["slurm_job_id"], "passed": True}
    return result


def verify_a2_binding(manifest: Mapping[str, Any], expected_sha: str, current_job: str) -> None:
    require(POSITIVE_JOB.fullmatch(current_job) is not None, "current job must be ASCII positive decimal")
    require(manifest.get("allocation_id") == "A1" and boolean(manifest.get("allocation_valid"), "A1 allocation_valid"), "A1 manifest invalid")
    require(string(manifest.get("slurm_job_id"), "A1 job") != current_job, "A2 must use a distinct Slurm job")
    sha256(expected_sha, "A1 manifest SHA")


def manifest_raw_inputs(manifest: Mapping[str, Any], label: str) -> tuple[list[Path], list[str]]:
    records = manifest.get("raw_runner_artifacts")
    require(type(records) is list and len(records) == 2, label + ": requires exactly two raw records")
    paths: list[Path] = []
    digests: list[str] = []
    for index, record in enumerate(records):
        item = obj(record, label + f".raw[{index}]")
        path = Path(string(item.get("path"), label + f".raw[{index}].path"))
        paths.append(path)
        digests.append(sha256(item.get("sha256"), label + f".raw[{index}].sha"))
    return paths, digests


def recompute_manifest(manifest: Mapping[str, Any], link: Mapping[str, str], current: Mapping[str, Any], label: str, *, a1_loaded: tuple[Mapping[str, Any], Mapping[str, str]] | None = None) -> dict[str, object]:
    allocation_id = string(manifest.get("allocation_id"), label + ".allocation_id")
    paths, digests = manifest_raw_inputs(manifest, label)
    if allocation_id == "A1":
        computed = build_allocation(allocation_id="A1", paths=paths, expected_shas=digests, current=current)
    else:
        binding = obj(manifest.get("a1_binding"), label + ".a1_binding")
        computed = build_allocation(
            allocation_id="A2",
            paths=paths,
            expected_shas=digests,
            current=current,
            a1_path=Path(string(binding.get("path"), label + ".a1_binding.path")),
            a1_sha=sha256(binding.get("sha256"), label + ".a1_binding.sha"),
            a1_loaded=a1_loaded,
        )
    strict_equal(manifest, computed, label + ".stored_vs_recomputed")
    require(link["sha256"] == sha256(link["sha256"], label + ".link.sha"), label + ": link SHA type drift")
    return computed


def command_allocation(args: argparse.Namespace) -> None:
    current = current_identity()
    current_job = os.environ.get("SLURM_JOB_ID", "")
    require(POSITIVE_JOB.fullmatch(current_job) is not None, "SLURM_JOB_ID must be positive ASCII decimal")
    result = build_allocation(allocation_id=args.allocation_id, paths=args.runner_json, expected_shas=args.expected_runner_sha256s, current=current, a1_path=args.a1_allocation_manifest, a1_sha=args.expected_a1_allocation_manifest_sha256, current_job=current_job)
    write_json(args.json, result)
    if args.require_pass:
        require(boolean(result.get("allocation_valid"), "allocation valid"), "allocation invalid")
    print(f"wrote {args.json}")


def command_verify(args: argparse.Namespace) -> None:
    current = current_identity()
    manifest, link = read_json_once(args.allocation_manifest, args.expected_allocation_sha256, "allocation manifest")
    require(manifest.get("current_identity") == current, "A1 manifest current source identity drift")
    recompute_manifest(manifest, link, current, "A2 prereq A1")
    verify_a2_binding(manifest, link["sha256"], args.current_slurm_job_id)
    print("A2_A1_BINDING_PASS")


def command_freeze(args: argparse.Namespace) -> None:
    current = current_identity()
    a1, a1_link = read_json_once(args.allocation_a, args.expected_allocation_a_sha256, "freeze A1")
    a2, a2_link = read_json_once(args.allocation_b, args.expected_allocation_b_sha256, "freeze A2")
    for label, value, expected in (("A1", a1, "A1"), ("A2", a2, "A2")):
        require(value.get("allocation_id") == expected and boolean(value.get("allocation_valid"), label + ".allocation_valid") and value.get("current_identity") == current, label + " allocation identity/validity drift")
    require(string(a1.get("slurm_job_id"), "freeze A1 job") != string(a2.get("slurm_job_id"), "freeze A2 job"), "freeze requires independent A1/A2 Slurm jobs")
    require(sha256(a1.get("production_map_sha256"), "freeze A1 map digest") == sha256(a2.get("production_map_sha256"), "freeze A2 map digest"), "freeze requires the exact same canonical production-map digest across A1/A2")
    recompute_manifest(a1, a1_link, current, "freeze A1")
    recompute_manifest(a2, a2_link, current, "freeze A2", a1_loaded=(a1, a1_link))
    binding = obj(a2.get("a1_binding"), "freeze A2 binding")
    strict_equal(binding, {**a1_link, "slurm_job_id": a1["slurm_job_id"], "passed": True}, "freeze A2 exact A1 binding")
    result = {"schema_version": SCHEMA, "purpose": "production skew FP32-both functional/route freeze decision", "allocation_a": a1_link, "allocation_b": a2_link, "current_identity": current, "eligible_for_production_freeze": True, "production_action": "already-integrated route confirmed; no mutation by freeze", "complete": True}
    write_json(args.json, result)
    if args.require_eligible:
        require(boolean(result.get("eligible_for_production_freeze"), "freeze eligible"), "freeze ineligible")
    print(f"wrote {args.json}")


def self_test() -> None:
    try:
        strict_equal(True, 1, "bool/int")
    except AuditError:
        pass
    else:
        raise AssertionError("strict JSON equality accepted bool/int forgery")
    control_evidence = {
        "isolation": {
            "clear_handoff_api": "C1B300FlashKDABackend._clear_varlen_handoff",
            "metadata_clear_api": "varlen_metadata.clear_cache",
            "cache_before_clear": {"entries": 1, "hits": 0},
            "cache_after_clear": {"entries": 0, "hits": 0},
            "handoff_empty_after_clear": True,
            "handoff_empty_after_public": True,
            "cache_after_cleanup": {"entries": 0, "hits": 0},
            "passed": True,
        },
        "verifier": {
            "call_count": 1,
            "q_tensor_identity": True,
            "gpu_offsets_tensor_identity": True,
            "cpu_offsets_tensor_identity": True,
            "accepted": False,
            "reason": "C1 packed-varlen preflight rejected: " + FALLBACK_REASON,
            "issuer_call_count": 1,
            "issuer_cpu_offsets_tensor_identity": True,
            "certified_offsets": list(FALLBACK_OFFSETS),
            "issuer_spy_restored": True,
            "verifier_spy_restored": True,
            "descriptor_object_id": 1,
            "passed": True,
        },
    }
    validate_control_evidence(control_evidence, "self control", FALLBACK_OFFSETS, False, "C1 packed-varlen preflight rejected: " + FALLBACK_REASON)
    forged_control = json.loads(json.dumps(control_evidence))
    forged_control["isolation"]["cache_after_clear"]["entries"] = True
    try:
        validate_control_evidence(forged_control, "self forged control", FALLBACK_OFFSETS, False, "C1 packed-varlen preflight rejected: " + FALLBACK_REASON)
    except AuditError:
        pass
    else:
        raise AssertionError("control evidence validator accepted bool cache statistic")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "one.json"
        payload = b'{"number":1}'
        path.write_bytes(payload)
        parsed, _ = read_json_once(path, _sha_bytes(payload), "self")
        require(parsed == {"number": 1}, "single-read parse self-test")
        try:
            read_json_once(path, "0" * 64, "forged")
        except AuditError:
            pass
        else:
            raise AssertionError("TOCTOU hash gate self-test accepted forged SHA")
    try:
        verify_a2_binding({"allocation_id": "A1", "allocation_valid": True, "slurm_job_id": "77"}, "0" * 64, "77")
    except AuditError:
        pass
    else:
        raise AssertionError("A2 binding self-test accepted same-job forgery")
    helper_ledger = {"pinned_reference_helper": {"path": "/pinned/sigmoid_ext.so", "sha256": STATIC_LEDGER["pinned_reference_helper"]}}
    good_load = {"path": "/pinned/sigmoid_ext.so", "sha256": STATIC_LEDGER["pinned_reference_helper"], "load_contract": PINNED_LOAD_CONTRACT, "intercepted_names": ["sigmoid_ext"], "no_build": True}
    validate_pinned_reference_load(good_load, helper_ledger, "self helper")
    forged_load = dict(good_load)
    forged_load["no_build"] = False
    try:
        validate_pinned_reference_load(forged_load, helper_ledger, "self forged helper")
    except AuditError:
        pass
    else:
        raise AssertionError("pinned helper self-test accepted no_build forgery")
    try:
        canonical_map_entries([{"offsets": [0, True], "contract": "none", "variant": "vshard2_p2"}], "self forged map")
    except AuditError:
        pass
    else:
        raise AssertionError("production-map self-test accepted bool offset forgery")
    entries = [
        {"offsets": list(TARGET_OFFSETS), "contract": "fp32_both", "variant": TARGET_VARIANT},
        {"offsets": list(TARGET_OFFSETS), "contract": "fp32_final_only", "variant": "vshard2_p2"},
        {"offsets": list(TARGET_OFFSETS), "contract": "none", "variant": "vshard2_p2"},
    ]
    _, digest = canonical_map_entries(entries, "self map")
    map_evidence = {
        "checked_before_cuda_initialization": True,
        "map_object_id": 7,
        "all_entries": entries,
        "all_entries_sha256": digest,
        "required_target_entries": [
            {"offsets": list(TARGET_OFFSETS), "contract": "none", "variant": "vshard2_p2"},
            {"offsets": list(TARGET_OFFSETS), "contract": "fp32_final_only", "variant": "vshard2_p2"},
            {"offsets": list(TARGET_OFFSETS), "contract": "fp32_both", "variant": TARGET_VARIANT},
        ],
        "runner_mutates_production_map": False,
        "passed": True,
        "post_all_entries": entries,
        "post_all_entries_sha256": digest,
        "same_map_object_after_workload": True,
        "canonical_entries_unchanged": True,
        "canonical_digest_unchanged": True,
        "passed_after_workload": True,
    }
    require(validate_production_map(map_evidence, "self map evidence") == digest, "map post invariance self-test")
    forged_evidence = dict(map_evidence)
    forged_evidence["post_all_entries_sha256"] = "0" * 64
    try:
        validate_production_map(forged_evidence, "self forged map evidence")
    except AuditError:
        pass
    else:
        raise AssertionError("production-map self-test accepted post-digest forgery")
    print("ANALYZER_SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    allocation = sub.add_parser("allocation")
    allocation.add_argument("--allocation-id", choices=("A1", "A2"), required=True)
    allocation.add_argument("--runner-json", type=Path, nargs=2, required=True)
    allocation.add_argument("--expected-runner-sha256s", nargs=2, required=True)
    allocation.add_argument("--a1-allocation-manifest", type=Path)
    allocation.add_argument("--expected-a1-allocation-manifest-sha256")
    allocation.add_argument("--json", type=Path, required=True)
    allocation.add_argument("--require-pass", action="store_true")
    verify = sub.add_parser("verify-allocation")
    verify.add_argument("--allocation-manifest", type=Path, required=True)
    verify.add_argument("--expected-allocation-sha256", required=True)
    verify.add_argument("--current-slurm-job-id", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--allocation-a", type=Path, required=True)
    freeze.add_argument("--expected-allocation-a-sha256", required=True)
    freeze.add_argument("--allocation-b", type=Path, required=True)
    freeze.add_argument("--expected-allocation-b-sha256", required=True)
    freeze.add_argument("--json", type=Path, required=True)
    freeze.add_argument("--require-eligible", action="store_true")
    sub.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "allocation":
        command_allocation(args)
    elif args.command == "verify-allocation":
        command_verify(args)
    elif args.command == "freeze":
        command_freeze(args)
    else:
        self_test()


if __name__ == "__main__":
    main()
