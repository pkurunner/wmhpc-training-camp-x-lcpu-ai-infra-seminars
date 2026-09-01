#!/usr/bin/env python3
"""Fresh-PID production-route freeze for the released skew FP32-both cell.

This program only observes the imported production dispatcher.  In particular,
it never clears, updates, replaces, or otherwise mutates the varlen policy
table, and it never replaces ``auto_dispatch.fwd``.  The only instrumentation
is a pair of instance-local FLA-registry backend counters, restored before the
process exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_seqcount_dispatch import run_seqcount_dispatch as shared
from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch, varlen_metadata
from assignment02.team.c1_flashkda.challenge_varlen_dispatch import run_varlen_dispatch_confirmation as confirmation
from assignment02.team.c1_flashkda.challenge_varlen_dispatch import run_varlen_fla_handoff_candidate as candidate


SCHEMA_VERSION = 1
ALLOCATION_IDS = ("A1", "A2")
TARGET_OFFSETS = (0, 1, 2, 3, 4, 5, 12288)
TARGET_CASE_NAME = "skew_n6_h12_t12288"
TARGET_CONTRACT = "fp32_both"
TARGET_VARIANT = "vshard4_p2"
TARGET_REASON = "varlen_skew_n6_h12_t12288_fp32_both_whitelist_hit"
EXPECTED_TARGET_ENTRIES = {
    (TARGET_OFFSETS, "none"): "vshard2_p2",
    (TARGET_OFFSETS, "fp32_final_only"): "vshard2_p2",
    (TARGET_OFFSETS, "fp32_both"): TARGET_VARIANT,
}
FALLBACK_OFFSETS = (0, 1, 2, 3, 4, 6, 12288)
FALLBACK_REASON = "varlen_offsets_not_whitelisted"
RUNNER_SHA_ENV = "C1_SKEW_PRODUCTION_FREEZE_RUNNER_SHA256"
ANALYZER_SHA_ENV = "C1_SKEW_PRODUCTION_FREEZE_ANALYZER_SHA256"
SHELL_SHA_ENV = "EXPECTED_PROTOCOL_SHELL_SHA256"
SHELL_PATH_ENV = "C1_SKEW_PRODUCTION_FREEZE_SHELL_PATH"
AUTO_SHA_ENV = "C1_SKEW_PRODUCTION_AUTO_DISPATCH_SHA256"
FLA_BACKEND_SHA_ENV = "C1_SKEW_PRODUCTION_FLA_BACKEND_SHA256"
CLEAN_ENV = "C1_SKEW_PRODUCTION_FREEZE_CLEAN_GPU"
SLURM_JOB_ID = re.compile(r"[1-9][0-9]*\Z")
PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
PRODUCTION_WRAPPER_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
PATCHED_EXACT_DIRTY_FILES = {
    "csrc/flash_kda.cpp": "38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4",
    "csrc/fwd.h": "613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083",
    "csrc/smxx/fwd_launch.cu": "a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928",
}
CANDIDATE_HELPER_SHA256 = "e07481e7e66551f1f9eaafa7f1fb9db9ef193f1694ff831ee0a07802c8ffbe14"
STATIC_LEDGER_SHA256 = {
    "varlen_metadata": "f89a97ba284c7a24a3df54efca7bb60eb70f9cbda659200bd6cb3e8dfdaf4ccd",
    "confirmation_runner": "9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b",
    "shared_seqcount_runner": "4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f",
    "candidate_helper": CANDIDATE_HELPER_SHA256,
    "prefetch2": "752126488487ac317a7ee167b660b0895562e4877aedbc4ce2a599c1f59a10d0",
    "vshard4_prefetch2": "445c90815919fb8c1db3bc79289d10029ea77b1d9accb393c326c120ba1f8385",
    "harness": "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52",
    "pinned_torch_ref": "bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5",
    "pinned_reference_helper": "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Mapping[str, object]) -> None:
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


def _lower_sha_from_env(name: str) -> str:
    value = os.environ.get(name, "")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"{name} must be a lowercase SHA-256")
    return value


def _file_identity(path: Path, expected_path: Path, expected_sha: str, label: str) -> dict[str, object]:
    actual = path.resolve(strict=True)
    expected = expected_path.resolve(strict=True)
    if actual != expected or _sha(actual) != expected_sha:
        raise RuntimeError(f"{label} path/SHA identity gate failed")
    return {"path": str(actual), "expected_path": str(expected), "sha256": expected_sha, "sha256_gate_pass": True}


def _runner_identity() -> dict[str, object]:
    path = Path(__file__).resolve(strict=True)
    return _file_identity(
        path,
        REPO_ROOT / "assignment02/team/c1_flashkda/challenge_varlen_fp32_both_production_freeze/run_varlen_fp32_both_production_freeze.py",
        _lower_sha_from_env(RUNNER_SHA_ENV),
        "runner",
    )


def _shell_identity() -> dict[str, object]:
    path_text = os.environ.get(SHELL_PATH_ENV, "")
    if not path_text:
        raise RuntimeError(f"{SHELL_PATH_ENV} is required")
    return _file_identity(
        Path(path_text),
        REPO_ROOT / "assignment02/team/c1_flashkda/challenge_varlen_fp32_both_production_freeze/run_clean_varlen_fp32_both_production_freeze.sh",
        _lower_sha_from_env(SHELL_SHA_ENV),
        "protocol shell",
    )


def _patched_identity(patched_root: Path) -> dict[str, object]:
    root = patched_root.resolve(strict=True)
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"], check=True, capture_output=True, text=True).stdout.splitlines()
    expected_status = {f" M {relative}" for relative in PATCHED_EXACT_DIRTY_FILES}
    if commit != PATCHED_COMMIT or set(status) != expected_status:
        raise RuntimeError("patched tree commit/dirty-set drift")
    files = {
        relative: _file_identity(root / relative, root / relative, digest, f"patched {relative}")
        for relative, digest in PATCHED_EXACT_DIRTY_FILES.items()
    }
    wrapper = _file_identity(root / "flash_kda/__init__.py", root / "flash_kda/__init__.py", PRODUCTION_WRAPPER_SHA256, "production wrapper")
    return {"root": str(root), "commit": commit, "dirty_files": files, "production_wrapper": wrapper, "passed": True}


def _strict_data_equal(left: object, right: object, label: str) -> None:
    """Recursive comparison that cannot coerce a bool to an integer."""

    if type(left) is dict or type(right) is dict:
        if type(left) is not dict or type(right) is not dict or set(left) != set(right):
            raise AssertionError(label + ": mapping type/key drift")
        for key in left:
            _strict_data_equal(left[key], right[key], label + "." + str(key))
        return
    if type(left) is list or type(right) is list:
        if type(left) is not list or type(right) is not list or len(left) != len(right):
            raise AssertionError(label + ": list type/length drift")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _strict_data_equal(left_item, right_item, label + f"[{index}]")
        return
    if type(left) is not type(right) or left != right:
        raise AssertionError(label + ": scalar type/value drift")


def _canonical_production_map(raw: object, label: str) -> tuple[dict[tuple[tuple[int, ...], str], str], list[dict[str, object]], str]:
    """Type-check and canonically serialize the live production map."""

    if type(raw) is not dict:
        raise RuntimeError(label + ": production varlen map must be a built-in dict")
    normalized: dict[tuple[tuple[int, ...], str], str] = {}
    for key, value in raw.items():
        if type(key) is not tuple or len(key) != 2 or type(value) is not str:
            raise RuntimeError(label + ": production varlen map entry type drift")
        offsets, contract = key
        if type(offsets) is not tuple or not offsets or any(type(item) is not int for item in offsets) or type(contract) is not str:
            raise RuntimeError(label + ": production varlen map key type drift")
        normalized[(offsets, contract)] = value
    serialized: list[dict[str, object]] = [
        {"offsets": list(offsets), "contract": contract, "variant": variant}
        for (offsets, contract), variant in sorted(normalized.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    encoded = json.dumps(serialized, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return normalized, serialized, hashlib.sha256(encoded).hexdigest()


def _production_map_gate() -> dict[str, object]:
    """Snapshot the real map before CUDA initialization without mutating it."""

    raw = getattr(auto_dispatch, "_VARLEN_PUBLIC_VARIANTS", None)
    normalized, serialized, digest = _canonical_production_map(raw, "pre-CUDA")
    target_entries = {key: normalized.get(key) for key in EXPECTED_TARGET_ENTRIES}
    if target_entries != EXPECTED_TARGET_ENTRIES:
        raise RuntimeError(f"released target production-map entries drift: {target_entries!r}")
    return {
        "checked_before_cuda_initialization": True,
        "map_object_id": id(raw),
        "all_entries": serialized,
        "all_entries_sha256": digest,
        "required_target_entries": [
            {"offsets": list(offsets), "contract": contract, "variant": variant}
            for (offsets, contract), variant in EXPECTED_TARGET_ENTRIES.items()
        ],
        "runner_mutates_production_map": False,
        "passed": True,
    }


def _production_map_post_gate(pre: Mapping[str, object]) -> dict[str, object]:
    """Require the post-workload content as well as object identity to match."""

    raw = getattr(auto_dispatch, "_VARLEN_PUBLIC_VARIANTS", None)
    _normalized, post_entries, post_digest = _canonical_production_map(raw, "post-workload")
    pre_entries = pre.get("all_entries")
    pre_digest = pre.get("all_entries_sha256")
    _strict_data_equal(post_entries, pre_entries, "production map canonical entries pre/post")
    if type(pre_digest) is not str or post_digest != pre_digest:
        raise AssertionError("production map canonical digest changed")
    same_object = id(raw) == pre.get("map_object_id")
    if same_object is not True:
        raise AssertionError("production map object identity changed")
    return {
        "post_all_entries": post_entries,
        "post_all_entries_sha256": post_digest,
        "same_map_object_after_workload": True,
        "canonical_entries_unchanged": True,
        "canonical_digest_unchanged": True,
        "passed_after_workload": True,
        "passed": True,
    }


def _pre_cuda_runtime_ledger(args: argparse.Namespace) -> dict[str, object]:
    owned = REPO_ROOT / "assignment02/team/c1_flashkda"
    helper_text = os.environ.get(confirmation.REFERENCE_HELPER_PATH_ENV, "")
    helper_sha = os.environ.get(confirmation.REFERENCE_HELPER_SHA_ENV, "")
    if not helper_text or helper_sha != STATIC_LEDGER_SHA256["pinned_reference_helper"]:
        raise RuntimeError("pinned reference helper environment identity drift")
    expected = {
        "auto_dispatch": (Path(auto_dispatch.__file__), owned / "challenge_tp8_dispatch/auto_dispatch.py", _lower_sha_from_env(AUTO_SHA_ENV)),
        "fla_backend": (owned / "challenge_tp8_dispatch/fla_backend.py", owned / "challenge_tp8_dispatch/fla_backend.py", _lower_sha_from_env(FLA_BACKEND_SHA_ENV)),
        "varlen_metadata": (Path(varlen_metadata.__file__), owned / "challenge_tp8_dispatch/varlen_metadata.py", STATIC_LEDGER_SHA256["varlen_metadata"]),
        "confirmation_runner": (Path(confirmation.__file__), owned / "challenge_varlen_dispatch/run_varlen_dispatch_confirmation.py", STATIC_LEDGER_SHA256["confirmation_runner"]),
        "shared_seqcount_runner": (Path(shared.__file__), owned / "challenge_seqcount_dispatch/run_seqcount_dispatch.py", STATIC_LEDGER_SHA256["shared_seqcount_runner"]),
        "candidate_helper": (Path(candidate.__file__), owned / "challenge_varlen_dispatch/run_varlen_fla_handoff_candidate.py", STATIC_LEDGER_SHA256["candidate_helper"]),
        "prefetch2": (owned / "challenge_prefetch2/prefetch2.py", owned / "challenge_prefetch2/prefetch2.py", STATIC_LEDGER_SHA256["prefetch2"]),
        "vshard4_prefetch2": (owned / "challenge_vshard4_prefetch2/vshard4_prefetch2.py", owned / "challenge_vshard4_prefetch2/vshard4_prefetch2.py", STATIC_LEDGER_SHA256["vshard4_prefetch2"]),
        "harness": (owned / "harness/validate_and_bench.py", owned / "harness/validate_and_bench.py", STATIC_LEDGER_SHA256["harness"]),
        "pinned_torch_ref": (args.reference_root / "tests/torch_ref.py", args.reference_root / "tests/torch_ref.py", STATIC_LEDGER_SHA256["pinned_torch_ref"]),
        "pinned_reference_helper": (Path(helper_text), Path(helper_text), STATIC_LEDGER_SHA256["pinned_reference_helper"]),
    }
    if set(expected) != {"auto_dispatch", "fla_backend", *STATIC_LEDGER_SHA256}:
        raise AssertionError("runtime ledger key drift")
    return {name: _file_identity(actual, wanted, digest, name) for name, (actual, wanted, digest) in expected.items()}


def _python_clean_gpu_gate() -> dict[str, object]:
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    if query.returncode != 0:
        raise RuntimeError("pre-Torch nvidia-smi query failed")
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("pre-Torch gate requires exactly one visible GPU")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 5 or not fields[1] or int(fields[-1]) != 0:
        raise RuntimeError(f"pre-Torch GPU identity/cleanliness drift: {rows!r}")
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    app_rows = [line.strip() for line in apps.stdout.splitlines() if line.strip()]
    if apps.returncode != 0 or app_rows:
        raise RuntimeError("pre-Torch clean GPU requires no compute apps")
    return {"index": fields[0], "uuid": fields[1], "name": fields[2], "compute_capability": fields[3], "memory_used_mib": 0, "compute_apps": [], "passed": True}


def _exact_offsets(value: object, label: str) -> list[int]:
    expected = list(TARGET_OFFSETS)
    if type(value) is not list or value != expected or any(type(item) is not int for item in value):
        raise AssertionError(label + ": canonical offsets/type drift")
    return expected


def _target_case() -> object:
    matches = [case for case in confirmation.CASES if case.name == TARGET_CASE_NAME]
    if len(matches) != 1:
        raise RuntimeError("target case missing/ambiguous")
    case = matches[0]
    if tuple(case.lengths) != (1, 1, 1, 1, 1, 12283) or case.total_tokens != 12288 or case.batch != 1 or case.heads != 12:
        raise RuntimeError("target case structural drift")
    return case


def _target_cell() -> object:
    return candidate.Cell(_target_case(), TARGET_CONTRACT, TARGET_VARIANT)


def _missing_v4_prelaunch_negative_control() -> dict[str, object]:
    """Exercise the real prelaunch selector with a v2-only symbol inventory.

    This changes neither the production map nor ``auto_dispatch.fwd``.  The
    loader seam is restored in a ``finally`` block before any public workload;
    the synthetic inventory is used solely to prove that an evidence-scoped v4
    route cannot silently become v2 when its exact symbol is absent.
    """

    original_loader = auto_dispatch._load_extension_and_symbols

    def v4_missing_loader() -> tuple[object, frozenset[str], str, None]:
        return object(), frozenset(("fwd_vshard_p2",)), "0" * 64, None

    auto_dispatch._load_extension_and_symbols = v4_missing_loader
    try:
        decision, extension, digest = auto_dispatch._choose_available_variant(
            auto_dispatch.DispatchDecision(TARGET_VARIANT, TARGET_VARIANT, TARGET_REASON)
        )
    finally:
        auto_dispatch._load_extension_and_symbols = original_loader
    if auto_dispatch._load_extension_and_symbols is not original_loader:
        raise AssertionError("prelaunch negative control loader restoration failed")
    expected_reason = TARGET_REASON + "; fwd_vshard4_p2_missing_prelaunch_fallback_to_baseline"
    if decision.chosen_variant != "baseline" or decision.requested_variant != TARGET_VARIANT or decision.reason != expected_reason or extension is not None or digest != "0" * 64:
        raise AssertionError("v4-symbol-missing control did not fail closed to baseline")
    return {"scope": "synthetic loader inventory only; no map mutation and no auto_dispatch.fwd replacement", "requested_variant": TARGET_VARIANT, "available_symbols": ["fwd_vshard_p2"], "chosen_variant": "baseline", "reason": expected_reason, "extension_is_none": True, "loader_restored": True, "passed": True}


def _descriptor_handoff(*, x: object, cpu: object, gpu: object, initial: object, public_fn: Callable[..., Any], counts: Mapping[str, int], sequences: int) -> dict[str, object]:
    """Prove one real registry call consumes a fresh CPU-authoritative descriptor."""

    import torch

    original_issue = varlen_metadata.issue_descriptor
    issued: list[tuple[object, object, object]] = []

    def issue_spy(*args: object, **kwargs: object) -> object:
        descriptor = original_issue(*args, **kwargs)
        q_arg = args[0] if args else kwargs.get("q")
        cpu_arg = args[1] if len(args) > 1 else kwargs.get("cu_seqlens_cpu")
        if q_arg is not x.q or cpu_arg is not cpu:
            raise AssertionError("descriptor issuer argument identity drift")
        issued.append((descriptor, cpu_arg, varlen_metadata.verify_descriptor(descriptor, cpu_arg)))
        return descriptor

    prior_env = os.environ.get("C1_B300_FLASH_KDA")
    snapshot = candidate._snapshot_input_tensors(x, gpu, cpu, initial)
    varlen_metadata.clear_cache()
    varlen_metadata.issue_descriptor = issue_spy
    try:
        os.environ["C1_B300_FLASH_KDA"] = "1"
        before = dict(counts)
        with torch.inference_mode():
            output = candidate._call(public_fn, x, initial, True, gpu, cpu)
            torch.cuda.synchronize()
        delta = {name: counts[name] - before[name] for name in before}
        decision = dict(auto_dispatch.get_last_decision())
    finally:
        varlen_metadata.issue_descriptor = original_issue
        candidate._restore_env("C1_B300_FLASH_KDA", prior_env)
    if varlen_metadata.issue_descriptor is not original_issue or delta != {"c1": 1, "pinned": 0} or len(issued) != 1:
        raise AssertionError("CPU descriptor/registry handoff counter or restoration drift")
    descriptor, issued_cpu, facts = issued[0]
    offsets = _exact_offsets(list(facts.offsets), "descriptor")
    certified = _exact_offsets(decision.get("certified_varlen_offsets"), "decision")
    if issued_cpu is not cpu or decision.get("chosen_variant") != TARGET_VARIANT or decision.get("reason") != TARGET_REASON:
        raise AssertionError("production descriptor handoff route decision drift")
    immutability = candidate._assert_input_immutability("production/descriptor", snapshot, x, gpu, cpu, initial)
    output_contract = candidate._output_contract(output[0], "production/descriptor")
    final_contract = candidate._final_contract(output[1], sequences, True, "production/descriptor")
    return {"real_public_registry_call": "fla.ops.kda.chunk_kda", "public_call_count": 1, "issue_descriptor_call_count": 1, "route_spy_delta": delta, "chosen_variant": TARGET_VARIANT, "reason": TARGET_REASON, "descriptor_object_id": id(descriptor), "cpu_offsets_object_id": id(cpu), "descriptor_cpu_tensor_identity": True, "offsets": offsets, "certified_offsets": certified, "output_contract": output_contract, "final_state_contract": final_contract, "input_immutability_exact": immutability.get("input_immutability_exact") is True, "input_immutability_fields": immutability.get("fields"), "issue_descriptor_spy_restored": True, "passed": True}


def _zero_cache_stats(value: object, label: str) -> dict[str, int]:
    """Require a literal all-zero metadata-cache observation."""

    if type(value) is not dict or not value:
        raise AssertionError(f"{label}: cache statistics are missing")
    normalized: dict[str, int] = {}
    for name, count in value.items():
        if type(name) is not str or type(count) is not int or count != 0:
            raise AssertionError(f"{label}: cache statistics are not exact zeros: {value!r}")
        normalized[name] = count
    return normalized


def _strict_offsets_for_control(value: object, expected: tuple[int, ...], label: str) -> list[int]:
    """Canonicalize one descriptor fact without accepting bool/float aliases."""

    if type(value) not in (list, tuple) or list(value) != list(expected) or any(type(item) is not int for item in value):
        raise AssertionError(f"{label}: CPU descriptor offsets/type drift")
    return list(expected)


def _state_and_offset_controls(*, x: object, cpu: object, gpu: object, c1: object, public_fn: Callable[..., Any], originals: Mapping[str, Callable[..., Any]], counts: Mapping[str, int], sequences: int) -> dict[str, object]:
    """Keep an adjacent released state distinct and a neighbor layout pinned."""

    import torch

    def public_call(
        label: str,
        initial: object,
        final: bool,
        current_cpu: object,
        current_gpu: object,
        expect: dict[str, int],
        expected_offsets: tuple[int, ...],
        verifier_accepted: bool,
        verifier_reason: str | None,
    ) -> tuple[object, dict[str, object] | None, dict[str, object], dict[str, object]]:
        """Call the real registry after clearing all protocol-local handoff state.

        ``auto_dispatch.get_last_decision`` is launch-side evidence, not a
        rejection-side observation: a correctly rejected C1 backend never
        calls ``auto_dispatch.fwd``.  The verifier spy records that branch
        directly and the issuer spy proves the current CPU tensor was freshly
        certified before the public registry made its choice.
        """

        clear_handoff = getattr(c1, "_clear_varlen_handoff", None)
        handoff_local = getattr(c1, "_handoff_local", None)
        cache_stats = getattr(varlen_metadata, "cache_stats", None)
        if not callable(clear_handoff) or not callable(handoff_local) or not callable(cache_stats):
            raise RuntimeError(f"{label}: required C1 handoff/cache isolation API is unavailable")
        if "chunk_kda_verifier" in vars(c1):
            raise RuntimeError(f"{label}: C1 verifier already has an instance shadow")

        cache_before_clear = dict(cache_stats())
        clear_handoff()
        varlen_metadata.clear_cache()
        cache_after_clear = _zero_cache_stats(dict(cache_stats()), label + "/cache-after-clear")
        if hasattr(handoff_local(), "plan"):
            raise AssertionError(f"{label}: C1 handoff remained pending after explicit clear")

        prior = os.environ.get("C1_B300_FLASH_KDA")
        snapshot = candidate._snapshot_input_tensors(x, current_gpu, current_cpu, initial)
        original_issue = varlen_metadata.issue_descriptor
        original_verifier = c1.chunk_kda_verifier
        issued: list[tuple[object, object, object]] = []
        verifier_calls: list[dict[str, object]] = []
        pending_after_public = False
        cache_after_cleanup: dict[str, int] | None = None

        def arg_at(args: tuple[object, ...], kwargs: Mapping[str, object], index: int, name: str) -> object:
            return args[index] if len(args) > index else kwargs.get(name)

        def issue_spy(*args: object, **kwargs: object) -> object:
            descriptor = original_issue(*args, **kwargs)
            q_arg = args[0] if args else kwargs.get("q")
            cpu_arg = args[1] if len(args) > 1 else kwargs.get("cu_seqlens_cpu")
            if q_arg is not x.q or cpu_arg is not current_cpu:
                raise AssertionError(f"{label}: descriptor issuer input identity drift")
            facts = varlen_metadata.verify_descriptor(descriptor, cpu_arg)
            issued.append((descriptor, cpu_arg, facts))
            return descriptor

        def verifier_spy(*args: object, **kwargs: object) -> object:
            q_arg = arg_at(args, kwargs, 0, "q")
            gpu_arg = arg_at(args, kwargs, 17, "cu_seqlens")
            cpu_arg = arg_at(args, kwargs, 18, "cu_seqlens_cpu")
            if q_arg is not x.q or gpu_arg is not current_gpu or cpu_arg is not current_cpu:
                raise AssertionError(f"{label}: public verifier input identity drift")
            outcome = original_verifier(*args, **kwargs)
            if type(outcome) is not tuple or len(outcome) != 2 or type(outcome[0]) is not bool or outcome[1] is not None and type(outcome[1]) is not str:
                raise AssertionError(f"{label}: verifier return schema drift")
            verifier_calls.append({"accepted": outcome[0], "reason": outcome[1]})
            return outcome

        try:
            varlen_metadata.issue_descriptor = issue_spy
            c1.chunk_kda_verifier = verifier_spy
            os.environ["C1_B300_FLASH_KDA"] = "1"
            before = dict(counts)
            with torch.inference_mode():
                value = candidate._call(public_fn, x, initial, final, current_gpu, current_cpu)
                torch.cuda.synchronize()
            delta = {name: counts[name] - before[name] for name in before}
            decision = dict(auto_dispatch.get_last_decision()) if expect["c1"] == 1 else None
            pending_after_public = hasattr(handoff_local(), "plan")
        finally:
            candidate._restore_env("C1_B300_FLASH_KDA", prior)
            varlen_metadata.issue_descriptor = original_issue
            if "chunk_kda_verifier" in vars(c1):
                delattr(c1, "chunk_kda_verifier")
            restored_verifier = c1.chunk_kda_verifier
            if (
                getattr(original_verifier, "__self__", None) is not c1
                or getattr(restored_verifier, "__self__", None) is not c1
                or getattr(original_verifier, "__func__", None) is not getattr(restored_verifier, "__func__", None)
            ):
                raise AssertionError(f"{label}: C1 verifier binding restoration drift")
            clear_handoff()
            varlen_metadata.clear_cache()
            cache_after_cleanup = _zero_cache_stats(dict(cache_stats()), label + "/cache-after-cleanup")
        if delta != expect:
            raise AssertionError(f"{label}: registry route delta {delta!r} != {expect!r}")
        if varlen_metadata.issue_descriptor is not original_issue or len(issued) != 1 or len(verifier_calls) != 1:
            raise AssertionError(f"{label}: fresh descriptor/verifier call count drift")
        descriptor, issued_cpu, facts = issued[0]
        certified_offsets = _strict_offsets_for_control(facts.offsets, expected_offsets, label + "/issuer")
        verifier = verifier_calls[0]
        if verifier["accepted"] is not verifier_accepted or verifier["reason"] != verifier_reason:
            raise AssertionError(f"{label}: verifier outcome drift: {verifier!r}")
        if pending_after_public:
            raise AssertionError(f"{label}: public call left a stale C1 handoff pending")
        isolation = {
            "clear_handoff_api": "C1B300FlashKDABackend._clear_varlen_handoff",
            "metadata_clear_api": "varlen_metadata.clear_cache",
            "cache_before_clear": cache_before_clear,
            "cache_after_clear": cache_after_clear,
            "handoff_empty_after_clear": True,
            "handoff_empty_after_public": True,
            "cache_after_cleanup": cache_after_cleanup,
            "passed": True,
        }
        public_verifier = {
            "call_count": 1,
            "q_tensor_identity": True,
            "gpu_offsets_tensor_identity": True,
            "cpu_offsets_tensor_identity": issued_cpu is current_cpu,
            "accepted": verifier_accepted,
            "reason": verifier_reason,
            "issuer_call_count": 1,
            "issuer_cpu_offsets_tensor_identity": issued_cpu is current_cpu,
            "certified_offsets": certified_offsets,
            "issuer_spy_restored": True,
            "verifier_spy_restored": True,
            "descriptor_object_id": id(descriptor),
            "passed": True,
        }
        return value, decision, candidate._assert_input_immutability(label, snapshot, x, current_gpu, current_cpu, initial), {"isolation": isolation, "verifier": public_verifier}

    # This is the immediate public state neighbour, already deliberately
    # released as vshard2 rather than inheriting the fp32-both vshard4 route.
    state_public, state_decision, state_immutable, state_evidence = public_call(
        "adjacent_fp32_final_only", None, True, cpu, gpu, {"c1": 1, "pinned": 0}, TARGET_OFFSETS, True, None
    )
    if state_decision is None or state_decision.get("chosen_variant") != "vshard2_p2" or state_decision.get("reason") != "varlen_skew_n6_h12_t12288_fp32_final_only_whitelist_hit":
        raise AssertionError("adjacent state route drift")
    state_output_contract = candidate._output_contract(state_public[0], "adjacent_fp32_final_only")
    state_final_contract = candidate._final_contract(state_public[1], sequences, True, "adjacent_fp32_final_only")
    fallback_cpu = torch.tensor(FALLBACK_OFFSETS, dtype=torch.int64, device="cpu")
    fallback_gpu = torch.tensor(FALLBACK_OFFSETS, dtype=torch.int64, device="cuda")
    fallback_initial = candidate._initial_state(TARGET_CONTRACT, sequences)
    fallback_public, fallback_decision, fallback_immutable, fallback_evidence = public_call(
        "adjacent_offset_fallback", fallback_initial, True, fallback_cpu, fallback_gpu, {"c1": 0, "pinned": 1}, FALLBACK_OFFSETS, False, f"C1 packed-varlen preflight rejected: {FALLBACK_REASON}"
    )
    with torch.inference_mode():
        pinned = candidate._call(originals["pinned"], x, fallback_initial.clone(), True, fallback_gpu, fallback_cpu)
    if fallback_decision is not None:
        raise AssertionError("adjacent offset fallback unexpectedly launched the C1 dispatcher")
    exact = candidate._exact(fallback_public, pinned, sequences, True, "adjacent_offset_fallback/pinned")
    return {
        "adjacent_state_control": {"contract": "fp32_final_only", "expected_variant": "vshard2_p2", "route_spy_delta": {"c1": 1, "pinned": 0}, "decision": state_decision, "public_verifier": state_evidence["verifier"], "handoff_and_cache_isolation": state_evidence["isolation"], "output_contract": state_output_contract, "final_state_contract": state_final_contract, "input_immutability_exact": state_immutable.get("input_immutability_exact") is True, "passed": True},
        "adjacent_offset_fallback": {"offsets": list(FALLBACK_OFFSETS), "route_spy_delta": {"c1": 0, "pinned": 1}, "public_verifier": fallback_evidence["verifier"], "handoff_and_cache_isolation": fallback_evidence["isolation"], "public_vs_pinned": exact, "input_immutability_exact": fallback_immutable.get("input_immutability_exact") is True, "passed": True},
        "passed": True,
    }


def _initial_result(args: argparse.Namespace) -> dict[str, object]:
    job = os.environ.get("SLURM_JOB_ID", "")
    if SLURM_JOB_ID.fullmatch(job) is None:
        raise RuntimeError("SLURM_JOB_ID must be a strictly positive decimal allocation identity")
    return {"schema_version": SCHEMA_VERSION, "purpose": "production packed-varlen skew FP32-both functional and registry-route freeze; no production policy mutation", "allocation_id": args.allocation_id, "process_index": args.process_index, "allocation": {"slurm_job_id": job, "hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}, "process": {"pid": os.getpid(), "fresh_python_process_required": True}, "target": {"offsets": list(TARGET_OFFSETS), "contract": TARGET_CONTRACT, "variant": TARGET_VARIANT, "reason": TARGET_REASON}, "production_map": {}, "identity": {}, "prelaunch_negative_control": {}, "registry": {}, "correctness": {}, "descriptor_handoff": {}, "controls": {}, "gates": {}, "complete": False}


def _self_test() -> None:
    if TARGET_OFFSETS == FALLBACK_OFFSETS or any(type(value) is not int for value in TARGET_OFFSETS):
        raise AssertionError("constant tuple schema drift")
    if _zero_cache_stats({"entries": 0, "hits": 0}, "self-test/cache") != {"entries": 0, "hits": 0}:
        raise AssertionError("self-test cache normalization drift")
    try:
        _zero_cache_stats({"entries": True}, "self-test/forged-cache")
    except AssertionError:
        pass
    else:
        raise AssertionError("self-test accepted bool cache statistic")
    if _strict_offsets_for_control(TARGET_OFFSETS, TARGET_OFFSETS, "self-test/target-offsets") != list(TARGET_OFFSETS):
        raise AssertionError("self-test control offset normalization drift")
    try:
        _strict_offsets_for_control([0, 1, 2, 3, 4, 5, 12288.0], TARGET_OFFSETS, "self-test/forged-control-offsets")
    except AssertionError:
        pass
    else:
        raise AssertionError("self-test accepted float control offset")
    for forged in ([0, True, 2, 3, 4, 5, 12288], [0, 1, 2, 3, 4, 5, 12288.0]):
        try:
            _exact_offsets(forged, "self-test")
        except AssertionError:
            continue
        raise AssertionError("self-test accepted non-exact integer offsets")
    try:
        _canonical_production_map({((0, True), "none"): "vshard2_p2"}, "self forged map")
    except RuntimeError:
        pass
    else:
        raise AssertionError("self-test accepted bool in production-map offsets")
    print("RUNNER_SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-id", choices=ALLOCATION_IDS)
    parser.add_argument("--process-index", type=int, choices=(0, 1))
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.allocation_id, args.process_index, args.reference_root, args.json)):
            raise RuntimeError("--self-test cannot combine with GPU arguments")
        _self_test()
        return
    if None in (args.allocation_id, args.process_index, args.reference_root, args.json):
        parser.error("--allocation-id, --process-index, --reference-root, and --json are required")
    if any(os.environ.get(name) != "1" for name in (CLEAN_ENV, "C1_B300_FLASH_KDA", "C1_B300_VARLEN_CPU_DESCRIPTOR", "FLA_FLASH_KDA")):
        raise RuntimeError("clean-GPU plus C1 CPU-descriptor and FLA opt-ins are required")
    patched_text, fla_text = os.environ.get("PATCHED_ROOT"), os.environ.get("FLA_ROOT")
    if not patched_text or not fla_text:
        raise RuntimeError("PATCHED_ROOT and FLA_ROOT are required")
    _target_cell()
    result = _initial_result(args)
    _write(args.json, result)
    # All dispatcher/source/hash gates below are CPU-only and precede CUDA.
    result["production_map"] = _production_map_gate()
    result["prelaunch_negative_control"] = _missing_v4_prelaunch_negative_control()
    result["identity"] = {"runner": _runner_identity(), "protocol_shell": _shell_identity(), "patched": _patched_identity(Path(patched_text)), "runtime_import_ledger": _pre_cuda_runtime_ledger(args), "production_source_hashes_external": {"auto_dispatch_env": AUTO_SHA_ENV, "auto_dispatch_sha256": _lower_sha_from_env(AUTO_SHA_ENV), "fla_backend_env": FLA_BACKEND_SHA_ENV, "fla_backend_sha256": _lower_sha_from_env(FLA_BACKEND_SHA_ENV), "passed": True}}
    result["gates"]["pre_cuda_source_and_map"] = {"passed": True}
    result["gates"]["v4_symbol_missing_prelaunch_baseline"] = {"passed": True}
    _write(args.json, result)
    result["gates"]["python_pre_torch_clean"] = _python_clean_gpu_gate()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shared.torch = torch
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import fla_backend
    from fla.ops.kda import chunk_kda
    shared.common = common
    # candidate._identity pins the imported B300 extension and public FLA
    # registry callable identity after the CPU-only source ledger above.
    result["identity"]["b300_and_fla"] = candidate._identity(Path(patched_text), Path(fla_text), args.reference_root)
    if result["identity"]["b300_and_fla"]["device"].get("passed") is not True:
        raise RuntimeError("B300 device gate failed")
    c1, pinned, _registry, registry = candidate._registry_backends()
    originals, counts = candidate._install_spies(c1, pinned)
    result["registry"] = {"public_callable": "fla.ops.kda.chunk_kda", "snapshot": registry, "instrumentation": "instance-local C1/pinned backend counters only; dispatcher unmodified"}
    primary: BaseException | None = None
    try:
        torch_ref, helper_identity = confirmation._load_pinned_reference_without_build(common, args.reference_root)
        result["identity"]["pinned_reference_load"] = helper_identity
        cell = _target_cell()
        x = shared._make_inputs(cell.case, args.seed + args.process_index * 100_003)
        try:
            cpu, gpu = candidate._cpu_offsets(cell.case.lengths), x.cu_seqlens
            if gpu is None:
                raise AssertionError("target GPU offsets missing")
            with torch.inference_mode():
                varlen_metadata.clear_cache()
                correctness = candidate._positive_cell(cell, x, cpu, gpu, originals, counts, chunk_kda, c1, pinned, torch_ref, args.seed)
            if correctness.get("passed") is not True:
                raise AssertionError("target exact correctness failed")
            decision = correctness.get("public_decision")
            public_spy = correctness.get("public_c1_spy")
            handoff = correctness.get("public_handoff_prepare")
            if not isinstance(decision, Mapping) or decision.get("chosen_variant") != TARGET_VARIANT or decision.get("reason") != TARGET_REASON:
                raise AssertionError("target public decision exactness failed")
            if not isinstance(public_spy, Mapping) or public_spy.get("delta") != {"c1": 1, "pinned": 0}:
                raise AssertionError("target public C1 spy must be +1/pinned +0")
            if not isinstance(handoff, Mapping) or not isinstance(handoff.get("c1"), Mapping) or not isinstance(handoff.get("pinned"), Mapping) or handoff["c1"].get("prepare_delta") != 1 or handoff["pinned"].get("prepare_delta") != 0:
                raise AssertionError("target C1/pinned prepare handoff deltas drift")
            result["correctness"] = correctness
            descriptor_initial = candidate._initial_state(TARGET_CONTRACT, cell.case.sequences)
            result["descriptor_handoff"] = _descriptor_handoff(x=x, cpu=cpu, gpu=gpu, initial=descriptor_initial, public_fn=chunk_kda, counts=counts, sequences=cell.case.sequences)
            result["controls"] = _state_and_offset_controls(x=x, cpu=cpu, gpu=gpu, c1=c1, public_fn=chunk_kda, originals=originals, counts=counts, sequences=cell.case.sequences)
            result["gates"].update({"target_public_route": {"c1": 1, "pinned": 0, "passed": True}, "exact_output_and_fp32_final": {"passed": True}, "input_and_initial_immutability": {"passed": True}, "cpu_descriptor_handoff": {"passed": True}, "adjacent_state_and_offset_controls": {"passed": True}})
        finally:
            del x
            torch.cuda.empty_cache()
    except BaseException as exc:
        primary = exc
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        c1.chunk_kda, pinned.chunk_kda = originals["c1"], originals["pinned"]
        result["gates"]["backend_spies_restored"] = {"passed": c1.chunk_kda is originals["c1"] and pinned.chunk_kda is originals["pinned"]}
        result["gates"]["prepare_descriptor_binding_restored"] = candidate._assert_no_prepare_instance_shadow(c1, "production-freeze restoration")
        try:
            result["production_map"].update(_production_map_post_gate(result["production_map"]))
        except BaseException as map_error:
            result["production_map_post_failure"] = {"type": type(map_error).__name__, "message": str(map_error)}
            if primary is None:
                primary = map_error
        _write(args.json, result)
    if primary is not None:
        raise primary
    if result["production_map"].get("passed_after_workload") is not True:
        raise RuntimeError("production map object identity drifted")
    result["complete"] = True
    _write(args.json, result)
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
