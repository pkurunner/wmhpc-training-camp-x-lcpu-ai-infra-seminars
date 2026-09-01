#!/usr/bin/env python3
"""Fresh-PID relative-only public-FLA release protocol for one skew varlen cell.

This test never edits a production whitelist.  It temporarily adds the target
entry only to the process-local, existing mutable dispatch-map object and
proves same-object restoration before the Python process exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
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


SCHEMA_VERSION = 3
ALLOCATION_IDS = ("A1", "A2")
TARGET_OFFSETS = (0, 1, 2, 3, 4, 5, 12288)
TARGET_CASE_NAME, TARGET_CONTRACT, TARGET_VARIANT = "skew_n6_h12_t12288", "fp32_both", "vshard4_p2"
PATHS = ("public_registry_c1", "public_registry_pinned")
SAMPLES, REPEATS, WARMUP = 1000, 2, 100
MIN_SPEEDUP_X = 1.02
TELEMETRY_SM_CLOCK_POLICY = {
    "all_positive": True,
    "minimum_median_mhz": 1000.0,
    "near_median_ratio": 0.95,
    "minimum_near_median_fraction": 0.80,
}
RUNNER_SHA_ENV = "C1_VARLEN_FP32_BOTH_RUNNER_SHA256"
PROTOCOL_SHELL_SHA_ENV = "EXPECTED_PROTOCOL_SHELL_SHA256"
PROTOCOL_SHELL_PATH_ENV = "C1_VARLEN_FP32_BOTH_PROTOCOL_SHELL_PATH"
CANDIDATE_HELPER_SHA256 = "e07481e7e66551f1f9eaafa7f1fb9db9ef193f1694ff831ee0a07802c8ffbe14"
PRODUCTION_WRAPPER_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
PATCHED_EXACT_DIRTY_FILES = {
    "csrc/flash_kda.cpp": "38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4",
    "csrc/fwd.h": "613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083",
    "csrc/smxx/fwd_launch.cu": "a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928",
}
CANDIDATE_RUNTIME_IDENTITY_KEYS = frozenset((
    "auto_dispatch", "fla_backend", "varlen_metadata", "confirmation_runner",
    "shared_seqcount_runner", "prefetch2", "vshard4_prefetch2", "harness",
    "pinned_torch_ref", "pinned_reference_helper",
))
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
RUNTIME_LEDGER_AUTHORITY = "owned_runner_current_runtime_import_ledger"
SLURM_JOB_ID = re.compile(r"[1-9][0-9]*\Z")
CLEAN_ENV = "C1_VARLEN_FP32_BOTH_CLEAN_GPU"
GPU_FIELDS = ("index", "uuid", "pstate", "clocks.current.sm", "clocks.current.memory", "power.draw", "temperature.gpu", "power.limit")
PRODUCTION_MAP = {(TARGET_OFFSETS, "none"): "vshard2_p2", (TARGET_OFFSETS, "fp32_final_only"): "vshard2_p2"}
TEMPORARY_MAP = {**PRODUCTION_MAP, (TARGET_OFFSETS, TARGET_CONTRACT): TARGET_VARIANT}
TIMING_CONTRACT = "after start.record/start.synchronize and before end.record/end.synchronize, the sole invocation is candidate._call(public chunk_kda, x, initial, True, gpu_offsets, cpu_offsets), including its symmetric backend route-counter instrumentation; route selection, counter snapshots/deltas, decision inspection, event construction, elapsed_time, and raw recording are outside"


def write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runner_identity() -> dict[str, object]:
    path = Path(__file__).resolve(strict=True)
    expected = os.environ.get(RUNNER_SHA_ENV, "")
    actual = sha(path)
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected) or actual != expected:
        raise RuntimeError("runner SHA authorization failed")
    expected_path = (REPO_ROOT / "assignment02/team/c1_flashkda/challenge_varlen_fp32_both/run_varlen_fp32_both_release.py").resolve(strict=True)
    if path != expected_path:
        raise RuntimeError("runner resolved path authorization failed")
    return {"path": str(path), "expected_path": str(expected_path), "expected_root": str((REPO_ROOT / "assignment02").resolve(strict=True)), "sha256": actual, "sha256_gate_pass": True}


def protocol_shell_identity() -> dict[str, object]:
    """Record the *caller-authorized* shell, without self-hashing it in shell source."""

    expected = os.environ.get(PROTOCOL_SHELL_SHA_ENV, "")
    shell_text = os.environ.get(PROTOCOL_SHELL_PATH_ENV, "")
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected) or not shell_text:
        raise RuntimeError("external protocol-shell authorization missing")
    path = Path(shell_text).resolve(strict=True)
    actual = sha(path)
    if actual != expected:
        raise RuntimeError("external protocol-shell SHA authorization failed")
    expected_path = (REPO_ROOT / "assignment02/team/c1_flashkda/challenge_varlen_fp32_both/run_clean_varlen_fp32_both_release.sh").resolve(strict=True)
    if path != expected_path:
        raise RuntimeError("external protocol-shell path authorization failed")
    return {"path": str(path), "expected_path": str(expected_path), "expected_root": str((REPO_ROOT / "assignment02").resolve(strict=True)), "sha256": actual, "sha256_gate_pass": True}


def candidate_helper_identity() -> dict[str, object]:
    """Pin the imported public-handoff helper, not merely its module name."""

    path = Path(candidate.__file__).resolve(strict=True)
    actual = sha(path)
    expected_path = (REPO_ROOT / "assignment02/team/c1_flashkda/challenge_varlen_dispatch/run_varlen_fla_handoff_candidate.py").resolve(strict=True)
    if path != expected_path or actual != CANDIDATE_HELPER_SHA256:
        raise RuntimeError("candidate handoff helper SHA authorization failed")
    return {"path": str(path), "expected_path": str(expected_path), "expected_root": str((REPO_ROOT / "assignment02").resolve(strict=True)), "sha256": actual, "sha256_gate_pass": True}


def production_wrapper_identity(patched_root: Path) -> dict[str, object]:
    root = patched_root.resolve(strict=True)
    path = (root / "flash_kda/__init__.py").resolve(strict=True)
    expected_path = root / "flash_kda/__init__.py"
    if path != expected_path or sha(path) != PRODUCTION_WRAPPER_SHA256:
        raise RuntimeError("patched flash_kda wrapper identity authorization failed")
    return {"path": str(path), "expected_path": str(expected_path), "expected_root": str(root), "sha256": PRODUCTION_WRAPPER_SHA256, "sha256_gate_pass": True}


def patched_tracked_identity(patched_root: Path) -> dict[str, object]:
    """The patched tree is intentionally dirty, but only in this frozen trio."""

    root = patched_root.resolve(strict=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"], check=True, capture_output=True, text=True).stdout.splitlines()
    records: dict[str, dict[str, object]] = {}
    for line in status:
        if len(line) < 4 or line[:2] != " M" or line[2] != " ":
            raise RuntimeError("patched tracked-tree status is not the exact permitted unstaged set")
        relative = line[3:]
        if relative not in PATCHED_EXACT_DIRTY_FILES or relative in records:
            raise RuntimeError("patched tracked-tree path is outside the exact permitted dirty set")
        path = (root / relative).resolve(strict=True)
        records[relative] = {"status": " M", "path": str(path), "sha256": sha(path)}
    if head != PATCHED_COMMIT or set(records) != set(PATCHED_EXACT_DIRTY_FILES):
        raise RuntimeError("patched tracked-tree commit/dirty-set drift")
    for relative, digest in PATCHED_EXACT_DIRTY_FILES.items():
        if records[relative]["sha256"] != digest:
            raise RuntimeError("patched tracked-tree file SHA drift: " + relative)
    return {"root": str(root), "head": PATCHED_COMMIT, "dirty_files": records, "gate_pass": True}


def merge_owned_identity(candidate_identity: Mapping[str, object], owned_identity: Mapping[str, object]) -> dict[str, object]:
    """Never let helper identity replacement erase protocol-owned authority."""

    required = {"runner", "protocol_shell", "candidate_helper", "production_wrapper", "patched_tracked_identity"}
    if set(owned_identity) != required:
        raise RuntimeError("owned identity key set drift")
    merged = dict(candidate_identity)
    merged.update(owned_identity)
    if not required.issubset(merged):
        raise RuntimeError("owned identity was lost while merging helper identity")
    return merged


def exact_offsets(value: object, label: str) -> list[int]:
    """Reject bool/int lookalikes before recording any canonical offset claim."""

    if type(value) is not list or len(value) != len(TARGET_OFFSETS) or any(type(item) is not int for item in value) or value != list(TARGET_OFFSETS):
        raise AssertionError(label + ": exact canonical offsets/type drift")
    return list(TARGET_OFFSETS)


def identity_schema_self_test() -> None:
    owned = {
        "runner": {"tag": "runner"}, "protocol_shell": {"tag": "shell"}, "candidate_helper": {"tag": "helper"},
        "production_wrapper": {"tag": "wrapper"}, "patched_tracked_identity": {"tag": "dirty"},
    }
    merged = merge_owned_identity({"device": {"tag": "candidate"}}, owned)
    if set(owned) - set(merged) or merged["protocol_shell"] != owned["protocol_shell"] or merged["production_wrapper"] != owned["production_wrapper"]:
        raise AssertionError("merged raw identity schema lost protocol shell/wrapper authority")
    try:
        merge_owned_identity({}, {name: value for name, value in owned.items() if name != "protocol_shell"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("identity schema self-test accepted missing protocol shell")
    for label, forged in (
        ("descriptor-false-offset", [False, 1, 2, 3, 4, 5, 12288]),
        ("decision-true-offset", [0, True, 2, 3, 4, 5, 12288]),
    ):
        try:
            exact_offsets(forged, label)
        except AssertionError:
            continue
        raise AssertionError("offset schema self-test accepted bool forgery: " + label)


def runtime_import_identities(args: argparse.Namespace, common: object, fla_backend: object) -> dict[str, dict[str, object]]:
    """Pin the complete imported public path after the production-table update.

    The shared candidate helper remains intentionally pinned at its published
    SHA.  Its older embedded dependency ledger cannot authorize a newer
    production dispatcher, so this isolated protocol owns an equally strict,
    current ledger rather than weakening or monkey-patching that helper.
    """

    owned = REPO_ROOT / "assignment02/team/c1_flashkda"
    helper_text = os.environ.get(confirmation.REFERENCE_HELPER_PATH_ENV)
    helper_sha = os.environ.get(confirmation.REFERENCE_HELPER_SHA_ENV)
    if not helper_text or helper_sha != RUNTIME_IMPORT_SHA256["pinned_reference_helper"]:
        raise RuntimeError("pinned reference helper identity environment drift")
    files = {
        "auto_dispatch": (Path(auto_dispatch.__file__), owned / "challenge_tp8_dispatch/auto_dispatch.py", owned),
        "fla_backend": (Path(fla_backend.__file__), owned / "challenge_tp8_dispatch/fla_backend.py", owned),
        "varlen_metadata": (Path(varlen_metadata.__file__), owned / "challenge_tp8_dispatch/varlen_metadata.py", owned),
        "confirmation_runner": (Path(confirmation.__file__), owned / "challenge_varlen_dispatch/run_varlen_dispatch_confirmation.py", owned),
        "shared_seqcount_runner": (Path(shared.__file__), owned / "challenge_seqcount_dispatch/run_seqcount_dispatch.py", owned),
        "prefetch2": (owned / "challenge_prefetch2/prefetch2.py", owned / "challenge_prefetch2/prefetch2.py", owned),
        "vshard4_prefetch2": (owned / "challenge_vshard4_prefetch2/vshard4_prefetch2.py", owned / "challenge_vshard4_prefetch2/vshard4_prefetch2.py", owned),
        "harness": (Path(common.__file__), owned / "harness/validate_and_bench.py", owned),
        "pinned_torch_ref": (args.reference_root / "tests/torch_ref.py", args.reference_root / "tests/torch_ref.py", args.reference_root),
        "pinned_reference_helper": (Path(helper_text), Path(helper_text), Path(helper_text).parent),
    }
    if set(files) != CANDIDATE_RUNTIME_IDENTITY_KEYS or set(files) != set(RUNTIME_IMPORT_SHA256):
        raise RuntimeError("runtime import identity ledger key drift")
    identities: dict[str, dict[str, object]] = {}
    for name, (actual_path, expected_path, expected_root) in files.items():
        actual, expected, root = actual_path.resolve(strict=True), expected_path.resolve(strict=True), expected_root.resolve(strict=True)
        if actual != expected:
            raise RuntimeError(f"{name}: imported from {actual}, expected {expected}")
        digest = sha(actual)
        if digest != RUNTIME_IMPORT_SHA256[name]:
            raise RuntimeError(f"{name}: SHA authorization failed")
        identities[name] = {"path": str(actual), "expected_path": str(expected), "expected_root": str(root), "sha256": digest, "sha256_gate_pass": True}
    return identities


def target_case() -> object:
    matches = [case for case in confirmation.CASES if case.name == TARGET_CASE_NAME]
    if len(matches) != 1:
        raise RuntimeError("target case missing or ambiguous")
    case = matches[0]
    if tuple(case.lengths) != (1, 1, 1, 1, 1, 12283) or case.total_tokens != 12288 or case.batch != 1 or case.heads != 12:
        raise RuntimeError("target case structural drift")
    return case


def target_cell() -> object:
    return candidate.Cell(target_case(), TARGET_CONTRACT, TARGET_VARIANT)


def map_gate(expected: Mapping[tuple[tuple[int, ...], str], str], label: str) -> dict[str, object]:
    evidence = candidate._assert_map_values_and_behavior(expected, label)
    if evidence.get("passed") is not True:
        raise RuntimeError(label + ": dispatch-map behavior failed")
    return evidence


def install_target() -> tuple[object, dict[str, object]]:
    original = auto_dispatch._VARLEN_PUBLIC_VARIANTS
    if not isinstance(original, dict) or dict(original) != PRODUCTION_MAP:
        raise RuntimeError("frozen production varlen map drift")
    before = map_gate(PRODUCTION_MAP, "relative-release/production-before")
    try:
        original.clear(); original.update(TEMPORARY_MAP)
        if auto_dispatch._VARLEN_PUBLIC_VARIANTS is not original:
            raise RuntimeError("temporary map replaced object")
        installed = map_gate(TEMPORARY_MAP, "relative-release/temporary-target")
    except BaseException:
        if auto_dispatch._VARLEN_PUBLIC_VARIANTS is original:
            original.clear(); original.update(PRODUCTION_MAP)
        raise
    return original, {"production_before": before, "temporary_target": installed, "map_object_id": id(original), "passed": True}


def restore_target(original: object) -> dict[str, object]:
    restored = candidate._restore_production_map(original, PRODUCTION_MAP)
    if restored.get("passed") is not True or restored.get("map_object_id") != id(original):
        raise RuntimeError("same-object production map restoration failed")
    return restored


def gpu_state(stage: str) -> dict[str, object]:
    proc = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{stage}: nvidia-smi failed")
    rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"{stage}: expected exactly one visible GPU")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != len(GPU_FIELDS) or not all(values):
        raise RuntimeError(f"{stage}: malformed telemetry row")
    return {"stage": stage, "fields": list(GPU_FIELDS), "values": dict(zip(GPU_FIELDS, values, strict=True)), "passed": True}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] * (1.0 - position + lower) + ordered[upper] * (position - lower)


def summary(values: list[float]) -> dict[str, float | int]:
    if len(values) != SAMPLES or not all(math.isfinite(value) and value > 0.0 for value in values):
        raise RuntimeError("invalid CUDA-event raw samples")
    return {"samples": len(values), "mean_ms": statistics.fmean(values), "p50_ms": percentile(values, .5), "p95_ms": percentile(values, .95), "p99_ms": percentile(values, .99), "min_ms": min(values), "max_ms": max(values)}


def select(path: str) -> None:
    os.environ["C1_B300_FLASH_KDA"] = "1" if path == PATHS[0] else "0" if path == PATHS[1] else (_ for _ in ()).throw(ValueError(path))


def expected_route(path: str) -> dict[str, int]:
    return {"c1": 1, "pinned": 0} if path == PATHS[0] else {"c1": 0, "pinned": 1}


def repeat(*, repeat_index: int, x: object, cpu: object, gpu: object, initial: object, public_fn: Callable[..., Any], counts: dict[str, int], cell: object) -> dict[str, object]:
    import torch

    if SAMPLES % 2 or WARMUP % 2:
        raise RuntimeError("unbalanced preregistered schedule")
    snapshot = candidate._snapshot_input_tensors(x, gpu, cpu, initial)
    def invoke() -> object:
        return candidate._call(public_fn, x, initial, True, gpu, cpu)
    def order(index: int) -> tuple[str, str]:
        return PATHS if index % 2 == 0 else tuple(reversed(PATHS))
    with torch.inference_mode():
        warm_before = dict(counts)
        first_c1: dict[str, object] | None = None
        last_c1: dict[str, object] | None = None
        for index in range(WARMUP):
            for path in order(index):
                select(path); before = dict(counts); output = invoke(); del output
                delta = {name: counts[name] - before[name] for name in before}
                if delta != expected_route(path):
                    raise AssertionError(f"warm route drift {delta}")
                if path == PATHS[0]:
                    last_c1 = auto_dispatch.get_last_decision()
                    first_c1 = first_c1 or last_c1
                    if last_c1.get("chosen_variant") != TARGET_VARIANT:
                        raise AssertionError("warm C1 variant drift")
        torch.cuda.synchronize()
        warm_delta = {name: counts[name] - warm_before[name] for name in warm_before}
        if warm_delta != {"c1": WARMUP, "pinned": WARMUP} or first_c1 is None or last_c1 is None or last_c1.get("canonical_cache_hit") is not True:
            raise AssertionError("warm route/cache contract drift")
        raw = {path: [] for path in PATHS}
        timed_before = dict(counts)
        stream = torch.cuda.current_stream()
        for index in range(SAMPLES):
            for path in order(index):
                select(path); before = dict(counts)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(stream); start.synchronize()
                output = invoke()
                end.record(stream); end.synchronize()
                delta = {name: counts[name] - before[name] for name in before}
                if delta != expected_route(path):
                    raise AssertionError(f"sample {index}/{path}: route drift {delta}")
                if path == PATHS[0] and auto_dispatch.get_last_decision().get("chosen_variant") != TARGET_VARIANT:
                    raise AssertionError("timed C1 variant drift")
                raw[path].append(float(start.elapsed_time(end)))
                del output, start, end
    timed_delta = {name: counts[name] - timed_before[name] for name in timed_before}
    if timed_delta != {"c1": SAMPLES, "pinned": SAMPLES}:
        raise AssertionError("timed route count drift")
    immutability = candidate._assert_input_immutability(f"relative/repeat{repeat_index}", snapshot, x, gpu, cpu, initial)
    paths = {path: summary(values) for path, values in raw.items()}
    speedups = {q: float(paths[PATHS[1]][q]) / float(paths[PATHS[0]][q]) for q in ("p50_ms", "p95_ms", "p99_ms")}
    passed = all(value >= MIN_SPEEDUP_X for value in speedups.values())
    return {"repeat_index": repeat_index, "timing_contract": TIMING_CONTRACT, "schedule": "even sample C1->pinned; odd sample pinned->C1; route selection outside events", "raw_samples_ms": raw, "paths": paths, "speedup_c1_over_pinned_x": speedups, "relative_gate": {"minimum_speedup_x": MIN_SPEEDUP_X, "percentiles": ["p50_ms", "p95_ms", "p99_ms"], "passed": passed}, "warmup_route_spy_delta": warm_delta, "timed_route_spy_delta": timed_delta, "per_sample_route_spy_assertions": {PATHS[0]: SAMPLES, PATHS[1]: SAMPLES, "passed": True}, "first_warm_c1_decision": first_c1, "last_warm_c1_decision": last_c1, "input_immutability_exact": immutability.get("input_immutability_exact") is True, "input_immutability_fields": immutability.get("fields"), "passed": True}


def descriptor_freshness(*, repeat_index: int, x: object, cpu: object, gpu: object, initial: object, public_fn: Callable[..., Any], counts: dict[str, int], sequences: int, prior: list[object]) -> dict[str, object]:
    """Audit one *real* public C1 call while observing descriptor issuance.

    This is deliberately outside CUDA-event timing.  The wrapped function is
    the same module-level issuer used by the C1 backend, and no descriptor is
    constructed by this runner.  A fresh cache is used for the probe, then
    cleared again before warm-up/timing.
    """

    import torch

    original_issue = varlen_metadata.issue_descriptor
    issued: list[tuple[object, object, object]] = []

    def issue_spy(*args: object, **kwargs: object) -> object:
        descriptor = original_issue(*args, **kwargs)
        if len(args) >= 2:
            q_arg, cpu_arg = args[0], args[1]
        else:
            try:
                q_arg, cpu_arg = kwargs["q"], kwargs["cu_seqlens_cpu"]
            except KeyError as exc:
                raise AssertionError("public descriptor issuer argument drift") from exc
        facts = varlen_metadata.verify_descriptor(descriptor, cpu_arg)
        if q_arg is not x.q:
            raise AssertionError("public descriptor issuer q identity drift")
        issued.append((descriptor, cpu_arg, facts))
        return descriptor

    previous_route = os.environ.get("C1_B300_FLASH_KDA")
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
        candidate._restore_env("C1_B300_FLASH_KDA", previous_route)
    if varlen_metadata.issue_descriptor is not original_issue:
        raise AssertionError("descriptor issue spy restoration failed")
    if delta != {"c1": 1, "pinned": 0}:
        raise AssertionError(f"descriptor probe route drift: {delta}")
    if decision.get("chosen_variant") != TARGET_VARIANT:
        raise AssertionError("descriptor probe did not take C1 target route")
    if len(issued) != 1:
        raise AssertionError(f"descriptor probe issued {len(issued)} descriptors, expected exactly one")
    descriptor, issued_cpu, facts = issued[0]
    descriptor_offsets = exact_offsets(list(facts.offsets), "public descriptor offsets")
    decision_offsets = exact_offsets(decision.get("certified_varlen_offsets"), "public decision certified offsets")
    if issued_cpu is not cpu or any(descriptor is item for item in prior) or descriptor_offsets != list(TARGET_OFFSETS) or decision_offsets != list(TARGET_OFFSETS):
        raise AssertionError("public descriptor freshness/canonical offsets drift")
    immutability = candidate._assert_input_immutability(
        f"relative/descriptor-probe{repeat_index}", snapshot, x, gpu, cpu, initial
    )
    output_contract = candidate._output_contract(output[0], f"relative/descriptor-probe{repeat_index}")
    final_contract = candidate._final_contract(output[1], sequences, True, f"relative/descriptor-probe{repeat_index}")
    prior.append(descriptor)
    return {
        "repeat_index": repeat_index,
        "probe": "one timing-external public fla.ops.kda.chunk_kda C1 call",
        "public_chunk_kda_call_count": 1,
        "issue_descriptor_call_count": len(issued),
        "route_spy_delta": delta,
        "chosen_variant": decision.get("chosen_variant"),
        "decision": decision,
        "descriptor_object_id": id(descriptor),
        "cpu_offsets_object_id": id(cpu),
        "descriptor_cpu_tensor_identity": issued_cpu is cpu,
        "offsets": descriptor_offsets,
        "fresh_against_prior": True,
        "output_contract": output_contract,
        "final_state_contract": final_contract,
        "input_immutability_exact": immutability.get("input_immutability_exact") is True,
        "input_immutability_fields": immutability.get("fields"),
        "issue_descriptor_spy_restored": True,
        "cache_cleared_before_probe": True,
        "passed": True,
    }


def initial(args: argparse.Namespace) -> dict[str, object]:
    job = os.environ.get("SLURM_JOB_ID", "")
    if SLURM_JOB_ID.fullmatch(job) is None:
        raise RuntimeError("SLURM_JOB_ID must be a strictly positive decimal allocation identity")
    return {"schema_version": SCHEMA_VERSION, "purpose": "relative-only packed-varlen release evidence; no production mutation authority", "allocation_id": args.allocation_id, "process_index": args.process_index, "allocation": {"slurm_job_id": job, "hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}, "process": {"pid": os.getpid(), "fresh_python_process_required": True}, "target": {"cell": f"{TARGET_CASE_NAME}/{TARGET_CONTRACT}", "offsets": list(TARGET_OFFSETS), "variant": TARGET_VARIANT}, "pre_registered": {"repeats": REPEATS, "samples_per_path_per_repeat": SAMPLES, "warmup_per_path_per_repeat": WARMUP, "minimum_relative_speedup_x": MIN_SPEEDUP_X, "telemetry_sm_clock_policy": dict(TELEMETRY_SM_CLOCK_POLICY), "policy": "relative-only; no absolute-latency rule exists in this protocol"}, "identity": {}, "gates": {}, "correctness": {}, "fallback": {}, "descriptor_freshness": [], "performance": {"repeats": []}, "map": {"installed": False, "restored": False}, "complete": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-id", choices=ALLOCATION_IDS); parser.add_argument("--process-index", type=int, choices=(0, 1))
    parser.add_argument("--reference-root", type=Path); parser.add_argument("--json", type=Path); parser.add_argument("--seed", type=int, default=20260910); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.allocation_id, args.process_index, args.reference_root, args.json)):
            raise RuntimeError("--self-test cannot combine with GPU-run arguments")
        identity_schema_self_test(); print("RUNNER_IDENTITY_SCHEMA_SELF_TEST_PASS"); return
    if args.allocation_id is None or args.process_index is None or args.reference_root is None or args.json is None:
        parser.error("--allocation-id, --process-index, --reference-root, and --json are required unless --self-test")
    target_cell(); result = initial(args); write(args.json, result)
    if any(os.environ.get(name) != "1" for name in (CLEAN_ENV, "C1_B300_FLASH_KDA", "C1_B300_VARLEN_CPU_DESCRIPTOR", "FLA_FLASH_KDA")):
        raise RuntimeError("clean shell plus explicit C1 CPU-descriptor and FLA opt-ins required")
    patched_root, fla_root = os.environ.get("PATCHED_ROOT"), os.environ.get("FLA_ROOT")
    if not patched_root or not fla_root:
        raise RuntimeError("PATCHED_ROOT and FLA_ROOT required")
    owned_identity = {
        "runner": runner_identity(),
        "protocol_shell": protocol_shell_identity(),
        "candidate_helper": candidate_helper_identity(),
        "production_wrapper": production_wrapper_identity(Path(patched_root)),
        "patched_tracked_identity": patched_tracked_identity(Path(patched_root)),
    }
    result["identity"] = owned_identity
    preclean = candidate._python_clean_gpu_gate()
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    shared.torch = torch
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import fla_backend
    from fla.ops.kda import chunk_kda
    shared.common = common
    identity = merge_owned_identity(candidate._identity(Path(patched_root), Path(fla_root), args.reference_root), owned_identity)
    runtime_ledger = runtime_import_identities(args, common, fla_backend)
    if set(runtime_ledger) != CANDIDATE_RUNTIME_IDENTITY_KEYS or not all(
        isinstance(item, Mapping) and item.get("sha256_gate_pass") is True
        for item in runtime_ledger.values()
    ):
        raise RuntimeError("runtime import identity ledger drift")
    identity["runtime_import_identities"] = runtime_ledger
    identity["runtime_import_ledger_authority"] = {
        "provider": RUNTIME_LEDGER_AUTHORITY,
        "candidate_helper_embedded_ledger_trusted": False,
        "reason": "helper SHA is pinned, but its embedded ledger predates the current production registration token",
        "expected_keys": sorted(CANDIDATE_RUNTIME_IDENTITY_KEYS),
    }
    identity["python_pre_torch_clean"] = preclean
    result["identity"] = identity
    c1, pinned, _registry, registry = candidate._registry_backends(); originals, counts = candidate._install_spies(c1, pinned)
    result["registry"] = {"snapshot": registry, "spies": "symmetric backend instrumentation; counter snapshots/deltas outside timed events"}
    original_map: object | None = None; primary: BaseException | None = None
    try:
        original_map, installed = install_target(); result["map"] = {"installed": True, "restored": False, "installation": installed}
        torch_ref, helper = confirmation._load_pinned_reference_without_build(common, args.reference_root); result["identity"]["pinned_reference_helper"] = helper
        cell = target_cell(); x = shared._make_inputs(cell.case, args.seed + args.process_index * 100_003)
        try:
            # The candidate helper's positive correctness and mismatch-fallback
            # probes are part of the public inference contract too.  Keep their
            # full body (not merely the timed closure) inside inference_mode.
            with torch.inference_mode():
                cpu, gpu = candidate._cpu_offsets(cell.case.lengths), x.cu_seqlens
                if gpu is None: raise AssertionError("target lost GPU offsets")
                varlen_metadata.clear_cache()
                correctness = candidate._positive_cell(cell, x, cpu, gpu, originals, counts, chunk_kda, c1, pinned, torch_ref, args.seed)
                if correctness.get("passed") is not True: raise AssertionError("exact public correctness failed")
                result["correctness"] = correctness
                result["fallback"] = candidate._gpu_structural_mismatch_fallback(x, cpu, gpu, c1, pinned, counts)
                if result["fallback"].get("passed") is not True: raise AssertionError("fallback proof failed")
        finally:
            del x; torch.cuda.empty_cache()
        result["gates"]["prepare_spy_restored"] = candidate._assert_no_prepare_instance_shadow(c1, "before timing")
        result["performance"]["gpu_state_before_timing"] = gpu_state("before_timing")
        descriptors: list[object] = []
        for index in range(REPEATS):
            x = shared._make_inputs(cell.case, args.seed + args.process_index * 100_003 + (index + 1) * 1009)
            try:
                cpu, gpu = candidate._cpu_offsets(cell.case.lengths), x.cu_seqlens
                if gpu is None: raise AssertionError("timed target lost GPU offsets")
                probe = descriptor_freshness(
                    repeat_index=index, x=x, cpu=cpu, gpu=gpu,
                    initial=candidate._initial_state(TARGET_CONTRACT, cell.case.sequences),
                    public_fn=chunk_kda, counts=counts, sequences=cell.case.sequences,
                    prior=descriptors,
                )
                varlen_metadata.clear_cache()
                cache_after_probe = varlen_metadata.cache_stats()
                if cache_after_probe != {"entries": 0, "hits": 0, "misses": 0, "capture_miss_rejections": 0, "capture_hit_rejections": 0}:
                    raise AssertionError("descriptor probe cache was not cleared before timing")
                probe["cache_cleared_after_probe_before_timing"] = True
                probe["cache_after_probe_clear"] = cache_after_probe
                result["descriptor_freshness"].append(probe)
                item = repeat(repeat_index=index, x=x, cpu=cpu, gpu=gpu, initial=candidate._initial_state(TARGET_CONTRACT, cell.case.sequences), public_fn=chunk_kda, counts=counts, cell=cell)
                result["performance"]["repeats"].append(item); write(args.json, result)
            finally:
                del x; torch.cuda.empty_cache()
        result["performance"]["gpu_state_after_timing"] = gpu_state("after_timing")
        result["gates"].update({"target_exact_public_route": {"passed": True}, "fallback": {"passed": True}, "descriptor_freshness": {"passed": True}, "route": {"passed": True}, "identity": {"passed": True}})
    except BaseException as exc:
        primary = exc; result["failure"] = {"type": type(exc).__name__, "message": str(exc)}; raise
    finally:
        restoration_error: BaseException | None = None
        try:
            if original_map is not None:
                restored = restore_target(original_map); result["map"]["restored"] = True; result["map"]["restoration"] = restored
        except BaseException as exc:
            restoration_error = exc; result["map_restoration_failure"] = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            c1.chunk_kda, pinned.chunk_kda = originals["c1"], originals["pinned"]
            result["gates"]["backend_spies_restored"] = {"passed": c1.chunk_kda is originals["c1"] and pinned.chunk_kda is originals["pinned"]}
            result["gates"]["prepare_spy_restored_after"] = candidate._assert_no_prepare_instance_shadow(c1, "after restoration")
            write(args.json, result)
        if restoration_error is not None and primary is None: raise restoration_error
    if result["map"].get("restored") is not True: raise RuntimeError("map not restored")
    result["complete"] = True; write(args.json, result); print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
