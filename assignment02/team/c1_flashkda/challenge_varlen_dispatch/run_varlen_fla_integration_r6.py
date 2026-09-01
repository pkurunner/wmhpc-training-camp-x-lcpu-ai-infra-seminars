#!/usr/bin/env python3
"""B300 r6 production-freeze audit for CPU-authoritative packed varlen.

No dispatcher, FLA, backend, or metadata source is changed here.  The audit
registers the opt-in C1 backend only in this process and proves public registry
selection with instance-local spies on both the C1 and pinned backends.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_seqcount_dispatch import run_seqcount_dispatch as shared  # noqa: E402
from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch, varlen_metadata  # noqa: E402
from assignment02.team.c1_flashkda.challenge_varlen_dispatch import run_varlen_dispatch_confirmation as confirmation  # noqa: E402


DIM = 128
HEADS = 12
PERFORMANCE_SEED = 20260901
PERFORMANCE_WARMUP = 100
PERFORMANCE_SAMPLES = 1000
PERFORMANCE_REPEATS = 2
PERFORMANCE_MIN_MARGIN = 0.02
CLEAN_GPU_GATE_ENV = "C1_VARLEN_FLA_INTEGRATION_R6_CLEAN_GPU"
RUNNER_SHA256_ENV = "C1_VARLEN_FLA_INTEGRATION_R6_RUNNER_SHA256"
AUDITED_EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
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
RUNTIME_DEPENDENCY_SHA256 = {
    "auto_dispatch": "2b817adb7d21d1f223e8df4616eeccd74e34a5b1944492211f0f0254147ba883",
    "fla_backend": "8555995c04ecd666a580ddee02eae1d34820ef1a601cbad5d10f9c6b8505974b",
    "varlen_metadata": "f89a97ba284c7a24a3df54efca7bb60eb70f9cbda659200bd6cb3e8dfdaf4ccd",
    "confirmation_runner": "9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b",
    "shared_seqcount_runner": "4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f",
    "prefetch2": "752126488487ac317a7ee167b660b0895562e4877aedbc4ce2a599c1f59a10d0",
    "vshard4_prefetch2": "445c90815919fb8c1db3bc79289d10029ea77b1d9accb393c326c120ba1f8385",
    "harness": "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52",
    "pinned_torch_ref": "bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5",
    "pinned_reference_helper": "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f",
}


@dataclass(frozen=True)
class Cell:
    case: shared.Case
    contract: str
    expected_variant: str | None

    @property
    def key(self) -> str:
        return f"{self.case.name}/{self.contract}"


# The production whitelist is deliberately the public-performance release
# intersection, not the broader raw-ABI confirmation matrix.  The r6 runner
# checks the imported production map against this literal before any CUDA
# initialization and never mutates that map in-process.
RELEASED_PUBLIC_VARIANTS = {
    "skew_n6_h12_t12288/none": "vshard2_p2",
    "skew_n6_h12_t12288/fp32_final_only": "vshard2_p2",
}
_RELEASED_PRODUCTION_MAP = {
    ((0, 1, 2, 3, 4, 5, 12288), "none"): "vshard2_p2",
    ((0, 1, 2, 3, 4, 5, 12288), "fp32_final_only"): "vshard2_p2",
}
RAW_RELEASE_FAILED_CELL = Cell(confirmation.CASES[1], "fp32_both", None)
RECORD_ONLY_CELL = Cell(confirmation.CASES[2], "fp32_both", None)
_SPECIAL_FALLBACK_CLASSIFICATIONS = {
    RAW_RELEASE_FAILED_CELL.key: "raw_release_failed",
    RECORD_ONLY_CELL.key: "record_only",
}
_LAYOUT_NAME_BY_CASE = {
    "equal_n2_h12_t2048": "equal_n2_h12_t4096",
    "equal_n4_h12_t2048": "equal_n4_h12_t8192",
    "mixed_n6_h12_t8192": "mixed_n6_h12_t8192",
    "skew_n6_h12_t12288": "skew_n6_h12_t12288",
}
POSITIVE_CELLS = tuple(
    Cell(case, contract, RELEASED_PUBLIC_VARIANTS[f"{case.name}/{contract}"])
    for case in confirmation.CASES
    for contract in confirmation.FLA_PUBLIC_CONTRACTS
    if f"{case.name}/{contract}" in RELEASED_PUBLIC_VARIANTS
)
POLICY_FALLBACK_CELLS = tuple(
    Cell(case, contract, None)
    for case in confirmation.CASES
    for contract in confirmation.FLA_PUBLIC_CONTRACTS
    if f"{case.name}/{contract}" not in RELEASED_PUBLIC_VARIANTS
)
POLICY_FALLBACK_MANIFEST = {
    cell.key: {
        "classification": _SPECIAL_FALLBACK_CLASSIFICATIONS.get(cell.key, "public_release_failed"),
        "action": "pinned_baseline_only",
    }
    for cell in POLICY_FALLBACK_CELLS
}
POLICY_FALLBACK_C1_REASONS = {
    cell.key: (
        "C1 packed-varlen preflight rejected: "
        f"varlen_{_LAYOUT_NAME_BY_CASE[cell.case.name]}_{cell.contract}_not_whitelisted"
    )
    for cell in POLICY_FALLBACK_CELLS
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
FIXED_REPRESENTATIVE = "b2_h12_t2048/none"


def _production_public_map_gate() -> dict[str, object]:
    """Freeze the imported production map before this runner can initialize CUDA.

    The test deliberately observes the exact private policy table rather than
    inferring it from later routing.  This makes additions, removals, aliases,
    or a mutable-key representation fail before the first GPU-side action.
    """

    raw = getattr(auto_dispatch, "_VARLEN_PUBLIC_VARIANTS", None)
    if type(raw) is not dict:
        raise RuntimeError("production varlen public map must be a built-in dict")
    normalized: dict[tuple[tuple[int, ...], str], str] = {}
    for key, variant in raw.items():
        if type(key) is not tuple or len(key) != 2:
            raise RuntimeError("production varlen public map key must be (offsets, contract)")
        offsets, contract = key
        if type(offsets) is not tuple or not offsets or any(type(value) is not int for value in offsets):
            raise RuntimeError("production varlen public offsets must be a nonempty built-in int tuple")
        if type(contract) is not str or type(variant) is not str:
            raise RuntimeError("production varlen public contract and variant must be built-in strings")
        normalized[(offsets, contract)] = variant
    if normalized != _RELEASED_PRODUCTION_MAP:
        raise RuntimeError(
            "production packed-varlen map drift: "
            f"expected {_RELEASED_PRODUCTION_MAP!r}, got {normalized!r}"
        )
    return {
        "passed": True,
        "exact_entries": [
            {"offsets": list(offsets), "contract": contract, "variant": variant}
            for (offsets, contract), variant in _RELEASED_PRODUCTION_MAP.items()
        ],
        "entry_count": len(normalized),
        "checked_before_cuda_initialization": True,
        "runner_mutates_production_map": False,
    }


def _capture_prepare_spy_state(backend: object) -> tuple[bool, object | None, object]:
    """Capture both instance-shadow state and normal descriptor resolution."""

    state = vars(backend)
    had_instance_shadow = "_prepare_varlen" in state
    prior_instance_value = state.get("_prepare_varlen")
    resolved = backend._prepare_varlen
    if not callable(resolved):
        raise RuntimeError("C1 backend _prepare_varlen is not callable")
    return had_instance_shadow, prior_instance_value, resolved


def _restore_prepare_spy_state(
    backend: object,
    *,
    had_instance_shadow: bool,
    prior_instance_value: object | None,
    normally_resolved: object,
) -> dict[str, object]:
    """Undo temporary instrumentation without retaining a bound-method shadow."""

    state = vars(backend)
    if had_instance_shadow:
        state["_prepare_varlen"] = prior_instance_value
        if state.get("_prepare_varlen") is not prior_instance_value:
            raise RuntimeError("C1 prepare spy did not restore the exact prior instance value")
        return {"had_instance_shadow": True, "descriptor_binding_restored": None, "passed": True}
    if "_prepare_varlen" in state:
        delattr(backend, "_prepare_varlen")
    if "_prepare_varlen" in vars(backend):
        raise RuntimeError("C1 prepare spy left an instance _prepare_varlen shadow")
    restored = backend._prepare_varlen
    if (
        getattr(normally_resolved, "__self__", None) is not backend
        or getattr(restored, "__self__", None) is not backend
        or getattr(normally_resolved, "__func__", None) is not getattr(restored, "__func__", None)
    ):
        raise RuntimeError("C1 prepare spy did not restore normal class-descriptor binding")
    return {"had_instance_shadow": False, "descriptor_binding_restored": True, "passed": True}


def _assert_no_prepare_instance_shadow(backend: object, label: str) -> dict[str, object]:
    """Fail closed if non-timed instrumentation leaks into performance."""

    if "_prepare_varlen" in vars(backend):
        raise RuntimeError(f"{label}: C1 _prepare_varlen remains shadowed on the instance")
    bound = backend._prepare_varlen
    if not callable(bound) or getattr(bound, "__self__", None) is not backend or getattr(bound, "__func__", None) is None:
        raise RuntimeError(f"{label}: C1 _prepare_varlen is not normally class-bound")
    return {"instance_shadow": False, "descriptor_binding": True, "passed": True}


def _write(path: Path, result: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integration_runner_identity() -> dict[str, object]:
    """Bind the result to the exact runner authorized by the outer audit."""

    path = Path(__file__).resolve(strict=True)
    actual = _sha(path)
    expected = os.environ.get(RUNNER_SHA256_ENV)
    if expected is None:
        raise RuntimeError(f"{RUNNER_SHA256_ENV} is required for a GPU integration experiment")
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise RuntimeError(f"{RUNNER_SHA256_ENV} must be a lowercase SHA256")
    if actual != expected:
        raise RuntimeError(
            "integration runner SHA256 mismatch: "
            f"outer audit expected {expected}, loaded {actual} at {path}"
        )
    return {
        "path": str(path),
        "sha256": actual,
        "sha256_gate_pass": True,
        "expected_sha256_environment": RUNNER_SHA256_ENV,
    }


def _file_identity(actual_path: Path, expected_path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    """Require both import/source path and bytes, not just a matching basename."""

    actual = actual_path.resolve(strict=True)
    expected = expected_path.resolve(strict=True)
    if actual != expected:
        raise RuntimeError(f"loaded {label} from {actual}, expected {expected}")
    digest = _sha(actual)
    if digest != expected_sha256:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected_sha256}, got {digest}")
    return {"path": str(actual), "sha256": digest, "sha256_gate_pass": True}


def _runtime_dependency_identities(
    args: argparse.Namespace,
    common: object,
    fla_backend: object,
) -> dict[str, object]:
    """Freeze the complete C1/public-path source ledger used by this process."""

    owned = REPO_ROOT / "assignment02/team/c1_flashkda"
    helper_text = os.environ.get(confirmation.REFERENCE_HELPER_PATH_ENV)
    helper_sha = os.environ.get(confirmation.REFERENCE_HELPER_SHA_ENV)
    if not helper_text:
        raise RuntimeError(f"{confirmation.REFERENCE_HELPER_PATH_ENV} is required")
    if helper_sha != RUNTIME_DEPENDENCY_SHA256["pinned_reference_helper"]:
        raise RuntimeError("pinned reference helper SHA environment drift")
    files = {
        "auto_dispatch": (Path(auto_dispatch.__file__), owned / "challenge_tp8_dispatch/auto_dispatch.py"),
        "fla_backend": (Path(fla_backend.__file__), owned / "challenge_tp8_dispatch/fla_backend.py"),
        "varlen_metadata": (Path(varlen_metadata.__file__), owned / "challenge_tp8_dispatch/varlen_metadata.py"),
        "confirmation_runner": (Path(confirmation.__file__), owned / "challenge_varlen_dispatch/run_varlen_dispatch_confirmation.py"),
        "shared_seqcount_runner": (Path(shared.__file__), owned / "challenge_seqcount_dispatch/run_seqcount_dispatch.py"),
        "prefetch2": (owned / "challenge_prefetch2/prefetch2.py", owned / "challenge_prefetch2/prefetch2.py"),
        "vshard4_prefetch2": (owned / "challenge_vshard4_prefetch2/vshard4_prefetch2.py", owned / "challenge_vshard4_prefetch2/vshard4_prefetch2.py"),
        "harness": (Path(common.__file__), owned / "harness/validate_and_bench.py"),
        "pinned_torch_ref": (args.reference_root / "tests/torch_ref.py", args.reference_root / "tests/torch_ref.py"),
        "pinned_reference_helper": (Path(helper_text), Path(helper_text)),
    }
    if set(files) != set(RUNTIME_DEPENDENCY_SHA256):
        raise AssertionError("runtime dependency ledger key drift")
    return {
        key: _file_identity(actual, expected, RUNTIME_DEPENDENCY_SHA256[key], key)
        for key, (actual, expected) in files.items()
    }


def _initial_result(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 4,
        "purpose": "r6 public FLA packed-varlen C1/pinned production freeze; no policy mutation",
        "seed": args.seed,
        "positive_cells": [
            {"cell": cell.key, "expected_variant": cell.expected_variant} for cell in POSITIVE_CELLS
        ],
        "record_only_cell": {"cell": RECORD_ONLY_CELL.key, "action": "pinned_baseline_only"},
        "policy_fallback_cells": {
            key: dict(value) for key, value in POLICY_FALLBACK_MANIFEST.items()
        },
        "performance": {
            "pre_registered": {
                "cells": [cell.key for cell in POSITIVE_CELLS],
                "fixed_measurement_seed": PERFORMANCE_SEED,
                "paths": ["public_registry_c1", "public_registry_pinned"],
                "repeats": PERFORMANCE_REPEATS,
                "warmup_per_path_per_repeat": PERFORMANCE_WARMUP,
                "cyclic_cuda_event_samples_per_path_per_repeat": PERFORMANCE_SAMPLES,
                "percentiles": list(shared.PERCENTILES),
                "minimum_c1_margin_over_pinned": PERFORMANCE_MIN_MARGIN,
                "prepare_spy_in_timed_region": False,
                "release_rule": "A cell releases only if both repeats have C1 as P50/P95/P99 winner with every margin >=2%; failures remain recorded and do not invalidate completed correctness evidence.",
                "timing_contract": "one current-stream CUDA-event pair wraps each complete public chunk_kda call; start.record is immediately synchronized before the public call and end.record is immediately synchronized before elapsed_time, so the start marker precedes host dispatch/preflight and neither synchronization is in the sample value; samples alternate C1->pinned then pinned->C1 so each path is first exactly 500 times; C1_B300_FLASH_KDA selection changes happen before event record and never inside a sample; no direct raw wrapper is timed.",
                "timed_public_calls": len(POSITIVE_CELLS) * PERFORMANCE_REPEATS * PERFORMANCE_SAMPLES * 2,
                "warmup_public_calls": len(POSITIVE_CELLS) * PERFORMANCE_REPEATS * PERFORMANCE_WARMUP * 2,
                "timed_event_synchronizations": len(POSITIVE_CELLS) * PERFORMANCE_REPEATS * PERFORMANCE_SAMPLES * 2 * 2,
                "one_hour_mean_call_budget_ms_including_warmup": 3600_000 / (len(POSITIVE_CELLS) * PERFORMANCE_REPEATS * (PERFORMANCE_SAMPLES + PERFORMANCE_WARMUP) * 2),
            },
            "cold_miss_observations": {},
            "cells": {},
            "performance_release_cells": [],
            "failed_cells": [],
            "complete": False,
        },
        "negative_cases": [
            "same_n_total_different_split", "varlen_env_unset", "cpu_missing_or_malformed",
            "gpu_structural_mismatch_preflight", "allow_neg_eigval_semantic_fallback",
            "fixed_representative_non_regression",
        ],
        "identity": {},
        "registry": {},
        "positive_results": {},
        "negative_results": {},
        "cache_observations": {},
        "performance_release_cells": [],
        "performance_failed_cells": [],
        "gates": {
            "scope": {
                "positive_cells": len(POSITIVE_CELLS),
                "policy_fallback_cells": len(POLICY_FALLBACK_CELLS),
                "required_positive_cells": 2,
                "required_policy_fallback_cells": 10,
                "passed": len(POSITIVE_CELLS) == 2 and len(POLICY_FALLBACK_CELLS) == 10,
            },
            "production_map": {"passed": False},
            "prepare_spy_restored": {"passed": False},
            "clean_gpu": {"passed": False},
            "device": {"passed": False},
            "extension": {"passed": False},
            "fla_pin": {"passed": False},
            "inference_mode": {"passed": False},
            "registry_spy_identity": {"passed": False},
            "python_nvidia_clean": {"passed": False},
        },
        "complete": False,
    }


def _python_clean_gpu_gate() -> dict[str, object]:
    """Recheck the shell's one-device/idle assertion before Torch imports CUDA."""
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,compute_cap,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if gpu.returncode != 0:
        raise RuntimeError(f"pre-Torch nvidia-smi GPU query failed: {gpu.stderr.strip()}")
    gpu_lines = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    if len(gpu_lines) != 1:
        raise RuntimeError(f"pre-Torch gate requires exactly one visible GPU, got {gpu_lines!r}")
    fields = [field.strip() for field in gpu_lines[0].split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"unparseable pre-Torch GPU identity: {gpu_lines[0]!r}")
    try:
        used_mib = int(fields[4])
    except ValueError as exc:
        raise RuntimeError(f"unparseable pre-Torch GPU memory usage: {fields[4]!r}") from exc
    if used_mib != 0:
        raise RuntimeError(f"pre-Torch clean-GPU gate requires 0 MiB, got {used_mib} MiB")
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if apps.returncode != 0:
        raise RuntimeError(f"pre-Torch nvidia-smi compute-app query failed: {apps.stderr.strip()}")
    app_lines = [
        line.strip()
        for line in apps.stdout.splitlines()
        if line.strip() and "No running compute processes found" not in line
    ]
    if app_lines:
        raise RuntimeError(f"pre-Torch clean-GPU gate found compute apps: {app_lines!r}")
    return {
        "index": fields[0],
        "uuid": fields[1],
        "name": fields[2],
        "compute_capability": fields[3],
        "memory_used_mib": used_mib,
        "compute_apps": app_lines,
        "passed": True,
    }


def _device_identity() -> dict[str, object]:
    import torch

    name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if "B300" not in name.upper() or capability != (10, 3) or sms != 148:
        raise RuntimeError(f"B300 gate failed: name={name!r}, cc={capability}, SMs={sms}")
    return {"name": name, "capability": list(capability), "multiprocessor_count": sms, "passed": True}


def _identity(patched_root: Path, fla_root: Path, reference_root: Path) -> dict[str, object]:
    import flash_kda
    import flash_kda_C

    so = Path(flash_kda_C.__file__).resolve(strict=True)
    py = Path(flash_kda.__file__).resolve(strict=True)
    root = patched_root.resolve(strict=True)
    for path, label in ((so, "extension"), (py, "flash_kda Python")):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"loaded {label} outside PATCHED_ROOT: {path}") from exc
    if _sha(so) != AUDITED_EXTENSION_SHA256:
        raise RuntimeError("audited extension SHA mismatch")
    patched_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if patched_commit != PATCHED_COMMIT:
        raise RuntimeError(f"patched worktree commit mismatch: {patched_commit}")
    reference = reference_root.resolve(strict=True)
    reference_commit = subprocess.run(
        ["git", "-C", str(reference), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if reference_commit != PATCHED_COMMIT:
        raise RuntimeError(f"reference worktree commit mismatch: {reference_commit}")
    reference_status = subprocess.run(
        ["git", "-C", str(reference), "status", "--short", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if reference_status:
        raise RuntimeError(f"reference tracked/staged diff is not clean: {reference_status}")
    commit = subprocess.run(["git", "-C", str(fla_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if commit != FLA_COMMIT:
        raise RuntimeError(f"FLA commit mismatch: {commit}")
    tracked_status = subprocess.run(
        ["git", "-C", str(fla_root), "status", "--short", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_status:
        raise RuntimeError(f"FLA tracked/staged diff is not clean: {tracked_status}")
    hashes = {}
    for relative, expected in FLA_FILE_SHA256.items():
        actual = _sha(fla_root / relative)
        if actual != expected:
            raise RuntimeError(f"FLA source SHA mismatch: {relative}")
        hashes[relative] = actual
    module_paths = {}
    loaded_fla_modules = {}
    for module, relative in {
        "fla": "fla/__init__.py",
        "fla.ops.backends": "fla/ops/backends/__init__.py",
        "fla.ops.kda": "fla/ops/kda/__init__.py",
        "fla.ops.kda.backends": "fla/ops/kda/backends/__init__.py",
        "fla.ops.kda.backends.flash_kda": "fla/ops/kda/backends/flash_kda.py",
        "fla.ops.kda.chunk": "fla/ops/kda/chunk.py",
    }.items():
        loaded_module = importlib.import_module(module)
        loaded = Path(loaded_module.__file__).resolve(strict=True)
        if loaded != (fla_root / relative).resolve(strict=True):
            raise RuntimeError(f"loaded {module} is not from FLA_ROOT")
        loaded_fla_modules[module] = loaded_module
        module_paths[module] = str(loaded)
    public_chunk_kda = getattr(loaded_fla_modules["fla.ops.kda"], "chunk_kda", None)
    implementation_chunk_kda = getattr(loaded_fla_modules["fla.ops.kda.chunk"], "chunk_kda", None)
    if not callable(public_chunk_kda) or not callable(implementation_chunk_kda):
        raise RuntimeError("public or implementation fla.ops.kda.chunk_kda is not callable")
    if public_chunk_kda is not implementation_chunk_kda:
        raise RuntimeError("raw public fla.ops.kda.chunk_kda object is not the implementation export")
    try:
        canonical_public = inspect.unwrap(public_chunk_kda)
        canonical_implementation = inspect.unwrap(implementation_chunk_kda)
        canonical_source_text = inspect.getsourcefile(canonical_public)
        if canonical_source_text is None:
            raise RuntimeError("canonical public chunk_kda has no inspectable source file")
        canonical_source = Path(canonical_source_text).resolve(strict=True)
    except Exception as exc:
        raise RuntimeError("cannot unwrap and resolve canonical public fla.ops.kda.chunk_kda") from exc
    expected_public_source = (fla_root / "fla/ops/kda/chunk.py").resolve(strict=True)
    if (
        canonical_public is not canonical_implementation
        or getattr(canonical_public, "__module__", None) != "fla.ops.kda.chunk"
        or getattr(canonical_public, "__qualname__", None) != "chunk_kda"
        or canonical_source != expected_public_source
    ):
        raise RuntimeError("canonical public fla.ops.kda.chunk_kda export identity drift")
    return {
        "device": _device_identity(),
        "extension": {"path": str(so), "sha256": _sha(so), "passed": True},
        "flash_kda_python": {"path": str(py), "sha256": _sha(py)},
        "source_trees": {
            "patched": {"root": str(root), "commit": patched_commit, "passed": True},
            "reference": {
                "root": str(reference),
                "commit": reference_commit,
                "tracked_status_clean": True,
                "passed": True,
            },
        },
        "fla": {
            "root": str(fla_root.resolve()),
            "commit": commit,
            "tracked_status_clean": True,
            "files": hashes,
            "loaded_modules": module_paths,
            "public_callables": {
                "fla.ops.kda.chunk_kda": {
                    "implementation_identity_match": True,
                    "module": "fla.ops.kda.chunk",
                    "qualname": "chunk_kda",
                    "source_path": str(canonical_source),
                    "passed": True,
                }
            },
            "passed": True,
        },
    }


def _cpu_offsets(lengths: tuple[int, ...]) -> torch.Tensor:
    import torch
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    return torch.tensor(cumulative, dtype=torch.int64, device="cpu")


def _initial_state(contract: str, sequences: int) -> torch.Tensor | None:
    if contract != "fp32_both":
        return None
    import torch
    count = sequences * HEADS * DIM * DIM
    return torch.arange(count, dtype=torch.float32, device="cuda").reshape(sequences, HEADS, DIM, DIM).mul_(1.0 / 8192.0).add_(0.125).contiguous()


def _call_kwargs(x: object, initial: torch.Tensor | None, final: bool, gpu_offsets: torch.Tensor | None, cpu_offsets: torch.Tensor | None, *, allow_neg_eigval: bool = False) -> dict[str, object]:
    return {
        "scale": x.scale, "initial_state": initial, "output_final_state": final,
        "use_qk_l2norm_in_kernel": True, "use_gate_in_kernel": True, "use_beta_sigmoid_in_kernel": True,
        "allow_neg_eigval": allow_neg_eigval, "state_v_first": True,
        "cu_seqlens": gpu_offsets, "cu_seqlens_cpu": cpu_offsets,
        "safe_gate": True, "lower_bound": x.lower_bound, "disable_recompute": False,
        "return_intermediate_states": False, "cp_context": None, "A_log": x.a_log, "dt_bias": x.dt_bias,
    }


def _require_inference_mode(label: str) -> None:
    """Fail closed if an FLA verifier/call escapes the production context."""
    import torch
    if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
        raise RuntimeError(f"{label} requires torch.inference_mode()")


def _call(fn: Callable[..., Any], x: object, initial: torch.Tensor | None, final: bool, gpu_offsets: torch.Tensor | None, cpu_offsets: torch.Tensor | None, *, allow_neg_eigval: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    _require_inference_mode("FLA call")
    return fn(x.q, x.k, x.v, x.g, x.beta, **_call_kwargs(x, initial, final, gpu_offsets, cpu_offsets, allow_neg_eigval=allow_neg_eigval))


def _verify(backend: object, x: object, initial: torch.Tensor | None, final: bool, gpu_offsets: torch.Tensor | None, cpu_offsets: torch.Tensor | None, *, allow_neg_eigval: bool = False) -> tuple[bool, object]:
    _require_inference_mode("FLA verifier")
    verdict = backend.verify("chunk_kda", x.q, x.k, x.v, x.g, x.beta, **_call_kwargs(x, initial, final, gpu_offsets, cpu_offsets, allow_neg_eigval=allow_neg_eigval))
    return bool(verdict[0]), verdict[1]


def _final_contract(value: torch.Tensor | None, sequences: int, required: bool, label: str) -> dict[str, object]:
    import torch
    if not required:
        if value is not None:
            raise AssertionError(f"{label}: unexpected final state")
        return {"present": False}
    if value is None or value.dtype != torch.float32 or tuple(value.shape) != (sequences, HEADS, DIM, DIM) or not value.is_contiguous():
        raise AssertionError(f"{label}: final state must be contiguous FP32 [{sequences},12,128,128]")
    return {"present": True, "dtype": str(value.dtype), "shape": list(value.shape), "contiguous": True}


def _output_contract(value: torch.Tensor, label: str) -> dict[str, object]:
    """Require the public KDA output ABI in addition to numerical equality."""

    import torch
    shape = tuple(value.shape)
    if (
        value.dtype != torch.bfloat16
        or len(shape) != 4
        or shape[0] < 1
        or shape[1] < 1
        or shape[2:] != (HEADS, DIM)
        or not value.is_contiguous()
    ):
        raise AssertionError(f"{label}: output must be contiguous BF16 [B,T,12,128]")
    return {"shape": list(shape), "dtype": str(value.dtype), "contiguous": True}


def _exact(actual: tuple[torch.Tensor, torch.Tensor | None], expected: tuple[torch.Tensor, torch.Tensor | None], sequences: int, final: bool, label: str) -> dict[str, object]:
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common
    actual_output = _output_contract(actual[0], f"{label}/actual_output")
    expected_output = _output_contract(expected[0], f"{label}/expected_output")
    if actual_output != expected_output:
        raise AssertionError(f"{label}: output ABI differs despite equal values")
    common.require_exact(f"{label}/output", actual[0], expected[0])
    result = {
        "output_exact": True,
        "output_max_abs": common.max_abs(actual[0], expected[0]),
        "actual_output": actual_output,
        "expected_output": expected_output,
        "actual_final": _final_contract(actual[1], sequences, final, f"{label}/actual"),
        "expected_final": _final_contract(expected[1], sequences, final, f"{label}/expected"),
    }
    if final:
        assert actual[1] is not None and expected[1] is not None
        common.require_exact(f"{label}/final", actual[1], expected[1])
        result.update({"final_exact": True, "final_max_abs": common.max_abs(actual[1], expected[1])})
    return result


def _input_tensor_map(x: object, gpu_offsets: torch.Tensor | None, cpu_offsets: torch.Tensor | None, initial: torch.Tensor | None) -> dict[str, torch.Tensor]:
    tensors = {
        "q": x.q,
        "k": x.k,
        "v": x.v,
        "g": x.g,
        "beta": x.beta,
        "A_log": x.a_log,
        "dt_bias": x.dt_bias,
    }
    if gpu_offsets is not None:
        tensors["cu_seqlens"] = gpu_offsets
    if cpu_offsets is not None:
        tensors["cu_seqlens_cpu"] = cpu_offsets
    if initial is not None:
        tensors["initial_state"] = initial
    return tensors


def _snapshot_input_tensors(x: object, gpu_offsets: torch.Tensor | None, cpu_offsets: torch.Tensor | None, initial: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Make an exact tensor snapshot before an FLA path receives these inputs."""
    return {name: tensor.clone() for name, tensor in _input_tensor_map(x, gpu_offsets, cpu_offsets, initial).items()}


def _assert_input_immutability(label: str, snapshot: Mapping[str, torch.Tensor], x: object, gpu_offsets: torch.Tensor | None, cpu_offsets: torch.Tensor | None, initial: torch.Tensor | None) -> dict[str, object]:
    import torch

    current = _input_tensor_map(x, gpu_offsets, cpu_offsets, initial)
    if set(current) != set(snapshot):
        raise AssertionError(f"{label}: input snapshot field coverage drift")
    for name, before in snapshot.items():
        after = current[name]
        if (
            before.dtype != after.dtype
            or before.device != after.device
            or tuple(before.shape) != tuple(after.shape)
            or not torch.equal(before, after)
        ):
            raise AssertionError(f"{label}: {name} was mutated")
    return {"input_immutability_exact": True, "fields": sorted(snapshot)}


def _call_with_immutability(label: str, fn: Callable[..., Any], x: object, initial_template: torch.Tensor | None, final: bool, gpu_offsets: torch.Tensor | None, cpu_offsets: torch.Tensor | None) -> tuple[tuple[torch.Tensor, torch.Tensor | None], dict[str, object]]:
    state = None if initial_template is None else initial_template.clone()
    snapshot = _snapshot_input_tensors(x, gpu_offsets, cpu_offsets, state)
    output = _call(fn, x, state, final, gpu_offsets, cpu_offsets)
    return output, _assert_input_immutability(label, snapshot, x, gpu_offsets, cpu_offsets, state)


def _registry_backends() -> tuple[object, object, object, list[dict[str, object]]]:
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import fla_backend
    from fla.ops.kda.backends import kda_registry

    custom = fla_backend.register_backend()
    if fla_backend.register_backend() is not custom:
        raise RuntimeError("C1 registration is not idempotent")
    backends = list(kda_registry._get_sorted_backends())
    c1 = [backend for backend in backends if getattr(backend, "backend_type", None) == "c1_b300_flash_kda"]
    pinned = [backend for backend in backends if getattr(backend, "backend_type", None) == "flash_kda"]
    if c1 != [custom] or len(pinned) != 1:
        raise RuntimeError("cannot reliably distinguish exactly one C1 and one pinned registry backend")
    if backends.index(custom) >= backends.index(pinned[0]):
        raise RuntimeError("C1 backend is not ordered before pinned flash_kda")
    snapshot = [{"backend_type": b.backend_type, "priority": b.priority, "id": id(b)} for b in backends]
    return custom, pinned[0], kda_registry, snapshot


def _install_spies(c1: object, pinned: object) -> tuple[dict[str, Callable[..., Any]], dict[str, int]]:
    originals = {"c1": c1.chunk_kda, "pinned": pinned.chunk_kda}
    counts = {"c1": 0, "pinned": 0}
    def c1_spy(*args: object, **kwargs: object) -> object:
        counts["c1"] += 1
        return originals["c1"](*args, **kwargs)
    def pinned_spy(*args: object, **kwargs: object) -> object:
        counts["pinned"] += 1
        return originals["pinned"](*args, **kwargs)
    c1.chunk_kda, pinned.chunk_kda = c1_spy, pinned_spy
    return originals, counts


def _spy_public(public_fn: Callable[..., Any], x: object, initial: torch.Tensor | None, final: bool, gpu: torch.Tensor | None, cpu: torch.Tensor | None, counts: Mapping[str, int], expect_c1: bool | None, label: str, *, allow_neg_eigval: bool = False) -> tuple[tuple[torch.Tensor, torch.Tensor | None], dict[str, object]]:
    before = dict(counts)
    output = _call(public_fn, x, initial, final, gpu, cpu, allow_neg_eigval=allow_neg_eigval)
    after = dict(counts)
    expected = ({"c1": 1, "pinned": 0} if expect_c1 else {"c1": 0, "pinned": 1}) if expect_c1 is not None else {"c1": 0, "pinned": 0}
    delta = {key: after[key] - before[key] for key in expected}
    if delta != expected:
        raise AssertionError(f"{label}: registry spy delta={delta}, expected={expected}")
    return output, {"before": before, "after": after, "delta": delta, "passed": True}


def _public_handoff_probe(
    cell: Cell,
    x: object,
    initial: torch.Tensor | None,
    final: bool,
    gpu: torch.Tensor,
    cpu: torch.Tensor,
    public_fn: Callable[..., Any],
    c1: object,
    counts: Mapping[str, int],
) -> tuple[
    tuple[torch.Tensor, torch.Tensor | None],
    dict[str, object],
    dict[str, object],
    tuple[torch.Tensor, torch.Tensor | None],
    dict[str, object],
    dict[str, object],
]:
    """Prove verifier-to-body handoff exactly once outside all timed calls."""

    had_instance_shadow, prior_instance_value, original_prepare = _capture_prepare_spy_state(c1)
    prepared_calls = 0

    def prepare_spy(*args: object, **kwargs: object) -> object:
        nonlocal prepared_calls
        prepared_calls += 1
        return original_prepare(*args, **kwargs)

    previous_c1 = os.environ.get("C1_B300_FLASH_KDA")
    c1._prepare_varlen = prepare_spy
    try:
        os.environ["C1_B300_FLASH_KDA"] = "1"
        c1_initial = None if initial is None else initial.clone()
        c1_snapshot = _snapshot_input_tensors(x, gpu, cpu, c1_initial)
        before_c1 = prepared_calls
        public_c1, c1_route = _spy_public(
            public_fn, x, c1_initial, final, gpu, cpu, counts, True, f"{cell.key}/handoff-c1"
        )
        c1_delta = prepared_calls - before_c1
        if c1_delta != 1:
            raise AssertionError(f"{cell.key}: public C1 handoff prepare delta={c1_delta}, expected 1")
        c1_immutability = _assert_input_immutability(
            f"{cell.key}/handoff-c1", c1_snapshot, x, gpu, cpu, c1_initial
        )

        os.environ["C1_B300_FLASH_KDA"] = "0"
        pinned_initial = None if initial is None else initial.clone()
        pinned_snapshot = _snapshot_input_tensors(x, gpu, cpu, pinned_initial)
        before_pinned = prepared_calls
        public_pinned, pinned_route = _spy_public(
            public_fn, x, pinned_initial, final, gpu, cpu, counts, False, f"{cell.key}/handoff-pinned"
        )
        pinned_delta = prepared_calls - before_pinned
        if pinned_delta != 0:
            raise AssertionError(f"{cell.key}: public pinned handoff prepare delta={pinned_delta}, expected 0")
        pinned_immutability = _assert_input_immutability(
            f"{cell.key}/handoff-pinned", pinned_snapshot, x, gpu, cpu, pinned_initial
        )
    finally:
        restore_evidence = _restore_prepare_spy_state(
            c1,
            had_instance_shadow=had_instance_shadow,
            prior_instance_value=prior_instance_value,
            normally_resolved=original_prepare,
        )
        if restore_evidence.get("passed") is not True:
            raise RuntimeError("C1 prepare spy restore evidence failed")
        _restore_env("C1_B300_FLASH_KDA", previous_c1)

    return (
        public_c1,
        c1_route,
        {"prepare_delta": c1_delta, "prepare_calls_total": prepared_calls, "passed": True},
        public_pinned,
        pinned_route,
        {
            "prepare_delta": pinned_delta,
            "prepare_calls_total": prepared_calls,
            "c1_immutability": c1_immutability,
            "pinned_immutability": pinned_immutability,
            "passed": True,
        },
    )


def _reference(torch_ref: Callable[..., None], x: object, initial: torch.Tensor | None, final: bool, sequences: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the pinned reference with exactly the FLA call's FP32 state input."""
    import torch
    final_state = (
        torch.zeros(sequences, HEADS, DIM, DIM, dtype=torch.float32, device="cuda")
        if final
        else None
    )
    return shared._invoke(torch_ref, x, None if initial is None else initial.clone(), final_state)


def _positive_cell(cell: Cell, x: object, cpu: torch.Tensor, gpu: torch.Tensor, originals: Mapping[str, Callable[..., Any]], counts: Mapping[str, int], public_fn: Callable[..., Any], c1: object, pinned: object, torch_ref: Callable[..., None], seed: int) -> dict[str, object]:
    import torch
    final = cell.contract != "none"
    initial = _initial_state(cell.contract, cell.case.sequences)
    c1_ok, c1_reason = _verify(c1, x, None if initial is None else initial.clone(), final, gpu, cpu)
    pinned_ok, pinned_reason = _verify(pinned, x, None if initial is None else initial.clone(), final, gpu, cpu)
    if not c1_ok or not pinned_ok:
        raise AssertionError(f"{cell.key}: verifier failure c1={c1_reason!r}, pinned={pinned_reason!r}")
    with torch.inference_mode():
        reference = _reference(torch_ref, x, initial, final, cell.case.sequences)
        varlen_metadata.clear_cache()
        pinned_output, pinned_immutability = _call_with_immutability(f"{cell.key}/pinned", originals["pinned"], x, initial, final, gpu, cpu)
        direct, direct_immutability = _call_with_immutability(f"{cell.key}/direct", originals["c1"], x, initial, final, gpu, cpu)
        direct_decision = auto_dispatch.get_last_decision()
        (
            public,
            spy,
            handoff_c1,
            public_pinned,
            public_pinned_spy,
            handoff_pinned,
        ) = _public_handoff_probe(cell, x, initial, final, gpu, cpu, public_fn, c1, counts)
        public_immutability = handoff_pinned["c1_immutability"]
        public_pinned_immutability = handoff_pinned["pinned_immutability"]
        decision = auto_dispatch.get_last_decision()
        torch.cuda.synchronize()
    if direct_decision.get("chosen_variant") != cell.expected_variant or decision.get("chosen_variant") != cell.expected_variant:
        raise AssertionError(f"{cell.key}: C1 decision drift direct={direct_decision}, public={decision}")
    if direct_decision.get("canonical_cache_hit") is not False or decision.get("canonical_cache_hit") is not True:
        raise AssertionError(f"{cell.key}: expected miss then hot hit, got direct={direct_decision}, public={decision}")
    return {
        "expected_variant": cell.expected_variant,
        "verifier": {"c1": {"passed": c1_ok, "reason": c1_reason}, "pinned": {"passed": pinned_ok, "reason": pinned_reason}},
        "pinned_vs_torch_ref": _exact(pinned_output, reference, cell.case.sequences, final, f"{cell.key}/pinned"),
        "direct_c1_vs_pinned": _exact(direct, pinned_output, cell.case.sequences, final, f"{cell.key}/direct-pinned"),
        "public_vs_pinned": _exact(public, pinned_output, cell.case.sequences, final, f"{cell.key}/public-pinned"),
        "public_pinned_vs_torch_ref": _exact(public_pinned, reference, cell.case.sequences, final, f"{cell.key}/public-pinned-torch-ref"),
        "direct_c1_vs_torch_ref": _exact(direct, reference, cell.case.sequences, final, f"{cell.key}/direct"),
        "public_vs_torch_ref": _exact(public, reference, cell.case.sequences, final, f"{cell.key}/public"),
        "public_c1_spy": spy,
        "public_pinned_spy": public_pinned_spy,
        "public_handoff_prepare": {"c1": handoff_c1, "pinned": handoff_pinned},
        "direct_decision": direct_decision,
        "public_decision": decision,
        "input_immutability_exact": True,
        "input_immutability_fields": sorted({field for evidence in (pinned_immutability, direct_immutability, public_immutability, public_pinned_immutability) for field in evidence["fields"]}),
        "input_immutability_by_path": {"pinned": pinned_immutability, "direct_c1": direct_immutability, "public_c1": public_immutability, "public_pinned": public_pinned_immutability},
        "cache_stats": varlen_metadata.cache_stats(),
        "passed": True,
    }


def _cpu_canonical_gpu_ignored(x: object, case: shared.Case, cpu: torch.Tensor, gpu: torch.Tensor, originals: Mapping[str, Callable[..., Any]], counts: Mapping[str, int], public_fn: Callable[..., Any], c1: object, torch_ref: Callable[..., None], seed: int) -> dict[str, object]:
    """Same GPU numel but different values must not override the CPU descriptor."""
    import torch
    caller_gpu = gpu.clone()
    alternate = list(cpu.tolist())
    alternate[1] += 1
    alternate[2] -= 1
    caller_gpu.copy_(torch.tensor(alternate, device="cuda", dtype=torch.int64))
    if torch.equal(caller_gpu, gpu):
        raise AssertionError("semantic test did not create distinct caller GPU offsets")
    with torch.inference_mode():
        reference = _reference(torch_ref, x, None, False, case.sequences)
        varlen_metadata.clear_cache()
        direct = _call(originals["c1"], x, None, False, caller_gpu, cpu)
        direct_decision = auto_dispatch.get_last_decision()
        public, spy = _spy_public(public_fn, x, None, False, caller_gpu, cpu, counts, True, "cpu_canonical_gpu_ignored")
        public_decision = auto_dispatch.get_last_decision()
        torch.cuda.synchronize()
    if direct_decision.get("certified_varlen_offsets") != cpu.tolist() or public_decision.get("certified_varlen_offsets") != cpu.tolist():
        raise AssertionError("C1 decision did not record CPU-canonical offsets")
    return {
        "caller_gpu_values_equal_cpu": False,
        "caller_gpu_numel": int(caller_gpu.numel()),
        "cpu_canonical_offsets": cpu.tolist(),
        "direct_vs_cpu_canonical_torch_ref": _exact(direct, reference, case.sequences, False, "cpu_canonical/direct"),
        "public_vs_cpu_canonical_torch_ref": _exact(public, reference, case.sequences, False, "cpu_canonical/public"),
        "public_spy": spy,
        "direct_decision": direct_decision,
        "public_decision": public_decision,
        "passed": True,
    }


def _fixed_representative(originals: Mapping[str, Callable[..., Any]], counts: Mapping[str, int], public_fn: Callable[..., Any], c1: object, pinned: object, seed: int) -> dict[str, object]:
    """A C1 fixed-B2 public call must retain the existing v4 integration path."""
    import torch
    case = shared.Case("b2_h12_t2048", "fixed", 2, HEADS, (2048, 2048), "varlen_integration_fixed_control")
    x = shared._make_inputs(case, seed)
    try:
        c1_ok, c1_reason = _verify(c1, x, None, False, None, None)
        pinned_ok, pinned_reason = _verify(pinned, x, None, False, None, None)
        if not c1_ok or not pinned_ok:
            raise AssertionError(f"fixed representative verifier failure: C1={c1_reason!r}, pinned={pinned_reason!r}")
        with torch.inference_mode():
            pinned_output = _call(originals["pinned"], x, None, False, None, None)
            direct = _call(originals["c1"], x, None, False, None, None)
            direct_decision = auto_dispatch.get_last_decision()
            public, spy = _spy_public(public_fn, x, None, False, None, None, counts, True, FIXED_REPRESENTATIVE)
            public_decision = auto_dispatch.get_last_decision()
            torch.cuda.synchronize()
        if direct_decision.get("chosen_variant") != "vshard4_p2" or public_decision.get("chosen_variant") != "vshard4_p2":
            raise AssertionError(f"fixed representative decision drift: {direct_decision}, {public_decision}")
        return {"expected_variant": "vshard4_p2", "direct_vs_pinned": _exact(direct, pinned_output, 2, False, "fixed/direct"), "public_vs_pinned": _exact(public, pinned_output, 2, False, "fixed/public"), "public_spy": spy, "passed": True}
    finally:
        del x


def _negative_public(
    label: str,
    x: object,
    contract: str,
    cpu_for_public: torch.Tensor | None,
    gpu_valid: torch.Tensor,
    c1: object,
    pinned: object,
    public_fn: Callable[..., Any],
    counts: Mapping[str, int],
    expected_c1_reason: str | None = None,
    *,
    require_exact_fallback: bool = False,
    originals: Mapping[str, Callable[..., Any]] | None = None,
    torch_ref: Callable[..., None] | None = None,
) -> dict[str, object]:
    """Prove a safe pinned fallback, with exact evidence for policy misses.

    The broad structural negatives only establish safe registry takeover.  A
    policy fallback, however, is production behavior: it must additionally
    match both the direct pinned backend and the frozen Torch reference while
    preserving every input supplied to each public/direct call.
    """
    final = contract != "none"
    sequences = len(cpu_for_public) - 1 if cpu_for_public is not None else len(x.cu_seqlens) - 1
    initial = _initial_state(contract, sequences)
    c1_ok, c1_reason = _verify(c1, x, None if initial is None else initial.clone(), final, gpu_valid, cpu_for_public)
    if c1_ok:
        raise AssertionError(f"{label}: C1 verifier unexpectedly accepted")
    if expected_c1_reason is not None and c1_reason != expected_c1_reason:
        raise AssertionError(
            f"{label}: C1 rejected for the wrong reason: {c1_reason!r}"
        )
    pinned_ok, pinned_reason = _verify(pinned, x, None if initial is None else initial.clone(), final, gpu_valid, cpu_for_public)
    if not pinned_ok:
        raise AssertionError(f"{label}: pinned verifier rejected safe fallback: {pinned_reason!r}")
    if expected_c1_reason is not None and pinned_reason is not None:
        raise AssertionError(f"{label}: pinned verifier returned an unexpected reason: {pinned_reason!r}")
    if not require_exact_fallback:
        output, spy = _spy_public(public_fn, x, initial, final, gpu_valid, cpu_for_public, counts, False, label)
        return {
            "c1_verifier": {"passed": False, "reason": c1_reason},
            "pinned_verifier": {"passed": True, "reason": pinned_reason},
            "public_pinned_spy": spy,
            "final": _final_contract(output[1], sequences, final, label),
            "passed": True,
        }

    if expected_c1_reason is None or originals is None or torch_ref is None:
        raise AssertionError(f"{label}: policy fallback exactness reason and dependencies are required")
    import torch

    with torch.inference_mode():
        reference_snapshot = _snapshot_input_tensors(x, gpu_valid, cpu_for_public, initial)
        reference = _reference(torch_ref, x, initial, final, sequences)
        reference_immutability = _assert_input_immutability(
            f"{label}/torch-ref", reference_snapshot, x, gpu_valid, cpu_for_public, initial
        )
        direct, direct_immutability = _call_with_immutability(
            f"{label}/direct-pinned", originals["pinned"], x, initial, final, gpu_valid, cpu_for_public
        )
        public_initial = None if initial is None else initial.clone()
        public_snapshot = _snapshot_input_tensors(x, gpu_valid, cpu_for_public, public_initial)
        output, spy = _spy_public(
            public_fn, x, public_initial, final, gpu_valid, cpu_for_public, counts, False, label
        )
        public_immutability = _assert_input_immutability(
            f"{label}/public-pinned", public_snapshot, x, gpu_valid, cpu_for_public, public_initial
        )
        torch.cuda.synchronize()
    immutability_fields = sorted({
        field
        for evidence in (reference_immutability, direct_immutability, public_immutability)
        for field in evidence["fields"]
    })
    return {
        "c1_verifier": {"passed": False, "reason": c1_reason},
        "pinned_verifier": {"passed": True, "reason": pinned_reason},
        "public_pinned_spy": spy,
        "direct_pinned_vs_torch_ref": _exact(direct, reference, sequences, final, f"{label}/direct-pinned"),
        "public_vs_direct_pinned": _exact(output, direct, sequences, final, f"{label}/public-direct-pinned"),
        "public_vs_torch_ref": _exact(output, reference, sequences, final, f"{label}/public-torch-ref"),
        "final": _final_contract(output[1], sequences, final, label),
        "input_immutability_exact": True,
        "input_immutability_fields": immutability_fields,
        "input_immutability_by_path": {
            "torch_ref": reference_immutability,
            "direct_pinned": direct_immutability,
            "public_pinned": public_immutability,
        },
        "passed": True,
    }


def _allow_neg_eigval_fallback(x: object, cpu: torch.Tensor, gpu: torch.Tensor, c1: object, pinned: object, public_fn: Callable[..., Any], counts: Mapping[str, int]) -> dict[str, object]:
    """C1 must catch doubled-beta before allocation/cache/raw-launch, not fall through."""
    import flash_kda
    import torch

    c1_ok, c1_reason = _verify(c1, x, None, False, gpu, cpu, allow_neg_eigval=True)
    pinned_ok, pinned_reason = _verify(pinned, x, None, False, gpu, cpu, allow_neg_eigval=True)
    result: dict[str, object] = {
        "allow_neg_eigval": True,
        "semantic": "beta=2*sigmoid(beta); raw FlashKDA/C1 implement sigmoid(beta)",
        "c1_verifier": {"passed": c1_ok, "reason": c1_reason},
        "pinned_verifier": {"passed": pinned_ok, "reason": pinned_reason},
        "expected": "C1 priority capture then ValueError before cache/allocation/raw launch",
    }
    if not c1_ok:
        result["failure"] = "C1 verifier did not capture allow_neg_eigval=True before pinned fallback"
        result["passed"] = False
        return result

    original_dispatch = auto_dispatch.fwd
    original_pinned_raw = flash_kda.fwd
    kernel_counts = {"c1_raw_dispatch": 0, "pinned_raw_flash_kda": 0}

    def c1_raw_spy(*args: object, **kwargs: object) -> object:
        kernel_counts["c1_raw_dispatch"] += 1
        return original_dispatch(*args, **kwargs)

    def pinned_raw_spy(*args: object, **kwargs: object) -> object:
        kernel_counts["pinned_raw_flash_kda"] += 1
        return original_pinned_raw(*args, **kwargs)

    def positional_public() -> object:
        # This follows the pinned public chunk_kda positional order exactly
        # through allow_neg_eigval, so a signature/binding regression cannot
        # turn the guard into a different flag.
        return public_fn(
            x.q, x.k, x.v, x.g, x.beta, x.scale, None, False,
            True, True, True, True, True, x.lower_bound, False, False,
            True, gpu, cpu, None, A_log=x.a_log, dt_bias=x.dt_bias,
        )

    def expect_guard(label: str, invoke: Callable[[], object]) -> dict[str, object]:
        before_backend = dict(counts)
        before_kernel = dict(kernel_counts)
        try:
            invoke()
        except ValueError as exc:
            message = str(exc)
            if "allow_neg_eigval" not in message:
                raise AssertionError(f"{label}: wrong C1 guard error: {message!r}") from exc
        else:
            raise AssertionError(f"{label}: public call did not raise C1 allow_neg_eigval guard")
        backend_delta = {key: counts[key] - before_backend[key] for key in before_backend}
        kernel_delta = {key: kernel_counts[key] - before_kernel[key] for key in before_kernel}
        if backend_delta != {"c1": 1, "pinned": 0}:
            raise AssertionError(f"{label}: registry capture spy drift: {backend_delta}")
        if kernel_delta != {"c1_raw_dispatch": 0, "pinned_raw_flash_kda": 0}:
            raise AssertionError(f"{label}: raw kernel path was entered: {kernel_delta}")
        return {"registry_backend_delta": backend_delta, "kernel_launch_delta": kernel_delta, "passed": True}

    auto_dispatch.fwd, flash_kda.fwd = c1_raw_spy, pinned_raw_spy
    try:
        with torch.inference_mode():
            result["keyword_public_call"] = expect_guard(
                "allow_neg_eigval/keyword",
                lambda: _call(public_fn, x, None, False, gpu, cpu, allow_neg_eigval=True),
            )
            result["positional_public_call"] = expect_guard("allow_neg_eigval/positional", positional_public)
    except (AssertionError, ValueError) as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        result["passed"] = False
        return result
    finally:
        auto_dispatch.fwd, flash_kda.fwd = original_dispatch, original_pinned_raw
    result["passed"] = True
    return result


def _gpu_structural_mismatch_fallback(x: object, cpu: torch.Tensor, gpu: torch.Tensor, c1: object, pinned: object, counts: Mapping[str, int]) -> dict[str, object]:
    """Keep malformed GPU offsets out of public FLA, then prove safe pinned takeover."""
    import torch
    malformed_gpu = gpu.to(torch.int32)
    c1_ok, c1_reason = _verify(c1, x, None, False, malformed_gpu, cpu)
    if c1_ok:
        raise AssertionError("GPU structural mismatch was accepted by C1 verifier")
    pinned_ok, pinned_reason = _verify(pinned, x, None, False, gpu, cpu)
    if not pinned_ok:
        raise AssertionError(f"pinned verifier rejected valid GPU offsets: {pinned_reason!r}")
    before = dict(counts)
    output = _call(pinned.chunk_kda, x, None, False, gpu, cpu)
    after = dict(counts)
    delta = {key: after[key] - before[key] for key in before}
    if delta != {"c1": 0, "pinned": 1}:
        raise AssertionError(f"valid pinned takeover spy drift: {delta}")
    return {
        "c1_verifier_malformed_gpu": {"passed": False, "reason": c1_reason},
        "pinned_verifier_valid_gpu": {"passed": True, "reason": pinned_reason},
        "malformed_gpu": {"dtype": str(malformed_gpu.dtype), "numel": int(malformed_gpu.numel())},
        "pinned_direct_takeover_spy": {"before": before, "after": after, "delta": delta, "passed": True},
        "final": _final_contract(output[1], len(cpu) - 1, False, "gpu_structural_mismatch"),
        "passed": True,
    }


def _cache_concurrency_and_capture(case: shared.Case, x: object, cpu: torch.Tensor, gpu: torch.Tensor, counts: Mapping[str, int]) -> dict[str, object]:
    import flash_kda
    import torch
    varlen_metadata.clear_cache()
    descriptor = varlen_metadata.issue_descriptor(x.q, cpu, opt_in=True)
    stream1, stream2 = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(stream1):
        first = varlen_metadata.cached_gpu_offsets(x.q, gpu, cpu, descriptor)
    with torch.cuda.stream(stream2):
        second = varlen_metadata.cached_gpu_offsets(x.q, gpu, cpu, descriptor)
    torch.cuda.synchronize()
    stats = varlen_metadata.cache_stats()
    if first.cache_hit or not second.cache_hit or first.key != second.key or first.tensor is not second.tensor or stats["misses"] != 1 or stats["hits"] < 1:
        raise AssertionError(f"two-stream cache event ordering failed: first={first.cache_hit}, second={second.cache_hit}, stats={stats}")
    graph_factory = getattr(torch.cuda, "CUDAGraph", None)
    graph_context = getattr(torch.cuda, "graph", None)
    capture_probe = getattr(torch.cuda, "is_current_stream_capturing", None)
    if not callable(graph_factory) or not callable(graph_context) or not callable(capture_probe):
        raise RuntimeError("CUDA graph capture APIs unavailable; fail closed")
    original_dispatch, original_pinned_raw = auto_dispatch.fwd, flash_kda.fwd
    raw_counts = {"c1_raw_dispatch": 0, "pinned_raw_flash_kda": 0}

    def unexpected_c1_raw(*args: object, **kwargs: object) -> object:
        raw_counts["c1_raw_dispatch"] += 1
        raise AssertionError("CUDA graph cache test attempted C1 raw dispatch")

    def unexpected_pinned_raw(*args: object, **kwargs: object) -> object:
        raw_counts["pinned_raw_flash_kda"] += 1
        raise AssertionError("CUDA graph cache test attempted pinned raw FlashKDA")

    def capture_rejection(label: str, expected_error: type[BaseException]) -> tuple[dict[str, object], dict[str, int]]:
        before_backend = dict(counts)
        before_raw = dict(raw_counts)
        graph = graph_factory()
        auto_dispatch.fwd, flash_kda.fwd = unexpected_c1_raw, unexpected_pinned_raw
        try:
            try:
                with graph_context(graph):
                    varlen_metadata.cached_gpu_offsets(x.q, gpu, cpu, descriptor)
            except expected_error as exc:
                reason = str(exc)
            else:
                raise AssertionError(f"{label}: cache access during CUDA graph capture was not rejected")
        finally:
            auto_dispatch.fwd, flash_kda.fwd = original_dispatch, original_pinned_raw
        backend_delta = {key: counts[key] - before_backend[key] for key in before_backend}
        raw_delta = {key: raw_counts[key] - before_raw[key] for key in before_raw}
        if backend_delta != {"c1": 0, "pinned": 0}:
            raise AssertionError(f"{label}: C1/pinned backend was entered during metadata-only capture test: {backend_delta}")
        if raw_delta != {"c1_raw_dispatch": 0, "pinned_raw_flash_kda": 0}:
            raise AssertionError(f"{label}: C1/pinned raw kernel path was entered: {raw_delta}")
        return {
            "rejected": True,
            "error_type": expected_error.__name__,
            "reason": reason,
            "c1_pinned_backend_delta": backend_delta,
            "c1_pinned_kernel_launch_delta": raw_delta,
            "passed": True,
        }, varlen_metadata.cache_stats()

    varlen_metadata.clear_cache()
    cold_capture, cold_stats = capture_rejection("cache_capture_cold", varlen_metadata.CaptureCacheMissError)
    if cold_stats.get("capture_miss_rejections") != 1:
        raise AssertionError(f"cold capture rejection was not accounted: {cold_stats}")
    varlen_metadata.clear_cache()
    post_clear = varlen_metadata.cache_stats()
    if post_clear.get("entries") != 0:
        raise AssertionError(f"clear_cache did not safely drop capture-test entry: {post_clear}")
    warm = varlen_metadata.cached_gpu_offsets(x.q, gpu, cpu, descriptor)
    if warm.cache_hit:
        raise AssertionError("expected the explicit warm-cache setup to be a miss")
    capture_hit_error = getattr(varlen_metadata, "CaptureCacheHitError", None)
    if not isinstance(capture_hit_error, type) or not issubclass(capture_hit_error, BaseException):
        raise RuntimeError("CaptureCacheHitError is unavailable; cache hit capture must fail closed")
    hot_capture, hot_stats = capture_rejection("cache_capture_hot", capture_hit_error)
    if hot_stats.get("capture_hit_rejections") != 1:
        raise AssertionError(f"hot capture rejection was not accounted: {hot_stats}")
    return {
        "two_stream_same_tuple": {"first_cache_hit": first.cache_hit, "second_cache_hit": second.cache_hit, "stats": stats, "passed": True},
        "capture": {
            "cold_miss": {**cold_capture, "stats": cold_stats},
            "clear_after_cold": post_clear,
            "warm_before_hot": {"cache_hit": warm.cache_hit, "key": str(warm.key)},
            "hot_hit": {**hot_capture, "stats": hot_stats},
        },
    }


def _cpu_construction_check() -> dict[str, object]:
    """Exercise descriptor issuance without importing Torch or touching CUDA."""
    class Device:
        type = "cpu"
        index = None
    class Cpu:
        dtype = "torch.int64"
        shape = (3,)
        device = Device()
        def is_contiguous(self) -> bool: return True
        def tolist(self) -> list[int]: return [0, 3, 8]
    class QDevice:
        type = "cuda"
        index = 0
    class Q:
        shape = (1, 8, HEADS, DIM)
        device = QDevice()
    cpu = Cpu()
    descriptor = varlen_metadata.issue_descriptor(Q(), cpu, opt_in=True)
    facts = varlen_metadata.verify_descriptor(descriptor, cpu)
    if facts.offsets != (0, 3, 8) or facts.sequence_count != 2 or facts.total_tokens != 8:
        raise AssertionError("CPU descriptor construction drift")
    class ProbeBackend:
        def _prepare_varlen(self) -> str:
            return "class-descriptor"
    normal = ProbeBackend()
    normal_state = _capture_prepare_spy_state(normal)
    normal._prepare_varlen = lambda: "temporary-spy"
    normal_restore = _restore_prepare_spy_state(
        normal,
        had_instance_shadow=normal_state[0],
        prior_instance_value=normal_state[1],
        normally_resolved=normal_state[2],
    )
    normal_gate = _assert_no_prepare_instance_shadow(normal, "CPU normal probe restore")
    overridden = ProbeBackend()
    sentinel = lambda: "preexisting-instance-override"
    overridden._prepare_varlen = sentinel
    override_state = _capture_prepare_spy_state(overridden)
    overridden._prepare_varlen = lambda: "temporary-spy"
    override_restore = _restore_prepare_spy_state(
        overridden,
        had_instance_shadow=override_state[0],
        prior_instance_value=override_state[1],
        normally_resolved=override_state[2],
    )
    if vars(overridden).get("_prepare_varlen") is not sentinel:
        raise AssertionError("CPU override restore did not preserve exact instance value")
    return {
        "offsets": list(facts.offsets),
        "sequence_count": facts.sequence_count,
        "total_tokens": facts.total_tokens,
        "prepare_spy_restore": {
            "normal": {"restore": normal_restore, "no_shadow_gate": normal_gate},
            "preexisting_instance_override": {"restore": override_restore, "exact_value_restored": True},
        },
        "passed": True,
    }


def _hot_sync(public_fn: Callable[..., Any], x: object, cpu: torch.Tensor, gpu: torch.Tensor, counts: Mapping[str, int]) -> dict[str, object]:
    import torch
    sleep = getattr(torch.cuda, "_sleep", None)
    if not callable(sleep):
        raise RuntimeError("torch.cuda._sleep unavailable; cannot safely attest hot-return behavior")
    # Warm the exact C1 cache/public route before placing work ahead of it.
    _spy_public(public_fn, x, None, False, gpu, cpu, counts, True, "hot_sync/warm")
    torch.cuda.synchronize()
    cycles, return_threshold_s, sync_floor_s = 50_000_000, 0.25, 0.001
    sleep(cycles)
    start = time.perf_counter()
    _spy_public(public_fn, x, None, False, gpu, cpu, counts, True, "hot_sync/public")
    returned_s = time.perf_counter() - start
    sync_start = time.perf_counter()
    torch.cuda.synchronize()
    sync_s = time.perf_counter() - sync_start
    if returned_s > return_threshold_s or sync_s < sync_floor_s:
        raise AssertionError(f"hot sync behavior failed: return={returned_s}s sync={sync_s}s")
    return {"cycles": cycles, "public_return_wall_s": returned_s, "explicit_sync_wall_s": sync_s, "return_threshold_s": return_threshold_s, "sync_floor_s": sync_floor_s, "passed": True}


def _timing_percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty performance sample vector")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timing_summary(values: list[float]) -> dict[str, float | int]:
    if len(values) != PERFORMANCE_SAMPLES:
        raise ValueError(f"expected {PERFORMANCE_SAMPLES} CUDA-event samples, got {len(values)}")
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("public-registry CUDA-event samples must be finite positive milliseconds")
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": _timing_percentile(values, 0.50),
        "p95_ms": _timing_percentile(values, 0.95),
        "p99_ms": _timing_percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _cold_c1_observation(cell: Cell, x: object, cpu: torch.Tensor, gpu: torch.Tensor, public_fn: Callable[..., Any], counts: Mapping[str, int]) -> dict[str, object]:
    """A single un-gated cold cache observation, outside all timed release samples."""
    import torch

    final = cell.contract != "none"
    initial = _initial_state(cell.contract, cell.case.sequences)
    varlen_metadata.clear_cache()
    torch.cuda.synchronize()
    os.environ["C1_B300_FLASH_KDA"] = "1"
    before = dict(counts)
    started = time.perf_counter()
    state = None if initial is None else initial.clone()
    snapshot = _snapshot_input_tensors(x, gpu, cpu, state)
    with torch.inference_mode():
        output = _call(public_fn, x, state, final, gpu, cpu)
    returned_s = time.perf_counter() - started
    sync_started = time.perf_counter()
    torch.cuda.synchronize()
    sync_s = time.perf_counter() - sync_started
    after = dict(counts)
    delta = {key: after[key] - before[key] for key in before}
    decision = auto_dispatch.get_last_decision()
    if delta != {"c1": 1, "pinned": 0}:
        raise AssertionError(f"{cell.key}: cold observation route drift: {delta}")
    if decision.get("chosen_variant") != cell.expected_variant or decision.get("canonical_cache_hit") is not False:
        raise AssertionError(f"{cell.key}: cold C1 decision drift: {decision}")
    immutability = _assert_input_immutability(f"{cell.key}/cold", snapshot, x, gpu, cpu, state)
    return {
        "route_spy_delta": delta,
        "wall_return_s": returned_s,
        "explicit_sync_s": sync_s,
        "decision": decision,
        "final": _final_contract(output[1], cell.case.sequences, final, f"{cell.key}/cold"),
        "input_immutability": immutability,
        "not_part_of_performance_gate": True,
        "passed": True,
    }


def _public_performance_repeat(cell: Cell, x: object, cpu: torch.Tensor, gpu: torch.Tensor, public_fn: Callable[..., Any], counts: Mapping[str, int], repeat_index: int) -> dict[str, object]:
    """Cyclic, public-registry-only C1/pinned timing on one current CUDA stream."""
    import torch

    final = cell.contract != "none"
    initial = _initial_state(cell.contract, cell.case.sequences)
    repeat_snapshot = _snapshot_input_tensors(x, gpu, cpu, initial)

    def invoke() -> tuple[torch.Tensor, torch.Tensor | None]:
        # This is the stable, immutable FP32 state for the entire measured
        # path.  It mirrors the audited raw benchmark rather than inserting a
        # per-sample state clone outside the public FLA call.
        return _call(public_fn, x, initial, final, gpu, cpu)

    def select_path(path: str) -> None:
        if path == "public_registry_c1":
            os.environ["C1_B300_FLASH_KDA"] = "1"
        elif path == "public_registry_pinned":
            os.environ["C1_B300_FLASH_KDA"] = "0"
        else:
            raise ValueError(f"unknown public performance path {path!r}")

    paths = ("public_registry_c1", "public_registry_pinned")
    reversed_paths = tuple(reversed(paths))
    if PERFORMANCE_WARMUP % 2 or PERFORMANCE_SAMPLES % 2:
        raise RuntimeError("balanced public-registry schedule requires even warmup and sample counts")

    def path_order(index: int) -> tuple[str, str]:
        return paths if index % 2 == 0 else reversed_paths

    expected_deltas = {
        "public_registry_c1": {"c1": 1, "pinned": 0},
        "public_registry_pinned": {"c1": 0, "pinned": 1},
    }
    with torch.inference_mode():
        warm_before = dict(counts)
        first_warm_c1_decision: dict[str, object] | None = None
        for warm_index in range(PERFORMANCE_WARMUP):
            for path in path_order(warm_index):
                select_path(path)
                invoke()
                if path == "public_registry_c1" and warm_index == 0:
                    first_warm_c1_decision = auto_dispatch.get_last_decision()
        torch.cuda.synchronize()
        warm_after = dict(counts)

        raw_samples: dict[str, list[float]] = {path: [] for path in paths}
        timed_before = dict(counts)
        current_stream = torch.cuda.current_stream()
        for sample_index in range(PERFORMANCE_SAMPLES):
            for path in path_order(sample_index):
                # Selection is intentionally outside the CUDA event pair; the
                # complete *public* call, including registry/C1 preflight,
                # cache lookup and event recording, is between these events.
                select_path(path)
                sample_before = dict(counts)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(current_stream)
                # Do not allow a queued start marker to execute after Python
                # has already done preflight.  The sync is deliberately out
                # of the elapsed interval; once it returns, the start marker
                # is consumed and host public-dispatch delay is observable.
                start.synchronize()
                invoke()
                end.record(current_stream)
                # Synchronize this exact terminal event before elapsed_time;
                # synchronization itself is deliberately not a sample value.
                end.synchronize()
                sample_after = dict(counts)
                sample_delta = {key: sample_after[key] - sample_before[key] for key in sample_before}
                if sample_delta != expected_deltas[path]:
                    raise AssertionError(f"{cell.key}/repeat{repeat_index}/{path}: per-sample public route drift: {sample_delta}")
                raw_samples[path].append(float(start.elapsed_time(end)))
                del start, end
        timed_after = dict(counts)

    warm_delta = {key: warm_after[key] - warm_before[key] for key in warm_before}
    expected_warm_delta = {"c1": PERFORMANCE_WARMUP, "pinned": PERFORMANCE_WARMUP}
    if warm_delta != expected_warm_delta:
        raise AssertionError(f"{cell.key}/repeat{repeat_index}: warm public route drift: {warm_delta}")
    if (
        first_warm_c1_decision is None
        or first_warm_c1_decision.get("chosen_variant") != cell.expected_variant
        or first_warm_c1_decision.get("canonical_cache_hit") is not True
    ):
        raise AssertionError(f"{cell.key}/repeat{repeat_index}: C1 warm-cache decision drift: {first_warm_c1_decision}")
    timed_delta = {key: timed_after[key] - timed_before[key] for key in timed_before}
    expected_timed_delta = {"c1": PERFORMANCE_SAMPLES, "pinned": PERFORMANCE_SAMPLES}
    if timed_delta != expected_timed_delta:
        raise AssertionError(f"{cell.key}/repeat{repeat_index}: timed public route drift: {timed_delta}")
    repeat_immutability = _assert_input_immutability(
        f"{cell.key}/repeat{repeat_index}/performance", repeat_snapshot, x, gpu, cpu, initial
    )
    summaries = {path: _timing_summary(raw_samples[path]) for path in paths}
    winners: dict[str, str] = {}
    margins: dict[str, float | None] = {}
    rankings: dict[str, list[dict[str, object]]] = {}
    for percentile in shared.PERCENTILES:
        metric = f"{percentile}_ms"
        ordered = sorted((float(summaries[path][metric]), path) for path in paths)
        winner_latency, winner = ordered[0]
        runner_up_latency, runner_up = ordered[1]
        winners[percentile] = winner
        margins[percentile] = runner_up_latency / winner_latency - 1.0 if winner == "public_registry_c1" else None
        rankings[percentile] = [
            {"path": path, "latency_ms": latency} for latency, path in ordered
        ]
    winner_pass = all(winners[p] == "public_registry_c1" for p in shared.PERCENTILES)
    margin_pass = all(
        margins[p] is not None and float(margins[p]) >= PERFORMANCE_MIN_MARGIN
        for p in shared.PERCENTILES
    )
    return {
        "repeat_index": repeat_index,
        "event_contract": "current-stream start event -> immediate start.synchronize -> public FLA chunk_kda -> end event -> immediate end.synchronize -> elapsed_time; both synchronizations are excluded from the sample value",
        "schedule": "balanced alternating order: even index C1->pinned, odd index pinned->C1; C1_B300_FLASH_KDA is selected before each event pair and is outside timing",
        "path_order": {
            "even_sample": list(paths),
            "odd_sample": list(reversed_paths),
            "timed_first_path_counts": {"public_registry_c1": PERFORMANCE_SAMPLES // 2, "public_registry_pinned": PERFORMANCE_SAMPLES // 2},
            "warmup_first_path_counts": {"public_registry_c1": PERFORMANCE_WARMUP // 2, "public_registry_pinned": PERFORMANCE_WARMUP // 2},
        },
        "path_environment": {
            "public_registry_c1": {"C1_B300_FLASH_KDA": "1", "FLA_FLASH_KDA": "1"},
            "public_registry_pinned": {"C1_B300_FLASH_KDA": "0", "FLA_FLASH_KDA": "1"},
        },
        "warmup_route_spy_delta": warm_delta,
        "first_warm_c1_decision": first_warm_c1_decision,
        "timed_route_spy_delta": timed_delta,
        "expected_route_delta_per_call": expected_deltas,
        "per_sample_route_spy_assertions": {"public_registry_c1": PERFORMANCE_SAMPLES, "public_registry_pinned": PERFORMANCE_SAMPLES, "passed": True},
        "input_immutability_exact": True,
        "input_immutability_fields": repeat_immutability["fields"],
        "raw_samples_ms": raw_samples,
        "paths": summaries,
        "winner_by_percentile": winners,
        "c1_margin_over_pinned_by_percentile": margins,
        "ranked_paths_by_percentile": rankings,
        "winner_gate_pass": winner_pass,
        "margin_gate_pass": margin_pass,
        "repeat_gate_pass": winner_pass and margin_pass,
    }


def _performance_release(public_fn: Callable[..., Any], c1: object, counts: Mapping[str, int], result: dict[str, object], json_path: Path) -> None:
    """Measure every pre-registered exact cell; performance misses never short-circuit peers."""
    import torch

    if os.environ.get("C1_B300_VARLEN_CPU_DESCRIPTOR") != "1":
        raise RuntimeError("performance gate requires explicit CPU-authoritative packed-varlen opt-in")
    previous_c1 = os.environ.get("C1_B300_FLASH_KDA")
    previous_pinned = os.environ.get("FLA_FLASH_KDA")
    performance = result["performance"]
    if not isinstance(performance, dict):
        raise TypeError("performance result schema corrupted")
    shadow_gate = _assert_no_prepare_instance_shadow(c1, "before r6 performance")
    gates = result["gates"]
    if not isinstance(gates, dict):
        raise TypeError("r6 gate result schema corrupted")
    gates["prepare_spy_restored"] = shadow_gate
    _write(json_path, result)
    try:
        # Pinned remains eligible whenever the public registry is switched to
        # it; only the C1 opt-in value alternates outside event timing.
        os.environ["FLA_FLASH_KDA"] = "1"
        for cell_index, cell in enumerate(POSITIVE_CELLS):
            cell_record: dict[str, object] = {"expected_winner": "public_registry_c1", "repeats": []}
            for repeat_index in range(PERFORMANCE_REPEATS):
                seed = PERFORMANCE_SEED + cell_index * 1009 + repeat_index * 101
                x = shared._make_inputs(cell.case, seed)
                try:
                    cpu = _cpu_offsets(cell.case.lengths)
                    gpu = x.cu_seqlens
                    if gpu is None:
                        raise AssertionError(f"{cell.key}: packed performance case lost GPU offsets")
                    if repeat_index == 0:
                        cold = _cold_c1_observation(cell, x, cpu, gpu, public_fn, counts)
                        performance["cold_miss_observations"][cell.key] = cold  # type: ignore[index]
                    repeat = _public_performance_repeat(cell, x, cpu, gpu, public_fn, counts, repeat_index)
                    cell_record["repeats"].append(repeat)  # type: ignore[index]
                finally:
                    del x
                    torch.cuda.empty_cache()
            repeats = cell_record["repeats"]
            if not isinstance(repeats, list) or len(repeats) != PERFORMANCE_REPEATS:
                raise AssertionError(f"{cell.key}: incomplete performance repeats")
            if not all(bool(repeat.get("input_immutability_exact")) for repeat in repeats):
                raise AssertionError(f"{cell.key}: performance input immutability evidence is incomplete")
            cell_record["input_immutability_exact"] = True
            cell_record["input_immutability_fields"] = sorted(
                {field for repeat in repeats for field in repeat["input_immutability_fields"]}
            )
            released = all(bool(repeat.get("repeat_gate_pass")) for repeat in repeats)
            cell_record["performance_release"] = released
            performance["cells"][cell.key] = cell_record  # type: ignore[index]
            if released:
                performance["performance_release_cells"].append(cell.key)  # type: ignore[index]
                result["performance_release_cells"].append(cell.key)  # type: ignore[index]
            else:
                performance["failed_cells"].append(cell.key)  # type: ignore[index]
                result["performance_failed_cells"].append(cell.key)  # type: ignore[index]
            _write(json_path, result)
    finally:
        _restore_env("C1_B300_FLASH_KDA", previous_c1)
        _restore_env("FLA_FLASH_KDA", previous_pinned)
    performance["complete"] = True
    if result["performance_release_cells"] != performance["performance_release_cells"] or result["performance_failed_cells"] != performance["failed_cells"]:
        raise AssertionError("top-level performance release mapping drift")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--describe", action="store_true", help="write matrix without importing Torch/FLA")
    parser.add_argument("--cpu-construction-check", action="store_true", help="exercise CPU descriptor issuance without CUDA")
    args = parser.parse_args()
    if args.json.suffix.lower() != ".json":
        raise ValueError("--json output must use a .json suffix")
    expected_public_keys = {
        f"{case.name}/{contract}"
        for case in confirmation.CASES
        for contract in confirmation.FLA_PUBLIC_CONTRACTS
    }
    positive_keys = {cell.key for cell in POSITIVE_CELLS}
    fallback_keys = {cell.key for cell in POLICY_FALLBACK_CELLS}
    if len(POSITIVE_CELLS) != 2 or positive_keys != set(RELEASED_PUBLIC_VARIANTS):
        raise RuntimeError("positive varlen scope must be exactly the two released skew cells")
    if (
        len(POLICY_FALLBACK_CELLS) != 10
        or fallback_keys != set(POLICY_FALLBACK_MANIFEST)
        or positive_keys.intersection(fallback_keys)
        or positive_keys.union(fallback_keys) != expected_public_keys
        or set(POLICY_FALLBACK_C1_REASONS) != set(POLICY_FALLBACK_MANIFEST)
    ):
        raise RuntimeError("policy fallback scope must be exactly the ten non-released public cells")
    classifications = [entry["classification"] for entry in POLICY_FALLBACK_MANIFEST.values()]
    if (
        classifications.count("public_release_failed") != 8
        or classifications.count("raw_release_failed") != 1
        or classifications.count("record_only") != 1
    ):
        raise RuntimeError("r6 freeze must retain the inherited 8/1/1 policy-fallback classification split")
    production_map_gate = _production_public_map_gate()
    result = _initial_result(args)
    result["production_map_before_gpu"] = production_map_gate
    result["gates"]["production_map"] = production_map_gate  # type: ignore[index]
    if args.describe:
        result["describe_only"] = True
        _write(args.json, result)
        print(f"wrote packed-varlen FLA integration plan {args.json}")
        return
    if args.cpu_construction_check:
        result["cpu_construction_check"] = _cpu_construction_check()
        result["cpu_only"] = True
        _write(args.json, result)
        print(f"wrote packed-varlen FLA CPU construction check {args.json}")
        return
    if args.reference_root is None:
        raise ValueError("--reference-root is required")
    if os.environ.get(CLEAN_GPU_GATE_ENV) != "1" or os.environ.get("C1_B300_FLASH_KDA") != "1" or os.environ.get("C1_B300_VARLEN_CPU_DESCRIPTOR") != "1":
        raise RuntimeError("clean shell plus C1_B300_FLASH_KDA=1 and C1_B300_VARLEN_CPU_DESCRIPTOR=1 are required")
    patched_text, fla_text = os.environ.get("PATCHED_ROOT"), os.environ.get("FLA_ROOT")
    if not patched_text or not fla_text:
        raise RuntimeError("PATCHED_ROOT and FLA_ROOT are required")
    integration_runner_identity = _integration_runner_identity()
    result["identity"] = {"integration_runner": integration_runner_identity}
    _write(args.json, result)
    pre_torch_clean = _python_clean_gpu_gate()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shared.torch = torch
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import fla_backend
    shared.common = common
    result["identity"] = _identity(Path(patched_text), Path(fla_text), args.reference_root)
    result["identity"]["integration_runner"] = integration_runner_identity  # type: ignore[index]
    result["identity"]["runtime_import_identities"] = _runtime_dependency_identities(  # type: ignore[index]
        args, common, fla_backend
    )
    result["identity"]["python_pre_torch_nvidia_smi"] = pre_torch_clean  # type: ignore[index]
    result["gates"].update({"clean_gpu": {"passed": True}, "python_nvidia_clean": {"passed": True}, "device": {"passed": True}, "extension": {"passed": True}, "fla_pin": {"passed": True}})
    from fla.ops.kda import chunk_kda
    torch_ref, helper = confirmation._load_pinned_reference_without_build(common, args.reference_root)
    result["identity"]["pinned_reference_helper"] = helper  # type: ignore[index]
    c1, pinned, registry, snapshot = _registry_backends()
    originals, counts = _install_spies(c1, pinned)
    result["registry"] = {"snapshot": snapshot, "c1_id": id(c1), "pinned_id": id(pinned), "spies": "instance-local c1/pinned counters"}
    result["gates"]["registry_spy_identity"] = {"passed": True}  # type: ignore[index]
    try:
        # The pinned FLA backend deliberately rejects grad-enabled calls.  Keep
        # every verifier and public/direct execution path under one auditable
        # inference-mode scope, matching the production inference contract.
        with torch.inference_mode():
            if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
                raise RuntimeError("failed to enter the packed-varlen inference-mode audit scope")
            result["gates"]["inference_mode"] = {  # type: ignore[index]
                "scope": "single main-thread torch.inference_mode covers positive, policy fallback, cache/capture, hot-sync, fixed control, and performance",
                "grad_enabled": False,
                "inference_mode_enabled": True,
                "passed": True,
            }
            _write(args.json, result)
            for index, case in enumerate(confirmation.CASES):
                x = shared._make_inputs(case, args.seed + index * 1009)
                cpu = _cpu_offsets(case.lengths)
                gpu = x.cu_seqlens
                assert gpu is not None
                varlen_metadata.clear_cache()
                for cell in (candidate for candidate in POSITIVE_CELLS if candidate.case.name == case.name):
                    result["positive_results"][cell.key] = _positive_cell(cell, x, cpu, gpu, originals, counts, chunk_kda, c1, pinned, torch_ref, args.seed + index * 1009)  # type: ignore[index]
                    _write(args.json, result)
                for fallback in (candidate for candidate in POLICY_FALLBACK_CELLS if candidate.case.name == case.name):
                    result["negative_results"][fallback.key] = _negative_public(
                        fallback.key,
                        x,
                        fallback.contract,
                        cpu,
                        gpu,
                        c1,
                        pinned,
                        chunk_kda,
                        counts,
                        POLICY_FALLBACK_C1_REASONS[fallback.key],
                        require_exact_fallback=True,
                        originals=originals,
                        torch_ref=torch_ref,
                    )  # type: ignore[index]
                    _write(args.json, result)
                del x
                torch.cuda.empty_cache()
            # Every general negative starts from the released skew/none cell.
            # Otherwise a production policy miss could hide the preflight rejection
            # the test claims to isolate.
            skew = confirmation.CASES[3]
            x = shared._make_inputs(skew, args.seed + 99991)
            cpu, gpu = _cpu_offsets(skew.lengths), x.cu_seqlens
            assert gpu is not None
            split_cpu = _cpu_offsets((18, 510, 1024, 1300, 2049, 7387))
            result["negative_results"]["same_n_total_different_split"] = _negative_public("same_n_total_different_split", x, "none", split_cpu, gpu, c1, pinned, chunk_kda, counts, GENERAL_NEGATIVE_C1_REASONS["same_n_total_different_split"])  # type: ignore[index]
            _write(args.json, result)
            previous = os.environ.pop("C1_B300_VARLEN_CPU_DESCRIPTOR")
            try:
                result["negative_results"]["varlen_env_unset"] = _negative_public("varlen_env_unset", x, "none", cpu, gpu, c1, pinned, chunk_kda, counts, GENERAL_NEGATIVE_C1_REASONS["varlen_env_unset"])  # type: ignore[index]
                _write(args.json, result)
            finally:
                os.environ["C1_B300_VARLEN_CPU_DESCRIPTOR"] = previous
            result["negative_results"]["cpu_missing"] = _negative_public("cpu_missing", x, "none", None, gpu, c1, pinned, chunk_kda, counts, GENERAL_NEGATIVE_C1_REASONS["cpu_missing"])  # type: ignore[index]
            _write(args.json, result)
            malformed_cpu = torch.tensor([0, skew.total_tokens, skew.total_tokens], dtype=torch.int64, device="cpu")
            result["negative_results"]["cpu_malformed"] = _negative_public("cpu_malformed", x, "none", malformed_cpu, gpu, c1, pinned, chunk_kda, counts, GENERAL_NEGATIVE_C1_REASONS["cpu_malformed"])  # type: ignore[index]
            _write(args.json, result)
            result["negative_results"]["gpu_structural_mismatch_preflight"] = _gpu_structural_mismatch_fallback(x, cpu, gpu, c1, pinned, counts)  # type: ignore[index]
            _write(args.json, result)

            # The CPU-canonical, guard, cache/capture, and hot-sync probes
            # intentionally retain this released skew C1 route.
            result["cache_observations"]["cpu_canonical_gpu_values_ignored"] = _cpu_canonical_gpu_ignored(  # type: ignore[index]
                x, skew, cpu, gpu, originals, counts, chunk_kda, c1, torch_ref, args.seed + 99993
            )
            _write(args.json, result)
            neg_eigval = _allow_neg_eigval_fallback(x, cpu, gpu, c1, pinned, chunk_kda, counts)  # type: ignore[index]
            result["negative_results"]["allow_neg_eigval_semantic_fallback"] = neg_eigval
            _write(args.json, result)
            if not bool(neg_eigval["passed"]):
                raise RuntimeError(f"allow_neg_eigval semantic gate failed: {neg_eigval.get('failure')}")
            cache_observations = result["cache_observations"]
            if not isinstance(cache_observations, dict):
                raise TypeError("cache observation result schema corrupted")
            cache_observations["concurrency_and_capture"] = _cache_concurrency_and_capture(skew, x, cpu, gpu, counts)
            _write(args.json, result)
            cache_observations["hot_sync"] = _hot_sync(chunk_kda, x, cpu, gpu, counts)
            _write(args.json, result)
            result["negative_results"]["fixed_representative"] = _fixed_representative(originals, counts, chunk_kda, c1, pinned, args.seed + 199_997)  # type: ignore[index]
            _write(args.json, result)
            del x
            torch.cuda.empty_cache()
            _performance_release(chunk_kda, c1, counts, result, args.json)
    finally:
        c1.chunk_kda, pinned.chunk_kda = originals["c1"], originals["pinned"]
    result["complete"] = True
    _write(args.json, result)
    print(f"wrote {args.json}; positive={len(result['positive_results'])}, negative={len(result['negative_results'])}")


if __name__ == "__main__":
    main()
