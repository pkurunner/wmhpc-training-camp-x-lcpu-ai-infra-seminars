#!/usr/bin/env python3
"""Fail-closed B300 production-freeze runner for the released T=8191 route.

This runner *never* installs or monkey-patches a dispatch route.  Timed calls
are the real pinned ``fla.ops.kda.chunk_kda`` public API, with the existing C1
production backend selected through its normal registry entry.  It is useful
only after the production source has added its deliberately narrow T=8191
whitelist; a source miss is an error, not a test-only substitute.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import importlib.util
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DIM = 128
BATCH = 1
HEADS = 12
TOKENS = 8191
CONTRACTS = ("none", "fp32_final_only")
PATHS = ("pinned_public", "c1_production_public")
PERCENTILES = ("p50", "p95", "p99")
SAMPLES, REPEATS, WARMUP, MIN_MARGIN = 1000, 2, 100, 0.02
CLEAN_ENV = "C1_TAIL8191_PRODUCTION_FREEZE_CLEAN_GPU"
SCHEMA_VERSION = 4
REFERENCE_HELPER_PATH_ENV = "C1_PINNED_REFERENCE_HELPER_PATH"
REFERENCE_HELPER_SHA_ENV = "C1_PINNED_REFERENCE_HELPER_SHA256"
EXPECTED_EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_AUTO_DISPATCH_SHA256 = "9633179c6a5e8ce61b1857a102f81074c6bdfdc63163e13980f40c24d6583c29"
EXPECTED_FLA_BACKEND_SHA256 = "152ac4ba897c1887963553d592456b7f988c89105e6f6cb507a44757058d5fd1"
EXPECTED_FLASH_KDA_PYTHON_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
EXPECTED_HARNESS_SHA256 = "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"
EXPECTED_REFERENCE_TORCH_REF_SHA256 = "bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5"
EXPECTED_PINNED_REFERENCE_HELPER_PATH = "/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so"
EXPECTED_PINNED_REFERENCE_HELPER_SHA256 = "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
PINNED_REFERENCE_HELPER_LOAD_CONTRACT = "direct cached binary; exactly one pinned load_inline('sigmoid_ext') intercepted"
EXPECTED_PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_REFERENCE_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
EXPECTED_FLA_SOURCE_SHA256 = {
    "fla/__init__.py": "b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d",
    "fla/ops/backends/__init__.py": "a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635",
    "fla/ops/kda/__init__.py": "24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb",
    "fla/ops/kda/backends/__init__.py": "86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797",
    "fla/ops/kda/backends/flash_kda.py": "0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2",
    "fla/ops/kda/chunk.py": "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8",
}
EXPECTED_PATCHED_TRACKED_SHA256 = {
    "csrc/flash_kda.cpp": "38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4",
    "csrc/fwd.h": "613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083",
    "csrc/smxx/fwd_launch.cu": "a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928",
}
EXPECTED_PATCHED_TRACKED_STATUS = [f" M {relative}" for relative in EXPECTED_PATCHED_TRACKED_SHA256]
EXPECTED_REASON = {
    "none": "fixed_single_batch_b1_h12_t8191_none_whitelist_hit",
    "fp32_final_only": "fixed_single_batch_b1_h12_t8191_fp32_final_only_whitelist_hit",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: Mapping[str, object]) -> str:
    """Type-preserving canonical digest for a JSON evidence object."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _commit(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()


def _tracked_status(path: Path) -> list[str]:
    return subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=no"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lo, hi = int(point), min(int(point) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def _summary(values: list[float]) -> dict[str, float | int]:
    if len(values) != SAMPLES or not all(math.isfinite(value) and value > 0 for value in values):
        raise AssertionError("CUDA-event sample count/value contract failed")
    return {"samples": len(values), "mean_ms": statistics.fmean(values), "p50_ms": _percentile(values, .5), "p95_ms": _percentile(values, .95), "p99_ms": _percentile(values, .99)}


def _make_inputs(seed: int, *, batch: int = BATCH, tokens: int = TOKENS, heads: int = HEADS) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch, tokens, heads, DIM)
    q = functional.normalize(torch.randn(shape, device="cuda", generator=generator), p=2, dim=-1).to(torch.bfloat16).contiguous()
    k = functional.normalize(torch.randn(shape, device="cuda", generator=generator), p=2, dim=-1).to(torch.bfloat16).contiguous()
    return {"q": q, "k": k, "v": torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator).contiguous(), "g": torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator).contiguous(), "beta": torch.randn((batch, tokens, heads), dtype=torch.bfloat16, device="cuda", generator=generator).contiguous(), "A_log": torch.rand(heads, dtype=torch.float32, device="cuda", generator=generator), "dt_bias": torch.rand((heads, DIM), dtype=torch.float32, device="cuda", generator=generator).contiguous(), "scale": 1.0 / math.sqrt(DIM), "lower_bound": -5.0}


def _states(contract: str) -> tuple[Any | None, Any | None]:
    import torch
    if contract == "none":
        return None, None
    if contract == "fp32_final_only":
        return None, torch.zeros((BATCH, HEADS, DIM, DIM), dtype=torch.float32, device="cuda")
    raise ValueError(contract)


def _snapshot_inputs(x: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value.clone() if hasattr(value, "clone") else value for name, value in x.items()}


def _immutability(before: Mapping[str, Any], after: Mapping[str, Any], initial_before: Any | None, initial_after: Any | None, label: str) -> dict[str, object]:
    import torch
    fields = {name: bool(torch.equal(value, after[name])) if hasattr(value, "shape") else value == after[name] for name, value in before.items()}
    if not all(fields.values()):
        raise AssertionError(f"{label}: public/raw call mutated input")
    if initial_before is not None and not torch.equal(initial_before, initial_after):
        raise AssertionError(f"{label}: call mutated initial state")
    if initial_before is None and initial_after is not None:
        raise AssertionError(f"{label}: absent initial state changed")
    return {"input_immutability_exact": True, "input_immutability_fields": fields, "initial_state_immutability_exact": True}


def _invoke_raw(fn: Callable[..., Any], x: Mapping[str, Any], initial: Any | None, final: Any | None) -> tuple[Any, Any | None]:
    import torch
    out = torch.empty_like(x["v"])
    fn(x["q"], x["k"], x["v"], x["g"], x["beta"], x["scale"], out, A_log=x["A_log"], dt_bias=x["dt_bias"], lower_bound=x["lower_bound"], initial_state=initial, final_state=final, cu_seqlens=None)
    torch.cuda.synchronize()
    return out, final


def _exact(label: str, left: tuple[Any, Any | None], right: tuple[Any, Any | None]) -> dict[str, object]:
    import torch
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common
    common.require_exact(f"{label}/output", left[0], right[0])
    evidence: dict[str, object] = {"output_exact": True, "output_max_abs": float(common.max_abs(left[0], right[0]))}
    if left[1] is None or right[1] is None:
        if left[1] is not None or right[1] is not None:
            raise AssertionError(f"{label}: final-state presence mismatch")
        evidence["final_state_present"] = False
    else:
        if left[1].dtype != torch.float32 or tuple(left[1].shape) != (BATCH, HEADS, DIM, DIM) or not left[1].is_contiguous():
            raise AssertionError(f"{label}: final-state ABI drift")
        common.require_exact(f"{label}/final", left[1], right[1])
        evidence.update({"final_state_present": True, "final_state_exact": True, "final_state_max_abs": float(common.max_abs(left[1], right[1]))})
    return evidence


def _device_identity() -> dict[str, object]:
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device required")
    properties = torch.cuda.get_device_properties(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if "B300" not in properties.name.upper() or capability != (10, 3) or properties.multi_processor_count != 148:
        raise RuntimeError(f"B300-only gate failed: {properties.name}, {capability}, {properties.multi_processor_count}")
    return {"name": properties.name, "capability": list(capability), "multiprocessor_count": properties.multi_processor_count, "gate_pass": True}


def _gpu_uuid() -> str:
    values = [line.strip() for line in subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], check=True, text=True, capture_output=True).stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise RuntimeError(f"exactly one visible GPU UUID required, got {values}")
    return values[0]


def _ledger_entry(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    actual = _sha(resolved)
    if actual != expected_sha256:
        raise RuntimeError(f"{label} SHA256 gate failed: {actual}")
    return {"path": str(resolved), "sha256": actual, "gate_pass": True}


def _pinned_reference_helper_identity() -> dict[str, object]:
    """Fail closed on the ABI-matched prebuilt Torch-reference helper."""

    raw_path = os.environ.get(REFERENCE_HELPER_PATH_ENV)
    if type(raw_path) is not str or raw_path != EXPECTED_PINNED_REFERENCE_HELPER_PATH:
        raise RuntimeError(f"{REFERENCE_HELPER_PATH_ENV} must name the pinned helper path exactly")
    expected_sha = os.environ.get(REFERENCE_HELPER_SHA_ENV)
    if type(expected_sha) is not str or expected_sha != EXPECTED_PINNED_REFERENCE_HELPER_SHA256:
        raise RuntimeError(f"{REFERENCE_HELPER_SHA_ENV} must match the pinned helper SHA256 exactly")
    helper = Path(raw_path).resolve(strict=True)
    if not helper.is_file() or str(helper).replace("\\", "/") != EXPECTED_PINNED_REFERENCE_HELPER_PATH:
        raise RuntimeError("pinned reference helper canonical path drift")
    identity = _ledger_entry(helper, EXPECTED_PINNED_REFERENCE_HELPER_SHA256, "pinned reference helper")
    return {"path": identity["path"], "sha256": identity["sha256"]}


def _validate_pinned_reference_helper_load(
    helper_identity: Mapping[str, object], helper_load_proof: Mapping[str, object]
) -> None:
    """Require the precise no-build interception proof in runtime evidence."""

    if set(helper_identity) != {"path", "sha256"}:
        raise RuntimeError("pinned reference helper identity schema drift")
    if (
        helper_identity.get("path") != EXPECTED_PINNED_REFERENCE_HELPER_PATH
        or helper_identity.get("sha256") != EXPECTED_PINNED_REFERENCE_HELPER_SHA256
        or set(helper_load_proof) != {"path", "sha256", "load_contract", "intercepted_names", "no_build"}
        or helper_load_proof.get("path") != helper_identity["path"]
        or helper_load_proof.get("sha256") != helper_identity["sha256"]
        or helper_load_proof.get("load_contract") != PINNED_REFERENCE_HELPER_LOAD_CONTRACT
        or helper_load_proof.get("intercepted_names") != ["sigmoid_ext"]
        or helper_load_proof.get("no_build") is not True
    ):
        raise RuntimeError("pinned reference helper no-build load proof drift")


def _load_pinned_reference_without_build(
    common: object,
    reference_root: Path,
    expected_helper_identity: Mapping[str, object],
) -> tuple[Callable[..., Any], dict[str, object]]:
    """Load the fixed helper and intercept exactly one upstream ``load_inline``.

    ``tests/torch_ref.py`` requests ``sigmoid_ext`` using Torch's inline
    builder.  This replacement returns the previously audited shared object
    for that exact request and rejects every other request (including a
    second one).  It never delegates to the builder, and restores both the
    builder function and the ``sys.modules`` slot in ``finally`` blocks.
    """

    import torch.utils.cpp_extension as cpp_extension

    helper_identity = _pinned_reference_helper_identity()
    if dict(expected_helper_identity) != helper_identity:
        raise RuntimeError("pinned reference helper protocol identity drift before import")
    helper_path = Path(str(helper_identity["path"])).resolve(strict=True)
    helper_spec = importlib.util.spec_from_file_location("sigmoid_ext", helper_path)
    if helper_spec is None or helper_spec.loader is None:
        raise RuntimeError(f"cannot import pinned reference helper {helper_path}")
    helper_module = importlib.util.module_from_spec(helper_spec)
    sentinel = object()
    prior_module = sys.modules.get("sigmoid_ext", sentinel)
    intercepted_names: list[str] = []
    torch_ref: Callable[..., Any] | None = None

    def cached_load_inline(*args: object, **kwargs: object) -> object:
        name = kwargs.get("name", args[0] if args else None)
        if name != "sigmoid_ext":
            raise RuntimeError(f"unexpected load_inline request from pinned reference: {name!r}")
        if intercepted_names:
            raise RuntimeError("pinned reference requested sigmoid_ext more than once")
        intercepted_names.append("sigmoid_ext")
        return helper_module

    try:
        sys.modules["sigmoid_ext"] = helper_module
        helper_spec.loader.exec_module(helper_module)
        original_load_inline = cpp_extension.load_inline
        try:
            cpp_extension.load_inline = cached_load_inline
            loader = getattr(common, "_load_torch_ref", None)
            if not callable(loader):
                raise RuntimeError("pinned harness has no callable _load_torch_ref")
            candidate = loader(reference_root)
            if not callable(candidate):
                raise RuntimeError("pinned torch-reference loader returned a non-callable")
            torch_ref = candidate
        finally:
            cpp_extension.load_inline = original_load_inline
    finally:
        if prior_module is sentinel:
            sys.modules.pop("sigmoid_ext", None)
        else:
            sys.modules["sigmoid_ext"] = prior_module
    if intercepted_names != ["sigmoid_ext"] or torch_ref is None:
        raise RuntimeError(f"expected exactly one intercepted sigmoid_ext request, got {intercepted_names!r}")
    proof: dict[str, object] = {
        "path": str(helper_path),
        "sha256": helper_identity["sha256"],
        "load_contract": PINNED_REFERENCE_HELPER_LOAD_CONTRACT,
        "intercepted_names": intercepted_names,
        "no_build": True,
    }
    _validate_pinned_reference_helper_load(helper_identity, proof)
    return torch_ref, proof


def _identity(
    args: argparse.Namespace,
    helper_identity: Mapping[str, object],
    helper_load_proof: Mapping[str, object],
) -> dict[str, object]:
    import flash_kda
    import flash_kda_C
    import fla
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch, fla_backend
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common

    runner = Path(__file__).resolve(strict=True)
    analyzer = Path(args.analyzer_path).resolve(strict=True)
    protocol_shell = Path(args.protocol_shell_path).resolve(strict=True)
    expected_shell = Path(__file__).with_name("run_clean_tail8191_production_freeze.sh").resolve(strict=True)
    if protocol_shell != expected_shell:
        raise RuntimeError("protocol shell path gate failed")
    if (_commit(args.patched_root), _commit(args.reference_root), _commit(args.fla_root)) != (EXPECTED_PATCHED_COMMIT, EXPECTED_REFERENCE_COMMIT, EXPECTED_FLA_COMMIT):
        raise RuntimeError("pinned worktree commit gate failed")
    patched_status = _tracked_status(args.patched_root)
    if patched_status != EXPECTED_PATCHED_TRACKED_STATUS or _tracked_status(args.reference_root) or _tracked_status(args.fla_root):
        raise RuntimeError(f"tracked worktree status gate failed: patched={patched_status}")
    ext_path = Path(flash_kda_C.__file__).resolve(strict=True)
    patched_root = args.patched_root.resolve(strict=True)
    fla_root = args.fla_root.resolve(strict=True)
    if not ext_path.is_relative_to(patched_root):
        raise RuntimeError("extension identity gate failed")
    if not all(callable(getattr(flash_kda_C, name, None)) for name in ("fwd", "fwd_vshard4_p2", "get_workspace_size")):
        raise RuntimeError("extension symbol gate failed")
    auto_path = REPO_ROOT / "assignment02/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py"
    backend_path = REPO_ROOT / "assignment02/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py"
    if Path(auto_dispatch.__file__).resolve(strict=True) != auto_path.resolve(strict=True) or Path(fla_backend.__file__).resolve(strict=True) != backend_path.resolve(strict=True):
        raise RuntimeError("loaded production source is not the audited file")
    flash_python = Path(flash_kda.__file__).resolve(strict=True)
    if flash_python != (patched_root / "flash_kda/__init__.py").resolve(strict=True):
        raise RuntimeError("loaded flash_kda Python wrapper path gate failed")
    if Path(fla.__file__).resolve(strict=True) != (fla_root / "fla/__init__.py").resolve(strict=True):
        raise RuntimeError("loaded FLA package path gate failed")
    harness = Path(common.__file__).resolve(strict=True)
    expected_harness = (REPO_ROOT / "assignment02/team/c1_flashkda/harness/validate_and_bench.py").resolve(strict=True)
    if harness != expected_harness:
        raise RuntimeError("loaded validation harness path gate failed")
    reference = (args.reference_root / "tests/torch_ref.py").resolve(strict=True)
    if dict(helper_identity) != _pinned_reference_helper_identity():
        raise RuntimeError("pinned reference helper identity changed after no-build load")
    _validate_pinned_reference_helper_load(helper_identity, helper_load_proof)
    helper_path = Path(str(helper_identity["path"])).resolve(strict=True)
    ledger: dict[str, dict[str, object]] = {
        "protocol_shell": _ledger_entry(protocol_shell, args.expected_protocol_shell_sha256, "protocol shell"),
        "runner": _ledger_entry(runner, args.expected_runner_sha256, "runner"),
        "analyzer": _ledger_entry(analyzer, args.expected_analyzer_sha256, "analyzer"),
        "auto_dispatch": _ledger_entry(auto_path, EXPECTED_AUTO_DISPATCH_SHA256, "auto_dispatch"),
        "fla_backend": _ledger_entry(backend_path, EXPECTED_FLA_BACKEND_SHA256, "fla_backend"),
        "harness": _ledger_entry(harness, EXPECTED_HARNESS_SHA256, "validation harness"),
        "extension": _ledger_entry(ext_path, EXPECTED_EXTENSION_SHA256, "extension"),
        "flash_kda_python": _ledger_entry(flash_python, EXPECTED_FLASH_KDA_PYTHON_SHA256, "flash_kda Python wrapper"),
        "reference_torch_ref": _ledger_entry(reference, EXPECTED_REFERENCE_TORCH_REF_SHA256, "reference torch_ref"),
        "pinned_reference_helper": _ledger_entry(helper_path, EXPECTED_PINNED_REFERENCE_HELPER_SHA256, "pinned reference helper"),
    }
    for relative, digest in EXPECTED_FLA_SOURCE_SHA256.items():
        ledger[f"fla:{relative}"] = _ledger_entry(fla_root / relative, digest, f"FLA {relative}")
    for relative, digest in EXPECTED_PATCHED_TRACKED_SHA256.items():
        ledger[f"patched:{relative}"] = _ledger_entry(patched_root / relative, digest, f"patched {relative}")
    return {
        "roots": {
            "repo": str(REPO_ROOT.resolve(strict=True)),
            "patched": str(patched_root),
            "reference": str(args.reference_root.resolve(strict=True)),
            "fla": str(fla_root),
        },
        "source_ledger": ledger,
        "patched_tracked_status": patched_status,
        "device": _device_identity(),
        "gpu_uuid": _gpu_uuid(),
        "pinned_reference_helper_load": dict(helper_load_proof),
        "commits": {
            "patched": EXPECTED_PATCHED_COMMIT,
            "reference": EXPECTED_REFERENCE_COMMIT,
            "fla": EXPECTED_FLA_COMMIT,
        },
    }


def _raw_correctness(x: Mapping[str, Any], contract: str, torch_ref: Callable[..., Any]) -> dict[str, object]:
    import flash_kda
    import flash_kda_C
    snapshot = _snapshot_inputs(x)
    initial, final = _states(contract); initial_before = None if initial is None else initial.clone()
    reference = _invoke_raw(torch_ref, x, initial, final)
    ref_immutable = _immutability(snapshot, x, initial_before, initial, f"raw/{contract}/reference")
    initial, final = _states(contract); initial_before = None if initial is None else initial.clone()
    baseline = _invoke_raw(flash_kda.fwd, x, initial, final)
    base_immutable = _immutability(snapshot, x, initial_before, initial, f"raw/{contract}/baseline")
    def v4(*call_args: Any, **kwargs: Any) -> None:
        import torch
        out = call_args[6]
        workspace = torch.empty(flash_kda_C.get_workspace_size(BATCH * TOKENS, HEADS, BATCH), dtype=torch.uint8, device=out.device)
        flash_kda_C.fwd_vshard4_p2(*call_args[:6], out, workspace, kwargs["A_log"], kwargs["dt_bias"], kwargs["lower_bound"], initial_state=kwargs.get("initial_state"), final_state=kwargs.get("final_state"), cu_seqlens=None)
    initial, final = _states(contract); initial_before = None if initial is None else initial.clone()
    candidate = _invoke_raw(v4, x, initial, final)
    v4_immutable = _immutability(snapshot, x, initial_before, initial, f"raw/{contract}/vshard4")
    return {"baseline_vs_pinned_torch_reference": _exact(f"raw/{contract}/baseline_reference", baseline, reference), "vshard4_vs_pinned_torch_reference": _exact(f"raw/{contract}/v4_reference", candidate, reference), "vshard4_vs_baseline": _exact(f"raw/{contract}/v4_baseline", candidate, baseline), "immutability": {"reference": ref_immutable, "baseline": base_immutable, "vshard4": v4_immutable}, "passed": True}


def _public_kwargs(x: Mapping[str, Any], contract: str) -> dict[str, object]:
    return {"scale": x["scale"], "initial_state": None, "output_final_state": contract != "none", "use_qk_l2norm_in_kernel": True, "use_gate_in_kernel": True, "use_beta_sigmoid_in_kernel": True, "allow_neg_eigval": False, "state_v_first": True, "cu_seqlens": None, "cu_seqlens_cpu": None, "safe_gate": True, "lower_bound": x["lower_bound"], "disable_recompute": False, "return_intermediate_states": False, "cp_context": None, "A_log": x["A_log"], "dt_bias": x["dt_bias"]}


@contextmanager
def _c1_enabled(enabled: bool) -> Iterator[None]:
    prior = os.environ.get("C1_B300_FLASH_KDA"); os.environ["C1_B300_FLASH_KDA"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if prior is None: os.environ.pop("C1_B300_FLASH_KDA", None)
        else: os.environ["C1_B300_FLASH_KDA"] = prior


def _registry_setup() -> tuple[Callable[..., Any], dict[str, int], dict[str, int], dict[str, bool], Callable[[], None], Callable[[], None]]:
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import fla_backend
    from fla.ops.kda.backends import kda_registry
    from fla.ops.kda import chunk_kda
    custom = fla_backend.register_backend(); ordered = kda_registry._get_sorted_backends()
    c1 = [backend for backend in ordered if getattr(backend, "backend_type", None) == "c1_b300_flash_kda"]
    pinned = [backend for backend in ordered if getattr(backend, "backend_type", None) == "flash_kda"]
    if c1 != [custom] or len(pinned) != 1 or ordered.index(custom) >= ordered.index(pinned[0]):
        raise RuntimeError("C1/pinned registry order gate failed")
    c1_count, pinned_count = {"calls": 0}, {"calls": 0}; original_c1, original_pinned = custom.chunk_kda, pinned[0].chunk_kda
    missing = object()
    original_c1_slot = vars(custom).get("chunk_kda", missing)
    original_pinned_slot = vars(pinned[0]).get("chunk_kda", missing)
    def c1_spy(*call_args: object, **kwargs: object) -> object: c1_count["calls"] += 1; return original_c1(*call_args, **kwargs)
    def pinned_spy(*call_args: object, **kwargs: object) -> object: pinned_count["calls"] += 1; return original_pinned(*call_args, **kwargs)
    restore_state = {"restored": True, "c1_instance_slot_restored": True, "pinned_instance_slot_restored": True}
    def slot_matches(obj: object, original_slot: object) -> bool:
        current = vars(obj).get("chunk_kda", missing)
        return current is original_slot if original_slot is not missing else current is missing
    def install() -> None:
        if restore_state["restored"] is not True or not slot_matches(custom, original_c1_slot) or not slot_matches(pinned[0], original_pinned_slot):
            raise RuntimeError("registry spies are already installed")
        if custom.chunk_kda != original_c1 or pinned[0].chunk_kda != original_pinned:
            raise RuntimeError("registry methods drifted before spy installation")
        custom.chunk_kda, pinned[0].chunk_kda = c1_spy, pinned_spy
        restore_state.update({"restored": False, "c1_instance_slot_restored": False, "pinned_instance_slot_restored": False})
    def restore() -> None:
        if original_c1_slot is missing:
            vars(custom).pop("chunk_kda", None)
        else:
            custom.chunk_kda = original_c1_slot
        if original_pinned_slot is missing:
            vars(pinned[0]).pop("chunk_kda", None)
        else:
            pinned[0].chunk_kda = original_pinned_slot
        c1_slot_restored = slot_matches(custom, original_c1_slot)
        pinned_slot_restored = slot_matches(pinned[0], original_pinned_slot)
        if not c1_slot_restored or not pinned_slot_restored or custom.chunk_kda != original_c1 or pinned[0].chunk_kda != original_pinned:
            raise RuntimeError("registry spy restoration failed")
        restore_state.update({"restored": True, "c1_instance_slot_restored": c1_slot_restored, "pinned_instance_slot_restored": pinned_slot_restored})
    return chunk_kda, c1_count, pinned_count, restore_state, install, restore


def _exact_c1_decision(decision: Mapping[str, object], contract: str) -> bool:
    return (
        set(decision) == {
            "requested_variant", "chosen_variant", "reason", "extension_sha256",
            "varlen_cpu_authoritative", "certified_varlen_offsets", "canonical_cache_hit",
        }
        and decision.get("requested_variant") == "vshard4_p2"
        and decision.get("chosen_variant") == "vshard4_p2"
        and decision.get("reason") == EXPECTED_REASON[contract]
        and decision.get("extension_sha256") == EXPECTED_EXTENSION_SHA256
        and decision.get("varlen_cpu_authoritative") is False
        and decision.get("certified_varlen_offsets") is None
        and decision.get("canonical_cache_hit") is None
        and "test_only_route" not in decision
    )


def _public_call(public_fn: Callable[..., Any], x: Mapping[str, Any], contract: str, c1: bool, c1_count: Mapping[str, int], pinned_count: Mapping[str, int], dispatch: Any, *, synchronize: bool = True) -> tuple[tuple[Any, Any | None], dict[str, object]]:
    import torch
    before_c1, before_pinned = int(c1_count["calls"]), int(pinned_count["calls"])
    with _c1_enabled(c1), torch.inference_mode(): result = public_fn(x["q"], x["k"], x["v"], x["g"], x["beta"], **_public_kwargs(x, contract))
    if synchronize: torch.cuda.synchronize()
    c1_delta, pinned_delta = int(c1_count["calls"]) - before_c1, int(pinned_count["calls"]) - before_pinned
    if c1:
        decision = dispatch.get_last_decision()
        if c1_delta != 1 or pinned_delta != 0 or not _exact_c1_decision(decision, contract):
            raise AssertionError(f"production C1 route proof failed: {c1_delta}, {pinned_delta}, {decision}")
        return result, {"c1_spy_delta": c1_delta, "pinned_spy_delta": pinned_delta, "decision": decision, "passed": True}
    if c1_delta != 0 or pinned_delta != 1: raise AssertionError(f"pinned route proof failed: {c1_delta}, {pinned_delta}")
    return result, {"c1_spy_delta": c1_delta, "pinned_spy_delta": pinned_delta, "passed": True}


def _uninstrumented_public_call(public_fn: Callable[..., Any], x: Mapping[str, Any], contract: str, c1: bool, dispatch: Any, *, synchronize: bool) -> tuple[Any, Any | None]:
    import torch
    with _c1_enabled(c1), torch.inference_mode():
        result = public_fn(x["q"], x["k"], x["v"], x["g"], x["beta"], **_public_kwargs(x, contract))
    if synchronize:
        torch.cuda.synchronize()
    if c1 and not _exact_c1_decision(dispatch.get_last_decision(), contract):
        raise AssertionError("uninstrumented production route decision drift")
    return result


def _timed_public_call(
    public_fn: Callable[..., Any],
    x: Mapping[str, Any],
    contract: str,
    c1: bool,
    dispatch: Any,
    start: Any,
    end: Any,
    stream: Any,
) -> tuple[float, dict[str, object]]:
    """Time exactly one real public call; all Python audit work is outside events."""
    import torch

    q, k, v, g, beta = x["q"], x["k"], x["v"], x["g"], x["beta"]
    kwargs = _public_kwargs(x, contract)
    with _c1_enabled(c1), torch.inference_mode():
        start.record(stream)
        start.synchronize()
        result = public_fn(q, k, v, g, beta, **kwargs)
        end.record(stream)
    end.synchronize()
    del result
    if c1:
        decision = dispatch.get_last_decision()
        if not _exact_c1_decision(decision, contract):
            raise AssertionError(f"timed production C1 route proof failed: {decision}")
        proof = {"registry_spy_present": False, "decision": decision, "passed": True}
    else:
        proof = {"registry_spy_present": False, "c1_disabled": True, "pinned_route_prechecked": True, "passed": True}
    return float(start.elapsed_time(end)), proof


def _selector_controls() -> dict[str, object]:
    import torch
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch
    cases = {"h11": (1, 8191, 11), "t8190": (1, 8190, 12), "b2": (2, 8191, 12)}; result: dict[str, object] = {}
    for index, (name, shape) in enumerate(cases.items()):
        x = _make_inputs(551900 + index, batch=shape[0], tokens=shape[1], heads=shape[2]); out = torch.empty_like(x["v"])
        contracts: dict[str, object] = {}
        for contract in CONTRACTS:
            final = None if contract == "none" else torch.zeros((shape[0], shape[2], DIM, DIM), dtype=torch.float32, device="cuda")
            d = auto_dispatch.select_variant(auto_dispatch._read_device_metadata(x["q"]), x["q"], x["k"], x["v"], x["g"], x["beta"], out, x["A_log"], x["dt_bias"], x["scale"], x["lower_bound"], None, final, None, None)
            if d.requested_variant != "baseline" or d.chosen_variant != "baseline": raise AssertionError(f"neighborhood {name}/{contract} broadened production policy: {d}")
            contracts[contract] = {"requested_variant": d.requested_variant, "chosen_variant": d.chosen_variant, "reason": d.reason, "passed": True}
        result[name] = {"shape": {"B": shape[0], "T": shape[1], "H": shape[2]}, "scale": x["scale"], "lower_bound": x["lower_bound"], "contracts": contracts, "passed": True}
    return {"selector_neighborhoods": result, "passed": True}


def _negative_controls() -> dict[str, object]:
    import torch
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch
    x = _make_inputs(5518191); out = torch.empty_like(x["v"]); meta = auto_dispatch._read_device_metadata(x["q"]); records: dict[str, object] = {}
    for contract, dtype in (("fp32_both", torch.float32), ("bf16_both", torch.bfloat16)):
        state = torch.zeros((BATCH, HEADS, DIM, DIM), dtype=dtype, device="cuda")
        d = auto_dispatch.select_variant(meta, x["q"], x["k"], x["v"], x["g"], x["beta"], out, x["A_log"], x["dt_bias"], x["scale"], x["lower_bound"], state, torch.zeros_like(state), None, None)
        if d.requested_variant != "baseline" or d.chosen_variant != "baseline": raise AssertionError(f"{contract} must remain baseline: {d}")
        records[contract] = {"requested_variant": d.requested_variant, "chosen_variant": d.chosen_variant, "reason": d.reason, "scale": x["scale"], "lower_bound": x["lower_bound"], "passed": True}
    return {"negative_contracts": records, "neighborhoods": _selector_controls(), "passed": True}


def _repeat(process_index: int, repeat_index: int, x: Mapping[str, Any], contract: str, public_fn: Callable[..., Any], c1_count: Mapping[str, int], pinned_count: Mapping[str, int], dispatch: Any, restore_state: Mapping[str, bool], install_spies: Callable[[], None], restore_spies: Callable[[], None]) -> dict[str, object]:
    import torch
    snapshot = _snapshot_inputs(x)
    install_spies()
    try:
        pinned, pinned_proof = _public_call(public_fn, x, contract, False, c1_count, pinned_count, dispatch)
        c1, c1_proof = _public_call(public_fn, x, contract, True, c1_count, pinned_count, dispatch)
    finally:
        restore_spies()
    if restore_state["restored"] is not True:
        raise AssertionError("registry spies remained installed before warm-up/timing")
    exact = _exact(f"public/{process_index}/{repeat_index}/{contract}", c1, pinned)
    calls = {"pinned_public": lambda: _uninstrumented_public_call(public_fn, x, contract, False, dispatch, synchronize=False), "c1_production_public": lambda: _uninstrumented_public_call(public_fn, x, contract, True, dispatch, synchronize=False)}
    for warmup in range(WARMUP):
        for path in PATHS[warmup % 2:] + PATHS[:warmup % 2]: calls[path]()
    torch.cuda.synchronize()
    raw: dict[str, list[float]] = {path: [] for path in PATHS}; first = {path: 0 for path in PATHS}; verified = {path: 0 for path in PATHS}; stream = torch.cuda.current_stream()
    for sample in range(SAMPLES):
        for path in PATHS[sample % 2:] + PATHS[:sample % 2]:
            if path == (PATHS[sample % 2:] + PATHS[:sample % 2])[0]: first[path] += 1
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            elapsed, proof = _timed_public_call(public_fn, x, contract, path == "c1_production_public", dispatch, start, end, stream)
            if proof.get("passed") is not True:
                raise AssertionError(f"timed route proof failed for {path}")
            verified[path] += 1
            raw[path].append(elapsed)
    if first != {"pinned_public": 500, "c1_production_public": 500}: raise AssertionError(f"schedule drift: {first}")
    if verified != {"pinned_public": SAMPLES, "c1_production_public": SAMPLES}: raise AssertionError(f"timed route proof coverage drift: {verified}")
    immutability = _immutability(snapshot, x, None, None, f"performance/{contract}"); paths = {path: _summary(raw[path]) for path in PATHS}; margins = {p: float(paths["pinned_public"][f"{p}_ms"]) / float(paths["c1_production_public"][f"{p}_ms"]) - 1.0 for p in PERCENTILES}
    return {"process_index": process_index, "repeat_index": repeat_index, "event_contract": "unmodified registry/context/kwargs/events prepared; start.record+start.synchronize; exactly one real uninstrumented public chunk_kda call; immediate end.record; route checks and end.synchronize excluded", "schedule": "alternating two-path; 100 warmups, one post-warmup synchronize, 1000 samples per path", "first_path_counts": first, "timed_route_proof_counts": verified, "timed_call_audit_checks_outside_event": True, "timed_registry_spy_present": False, "public_precheck": {"pinned": pinned_proof, "c1_production": c1_proof, "exact": exact, "registry_spy_restored_before_timing": True}, **immutability, "raw_samples_ms": raw, "paths": paths, "c1_margin_over_pinned_by_percentile": margins, "repeat_gate_pass": all(value >= MIN_MARGIN for value in margins.values()), "passed": True}


def _describe(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "two-allocation real C1 production public-route T8191 freeze",
        "describe_only": True,
        "allocation_id": args.allocation,
        "process_index": args.process_index,
        "shape": {"B": 1, "H": 12, "T": 8191, "K": 128, "V": 128},
        "scale": 1.0 / math.sqrt(DIM),
        "lower_bound": -5.0,
        "contracts": list(CONTRACTS),
        "public_paths": list(PATHS),
        "fresh_pids_per_allocation": 2,
        "repeats_per_pid": REPEATS,
        "samples_per_path_repeat": SAMPLES,
        "required_percentiles": list(PERCENTILES),
        "minimum_c1_margin": MIN_MARGIN,
        "test_only_route_installed": False,
        "pinned_reference_helper": {
            "path": EXPECTED_PINNED_REFERENCE_HELPER_PATH,
            "sha256": EXPECTED_PINNED_REFERENCE_HELPER_SHA256,
            "load_contract": PINNED_REFERENCE_HELPER_LOAD_CONTRACT,
            "intercepted_names": ["sigmoid_ext"],
            "no_build": True,
        },
    }


def _self_test() -> None:
    """CPU-only fail-closed checks for the helper identity/proof schema."""

    helper = {"path": EXPECTED_PINNED_REFERENCE_HELPER_PATH, "sha256": EXPECTED_PINNED_REFERENCE_HELPER_SHA256}
    proof = {
        "path": EXPECTED_PINNED_REFERENCE_HELPER_PATH,
        "sha256": EXPECTED_PINNED_REFERENCE_HELPER_SHA256,
        "load_contract": PINNED_REFERENCE_HELPER_LOAD_CONTRACT,
        "intercepted_names": ["sigmoid_ext"],
        "no_build": True,
    }
    _validate_pinned_reference_helper_load(helper, proof)
    for field, forged in (("intercepted_names", ["sigmoid_ext", "sigmoid_ext"]), ("no_build", False)):
        candidate = dict(proof)
        candidate[field] = forged
        try:
            _validate_pinned_reference_helper_load(helper, candidate)
        except RuntimeError:
            continue
        raise AssertionError(f"forged pinned-reference helper {field} proof was accepted")
    print("runner self-test PASS: helper proof requires one intercepted sigmoid_ext request and no build")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", choices=("A1", "A2"))
    parser.add_argument("--process-index", type=int, choices=(0, 1))
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--patched-root", type=Path)
    parser.add_argument("--fla-root", type=Path)
    parser.add_argument("--analyzer-path", type=Path, default=Path(__file__).with_name("analyze_tail8191_production_freeze.py"))
    parser.add_argument("--protocol-shell-path", type=Path, default=Path(__file__).with_name("run_clean_tail8191_production_freeze.sh"))
    parser.add_argument("--expected-protocol-shell-sha256")
    parser.add_argument("--expected-runner-sha256")
    parser.add_argument("--expected-analyzer-sha256")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.describe or any(
            value is not None
            for value in (
                args.allocation,
                args.process_index,
                args.reference_root,
                args.patched_root,
                args.fla_root,
                args.expected_protocol_shell_sha256,
                args.expected_runner_sha256,
                args.expected_analyzer_sha256,
                args.json,
            )
        ):
            raise RuntimeError("--self-test cannot combine with runner modes or identity arguments")
        _self_test()
        return
    if any(
        value is None
        for value in (
            args.allocation,
            args.process_index,
            args.expected_protocol_shell_sha256,
            args.expected_runner_sha256,
            args.expected_analyzer_sha256,
            args.json,
        )
    ):
        raise RuntimeError("allocation/process/identity/JSON arguments are required")
    if args.describe:
        _write(args.json, _describe(args)); print(f"wrote production-freeze plan {args.json}"); return
    if os.environ.get(CLEAN_ENV) != "1" or not all((args.reference_root, args.patched_root, args.fla_root)): raise RuntimeError("refusing direct GPU run: use the clean shell")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not (job_id.isascii() and job_id.isdecimal() and job_id[:1] != "0" and int(job_id) > 0) or os.environ.get("FLA_FLASH_KDA") != "1":
        raise RuntimeError("positive canonical-decimal SLURM_JOB_ID and FLA_FLASH_KDA=1 required")
    helper_identity = _pinned_reference_helper_identity()
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch
    expected_harness = (REPO_ROOT / "assignment02/team/c1_flashkda/harness/validate_and_bench.py").resolve(strict=True)
    common_file = getattr(common, "__file__", None)
    if type(common_file) is not str or Path(common_file).resolve(strict=True) != expected_harness or _sha(expected_harness) != EXPECTED_HARNESS_SHA256:
        raise RuntimeError("loaded pinned harness path/SHA mismatch before torch-reference import")
    torch_ref, helper_load_proof = _load_pinned_reference_without_build(common, args.reference_root, helper_identity)
    identity = _identity(args, helper_identity, helper_load_proof)
    pid = os.getpid()
    result: dict[str, object] = {
        **_describe(args),
        "describe_only": False,
        "pid": pid,
        "slurm_job_id": job_id,
        "identity": identity,
        "artifact_content_identity": {
            "allocation_id": args.allocation,
            "process_index": args.process_index,
            "pid": pid,
            "slurm_job_id": job_id,
            "gpu_uuid": identity["gpu_uuid"],
            "identity_sha256": _canonical_sha(identity),
            "pinned_reference_helper_load": dict(helper_load_proof),
        },
        "raw_abi_correctness": {},
        "public_benchmarks": {},
        "negative_controls": _negative_controls(),
        "registry_spy_restored": False,
        "registry_spy_restore_proof": {"restored": False, "c1_instance_slot_restored": False, "pinned_instance_slot_restored": False},
        "complete": False,
    }
    _write(args.json, result)
    with _c1_enabled(True): public_fn, c1_count, pinned_count, restore_state, install_spies, restore = _registry_setup()
    try:
        for contract_index, contract in enumerate(CONTRACTS):
            x = _make_inputs(args.seed + args.process_index * 1_000_003 + contract_index * 10_007); result["raw_abi_correctness"][contract] = _raw_correctness(x, contract, torch_ref)  # type: ignore[index]
            repeats: list[dict[str, object]] = []; result["public_benchmarks"][contract] = repeats  # type: ignore[index]; _write(args.json, result)
            for repeat_index in range(REPEATS):
                repeat_x = _make_inputs(args.seed + args.process_index * 1_000_003 + contract_index * 10_007 + repeat_index * 101); repeats.append(_repeat(args.process_index, repeat_index, repeat_x, contract, public_fn, c1_count, pinned_count, auto_dispatch, restore_state, install_spies, restore)); _write(args.json, result)
    finally:
        restore()
    if restore_state != {"restored": True, "c1_instance_slot_restored": True, "pinned_instance_slot_restored": True}:
        raise RuntimeError("registry spy restoration proof missing")
    result["registry_spy_restored"] = True; result["registry_spy_restore_proof"] = dict(restore_state); result["complete"] = True; _write(args.json, result); print(f"wrote production-freeze artifact {args.json}")


if __name__ == "__main__": main()
