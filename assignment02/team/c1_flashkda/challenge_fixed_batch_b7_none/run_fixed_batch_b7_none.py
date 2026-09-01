#!/usr/bin/env python3
"""Fresh A1/A2 public-API release protocol for only B=7,H=12,T=2048,none.

The historic B=7 discovery artifact is deliberately not loaded or referenced.
This runner installs an exact, process-local test route to vshard2-P2, proves
the real FLA registry used that route, then restores the module attribute on
exit.  It never changes a production dispatcher or FLA source file.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BATCH, HEADS, TOKENS, DIM = 7, 12, 2048, 128
RAW_CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
PUBLIC_CONTRACT = "none"
PATHS = ("pinned_public", "c1_test_route_public")
PERCENTILES = ("p50", "p95", "p99")
SAMPLES, REPEATS, WARMUP, MIN_MARGIN = 1000, 2, 100, 0.02
SCHEMA_VERSION = 3
CLEAN_ENV = "C1_FIXED_BATCH_B7_NONE_CLEAN_GPU"
SHELL_PATH_ENV = "C1_FIXED_BATCH_B7_NONE_PROTOCOL_SHELL_PATH"
SHELL_SHA_ENV = "C1_FIXED_BATCH_B7_NONE_PROTOCOL_SHELL_SHA256"
REFERENCE_HELPER_PATH_ENV = "C1_PINNED_REFERENCE_HELPER_PATH"
REFERENCE_HELPER_SHA_ENV = "C1_PINNED_REFERENCE_HELPER_SHA256"
TEST_ROUTE_REASON = "test_only_b7_h12_t2048_none_exact_route"
NEGATIVE_REASON = "fixed_batch_shape_not_whitelisted"
TIMED_EVENT_CONTRACT = "CUDA current-stream: prepared environment/context/kwargs/counters/events and start.record+start.synchronize before interval; interval exactly one public chunk_kda -> end.record; host-only audit then end.synchronize"
TIMED_SCHEDULE = "two-path cyclic; 100 warmups/path; 1000 CUDA-event samples/path"
EXPECTED_SCALE = DIM ** -0.5
EXPECTED_LOWER_BOUND = -5.0
EXPECTED_EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_REFERENCE_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
EXPECTED_AUTO_DISPATCH_SHA256 = "9cdd460058254016af58723875bdf99ebe74f8e016a4c6027eb7fb38c8e9a88c"
EXPECTED_FLA_BACKEND_SHA256 = "206e448abcd3d64826f87a20e7d57c790fef6adacd91e26edcb10a3711b9b656"
EXPECTED_HARNESS_SHA256 = "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"
EXPECTED_FLASH_KDA_PYTHON_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
PINNED_REFERENCE_HELPER_PATH = "/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so"
PINNED_REFERENCE_HELPER_SHA256 = "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
PINNED_REFERENCE_HELPER_LOAD_CONTRACT = "direct cached binary; exactly one pinned load_inline('sigmoid_ext') intercepted"
FLA_FILES = {
    "fla/__init__.py": "b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d",
    "fla/ops/backends/__init__.py": "a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635",
    "fla/ops/kda/__init__.py": "24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb",
    "fla/ops/kda/backends/__init__.py": "86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797",
    "fla/ops/kda/backends/flash_kda.py": "0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2",
    "fla/ops/kda/chunk.py": "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8",
}
PATCHED_DIRTY_FILES = {
    "csrc/flash_kda.cpp": "38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4",
    "csrc/fwd.h": "613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083",
    "csrc/smxx/fwd_launch.cu": "a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928",
}
PATCHED_DIRTY_STATUS = " M"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_text(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"{label} must be a lowercase SHA256")
    return value


def canonical_sha(value: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": sha(resolved)}


def write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def commit(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()


def patched_dirty_identity(root: Path) -> dict[str, object]:
    """Accept only the preregistered tracked overlay atop the pinned commit."""

    resolved = root.resolve(strict=True)
    actual_lines = [line for line in subprocess.run(["git", "-C", str(resolved), "status", "--porcelain=v1", "--untracked-files=no"], check=True, text=True, capture_output=True).stdout.splitlines() if line]
    expected_lines = {f"{PATCHED_DIRTY_STATUS} {relative}" for relative in PATCHED_DIRTY_FILES}
    if len(actual_lines) != len(expected_lines) or set(actual_lines) != expected_lines:
        raise RuntimeError(f"patched tracked dirty set drift: {actual_lines!r}")
    files = {relative: file_identity(resolved / relative) for relative in PATCHED_DIRTY_FILES}
    if {relative: item["sha256"] for relative, item in files.items()} != PATCHED_DIRTY_FILES:
        raise RuntimeError("patched tracked dirty overlay SHA drift")
    return {
        "root": str(resolved),
        "git_status_porcelain_v1": {relative: PATCHED_DIRTY_STATUS for relative in PATCHED_DIRTY_FILES},
        "files": files,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty sample set")
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lo = int(point)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def summary(values: list[float]) -> dict[str, float | int]:
    if len(values) != SAMPLES or not all(math.isfinite(value) and value > 0.0 for value in values):
        raise AssertionError("sample count/value gate failed")
    return {"samples": len(values), "mean_ms": statistics.fmean(values), "p50_ms": percentile(values, .50), "p95_ms": percentile(values, .95), "p99_ms": percentile(values, .99), "min_ms": min(values), "max_ms": max(values)}


def make_inputs(seed: int, batch: int = BATCH) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch, TOKENS, HEADS, DIM)
    q = functional.normalize(torch.randn(shape, device="cuda", generator=generator), p=2, dim=-1).to(torch.bfloat16)
    k = functional.normalize(torch.randn(shape, device="cuda", generator=generator), p=2, dim=-1).to(torch.bfloat16)
    return {"q": q.contiguous(), "k": k.contiguous(), "v": torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator).contiguous(), "g": torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator).contiguous(), "beta": torch.randn((batch, TOKENS, HEADS), dtype=torch.bfloat16, device="cuda", generator=generator).contiguous(), "A_log": torch.rand(HEADS, dtype=torch.float32, device="cuda", generator=generator), "dt_bias": torch.rand((HEADS, DIM), dtype=torch.float32, device="cuda", generator=generator).contiguous(), "scale": EXPECTED_SCALE, "lower_bound": EXPECTED_LOWER_BOUND}


def states(contract: str, batch: int = BATCH) -> tuple[Any | None, Any | None]:
    import torch
    if contract == "none":
        return None, None
    dtype = torch.bfloat16 if contract == "bf16_both" else torch.float32
    final = torch.zeros((batch, HEADS, DIM, DIM), dtype=dtype, device="cuda")
    if contract == "fp32_final_only":
        return None, final
    generator = torch.Generator(device="cuda").manual_seed(40_000 + batch + (1 if contract == "bf16_both" else 2))
    return torch.randn(final.shape, dtype=dtype, device="cuda", generator=generator).contiguous(), final


def snapshot_inputs(x: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value.clone() if hasattr(value, "clone") else value for name, value in x.items()}


def immutable(snapshot: Mapping[str, Any], x: Mapping[str, Any], initial_before: Any | None, initial_after: Any | None, label: str) -> dict[str, object]:
    import torch
    fields = {name: (bool(torch.equal(before, x[name])) if hasattr(before, "shape") else before == x[name]) for name, before in snapshot.items()}
    if not all(fields.values()):
        raise AssertionError(f"{label}: input mutation {sorted(name for name, ok in fields.items() if not ok)}")
    if initial_before is None:
        initial_ok = initial_after is None
    else:
        initial_ok = bool(torch.equal(initial_before, initial_after))
    if not initial_ok:
        raise AssertionError(f"{label}: initial-state mutation")
    return {"input_immutability_exact": True, "input_immutability_fields": fields, "initial_state_immutability_exact": True}


def invoke_raw(fn: Callable[..., Any], x: Mapping[str, Any], initial: Any | None, final: Any | None) -> tuple[Any, Any | None]:
    import torch
    out = torch.empty_like(x["v"])
    fn(x["q"], x["k"], x["v"], x["g"], x["beta"], x["scale"], out, A_log=x["A_log"], dt_bias=x["dt_bias"], lower_bound=x["lower_bound"], initial_state=initial, final_state=final, cu_seqlens=None)
    torch.cuda.synchronize()
    return out, final


def exact(label: str, actual: tuple[Any, Any | None], expected: tuple[Any, Any | None]) -> dict[str, object]:
    import torch
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common
    common.require_exact(f"{label}/output", actual[0], expected[0])
    answer: dict[str, object] = {"output_exact": True, "output_max_abs": float(common.max_abs(actual[0], expected[0]))}
    if actual[1] is None or expected[1] is None:
        if actual[1] is not None or expected[1] is not None:
            raise AssertionError(f"{label}: final-state presence mismatch")
        answer["final_state_present"] = False
    else:
        if actual[1].dtype != expected[1].dtype or tuple(actual[1].shape) != (BATCH, HEADS, DIM, DIM) or not actual[1].is_contiguous():
            raise AssertionError(f"{label}: final-state ABI drift")
        common.require_exact(f"{label}/final_state", actual[1], expected[1])
        answer.update({"final_state_present": True, "final_state_exact": True, "final_state_max_abs": float(common.max_abs(actual[1], expected[1]))})
    return answer


def device_identity() -> dict[str, object]:
    import torch
    name, capability, sm_count = torch.cuda.get_device_name(0), tuple(torch.cuda.get_device_capability(0)), torch.cuda.get_device_properties(0).multi_processor_count
    if "B300" not in name.upper() or capability != (10, 3) or sm_count != 148:
        raise RuntimeError(f"B300-only protocol got name={name!r}, capability={capability}, SMs={sm_count}")
    return {"name": name, "capability": list(capability), "multiprocessor_count": sm_count, "gate_pass": True}


def gpu_uuid() -> str:
    values = [line.strip() for line in subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], check=True, text=True, capture_output=True).stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise RuntimeError(f"requires one visible GPU UUID, got {values!r}")
    return values[0]


def protocol_shell_identity(args: argparse.Namespace) -> dict[str, object]:
    expected = sha_text(args.expected_protocol_shell_sha256, "expected protocol-shell SHA")
    environment_sha = sha_text(os.environ.get(SHELL_SHA_ENV), f"{SHELL_SHA_ENV}")
    if environment_sha != expected:
        raise RuntimeError("external protocol-shell SHA does not match runner argument")
    raw_path = os.environ.get(SHELL_PATH_ENV)
    if type(raw_path) is not str or not raw_path:
        raise RuntimeError(f"{SHELL_PATH_ENV} is required")
    shell = Path(raw_path).resolve(strict=True)
    if not shell.is_file() or sha(shell) != expected:
        raise RuntimeError("external protocol-shell identity mismatch")
    return file_identity(shell)


def pinned_reference_helper_identity() -> dict[str, object]:
    """Fail closed on the prebuilt ABI-matched Torch-reference helper."""

    raw_path = os.environ.get(REFERENCE_HELPER_PATH_ENV)
    if type(raw_path) is not str or raw_path != PINNED_REFERENCE_HELPER_PATH:
        raise RuntimeError(f"{REFERENCE_HELPER_PATH_ENV} must name the pinned helper path exactly")
    expected_sha = sha_text(os.environ.get(REFERENCE_HELPER_SHA_ENV), REFERENCE_HELPER_SHA_ENV)
    if expected_sha != PINNED_REFERENCE_HELPER_SHA256:
        raise RuntimeError("pinned reference helper SHA environment drift")
    helper = Path(raw_path).resolve(strict=True)
    if not helper.is_file() or str(helper).replace("\\", "/") != PINNED_REFERENCE_HELPER_PATH:
        raise RuntimeError("pinned reference helper canonical path drift")
    identity = file_identity(helper)
    if identity["sha256"] != PINNED_REFERENCE_HELPER_SHA256:
        raise RuntimeError("pinned reference helper SHA mismatch")
    return identity


def load_pinned_reference_without_build(
    common: object,
    reference_root: Path,
    expected_helper_identity: Mapping[str, object],
) -> tuple[Callable[..., Any], dict[str, object]]:
    """Directly load the pinned helper and permit one reference ``load_inline``.

    The upstream reference imports ``sigmoid_ext`` through ``load_inline``.
    Replacing exactly that one request with the SHA-pinned shared object avoids
    any JIT compilation before raw correctness or timed work begins.  There
    is deliberately no fallback to Torch's builder.
    """

    import torch.utils.cpp_extension as cpp_extension

    helper_identity = pinned_reference_helper_identity()
    if helper_identity != dict(expected_helper_identity):
        raise RuntimeError("pinned reference helper protocol identity drift before import")
    helper_path = Path(str(helper_identity["path"])).resolve(strict=True)
    helper_spec = importlib.util.spec_from_file_location("sigmoid_ext", helper_path)
    if helper_spec is None or helper_spec.loader is None:
        raise RuntimeError(f"cannot import pinned reference helper {helper_path}")
    helper_module = importlib.util.module_from_spec(helper_spec)
    sentinel = object()
    prior_module = sys.modules.get("sigmoid_ext", sentinel)
    intercepted_names: list[str] = []

    def cached_load_inline(*args: object, **kwargs: object) -> object:
        name = kwargs.get("name", args[0] if args else None)
        if name != "sigmoid_ext":
            raise RuntimeError(f"unexpected load_inline request from pinned reference: {name!r}")
        intercepted_names.append("sigmoid_ext")
        return helper_module

    try:
        sys.modules["sigmoid_ext"] = helper_module
        helper_spec.loader.exec_module(helper_module)
        original_load_inline = cpp_extension.load_inline
        cpp_extension.load_inline = cached_load_inline
        try:
            loader = getattr(common, "_load_torch_ref", None)
            if not callable(loader):
                raise RuntimeError("pinned harness has no callable _load_torch_ref")
            torch_ref = loader(reference_root)
        finally:
            cpp_extension.load_inline = original_load_inline
    finally:
        if prior_module is sentinel:
            sys.modules.pop("sigmoid_ext", None)
        else:
            sys.modules["sigmoid_ext"] = prior_module
    if intercepted_names != ["sigmoid_ext"]:
        raise RuntimeError(f"expected one intercepted sigmoid_ext request, got {intercepted_names!r}")
    if not callable(torch_ref):
        raise RuntimeError("pinned torch-reference loader returned a non-callable")
    return torch_ref, {
        "path": str(helper_path),
        "sha256": helper_identity["sha256"],
        "load_contract": PINNED_REFERENCE_HELPER_LOAD_CONTRACT,
        "intercepted_names": intercepted_names,
        "no_build": True,
    }


def protocol_identity(args: argparse.Namespace) -> dict[str, object]:
    """Validate every non-runtime input and return a cross-allocation identity."""

    runner = Path(__file__).resolve(strict=True)
    analyzer = runner.with_name("analyze_fixed_batch_b7_none.py").resolve(strict=True)
    if sha(runner) != sha_text(args.expected_runner_sha256, "expected runner SHA"):
        raise RuntimeError("runner SHA mismatch")
    if sha(analyzer) != sha_text(args.expected_analyzer_sha256, "expected analyzer SHA"):
        raise RuntimeError("analyzer SHA mismatch")
    if (commit(args.patched_root), commit(args.reference_root), commit(args.fla_root)) != (EXPECTED_PATCHED_COMMIT, EXPECTED_REFERENCE_COMMIT, EXPECTED_FLA_COMMIT):
        raise RuntimeError("pinned commit mismatch")
    extension_candidates = [path.resolve(strict=True) for path in args.patched_root.glob("flash_kda_C.cpython-*-linux-gnu.so")]
    if len(extension_candidates) != 1 or sha(extension_candidates[0]) != EXPECTED_EXTENSION_SHA256:
        raise RuntimeError("audited extension path/SHA mismatch")
    extension = extension_candidates[0]
    flash_path = (args.patched_root / "flash_kda" / "__init__.py").resolve(strict=True)
    if sha(flash_path) != EXPECTED_FLASH_KDA_PYTHON_SHA256:
        raise RuntimeError("flash_kda Python identity mismatch")
    sources = {
        "auto_dispatch": REPO_ROOT / "assignment02/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py",
        "fla_backend": REPO_ROOT / "assignment02/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py",
        "harness": REPO_ROOT / "assignment02/team/c1_flashkda/harness/validate_and_bench.py",
    }
    expected_sources = {
        "auto_dispatch": EXPECTED_AUTO_DISPATCH_SHA256,
        "fla_backend": EXPECTED_FLA_BACKEND_SHA256,
        "harness": EXPECTED_HARNESS_SHA256,
    }
    actual_sources = {name: file_identity(path) for name, path in sources.items()}
    if {name: item["sha256"] for name, item in actual_sources.items()} != expected_sources:
        raise RuntimeError("C1 source SHA mismatch")
    fla_source_map = {relative: file_identity(args.fla_root / relative) for relative in FLA_FILES}
    if {relative: item["sha256"] for relative, item in fla_source_map.items()} != FLA_FILES:
        raise RuntimeError("pinned FLA source SHA mismatch")
    reference = (args.reference_root / "tests" / "torch_ref.py").resolve(strict=True)
    helper = pinned_reference_helper_identity()
    return {
        "runner": file_identity(runner),
        "analyzer": file_identity(analyzer),
        "protocol_shell": protocol_shell_identity(args),
        "extension": file_identity(extension),
        "flash_kda_python": file_identity(flash_path),
        "auto_dispatch": actual_sources["auto_dispatch"],
        "fla_backend": actual_sources["fla_backend"],
        "harness": actual_sources["harness"],
        "reference_torch_ref": file_identity(reference),
        "pinned_reference_helper": helper,
        "commits": {
            "patched": {"root": str(args.patched_root.resolve(strict=True)), "head": EXPECTED_PATCHED_COMMIT},
            "reference": {"root": str(args.reference_root.resolve(strict=True)), "head": EXPECTED_REFERENCE_COMMIT},
            "fla": {"root": str(args.fla_root.resolve(strict=True)), "head": EXPECTED_FLA_COMMIT},
        },
        "patched_dirty_overlay": patched_dirty_identity(args.patched_root),
        "fla_source_map": fla_source_map,
    }


def identity(protocol: Mapping[str, object], helper_load_proof: Mapping[str, object]) -> dict[str, object]:
    import flash_kda
    import flash_kda_C
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch, fla_backend
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common

    extension = Path(str(protocol["extension"]["path"])).resolve(strict=True)  # type: ignore[index]
    if Path(flash_kda_C.__file__).resolve(strict=True) != extension or not callable(getattr(flash_kda_C, "fwd_vshard_p2", None)) or not callable(getattr(flash_kda_C, "get_workspace_size", None)):
        raise RuntimeError("loaded extension symbol/path mismatch")
    flash_path = Path(str(protocol["flash_kda_python"]["path"])).resolve(strict=True)  # type: ignore[index]
    if Path(flash_kda.__file__).resolve(strict=True) != flash_path:
        raise RuntimeError("loaded flash_kda Python is not pinned")
    source_modules = {"auto_dispatch": auto_dispatch, "fla_backend": fla_backend, "harness": common}
    for name, module in source_modules.items():
        if Path(module.__file__).resolve(strict=True) != Path(str(protocol[name]["path"])).resolve(strict=True):  # type: ignore[index]
            raise RuntimeError(f"loaded {name} is not the pinned file")
    module_map = {
        "fla": "fla/__init__.py",
        "fla.ops.backends": "fla/ops/backends/__init__.py",
        "fla.ops.kda": "fla/ops/kda/__init__.py",
        "fla.ops.kda.backends": "fla/ops/kda/backends/__init__.py",
        "fla.ops.kda.backends.flash_kda": "fla/ops/kda/backends/flash_kda.py",
        "fla.ops.kda.chunk": "fla/ops/kda/chunk.py",
    }
    for module_name, relative in module_map.items():
        expected_path = Path(str(protocol["fla_source_map"][relative]["path"])).resolve(strict=True)  # type: ignore[index]
        if Path(importlib.import_module(module_name).__file__).resolve(strict=True) != expected_path:
            raise RuntimeError(f"loaded FLA {module_name} is outside FLA_ROOT")
    expected_helper = protocol.get("pinned_reference_helper")
    if not isinstance(expected_helper, Mapping):
        raise RuntimeError("pinned reference helper protocol schema drift")
    expected_path = expected_helper.get("path")
    expected_sha = expected_helper.get("sha256")
    if (
        helper_load_proof.get("path") != expected_path
        or helper_load_proof.get("sha256") != expected_sha
        or helper_load_proof.get("load_contract") != PINNED_REFERENCE_HELPER_LOAD_CONTRACT
        or helper_load_proof.get("intercepted_names") != ["sigmoid_ext"]
        or helper_load_proof.get("no_build") is not True
    ):
        raise RuntimeError("pinned reference helper no-build load proof drift")
    return {
        "protocol": dict(protocol),
        "runtime": {
            "device": device_identity(),
            "gpu_uuid": gpu_uuid(),
            "pinned_reference_helper_load": dict(helper_load_proof),
        },
    }


def route_contract(q: Any, k: Any, v: Any, g: Any, beta: Any, out: Any, A_log: Any, dt_bias: Any, initial: Any | None, final: Any | None, cu_seqlens: Any | None) -> None:
    import torch
    for tensor in (q, k, v, g, out):
        if tuple(tensor.shape) != (BATCH, TOKENS, HEADS, DIM) or tensor.dtype != torch.bfloat16 or not tensor.is_cuda or not tensor.is_contiguous():
            raise RuntimeError("test route rejects non-exact B7 tensor contract")
    same_device = all(tensor.device == q.device for tensor in (k, v, g, beta, out, A_log, dt_bias))
    if cu_seqlens is not None or tuple(beta.shape) != (BATCH, TOKENS, HEADS) or beta.dtype != torch.bfloat16 or not beta.is_cuda or not beta.is_contiguous() or tuple(A_log.shape) != (HEADS,) or A_log.dtype != torch.float32 or not A_log.is_cuda or not A_log.is_contiguous() or tuple(dt_bias.shape) != (HEADS, DIM) or dt_bias.dtype != torch.float32 or not dt_bias.is_cuda or not dt_bias.is_contiguous() or not same_device or initial is not None or final is not None:
        raise RuntimeError("test route permits only B7 none state contract")


@contextmanager
def install_test_route() -> Iterator[tuple[Any, dict[str, object]]]:
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch, fla_backend
    original = fla_backend.auto_dispatch.fwd
    if original is not auto_dispatch.fwd:
        raise RuntimeError("unexpected C1 dispatcher module identity")
    def route(q: Any, k: Any, v: Any, g: Any, beta: Any, scale: float, out: Any, A_log: Any, dt_bias: Any, lower_bound: float, initial_state: Any | None = None, final_state: Any | None = None, cu_seqlens: Any | None = None, cu_seqlens_cpu: Any | None = None, _varlen_descriptor: Any | None = None) -> Any:
        if cu_seqlens_cpu is not None or _varlen_descriptor is not None or type(scale) is not float or type(lower_bound) is not float or scale != EXPECTED_SCALE or lower_bound != EXPECTED_LOWER_BOUND:
            raise RuntimeError("test route rejects descriptor/scalar drift")
        route_contract(q, k, v, g, beta, out, A_log, dt_bias, initial_state, final_state, cu_seqlens)
        extension, symbols, digest, rejection = auto_dispatch._load_extension_and_symbols()
        if rejection is not None or extension is None or "fwd_vshard_p2" not in symbols or digest != EXPECTED_EXTENSION_SHA256:
            raise RuntimeError("test route refuses extension fallback")
        decision = auto_dispatch.DispatchDecision("vshard2_p2", "vshard2_p2", TEST_ROUTE_REASON)
        auto_dispatch._record(decision, extension_sha256=digest, test_only_route=True, production_source_mutated=False)
        return auto_dispatch._launch_sharded("vshard2_p2", extension, q=q, k=k, v=v, g=g, beta=beta, scale=scale, out=out, A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound, initial_state=None, final_state=None, cu_seqlens=None)
    restore_proof: dict[str, object] = {
        "test_route_dispatcher_restored": False,
        "dispatcher_identity_matches_production": False,
    }
    fla_backend.auto_dispatch.fwd = route
    try:
        yield auto_dispatch, restore_proof
    finally:
        fla_backend.auto_dispatch.fwd = original
        restored = fla_backend.auto_dispatch.fwd is original
        production_identity = original is auto_dispatch.fwd
        if not restored or not production_identity:
            raise RuntimeError("test-only dispatcher restore proof failed")
        restore_proof.update({
            "test_route_dispatcher_restored": True,
            "dispatcher_identity_matches_production": True,
        })


@contextmanager
def c1_enabled(value: bool) -> Iterator[None]:
    previous = os.environ.get("C1_B300_FLASH_KDA")
    os.environ["C1_B300_FLASH_KDA"] = "1" if value else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("C1_B300_FLASH_KDA", None)
        else:
            os.environ["C1_B300_FLASH_KDA"] = previous


def registry_setup() -> tuple[Callable[..., Any], dict[str, int], Callable[[], dict[str, object]]]:
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import fla_backend
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.backends import kda_registry
    custom = fla_backend.register_backend()
    ordered = kda_registry._get_sorted_backends()
    pinned = [backend for backend in ordered if getattr(backend, "backend_type", None) == "flash_kda"]
    if [backend for backend in ordered if getattr(backend, "backend_type", None) == "c1_b300_flash_kda"] != [custom] or len(pinned) != 1 or ordered.index(custom) >= ordered.index(pinned[0]):
        raise RuntimeError("FLA registry gate failed")
    counters = {"c1": 0, "pinned": 0}
    custom_original, pinned_original = custom.chunk_kda, pinned[0].chunk_kda
    def custom_spy(*args: object, **kwargs: object) -> object:
        counters["c1"] += 1
        return custom_original(*args, **kwargs)
    def pinned_spy(*args: object, **kwargs: object) -> object:
        counters["pinned"] += 1
        return pinned_original(*args, **kwargs)
    custom.chunk_kda, pinned[0].chunk_kda = custom_spy, pinned_spy

    def restore_spies() -> dict[str, object]:
        custom.chunk_kda, pinned[0].chunk_kda = custom_original, pinned_original
        proof = {
            "c1_backend_spy_restored": custom.chunk_kda is custom_original,
            "pinned_backend_spy_restored": pinned[0].chunk_kda is pinned_original,
        }
        if not all(proof.values()):
            raise RuntimeError("public backend spy restore proof failed")
        return proof

    return chunk_kda, counters, restore_spies


def public_kwargs(x: Mapping[str, Any]) -> dict[str, object]:
    return {"scale": x["scale"], "initial_state": None, "output_final_state": False, "use_qk_l2norm_in_kernel": True, "use_gate_in_kernel": True, "use_beta_sigmoid_in_kernel": True, "allow_neg_eigval": False, "state_v_first": True, "cu_seqlens": None, "cu_seqlens_cpu": None, "safe_gate": True, "lower_bound": x["lower_bound"], "disable_recompute": False, "return_intermediate_states": False, "cp_context": None, "A_log": x["A_log"], "dt_bias": x["dt_bias"]}


@contextmanager
def prepared_public_call(public: Callable[..., Any], x: Mapping[str, Any], enable: bool, counters: Mapping[str, int]) -> Iterator[tuple[Callable[[], tuple[Any, Any | None]], dict[str, object]]]:
    """Prepare all host state before a timed interval.

    The yielded callable performs exactly one real public `chunk_kda` call.
    Environment selection, inference context entry, kwargs construction, and
    route-counter snapshotting happen before the caller records its start
    event.  The callable itself contains no synchronization or audit work.
    """

    import torch

    kwargs = public_kwargs(x)
    observation = {"enable": enable, "before_c1": int(counters["c1"]), "before_pinned": int(counters["pinned"])}
    with c1_enabled(enable), torch.inference_mode():
        def call_once() -> tuple[Any, Any | None]:
            return public(x["q"], x["k"], x["v"], x["g"], x["beta"], **kwargs)

        yield call_once, observation


def prove_public_launch(observation: Mapping[str, object], counters: Mapping[str, int], dispatch: Any) -> dict[str, object]:
    """Verify route spies/decision after a public call without GPU sync."""

    enable = observation.get("enable")
    if type(enable) is not bool:
        raise AssertionError("public launch observation drift")
    before_c1, before_pinned = observation.get("before_c1"), observation.get("before_pinned")
    if type(before_c1) is not int or type(before_pinned) is not int:
        raise AssertionError("public launch counter observation drift")
    c1_delta, pinned_delta = int(counters["c1"]) - before_c1, int(counters["pinned"]) - before_pinned
    if enable:
        decision = dispatch.get_last_decision()
        expected = {
            "requested_variant": "vshard2_p2",
            "chosen_variant": "vshard2_p2",
            "reason": TEST_ROUTE_REASON,
            "extension_sha256": EXPECTED_EXTENSION_SHA256,
            "test_only_route": True,
            "production_source_mutated": False,
        }
        if c1_delta != 1 or pinned_delta != 0 or decision != expected:
            raise AssertionError(f"C1 public routing proof failed: {decision}")
        return {"c1_spy_delta": c1_delta, "pinned_spy_delta": pinned_delta, "decision": decision, "passed": True}
    if c1_delta != 0 or pinned_delta != 1:
        raise AssertionError("pinned public routing proof failed")
    return {"c1_spy_delta": c1_delta, "pinned_spy_delta": pinned_delta, "decision": None, "passed": True}


def public_call_precheck(public: Callable[..., Any], x: Mapping[str, Any], enable: bool, counters: Mapping[str, int], dispatch: Any) -> tuple[tuple[Any, Any | None], dict[str, object]]:
    """Synchronous correctness precheck; never use this function in timing."""

    import torch

    with prepared_public_call(public, x, enable, counters) as (call_once, observation):
        result = call_once()
        torch.cuda.synchronize()
    return result, prove_public_launch(observation, counters, dispatch)


def raw_correctness(x: Mapping[str, Any], contract: str, reference_fn: Callable[..., Any]) -> dict[str, object]:
    import flash_kda
    import flash_kda_C
    snapshot = snapshot_inputs(x)
    def one(label: str, fn: Callable[..., Any]) -> tuple[tuple[Any, Any | None], dict[str, object]]:
        initial, final = states(contract)
        initial_before = None if initial is None else initial.clone()
        result = invoke_raw(fn, x, initial, final)
        return result, immutable(snapshot, x, initial_before, initial, f"raw/{contract}/{label}")
    def v2(*args: Any, **kwargs: Any) -> None:
        import torch
        workspace = torch.empty(flash_kda_C.get_workspace_size(BATCH * TOKENS, HEADS, BATCH), dtype=torch.uint8, device=args[6].device)
        flash_kda_C.fwd_vshard_p2(*args[:6], args[6], workspace, kwargs["A_log"], kwargs["dt_bias"], kwargs["lower_bound"], initial_state=kwargs.get("initial_state"), final_state=kwargs.get("final_state"), cu_seqlens=None)
    reference, reference_immutable = one("torch_reference", reference_fn)
    baseline, baseline_immutable = one("baseline", flash_kda.fwd)
    candidate, candidate_immutable = one("vshard2_p2", v2)
    return {"baseline_vs_pinned_torch_reference": exact(f"raw/{contract}/baseline_vs_reference", baseline, reference), "vshard2_vs_pinned_torch_reference": exact(f"raw/{contract}/v2_vs_reference", candidate, reference), "vshard2_vs_baseline": exact(f"raw/{contract}/v2_vs_baseline", candidate, baseline), "immutability": {"reference": reference_immutable, "baseline": baseline_immutable, "vshard2_p2": candidate_immutable}, "passed": True}


def negative_controls() -> dict[str, object]:
    import torch
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch
    records: dict[str, object] = {}
    for batch in (7, 8):
        x = make_inputs(700_000 + batch, batch)
        out = torch.empty_like(x["v"])
        metadata = auto_dispatch._read_device_metadata(x["q"])
        for contract in RAW_CONTRACTS:
            initial, final = states(contract, batch)
            decision = auto_dispatch.select_variant(metadata, x["q"], x["k"], x["v"], x["g"], x["beta"], out, x["A_log"], x["dt_bias"], x["scale"], x["lower_bound"], initial, final, None, None)
            if decision.requested_variant != "baseline" or decision.chosen_variant != "baseline" or decision.reason != NEGATIVE_REASON:
                raise AssertionError(f"production selector must keep B{batch}/{contract} baseline: {decision}")
            records[f"b{batch}/{contract}"] = {"requested_variant": decision.requested_variant, "chosen_variant": decision.chosen_variant, "reason": decision.reason, "passed": True}
        del x, out
        torch.cuda.empty_cache()
    return {"production_source_unmodified": True, "controls": records, "passed": True}


def benchmark_repeat(process_index: int, repeat_index: int, x: Mapping[str, Any], public: Callable[..., Any], counters: Mapping[str, int], dispatch: Any) -> dict[str, object]:
    import torch
    snapshot = snapshot_inputs(x)
    pinned, pinned_proof = public_call_precheck(public, x, False, counters, dispatch)
    candidate, candidate_proof = public_call_precheck(public, x, True, counters, dispatch)
    precheck = exact(f"public/{process_index}/{repeat_index}/v2_vs_pinned", candidate, pinned)
    enabled = {"pinned_public": False, "c1_test_route_public": True}
    warmup_counts = {path: 0 for path in PATHS}
    for warmup_index in range(WARMUP):
        order = PATHS[warmup_index % 2:] + PATHS[:warmup_index % 2]
        for path in order:
            with prepared_public_call(public, x, enabled[path], counters) as (call_once, observation):
                _result = call_once()
            prove_public_launch(observation, counters, dispatch)
            warmup_counts[path] += 1
    torch.cuda.synchronize()
    if warmup_counts != {"pinned_public": WARMUP, "c1_test_route_public": WARMUP}:
        raise AssertionError(f"warm-up path count drift: {warmup_counts}")
    raw, first = {path: [] for path in PATHS}, {path: 0 for path in PATHS}
    timed_counts = {path: 0 for path in PATHS}
    timed_checks: dict[str, dict[str, int]] = {
        "pinned_public": {"calls": 0, "c1_spy_delta_total": 0, "pinned_spy_delta_total": 0, "decision_checks": 0},
        "c1_test_route_public": {"calls": 0, "c1_spy_delta_total": 0, "pinned_spy_delta_total": 0, "decision_checks": 0},
    }
    stream = torch.cuda.current_stream()
    for sample_index in range(SAMPLES):
        order = PATHS[sample_index % 2:] + PATHS[:sample_index % 2]
        for path in order:
            first[path] += int(path == order[0])
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # All host setup precedes start; after it, only one ready public
            # call and its end event are issued before the audit/synchronize.
            with prepared_public_call(public, x, enabled[path], counters) as (call_once, observation):
                start.record(stream)
                start.synchronize()
                _result = call_once()
                end.record(stream)
            proof = prove_public_launch(observation, counters, dispatch)
            end.synchronize()
            raw[path].append(float(start.elapsed_time(end)))
            timed_counts[path] += 1
            check = timed_checks[path]
            check["calls"] += 1
            check["c1_spy_delta_total"] += int(proof["c1_spy_delta"])
            check["pinned_spy_delta_total"] += int(proof["pinned_spy_delta"])
            check["decision_checks"] += int(path == "c1_test_route_public")
    if first != {"pinned_public": 500, "c1_test_route_public": 500}:
        raise AssertionError(f"cyclic ordering drift: {first}")
    if timed_counts != {"pinned_public": SAMPLES, "c1_test_route_public": SAMPLES}:
        raise AssertionError(f"timed path count drift: {timed_counts}")
    expected_checks = {
        "pinned_public": {"calls": SAMPLES, "c1_spy_delta_total": 0, "pinned_spy_delta_total": SAMPLES, "decision_checks": 0},
        "c1_test_route_public": {"calls": SAMPLES, "c1_spy_delta_total": SAMPLES, "pinned_spy_delta_total": 0, "decision_checks": SAMPLES},
    }
    if timed_checks != expected_checks:
        raise AssertionError(f"timed host-route proof drift: {timed_checks}")
    paths = {path: summary(raw[path]) for path in PATHS}
    margins = {name: float(paths["pinned_public"][f"{name}_ms"]) / float(paths["c1_test_route_public"][f"{name}_ms"]) - 1.0 for name in PERCENTILES}
    return {"process_index": process_index, "repeat_index": repeat_index, "event_contract": TIMED_EVENT_CONTRACT, "schedule": TIMED_SCHEDULE, "first_path_counts": first, "warmup_public_call_counts": warmup_counts, "timed_public_call_counts": timed_counts, "timed_route_checks": timed_checks, "timed_route_checks_without_gpu_sync": True, "public_precheck": {"pinned": pinned_proof, "c1_test_route": candidate_proof, "exact": precheck}, **immutable(snapshot, x, None, None, f"performance/{process_index}/{repeat_index}"), "raw_samples_ms": raw, "paths": paths, "c1_margin_over_pinned_by_percentile": margins, "winner_by_percentile": {name: "c1_test_route_public" if margins[name] > 0 else "pinned_public" for name in PERCENTILES}, "repeat_gate_pass": all(value >= MIN_MARGIN for value in margins.values()), "passed": True}


def descriptor(args: argparse.Namespace) -> dict[str, object]:
    return {
        "shape": {"B": BATCH, "H": HEADS, "T": TOKENS, "K": DIM, "V": DIM},
        "public_contract": PUBLIC_CONTRACT,
        "raw_abi_contracts": list(RAW_CONTRACTS),
    }


def describe(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "fresh_b7_none_vshard2_p2_plan",
        "purpose": "fresh B7 none vshard2 public-route A1/A2 protocol; historic discovery is excluded",
        "describe_only": True,
        "allocation_id": args.allocation,
        "process_index": args.process_index,
        **descriptor(args),
        "test_route": "exact B7,H12,T2048,none -> vshard2_p2 only",
        "negative_controls": "unmodified production selector: all B7 and B8 state contracts must remain pre-launch baseline",
        "fresh_pids_per_allocation": 2,
        "repeats_per_pid": REPEATS,
        "samples_per_path_repeat": SAMPLES,
        "required_percentiles": list(PERCENTILES),
        "minimum_margin": MIN_MARGIN,
        "timed_event_contract": TIMED_EVENT_CONTRACT,
        "pinned_reference_helper": {
            "path": PINNED_REFERENCE_HELPER_PATH,
            "sha256": PINNED_REFERENCE_HELPER_SHA256,
            "load_contract": PINNED_REFERENCE_HELPER_LOAD_CONTRACT,
        },
        "freeze_gate": "both clean allocations must pass in separately hashed cross-allocation chain",
        "protocol_identity": protocol_identity(args),
    }


def self_test() -> None:
    """CPU-only adversarial check: no missing helper identity can fall back to JIT."""

    original = {name: os.environ.get(name) for name in (REFERENCE_HELPER_PATH_ENV, REFERENCE_HELPER_SHA_ENV)}
    try:
        for missing in (REFERENCE_HELPER_PATH_ENV, REFERENCE_HELPER_SHA_ENV):
            os.environ[REFERENCE_HELPER_PATH_ENV] = PINNED_REFERENCE_HELPER_PATH
            os.environ[REFERENCE_HELPER_SHA_ENV] = PINNED_REFERENCE_HELPER_SHA256
            os.environ.pop(missing, None)
            try:
                pinned_reference_helper_identity()
            except RuntimeError:
                continue
            raise AssertionError(f"missing {missing} was accepted by helper identity gate")
    finally:
        for name, prior in original.items():
            if prior is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prior
    print("runner self-test PASS (pinned reference helper missing-environment gates reject before JIT)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", choices=("A1", "A2"))
    parser.add_argument("--process-index", type=int, choices=(0, 1))
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--patched-root", type=Path)
    parser.add_argument("--fla-root", type=Path)
    parser.add_argument("--expected-runner-sha256")
    parser.add_argument("--expected-analyzer-sha256")
    parser.add_argument("--expected-protocol-shell-sha256")
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
                args.expected_runner_sha256,
                args.expected_analyzer_sha256,
                args.expected_protocol_shell_sha256,
                args.json,
            )
        ):
            raise RuntimeError("--self-test cannot combine with runner modes or identity arguments")
        self_test()
        return
    if any(
        value is None
        for value in (
            args.allocation,
            args.process_index,
            args.reference_root,
            args.patched_root,
            args.fla_root,
            args.expected_runner_sha256,
            args.expected_analyzer_sha256,
            args.expected_protocol_shell_sha256,
            args.json,
        )
    ):
        raise RuntimeError("allocation/process/root/identity/JSON arguments are required")
    if args.describe:
        write(args.json, describe(args)); print(f"wrote B7-none plan {args.json}"); return
    if os.environ.get(CLEAN_ENV) != "1":
        raise RuntimeError("refusing direct GPU run; use clean audit shell")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not (job_id.isascii() and job_id.isdecimal() and job_id[:1] != "0" and int(job_id) > 0) or not os.environ.get("FLA_FLASH_KDA"):
        raise RuntimeError("positive canonical-decimal SLURM_JOB_ID and FLA_FLASH_KDA are required")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common
    protocol = protocol_identity(args)
    common_file = getattr(common, "__file__", None)
    expected_harness = Path(str(protocol["harness"]["path"])).resolve(strict=True)  # type: ignore[index]
    if type(common_file) is not str or Path(common_file).resolve(strict=True) != expected_harness:
        raise RuntimeError("loaded pinned harness path mismatch before torch-reference import")
    reference_fn, helper_load_proof = load_pinned_reference_without_build(
        common,
        args.reference_root,
        protocol["pinned_reference_helper"],  # type: ignore[arg-type]
    )
    identity_record = identity(protocol, helper_load_proof)
    protocol = identity_record["protocol"]
    runtime = identity_record["runtime"]
    if not isinstance(protocol, Mapping) or not isinstance(runtime, Mapping):
        raise RuntimeError("identity schema drift")
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "fresh_b7_none_vshard2_p2_main",
        "allocation_id": args.allocation,
        "process_index": args.process_index,
        "pid": os.getpid(),
        "slurm_job_id": job_id,
        **descriptor(args),
        "identity": identity_record,
        "artifact_content_identity": {
            "allocation_id": args.allocation,
            "process_index": args.process_index,
            "protocol_identity_sha256": canonical_sha(protocol),
            "runtime_identity": runtime,
        },
        "raw_abi_correctness": {},
        "negative_controls": negative_controls(),
        "public_benchmarks": [],
        "post_restore_proof": {},
        "complete": False,
    }
    write(args.json, result)
    route_restore_proof: dict[str, object] = {}
    spy_restore_proof: dict[str, object] = {}
    with install_test_route() as (dispatch, route_restore_proof):
        restore: Callable[[], dict[str, object]] | None = None
        try:
            with c1_enabled(True):
                public, counters, restore = registry_setup()
            for contract_index, contract in enumerate(RAW_CONTRACTS):
                x = make_inputs(args.seed + args.process_index * 1_000_003 + contract_index * 10_007)
                result["raw_abi_correctness"][contract] = raw_correctness(x, contract, reference_fn)  # type: ignore[index]
                del x; torch.cuda.empty_cache(); write(args.json, result)
            for repeat_index in range(REPEATS):
                x = make_inputs(args.seed + args.process_index * 1_000_003 + repeat_index * 101)
                result["public_benchmarks"].append(benchmark_repeat(args.process_index, repeat_index, x, public, counters, dispatch))  # type: ignore[index]
                del x; torch.cuda.empty_cache(); write(args.json, result)
        finally:
            if restore is not None:
                spy_restore_proof = restore()
    post_restore_proof = {
        "test_route_dispatcher_restored": route_restore_proof.get("test_route_dispatcher_restored"),
        "dispatcher_identity_matches_production": route_restore_proof.get("dispatcher_identity_matches_production"),
        "c1_backend_spy_restored": spy_restore_proof.get("c1_backend_spy_restored"),
        "pinned_backend_spy_restored": spy_restore_proof.get("pinned_backend_spy_restored"),
        "passed": True,
    }
    if not all(type(value) is bool and value for value in post_restore_proof.values()):
        raise RuntimeError(f"post-restore proof failed: {post_restore_proof}")
    result["post_restore_proof"] = post_restore_proof
    result["complete"] = True
    write(args.json, result)
    print(f"wrote B7-none public artifact {args.json}")


if __name__ == "__main__":
    main()
