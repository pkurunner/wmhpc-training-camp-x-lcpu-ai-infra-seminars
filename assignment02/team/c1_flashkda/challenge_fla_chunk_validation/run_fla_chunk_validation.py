#!/usr/bin/env python3
"""Run the two upstream FLA-comparison tests with direct-call evidence.

This runner intentionally does not benchmark, register a C1 FLA backend, or
modify a source tree.  It loads the official ``tests/test_fwd.py`` from the
already-built candidate worktree, wraps only its three call boundaries, then
calls ``test_fwd_vs_fla`` and ``test_fwd_varlen_vs_fla`` by their official
names.  The wrappers return the original values unchanged and record that the
candidate, fused-recurrent reference, and direct Triton chunk path all
returned both output and final-state tensors.

``FLA_DISABLE_BACKEND_DISPATCH=1`` must already be present before this process
imports FLA.  That makes the ``@dispatch('kda')`` decorator return the original
``fla.ops.kda.chunk.chunk_kda`` implementation instead of any registered
FlashKDA/C1 backend.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable


class ValidationError(RuntimeError):
    """A fail-closed audit condition."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def tensor_record(value: Any, label: str) -> dict[str, object]:
    """Record actual returned/mutated tensors without copying their contents."""

    import torch

    require(isinstance(value, torch.Tensor), f"{label} is not a tensor")
    require(value.is_cuda, f"{label} is not CUDA-resident")
    require(value.numel() > 0, f"{label} is empty")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": value.numel(),
    }


def result_pair(value: Any, label: str) -> tuple[dict[str, object], dict[str, object]]:
    require(isinstance(value, tuple) and len(value) == 2, f"{label} did not return (output, final_state)")
    return tensor_record(value[0], f"{label}.output"), tensor_record(value[1], f"{label}.final_state")


def append_observation(
    observations: list[dict[str, object]],
    *,
    path: str,
    value: Any,
    varlen: bool,
) -> None:
    output, final_state = result_pair(value, path)
    observations.append(
        {
            "path": path,
            "varlen": varlen,
            "output": output,
            "final_state": final_state,
        }
    )


def load_test_module(test_file: Path) -> Any:
    """Load the upstream test as a module while preserving its sibling imports."""

    require(test_file.is_file(), f"missing official test file: {test_file}")
    test_parent = str(test_file.parent.resolve())
    if test_parent not in sys.path:
        sys.path.insert(0, test_parent)
    spec = importlib.util.spec_from_file_location("c1_fla_chunk_upstream_test_fwd", test_file)
    require(spec is not None and spec.loader is not None, f"cannot load test module: {test_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def count_by_test(observations: list[dict[str, object]], offset: int) -> int:
    return len(observations) - offset


def validate_coverage(
    observations: dict[str, list[dict[str, object]]],
    per_test: dict[str, dict[str, int]],
) -> dict[str, object]:
    required_tests = ("test_fwd_vs_fla", "test_fwd_varlen_vs_fla")
    for test_name in required_tests:
        require(test_name in per_test, f"missing execution record for {test_name}")
        for path in ("candidate_flash_kda", "fused_recurrent_gold", "triton_chunk_kda"):
            require(per_test[test_name].get(path, 0) > 0, f"{test_name}: {path} was not called")
    for path, entries in observations.items():
        require(entries, f"{path}: no observed calls")
        require(any(not bool(item["varlen"]) for item in entries), f"{path}: fixed-length call absent")
        require(any(bool(item["varlen"]) for item in entries), f"{path}: packed-varlen call absent")
        for item in entries:
            require("output" in item and "final_state" in item, f"{path}: incomplete tensor evidence")
    return {
        "passed": True,
        "required_tests": list(required_tests),
        "per_test_call_counts": per_test,
        "total_call_counts": {path: len(entries) for path, entries in observations.items()},
        "fixed_and_varlen_covered_for_each_path": True,
        "output_and_final_state_observed_for_each_call": True,
    }


def atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True, help="candidate FlashKDA tests/test_fwd.py")
    parser.add_argument("--patched-root", type=Path, required=True, help="already-built candidate worktree")
    parser.add_argument("--fla-root", type=Path, required=True, help="pinned FLA checkout")
    parser.add_argument("--json", type=Path, required=True, help="audit output written atomically")
    args = parser.parse_args()

    require(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", "FLA_DISABLE_BACKEND_DISPATCH must equal 1 before import")
    patched_root = args.patched_root.resolve()
    fla_root = args.fla_root.resolve()
    test_file = args.test_file.resolve()
    output_path = args.json.resolve()
    require(under(test_file, patched_root), "official test file must be inside PATCHED_ROOT")
    require(output_path != test_file, "audit output cannot overwrite test source")

    # Imports below intentionally happen only after the dispatch environment
    # gate.  The installed C1 backend is not imported or registered here.
    import torch
    import fla.ops.backends as backend_core
    import fla.ops.kda as kda
    import flash_kda
    import flash_kda_C
    chunk_module = importlib.import_module("fla.ops.kda.chunk")
    fused_module = importlib.import_module("fla.ops.kda.fused_recurrent")

    require(getattr(backend_core, "_DISPATCH_DISABLED", False) is True, "FLA backend dispatch was not disabled at import")
    # ``torch.compiler.disable`` replaces a function's code location with a
    # PyTorch wrapper.  Module identity is the stable evidence: the public
    # export must be the exact callable exported by pinned FLA's chunk/fused
    # module, whose files are then hashed below.
    chunk_source = Path(chunk_module.__file__ or "").resolve()
    chunk_impl_source = chunk_source
    fused_source = Path(fused_module.__file__ or "").resolve()
    require(chunk_source == fla_root / "fla" / "ops" / "kda" / "chunk.py", f"chunk_kda origin drift: {chunk_source}")
    require(chunk_impl_source == fla_root / "fla" / "ops" / "kda" / "chunk.py", f"ChunkKDAFunction origin drift: {chunk_impl_source}")
    require(under(fused_source, fla_root), f"fused_recurrent_kda origin is outside FLA_ROOT: {fused_source}")
    require(kda.chunk_kda is chunk_module.chunk_kda, "public chunk_kda is not the pinned FLA chunk module export")
    require(kda.fused_recurrent_kda is fused_module.fused_recurrent_kda, "public fused_recurrent_kda is not the pinned FLA fused module export")
    flash_module_path = Path(flash_kda.__file__ or "").resolve()
    extension_path = Path(flash_kda_C.__file__ or "").resolve()
    require(under(flash_module_path, patched_root), f"flash_kda package is outside PATCHED_ROOT: {flash_module_path}")
    require(under(extension_path, patched_root), f"flash_kda extension is outside PATCHED_ROOT: {extension_path}")
    require(callable(getattr(flash_kda, "fwd", None)), "candidate flash_kda.fwd missing")

    test_module = load_test_module(test_file)
    require(getattr(test_module, "flash_kda", None) is flash_kda, "official test did not import the candidate flash_kda module")
    require(callable(getattr(test_module, "test_fwd_vs_fla", None)), "official fixed FLA test missing")
    require(callable(getattr(test_module, "test_fwd_varlen_vs_fla", None)), "official varlen FLA test missing")

    observations: dict[str, list[dict[str, object]]] = {
        "candidate_flash_kda": [],
        "fused_recurrent_gold": [],
        "triton_chunk_kda": [],
    }
    original_candidate: Callable[..., Any] = flash_kda.fwd
    original_fused: Callable[..., Any] = kda.fused_recurrent_kda
    original_chunk: Callable[..., Any] = kda.chunk_kda

    def candidate_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
        value = original_candidate(*call_args, **call_kwargs)
        require(len(call_args) >= 7, "candidate fwd output argument is missing")
        output = call_args[6]
        final_state = call_kwargs.get("final_state")
        append_observation(
            observations["candidate_flash_kda"],
            path="candidate_flash_kda",
            value=(output, final_state),
            varlen=call_kwargs.get("cu_seqlens") is not None,
        )
        return value

    def fused_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
        value = original_fused(*call_args, **call_kwargs)
        append_observation(
            observations["fused_recurrent_gold"],
            path="fused_recurrent_gold",
            value=value,
            varlen=call_kwargs.get("cu_seqlens") is not None,
        )
        return value

    def chunk_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
        value = original_chunk(*call_args, **call_kwargs)
        append_observation(
            observations["triton_chunk_kda"],
            path="triton_chunk_kda",
            value=value,
            varlen=call_kwargs.get("cu_seqlens") is not None,
        )
        return value

    flash_kda.fwd = candidate_wrapper
    kda.fused_recurrent_kda = fused_wrapper
    kda.chunk_kda = chunk_wrapper
    per_test: dict[str, dict[str, int]] = {}
    try:
        for test_name in ("test_fwd_vs_fla", "test_fwd_varlen_vs_fla"):
            before = {path: len(entries) for path, entries in observations.items()}
            print(f"===== OFFICIAL_TEST {test_name} =====", flush=True)
            getattr(test_module, test_name)()
            torch.cuda.synchronize()
            per_test[test_name] = {
                path: count_by_test(entries, before[path])
                for path, entries in observations.items()
            }
    finally:
        # No altered callable leaks into a later, unrelated test in the same
        # allocation even if an assertion above fails.
        flash_kda.fwd = original_candidate
        kda.fused_recurrent_kda = original_fused
        kda.chunk_kda = original_chunk

    coverage = validate_coverage(observations, per_test)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload: dict[str, object] = {
        "schema_version": 1,
        "official_tests": ["tests/test_fwd.py::test_fwd_vs_fla", "tests/test_fwd.py::test_fwd_varlen_vs_fla"],
        "dispatch": {
            "FLA_DISABLE_BACKEND_DISPATCH": os.environ.get("FLA_DISABLE_BACKEND_DISPATCH"),
            "backend_dispatch_disabled_at_import": True,
            "chunk_kda_source": str(chunk_source),
            "chunk_function_class_source": str(chunk_impl_source),
            "fused_recurrent_source": str(fused_source),
            "dispatch_bypass_means": "direct fla.ops.kda.chunk.chunk_kda; no registered FLA backend can intercept",
        },
        "runtime_identity": {
            "test_file": str(test_file),
            "test_file_sha256": sha256(test_file),
            "flash_kda_module": str(flash_module_path),
            "flash_kda_module_sha256": sha256(flash_module_path),
            "flash_kda_extension": str(extension_path),
            "flash_kda_extension_sha256": sha256(extension_path),
            "fla_chunk_py_sha256": sha256(chunk_source),
            "fla_fused_recurrent_py_sha256": sha256(fused_source),
        },
        "gpu": {
            "name": props.name,
            "capability": [props.major, props.minor],
            "multiprocessor_count": props.multi_processor_count,
        },
        "coverage": coverage,
        "observed_calls": observations,
        "plot_outputs": {
            name: {"exists": (Path.cwd() / name).is_file(), "bytes": (Path.cwd() / name).stat().st_size if (Path.cwd() / name).is_file() else 0}
            for name in ("plot.png", "plot_varlen.png")
        },
    }
    atomic_write(output_path, payload)
    print(json.dumps({"coverage": coverage, "json": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, OSError, AssertionError, RuntimeError, TypeError, ValueError) as error:
        print(f"FAIL-CLOSED: {error}", file=sys.stderr)
        raise SystemExit(2) from error
