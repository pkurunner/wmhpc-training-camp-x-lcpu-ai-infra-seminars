#!/usr/bin/env python3
"""Audit fixed-batch B300 dispatch through the real pinned FLA public API.

The runner keeps the release scope intentionally narrow: it checks only
``H=12,T=2048,B in {2,3,4,5,6,8}`` and the three FLA-public state contracts.
It does not modify FLA source files or C1's dispatcher; it registers the
opt-in custom backend only in this audit process.  A public call is accepted
as evidence only when a spy on that registered instance proves that the
registry invoked it exactly once immediately before the observed
``auto_dispatch`` decision is read.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch  # noqa: E402
from assignment02.team.c1_flashkda.challenge_seqcount_dispatch import (  # noqa: E402
    run_seqcount_dispatch as shared,
)


DIM = 128
HEADS = 12
TOKENS = 2048
BATCHES = (2, 3, 4, 5, 6, 8)
CONTRACTS = ("none", "fp32_final_only", "fp32_both")
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
FLASH_KDA_PYTHON_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
CLEAN_GPU_GATE_ENV = "C1_FIXED_BATCH_FLA_INTEGRATION_CLEAN_GPU"


@dataclass(frozen=True)
class Cell:
    batch: int
    contract: str
    expected_variant: str

    @property
    def key(self) -> str:
        return f"b{self.batch}_h12_t2048/{self.contract}"


def _expected_variant(batch: int, contract: str) -> str:
    if batch in (2, 3):
        return "vshard4_p2"
    if batch == 5:
        return "vshard2_p2"
    if batch in (4, 6) and contract in ("none", "fp32_final_only"):
        return "vshard2_p2"
    return "baseline"


CELLS = tuple(Cell(batch, contract, _expected_variant(batch, contract)) for batch in BATCHES for contract in CONTRACTS)


def _case(batch: int) -> shared.Case:
    return shared.Case(
        name=f"b{batch}_h12_t2048",
        form="fixed",
        sequences=batch,
        heads=HEADS,
        lengths=(TOKENS,) * batch,
        family="fixed_batch_fla_integration",
    )


CASES = {batch: _case(batch) for batch in BATCHES}


def _write(path: Path, result: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _matrix() -> list[dict[str, object]]:
    return [
        {"cell": cell.key, "expected_variant": cell.expected_variant, "release_role": "positive" if cell.expected_variant != "baseline" else "negative"}
        for cell in CELLS
    ]


def _initial_result(args: argparse.Namespace) -> dict[str, object]:
    positive = [cell.key for cell in CELLS if cell.expected_variant != "baseline"]
    negative = [cell.key for cell in CELLS if cell.expected_variant == "baseline"]
    return {
        "schema_version": 1,
        "purpose": "real pinned-FLA/public-registry fixed-batch correctness integration audit",
        "shape": {"H": HEADS, "T": TOKENS, "K": DIM, "V": DIM},
        "seed": args.seed,
        "matrix": _matrix(),
        "positive_cells": positive,
        "negative_cells": negative,
        "performance_observation": "not_run; this is a correctness/registry integration gate, not a release latency gate",
        "identity": {},
        "registry": {},
        "cells": {},
        "gates": {
            "scope": {"required_cells": 18, "actual_cells": len(CELLS), "passed": len(CELLS) == 18},
            "clean_gpu_shell_gate": {"required": True, "passed": False},
            "device": {"required": "B300, capability 10.3, 148 SM", "passed": False},
            "extension": {"required_sha256": AUDITED_EXTENSION_SHA256, "passed": False},
            "fla_pin": {"commit": FLA_COMMIT, "file_hashes": FLA_FILE_SHA256, "passed": False},
            "registry": {"passed": False},
        },
        "complete": False,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extension_identity(patched_root: Path) -> dict[str, object]:
    import flash_kda_C

    required = ("fwd", "fwd_vshard_p2", "fwd_vshard4_p2", "get_workspace_size")
    missing = [symbol for symbol in required if not callable(getattr(flash_kda_C, symbol, None))]
    if missing:
        raise RuntimeError(f"loaded extension lacks required symbols: {missing}")
    path = Path(flash_kda_C.__file__).resolve(strict=True)
    try:
        path.relative_to(patched_root.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"loaded extension is outside PATCHED_ROOT: {path}") from exc
    digest = _sha256(path)
    if digest != AUDITED_EXTENSION_SHA256:
        raise RuntimeError(f"unaudited extension: expected {AUDITED_EXTENSION_SHA256}, got {digest}")
    return {"path": str(path), "sha256": digest, "required_symbols": list(required), "gate_pass": True}


def _flash_kda_python_identity(patched_root: Path) -> dict[str, object]:
    import flash_kda

    path = Path(flash_kda.__file__).resolve(strict=True)
    try:
        path.relative_to(patched_root.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"loaded flash_kda package is outside PATCHED_ROOT: {path}") from exc
    digest = _sha256(path)
    if digest != FLASH_KDA_PYTHON_SHA256:
        raise RuntimeError(
            f"unaudited flash_kda Python package: expected {FLASH_KDA_PYTHON_SHA256}, got {digest}"
        )
    return {"path": str(path), "sha256": digest, "gate_pass": True}


def _fla_identity(fla_root: Path) -> dict[str, object]:
    if not fla_root.is_dir():
        raise RuntimeError(f"FLA_ROOT is not a directory: {fla_root}")
    completed = subprocess.run(
        ["git", "-C", str(fla_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != FLA_COMMIT:
        raise RuntimeError(f"pinned FLA commit mismatch: expected {FLA_COMMIT}, got {commit}")
    files: dict[str, str] = {}
    for relative, expected in FLA_FILE_SHA256.items():
        path = fla_root / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"pinned FLA file mismatch for {relative}: expected {expected}, got {actual}")
        files[relative] = actual
    module_files = {
        "fla": "fla/__init__.py",
        "fla.ops.backends": "fla/ops/backends/__init__.py",
        "fla.ops.kda": "fla/ops/kda/__init__.py",
        "fla.ops.kda.backends": "fla/ops/kda/backends/__init__.py",
        "fla.ops.kda.backends.flash_kda": "fla/ops/kda/backends/flash_kda.py",
        "fla.ops.kda.chunk": "fla/ops/kda/chunk.py",
    }
    loaded_modules: dict[str, str] = {}
    for module_name, relative in module_files.items():
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError(f"loaded FLA module has no __file__: {module_name}")
        actual_path = Path(module_file).resolve(strict=True)
        expected_path = (fla_root / relative).resolve(strict=True)
        if actual_path != expected_path:
            raise RuntimeError(
                f"loaded FLA module is not from FLA_ROOT: {module_name} -> {actual_path}, "
                f"expected {expected_path}"
            )
        loaded_modules[module_name] = str(actual_path)
    return {
        "root": str(fla_root.resolve()),
        "commit": commit,
        "files": files,
        "loaded_modules": loaded_modules,
        "gate_pass": True,
    }


def _device_identity() -> dict[str, object]:
    import torch

    name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if "B300" not in name.upper() or capability != (10, 3) or sm_count != 148:
        raise RuntimeError(
            f"B300-only audit got name={name!r}, capability={capability}, SMs={sm_count}"
        )
    return {"name": name, "capability": list(capability), "multiprocessor_count": sm_count, "gate_pass": True}


def _initial_state(contract: str, batch: int) -> torch.Tensor | None:
    """Return a deterministic, non-symmetric contiguous FP32 initial state."""
    if contract != "fp32_both":
        return None
    import torch

    count = batch * HEADS * DIM * DIM
    # Consecutive affine values are deliberate: every element is distinct,
    # continuous in storage order, and no symmetric random pattern can mask an
    # accidental transpose or sequence broadcast.
    return (
        torch.arange(count, dtype=torch.float32, device="cuda")
        .reshape(batch, HEADS, DIM, DIM)
        .mul_(1.0 / 8192.0)
        .add_(0.125)
        .contiguous()
    )


def _call_kwargs(x: object, initial_state: torch.Tensor | None, output_final_state: bool) -> dict[str, object]:
    return {
        "scale": x.scale,
        "initial_state": initial_state,
        "output_final_state": output_final_state,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "allow_neg_eigval": False,
        "state_v_first": True,
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
        "safe_gate": True,
        "lower_bound": x.lower_bound,
        "disable_recompute": False,
        "return_intermediate_states": False,
        "cp_context": None,
        "A_log": x.a_log,
        "dt_bias": x.dt_bias,
    }


def _call(
    fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    x: object,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return fn(x.q, x.k, x.v, x.g, x.beta, **_call_kwargs(x, initial_state, output_final_state))


def _verify_backend(backend: object, x: object, initial_state: torch.Tensor | None, output_final_state: bool, label: str) -> dict[str, object]:
    verdict = backend.verify(
        "chunk_kda", x.q, x.k, x.v, x.g, x.beta, **_call_kwargs(x, initial_state, output_final_state)
    )
    if not isinstance(verdict, tuple) or len(verdict) != 2 or verdict[0] is not True:
        raise AssertionError(f"{label} verifier rejected this audited cell: {verdict!r}")
    return {"passed": True, "reason": verdict[1]}


def _final_contract(value: torch.Tensor | None, batch: int, expected_present: bool, label: str) -> dict[str, object]:
    import torch

    if not expected_present:
        if value is not None:
            raise AssertionError(f"{label}: expected final_state=None")
        return {"present": False}
    if value is None:
        raise AssertionError(f"{label}: expected a final_state")
    expected_shape = (batch, HEADS, DIM, DIM)
    if value.dtype != torch.float32 or tuple(value.shape) != expected_shape or not value.is_contiguous():
        raise AssertionError(
            f"{label}: final_state must be contiguous FP32 {expected_shape}, got dtype={value.dtype}, shape={tuple(value.shape)}"
        )
    return {"present": True, "dtype": str(value.dtype), "shape": list(value.shape), "contiguous": True}


def _assert_exact(
    actual: tuple[torch.Tensor, torch.Tensor | None],
    expected: tuple[torch.Tensor, torch.Tensor | None],
    batch: int,
    expected_final: bool,
    label: str,
) -> dict[str, object]:
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common

    common.require_exact(f"{label}/output", actual[0], expected[0])
    result: dict[str, object] = {
        "output_exact": True,
        "output_max_abs": common.max_abs(actual[0], expected[0]),
        "actual_final": _final_contract(actual[1], batch, expected_final, f"{label}/actual"),
        "pinned_final": _final_contract(expected[1], batch, expected_final, f"{label}/pinned"),
    }
    if expected_final:
        if actual[1] is None or expected[1] is None:  # guarded above, retains type narrowing.
            raise AssertionError(f"{label}: missing final state")
        common.require_exact(f"{label}/final_state", actual[1], expected[1])
        result.update({"final_state_exact": True, "final_state_max_abs": common.max_abs(actual[1], expected[1])})
    return result


def _registry_snapshot(registry: object) -> list[dict[str, object]]:
    return [
        {
            "backend_type": backend.backend_type,
            "priority": backend.priority,
            "available": backend.is_available(),
            "enabled": backend.is_enabled(),
        }
        for backend in registry._get_sorted_backends()
    ]


def _register_and_check_registry() -> tuple[object, object, list[dict[str, object]]]:
    from fla.ops.kda.backends import kda_registry
    from fla.ops.kda.backends.flash_kda import FlashKDABackend

    custom = fla_backend.register_backend()
    if fla_backend.register_backend() is not custom:
        raise AssertionError("custom FLA backend registration is not idempotent")
    sorted_backends = kda_registry._get_sorted_backends()
    types = [backend.backend_type for backend in sorted_backends]
    if types.count("c1_b300_flash_kda") != 1:
        raise AssertionError(f"custom backend duplicate/missing in registry: {types}")
    if "flash_kda" not in types or types.index("c1_b300_flash_kda") >= types.index("flash_kda"):
        raise AssertionError(f"custom backend is not ordered ahead of flash_kda: {types}")
    if not custom.is_available() or not custom.is_enabled():
        raise AssertionError("custom backend must be available and opted in")
    return custom, FlashKDABackend(), _registry_snapshot(kda_registry)


def _spy_registered_backend(custom_backend: object) -> tuple[Callable[..., object], dict[str, int]]:
    """Install an instance-local spy; FLA registry lookup observes this exact method."""
    original = custom_backend.chunk_kda
    counter = {"calls": 0}

    def spy(*args: object, **kwargs: object) -> object:
        counter["calls"] += 1
        return original(*args, **kwargs)

    custom_backend.chunk_kda = spy
    return original, counter


def _run_cell(
    cell: Cell,
    x: object,
    pinned_fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    custom_direct_fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    public_fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    pinned_backend: object,
    custom_backend: object,
    spy_counter: Mapping[str, int],
) -> dict[str, object]:
    import torch

    output_final = cell.contract != "none"
    initial = _initial_state(cell.contract, cell.batch)
    with torch.inference_mode():
        # The pinned verifier deliberately rejects grad-enabled calls.  Check
        # it under the exact inference-mode context used for all three paths.
        verifiers = {
            "pinned_flash_kda": _verify_backend(
                pinned_backend, x, None if initial is None else initial.clone(), output_final, f"{cell.key}/pinned"
            ),
            "custom_backend": _verify_backend(
                custom_backend, x, None if initial is None else initial.clone(), output_final, f"{cell.key}/custom"
            ),
        }
        pinned = _call(pinned_fn, x, None if initial is None else initial.clone(), output_final)
        direct_custom = _call(custom_direct_fn, x, None if initial is None else initial.clone(), output_final)
        spy_before = int(spy_counter["calls"])
        public = _call(public_fn, x, None if initial is None else initial.clone(), output_final)
        spy_after = int(spy_counter["calls"])
        # This must stay directly adjacent to the public call.  A decision
        # left by direct_custom would otherwise be indistinguishable from a
        # genuine registry call that skipped the custom backend.
        decision = auto_dispatch.get_last_decision()
        torch.cuda.synchronize()
    if spy_after != spy_before + 1:
        raise AssertionError(f"{cell.key}: public chunk_kda changed custom backend spy {spy_before}->{spy_after}, expected +1")
    if decision.get("chosen_variant") != cell.expected_variant:
        raise AssertionError(f"{cell.key}: public decision {decision} != expected {cell.expected_variant}")
    expected_final_shape = [cell.batch, HEADS, DIM, DIM] if output_final else None
    return {
        "expected_variant": cell.expected_variant,
        "verifiers": verifiers,
        "initial_state": {
            "present": initial is not None,
            "dtype": str(initial.dtype) if initial is not None else None,
            "shape": list(initial.shape) if initial is not None else None,
            "construction": "deterministic non-symmetric contiguous affine FP32 values" if initial is not None else None,
        },
        "expected_final_state": {"present": output_final, "dtype": "torch.float32" if output_final else None, "shape": expected_final_shape},
        "direct_custom_vs_pinned": _assert_exact(direct_custom, pinned, cell.batch, output_final, f"{cell.key}/direct_custom_vs_pinned"),
        "public_registry_vs_pinned": _assert_exact(public, pinned, cell.batch, output_final, f"{cell.key}/public_registry_vs_pinned"),
        "public_custom_backend_spy": {"before": spy_before, "after": spy_after, "delta": spy_after - spy_before, "passed": True},
        "public_decision": decision,
        "cell_gate_pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--describe", action="store_true", help="write the 18-cell matrix without importing torch or FLA")
    args = parser.parse_args()
    if len(CELLS) != 18 or sum(cell.expected_variant != "baseline" for cell in CELLS) != 13:
        raise RuntimeError("fixed FLA integration scope corruption")
    result = _initial_result(args)
    if args.describe:
        result["describe_only"] = True
        _write(args.json, result)
        print(f"wrote fixed-batch FLA integration matrix {args.json}")
        return
    if os.environ.get(CLEAN_GPU_GATE_ENV) != "1":
        raise RuntimeError(
            "refusing a direct GPU run: use run_clean_fixed_batch_fla_integration_audit.sh so "
            f"{CLEAN_GPU_GATE_ENV}=1 is set only after its PRE clean-GPU check"
        )
    if os.environ.get("C1_B300_FLASH_KDA") != "1":
        raise RuntimeError("C1_B300_FLASH_KDA=1 is required for the opt-in custom FLA backend")
    patched_root_text = os.environ.get("PATCHED_ROOT")
    fla_root_text = os.environ.get("FLA_ROOT")
    if not patched_root_text or not fla_root_text:
        raise RuntimeError("PATCHED_ROOT and FLA_ROOT are required for identity gates")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shared.torch = torch
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common
    shared.common = common

    patched_root = Path(patched_root_text)
    fla_root = Path(fla_root_text)
    result["identity"] = {
        "device": _device_identity(),
        "extension": _extension_identity(patched_root),
        "flash_kda_python": _flash_kda_python_identity(patched_root),
        "fla": _fla_identity(fla_root),
    }
    result["gates"]["clean_gpu_shell_gate"]["passed"] = True  # type: ignore[index]
    result["gates"]["device"]["passed"] = True  # type: ignore[index]
    result["gates"]["extension"]["passed"] = True  # type: ignore[index]
    result["gates"]["fla_pin"]["passed"] = True  # type: ignore[index]

    # _fla_identity already hashed the pinned source and proved every relevant
    # imported module resolves to that checkout.
    from fla.ops.kda import chunk_kda
    global fla_backend
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import fla_backend

    custom_backend, pinned_backend, backend_order = _register_and_check_registry()
    custom_direct_fn, spy_counter = _spy_registered_backend(custom_backend)
    result["registry"] = {"backend_order": backend_order, "registration_idempotent": True, "custom_before_pinned": True}
    result["gates"]["registry"]["passed"] = True  # type: ignore[index]
    _write(args.json, result)

    try:
        for batch_index, batch in enumerate(BATCHES):
            case = CASES[batch]
            x = shared._make_inputs(case, args.seed + batch_index * 10_007)
            for contract_index, contract in enumerate(CONTRACTS):
                cell = Cell(batch, contract, _expected_variant(batch, contract))
                print(f"FLA public integration {cell.key}: expected={cell.expected_variant}")
                result["cells"][cell.key] = _run_cell(  # type: ignore[index]
                    cell,
                    x,
                    pinned_backend.chunk_kda,
                    custom_direct_fn,
                    chunk_kda,
                    pinned_backend,
                    custom_backend,
                    spy_counter,
                )
                _write(args.json, result)
            del x
            torch.cuda.empty_cache()
    finally:
        # The process normally exits after the audit, but restoring the method
        # avoids leaking instrumentation if this runner is imported by a test.
        custom_backend.chunk_kda = custom_direct_fn

    result["complete"] = True
    _write(args.json, result)
    print(f"wrote {args.json}; 18/18 public FLA cells passed")


if __name__ == "__main__":
    main()
