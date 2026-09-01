#!/usr/bin/env python3
"""Fail-closed v5 public-registry cross-map regression runner.

The runner is intentionally read-only with respect to the v5 dispatcher maps
and production source.  It proves four already published cells through the
real FLA registry, and three neighbouring baseline/rejection controls through
that same public registry.  It is a regression gate, never a publication gate.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import types
from typing import Any, Callable, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Deliberately no FLA, Torch, dispatcher, or production-helper imports here.
# ``_load_runtime_modules_after_clean_gate`` is the sole heavy-import gateway.
SCHEMA_VERSION = 4
DIM, HEADS = 128, 12
SAMPLES, WARMUP = 100, 12
PERCENTILES = ("p50", "p95", "p99")
EXPECTED_AUTO_SHA = "9633179c6a5e8ce61b1857a102f81074c6bdfdc63163e13980f40c24d6583c29"
EXPECTED_BACKEND_SHA = "152ac4ba897c1887963553d592456b7f988c89105e6f6cb507a44757058d5fd1"
EXPECTED_EXTENSION_SHA = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_HELPER_PATH = "/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so"
EXPECTED_HELPER_SHA = "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
EXPECTED_PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
EXPECTED_PATCHED_DIRTY = {
    "csrc/flash_kda.cpp": "38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4",
    "csrc/fwd.h": "613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083",
    "csrc/smxx/fwd_launch.cu": "a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928",
}
SUPPORTING_HELPER_SHA256 = {
    "varlen_helper": "e07481e7e66551f1f9eaafa7f1fb9db9ef193f1694ff831ee0a07802c8ffbe14",
    "tail_helper": "f4144f5fbdd61396ff907c6290b767b5570e04d19087f8332f9db10e56e7b1dc",
    "shared_seqcount": "4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f",
    "confirmation": "9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b",
    "harness": "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52",
    "varlen_metadata": "f89a97ba284c7a24a3df54efca7bb60eb70f9cbda659200bd6cb3e8dfdaf4ccd",
    "reference_torch_ref": "bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5",
}
FLA_SOURCE_SHA256 = {
    "fla/__init__.py": "b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d",
    "fla/ops/backends/__init__.py": "a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635",
    "fla/ops/kda/__init__.py": "24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb",
    "fla/ops/kda/backends/__init__.py": "86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797",
    "fla/ops/kda/backends/flash_kda.py": "0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2",
    "fla/ops/kda/chunk.py": "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8",
}

# These are intentionally names rather than eagerly imported module objects.
# Every member may indirectly load Torch/CUDA, so importing any one before
# the runner's own 0-MiB/no-compute-app gate is forbidden.
HEAVY_RUNTIME_IMPORTS = {
    "auto_dispatch": "assignment02.team.c1_flashkda.challenge_tp8_dispatch.auto_dispatch",
    "fla_backend": "assignment02.team.c1_flashkda.challenge_tp8_dispatch.fla_backend",
    "varlen_metadata": "assignment02.team.c1_flashkda.challenge_tp8_dispatch.varlen_metadata",
    "shared_seqcount": "assignment02.team.c1_flashkda.challenge_seqcount_dispatch.run_seqcount_dispatch",
    "confirmation": "assignment02.team.c1_flashkda.challenge_varlen_dispatch.run_varlen_dispatch_confirmation",
    "varlen_helper": "assignment02.team.c1_flashkda.challenge_varlen_dispatch.run_varlen_fla_handoff_candidate",
    "tail_helper": "assignment02.team.c1_flashkda.challenge_tail8191_production_freeze.run_tail8191_production_freeze",
    "harness": "assignment02.team.c1_flashkda.harness.validate_and_bench",
    "fla_ops_kda": "fla.ops.kda",
}
# The cross-map runner directly calls only this TYPE_CHECKING-only dependency.
# It is the sole allowed runtime-global hydration in this protocol; no artifact
# claims a static proof for other helper execution paths.
SHARED_SEQCOUNT_LABEL = "shared_seqcount"
SHARED_MAKE_INPUTS = "_make_inputs"
BOOTSTRAP_STAGES = [
    "source_ledger_pre_torch",
    "pre_torch_clean_gpu",
    "heavy_runtime_import",
    "loaded_module_identity",
    "shared_make_inputs_hydration",
    "canonical_map_pre",
]


@dataclass(frozen=True)
class Cell:
    name: str
    form: str
    batch: int
    tokens: int
    lengths: tuple[int, ...]
    contract: str
    variant: str
    reason: str

    @property
    def sequences(self) -> int:
        return len(self.lengths)


POSITIVES = (
    Cell("fixed_b2_h12_t2048_fp32_both", "fixed", 2, 2048, (2048, 2048), "fp32_both", "vshard4_p2", "fixed_batch_b2_h12_t2048_fp32_both_whitelist_hit"),
    Cell("fixed_b5_h12_t2048_fp32_both", "fixed", 5, 2048, (2048,) * 5, "fp32_both", "vshard2_p2", "fixed_batch_b5_h12_t2048_fp32_both_whitelist_hit"),
    Cell("fixed_b1_h12_t8191_none", "fixed", 1, 8191, (8191,), "none", "vshard4_p2", "fixed_single_batch_b1_h12_t8191_none_whitelist_hit"),
    Cell("varlen_skew_n6_h12_t12288_fp32_both", "varlen", 1, 12288, (1, 1, 1, 1, 1, 12283), "fp32_both", "vshard4_p2", "varlen_skew_n6_h12_t12288_fp32_both_whitelist_hit"),
)
EXPECTED_POSITIVE_KEYS = {cell.name for cell in POSITIVES}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha_env(name: str, expected: str) -> str:
    value = os.environ.get(name)
    if value != expected:
        raise RuntimeError(f"{name} must equal the frozen SHA256")
    return value


def _external_file_sha(name: str, path: Path) -> str:
    """Bind a protocol file to a SHA supplied outside the file being run."""
    expected = os.environ.get(name)
    if type(expected) is not str or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise RuntimeError(f"{name} must be a lowercase SHA256")
    actual = _sha(path)
    if actual != expected:
        raise RuntimeError(f"{name} does not match bytes at {path}")
    return actual


def _canonical_maps() -> dict[str, object]:
    """Serialize all three public maps without coercing keys or values."""
    maps = {
        "fixed_batch": getattr(auto_dispatch, "_FIXED_BATCH_PUBLIC_VARIANTS", None),
        "fixed_single_batch": getattr(auto_dispatch, "_FIXED_SINGLE_BATCH_PUBLIC_VARIANTS", None),
        "varlen": getattr(auto_dispatch, "_VARLEN_PUBLIC_VARIANTS", None),
    }
    rows: dict[str, list[dict[str, object]]] = {}
    for name, raw in maps.items():
        if type(raw) is not dict:
            raise RuntimeError(f"{name} public map is not a built-in dict")
        current: list[dict[str, object]] = []
        for key, variant in raw.items():
            if type(key) is not tuple or type(variant) is not str:
                raise RuntimeError(f"{name} entry type drift")
            if name == "varlen":
                if len(key) != 2 or type(key[0]) is not tuple or type(key[1]) is not str or any(type(i) is not int for i in key[0]):
                    raise RuntimeError("varlen key type drift")
                current.append({"offsets": list(key[0]), "contract": key[1], "variant": variant})
            else:
                if len(key) != 2 or type(key[0]) is not int or type(key[1]) is not str:
                    raise RuntimeError(f"{name} key type drift")
                current.append({"first": key[0], "contract": key[1], "variant": variant})
        rows[name] = sorted(current, key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))
    expected = {
        ("fixed_batch", 2, "fp32_both"): "vshard4_p2",
        ("fixed_batch", 5, "fp32_both"): "vshard2_p2",
        ("fixed_single_batch", 8191, "none"): "vshard4_p2",
        ("varlen", (0, 1, 2, 3, 4, 5, 12288), "fp32_both"): "vshard4_p2",
    }
    for key, variant in expected.items():
        map_name, first, contract = key
        map_key = (first, contract)
        if maps[map_name].get(map_key) != variant:  # type: ignore[union-attr]
            raise RuntimeError(f"v5 public map missing/changed required cell {key!r}")
    payload = {"entries": rows, "object_ids": {name: id(value) for name, value in maps.items()}}
    payload["digest"] = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return payload


def _content_map(snapshot: Mapping[str, object]) -> dict[str, object]:
    """The portable, typed map identity.  ``id`` is local-process evidence only."""
    return {"entries": snapshot["entries"], "digest": snapshot["digest"]}


def _assert_map_readonly(pre: Mapping[str, object], post: Mapping[str, object]) -> dict[str, object]:
    if _content_map(pre) != _content_map(post):
        raise RuntimeError("public map content/digest changed during workload")
    if pre.get("object_ids") != post.get("object_ids"):
        raise RuntimeError("public map object identity changed inside one raw PID")
    return {"content_unchanged": True, "object_ids_unchanged_within_raw_pid": True, "passed": True}


def _source_ledger_pre_torch(args: argparse.Namespace) -> dict[str, object]:
    """Physically hash all bound sources without importing any runtime helper.

    This deliberately runs before the runner's first CUDA cleanliness query.
    In particular, it must never reach for ``module.__file__``: importing a
    module merely to obtain that attribute was the P0 defect in jobs 12828 and
    12882.  Runtime module identities are checked separately, only after the
    clean gate passes.
    """
    auto_path = REPO_ROOT / "assignment02/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py"
    backend_path = REPO_ROOT / "assignment02/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py"
    if _sha(auto_path) != _sha_env("C1_V5_CROSSMAP_AUTO_DISPATCH_SHA256", EXPECTED_AUTO_SHA):
        raise RuntimeError("auto_dispatch bytes do not match external v5 binding")
    if _sha(backend_path) != _sha_env("C1_V5_CROSSMAP_FLA_BACKEND_SHA256", EXPECTED_BACKEND_SHA):
        raise RuntimeError("fla_backend bytes do not match external v5 binding")
    runner = Path(__file__).resolve(strict=True)
    analyzer = Path(args.analyzer_path).resolve(strict=True)
    shell = Path(args.protocol_shell_path).resolve(strict=True)
    if shell != Path(__file__).with_name("run_clean_v5_crossmap_regression.sh").resolve(strict=True):
        raise RuntimeError("non-canonical protocol shell path")
    runner_sha = _external_file_sha("C1_V5_CROSSMAP_RUNNER_SHA256", runner)
    analyzer_sha = _external_file_sha("C1_V5_CROSSMAP_ANALYZER_SHA256", analyzer)
    shell_sha = _external_file_sha("EXPECTED_PROTOCOL_SHELL_SHA256", shell)
    owned = REPO_ROOT / "assignment02/team/c1_flashkda"
    helper_specs = {
        "varlen_helper": owned / "challenge_varlen_dispatch/run_varlen_fla_handoff_candidate.py",
        "tail_helper": owned / "challenge_tail8191_production_freeze/run_tail8191_production_freeze.py",
        "shared_seqcount": owned / "challenge_seqcount_dispatch/run_seqcount_dispatch.py",
        "confirmation": owned / "challenge_varlen_dispatch/run_varlen_dispatch_confirmation.py",
        "varlen_metadata": owned / "challenge_tp8_dispatch/varlen_metadata.py",
        "harness": owned / "harness/validate_and_bench.py",
        "reference_torch_ref": args.reference_root / "tests/torch_ref.py",
    }
    helpers: dict[str, dict[str, str]] = {}
    for name, canonical_path in helper_specs.items():
        actual = canonical_path.resolve(strict=True)
        if _sha(actual) != SUPPORTING_HELPER_SHA256[name]:
            raise RuntimeError(f"supporting helper identity drift: {name}")
        helpers[name] = {"path": str(actual), "sha256": SUPPORTING_HELPER_SHA256[name]}
    fla_sources: dict[str, dict[str, str]] = {}
    for relative, expected in FLA_SOURCE_SHA256.items():
        path = (args.fla_root / relative).resolve(strict=True)
        if _sha(path) != expected:
            raise RuntimeError(f"FLA source ledger drift: {relative}")
        fla_sources[relative] = {"path": str(path), "sha256": expected}
    return {
        "auto_dispatch": {"path": str(auto_path.resolve()), "sha256": EXPECTED_AUTO_SHA},
        "fla_backend": {"path": str(backend_path.resolve()), "sha256": EXPECTED_BACKEND_SHA},
        "runner": {"path": str(runner), "sha256": runner_sha},
        "analyzer": {"path": str(analyzer), "sha256": analyzer_sha},
        "protocol_shell": {"path": str(shell), "sha256": shell_sha},
        "supporting_helpers": helpers,
        "fla_sources": fla_sources,
        "passed": True,
    }


def _pre_torch_import_state() -> dict[str, object]:
    """Fail if an import has already invalidated a pre-Torch clean-GPU gate."""
    torch_modules = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch."))
    heavy_modules = sorted(
        name
        for name in sys.modules
        if name == "fla" or name.startswith("fla.") or name in HEAVY_RUNTIME_IMPORTS.values()
    )
    if torch_modules or heavy_modules:
        raise RuntimeError(
            "pre-Torch gate invalid: Torch/FLA/production helper was already imported "
            f"(torch={torch_modules!r}, heavy={heavy_modules!r})"
        )
    return {"torch_modules_before_gate": [], "heavy_modules_before_gate": [], "passed": True}


def _require_clean_gate_before_import(gate: Mapping[str, object]) -> None:
    expected = {
        "index", "uuid", "name", "compute_capability", "memory_used_mib", "compute_apps",
        "torch_modules_before_gate", "heavy_modules_before_gate", "passed",
    }
    if (
        set(gate) != expected
        or gate.get("memory_used_mib") != 0
        or gate.get("compute_apps") != []
        or gate.get("torch_modules_before_gate") != []
        or gate.get("heavy_modules_before_gate") != []
        or gate.get("passed") is not True
    ):
        raise RuntimeError("heavy runtime import attempted without a valid earliest clean-GPU gate")


def _import_named_modules_after_clean_gate(
    gate: Mapping[str, object], names: Mapping[str, str], importer: Callable[[str], object] = importlib.import_module,
) -> dict[str, object]:
    """The sole import primitive for FLA/dispatcher/helper runtime modules."""
    _require_clean_gate_before_import(gate)
    return {label: importer(module_name) for label, module_name in names.items()}


def _loaded_module_identity(modules: Mapping[str, object], ledger: Mapping[str, object]) -> dict[str, object]:
    expected_paths = {
        "auto_dispatch": Path(str(ledger["auto_dispatch"]["path"])),  # type: ignore[index]
        "fla_backend": Path(str(ledger["fla_backend"]["path"])),  # type: ignore[index]
        "varlen_metadata": Path(str(ledger["supporting_helpers"]["varlen_metadata"]["path"])),  # type: ignore[index]
        "shared_seqcount": Path(str(ledger["supporting_helpers"]["shared_seqcount"]["path"])),  # type: ignore[index]
        "confirmation": Path(str(ledger["supporting_helpers"]["confirmation"]["path"])),  # type: ignore[index]
        "varlen_helper": Path(str(ledger["supporting_helpers"]["varlen_helper"]["path"])),  # type: ignore[index]
        "tail_helper": Path(str(ledger["supporting_helpers"]["tail_helper"]["path"])),  # type: ignore[index]
        "harness": Path(str(ledger["supporting_helpers"]["harness"]["path"])),  # type: ignore[index]
        "fla_ops_kda": Path(str(ledger["fla_sources"]["fla/ops/kda/__init__.py"]["path"])),  # type: ignore[index]
    }
    if set(modules) != set(HEAVY_RUNTIME_IMPORTS) or set(expected_paths) != set(HEAVY_RUNTIME_IMPORTS):
        raise RuntimeError("heavy runtime module scope drift")
    rows: dict[str, dict[str, str]] = {}
    for label, module_name in HEAVY_RUNTIME_IMPORTS.items():
        module = modules[label]
        actual_path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
        expected_path = expected_paths[label].resolve(strict=True)
        if actual_path != expected_path:
            raise RuntimeError(f"loaded module is not bound to canonical source: {label}")
        rows[label] = {"module": module_name, "path": str(actual_path)}
    return {"modules": rows, "passed": True}


def _load_runtime_modules_after_clean_gate(
    gate: Mapping[str, object], ledger: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    modules = _import_named_modules_after_clean_gate(gate, HEAVY_RUNTIME_IMPORTS)
    return modules, _loaded_module_identity(modules, ledger)


def _shared_make_inputs_binding(
    module: object,
    torch_module: object,
    *,
    phase: str,
    expect_torch_global: bool,
    expected_module: object | None = None,
    expected_function: object | None = None,
) -> tuple[dict[str, object], object]:
    """Validate the sole deliberately hydrated helper global.

    ``run_seqcount_dispatch`` has a TYPE_CHECKING-only ``torch`` import, while
    this runner directly calls its ``_make_inputs`` function.  No static claim
    is made about other helper execution paths: this narrow check only
    proves that the one known missing global is on the canonical module and
    function object, is absent before the one permitted binding, and remains
    the canonical ``sys.modules['torch']`` object afterwards.
    """
    if phase not in {"pre_hydration", "bound_pre_workload", "post_workload"}:
        raise RuntimeError(f"shared make-inputs binding has an invalid phase: {phase}")
    module_name = HEAVY_RUNTIME_IMPORTS[SHARED_SEQCOUNT_LABEL]
    if module is not sys.modules.get(module_name):
        raise RuntimeError("shared seqcount module is detached from sys.modules")
    if getattr(module, "__name__", None) != module_name:
        raise RuntimeError("shared seqcount module name is non-canonical")
    if expected_module is not None and module is not expected_module:
        raise RuntimeError("shared seqcount module object drifted after hydration")
    try:
        module_dict = vars(module)
        module_path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    except (TypeError, OSError) as exc:
        raise RuntimeError("shared seqcount canonical module identity is unavailable") from exc
    function = getattr(module, SHARED_MAKE_INPUTS, None)
    globals_dict = getattr(function, "__globals__", None)
    if not callable(function) or type(globals_dict) is not dict:
        raise RuntimeError("shared seqcount _make_inputs function/global namespace is unavailable")
    if getattr(function, "__module__", None) != module_name:
        raise RuntimeError("shared seqcount _make_inputs function module is non-canonical")
    if globals_dict is not module_dict:
        raise RuntimeError("shared seqcount _make_inputs globals are detached from canonical module")
    if expected_function is not None and function is not expected_function:
        raise RuntimeError("shared seqcount _make_inputs function object drifted after hydration")
    if sys.modules.get("torch") is not torch_module or getattr(torch_module, "__name__", None) != "torch":
        raise RuntimeError("canonical torch module is unavailable after clean-gate import")
    torch_present = "torch" in globals_dict
    if torch_present != expect_torch_global:
        raise RuntimeError("shared seqcount _make_inputs torch-global presence drift")
    if expect_torch_global and globals_dict.get("torch") is not torch_module:
        raise RuntimeError("shared seqcount _make_inputs torch global is non-canonical")
    record = {
        "phase": phase,
        "module_label": SHARED_SEQCOUNT_LABEL,
        "module": module_name,
        "module_path": str(module_path),
        "module_object_id": id(module),
        "function": SHARED_MAKE_INPUTS,
        "function_module": module_name,
        "function_object_id": id(function),
        "function_globals_is_module_dict": True,
        "torch_global_present": torch_present,
        "torch_object_id": id(torch_module) if torch_present else None,
        "sys_modules_torch_object_id": id(torch_module),
        "torch_is_sys_modules_canonical": torch_present,
        "passed": True,
    }
    return record, function


def _hydrate_shared_make_inputs_after_clean_gate(
    gate: Mapping[str, object], modules: Mapping[str, object], torch_module: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Perform the only allowed post-clean-gate runtime-global binding."""
    _require_clean_gate_before_import(gate)
    if set(modules) != set(HEAVY_RUNTIME_IMPORTS):
        raise RuntimeError("shared hydration saw a drifted heavy-module scope")
    module = modules.get(SHARED_SEQCOUNT_LABEL)
    if module is None:
        raise RuntimeError("shared seqcount helper is absent from the post-gate module set")
    pre, function = _shared_make_inputs_binding(
        module, torch_module, phase="pre_hydration", expect_torch_global=False,
    )
    globals_dict = getattr(function, "__globals__")
    globals_dict["torch"] = torch_module
    bound, bound_function = _shared_make_inputs_binding(
        module,
        torch_module,
        phase="bound_pre_workload",
        expect_torch_global=True,
        expected_module=module,
        expected_function=function,
    )
    if bound_function is not function:
        raise RuntimeError("shared seqcount _make_inputs identity changed while binding torch")
    return (
        {"pre_hydration": pre, "bound_pre_workload": bound, "passed": True},
        {"module": module, "function": function},
    )


def _verify_shared_make_inputs_post_workload(
    binding: Mapping[str, object], torch_module: object,
) -> dict[str, object]:
    """Re-open the exact allowed binding after all workload/source/map gates."""
    if set(binding) != {"module", "function"}:
        raise RuntimeError("shared make-inputs binding state scope drift")
    module, function = binding["module"], binding["function"]
    post, observed_function = _shared_make_inputs_binding(
        module,
        torch_module,
        phase="post_workload",
        expect_torch_global=True,
        expected_module=module,
        expected_function=function,
    )
    if observed_function is not function:
        raise RuntimeError("shared seqcount _make_inputs identity changed after workload")
    return post


def _runtime_source(ledger: Mapping[str, object], loaded_identity: Mapping[str, object]) -> dict[str, object]:
    if loaded_identity.get("passed") is not True:
        raise RuntimeError("loaded runtime module identity did not pass")
    return {"ledger": dict(ledger), "loaded_modules": dict(loaded_identity), "passed": True}


def _pre_torch_clean_gpu(pre_import_state: Mapping[str, object]) -> dict[str, object]:
    if pre_import_state != {"torch_modules_before_gate": [], "heavy_modules_before_gate": [], "passed": True}:
        raise RuntimeError("pre-Torch import-state evidence drift")
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap,memory.used", "--format=csv,noheader,nounits"]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines()
    lines = [line.strip() for line in output if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"requires exactly one visible GPU: {lines!r}")
    fields = [part.strip() for part in lines[0].split(",")]
    if len(fields) != 5 or int(fields[4]) != 0:
        raise RuntimeError(f"clean GPU 0-MiB gate failed: {lines!r}")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    active = [line.strip() for line in processes.splitlines() if line.strip() and "No running" not in line]
    if active:
        raise RuntimeError(f"clean GPU gate found active compute processes: {active!r}")
    return {
        "index": fields[0], "uuid": fields[1], "name": fields[2], "compute_capability": fields[3],
        "memory_used_mib": 0, "compute_apps": [], **dict(pre_import_state),
    }


def _dirty_set(root: Path, expected_commit: str, *, require_dirty: bool) -> dict[str, object]:
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if commit != expected_commit:
        raise RuntimeError(f"worktree commit drift: {commit}")
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"], check=True, capture_output=True, text=True).stdout.splitlines()
    expected_status = [f" M {name}" for name in EXPECTED_PATCHED_DIRTY] if require_dirty else []
    if status != expected_status:
        raise RuntimeError(f"tracked dirty-set drift: {status!r}")
    files = {}
    if require_dirty:
        for relative, digest in EXPECTED_PATCHED_DIRTY.items():
            path = root / relative
            if _sha(path) != digest:
                raise RuntimeError(f"patched dirty file hash drift: {relative}")
            files[relative] = digest
    return {"root": str(root.resolve()), "commit": commit, "tracked_status": status, "tracked_dirty_sha256": files, "passed": True}


def _initial_state(contract: str, sequences: int) -> Any | None:
    if contract != "fp32_both":
        return None
    import torch
    count = sequences * HEADS * DIM * DIM
    return torch.arange(count, dtype=torch.float32, device="cuda").reshape(sequences, HEADS, DIM, DIM).mul_(1.0 / 8192.0).add_(0.125).contiguous()


def _case(cell: Cell) -> Any:
    return shared.Case(cell.name, cell.form, cell.sequences, HEADS, cell.lengths, "v5_crossmap_public_registry")


@contextmanager
def _c1_enabled(enabled: bool) -> Iterator[None]:
    prior = os.environ.get("C1_B300_FLASH_KDA")
    os.environ["C1_B300_FLASH_KDA"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("C1_B300_FLASH_KDA", None)
        else:
            os.environ["C1_B300_FLASH_KDA"] = prior


def _summary(values: list[float]) -> dict[str, object]:
    if len(values) != SAMPLES or any(type(value) is not float or not math.isfinite(value) or value <= 0.0 for value in values):
        raise AssertionError("invalid CUDA-event sample list")
    ordered = sorted(values)
    def p(q: float) -> float:
        point = (len(ordered) - 1) * q
        low, high = int(point), min(int(point) + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (point - low)
    return {"samples": SAMPLES, "mean_ms": float(statistics.fmean(values)), "p50_ms": float(p(.50)), "p95_ms": float(p(.95)), "p99_ms": float(p(.99))}


def _decision(decision: Mapping[str, object], cell: Cell, *, expected_cache_hit: bool | None) -> dict[str, object]:
    required = {"requested_variant", "chosen_variant", "reason", "extension_sha256", "varlen_cpu_authoritative", "certified_varlen_offsets", "canonical_cache_hit"}
    if set(decision) != required or decision.get("requested_variant") != cell.variant or decision.get("chosen_variant") != cell.variant or decision.get("reason") != cell.reason or decision.get("extension_sha256") != EXPECTED_EXTENSION_SHA:
        raise AssertionError(f"{cell.name}: public C1 decision drift: {decision!r}")
    if "test_only_route" in decision:
        raise AssertionError("test-only route marker is forbidden")
    if cell.form == "fixed":
        if decision.get("varlen_cpu_authoritative") is not False or decision.get("certified_varlen_offsets") is not None or decision.get("canonical_cache_hit") is not None:
            raise AssertionError(f"{cell.name}: fixed decision provenance drift")
    else:
        expected_offsets = [0]
        for length in cell.lengths:
            expected_offsets.append(expected_offsets[-1] + length)
        if decision.get("varlen_cpu_authoritative") is not True or decision.get("certified_varlen_offsets") != expected_offsets:
            raise AssertionError(f"{cell.name}: varlen CPU descriptor provenance drift")
        cache_hit = decision.get("canonical_cache_hit")
        if type(cache_hit) is not bool or expected_cache_hit is not None and cache_hit is not expected_cache_hit:
            raise AssertionError(f"{cell.name}: varlen canonical-cache provenance drift")
    return dict(decision)


def _baseline_decision(decision: Mapping[str, object], cell: Cell) -> dict[str, object]:
    required = {"requested_variant", "chosen_variant", "reason", "extension_sha256", "varlen_cpu_authoritative", "certified_varlen_offsets", "canonical_cache_hit"}
    if set(decision) != required or decision.get("requested_variant") != "baseline" or decision.get("chosen_variant") != "baseline" or decision.get("reason") != cell.reason:
        raise AssertionError(f"{cell.name}: baseline decision drift: {decision!r}")
    if decision.get("extension_sha256") is not None or decision.get("varlen_cpu_authoritative") is not False or decision.get("certified_varlen_offsets") is not None or decision.get("canonical_cache_hit") is not None:
        raise AssertionError(f"{cell.name}: fixed baseline provenance drift")
    return dict(decision)


def _tracked_torch_reference(label: str, torch_ref: Callable[..., Any], x: object, gpu_offsets: object | None, cpu_offsets: object | None) -> tuple[Callable[..., Any], Callable[[], dict[str, object]]]:
    """Wrap exactly one pinned Torch-reference call with input immutability proof.

    The wrapper is deliberately at the callable boundary, so the snapshot covers
    the actual ``initial_state`` object that the reference receives (rather than
    only a caller template).  CPU offsets are captured because Torch ref's
    ABI accepts the GPU offsets only; both are nevertheless included in the
    immutability ledger.
    """
    records: list[dict[str, object]] = []
    def tracked(*args: object, **kwargs: object) -> object:
        if len(args) != 7 or args[0] is not x.q or args[1] is not x.k or args[2] is not x.v or args[3] is not x.g or args[4] is not x.beta or args[5] is not x.scale:
            raise AssertionError(f"{label}: Torch-reference positional input identity drift")
        if kwargs.get("A_log") is not x.a_log or kwargs.get("dt_bias") is not x.dt_bias or kwargs.get("lower_bound") is not x.lower_bound or kwargs.get("cu_seqlens") is not gpu_offsets:
            raise AssertionError(f"{label}: Torch-reference keyword input identity drift")
        initial = kwargs.get("initial_state")
        snapshot = varlen_helper._snapshot_input_tensors(x, gpu_offsets, cpu_offsets, initial)
        result = torch_ref(*args, **kwargs)
        records.append(varlen_helper._assert_input_immutability(label, snapshot, x, gpu_offsets, cpu_offsets, initial))
        return result
    def evidence() -> dict[str, object]:
        if len(records) != 1:
            raise AssertionError(f"{label}: expected exactly one Torch-reference call, observed {len(records)}")
        return records[0]
    return tracked, evidence


def _require_inference_flags(grad_enabled: bool, inference_enabled: bool) -> None:
    if grad_enabled or not inference_enabled:
        raise RuntimeError("packed-varlen registry path must execute inside torch.inference_mode()")


def _require_inference_mode() -> None:
    import torch
    _require_inference_flags(bool(torch.is_grad_enabled()), bool(torch.is_inference_mode_enabled()))


def _call_varlen_positive_inference_guard(call: Callable[[], dict[str, object]], *, mode_check: Callable[[], None] = _require_inference_mode) -> dict[str, object]:
    """Invoke the complete packed-varlen correctness helper only in inference mode.

    Keeping the guard immediately around the delegated helper is intentional:
    the imported helper contains reference, verifier, direct-backend and public
    registry calls, all of which must inherit the same inference-mode contract.
    The runner self-test invokes this guard directly from adversarial contexts.
    """
    mode_check()
    result = call()
    mode_check()
    return result


def _fixed_positive(cell: Cell, x: object, public: Callable[..., Any], originals: Mapping[str, Callable[..., Any]], counts: Mapping[str, int], c1: object, pinned: object, torch_ref: Callable[..., Any]) -> dict[str, object]:
    import torch
    final = cell.contract != "none"
    initial = _initial_state(cell.contract, cell.sequences)
    with torch.inference_mode():
        c1_ok, c1_reason = varlen_helper._verify(c1, x, None if initial is None else initial.clone(), final, None, None)
        pinned_ok, pinned_reason = varlen_helper._verify(pinned, x, None if initial is None else initial.clone(), final, None, None)
        if not c1_ok or not pinned_ok:
            raise AssertionError(f"{cell.name}: backend verifier rejected c1={c1_reason!r} pinned={pinned_reason!r}")
        tracked_reference, reference_evidence = _tracked_torch_reference(cell.name + "/torch_reference", torch_ref, x, None, None)
        reference = varlen_helper._reference(tracked_reference, x, initial, final, cell.sequences)
        torch_reference_immutability = reference_evidence()
        pinned_direct, pinned_immutable = varlen_helper._call_with_immutability(cell.name + "/direct_pinned", originals["pinned"], x, initial, final, None, None)
        with _c1_enabled(True):
            direct, direct_immutable = varlen_helper._call_with_immutability(cell.name + "/direct_c1", originals["c1"], x, initial, final, None, None)
            direct_decision = _decision(auto_dispatch.get_last_decision(), cell, expected_cache_hit=None)
            public_c1_initial = None if initial is None else initial.clone()
            public_c1_snapshot = varlen_helper._snapshot_input_tensors(x, None, None, public_c1_initial)
            public_c1, c1_spy = varlen_helper._spy_public(public, x, public_c1_initial, final, None, None, counts, True, cell.name + "/public_c1")
            public_c1_immutable = varlen_helper._assert_input_immutability(cell.name + "/public_c1", public_c1_snapshot, x, None, None, public_c1_initial)
            public_decision = _decision(auto_dispatch.get_last_decision(), cell, expected_cache_hit=None)
        with _c1_enabled(False):
            public_pinned_initial = None if initial is None else initial.clone()
            public_pinned_snapshot = varlen_helper._snapshot_input_tensors(x, None, None, public_pinned_initial)
            public_pinned, pinned_spy = varlen_helper._spy_public(public, x, public_pinned_initial, final, None, None, counts, False, cell.name + "/public_pinned")
            public_pinned_immutable = varlen_helper._assert_input_immutability(cell.name + "/public_pinned", public_pinned_snapshot, x, None, None, public_pinned_initial)
        torch.cuda.synchronize()
    return {
        "expected_variant": cell.variant, "expected_reason": cell.reason,
        "verifier": {"c1": {"accepted": True, "reason": c1_reason}, "pinned": {"accepted": True, "reason": pinned_reason}},
        "pinned_direct_vs_reference": varlen_helper._exact(pinned_direct, reference, cell.sequences, final, cell.name + "/pinned-ref"),
        "direct_c1_vs_pinned": varlen_helper._exact(direct, pinned_direct, cell.sequences, final, cell.name + "/direct-pinned"),
        "public_c1_vs_pinned": varlen_helper._exact(public_c1, pinned_direct, cell.sequences, final, cell.name + "/public-pinned"),
        "public_pinned_vs_reference": varlen_helper._exact(public_pinned, reference, cell.sequences, final, cell.name + "/public-pinned-ref"),
        "direct_c1_vs_reference": varlen_helper._exact(direct, reference, cell.sequences, final, cell.name + "/direct-ref"),
        "public_c1_vs_reference": varlen_helper._exact(public_c1, reference, cell.sequences, final, cell.name + "/public-ref"),
        "public_c1_spy": c1_spy, "public_pinned_spy": pinned_spy,
        "direct_decision": direct_decision, "public_decision": public_decision,
        "torch_reference_immutability": torch_reference_immutability,
        "input_immutability": {"direct_pinned": pinned_immutable, "direct_c1": direct_immutable, "public_c1": public_c1_immutable, "public_pinned": public_pinned_immutable},
        "handoff_cache": {"applicable": False, "reason": "fixed_batch_has_no_CPU_varlen_descriptor", "passed": True}, "passed": True,
    }


def _varlen_positive(cell: Cell, x: object, public: Callable[..., Any], originals: Mapping[str, Callable[..., Any]], counts: Mapping[str, int], c1: object, pinned: object, torch_ref: Callable[..., Any], seed: int) -> dict[str, object]:
    import torch
    # CPU descriptor construction is also part of the packed-varlen call
    # chain, so it stays inside this scope rather than preceding it.
    with torch.inference_mode(), _c1_enabled(True):
        _require_inference_mode()
        cpu = varlen_helper._cpu_offsets(cell.lengths)
        gpu = x.cu_seqlens
        if gpu is None:
            raise AssertionError("varlen GPU offsets absent")
        helper_cell = varlen_helper.Cell(_case(cell), cell.contract, cell.variant)
        tracked_reference, reference_evidence = _tracked_torch_reference(cell.name + "/torch_reference", torch_ref, x, gpu, cpu)
        record = _call_varlen_positive_inference_guard(
            lambda: varlen_helper._positive_cell(helper_cell, x, cpu, gpu, originals, counts, public, c1, pinned, tracked_reference, seed)
        )
    if record.get("passed") is not True:
        raise AssertionError("varlen production correctness helper did not pass")
    _decision(record["direct_decision"], cell, expected_cache_hit=False)
    _decision(record["public_decision"], cell, expected_cache_hit=True)
    if record.get("public_c1_spy", {}).get("delta") != {"c1": 1, "pinned": 0} or record.get("public_pinned_spy", {}).get("delta") != {"c1": 0, "pinned": 1}:
        raise AssertionError("varlen public registry route spy drift")
    if not isinstance(record.get("public_handoff_prepare"), Mapping) or not isinstance(record.get("cache_stats"), Mapping):
        raise AssertionError("varlen handoff/cache evidence missing")
    verifier = record.get("verifier")
    if not isinstance(verifier, Mapping) or not all(isinstance(verifier.get(name), Mapping) and verifier[name].get("passed") is True for name in ("c1", "pinned")):
        raise AssertionError("varlen verifier evidence missing")
    normalized_verifier = {name: {"accepted": True, "reason": verifier[name].get("reason")} for name in ("c1", "pinned")}
    handoff_cache = {
        "public_prepare": record["public_handoff_prepare"],
        "cache_stats": record["cache_stats"],
        "direct_canonical_cache_hit": False,
        "public_canonical_cache_hit": True,
        "direct_miss_to_public_hit": True,
        "passed": True,
    }
    return {
        "expected_variant": cell.variant,
        "expected_reason": cell.reason,
        "verifier": normalized_verifier,
        "pinned_vs_torch_ref": record["pinned_vs_torch_ref"],
        "direct_c1_vs_pinned": record["direct_c1_vs_pinned"],
        "public_vs_pinned": record["public_vs_pinned"],
        "public_pinned_vs_torch_ref": record["public_pinned_vs_torch_ref"],
        "direct_c1_vs_torch_ref": record["direct_c1_vs_torch_ref"],
        "public_vs_torch_ref": record["public_vs_torch_ref"],
        "public_c1_spy": record["public_c1_spy"],
        "public_pinned_spy": record["public_pinned_spy"],
        "direct_decision": record["direct_decision"],
        "public_decision": record["public_decision"],
        "input_immutability_by_path": record["input_immutability_by_path"],
        "torch_reference_immutability": reference_evidence(),
        "handoff_cache": handoff_cache,
        "passed": True,
    }


def _timing(cell: Cell, x: object, public: Callable[..., Any]) -> dict[str, object]:
    import torch
    final = cell.contract != "none"
    gpu = x.cu_seqlens if cell.form == "varlen" else None
    cpu = varlen_helper._cpu_offsets(cell.lengths) if cell.form == "varlen" else None
    for warmup in range(WARMUP):
        c1 = warmup % 2 == 0
        with _c1_enabled(c1), torch.inference_mode():
            varlen_helper._call(public, x, _initial_state(cell.contract, cell.sequences), final, gpu, cpu)
        if c1:
            _decision(auto_dispatch.get_last_decision(), cell, expected_cache_hit=None)
    torch.cuda.synchronize()
    samples = {"pinned_public": [], "c1_public": []}
    first = {"pinned_public": 0, "c1_public": 0}
    stream = torch.cuda.current_stream()
    for index in range(SAMPLES):
        order = ("pinned_public", "c1_public") if index % 2 == 0 else ("c1_public", "pinned_public")
        first[order[0]] += 1
        for path in order:
            enabled = path == "c1_public"
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            initial = _initial_state(cell.contract, cell.sequences)
            with _c1_enabled(enabled), torch.inference_mode():
                start.record(stream); start.synchronize()
                result = varlen_helper._call(public, x, initial, final, gpu, cpu)
                end.record(stream)
            end.synchronize()
            del result, initial
            if enabled:
                _decision(auto_dispatch.get_last_decision(), cell, expected_cache_hit=None)
            samples[path].append(float(start.elapsed_time(end)))
    paths = {name: _summary(values) for name, values in samples.items()}
    ratios = {q: float(paths["pinned_public"][q + "_ms"]) / float(paths["c1_public"][q + "_ms"]) for q in PERCENTILES}
    if not all(value > 1.0 for value in ratios.values()):
        raise AssertionError(f"{cell.name}: regression gate requires pinned/C1 > 1 at every percentile: {ratios!r}")
    return {"round_index": 0, "event_contract": "one uninstrumented real FLA public call per CUDA-event sample; spy and source checks excluded", "warmup": WARMUP, "samples_per_path": SAMPLES, "first_path_counts": first, "raw_samples_ms": samples, "paths": paths, "pinned_over_c1_by_percentile": ratios, "regression_gate": "pinned/C1 > 1 at P50/P95/P99 (non-release gate)", "passed": True}


def _fixed_baseline_control(cell: Cell, x: object, public: Callable[..., Any], originals: Mapping[str, Callable[..., Any]], counts: Mapping[str, int], c1: object, pinned: object, torch_ref: Callable[..., Any]) -> dict[str, object]:
    """A real registry call may enter C1, but dispatcher policy must stay baseline."""
    import torch
    final = cell.contract != "none"
    initial = _initial_state(cell.contract, cell.sequences)
    with torch.inference_mode():
        c1_ok, c1_reason = varlen_helper._verify(c1, x, None if initial is None else initial.clone(), final, None, None)
        pinned_ok, pinned_reason = varlen_helper._verify(pinned, x, None if initial is None else initial.clone(), final, None, None)
        if not c1_ok or not pinned_ok:
            raise AssertionError(f"{cell.name}: baseline verifier drift c1={c1_reason!r} pinned={pinned_reason!r}")
        tracked_reference, reference_evidence = _tracked_torch_reference(cell.name + "/torch_reference", torch_ref, x, None, None)
        reference = varlen_helper._reference(tracked_reference, x, initial, final, cell.sequences)
        torch_reference_immutability = reference_evidence()
        direct_pinned, direct_pinned_immutable = varlen_helper._call_with_immutability(cell.name + "/direct_pinned", originals["pinned"], x, initial, final, None, None)
        with _c1_enabled(True):
            direct_c1, direct_c1_immutable = varlen_helper._call_with_immutability(cell.name + "/direct_c1", originals["c1"], x, initial, final, None, None)
            direct_decision = _baseline_decision(auto_dispatch.get_last_decision(), cell)
            c1_initial = None if initial is None else initial.clone()
            c1_snapshot = varlen_helper._snapshot_input_tensors(x, None, None, c1_initial)
            public_c1, c1_spy = varlen_helper._spy_public(public, x, c1_initial, final, None, None, counts, True, cell.name + "/public_c1_baseline")
            c1_immutable = varlen_helper._assert_input_immutability(cell.name + "/public_c1_baseline", c1_snapshot, x, None, None, c1_initial)
            public_decision = _baseline_decision(auto_dispatch.get_last_decision(), cell)
        with _c1_enabled(False):
            pinned_initial = None if initial is None else initial.clone()
            pinned_snapshot = varlen_helper._snapshot_input_tensors(x, None, None, pinned_initial)
            public_pinned, pinned_spy = varlen_helper._spy_public(public, x, pinned_initial, final, None, None, counts, False, cell.name + "/public_pinned")
            pinned_immutable = varlen_helper._assert_input_immutability(cell.name + "/public_pinned", pinned_snapshot, x, None, None, pinned_initial)
        torch.cuda.synchronize()
    return {
        "expected_variant": "baseline", "expected_reason": cell.reason,
        "verifier": {"c1": {"accepted": True, "reason": c1_reason}, "pinned": {"accepted": True, "reason": pinned_reason}},
        "direct_pinned_vs_reference": varlen_helper._exact(direct_pinned, reference, cell.sequences, final, cell.name + "/pinned-ref"),
        "direct_c1_vs_pinned": varlen_helper._exact(direct_c1, direct_pinned, cell.sequences, final, cell.name + "/direct-pinned"),
        "public_c1_vs_pinned": varlen_helper._exact(public_c1, direct_pinned, cell.sequences, final, cell.name + "/public-pinned"),
        "public_pinned_vs_reference": varlen_helper._exact(public_pinned, reference, cell.sequences, final, cell.name + "/public-pinned-ref"),
        "direct_c1_vs_reference": varlen_helper._exact(direct_c1, reference, cell.sequences, final, cell.name + "/direct-ref"),
        "public_c1_vs_reference": varlen_helper._exact(public_c1, reference, cell.sequences, final, cell.name + "/public-ref"),
        "direct_decision": direct_decision, "public_decision": public_decision,
        "public_c1_spy": c1_spy, "public_pinned_spy": pinned_spy,
        "torch_reference_immutability": torch_reference_immutability,
        "c1_backend_entry": {"direct": True, "public": True, "passed": True},
        "accelerated_variants_forbidden": ["vshard2_p2", "vshard4_p2"],
        "no_accelerated_variant_selected_or_launched": True,
        "input_immutability": {"direct_pinned": direct_pinned_immutable, "direct_c1": direct_c1_immutable, "public_c1": c1_immutable, "public_pinned": pinned_immutable},
        "handoff_cache": {"applicable": False, "reason": "fixed_batch_has_no_CPU_varlen_descriptor", "passed": True}, "passed": True,
    }


def _zero_cache(value: object, label: str) -> dict[str, int]:
    if type(value) is not dict or not value or any(type(name) is not str or type(count) is not int or count != 0 for name, count in value.items()):
        raise AssertionError(f"{label}: cache must be exact zero integer statistics")
    return dict(value)


def _varlen_adjacent_baseline_control(cell: Cell, x: object, public: Callable[..., Any], originals: Mapping[str, Callable[..., Any]], counts: Mapping[str, int], c1: object, pinned: object, torch_ref: Callable[..., Any]) -> dict[str, object]:
    """Prove a current neighbour is rejected by C1 verifier and registry-falls back.

    This code deliberately never reads ``get_last_decision``: a verifier rejection
    does not launch the C1 dispatcher, so a prior thread-local decision would be
    stale evidence.
    """
    import torch
    cpu = varlen_helper._cpu_offsets(cell.lengths)
    gpu = x.cu_seqlens
    if gpu is None:
        raise AssertionError("adjacent varlen GPU offsets absent")
    clear = getattr(c1, "_clear_varlen_handoff", None)
    local = getattr(c1, "_handoff_local", None)
    if not callable(clear) or not callable(local):
        raise AssertionError("adjacent control requires C1 handoff clear API")
    final, initial = True, _initial_state(cell.contract, cell.sequences)
    original_issue, original_verifier = varlen_metadata.issue_descriptor, c1.chunk_kda_verifier
    issued: list[dict[str, object]] = []
    verifier_calls: list[dict[str, object]] = []
    def argument(args: tuple[object, ...], kwargs: Mapping[str, object], index: int, name: str) -> object:
        return args[index] if len(args) > index else kwargs.get(name)
    def issue_spy(*call_args: object, **kwargs: object) -> object:
        descriptor = original_issue(*call_args, **kwargs)
        q_arg = argument(call_args, kwargs, 0, "q")
        cpu_arg = argument(call_args, kwargs, 1, "cu_seqlens_cpu")
        if q_arg is not x.q or cpu_arg is not cpu:
            raise AssertionError("adjacent issuer did not receive current CPU offsets")
        facts = varlen_metadata.verify_descriptor(descriptor, cpu_arg)
        if tuple(facts.offsets) != tuple(cpu.tolist()):
            raise AssertionError("adjacent issuer certified stale/wrong offsets")
        issued.append({"cpu_tensor_identity": True, "certified_offsets": list(facts.offsets)})
        return descriptor
    def verifier_spy(*call_args: object, **kwargs: object) -> object:
        q_arg = argument(call_args, kwargs, 0, "q")
        gpu_arg = argument(call_args, kwargs, 17, "cu_seqlens")
        cpu_arg = argument(call_args, kwargs, 18, "cu_seqlens_cpu")
        if q_arg is not x.q or gpu_arg is not gpu or cpu_arg is not cpu:
            raise AssertionError("adjacent verifier did not receive current offset tensors")
        outcome = original_verifier(*call_args, **kwargs)
        if type(outcome) is not tuple or len(outcome) != 2 or type(outcome[0]) is not bool or (outcome[1] is not None and type(outcome[1]) is not str):
            raise AssertionError("adjacent verifier return schema drift")
        verifier_calls.append({"accepted": outcome[0], "reason": outcome[1], "cpu_tensor_identity": True, "gpu_tensor_identity": True})
        return outcome
    cache_before_clear = dict(varlen_metadata.cache_stats())
    handoff_empty_after_clear = False
    handoff_empty_after_public = False
    cache_after_cleanup: dict[str, int] | None = None
    issuer_spy_restored = False
    verifier_spy_restored = False
    try:
        clear(); varlen_metadata.clear_cache()
        cache_after_clear = _zero_cache(varlen_metadata.cache_stats(), "adjacent/cache_after_clear")
        if hasattr(local(), "plan"):
            raise AssertionError("adjacent control retained a stale C1 handoff before public call")
        handoff_empty_after_clear = True
        varlen_metadata.issue_descriptor, c1.chunk_kda_verifier = issue_spy, verifier_spy
        with torch.inference_mode(), _c1_enabled(True):
            _require_inference_mode()
            tracked_reference, reference_evidence = _tracked_torch_reference(cell.name + "/torch_reference", torch_ref, x, gpu, cpu)
            reference = varlen_helper._reference(tracked_reference, x, initial, final, cell.sequences)
            torch_reference_immutability = reference_evidence()
            direct_pinned, direct_pinned_immutable = varlen_helper._call_with_immutability(cell.name + "/direct_pinned", originals["pinned"], x, initial, final, gpu, cpu)
            public_initial = initial.clone()
            public_snapshot = varlen_helper._snapshot_input_tensors(x, gpu, cpu, public_initial)
            public_c1, public_c1_spy = varlen_helper._spy_public(public, x, public_initial, final, gpu, cpu, counts, False, cell.name + "/public_C1_rejected")
            public_c1_immutable = varlen_helper._assert_input_immutability(cell.name + "/public_C1_rejected", public_snapshot, x, gpu, cpu, public_initial)
        if len(issued) != 1 or len(verifier_calls) != 1:
            raise AssertionError("adjacent control requires exactly one current issuer/verifier call")
        expected_reason = "C1 packed-varlen preflight rejected: varlen_offsets_not_whitelisted"
        if verifier_calls[0]["accepted"] is not False or verifier_calls[0]["reason"] != expected_reason:
            raise AssertionError(f"adjacent verifier wrong rejection: {verifier_calls!r}")
        if hasattr(local(), "plan"):
            raise AssertionError("adjacent public fallback left a stale C1 handoff")
        handoff_empty_after_public = True
        with torch.inference_mode(), _c1_enabled(False):
            pinned_initial = initial.clone()
            pinned_snapshot = varlen_helper._snapshot_input_tensors(x, gpu, cpu, pinned_initial)
            public_pinned, public_pinned_spy = varlen_helper._spy_public(public, x, pinned_initial, final, gpu, cpu, counts, False, cell.name + "/public_pinned")
            public_pinned_immutable = varlen_helper._assert_input_immutability(cell.name + "/public_pinned", pinned_snapshot, x, gpu, cpu, pinned_initial)
        torch.cuda.synchronize()
    finally:
        varlen_metadata.issue_descriptor = original_issue
        issuer_spy_restored = varlen_metadata.issue_descriptor is original_issue
        if "chunk_kda_verifier" in vars(c1):
            delattr(c1, "chunk_kda_verifier")
        verifier_spy_restored = getattr(original_verifier, "__self__", None) is c1 and getattr(c1.chunk_kda_verifier, "__self__", None) is c1 and getattr(original_verifier, "__func__", None) is getattr(c1.chunk_kda_verifier, "__func__", None)
        clear(); varlen_metadata.clear_cache()
        cache_after_cleanup = _zero_cache(varlen_metadata.cache_stats(), "adjacent/cache_after_cleanup")
    if not issuer_spy_restored or not verifier_spy_restored:
        raise AssertionError("adjacent verifier spy restoration drift")
    if len(issued) != 1 or len(verifier_calls) != 1:
        raise AssertionError("adjacent control did not retain exactly one issuer/verifier observation")
    verifier_evidence = {
        "call_count": 1,
        "q_tensor_identity": True,
        "gpu_offsets_tensor_identity": True,
        "cpu_offsets_tensor_identity": True,
        "accepted": verifier_calls[0]["accepted"],
        "reason": verifier_calls[0]["reason"],
        "issuer_call_count": 1,
        "issuer_cpu_offsets_tensor_identity": True,
        "certified_offsets": issued[0]["certified_offsets"],
        "issuer_spy_restored": True,
        "verifier_spy_restored": True,
        "passed": True,
    }
    return {
        "expected_variant": "baseline", "expected_reason": "varlen_offsets_not_whitelisted",
        "c1_verifier": verifier_evidence, "public_c1_spy": public_c1_spy, "public_pinned_spy": public_pinned_spy,
        "direct_pinned_vs_reference": varlen_helper._exact(direct_pinned, reference, cell.sequences, True, cell.name + "/pinned-ref"),
        "public_c1_fallback_vs_pinned": varlen_helper._exact(public_c1, direct_pinned, cell.sequences, True, cell.name + "/c1fallback-pinned"),
        "public_c1_fallback_vs_reference": varlen_helper._exact(public_c1, reference, cell.sequences, True, cell.name + "/c1fallback-ref"),
        "public_pinned_vs_reference": varlen_helper._exact(public_pinned, reference, cell.sequences, True, cell.name + "/public-pinned-ref"),
        "input_immutability": {"direct_pinned": direct_pinned_immutable, "public_c1_fallback": public_c1_immutable, "public_pinned": public_pinned_immutable},
        "torch_reference_immutability": torch_reference_immutability,
        "handoff_cache": {
            "clear_handoff_api": "C1B300FlashKDABackend._clear_varlen_handoff",
            "metadata_clear_api": "varlen_metadata.clear_cache",
            "cache_before_clear": cache_before_clear,
            "cache_after_clear": cache_after_clear,
            "handoff_empty_after_clear": handoff_empty_after_clear,
            "handoff_empty_after_public": handoff_empty_after_public,
            "cache_after_cleanup": cache_after_cleanup,
            "issuer": issued[0],
            "verifier": verifier_evidence,
            "last_decision_read": False,
            "spies_restored": issuer_spy_restored and verifier_spy_restored,
            "passed": True,
        },
        "no_accelerated_variant_selected_or_launched": True, "passed": True,
    }


def _public_registry_negative_controls(seed: int, public: Callable[..., Any], originals: Mapping[str, Callable[..., Any]], counts: Mapping[str, int], c1: object, pinned: object, torch_ref: Callable[..., Any]) -> dict[str, object]:
    specs = (
        Cell("b7_none", "fixed", 7, 2048, (2048,) * 7, "none", "baseline", "fixed_batch_shape_not_whitelisted"),
        Cell("t8191_fp32_both", "fixed", 1, 8191, (8191,), "fp32_both", "baseline", "state_contract_fp32_both_h12_length_not_whitelisted"),
        Cell("adjacent_offsets_fp32_both", "varlen", 1, 12288, (1, 1, 1, 1, 2, 12282), "fp32_both", "baseline", "varlen_offsets_not_whitelisted"),
    )
    controls: dict[str, object] = {}
    for index, cell in enumerate(specs):
        x = shared._make_inputs(_case(cell), seed + index)
        try:
            controls[cell.name] = _varlen_adjacent_baseline_control(cell, x, public, originals, counts, c1, pinned, torch_ref) if cell.form == "varlen" else _fixed_baseline_control(cell, x, public, originals, counts, c1, pinned, torch_ref)
        finally:
            del x
    return controls


def _initial(args: argparse.Namespace) -> dict[str, object]:
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdecimal() or int(job) <= 0:
        raise RuntimeError("SLURM_JOB_ID must be a positive decimal string")
    return {"schema_version": SCHEMA_VERSION, "purpose": "v5 public-registry cross-map read-only regression; no production source or map mutation", "allocation_id": args.allocation_id, "process_index": args.process_index, "allocation": {"slurm_job_id": job, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}, "process": {"pid": os.getpid(), "fresh_python_process_required": True}, "positive_cells": {}, "public_registry_negative_controls": {}, "source_pre_torch": {}, "bootstrap": {}, "source_pre": {}, "source_post": {}, "map_pre": {}, "map_post": {}, "map_readonly": {}, "identity": {}, "performance": {}, "complete": False}


def _assert_no_top_level_heavy_imports() -> None:
    """Reject an accidental return to module-scope Torch/FLA/helper imports."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=str(Path(__file__)))
    forbidden = tuple(HEAVY_RUNTIME_IMPORTS.values())
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            if name == "torch" or name.startswith("torch.") or name == "fla" or name.startswith("fla.") or name in forbidden:
                raise RuntimeError(f"top-level heavy import is forbidden before clean gate: {name}")


def _self_test() -> None:
    if len(POSITIVES) != 4 or len(EXPECTED_POSITIVE_KEYS) != 4 or any(cell.variant == "baseline" for cell in POSITIVES):
        raise AssertionError("positive preregistration scope drift")
    for expected in (EXPECTED_AUTO_SHA, EXPECTED_BACKEND_SHA, EXPECTED_EXTENSION_SHA, EXPECTED_HELPER_SHA):
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise AssertionError("SHA literal drift")
    try:
        _summary([1.0] * (SAMPLES - 1))
    except AssertionError:
        pass
    else:
        raise AssertionError("summary accepted wrong sample count")
    for grad_enabled, inference_enabled in ((True, True), (False, False)):
        try:
            _require_inference_flags(grad_enabled, inference_enabled)
        except RuntimeError:
            pass
        else:
            raise AssertionError("varlen inference guard accepted an adversarial non-inference context")
    _require_inference_flags(False, True)
    baseline_cell = Cell("selftest_baseline", "fixed", 7, 2048, (2048,) * 7, "none", "baseline", "fixed_batch_shape_not_whitelisted")
    baseline = {
        "requested_variant": "baseline", "chosen_variant": "baseline", "reason": baseline_cell.reason,
        "extension_sha256": None, "varlen_cpu_authoritative": False,
        "certified_varlen_offsets": None, "canonical_cache_hit": None,
    }
    _baseline_decision(baseline, baseline_cell)
    try:
        _baseline_decision({**baseline, "extension_sha256": EXPECTED_EXTENSION_SHA}, baseline_cell)
    except AssertionError:
        pass
    else:
        raise AssertionError("baseline decision accepted a non-null extension provenance")
    _assert_no_top_level_heavy_imports()
    pristine = _pre_torch_import_state()
    if pristine != {"torch_modules_before_gate": [], "heavy_modules_before_gate": [], "passed": True}:
        raise AssertionError("runner self-test did not start before every Torch/FLA/helper import")
    import_events: list[str] = []
    invalid_gate = {**pristine, "passed": False}
    try:
        _import_named_modules_after_clean_gate(invalid_gate, {"heavy_helper": "fake.heavy_helper"}, lambda name: import_events.append(name))
    except RuntimeError:
        pass
    else:
        raise AssertionError("heavy import gateway accepted a missing clean gate")
    if import_events:
        raise AssertionError("heavy import gateway invoked an importer before validating clean GPU evidence")
    valid_gate = {
        "index": "0", "uuid": "GPU-selftest", "name": "B300", "compute_capability": "10.3",
        "memory_used_mib": 0, "compute_apps": [], **pristine,
    }
    imported = _import_named_modules_after_clean_gate(
        valid_gate, {"heavy_helper": "fake.heavy_helper"}, lambda name: import_events.append(name) or object(),
    )
    if list(imported) != ["heavy_helper"] or import_events != ["fake.heavy_helper"]:
        raise AssertionError("heavy helper import did not occur strictly after the clean-gate validation")
    # Direct adversarial probes for job12911's exact failure mode.  This
    # validates only the one function that this runner directly calls; it
    # intentionally makes no assertion beyond that narrow binding.
    module_name = HEAVY_RUNTIME_IMPORTS[SHARED_SEQCOUNT_LABEL]
    saved_shared = sys.modules.get(module_name)
    saved_torch = sys.modules.get("torch")
    fake_torch = types.ModuleType("torch")
    fake_shared = types.ModuleType(module_name)
    fake_shared.__file__ = str(Path(__file__).resolve())
    try:
        sys.modules[module_name] = fake_shared
        sys.modules["torch"] = fake_torch
        exec("def _make_inputs():\n    return torch\n", fake_shared.__dict__)
        try:
            fake_shared._make_inputs()
        except NameError:
            pass
        else:
            raise AssertionError("TYPE_CHECKING-only shared helper unexpectedly ran without torch")
        pre, _ = _shared_make_inputs_binding(
            fake_shared, fake_torch, phase="pre_hydration", expect_torch_global=False,
        )
        if pre["torch_global_present"] is not False or pre["torch_object_id"] is not None:
            raise AssertionError("shared hydration pre-record did not prove missing torch")
        # A detached callable can have the right name but must not be accepted
        # as the canonical module function.
        detached_namespace = {"__name__": module_name}
        exec("def _make_inputs():\n    return torch\n", detached_namespace)
        canonical_function = fake_shared._make_inputs
        fake_shared._make_inputs = detached_namespace["_make_inputs"]
        try:
            _shared_make_inputs_binding(
                fake_shared, fake_torch, phase="pre_hydration", expect_torch_global=False,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("shared hydration accepted a detached function global namespace")
        fake_shared._make_inputs = canonical_function
        # A module object that is no longer the canonical sys.modules object is
        # also forbidden, even though its source/name look correct.
        sys.modules[module_name] = object()
        try:
            _shared_make_inputs_binding(
                fake_shared, fake_torch, phase="pre_hydration", expect_torch_global=False,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("shared hydration accepted a detached canonical module")
        sys.modules[module_name] = fake_shared
        # A noncanonical value under the exact global name must fail rather
        # than silently replacing it.
        fake_shared.__dict__["torch"] = object()
        try:
            _shared_make_inputs_binding(
                fake_shared, fake_torch, phase="bound_pre_workload", expect_torch_global=True,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("shared hydration accepted a non-canonical torch global")
        del fake_shared.__dict__["torch"]
        fake_heavy_modules = {key: object() for key in HEAVY_RUNTIME_IMPORTS} | {SHARED_SEQCOUNT_LABEL: fake_shared}
        try:
            _hydrate_shared_make_inputs_after_clean_gate({**valid_gate, "passed": False}, fake_heavy_modules, fake_torch)
        except RuntimeError:
            pass
        else:
            raise AssertionError("shared hydration ran without the clean-gate evidence")
        if "torch" in fake_shared.__dict__:
            raise AssertionError("shared hydration mutated a helper before validating clean-gate evidence")
        hydration, state = _hydrate_shared_make_inputs_after_clean_gate(valid_gate, fake_heavy_modules, fake_torch)
        if hydration["bound_pre_workload"]["torch_object_id"] != id(fake_torch):
            raise AssertionError("shared hydration did not bind the canonical torch object")
        sys.modules["torch"] = types.ModuleType("torch")
        try:
            _verify_shared_make_inputs_post_workload(state, fake_torch)
        except RuntimeError:
            pass
        else:
            raise AssertionError("shared hydration accepted sys.modules torch identity drift")
        sys.modules["torch"] = fake_torch
        # Replacing the function after binding is a detached/noncanonical
        # artifact and must be detected by the post-workload reopen.
        exec("def _make_inputs():\n    return torch\n", fake_shared.__dict__)
        try:
            _verify_shared_make_inputs_post_workload(state, fake_torch)
        except RuntimeError:
            pass
        else:
            raise AssertionError("shared hydration accepted a post-workload function replacement")
        fake_shared._make_inputs = state["function"]
        post = _verify_shared_make_inputs_post_workload(state, fake_torch)
        if post["torch_object_id"] != id(fake_torch):
            raise AssertionError("shared post-workload binding lost canonical torch identity")
    finally:
        if saved_shared is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = saved_shared
        if saved_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved_torch
    # Direct adversarial probe: the same helper-level guard used by
    # ``_varlen_positive`` must reject calls outside inference-mode, while the
    # enclosing inference scope must propagate all the way into the callback.
    def reject_non_inference() -> None:
        raise RuntimeError("adversarial non-inference context")
    try:
        _call_varlen_positive_inference_guard(lambda: {"should_not": "run"}, mode_check=reject_non_inference)
    except RuntimeError:
        pass
    else:
        raise AssertionError("varlen positive helper guard accepted direct non-inference call")
    guard_calls: list[str] = []
    def observe_inference() -> None:
        guard_calls.append("checked")
    _call_varlen_positive_inference_guard(lambda: {"passed": True}, mode_check=observe_inference)
    if guard_calls != ["checked", "checked"]:
        raise AssertionError("varlen positive helper guard failed to cover pre/post helper execution")
    map_pre = {"entries": {"fixed_batch": [], "fixed_single_batch": [], "varlen": []}, "object_ids": {"fixed_batch": 1, "fixed_single_batch": 2, "varlen": 3}, "digest": "0" * 64}
    map_post = {**map_pre, "object_ids": {"fixed_batch": 4, "fixed_single_batch": 2, "varlen": 3}}
    try:
        _assert_map_readonly(map_pre, map_post)
    except RuntimeError:
        pass
    else:
        raise AssertionError("map identity guard accepted a raw-PID object replacement")
    print("RUNNER_SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-id", choices=("A1", "A2"))
    parser.add_argument("--process-index", type=int, choices=(0, 1))
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--patched-root", type=Path)
    parser.add_argument("--fla-root", type=Path)
    parser.add_argument("--analyzer-path", type=Path)
    parser.add_argument("--protocol-shell-path", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.allocation_id, args.process_index, args.reference_root, args.json)):
            parser.error("--self-test cannot combine with GPU arguments")
        _self_test(); return
    if any(value is None for value in (args.allocation_id, args.process_index, args.reference_root, args.patched_root, args.fla_root, args.analyzer_path, args.protocol_shell_path, args.json)):
        parser.error("all allocation/source/json arguments are required")
    if os.environ.get("C1_V5_CROSSMAP_CLEAN_GPU") != "1" or os.environ.get("C1_B300_FLASH_KDA") != "1" or os.environ.get("FLA_FLASH_KDA") != "1" or os.environ.get("C1_B300_VARLEN_CPU_DESCRIPTOR") != "1":
        raise RuntimeError("clean shell plus C1/FLA/CPU-descriptor opt-ins are required")
    result = _initial(args); _write(args.json, result)
    # This is intentionally the first point where any source-dependent action
    # occurs.  It uses only canonical paths plus read_bytes/SHA; the assertions
    # prove that no Torch/FLA/helper import has preceded the 0-MiB GPU gate.
    _assert_no_top_level_heavy_imports()
    pre_import_state = _pre_torch_import_state()
    result["source_pre_torch"] = _source_ledger_pre_torch(args)
    result["identity"]["pre_torch_clean_gpu"] = _pre_torch_clean_gpu(pre_import_state)
    modules, loaded_identity_pre = _load_runtime_modules_after_clean_gate(
        result["identity"]["pre_torch_clean_gpu"], result["source_pre_torch"],
    )
    torch = _import_named_modules_after_clean_gate(
        result["identity"]["pre_torch_clean_gpu"], {"torch": "torch"},
    )["torch"]
    helper_runtime_globals_pre, shared_make_inputs_binding = _hydrate_shared_make_inputs_after_clean_gate(
        result["identity"]["pre_torch_clean_gpu"], modules, torch,
    )
    global auto_dispatch, fla_backend, varlen_metadata, shared, confirmation, varlen_helper, tail_helper
    auto_dispatch = modules["auto_dispatch"]
    fla_backend = modules["fla_backend"]
    varlen_metadata = modules["varlen_metadata"]
    shared = modules["shared_seqcount"]
    confirmation = modules["confirmation"]
    varlen_helper = modules["varlen_helper"]
    tail_helper = modules["tail_helper"]
    common = modules["harness"]
    chunk_kda = getattr(modules["fla_ops_kda"], "chunk_kda", None)
    if not callable(chunk_kda):
        raise RuntimeError("canonical fla.ops.kda.chunk_kda public callable is absent")
    result["bootstrap"] = {
        "source_ledger_mode": "canonical_path_sha256_without_module_import",
        "stages": BOOTSTRAP_STAGES,
        "heavy_runtime_import_after_clean_gate": True,
        "passed": True,
    }
    result["source_pre"] = _runtime_source(result["source_pre_torch"], loaded_identity_pre)
    result["map_pre"] = _canonical_maps(); _write(args.json, result)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result["identity"]["runtime"] = varlen_helper._identity(args.patched_root, args.fla_root, args.reference_root)
    result["identity"]["patched_dirty_set"] = _dirty_set(args.patched_root, EXPECTED_PATCHED_COMMIT, require_dirty=True)
    result["identity"]["reference_clean"] = _dirty_set(args.reference_root, EXPECTED_PATCHED_COMMIT, require_dirty=False)
    result["identity"]["fla_clean"] = _dirty_set(args.fla_root, EXPECTED_FLA_COMMIT, require_dirty=False)
    helper_identity = tail_helper._pinned_reference_helper_identity()
    if helper_identity != {"path": EXPECTED_HELPER_PATH, "sha256": EXPECTED_HELPER_SHA}:
        raise RuntimeError("pinned helper identity drift")
    torch_ref, helper_proof = tail_helper._load_pinned_reference_without_build(common, args.reference_root, helper_identity)
    tail_helper._validate_pinned_reference_helper_load(helper_identity, helper_proof)
    result["identity"]["pinned_reference_helper"] = {"identity": helper_identity, "load_proof": helper_proof}
    c1, pinned, _registry, registry = varlen_helper._registry_backends()
    originals = {"c1": c1.chunk_kda, "pinned": pinned.chunk_kda}
    original_slots = {"c1": vars(c1).get("chunk_kda"), "pinned": vars(pinned).get("chunk_kda")}
    counts = {"c1": 0, "pinned": 0}
    def c1_spy(*call_args: object, **kwargs: object) -> object:
        counts["c1"] += 1; return originals["c1"](*call_args, **kwargs)
    def pinned_spy(*call_args: object, **kwargs: object) -> object:
        counts["pinned"] += 1; return originals["pinned"](*call_args, **kwargs)
    c1.chunk_kda, pinned.chunk_kda = c1_spy, pinned_spy
    result["identity"]["registry"] = {"public_callable": "fla.ops.kda.chunk_kda", "snapshot": registry, "test_only_route_installed": False}
    primary: BaseException | None = None
    try:
        result["public_registry_negative_controls"] = _public_registry_negative_controls(args.seed + args.process_index * 1000, chunk_kda, originals, counts, c1, pinned, torch_ref)
        for index, cell in enumerate(POSITIVES):
            x = shared._make_inputs(_case(cell), args.seed + args.process_index * 100_000 + index)
            try:
                correctness = _varlen_positive(cell, x, chunk_kda, originals, counts, c1, pinned, torch_ref, args.seed) if cell.form == "varlen" else _fixed_positive(cell, x, chunk_kda, originals, counts, c1, pinned, torch_ref)
                result["positive_cells"][cell.name] = correctness
                # Timed calls must run after the correctness spies have been removed.
                c1.chunk_kda, pinned.chunk_kda = originals["c1"], originals["pinned"]
                result["performance"][cell.name] = [_timing(cell, x, chunk_kda)]
                c1.chunk_kda, pinned.chunk_kda = c1_spy, pinned_spy
            finally:
                del x; torch.cuda.empty_cache()
    except BaseException as exc:
        primary = exc; result["failure"] = {"type": type(exc).__name__, "message": str(exc)}; raise
    finally:
        if original_slots["c1"] is None:
            vars(c1).pop("chunk_kda", None)
        else:
            c1.chunk_kda = original_slots["c1"]
        if original_slots["pinned"] is None:
            vars(pinned).pop("chunk_kda", None)
        else:
            pinned.chunk_kda = original_slots["pinned"]
        result["identity"]["registry_spies_restored"] = {"c1": c1.chunk_kda == originals["c1"], "pinned": pinned.chunk_kda == originals["pinned"], "passed": c1.chunk_kda == originals["c1"] and pinned.chunk_kda == originals["pinned"]}
        try:
            result["map_post"] = _canonical_maps()
            source_post_ledger = _source_ledger_pre_torch(args)
            result["source_post"] = _runtime_source(source_post_ledger, _loaded_module_identity(modules, source_post_ledger))
            result["map_readonly"] = _assert_map_readonly(result["map_pre"], result["map_post"])
            if result["source_pre"] != result["source_post"]:
                raise RuntimeError("read-only source identity changed during workload")
            helper_runtime_globals_post = _verify_shared_make_inputs_post_workload(shared_make_inputs_binding, torch)
            result["identity"]["helper_runtime_globals"] = {
                "pre_hydration": helper_runtime_globals_pre["pre_hydration"],
                "bound_pre_workload": helper_runtime_globals_pre["bound_pre_workload"],
                "post_workload": helper_runtime_globals_post,
                "passed": True,
            }
        finally:
            _write(args.json, result)
    if primary is not None:
        raise primary
    if set(result["positive_cells"]) != EXPECTED_POSITIVE_KEYS or not all(rounds[0].get("passed") is True for rounds in result["performance"].values()):
        raise RuntimeError("positive/performance coverage incomplete")
    result["complete"] = True; _write(args.json, result); print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
