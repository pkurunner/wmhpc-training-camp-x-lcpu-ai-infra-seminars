#!/usr/bin/env python3
"""Per-cell release gate for the passed packed-varlen confirmation matrix.

This runner is deliberately a second raw-wrapper allocation/seed experiment,
not a dispatcher mutation.  It freezes the two discovery artifacts, the
passed confirmation artifact, and the confirmation runner source.  Every one
of the eleven eligible cells is released or rejected independently; a failed
cell never promotes itself, but cannot invalidate a different passing cell.
The mixed-N6 FP32-both cell is measured for record only and remains baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_seqcount_dispatch import (  # noqa: E402
    run_seqcount_dispatch as shared,
)
from assignment02.team.c1_flashkda.challenge_varlen_dispatch import (  # noqa: E402
    run_varlen_dispatch_confirmation as confirmation,
)


DEFAULT_SEED = 20260830
WARMUP = 100
SAMPLES_PER_REPEAT = 1000
SEQCOUNT_DISCOVERY_SAMPLES = 1000
MIXED_DISCOVERY_SAMPLES = 300
REPEATS = 2
MIN_WINNER_MARGIN = 0.02
VARIANTS = ("baseline", "vshard2_p2", "vshard4_p2")
PERCENTILES = (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
CLEAN_GPU_GATE_ENV = "C1_VARLEN_DISPATCH_RELEASE_CLEAN_GPU_GATES"
RUNNER_SHA256_ENV = "C1_VARLEN_DISPATCH_RELEASE_RUNNER_SHA256"

SEQCOUNT_DISCOVERY_SHA256 = "46cd27f2fbdcaeeb61011c49c6175a0c05d15d4365bfda800cf52040dbe414f7"
MIXED_DISCOVERY_SHA256 = "b2dae8d42f43c3e42c44ca20fdc2c8443ec8b6b1b1ff2b81aff74be5b877fcd3"
CONFIRMATION_SHA256 = "447d7f49a624fa5b92adc431b350450f99d53f5b20f3a07a1bf4d2f76a64e51c"
CONFIRMATION_RUNNER_SHA256 = "9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b"
PINNED_REFERENCE_HELPER_SHA256 = "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
PINNED_TORCH_REF_SHA256 = "bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5"
PATCHED_FLASH_KDA_INIT_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
SHARED_SEQCOUNT_RUNNER_SHA256 = "4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f"
VARLEN_TAIL_RUNNER_SHA256 = "ff771c0b2f1b66f3062bc310c14634bf23830f706aec39f1b8ff03ff8b567621"
PREFETCH2_SHA256 = "752126488487ac317a7ee167b660b0895562e4877aedbc4ce2a599c1f59a10d0"
VSHARD4_PREFETCH2_SHA256 = "445c90815919fb8c1db3bc79289d10029ea77b1d9accb393c326c120ba1f8385"
HARNESS_SHA256 = "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_SEQCOUNT_DISCOVERY_JSON = (
    REPO_ROOT
    / "assignment02/team/c1_flashkda/challenge_seqcount_dispatch/results/c1_seqcount_dispatch_b300_sm103a_r2.json"
)
DEFAULT_MIXED_DISCOVERY_JSON = (
    REPO_ROOT
    / "assignment02/team/c1_flashkda/challenge_varlen_tail/results/c1_varlen_tail_b300_sm103a_r1.json"
)
DEFAULT_CONFIRMATION_JSON = RESULTS_DIR / "c1_varlen_dispatch_confirmation_b300_sm103a_r4.json"
CONFIRMATION_RUNNER_PATH = Path(confirmation.__file__).resolve()

# Case-level discovery source is fixed rather than inferred from a name.
DISCOVERY_SOURCE = {
    "equal_n2_h12_t2048": ("seqcount", "m024_n02_h12_balanced_varlen"),
    "equal_n4_h12_t2048": ("seqcount", "m048_n04_h12_balanced_varlen"),
    "mixed_n6_h12_t8192": ("mixed", "varlen_mixed_t8192"),
    "skew_n6_h12_t12288": ("seqcount", "m072_n06_h12_skewed_varlen"),
}


def _write(path: Path, result: Mapping[str, object]) -> None:
    """Atomically publish each checkpoint, preserving the last valid JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _clean_gpu_preflight() -> dict[str, object]:
    """Independently prove a single zero-allocation visible GPU before torch import."""

    binary = shutil.which("nvidia-smi")
    if binary is None:
        raise RuntimeError("nvidia-smi is unavailable for runner clean-GPU preflight")

    def query(arguments: list[str]) -> str:
        completed = subprocess.run(
            [binary, *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return completed.stdout.strip()

    applications = query([
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    memory_text = query([
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ])
    try:
        memory_mib = [int(line.strip()) for line in memory_text.splitlines() if line.strip()]
    except ValueError as exc:
        raise RuntimeError(f"invalid nvidia-smi memory.used output: {memory_text!r}") from exc
    if applications:
        raise RuntimeError(f"runner PRE clean-GPU gate found compute applications: {applications}")
    if memory_mib != [0]:
        raise RuntimeError(f"runner PRE clean-GPU gate requires one visible GPU at 0 MiB, got {memory_mib}")
    return {
        "passed": True,
        "nvidia_smi": str(Path(binary).resolve()),
        "compute_applications": [],
        "memory_used_mib": memory_mib,
        "checked_before_torch_import": True,
    }


def _as_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _as_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a JSON array")
    return value


def _load_json_with_hash(path: Path, expected_sha256: str, label: str) -> tuple[Mapping[str, Any], dict[str, object]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read frozen {label} artifact at {path}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} is invalid JSON despite matching its SHA256") from exc
    return _as_mapping(parsed, label), {
        "path": str(path), "sha256": actual, "sha256_gate_pass": True,
    }


def _confirmation_runner_identity() -> dict[str, object]:
    payload = CONFIRMATION_RUNNER_PATH.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != CONFIRMATION_RUNNER_SHA256:
        raise RuntimeError(
            "confirmation runner SHA256 mismatch: "
            f"expected {CONFIRMATION_RUNNER_SHA256}, got {actual} at {CONFIRMATION_RUNNER_PATH}"
        )
    return {
        "path": str(CONFIRMATION_RUNNER_PATH),
        "sha256": actual,
        "sha256_gate_pass": True,
    }


def _release_runner_identity() -> dict[str, object]:
    """Bind the JSON to the exact runner approved by the outer audit."""

    path = Path(__file__).resolve(strict=True)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = os.environ.get(RUNNER_SHA256_ENV)
    if expected is None:
        raise RuntimeError(
            f"{RUNNER_SHA256_ENV} is required for a GPU release experiment"
        )
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise RuntimeError(f"{RUNNER_SHA256_ENV} must be a lowercase SHA256")
    if actual != expected:
        raise RuntimeError(
            "release runner SHA256 mismatch: "
            f"outer audit expected {expected}, loaded {actual} at {path}"
        )
    return {
        "path": str(path),
        "sha256": actual,
        "sha256_gate_pass": True,
        "expected_sha256_environment": RUNNER_SHA256_ENV,
    }


def _require_file_sha256(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"cannot read runtime dependency {label} at {path}") from exc
    if actual != expected_sha256:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual} at {path}")
    return {"path": str(path.resolve()), "sha256": actual, "sha256_gate_pass": True}


def _verify_runtime_import_identities(
    args: argparse.Namespace,
    prefetch2: object,
    vshard4_prefetch2: object,
    common: object,
    flash_kda: object,
) -> dict[str, object]:
    """Fail closed before any CUDA work when a loaded/source dependency drifts."""
    helper_text = os.environ.get(confirmation.REFERENCE_HELPER_PATH_ENV)
    helper_env_sha = os.environ.get(confirmation.REFERENCE_HELPER_SHA_ENV)
    if not helper_text:
        raise RuntimeError("pinned reference helper path environment is missing")
    if helper_env_sha != PINNED_REFERENCE_HELPER_SHA256:
        raise RuntimeError(
            "pinned reference helper SHA environment drift: "
            f"expected {PINNED_REFERENCE_HELPER_SHA256}, got {helper_env_sha!r}"
        )
    return {
        "confirmation_runner": _confirmation_runner_identity(),
        "shared_seqcount_runner": _require_file_sha256(Path(shared.__file__).resolve(), SHARED_SEQCOUNT_RUNNER_SHA256, "shared seqcount runner"),
        "varlen_tail_runner": _require_file_sha256(
            REPO_ROOT / "assignment02/team/c1_flashkda/challenge_varlen_tail/run_varlen_tail.py",
            VARLEN_TAIL_RUNNER_SHA256,
            "varlen-tail historical runner",
        ),
        "prefetch2": _require_file_sha256(Path(prefetch2.__file__).resolve(), PREFETCH2_SHA256, "prefetch2 wrapper"),
        "vshard4_prefetch2": _require_file_sha256(Path(vshard4_prefetch2.__file__).resolve(), VSHARD4_PREFETCH2_SHA256, "vshard4 wrapper"),
        "harness": _require_file_sha256(Path(common.__file__).resolve(), HARNESS_SHA256, "harness"),
        "pinned_torch_ref": _require_file_sha256(args.reference_root / "tests/torch_ref.py", PINNED_TORCH_REF_SHA256, "pinned Torch reference"),
        "patched_flash_kda_init": _require_file_sha256(Path(flash_kda.__file__).resolve(), PATCHED_FLASH_KDA_INIT_SHA256, "loaded patched flash_kda.__init__"),
        "pinned_reference_helper": _require_file_sha256(Path(helper_text).resolve(), PINNED_REFERENCE_HELPER_SHA256, "pinned reference helper"),
    }


def _tensor_contract_evidence(
    functions: Mapping[str, Callable[..., None]],
    torch_ref: Callable[..., None],
    case: object,
    x: object,
    contract: str,
    seed: int,
) -> dict[str, object]:
    """Record actual packed offsets and tensor shapes omitted by ``_compare``."""

    lengths = tuple(int(length) for length in getattr(case, "lengths"))
    sequences = int(getattr(case, "sequences"))
    heads = int(getattr(case, "heads"))
    canonical_offsets = [0]
    for length in lengths:
        canonical_offsets.append(canonical_offsets[-1] + length)
    gpu_offsets = getattr(x, "cu_seqlens")
    if gpu_offsets is None:
        raise AssertionError(f"{getattr(case, 'name')}: packed-varlen evidence requires cu_seqlens")
    observed_offsets = [int(value) for value in gpu_offsets.detach().cpu().tolist()]
    if observed_offsets != canonical_offsets:
        raise AssertionError(
            f"{getattr(case, 'name')}: GPU offsets {observed_offsets} != canonical {canonical_offsets}"
        )
    if tuple(gpu_offsets.shape) != (sequences + 1,) or str(gpu_offsets.dtype) != "torch.int64":
        raise AssertionError(f"{getattr(case, 'name')}: invalid packed-offset tensor contract")

    expected_output_shape = list(getattr(x, "v").shape)
    expected_output_dtype = str(getattr(x, "v").dtype)
    final_required = contract != "none"
    expected_final_shape = [sequences, heads, shared.DIM, shared.DIM] if final_required else None
    expected_final_dtype = (
        "torch.bfloat16" if contract == "bf16_both" else "torch.float32"
    ) if final_required else None
    implementations: dict[str, object] = {}
    calls: dict[str, Callable[..., None]] = dict(functions)
    if contract in confirmation.FLA_PUBLIC_CONTRACTS:
        calls["pinned_torch_ref"] = torch_ref
    snapshot = confirmation._snapshot_inputs(x)
    for label, function in calls.items():
        initial, final = shared._states(contract, case, seed)
        output, final_state = shared._invoke(
            function, x, shared._clone(initial), shared._clone(final)
        )
        output_shape = list(output.shape)
        output_dtype = str(output.dtype)
        output_contiguous = bool(output.is_contiguous())
        if (
            output_shape != expected_output_shape
            or output_dtype != expected_output_dtype
            or not output_contiguous
        ):
            raise AssertionError(f"{getattr(case, 'name')}/{contract}/{label}: output contract mismatch")
        entry: dict[str, object] = {
            "output_shape": output_shape,
            "output_dtype": output_dtype,
            "output_contiguous": output_contiguous,
            "final_state_present": final_state is not None,
            "passed": True,
        }
        if final_required:
            if final_state is None:
                raise AssertionError(f"{getattr(case, 'name')}/{contract}/{label}: missing final state")
            final_shape = list(final_state.shape)
            final_dtype = str(final_state.dtype)
            final_contiguous = bool(final_state.is_contiguous())
            if (
                final_shape != expected_final_shape
                or final_dtype != expected_final_dtype
                or not final_contiguous
            ):
                raise AssertionError(f"{getattr(case, 'name')}/{contract}/{label}: final-state contract mismatch")
            entry.update(
                {
                    "final_state_shape": final_shape,
                    "final_state_dtype": final_dtype,
                    "final_state_contiguous": final_contiguous,
                }
            )
        elif final_state is not None:
            raise AssertionError(f"{getattr(case, 'name')}/{contract}/{label}: unexpected final state")
        confirmation._assert_inputs_unchanged(
            f"{getattr(case, 'name')}/{contract}/{label}/tensor_contract", x, snapshot
        )
        implementations[label] = entry
    return {
        "case": str(getattr(case, "name")),
        "contract": contract,
        "lengths": list(lengths),
        "canonical_offsets": canonical_offsets,
        "observed_gpu_offsets": observed_offsets,
        "sequence_count": sequences,
        "cu_seqlens_shape": list(gpu_offsets.shape),
        "cu_seqlens_dtype": str(gpu_offsets.dtype),
        "cu_seqlens_contiguous": bool(gpu_offsets.is_contiguous()),
        "expected_output_shape": expected_output_shape,
        "expected_output_dtype": expected_output_dtype,
        "expected_final_state_present": final_required,
        "expected_final_state_shape": expected_final_shape,
        "expected_final_state_dtype": expected_final_dtype,
        "implementations": implementations,
        "passed": True,
    }


def _require_b300_identity(data: Mapping[str, Any], label: str) -> dict[str, object]:
    """Accept the two frozen discovery layouts but reject every identity drift."""
    identity = _as_mapping(data.get("identity"), f"{label}.identity") if "identity" in data else data
    device_value = identity.get("device")
    device = _as_mapping(device_value, f"{label}.device") if isinstance(device_value, Mapping) else None
    name = str(device.get("name", "")) if device is not None else str(data.get("device", ""))
    capability = device.get("capability") if device is not None else data.get("capability")
    sm_count = device.get("multiprocessor_count") if device is not None else data.get("multiprocessor_count")
    extension = _as_mapping(identity.get("extension", data.get("extension")), f"{label}.extension")
    extension_sha = str(extension.get("sha256", ""))
    if (
        "B300" not in name.upper()
        or capability != [10, 3]
        or sm_count != 148
        or extension_sha != shared.AUDITED_EXTENSION_SHA256
    ):
        raise RuntimeError(
            f"{label} is not an audited B300 artifact: name={name!r}, capability={capability!r}, "
            f"SMs={sm_count!r}, extension={extension_sha!r}"
        )
    return {
        "name": name,
        "capability": capability,
        "multiprocessor_count": sm_count,
        "extension_sha256": extension_sha,
        "identity_gate_pass": True,
    }


def _validate_exact_record(value: object, label: str) -> None:
    record = _as_mapping(value, label)
    if record.get("output_exact") is not True or float(record.get("output_max_abs", float("nan"))) != 0.0:
        raise RuntimeError(f"{label}: output is not bitwise exact")
    final_present = record.get("final_state_present")
    if final_present is True:
        if record.get("final_state_exact") is not True or float(record.get("final_state_max_abs", float("nan"))) != 0.0:
            raise RuntimeError(f"{label}: final state is not bitwise exact")
    elif final_present is not False:
        raise RuntimeError(f"{label}: final_state_present is malformed")


def _validate_raw_exactness(correctness: Mapping[str, Any], case_name: str, label: str) -> dict[str, object]:
    case = _as_mapping(correctness.get(case_name), f"{label}.{case_name}")
    for contract in confirmation.RAW_CONTRACTS:
        contract_data = _as_mapping(case.get(contract), f"{label}.{case_name}.{contract}")
        for variant in ("vshard2_p2", "vshard4_p2"):
            _validate_exact_record(contract_data.get(variant), f"{label}.{case_name}.{contract}.{variant}")
    return {"case": case_name, "contracts": list(confirmation.RAW_CONTRACTS), "exact_gate_pass": True}


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _recompute_gate(
    benchmark: Mapping[str, Any], expected_winner: str, label: str, *, expected_samples: int
) -> dict[str, object]:
    raw = _as_mapping(benchmark.get("raw_samples_ms"), f"{label}.raw_samples_ms")
    if set(raw) != set(VARIANTS):
        raise RuntimeError(f"{label} must have raw samples for exactly {VARIANTS}")
    summaries: dict[str, dict[str, float | int]] = {}
    for variant in VARIANTS:
        samples = _as_list(raw[variant], f"{label}.{variant}")
        if len(samples) != expected_samples:
            raise RuntimeError(f"{label}.{variant}: expected {expected_samples} raw samples")
        values = [float(value) for value in samples]
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise RuntimeError(f"{label}.{variant}: invalid CUDA-event sample")
        summaries[variant] = {
            "samples": len(values),
            "mean_ms": statistics.fmean(values),
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
            "p99_ms": _percentile(values, 0.99),
        }
    winners: dict[str, str] = {}
    margins: dict[str, float] = {}
    for name, _ in PERCENTILES:
        ranked = sorted((float(summaries[variant][f"{name}_ms"]), variant) for variant in VARIANTS)
        winners[name] = ranked[0][1]
        margins[name] = ranked[1][0] / ranked[0][0] - 1.0
    expected_at_all = all(winners[name] == expected_winner for name, _ in PERCENTILES)
    margin_pass = all(margins[name] >= MIN_WINNER_MARGIN for name, _ in PERCENTILES)
    return {
        "recomputed_from_raw_samples": True,
        "expected_raw_sample_count": expected_samples,
        "summaries": summaries,
        "winner_by_percentile": winners,
        "runner_up_margin_by_percentile": margins,
        "expected_winner": expected_winner,
        "expected_winner_at_all_percentiles": expected_at_all,
        "minimum_required_margin": MIN_WINNER_MARGIN,
        "margin_gate_pass": margin_pass,
        "gate_pass": expected_at_all and margin_pass,
    }


def _confirmation_mapping() -> dict[str, dict[str, str]]:
    return {case: dict(contracts) for case, contracts in confirmation.PROMOTION_CELLS.items()}


def _validate_confirmation(
    data: Mapping[str, Any], artifact: Mapping[str, object]
) -> dict[str, object]:
    if data.get("complete") is not True:
        raise RuntimeError("confirmation artifact is incomplete")
    prereg = _as_mapping(data.get("preregistration"), "confirmation.preregistration")
    if (
        prereg.get("fixed_seed") != confirmation.DEFAULT_SEED
        or prereg.get("promotion_cell_count") != 11
        or prereg.get("repeats_per_public_cell") != REPEATS
        or prereg.get("cyclic_cuda_event_samples_per_repeat") != SAMPLES_PER_REPEAT
        or prereg.get("promotion_cells") != _confirmation_mapping()
    ):
        raise RuntimeError("confirmation preregistration mapping/count/seed drift")
    gates = _as_mapping(data.get("gates"), "confirmation.gates")
    required_gates = (
        "clean_gpu_shell_gate", "device_gate", "audited_extension_sha256_gate",
        "raw_abi_exact_gate", "pinned_torch_reference_gate", "confirmation_gate",
    )
    if any(_as_mapping(gates.get(name), f"confirmation.gates.{name}").get("passed") is not True for name in required_gates):
        raise RuntimeError("confirmation has a failed required gate")
    assessment = _as_mapping(data.get("confirmation_assessment"), "confirmation.assessment")
    scope_cells = _as_mapping(assessment.get("scope_cells"), "confirmation.assessment.scope_cells")
    if (
        assessment.get("scope_cell_count") != 11
        or assessment.get("confirmation_gate_pass") is not True
        or set(scope_cells) != {f"{case}/{contract}" for case, contracts in confirmation.PROMOTION_CELLS.items() for contract in contracts}
    ):
        raise RuntimeError("confirmation assessment is not the complete passed 11-cell scope")
    for key, expected in ((f"{case}/{contract}", winner) for case, contracts in confirmation.PROMOTION_CELLS.items() for contract, winner in contracts.items()):
        cell = _as_mapping(scope_cells.get(key), f"confirmation.assessment.{key}")
        if cell.get("expected_winner") != expected or cell.get("cell_gate_pass") is not True:
            raise RuntimeError(f"confirmation assessment cell drift: {key}")
    identity = _require_b300_identity(_as_mapping(data.get("identity"), "confirmation.identity"), "confirmation")
    exact = _as_mapping(data.get("raw_abi_correctness"), "confirmation.raw_abi_correctness")
    if set(exact) != set(confirmation.PROMOTION_CELLS):
        raise RuntimeError("confirmation raw exactness case scope drift")
    exact_summary = {
        case.name: _validate_raw_exactness(exact, case.name, "confirmation.raw_abi_correctness")
        for case in confirmation.CASES
    }
    return {"artifact": dict(artifact), "identity": identity, "raw_abi_exactness": exact_summary, "gate_pass": True}


def _load_history(args: argparse.Namespace) -> dict[str, object]:
    seqcount, seq_artifact = _load_json_with_hash(
        args.seqcount_discovery_json, SEQCOUNT_DISCOVERY_SHA256, "seqcount discovery"
    )
    mixed, mixed_artifact = _load_json_with_hash(
        args.mixed_discovery_json, MIXED_DISCOVERY_SHA256, "mixed discovery"
    )
    confirmed, confirmation_artifact = _load_json_with_hash(
        args.confirmation_json, CONFIRMATION_SHA256, "confirmation"
    )
    if seqcount.get("complete") is not True or seqcount.get("exact_gate_pass") is not True:
        raise RuntimeError("seqcount discovery is incomplete or its exact gate failed")
    if mixed.get("exact_gate_pass") is not True:
        raise RuntimeError("mixed discovery exact gate failed")
    history = {
        "confirmation_runner": _confirmation_runner_identity(),
        "seqcount_discovery": {"artifact": seq_artifact, "identity": _require_b300_identity(seqcount, "seqcount")},
        "mixed_discovery": {"artifact": mixed_artifact, "identity": _require_b300_identity(mixed, "mixed")},
        "confirmation": _validate_confirmation(confirmed, confirmation_artifact),
        "cells": {},
    }
    discoveries = {"seqcount": seqcount, "mixed": mixed}
    for case_name, contracts in confirmation.PROMOTION_CELLS.items():
        source_name, source_case = DISCOVERY_SOURCE[case_name]
        discovery_expected_samples = (
            SEQCOUNT_DISCOVERY_SAMPLES if source_name == "seqcount" else MIXED_DISCOVERY_SAMPLES
        )
        source = discoveries[source_name]
        source_benchmark = _as_mapping(source.get("benchmark"), f"{source_name}.benchmark")
        source_correctness = _as_mapping(source.get("correctness"), f"{source_name}.correctness")
        discovery_exact = _validate_raw_exactness(source_correctness, source_case, f"{source_name}.correctness")
        case_benchmark = _as_mapping(source_benchmark.get(source_case), f"{source_name}.benchmark.{source_case}")
        confirmation_benchmarks = _as_mapping(
            confirmed.get("raw_wrapper_public_contract_benchmarks"), "confirmation.benchmarks"
        )
        confirmation_case = _as_mapping(confirmation_benchmarks.get(case_name), f"confirmation.benchmarks.{case_name}")
        for contract, expected_winner in contracts.items():
            discovery_gate = _recompute_gate(
                _as_mapping(case_benchmark.get(contract), f"{source_name}.{case_name}.{contract}"),
                expected_winner, f"discovery/{case_name}/{contract}", expected_samples=discovery_expected_samples,
            )
            confirmation_cell = _as_mapping(confirmation_case.get(contract), f"confirmation.{case_name}.{contract}")
            repeats = _as_list(confirmation_cell.get("repeats"), f"confirmation.{case_name}.{contract}.repeats")
            if len(repeats) != REPEATS:
                raise RuntimeError(f"confirmation {case_name}/{contract} repeat count drift")
            confirmation_repeats = [
                _recompute_gate(
                    _as_mapping(_as_mapping(repeat, "confirmation.repeat").get("benchmark"), "confirmation.repeat.benchmark"),
                    expected_winner, f"confirmation/{case_name}/{contract}/repeat{index}", expected_samples=SAMPLES_PER_REPEAT,
                )
                for index, repeat in enumerate(repeats)
            ]
            key = f"{case_name}/{contract}"
            history["cells"][key] = {  # type: ignore[index]
                "case": case_name,
                "contract": contract,
                "expected_winner": expected_winner,
                "discovery_source": source_name,
                "discovery_raw_exactness": discovery_exact,
                "discovery_single_repeat": discovery_gate,
                "confirmation_two_repeats": confirmation_repeats,
                "historical_gate_pass": bool(discovery_gate["gate_pass"]) and all(
                    bool(repeat["gate_pass"]) for repeat in confirmation_repeats
                ),
            }
    if len(history["cells"]) != 11:  # type: ignore[arg-type]
        raise RuntimeError("historical eligible release scope is not exactly 11 cells")
    return history


def _preregistration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "confirmation_runner_sha256": CONFIRMATION_RUNNER_SHA256,
        "release_seed": DEFAULT_SEED,
        "warmup_per_path": WARMUP,
        "samples_per_repeat": SAMPLES_PER_REPEAT,
        "repeats": REPEATS,
        "percentiles": [name for name, _ in PERCENTILES],
        "minimum_runner_up_margin": MIN_WINNER_MARGIN,
        "cases": [shared._case_dict(case) for case in confirmation.CASES],
        "eligible_cells": _confirmation_mapping(),
        "eligible_cell_count": 11,
        "record_only": dict(confirmation.RECORD_ONLY_CELLS[0]),
        "frozen_artifacts": {
            "seqcount_discovery": {"path": str(args.seqcount_discovery_json), "sha256": SEQCOUNT_DISCOVERY_SHA256, "raw_samples_per_variant": SEQCOUNT_DISCOVERY_SAMPLES},
            "mixed_discovery": {"path": str(args.mixed_discovery_json), "sha256": MIXED_DISCOVERY_SHA256, "raw_samples_per_variant": MIXED_DISCOVERY_SAMPLES},
            "confirmation": {"path": str(args.confirmation_json), "sha256": CONFIRMATION_SHA256},
        },
        "per_cell_release_rule": (
            "A cell is released only when frozen discovery, both frozen confirmation repeats, and both "
            "new-allocation release repeats independently select the pre-registered winner at P50/P95/P99 "
            "with >=2% runner-up margin. Rejection is isolated to that cell."
        ),
    }


def _initial_result(args: argparse.Namespace) -> dict[str, object]:
    eligible_keys = [f"{case}/{contract}" for case, contracts in confirmation.PROMOTION_CELLS.items() for contract in contracts]
    return {
        "schema_version": 1,
        "purpose": "per-cell packed-varlen raw-wrapper release gate; no dispatcher mutation",
        "preregistration": _preregistration(args),
        "history_evidence": {},
        "identity": {},
        "raw_abi_correctness": {},
        "new_measurements": {},
        "cell_status": {key: {"status": "rejected", "reason": "not_run"} for key in eligible_keys},
        "released_mapping": {},
        "fallback_mapping": {key: "baseline" for key in eligible_keys},
        "record_only": {},
        "gates": {
            "scope_count": {"required": 11, "actual": len(eligible_keys), "passed": len(eligible_keys) == 11},
            "confirmation_runner_identity": {"required": CONFIRMATION_RUNNER_SHA256, "passed": False},
            "historical_artifacts": {"passed": False},
            "clean_gpu_shell_gate": {"required": True, "passed": False},
            "clean_gpu_runner_preflight_gate": {
                "required": "one visible GPU at 0 MiB before torch import", "passed": False,
            },
            "device_gate": {"required": "B300, capability 10.3, 148 SM", "passed": False},
            "audited_extension_sha256_gate": {"required": shared.AUDITED_EXTENSION_SHA256, "passed": False},
            "current_raw_abi_exact_gate": {"passed": False},
            "pinned_torch_reference_gate": {"passed": False},
            "runtime_import_identity_gate": {"passed": False},
        },
        "complete": False,
    }


def _check_args(args: argparse.Namespace) -> None:
    if (
        args.seed != DEFAULT_SEED or args.warmup != WARMUP or args.samples != SAMPLES_PER_REPEAT or args.repeats != REPEATS
    ):
        raise ValueError("release is fixed at --seed=20260830, --warmup=100, --samples=1000, --repeats=2")
    if args.json.suffix.lower() != ".json":
        raise ValueError("--json output must use a .json suffix")
    output = args.json.resolve()
    frozen_inputs = {
        "seqcount discovery": args.seqcount_discovery_json.resolve(),
        "mixed discovery": args.mixed_discovery_json.resolve(),
        "confirmation": args.confirmation_json.resolve(),
    }
    collisions = [label for label, path in frozen_inputs.items() if path == output]
    if collisions:
        raise ValueError(f"--json output collides with frozen input artifact(s): {collisions}")
    confirmation._assert_preregistration_scope()


def _synthetic_assessment() -> dict[str, object]:
    def synthetic_raw(count: int) -> dict[str, object]:
        return {"raw_samples_ms": {"baseline": [3.0] * count, "vshard2_p2": [2.0] * count, "vshard4_p2": [2.5] * count}}
    evidence_1000 = _recompute_gate(synthetic_raw(SAMPLES_PER_REPEAT), "vshard2_p2", "synthetic/1000", expected_samples=SAMPLES_PER_REPEAT)
    evidence_300 = _recompute_gate(synthetic_raw(MIXED_DISCOVERY_SAMPLES), "vshard2_p2", "synthetic/300", expected_samples=MIXED_DISCOVERY_SAMPLES)
    if evidence_1000["gate_pass"] is not True or evidence_300["gate_pass"] is not True:
        raise AssertionError("synthetic expected-winner assessment unexpectedly failed")
    return {"samples_1000": evidence_1000, "samples_300": evidence_300}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_REPEAT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--seqcount-discovery-json", type=Path, default=DEFAULT_SEQCOUNT_DISCOVERY_JSON)
    parser.add_argument("--mixed-discovery-json", type=Path, default=DEFAULT_MIXED_DISCOVERY_JSON)
    parser.add_argument("--confirmation-json", type=Path, default=DEFAULT_CONFIRMATION_JSON)
    parser.add_argument("--describe", action="store_true", help="write the fixed release plan without importing CUDA")
    parser.add_argument("--synthetic-assessment", action="store_true", help="exercise raw-sample gate logic without CUDA")
    args = parser.parse_args()
    _check_args(args)
    result = _initial_result(args)
    if args.describe:
        result["describe_only"] = True
        _write(args.json, result)
        print(f"wrote packed-varlen release preregistration {args.json}")
        return
    if args.synthetic_assessment:
        result["synthetic_assessment"] = _synthetic_assessment()
        result["synthetic_only"] = True
        _write(args.json, result)
        print(f"wrote synthetic packed-varlen release assessment {args.json}")
        return
    if args.reference_root is None:
        raise ValueError("--reference-root is required for a GPU release experiment")
    if os.environ.get(CLEAN_GPU_GATE_ENV) != "1":
        raise RuntimeError(
            "refusing a direct GPU run: use run_clean_varlen_dispatch_release_audit.sh so "
            f"{CLEAN_GPU_GATE_ENV}=1 is set only after PRE clean-GPU verification"
        )

    release_runner_identity = _release_runner_identity()
    result["identity"] = {"release_runner": release_runner_identity}
    _write(args.json, result)

    clean_gpu_preflight = _clean_gpu_preflight()
    result["gates"]["clean_gpu_runner_preflight_gate"]["passed"] = True  # type: ignore[index]

    history = _load_history(args)
    result["history_evidence"] = history
    result["gates"]["confirmation_runner_identity"]["passed"] = True  # type: ignore[index]
    result["gates"]["historical_artifacts"]["passed"] = True  # type: ignore[index]
    _write(args.json, result)

    import torch
    from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2
    from assignment02.team.c1_flashkda.challenge_vshard4_prefetch2 import vshard4_prefetch2
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shared.torch = torch
    shared.common = common
    import flash_kda

    functions: dict[str, Callable[..., None]] = {
        "baseline": flash_kda.fwd,
        "vshard2_p2": prefetch2.fwd,
        "vshard4_p2": vshard4_prefetch2.fwd,
    }
    runtime_import_identities = _verify_runtime_import_identities(
        args, prefetch2, vshard4_prefetch2, common, flash_kda
    )
    torch_ref, helper_identity = confirmation._load_pinned_reference_without_build(common, args.reference_root)
    result["identity"] = {
        "release_runner": release_runner_identity,
        "device": shared._device_identity(),
        "extension": shared._identity(),
        "clean_gpu_shell_gate": {"environment_name": CLEAN_GPU_GATE_ENV, "passed": True},
        "clean_gpu_runner_preflight_gate": clean_gpu_preflight,
        "pinned_torch_reference_root": str(args.reference_root.resolve()),
        "pinned_torch_reference_helper": helper_identity,
        "runtime_import_identities": runtime_import_identities,
    }
    result["gates"]["clean_gpu_shell_gate"]["passed"] = True  # type: ignore[index]
    result["gates"]["device_gate"]["passed"] = True  # type: ignore[index]
    result["gates"]["audited_extension_sha256_gate"]["passed"] = True  # type: ignore[index]
    result["gates"]["runtime_import_identity_gate"]["passed"] = True  # type: ignore[index]
    _write(args.json, result)

    for case_index, case in enumerate(confirmation.CASES):
        x = shared._make_inputs(case, args.seed + case_index * 10_007)
        result["raw_abi_correctness"][case.name] = {}  # type: ignore[index]
        for contract_index, contract in enumerate(confirmation.RAW_CONTRACTS):
            cell_evidence = confirmation._raw_abi_exactness(
                functions, torch_ref, case, x, contract, args.seed + case_index * 10_007 + contract_index * 101
            )
            cell_evidence["tensor_contract"] = _tensor_contract_evidence(
                functions,
                torch_ref,
                case,
                x,
                contract,
                args.seed + case_index * 10_007 + contract_index * 101,
            )
            result["raw_abi_correctness"][case.name][contract] = cell_evidence  # type: ignore[index]
        del x
        torch.cuda.empty_cache()
        _write(args.json, result)
    result["gates"]["current_raw_abi_exact_gate"]["passed"] = True  # type: ignore[index]
    result["gates"]["pinned_torch_reference_gate"]["passed"] = True  # type: ignore[index]
    _write(args.json, result)

    history_cells = _as_mapping(history["cells"], "history.cells")
    for case_index, case in enumerate(confirmation.CASES):
        result["new_measurements"][case.name] = {}  # type: ignore[index]
        for contract_index, contract in enumerate(confirmation.FLA_PUBLIC_CONTRACTS):
            expected = confirmation.PROMOTION_CELLS.get(case.name, {}).get(contract)
            record_only = expected is None
            repeats: list[dict[str, object]] = []
            result["new_measurements"][case.name][contract] = {  # type: ignore[index]
                "expected_winner": expected, "record_only": record_only, "repeats": repeats,
            }
            _write(args.json, result)
            for repeat_index in range(REPEATS):
                repeat_seed = args.seed + case_index * 100_003 + contract_index * 10_007 + repeat_index * 1_009
                x = shared._make_inputs(case, repeat_seed)
                snapshot = confirmation._snapshot_inputs(x)
                benchmark = shared._benchmark(functions, case, x, contract, repeat_seed + 101, args.warmup, args.samples)
                confirmation._assert_inputs_unchanged(f"release/{case.name}/{contract}/repeat{repeat_index}", x, snapshot)
                repeat = {
                    "repeat_index": repeat_index,
                    "input_seed": repeat_seed,
                    "state_seed": repeat_seed + 101,
                    "input_immutability_exact": True,
                    "benchmark": benchmark,
                    "recomputed_gate": (
                        _recompute_gate(
                            _as_mapping(benchmark, "new benchmark"), expected,
                            f"new/{case.name}/{contract}/repeat{repeat_index}", expected_samples=SAMPLES_PER_REPEAT,
                        )
                        if expected is not None else confirmation._independent_summary(benchmark)
                    ),
                }
                repeats.append(repeat)
                del x
                torch.cuda.empty_cache()
                _write(args.json, result)
            key = f"{case.name}/{contract}"
            if record_only:
                result["record_only"][key] = {  # type: ignore[index]
                    "status": "baseline", "reason": "pre-registered record-only; never eligible", "repeats": repeats,
                }
                continue
            historical = _as_mapping(history_cells.get(key), f"history.cells.{key}")
            new_pass = all(bool(_as_mapping(repeat["recomputed_gate"], "new repeat gate").get("gate_pass")) for repeat in repeats)
            historical_pass = bool(historical.get("historical_gate_pass"))
            if historical_pass and new_pass:
                status, reason = "released", "frozen history and both independent new repeats passed"
                result["released_mapping"][key] = expected  # type: ignore[index]
                result["fallback_mapping"].pop(key, None)  # type: ignore[index]
            elif not historical_pass:
                status, reason = "rejected", "frozen discovery or confirmation evidence failed this cell"
            else:
                status, reason = "rejected", "at least one new-allocation release repeat failed this cell"
            result["cell_status"][key] = {  # type: ignore[index]
                "status": status, "expected_winner": expected, "historical_gate_pass": historical_pass,
                "new_gate_pass": new_pass, "reason": reason,
            }
            _write(args.json, result)

    result["complete"] = True
    _write(args.json, result)
    print(f"wrote {args.json}; released={result['released_mapping']}; fallback={result['fallback_mapping']}")


if __name__ == "__main__":
    main()
