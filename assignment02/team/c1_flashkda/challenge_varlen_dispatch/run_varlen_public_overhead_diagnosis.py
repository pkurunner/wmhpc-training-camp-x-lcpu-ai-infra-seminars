#!/usr/bin/env python3
"""Diagnostic-only attribution of packed-varlen public FLA dispatch overhead.

This is deliberately not a release runner.  It first requires the frozen r5
two-cell production map, then installs the frozen r4 ten-cell map only in this
fresh Python process.  The original module object and environment are restored
in ``finally`` before the process exits.  Results explicitly have no release
authority.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_seqcount_dispatch import run_seqcount_dispatch as shared  # noqa: E402
from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch, varlen_metadata  # noqa: E402
from assignment02.team.c1_flashkda.challenge_varlen_dispatch import run_varlen_dispatch_confirmation as confirmation  # noqa: E402
from assignment02.team.c1_flashkda.challenge_varlen_dispatch import run_varlen_fla_integration as integration  # noqa: E402


RUNNER_SHA256_ENV = "C1_VARLEN_PUBLIC_OVERHEAD_RUNNER_SHA256"
CLEAN_GPU_GATE_ENV = "C1_VARLEN_PUBLIC_OVERHEAD_CLEAN_GPU"
GPU_AUTHORIZATION_ENV = "C1_VARLEN_PUBLIC_OVERHEAD_GPU_AUTHORIZED"
CURRENT_INTEGRATION_RUNNER_SHA256 = "5db71f29335220496ca9540924e17c5f160b0bc8237060921cffaecb708f22bb"
SEED = 20260902
WARMUP = 100
SAMPLES = 1000
REPEATS = 2


# r5 production policy: this diagnostic must fail closed if it sees either a
# broader map or a different two-cell map before its temporary installation.
PRODUCTION_PUBLIC_VARIANTS = {
    ((0, 1, 2, 3, 4, 5, 12288), "none"): "vshard2_p2",
    ((0, 1, 2, 3, 4, 5, 12288), "fp32_final_only"): "vshard2_p2",
}

# Frozen r4 raw-release candidates.  This map is diagnostic-only and never
# written to auto_dispatch.py; it exists solely in this fresh child process.
FROZEN_R4_DIAGNOSTIC_VARIANTS = {
    ((0, 2048, 4096), "none"): "vshard4_p2",
    ((0, 2048, 4096), "fp32_final_only"): "vshard4_p2",
    ((0, 2048, 4096), "fp32_both"): "vshard4_p2",
    ((0, 2048, 4096, 6144, 8192), "none"): "vshard2_p2",
    ((0, 2048, 4096, 6144, 8192), "fp32_final_only"): "vshard2_p2",
    ((0, 17, 528, 1552, 2852, 4901, 8192), "none"): "vshard2_p2",
    ((0, 17, 528, 1552, 2852, 4901, 8192), "fp32_final_only"): "vshard2_p2",
    ((0, 1, 2, 3, 4, 5, 12288), "none"): "vshard2_p2",
    ((0, 1, 2, 3, 4, 5, 12288), "fp32_final_only"): "vshard2_p2",
    ((0, 1, 2, 3, 4, 5, 12288), "fp32_both"): "vshard4_p2",
}


@dataclass(frozen=True)
class DiagnosticCell:
    case: shared.Case
    expected_variant: str

    @property
    def key(self) -> str:
        return f"{self.case.name}/none"


REPRESENTATIVE_CELLS = (
    DiagnosticCell(confirmation.CASES[0], "vshard4_p2"),
    DiagnosticCell(confirmation.CASES[2], "vshard2_p2"),
    DiagnosticCell(confirmation.CASES[3], "vshard2_p2"),
)
PATHS = ("public_c1", "direct_c1", "public_pinned", "direct_pinned")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, result: Mapping[str, object]) -> None:
    integration._write(path, result)


def _serializable_map(mapping: Mapping[tuple[tuple[int, ...], str], str]) -> list[dict[str, object]]:
    return [
        {"offsets": list(offsets), "contract": contract, "variant": variant}
        for (offsets, contract), variant in sorted(mapping.items())
    ]


def _offsets(lengths: tuple[int, ...]) -> tuple[int, ...]:
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + int(length))
    return tuple(cumulative)


def _require_static_contracts() -> None:
    if len(PRODUCTION_PUBLIC_VARIANTS) != 2:
        raise AssertionError("production map literal must contain exactly two cells")
    if len(FROZEN_R4_DIAGNOSTIC_VARIANTS) != 10:
        raise AssertionError("frozen r4 diagnostic map must contain exactly ten cells")
    representatives = {(_offsets(cell.case.lengths), "none"): cell.expected_variant for cell in REPRESENTATIVE_CELLS}
    if len(REPRESENTATIVE_CELLS) != 3 or representatives != {
        ((0, 2048, 4096), "none"): "vshard4_p2",
        ((0, 17, 528, 1552, 2852, 4901, 8192), "none"): "vshard2_p2",
        ((0, 1, 2, 3, 4, 5, 12288), "none"): "vshard2_p2",
    }:
        raise AssertionError("diagnostic representatives drifted")
    if not set(PRODUCTION_PUBLIC_VARIANTS).issubset(FROZEN_R4_DIAGNOSTIC_VARIANTS):
        raise AssertionError("r4 diagnostic map must include the frozen r5 production cells")
    if WARMUP % len(PATHS) or SAMPLES % len(PATHS):
        raise AssertionError("four-path cyclic schedule requires warmup and samples divisible by four")


def _assert_production_map() -> dict[tuple[tuple[int, ...], str], str]:
    actual = getattr(auto_dispatch, "_VARLEN_PUBLIC_VARIANTS", None)
    if not isinstance(actual, dict):
        raise RuntimeError("auto_dispatch has no mutable packed-varlen public map")
    snapshot = dict(actual)
    if snapshot != PRODUCTION_PUBLIC_VARIANTS:
        raise RuntimeError(
            "production packed-varlen map drift: diagnostic requires exactly the frozen r5 two-cell map"
        )
    return snapshot


def _runner_identity() -> dict[str, object]:
    path = Path(__file__).resolve(strict=True)
    actual = _sha(path)
    expected = os.environ.get(RUNNER_SHA256_ENV)
    if expected is None:
        raise RuntimeError(f"{RUNNER_SHA256_ENV} is required for a GPU diagnostic")
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise RuntimeError(f"{RUNNER_SHA256_ENV} must be a lowercase SHA256")
    if actual != expected:
        raise RuntimeError(
            f"diagnostic runner SHA256 mismatch: outer audit expected {expected}, loaded {actual} at {path}"
        )
    integration_path = Path(integration.__file__).resolve(strict=True)
    integration_sha = _sha(integration_path)
    if integration_sha != CURRENT_INTEGRATION_RUNNER_SHA256:
        raise RuntimeError(
            "current integration runner SHA256 drift: "
            f"expected {CURRENT_INTEGRATION_RUNNER_SHA256}, got {integration_sha}"
        )
    return {
        "path": str(path),
        "sha256": actual,
        "sha256_gate_pass": True,
        "expected_sha256_environment": RUNNER_SHA256_ENV,
        "current_integration_runner": {
            "path": str(integration_path),
            "sha256": integration_sha,
            "passed": True,
        },
    }


def _initial_result(args: argparse.Namespace, production_map: Mapping[tuple[tuple[int, ...], str], str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "purpose": "diagnostic-only attribution of public FLA packed-varlen overhead",
        "diagnostic_only": True,
        "no_release_authority": True,
        "no_policy_mutation": "auto_dispatch.py is never written; the r4 map is process-local and finally-restored",
        "seed": args.seed,
        "representative_cells": [
            {"cell": cell.key, "expected_diagnostic_variant": cell.expected_variant}
            for cell in REPRESENTATIVE_CELLS
        ],
        "maps": {
            "production_r5_before_temporary_install": _serializable_map(production_map),
            "temporary_r4_diagnostic": _serializable_map(FROZEN_R4_DIAGNOSTIC_VARIANTS),
            "temporary_installation": {"attempted": False, "passed": False},
            "finally_restored_r5": {"attempted": False, "passed": False},
        },
        "measurement": {
            "paths": list(PATHS),
            "repeats": REPEATS,
            "warmup_per_path_per_repeat": WARMUP,
            "samples_per_path_per_repeat": SAMPLES,
            "percentiles": ["p50", "p95", "p99"],
            "event_contract": "for every path: start.record(current_stream) -> start.synchronize() -> complete call -> end.record(current_stream) -> end.synchronize() -> elapsed_time; both synchronizations are excluded from the sample value",
            "schedule": "cyclic four-path rotation; each path is first exactly 25 warmup rounds and 250 timed rounds per repeat",
            "differential": "(public_c1 - direct_c1) - (public_pinned - direct_pinned)",
        },
        "identity": {},
        "gates": {
            "production_map_exact_two": {"passed": True},
            "clean_gpu": {"passed": False},
            "device": {"passed": False},
            "extension": {"passed": False},
            "fla_pin": {"passed": False},
            "runtime_dependencies": {"passed": False},
            "inference_mode": {"passed": False},
            "temporary_map_restored": {"passed": False},
        },
        "cells": {},
        "complete": False,
    }


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _select_path(path: str) -> None:
    if path in ("public_c1", "direct_c1"):
        os.environ["C1_B300_FLASH_KDA"] = "1"
    elif path in ("public_pinned", "direct_pinned"):
        os.environ["C1_B300_FLASH_KDA"] = "0"
    else:
        raise ValueError(f"unknown timing path {path!r}")
    os.environ["FLA_FLASH_KDA"] = "1"


def _call_path(
    path: str,
    public_fn: Callable[..., Any],
    direct_c1: Callable[..., Any],
    direct_pinned: Callable[..., Any],
    x: object,
    gpu: object,
    cpu: object,
    *,
    select_path: bool = True,
) -> tuple[object, object | None]:
    if select_path:
        _select_path(path)
    fn = {
        "public_c1": public_fn,
        "direct_c1": direct_c1,
        "public_pinned": public_fn,
        "direct_pinned": direct_pinned,
    }[path]
    return integration._call(fn, x, None, False, gpu, cpu)


@contextmanager
def _backend_chunk_spies(c1: object, pinned: object) -> Iterator[tuple[dict[str, Callable[..., Any]], dict[str, int]]]:
    """Use instance-local spies only for non-timed route probes."""

    originals = {"c1": c1.chunk_kda, "pinned": pinned.chunk_kda}
    counts = {"c1": 0, "pinned": 0}

    def c1_spy(*args: object, **kwargs: object) -> object:
        counts["c1"] += 1
        return originals["c1"](*args, **kwargs)

    def pinned_spy(*args: object, **kwargs: object) -> object:
        counts["pinned"] += 1
        return originals["pinned"](*args, **kwargs)

    c1.chunk_kda = c1_spy
    pinned.chunk_kda = pinned_spy
    try:
        yield originals, counts
    finally:
        c1.chunk_kda = originals["c1"]
        pinned.chunk_kda = originals["pinned"]


@contextmanager
def _prepare_varlen_spy(c1: object) -> Iterator[dict[str, int]]:
    """Count C1's public verifier/preflight calls without timing it."""

    original = c1._prepare_varlen
    counts = {"prepare_varlen": 0}

    def spy(*args: object, **kwargs: object) -> object:
        counts["prepare_varlen"] += 1
        return original(*args, **kwargs)

    c1._prepare_varlen = spy
    try:
        yield counts
    finally:
        c1._prepare_varlen = original


def _delta(after: Mapping[str, int], before: Mapping[str, int]) -> dict[str, int]:
    return {key: int(after[key] - before[key]) for key in before}


def _call_probe(
    path: str,
    public_fn: Callable[..., Any],
    direct_c1: Callable[..., Any],
    direct_pinned: Callable[..., Any],
    x: object,
    gpu: object,
    cpu: object,
    backend_counts: Mapping[str, int],
    prepare_counts: Mapping[str, int],
) -> tuple[tuple[object, object | None], dict[str, object]]:
    before_backend = dict(backend_counts)
    before_prepare = dict(prepare_counts)
    output = _call_path(path, public_fn, direct_c1, direct_pinned, x, gpu, cpu)
    return output, {
        "backend_chunk_spy_delta": _delta(backend_counts, before_backend),
        "prepare_varlen_spy_delta": _delta(prepare_counts, before_prepare),
    }


def _check_probe_expectations(cell: DiagnosticCell, probes: Mapping[str, Mapping[str, object]]) -> None:
    expected_prepare = {
        "public_c1": {"prepare_varlen": 2},
        "direct_c1": {"prepare_varlen": 1},
        "public_pinned": {"prepare_varlen": 0},
        "direct_pinned": {"prepare_varlen": 0},
    }
    expected_chunk = {
        "public_c1": {"c1": 1, "pinned": 0},
        "direct_c1": {"c1": 0, "pinned": 0},
        "public_pinned": {"c1": 0, "pinned": 1},
        "direct_pinned": {"c1": 0, "pinned": 0},
    }
    for path in PATHS:
        probe = probes[path]
        if probe["prepare_varlen_spy_delta"] != expected_prepare[path]:
            raise AssertionError(
                f"{cell.key}/{path}: _prepare_varlen probe drift: {probe['prepare_varlen_spy_delta']}"
            )
        if probe["backend_chunk_spy_delta"] != expected_chunk[path]:
            raise AssertionError(
                f"{cell.key}/{path}: public backend route probe drift: {probe['backend_chunk_spy_delta']}"
            )


def _verify_backends(c1: object, pinned: object, x: object, gpu: object, cpu: object, cell: DiagnosticCell) -> dict[str, object]:
    _select_path("public_c1")
    c1_ok, c1_reason = integration._verify(c1, x, None, False, gpu, cpu)
    pinned_ok, pinned_reason = integration._verify(pinned, x, None, False, gpu, cpu)
    if not c1_ok or not pinned_ok:
        raise AssertionError(
            f"{cell.key}: verifier failure c1={c1_reason!r}, pinned={pinned_reason!r}"
        )
    return {
        "c1": {"passed": True, "reason": c1_reason},
        "pinned": {"passed": True, "reason": pinned_reason},
    }


def _probe_correctness_and_route(
    cell: DiagnosticCell,
    x: object,
    cpu: object,
    gpu: object,
    public_fn: Callable[..., Any],
    c1: object,
    pinned: object,
) -> dict[str, object]:
    """Before timing, prove four paths are bitwise equal and actually routed."""

    import torch

    verifier = _verify_backends(c1, pinned, x, gpu, cpu, cell)
    snapshot = integration._snapshot_input_tensors(x, gpu, cpu, None)
    varlen_metadata.clear_cache()
    with _backend_chunk_spies(c1, pinned) as (originals, backend_counts):
        with _prepare_varlen_spy(c1) as prepare_counts:
            outputs: dict[str, tuple[object, object | None]] = {}
            probes: dict[str, dict[str, object]] = {}
            direct_decision: dict[str, object] | None = None
            public_decision: dict[str, object] | None = None
            for path in PATHS:
                output, probe = _call_probe(
                    path,
                    public_fn,
                    originals["c1"],
                    originals["pinned"],
                    x,
                    gpu,
                    cpu,
                    backend_counts,
                    prepare_counts,
                )
                outputs[path] = output
                probes[path] = probe
                if path == "direct_c1":
                    direct_decision = auto_dispatch.get_last_decision()
                elif path == "public_c1":
                    public_decision = auto_dispatch.get_last_decision()
            torch.cuda.synchronize()
    _check_probe_expectations(cell, probes)
    if direct_decision is None or public_decision is None:
        raise AssertionError(f"{cell.key}: missing C1 dispatch decisions")
    for label, decision in (("direct_c1", direct_decision), ("public_c1", public_decision)):
        if decision.get("chosen_variant") != cell.expected_variant:
            raise AssertionError(f"{cell.key}/{label}: diagnostic map decision drift: {decision}")
        certified = decision.get("certified_varlen_offsets")
        if certified != list(_offsets(cell.case.lengths)):
            raise AssertionError(f"{cell.key}/{label}: CPU-canonical offsets drift: {decision}")
    immutability = integration._assert_input_immutability(
        f"{cell.key}/four_path_probe", snapshot, x, gpu, cpu, None
    )
    exact = {
        "direct_c1_vs_direct_pinned": integration._exact(
            outputs["direct_c1"], outputs["direct_pinned"], cell.case.sequences, False, f"{cell.key}/direct-c1-pinned"
        ),
        "public_c1_vs_direct_c1": integration._exact(
            outputs["public_c1"], outputs["direct_c1"], cell.case.sequences, False, f"{cell.key}/public-direct-c1"
        ),
        "public_pinned_vs_direct_pinned": integration._exact(
            outputs["public_pinned"], outputs["direct_pinned"], cell.case.sequences, False, f"{cell.key}/public-direct-pinned"
        ),
        "public_c1_vs_public_pinned": integration._exact(
            outputs["public_c1"], outputs["public_pinned"], cell.case.sequences, False, f"{cell.key}/public-c1-pinned"
        ),
    }
    return {
        "verifier": verifier,
        "four_path_bitwise_exact": exact,
        "input_immutability": immutability,
        "non_timed_prepare_varlen_probe": probes,
        "public_route_proof": {
            "public_c1": probes["public_c1"]["backend_chunk_spy_delta"],
            "public_pinned": probes["public_pinned"]["backend_chunk_spy_delta"],
            "passed": True,
        },
        "c1_dispatch_decisions": {
            "direct_c1": direct_decision,
            "public_c1": public_decision,
            "expected_variant": cell.expected_variant,
            "passed": True,
        },
        "passed": True,
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample vector")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float | int]:
    if len(values) != SAMPLES:
        raise ValueError(f"expected {SAMPLES} samples, got {len(values)}")
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("CUDA-event samples must be finite positive milliseconds")
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _cyclic_order(index: int) -> tuple[str, str, str, str]:
    shift = index % len(PATHS)
    return PATHS[shift:] + PATHS[:shift]


def _timing_repeat(
    cell: DiagnosticCell,
    x: object,
    cpu: object,
    gpu: object,
    public_fn: Callable[..., Any],
    direct_c1: Callable[..., Any],
    direct_pinned: Callable[..., Any],
    repeat_index: int,
) -> dict[str, object]:
    """Four-path cyclic CUDA-event timing, with no spies in the sample path."""

    import torch

    integration._require_inference_mode("public overhead timing")
    varlen_metadata.clear_cache()
    warm_first = {path: 0 for path in PATHS}
    for warm_index in range(WARMUP):
        order = _cyclic_order(warm_index)
        warm_first[order[0]] += 1
        for path in order:
            _call_path(path, public_fn, direct_c1, direct_pinned, x, gpu, cpu)
    torch.cuda.synchronize()
    expected_first = WARMUP // len(PATHS)
    if set(warm_first.values()) != {expected_first}:
        raise AssertionError(f"{cell.key}/repeat{repeat_index}: warmup rotation imbalance {warm_first}")

    raw_samples: dict[str, list[float]] = {path: [] for path in PATHS}
    timed_first = {path: 0 for path in PATHS}
    current_stream = torch.cuda.current_stream()
    for sample_index in range(SAMPLES):
        order = _cyclic_order(sample_index)
        timed_first[order[0]] += 1
        for path in order:
            _select_path(path)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(current_stream)
            start.synchronize()
            _call_path(
                path, public_fn, direct_c1, direct_pinned, x, gpu, cpu, select_path=False
            )
            end.record(current_stream)
            end.synchronize()
            raw_samples[path].append(float(start.elapsed_time(end)))
            del start, end
    expected_timed_first = SAMPLES // len(PATHS)
    if set(timed_first.values()) != {expected_timed_first}:
        raise AssertionError(f"{cell.key}/repeat{repeat_index}: timed rotation imbalance {timed_first}")
    summaries = {path: _summary(values) for path, values in raw_samples.items()}
    differential_raw = [
        raw_samples["public_c1"][index]
        - raw_samples["direct_c1"][index]
        - raw_samples["public_pinned"][index]
        + raw_samples["direct_pinned"][index]
        for index in range(SAMPLES)
    ]
    differential = {
        percentile: (
            float(summaries["public_c1"][f"{percentile}_ms"])
            - float(summaries["direct_c1"][f"{percentile}_ms"])
            - float(summaries["public_pinned"][f"{percentile}_ms"])
            + float(summaries["direct_pinned"][f"{percentile}_ms"])
        )
        for percentile in ("p50", "p95", "p99")
    }
    return {
        "repeat_index": repeat_index,
        "event_contract": "start.record -> start.synchronize -> complete selected path -> end.record -> end.synchronize -> elapsed_time",
        "path_order": {
            "cycle": list(PATHS),
            "warmup_first_path_counts": warm_first,
            "timed_first_path_counts": timed_first,
            "passed": True,
        },
        "raw_samples_ms": raw_samples,
        "paths": summaries,
        "differential_formula": "(public_c1-direct_c1)-(public_pinned-direct_pinned)",
        "differential_raw_samples_ms": differential_raw,
        "differential_ms": differential,
        "passed": True,
    }


def _run_cell(
    cell: DiagnosticCell,
    public_fn: Callable[..., Any],
    c1: object,
    pinned: object,
    seed: int,
) -> dict[str, object]:
    import torch

    x = shared._make_inputs(cell.case, seed)
    try:
        cpu = integration._cpu_offsets(cell.case.lengths)
        gpu = x.cu_seqlens
        if gpu is None:
            raise AssertionError(f"{cell.key}: packed representative lost GPU offsets")
        correctness = _probe_correctness_and_route(cell, x, cpu, gpu, public_fn, c1, pinned)
        direct_c1, direct_pinned = c1.chunk_kda, pinned.chunk_kda
        repeats = [
            _timing_repeat(
                cell, x, cpu, gpu, public_fn, direct_c1, direct_pinned, repeat_index
            )
            for repeat_index in range(REPEATS)
        ]
        if len(repeats) != REPEATS or not all(bool(repeat["passed"]) for repeat in repeats):
            raise AssertionError(f"{cell.key}: incomplete diagnostic repeats")
        return {
            "expected_diagnostic_variant": cell.expected_variant,
            "contract": "none",
            "correctness_and_route": correctness,
            "repeats": repeats,
            "passed": True,
        }
    finally:
        del x
        torch.cuda.empty_cache()


def _gpu_main(args: argparse.Namespace, result: dict[str, object]) -> None:
    if (
        os.environ.get(CLEAN_GPU_GATE_ENV) != "1"
        or os.environ.get(GPU_AUTHORIZATION_ENV) != "1"
        or os.environ.get("C1_B300_FLASH_KDA") != "1"
        or os.environ.get("C1_B300_VARLEN_CPU_DESCRIPTOR") != "1"
    ):
        raise RuntimeError(
            "clean shell, explicit diagnostic authorization, C1 opt-in, and CPU-authoritative varlen opt-in are required"
        )
    if args.reference_root is None:
        raise ValueError("--reference-root is required")
    patched_text, fla_text = os.environ.get("PATCHED_ROOT"), os.environ.get("FLA_ROOT")
    if not patched_text or not fla_text:
        raise RuntimeError("PATCHED_ROOT and FLA_ROOT are required")

    result["identity"] = {"diagnostic_runner": _runner_identity()}
    _write(args.json, result)
    pre_torch_clean = integration._python_clean_gpu_gate()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shared.torch = torch
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import fla_backend
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common

    shared.common = common
    identity = integration._identity(Path(patched_text), Path(fla_text), args.reference_root)
    identity["diagnostic_runner"] = result["identity"]["diagnostic_runner"]
    identity["runtime_import_identities"] = integration._runtime_dependency_identities(
        args, common, fla_backend
    )
    _, helper = confirmation._load_pinned_reference_without_build(common, args.reference_root)
    identity["pinned_reference_helper"] = helper
    identity["python_pre_torch_nvidia_smi"] = pre_torch_clean
    result["identity"] = identity
    result["gates"].update({
        "clean_gpu": {"passed": True},
        "device": {"passed": True},
        "extension": {"passed": True},
        "fla_pin": {"passed": True},
        "runtime_dependencies": {"passed": True},
    })
    _write(args.json, result)

    from fla.ops.kda import chunk_kda

    c1, pinned, _registry, snapshot = integration._registry_backends()
    result["registry"] = {
        "snapshot": snapshot,
        "c1_id": id(c1),
        "pinned_id": id(pinned),
        "timing_has_no_spies": True,
    }
    _assert_production_map()
    original_map = getattr(auto_dispatch, "_VARLEN_PUBLIC_VARIANTS")
    if not isinstance(original_map, dict) or dict(original_map) != PRODUCTION_PUBLIC_VARIANTS:
        raise RuntimeError("lost exact production map object before temporary diagnostic installation")
    previous_c1 = os.environ.get("C1_B300_FLASH_KDA")
    previous_pinned = os.environ.get("FLA_FLASH_KDA")
    installed = False
    restore_error: Exception | None = None
    try:
        auto_dispatch._VARLEN_PUBLIC_VARIANTS = dict(FROZEN_R4_DIAGNOSTIC_VARIANTS)
        installed = True
        if dict(auto_dispatch._VARLEN_PUBLIC_VARIANTS) != FROZEN_R4_DIAGNOSTIC_VARIANTS:
            raise RuntimeError("temporary r4 diagnostic map installation drift")
        result["maps"]["temporary_installation"] = {"attempted": True, "passed": True}
        with torch.inference_mode():
            if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
                raise RuntimeError("failed to enter diagnostic inference mode")
            result["gates"]["inference_mode"] = {
                "grad_enabled": False,
                "inference_mode_enabled": True,
                "passed": True,
            }
            _write(args.json, result)
            for cell_index, cell in enumerate(REPRESENTATIVE_CELLS):
                result["cells"][cell.key] = _run_cell(
                    cell, chunk_kda, c1, pinned, args.seed + cell_index * 1009
                )
                _write(args.json, result)
        torch.cuda.synchronize()
        result["complete"] = True
    finally:
        try:
            if installed:
                auto_dispatch._VARLEN_PUBLIC_VARIANTS = original_map
            restored = dict(getattr(auto_dispatch, "_VARLEN_PUBLIC_VARIANTS", {})) == PRODUCTION_PUBLIC_VARIANTS
            result["maps"]["finally_restored_r5"] = {"attempted": True, "passed": restored}
            result["gates"]["temporary_map_restored"] = {"passed": restored}
            if not restored:
                raise RuntimeError("failed to restore exact r5 production packed-varlen map")
        except Exception as exc:  # Preserve the original diagnostic exception as context.
            restore_error = exc
        finally:
            _restore_env("C1_B300_FLASH_KDA", previous_c1)
            _restore_env("FLA_FLASH_KDA", previous_pinned)
            _write(args.json, result)
        if restore_error is not None:
            raise restore_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--describe", action="store_true", help="write the diagnostic plan without importing Torch/FLA")
    parser.add_argument("--cpu-construction-check", action="store_true", help="exercise CPU descriptor issuance without CUDA")
    args = parser.parse_args()
    if args.json.suffix.lower() != ".json":
        raise ValueError("--json output must use a .json suffix")
    _require_static_contracts()
    production_map = _assert_production_map()
    result = _initial_result(args, production_map)
    if args.describe:
        result["describe_only"] = True
        _write(args.json, result)
        print(f"wrote public-overhead diagnostic plan {args.json}")
        return
    if args.cpu_construction_check:
        result["cpu_construction_check"] = integration._cpu_construction_check()
        result["cpu_only"] = True
        _write(args.json, result)
        print(f"wrote public-overhead CPU construction check {args.json}")
        return
    _gpu_main(args, result)
    print(f"wrote public-overhead diagnostic {args.json}")


if __name__ == "__main__":
    main()
