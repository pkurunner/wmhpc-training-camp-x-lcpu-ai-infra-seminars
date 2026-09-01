#!/usr/bin/env python3
"""Fail-closed, discovery-only B=5 fixed-batch raw-wrapper experiment.

This runner deliberately calls the three already-audited raw ABI wrappers
directly.  It neither imports nor mutates the public dispatcher, its maps, or
any production source.  A valid result is evidence for a *separate*
confirmation allocation only; it never produces a production mapping.

The outer clean-audit shell freezes all source identities and invokes two
fresh Python processes.  This file records enough raw data for the companion
stdlib-only analyzer to independently recompute every latency decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_seqcount_dispatch import (  # noqa: E402
    run_seqcount_dispatch as shared,
)


DIM = 128
BATCH = 5
HEADS = 12
TOKENS = 2048
RAW_CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
PERFORMANCE_CONTRACTS = ("none", "fp32_final_only", "fp32_both")
VARIANTS = ("baseline", "vshard2_p2", "vshard4_p2")
PERCENTILES = ("p50", "p95", "p99")
WARMUP_PER_PATH = 100
SAMPLES_PER_PATH = 1000
REPEATS_PER_PROCESS = 2
MIN_WINNER_MARGIN = 0.02
CLEAN_GPU_GATE_ENV = "C1_FIXED_BATCH_B5_DISCOVERY_CLEAN_GPU"
EXPECTED_PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_REFERENCE_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
EXPECTED_EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_HELPER_SHA256 = "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
EXPECTED_FLASH_KDA_PYTHON_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
EXPECTED_PINNED_LOADER_SHA256 = "9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b"


def _write(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically persist even a long-running process's completed evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_lowercase_sha(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA256")


def _case() -> shared.Case:
    return shared.Case(
        name="b5_h12_t2048",
        form="fixed",
        sequences=BATCH,
        heads=HEADS,
        lengths=(TOKENS,) * BATCH,
        family="fixed_batch_b5_discovery_only",
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: list[float], expected_samples: int) -> dict[str, float | int]:
    if len(values) != expected_samples:
        raise ValueError(f"expected {expected_samples} samples, got {len(values)}")
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("CUDA-event samples must be finite and positive")
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _winner_and_margins(paths: Mapping[str, Mapping[str, float | int]]) -> dict[str, object]:
    winners: dict[str, str] = {}
    margins: dict[str, float] = {}
    for percentile_name in PERCENTILES:
        metric = f"{percentile_name}_ms"
        ranked = sorted(
            ((float(paths[label][metric]), label) for label in VARIANTS),
            key=lambda item: item[0],
        )
        winner_value, winner = ranked[0]
        runner_up_value, _ = ranked[1]
        winners[percentile_name] = winner
        margins[percentile_name] = runner_up_value / winner_value - 1.0
    single_winner = len(set(winners.values())) == 1
    margin_gate = all(value >= MIN_WINNER_MARGIN for value in margins.values())
    return {
        "winner_by_percentile": winners,
        "winner_margin_over_runner_up": margins,
        "single_winner_all_percentiles": single_winner,
        "margin_gate_pass": margin_gate,
        "repeat_gate_pass": single_winner and margin_gate,
    }


def _snapshot_inputs(x: Any) -> dict[str, object]:
    """Retain exact pre-call copies of every immutable raw input tensor."""

    tensor_fields = ("q", "k", "v", "g", "beta", "a_log", "dt_bias", "cu_seqlens")
    snapshot: dict[str, object] = {
        "scale": x.scale,
        "lower_bound": x.lower_bound,
    }
    for field in tensor_fields:
        value = getattr(x, field)
        snapshot[field] = None if value is None else value.clone()
    return snapshot


def _assert_inputs_immutable(label: str, snapshot: Mapping[str, object], x: Any) -> dict[str, object]:
    import torch

    fields: dict[str, bool] = {}
    for field in ("q", "k", "v", "g", "beta", "a_log", "dt_bias", "cu_seqlens"):
        before = snapshot[field]
        after = getattr(x, field)
        if before is None:
            passed = after is None
        else:
            passed = torch.equal(before, after)
        fields[field] = passed
    fields["scale"] = x.scale == snapshot["scale"]
    fields["lower_bound"] = x.lower_bound == snapshot["lower_bound"]
    if not all(fields.values()):
        failed = [name for name, passed in fields.items() if not passed]
        raise AssertionError(f"{label}: input mutation detected in {failed}")
    return {"input_immutability_exact": True, "input_immutability_fields": fields}


def _assert_initial_immutable(label: str, before: Any, after: Any) -> dict[str, object]:
    import torch

    if before is None:
        passed = after is None
    else:
        passed = torch.equal(before, after)
    if not passed:
        raise AssertionError(f"{label}: initial_state mutation detected")
    return {"initial_state_immutability_exact": True}


def _invoke_checked(
    label: str,
    fn: Callable[..., None],
    x: Any,
    initial_template: Any,
    final_template: Any,
    input_snapshot: Mapping[str, object],
) -> tuple[tuple[Any, Any], dict[str, object]]:
    """One raw ABI invocation plus input/initial immutability proof."""

    initial = shared._clone(initial_template)
    initial_before = shared._clone(initial)
    final = shared._clone(final_template)
    output = shared._invoke(fn, x, initial, final)
    evidence = _assert_inputs_immutable(label, input_snapshot, x)
    evidence.update(_assert_initial_immutable(label, initial_before, initial))
    return output, evidence


def _compare(
    label: str,
    actual: tuple[Any, Any],
    expected: tuple[Any, Any],
) -> dict[str, object]:
    """Keep the raw output/final-state exactness contract explicit in JSON."""

    from assignment02.team.c1_flashkda.harness import validate_and_bench as common

    common.require_exact(f"{label}/output", actual[0], expected[0])
    evidence: dict[str, object] = {
        "output_exact": True,
        "output_max_abs": common.max_abs(actual[0], expected[0]),
    }
    if actual[1] is None or expected[1] is None:
        if actual[1] is not None or expected[1] is not None:
            raise AssertionError(f"{label}: final-state presence mismatch")
        evidence["final_state_present"] = False
    else:
        common.require_exact(f"{label}/final_state", actual[1], expected[1])
        evidence.update(
            {
                "final_state_present": True,
                "final_state_exact": True,
                "final_state_max_abs": common.max_abs(actual[1], expected[1]),
            }
        )
    return evidence


def _correctness(
    *,
    case: shared.Case,
    x: Any,
    contract: str,
    seed: int,
    functions: Mapping[str, Callable[..., None]],
    torch_ref: Callable[..., None],
) -> dict[str, object]:
    """Exact raw ABI checks for all wrappers and the pinned Torch oracle."""

    initial_template, final_template = shared._states(contract, case, seed)
    input_snapshot = _snapshot_inputs(x)
    outputs: dict[str, tuple[Any, Any]] = {}
    invocation_immutability: dict[str, object] = {}
    for variant in VARIANTS:
        output, immutable = _invoke_checked(
            f"correctness/{case.name}/{contract}/{variant}",
            functions[variant],
            x,
            initial_template,
            final_template,
            input_snapshot,
        )
        outputs[variant] = output
        invocation_immutability[variant] = immutable
    baseline = outputs["baseline"]
    direct = {
        "baseline_vs_vshard2_p2": _compare(
            f"{case.name}/{contract}/baseline_vs_vshard2_p2", outputs["vshard2_p2"], baseline
        ),
        "baseline_vs_vshard4_p2": _compare(
            f"{case.name}/{contract}/baseline_vs_vshard4_p2", outputs["vshard4_p2"], baseline
        ),
    }
    reference_evidence: dict[str, object] | None = None
    if contract in PERFORMANCE_CONTRACTS:
        reference_output, reference_immutable = _invoke_checked(
            f"correctness/{case.name}/{contract}/pinned_torch_reference",
            torch_ref,
            x,
            initial_template,
            final_template,
            input_snapshot,
        )
        reference_evidence = {
            "baseline_vs_pinned_torch_reference": _compare(
                f"{case.name}/{contract}/baseline_vs_pinned_torch_reference", reference_output, baseline
            ),
            "invocation_immutability": reference_immutable,
        }
    return {
        "contract": contract,
        "direct_wrapper_exactness": direct,
        "pinned_torch_reference_exactness": reference_evidence,
        "invocation_immutability": invocation_immutability,
        "passed": True,
    }


def _benchmark_repeat(
    *,
    process_index: int,
    repeat_index: int,
    case: shared.Case,
    contract: str,
    seed: int,
    functions: Mapping[str, Callable[..., None]],
) -> dict[str, object]:
    """Cyclic three-path raw-wrapper CUDA-event timing with raw samples."""

    import torch

    x = shared._make_inputs(case, seed)
    input_snapshot = _snapshot_inputs(x)
    initial_template, final_template = shared._states(contract, case, seed + 101)
    calls: dict[str, Callable[[], None]] = {}
    initial_snapshots: dict[str, object] = {}
    initial_values: dict[str, object] = {}
    for variant in VARIANTS:
        initial = shared._clone(initial_template)
        final = shared._clone(final_template)
        initial_values[variant] = initial
        initial_snapshots[variant] = shared._clone(initial)
        output = torch.empty_like(x.v)

        def call(
            fn: Callable[..., None] = functions[variant],
            initial_state: Any = initial,
            final_state: Any = final,
            result: Any = output,
        ) -> None:
            fn(
                x.q,
                x.k,
                x.v,
                x.g,
                x.beta,
                x.scale,
                result,
                A_log=x.a_log,
                dt_bias=x.dt_bias,
                lower_bound=x.lower_bound,
                initial_state=initial_state,
                final_state=final_state,
                cu_seqlens=x.cu_seqlens,
            )

        calls[variant] = call

    # Every path receives exactly 100 warm-up calls, but its position rotates
    # to avoid systematically assigning host-side setup to one variant.
    for warm_index in range(WARMUP_PER_PATH):
        offset = warm_index % len(VARIANTS)
        for variant in VARIANTS[offset:] + VARIANTS[:offset]:
            calls[variant]()
    torch.cuda.synchronize()

    raw_samples = {variant: [] for variant in VARIANTS}
    first_path_counts = {variant: 0 for variant in VARIANTS}
    stream = torch.cuda.current_stream()
    for sample_index in range(SAMPLES_PER_PATH):
        offset = sample_index % len(VARIANTS)
        order = VARIANTS[offset:] + VARIANTS[:offset]
        first_path_counts[order[0]] += 1
        for variant in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            calls[variant]()
            end.record(stream)
            end.synchronize()
            raw_samples[variant].append(float(start.elapsed_time(end)))
            del start, end
    torch.cuda.synchronize()

    input_evidence = _assert_inputs_immutable(
        f"performance/{process_index}/{repeat_index}/{contract}", input_snapshot, x
    )
    initial_fields: dict[str, bool] = {}
    for variant in VARIANTS:
        _assert_initial_immutable(
            f"performance/{process_index}/{repeat_index}/{contract}/{variant}",
            initial_snapshots[variant],
            initial_values[variant],
        )
        initial_fields[variant] = True
    paths = {variant: _summary(raw_samples[variant], SAMPLES_PER_PATH) for variant in VARIANTS}
    evidence = _winner_and_margins(paths)
    expected_first_counts = {
        "baseline": 334,
        "vshard2_p2": 333,
        "vshard4_p2": 333,
    }
    if first_path_counts != expected_first_counts:
        raise AssertionError(f"cyclic first-path schedule drift: {first_path_counts}")
    return {
        "process_index": process_index,
        "repeat_index": repeat_index,
        "input_seed": seed,
        "state_seed": seed + 101,
        "event_contract": (
            "current-stream start event -> one direct raw ABI wrapper call -> end event -> "
            "end.synchronize -> elapsed_time; synchronization is not included as a sample"
        ),
        "schedule": "three-path cyclic rotation; every path receives 100 warmups and 1000 timed samples",
        "path_order": {
            "variants": list(VARIANTS),
            "offset_rule": "sample_or_warmup_index modulo three rotates the first path",
            "timed_first_path_counts": first_path_counts,
        },
        "warmup_calls_per_path": {variant: WARMUP_PER_PATH for variant in VARIANTS},
        "timed_calls_per_path": {variant: SAMPLES_PER_PATH for variant in VARIANTS},
        **input_evidence,
        "initial_state_immutability_exact": all(initial_fields.values()),
        "initial_state_immutability_by_variant": initial_fields,
        "raw_samples_ms": raw_samples,
        "paths": paths,
        **evidence,
        "passed": True,
    }


def _gpu_uuid() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one visible GPU UUID, got {values!r}")
    return values[0]


def _runner_identity(
    *,
    expected_runner_sha256: str,
    reference_root: Path,
    patched_root: Path,
    fla_root: Path,
    reference_helper_identity: Mapping[str, object],
    flash_kda_module: Any,
    pinned_reference_loader_module: Any,
) -> dict[str, object]:
    _require_lowercase_sha(expected_runner_sha256, "--expected-runner-sha256")
    runner_path = Path(__file__).resolve(strict=True)
    runner_sha = _sha(runner_path)
    if runner_sha != expected_runner_sha256:
        raise RuntimeError(
            f"runner SHA mismatch: expected {expected_runner_sha256}, got {runner_sha}"
        )
    helper_path_text = os.environ.get("C1_PINNED_REFERENCE_HELPER_PATH")
    if not helper_path_text:
        raise RuntimeError("C1_PINNED_REFERENCE_HELPER_PATH is required")
    helper_path = Path(helper_path_text)
    helper_sha = _sha(helper_path)
    if helper_sha != EXPECTED_HELPER_SHA256:
        raise RuntimeError("pinned reference helper SHA mismatch")
    if os.environ.get("C1_PINNED_REFERENCE_HELPER_SHA256") != EXPECTED_HELPER_SHA256:
        raise RuntimeError("pinned reference helper SHA environment drift")
    if reference_helper_identity.get("path") != str(helper_path.resolve()):
        raise RuntimeError("pinned reference helper load path drift")
    if reference_helper_identity.get("sha256") != EXPECTED_HELPER_SHA256:
        raise RuntimeError("pinned reference helper load SHA drift")
    if reference_helper_identity.get("no_build") is not True:
        raise RuntimeError("pinned reference helper loader did not prove no-build operation")
    if reference_helper_identity.get("intercepted_names") != ["sigmoid_ext"]:
        raise RuntimeError("pinned reference helper load_inline interception drift")
    expected_flash_kda_path = (patched_root / "flash_kda" / "__init__.py").resolve(strict=True)
    flash_kda_path = Path(flash_kda_module.__file__).resolve(strict=True)
    if flash_kda_path != expected_flash_kda_path:
        raise RuntimeError(
            f"loaded flash_kda Python path drift: expected {expected_flash_kda_path}, got {flash_kda_path}"
        )
    flash_kda_sha = _sha(flash_kda_path)
    if flash_kda_sha != EXPECTED_FLASH_KDA_PYTHON_SHA256:
        raise RuntimeError("loaded flash_kda Python SHA drift")
    expected_loader_path = (
        REPO_ROOT
        / "assignment02"
        / "team"
        / "c1_flashkda"
        / "challenge_varlen_dispatch"
        / "run_varlen_dispatch_confirmation.py"
    ).resolve(strict=True)
    loader_path = Path(pinned_reference_loader_module.__file__).resolve(strict=True)
    if loader_path != expected_loader_path:
        raise RuntimeError(
            f"loaded pinned-reference loader path drift: expected {expected_loader_path}, got {loader_path}"
        )
    loader_sha = _sha(loader_path)
    if loader_sha != EXPECTED_PINNED_LOADER_SHA256:
        raise RuntimeError("loaded pinned-reference loader SHA drift")
    torch_ref_path = reference_root / "tests" / "torch_ref.py"
    if not torch_ref_path.is_file():
        raise RuntimeError(f"missing pinned Torch reference: {torch_ref_path}")
    commits = {
        "patched": _git_commit(patched_root),
        "reference": _git_commit(reference_root),
        "fla": _git_commit(fla_root),
    }
    expected_commits = {
        "patched": EXPECTED_PATCHED_COMMIT,
        "reference": EXPECTED_REFERENCE_COMMIT,
        "fla": EXPECTED_FLA_COMMIT,
    }
    if commits != expected_commits:
        raise RuntimeError(f"pinned commit drift: expected {expected_commits}, got {commits}")
    return {
        "runner": {
            "path": str(runner_path),
            "sha256": runner_sha,
            "sha256_gate_pass": True,
        },
        "device": shared._device_identity(),
        "gpu_uuid": _gpu_uuid(),
        "extension": shared._identity(),
        "flash_kda_python": {
            "path": str(flash_kda_path),
            "sha256": flash_kda_sha,
            "sha256_gate_pass": True,
        },
        "pinned_reference_loader": {
            "path": str(loader_path),
            "sha256": loader_sha,
            "sha256_gate_pass": True,
        },
        "commits": commits,
        "reference_torch_ref": {"path": str(torch_ref_path), "sha256": _sha(torch_ref_path)},
        "pinned_reference_helper": {
            "path": str(helper_path.resolve()),
            "sha256": helper_sha,
            "sha256_gate_pass": True,
            "load_contract": reference_helper_identity.get("load_contract"),
            "intercepted_names": reference_helper_identity.get("intercepted_names"),
            "no_build": True,
        },
    }


def _initial_result(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "purpose": (
            "B=5 fixed-batch raw-wrapper discovery only; direct calls only, no dispatcher/map "
            "mutation, no production mapping, and no release authority"
        ),
        "diagnostic_only": True,
        "discovery_only": True,
        "no_release_authority": True,
        "no_production_mapping": True,
        "process": {
            "pid": os.getpid(),
            "process_index": args.process_index,
            "fresh_python_process_required": True,
        },
        "target": {
            "B": BATCH,
            "H": HEADS,
            "T": TOKENS,
            "K": DIM,
            "V": DIM,
            "form": "fixed",
            "case_name": _case().name,
            "lengths": [TOKENS] * BATCH,
        },
        "pre_registered": {
            "raw_abi_contracts": list(RAW_CONTRACTS),
            "pinned_torch_reference_contracts": list(PERFORMANCE_CONTRACTS),
            "performance_contracts": list(PERFORMANCE_CONTRACTS),
            "variants": list(VARIANTS),
            "fresh_main_processes": 2,
            "required_process_indices": [0, 1],
            "repeats_per_process": REPEATS_PER_PROCESS,
            "warmup_per_path_per_repeat": WARMUP_PER_PATH,
            "cuda_event_samples_per_path_per_repeat": SAMPLES_PER_PATH,
            "required_percentiles": list(PERCENTILES),
            "minimum_runner_up_margin": MIN_WINNER_MARGIN,
            "optimized_winner_required": ["vshard2_p2", "vshard4_p2"],
            "performance_success_rule": (
                "For each public state contract, all four repeats across the two fresh processes "
                "must choose one identical non-baseline winner at P50/P95/P99, with that custom "
                "winner's runner-up margin at least 2% at every percentile.  A baseline winner is "
                "a valid negative and is not confirmation-eligible.  Passing only permits "
                "independent confirmation, never a production mapping."
            ),
        },
        "identity": {},
        "gates": {
            "clean_gpu_shell": {"required": True, "passed": False},
            "device": {"required": "B300 capability 10.3 with 148 SMs", "passed": False},
            "extension": {"required_sha256": EXPECTED_EXTENSION_SHA256, "passed": False},
            "pinned_commits": {"required": True, "passed": False},
            "runner_source": {"required": True, "passed": False},
            "python_sources": {
                "required_flash_kda_sha256": EXPECTED_FLASH_KDA_PYTHON_SHA256,
                "required_pinned_loader_sha256": EXPECTED_PINNED_LOADER_SHA256,
                "passed": False,
            },
            "pinned_reference_helper": {"required_sha256": EXPECTED_HELPER_SHA256, "passed": False},
            "no_dispatcher_or_map_mutation": {
                "method": "direct raw ABI wrapper calls only; auto_dispatch is not imported",
                "passed": True,
            },
            "raw_abi_exact": {"required": True, "passed": False},
            "pinned_torch_reference_exact": {"required": True, "passed": False},
        },
        "correctness": {},
        "performance": {},
        "complete": False,
    }


def _check_args(args: argparse.Namespace) -> None:
    if args.process_index not in (0, 1):
        raise ValueError("--process-index must be 0 or 1")
    _require_lowercase_sha(args.expected_runner_sha256, "--expected-runner-sha256")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--patched-root", type=Path)
    parser.add_argument("--fla-root", type=Path)
    parser.add_argument("--describe", action="store_true", help="write the design without importing Torch/CUDA")
    args = parser.parse_args()
    _check_args(args)

    result = _initial_result(args)
    if args.describe:
        result["describe_only"] = True
        result["identity"] = {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha(Path(__file__).resolve()),
                "sha256_gate_pass": _sha(Path(__file__).resolve()) == args.expected_runner_sha256,
            }
        }
        if result["identity"]["runner"]["sha256_gate_pass"] is not True:  # type: ignore[index]
            raise RuntimeError("runner SHA mismatch in describe mode")
        _write(args.json, result)
        print(f"wrote B=5 discovery description {args.json}")
        return

    if os.environ.get(CLEAN_GPU_GATE_ENV) != "1":
        raise RuntimeError(
            "refusing a direct GPU run: use run_clean_fixed_batch_b5_discovery.sh so the "
            "clean-GPU authorization is established before this process starts"
        )
    if args.reference_root is None or args.patched_root is None or args.fla_root is None:
        raise ValueError("--reference-root, --patched-root, and --fla-root are required for a GPU run")

    # Keep planning mode import-free.  The direct GPU path deliberately uses
    # the same audited input/state constructors and wrapper ABI as r2/r6.
    import torch
    from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2
    from assignment02.team.c1_flashkda.challenge_vshard4_prefetch2 import vshard4_prefetch2
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common
    from assignment02.team.c1_flashkda.challenge_varlen_dispatch import (
        run_varlen_dispatch_confirmation as pinned_reference_loader,
    )

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
    # The upstream reference imports a CUDA helper through load_inline.  Reuse
    # the existing audited loader, which direct-loads the pinned binary and
    # intercepts exactly that one request; no compilation is possible here.
    torch_ref, reference_helper_identity = pinned_reference_loader._load_pinned_reference_without_build(
        common, args.reference_root
    )
    result["identity"] = _runner_identity(
        expected_runner_sha256=args.expected_runner_sha256,
        reference_root=args.reference_root,
        patched_root=args.patched_root,
        fla_root=args.fla_root,
        reference_helper_identity=reference_helper_identity,
        flash_kda_module=flash_kda,
        pinned_reference_loader_module=pinned_reference_loader,
    )
    result["gates"]["clean_gpu_shell"]["passed"] = True  # type: ignore[index]
    result["gates"]["device"]["passed"] = True  # type: ignore[index]
    result["gates"]["extension"]["passed"] = True  # type: ignore[index]
    result["gates"]["pinned_commits"]["passed"] = True  # type: ignore[index]
    result["gates"]["runner_source"]["passed"] = True  # type: ignore[index]
    result["gates"]["python_sources"]["passed"] = True  # type: ignore[index]
    result["gates"]["pinned_reference_helper"]["passed"] = True  # type: ignore[index]
    _write(args.json, result)

    case = _case()
    for contract_index, contract in enumerate(RAW_CONTRACTS):
        print(f"B=5 raw ABI exactness {contract}")
        x = shared._make_inputs(case, args.seed + args.process_index * 100_003 + contract_index * 1_009)
        result["correctness"][contract] = _correctness(  # type: ignore[index]
            case=case,
            x=x,
            contract=contract,
            seed=args.seed + args.process_index * 100_003 + contract_index * 1_009 + 101,
            functions=functions,
            torch_ref=torch_ref,
        )
        del x
        torch.cuda.empty_cache()
        _write(args.json, result)
    result["gates"]["raw_abi_exact"]["passed"] = True  # type: ignore[index]
    result["gates"]["pinned_torch_reference_exact"]["passed"] = True  # type: ignore[index]
    _write(args.json, result)

    for contract_index, contract in enumerate(PERFORMANCE_CONTRACTS):
        repeats: list[dict[str, object]] = []
        result["performance"][contract] = {"repeats": repeats}  # type: ignore[index]
        _write(args.json, result)
        for repeat_index in range(REPEATS_PER_PROCESS):
            seed = (
                args.seed
                + args.process_index * 1_000_003
                + contract_index * 10_007
                + repeat_index * 1_009
            )
            print(
                f"B=5 direct cyclic benchmark process={args.process_index} "
                f"contract={contract} repeat={repeat_index} samples={SAMPLES_PER_PATH}"
            )
            repeats.append(
                _benchmark_repeat(
                    process_index=args.process_index,
                    repeat_index=repeat_index,
                    case=case,
                    contract=contract,
                    seed=seed,
                    functions=functions,
                )
            )
            torch.cuda.empty_cache()
            _write(args.json, result)

    result["complete"] = True
    _write(args.json, result)
    print(f"wrote B=5 discovery artifact {args.json}; discovery-only/no-release-authority")


if __name__ == "__main__":
    main()
