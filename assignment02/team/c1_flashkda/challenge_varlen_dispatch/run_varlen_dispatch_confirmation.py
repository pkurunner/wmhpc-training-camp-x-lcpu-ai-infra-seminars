#!/usr/bin/env python3
"""Pre-registered raw-wrapper confirmation for four exact packed-varlen cases.

This is an evidence runner, not a dispatcher change.  It reuses the audited
raw ABI wrappers and state construction from the sequence-count experiment,
then independently derives percentile winners from the recorded CUDA-event
samples.  Only the eleven explicitly listed cells can ever be promotion
candidates; the mixed-N6 FP32-both cell is collected as record-only evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_seqcount_dispatch import (  # noqa: E402
    run_seqcount_dispatch as shared,
)


DIM = 128
DEFAULT_SEED = 20260829
SAMPLES_PER_REPEAT = 1000
REPEATS = 2
WARMUP = 100
FLA_PUBLIC_CONTRACTS = ("none", "fp32_final_only", "fp32_both")
RAW_CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
MIN_WINNER_MARGIN = 0.02
CLEAN_GPU_GATE_ENV = "C1_VARLEN_DISPATCH_CLEAN_GPU_GATES"
REFERENCE_HELPER_PATH_ENV = "C1_PINNED_REFERENCE_HELPER_PATH"
REFERENCE_HELPER_SHA_ENV = "C1_PINNED_REFERENCE_HELPER_SHA256"


def _case(name: str, form: str, lengths: tuple[int, ...]) -> shared.Case:
    return shared.Case(
        name=name,
        form=form,
        sequences=len(lengths),
        heads=12,
        lengths=lengths,
        family="varlen_dispatch_confirmation",
    )


CASES = (
    _case("equal_n2_h12_t2048", "balanced_varlen", (2048, 2048)),
    _case("equal_n4_h12_t2048", "balanced_varlen", (2048, 2048, 2048, 2048)),
    _case("mixed_n6_h12_t8192", "mixed_varlen", (17, 511, 1024, 1300, 2049, 3291)),
    _case("skew_n6_h12_t12288", "skewed_varlen", (1, 1, 1, 1, 1, 12283)),
)

# This table is the complete eligible population.  Do not infer additional
# shapes or generalize its result to a dispatcher policy from this runner.
PROMOTION_CELLS = {
    "equal_n2_h12_t2048": {
        "none": "vshard4_p2",
        "fp32_final_only": "vshard4_p2",
        "fp32_both": "vshard4_p2",
    },
    "equal_n4_h12_t2048": {
        "none": "vshard2_p2",
        "fp32_final_only": "vshard2_p2",
        "fp32_both": "vshard4_p2",
    },
    "mixed_n6_h12_t8192": {
        "none": "vshard2_p2",
        "fp32_final_only": "vshard2_p2",
    },
    "skew_n6_h12_t12288": {
        "none": "vshard2_p2",
        "fp32_final_only": "vshard2_p2",
        "fp32_both": "vshard4_p2",
    },
}
RECORD_ONLY_CELLS = (
    {
        "case": "mixed_n6_h12_t8192",
        "contract": "fp32_both",
        "promotion_gate_scope": False,
        "future_dispatch_variant": "baseline",
        "reason": "pre-registered discovery-only observation; it is excluded from the 11-cell promotion gate",
    },
)


def _assert_preregistration_scope() -> None:
    case_names = {case.name for case in CASES}
    public_contracts = set(FLA_PUBLIC_CONTRACTS)
    cells = [(case, contract) for case, contracts in PROMOTION_CELLS.items() for contract in contracts]
    if len(cells) != 11 or len(set(cells)) != 11:
        raise AssertionError(f"pre-registered promotion scope must contain exactly 11 cells, got {cells!r}")
    if any(case not in case_names or contract not in public_contracts for case, contract in cells):
        raise AssertionError("promotion cells are outside the fixed case/public-contract matrix")
    records = {(str(cell["case"]), str(cell["contract"])) for cell in RECORD_ONLY_CELLS}
    if records != {("mixed_n6_h12_t8192", "fp32_both")}:
        raise AssertionError(f"unexpected record-only scope: {records!r}")
    if set(cells) & records:
        raise AssertionError("a record-only cell must not be promotion-gated")
    measured = set(cells) | records
    full = {(case.name, contract) for case in CASES for contract in FLA_PUBLIC_CONTRACTS}
    if measured != full:
        raise AssertionError("the 12-cell public matrix must be 11 promotion cells plus one record-only cell")


def _write(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_pinned_reference_without_build(
    common: object, reference_root: Path
) -> tuple[Callable[..., None], dict[str, object]]:
    """Load the pinned helper binary directly and intercept exactly one load_inline call."""
    import torch.utils.cpp_extension as cpp_extension

    helper_text = os.environ.get(REFERENCE_HELPER_PATH_ENV)
    expected_sha = os.environ.get(REFERENCE_HELPER_SHA_ENV)
    if not helper_text or not expected_sha:
        raise RuntimeError("pinned reference helper identity environment is missing")
    helper_path = Path(helper_text).resolve()
    if not helper_path.is_file():
        raise FileNotFoundError(f"missing pinned reference helper: {helper_path}")
    actual_sha = hashlib.sha256(helper_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"pinned reference helper SHA mismatch: expected={expected_sha} actual={actual_sha}"
        )

    helper_spec = importlib.util.spec_from_file_location("sigmoid_ext", helper_path)
    if helper_spec is None or helper_spec.loader is None:
        raise RuntimeError(f"cannot import pinned reference helper {helper_path}")
    helper_module = importlib.util.module_from_spec(helper_spec)
    sys.modules["sigmoid_ext"] = helper_module
    helper_spec.loader.exec_module(helper_module)

    original_load_inline = cpp_extension.load_inline
    intercepted_names: list[str] = []

    def cached_load_inline(*args: object, **kwargs: object) -> object:
        name = kwargs.get("name", args[0] if args else None)
        if name != "sigmoid_ext":
            raise RuntimeError(f"unexpected load_inline request from pinned reference: {name!r}")
        intercepted_names.append(str(name))
        return helper_module

    cpp_extension.load_inline = cached_load_inline
    try:
        torch_ref = common._load_torch_ref(reference_root)
    finally:
        cpp_extension.load_inline = original_load_inline
    if intercepted_names != ["sigmoid_ext"]:
        raise RuntimeError(f"expected one intercepted sigmoid_ext request, got {intercepted_names!r}")
    return torch_ref, {
        "path": str(helper_path),
        "sha256": actual_sha,
        "load_contract": "direct cached binary; exactly one pinned load_inline('sigmoid_ext') intercepted",
        "intercepted_names": intercepted_names,
        "no_build": True,
    }


def _preregistration() -> dict[str, object]:
    return {
        "representation": "packed varlen only; B=1 with CUDA int64 cu_seqlens",
        "fixed_shape": {"H": 12, "K": DIM, "V": DIM},
        "cases": [shared._case_dict(case) for case in CASES],
        "raw_abi_contracts": list(RAW_CONTRACTS),
        "public_fla_contracts": list(FLA_PUBLIC_CONTRACTS),
        "promotion_cells": PROMOTION_CELLS,
        "promotion_cell_count": 11,
        "record_only_cells": list(RECORD_ONLY_CELLS),
        "repeats_per_public_cell": REPEATS,
        "fixed_seed": DEFAULT_SEED,
        "warmup_per_path": WARMUP,
        "cyclic_cuda_event_samples_per_repeat": SAMPLES_PER_REPEAT,
        "required_percentiles": list(shared.PERCENTILES),
        "minimum_expected_winner_margin_over_runner_up": MIN_WINNER_MARGIN,
        "stop_rule": (
            "A promotion cell fails if either independent repeat has a P50, P95, or P99 winner "
            "other than its pre-registered variant, or if the expected winner's margin over its "
            "runner-up is below 2% at any required percentile. Any failed promotion cell makes the "
            "global decision STOP; the record-only cell never changes that decision."
        ),
        "correctness_rule": (
            "For every case and raw ABI contract, vshard2_p2 and vshard4_p2 must be bitwise exact "
            "to baseline for output and, when present, final state. For every case and public contract, "
            "baseline must also be bitwise exact to the pinned Torch reference."
        ),
    }


def _initial_result(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "purpose": "pre-registered raw-wrapper packed-varlen B300 confirmation; no dispatcher mutation",
        "preregistration": _preregistration(),
        "seed": args.seed,
        "raw_abi_correctness": {},
        "raw_wrapper_public_contract_benchmarks": {},
        "identity": {},
        "gates": {
            "clean_gpu_shell_gate": {
                "environment_name": CLEAN_GPU_GATE_ENV,
                "required": True,
                "passed": False,
            },
            "device_gate": {"required": "B300, capability 10.3, 148 SM", "passed": False},
            "audited_extension_sha256_gate": {
                "required": shared.AUDITED_EXTENSION_SHA256,
                "passed": False,
            },
            "raw_abi_exact_gate": {"required": "four contracts across all four cases", "passed": False},
            "pinned_torch_reference_gate": {
                "required": "all four cases times all three public contracts", "passed": False
            },
            "confirmation_gate": {
                "scope_cell_count": 11,
                "passed": False,
                "decision": "not_run",
            },
        },
        "complete": False,
    }


def _raw_abi_exactness(
    functions: dict[str, Callable[..., None]],
    torch_ref: Callable[..., None],
    case: shared.Case,
    x: object,
    contract: str,
    seed: int,
) -> dict[str, object]:
    """Compare every raw wrapper, and public baseline, from the same cloned state."""
    initial, final = shared._states(contract, case, seed)
    input_snapshot = _snapshot_inputs(x)
    outputs = {}
    for label, fn in functions.items():
        outputs[label] = shared._invoke(fn, x, shared._clone(initial), shared._clone(final))
        _assert_inputs_unchanged(f"{case.name}/{contract}/{label}", x, input_snapshot)
    baseline = outputs["baseline"]
    result: dict[str, object] = {
        label: shared._compare(
            f"varlen_dispatch_confirmation/{case.name}/{contract}/{label}_vs_baseline",
            outputs[label],
            baseline,
        )
        for label in ("vshard2_p2", "vshard4_p2")
    }
    if contract in FLA_PUBLIC_CONTRACTS:
        reference = shared._invoke(torch_ref, x, shared._clone(initial), shared._clone(final))
        _assert_inputs_unchanged(f"{case.name}/{contract}/pinned_torch_ref", x, input_snapshot)
        result["baseline_vs_pinned_torch_ref"] = shared._compare(
            f"varlen_dispatch_confirmation/{case.name}/{contract}/baseline_vs_pinned_torch_ref",
            baseline,
            reference,
        )
    result["input_immutability_exact"] = True
    return result


_INPUT_TENSOR_FIELDS = ("q", "k", "v", "g", "beta", "a_log", "dt_bias", "cu_seqlens")


def _snapshot_inputs(x: object) -> dict[str, object]:
    return {
        field: (None if getattr(x, field) is None else getattr(x, field).clone())
        for field in _INPUT_TENSOR_FIELDS
    }


def _assert_inputs_unchanged(label: str, x: object, snapshot: dict[str, object]) -> None:
    for field, expected in snapshot.items():
        actual = getattr(x, field)
        if actual is None or expected is None:
            if actual is not None or expected is not None:
                raise AssertionError(f"{label}: input {field} presence changed")
            continue
        shared.common.require_exact(f"{label}/input_immutability/{field}", actual, expected)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: list[float]) -> dict[str, float | int]:
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("CUDA-event samples must all be finite positive milliseconds")
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _independent_summary(benchmark: dict[str, object]) -> dict[str, object]:
    """Derive all gate evidence solely from the raw event samples.

    The shared runner's summaries are cross-checked for auditability, but are
    never used to choose the winner or calculate the margin below.
    """
    raw = benchmark.get("raw_samples_ms")
    shared_paths = benchmark.get("paths")
    if not isinstance(raw, dict) or not isinstance(shared_paths, dict):
        raise TypeError("shared benchmark lacks raw samples or its audit summary")
    paths: dict[str, dict[str, float | int]] = {}
    for label in shared.VARIANTS:
        values = raw.get(label)
        if not isinstance(values, list) or len(values) != SAMPLES_PER_REPEAT:
            raise ValueError(f"{label}: expected exactly {SAMPLES_PER_REPEAT} raw samples")
        numeric = [float(value) for value in values]
        derived = _summary(numeric)
        recorded = shared_paths.get(label)
        if not isinstance(recorded, dict):
            raise TypeError(f"{label}: missing shared audit summary")
        for key, value in derived.items():
            recorded_value = recorded.get(key)
            if key == "samples":
                matches = recorded_value == value
            else:
                matches = isinstance(recorded_value, (float, int)) and math.isclose(
                    float(recorded_value), float(value), rel_tol=0.0, abs_tol=1e-12
                )
            if not matches:
                raise AssertionError(
                    f"{label}: shared summary disagrees with independently recomputed {key}: "
                    f"recorded={recorded_value!r}, recomputed={value!r}"
                )
        paths[label] = derived

    ranked: dict[str, list[dict[str, object]]] = {}
    winners: dict[str, str] = {}
    margins: dict[str, float] = {}
    for percentile in shared.PERCENTILES:
        metric = f"{percentile}_ms"
        ordered = sorted(
            ((float(paths[label][metric]), label) for label in shared.VARIANTS), key=lambda item: item[0]
        )
        if ordered[0][0] <= 0.0:
            raise AssertionError(f"{percentile}: nonpositive winner measurement")
        winners[percentile] = ordered[0][1]
        margins[percentile] = ordered[1][0] / ordered[0][0] - 1.0
        ranked[percentile] = [
            {"variant": label, "latency_ms": latency} for latency, label in ordered
        ]
    return {
        "derivation": "linear-interpolated percentiles independently recomputed from raw_samples_ms",
        "paths": paths,
        "winner_by_percentile": winners,
        "winner_margin_over_runner_up": margins,
        "ranked_paths_by_percentile": ranked,
        "shared_summary_crosscheck_pass": True,
    }


def _repeat_gate(independent: dict[str, object], expected_winner: str) -> dict[str, object]:
    winners = independent["winner_by_percentile"]
    rankings = independent["ranked_paths_by_percentile"]
    if not isinstance(winners, dict) or not isinstance(rankings, dict):
        raise TypeError("independent summary is malformed")
    expected_at_every_percentile = all(
        winners.get(percentile) == expected_winner for percentile in shared.PERCENTILES
    )
    expected_margins: dict[str, float | None] = {}
    margin_pass = True
    for percentile in shared.PERCENTILES:
        ordered = rankings.get(percentile)
        if not isinstance(ordered, list) or len(ordered) != len(shared.VARIANTS):
            raise TypeError(f"{percentile}: malformed independent ranking")
        first, second = ordered[0], ordered[1]
        if first.get("variant") != expected_winner:
            expected_margins[percentile] = None
            margin_pass = False
            continue
        first_ms = float(first["latency_ms"])
        second_ms = float(second["latency_ms"])
        margin = second_ms / first_ms - 1.0
        expected_margins[percentile] = margin
        margin_pass = margin_pass and margin >= MIN_WINNER_MARGIN
    return {
        "expected_winner": expected_winner,
        "winner_by_percentile": winners,
        "expected_winner_at_every_percentile": expected_at_every_percentile,
        "expected_winner_margin_over_runner_up": expected_margins,
        "minimum_required_margin": MIN_WINNER_MARGIN,
        "margin_gate_pass": margin_pass,
        "repeat_gate_pass": expected_at_every_percentile and margin_pass,
    }


def _assess_confirmation(result: dict[str, object]) -> dict[str, object]:
    benchmarks = result["raw_wrapper_public_contract_benchmarks"]
    if not isinstance(benchmarks, dict):
        raise TypeError("benchmark record is malformed")
    all_pass = True
    cells: dict[str, object] = {}
    for case_name, contracts in PROMOTION_CELLS.items():
        for contract, expected_winner in contracts.items():
            cell = benchmarks[case_name][contract]  # type: ignore[index]
            repeats = cell["repeats"]  # type: ignore[index]
            if not isinstance(repeats, list):
                raise TypeError(f"{case_name}/{contract}: repeat record is malformed")
            repeat_gates = [_repeat_gate(repeat["independent_summary"], expected_winner) for repeat in repeats]
            cell_pass = len(repeat_gates) == REPEATS and all(
                bool(gate["repeat_gate_pass"]) for gate in repeat_gates
            )
            cells[f"{case_name}/{contract}"] = {
                "case": case_name,
                "contract": contract,
                "expected_winner": expected_winner,
                "repeats": repeat_gates,
                "cell_gate_pass": cell_pass,
            }
            all_pass = all_pass and cell_pass
    record_evidence = []
    for record in RECORD_ONLY_CELLS:
        cell = benchmarks[record["case"]][record["contract"]]  # type: ignore[index]
        record_evidence.append(
            {
                **record,
                "repeats": [
                    {
                        "repeat_index": repeat["repeat_index"],
                        "winner_by_percentile": repeat["independent_summary"]["winner_by_percentile"],
                        "winner_margin_over_runner_up": repeat["independent_summary"]["winner_margin_over_runner_up"],
                    }
                    for repeat in cell["repeats"]  # type: ignore[index]
                ],
                "decision": "record_only_remain_baseline_regardless_of_observation",
            }
        )
    return {
        "scope_cells": cells,
        "scope_cell_count": len(cells),
        "record_only_cells": record_evidence,
        "confirmation_gate_pass": all_pass,
        "decision": (
            "eligible_for_separate_packed_varlen_dispatch_integration_review"
            if all_pass
            else "STOP_do_not_promote_packed_varlen_custom_paths"
        ),
    }


def _check_args(args: argparse.Namespace) -> None:
    if (
        args.seed != DEFAULT_SEED
        or args.samples != SAMPLES_PER_REPEAT
        or args.repeats != REPEATS
        or args.warmup != WARMUP
    ):
        raise ValueError(
            "this pre-registered runner fixes --seed=20260829, --samples=1000, "
            "--repeats=2, and --warmup=100"
        )


def main() -> None:
    _assert_preregistration_scope()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_REPEAT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--describe", action="store_true", help="write preregistration without importing CUDA")
    args = parser.parse_args()
    _check_args(args)

    result = _initial_result(args)
    if args.describe:
        result["describe_only"] = True
        _write(args.json, result)
        print(f"wrote packed-varlen confirmation preregistration {args.json}")
        return
    if args.reference_root is None:
        raise ValueError("--reference-root is required for a GPU experiment")
    if os.environ.get(CLEAN_GPU_GATE_ENV) != "1":
        raise RuntimeError(
            "refusing a direct GPU run: use run_clean_varlen_dispatch_confirmation_audit.sh "
            f"so {CLEAN_GPU_GATE_ENV}=1 is set only after its PRE clean-GPU check"
        )

    # Imports are intentionally after --describe and both authority gates.
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
    torch_ref, reference_helper_identity = _load_pinned_reference_without_build(
        common, args.reference_root
    )
    result["identity"] = {
        "device": shared._device_identity(),
        "extension": shared._identity(),
        "clean_gpu_shell_gate": {
            "environment_name": CLEAN_GPU_GATE_ENV,
            "value": os.environ[CLEAN_GPU_GATE_ENV],
            "passed": True,
        },
        "pinned_torch_reference_root": str(args.reference_root.resolve()),
        "pinned_torch_reference_helper": reference_helper_identity,
    }
    result["gates"]["clean_gpu_shell_gate"]["passed"] = True  # type: ignore[index]
    result["gates"]["device_gate"]["passed"] = True  # type: ignore[index]
    result["gates"]["audited_extension_sha256_gate"]["passed"] = True  # type: ignore[index]
    _write(args.json, result)

    for case_index, case in enumerate(CASES):
        print(f"raw ABI and pinned-reference exactness {case.name}: contracts={RAW_CONTRACTS}")
        x = shared._make_inputs(case, args.seed + case_index * 1009)
        result["raw_abi_correctness"][case.name] = {}  # type: ignore[index]
        for contract_index, contract in enumerate(RAW_CONTRACTS):
            result["raw_abi_correctness"][case.name][contract] = _raw_abi_exactness(  # type: ignore[index]
                functions,
                torch_ref,
                case,
                x,
                contract,
                args.seed + case_index * 1009 + contract_index * 101,
            )
        del x
        torch.cuda.empty_cache()
        _write(args.json, result)
    result["gates"]["raw_abi_exact_gate"]["passed"] = True  # type: ignore[index]
    result["gates"]["pinned_torch_reference_gate"]["passed"] = True  # type: ignore[index]
    _write(args.json, result)

    for case_index, case in enumerate(CASES):
        result["raw_wrapper_public_contract_benchmarks"][case.name] = {}  # type: ignore[index]
        for contract_index, contract in enumerate(FLA_PUBLIC_CONTRACTS):
            expected_winner = PROMOTION_CELLS.get(case.name, {}).get(contract)
            record_only = expected_winner is None
            print(
                f"raw-wrapper public-contract benchmark {case.name}/{contract}: repeats={REPEATS}, "
                f"samples={SAMPLES_PER_REPEAT}, record_only={record_only}"
            )
            repeats: list[dict[str, object]] = []
            result["raw_wrapper_public_contract_benchmarks"][case.name][contract] = {  # type: ignore[index]
                "promotion_gate_scope": not record_only,
                "expected_winner": expected_winner,
                "record_only": record_only,
                "repeats": repeats,
            }
            _write(args.json, result)
            for repeat_index in range(REPEATS):
                repeat_seed = (
                    args.seed
                    + case_index * 100_003
                    + contract_index * 10_007
                    + repeat_index * 1_009
                )
                x = shared._make_inputs(case, repeat_seed)
                input_snapshot = _snapshot_inputs(x)
                benchmark = shared._benchmark(
                    functions, case, x, contract, repeat_seed + 101, args.warmup, args.samples
                )
                _assert_inputs_unchanged(
                    f"{case.name}/{contract}/repeat_{repeat_index}/benchmark", x, input_snapshot
                )
                repeats.append(
                    {
                        "repeat_index": repeat_index,
                        "input_seed": repeat_seed,
                        "state_seed": repeat_seed + 101,
                        "benchmark": benchmark,
                        "independent_summary": _independent_summary(benchmark),
                        "input_immutability_exact": True,
                    }
                )
                del x
                torch.cuda.empty_cache()
                _write(args.json, result)

    assessment = _assess_confirmation(result)
    result["confirmation_assessment"] = assessment
    result["gates"]["confirmation_gate"]["passed"] = assessment["confirmation_gate_pass"]  # type: ignore[index]
    result["gates"]["confirmation_gate"]["decision"] = assessment["decision"]  # type: ignore[index]
    result["complete"] = True
    _write(args.json, result)
    print(f"wrote {args.json}; {assessment['decision']}")


if __name__ == "__main__":
    main()
