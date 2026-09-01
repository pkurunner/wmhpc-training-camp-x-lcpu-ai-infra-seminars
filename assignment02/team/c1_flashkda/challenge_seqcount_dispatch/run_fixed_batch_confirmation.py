#!/usr/bin/env python3
"""Pre-registered B300 confirmation for five exact fixed-batch FLA shapes.

This runner is deliberately narrower than ``run_seqcount_dispatch.py``.  It
does not infer a general sequence-count rule and it does not modify the
dispatcher.  It only decides whether twelve *already measured* public FLA
shape/contract cells survive two new independent cyclic CUDA-event repeats.

The three B=4/B=8 FP32 cells that lacked a stable prior result are recorded
but deliberately excluded from the promotion gate.  Their future dispatcher
action stays the upstream baseline regardless of this runner's observations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
SAMPLES_PER_REPEAT = 1000
REPEATS = 2
WARMUP = 100
FLA_PUBLIC_CONTRACTS = ("none", "fp32_final_only", "fp32_both")
RAW_CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
MIN_WINNER_MARGIN = 0.02
CLEAN_GPU_GATE_ENV = "C1_FIXED_BATCH_CLEAN_GPU_GATES"


def _case(name: str, batch: int) -> shared.Case:
    return shared.Case(
        name=name,
        form="fixed",
        sequences=batch,
        heads=12,
        lengths=(2048,) * batch,
        family="fixed_batch_confirmation",
    )


CASES = (
    _case("b2_h12_t2048", 2),
    _case("b3_h12_t2048", 3),
    _case("b4_h12_t2048", 4),
    _case("b6_h12_t2048", 6),
    _case("b8_h12_t2048", 8),
)

# The scope is intentionally explicit.  Every included cell needs both
# repeats to select its prescribed winner at P50/P95/P99, and each of those
# six winner measurements must beat its runner-up by at least 2%.
PROMOTION_CELLS = {
    "b2_h12_t2048": {
        "none": "vshard4_p2",
        "fp32_final_only": "vshard4_p2",
        "fp32_both": "vshard4_p2",
    },
    "b3_h12_t2048": {
        "none": "vshard4_p2",
        "fp32_final_only": "vshard4_p2",
        "fp32_both": "vshard4_p2",
    },
    "b4_h12_t2048": {
        "none": "vshard2_p2",
        "fp32_final_only": "vshard2_p2",
    },
    "b6_h12_t2048": {
        "none": "vshard2_p2",
        "fp32_final_only": "vshard2_p2",
        "fp32_both": "vshard2_p2",
    },
    "b8_h12_t2048": {
        "none": "vshard2_p2",
    },
}
RECORD_ONLY_CELLS = (
    {
        "case": "b4_h12_t2048",
        "contract": "fp32_both",
        "future_dispatch_variant": "baseline",
        "promotion_gate_scope": False,
        "reason": (
            "r2 did not establish one winner at all P50/P95/P99 with a 2% margin; "
            "this confirmation records another observation but cannot promote any custom path"
        ),
    },
    {
        "case": "b8_h12_t2048",
        "contract": "fp32_final_only",
        "future_dispatch_variant": "baseline",
        "promotion_gate_scope": False,
        "reason": "r2 runner-up margin was below 2%; preserve fail-closed baseline fallback",
    },
    {
        "case": "b8_h12_t2048",
        "contract": "fp32_both",
        "future_dispatch_variant": "baseline",
        "promotion_gate_scope": False,
        "reason": "r2 runner-up margin was below 2%; preserve fail-closed baseline fallback",
    },
)


def _write(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _preregistration() -> dict[str, object]:
    return {
        "fixed_shape": {"H": 12, "T": 2048, "K": DIM, "V": DIM},
        "raw_abi_contracts": list(RAW_CONTRACTS),
        "public_fla_contracts": list(FLA_PUBLIC_CONTRACTS),
        "repeats": REPEATS,
        "samples_per_repeat": SAMPLES_PER_REPEAT,
        "warmup_per_path": WARMUP,
        "required_percentiles": list(shared.PERCENTILES),
        "minimum_runner_up_margin": MIN_WINNER_MARGIN,
        "promotion_cells": PROMOTION_CELLS,
        "record_only_cells": list(RECORD_ONLY_CELLS),
        "stop_rule": (
            "For any one of the twelve promotion cells, STOP if either repeat has a "
            "different winner at any required percentile, or if that expected winner's "
            "runner-up margin is below 2% at any required percentile."
        ),
    }


def _initial_result(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1,
        "purpose": "pre-registered exact-shape fixed-batch B300 confirmation; no dispatcher mutation",
        "preregistration": _preregistration(),
        "cases": [shared._case_dict(case) for case in CASES],
        "seed": args.seed,
        "raw_abi_correctness": {},
        "raw_wrapper_public_contract_benchmarks": {},
        "identity": {},
        "gates": {
            "clean_gpu_shell_gate": {
                "environment_name": CLEAN_GPU_GATE_ENV,
                "required": True,
                "passed": False,
                "evidence": "the clean submitting shell records PRE/POST nvidia-smi checks in its log",
            },
            "device_gate": {"required": "B300, capability 10.3, 148 SM", "passed": False},
            "audited_extension_sha256_gate": {
                "required": shared.AUDITED_EXTENSION_SHA256,
                "passed": False,
            },
            "raw_abi_exact_gate": {"required": "all candidates exact to baseline", "passed": False},
            "confirmation_gate": {
                "scope_cell_count": 12,
                "passed": False,
                "decision": "not_run",
            },
        },
        "complete": False,
    }


def _raw_abi_exactness(
    functions: dict[str, Callable[..., None]],
    case: shared.Case,
    x: object,
    contract: str,
    seed: int,
) -> dict[str, object]:
    """Check the two custom raw ABI paths bitwise against upstream baseline."""
    initial, final = shared._states(contract, case, seed)
    outputs = {
        label: shared._invoke(fn, x, shared._clone(initial), shared._clone(final))
        for label, fn in functions.items()
    }
    baseline = outputs["baseline"]
    return {
        label: shared._compare(
            f"fixed_batch_confirmation/{case.name}/{contract}/{label}_vs_baseline",
            outputs[label],
            baseline,
        )
        for label in ("vshard2_p2", "vshard4_p2")
    }


def _repeat_gate(benchmark: dict[str, object], expected_winner: str) -> dict[str, object]:
    """Assess one independent repeat against its pre-registered path target."""
    stable = shared._winner_evidence(benchmark)
    winner_by_percentile = stable["winner_by_percentile"]
    if not isinstance(winner_by_percentile, dict):
        raise TypeError("benchmark winner evidence is malformed")
    expected_at_every_percentile = all(
        winner_by_percentile.get(percentile) == expected_winner
        for percentile in shared.PERCENTILES
    )
    return {
        "expected_winner": expected_winner,
        "winner_by_percentile": winner_by_percentile,
        "expected_winner_at_every_percentile": expected_at_every_percentile,
        "runner_up_margin_by_percentile": stable["winner_margin_over_runner_up"],
        "minimum_required_margin": MIN_WINNER_MARGIN,
        "margin_gate_pass": stable["margin_gate_pass"],
        "repeat_gate_pass": expected_at_every_percentile and bool(stable["margin_gate_pass"]),
    }


def _assess_confirmation(result: dict[str, object]) -> dict[str, object]:
    all_pass = True
    cells: dict[str, object] = {}
    benchmarks = result["raw_wrapper_public_contract_benchmarks"]
    for case_name, contracts in PROMOTION_CELLS.items():
        for contract, expected_winner in contracts.items():
            repeats = benchmarks[case_name][contract]["repeats"]  # type: ignore[index]
            repeat_gates = [
                _repeat_gate(repeat["benchmark"], expected_winner)  # type: ignore[index]
                for repeat in repeats
            ]
            cell_pass = len(repeat_gates) == REPEATS and all(
                bool(repeat_gate["repeat_gate_pass"]) for repeat_gate in repeat_gates
            )
            cells[f"{case_name}/{contract}"] = {
                "case": case_name,
                "contract": contract,
                "expected_winner": expected_winner,
                "repeats": repeat_gates,
                "cell_gate_pass": cell_pass,
            }
            all_pass = all_pass and cell_pass

    record_only_evidence = []
    for record_only in RECORD_ONLY_CELLS:
        record = benchmarks[record_only["case"]][record_only["contract"]]  # type: ignore[index]
        record_only_evidence.append(
            {
                **record_only,
                "repeats": [
                    {
                        "repeat_index": repeat["repeat_index"],
                        "winner_evidence": shared._winner_evidence(repeat["benchmark"]),
                    }
                    for repeat in record["repeats"]  # type: ignore[index]
                ],
                "decision": "remain_baseline_regardless_of_observation",
            }
        )
    return {
        "scope_cells": cells,
        "record_only_cells": record_only_evidence,
        "scope_cell_count": len(cells),
        "confirmation_gate_pass": all_pass,
        "decision": (
            "eligible_for_separate_fixed_batch_dispatch_integration_review"
            if all_pass
            else "STOP_do_not_promote_fixed_batch_custom_paths"
        ),
    }


def _check_describe_args(args: argparse.Namespace) -> None:
    if args.samples != SAMPLES_PER_REPEAT or args.repeats != REPEATS or args.warmup != WARMUP:
        raise ValueError(
            "this pre-registered runner fixes --samples=1000, --repeats=2, and --warmup=100; "
            "changing any count would invalidate the stated confirmation gate"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_REPEAT)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--describe", action="store_true", help="write preregistration without importing CUDA")
    args = parser.parse_args()
    _check_describe_args(args)

    result = _initial_result(args)
    if args.describe:
        result["describe_only"] = True
        _write(args.json, result)
        print(f"wrote fixed-batch confirmation preregistration {args.json}")
        return

    if os.environ.get(CLEAN_GPU_GATE_ENV) != "1":
        raise RuntimeError(
            "refusing a direct GPU run: use run_clean_fixed_batch_confirmation_audit.sh "
            f"so {CLEAN_GPU_GATE_ENV}=1 is set only after its PRE clean-GPU check"
        )

    # Import only after --describe and the shell-authority gate.  The helpers
    # below intentionally share the already-audited wrappers/state construction
    # with the r2 sequence-count runner rather than recreating an ABI path.
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
    result["identity"] = {
        "device": shared._device_identity(),
        "extension": shared._identity(),
        "clean_gpu_shell_gate": {
            "environment_name": CLEAN_GPU_GATE_ENV,
            "value": os.environ[CLEAN_GPU_GATE_ENV],
            "passed": True,
        },
    }
    result["gates"]["clean_gpu_shell_gate"]["passed"] = True  # type: ignore[index]
    result["gates"]["device_gate"]["passed"] = True  # type: ignore[index]
    result["gates"]["audited_extension_sha256_gate"]["passed"] = True  # type: ignore[index]
    _write(args.json, result)

    for case_index, case in enumerate(CASES):
        print(f"raw ABI exactness {case.name}: contracts={RAW_CONTRACTS}")
        x = shared._make_inputs(case, args.seed + case_index * 1009)
        result["raw_abi_correctness"][case.name] = {}  # type: ignore[index]
        for contract_index, contract in enumerate(RAW_CONTRACTS):
            result["raw_abi_correctness"][case.name][contract] = _raw_abi_exactness(  # type: ignore[index]
                functions,
                case,
                x,
                contract,
                args.seed + case_index * 1009 + contract_index * 101,
            )
        del x
        torch.cuda.empty_cache()
        _write(args.json, result)
    result["gates"]["raw_abi_exact_gate"]["passed"] = True  # type: ignore[index]
    _write(args.json, result)

    for case_index, case in enumerate(CASES):
        result["raw_wrapper_public_contract_benchmarks"][case.name] = {}  # type: ignore[index]
        for contract_index, contract in enumerate(FLA_PUBLIC_CONTRACTS):
            print(
                f"raw-wrapper public-contract benchmark {case.name}/{contract}: repeats={REPEATS}, "
                f"samples={SAMPLES_PER_REPEAT}"
            )
            repeats: list[dict[str, object]] = []
            expected_winner = PROMOTION_CELLS.get(case.name, {}).get(contract)
            # Create the JSON slot before executing a repeat so a scheduler
            # interruption cannot discard completed 1000-sample evidence.
            result["raw_wrapper_public_contract_benchmarks"][case.name][contract] = {  # type: ignore[index]
                "promotion_gate_scope": expected_winner is not None,
                "expected_winner": expected_winner,
                "repeats": repeats,
            }
            _write(args.json, result)
            for repeat_index in range(REPEATS):
                # New input/state seeds make the two measurements independent;
                # each event still sees exactly one wrapper invocation.
                repeat_seed = (
                    args.seed
                    + case_index * 100_003
                    + contract_index * 10_007
                    + repeat_index * 1_009
                )
                x = shared._make_inputs(case, repeat_seed)
                benchmark = shared._benchmark(
                    functions,
                    case,
                    x,
                    contract,
                    repeat_seed + 101,
                    args.warmup,
                    args.samples,
                )
                repeats.append(
                    {
                        "repeat_index": repeat_index,
                        "input_seed": repeat_seed,
                        "state_seed": repeat_seed + 101,
                        "benchmark": benchmark,
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
