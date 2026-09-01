"""Frozen AB/BA check for prepared BF16 C=1, Triton stage 5 versus stage 3.

This is a confirmation experiment, not a tuning interface.  Both runners use
the same :class:`TunedPreparedSparseDecode` implementation, the same problem,
and caller-owned outputs.  The candidate differs only in
``decode_num_stages=5``; with ``C=1`` the merge kernel is bypassed, so its
configuration remains unchanged and is recorded explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch

from harness.data import make_decode_problem
from harness.reference import dense_sparse_attention_reference
from .cli import ATOL, BATCHES, RTOL, _environment, _verify_output
from .prepared_tuned import TuningConfig, TunedPreparedSparseDecode


WARMUP = 30
ABBA_PAIRS = 101
SAMPLES_PER_RUNNER = 2 * ABBA_PAIRS

# C=1 means merge is bypassed.  Retaining the same merge settings makes the
# sole measured change unambiguous even if that invariant changes in future.
FROZEN_CONTROL = TuningConfig(
    num_topk_chunks=1,
    decode_num_warps=4,
    merge_num_warps=4,
    decode_num_stages=3,
    merge_num_stages=3,
    pdl_mode="auto",
    decode_maxnreg=None,
    merge_maxnreg=None,
)
FROZEN_STAGE5_CANDIDATE = TuningConfig(
    num_topk_chunks=1,
    decode_num_warps=4,
    merge_num_warps=4,
    decode_num_stages=5,
    merge_num_stages=3,
    pdl_mode="auto",
    decode_maxnreg=None,
    merge_maxnreg=None,
)


def _stats_us(values_ms: list[float]) -> dict[str, float]:
    """Return fixed nearest-rank percentiles in microseconds."""
    ordered = sorted(value * 1000.0 for value in values_ms)
    count = len(ordered)
    if not count:
        raise ValueError("cannot summarize an empty timing series")
    return {
        "p10_us": ordered[max(0, math.ceil(0.10 * count) - 1)],
        "median_us": float(statistics.median(ordered)),
        "p90_us": ordered[min(count - 1, math.ceil(0.90 * count) - 1)],
    }


def _one_event(
    runner: Callable[[], torch.Tensor],
) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    """Place exactly one persistent runner call between a CUDA-event pair."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    runner()
    end.record()
    return start, end


def _abba_events(
    control: Callable[[], torch.Tensor], candidate: Callable[[], torch.Tensor]
) -> dict[str, object]:
    """Warm up, then collect control->candidate->candidate->control blocks."""
    for _ in range(WARMUP):
        control()
        candidate()
    torch.cuda.synchronize()

    # (runner label, half-block order, start event, end event)
    events: list[tuple[str, str, torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(ABBA_PAIRS):
        start, end = _one_event(control)
        events.append(("prepared_stage3_control", "AB", start, end))
        start, end = _one_event(candidate)
        events.append(("prepared_stage5_candidate", "AB", start, end))
        start, end = _one_event(candidate)
        events.append(("prepared_stage5_candidate", "BA", start, end))
        start, end = _one_event(control)
        events.append(("prepared_stage3_control", "BA", start, end))
    torch.cuda.synchronize()

    samples: dict[str, dict[str, list[float]]] = {
        "prepared_stage3_control": {"AB": [], "BA": []},
        "prepared_stage5_candidate": {"AB": [], "BA": []},
    }
    for runner_name, order, start, end in events:
        samples[runner_name][order].append(float(start.elapsed_time(end)))

    timing: dict[str, object] = {
        "protocol": "warmup_each_then_101_ABBA_pairs_one_runner_call_per_cuda_event_pair_one_stream",
        "warmup_each": WARMUP,
        "abba_pairs": ABBA_PAIRS,
        "samples_per_runner": SAMPLES_PER_RUNNER,
        "AB_BA_interleaved": True,
        "raw_samples_us": {
            runner_name: {
                order: [value * 1000.0 for value in values_ms]
                for order, values_ms in by_order.items()
            }
            for runner_name, by_order in samples.items()
        },
    }
    for runner_name, by_order in samples.items():
        timing[runner_name] = {
            "all": _stats_us([*by_order["AB"], *by_order["BA"]]),
            "when_launch_order_is_AB": _stats_us(by_order["AB"]),
            "when_launch_order_is_BA": _stats_us(by_order["BA"]),
        }
    return timing


def _source_hashes() -> dict[str, str]:
    """Hash the complete local identity of this frozen comparison."""
    root = Path(__file__).resolve().parents[1]
    items = {
        "challenge_v2/prepared_stage_abba_cli.py": root / "challenge_v2" / "prepared_stage_abba_cli.py",
        "challenge_v2/prepared_tuned.py": root / "challenge_v2" / "prepared_tuned.py",
        "challenge_v2/cli.py": root / "challenge_v2" / "cli.py",
        "challenge_v2/run_prepared_stage5_abba_clean.sh": root / "challenge_v2" / "run_prepared_stage5_abba_clean.sh",
        "challenge/prepared_decode.py": root / "challenge" / "prepared_decode.py",
        "harness/data.py": root / "harness" / "data.py",
        "harness/reference.py": root / "harness" / "reference.py",
        "harness/triton_baseline.py": root / "harness" / "triton_baseline.py",
        "vllm_msa_ref/sparse_attn.py": root / "vllm_msa_ref" / "sparse_attn.py",
    }
    return {
        relative: hashlib.sha256(path.read_bytes()).hexdigest()
        for relative, path in items.items()
    }


def _context(args: argparse.Namespace, batch: int) -> dict[str, object]:
    """Build one same-input B context, gate both runners, then time it."""
    seed = args.seed + batch
    problem = make_decode_problem(
        batch_size=batch,
        device="cuda",
        storage_dtype="bf16",
        seed=seed,
        max_seq_len=args.max_seq_len,
    )
    expected = dense_sparse_attention_reference(problem)
    control_output = torch.empty_like(problem.q)
    candidate_output = torch.empty_like(problem.q)
    control = TunedPreparedSparseDecode(problem, control_output, config=FROZEN_CONTROL)
    candidate = TunedPreparedSparseDecode(
        problem, candidate_output, config=FROZEN_STAGE5_CANDIDATE
    )
    row: dict[str, object] = {
        "batch": batch,
        "storage": "bf16",
        "seed": seed,
        "problem": {
            "q_shape": list(problem.q.shape),
            "kv_dtype": str(problem.kv_cache.dtype),
            "max_seq_len": args.max_seq_len,
        },
        "control": {
            "config": FROZEN_CONTROL.as_dict(),
            "metadata": control.metadata.as_dict(),
        },
        "candidate": {
            "config": FROZEN_STAGE5_CANDIDATE.as_dict(),
            "metadata": candidate.metadata.as_dict(),
        },
    }
    try:
        # These calls compile and synchronize before timing.  Each is compared
        # separately to the independent dense FP32 selected-page oracle.
        row["control"]["correctness"] = _verify_output(control, control_output, expected)  # type: ignore[index]
        row["candidate"]["correctness"] = _verify_output(candidate, candidate_output, expected)  # type: ignore[index]
        timing = _abba_events(control, candidate)
        control_us = float(timing["prepared_stage3_control"]["all"]["median_us"])  # type: ignore[index]
        candidate_us = float(timing["prepared_stage5_candidate"]["all"]["median_us"])  # type: ignore[index]
        speedup = control_us / candidate_us
        row.update(
            {
                "status": "pass",
                "timing": timing,
                "control_median_us": control_us,
                "candidate_median_us": candidate_us,
                "speedup_stage5_vs_stage3": speedup,
                "strict_10_percent_target_met": speedup >= 1.10,
            }
        )
    except Exception as exc:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        row.update(
            {
                "status": "rejected",
                "error": f"{type(exc).__name__}: {exc}",
                "strict_10_percent_target_met": False,
            }
        )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=BATCHES)
    parser.add_argument("--all-batches", action="store_true")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--require-strict-10", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.batch is None and not args.all_batches:
        parser.error("one of --batch or --all-batches is required")
    if args.batch is not None and args.all_batches:
        parser.error("--batch and --all-batches are mutually exclusive")
    if args.max_seq_len < 2048 or args.max_seq_len % 128:
        parser.error("--max-seq-len must be page-aligned and >=2048")
    if not torch.cuda.is_available():
        raise RuntimeError("prepared stage-5 AB/BA check requires CUDA")

    batches = BATCHES if args.all_batches else (args.batch,)
    results = [_context(args, batch) for batch in batches]
    strict = all(bool(row.get("strict_10_percent_target_met", False)) for row in results)
    payload: dict[str, object] = {
        "schema": "c2-prepared-stage5-abba-v1",
        "environment": _environment(),
        "source_sha256": _source_hashes(),
        "frozen_configuration": {
            "control": FROZEN_CONTROL.as_dict(),
            "candidate": FROZEN_STAGE5_CANDIDATE.as_dict(),
            "changed_field": "decode_num_stages",
            "selection_rule": "specified before this AB/BA run; not reselected from its event samples",
            "bf16_c1_only": True,
        },
        "fairness_contract": {
            "same_problem_instance_per_batch": True,
            "same_input_seed_per_batch": True,
            "caller_owned_output": True,
            "persistent_workspace_outside_timing": True,
            "selected_chunks": 1,
            "merge_bypassed": True,
            "oracle": "independent harness.reference dense FP32 selected-page causal attention",
            "tolerance": {"rtol": RTOL, "atol": ATOL},
            "single_call_per_cuda_event": True,
            "AB_BA_interleaved": True,
            "raw_event_samples_recorded": True,
        },
        "results": results,
        "all_contexts_strict_10_percent": strict,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 3 if args.require_strict_10 and not strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
