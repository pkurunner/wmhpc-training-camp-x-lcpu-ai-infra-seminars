"""Fixed-config AB/BA interleaved final check for BF16 C=1 no-LSE decode.

This is intentionally not a tuning sweep.  It compares exactly the frozen
``shards=2, stages=4, PDL=off, warps=2, maxnreg=None`` candidate with the
current prepared C=1 control.  Every event still surrounds precisely one
runner call, but each block records control->candidate and then
candidate->control so a fixed launch order cannot select the winner.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

from harness.data import make_decode_problem
from harness.reference import dense_sparse_attention_reference
from .c1_no_lse import C1NoLseConfig, C1NoLseSparseDecode
from .cli import ATOL, BATCHES, RTOL, _environment, _source_hashes, _verify_output
from .prepared_tuned import TuningConfig, TunedPreparedSparseDecode


DEFAULT_CANDIDATE = C1NoLseConfig(
    gqa_shards=2, num_warps=2, num_stages=4, pdl_mode="off", maxnreg=None
)
FROZEN_CONTROL = TuningConfig(
    num_topk_chunks=1, decode_num_warps=4, decode_num_stages=3,
    pdl_mode="auto", decode_maxnreg=None,
)


def _stats_us(values_ms: list[float]) -> dict[str, float]:
    ordered = sorted(value * 1000.0 for value in values_ms)
    count = len(ordered)
    assert count > 0
    return {
        "p10_us": ordered[max(0, int(0.10 * count) - 1)],
        "median_us": float(statistics.median(ordered)),
        "p90_us": ordered[min(count - 1, int(0.90 * count))],
    }


def _one_event(function: Callable[[], torch.Tensor]) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    return start, end


def _abba_events(
    control: Callable[[], torch.Tensor], candidate: Callable[[], torch.Tensor], *, warmup: int, pairs: int
) -> dict[str, object]:
    for _ in range(warmup):
        control()
        candidate()
    torch.cuda.synchronize()
    # Each tuple is (runner label, launch-order label, start, end).  AB denotes
    # the control then candidate half; BA denotes candidate then control.
    events: list[tuple[str, str, torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(pairs):
        start, end = _one_event(control)
        events.append(("current_prepared_control", "AB", start, end))
        start, end = _one_event(candidate)
        events.append(("c1_online_softmax_no_lse", "AB", start, end))
        start, end = _one_event(candidate)
        events.append(("c1_online_softmax_no_lse", "BA", start, end))
        start, end = _one_event(control)
        events.append(("current_prepared_control", "BA", start, end))
    torch.cuda.synchronize()
    collected: dict[str, dict[str, list[float]]] = {
        "current_prepared_control": {"AB": [], "BA": []},
        "c1_online_softmax_no_lse": {"AB": [], "BA": []},
    }
    for runner_name, order, start, end in events:
        collected[runner_name][order].append(float(start.elapsed_time(end)))
    result: dict[str, object] = {
        "protocol": "warmup_each_then_ABBA_pairs_one_runner_call_per_cuda_event_pair_one_stream",
        "warmup_each": warmup,
        "abba_pairs": pairs,
        "samples_per_runner": 2 * pairs,
        "AB_BA_interleaved": True,
    }
    for runner_name, by_order in collected.items():
        merged = [*by_order["AB"], *by_order["BA"]]
        result[runner_name] = {
            "all": _stats_us(merged),
            "when_launch_order_is_AB": _stats_us(by_order["AB"]),
            "when_launch_order_is_BA": _stats_us(by_order["BA"]),
        }
    return result


def _candidate_config(args: argparse.Namespace) -> C1NoLseConfig:
    return C1NoLseConfig(
        gqa_shards=args.shards, num_warps=args.warps, num_stages=args.stages,
        pdl_mode=args.pdl, maxnreg=args.maxnreg,
    )


def _context(args: argparse.Namespace, batch: int, candidate_config: C1NoLseConfig) -> dict[str, object]:
    problem = make_decode_problem(
        batch_size=batch, device="cuda", storage_dtype="bf16", seed=args.seed + batch,
        max_seq_len=args.max_seq_len,
    )
    expected = dense_sparse_attention_reference(problem)
    control_output = torch.empty_like(problem.q)
    candidate_output = torch.empty_like(problem.q)
    control = TunedPreparedSparseDecode(problem, control_output, config=FROZEN_CONTROL)
    candidate = C1NoLseSparseDecode(problem, candidate_output, config=candidate_config)
    row: dict[str, object] = {
        "batch": batch,
        "storage": "bf16",
        "seed": args.seed + batch,
        "problem": {"q_shape": list(problem.q.shape), "kv_dtype": str(problem.kv_cache.dtype)},
        "control": {"config": FROZEN_CONTROL.as_dict(), "metadata": control.metadata.as_dict()},
        "candidate": {"config": candidate_config.as_dict(), "metadata": candidate.metadata},
    }
    try:
        row["control"]["correctness"] = _verify_output(control, control_output, expected)  # type: ignore[index]
        row["candidate"]["correctness"] = _verify_output(candidate, candidate_output, expected)  # type: ignore[index]
        timing = _abba_events(control, candidate, warmup=args.warmup, pairs=args.pairs)
        control_us = float(timing["current_prepared_control"]["all"]["median_us"])  # type: ignore[index]
        candidate_us = float(timing["c1_online_softmax_no_lse"]["all"]["median_us"])  # type: ignore[index]
        speedup = control_us / candidate_us
        row.update({
            "status": "pass", "timing": timing,
            "control_median_us": control_us, "candidate_median_us": candidate_us,
            "speedup_vs_current_prepared_control": speedup,
            "strict_10_percent_target_met": speedup >= 1.10,
        })
    except Exception as exc:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        row.update({"status": "rejected", "error": f"{type(exc).__name__}: {exc}",
                    "strict_10_percent_target_met": False})
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=BATCHES)
    parser.add_argument("--all-batches", action="store_true")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--pairs", type=int, default=101)
    parser.add_argument("--shards", type=int, choices=(1, 2, 4), default=DEFAULT_CANDIDATE.gqa_shards)
    parser.add_argument("--warps", type=int, choices=(1, 2, 4, 8), default=DEFAULT_CANDIDATE.num_warps)
    parser.add_argument("--stages", type=int, choices=(1, 2, 3, 4, 5, 6), default=DEFAULT_CANDIDATE.num_stages)
    parser.add_argument("--pdl", choices=("auto", "on", "off"), default=DEFAULT_CANDIDATE.pdl_mode)
    parser.add_argument("--maxnreg", choices=("none", "64", "96", "128", "160"), default="none")
    parser.add_argument("--require-strict-10", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.batch is None and not args.all_batches:
        parser.error("one of --batch or --all-batches is required")
    if args.max_seq_len < 2048 or args.max_seq_len % 128:
        parser.error("--max-seq-len must be page-aligned and >=2048")
    if args.warmup < 0 or args.pairs < 1:
        parser.error("--warmup must be >=0 and --pairs must be >=1")
    if not torch.cuda.is_available():
        raise RuntimeError("C1 no-LSE AB/BA check requires CUDA")
    args.maxnreg = None if args.maxnreg == "none" else int(args.maxnreg)
    candidate_config = _candidate_config(args)
    batches = BATCHES if args.all_batches else (args.batch,)
    results = [_context(args, batch, candidate_config) for batch in batches]
    strict = all(bool(item.get("strict_10_percent_target_met", False)) for item in results)
    payload: dict[str, object] = {
        "schema": "c2-c1-online-softmax-no-lse-abba-v1",
        "environment": _environment(), "source_sha256": _source_hashes(),
        "frozen_configuration": {
            "control": FROZEN_CONTROL.as_dict(), "candidate": candidate_config.as_dict(),
            "selection_rule": "specified before this AB/BA run; not reselected from its event samples",
            "bf16_c1_only": True,
        },
        "fairness_contract": {
            "same_input_seed_per_batch": True, "caller_owned_output": True,
            "persistent_workspace_outside_timing": True, "selected_chunks": 1, "no_merge": True,
            "oracle": "independent harness.reference dense FP32 selected-page causal attention",
            "tolerance": {"rtol": RTOL, "atol": ATOL}, "single_call_per_cuda_event": True,
            "AB_BA_interleaved": True,
        },
        "results": results, "all_contexts_strict_10_percent": strict,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 3 if args.require_strict_10 and not strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
