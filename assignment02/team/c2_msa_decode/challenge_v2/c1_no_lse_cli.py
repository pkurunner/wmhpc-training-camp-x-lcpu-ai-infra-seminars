"""FP32-gated sweep of the C=1 no-LSE online-softmax decode candidate."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from harness.data import make_decode_problem
from harness.reference import dense_sparse_attention_reference
from .c1_no_lse import C1NoLseConfig, C1NoLseSparseDecode
from .cli import BATCHES, MODES, RTOL, ATOL, _environment, _one_call_events, _source_hashes, _verify_output
from .prepared_tuned import TuningConfig, TunedPreparedSparseDecode


def _csv_ints(value: str, *, none: bool = False) -> tuple[int | None, ...]:
    return tuple(None if none and item.strip().lower() == "none" else int(item) for item in value.split(",") if item.strip())


def _pdls(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or any(item not in ("auto", "on", "off") for item in result):
        raise argparse.ArgumentTypeError("PDL choices must be auto,on,off")
    return result


def _warps(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item not in (1, 2, 4, 8) for item in result):
        raise argparse.ArgumentTypeError("warps must be 1,2,4,8")
    return result


def _context(args: argparse.Namespace, batch: int) -> dict[str, object]:
    problem = make_decode_problem(batch_size=batch, device="cuda", storage_dtype=args.storage_mode,
                                  seed=args.seed + batch, max_seq_len=args.max_seq_len)
    expected = dense_sparse_attention_reference(problem)
    base_config = TuningConfig(num_topk_chunks=1, decode_num_warps=args.baseline_warps,
                               decode_num_stages=args.baseline_stages, pdl_mode=args.baseline_pdl,
                               decode_maxnreg=args.baseline_maxnreg)
    candidates: list[tuple[str, dict[str, object], Any, torch.Tensor]] = []
    control_output = torch.empty_like(problem.q)
    candidates.append(("current_prepared_control", base_config.as_dict(),
                       TunedPreparedSparseDecode(problem, control_output, config=base_config), control_output))
    for shards, stages, pdl, maxnreg, warps in itertools.product(
        args.shards, args.stages, args.pdl_modes, args.maxnregs, args.warps
    ):
        config = C1NoLseConfig(gqa_shards=shards, num_stages=stages, pdl_mode=pdl,
                                maxnreg=maxnreg, num_warps=warps)
        output = torch.empty_like(problem.q)
        candidates.append(("c1_online_softmax_no_lse", config.as_dict(),
                           C1NoLseSparseDecode(problem, output, config=config), output))
    rows: list[dict[str, object]] = []
    for kind, config, runner, output in candidates:
        row: dict[str, object] = {"implementation": kind, "config": config}
        try:
            correct = _verify_output(runner, output, expected)
            timing = _one_call_events(runner, warmup=args.warmup, repetitions=args.repetitions)
            row.update({"status": "pass", "correctness": correct, "timing": timing,
                        "metadata": runner.metadata.as_dict() if hasattr(runner.metadata, "as_dict") else runner.metadata})
        except Exception as exc:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            row.update({"status": "rejected", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    control = rows[0]
    if control["status"] != "pass":
        summary: dict[str, object] = {"status": "control_rejected"}
    else:
        base_us = float(control["timing"]["median_us"])  # type: ignore[index]
        passing = [row for row in rows if row["status"] == "pass"]
        for row in passing:
            row["speedup_vs_current_prepared_control"] = base_us / float(row["timing"]["median_us"])  # type: ignore[index]
        no_lse = [row for row in passing if row["implementation"] == "c1_online_softmax_no_lse"]
        if no_lse:
            winner = min(no_lse, key=lambda row: float(row["timing"]["median_us"]))  # type: ignore[index]
            speed = float(winner["speedup_vs_current_prepared_control"])
            summary = {"status": "pass", "control_median_us": base_us,
                       "no_lse_winner_median_us": float(winner["timing"]["median_us"]),
                       "no_lse_winner_speedup": speed, "strict_10_percent_target_met": speed >= 1.10,
                       "winner_config": winner["config"]}
        else:
            summary = {"status": "all_no_lse_rejected", "control_median_us": base_us}
    return {"batch": batch, "storage": args.storage_mode, "seed": args.seed + batch,
            "problem": {"q_shape": list(problem.q.shape), "kv_dtype": str(problem.kv_cache.dtype)},
            "candidates": rows, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=BATCHES, default=4)
    parser.add_argument("--all-batches", action="store_true")
    parser.add_argument("--storage-mode", choices=MODES, default="bf16")
    parser.add_argument("--shards", type=lambda value: _csv_ints(value), default=(1, 2, 4))
    parser.add_argument("--stages", type=lambda value: _csv_ints(value), default=(2, 3, 4))
    parser.add_argument("--pdl-modes", type=_pdls, default=("off",))
    parser.add_argument("--maxnregs", type=lambda value: _csv_ints(value, none=True), default=(None,))
    parser.add_argument("--warps", type=_warps, default=(4,))
    parser.add_argument("--baseline-stages", type=int, default=3)
    parser.add_argument("--baseline-pdl", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--baseline-maxnreg", type=lambda value: _csv_ints(value, none=True)[0], default=None)
    parser.add_argument("--baseline-warps", choices=(1, 2, 4, 8), type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=41)
    parser.add_argument(
        "--require-strict-10", action="store_true",
        help=("write the complete JSON as usual, then return nonzero unless every requested "
              "same-contract context has a PASS control and a no-LSE winner >=1.10x"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("C1 no-LSE sweep requires CUDA")
    if args.max_seq_len < 2048 or args.max_seq_len % 128:
        raise ValueError("--max-seq-len must be page-aligned and >=2048")
    results = [_context(args, batch) for batch in (BATCHES if args.all_batches else (args.batch,))]
    payload = {
        "schema": "c2-c1-online-softmax-no-lse-v1", "environment": _environment(), "source_sha256": _source_hashes(),
        "fairness_contract": {
            "same_input_seed_per_batch": True, "caller_owned_output": True,
            "persistent_workspace_outside_timing": True, "selected_chunks": 1,
            "no_merge": True, "no_lse_workspace_or_store": True,
            "oracle": "independent harness.reference dense FP32 selected-page causal attention",
            "tolerance": {"rtol": RTOL, "atol": ATOL}, "single_call_per_cuda_event": True,
        }, "results": results,
        "all_contexts_strict_10_percent": all(bool(row["summary"].get("strict_10_percent_target_met", False)) for row in results),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    # Exploratory sweeps remain RC=0 by default so their complete rejection
    # evidence can be inspected.  A final audit opts in to a hard gate: an
    # absent/rejected control or any <10% requested context must not look like
    # a successful shell job merely because JSON serialization completed.
    return 3 if args.require_strict_10 and not payload["all_contexts_strict_10_percent"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
