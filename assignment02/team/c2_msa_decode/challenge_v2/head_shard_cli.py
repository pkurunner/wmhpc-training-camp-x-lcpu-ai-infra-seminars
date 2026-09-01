"""Correctness-gated GQA head-shard sweep for the C=1 prepared decode path."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from harness.data import make_decode_problem
from harness.reference import dense_sparse_attention_reference
from .cli import BATCHES, MODES, RTOL, ATOL, _environment, _one_call_events, _source_hashes, _verify_output
from .head_shard import HeadShardConfig, HeadShardedSparseDecode
from .prepared_tuned import TuningConfig, TunedPreparedSparseDecode


def _csv_ints(value: str, *, allow_none: bool = False) -> tuple[int | None, ...]:
    result: list[int | None] = []
    for raw in value.split(","):
        text = raw.strip().lower()
        result.append(None if allow_none and text == "none" else int(text))
    return tuple(result)


def _csv_pdl(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or any(item not in ("auto", "on", "off") for item in result):
        raise argparse.ArgumentTypeError("PDL choices are auto,on,off")
    return result


def _csv_warps(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item not in (1, 2, 4, 8) for item in result):
        raise argparse.ArgumentTypeError("warps must be one of 1,2,4,8")
    return result


def _run(args: argparse.Namespace, *, batch: int) -> dict[str, object]:
    problem = make_decode_problem(
        batch_size=batch, device="cuda", storage_dtype=args.storage_mode,
        seed=args.seed + batch, max_seq_len=args.max_seq_len,
    )
    expected = dense_sparse_attention_reference(problem)
    base_config = TuningConfig(
        num_topk_chunks=1, decode_num_warps=args.baseline_warps,
        decode_num_stages=args.baseline_stages, pdl_mode=args.baseline_pdl,
        decode_maxnreg=args.baseline_maxnreg,
    )
    candidates: list[tuple[str, dict[str, object], Any, torch.Tensor]] = []
    output = torch.empty_like(problem.q)
    candidates.append(("current_prepared_control", base_config.as_dict(), TunedPreparedSparseDecode(problem, output, config=base_config), output))
    for shards, stages, pdl, maxnreg, warps in itertools.product(
        args.shards, args.stages, args.pdl_modes, args.maxnregs, args.warps
    ):
        config = HeadShardConfig(
            gqa_shards=shards, num_stages=stages, pdl_mode=pdl, maxnreg=maxnreg, num_warps=warps
        )
        output = torch.empty_like(problem.q)
        candidates.append(("gqa_head_shard", config.as_dict(), HeadShardedSparseDecode(problem, output, config=config), output))
    rows: list[dict[str, object]] = []
    for kind, config, runner, output in candidates:
        row: dict[str, object] = {"implementation": kind, "config": config}
        try:
            correctness = _verify_output(runner, output, expected)
            timing = _one_call_events(runner, warmup=args.warmup, repetitions=args.repetitions)
            metadata = runner.metadata.as_dict() if hasattr(runner.metadata, "as_dict") else runner.metadata
            row.update({"status": "pass", "correctness": correctness, "timing": timing, "metadata": metadata})
        except Exception as exc:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            row.update({"status": "rejected", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    control = rows[0]
    passing = [item for item in rows if item["status"] == "pass"]
    summary: dict[str, object]
    if control["status"] != "pass":
        summary = {"status": "control_rejected"}
    else:
        control_us = float(control["timing"]["median_us"])  # type: ignore[index]
        for item in passing:
            item["speedup_vs_current_prepared_control"] = control_us / float(item["timing"]["median_us"])  # type: ignore[index]
        shard_passing = [item for item in passing if item["implementation"] == "gqa_head_shard"]
        if shard_passing:
            winner = min(shard_passing, key=lambda item: float(item["timing"]["median_us"]))  # type: ignore[index]
            speedup = float(winner["speedup_vs_current_prepared_control"])
            summary = {
                "status": "pass", "control_median_us": control_us,
                "head_shard_winner_median_us": float(winner["timing"]["median_us"]),  # type: ignore[index]
                "head_shard_winner_speedup": speedup,
                "strict_10_percent_target_met": speedup >= 1.10,
                "winner_config": winner["config"],
            }
        else:
            summary = {"status": "all_head_shards_rejected", "control_median_us": control_us}
    return {"batch": batch, "storage": args.storage_mode, "seed": args.seed + batch,
            "problem": {"q_shape": list(problem.q.shape), "kv_dtype": str(problem.kv_cache.dtype)},
            "candidates": rows, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=BATCHES, default=4)
    parser.add_argument("--all-batches", action="store_true")
    parser.add_argument("--storage-mode", choices=MODES, default="bf16")
    parser.add_argument("--shards", type=lambda v: _csv_ints(v), default=(2, 4))
    parser.add_argument("--stages", type=lambda v: _csv_ints(v), default=(2, 3, 4))
    parser.add_argument("--pdl-modes", type=_csv_pdl, default=("off",))
    parser.add_argument("--maxnregs", type=lambda v: _csv_ints(v, allow_none=True), default=(None,))
    parser.add_argument("--warps", type=_csv_warps, default=(4,))
    parser.add_argument("--baseline-stages", type=int, default=3)
    parser.add_argument("--baseline-pdl", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--baseline-maxnreg", type=lambda v: _csv_ints(v, allow_none=True)[0], default=None)
    parser.add_argument("--baseline-warps", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=41)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("head-shard sweep requires CUDA")
    if args.max_seq_len % 128 or args.max_seq_len < 2048:
        raise ValueError("max sequence length must be a page-aligned value >=2048")
    batches = BATCHES if args.all_batches else (args.batch,)
    results = [_run(args, batch=batch) for batch in batches]
    payload = {
        "schema": "c2-gqa-head-shard-c1-v1", "environment": _environment(), "source_sha256": _source_hashes(),
        "fairness_contract": {
            "same_input_seed_per_batch": True, "caller_owned_output": True,
            "persistent_workspace_outside_timing": True, "chunks": 1, "merge_bypassed": True,
            "oracle": "independent harness.reference dense FP32 selected-page causal attention",
            "tolerance": {"rtol": RTOL, "atol": ATOL}, "single_call_per_cuda_event": True,
        },
        "results": results,
        "all_contexts_head_shard_strict_10_percent": all(bool(item["summary"].get("strict_10_percent_target_met", False)) for item in results),
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
