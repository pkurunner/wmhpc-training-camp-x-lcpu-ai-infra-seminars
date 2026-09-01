"""Correctness-gated probe of the fixed BF16 static-topk16 C=1 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from harness.data import make_decode_problem
from harness.reference import dense_sparse_attention_reference
from .c1_static_topk16 import StaticTopk16Bf16Decode, StaticTopk16Config
from .cli import ATOL, RTOL, _environment, _one_call_events, _source_hashes, _verify_output
from .prepared_tuned import TuningConfig, TunedPreparedSparseDecode


CONTROL = TuningConfig(
    num_topk_chunks=1, decode_num_warps=4, decode_num_stages=3,
    pdl_mode="auto", decode_maxnreg=None,
)
CANDIDATE = StaticTopk16Config(gqa_shards=2, num_warps=2, num_stages=4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(1, 4, 8, 16), default=4)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=41)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repetitions < 1:
        parser.error("--warmup must be >=0 and --repetitions >=1")
    if not torch.cuda.is_available():
        raise RuntimeError("static-topk16 probe requires CUDA")
    problem = make_decode_problem(batch_size=args.batch, device="cuda", storage_dtype="bf16", seed=args.seed + args.batch)
    expected = dense_sparse_attention_reference(problem)
    control_out = torch.empty_like(problem.q)
    candidate_out = torch.empty_like(problem.q)
    control = TunedPreparedSparseDecode(problem, control_out, config=CONTROL)
    candidate = StaticTopk16Bf16Decode(problem, candidate_out, config=CANDIDATE)
    rows: list[dict[str, object]] = []
    for name, config, runner, output in (
        ("current_prepared_control", CONTROL.as_dict(), control, control_out),
        ("static_topk16_bf16", CANDIDATE.as_dict(), candidate, candidate_out),
    ):
        row: dict[str, object] = {"implementation": name, "config": config, "metadata": runner.metadata if name != "current_prepared_control" else runner.metadata.as_dict()}
        try:
            row["correctness"] = _verify_output(runner, output, expected)
            row["timing"] = _one_call_events(runner, warmup=args.warmup, repetitions=args.repetitions)
            row["status"] = "pass"
        except Exception as exc:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            row.update({"status": "rejected", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    strict = False
    summary: dict[str, object]
    if rows[0]["status"] == "pass" and rows[1]["status"] == "pass":
        control_us = float(rows[0]["timing"]["median_us"])  # type: ignore[index]
        candidate_us = float(rows[1]["timing"]["median_us"])  # type: ignore[index]
        speedup = control_us / candidate_us
        rows[0]["speedup_vs_current_prepared_control"] = 1.0
        rows[1]["speedup_vs_current_prepared_control"] = speedup
        strict = speedup >= 1.10
        summary = {"status": "pass", "control_median_us": control_us, "candidate_median_us": candidate_us,
                   "speedup": speedup, "strict_10_percent_target_met": strict}
    else:
        summary = {"status": "rejected", "strict_10_percent_target_met": False}
    payload = {
        "schema": "c2-static-topk16-probe-v1", "environment": _environment(), "source_sha256": _source_hashes(),
        "fairness_contract": {"storage": "bf16", "selected_chunks": 1, "no_merge": True,
                              "caller_owned_output": True, "persistent_workspace_outside_timing": True,
                              "oracle": "independent harness.reference dense FP32 selected-page causal attention",
                              "tolerance": {"rtol": RTOL, "atol": ATOL}, "single_call_per_cuda_event": True},
        "batch": args.batch, "seed": args.seed + args.batch, "rows": rows, "summary": summary,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if summary["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
