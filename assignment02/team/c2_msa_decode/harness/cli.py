"""Correctness, benchmark, and profiler CLI for the Team C2 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Iterable

import torch

from .data import DecodeDType, make_decode_problem
from .reference import dense_sparse_attention_reference
from .triton_baseline import run_triton_baseline


OFFICIAL_BATCHES = (1, 4, 8, 16)
TOLERANCES: dict[str, tuple[float, float]] = {
    "bf16": (3e-2, 3e-2),
    # Upstream FP8-scale coverage uses 2e-2 against a dequantized BF16 cache.
    # We allow one additional BF16 reduction/merge rounding margin here.
    "fp8-scalar": (3e-2, 3e-2),
    "fp8-token": (3e-2, 3e-2),
}


def _device(args: argparse.Namespace) -> str:
    return args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")


def _batches(args: argparse.Namespace) -> Iterable[int]:
    return OFFICIAL_BATCHES if args.all_batches else (args.batch,)


def _make(args: argparse.Namespace, batch: int, mode: DecodeDType):
    return make_decode_problem(
        batch_size=batch,
        device=_device(args),
        storage_dtype=mode,
        seed=args.seed + batch,
        decode_query_len=args.decode_query_len,
        max_seq_len=args.max_seq_len,
    )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _correctness(args: argparse.Namespace) -> int:
    if _device(args) != "cuda":
        raise RuntimeError("correctness needs CUDA because the baseline is Triton")
    records = []
    for batch in _batches(args):
        for mode in args.storage_modes:
            problem = _make(args, batch, mode)
            reference = dense_sparse_attention_reference(problem)
            actual = run_triton_baseline(problem)
            _sync()
            rtol, atol = TOLERANCES[mode]
            torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
            diff = (actual.float() - reference.float()).abs()
            records.append(
                {
                    "batch": batch,
                    "storage": mode,
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                    "rtol": rtol,
                    "atol": atol,
                    "status": "pass",
                }
            )
    print(json.dumps({"correctness": records}, ensure_ascii=False, indent=2))
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    if _device(args) != "cuda":
        raise RuntimeError("benchmark needs CUDA")
    results = []
    for batch in _batches(args):
        problem = _make(args, batch, args.storage_mode)
        for _ in range(args.warmup):
            run_triton_baseline(problem)
        _sync()
        elapsed = []
        for _ in range(args.repetitions):
            start = time.perf_counter_ns()
            run_triton_baseline(problem)
            _sync()
            elapsed.append((time.perf_counter_ns() - start) / 1e6)
        elapsed.sort()
        results.append(
            {
                "batch": batch,
                "storage": args.storage_mode,
                "warmup": args.warmup,
                "repetitions": args.repetitions,
                "median_ms": elapsed[len(elapsed) // 2],
                "p10_ms": elapsed[max(0, int(len(elapsed) * 0.10) - 1)],
                "p90_ms": elapsed[min(len(elapsed) - 1, int(len(elapsed) * 0.90))],
            }
        )
    print(json.dumps({"benchmark": results}, ensure_ascii=False, indent=2))
    return 0


def _profile(args: argparse.Namespace) -> int:
    if _device(args) != "cuda":
        raise RuntimeError("profile needs CUDA")
    problem = _make(args, args.batch, args.storage_mode)
    for _ in range(args.warmup):
        run_triton_baseline(problem)
    _sync()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
        for _ in range(args.profile_steps):
            run_triton_baseline(problem)
        _sync()
    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=args.row_limit))
    if args.trace is not None:
        trace = Path(args.trace)
        trace.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace))
        print(json.dumps({"trace": str(trace.resolve())}, ensure_ascii=False))
    return 0


def _once(args: argparse.Namespace) -> int:
    """Launch one baseline decode for Nsight Systems/Compute wrappers."""
    if _device(args) != "cuda":
        raise RuntimeError("once needs CUDA")
    problem = _make(args, args.batch, args.storage_mode)
    actual = run_triton_baseline(problem)
    _sync()
    print(
        json.dumps(
            {
                "once": "complete",
                "batch": args.batch,
                "storage": args.storage_mode,
                "shape": list(actual.shape),
                "finite": bool(torch.isfinite(actual.float()).all().item()),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("correctness", "benchmark", "profile", "once", "dry-run")
    )
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto (default)")
    parser.add_argument("--batch", type=int, choices=OFFICIAL_BATCHES, default=1)
    parser.add_argument("--all-batches", action="store_true", help="run B=1,4,8,16")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--decode-query-len", type=int, default=1)
    parser.add_argument("--storage-mode", choices=("bf16", "fp8-scalar", "fp8-token"), default="bf16")
    parser.add_argument(
        "--storage-modes",
        choices=("bf16", "fp8-scalar", "fp8-token"),
        nargs="+",
        default=("bf16", "fp8-scalar", "fp8-token"),
        help="correctness modes (default: all)",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--trace", default=None, help="optional Chrome trace JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "dry-run":
        print(json.dumps({"status": "parser-ok", "official_batches": OFFICIAL_BATCHES}))
        return 0
    if args.command == "correctness":
        return _correctness(args)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "profile":
        return _profile(args)
    return _once(args)


if __name__ == "__main__":
    raise SystemExit(main())
