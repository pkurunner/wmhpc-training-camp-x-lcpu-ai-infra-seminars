"""Correctness-gated, same-contract Triton configuration sweep for Team C2.

The benchmark deliberately measures exactly one decode invocation per CUDA
event pair.  Input seed, caller-owned output, and persistent workspace are
held fixed while a candidate changes only Triton ``num_stages``, PDL, or
``maxnreg``.  Compilation and warmup occur before event recording.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import statistics
from typing import Any, Callable, Iterable

import torch
import triton

from harness.data import make_decode_problem
from harness.reference import dense_sparse_attention_reference
from .prepared_tuned import TuningConfig, TunedPreparedSparseDecode


BATCHES = (1, 4, 8, 16)
MODES = ("bf16", "fp8-scalar", "fp8-token")
RTOL = ATOL = 3e-2


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _comma_ints(value: str, *, allow_none: bool = False) -> tuple[int | None, ...]:
    result: list[int | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        if allow_none and item == "none":
            result.append(None)
        else:
            result.append(int(item))
    if not result:
        raise argparse.ArgumentTypeError("list must not be empty")
    return tuple(result)


def _warps(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item not in (1, 2, 4, 8) for item in result):
        raise argparse.ArgumentTypeError("warp list must contain only 1,2,4,8")
    return result


def _chunks(value: str, storage: str, batch: int) -> int | None:
    if value == "auto":
        return None
    if value == "selected":
        return {
            "bf16": {1: 1, 4: 1, 8: 1, 16: 1},
            "fp8-scalar": {1: 16, 4: 4, 8: 8, 16: 4},
            "fp8-token": {1: 4, 4: 16, 8: 16, 16: 4},
        }[storage][batch]
    return int(value)


def _config_key(config: TuningConfig) -> tuple[object, ...]:
    return tuple(config.as_dict().items())


def _unique(configs: Iterable[TuningConfig]) -> list[TuningConfig]:
    seen: set[tuple[object, ...]] = set()
    result: list[TuningConfig] = []
    for config in configs:
        key = _config_key(config)
        if key not in seen:
            seen.add(key)
            result.append(config)
    return result


def _candidate_configs(args: argparse.Namespace, *, chunks: int | None) -> list[TuningConfig]:
    """Return baseline plus either an economical one-factor or full grid scan."""
    base = TuningConfig(num_topk_chunks=chunks)
    if args.sweep == "compact":
        # The current implementation is (stage=3, pdl=auto, no maxnreg).
        # Vary one independent compiler/runtime control at a time first; this
        # keeps the B300 reservation usable for evidence rather than JIT churn.
        configs: list[TuningConfig] = [base]
        configs.extend(
            TuningConfig(num_topk_chunks=chunks, decode_num_stages=stage, merge_num_stages=stage)
            for stage in args.stages
        )
        configs.extend(TuningConfig(num_topk_chunks=chunks, pdl_mode=mode) for mode in args.pdl_modes)
        configs.extend(
            TuningConfig(num_topk_chunks=chunks, decode_maxnreg=value, merge_maxnreg=value)
            for value in args.maxnregs
        )
        configs.extend(
            TuningConfig(num_topk_chunks=chunks, decode_num_warps=decode_warps, merge_num_warps=merge_warps)
            for decode_warps, merge_warps in itertools.product(args.decode_warps, args.merge_warps)
        )
        return _unique(configs)
    return _unique(
        TuningConfig(
            num_topk_chunks=chunks,
            decode_num_warps=decode_warps,
            merge_num_warps=merge_warps,
            decode_num_stages=stage,
            merge_num_stages=stage,
            pdl_mode=mode,
            decode_maxnreg=maxnreg,
            merge_maxnreg=maxnreg,
        )
        for stage, mode, maxnreg, decode_warps, merge_warps in itertools.product(
            args.stages, args.pdl_modes, args.maxnregs, args.decode_warps, args.merge_warps
        )
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    items = {
        "challenge_v2/cli.py": root / "challenge_v2" / "cli.py",
        "challenge_v2/prepared_tuned.py": root / "challenge_v2" / "prepared_tuned.py",
        "challenge/prepared_decode.py": root / "challenge" / "prepared_decode.py",
        "harness/triton_baseline.py": root / "harness" / "triton_baseline.py",
        "harness/data.py": root / "harness" / "data.py",
        "harness/reference.py": root / "harness" / "reference.py",
        "vllm_msa_ref/sparse_attn.py": root / "vllm_msa_ref" / "sparse_attn.py",
    }
    # Candidate source is part of the measurement identity even though each
    # candidate has its own CLI. Keep optional entries conditional so this
    # shared helper remains usable during incremental development.
    for relative in (
        "challenge_v2/head_shard.py",
        "challenge_v2/head_shard_cli.py",
        "challenge_v2/c1_no_lse.py",
        "challenge_v2/c1_no_lse_cli.py",
        "challenge_v2/c1_no_lse_abba_cli.py",
        "challenge_v2/c1_static_topk16.py",
        "challenge_v2/c1_static_topk16_probe_cli.py",
    ):
        path = root / relative
        if path.is_file():
            items[relative] = path
    return {name: _sha(path) for name, path in items.items()}


def _environment() -> dict[str, object]:
    return {
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
    }


def _verify_output(
    runner: TunedPreparedSparseDecode, output: torch.Tensor, expected: torch.Tensor
) -> dict[str, float | bool]:
    result = runner()
    if result.data_ptr() != output.data_ptr():
        raise RuntimeError("candidate did not return the caller-owned output")
    torch.cuda.synchronize()
    actual = output.float()
    target = expected.float()
    diff = (actual - target).abs()
    finite = bool(torch.isfinite(actual).all().item())
    passed = finite and bool(torch.isclose(actual, target, rtol=RTOL, atol=ATOL).all().item())
    if not passed:
        raise AssertionError(
            f"independent FP32 gate failed: finite={finite}, max_abs={float(diff.max())}"
        )
    return {"finite": finite, "max_abs": float(diff.max()), "mean_abs": float(diff.mean())}


def _one_call_events(
    function: Callable[[], torch.Tensor], *, warmup: int, repetitions: int
) -> dict[str, object]:
    """Each sample has exactly one call between its own CUDA event pair."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repetitions)]
    for start, end in events:
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    values = sorted(float(start.elapsed_time(end)) for start, end in events)
    p10 = values[max(0, int(0.10 * repetitions) - 1)]
    p90 = values[min(repetitions - 1, int(0.90 * repetitions))]
    return {
        "protocol": "warmup_then_per_call_cuda_event_pair_one_stream_no_intercall_sync",
        "warmup": warmup,
        "repetitions": repetitions,
        "p10_us": p10 * 1000.0,
        "median_us": statistics.median(values) * 1000.0,
        "p90_us": p90 * 1000.0,
    }


def _run_context(args: argparse.Namespace, *, batch: int, storage: str) -> dict[str, object]:
    # Constructed only once: every candidate observes the exact same tensor
    # addresses and values. The independent reference is timing-excluded.
    problem = make_decode_problem(
        batch_size=batch, device="cuda", storage_dtype=storage,
        seed=args.seed + batch, max_seq_len=args.max_seq_len,
    )
    expected = dense_sparse_attention_reference(problem)
    chunks = _chunks(args.chunks, storage, batch)
    candidates: list[dict[str, object]] = []
    for config in _candidate_configs(args, chunks=chunks):
        output = torch.empty_like(problem.q)
        row: dict[str, object] = {"config": config.as_dict()}
        try:
            runner = TunedPreparedSparseDecode(problem, output, config=config)
            correctness = _verify_output(runner, output, expected)
            timing = _one_call_events(runner, warmup=args.warmup, repetitions=args.repetitions)
            row.update(
                {
                    "status": "pass",
                    "correctness": correctness,
                    "timing": timing,
                    "contract": {
                        "caller_owned_output": True,
                        "persistent_workspace_created_before_timing": True,
                        "same_problem_seed": args.seed + batch,
                        "single_call_per_event": True,
                    },
                    "metadata": runner.metadata.as_dict(),
                }
            )
        except Exception as exc:  # Retain rejected JIT configurations as evidence.
            # A rejected asynchronous launch can make this synchronization
            # raise the same CUDA error.  Preserve the original candidate
            # failure rather than aborting the entire evidence sweep.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            row.update({"status": "rejected", "error": f"{type(exc).__name__}: {exc}"})
        candidates.append(row)

    passing = [row for row in candidates if row["status"] == "pass"]
    baseline = next(
        (row for row in passing if row["config"] == TuningConfig(num_topk_chunks=chunks).as_dict()),
        None,
    )
    if baseline is None:
        summary: dict[str, object] = {"status": "baseline_rejected"}
    else:
        base_us = float(baseline["timing"]["median_us"])  # type: ignore[index]
        for row in passing:
            candidate_us = float(row["timing"]["median_us"])  # type: ignore[index]
            row["speedup_vs_same_contract_baseline"] = base_us / candidate_us
        winner = min(passing, key=lambda row: float(row["timing"]["median_us"]))  # type: ignore[index]
        speedup = float(winner["speedup_vs_same_contract_baseline"])
        summary = {
            "status": "pass",
            "baseline_median_us": base_us,
            "winner_median_us": float(winner["timing"]["median_us"]),  # type: ignore[index]
            "winner_speedup": speedup,
            "strict_10_percent_target_met": speedup >= 1.10,
            "winner_config": winner["config"],
        }
    return {
        "batch": batch,
        "storage": storage,
        "seed": args.seed + batch,
        "resolved_chunks": chunks if chunks is not None else "baseline_auto",
        "problem": {"q_shape": list(problem.q.shape), "kv_dtype": str(problem.kv_cache.dtype)},
        "candidates": candidates,
        "summary": summary,
    }


def static_check(_: argparse.Namespace) -> int:
    from triton.runtime.autotuner import Config

    try:
        from triton.backends.nvidia.compiler import CUDAOptions
        cuda_options = getattr(CUDAOptions, "__dataclass_fields__", {})
        cuda_option_has_maxnreg = "maxnreg" in cuda_options
        cuda_option_has_pdl = "launch_pdl" in cuda_options
    except ImportError:
        cuda_option_has_maxnreg = cuda_option_has_pdl = False
    payload = {
        "status": "pass",
        "triton": str(triton.__version__),
        "Config_signature": str(inspect.signature(Config)),
        "Config_supports_maxnreg": "maxnreg" in inspect.signature(Config).parameters,
        "nvidia_CUDAOptions_supports_maxnreg": cuda_option_has_maxnreg,
        "nvidia_CUDAOptions_supports_launch_pdl": cuda_option_has_pdl,
        "note": "This verifies API availability only; sweep performs the actual compile/run gate on the target GPU.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def sweep(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("sweep requires CUDA")
    modes = MODES if args.all_modes else (args.storage_mode,)
    batches = BATCHES if args.all_batches else (args.batch,)
    results = [_run_context(args, batch=batch, storage=mode) for mode in modes for batch in batches]
    all_pass = all(item["summary"].get("status") == "pass" for item in results)
    strict = all(bool(item["summary"].get("strict_10_percent_target_met", False)) for item in results)
    payload = {
        "schema": "c2-tuned-prepared-sweep-v1",
        "environment": _environment(),
        "source_sha256": _source_hashes(),
        "fairness_contract": {
            "same_seed_and_problem_within_context": True,
            "caller_owned_output": True,
            "persistent_workspace_outside_events": True,
            "oracle": "harness.reference.dense_sparse_attention_reference FP32 selected-page causal attention",
            "tolerance": {"rtol": RTOL, "atol": ATOL},
            "event_protocol": "one kernel-call path invocation per CUDA event pair",
            "compile_and_warmup_excluded": True,
        },
        "sweep": {
            "style": args.sweep,
            "chunks": args.chunks,
            "stages": list(args.stages),
            "pdl_modes": list(args.pdl_modes),
            "maxnregs": list(args.maxnregs),
            "decode_warps": list(args.decode_warps),
            "merge_warps": list(args.merge_warps),
            "seed": args.seed,
            "max_seq_len": args.max_seq_len,
        },
        "results": results,
        "all_context_baselines_pass": all_pass,
        "strict_10_percent_target_met_in_every_requested_context": strict,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if all_pass else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("static-check", "sweep"))
    result.add_argument("--batch", type=int, choices=BATCHES, default=1)
    result.add_argument("--all-batches", action="store_true")
    result.add_argument("--storage-mode", choices=MODES, default="bf16")
    result.add_argument("--all-modes", action="store_true")
    result.add_argument("--chunks", choices=("selected", "auto", "1", "2", "4", "8", "16"), default="selected")
    result.add_argument("--sweep", choices=("compact", "grid"), default="compact")
    result.add_argument("--stages", type=lambda v: _comma_ints(v), default=(1, 2, 3, 4, 5))
    result.add_argument("--pdl-modes", type=lambda v: tuple(v.strip() for v in v.split(",") if v.strip()), default=("auto", "on", "off"))
    result.add_argument("--maxnregs", type=lambda v: _comma_ints(v, allow_none=True), default=(None, 64, 96, 128, 160))
    result.add_argument("--decode-warps", type=_warps, default=(4,))
    result.add_argument("--merge-warps", type=_warps, default=(4,))
    result.add_argument("--seed", type=_nonnegative, default=20260819)
    result.add_argument("--max-seq-len", type=_positive, default=4096)
    result.add_argument("--warmup", type=_nonnegative, default=20)
    result.add_argument("--repetitions", type=_positive, default=41)
    result.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "static-check":
        return static_check(args)
    if any(mode not in ("auto", "on", "off") for mode in args.pdl_modes):
        raise ValueError("--pdl-modes must contain only auto,on,off")
    if args.max_seq_len % 128:
        raise ValueError("--max-seq-len must be page aligned (multiple of 128)")
    return sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
