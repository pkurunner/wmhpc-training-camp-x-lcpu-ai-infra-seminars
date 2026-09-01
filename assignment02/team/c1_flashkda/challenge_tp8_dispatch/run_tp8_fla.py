#!/usr/bin/env python3
"""Exercise the opt-in dispatcher through FLA on one eight-GPU TP shard.

The KDA operation itself is head-local, so this harness launches one NCCL
rank per GPU with H=12 (global H=96 at TP=8).  It validates the pinned FLA
backend, the custom backend, and FLA's public ``chunk_kda`` call bit-for-bit,
then reports the per-sample maximum latency across all eight ranks.  A small
non-whitelisted call proves that the same public registration fails closed to
the upstream FlashKDA implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_tp8_dispatch import (  # noqa: E402
    auto_dispatch,
    fla_backend,
)
from assignment02.team.c1_flashkda.harness import validate_and_bench as common  # noqa: E402


CONTRACTS = ("none", "fp32_final_only", "fp32_both")
PATHS = ("pinned_flash_kda", "c1_auto_backend", "public_chunk_kda")


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _extension_identity() -> dict[str, object]:
    import flash_kda_C

    required = ("fwd", "fwd_vshard_p2", "fwd_vshard4_p2", "get_workspace_size")
    missing = [symbol for symbol in required if not callable(getattr(flash_kda_C, symbol, None))]
    if missing:
        raise RuntimeError(f"loaded extension lacks required symbols: {missing}")
    path = Path(flash_kda_C.__file__).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "required_symbols": list(required),
    }


def _initial_state(contract: str, heads: int, seed: int) -> torch.Tensor | None:
    if contract != "fp32_both":
        return None
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        1,
        heads,
        128,
        128,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ).contiguous()


def _call(
    fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    x: common.Inputs,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return fn(
        x.q,
        x.k,
        x.v,
        x.g,
        x.beta,
        scale=x.scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        safe_gate=True,
        lower_bound=x.lower_bound,
        A_log=x.a_log,
        dt_bias=x.dt_bias,
    )


def _assert_exact(
    label: str,
    actual: tuple[torch.Tensor, torch.Tensor | None],
    expected: tuple[torch.Tensor, torch.Tensor | None],
) -> dict[str, object]:
    common.require_exact(f"{label}/output", actual[0], expected[0])
    item: dict[str, object] = {
        "output_exact": True,
        "output_max_abs": common.max_abs(actual[0], expected[0]),
    }
    if actual[1] is None or expected[1] is None:
        if actual[1] is not None or expected[1] is not None:
            raise AssertionError(f"{label}: final-state presence mismatch")
        item["final_state_present"] = False
    else:
        common.require_exact(f"{label}/final_state", actual[1], expected[1])
        item.update(
            {
                "final_state_present": True,
                "final_state_exact": True,
                "final_state_max_abs": common.max_abs(actual[1], expected[1]),
                "final_state_dtype": str(actual[1].dtype),
            }
        )
    return item


def _validate_contract(
    functions: dict[str, Callable[..., tuple[torch.Tensor, torch.Tensor | None]]],
    x: common.Inputs,
    contract: str,
    heads: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    initial = _initial_state(contract, heads, seed)
    output_final = contract != "none"
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    with torch.inference_mode():
        for label in PATHS:
            state = None if initial is None else initial.clone()
            outputs[label] = _call(functions[label], x, state, output_final)
        torch.cuda.synchronize()
    baseline = outputs["pinned_flash_kda"]
    checks = {
        label: _assert_exact(f"{contract}/{label}_vs_pinned", outputs[label], baseline)
        for label in ("c1_auto_backend", "public_chunk_kda")
    }
    decision = auto_dispatch.get_last_decision()
    if decision.get("chosen_variant") != "vshard4_p2":
        raise AssertionError(f"{contract}: public call did not select vshard4_p2: {decision}")
    return checks, decision


def _benchmark_contract(
    functions: dict[str, Callable[..., tuple[torch.Tensor, torch.Tensor | None]]],
    x: common.Inputs,
    contract: str,
    heads: int,
    seed: int,
    warmup: int,
    samples: int,
) -> dict[str, list[float]]:
    initial = _initial_state(contract, heads, seed)
    output_final = contract != "none"

    def make_call(
        fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    ) -> Callable[[], None]:
        def call() -> None:
            _call(fn, x, initial, output_final)

        return call

    calls = {label: make_call(functions[label]) for label in PATHS}
    with torch.inference_mode():
        for index in range(warmup):
            offset = index % len(PATHS)
            order = PATHS[offset:] + PATHS[:offset]
            for label in order:
                calls[label]()
        torch.cuda.synchronize()
        dist.barrier()
        raw = {label: [] for label in PATHS}
        for index in range(samples):
            dist.barrier()  # Outside the CUDA events; aligns TP sample indices.
            offset = index % len(PATHS)
            order = PATHS[offset:] + PATHS[:offset]
            for label in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                calls[label]()
                end.record()
                end.synchronize()
                raw[label].append(float(start.elapsed_time(end)))
        dist.barrier()
    return raw


def _fallback_smoke(
    baseline_fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    public_fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    heads: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    x = common.make_inputs(257, heads, seed)
    with torch.inference_mode():
        baseline = _call(baseline_fn, x, None, False)
        public = _call(public_fn, x, None, False)
        torch.cuda.synchronize()
    check = _assert_exact("fallback_t257/public_vs_pinned", public, baseline)
    decision = auto_dispatch.get_last_decision()
    if decision.get("chosen_variant") != "baseline":
        raise AssertionError(f"T=257 did not fail closed: {decision}")
    if decision.get("reason") != "state_contract_none_h12_length_not_whitelisted":
        raise AssertionError(f"unexpected T=257 fallback reason: {decision}")
    return check, decision


def _critical_path(rank_reports: list[dict[str, Any]], samples: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for contract in CONTRACTS:
        paths: dict[str, object] = {}
        critical_raw: dict[str, list[float]] = {}
        for label in PATHS:
            per_rank = [report["raw_samples_ms"][contract][label] for report in rank_reports]
            if any(len(values) != samples for values in per_rank):
                raise AssertionError(f"{contract}/{label}: rank sample-count mismatch")
            values = [max(rank_values[index] for rank_values in per_rank) for index in range(samples)]
            critical_raw[label] = values
            paths[label] = _summary(values)
        baseline_p50 = float(paths["pinned_flash_kda"]["p50_ms"])  # type: ignore[index]
        auto_p50 = float(paths["c1_auto_backend"]["p50_ms"])  # type: ignore[index]
        public_p50 = float(paths["public_chunk_kda"]["p50_ms"])  # type: ignore[index]
        result[contract] = {
            "paths": paths,
            "raw_samples_ms": critical_raw,
            "pinned_over_auto_p50_x": baseline_p50 / auto_p50,
            "pinned_over_public_p50_x": baseline_p50 / public_p50,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--H", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--target-tp-degree", type=int, default=8)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.T != 8192 or args.H != 12:
        raise ValueError("this audited TP8 harness is pinned to local T=8192,H=12")
    if args.warmup < 0 or args.samples <= 0:
        raise ValueError("warmup must be nonnegative and samples positive")
    if args.expected_world_size <= 0 or args.target_tp_degree < args.expected_world_size:
        raise ValueError("target TP degree must be at least the positive observed world size")
    if os.environ.get("C1_B300_FLASH_KDA") != "1":
        raise RuntimeError("set C1_B300_FLASH_KDA=1 for the opt-in integration test")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != args.expected_world_size:
            raise RuntimeError(
                f"audit expected {args.expected_world_size} ranks, got {world_size}"
            )

        from fla.ops.kda import chunk_kda
        from fla.ops.kda.backends import kda_registry
        from fla.ops.kda.backends.flash_kda import FlashKDABackend

        custom_backend = fla_backend.register_backend()
        if fla_backend.register_backend() is not custom_backend:
            raise AssertionError("custom backend registration is not idempotent")
        backend_order = [
            {
                "backend_type": backend.backend_type,
                "priority": backend.priority,
                "available": backend.is_available(),
                "enabled": backend.is_enabled(),
            }
            for backend in kda_registry._get_sorted_backends()
        ]
        types = [item["backend_type"] for item in backend_order]
        if types.index("c1_b300_flash_kda") > types.index("flash_kda"):
            raise AssertionError(f"custom backend is ordered after pinned backend: {backend_order}")

        baseline_backend = FlashKDABackend()
        functions = {
            "pinned_flash_kda": baseline_backend.chunk_kda,
            "c1_auto_backend": custom_backend.chunk_kda,
            "public_chunk_kda": chunk_kda,
        }
        x = common.make_inputs(args.T, args.H, args.seed + rank * 1009)
        correctness: dict[str, object] = {}
        decisions: dict[str, object] = {}
        raw_samples: dict[str, dict[str, list[float]]] = {}
        for index, contract in enumerate(CONTRACTS):
            checks, decision = _validate_contract(
                functions, x, contract, args.H, args.seed + rank * 1009 + index * 101
            )
            correctness[contract] = checks
            decisions[contract] = decision
            raw_samples[contract] = _benchmark_contract(
                functions,
                x,
                contract,
                args.H,
                args.seed + rank * 1009 + index * 101,
                args.warmup,
                args.samples,
            )

        fallback_check, fallback_decision = _fallback_smoke(
            baseline_backend.chunk_kda,
            chunk_kda,
            args.H,
            args.seed + rank * 1009 + 909,
        )
        props = torch.cuda.get_device_properties(local_rank)
        rank_report: dict[str, Any] = {
            "rank": rank,
            "local_rank": local_rank,
            "current_device": torch.cuda.current_device(),
            "device": props.name,
            "capability": [props.major, props.minor],
            "multiprocessor_count": props.multi_processor_count,
            "backend_order": backend_order,
            "extension": _extension_identity(),
            "correctness": correctness,
            "whitelist_decisions": decisions,
            "fallback_t257": {"correctness": fallback_check, "decision": fallback_decision},
            "raw_samples_ms": raw_samples,
        }
        gathered: list[dict[str, Any] | None] | None = [None] * world_size if rank == 0 else None
        dist.gather_object(rank_report, gathered, dst=0)
        if rank == 0:
            assert gathered is not None and all(item is not None for item in gathered)
            reports = [item for item in gathered if item is not None]
            if sorted(item["local_rank"] for item in reports) != list(range(world_size)):
                raise AssertionError("local ranks do not cover the allocated GPUs exactly once")
            extension_hashes = {item["extension"]["sha256"] for item in reports}
            if len(extension_hashes) != 1:
                raise AssertionError(f"ranks loaded different extensions: {extension_hashes}")
            result = {
                "exact_gate_pass": True,
                "tp8_concurrent_gate_pass": (
                    world_size == args.target_tp_degree == 8
                ),
                "integration": {
                    "framework": "torch.distributed NCCL + FLA public chunk_kda",
                    "observed_concurrent_ranks": world_size,
                    "target_tp_degree": args.target_tp_degree,
                    "target_global_heads": args.target_tp_degree * args.H,
                    "local_heads": args.H,
                    "shape_per_rank": {"B": 1, "T": args.T, "H": args.H, "K": 128, "V": 128},
                    "coverage": (
                        "full_concurrent_tp8"
                        if world_size == args.target_tp_degree == 8
                        else "local_tp_shard_only_due_to_scheduler_quota"
                    ),
                    "semantics": "KDA is head-local; one H=12 shard per rank. Rank-max latency is a full TP critical path only when all target ranks run concurrently.",
                },
                "contracts": list(CONTRACTS),
                "timing_contract": "three-path cyclic order; NCCL barrier before each sample and outside CUDA events; per-sample max across observed ranks",
                "warmup": args.warmup,
                "samples": args.samples,
                "critical_path": _critical_path(reports, args.samples),
                "ranks": reports,
            }
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"wrote {args.json}")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
