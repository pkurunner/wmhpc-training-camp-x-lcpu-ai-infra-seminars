#!/usr/bin/env python3
"""Validate and benchmark the state contracts used by the FLA inference path.

This prerequisite runner compares the upstream public wrapper, vshard2-P2,
and vshard4-P2 from one already-built extension.  It intentionally benchmarks
the exact ``None/None`` and ``None/FP32-final`` contracts emitted by the real
FLA backend before either contract can be added to the automatic whitelist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2  # noqa: E402
from assignment02.team.c1_flashkda.challenge_vshard4_prefetch2 import (  # noqa: E402
    vshard4_prefetch2,
)
from assignment02.team.c1_flashkda.harness import validate_and_bench as common  # noqa: E402


VARIANTS = ("baseline", "vshard2_p2", "vshard4_p2")
CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")


def _csv(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or any(item not in allowed for item in items):
        raise ValueError(f"invalid comma-separated value: {value!r}")
    return items


def _identity() -> dict[str, object]:
    import flash_kda_C

    required = ("fwd", "fwd_vshard_p2", "fwd_vshard4_p2", "get_workspace_size")
    missing = [symbol for symbol in required if not hasattr(flash_kda_C, symbol)]
    if missing:
        raise RuntimeError(f"loaded extension lacks required symbols: {missing}")
    path = Path(flash_kda_C.__file__).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "required_symbols": list(required),
    }


def _states(
    contract: str, heads: int, seed: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if contract == "none":
        return None, None
    if contract == "fp32_final_only":
        return None, torch.zeros(1, heads, 128, 128, dtype=torch.float32, device="cuda")
    dtype = torch.bfloat16 if contract == "bf16_both" else torch.float32
    generator = torch.Generator(device="cuda").manual_seed(seed)
    initial = torch.randn(
        1, heads, 128, 128, dtype=dtype, device="cuda", generator=generator
    ).contiguous()
    return initial, torch.zeros_like(initial)


def _clone(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.clone()


def _invoke(
    fn: Callable[..., None],
    x: common.Inputs,
    initial: torch.Tensor | None,
    final: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    out = torch.zeros_like(x.v)
    fn(
        x.q,
        x.k,
        x.v,
        x.g,
        x.beta,
        x.scale,
        out,
        A_log=x.a_log,
        dt_bias=x.dt_bias,
        lower_bound=x.lower_bound,
        initial_state=initial,
        final_state=final,
    )
    torch.cuda.synchronize()
    return out, final


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
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


def _validate_contract(
    functions: dict[str, Callable[..., None]],
    torch_ref: Callable[..., None],
    x: common.Inputs,
    contract: str,
    seed: int,
) -> dict[str, object]:
    initial, final = _states(contract, x.q.shape[2], seed)
    outputs = {
        label: _invoke(fn, x, _clone(initial), _clone(final))
        for label, fn in functions.items()
    }
    reference = _invoke(torch_ref, x, _clone(initial), _clone(final))
    baseline_out, baseline_state = outputs["baseline"]
    result: dict[str, object] = {}
    for label in ("vshard2_p2", "vshard4_p2"):
        out, state = outputs[label]
        common.require_exact(f"{contract}/baseline_vs_{label}/output", out, baseline_out)
        common.require_exact(f"{contract}/{label}_vs_torch_ref/output", out, reference[0])
        item: dict[str, object] = {
            "output_exact_baseline": True,
            "output_exact_torch_ref": True,
            "output_max_abs_baseline": _max_abs(out, baseline_out),
            "output_max_abs_torch_ref": _max_abs(out, reference[0]),
        }
        if state is not None:
            if baseline_state is None or reference[1] is None:
                raise AssertionError(f"{contract}: missing state comparison target")
            common.require_exact(
                f"{contract}/baseline_vs_{label}/final_state", state, baseline_state
            )
            common.require_exact(
                f"{contract}/{label}_vs_torch_ref/final_state", state, reference[1]
            )
            item.update(
                {
                    "final_state_exact_baseline": True,
                    "final_state_exact_torch_ref": True,
                    "final_state_max_abs_baseline": _max_abs(state, baseline_state),
                    "final_state_max_abs_torch_ref": _max_abs(state, reference[1]),
                }
            )
        result[label] = item
    return result


def _benchmark_contract(
    functions: dict[str, Callable[..., None]],
    x: common.Inputs,
    contract: str,
    seed: int,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    calls: dict[str, Callable[[], None]] = {}
    for label, fn in functions.items():
        initial, final = _states(contract, x.q.shape[2], seed)
        out = torch.empty_like(x.v)

        def call(
            fn: Callable[..., None] = fn,
            initial: torch.Tensor | None = initial,
            final: torch.Tensor | None = final,
            out: torch.Tensor = out,
        ) -> None:
            fn(
                x.q,
                x.k,
                x.v,
                x.g,
                x.beta,
                x.scale,
                out,
                A_log=x.a_log,
                dt_bias=x.dt_bias,
                lower_bound=x.lower_bound,
                initial_state=initial,
                final_state=final,
            )

        calls[label] = call
    labels = tuple(calls)
    for index in range(warmup):
        order = labels[index % len(labels):] + labels[:index % len(labels)]
        for label in order:
            calls[label]()
    torch.cuda.synchronize()
    raw = {label: [] for label in labels}
    for index in range(samples):
        order = labels[index % len(labels):] + labels[:index % len(labels)]
        for label in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            calls[label]()
            end.record()
            torch.cuda.synchronize()
            raw[label].append(float(start.elapsed_time(end)))
    summaries = {label: _summary(values) for label, values in raw.items()}
    v2 = float(summaries["vshard2_p2"]["p50_ms"])
    v4 = float(summaries["vshard4_p2"]["p50_ms"])
    return {
        "paths": summaries,
        "raw_samples_ms": raw,
        "vshard2_p2_over_vshard4_p2_p50_x": v2 / v4,
        "vshard4_p2_faster_than_vshard2_p2": v4 < v2,
        "event_contract": "three-path cyclic rotation; one public-wrapper call per event; workspace allocation included",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--H", type=int, default=12)
    parser.add_argument("--contracts", default="none,bf16_both,fp32_both,fp32_final_only")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.T <= 0 or args.H <= 0 or args.T % 16:
        raise ValueError("T/H must be positive and fixed T must be divisible by 16")
    if args.warmup < 0 or args.samples <= 0:
        raise ValueError("warmup must be nonnegative and samples positive")
    contracts = _csv(args.contracts, CONTRACTS)
    import flash_kda

    functions: dict[str, Callable[..., None]] = {
        "baseline": flash_kda.fwd,
        "vshard2_p2": prefetch2.fwd,
        "vshard4_p2": vshard4_prefetch2.fwd,
    }
    torch_ref = common._load_torch_ref(args.reference_root)
    identity = _identity()
    x = common.make_inputs(args.T, args.H, args.seed)
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "shape": {"B": 1, "T": args.T, "H": args.H, "K": 128, "V": 128},
        "contracts": list(contracts),
        "extension": identity,
        "correctness": {},
        "exact_gate_pass": False,
    }
    for index, contract in enumerate(contracts):
        result["correctness"][contract] = _validate_contract(  # type: ignore[index]
            functions, torch_ref, x, contract, args.seed + index * 101
        )
    result["exact_gate_pass"] = True
    if not args.no_bench:
        result["benchmark"] = {
            contract: _benchmark_contract(
                functions,
                x,
                contract,
                args.seed + index * 101,
                args.warmup,
                args.samples,
            )
            for index, contract in enumerate(contracts)
        }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
