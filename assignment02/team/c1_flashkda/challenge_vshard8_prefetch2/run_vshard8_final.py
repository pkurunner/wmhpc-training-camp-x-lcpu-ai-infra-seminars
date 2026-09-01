#!/usr/bin/env python3
"""Exact gate and same-SO cyclic benchmark for the V=16 vshard8-P2 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2  # noqa: E402
from assignment02.team.c1_flashkda.challenge_vshard4_prefetch2 import (  # noqa: E402
    vshard4_prefetch2,
)
from assignment02.team.c1_flashkda.challenge_vshard8_prefetch2 import (  # noqa: E402
    vshard8_prefetch2,
)
from assignment02.team.c1_flashkda.harness import validate_and_bench as common  # noqa: E402


CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
BASE_VARIANTS = ("baseline", "vshard2_p2", "vshard4_p2")
CANDIDATES = ("vshard8_p1", "vshard8_p2")


def _csv_strings(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or any(item not in allowed for item in items):
        raise ValueError(f"invalid comma-separated value: {value!r}")
    return items


def _csv_ints(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items or any(item <= 0 for item in items):
        raise ValueError(f"invalid positive integer list: {value!r}")
    return items


def _identity(candidate: str) -> dict[str, object]:
    import flash_kda_C

    candidate_symbol = {
        "vshard8_p1": "fwd_vshard8",
        "vshard8_p2": "fwd_vshard8_p2",
    }[candidate]
    required = (
        "fwd",
        "fwd_vshard_p2",
        "fwd_vshard4_p2",
        candidate_symbol,
        "get_workspace_size",
    )
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
        return None, torch.zeros(
            1, heads, 128, 128, dtype=torch.float32, device="cuda"
        )
    dtype = torch.bfloat16 if contract == "bf16_both" else torch.float32
    generator = torch.Generator(device="cuda").manual_seed(seed)
    initial = torch.randn(
        1,
        heads,
        128,
        128,
        dtype=dtype,
        device="cuda",
        generator=generator,
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


def _require_pair_exact(
    label: str,
    actual: tuple[torch.Tensor, torch.Tensor | None],
    expected: tuple[torch.Tensor, torch.Tensor | None],
) -> dict[str, object]:
    common.require_exact(label + "/output", actual[0], expected[0])
    result: dict[str, object] = {
        "output_exact": True,
        "output_max_abs": _max_abs(actual[0], expected[0]),
    }
    if actual[1] is None or expected[1] is None:
        if actual[1] is not None or expected[1] is not None:
            raise AssertionError(label + ": final-state presence mismatch")
        result["final_state_present"] = False
    else:
        common.require_exact(label + "/final_state", actual[1], expected[1])
        result.update(
            {
                "final_state_present": True,
                "final_state_exact": True,
                "final_state_max_abs": _max_abs(actual[1], expected[1]),
            }
        )
    return result


def _validate(
    functions: dict[str, Callable[..., None]],
    variants: tuple[str, ...],
    torch_ref: Callable[..., None] | None,
    x: common.Inputs,
    contract: str,
    seed: int,
) -> dict[str, object]:
    initial, final = _states(contract, x.q.shape[2], seed)
    outputs = {
        label: _invoke(fn, x, _clone(initial), _clone(final))
        for label, fn in functions.items()
    }
    baseline = outputs["baseline"]
    result: dict[str, object] = {"against_baseline": {}}
    for label in variants[1:]:
        result["against_baseline"][label] = _require_pair_exact(  # type: ignore[index]
            f"{contract}/{label}_vs_baseline", outputs[label], baseline
        )
    if torch_ref is not None:
        reference = _invoke(torch_ref, x, _clone(initial), _clone(final))
        result["against_torch_ref"] = {
            label: _require_pair_exact(
                f"{contract}/{label}_vs_torch_ref", outputs[label], reference
            )
            for label in variants
        }
    return result


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


def _benchmark(
    functions: dict[str, Callable[..., None]],
    candidate: str,
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
        rotation = index % len(labels)
        for label in labels[rotation:] + labels[:rotation]:
            calls[label]()
    torch.cuda.synchronize()
    raw = {label: [] for label in labels}
    for index in range(samples):
        rotation = index % len(labels)
        for label in labels[rotation:] + labels[:rotation]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            calls[label]()
            end.record()
            torch.cuda.synchronize()
            raw[label].append(float(start.elapsed_time(end)))
    summaries = {label: _summary(values) for label, values in raw.items()}
    v4 = summaries["vshard4_p2"]
    v8 = summaries[candidate]
    quantile_wins = {
        quantile: float(v8[quantile + "_ms"]) < float(v4[quantile + "_ms"])
        for quantile in ("p50", "p95", "p99")
    }
    return {
        "paths": summaries,
        "raw_samples_ms": raw,
        "candidate": candidate,
        "vshard4_over_candidate_p50_x": float(v4["p50_ms"]) / float(v8["p50_ms"]),
        "candidate_quantile_wins_vs_vshard4": quantile_wins,
        "candidate_wins_p50_p95_p99": all(quantile_wins.values()),
        "event_contract": (
            "four-path cyclic rotation; one public-wrapper call per event; "
            "workspace allocation included"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, default="vshard8_p2")
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--heads", default="12")
    parser.add_argument(
        "--contracts", default="none,bf16_both,fp32_both,fp32_final_only"
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--torch-ref", action="store_true")
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.T <= 0 or args.T % 16:
        raise ValueError("fixed T must be positive and divisible by 16")
    if args.warmup < 0 or args.samples <= 0:
        raise ValueError("warmup must be nonnegative and samples positive")
    heads = _csv_ints(args.heads)
    contracts = _csv_strings(args.contracts, CONTRACTS)

    import flash_kda

    if args.candidate == "vshard8_p1":
        from assignment02.team.c1_flashkda.challenge_vshard8 import vshard8

        candidate_fn = vshard8.fwd
    else:
        candidate_fn = vshard8_prefetch2.fwd
    variants = BASE_VARIANTS + (args.candidate,)

    functions: dict[str, Callable[..., None]] = {
        "baseline": flash_kda.fwd,
        "vshard2_p2": prefetch2.fwd,
        "vshard4_p2": vshard4_prefetch2.fwd,
        args.candidate: candidate_fn,
    }
    reference = (
        common._load_torch_ref(args.reference_root) if args.torch_ref else None
    )
    result: dict[str, object] = {
        "schema": "c1_vshard8/v2",
        "candidate": args.candidate,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "configuration": {"B": 1, "T": args.T, "heads": list(heads), "K": 128, "V": 128},
        "contracts": list(contracts),
        "extension": _identity(args.candidate),
        "correctness": {},
        "exact_gate_pass": False,
        "torch_ref_enabled": bool(reference),
    }
    for head_index, head_count in enumerate(heads):
        x = common.make_inputs(args.T, head_count, args.seed + head_index * 1009)
        head_result: dict[str, object] = {"contracts": {}}
        for contract_index, contract in enumerate(contracts):
            contract_seed = args.seed + head_index * 1009 + contract_index * 101
            head_result["contracts"][contract] = _validate(  # type: ignore[index]
                functions, variants, reference, x, contract, contract_seed
            )
        if not args.no_bench:
            head_result["benchmark"] = {
                contract: _benchmark(
                    functions,
                    args.candidate,
                    x,
                    contract,
                    args.seed + head_index * 1009 + contract_index * 101,
                    args.warmup,
                    args.samples,
                )
                for contract_index, contract in enumerate(contracts)
            }
        result["correctness"][str(head_count)] = head_result  # type: ignore[index]
    result["exact_gate_pass"] = True
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
