#!/usr/bin/env python3
"""Build a conservative B300 length/head/state performance map."""

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
from assignment02.team.c1_flashkda.harness import validate_and_bench as common  # noqa: E402


VARIANTS = ("baseline", "vshard2_p2", "vshard4_p2")
CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
TP8_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
INTERACTION_HEADS = (1, 12, 37, 38, 64, 96)
INTERACTION_LENGTHS = (2048, 32768)


def _states(
    contract: str, heads: int, seed: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if contract == "none":
        return None, None
    dtype = torch.float32 if contract in ("fp32_both", "fp32_final_only") else torch.bfloat16
    final = torch.zeros(1, heads, 128, 128, dtype=dtype, device="cuda")
    if contract == "fp32_final_only":
        return None, final
    generator = torch.Generator(device="cuda").manual_seed(seed)
    initial = torch.randn(
        final.shape, dtype=dtype, device="cuda", generator=generator
    ).contiguous()
    return initial, final


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


def _compare(
    label: str,
    actual: tuple[torch.Tensor, torch.Tensor | None],
    baseline: tuple[torch.Tensor, torch.Tensor | None],
) -> dict[str, object]:
    common.require_exact(f"{label}/output", actual[0], baseline[0])
    result: dict[str, object] = {
        "output_exact": True,
        "output_max_abs": common.max_abs(actual[0], baseline[0]),
    }
    if actual[1] is None or baseline[1] is None:
        if actual[1] is not None or baseline[1] is not None:
            raise AssertionError(f"{label}: state presence mismatch")
        result["final_state_present"] = False
    else:
        common.require_exact(f"{label}/final_state", actual[1], baseline[1])
        result.update(
            {
                "final_state_present": True,
                "final_state_exact": True,
                "final_state_max_abs": common.max_abs(actual[1], baseline[1]),
            }
        )
    return result


def _validate(
    functions: dict[str, Callable[..., None]],
    x: common.Inputs,
    contract: str,
    seed: int,
) -> dict[str, object]:
    heads = x.q.shape[2]
    initial, final = _states(contract, heads, seed)
    outputs = {
        label: _invoke(fn, x, _clone(initial), _clone(final))
        for label, fn in functions.items()
    }
    return {
        label: _compare(f"{contract}/{label}_vs_baseline", outputs[label], outputs["baseline"])
        for label in ("vshard2_p2", "vshard4_p2")
    }


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


def _benchmark(
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
    for index in range(warmup):
        offset = index % len(VARIANTS)
        for label in VARIANTS[offset:] + VARIANTS[:offset]:
            calls[label]()
    torch.cuda.synchronize()
    raw = {label: [] for label in VARIANTS}
    for index in range(samples):
        offset = index % len(VARIANTS)
        for label in VARIANTS[offset:] + VARIANTS[:offset]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            calls[label]()
            end.record()
            end.synchronize()
            raw[label].append(float(start.elapsed_time(end)))
    paths = {label: _summary(values) for label, values in raw.items()}
    p50 = {label: float(paths[label]["p50_ms"]) for label in VARIANTS}
    return {
        "paths": paths,
        "raw_samples_ms": raw,
        "winner_p50": min(VARIANTS, key=p50.get),
        "baseline_over_vshard2_p2_p50_x": p50["baseline"] / p50["vshard2_p2"],
        "baseline_over_vshard4_p2_p50_x": p50["baseline"] / p50["vshard4_p2"],
        "vshard2_p2_over_vshard4_p2_p50_x": p50["vshard2_p2"] / p50["vshard4_p2"],
        "event_contract": "three-path cyclic rotation; one wrapper call per event; workspace allocation included",
    }


def _identity() -> dict[str, object]:
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


def _run_case(
    functions: dict[str, Callable[..., None]],
    tokens: int,
    heads: int,
    contracts: tuple[str, ...],
    seed: int,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    x = common.make_inputs(tokens, heads, seed)
    result: dict[str, object] = {
        "shape": {"B": 1, "T": tokens, "H": heads, "K": 128, "V": 128},
        "contracts": {},
    }
    for index, contract in enumerate(contracts):
        contract_seed = seed + index * 101
        result["contracts"][contract] = {  # type: ignore[index]
            "correctness": _validate(functions, x, contract, contract_seed),
            "benchmark": _benchmark(
                functions, x, contract, contract_seed, warmup, samples
            ),
        }
    return result


def _write(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 0 or args.samples <= 0:
        raise ValueError("warmup must be nonnegative and samples positive")
    import flash_kda

    functions: dict[str, Callable[..., None]] = {
        "baseline": flash_kda.fwd,
        "vshard2_p2": prefetch2.fwd,
        "vshard4_p2": vshard4_prefetch2.fwd,
    }
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "multiprocessor_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "extension": _identity(),
        "warmup": args.warmup,
        "samples": args.samples,
        "tp8_h12_length_axis": [],
        "length_head_interactions_bf16": [],
        "exact_gate_pass": False,
        "complete": False,
    }
    _write(args.json, result)
    for index, tokens in enumerate(TP8_LENGTHS):
        print(f"TP8-local length sweep T={tokens}, H=12, contracts={CONTRACTS}")
        result["tp8_h12_length_axis"].append(  # type: ignore[union-attr]
            _run_case(
                functions,
                tokens,
                12,
                CONTRACTS,
                args.seed + index * 1009,
                args.warmup,
                args.samples,
            )
        )
        _write(args.json, result)
    interaction_index = 0
    for tokens in INTERACTION_LENGTHS:
        for heads in INTERACTION_HEADS:
            print(f"interaction sweep T={tokens}, H={heads}, contract=bf16_both")
            result["length_head_interactions_bf16"].append(  # type: ignore[union-attr]
                _run_case(
                    functions,
                    tokens,
                    heads,
                    ("bf16_both",),
                    args.seed + 50000 + interaction_index * 1009,
                    args.warmup,
                    args.samples,
                )
            )
            interaction_index += 1
            _write(args.json, result)
    result["exact_gate_pass"] = True
    result["complete"] = True
    _write(args.json, result)
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
