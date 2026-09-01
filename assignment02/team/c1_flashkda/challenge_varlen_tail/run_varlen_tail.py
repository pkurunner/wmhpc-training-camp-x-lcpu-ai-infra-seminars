#!/usr/bin/env python3
"""Audit vshard2-P2/vshard4-P2 on tails, batches, and varlen inputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2  # noqa: E402
from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch  # noqa: E402
from assignment02.team.c1_flashkda.challenge_vshard4_prefetch2 import (  # noqa: E402
    vshard4_prefetch2,
)
from assignment02.team.c1_flashkda.harness import validate_and_bench as common  # noqa: E402


VARIANTS = ("baseline", "vshard2_p2", "vshard4_p2")
CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
BENCH_CONTRACTS = ("none", "fp32_final_only")


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    tokens: int
    heads: int = 12
    lengths: tuple[int, ...] | None = None

    @property
    def sequences(self) -> int:
        return len(self.lengths) if self.lengths is not None else self.batch

    @property
    def total_tokens(self) -> int:
        return self.batch * self.tokens


@dataclass(frozen=True)
class Inputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    scale: float
    lower_bound: float
    cu_seqlens: torch.Tensor | None


CASES = (
    Case("fixed_t1", 1, 1),
    Case("fixed_t15", 1, 15),
    Case("fixed_t17", 1, 17),
    Case("fixed_t31", 1, 31),
    Case("fixed_t127", 1, 127),
    Case("fixed_tail_t8191", 1, 8191),
    Case("batch_b2_t17", 2, 17),
    Case("batch_b4_t127", 4, 127),
    Case("batch_b4_t2048", 4, 2048),
    Case("varlen_short", 1, 64, lengths=(1, 15, 17, 31)),
    Case("varlen_mixed_t8192", 1, 8192, lengths=(17, 511, 1024, 1300, 2049, 3291)),
)
REFERENCE_CONTRACTS = {
    "fixed_t17": set(CONTRACTS),
    "batch_b2_t17": set(CONTRACTS),
    "varlen_short": set(CONTRACTS),
    "varlen_mixed_t8192": {"none"},
}
BENCH_CASES = {"fixed_tail_t8191", "batch_b4_t2048", "varlen_mixed_t8192"}


def _case_dict(case: Case) -> dict[str, object]:
    return {
        "name": case.name,
        "B": case.batch,
        "T": case.tokens,
        "H": case.heads,
        "K": 128,
        "V": 128,
        "N": case.sequences,
        "total_tokens": case.total_tokens,
        "lengths": None if case.lengths is None else list(case.lengths),
        "has_tail_chunk": (
            case.tokens % 16 != 0
            if case.lengths is None
            else any(length % 16 != 0 for length in case.lengths)
        ),
    }


def _make_inputs(case: Case, seed: int) -> Inputs:
    if case.lengths is not None and sum(case.lengths) != case.total_tokens:
        raise ValueError(f"{case.name}: lengths do not sum to B*T")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (case.batch, case.tokens, case.heads, 128)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda", generator=generator)
    cu_seqlens = None
    if case.lengths is not None:
        cumulative = [0]
        for length in case.lengths:
            cumulative.append(cumulative[-1] + length)
        cu_seqlens = torch.tensor(cumulative, dtype=torch.long, device="cuda")
    return Inputs(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        a_log=torch.rand(case.heads, dtype=torch.float32, device="cuda", generator=generator),
        dt_bias=torch.rand(
            case.heads, 128, dtype=torch.float32, device="cuda", generator=generator
        ).contiguous(),
        scale=1.0 / math.sqrt(128),
        lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
    )


def _states(
    contract: str, case: Case, seed: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if contract == "none":
        return None, None
    dtype = torch.float32 if contract in ("fp32_both", "fp32_final_only") else torch.bfloat16
    final = torch.zeros(
        case.sequences,
        case.heads,
        128,
        128,
        dtype=dtype,
        device="cuda",
    )
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
    x: Inputs,
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
        cu_seqlens=x.cu_seqlens,
    )
    torch.cuda.synchronize()
    return out, final


def _compare(
    name: str,
    actual: tuple[torch.Tensor, torch.Tensor | None],
    expected: tuple[torch.Tensor, torch.Tensor | None],
) -> dict[str, object]:
    common.require_exact(f"{name}/output", actual[0], expected[0])
    result: dict[str, object] = {
        "output_exact": True,
        "output_max_abs": common.max_abs(actual[0], expected[0]),
    }
    if actual[1] is None or expected[1] is None:
        if actual[1] is not None or expected[1] is not None:
            raise AssertionError(f"{name}: final-state presence mismatch")
        result["final_state_present"] = False
    else:
        common.require_exact(f"{name}/final_state", actual[1], expected[1])
        result.update(
            {
                "final_state_present": True,
                "final_state_exact": True,
                "final_state_max_abs": common.max_abs(actual[1], expected[1]),
            }
        )
    return result


def _validate(
    functions: dict[str, Callable[..., None]],
    torch_ref: Callable[..., None],
    case: Case,
    x: Inputs,
    contract: str,
    seed: int,
) -> dict[str, object]:
    initial, final = _states(contract, case, seed)
    outputs = {
        label: _invoke(fn, x, _clone(initial), _clone(final))
        for label, fn in functions.items()
    }
    baseline = outputs["baseline"]
    result: dict[str, object] = {
        label: _compare(f"{case.name}/{contract}/{label}_vs_baseline", outputs[label], baseline)
        for label in ("vshard2_p2", "vshard4_p2")
    }
    if contract in REFERENCE_CONTRACTS.get(case.name, set()):
        reference = _invoke(torch_ref, x, _clone(initial), _clone(final))
        result["baseline_vs_torch_ref"] = _compare(
            f"{case.name}/{contract}/baseline_vs_torch_ref", baseline, reference
        )
    return result


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
    case: Case,
    x: Inputs,
    contract: str,
    seed: int,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    calls: dict[str, Callable[[], None]] = {}
    for label, fn in functions.items():
        initial, final = _states(contract, case, seed)
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
                cu_seqlens=x.cu_seqlens,
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
    summaries = {label: _summary(values) for label, values in raw.items()}
    baseline = float(summaries["baseline"]["p50_ms"])
    v2 = float(summaries["vshard2_p2"]["p50_ms"])
    v4 = float(summaries["vshard4_p2"]["p50_ms"])
    return {
        "paths": summaries,
        "raw_samples_ms": raw,
        "baseline_over_vshard2_p2_p50_x": baseline / v2,
        "baseline_over_vshard4_p2_p50_x": baseline / v4,
        "vshard2_p2_over_vshard4_p2_p50_x": v2 / v4,
        "winner_p50": min(VARIANTS, key=lambda label: float(summaries[label]["p50_ms"])),
        "event_contract": "three-path cyclic rotation; one wrapper call per event; workspace allocation included",
    }


def _runtime_fallback(functions: dict[str, Callable[..., None]], seed: int) -> dict[str, object]:
    case = next(item for item in CASES if item.name == "varlen_mixed_t8192")
    x = _make_inputs(case, seed)
    baseline = _invoke(functions["baseline"], x, None, None)
    out = torch.zeros_like(x.v)
    auto_dispatch.fwd(
        x.q,
        x.k,
        x.v,
        x.g,
        x.beta,
        x.scale,
        out,
        x.a_log,
        x.dt_bias,
        x.lower_bound,
        cu_seqlens=x.cu_seqlens,
    )
    torch.cuda.synchronize()
    check = _compare("runtime_dispatch_varlen/public_vs_baseline", (out, None), baseline)
    decision = auto_dispatch.get_last_decision()
    if decision.get("chosen_variant") != "baseline" or decision.get("reason") != "varlen_cu_seqlens_not_whitelisted":
        raise AssertionError(f"varlen dispatcher did not fail closed: {decision}")
    return {"correctness": check, "decision": decision}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
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
    torch_ref = common._load_torch_ref(args.reference_root)
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "multiprocessor_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "extension": _identity(),
        "chunk": 16,
        "cases": [_case_dict(case) for case in CASES],
        "contracts": list(CONTRACTS),
        "correctness": {},
        "exact_gate_pass": False,
    }
    inputs: dict[str, Inputs] = {}
    for case_index, case in enumerate(CASES):
        x = _make_inputs(case, args.seed + case_index * 1009)
        inputs[case.name] = x
        result["correctness"][case.name] = {}  # type: ignore[index]
        for contract_index, contract in enumerate(CONTRACTS):
            result["correctness"][case.name][contract] = _validate(  # type: ignore[index]
                functions,
                torch_ref,
                case,
                x,
                contract,
                args.seed + case_index * 1009 + contract_index * 101,
            )
    result["runtime_dispatch_fallback"] = _runtime_fallback(functions, args.seed + 99991)
    result["exact_gate_pass"] = True
    result["benchmark"] = {}
    for case_index, case in enumerate(CASES):
        if case.name not in BENCH_CASES:
            continue
        result["benchmark"][case.name] = {}  # type: ignore[index]
        for contract_index, contract in enumerate(BENCH_CONTRACTS):
            result["benchmark"][case.name][contract] = _benchmark(  # type: ignore[index]
                functions,
                case,
                inputs[case.name],
                contract,
                args.seed + case_index * 1009 + contract_index * 101,
                args.warmup,
                args.samples,
            )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
