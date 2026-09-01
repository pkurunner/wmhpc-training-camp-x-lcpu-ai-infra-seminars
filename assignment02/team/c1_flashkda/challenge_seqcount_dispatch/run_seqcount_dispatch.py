#!/usr/bin/env python3
"""Test whether M=N_seq*H explains fixed-batch and varlen path winners.

The matrix intentionally keeps each logical sequence at 2048 tokens.  A fixed
batch therefore has shape [N_seq, 2048, H, 128], while the matching packed
balanced-varlen case has shape [1, N_seq*2048, H, 128] and N_seq equal spans.
The skewed cases retain N_seq, H and total tokens but concentrate nearly all
tokens in the last sequence.

This is an evidence-gathering runner, not a dispatcher.  It never changes the
runtime selection policy.  The JSON contains a predeclared M-only promotion
gate; a disagreement between forms or percentile winners rejects that policy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHUNK = 16
DIM = 128
VARIANTS = ("baseline", "vshard2_p2", "vshard4_p2")
CONTRACTS = ("none", "bf16_both", "fp32_both", "fp32_final_only")
FLA_PUBLIC_CONTRACTS = ("none", "fp32_final_only", "fp32_both")
BENCH_CONTRACTS = FLA_PUBLIC_CONTRACTS
PERCENTILES = ("p50", "p95", "p99")
MIN_WINNER_MARGIN = 0.02
AUDITED_EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
B300_CAPABILITY = (10, 3)
B300_SM_COUNT = 148


@dataclass(frozen=True)
class Case:
    """One fixed or packed representation of an N_seq/H experiment point."""

    name: str
    form: str
    sequences: int
    heads: int
    lengths: tuple[int, ...]
    family: str = "main_t2048"

    @property
    def m(self) -> int:
        return self.sequences * self.heads

    @property
    def is_varlen(self) -> bool:
        return self.form != "fixed"

    @property
    def batch(self) -> int:
        return 1 if self.is_varlen else self.sequences

    @property
    def tokens_per_batch_item(self) -> int:
        return sum(self.lengths) if self.is_varlen else self.lengths[0]

    @property
    def total_tokens(self) -> int:
        return sum(self.lengths)


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


# (N_seq, H): every tested M has two different factorizations.  Without those
# pairs, agreement between fixed and packed forms would test representation,
# not the stronger M=N_seq*H-only hypothesis.
POINTS = (
    (1, 24),
    (2, 12),   # M=24
    (1, 36),
    (3, 12),   # M=36
    (1, 37),
    (37, 1),   # M=37
    (1, 38),
    (38, 1),   # M=38
    (1, 39),
    (39, 1),   # M=39
    (1, 40),
    (40, 1),   # M=40
    (1, 48),
    (4, 12),   # M=48
    (1, 72),
    (6, 12),   # M=72
    (1, 75),
    (75, 1),   # M=75
    (1, 76),
    (76, 1),   # M=76
    (1, 96),
    (8, 12),   # M=96
)
SKEW_POINTS = {(3, 12), (38, 1), (6, 12), (76, 1)}


def _balanced_lengths(sequences: int) -> tuple[int, ...]:
    return (2048,) * sequences


def _skewed_lengths(sequences: int) -> tuple[int, ...]:
    """Use N-1 one-token sequences and one long tail at fixed total tokens."""
    if sequences < 3:
        raise ValueError("the designed skew matrix requires at least three sequences")
    return (1,) * (sequences - 1) + (2048 * sequences - (sequences - 1),)


def _t257_control_skewed_lengths(sequences: int) -> tuple[int, ...]:
    """Preserve total tokens, length mod 16, and actual K1 tiles at T=257.

    N=37 gives [17]*36+[8897]; N=38 gives [17]*37+[9137].  Every component is
    congruent to one modulo 16, and the sum of ceil(length/16) values is the
    same as N * ceil(257/16).
    """
    if sequences not in (37, 38):
        raise ValueError("T=257 control is intentionally defined only for N=37/38")
    return (17,) * (sequences - 1) + (257 * sequences - 17 * (sequences - 1),)


def _make_cases() -> tuple[Case, ...]:
    cases: list[Case] = []
    for sequences, heads in POINTS:
        stem = f"m{sequences * heads:03d}_n{sequences:02d}_h{heads:02d}"
        cases.append(Case(f"{stem}_fixed", "fixed", sequences, heads, _balanced_lengths(sequences)))
        cases.append(
            Case(
                f"{stem}_balanced_varlen",
                "balanced_varlen",
                sequences,
                heads,
                _balanced_lengths(sequences),
            )
        )
        if (sequences, heads) in SKEW_POINTS:
            cases.append(
                Case(
                    f"{stem}_skewed_varlen",
                    "skewed_varlen",
                    sequences,
                    heads,
                    _skewed_lengths(sequences),
                )
            )
    # A necessary-control family around M=37/38.  Each representation has the
    # same N, H, total tokens, every sequence length mod 16, and *actual* K1
    # tile count.  Only the packed representation/length distribution differs.
    for sequences in (37, 38):
        heads = 1
        stem = f"m{sequences:03d}_n{sequences:02d}_h01_t257_control"
        uniform = (257,) * sequences
        cases.extend(
            (
                Case(f"{stem}_fixed", "fixed", sequences, heads, uniform, "t257_mod16_control"),
                Case(
                    f"{stem}_balanced_varlen",
                    "balanced_varlen",
                    sequences,
                    heads,
                    uniform,
                    "t257_mod16_control",
                ),
                Case(
                    f"{stem}_skewed_varlen",
                    "skewed_varlen",
                    sequences,
                    heads,
                    _t257_control_skewed_lengths(sequences),
                    "t257_mod16_control",
                ),
            )
        )
    return tuple(cases)


CASES = _make_cases()
# Pinned Torch-reference checks deliberately span fixed, balanced packed, and
# severe-skew forms, without turning the Python chunk-by-chunk reference into
# the dominant cost of the full matrix.
REFERENCE_CONTRACTS = {
    "m024_n02_h12_fixed": set(CONTRACTS),
    "m024_n02_h12_balanced_varlen": set(CONTRACTS),
    "m036_n03_h12_skewed_varlen": {"none"},
    "m037_n37_h01_t257_control_fixed": set(CONTRACTS),
    "m037_n37_h01_t257_control_balanced_varlen": {"none"},
    "m037_n37_h01_t257_control_skewed_varlen": {"none"},
    "m038_n38_h01_t257_control_fixed": set(CONTRACTS),
    "m038_n38_h01_t257_control_balanced_varlen": {"none", "fp32_both"},
    "m038_n38_h01_t257_control_skewed_varlen": {"none", "fp32_both"},
}
FIXED_TP8_BATCH_CASES = {
    "m024_n02_h12_fixed",
    "m036_n03_h12_fixed",
    "m048_n04_h12_fixed",
}


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _grid_sizes(case: Case) -> dict[str, object]:
    """Record launch dimensions implied by the audited C++ launch entries."""
    actual_tiles = sum(_ceil_div(length, CHUNK) for length in case.lengths)
    if case.is_varlen:
        # flash_kda.cpp uses an upper bound for the packed varlen K1 grid.
        launch_tiles = _ceil_div(case.total_tokens, CHUNK) + case.sequences
    else:
        launch_tiles = case.sequences * _ceil_div(case.tokens_per_batch_item, CHUNK)
    if launch_tiles < actual_tiles:
        raise AssertionError(f"{case.name}: varlen upper bound is below actual tile count")
    result: dict[str, object] = {
        "chunk": CHUNK,
        "actual_tiles_all_sequences": actual_tiles,
        "launch_total_tiles": launch_tiles,
        "tile_prefix_kernel": (
            {"grid_xyz": [1, 1, 1], "block_xyz": [32, 1, 1]} if case.is_varlen else None
        ),
    }
    for label, shard_count in (("baseline", 1), ("vshard2_p2", 2), ("vshard4_p2", 4)):
        result[label] = {
            "kernel1_grid_xyz": [launch_tiles, case.heads, 1],
            "kernel2_grid_xyz": [case.sequences, case.heads * shard_count, 1],
            "kernel2_ctas": case.m * shard_count,
            "k2_value_shards_per_head": shard_count,
        }
    return result


def _benchmark_contracts_for(case: Case) -> tuple[str, ...]:
    """Use all FLA-public state modes for the proposed fixed TP8 batch points."""
    return FLA_PUBLIC_CONTRACTS if case.name in FIXED_TP8_BATCH_CASES else BENCH_CONTRACTS


def _case_dict(case: Case) -> dict[str, object]:
    return {
        "name": case.name,
        "family": case.family,
        "form": case.form,
        "B": case.batch,
        "T": case.tokens_per_batch_item,
        "H": case.heads,
        "N_seq": case.sequences,
        "M": case.m,
        "K": DIM,
        "V": DIM,
        "total_tokens": case.total_tokens,
        "lengths": list(case.lengths),
        "lengths_mod_chunk": [length % CHUNK for length in case.lengths],
        "grid_sizes": _grid_sizes(case),
    }


def _make_inputs(case: Case, seed: int) -> Inputs:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (case.batch, case.tokens_per_batch_item, case.heads, DIM)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda", generator=generator)
    cu_seqlens = None
    if case.is_varlen:
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
        dt_bias=torch.rand(case.heads, DIM, dtype=torch.float32, device="cuda", generator=generator).contiguous(),
        scale=1.0 / math.sqrt(DIM),
        lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
    )


def _states(
    contract: str, case: Case, seed: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if contract == "none":
        return None, None
    dtype = torch.float32 if contract in ("fp32_both", "fp32_final_only") else torch.bfloat16
    final = torch.zeros(case.sequences, case.heads, DIM, DIM, dtype=dtype, device="cuda")
    if contract == "fp32_final_only":
        return None, final
    generator = torch.Generator(device="cuda").manual_seed(seed)
    initial = torch.randn(final.shape, dtype=dtype, device="cuda", generator=generator).contiguous()
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
        result["baseline_vs_pinned_torch_ref"] = _compare(
            f"{case.name}/{contract}/baseline_vs_pinned_torch_ref", baseline, reference
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
    paths = {label: _summary(values) for label, values in raw.items()}
    winner_by_percentile = {
        percentile: min(VARIANTS, key=lambda label: float(paths[label][f"{percentile}_ms"]))
        for percentile in PERCENTILES
    }
    p50 = {label: float(paths[label]["p50_ms"]) for label in VARIANTS}
    return {
        "paths": paths,
        "raw_samples_ms": raw,
        "winner_by_percentile": winner_by_percentile,
        "winner_p50": winner_by_percentile["p50"],
        "winner_p95": winner_by_percentile["p95"],
        "winner_p99": winner_by_percentile["p99"],
        "baseline_over_vshard2_p2_p50_x": p50["baseline"] / p50["vshard2_p2"],
        "baseline_over_vshard4_p2_p50_x": p50["baseline"] / p50["vshard4_p2"],
        "vshard2_p2_over_vshard4_p2_p50_x": p50["vshard2_p2"] / p50["vshard4_p2"],
        "event_contract": (
            "three-path cyclic rotation; one wrapper call per CUDA event; "
            "the wrapper invokes its normal workspace allocation path, while CUDA-event latency "
            "does not include host allocator overhead"
        ),
    }


def _identity() -> dict[str, object]:
    import flash_kda_C

    required = ("fwd", "fwd_vshard_p2", "fwd_vshard4_p2", "get_workspace_size")
    missing = [symbol for symbol in required if not callable(getattr(flash_kda_C, symbol, None))]
    if missing:
        raise RuntimeError(f"loaded extension lacks required symbols: {missing}")
    path = Path(flash_kda_C.__file__).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != AUDITED_EXTENSION_SHA256:
        raise RuntimeError(
            "loaded extension is not the audited one-SO comparison binary: "
            f"expected {AUDITED_EXTENSION_SHA256}, got {digest} at {path}"
        )
    return {
        "path": str(path),
        "sha256": digest,
        "sha256_gate_pass": True,
        "required_symbols": list(required),
    }


def _device_identity() -> dict[str, object]:
    name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if "B300" not in name.upper() or capability != B300_CAPABILITY or sm_count != B300_SM_COUNT:
        raise RuntimeError(
            "this experiment is B300-only: "
            f"got name={name!r}, capability={capability}, SMs={sm_count}; "
            f"expected a B300 with capability={B300_CAPABILITY}, SMs={B300_SM_COUNT}"
        )
    return {
        "name": name,
        "capability": list(capability),
        "multiprocessor_count": sm_count,
        "b300_gate_pass": True,
    }


def _winner_evidence(benchmark: dict[str, object]) -> dict[str, object]:
    winner_by_percentile = benchmark["winner_by_percentile"]
    if not isinstance(winner_by_percentile, dict):
        raise TypeError("winner_by_percentile must be a dictionary")
    winner_names = [str(winner_by_percentile[p]) for p in PERCENTILES]
    single_winner = len(set(winner_names)) == 1
    margins: dict[str, float] = {}
    for percentile in PERCENTILES:
        metric = f"{percentile}_ms"
        ranked = sorted(
            (
                (float(benchmark["paths"][label][metric]), label)  # type: ignore[index]
                for label in VARIANTS
            ),
            key=lambda item: item[0],
        )
        margins[percentile] = ranked[1][0] / ranked[0][0] - 1.0
    margin_pass = all(value >= MIN_WINNER_MARGIN for value in margins.values())
    return {
        "winner_by_percentile": winner_by_percentile,
        "single_winner_all_percentiles": single_winner,
        "winner": winner_names[0] if single_winner else None,
        "winner_margin_over_runner_up": margins,
        "minimum_required_margin": MIN_WINNER_MARGIN,
        "margin_gate_pass": margin_pass,
        "stable_winner_gate_pass": single_winner and margin_pass,
    }


def _policy_definition() -> dict[str, object]:
    return {
        "hypothesis": "M=N_seq*H alone may predict the v2/v4/baseline winner.",
        "predeclared_stop_rule": (
            "For any measured FLA-public state contract, reject an M-only dispatch policy if any "
            "fixed, balanced-varlen, or skewed-varlen representation at the same M has a "
            "different winner at P50/P95/P99.  Raw latency equality is not required; the "
            "identity of the winner at each percentile is the comparison."
        ),
        "promotion_rule": (
            "M-only policy is eligible for a separate dispatcher decision only if every tested "
            "M has at least two distinct (N_seq,H) factorizations, every case has one winner across "
            "P50/P95/P99 with at least a 2% runner-up margin, and every same-M form/factorization "
            "selects that same winner for all measured contracts. Eligibility does not override unmeasured "
            "shapes or existing fail-closed guards."
        ),
        "measured_contracts": list(BENCH_CONTRACTS),
        "fixed_tp8_batch_focus": {
            "cases": sorted(FIXED_TP8_BATCH_CASES),
            "contracts": list(FLA_PUBLIC_CONTRACTS),
            "reason": "separate exact-shape gate before considering a fixed B=2/3/4,H=12,T=2048 FLA whitelist",
        },
        "required_percentiles": list(PERCENTILES),
        "minimum_winner_margin": MIN_WINNER_MARGIN,
    }


def _assess_m_only_policy(result: dict[str, object]) -> dict[str, object]:
    benchmark = result["benchmark"]
    by_m: dict[int, list[Case]] = {}
    for case in CASES:
        by_m.setdefault(case.m, []).append(case)
    assessment: dict[str, object] = {"per_m": {}, "m_only_policy_eligible": False}
    all_consistent = True
    for m, cases in sorted(by_m.items()):
        factorizations = sorted({(case.sequences, case.heads) for case in cases})
        factorization_coverage_pass = len(factorizations) >= 2
        per_contract: dict[str, object] = {}
        per_m_pass = factorization_coverage_pass
        for contract in BENCH_CONTRACTS:
            winner_vectors = {
                case.name: {
                    "form": case.form,
                    "family": case.family,
                    "N_seq": case.sequences,
                    "H": case.heads,
                    **_winner_evidence(benchmark[case.name][contract]),  # type: ignore[index]
                }
                for case in cases
            }
            vectors_only = [entry["winner_by_percentile"] for entry in winner_vectors.values()]
            canonical = vectors_only[0]
            consistent_across_cases = all(vector == canonical for vector in vectors_only)
            stable_with_margin = all(
                bool(entry["stable_winner_gate_pass"]) for entry in winner_vectors.values()
            )
            contract_pass = consistent_across_cases and stable_with_margin
            per_contract[contract] = {
                "cases": [case.name for case in cases],
                "winner_by_percentile_per_case": winner_vectors,
                "consistent_across_cases": consistent_across_cases,
                "all_cases_have_one_margin_stable_winner": stable_with_margin,
                "contract_gate_pass": contract_pass,
            }
            per_m_pass = per_m_pass and contract_pass
        assessment["per_m"][str(m)] = {  # type: ignore[index]
            "factorizations": [list(pair) for pair in factorizations],
            "factorization_coverage_pass": factorization_coverage_pass,
            "contracts": per_contract,
            "m_gate_pass": per_m_pass,
        }
        all_consistent = all_consistent and per_m_pass
    assessment["m_only_policy_eligible"] = all_consistent
    assessment["decision"] = (
        "eligible_for_separate_dispatch_review"
        if all_consistent
        else "STOP_do_not_promote_M_only_policy"
    )
    return assessment


def _assess_fixed_tp8_batch(result: dict[str, object]) -> dict[str, object]:
    benchmark = result["benchmark"]
    assessment: dict[str, object] = {"cases": {}, "all_contracts_gate_pass": True}
    all_pass = True
    for case_name in sorted(FIXED_TP8_BATCH_CASES):
        per_contract = {
            contract: _winner_evidence(benchmark[case_name][contract])  # type: ignore[index]
            for contract in FLA_PUBLIC_CONTRACTS
        }
        case_pass = all(
            bool(evidence["stable_winner_gate_pass"]) for evidence in per_contract.values()
        )
        assessment["cases"][case_name] = {  # type: ignore[index]
            "contracts": per_contract,
            "case_gate_pass": case_pass,
        }
        all_pass = all_pass and case_pass
    assessment["all_contracts_gate_pass"] = all_pass
    assessment["decision"] = (
        "eligible_for_exact_fixed_batch_dispatch_review"
        if all_pass
        else "STOP_do_not_expand_fixed_batch_dispatch"
    )
    return assessment


def _write(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--describe", action="store_true", help="write the matrix/policy without CUDA")
    args = parser.parse_args()
    if args.warmup < 0 or args.samples <= 0:
        raise ValueError("warmup must be nonnegative and samples positive")
    result: dict[str, object] = {
        "schema_version": 1,
        "purpose": "sequence-count/head boundary experiment; no dispatcher mutation",
        "chunk": CHUNK,
        "cases": [_case_dict(case) for case in CASES],
        "raw_state_contracts": list(CONTRACTS),
        "benchmark_contracts_default": list(BENCH_CONTRACTS),
        "benchmark_contracts_for_fixed_tp8_batch": list(FLA_PUBLIC_CONTRACTS),
        "samples": args.samples,
        "warmup": args.warmup,
        "policy_definition": _policy_definition(),
        "correctness": {},
        "benchmark": {},
        "exact_gate_pass": False,
        "complete": False,
    }
    if args.describe:
        result["describe_only"] = True
        _write(args.json, result)
        print(f"wrote matrix description {args.json}")
        return
    if args.reference_root is None:
        raise ValueError("--reference-root is required for a GPU experiment")
    # Keep --describe usable on a planning machine without PyTorch/CUDA.  The
    # execution path still imports exactly the audited wrappers.
    global torch, prefetch2, vshard4_prefetch2, common
    import torch
    from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2
    from assignment02.team.c1_flashkda.challenge_vshard4_prefetch2 import vshard4_prefetch2
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    import flash_kda

    functions: dict[str, Callable[..., None]] = {
        "baseline": flash_kda.fwd,
        "vshard2_p2": prefetch2.fwd,
        "vshard4_p2": vshard4_prefetch2.fwd,
    }
    torch_ref = common._load_torch_ref(args.reference_root)
    result.update({"device": _device_identity(), "extension": _identity()})
    _write(args.json, result)
    for case_index, case in enumerate(CASES):
        print(f"correctness {case.name}: M={case.m}, contracts={CONTRACTS}")
        x = _make_inputs(case, args.seed + case_index * 1009)
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
        del x
        torch.cuda.empty_cache()
        _write(args.json, result)
    result["exact_gate_pass"] = True
    _write(args.json, result)
    for case_index, case in enumerate(CASES):
        benchmark_contracts = _benchmark_contracts_for(case)
        print(
            f"benchmark {case.name}: M={case.m}, contracts={benchmark_contracts}, "
            f"samples={args.samples}"
        )
        x = _make_inputs(case, args.seed + case_index * 1009)
        result["benchmark"][case.name] = {}  # type: ignore[index]
        for contract_index, contract in enumerate(benchmark_contracts):
            result["benchmark"][case.name][contract] = _benchmark(  # type: ignore[index]
                functions,
                case,
                x,
                contract,
                args.seed + case_index * 1009 + contract_index * 101,
                args.warmup,
                args.samples,
            )
        del x
        torch.cuda.empty_cache()
        _write(args.json, result)
    result["m_only_policy_assessment"] = _assess_m_only_policy(result)
    result["fixed_tp8_batch_assessment"] = _assess_fixed_tp8_batch(result)
    result["complete"] = True
    _write(args.json, result)
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
