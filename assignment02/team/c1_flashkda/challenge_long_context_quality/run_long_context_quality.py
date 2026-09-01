#!/usr/bin/env python3
"""Stream long contexts to measure BF16 recurrent persistence against FP32 FLA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_vshard4_prefetch2 import (  # noqa: E402
    vshard4_prefetch2,
)
from assignment02.team.c1_flashkda.harness import validate_and_bench as common  # noqa: E402


LOG2E = 1.4426950408889634
LOWER_BOUND = -5.0
D = 128


def _load_torch_ref_module(reference_root: Path) -> Any:
    ref_file = reference_root / "tests" / "torch_ref.py"
    if not ref_file.is_file():
        raise FileNotFoundError(f"missing exact reference: {ref_file}")
    spec = importlib.util.spec_from_file_location("c1_long_context_torch_ref", ref_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {ref_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class RawInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor


@dataclass(frozen=True)
class ModelParams:
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    scale: float


def _params(heads: int, regime: str, seed: int) -> ModelParams:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if regime == "random_service":
        a_log = torch.rand(heads, dtype=torch.float32, device="cuda", generator=generator)
        dt_bias = torch.rand(
            heads, D, dtype=torch.float32, device="cuda", generator=generator
        )
    elif regime == "retention_stress_gm8_bm4":
        a_log = torch.zeros(heads, dtype=torch.float32, device="cuda")
        dt_bias = torch.zeros(heads, D, dtype=torch.float32, device="cuda")
    else:
        raise ValueError(f"unknown regime {regime}")
    return ModelParams(a_log.contiguous(), dt_bias.contiguous(), 1.0 / math.sqrt(D))


def _raw_inputs(tokens: int, heads: int, regime: str, generator: torch.Generator) -> RawInputs:
    shape = (1, tokens, heads, D)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    if regime == "random_service":
        g = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
        beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda", generator=generator)
    elif regime == "retention_stress_gm8_bm4":
        g = torch.full(shape, -8.0, dtype=torch.bfloat16, device="cuda")
        beta = torch.full(shape[:-1], -4.0, dtype=torch.bfloat16, device="cuda")
    else:
        raise ValueError(f"unknown regime {regime}")
    return RawInputs(
        q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous()
    )


def _raw_call(
    fn: Callable[..., None],
    raw: RawInputs,
    params: ModelParams,
    initial_state: torch.Tensor,
    final_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros_like(raw.v)
    final_state = torch.empty(
        initial_state.shape, dtype=final_dtype, device=initial_state.device
    )
    fn(
        raw.q,
        raw.k,
        raw.v,
        raw.g,
        raw.beta,
        params.scale,
        out,
        A_log=params.a_log,
        dt_bias=params.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=initial_state,
        final_state=final_state,
    )
    torch.cuda.synchronize()
    return out, final_state


def _torch_ref_call(
    torch_ref: Callable[..., None],
    raw: RawInputs,
    params: ModelParams,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros_like(raw.v)
    final_state = torch.empty_like(initial_state)
    torch_ref(
        raw.q,
        raw.k,
        raw.v,
        raw.g,
        raw.beta,
        params.scale,
        out,
        A_log=params.a_log,
        dt_bias=params.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=initial_state,
        final_state=final_state,
    )
    torch.cuda.synchronize()
    return out, final_state


def _preprocess_for_oracle(
    raw: RawInputs,
    params: ModelParams,
    torch_ref_module: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch_ref_module.l2_normalize_kernel_match(raw.q).float()
    k = torch_ref_module.l2_normalize_kernel_match(raw.k).float()
    gate_argument = raw.g.float() + params.dt_bias.view(1, 1, raw.g.shape[2], D)
    a_exp = torch_ref_module.fp32_ex2_ftz(params.a_log * LOG2E).view(
        1, 1, raw.g.shape[2], 1
    )
    gate = LOWER_BOUND * torch_ref_module.sigmoid_ext.sigmoid_tanh_fp32(
        a_exp * gate_argument
    )
    beta = torch_ref_module.sigmoid_ext.sigmoid_tanh_fp32(raw.beta.float())
    # Match the values actually consumed by FlashKDA's BF16 update while the
    # oracle itself retains an FP32 recurrent accumulator.
    beta = beta.to(torch.bfloat16).float()
    return q, k, raw.v.float(), gate, beta


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int]:
    actual_f = actual.float()
    reference_f = reference.float()
    if not torch.isfinite(actual_f).all() or not torch.isfinite(reference_f).all():
        raise FloatingPointError("non-finite value in quality metric inputs")
    diff = actual_f - reference_f
    flat_diff = diff.flatten()
    flat_ref = reference_f.flatten()
    numel = flat_diff.numel()
    sample_limit = 65536
    stride = max(1, math.ceil(numel / sample_limit))
    sampled_abs = flat_diff.abs()[::stride][:sample_limit]
    diff_sq = torch.sum(flat_diff * flat_diff)
    ref_sq = torch.sum(flat_ref * flat_ref)
    actual_sq = torch.sum(actual_f.flatten() * actual_f.flatten())
    dot = torch.sum(actual_f.flatten() * flat_ref)
    eps = 1e-12
    return {
        "numel": numel,
        "mae": float(flat_diff.abs().mean().item()),
        "rmse": float(torch.sqrt(diff_sq / numel).item()),
        "relative_l2": float(torch.sqrt(diff_sq / (ref_sq + eps)).item()),
        "max_abs": float(flat_diff.abs().max().item()),
        "p99_abs_sampled": float(torch.quantile(sampled_abs, 0.99).item()),
        "p99_sample_count": int(sampled_abs.numel()),
        "cosine": float((dot / torch.sqrt((actual_sq + eps) * (ref_sq + eps))).item()),
    }


def _summarize_segments(segments: list[dict[str, Any]]) -> dict[str, object]:
    labels = (
        "output_persist_vs_fp32",
        "output_reseed_vs_fp32",
        "output_persist_vs_reseed",
        "state_persist_vs_fp32",
        "state_reseed_vs_fp32",
        "state_persist_vs_reseed",
    )
    result: dict[str, object] = {}
    for label in labels:
        rel = [float(segment[label]["relative_l2"]) for segment in segments]
        max_abs = [float(segment[label]["max_abs"]) for segment in segments]
        p99 = [float(segment[label]["p99_abs_sampled"]) for segment in segments]
        result[label] = {
            "relative_l2_final": rel[-1],
            "relative_l2_mean": statistics.fmean(rel),
            "relative_l2_max": max(rel),
            "max_abs_max": max(max_abs),
            "p99_abs_sampled_max": max(p99),
        }
    return result


def _preflight_exact(torch_ref_module: Any) -> dict[str, object]:
    import flash_kda

    result: dict[str, object] = {"seeds": {}, "exact_gate_pass": False}
    for seed in (0, 1):
        heads, tokens = 1, 1024
        params = _params(heads, "random_service", 70000 + seed)
        generator = torch.Generator(device="cuda").manual_seed(80000 + seed)
        baseline_state = torch.zeros(1, heads, D, D, dtype=torch.bfloat16, device="cuda")
        candidate_state = baseline_state.clone()
        seed_result: dict[str, object] = {"segments": []}
        for segment in range(2):
            raw = _raw_inputs(tokens, heads, "random_service", generator)
            baseline = _raw_call(
                flash_kda.fwd, raw, params, baseline_state.clone(), torch.bfloat16
            )
            candidate = _raw_call(
                vshard4_prefetch2.fwd,
                raw,
                params,
                candidate_state.clone(),
                torch.bfloat16,
            )
            reference = _torch_ref_call(
                torch_ref_module.torch_ref, raw, params, baseline_state.clone()
            )
            common.require_exact(
                f"preflight/seed{seed}/segment{segment}/candidate_vs_baseline/output",
                candidate[0],
                baseline[0],
            )
            common.require_exact(
                f"preflight/seed{seed}/segment{segment}/candidate_vs_baseline/state",
                candidate[1],
                baseline[1],
            )
            common.require_exact(
                f"preflight/seed{seed}/segment{segment}/baseline_vs_torch_ref/output",
                baseline[0],
                reference[0],
            )
            common.require_exact(
                f"preflight/seed{seed}/segment{segment}/baseline_vs_torch_ref/state",
                baseline[1],
                reference[1],
            )
            if segment == 0:
                fp32_abi = _raw_call(
                    flash_kda.fwd,
                    raw,
                    params,
                    baseline_state.float().contiguous(),
                    torch.float32,
                )
                common.require_exact(
                    f"preflight/seed{seed}/bf16_vs_fp32_abi/output",
                    baseline[0],
                    fp32_abi[0],
                )
                common.require_exact(
                    f"preflight/seed{seed}/bf16_vs_fp32_abi/state",
                    baseline[1].float(),
                    fp32_abi[1],
                )
            baseline_state, candidate_state = baseline[1], candidate[1]
            seed_result["segments"].append(  # type: ignore[index]
                {
                    "segment": segment,
                    "candidate_exact_baseline": True,
                    "baseline_exact_torch_ref": True,
                    "fp32_abi_is_bf16_compute": segment == 0,
                }
            )
        result["seeds"][str(seed)] = seed_result  # type: ignore[index]
    result["exact_gate_pass"] = True
    return result


def _oracle_calibration(torch_ref_module: Any) -> dict[str, object]:
    from fla.ops.kda import fused_recurrent_kda
    from assignment02.team.c1_flashkda.fla_kda_ref.naive import naive_recurrent_kda

    heads, tokens = 1, 64
    params = _params(heads, "random_service", 91001)
    generator = torch.Generator(device="cuda").manual_seed(91002)
    raw = _raw_inputs(tokens, heads, "random_service", generator)
    q, k, v, gate, beta = _preprocess_for_oracle(raw, params, torch_ref_module)
    initial = torch.zeros(1, heads, D, D, dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        fused_o, fused_state = fused_recurrent_kda(
            q,
            k,
            v,
            gate,
            beta,
            scale=params.scale,
            initial_state=initial.clone(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            state_v_first=False,
        )
        naive_o, naive_state = naive_recurrent_kda(
            q,
            k,
            v,
            gate,
            beta,
            scale=params.scale,
            initial_state=initial.clone(),
            output_final_state=True,
        )
        torch.cuda.synchronize()
    output_metrics = _metrics(fused_o, naive_o)
    state_metrics = _metrics(fused_state, naive_state)
    gate_pass = (
        float(output_metrics["relative_l2"]) < 0.005
        and float(state_metrics["relative_l2"]) < 0.005
    )
    if not gate_pass:
        raise AssertionError(
            f"FLA fused-vs-naive calibration exceeded 0.005: "
            f"output={output_metrics}, state={state_metrics}"
        )
    return {
        "shape": {"B": 1, "T": tokens, "H": heads, "K": D, "V": D},
        "output": output_metrics,
        "state": state_metrics,
        "relative_l2_threshold": 0.005,
        "gate_pass": True,
    }


def _quality_run(
    torch_ref_module: Any,
    heads: int,
    tokens_per_segment: int,
    segments: int,
    seed: int,
    regime: str,
) -> dict[str, object]:
    from fla.ops.kda import fused_recurrent_kda

    params = _params(heads, regime, 100000 + seed * 97 + heads)
    generator = torch.Generator(device="cuda").manual_seed(
        200000 + seed * 193 + heads
    )
    persistent_state = torch.zeros(
        1, heads, D, D, dtype=torch.bfloat16, device="cuda"
    )
    oracle_state = torch.zeros(1, heads, D, D, dtype=torch.float32, device="cuda")
    segment_results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for segment in range(segments):
            raw = _raw_inputs(tokens_per_segment, heads, regime, generator)
            oracle_initial = oracle_state
            q, k, v, gate, beta = _preprocess_for_oracle(raw, params, torch_ref_module)
            oracle_out, oracle_next = fused_recurrent_kda(
                q,
                k,
                v,
                gate,
                beta,
                scale=params.scale,
                initial_state=oracle_initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
                use_gate_in_kernel=False,
                use_beta_sigmoid_in_kernel=False,
                state_v_first=True,
            )
            persistent_out, persistent_next = _raw_call(
                vshard4_prefetch2.fwd,
                raw,
                params,
                persistent_state,
                torch.bfloat16,
            )
            reseed_initial = oracle_initial.to(torch.bfloat16).contiguous()
            reseed_out, reseed_next = _raw_call(
                vshard4_prefetch2.fwd,
                raw,
                params,
                reseed_initial,
                torch.bfloat16,
            )
            torch.cuda.synchronize()
            item = {
                "segment": segment + 1,
                "tokens_seen": (segment + 1) * tokens_per_segment,
                "output_persist_vs_fp32": _metrics(persistent_out, oracle_out),
                "output_reseed_vs_fp32": _metrics(reseed_out, oracle_out),
                "output_persist_vs_reseed": _metrics(persistent_out, reseed_out),
                "state_persist_vs_fp32": _metrics(persistent_next, oracle_next),
                "state_reseed_vs_fp32": _metrics(reseed_next, oracle_next),
                "state_persist_vs_reseed": _metrics(persistent_next, reseed_next),
            }
            segment_results.append(item)
            persistent_state = persistent_next
            oracle_state = oracle_next
            if segment == 0 or (segment + 1) % 4 == 0 or segment + 1 == segments:
                print(
                    f"quality H={heads} seed={seed} regime={regime} "
                    f"segment={segment + 1}/{segments} "
                    f"output_rel={item['output_persist_vs_fp32']['relative_l2']:.7g} "
                    f"state_rel={item['state_persist_vs_fp32']['relative_l2']:.7g}"
                )
            del raw, q, k, v, gate, beta, oracle_out, persistent_out, reseed_out
    return {
        "shape": {
            "B": 1,
            "segment_tokens": tokens_per_segment,
            "segments": segments,
            "total_tokens": tokens_per_segment * segments,
            "H": heads,
            "K": D,
            "V": D,
        },
        "seed": seed,
        "regime": regime,
        "segments": segment_results,
        "summary": _summarize_segments(segment_results),
    }


def _write(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch_ref = _load_torch_ref_module(args.reference_root)
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "multiprocessor_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "claim": "BF16-persisted FlashKDA recurrence versus independent FP32 FLA recurrent oracle; FP32 FlashKDA state ABI is not FP32 compute",
        "preflight": _preflight_exact(torch_ref),
        "oracle_calibration": _oracle_calibration(torch_ref),
        "main_runs": [],
        "complete": False,
    }
    _write(args.json, result)
    if not args.skip_main:
        matrix = (
            (1, 8192, 32, (0, 1, 2, 3)),
            (12, 8192, 16, (0, 1)),
        )
        for heads, tokens, segments, seeds in matrix:
            for seed in seeds:
                for regime in ("random_service", "retention_stress_gm8_bm4"):
                    run = _quality_run(
                        torch_ref, heads, tokens, segments, seed, regime
                    )
                    result["main_runs"].append(run)  # type: ignore[union-attr]
                    _write(args.json, result)
    result["complete"] = len(result["main_runs"]) == 12  # type: ignore[arg-type]
    _write(args.json, result)
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
