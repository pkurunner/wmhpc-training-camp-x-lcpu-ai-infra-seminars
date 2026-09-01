"""Frozen FP8 AB/BA policy check for the complete prepared C>1 dispatch.

Each invocation measures one predeclared FP8 storage mode, batch size, and
base seed.  The two arms bind the *same* prepared problem and scale tensors to
different caller-owned outputs.  Both therefore execute decode followed by the
required split-K merge; only ``decode_num_stages`` changes from 3 to 5.

The boundary is the complete prepared sparse-decode dispatch, not an isolated
decode kernel and not a model/server end-to-end benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch
import triton

from harness.data import DecodeProblem, make_decode_problem
from harness.reference import dense_sparse_attention_reference
from .prepared_tuned import TuningConfig, TunedPreparedSparseDecode


STORAGES = ("fp8-scalar", "fp8-token")
BATCHES = (1, 4, 8, 16)
BASE_SEEDS = (20260828, 20260829)
MAX_SEQ_LEN = 4096
RTOL = ATOL = 3e-2
WARMUP = 30
ABBA_PAIRS = 101
SAMPLES_PER_RUNNER = 2 * ABBA_PAIRS

# This is the selected C policy, frozen before collecting any FP8 event data.
SELECTED_CHUNKS: dict[str, dict[int, int]] = {
    "fp8-scalar": {1: 16, 4: 4, 8: 8, 16: 4},
    "fp8-token": {1: 4, 4: 16, 8: 16, 16: 4},
}


def _config(*, chunks: int, decode_num_stages: int) -> TuningConfig:
    return TuningConfig(
        num_topk_chunks=chunks,
        decode_num_warps=4,
        merge_num_warps=4,
        decode_num_stages=decode_num_stages,
        merge_num_stages=3,
        pdl_mode="auto",
        decode_maxnreg=None,
        merge_maxnreg=None,
    )


def _environment() -> dict[str, object]:
    return {
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
    }


def _source_hashes() -> dict[str, str]:
    """Hash exactly the code that defines this FP8 comparison boundary."""
    root = Path(__file__).resolve().parents[1]
    items = {
        "challenge_v2/prepared_stage_fp8_abba_cli.py": root
        / "challenge_v2"
        / "prepared_stage_fp8_abba_cli.py",
        "challenge_v2/run_prepared_stage5_fp8_abba_clean.sh": root
        / "challenge_v2"
        / "run_prepared_stage5_fp8_abba_clean.sh",
        "challenge_v2/prepared_tuned.py": root / "challenge_v2" / "prepared_tuned.py",
        "challenge/prepared_decode.py": root / "challenge" / "prepared_decode.py",
        "harness/data.py": root / "harness" / "data.py",
        "harness/reference.py": root / "harness" / "reference.py",
        "harness/triton_baseline.py": root / "harness" / "triton_baseline.py",
        "vllm_msa_ref/sparse_attn.py": root / "vllm_msa_ref" / "sparse_attn.py",
    }
    return {
        relative: hashlib.sha256(path.read_bytes()).hexdigest()
        for relative, path in items.items()
    }


def _stats_us(values_ms: list[float]) -> dict[str, float]:
    """Fixed nearest-rank p10/p90 and ordinary median in microseconds."""
    ordered = sorted(value * 1000.0 for value in values_ms)
    count = len(ordered)
    if not count:
        raise ValueError("cannot summarize an empty timing series")
    return {
        "p10_us": ordered[max(0, math.ceil(0.10 * count) - 1)],
        "median_us": float(statistics.median(ordered)),
        "p90_us": ordered[min(count - 1, math.ceil(0.90 * count) - 1)],
    }


def _one_event(
    runner: Callable[[], torch.Tensor],
) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    """Put one complete prepared-dispatch call between a CUDA-event pair."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    runner()
    end.record()
    return start, end


def _abba_events(
    control: Callable[[], torch.Tensor], candidate: Callable[[], torch.Tensor]
) -> dict[str, object]:
    """Warm up, then collect 101 control/candidate/candidate/control blocks."""
    for _ in range(WARMUP):
        control()
        candidate()
    torch.cuda.synchronize()

    events: list[tuple[str, str, torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(ABBA_PAIRS):
        start, end = _one_event(control)
        events.append(("prepared_stage3_control", "AB", start, end))
        start, end = _one_event(candidate)
        events.append(("prepared_stage5_candidate", "AB", start, end))
        start, end = _one_event(candidate)
        events.append(("prepared_stage5_candidate", "BA", start, end))
        start, end = _one_event(control)
        events.append(("prepared_stage3_control", "BA", start, end))
    torch.cuda.synchronize()

    samples: dict[str, dict[str, list[float]]] = {
        "prepared_stage3_control": {"AB": [], "BA": []},
        "prepared_stage5_candidate": {"AB": [], "BA": []},
    }
    for runner_name, order, start, end in events:
        samples[runner_name][order].append(float(start.elapsed_time(end)))

    timing: dict[str, object] = {
        "protocol": "warmup_each_then_101_ABBA_pairs_one_complete_prepared_dispatch_per_cuda_event_pair_one_stream",
        "warmup_each": WARMUP,
        "abba_pairs": ABBA_PAIRS,
        "samples_per_runner": SAMPLES_PER_RUNNER,
        "AB_BA_interleaved": True,
        "raw_samples_us": {
            runner_name: {
                order: [value * 1000.0 for value in values_ms]
                for order, values_ms in by_order.items()
            }
            for runner_name, by_order in samples.items()
        },
    }
    for runner_name, by_order in samples.items():
        timing[runner_name] = {
            "all": _stats_us([*by_order["AB"], *by_order["BA"]]),
            "when_launch_order_is_AB": _stats_us(by_order["AB"]),
            "when_launch_order_is_BA": _stats_us(by_order["BA"]),
        }
    return timing


def _tensor_layout(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "numel": tensor.numel(),
        "contiguous": tensor.is_contiguous(),
        "device": str(tensor.device),
        "data_ptr": tensor.data_ptr(),
    }


def _scale_description(problem: DecodeProblem, storage: str) -> dict[str, object]:
    k_scale, v_scale = problem.k_scale, problem.v_scale
    assert k_scale is not None and v_scale is not None
    assert problem.kv_cache.dtype.is_floating_point
    assert str(problem.kv_cache.dtype).startswith("torch.float8")
    if storage == "fp8-scalar":
        assert k_scale.numel() == v_scale.numel() == 1
        return {
            "storage_mode": storage,
            "abi_mode": "scalar",
            "kv_scale_mode": 1,
            "dequantization": "FP8 cache value multiplied by its scalar K or V scale",
            "k_scale": {**_tensor_layout(k_scale), "value": float(k_scale.item())},
            "v_scale": {**_tensor_layout(v_scale), "value": float(v_scale.item())},
        }
    if storage == "fp8-token":
        expected = (problem.num_kv_heads, problem.kv_cache.shape[0] * 128)
        assert tuple(k_scale.shape) == expected and tuple(v_scale.shape) == expected
        return {
            "storage_mode": storage,
            "abi_mode": "per_token_head",
            "kv_scale_mode": 2,
            "dequantization": "FP8 cache value multiplied by scale[kv_head, physical_page*128 + token]",
            "k_scale": _tensor_layout(k_scale),
            "v_scale": _tensor_layout(v_scale),
        }
    raise AssertionError(f"unexpected FP8 storage mode: {storage}")


def _verify_output(
    runner: TunedPreparedSparseDecode, output: torch.Tensor, expected: torch.Tensor
) -> dict[str, float | bool]:
    result = runner()
    if result.data_ptr() != output.data_ptr():
        raise RuntimeError("prepared dispatch did not return its caller-owned output")
    torch.cuda.synchronize()
    actual, target = output.float(), expected.float()
    diff = (actual - target).abs()
    finite = bool(torch.isfinite(actual).all().item())
    passed = finite and bool(torch.isclose(actual, target, rtol=RTOL, atol=ATOL).all().item())
    if not passed:
        raise AssertionError(
            f"independent FP32 gate failed: finite={finite}, max_abs={float(diff.max())}"
        )
    return {"finite": finite, "max_abs": float(diff.max()), "mean_abs": float(diff.mean())}


def _context(args: argparse.Namespace) -> dict[str, object]:
    storage, batch = args.storage, args.batch
    chunks = SELECTED_CHUNKS[storage][batch]
    row_seed = args.seed + batch
    problem = make_decode_problem(
        batch_size=batch,
        device="cuda",
        storage_dtype=storage,
        seed=row_seed,
        max_seq_len=args.max_seq_len,
    )
    expected = dense_sparse_attention_reference(problem)
    control_output = torch.empty_like(problem.q)
    candidate_output = torch.empty_like(problem.q)
    control_config = _config(chunks=chunks, decode_num_stages=3)
    candidate_config = _config(chunks=chunks, decode_num_stages=5)
    control = TunedPreparedSparseDecode(problem, control_output, config=control_config)
    candidate = TunedPreparedSparseDecode(problem, candidate_output, config=candidate_config)
    control_metadata, candidate_metadata = control.metadata.as_dict(), candidate.metadata.as_dict()

    # The identity checks are part of the fairness contract, not timing data.
    assert control.problem is problem and candidate.problem is problem
    assert control.problem.k_scale is candidate.problem.k_scale
    assert control.problem.v_scale is candidate.problem.v_scale
    assert control_output.data_ptr() != candidate_output.data_ptr()
    assert control.num_topk_chunks == candidate.num_topk_chunks == chunks > 1
    assert control_metadata["merge_bypassed"] is False
    assert candidate_metadata["merge_bypassed"] is False
    assert {
        key: value for key, value in control_config.as_dict().items() if key != "decode_num_stages"
    } == {
        key: value for key, value in candidate_config.as_dict().items() if key != "decode_num_stages"
    }

    scale = _scale_description(problem, storage)
    shared_input_ptrs = {
        "q": problem.q.data_ptr(),
        "kv_cache": problem.kv_cache.data_ptr(),
        "block_table": problem.block_table.data_ptr(),
        "topk_idx": problem.topk_idx.data_ptr(),
        "seq_lens": problem.seq_lens.data_ptr(),
        "k_scale": problem.k_scale.data_ptr() if problem.k_scale is not None else None,
        "v_scale": problem.v_scale.data_ptr() if problem.v_scale is not None else None,
    }
    row: dict[str, object] = {
        "storage": storage,
        "batch": batch,
        "base_seed": args.seed,
        "seed": row_seed,
        "status": "rejected",
        "problem": {
            "q_shape": list(problem.q.shape),
            "kv_cache_shape": list(problem.kv_cache.shape),
            "kv_cache_dtype": str(problem.kv_cache.dtype),
            "max_seq_len": args.max_seq_len,
            "decode_query_len": problem.decode_query_len,
            "num_kv_heads": problem.num_kv_heads,
        },
        "scale": scale,
        "selected_num_topk_chunks": chunks,
        "dispatch": {
            "boundary": "complete prepared dispatch: decode plus required merge",
            "decode_executed": True,
            "merge_required": True,
            "merge_executed": True,
            "merge_bypassed": False,
            "kernels_per_timed_runner_call": 2,
        },
        "shared_inputs": {
            "same_problem_object": True,
            "same_scale_tensor_objects": True,
            "same_problem_buffers": True,
            "shared_input_data_ptrs": shared_input_ptrs,
        },
        "outputs": {
            "caller_owned_independent": True,
            "control_data_ptr": control_output.data_ptr(),
            "candidate_data_ptr": candidate_output.data_ptr(),
            "shape": list(control_output.shape),
            "dtype": str(control_output.dtype),
        },
        "control": {"config": control_config.as_dict(), "metadata": control_metadata},
        "candidate": {"config": candidate_config.as_dict(), "metadata": candidate_metadata},
    }
    try:
        # These dispatches compile before timing and each independently meets
        # the dense FP32 selected-page causal-attention oracle.
        row["control"]["correctness"] = _verify_output(control, control_output, expected)  # type: ignore[index]
        row["candidate"]["correctness"] = _verify_output(candidate, candidate_output, expected)  # type: ignore[index]
        timing = _abba_events(control, candidate)
        control_us = float(timing["prepared_stage3_control"]["all"]["median_us"])  # type: ignore[index]
        candidate_us = float(timing["prepared_stage5_candidate"]["all"]["median_us"])  # type: ignore[index]
        speedup = control_us / candidate_us
        row.update(
            {
                "status": "pass",
                "timing": timing,
                "control_median_us": control_us,
                "candidate_median_us": candidate_us,
                "speedup_stage5_vs_stage3": speedup,
                "strict_10_percent_target_met": speedup >= 1.10,
            }
        )
    except Exception as exc:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        row.update({"error": f"{type(exc).__name__}: {exc}", "strict_10_percent_target_met": False})
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", choices=STORAGES, required=True)
    parser.add_argument("--batch", choices=BATCHES, type=int, required=True)
    parser.add_argument("--seed", type=int, choices=BASE_SEEDS, required=True)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--require-strict-10", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_seq_len != MAX_SEQ_LEN:
        parser.error(f"--max-seq-len is frozen at {MAX_SEQ_LEN}")
    if not torch.cuda.is_available():
        raise RuntimeError("FP8 prepared stage-5 AB/BA check requires CUDA")

    chunks = SELECTED_CHUNKS[args.storage][args.batch]
    control = _config(chunks=chunks, decode_num_stages=3).as_dict()
    candidate = _config(chunks=chunks, decode_num_stages=5).as_dict()
    result = _context(args)
    strict = bool(result.get("strict_10_percent_target_met", False))
    payload: dict[str, object] = {
        "schema": "c2-prepared-stage5-fp8-abba-v1",
        "environment": _environment(),
        "source_sha256": _source_hashes(),
        "scope": {
            "boundary": "complete prepared sparse-decode dispatch: decode plus required merge; not pure decode-only and not model/server end-to-end",
            "cross_mode_comparison": "prohibited: each FP8 mode is evaluated only against its same-mode stage-3 control",
        },
        "frozen_configuration": {
            "storage": args.storage,
            "batch": args.batch,
            "base_seed": args.seed,
            "selected_num_topk_chunks": chunks,
            "selected_chunks_by_storage_and_batch": SELECTED_CHUNKS,
            "control": control,
            "candidate": candidate,
            "changed_field": "decode_num_stages",
            "selection_rule": "C mapping and launch configuration were specified before this FP8 AB/BA run; not reselected from event samples",
            "strict_10_percent_policy": "reported after validation only; it does not select or retune a configuration",
        },
        "fairness_contract": {
            "same_problem_object_per_context": True,
            "same_problem_buffers_per_context": True,
            "same_scale_tensor_objects_per_context": True,
            "caller_owned_independent_outputs": True,
            "persistent_workspace_outside_timing": True,
            "same_selected_chunks_per_context": True,
            "full_prepared_decode_and_merge_per_timed_call": True,
            "oracle": "independent harness.reference dense FP32 selected-page causal attention",
            "tolerance": {"rtol": RTOL, "atol": ATOL},
            "single_complete_dispatch_per_cuda_event": True,
            "AB_BA_interleaved": True,
            "raw_event_samples_recorded": True,
        },
        "results": [result],
        "all_contexts_strict_10_percent": strict,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if result["status"] != "pass":
        # rc=3 is reserved exclusively for a valid observation that missed the
        # policy target, so an exception/correctness rejection remains fatal.
        return 70
    return 3 if args.require_strict_10 and not strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
