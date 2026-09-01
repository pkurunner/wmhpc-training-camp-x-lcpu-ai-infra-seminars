#!/usr/bin/env python3
"""Directed correctness gate for the v6 register-resident PV numerator.

The v6 change replaces the producer CTA's shared ``numerator[head][dim]``
array with eight register accumulators per lane.  This test therefore keeps
the accepted v5 directed softmax/rank fixtures, and adds an adversarial V
coding fixture.  The latter selects a different target token for each GQA
head and gives every dimension a distinct finite E4M3 code; each head uses a
different cyclic order.  A head, warp, lane-parity, or dimension permutation
then changes an exact expected BF16 output location rather than merely
changing an aggregate statistic.

This is deliberately a direct-AOT test.  It loads exactly one reviewed DSO
and calls ``torch.ops._C.native_c2_msa_decode``.  No vLLM model import,
fallback path, or dispatch monkeypatch is involved.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


_SCHEMA = "c2-native-c2-v6-register-numerator-directed-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--base-harness", required=True, type=Path)
    parser.add_argument("--v5-directed-harness", required=True, type=Path)
    parser.add_argument("--sync-repeats", type=int, default=4)
    parser.add_argument("--max-iteration-seconds", type=float, default=15.0)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-3)
    return parser.parse_args()


def _load_module(path: Path, module_name: str) -> ModuleType:
    if not path.is_absolute():
        raise ValueError(f"{module_name} path must be absolute: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_v5_helpers(v5: ModuleType) -> None:
    required = (
        "_call_sync_repeatedly",
        "_contract",
        "_load_base_harness",
        "_make_rank_directed_inputs",
        "_make_softmax_directed_inputs",
    )
    for name in required:
        if not hasattr(v5, name):
            raise RuntimeError(f"v5 directed harness lacks required helper: {name}")


def _oracle_metrics(
    base: ModuleType,
    output: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    contract: Any,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Check the caller BF16 output independently against FP64 and FP32."""
    reference_fp64 = base._oracle(*inputs, contract, torch.float64)
    reference_fp32 = base._oracle(*inputs, contract, torch.float32)
    actual_fp64 = output.to(torch.float64)
    actual_fp32 = output.to(torch.float32)
    difference_fp64 = (actual_fp64 - reference_fp64).abs()
    difference_fp32 = (actual_fp32 - reference_fp32).abs()
    finite = bool(torch.isfinite(actual_fp64).all().item())
    fp64_allclose = bool(
        torch.allclose(actual_fp64, reference_fp64, atol=args.atol, rtol=args.rtol)
    )
    fp32_allclose = bool(
        torch.allclose(actual_fp32, reference_fp32, atol=args.atol, rtol=args.rtol)
    )
    reference_agrees = bool(
        torch.allclose(
            reference_fp64,
            reference_fp32.to(torch.float64),
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    return {
        "finite_output": finite,
        "fp64": {
            "allclose": fp64_allclose,
            "max_abs": float(difference_fp64.max().item()),
            "mean_abs": float(difference_fp64.mean().item()),
        },
        "fp32": {
            "allclose": fp32_allclose,
            "max_abs": float(difference_fp32.max().item()),
            "mean_abs": float(difference_fp32.mean().item()),
        },
        "reference_fp64_fp32_agree": reference_agrees,
    }


def _zero_oracle_metrics(output: torch.Tensor, args: argparse.Namespace) -> dict[str, object]:
    """All-invalid has a deliberately defined zero result, not softmax(NaN)."""
    actual_fp64 = output.to(torch.float64)
    actual_fp32 = output.to(torch.float32)
    zero_fp64 = torch.zeros_like(actual_fp64)
    zero_fp32 = torch.zeros_like(actual_fp32)
    difference_fp64 = actual_fp64.abs()
    difference_fp32 = actual_fp32.abs()
    finite = bool(torch.isfinite(actual_fp64).all().item())
    return {
        "finite_output": finite,
        "fp64": {
            "allclose": bool(torch.allclose(actual_fp64, zero_fp64, atol=args.atol, rtol=args.rtol)),
            "max_abs": float(difference_fp64.max().item()),
            "mean_abs": float(difference_fp64.mean().item()),
        },
        "fp32": {
            "allclose": bool(torch.allclose(actual_fp32, zero_fp32, atol=args.atol, rtol=args.rtol)),
            "max_abs": float(difference_fp32.max().item()),
            "mean_abs": float(difference_fp32.mean().item()),
        },
        "reference_fp64_fp32_agree": True,
    }


def _common_gates(observation: dict[str, object], oracle: dict[str, object]) -> bool:
    return bool(
        oracle["finite_output"]
        and oracle["fp64"]["allclose"]
        and oracle["fp32"]["allclose"]
        and oracle["reference_fp64_fp32_agree"]
        and observation["pointer_unchanged"]
        and observation["return_is_none"]
        and observation["bitwise_repeatable"]
    )


def _case_record(
    base: ModuleType,
    v5: ModuleType,
    op: Any,
    inputs: tuple[torch.Tensor, ...],
    contract: Any,
    args: argparse.Namespace,
    *,
    label: str,
    zero_oracle: bool = False,
) -> tuple[torch.Tensor, dict[str, object]]:
    output, observation = v5._call_sync_repeatedly(base, op, inputs, contract, args)
    oracle = (
        _zero_oracle_metrics(output, args)
        if zero_oracle
        else _oracle_metrics(base, output, inputs, contract, args)
    )
    return output, {
        "all_gates_pass": _common_gates(observation, oracle),
        "case": label,
        "caller_output": {
            "pointer_before": observation["pointer_before"],
            "pointer_after": observation["pointer_after"],
            "pointer_unchanged": observation["pointer_unchanged"],
            "return_is_none": observation["return_is_none"],
        },
        "oracle": oracle,
        "stability": {"bitwise_repeatable": observation["bitwise_repeatable"]},
        "sync_watchdog": {
            "elapsed_seconds": observation["elapsed_seconds"],
            "max_iteration_seconds": observation["max_iteration_seconds"],
        },
    }


def _make_head_dim_encoding_inputs(
    base: ModuleType, contract: Any
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, dict[str, object]]:
    """Make target-V codes that expose any head/dimension remapping.

    There are fewer finite E4M3 scalars than all 64 * 128 output positions,
    so scalar-global uniqueness is impossible.  Instead every dimension uses
    one of 128 distinct finite FP8 values and each of the 64 heads receives a
    different cyclic ordering.  The expected matrix is therefore a unique
    head signature at every output location, while remaining exactly
    representable through FP8 -> BF16 with ``v_scale == 1/2``.
    """
    if base._QUERY_HEADS != base._KV_HEADS * 16:
        raise RuntimeError("head/dimension encoding assumes 16-way GQA")
    if base._TOPK != 16 or base._PAGE_SIZE != 128 or base._HEAD_DIM != 128:
        raise RuntimeError("head/dimension encoding assumes the production C2 shape")
    device = torch.device("cuda")
    gqa = base._QUERY_HEADS // base._KV_HEADS
    raw = torch.arange(256, dtype=torch.uint8)
    finite_fp8 = raw.view(torch.float8_e4m3fn).to(torch.float32)
    candidates = finite_fp8[
        torch.isfinite(finite_fp8)
        & (finite_fp8.abs() >= 0.03125)
        & (finite_fp8.abs() <= 16.0)
    ]
    if candidates.numel() < base._HEAD_DIM:
        raise RuntimeError("insufficient finite E4M3 dimension codes")
    dim_codes = candidates[: base._HEAD_DIM].contiguous()
    if torch.unique(dim_codes).numel() != base._HEAD_DIM:
        raise RuntimeError("dimension code selection is not unique")

    query_f32 = torch.empty(
        (base._BATCH, base._QUERY_HEADS, base._HEAD_DIM), dtype=torch.float32
    )
    cache_f32 = torch.zeros(
        (contract.num_physical_pages, base._KV_HEADS, base._PAGE_SIZE, 2 * base._HEAD_DIM),
        dtype=torch.float32,
    )
    expected_cpu = torch.empty(
        (base._BATCH, base._QUERY_HEADS, base._HEAD_DIM), dtype=torch.bfloat16
    )
    head_offsets: list[int] = []
    target_tokens: list[dict[str, int]] = []
    for kv_head in range(base._KV_HEADS):
        for local_head in range(gqa):
            global_head = kv_head * gqa + local_head
            # Rows of the 16x128 Walsh matrix are orthogonal.  A target key
            # has score ~45.25 while every nonmatching encoded key has score
            # 0, so the selected V vector reaches output unchanged.
            signs = torch.tensor(
                [
                    8.0
                    if ((local_head & dim).bit_count() & 1) == 0
                    else -8.0
                    for dim in range(base._HEAD_DIM)
                ],
                dtype=torch.float32,
            )
            query_f32[:, global_head, :] = signs
            target_page = local_head
            target_token = local_head
            cache_f32[target_page, kv_head, target_token, : base._HEAD_DIM] = signs
            offset = (37 * global_head) % base._HEAD_DIM
            head_offsets.append(offset)
            coded_value = torch.roll(dim_codes, shifts=offset)
            cache_f32[
                target_page, kv_head, target_token, base._HEAD_DIM :
            ] = coded_value
            expected_cpu[:, global_head, :] = (
                coded_value.mul(contract.v_scale).to(torch.bfloat16)
            )
            target_tokens.append(
                {
                    "global_head": global_head,
                    "kv_head": kv_head,
                    "local_head": local_head,
                    "logical_page": target_page,
                    "token": target_token,
                }
            )
    if len(set(head_offsets)) != base._QUERY_HEADS:
        raise RuntimeError("head signature offsets are not unique")

    query = query_f32.to(device).to(torch.float8_e4m3fn).contiguous()
    kv_cache = cache_f32.to(device).to(torch.float8_e4m3fn).contiguous()
    block_row = torch.arange(contract.max_logical_pages, device=device, dtype=torch.int32)
    block_table = block_row.expand(base._BATCH, -1).contiguous()
    selected = torch.arange(base._TOPK, device=device, dtype=torch.int32)
    topk = selected.view(1, 1, -1).expand(base._BATCH, base._KV_HEADS, -1).contiguous()
    seq_lens = torch.full(
        (base._BATCH,), base._TOPK * base._PAGE_SIZE, device=device, dtype=torch.int32
    )
    return (
        (query, kv_cache, topk, block_table, seq_lens),
        expected_cpu.to(device),
        {
            "dim_code_count": int(torch.unique(dim_codes).numel()),
            "head_signature_count": len(set(head_offsets)),
            "head_signature_offsets": head_offsets,
            "target_tokens": target_tokens,
            "target_score_vs_orthogonal_score": 45.254833995939045,
        },
    )


def _head_dim_encoding_case(
    base: ModuleType, v5: ModuleType, op: Any, contract: Any, args: argparse.Namespace
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    inputs, expected, encoding = _make_head_dim_encoding_inputs(base, contract)
    output, result = _case_record(
        base, v5, op, inputs, contract, args, label="head-dim-v-encoding"
    )
    difference = (output.to(torch.float32) - expected.to(torch.float32)).abs()
    exact_location_match = bool(torch.equal(output, expected))
    result["all_gates_pass"] = bool(result["all_gates_pass"] and exact_location_match)
    result["head_dim_encoding"] = {
        **encoding,
        "exact_location_match": exact_location_match,
        "max_abs_to_exact_expected": float(difference.max().item()),
        "nonzero_mismatch_count": int(torch.count_nonzero(difference).item()),
    }
    return inputs, result


def _rank_directed_case(
    base: ModuleType,
    v5: ModuleType,
    op: Any,
    contract: Any,
    args: argparse.Namespace,
    rank: int,
) -> dict[str, object]:
    inputs = v5._make_rank_directed_inputs(base, contract, rank, all_invalid=False)
    output, result = _case_record(
        base, v5, op, inputs, contract, args, label=f"producer-rank-{rank}"
    )
    expected = torch.full_like(output.to(torch.float32), float(rank + 1))
    expected_difference = (output.to(torch.float32) - expected).abs()
    rank_value_match = bool(torch.allclose(output.to(torch.float32), expected, atol=args.atol, rtol=args.rtol))
    result["all_gates_pass"] = bool(result["all_gates_pass"] and rank_value_match)
    result["producer_rank"] = rank
    result["selected_page_indices"] = list(range(4 * rank, 4 * rank + 4))
    result["rank_expected_value"] = {
        "allclose": rank_value_match,
        "max_abs": float(expected_difference.max().item()),
        "mean_abs": float(expected_difference.mean().item()),
    }
    return result


def _softmax_directed_case(
    base: ModuleType,
    v5: ModuleType,
    op: Any,
    contract: Any,
    args: argparse.Namespace,
    label: str,
    sequence_length: int,
) -> dict[str, object]:
    inputs = v5._make_softmax_directed_inputs(base, contract, sequence_length)
    _, result = _case_record(base, v5, op, inputs, contract, args, label=label)
    result["sequence_length"] = sequence_length
    result["mixed_causal_subgroup"] = sequence_length % 16 != 0
    return result


def _all_invalid_case(
    base: ModuleType, v5: ModuleType, op: Any, contract: Any, args: argparse.Namespace
) -> dict[str, object]:
    inputs = v5._make_rank_directed_inputs(base, contract, 0, all_invalid=True)
    _, result = _case_record(
        base, v5, op, inputs, contract, args,
        label="all-invalid", zero_oracle=True,
    )
    result["all_zero"] = bool(
        result["oracle"]["fp64"]["max_abs"] == 0.0
        and result["oracle"]["fp32"]["max_abs"] == 0.0
    )
    result["all_gates_pass"] = bool(result["all_gates_pass"] and result["all_zero"])
    return result


def _profile_native_dispatch(
    base: ModuleType, op: Any, inputs: tuple[torch.Tensor, ...], contract: Any
) -> dict[str, object]:
    """Require both dispatcher and CUDA kernel names in a fresh profiler trace."""
    output = torch.full(
        (base._BATCH, base._QUERY_HEADS, base._HEAD_DIM),
        float("nan"), device="cuda", dtype=torch.bfloat16,
    )
    pointer_before = output.data_ptr()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as profiler:
        returned = op(
            output, *inputs, contract.scale, contract.q_scale,
            contract.k_scale, contract.v_scale,
        )
        torch.cuda.synchronize()
    events = list(profiler.key_averages())
    dispatcher_events = [
        event for event in events if str(event.key) == "_C::native_c2_msa_decode"
    ]
    kernel_events = [
        event
        for event in events
        if "native_c2_msa_decode_kernel" in str(event.key).lower()
    ]
    dispatcher_event_records = [
        {"key": str(event.key), "count": int(getattr(event, "count", 0))}
        for event in dispatcher_events
    ]
    kernel_event_records = [
        {"key": str(event.key), "count": int(getattr(event, "count", 0))}
        for event in kernel_events
    ]
    exactly_one_dispatcher = (
        len(dispatcher_event_records) == 1
        and dispatcher_event_records[0]["count"] == 1
    )
    exactly_one_kernel = (
        len(kernel_event_records) == 1 and kernel_event_records[0]["count"] == 1
    )
    return {
        "dispatcher_events": dispatcher_event_records,
        "kernel_events": kernel_event_records,
        "native_dispatcher_hit": exactly_one_dispatcher,
        "native_kernel_hit": exactly_one_kernel,
        "exactly_one_dispatcher_event": exactly_one_dispatcher,
        "exactly_one_native_kernel_event": exactly_one_kernel,
        "caller_output_pointer_unchanged": output.data_ptr() == pointer_before,
        "return_is_none": returned is None,
        "finite_output": bool(torch.isfinite(output.to(torch.float32)).all().item()),
    }


@torch.no_grad()
def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.sync_repeats < 3:
        raise ValueError("--sync-repeats must be at least three")
    if not math.isfinite(args.max_iteration_seconds) or not (0.0 < args.max_iteration_seconds <= 60.0):
        raise ValueError("--max-iteration-seconds must be finite in (0, 60]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 3):
        raise RuntimeError(f"native C2 requires B300 capability (10, 3), got {capability}")
    v5 = _load_module(args.v5_directed_harness, "native_c2_v6_directed_v5_base")
    _require_v5_helpers(v5)
    base = v5._load_base_harness(args.base_harness)
    contract = v5._contract(base, args)
    operator_library = base._require_registered_operator(args.library)
    op = torch.ops._C.native_c2_msa_decode

    encoding_inputs, head_dim_encoding = _head_dim_encoding_case(base, v5, op, contract, args)
    rank_directed = [
        _rank_directed_case(base, v5, op, contract, args, rank) for rank in range(4)
    ]
    online_rescale = _softmax_directed_case(
        base, v5, op, contract, args,
        label="all-valid-online-rescale", sequence_length=base._TOPK * base._PAGE_SIZE,
    )
    mixed_tail = _softmax_directed_case(
        base, v5, op, contract, args,
        label="mixed-causal-tail-37", sequence_length=37,
    )
    all_invalid = _all_invalid_case(base, v5, op, contract, args)
    profiler = _profile_native_dispatch(base, op, encoding_inputs, contract)
    profiler_gates = bool(
        profiler["native_dispatcher_hit"]
        and profiler["native_kernel_hit"]
        and profiler["caller_output_pointer_unchanged"]
        and profiler["return_is_none"]
        and profiler["finite_output"]
    )
    return {
        "all_gates_pass": bool(
            head_dim_encoding["all_gates_pass"]
            and all(case["all_gates_pass"] for case in rank_directed)
            and online_rescale["all_gates_pass"]
            and mixed_tail["all_gates_pass"]
            and all_invalid["all_gates_pass"]
            and profiler_gates
        ),
        "all_invalid": all_invalid,
        "contract": {
            "batch": base._BATCH,
            "head_dim": base._HEAD_DIM,
            "kv_heads": base._KV_HEADS,
            "page_size": base._PAGE_SIZE,
            "q_heads": base._QUERY_HEADS,
            "selected_pages": base._TOPK,
            "producer_ctas": 4,
            "pages_per_producer": 4,
        },
        "device": {"capability": list(capability), "name": torch.cuda.get_device_name()},
        "dispatch": "direct_torch.ops._C.native_c2_msa_decode",
        "head_dim_encoding": head_dim_encoding,
        "no_monkeypatch": True,
        "online_rescale": online_rescale,
        "operator_library": operator_library,
        "profiler": profiler,
        "rank_directed": rank_directed,
        "rank_directed_case_count": len(rank_directed),
        "schema": _SCHEMA,
        "softmax_directed": [online_rescale, mixed_tail],
        "softmax_directed_case_count": 2,
        "torch": {"cuda": torch.version.cuda, "version": torch.__version__},
    }


def _as_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    try:
        result = _run(_parse_args())
    except Exception as error:
        result = {
            "all_gates_pass": False,
            "error": f"{type(error).__name__}: {error}",
            "schema": _SCHEMA,
            "traceback": traceback.format_exc(),
        }
        print(_as_json(result))
        return 1
    print(_as_json(result))
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
