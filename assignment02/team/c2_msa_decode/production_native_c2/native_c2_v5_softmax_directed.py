#!/usr/bin/env python3
"""Directed liveness/correctness tests for warp-parallel-softmax C2 v5.

This is intentionally a direct-AOT test.  It loads exactly the reviewed
stable extension and calls ``torch.ops._C.native_c2_msa_decode`` without any
vLLM import, fallback, or monkeypatch.  The normal eight-seed stress harness
remains the primary broad oracle gate; this companion isolates each of the
four selected-page groups so a missing producer, wrong DSM source rank, or
incorrect four-way LSE merge cannot hide in random data.  It also adds
all-valid online-rescale and mixed causal-tail cases for the two 16-lane
softmax subgroups introduced by v5.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--base-harness", required=True, type=Path)
    parser.add_argument("--sync-repeats", type=int, default=4)
    parser.add_argument("--max-iteration-seconds", type=float, default=15.0)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-3)
    return parser.parse_args()


def _load_base_harness(path: Path) -> ModuleType:
    if not path.is_absolute():
        raise ValueError("--base-harness must be absolute")
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("native_c2_directed_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "_BATCH", "_QUERY_HEADS", "_KV_HEADS", "_HEAD_DIM", "_PAGE_SIZE",
        "_TOPK", "_oracle", "_require_registered_operator", "_validate_args",
    )
    for name in required:
        if not hasattr(module, name):
            raise RuntimeError(f"base harness lacks required helper: {name}")
    return module


def _contract(base: ModuleType, args: argparse.Namespace) -> Any:
    return base._validate_args(
        SimpleNamespace(
            num_physical_pages=64,
            max_logical_pages=32,
            scale=1.0 / math.sqrt(base._HEAD_DIM),
            q_scale=0.25,
            k_scale=0.25,
            v_scale=0.5,
            atol=args.atol,
            rtol=args.rtol,
        )
    )


def _make_rank_directed_inputs(
    base: ModuleType,
    contract: Any,
    target_rank: int,
    *,
    all_invalid: bool,
) -> tuple[torch.Tensor, ...]:
    """Give only one contiguous four-page producer group a high logit.

    Query values are +8 and the four pages belonging to ``target_rank`` use
    K=+8; every other selected page uses K=-8.  With the production scalar
    scales the logit separation is about 90.5, making the independent FP64
    reference essentially the target group's distinct V constant.  The group
    layout is the kernel contract: selected pages [4r, 4r+3] belong to rank r.
    """
    if target_rank not in range(4):
        raise ValueError("target_rank must be in [0, 3]")
    device = torch.device("cuda")
    batch = base._BATCH
    q_heads = base._QUERY_HEADS
    kv_heads = base._KV_HEADS
    head_dim = base._HEAD_DIM
    page_size = base._PAGE_SIZE
    topk_count = base._TOPK

    query = torch.full(
        (batch, q_heads, head_dim), 8.0, device=device, dtype=torch.float32
    ).to(torch.float8_e4m3fn).contiguous()
    cache_f32 = torch.zeros(
        (contract.num_physical_pages, kv_heads, page_size, 2 * head_dim),
        device=device,
        dtype=torch.float32,
    )
    for logical_page in range(topk_count):
        producer_rank = logical_page // 4
        key_value = 8.0 if producer_rank == target_rank else -8.0
        # After v_scale=0.5, the four producer groups carry 1,2,3,4.  This
        # makes an erroneous DSM source/merge rank observable in every dim.
        value_value = float(2 * (producer_rank + 1))
        cache_f32[logical_page, :, :, :head_dim].fill_(key_value)
        cache_f32[logical_page, :, :, head_dim:].fill_(value_value)
    kv_cache = cache_f32.to(torch.float8_e4m3fn).contiguous()

    block_row = torch.arange(
        contract.max_logical_pages, device=device, dtype=torch.int32
    )
    block_table = block_row.expand(batch, -1).contiguous()
    selected = torch.arange(topk_count, device=device, dtype=torch.int32)
    topk = selected.view(1, 1, -1).expand(batch, kv_heads, -1).contiguous()
    seq_value = 0 if all_invalid else topk_count * page_size
    seq_lens = torch.full((batch,), seq_value, device=device, dtype=torch.int32)
    return query, kv_cache, topk, block_table, seq_lens


def _make_softmax_directed_inputs(
    base: ModuleType,
    contract: Any,
    sequence_length: int,
) -> tuple[torch.Tensor, ...]:
    """Build deterministic per-head/per-token inputs for subgroup softmax.

    The eight-tile key pattern repeatedly raises and lowers the online maximum,
    while per-token values make a lost/duplicated weight observable.  A
    sequence length of 37 leaves five valid lanes in the third 16-token tile,
    explicitly exercising mixed valid/invalid lanes inside each subgroup.
    """
    full_length = base._TOPK * base._PAGE_SIZE
    if sequence_length not in (37, full_length):
        raise ValueError("sequence_length must be 37 or the full selected span")
    device = torch.device("cuda")
    batch = base._BATCH
    q_heads = base._QUERY_HEADS
    kv_heads = base._KV_HEADS
    head_dim = base._HEAD_DIM
    page_size = base._PAGE_SIZE
    topk_count = base._TOPK

    head_values = (0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0)
    query_f32 = torch.empty((batch, q_heads, head_dim), dtype=torch.float32)
    for head in range(q_heads):
        query_f32[:, head, :].fill_(head_values[head % len(head_values)])
    query = query_f32.to(device).to(torch.float8_e4m3fn).contiguous()

    cache_f32 = torch.zeros(
        (contract.num_physical_pages, kv_heads, page_size, 2 * head_dim),
        dtype=torch.float32,
    )
    key_pattern = (-4.0, -1.0, 2.0, 4.0, 1.0, -2.0, 3.0, 0.0)
    kv_bias = (torch.arange(kv_heads, dtype=torch.float32) - 1.5) * 0.125
    dim_pattern = (torch.arange(head_dim, dtype=torch.float32) % 8 - 3.5) * 0.125
    # Keep the token/head/dim code comfortably representable in FP8 while
    # making the unavoidable final BF16 quantization smaller than the frozen
    # 1e-4/1e-3 oracle gate.  This does not scale K, so the repeated online-max
    # rescale pattern and mixed-lane softmax control flow are unchanged.
    directed_value_scale = 1.0 / 32.0
    for logical_page in range(topk_count):
        for token in range(page_size):
            tile = token // 16
            token_in_tile = token % 16
            global_tile = logical_page * (page_size // 16) + tile
            key_value = (
                key_pattern[global_tile % len(key_pattern)]
                + (token_in_tile % 4 - 1.5) * 0.25
            )
            cache_f32[logical_page, :, token, :head_dim] = (
                key_value + kv_bias[:, None]
            )
            value_scalar = (
                (logical_page * 11 + tile * 5 + token_in_tile * 3) % 29 - 14
            ) / 4.0
            cache_f32[logical_page, :, token, head_dim:] = (
                value_scalar + kv_bias[:, None] + dim_pattern[None, :]
            ) * directed_value_scale
    kv_cache = cache_f32.to(device).to(torch.float8_e4m3fn).contiguous()

    block_row = torch.arange(
        contract.max_logical_pages, device=device, dtype=torch.int32
    )
    block_table = block_row.expand(batch, -1).contiguous()
    selected = torch.arange(topk_count, device=device, dtype=torch.int32)
    topk = selected.view(1, 1, -1).expand(batch, kv_heads, -1).contiguous()
    seq_lens = torch.full(
        (batch,), sequence_length, device=device, dtype=torch.int32
    )
    return query, kv_cache, topk, block_table, seq_lens


def _call_sync_repeatedly(
    base: ModuleType,
    op: Any,
    inputs: tuple[torch.Tensor, ...],
    contract: Any,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, object]]:
    shape = (base._BATCH, base._QUERY_HEADS, base._HEAD_DIM)
    output = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    pointer_before = output.data_ptr()
    outputs: list[torch.Tensor] = []
    elapsed_seconds: list[float] = []
    returns_none = True
    pointer_unchanged = True
    for _ in range(args.sync_repeats):
        output.fill_(float("nan"))
        started = time.monotonic()
        returned = op(
            output, *inputs, contract.scale, contract.q_scale,
            contract.k_scale, contract.v_scale,
        )
        # This is deliberately per invocation rather than one final sync: a
        # broken mbarrier/cluster protocol therefore cannot be masked by later
        # launches.  The Slurm wrapper adds an outer hard timeout for a true
        # device-side deadlock that cannot return control to this process.
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        elapsed_seconds.append(elapsed)
        if elapsed > args.max_iteration_seconds:
            raise RuntimeError(
                f"synchronized iteration exceeded liveness budget: {elapsed:.3f}s"
            )
        pointer_unchanged = pointer_unchanged and output.data_ptr() == pointer_before
        returns_none = returns_none and returned is None
        outputs.append(output.clone())
    bitwise_repeatable = all(torch.equal(outputs[0], item) for item in outputs[1:])
    return outputs[0], {
        "pointer_before": pointer_before,
        "pointer_after": output.data_ptr(),
        "pointer_unchanged": pointer_unchanged,
        "return_is_none": returns_none,
        "bitwise_repeatable": bitwise_repeatable,
        "elapsed_seconds": elapsed_seconds,
        "max_iteration_seconds": args.max_iteration_seconds,
    }


@torch.no_grad()
def _rank_directed_case(
    base: ModuleType, op: Any, contract: Any, args: argparse.Namespace, rank: int
) -> dict[str, object]:
    inputs = _make_rank_directed_inputs(base, contract, rank, all_invalid=False)
    output, observation = _call_sync_repeatedly(base, op, inputs, contract, args)
    reference = base._oracle(*inputs, contract, torch.float64)
    actual = output.to(torch.float64)
    difference = (actual - reference).abs()
    expected_value = float(rank + 1)
    expected = torch.full_like(actual, expected_value)
    expected_error = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all().item())
    oracle_allclose = bool(torch.allclose(actual, reference, atol=args.atol, rtol=args.rtol))
    expected_allclose = bool(torch.allclose(actual, expected, atol=args.atol, rtol=args.rtol))
    all_gates = bool(
        finite and oracle_allclose and expected_allclose
        and observation["pointer_unchanged"] and observation["return_is_none"]
        and observation["bitwise_repeatable"]
    )
    return {
        "all_gates_pass": all_gates,
        "producer_rank": rank,
        "selected_page_indices": [4 * rank + offset for offset in range(4)],
        "expected_output_value": expected_value,
        "caller_output": {
            "pointer_before": observation["pointer_before"],
            "pointer_after": observation["pointer_after"],
            "pointer_unchanged": observation["pointer_unchanged"],
            "return_is_none": observation["return_is_none"],
        },
        "oracle": {
            "allclose": oracle_allclose,
            "finite_output": finite,
            "max_abs": float(difference.max().item()),
            "mean_abs": float(difference.mean().item()),
        },
        "rank_expected_value": {
            "allclose": expected_allclose,
            "max_abs": float(expected_error.max().item()),
            "mean_abs": float(expected_error.mean().item()),
        },
        "stability": {"bitwise_repeatable": observation["bitwise_repeatable"]},
        "sync_watchdog": {
            "elapsed_seconds": observation["elapsed_seconds"],
            "max_iteration_seconds": observation["max_iteration_seconds"],
        },
    }


@torch.no_grad()
def _softmax_directed_case(
    base: ModuleType,
    op: Any,
    contract: Any,
    args: argparse.Namespace,
    *,
    label: str,
    sequence_length: int,
) -> dict[str, object]:
    inputs = _make_softmax_directed_inputs(base, contract, sequence_length)
    output, observation = _call_sync_repeatedly(base, op, inputs, contract, args)
    reference = base._oracle(*inputs, contract, torch.float64)
    actual = output.to(torch.float64)
    difference = (actual - reference).abs()
    finite = bool(torch.isfinite(actual).all().item())
    oracle_allclose = bool(
        torch.allclose(actual, reference, atol=args.atol, rtol=args.rtol)
    )
    all_gates = bool(
        finite and oracle_allclose and observation["pointer_unchanged"]
        and observation["return_is_none"] and observation["bitwise_repeatable"]
    )
    return {
        "all_gates_pass": all_gates,
        "case": label,
        "sequence_length": sequence_length,
        "mixed_causal_subgroup": sequence_length % 16 != 0,
        "caller_output": {
            "pointer_before": observation["pointer_before"],
            "pointer_after": observation["pointer_after"],
            "pointer_unchanged": observation["pointer_unchanged"],
            "return_is_none": observation["return_is_none"],
        },
        "oracle": {
            "allclose": oracle_allclose,
            "finite_output": finite,
            "max_abs": float(difference.max().item()),
            "mean_abs": float(difference.mean().item()),
        },
        "stability": {"bitwise_repeatable": observation["bitwise_repeatable"]},
        "sync_watchdog": {
            "elapsed_seconds": observation["elapsed_seconds"],
            "max_iteration_seconds": observation["max_iteration_seconds"],
        },
    }


@torch.no_grad()
def _all_invalid_case(
    base: ModuleType, op: Any, contract: Any, args: argparse.Namespace
) -> dict[str, object]:
    # Target rank is immaterial when every token is masked; use rank 0 only to
    # retain the same packed layout/scalars as the rank-directed cases.
    inputs = _make_rank_directed_inputs(base, contract, 0, all_invalid=True)
    output, observation = _call_sync_repeatedly(base, op, inputs, contract, args)
    actual = output.to(torch.float32)
    finite = bool(torch.isfinite(actual).all().item())
    all_zero = bool(torch.count_nonzero(actual).item() == 0)
    all_gates = bool(
        finite and all_zero and observation["pointer_unchanged"]
        and observation["return_is_none"] and observation["bitwise_repeatable"]
    )
    return {
        "all_gates_pass": all_gates,
        "finite_output": finite,
        "all_zero": all_zero,
        "max_abs": float(actual.abs().max().item()),
        "caller_output": {
            "pointer_before": observation["pointer_before"],
            "pointer_after": observation["pointer_after"],
            "pointer_unchanged": observation["pointer_unchanged"],
            "return_is_none": observation["return_is_none"],
        },
        "stability": {"bitwise_repeatable": observation["bitwise_repeatable"]},
        "sync_watchdog": {
            "elapsed_seconds": observation["elapsed_seconds"],
            "max_iteration_seconds": observation["max_iteration_seconds"],
        },
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
    base = _load_base_harness(args.base_harness)
    contract = _contract(base, args)
    operator_library = base._require_registered_operator(args.library)
    op = torch.ops._C.native_c2_msa_decode
    rank_directed = [_rank_directed_case(base, op, contract, args, rank) for rank in range(4)]
    all_invalid = _all_invalid_case(base, op, contract, args)
    softmax_directed = [
        _softmax_directed_case(
            base, op, contract, args,
            label="all-valid-online-rescale",
            sequence_length=base._TOPK * base._PAGE_SIZE,
        ),
        _softmax_directed_case(
            base, op, contract, args,
            label="mixed-causal-tail-37",
            sequence_length=37,
        ),
    ]
    return {
        "all_gates_pass": all(
            bool(case["all_gates_pass"]) for case in rank_directed
        ) and bool(all_invalid["all_gates_pass"]) and all(
            bool(case["all_gates_pass"]) for case in softmax_directed
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
        "no_monkeypatch": True,
        "operator_library": operator_library,
        "rank_directed": rank_directed,
        "rank_directed_case_count": len(rank_directed),
        "softmax_directed": softmax_directed,
        "softmax_directed_case_count": len(softmax_directed),
        "schema": "c2-native-c2-v5-softmax-directed-v2",
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
            "schema": "c2-native-c2-v5-softmax-directed-v2",
            "traceback": traceback.format_exc(),
        }
        print(_as_json(result))
        return 1
    print(_as_json(result))
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
