#!/usr/bin/env python3
"""Directed correctness gate for v16 intra-page K-chunk raw-FP8 lookahead.

The candidate changes only how K tiles are prefetched.  Its main new
correctness risks are stale or shifted K chunks at a 16-element transition,
an omitted final chunk, and accidentally crossing a metadata gap.  This
harness first executes the complete frozen v6 directed suite unchanged, then
adds two rank-0 selected-page gap cases and three intra-page K-chunk cases:

* ``[0, 1, -1, 2]`` verifies a logically invalid selected entry is skipped;
* ``[0, 1, 2, 3]`` with logical page 2 mapping to physical ``-1`` verifies a
  physically invalid mapping is skipped while logical page 3 -> physical 2
  remains reachable after the gap.

The extra oracle deliberately filters metadata before any cache indexing.
In particular, Python's negative-index convention is never permitted to
turn an invalid logical or physical page into a valid cache page.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # Lets the CPU-only metadata self-test run anywhere.
    torch = None  # type: ignore[assignment]


_SCHEMA = "c2-native-c2-v16-k-chunk-lookahead-directed-v1"
_POST_GAP_PHYSICAL_PAGE = 2
_EXPECTED_BF16_VALUE = 3.0
_K_CHUNK_WIDTH = 16
# Every code is exactly representable in E4M3 and at most 64 (well below its
# finite range). Distinct chunk weights make a one-chunk cyclic K shift
# observably different from the original order.
_K_CHUNK_QUERY_CODES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
_K_CHUNK_SENSITIVITY_MINIMUM = 1.0e-1


def _k_chunk_query_codes() -> tuple[float, ...]:
    """Return the exact raw FP8 codes shared by CPU and GPU K-chunk fixtures."""
    if len(_K_CHUNK_QUERY_CODES) != 8:
        raise RuntimeError("K-chunk query fixture must cover exactly eight chunks")
    return _K_CHUNK_QUERY_CODES


def _make_k_chunk_coded_query(base: ModuleType, device: Any) -> torch.Tensor:
    """Materialize the CPU-checked chunk-code vector as the GPU query tensor."""
    query_codes = _k_chunk_query_codes()
    if base._HEAD_DIM != len(query_codes) * _K_CHUNK_WIDTH:
        raise RuntimeError("K-chunk query fixture has an unexpected head dimension")
    query_f32 = torch.empty(
        (base._BATCH, base._QUERY_HEADS, base._HEAD_DIM),
        device=device,
        dtype=torch.float32,
    )
    for chunk, raw_query in enumerate(query_codes):
        query_f32[..., chunk * _K_CHUNK_WIDTH : (chunk + 1) * _K_CHUNK_WIDTH].fill_(
            raw_query
        )
    return query_f32.to(torch.float8_e4m3fn).contiguous()


def _no_grad(function: Any) -> Any:
    """Keep the metadata-only self-test importable without a PyTorch install."""
    return function if torch is None else torch.no_grad()(function)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--base-harness", type=Path)
    parser.add_argument("--v5-directed-harness", type=Path)
    parser.add_argument(
        "--v6-directed-harness",
        type=Path,
        help="absolute path to frozen native_c2_v6_register_numerator_directed.py",
    )
    parser.add_argument("--sync-repeats", type=int, default=4)
    parser.add_argument("--max-iteration-seconds", type=float, default=15.0)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-3)
    parser.add_argument(
        "--self-test-metadata",
        action="store_true",
        help="run the CPU-only metadata-filter/oracle-layout self-test",
    )
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


def _require_runtime_paths(args: argparse.Namespace) -> None:
    for attribute in (
        "library",
        "base_harness",
        "v5_directed_harness",
        "v6_directed_harness",
    ):
        path = getattr(args, attribute)
        if path is None:
            raise ValueError(f"--{attribute.replace('_', '-')} is required")
        if not path.is_absolute():
            raise ValueError(f"--{attribute.replace('_', '-')} must be absolute: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)


def _metadata_filtered_pairs(
    topk_values: list[int],
    block_table_values: list[int],
    num_physical_pages: int,
) -> tuple[list[dict[str, int]], list[dict[str, int | str]]]:
    """Return valid selected-page pairs without ever indexing invalid metadata."""
    if num_physical_pages <= 0:
        raise ValueError("num_physical_pages must be positive")
    accepted: list[dict[str, int]] = []
    rejected: list[dict[str, int | str]] = []
    for selected_slot, logical_page in enumerate(topk_values):
        logical_page = int(logical_page)
        if logical_page < 0 or logical_page >= len(block_table_values):
            rejected.append(
                {
                    "selected_slot": selected_slot,
                    "logical_page": logical_page,
                    "reason": "invalid-logical-page",
                }
            )
            continue
        # This lookup happens only after the logical-page range check above.
        physical_page = int(block_table_values[logical_page])
        if physical_page < 0 or physical_page >= num_physical_pages:
            rejected.append(
                {
                    "selected_slot": selected_slot,
                    "logical_page": logical_page,
                    "physical_page": physical_page,
                    "reason": "invalid-physical-page",
                }
            )
            continue
        accepted.append(
            {
                "selected_slot": selected_slot,
                "logical_page": logical_page,
                "physical_page": physical_page,
            }
        )
    return accepted, rejected


def _metadata_filter_self_test() -> dict[str, object]:
    """CPU-only proof that both gap encodings have the intended valid pages."""
    logical_topk = [0, 1, -1, 2] + [-1] * 12
    logical_block = [0, 1, 2] + [-1] * 29
    physical_topk = [0, 1, 2, 3] + [-1] * 12
    physical_block = [0, 1, -1, 2] + [-1] * 28
    logical_accepted, logical_rejected = _metadata_filtered_pairs(
        logical_topk, logical_block, 64
    )
    physical_accepted, physical_rejected = _metadata_filtered_pairs(
        physical_topk, physical_block, 64
    )
    expected_logical = [
        {"selected_slot": 0, "logical_page": 0, "physical_page": 0},
        {"selected_slot": 1, "logical_page": 1, "physical_page": 1},
        {"selected_slot": 3, "logical_page": 2, "physical_page": 2},
    ]
    expected_physical = [
        {"selected_slot": 0, "logical_page": 0, "physical_page": 0},
        {"selected_slot": 1, "logical_page": 1, "physical_page": 1},
        {"selected_slot": 3, "logical_page": 3, "physical_page": 2},
    ]
    logical_ok = logical_accepted == expected_logical and len(logical_rejected) == 13
    physical_ok = physical_accepted == expected_physical and len(physical_rejected) == 13
    return {
        "all_gates_pass": bool(logical_ok and physical_ok),
        "logical_invalid": {
            "accepted": logical_accepted,
            "accepted_matches_expected": logical_ok,
            "rejected": logical_rejected,
        },
        "physical_invalid": {
            "accepted": physical_accepted,
            "accepted_matches_expected": physical_ok,
            "rejected": physical_rejected,
        },
    }


def _k_chunk_raw_key(kind: str, token: int, chunk: int) -> float:
    """Return the raw E4M3 code used for one K chunk in a directed fixture."""
    if kind == "last-chunk":
        return 0.0 if chunk < 7 else (8.0 if token & 1 else -8.0)
    if kind == "transition":
        sign = 1.0 if token & (1 << chunk) else -1.0
        return sign * (1.0 + 0.5 * chunk)
    if kind == "stale-or-shifted":
        sign = 1.0 if ((token + 3 * chunk) % 5) < 2 else -1.0
        return sign * (1.0 + 0.5 * ((3 * chunk) % 8))
    raise ValueError(f"unsupported K-chunk case: {kind}")


def _k_chunk_raw_value(kind: str, token: int) -> float:
    if kind == "last-chunk":
        return 6.0 if token & 1 else -6.0
    if kind == "transition":
        return 6.0 if token & (1 << 3) else -6.0
    if kind == "stale-or-shifted":
        return 6.0 if token % 5 == 0 else -6.0
    raise ValueError(f"unsupported K-chunk case: {kind}")


def _k_chunk_math_sensitivity_self_test() -> dict[str, object]:
    """CPU-only proof that every directed fixture rejects its alternate K order."""
    query_codes = _k_chunk_query_codes()

    def output(kind: str, alternate: bool) -> float:
        logits: list[float] = []
        values: list[float] = []
        for token in range(128):
            dot = 0.0
            for chunk, q_code in enumerate(query_codes):
                key_chunk = (
                    0.0
                    if alternate and kind == "last-chunk"
                    else _k_chunk_raw_key(
                        kind, token, (chunk - 1) % len(query_codes)
                    )
                    if alternate
                    else _k_chunk_raw_key(kind, token, chunk)
                )
                dot += _K_CHUNK_WIDTH * q_code * key_chunk
            # The fixture uses production q/k scales 1/4 and scale=1/sqrt(128).
            logits.append(dot * 0.25 * 0.25 / math.sqrt(128.0))
            values.append(_k_chunk_raw_value(kind, token) * 0.5)
        maximum = max(logits)
        weights = [math.exp(value - maximum) for value in logits]
        return sum(weight * value for weight, value in zip(weights, values)) / sum(weights)

    def nearest_bfloat16(value: float) -> float:
        # Round through IEEE float32, then round its upper 16 bits to nearest
        # even exactly as a BF16 conversion does for these finite scalars.
        bits = struct.unpack(">I", struct.pack(">f", value))[0]
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
        return struct.unpack(">f", struct.pack(">I", rounded & 0xFFFF0000))[0]

    cases: dict[str, dict[str, float | bool]] = {}
    for kind in ("transition", "last-chunk", "stale-or-shifted"):
        expected = output(kind, alternate=False)
        alternate = output(kind, alternate=True)
        difference = abs(expected - alternate)
        nearest = nearest_bfloat16(expected)
        rounding_error = abs(expected - nearest)
        rounding_tolerance = 1.0e-4 + 1.0e-3 * abs(expected)
        rounding_passed = rounding_error <= rounding_tolerance
        cases[kind] = {
            "true_output": expected,
            "alternate_output": alternate,
            "alternate_reference_max_abs": difference,
            "minimum_required_max_abs": _K_CHUNK_SENSITIVITY_MINIMUM,
            "correct_to_nearest_bf16": nearest,
            "correct_to_nearest_bf16_abs": rounding_error,
            "correct_to_nearest_bf16_tolerance": rounding_tolerance,
            "correct_to_nearest_bf16_passed": rounding_passed,
            "passed": bool(
                difference > _K_CHUNK_SENSITIVITY_MINIMUM and rounding_passed
            ),
        }
    return {
        "all_gates_pass": all(bool(value["passed"]) for value in cases.values()),
        "cases": cases,
        "query_chunk_codes": list(query_codes),
        "schema": "c2-native-c2-v16-k-chunk-lookahead-math-sensitivity-v1",
    }


@_no_grad
def _metadata_filtered_dense_oracle(
    base: ModuleType,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    topk: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    contract: Any,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dense selected-page oracle with explicit logical/physical filtering."""
    output = torch.zeros(
        (base._BATCH, base._QUERY_HEADS, base._HEAD_DIM),
        device=query.device,
        dtype=dtype,
    )
    token_offsets = torch.arange(base._PAGE_SIZE, device=query.device, dtype=torch.int64)
    q = query.to(dtype).mul(contract.q_scale)
    packed = kv_cache.to(dtype)
    for batch in range(base._BATCH):
        seq_len = int(seq_lens[batch].item())
        block_values = [int(value) for value in block_table[batch].cpu().tolist()]
        for kv_head in range(base._KV_HEADS):
            selected_values = [
                int(value) for value in topk[batch, kv_head].cpu().tolist()
            ]
            accepted, _ = _metadata_filtered_pairs(
                selected_values, block_values, contract.num_physical_pages
            )
            if not accepted:
                continue
            logical_pages = torch.tensor(
                [item["logical_page"] for item in accepted],
                device=query.device,
                dtype=torch.int64,
            )
            physical_pages = torch.tensor(
                [item["physical_page"] for item in accepted],
                device=query.device,
                dtype=torch.int64,
            )
            # ``physical_pages`` comes only from _metadata_filtered_pairs;
            # no negative Python index can reach this tensor expression.
            selected = packed[physical_pages, kv_head]
            key = selected[..., : base._HEAD_DIM].reshape(-1, base._HEAD_DIM)
            key = key.mul(contract.k_scale)
            value = selected[..., base._HEAD_DIM :].reshape(-1, base._HEAD_DIM)
            value = value.mul(contract.v_scale)
            positions = (
                logical_pages[:, None] * base._PAGE_SIZE + token_offsets
            ).reshape(-1)
            causal = positions < seq_len
            if not bool(causal.any().item()):
                continue
            head_start = kv_head * base._GQA
            scores = q[batch, head_start : head_start + base._GQA] @ key.transpose(0, 1)
            scores.mul_(contract.scale)
            scores.masked_fill_(~causal.unsqueeze(0), -torch.inf)
            probabilities = torch.softmax(scores, dim=-1)
            output[batch, head_start : head_start + base._GQA] = probabilities @ value
    return output


def _metadata_oracle_metrics(
    base: ModuleType,
    output: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    contract: Any,
    args: argparse.Namespace,
) -> dict[str, object]:
    reference_fp64 = _metadata_filtered_dense_oracle(
        base, *inputs, contract, torch.float64
    )
    reference_fp32 = _metadata_filtered_dense_oracle(
        base, *inputs, contract, torch.float32
    )
    actual_fp64 = output.to(torch.float64)
    actual_fp32 = output.to(torch.float32)
    difference_fp64 = (actual_fp64 - reference_fp64).abs()
    difference_fp32 = (actual_fp32 - reference_fp32).abs()
    return {
        "finite_output": bool(torch.isfinite(actual_fp64).all().item()),
        "fp64": {
            "allclose": bool(
                torch.allclose(actual_fp64, reference_fp64, atol=args.atol, rtol=args.rtol)
            ),
            "max_abs": float(difference_fp64.max().item()),
            "mean_abs": float(difference_fp64.mean().item()),
        },
        "fp32": {
            "allclose": bool(
                torch.allclose(actual_fp32, reference_fp32, atol=args.atol, rtol=args.rtol)
            ),
            "max_abs": float(difference_fp32.max().item()),
            "mean_abs": float(difference_fp32.mean().item()),
        },
        "reference_fp64_fp32_agree": bool(
            torch.allclose(
                reference_fp64,
                reference_fp32.to(torch.float64),
                atol=args.atol,
                rtol=args.rtol,
            )
        ),
    }


def _make_metadata_gap_inputs(
    base: ModuleType,
    contract: Any,
    kind: str,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, dict[str, object]]:
    """Build an adversarial rank-0 gap whose post-gap physical page wins."""
    if kind not in ("logical-invalid", "physical-invalid"):
        raise ValueError(f"unsupported metadata-gap kind: {kind}")
    if base._TOPK != 16 or base._PAGE_SIZE != 128:
        raise RuntimeError("metadata-gap fixture assumes the production C2 shape")
    device = torch.device("cuda")
    query = torch.full(
        (base._BATCH, base._QUERY_HEADS, base._HEAD_DIM),
        8.0,
        device=device,
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn).contiguous()
    cache_f32 = torch.zeros(
        (
            contract.num_physical_pages,
            base._KV_HEADS,
            base._PAGE_SIZE,
            2 * base._HEAD_DIM,
        ),
        device=device,
        dtype=torch.float32,
    )
    # Pages 0/1 lose by roughly 90.5 logit units to page 2.  The selected
    # post-gap page therefore must return exactly raw V=6 times v_scale=1/2.
    page_codes = {
        0: (-8.0, -6.0),
        1: (-8.0, 0.0),
        _POST_GAP_PHYSICAL_PAGE: (8.0, 6.0),
        # Poison makes an accidental Python-style -1 physical interpretation
        # observable rather than silently numerically similar to the target.
        contract.num_physical_pages - 1: (8.0, -8.0),
    }
    for physical_page, (key_code, value_code) in page_codes.items():
        cache_f32[physical_page, :, :, : base._HEAD_DIM].fill_(key_code)
        cache_f32[physical_page, :, :, base._HEAD_DIM :].fill_(value_code)
    kv_cache = cache_f32.to(torch.float8_e4m3fn).contiguous()

    block_cpu = [-1] * contract.max_logical_pages
    if kind == "logical-invalid":
        topk_prefix = [0, 1, -1, 2]
        block_cpu[0:3] = [0, 1, _POST_GAP_PHYSICAL_PAGE]
        sequence_length = 3 * base._PAGE_SIZE
        expected_valid = [
            {"selected_slot": 0, "logical_page": 0, "physical_page": 0},
            {"selected_slot": 1, "logical_page": 1, "physical_page": 1},
            {
                "selected_slot": 3,
                "logical_page": 2,
                "physical_page": _POST_GAP_PHYSICAL_PAGE,
            },
        ]
    else:
        topk_prefix = [0, 1, 2, 3]
        block_cpu[0:4] = [0, 1, -1, _POST_GAP_PHYSICAL_PAGE]
        sequence_length = 4 * base._PAGE_SIZE
        expected_valid = [
            {"selected_slot": 0, "logical_page": 0, "physical_page": 0},
            {"selected_slot": 1, "logical_page": 1, "physical_page": 1},
            {
                "selected_slot": 3,
                "logical_page": 3,
                "physical_page": _POST_GAP_PHYSICAL_PAGE,
            },
        ]
    topk_cpu = topk_prefix + [-1] * (base._TOPK - len(topk_prefix))
    accepted, rejected = _metadata_filtered_pairs(
        topk_cpu, block_cpu, contract.num_physical_pages
    )
    if accepted != expected_valid:
        raise RuntimeError(f"metadata fixture construction failed: {kind}")
    if len(rejected) != 13:
        raise RuntimeError(f"metadata fixture rejection count changed: {kind}")
    block_table = torch.tensor(
        block_cpu, device=device, dtype=torch.int32
    ).view(1, -1).expand(base._BATCH, -1).contiguous()
    topk = torch.tensor(
        topk_cpu, device=device, dtype=torch.int32
    ).view(1, 1, -1).expand(base._BATCH, base._KV_HEADS, -1).contiguous()
    seq_lens = torch.full(
        (base._BATCH,), sequence_length, device=device, dtype=torch.int32
    )
    expected = torch.full(
        (base._BATCH, base._QUERY_HEADS, base._HEAD_DIM),
        _EXPECTED_BF16_VALUE,
        device=device,
        dtype=torch.bfloat16,
    )
    return (
        (query, kv_cache, topk, block_table, seq_lens),
        expected,
        {
            "kind": kind,
            "topk_prefix": topk_prefix,
            "topk_tail_invalid_count": base._TOPK - len(topk_prefix),
            "block_table_prefix": block_cpu[:4],
            "sequence_length": sequence_length,
            "valid_selected_pages": accepted,
            "rejected_selected_pages": rejected,
            "post_gap_physical_page": _POST_GAP_PHYSICAL_PAGE,
            "post_gap_raw_k": 8.0,
            "post_gap_raw_v": 6.0,
            "expected_bf16_value": _EXPECTED_BF16_VALUE,
        },
    )


def _gap_case(
    base: ModuleType,
    v5: ModuleType,
    op: Any,
    contract: Any,
    args: argparse.Namespace,
    kind: str,
) -> dict[str, object]:
    inputs, exact_expected, metadata = _make_metadata_gap_inputs(base, contract, kind)
    output, observation = v5._call_sync_repeatedly(base, op, inputs, contract, args)
    oracle = _metadata_oracle_metrics(base, output, inputs, contract, args)
    exact_match = bool(torch.equal(output, exact_expected))
    exact_difference = (output.to(torch.float32) - exact_expected.to(torch.float32)).abs()
    all_gates_pass = bool(
        oracle["finite_output"]
        and oracle["fp64"]["allclose"]
        and oracle["fp32"]["allclose"]
        and oracle["reference_fp64_fp32_agree"]
        and exact_match
        and observation["pointer_unchanged"]
        and observation["return_is_none"]
        and observation["bitwise_repeatable"]
    )
    return {
        "all_gates_pass": all_gates_pass,
        "caller_output": {
            "pointer_before": observation["pointer_before"],
            "pointer_after": observation["pointer_after"],
            "pointer_unchanged": observation["pointer_unchanged"],
            "return_is_none": observation["return_is_none"],
        },
        "exact_bf16_target": {
            "all_locations_match": exact_match,
            "expected_value": _EXPECTED_BF16_VALUE,
            "max_abs": float(exact_difference.max().item()),
            "nonzero_mismatch_count": int(torch.count_nonzero(exact_difference).item()),
        },
        "metadata": metadata,
        "metadata_filtered_dense_oracle": oracle,
        "stability": {"bitwise_repeatable": observation["bitwise_repeatable"]},
        "sync_watchdog": {
            "elapsed_seconds": observation["elapsed_seconds"],
            "max_iteration_seconds": observation["max_iteration_seconds"],
        },
    }


@_no_grad
def _make_k_chunk_inputs(
    base: ModuleType,
    contract: Any,
    kind: str,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], dict[str, object]]:
    """Build one-page fixtures that make K-chunk stale/shifted reads observable."""
    if kind not in ("transition", "last-chunk", "stale-or-shifted"):
        raise ValueError(f"unsupported K-chunk case: {kind}")
    if base._TOPK != 16 or base._PAGE_SIZE != 128 or base._HEAD_DIM != 128:
        raise RuntimeError("K-chunk fixtures assume the fixed production C2 shape")
    device = torch.device("cuda")
    target_physical_page = 3
    query = _make_k_chunk_coded_query(base, device)
    cache_f32 = torch.zeros(
        (
            contract.num_physical_pages,
            base._KV_HEADS,
            base._PAGE_SIZE,
            2 * base._HEAD_DIM,
        ),
        device=device,
        dtype=torch.float32,
    )
    for token in range(base._PAGE_SIZE):
        for chunk in range(base._HEAD_DIM // _K_CHUNK_WIDTH):
            raw_key = _k_chunk_raw_key(kind, token, chunk)
            cache_f32[
                target_physical_page,
                :,
                token,
                chunk * _K_CHUNK_WIDTH : (chunk + 1) * _K_CHUNK_WIDTH,
            ].fill_(raw_key)
        raw_value = _k_chunk_raw_value(kind, token)
        cache_f32[
            target_physical_page, :, token, base._HEAD_DIM :
        ].fill_(raw_value)

    alternate_f32 = cache_f32.clone()
    if kind == "last-chunk":
        alternate_f32[target_physical_page, :, :, : base._HEAD_DIM].zero_()
        alternate_description = "all-zero K reference (detects an omitted final chunk)"
    else:
        alternate_f32[target_physical_page, :, :, : base._HEAD_DIM] = torch.roll(
            alternate_f32[target_physical_page, :, :, : base._HEAD_DIM],
            shifts=16,
            dims=-1,
        )
        alternate_description = "one-16-element cyclic K-chunk shift reference"

    topk_cpu = [0] + [-1] * (base._TOPK - 1)
    block_cpu = [-1] * contract.max_logical_pages
    block_cpu[0] = target_physical_page
    topk = torch.tensor(
        topk_cpu, device=device, dtype=torch.int32
    ).view(1, 1, -1).expand(base._BATCH, base._KV_HEADS, -1).contiguous()
    block_table = torch.tensor(
        block_cpu, device=device, dtype=torch.int32
    ).view(1, -1).expand(base._BATCH, -1).contiguous()
    seq_lens = torch.full(
        (base._BATCH,), base._PAGE_SIZE, device=device, dtype=torch.int32
    )
    inputs = (
        query,
        cache_f32.to(torch.float8_e4m3fn).contiguous(),
        topk,
        block_table,
        seq_lens,
    )
    alternate_inputs = (
        query,
        alternate_f32.to(torch.float8_e4m3fn).contiguous(),
        topk,
        block_table,
        seq_lens,
    )
    return inputs, alternate_inputs, {
        "case": kind,
        "chunk_count": base._HEAD_DIM // 16,
        "chunk_width": 16,
        "query_chunk_codes": list(_k_chunk_query_codes()),
        "selected_slot": 0,
        "logical_page": 0,
        "physical_page": target_physical_page,
        "all_other_selected_slots_invalid": True,
        "sequence_length": base._PAGE_SIZE,
        "alternate_reference": alternate_description,
    }


@_no_grad
def _k_chunk_case(
    base: ModuleType,
    v5: ModuleType,
    op: Any,
    contract: Any,
    args: argparse.Namespace,
    kind: str,
) -> dict[str, object]:
    inputs, alternate_inputs, metadata = _make_k_chunk_inputs(base, contract, kind)
    output, observation = v5._call_sync_repeatedly(base, op, inputs, contract, args)
    oracle = _metadata_oracle_metrics(base, output, inputs, contract, args)
    reference = base._oracle(*inputs, contract, torch.float64)
    alternate_reference = base._oracle(*alternate_inputs, contract, torch.float64)
    shift_sensitivity_max_abs = float(
        (reference - alternate_reference).abs().max().item()
    )
    # The constructed alternate reference must be far beyond the runtime
    # allclose tolerance; otherwise a stale or shifted raw-FP8 chunk could
    # evade a correct-looking oracle comparison by numerical coincidence.
    sensitivity_passed = shift_sensitivity_max_abs > 1.0e-1
    all_gates_pass = bool(
        oracle["finite_output"]
        and oracle["fp64"]["allclose"]
        and oracle["fp32"]["allclose"]
        and oracle["reference_fp64_fp32_agree"]
        and observation["pointer_unchanged"]
        and observation["return_is_none"]
        and observation["bitwise_repeatable"]
        and sensitivity_passed
    )
    return {
        "all_gates_pass": all_gates_pass,
        "caller_output": {
            "pointer_before": observation["pointer_before"],
            "pointer_after": observation["pointer_after"],
            "pointer_unchanged": observation["pointer_unchanged"],
            "return_is_none": observation["return_is_none"],
        },
        "k_chunk_metadata": metadata,
        "metadata_filtered_dense_oracle": oracle,
        "stale_or_shifted_sensitivity": {
            "alternate_reference_max_abs": shift_sensitivity_max_abs,
            "minimum_required_max_abs": 1.0e-1,
            "passed": sensitivity_passed,
        },
        "stability": {"bitwise_repeatable": observation["bitwise_repeatable"]},
        "sync_watchdog": {
            "elapsed_seconds": observation["elapsed_seconds"],
            "max_iteration_seconds": observation["max_iteration_seconds"],
        },
    }

@_no_grad
def _run(args: argparse.Namespace) -> dict[str, object]:
    _require_runtime_paths(args)
    if args.sync_repeats < 3:
        raise ValueError("--sync-repeats must be at least three")
    if not math.isfinite(args.max_iteration_seconds) or not (0.0 < args.max_iteration_seconds <= 60.0):
        raise ValueError("--max-iteration-seconds must be finite in (0, 60]")
    if torch is None:
        raise RuntimeError("PyTorch is required for the GPU directed suite")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 3):
        raise RuntimeError(f"native C2 requires B300 capability (10, 3), got {capability}")

    # Execute the frozen v6 suite as an opaque complete gate: this retains its
    # original rank/softmax/all-invalid checks and its dispatch profiler gate.
    v6 = _load_module(args.v6_directed_harness, "native_c2_v16_frozen_v6_directed")
    if not hasattr(v6, "_run"):
        raise RuntimeError("frozen v6 directed harness lacks _run")
    v6_directed = v6._run(args)
    v5 = _load_module(args.v5_directed_harness, "native_c2_v16_v5_helpers")
    for helper in ("_call_sync_repeatedly", "_contract", "_load_base_harness"):
        if not hasattr(v5, helper):
            raise RuntimeError(f"v5 directed harness lacks required helper: {helper}")
    base = v5._load_base_harness(args.base_harness)
    contract = v5._contract(base, args)
    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "native_c2_msa_decode"):
        raise RuntimeError("frozen v6 suite did not register native C2 operator")
    op = torch.ops._C.native_c2_msa_decode
    logical_invalid_gap = _gap_case(
        base, v5, op, contract, args, "logical-invalid"
    )
    physical_invalid_gap = _gap_case(
        base, v5, op, contract, args, "physical-invalid"
    )
    k_chunk_transition = _k_chunk_case(
        base, v5, op, contract, args, "transition"
    )
    k_chunk_last_chunk = _k_chunk_case(
        base, v5, op, contract, args, "last-chunk"
    )
    k_chunk_stale_or_shifted = _k_chunk_case(
        base, v5, op, contract, args, "stale-or-shifted"
    )
    v6_pass = bool(v6_directed.get("all_gates_pass", False))
    return {
        "all_gates_pass": bool(
            v6_pass
            and logical_invalid_gap["all_gates_pass"]
            and physical_invalid_gap["all_gates_pass"]
            and k_chunk_transition["all_gates_pass"]
            and k_chunk_last_chunk["all_gates_pass"]
            and k_chunk_stale_or_shifted["all_gates_pass"]
        ),
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
        "gate_composition": {
            "logical_invalid_gap_passed": logical_invalid_gap["all_gates_pass"],
            "physical_invalid_gap_passed": physical_invalid_gap["all_gates_pass"],
            "k_chunk_transition_passed": k_chunk_transition["all_gates_pass"],
            "k_chunk_last_chunk_passed": k_chunk_last_chunk["all_gates_pass"],
            "k_chunk_stale_or_shifted_passed": k_chunk_stale_or_shifted["all_gates_pass"],
            "v6_directed_complete_gate_passed": v6_pass,
        },
        "logical_invalid_gap": logical_invalid_gap,
        "no_monkeypatch": True,
        "operator_library": v6_directed.get("operator_library"),
        "physical_invalid_gap": physical_invalid_gap,
        "k_chunk_directed_case_count": 3,
        "k_chunk_transition": k_chunk_transition,
        "k_chunk_last_chunk": k_chunk_last_chunk,
        "k_chunk_stale_or_shifted": k_chunk_stale_or_shifted,
        "schema": _SCHEMA,
        "torch": {"cuda": torch.version.cuda, "version": torch.__version__},
        "v6_directed_complete_gate": v6_directed,
        "v6_directed_complete_gate_passed": v6_pass,
    }


def _as_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    try:
        args = _parse_args()
        if args.self_test_metadata:
            metadata_self_test = _metadata_filter_self_test()
            k_chunk_math_sensitivity_self_test = _k_chunk_math_sensitivity_self_test()
            result = {
                "all_gates_pass": bool(
                    metadata_self_test["all_gates_pass"]
                    and k_chunk_math_sensitivity_self_test["all_gates_pass"]
                ),
                "k_chunk_math_sensitivity_self_test": k_chunk_math_sensitivity_self_test,
                "metadata_filter_self_test": metadata_self_test,
                "schema": _SCHEMA,
            }
        else:
            result = _run(args)
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
