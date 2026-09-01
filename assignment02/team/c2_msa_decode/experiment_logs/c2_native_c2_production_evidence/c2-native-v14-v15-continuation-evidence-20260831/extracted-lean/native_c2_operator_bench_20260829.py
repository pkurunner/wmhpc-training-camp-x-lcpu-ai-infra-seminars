#!/usr/bin/env python3
"""Direct correctness harness for the AOT ``_C.native_c2_msa_decode`` op.

This intentionally calls the registered operator directly.  It does not
monkeypatch vLLM's Python dispatch and it does not form K/V bridge tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch


_BATCH = 16
_QUERY_HEADS = 64
_KV_HEADS = 4
_GQA = _QUERY_HEADS // _KV_HEADS
_HEAD_DIM = 128
_PAGE_SIZE = 128
_TOPK = 16


@dataclass(frozen=True)
class Contract:
    num_physical_pages: int
    max_logical_pages: int
    scale: float
    q_scale: float
    k_scale: float
    v_scale: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--library",
        type=Path,
        help="absolute path to the derived _C_stable_libtorch.abi3.so",
    )
    parser.add_argument("--num-physical-pages", type=int, default=64)
    parser.add_argument("--max-logical-pages", type=int, default=32)
    parser.add_argument("--scale", type=float, default=1.0 / math.sqrt(_HEAD_DIM))
    parser.add_argument("--q-scale", type=float, default=0.25)
    parser.add_argument("--k-scale", type=float, default=0.25)
    parser.add_argument("--v-scale", type=float, default=0.5)
    parser.add_argument(
        "--oracle-dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--rtol", type=float, default=5.0e-2)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> Contract:
    if args.num_physical_pages < args.max_logical_pages:
        raise ValueError("num_physical_pages must be >= max_logical_pages")
    if args.max_logical_pages < _TOPK:
        raise ValueError("max_logical_pages must be >= 16")
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("atol and rtol must be non-negative")
    scales = (args.scale, args.q_scale, args.k_scale, args.v_scale)
    if not all(math.isfinite(value) and value > 0.0 for value in scales):
        raise ValueError("all scalar scales must be finite and positive")
    return Contract(
        num_physical_pages=args.num_physical_pages,
        max_logical_pages=args.max_logical_pages,
        scale=args.scale,
        q_scale=args.q_scale,
        k_scale=args.k_scale,
        v_scale=args.v_scale,
    )


def _require_registered_operator(library: Path | None) -> str:
    if library is not None:
        if not library.is_absolute():
            raise ValueError("--library must be an absolute path")
        if not library.is_file():
            raise FileNotFoundError(f"--library does not exist: {library}")
        # Do not import vLLM in library mode: an installed stable extension can
        # register the same namespace and make a second load ambiguous.
        torch.ops.load_library(str(library))
        source = str(library)
    else:
        # The normal installed-wheel path loads the stable libtorch extension.
        # The call below remains direct; no wrapper or dispatch monkeypatch is
        # involved.
        import vllm._custom_ops  # noqa: F401

        source = "installed-vllm"

    if not hasattr(torch.ops, "_C") or not hasattr(
        torch.ops._C, "native_c2_msa_decode"
    ):
        raise RuntimeError("torch.ops._C.native_c2_msa_decode is not registered")
    return source


def _make_inputs(contract: Contract, seed: int) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    query = (
        torch.randn(
            (_BATCH, _QUERY_HEADS, _HEAD_DIM),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    # The final dimension is the production packed [K | V] FP8 layout.
    kv_cache = (
        torch.randn(
            (
                contract.num_physical_pages,
                _KV_HEADS,
                _PAGE_SIZE,
                2 * _HEAD_DIM,
            ),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    seq_lens = torch.randint(
        low=_TOPK * _PAGE_SIZE,
        high=contract.max_logical_pages * _PAGE_SIZE + 1,
        size=(_BATCH,),
        generator=generator,
        device=device,
        dtype=torch.int32,
    ).contiguous()
    block_table = torch.empty(
        (_BATCH, contract.max_logical_pages), device=device, dtype=torch.int32
    )
    topk = torch.empty((_BATCH, _KV_HEADS, _TOPK), device=device, dtype=torch.int32)
    # Generate each row independently so physical pages are deliberately shared
    # across requests; no synthetic per-request physical-page partition exists.
    for batch in range(_BATCH):
        block_table[batch].copy_(
            torch.randperm(
                contract.num_physical_pages,
                generator=generator,
                device=device,
                dtype=torch.int32,
            )[: contract.max_logical_pages]
        )
        num_valid_pages = int((int(seq_lens[batch]) + _PAGE_SIZE - 1) // _PAGE_SIZE)
        for kv_head in range(_KV_HEADS):
            selected = torch.randperm(
                num_valid_pages,
                generator=generator,
                device=device,
                dtype=torch.int32,
            )[:_TOPK]
            topk[batch, kv_head].copy_(torch.sort(selected).values)
    return query, kv_cache, topk.contiguous(), block_table.contiguous(), seq_lens


@torch.no_grad()
def _oracle(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    topk: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    contract: Contract,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Independent dense FP32/FP64 oracle over the selected logical pages."""
    output = torch.empty(
        (_BATCH, _QUERY_HEADS, _HEAD_DIM), device=query.device, dtype=dtype
    )
    token_offsets = torch.arange(_PAGE_SIZE, device=query.device, dtype=torch.int64)
    q = query.to(dtype).mul(contract.q_scale)
    packed = kv_cache.to(dtype)
    for batch in range(_BATCH):
        seq_len = int(seq_lens[batch])
        for kv_head in range(_KV_HEADS):
            logical_pages = topk[batch, kv_head].to(torch.int64)
            physical_pages = block_table[batch, logical_pages].to(torch.int64)
            selected = packed[physical_pages, kv_head]
            key = selected[..., :_HEAD_DIM].reshape(-1, _HEAD_DIM).mul(contract.k_scale)
            value = selected[..., _HEAD_DIM:].reshape(-1, _HEAD_DIM).mul(
                contract.v_scale
            )
            positions = (
                logical_pages[:, None] * _PAGE_SIZE + token_offsets
            ).reshape(-1)
            causal = positions < seq_len
            head_start = kv_head * _GQA
            scores = q[batch, head_start : head_start + _GQA] @ key.transpose(0, 1)
            scores.mul_(contract.scale)
            scores.masked_fill_(~causal.unsqueeze(0), -torch.inf)
            probabilities = torch.softmax(scores, dim=-1)
            output[batch, head_start : head_start + _GQA] = probabilities @ value
    return output


def _as_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


@torch.no_grad()
def _run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 3):
        raise RuntimeError(
            f"native C2 requires B300 capability (10, 3), got {capability}"
        )
    contract = _validate_args(args)
    operator_library = _require_registered_operator(args.library)
    query, kv_cache, topk, block_table, seq_lens = _make_inputs(contract, args.seed)
    output = torch.full(
        (_BATCH, _QUERY_HEADS, _HEAD_DIM),
        float("nan"),
        device=query.device,
        dtype=torch.bfloat16,
    )
    pointer_before = output.data_ptr()
    torch.cuda.synchronize()
    returned = torch.ops._C.native_c2_msa_decode(
        output,
        query,
        kv_cache,
        topk,
        block_table,
        seq_lens,
        contract.scale,
        contract.q_scale,
        contract.k_scale,
        contract.v_scale,
    )
    torch.cuda.synchronize()
    oracle_dtype = torch.float64 if args.oracle_dtype == "float64" else torch.float32
    reference = _oracle(
        query,
        kv_cache,
        topk,
        block_table,
        seq_lens,
        contract,
        oracle_dtype,
    )
    actual = output.to(oracle_dtype)
    difference = (actual - reference).abs()
    denominator = reference.abs().clamp_min(torch.finfo(oracle_dtype).eps)
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    max_rel = float((difference / denominator).max().item())
    finite = bool(torch.isfinite(actual).all().item())
    correct = finite and bool(
        torch.allclose(actual, reference, atol=args.atol, rtol=args.rtol)
    )
    pointer_unchanged = output.data_ptr() == pointer_before
    return {
        "all_gates_pass": bool(correct and pointer_unchanged and returned is None),
        "caller_output": {
            "pointer_before": pointer_before,
            "pointer_after": output.data_ptr(),
            "pointer_unchanged": pointer_unchanged,
            "return_is_none": returned is None,
        },
        "contract": {
            "batch": _BATCH,
            "head_dim": _HEAD_DIM,
            "kv_heads": _KV_HEADS,
            "max_logical_pages": contract.max_logical_pages,
            "num_physical_pages": contract.num_physical_pages,
            "page_size": _PAGE_SIZE,
            "q_heads": _QUERY_HEADS,
            "q_scale": contract.q_scale,
            "k_scale": contract.k_scale,
            "v_scale": contract.v_scale,
            "scale": contract.scale,
            "topk": _TOPK,
        },
        "correctness": {
            "allclose": correct,
            "atol": args.atol,
            "finite_output": finite,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "mean_abs": mean_abs,
            "oracle_dtype": args.oracle_dtype,
            "rtol": args.rtol,
        },
        "device_capability": list(capability),
        "dispatch": "direct_torch.ops._C.native_c2_msa_decode",
        "no_monkeypatch": True,
        "operator_library": operator_library,
        "schema": "c2-native-c2-direct-operator-oracle-v1",
        "seed": args.seed,
    }


def main() -> int:
    try:
        result = _run(_parse_args())
    except Exception as error:  # Emit a single strict JSON failure record.
        result = {
            "all_gates_pass": False,
            "error": f"{type(error).__name__}: {error}",
            "schema": "c2-native-c2-direct-operator-oracle-v1",
            "traceback": traceback.format_exc(),
        }
        print(_as_json(result))
        return 1
    print(_as_json(result))
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
