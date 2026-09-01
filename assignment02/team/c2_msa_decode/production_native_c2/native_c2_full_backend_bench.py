#!/usr/bin/env python3
"""Strict B300 audit of the real MiniMax-M3 ``native_c2`` backend path.

| Symbol | Meaning | Required value |
| --- | --- | --- |
| B | decode requests / query tokens | 16 |
| Hq / Hkv / D | query heads / KV heads / head dimension | 64 / 4 / 128 |
| P / K | page size / selected logical pages | 128 / 16 |
| q_s, k_s, v_s | scalar FP8 dequantization scales | positive scalar |

The only measured execution route is ``MiniMaxM3SparseMSAImpl.forward``.  In
particular, this harness never calls the registered dispatcher directly and
does not replace vLLM functions at runtime.  A profiler trace is part of the
acceptance gate: it must contain both the dispatcher op and the native CUDA
kernel produced by that one ``forward`` call.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import traceback
from types import SimpleNamespace
from typing import Any

import torch


_BATCH = 16
_Q_HEADS = 64
_KV_HEADS = 4
_HEAD_DIM = 128
_PAGE_SIZE = 128
_TOPK = 16
_Q_SCALE = 0.25
_ATOL = 1.0e-4
_RTOL = 1.0e-3
_WARMUP = 1
_SCHEMA = "c2-native-c2-full-vllm-backend-v2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-root", required=True, type=Path)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--timing-repetitions", type=int, default=50)
    return parser.parse_args()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def _config(backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                num_attention_heads=_Q_HEADS,
                sparse_attention_config={"sparse_topk_blocks": _TOPK},
            )
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, decode_context_parallel_size=1
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"),
        attention_config=SimpleNamespace(
            minimax_m3_msa_decode_backend=backend
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=_BATCH),
        speculative_config=None,
    )


def _common_metadata(problem: Any, common_cls: type[Any]) -> Any:
    starts = torch.arange(_BATCH + 1, device="cuda", dtype=torch.int32)
    return common_cls(
        query_start_loc=starts,
        query_start_loc_cpu=starts.cpu(),
        seq_lens=problem.seq_lens,
        num_reqs=_BATCH,
        num_actual_tokens=_BATCH,
        max_query_len=1,
        max_seq_len=4096,
        block_table_tensor=problem.block_table,
        slot_mapping=torch.full((_BATCH,), -1, device="cuda", dtype=torch.int64),
        seq_lens_cpu_upper_bound=problem.seq_lens.cpu().contiguous(),
    )


def _native_fp32_oracle(
    problem: Any,
    query_fp8: torch.Tensor,
    *,
    q_scale: float,
    k_scale: float,
    v_scale: float,
    reference: Any,
) -> torch.Tensor:
    """Independent FP32 selected-page oracle for the exact native inputs.

    The native kernel first produces BF16 staging values from the FP8 tensors.
    Matching that documented input rounding here prevents a quantization-model
    mismatch from being misreported as an attention arithmetic error.  The
    imported reference then performs the selected-page attention in FP32.
    """
    q_bf16 = (query_fp8.float() * q_scale).to(torch.bfloat16).contiguous()
    kv_bf16 = torch.empty_like(problem.kv_cache, dtype=torch.bfloat16)
    kv_bf16[..., :_HEAD_DIM] = (
        problem.kv_cache[..., :_HEAD_DIM].float() * k_scale
    ).to(torch.bfloat16)
    kv_bf16[..., _HEAD_DIM:] = (
        problem.kv_cache[..., _HEAD_DIM:].float() * v_scale
    ).to(torch.bfloat16)
    exact_input_problem = replace(
        problem,
        q=q_bf16,
        kv_cache=kv_bf16.contiguous(),
        k_scale=None,
        v_scale=None,
        storage_dtype="bf16-native-staging",
    )
    return reference(exact_input_problem)


def _check_output(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    oracle: str,
) -> dict[str, Any]:
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    difference = (actual_fp32 - reference_fp32).abs()
    finite = bool(torch.isfinite(actual_fp32).all().item())
    allclose = bool(
        torch.allclose(actual_fp32, reference_fp32, atol=_ATOL, rtol=_RTOL)
    )
    return {
        "oracle": oracle,
        "atol": _ATOL,
        "rtol": _RTOL,
        "finite": finite,
        "allclose": allclose,
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "pass": bool(finite and allclose),
    }


def _event_number(event: Any, name: str) -> float:
    value = getattr(event, name, 0.0)
    return float(value) if value is not None else 0.0


def _event_record(event: Any) -> dict[str, Any]:
    return {
        "key": str(event.key),
        "count": int(getattr(event, "count", 0)),
        "cpu_total_us": _event_number(event, "cpu_time_total"),
        "self_cpu_us": _event_number(event, "self_cpu_time_total"),
        "cuda_total_us": _event_number(event, "cuda_time_total"),
        "self_cuda_us": _event_number(event, "self_cuda_time_total"),
    }


def _profile_checks(profiler: Any, trace: Path) -> dict[str, Any]:
    profiler.export_chrome_trace(str(trace))
    if not trace.is_file() or trace.stat().st_size == 0:
        raise RuntimeError(f"profiler did not write a nonempty trace: {trace}")
    events = list(profiler.key_averages())
    dispatcher = [event for event in events if event.key == "_C::native_c2_msa_decode"]
    kernels = [
        event
        for event in events
        if "native_c2_msa_decode_kernel" in str(event.key)
    ]
    nearby_keys = sorted(
        str(event.key)
        for event in events
        if "native_c2" in str(event.key) or "_C::" in str(event.key)
    )
    checks = {
        "one_cpu_dispatcher_event": len(dispatcher) == 1
        and int(getattr(dispatcher[0], "count", 0)) == 1,
        "one_cuda_native_kernel_event": len(kernels) == 1
        and int(getattr(kernels[0], "count", 0)) == 1,
        "trace_nonempty": True,
    }
    return {
        "trace": str(trace),
        "dispatcher_events": [_event_record(event) for event in dispatcher],
        "cuda_kernel_events": [_event_record(event) for event in kernels],
        "nearby_keys": nearby_keys,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _latency_summary(samples: list[float]) -> dict[str, Any]:
    if not samples or any(not math.isfinite(sample) or sample <= 0.0 for sample in samples):
        raise RuntimeError(f"invalid CUDA-event timing samples: {samples}")
    ordered = sorted(samples)
    p10_index = max(0, math.ceil(0.10 * len(ordered)) - 1)
    p90_index = min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1)
    return {
        "raw_ms": samples,
        "sample_count": len(samples),
        "p10_ms": ordered[p10_index],
        "median_ms": float(statistics.median(samples)),
        "p90_ms": ordered[p90_index],
    }


def _abba_timing(
    native_call: Any,
    triton_call: Any,
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    """CUDA-event ABBA comparison over pre-built real backend calls."""
    for _ in range(warmup):
        native_call()
        triton_call()
    torch.cuda.synchronize()
    entries: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(repetitions):
        for name, call in (
            ("native_a", native_call),
            ("triton_b", triton_call),
            ("triton_b", triton_call),
            ("native_a", native_call),
        ):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            stop.record()
            entries.append((name, start, stop))
    torch.cuda.synchronize()
    native_samples: list[float] = []
    triton_samples: list[float] = []
    cycles: list[dict[str, float]] = []
    for cycle in range(repetitions):
        row: dict[str, float] = {}
        cycle_entries = entries[cycle * 4 : cycle * 4 + 4]
        for position, (name, start, stop) in enumerate(cycle_entries):
            elapsed = float(start.elapsed_time(stop))
            row[f"{name}_{position}"] = elapsed
            (native_samples if name == "native_a" else triton_samples).append(elapsed)
        cycles.append(row)
    native = _latency_summary(native_samples)
    triton = _latency_summary(triton_samples)
    return {
        "protocol": "CUDA events; separate warmup excluded; ABBA=native,triton,triton,native; one default stream; input, metadata, output allocation, and oracle excluded",
        "warmup": warmup,
        "repetitions": repetitions,
        "raw_cycles_ms": cycles,
        "native": native,
        "triton": triton,
        "triton_over_native_median_speedup": triton["median_ms"]
        / native["median_ms"],
        "native_over_triton_median_latency_ratio": native["median_ms"]
        / triton["median_ms"],
        "ratio_interpretation": (
            "triton_over_native < 1 means native is slower; "
            "native_over_triton is the direct latency-regression factor"
        ),
        "pass": len(native_samples) == 2 * repetitions
        and len(triton_samples) == 2 * repetitions,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.timing_warmup < 10:
        raise ValueError("--timing-warmup must be at least 10")
    if args.timing_repetitions < 50:
        raise ValueError("--timing-repetitions must be at least 50")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name()
    if capability != (10, 3) or "B300" not in device_name.upper():
        raise RuntimeError(
            "requires B300 capability (10, 3), got "
            f"name={device_name!r}, capability={capability}"
        )
    if str(args.c2_root) not in sys.path:
        sys.path.insert(0, str(args.c2_root))

    import vllm
    from harness.data import make_decode_problem
    from harness.reference import dense_sparse_attention_reference
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.minimax_m3.nvidia.sparse_attention_msa import (
        MiniMaxM3SparseMSAImpl,
        MiniMaxM3SparseMSAMetadataBuilder,
        MiniMaxM3SparseNativeC2Backend,
    )
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode

    imported_vllm_root = Path(vllm.__file__).resolve().parent
    expected_vllm_root = (args.vllm_root / "vllm").resolve()
    if imported_vllm_root != expected_vllm_root:
        raise RuntimeError(
            "vLLM import did not use the derived checkout: "
            f"imported={imported_vllm_root}, expected={expected_vllm_root}"
        )

    raw_problem = make_decode_problem(
        batch_size=_BATCH,
        device="cuda",
        storage_dtype="fp8-scalar",
        seed=args.seed,
        decode_query_len=1,
        max_seq_len=4096,
    )
    problem = replace(
        raw_problem,
        topk_idx=torch.sort(raw_problem.topk_idx, dim=-1).values.contiguous(),
    )
    if not bool((problem.topk_idx[..., 1:] > problem.topk_idx[..., :-1]).all()):
        raise RuntimeError("top-k logical block indices must be strictly ascending")
    if (
        problem.k_scale is None
        or problem.v_scale is None
        or problem.k_scale.numel() != 1
        or problem.v_scale.numel() != 1
    ):
        raise RuntimeError("native C2 requires scalar FP8 K/V scales")

    spec = AttentionSpec(
        block_size=_PAGE_SIZE,
        num_kv_heads=_KV_HEADS,
        head_size=_HEAD_DIM,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    builder = MiniMaxM3SparseMSAMetadataBuilder(
        spec, ["native_c2_layer"], _config("native_c2"), torch.device("cuda")
    )
    metadata = builder.build(0, _common_metadata(problem, CommonAttentionMetadata))
    decode = metadata.decode
    plan_cache_size = len(builder.msa_cutlass_plan_cache.plans)
    builder_checks = {
        "decode_metadata_present": decode is not None,
        "native_backend_did_not_make_cutlass_plan": plan_cache_size == 0,
        "native_decode_has_no_cutlass_metadata": decode is not None
        and getattr(decode, "msa_cutlass", None) is None,
    }
    if not all(builder_checks.values()):
        raise RuntimeError(f"unexpected native builder state: {builder_checks}")
    triton_builder = MiniMaxM3SparseMSAMetadataBuilder(
        spec, ["triton_layer"], _config("triton"), torch.device("cuda")
    )
    triton_metadata = triton_builder.build(
        0, _common_metadata(problem, CommonAttentionMetadata)
    )
    triton_builder_checks = {
        "triton_decode_metadata_present": triton_metadata.decode is not None,
        "triton_plan_cache_empty": len(triton_builder.msa_cutlass_plan_cache.plans)
        == 0,
    }
    if not all(triton_builder_checks.values()):
        raise RuntimeError(f"unexpected Triton builder state: {triton_builder_checks}")

    query_fp8 = (problem.q.float() / _Q_SCALE).to(torch.float8_e4m3fn).contiguous()
    # Freeze one query tensor with exactly the BF16 values that the native op
    # stages from query_fp8.  Native still reads query_fp8 through its real ABI;
    # Triton receives those exact staged BF16 values, removing the prior query
    # value asymmetry without adding a bridge inside either timed call.
    shared_query_bf16 = (
        query_fp8.float() * _Q_SCALE
    ).to(torch.bfloat16).contiguous()
    token_major_topk = problem.topk_idx.permute(1, 0, 2).contiguous()
    if token_major_topk.shape != (_BATCH, _KV_HEADS, _TOPK):
        raise RuntimeError(f"bad native top-k ABI shape: {tuple(token_major_topk.shape)}")
    k_scale, v_scale = float(problem.k_scale.item()), float(problem.v_scale.item())
    native_layer = SimpleNamespace(
        layer_name="native_c2_layer",
        topk_indices_buffer=token_major_topk,
        _q_scale=torch.tensor(_Q_SCALE, device="cuda"),
        _k_scale=problem.k_scale,
        _v_scale=problem.v_scale,
        _q_scale_float=_Q_SCALE,
        _k_scale_float=k_scale,
        _v_scale_float=v_scale,
    )
    native_impl = MiniMaxM3SparseMSAImpl(
        _Q_HEADS,
        _HEAD_DIM,
        problem.sm_scale,
        _KV_HEADS,
        "fp8_e4m3",
        topk_blocks=_TOPK,
        sparse_block_size=_PAGE_SIZE,
        msa_decode_backend="native_c2",
    )
    native_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"native_c2_layer": metadata},
        slot_mapping={},
    )
    with override_forward_context(native_context):
        should_use = native_impl.should_use_msa_decode("native_c2_layer")
    static_checks = {
        "native_backend_alias": MiniMaxM3SparseNativeC2Backend.get_name()
        == "NATIVE_C2_MSA",
        "use_native_c2_decode": bool(native_impl.use_native_c2_decode),
        "use_cutlass_decode_is_false": not bool(native_impl.use_cutlass_decode),
        "should_use_msa_decode": should_use is True,
    }
    if not all(static_checks.values()):
        raise RuntimeError(f"native static selection failed: {static_checks}")

    output = torch.empty_like(problem.q)
    output_pointer = output.data_ptr()

    def call_forward(
        one_impl: Any,
        one_layer: Any,
        one_context: Any,
        one_query: torch.Tensor,
        one_output: torch.Tensor,
    ) -> torch.Tensor:
        pointer = one_output.data_ptr()
        with override_forward_context(one_context):
            result = one_impl.forward(
                one_layer,
                one_query,
                problem.kv_cache,
                one_output,
                query_fp8=query_fp8,
            )
        if result.data_ptr() != pointer:
            raise RuntimeError("impl.forward replaced its caller-owned output")
        return result

    def native_call() -> torch.Tensor:
        return call_forward(
            native_impl, native_layer, native_context, shared_query_bf16, output
        )

    # All input setup and oracle work stay outside both warmup and profiling.
    reference = _native_fp32_oracle(
        problem,
        query_fp8,
        q_scale=_Q_SCALE,
        k_scale=k_scale,
        v_scale=v_scale,
        reference=dense_sparse_attention_reference,
    )
    for _ in range(_WARMUP):
        native_call()
    torch.cuda.synchronize()

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        actual = native_call()
        # Keep the profiled CUDA work inside the profiler lifetime so the
        # kernel event is not dependent on implicit context-exit behavior.
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    profiling = _profile_checks(profiler, args.trace)
    correctness = _check_output(
        actual,
        reference,
        oracle=(
            "independent FP32 selected-page causal reference over exact "
            "native BF16-staged FP8 values"
        ),
    )
    caller_output = {
        "pointer_before": output_pointer,
        "pointer_after": actual.data_ptr(),
        "pointer_unchanged": actual.data_ptr() == output_pointer,
    }

    # The comparison path shares the packed KV, sparse metadata, scalar scales,
    # and exact staged query values.  It has separately-built metadata and
    # output storage so neither backend can reuse the other's mutable state.
    triton_layer = SimpleNamespace(
        layer_name="triton_layer",
        topk_indices_buffer=token_major_topk,
        _q_scale=torch.tensor(_Q_SCALE, device="cuda"),
        _k_scale=problem.k_scale,
        _v_scale=problem.v_scale,
        _q_scale_float=_Q_SCALE,
        _k_scale_float=k_scale,
        _v_scale_float=v_scale,
    )
    triton_impl = MiniMaxM3SparseMSAImpl(
        _Q_HEADS,
        _HEAD_DIM,
        problem.sm_scale,
        _KV_HEADS,
        "fp8_e4m3",
        topk_blocks=_TOPK,
        sparse_block_size=_PAGE_SIZE,
        msa_decode_backend="triton",
    )
    triton_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={"triton_layer": triton_metadata},
        slot_mapping={},
    )
    native_timing_output = torch.empty_like(problem.q)
    triton_timing_output = torch.empty_like(problem.q)

    def native_timing_call() -> torch.Tensor:
        return call_forward(
            native_impl,
            native_layer,
            native_context,
            shared_query_bf16,
            native_timing_output,
        )

    def triton_timing_call() -> torch.Tensor:
        return call_forward(
            triton_impl,
            triton_layer,
            triton_context,
            shared_query_bf16,
            triton_timing_output,
        )

    shared_query_problem = replace(problem, q=shared_query_bf16)
    triton_reference = dense_sparse_attention_reference(shared_query_problem)
    triton_actual = triton_timing_call()
    torch.cuda.synchronize()
    triton_correctness = _check_output(
        triton_actual,
        triton_reference,
        oracle=(
            "independent FP32 selected-page causal reference over the "
            "shared native-staged BF16 query/scalar-FP8-KV Triton inputs"
        ),
    )

    pre_timing_checks = {
        "native_hard_gates": bool(
            all(static_checks.values())
            and all(builder_checks.values())
            and caller_output["pointer_unchanged"]
            and correctness["pass"]
            and profiling["pass"]
        ),
        "triton_static_selection": not bool(triton_impl.use_native_c2_decode)
        and not bool(triton_impl.use_cutlass_decode),
        "triton_builder": all(triton_builder_checks.values()),
        "triton_correctness": triton_correctness["pass"],
        "distinct_output_buffers": native_timing_output.data_ptr()
        != triton_timing_output.data_ptr(),
    }
    timing: dict[str, Any]
    if all(pre_timing_checks.values()):
        timing = _abba_timing(
            native_timing_call,
            triton_timing_call,
            warmup=args.timing_warmup,
            repetitions=args.timing_repetitions,
        )
        timing["pre_timing_checks"] = pre_timing_checks
        timing["caller_owned_output"] = {
            "native": True,
            "triton": True,
        }
        timing["caller_owned_output_verification"] = (
            "every warmup and timed call checks result.data_ptr() against "
            "its caller-owned output"
        )
        timing["comparison_input_semantics"] = (
            "native stages query_fp8 with q_scale and Triton receives the same "
            "BF16 query values; packed FP8 KV, scalar scales, sparse metadata, "
            "and caller-owned output shapes are shared"
        )
        timing["pass"] = bool(
            timing["pass"] and all(timing["caller_owned_output"].values())
        )
    else:
        timing = {
            "pass": False,
            "skipped": True,
            "reason": "native correctness/profiler gate or fair-comparison setup failed",
            "pre_timing_checks": pre_timing_checks,
        }

    gates = {
        "native_static_selection": all(static_checks.values()),
        "no_cutlass_plan": all(builder_checks.values()),
        "caller_owned_output": caller_output["pointer_unchanged"],
        "fp32_oracle_correctness": correctness["pass"],
        "one_profiled_forward_reaches_dispatcher_and_cuda_kernel": profiling["pass"],
        "fair_native_vs_triton_abba_timing": timing["pass"],
    }
    return {
        "schema": _SCHEMA,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "boundary": "real MiniMaxM3SparseMSAImpl.forward backend layer; no model weights or server scheduler",
        "environment": {
            "torch": torch.__version__,
            "vllm_import_root": str(imported_vllm_root),
            "device": device_name,
            "compute_capability": list(capability),
        },
        "data_contract": {
            "batch": _BATCH,
            "query_heads": _Q_HEADS,
            "kv_heads": _KV_HEADS,
            "head_dim": _HEAD_DIM,
            "page_size": _PAGE_SIZE,
            "topk": _TOPK,
            "q_scale": _Q_SCALE,
            "k_scale": k_scale,
            "v_scale": v_scale,
            "input_checksums": {
                "q_bf16_original_not_timed": _tensor_sha(problem.q),
                "q_bf16_shared_native_staging": _tensor_sha(shared_query_bf16),
                "query_fp8": _tensor_sha(query_fp8),
                "packed_kv_fp8": _tensor_sha(problem.kv_cache),
                "block_table_i32": _tensor_sha(problem.block_table),
                "topk_token_major_i32": _tensor_sha(token_major_topk),
                "seq_lens_i32": _tensor_sha(problem.seq_lens),
            },
        },
        "static_selection": static_checks,
        "builder": {
            "checks": builder_checks,
            "cutlass_plan_cache_size": plan_cache_size,
            "triton_checks": triton_builder_checks,
            "triton_cutlass_plan_cache_size": len(triton_builder.msa_cutlass_plan_cache.plans),
        },
        "caller_output": caller_output,
        "correctness": correctness,
        "triton_correctness": triton_correctness,
        "profiling": profiling,
        "timing": timing,
        "warmup_calls": _WARMUP,
        "profiled_forward_calls": 1,
        "seed": args.seed,
    }


def main() -> int:
    args = _parse_args()
    for field in ("c2_root", "vllm_root", "output", "trace"):
        setattr(args, field, getattr(args, field).resolve())
    try:
        result = _run(args)
    except Exception as error:
        result = {
            "schema": _SCHEMA,
            "all_gates_pass": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
    _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("all_gates_pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
