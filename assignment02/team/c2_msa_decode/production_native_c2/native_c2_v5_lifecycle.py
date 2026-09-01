#!/usr/bin/env python3
"""Fail-closed production-lifecycle harness for the v5 native-C2 plugin.

This program intentionally exercises the installed ``native_c2`` adapter and
the real ``MiniMaxM3SparseMSAImpl.forward`` path.  It is not a direct-op
microbenchmark and it does not monkeypatch a vLLM function.  The supplied
``--base-harness`` is a frozen copy of ``native_c2_full_backend_bench.py``;
this harness imports its production fixture helpers, scalar contract, FP32
oracle, output check, and input checksum implementation instead of creating a
second attention model.

The JSON result is fail-closed: every listed lifecycle gate must be ``true``
for the process to exit successfully.  The scope is a B300 decode backend
layer, not a full model/server/scheduler integration test and not a proof of
Triton fallback execution.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from types import ModuleType, SimpleNamespace
from typing import Any

import torch


_SCHEMA = "c2-native-c2-v5-production-lifecycle-v2"
_NATIVE_OP = "_C::native_c2_msa_decode"
_THREAD_COUNT = 8
_MIN_STEADY_ITERATIONS = 1000
_MIN_GRAPH_REPLAYS = 100
_DEFAULT_MEMORY_BOUND_BYTES = 8 * 1024 * 1024


@dataclass
class _Fixture:
    """One exact-shape real-forward fixture and its independent oracle."""

    problem: Any
    metadata: Any
    impl: Any
    layer: Any
    context: Any
    query_fp8: torch.Tensor
    shared_query_bf16: torch.Tensor
    output: torch.Tensor
    reference: torch.Tensor
    static_checks: dict[str, bool]
    builder_checks: dict[str, bool]
    k_scale: float
    v_scale: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-root", required=True, type=Path)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument(
        "--base-harness",
        required=True,
        type=Path,
        help="absolute frozen native_c2_full_backend_bench.py",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--steady-iterations", type=int, default=_MIN_STEADY_ITERATIONS)
    parser.add_argument("--graph-replays", type=int, default=_MIN_GRAPH_REPLAYS)
    parser.add_argument("--memory-bound-bytes", type=int, default=_DEFAULT_MEMORY_BOUND_BYTES)
    return parser.parse_args()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _load_base_harness(path: Path) -> ModuleType:
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError(f"--base-harness must be an absolute file: {path}")
    spec = importlib.util.spec_from_file_location("native_c2_v5_lifecycle_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen base harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "_BATCH",
        "_Q_HEADS",
        "_KV_HEADS",
        "_HEAD_DIM",
        "_PAGE_SIZE",
        "_TOPK",
        "_Q_SCALE",
        "_config",
        "_common_metadata",
        "_native_fp32_oracle",
        "_check_output",
        "_tensor_sha",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"frozen base harness lacks required helpers: {missing}")
    return module


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gpu_identity() -> dict[str, Any]:
    """Record scheduler-visible UUIDs without guessing a host GPU mapping."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        uuids = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        error = None
    except Exception as exc:  # Environment evidence must not hide a failed run.
        uuids = []
        error = f"{type(exc).__name__}: {exc}"
    return {
        "cuda_device": "cuda:0",
        "name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_visible_uuids": uuids,
        "nvidia_smi_error": error,
    }


def _schema_key(schema: Any) -> str:
    return f"{schema.name}.{schema.overload_name}" if schema.overload_name else schema.name


def _dispatch_surface() -> dict[str, Any]:
    """Snapshot the complete private ``_C`` schema/op surface for equality checks."""
    required = (
        "_dispatch_get_all_op_names",
        "_dispatch_dump_table",
        "_jit_get_all_schemas",
        "_dispatch_has_kernel_for_dispatch_key",
    )
    if not all(callable(getattr(torch._C, name, None)) for name in required):
        raise RuntimeError("required PyTorch dispatcher inspection API is unavailable")
    ops = sorted(name for name in torch._C._dispatch_get_all_op_names() if name.startswith("_C::"))
    schemas: dict[str, list[str]] = {}
    for schema in torch._C._jit_get_all_schemas():
        if schema.name.startswith("_C::"):
            schemas.setdefault(_schema_key(schema), []).append(str(schema))
    schemas = {name: sorted(rows) for name, rows in sorted(schemas.items())}
    if set(ops) != set(schemas):
        raise RuntimeError("_C dispatcher op/schema surfaces disagree")
    dispatch: dict[str, dict[str, str]] = {}
    for op in ops:
        rows: dict[str, str] = {}
        for line in torch._C._dispatch_dump_table(op).splitlines():
            key, separator, registration = line.partition(": ")
            if not separator:
                continue
            # Preserve the complete registration record (implementation/source
            # plus registration kind), not merely its trailing ``[kernel]``
            # tag.  Same-process equality can then detect replacement of an
            # existing implementation at the same dispatch key.
            rows[key] = registration
        if not rows:
            raise RuntimeError(f"dispatcher table is empty for {op}")
        dispatch[op] = rows
    native_cuda = bool(
        _NATIVE_OP in ops
        and torch._C._dispatch_has_kernel_for_dispatch_key(_NATIVE_OP, "CUDA")
    )
    return {
        "ops": ops,
        "schemas": schemas,
        "dispatch_table_rows": dispatch,
        "op_count": len(ops),
        "native_cuda_registered": native_cuda,
    }


def _compact_surface(surface: dict[str, Any]) -> dict[str, Any]:
    """Keep enough evidence to audit the diff without copying every schema twice."""
    return {
        "op_count": surface["op_count"],
        "native_cuda_registered": surface["native_cuda_registered"],
        "native_schema": surface["schemas"].get(_NATIVE_OP, []),
        "ops_sha256": hashlib.sha256(
            "\n".join(surface["ops"]).encode("utf-8")
        ).hexdigest(),
        "schemas_sha256": hashlib.sha256(
            json.dumps(surface["schemas"], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "dispatch_sha256": hashlib.sha256(
            json.dumps(
                surface["dispatch_table_rows"], sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
    }


def _run_concurrent_first_load(adapter: ModuleType) -> dict[str, Any]:
    """Synchronize eight first callers and count the real DSO load boundary."""
    barrier = threading.Barrier(_THREAD_COUNT)
    values: list[bool | None] = [None] * _THREAD_COUNT
    errors: list[str | None] = [None] * _THREAD_COUNT
    load_library_calls = 0
    load_library_call_lock = threading.Lock()
    original_load_library = torch.ops.load_library

    def counted_load_library(path: str) -> Any:
        nonlocal load_library_calls
        with load_library_call_lock:
            load_library_calls += 1
        # Release the GIL while the adapter lock is held so the other seven
        # barrier-released callers can reach and contend on that lock.  The
        # real torch loader is still invoked; only its call boundary is counted.
        time.sleep(0.1)
        return original_load_library(path)

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=30)
            values[index] = bool(adapter._load_native_c2_plugin_once())
        except Exception as exc:
            errors[index] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=worker, args=(index,), daemon=False) for index in range(_THREAD_COUNT)]
    torch.ops.load_library = counted_load_library
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
    finally:
        torch.ops.load_library = original_load_library
    alive = [index for index, thread in enumerate(threads) if thread.is_alive()]
    checks = {
        "exactly_eight_threads": len(threads) == _THREAD_COUNT,
        "all_threads_joined": not alive,
        "no_thread_exception": all(error is None for error in errors),
        "all_loader_results_true": all(value is True for value in values),
        "exactly_one_real_load_library_call": load_library_calls == 1,
    }
    return {
        "thread_count": _THREAD_COUNT,
        "barrier": "threading.Barrier(8), then adapter._load_native_c2_plugin_once",
        "instrumentation": "counted torch.ops.load_library with 0.1 s GIL-releasing delay before the real call",
        "load_library_call_count": load_library_calls,
        "values": values,
        "errors": errors,
        "alive_thread_indices": alive,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _first_load_gate(adapter: ModuleType) -> dict[str, Any]:
    before = _dispatch_surface()
    if _NATIVE_OP in before["ops"]:
        # The test has to start in a clean interpreter; otherwise it cannot
        # prove that the eight callers contended for the first loader action.
        raise RuntimeError(f"native op was already registered before first-load gate: {_NATIVE_OP}")
    concurrent = _run_concurrent_first_load(adapter)
    after = _dispatch_surface()
    added_ops = sorted(set(after["ops"]) - set(before["ops"]))
    removed_ops = sorted(set(before["ops"]) - set(after["ops"]))
    added_schema_keys = sorted(set(after["schemas"]) - set(before["schemas"]))
    removed_schema_keys = sorted(set(before["schemas"]) - set(after["schemas"]))
    unchanged_common_schemas = all(
        before["schemas"][key] == after["schemas"][key]
        for key in set(before["schemas"]) & set(after["schemas"])
    )
    unchanged_common_dispatch = all(
        before["dispatch_table_rows"][key]
        == after["dispatch_table_rows"][key]
        for key in set(before["ops"]) & set(after["ops"])
    )
    checks = {
        "native_op_absent_before_loader": _NATIVE_OP not in before["ops"],
        "eight_thread_first_load": concurrent["pass"],
        "only_native_op_added": added_ops == [_NATIVE_OP] and not removed_ops,
        "only_native_schema_added": added_schema_keys == [_NATIVE_OP] and not removed_schema_keys,
        "preexisting_schemas_unchanged": unchanged_common_schemas,
        "preexisting_dispatch_tables_unchanged": unchanged_common_dispatch,
        "native_cuda_kernel_registered": after["native_cuda_registered"],
    }
    return {
        "before": _compact_surface(before),
        "after": _compact_surface(after),
        "added_ops": added_ops,
        "removed_ops": removed_ops,
        "added_schema_keys": added_schema_keys,
        "removed_schema_keys": removed_schema_keys,
        "concurrent_loader": concurrent,
        "checks": checks,
        "pass": all(checks.values()),
        "post_load_surface": after,
    }


def _make_fixture(
    base: ModuleType,
    *,
    sequence_length: int,
    seed: int,
    imports: dict[str, Any],
) -> _Fixture:
    """Build a real MSA forward fixture with all requests at one exact length."""
    if sequence_length not in (2048, 2049, 4095, 4096):
        raise ValueError(f"unsupported lifecycle boundary case: {sequence_length}")
    make_decode_problem = imports["make_decode_problem"]
    dense_reference = imports["dense_reference"]
    MetadataBuilder = imports["MetadataBuilder"]
    CommonAttentionMetadata = imports["CommonAttentionMetadata"]
    AttentionSpec = imports["AttentionSpec"]
    KVQuantMode = imports["KVQuantMode"]
    Impl = imports["Impl"]
    ForwardContext = imports["ForwardContext"]
    override_forward_context = imports["override_forward_context"]
    NativeBackend = imports["NativeBackend"]

    raw_problem = make_decode_problem(
        batch_size=base._BATCH,
        device="cuda",
        storage_dtype="fp8-scalar",
        seed=seed,
        decode_query_len=1,
        max_seq_len=sequence_length,
    )
    # make_decode_problem samples each request from [topk * page_size,
    # max_seq_len], so force the requested boundary explicitly.  Also construct
    # a strictly increasing top-k set that contains the last visible logical
    # page; otherwise a larger seq_lens value could be metadata-only and never
    # exercise the 2048/2049 or 4095/4096 causal boundary.
    exact_seq_lens = torch.full_like(raw_problem.seq_lens, sequence_length)
    last_visible_page = (sequence_length - 1) // base._PAGE_SIZE
    logical_pages = torch.cat(
        (
            torch.arange(base._TOPK - 1, device="cuda", dtype=torch.int32),
            torch.tensor([last_visible_page], device="cuda", dtype=torch.int32),
        )
    )
    if logical_pages.unique().numel() != base._TOPK:
        raise RuntimeError("boundary top-k pages are not distinct")
    exact_topk = logical_pages.view(1, 1, -1).expand(
        base._KV_HEADS, base._BATCH, base._TOPK
    ).contiguous()
    problem = replace(
        raw_problem,
        seq_lens=exact_seq_lens,
        topk_idx=exact_topk,
    )
    if not bool((problem.topk_idx[..., 1:] > problem.topk_idx[..., :-1]).all().item()):
        raise RuntimeError("top-k logical pages must be strictly ascending")
    if not bool((problem.topk_idx[..., -1] == last_visible_page).all().item()):
        raise RuntimeError("boundary top-k does not include the last visible page")
    if (
        problem.k_scale is None
        or problem.v_scale is None
        or problem.k_scale.numel() != 1
        or problem.v_scale.numel() != 1
    ):
        raise RuntimeError("fixture requires scalar FP8 K/V scales")

    spec = AttentionSpec(
        block_size=base._PAGE_SIZE,
        num_kv_heads=base._KV_HEADS,
        head_size=base._HEAD_DIM,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    layer_name = f"native_c2_lifecycle_len_{sequence_length}"
    builder = MetadataBuilder(spec, [layer_name], base._config("native_c2"), torch.device("cuda"))
    metadata = builder.build(0, base._common_metadata(problem, CommonAttentionMetadata))
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

    query_fp8 = (problem.q.float() / base._Q_SCALE).to(torch.float8_e4m3fn).contiguous()
    shared_query_bf16 = (query_fp8.float() * base._Q_SCALE).to(torch.bfloat16).contiguous()
    token_major_topk = problem.topk_idx.permute(1, 0, 2).contiguous()
    expected_topk_shape = (base._BATCH, base._KV_HEADS, base._TOPK)
    if tuple(token_major_topk.shape) != expected_topk_shape:
        raise RuntimeError(f"unexpected token-major top-k shape: {tuple(token_major_topk.shape)}")
    k_scale = float(problem.k_scale.item())
    v_scale = float(problem.v_scale.item())
    layer = SimpleNamespace(
        layer_name=layer_name,
        topk_indices_buffer=token_major_topk,
        _q_scale=torch.tensor(base._Q_SCALE, device="cuda"),
        _k_scale=problem.k_scale,
        _v_scale=problem.v_scale,
        _q_scale_float=base._Q_SCALE,
        _k_scale_float=k_scale,
        _v_scale_float=v_scale,
    )
    impl = Impl(
        base._Q_HEADS,
        base._HEAD_DIM,
        problem.sm_scale,
        base._KV_HEADS,
        "fp8_e4m3",
        topk_blocks=base._TOPK,
        sparse_block_size=base._PAGE_SIZE,
        msa_decode_backend="native_c2",
    )
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata={layer_name: metadata},
        slot_mapping={},
    )
    with override_forward_context(context):
        should_use = impl.should_use_msa_decode(layer_name)
    static_checks = {
        "native_backend_alias": NativeBackend.get_name() == "NATIVE_C2_MSA",
        "use_native_c2_decode": bool(impl.use_native_c2_decode),
        "use_cutlass_decode_is_false": not bool(impl.use_cutlass_decode),
        "should_use_msa_decode": should_use is True,
    }
    if not all(static_checks.values()):
        raise RuntimeError(f"native selection failed: {static_checks}")
    output = torch.empty_like(problem.q)
    reference = base._native_fp32_oracle(
        problem,
        query_fp8,
        q_scale=base._Q_SCALE,
        k_scale=k_scale,
        v_scale=v_scale,
        reference=dense_reference,
    )
    return _Fixture(
        problem=problem,
        metadata=metadata,
        impl=impl,
        layer=layer,
        context=context,
        query_fp8=query_fp8,
        shared_query_bf16=shared_query_bf16,
        output=output,
        reference=reference,
        static_checks=static_checks,
        builder_checks=builder_checks,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def _call_forward(fixture: _Fixture, override_forward_context: Any) -> tuple[torch.Tensor, bool]:
    pointer = fixture.output.data_ptr()
    with override_forward_context(fixture.context):
        result = fixture.impl.forward(
            fixture.layer,
            fixture.shared_query_bf16,
            fixture.problem.kv_cache,
            fixture.output,
            query_fp8=fixture.query_fp8,
        )
    return result, result.data_ptr() == pointer and fixture.output.data_ptr() == pointer


def _profile_native_forward(
    base: ModuleType, fixture: _Fixture, override_forward_context: Any
) -> dict[str, Any]:
    """Prove one actual ``forward`` reaches both dispatcher and native CUDA kernel."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        actual, pointer_unchanged = _call_forward(fixture, override_forward_context)
        torch.cuda.synchronize()
    events = list(profiler.key_averages())
    dispatcher = [event for event in events if str(event.key) == _NATIVE_OP]
    kernels = [event for event in events if "native_c2_msa_decode_kernel" in str(event.key)]
    correctness = base._check_output(
        actual,
        fixture.reference,
        oracle="frozen base FP32 selected-page oracle over exact native BF16-staged FP8 inputs",
    )
    checks = {
        "one_dispatcher_event": len(dispatcher) == 1 and int(getattr(dispatcher[0], "count", 0)) == 1,
        "one_native_cuda_kernel_event": len(kernels) == 1 and int(getattr(kernels[0], "count", 0)) == 1,
        "caller_owned_output_pointer": pointer_unchanged,
        "oracle_correct": bool(correctness["pass"]),
    }
    return {
        "checks": checks,
        "dispatcher_events": [{"key": str(event.key), "count": int(getattr(event, "count", 0))} for event in dispatcher],
        "cuda_kernel_events": [{"key": str(event.key), "count": int(getattr(event, "count", 0))} for event in kernels],
        "correctness": correctness,
        "pass": all(checks.values()),
    }


def _memory_summary(samples: list[int], *, baseline: int, bound: int) -> dict[str, Any]:
    if not samples:
        raise RuntimeError("memory sample set is empty")
    minimum = min(samples)
    maximum = max(samples)
    checks = {
        "within_absolute_bound_from_baseline": maximum - baseline <= bound,
        "within_observed_range_bound": maximum - minimum <= bound,
    }
    return {
        "baseline_bytes": baseline,
        "min_bytes": minimum,
        "max_bytes": maximum,
        "max_minus_baseline_bytes": maximum - baseline,
        "max_minus_min_bytes": maximum - minimum,
        "bound_bytes": bound,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _cuda_memory_snapshot() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
    }


def _cuda_memory_counter_summary(
    before: dict[str, int], after: dict[str, int], *, bound: int
) -> dict[str, Any]:
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    growth = {
        "end_allocated_bytes": max(0, after["allocated_bytes"] - before["allocated_bytes"]),
        "peak_allocated_bytes": max(0, peak_allocated - before["allocated_bytes"]),
        "end_reserved_bytes": max(0, after["reserved_bytes"] - before["reserved_bytes"]),
        "peak_reserved_bytes": max(0, peak_reserved - before["reserved_bytes"]),
        "device_free_drop_bytes": max(0, before["device_free_bytes"] - after["device_free_bytes"]),
    }
    checks = {
        "device_total_unchanged": before["device_total_bytes"] == after["device_total_bytes"],
        "end_allocated_growth_bounded": growth["end_allocated_bytes"] <= bound,
        "peak_allocated_growth_bounded": growth["peak_allocated_bytes"] <= bound,
        "end_reserved_growth_bounded": growth["end_reserved_bytes"] <= bound,
        "peak_reserved_growth_bounded": growth["peak_reserved_bytes"] <= bound,
        "device_free_drop_bounded": growth["device_free_drop_bytes"] <= bound,
    }
    return {
        "before": before,
        "after": after,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "positive_growth": growth,
        "bound_bytes": bound,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _steady_state_gate(
    base: ModuleType,
    fixture: _Fixture,
    override_forward_context: Any,
    *,
    iterations: int,
    memory_bound_bytes: int,
) -> dict[str, Any]:
    """Run the same real fixture at least 1000 times without a new output buffer."""
    for _ in range(3):
        _, pointer_unchanged = _call_forward(fixture, override_forward_context)
        if not pointer_unchanged:
            raise RuntimeError("warmup forward replaced caller-owned output")
    torch.cuda.synchronize()
    canonical = fixture.output.clone()
    canonical_check = base._check_output(
        canonical,
        fixture.reference,
        oracle="frozen base FP32 oracle before 1000-call lifecycle loop",
    )
    torch.cuda.synchronize()
    baseline_memory = int(torch.cuda.memory_allocated())
    memory_counters_before = _cuda_memory_snapshot()
    torch.cuda.reset_peak_memory_stats()
    pointer_all = True
    bitwise_all = True
    oracle_all = bool(canonical_check["pass"])
    max_abs = float(canonical_check["max_abs"])
    max_mean_abs = float(canonical_check["mean_abs"])
    memory_samples: list[int] = []
    output_checksums: list[str] = []
    for index in range(iterations):
        actual, pointer_unchanged = _call_forward(fixture, override_forward_context)
        torch.cuda.synchronize()
        pointer_all = pointer_all and pointer_unchanged
        bitwise_all = bitwise_all and bool(torch.equal(actual, canonical))
        correctness = base._check_output(
            actual,
            fixture.reference,
            oracle="frozen base FP32 oracle on every lifecycle forward",
        )
        oracle_all = oracle_all and bool(correctness["pass"])
        max_abs = max(max_abs, float(correctness["max_abs"]))
        max_mean_abs = max(max_mean_abs, float(correctness["mean_abs"]))
        memory_samples.append(int(torch.cuda.memory_allocated()))
        if index in (0, iterations - 1):
            output_checksums.append(base._tensor_sha(actual))
    memory = _memory_summary(
        memory_samples, baseline=baseline_memory, bound=memory_bound_bytes
    )
    memory_counters_after = _cuda_memory_snapshot()
    memory_counters = _cuda_memory_counter_summary(
        memory_counters_before,
        memory_counters_after,
        bound=memory_bound_bytes,
    )
    checks = {
        "at_least_1000_real_forward_calls": iterations >= _MIN_STEADY_ITERATIONS,
        "caller_owned_output_pointer_every_call": pointer_all,
        "bitwise_repeatable_every_call": bitwise_all,
        "fp32_oracle_every_call": oracle_all,
        "memory_allocated_samples_bounded": memory["pass"],
        "allocated_reserved_peak_and_device_free_bounded": memory_counters["pass"],
    }
    return {
        "iterations": iterations,
        "checks": checks,
        "memory_allocated": memory,
        "cuda_memory_counters": memory_counters,
        "canonical_output_checksum": base._tensor_sha(canonical),
        "first_and_last_output_checksums": output_checksums,
        "max_abs_over_calls": max_abs,
        "max_mean_abs_over_calls": max_mean_abs,
        "pass": all(checks.values()),
    }


def _run_graph_replays(
    base: ModuleType,
    graph: torch.cuda.CUDAGraph,
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    replays: int,
    memory_bound_bytes: int,
    label: str,
) -> dict[str, Any]:
    """Replay a captured real forward and check its output on every replay."""
    torch.cuda.synchronize()
    baseline_memory = int(torch.cuda.memory_allocated())
    memory_counters_before = _cuda_memory_snapshot()
    torch.cuda.reset_peak_memory_stats()
    canonical: torch.Tensor | None = None
    pointer = output.data_ptr()
    pointer_all = True
    bitwise_all = True
    oracle_all = True
    max_abs = 0.0
    max_mean_abs = 0.0
    samples: list[int] = []
    checksums: list[str] = []
    for index in range(replays):
        graph.replay()
        torch.cuda.synchronize()
        pointer_all = pointer_all and output.data_ptr() == pointer
        if canonical is None:
            canonical = output.clone()
        else:
            bitwise_all = bitwise_all and bool(torch.equal(output, canonical))
        correctness = base._check_output(output, reference, oracle=label)
        oracle_all = oracle_all and bool(correctness["pass"])
        max_abs = max(max_abs, float(correctness["max_abs"]))
        max_mean_abs = max(max_mean_abs, float(correctness["mean_abs"]))
        samples.append(int(torch.cuda.memory_allocated()))
        if index in (0, replays - 1):
            checksums.append(base._tensor_sha(output))
    assert canonical is not None
    memory = _memory_summary(samples, baseline=baseline_memory, bound=memory_bound_bytes)
    memory_counters_after = _cuda_memory_snapshot()
    memory_counters = _cuda_memory_counter_summary(
        memory_counters_before,
        memory_counters_after,
        bound=memory_bound_bytes,
    )
    checks = {
        "at_least_100_replays": replays >= _MIN_GRAPH_REPLAYS,
        "caller_owned_output_pointer_every_replay": pointer_all,
        "bitwise_repeatable_every_replay": bitwise_all,
        "fp32_oracle_every_replay": oracle_all,
        "memory_allocated_samples_bounded": memory["pass"],
        "allocated_reserved_peak_and_device_free_bounded": memory_counters["pass"],
    }
    return {
        "replays": replays,
        "checks": checks,
        "memory_allocated": memory,
        "cuda_memory_counters": memory_counters,
        "canonical_output_checksum": base._tensor_sha(canonical),
        "first_and_last_output_checksums": checksums,
        "max_abs_over_replays": max_abs,
        "max_mean_abs_over_replays": max_mean_abs,
        "pass": all(checks.values()),
    }


def _cuda_graph_gate(
    base: ModuleType,
    fixture: _Fixture,
    imports: dict[str, Any],
    *,
    seed: int,
    replays: int,
    memory_bound_bytes: int,
) -> dict[str, Any]:
    """Capture/replay the real forward, then prove static input mutation is read."""
    override_forward_context = imports["override_forward_context"]
    dense_reference = imports["dense_reference"]
    # Populate non-graph lazy state before capture; nothing in this warmup is a
    # substitute for the captured/replayed checks below.
    _call_forward(fixture, override_forward_context)
    torch.cuda.synchronize()
    graph_output = torch.empty_like(fixture.output)
    original_output = fixture.output
    fixture.output = graph_output
    query_fp8_pointer = fixture.query_fp8.data_ptr()
    query_bf16_pointer = fixture.shared_query_bf16.data_ptr()
    graph = torch.cuda.CUDAGraph()
    try:
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            result, pointer_unchanged_capture = _call_forward(fixture, override_forward_context)
            if not pointer_unchanged_capture or result.data_ptr() != graph_output.data_ptr():
                raise RuntimeError("captured forward did not preserve caller-owned output")
        torch.cuda.synchronize()
        old_reference = fixture.reference
        original_query_fp8_checksum = base._tensor_sha(fixture.query_fp8)
        original_shared_bf16_checksum = base._tensor_sha(fixture.shared_query_bf16)
        old_replays = _run_graph_replays(
            base,
            graph,
            graph_output,
            old_reference,
            replays=replays,
            memory_bound_bytes=memory_bound_bytes,
            label="frozen base FP32 oracle for captured original query",
        )

        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + 7919)
        replacement_fp8 = (
            torch.randn(
                fixture.query_fp8.shape,
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
            * 0.1
            / base._Q_SCALE
        ).to(torch.float8_e4m3fn).contiguous()
        if bool(torch.equal(replacement_fp8, fixture.query_fp8)):
            raise RuntimeError("query mutation generator unexpectedly reproduced static FP8 input")
        # These are in-place updates: CUDA Graph continues to read exactly the
        # captured static addresses.  The BF16 tensor is updated too because it
        # is the real forward argument and remains valid if the dispatch falls
        # through its native guard in a future implementation.
        fixture.query_fp8.copy_(replacement_fp8)
        fixture.shared_query_bf16.copy_(
            (fixture.query_fp8.float() * base._Q_SCALE).to(torch.bfloat16)
        )
        torch.cuda.synchronize()
        updated_reference = base._native_fp32_oracle(
            fixture.problem,
            fixture.query_fp8,
            q_scale=base._Q_SCALE,
            k_scale=fixture.k_scale,
            v_scale=fixture.v_scale,
            reference=dense_reference,
        )
        reference_changed = not bool(torch.equal(old_reference, updated_reference))
        updated_replays = _run_graph_replays(
            base,
            graph,
            graph_output,
            updated_reference,
            replays=replays,
            memory_bound_bytes=memory_bound_bytes,
            label="independent second FP32 oracle after in-place static-query update",
        )
        # A new-oracle pass alone would be weak if both references happened to
        # round identically.  Require both a changed oracle and a different
        # graph result from the old baseline checksum.
        graph_output_changed = (
            updated_replays["canonical_output_checksum"]
            != old_replays["canonical_output_checksum"]
        )
        checks = {
            "capture_preserved_output_pointer": pointer_unchanged_capture,
            "query_fp8_static_pointer_unchanged": fixture.query_fp8.data_ptr() == query_fp8_pointer,
            "shared_bf16_query_static_pointer_unchanged": fixture.shared_query_bf16.data_ptr() == query_bf16_pointer,
            "original_query_replays": old_replays["pass"],
            "in_place_query_update_second_oracle": updated_replays["pass"],
            "second_oracle_differs_from_first": reference_changed,
            "graph_output_changes_after_static_input_update": graph_output_changed,
        }
        return {
            "checks": checks,
            "capture": {
                "real_forward": "MiniMaxM3SparseMSAImpl.forward inside torch.cuda.graph",
                "query_fp8_pointer": query_fp8_pointer,
                "shared_bf16_query_pointer": query_bf16_pointer,
                "output_pointer": graph_output.data_ptr(),
            },
            "original_query": {
                "query_fp8_checksum": original_query_fp8_checksum,
                "shared_bf16_checksum": original_shared_bf16_checksum,
                "oracle_checksum": base._tensor_sha(old_reference),
                "replays": old_replays,
            },
            "updated_query": {
                "query_fp8_checksum": base._tensor_sha(fixture.query_fp8),
                "shared_bf16_checksum": base._tensor_sha(fixture.shared_query_bf16),
                "oracle_checksum": base._tensor_sha(updated_reference),
                "replays": updated_replays,
            },
            "pass": all(checks.values()),
        }
    finally:
        fixture.output = original_output


def _support_predicate_matrix(adapter: ModuleType, base: ModuleType) -> dict[str, Any]:
    """Exercise only the documented static selector; it does not claim fallback ran."""
    common = {
        "decode_backend": "native_c2",
        "num_q_heads": base._Q_HEADS,
        "num_kv_heads": base._KV_HEADS,
        "kv_cache_dtype": "fp8_e4m3",
        "page_size": base._PAGE_SIZE,
        "topk_blocks": base._TOPK,
    }
    definitions = [
        ("exact_native_contract", {}, True),
        ("triton_backend", {"decode_backend": "triton"}, False),
        ("wrong_query_heads", {"num_q_heads": base._Q_HEADS // 2}, False),
        ("wrong_kv_heads", {"num_kv_heads": base._KV_HEADS * 2}, False),
        ("unsupported_fp8_format", {"kv_cache_dtype": "fp8_e5m2"}, False),
        ("wrong_page_size", {"page_size": base._PAGE_SIZE // 2}, False),
        ("wrong_topk", {"topk_blocks": base._TOPK // 2}, False),
    ]
    rows: list[dict[str, Any]] = []
    for name, update, expected in definitions:
        arguments = dict(common)
        arguments.update(update)
        actual = bool(adapter.supports_native_c2_sparse_decode(**arguments))
        rows.append({"case": name, "arguments": arguments, "expected": expected, "actual": actual, "pass": actual is expected})
    return {
        "selector": "supports_native_c2_sparse_decode",
        "rows": rows,
        "fallback_boundary": (
            "False proves only that native selection rejects the contract. "
            "This harness does not claim that a vLLM Triton fallback kernel executed."
        ),
        "pass": all(row["pass"] for row in rows),
    }


def _dynamic_rejection_gate(adapter: ModuleType, fixture: _Fixture) -> dict[str, Any]:
    """Exercise every important runtime guard and prove rejection launches no native work.

    These calls intentionally stop at the native adapter's predicate.  They are
    *not* vLLM fallback tests: an unsupported input returning ``False`` gives
    the regular caller permission to choose Triton, but does not establish that
    a Triton kernel ran in this standalone harness.
    """
    topk = fixture.layer.topk_indices_buffer
    block_table = fixture.metadata.decode.block_table
    seq_lens = fixture.metadata.decode.seq_lens
    scale = float(fixture.problem.sm_scale)
    q_scale = float(getattr(fixture.layer, "_q_scale_float"))
    common = {
        "kv_cache": fixture.problem.kv_cache,
        "scale": scale,
        "q_scale": q_scale,
        "k_scale": fixture.k_scale,
        "v_scale": fixture.v_scale,
    }

    def batched_inputs(batch: int) -> dict[str, Any]:
        return {
            **common,
            # A finite sentinel is required here.  Comparing two clones of an
            # uninitialised CUDA allocation is not a valid no-write check:
            # either allocation may contain NaNs, for which torch.equal is
            # false even when every bit is unchanged.
            "output": torch.full_like(fixture.output[:batch], -123.5),
            "query_fp8": fixture.query_fp8[:batch].contiguous(),
            "topk": topk[:batch].contiguous(),
            "block_table": block_table[:batch].contiguous(),
            "seq_lens": seq_lens[:batch].contiguous(),
        }

    # A decode query length greater than one reaches the adapter as B*Q rows;
    # it must fail the fixed B=16 ABI predicate rather than launch native work.
    q2 = {
        **common,
        "output": torch.full_like(fixture.output.repeat(2, 1, 1), -123.5),
        "query_fp8": fixture.query_fp8.repeat(2, 1, 1).contiguous(),
        "topk": topk.repeat(2, 1, 1).contiguous(),
        "block_table": block_table.repeat(2, 1).contiguous(),
        "seq_lens": seq_lens.repeat(2).contiguous(),
    }
    exact = batched_inputs(16)
    definitions: list[tuple[str, dict[str, Any]]] = [
        ("batch_1", batched_inputs(1)),
        ("batch_4", batched_inputs(4)),
        ("batch_8", batched_inputs(8)),
        ("decode_query_len_2_rows_32", q2),
        ("topk_15", {**exact, "topk": topk[:, :, :15].contiguous()}),
        ("page_size_64", {**exact, "kv_cache": fixture.problem.kv_cache[:, :, :64, :].contiguous()}),
        ("query_dtype_bfloat16", {**exact, "query_fp8": fixture.query_fp8.to(torch.bfloat16)}),
        ("nonpositive_scale", {**exact, "scale": 0.0}),
    ]
    rows: list[dict[str, Any]] = []
    for name, kwargs in definitions:
        output_before = kwargs["output"].clone()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            selected = bool(adapter.native_c2_msa_decode(**kwargs))
            torch.cuda.synchronize()
        events = list(profiler.key_averages())
        dispatcher_events = [event for event in events if str(event.key) == _NATIVE_OP]
        native_kernels = [event for event in events if "native_c2_msa_decode_kernel" in str(event.key)]
        checks = {
            "native_selection_rejected": selected is False,
            "rejected_call_did_not_modify_output": bool(torch.equal(kwargs["output"], output_before)),
            "zero_native_dispatcher_events": len(dispatcher_events) == 0,
            "zero_native_cuda_kernel_events": len(native_kernels) == 0,
        }
        rows.append({
            "case": name,
            "checks": checks,
            "dispatcher_event_count": len(dispatcher_events),
            "native_cuda_kernel_event_count": len(native_kernels),
            "pass": all(checks.values()),
        })
    return {
        "mode": "selection_only",
        "fallback_boundary": "False is only a native selection/rejection result; Triton execution is not asserted.",
        "rows": rows,
        "pass": all(row["pass"] for row in rows),
    }


def _boundary_forward_cases(
    base: ModuleType,
    imports: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    override_forward_context = imports["override_forward_context"]
    for offset, sequence_length in enumerate((2048, 2049, 4095, 4096)):
        fixture = _make_fixture(
            base,
            sequence_length=sequence_length,
            seed=seed + 100 + offset,
            imports=imports,
        )
        profile = _profile_native_forward(
            base, fixture, override_forward_context
        )
        actual = fixture.output
        correctness = profile["correctness"]
        wrong_sequence_length = sequence_length - 1
        wrong_problem = replace(
            fixture.problem,
            seq_lens=torch.full_like(fixture.problem.seq_lens, wrong_sequence_length),
        )
        wrong_reference = base._native_fp32_oracle(
            wrong_problem,
            fixture.query_fp8,
            q_scale=base._Q_SCALE,
            k_scale=fixture.k_scale,
            v_scale=fixture.v_scale,
            reference=imports["dense_reference"],
        )
        negative_control = base._check_output(
            actual,
            wrong_reference,
            oracle=(
                "one-token-short FP32 negative control; this oracle must not "
                "pass the production tolerance"
            ),
        )
        exact_length = bool(torch.equal(
            fixture.problem.seq_lens,
            torch.full_like(fixture.problem.seq_lens, sequence_length),
        ))
        checks = {
            "exact_requested_sequence_length": exact_length,
            "native_static_selection": all(fixture.static_checks.values()),
            "native_metadata_without_cutlass_plan": all(fixture.builder_checks.values()),
            "profiled_native_dispatcher_and_kernel": profile["pass"],
            "caller_owned_output_pointer": profile["checks"][
                "caller_owned_output_pointer"
            ],
            "fp32_oracle_correctness": bool(correctness["pass"]),
            "one_token_boundary_is_tolerance_sensitive": not bool(
                negative_control["pass"]
            ),
        }
        rows.append({
            "sequence_length": sequence_length,
            "checks": checks,
            "correctness": correctness,
            "one_token_short_negative_control": {
                "sequence_length": wrong_sequence_length,
                "oracle_checksum": base._tensor_sha(wrong_reference),
                "comparison_against_actual": negative_control,
                "pass": not bool(negative_control["pass"]),
            },
            "profile": {
                "dispatcher_events": profile["dispatcher_events"],
                "cuda_kernel_events": profile["cuda_kernel_events"],
            },
            "input_checksums": {
                "query_fp8": base._tensor_sha(fixture.query_fp8),
                "kv_fp8": base._tensor_sha(fixture.problem.kv_cache),
                "topk_token_major": base._tensor_sha(fixture.layer.topk_indices_buffer),
                "seq_lens": base._tensor_sha(fixture.problem.seq_lens),
            },
            "output_checksum": base._tensor_sha(actual),
            "pass": all(checks.values()),
        })
    return {
        "execution_boundary": "Each row executes real MiniMaxM3SparseMSAImpl.forward. A selection pass is not a claimed Triton fallback test.",
        "rows": rows,
        "pass": all(row["pass"] for row in rows),
    }


def _imports_and_root_check(args: argparse.Namespace) -> tuple[dict[str, Any], ModuleType, ModuleType]:
    """Import only vLLM plus the adapter before the contested first load.

    In particular, this deliberately does *not* import the MSA implementation
    or invoke either support predicate before ``_first_load_gate``.  The
    dispatch snapshot directly below then proves the process began with no
    native schema, while the barrier makes the adapter loader's first action
    observable.
    """
    if str(args.c2_root) not in sys.path:
        sys.path.insert(0, str(args.c2_root))
    if str(args.vllm_root) not in sys.path:
        sys.path.insert(0, str(args.vllm_root))
    base = _load_base_harness(args.base_harness)
    import vllm

    expected_vllm_root = (args.vllm_root / "vllm").resolve()
    imported_vllm_root = Path(vllm.__file__).resolve().parent
    if imported_vllm_root != expected_vllm_root:
        raise RuntimeError(
            "vLLM import did not use the requested checkout: "
            f"imported={imported_vllm_root}, expected={expected_vllm_root}"
        )
    adapter = importlib.import_module("vllm.models.minimax_m3.nvidia.msa_native_c2_decode")
    required_adapter = (
        "_load_native_c2_plugin_once",
        "supports_native_c2_sparse_decode",
        "native_c2_msa_decode",
    )
    if any(not callable(getattr(adapter, name, None)) for name in required_adapter):
        raise RuntimeError("installed adapter lacks a required native-C2 lifecycle callable")
    return {"vllm_import_root": imported_vllm_root}, base, adapter


def _load_forward_imports(initial: dict[str, Any]) -> dict[str, Any]:
    """Load the real-forward dependencies only after the first-load gate."""
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
    return {
        "make_decode_problem": make_decode_problem,
        "dense_reference": dense_sparse_attention_reference,
        "ForwardContext": ForwardContext,
        "override_forward_context": override_forward_context,
        "Impl": MiniMaxM3SparseMSAImpl,
        "MetadataBuilder": MiniMaxM3SparseMSAMetadataBuilder,
        "NativeBackend": MiniMaxM3SparseNativeC2Backend,
        "CommonAttentionMetadata": CommonAttentionMetadata,
        "AttentionSpec": AttentionSpec,
        "KVQuantMode": KVQuantMode,
        "vllm_import_root": initial["vllm_import_root"],
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.steady_iterations < _MIN_STEADY_ITERATIONS:
        raise ValueError(f"--steady-iterations must be at least {_MIN_STEADY_ITERATIONS}")
    if args.graph_replays < _MIN_GRAPH_REPLAYS:
        raise ValueError(f"--graph-replays must be at least {_MIN_GRAPH_REPLAYS}")
    if args.memory_bound_bytes <= 0:
        raise ValueError("--memory-bound-bytes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    identity = _gpu_identity()
    if tuple(identity["compute_capability"]) != (10, 3) or "B300" not in identity["name"].upper():
        raise RuntimeError(f"requires B300 capability (10,3), got {identity}")
    initial_imports, base, adapter = _imports_and_root_check(args)
    first_load = _first_load_gate(adapter)
    if not first_load["pass"]:
        raise RuntimeError(f"adapter concurrent first-load gate failed: {first_load['checks']}")
    imports = _load_forward_imports(initial_imports)
    fixture = _make_fixture(base, sequence_length=4096, seed=args.seed, imports=imports)
    profile = _profile_native_forward(base, fixture, imports["override_forward_context"])
    steady = _steady_state_gate(
        base,
        fixture,
        imports["override_forward_context"],
        iterations=args.steady_iterations,
        memory_bound_bytes=args.memory_bound_bytes,
    )
    graph = _cuda_graph_gate(
        base,
        fixture,
        imports,
        seed=args.seed,
        replays=args.graph_replays,
        memory_bound_bytes=args.memory_bound_bytes,
    )
    support_matrix = _support_predicate_matrix(adapter, base)
    dynamic_rejection = _dynamic_rejection_gate(adapter, fixture)
    boundaries = _boundary_forward_cases(base, imports, seed=args.seed)
    after = _dispatch_surface()
    post_load_surface = first_load.pop("post_load_surface")
    dispatcher_stable_checks = {
        "op_names_unchanged_after_first_load": after["ops"] == post_load_surface["ops"],
        "schemas_unchanged_after_first_load": after["schemas"] == post_load_surface["schemas"],
        "complete_dispatch_tables_unchanged_after_first_load": (
            after["dispatch_table_rows"]
            == post_load_surface["dispatch_table_rows"]
        ),
        "native_cuda_kernel_still_registered": after["native_cuda_registered"],
    }
    dispatcher_stable = {
        "after_first_load": _compact_surface(post_load_surface),
        "after_all_lifecycle_checks": _compact_surface(after),
        "checks": dispatcher_stable_checks,
        "pass": all(dispatcher_stable_checks.values()),
    }
    gates = {
        "fresh_adapter_eight_thread_first_load": first_load["pass"],
        "real_forward_reaches_dispatcher_and_native_cuda_kernel": profile["pass"],
        "steady_1000_real_forward_lifecycle": steady["pass"],
        "cuda_graph_capture_replay_and_static_query_update": graph["pass"],
        "native_support_predicate_matrix": support_matrix["pass"],
        "dynamic_native_rejection_without_fallback_claim": dynamic_rejection["pass"],
        "native_forward_sequence_boundaries": boundaries["pass"],
        "dispatcher_surface_persists_through_lifecycle": dispatcher_stable["pass"],
    }
    return {
        "schema": _SCHEMA,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "boundary": {
            "covered": (
                "fresh adapter first load, real MiniMaxM3SparseMSAImpl.forward, "
                "caller-owned output, FP32 oracle, CUDA Graph replay/static input mutation, "
                "native selection matrix, and exact decode sequence-length boundaries"
            ),
            "not_covered": (
                "full model weights, scheduler/server integration, multi-process loading, "
                "and an assertion that an unsupported native shape actually executes Triton"
            ),
            "memory_scope": (
                "synchronized PyTorch allocated/reserved live and peak counters plus "
                "cudaMemGetInfo free-memory deltas; this is not a CUDA memory sanitizer"
            ),
        },
        "environment": {
            "gpu": identity,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm_import_root": str(imports["vllm_import_root"]),
            "base_harness": str(args.base_harness),
            "base_harness_sha256": _sha256_path(args.base_harness),
        },
        "arguments": {
            "seed": args.seed,
            "steady_iterations": args.steady_iterations,
            "graph_replays_per_static_query": args.graph_replays,
            "memory_bound_bytes": args.memory_bound_bytes,
        },
        "first_load": first_load,
        "profiled_real_forward": profile,
        "steady_lifecycle": steady,
        "cuda_graph": graph,
        "support_predicate_matrix": support_matrix,
        "dynamic_rejection_gate": dynamic_rejection,
        "sequence_length_boundaries": boundaries,
        "dispatcher_surface_after_lifecycle": dispatcher_stable,
    }


def main() -> int:
    args = _parse_args()
    for field in ("c2_root", "vllm_root", "base_harness", "output"):
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
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("all_gates_pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
