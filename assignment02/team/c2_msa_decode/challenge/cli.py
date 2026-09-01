"""Fair correctness, benchmark, graph replay, and final gate for C2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Callable

import torch
import triton

from harness.data import make_decode_problem
from harness.reference import dense_sparse_attention_reference
from harness.triton_baseline import run_triton_baseline_into
from .prepared_decode import PreparedSparseDecode, baseline_num_topk_chunks


BATCHES = (1, 4, 8, 16)
MODES = ("bf16", "fp8-scalar", "fp8-token")
GATE_SCHEMA = "c2-final-gate-v2-mode-aware"
RTOL = 3e-2
ATOL = 3e-2


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _chunks(value: str, storage_mode: str, batch: int) -> int | None:
    if value == "selected":
        return _selected_resolved_chunks(batch, storage_mode)
    return None if value == "auto" else int(value)


def _selected_resolved_chunks(batch: int, storage_mode: str) -> int:
    policy = {
        "bf16": {1: 1, 4: 1, 8: 1, 16: 1},
        "fp8-scalar": {1: 16, 4: 4, 8: 8, 16: 4},
        "fp8-token": {1: 4, 4: 16, 8: 16, 16: 4},
    }
    try:
        return policy[storage_mode][batch]
    except KeyError as exc:
        raise ValueError(f"selected policy has no entry for mode={storage_mode}, B={batch}") from exc


def _make(args: argparse.Namespace, batch: int, mode: str):
    return make_decode_problem(
        batch_size=batch,
        device="cuda",
        storage_dtype=mode,
        seed=args.seed + batch,
        max_seq_len=args.max_seq_len,
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return ordered[index]


def _summary(values: list[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_median_ms": statistics.median(values),
        f"{prefix}_p10_ms": _percentile(values, 0.10),
        f"{prefix}_p90_ms": _percentile(values, 0.90),
    }


def _warmup(function: Callable[[], object], count: int) -> None:
    for _ in range(count):
        function()
    torch.cuda.synchronize()


def _measure_single_step(
    function: Callable[[], object], *, warmup: int, samples: int
) -> dict[str, float | int | str]:
    """One call and one synchronization per independent host-latency sample."""
    _warmup(function, warmup)
    host_ms: list[float] = []
    cuda_ms: list[float] = []
    for _ in range(samples):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        host_start = time.perf_counter_ns()
        start_event.record()
        function()
        end_event.record()
        end_event.synchronize()
        host_ms.append((time.perf_counter_ns() - host_start) / 1e6)
        cuda_ms.append(start_event.elapsed_time(end_event))
    return {
        "latency_protocol": "one_call_then_device_synchronize_per_sample",
        "latency_samples": samples,
        **_summary(host_ms, "single_step_host_latency"),
        **_summary(cuda_ms, "single_step_cuda_latency"),
    }


def _measure_steady_state_cuda(
    function: Callable[[], object], *, warmup: int, samples: int, inner: int
) -> dict[str, float | int | str]:
    """Grouped CUDA-event timing; explicitly not a host single-step metric."""
    _warmup(function, warmup)
    per_decode_ms: list[float] = []
    for _ in range(samples):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(inner):
            function()
        end_event.record()
        end_event.synchronize()
        per_decode_ms.append(start_event.elapsed_time(end_event) / inner)
    median_ms = statistics.median(per_decode_ms)
    return {
        "throughput_protocol": "steady_state_grouped_cuda_events",
        "throughput_samples": samples,
        "throughput_inner_calls": inner,
        **_summary(per_decode_ms, "steady_state_cuda_time_per_decode"),
        "steady_state_cuda_decodes_per_second": 1000.0 / median_ms,
    }


def _measure(
    function: Callable[[], object], *, warmup: int, samples: int, inner: int
) -> dict[str, float | int | str]:
    return {
        **_measure_single_step(function, warmup=warmup, samples=samples),
        **_measure_steady_state_cuda(
            function, warmup=warmup, samples=samples, inner=inner
        ),
    }


def _check_once(
    function: Callable[[], torch.Tensor], output: torch.Tensor, expected: torch.Tensor
) -> None:
    returned = function()
    if returned.data_ptr() != output.data_ptr():
        raise RuntimeError("runner did not return the caller-owned output")
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=RTOL, atol=ATOL)


def _capture_once(function: Callable[[], object]) -> torch.cuda.CUDAGraph:
    """Capture one decode after all JIT/allocation warmup has completed."""
    _warmup(function, 3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    torch.cuda.synchronize()
    return graph


def _graph_result(
    function: Callable[[], torch.Tensor],
    output: torch.Tensor,
    expected: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Capture once, validate one replay, then measure replay-only behavior."""
    try:
        graph = _capture_once(function)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=RTOL, atol=ATOL)
        result: dict[str, object] = {
            "execution_mode": "capture_once_replay_many",
            "capture_status": "pass",
            "correctness_checked_after_replay": True,
        }
        result.update(
            _measure(
                graph.replay,
                warmup=args.warmup,
                samples=args.samples,
                inner=args.inner,
            )
        )
        return result
    except RuntimeError as exc:
        return {
            "execution_mode": "capture_once_replay_many",
            "capture_status": "unsupported",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _new_runner(
    problem, args: argparse.Namespace, chunk_text: str
) -> tuple[PreparedSparseDecode, torch.Tensor]:
    output = torch.empty_like(problem.q)
    runner = PreparedSparseDecode(
        problem,
        output,
        num_topk_chunks=_chunks(chunk_text, problem.storage_dtype, problem.batch_size),
        decode_num_warps=args.decode_warps,
        merge_num_warps=args.merge_warps,
    )
    return runner, output


def _correctness_matrix(
    args: argparse.Namespace,
    *,
    batches: tuple[int, ...],
    modes: tuple[str, ...],
    chunks: tuple[str, ...],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for batch in batches:
        for mode in modes:
            problem = _make(args, batch, mode)
            expected = dense_sparse_attention_reference(problem)
            for chunk_text in chunks:
                runner, output = _new_runner(problem, args, chunk_text)
                _check_once(runner, output, expected)
                error = (output.float() - expected.float()).abs()
                results.append(
                    {
                        "batch": batch,
                        "storage": mode,
                        "chunks": runner.num_topk_chunks,
                        "decode_warps": args.decode_warps,
                        "merge_warps": args.merge_warps,
                        "max_abs": float(error.max()),
                        "mean_abs": float(error.mean()),
                        "status": "pass",
                    }
                )
    return results


def correctness(args: argparse.Namespace) -> int:
    results = _correctness_matrix(
        args,
        batches=BATCHES if args.all_batches else (args.batch,),
        modes=MODES if args.all_modes else (args.storage_mode,),
        chunks=tuple(args.chunks),
    )
    print(json.dumps({"challenge_correctness": results}, indent=2, ensure_ascii=False))
    return 0


def _environment() -> dict[str, object]:
    capability = torch.cuda.get_device_capability()
    return {
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": list(capability),
    }


def _source_hashes() -> dict[str, str]:
    c2_root = Path(__file__).resolve().parents[1]
    sources = {
        "challenge/cli.py": c2_root / "challenge" / "cli.py",
        "challenge/prepared_decode.py": c2_root / "challenge" / "prepared_decode.py",
        "harness/triton_baseline.py": c2_root / "harness" / "triton_baseline.py",
        "harness/data.py": c2_root / "harness" / "data.py",
        "harness/reference.py": c2_root / "harness" / "reference.py",
        "vllm_msa_ref/sparse_attn.py": c2_root / "vllm_msa_ref" / "sparse_attn.py",
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}


def _gate_config(args: argparse.Namespace, chunk_text: str) -> dict[str, object]:
    return {
        "chunk_policy": chunk_text,
        "chunk_policy_definition": {
            "batch_order": list(BATCHES),
            "bf16": [1, 1, 1, 1],
            "fp8-scalar": [16, 4, 8, 4],
            "fp8-token": [4, 16, 16, 4],
        },
        "decode_warps": args.decode_warps,
        "merge_warps": args.merge_warps,
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "batches": list(BATCHES),
        "storage_modes": list(MODES),
        "rtol": RTOL,
        "atol": ATOL,
    }


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def final_gate(args: argparse.Namespace) -> int:
    chunk_text = args.chunks[0]
    path = Path(args.manifest)
    config = _gate_config(args, chunk_text)
    base: dict[str, object] = {
        "schema": GATE_SCHEMA,
        "status": "in_progress",
        "config": config,
        "environment": _environment(),
        "source_sha256": _source_hashes(),
    }
    _write_manifest(path, base)
    try:
        results = _correctness_matrix(
            args, batches=BATCHES, modes=MODES, chunks=(chunk_text,)
        )
    except Exception as exc:
        _write_manifest(
            path,
            {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    manifest = {**base, "status": "pass", "results": results}
    _write_manifest(path, manifest)
    print(json.dumps({"final_gate": "pass", "manifest": str(path.resolve())}, ensure_ascii=False))
    return 0


def _load_gate(args: argparse.Namespace) -> tuple[Path, dict[str, object], str]:
    path = Path(args.manifest)
    if not path.is_file():
        raise RuntimeError("final benchmark requires an existing final-gate manifest")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != GATE_SCHEMA or manifest.get("status") != "pass":
        raise RuntimeError("manifest is not a passing C2 final gate")
    expected_config = _gate_config(args, args.chunks[0])
    if manifest.get("config") != expected_config:
        raise RuntimeError("manifest configuration does not match benchmark configuration")
    if manifest.get("environment") != _environment():
        raise RuntimeError("manifest GPU/software environment does not match this benchmark")
    if manifest.get("source_sha256") != _source_hashes():
        raise RuntimeError("manifest source hashes do not match the current implementation")
    results = manifest.get("results")
    if not isinstance(results, list):
        raise RuntimeError("manifest results are missing")
    covered = {
        (item.get("batch"), item.get("storage"))
        for item in results
        if isinstance(item, dict) and item.get("status") == "pass"
    }
    required = {(batch, mode) for batch in BATCHES for mode in MODES}
    if covered != required or len(results) != len(required):
        raise RuntimeError("manifest does not cover exactly B={1,4,8,16} x all modes")
    expected_decode_warps = expected_config["decode_warps"]
    expected_merge_warps = expected_config["merge_warps"]
    if any(
        item.get("chunks")
        != _selected_resolved_chunks(int(item.get("batch")), str(item.get("storage")))
        or item.get("decode_warps") != expected_decode_warps
        or item.get("merge_warps") != expected_merge_warps
        for item in results
    ):
        raise RuntimeError("manifest rows do not match the gated launch configuration")
    return path, manifest, hashlib.sha256(raw).hexdigest()


def _benchmark(
    args: argparse.Namespace,
    *,
    batches: tuple[int, ...],
    evidence_class: str,
    gate: dict[str, object] | None = None,
) -> int:
    results: list[dict[str, object]] = []
    for batch in batches:
        problem = _make(args, batch, args.storage_mode)
        expected = dense_sparse_attention_reference(problem)
        baseline_output = torch.empty_like(problem.q)

        def baseline_call() -> torch.Tensor:
            return run_triton_baseline_into(problem, baseline_output)

        _check_once(baseline_call, baseline_output, expected)
        baseline_chunks = baseline_num_topk_chunks(
            problem.q.shape[0], problem.num_kv_heads, problem.topk_idx.shape[-1]
        )
        baseline_result: dict[str, object] = {
            "implementation": "baseline_caller_owned_output",
            "execution_mode": "eager",
            "batch": batch,
            "storage": args.storage_mode,
            "chunks": baseline_chunks,
            "caller_output_preallocated_outside_timing": True,
            "correctness_checked_before_timing": True,
            "evidence_class": evidence_class,
        }
        baseline_result.update(
            _measure(
                baseline_call,
                warmup=args.warmup,
                samples=args.samples,
                inner=args.inner,
            )
        )
        results.append(baseline_result)
        if not args.no_cudagraph:
            results.append(
                {
                    **baseline_result,
                    **_graph_result(baseline_call, baseline_output, expected, args=args),
                }
            )

        for chunk_text in args.chunks:
            runner, output = _new_runner(problem, args, chunk_text)
            _check_once(runner, output, expected)
            item: dict[str, object] = {
                "implementation": "prepared_persistent_workspace",
                "execution_mode": "eager",
                "batch": batch,
                "storage": args.storage_mode,
                "caller_output_preallocated_outside_timing": True,
                "correctness_checked_before_timing": True,
                "evidence_class": evidence_class,
                **runner.metadata.__dict__,
            }
            item.update(
                _measure(
                    runner,
                    warmup=args.warmup,
                    samples=args.samples,
                    inner=args.inner,
                )
            )
            results.append(item)
            if not args.no_cudagraph:
                results.append(
                    {
                        **item,
                        **_graph_result(runner, output, expected, args=args),
                    }
                )
    payload: dict[str, object] = {
        "evidence_class": evidence_class,
        "challenge_benchmark": results,
    }
    if gate is not None:
        payload["correctness_gate"] = gate
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def benchmark(args: argparse.Namespace) -> int:
    return _benchmark(
        args,
        batches=BATCHES if args.all_batches else (args.batch,),
        evidence_class="exploratory_ungated_not_for_final_claim",
    )


def final_benchmark(args: argparse.Namespace) -> int:
    path, _, digest = _load_gate(args)
    return _benchmark(
        args,
        batches=BATCHES,
        evidence_class="final_correctness_gated",
        gate={"manifest": str(path.resolve()), "sha256": digest},
    )


def profile(args: argparse.Namespace) -> int:
    problem = _make(args, args.batch, args.storage_mode)
    expected = dense_sparse_attention_reference(problem)
    runner, output = _new_runner(problem, args, args.chunks[0])
    _check_once(runner, output, expected)
    _warmup(runner, args.warmup)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as profiler:
        for _ in range(args.profile_steps):
            runner()
        torch.cuda.synchronize()
    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    if args.trace:
        path = Path(args.trace)
        path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(path))
        print(json.dumps({"trace": str(path.resolve())}, ensure_ascii=False))
    return 0


def once(args: argparse.Namespace) -> int:
    problem = _make(args, args.batch, args.storage_mode)
    runner, output = _new_runner(problem, args, args.chunks[0])
    runner()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "complete",
                "shape": list(output.shape),
                "finite": bool(output.float().isfinite().all()),
                **runner.metadata.__dict__,
            },
            ensure_ascii=False,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=(
            "dry-run",
            "correctness",
            "benchmark",
            "final-gate",
            "final-benchmark",
            "profile",
            "once",
        ),
    )
    result.add_argument("--batch", type=int, choices=BATCHES, default=1)
    result.add_argument("--all-batches", action="store_true")
    result.add_argument("--storage-mode", choices=MODES, default="bf16")
    result.add_argument("--all-modes", action="store_true")
    result.add_argument(
        "--chunks",
        nargs="+",
        choices=("selected", "auto", "1", "2", "4", "8", "16"),
        default=("auto",),
    )
    result.add_argument("--decode-warps", type=int, choices=(1, 2, 4, 8), default=4)
    result.add_argument("--merge-warps", type=int, choices=(1, 2, 4, 8), default=4)
    result.add_argument("--seed", type=_nonnegative_int, default=20260819)
    result.add_argument("--max-seq-len", type=_positive_int, default=4096)
    result.add_argument("--warmup", type=_nonnegative_int, default=20)
    result.add_argument("--samples", type=_positive_int, default=21)
    result.add_argument("--inner", type=_positive_int, default=20)
    result.add_argument("--profile-steps", type=_positive_int, default=10)
    result.add_argument("--trace")
    result.add_argument("--manifest", default="challenge/final_gate_manifest.json")
    result.add_argument("--no-cudagraph", action="store_true")
    return result


def _validate_cli(args: argparse.Namespace, parser_: argparse.ArgumentParser) -> None:
    single_chunk_commands = {"final-gate", "final-benchmark", "profile", "once"}
    if args.command in single_chunk_commands and len(args.chunks) != 1:
        parser_.error(f"{args.command} requires exactly one --chunks value")
    if args.command in {"final-gate", "final-benchmark"}:
        if tuple(args.chunks) != ("selected",):
            parser_.error("final commands require the measured --chunks selected policy")
        if args.decode_warps != 4 or args.merge_warps != 4:
            parser_.error("selected final configuration requires decode/merge warps=4")
    if args.all_modes and args.command not in {"correctness", "final-gate"}:
        parser_.error("--all-modes is only meaningful for correctness/final-gate")
    if args.all_batches and args.command in {"profile", "once"}:
        parser_.error("profile/once accept one --batch and do not use --all-batches")


def main(argv: list[str] | None = None) -> int:
    parser_ = parser()
    args = parser_.parse_args(argv)
    _validate_cli(args, parser_)
    if args.command == "dry-run":
        print(json.dumps({"status": "parser-ok", "batches": BATCHES, "modes": MODES}))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("Challenge execution requires CUDA")
    function_name = args.command.replace("-", "_")
    return globals()[function_name](args)


if __name__ == "__main__":
    raise SystemExit(main())
