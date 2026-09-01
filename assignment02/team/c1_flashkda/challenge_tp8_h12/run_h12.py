#!/usr/bin/env python3
"""Machine-readable fixed-length H=12 gate and latency runner for C1.

The ``p2`` variant requires one extension that exports the public baseline
``fwd`` together with ``fwd_vshard`` (P1) and ``fwd_vshard_p2`` (P2).  All
three paths are therefore called from one Python process and one imported
``flash_kda_C`` shared object.  The ``vshard4`` variant deliberately uses a
separate extension/runner invocation because its generated extension exports
``fwd_vshard4`` rather than the P1/P2 pair.

This runner covers fixed-length inputs only.  It neither exercises varlen nor
tail-token paths and it does not modify a kernel or public API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2  # noqa: E402
from assignment02.team.c1_flashkda.challenge_vshard import vshard  # noqa: E402
from assignment02.team.c1_flashkda.challenge_vshard4 import vshard4  # noqa: E402
from assignment02.team.c1_flashkda.harness import validate_and_bench as common  # noqa: E402


TORCH_REF_OUTPUT_RTOL = 2e-2
TORCH_REF_OUTPUT_ATOL = 2e-2
TORCH_REF_STATE_RTOL = 5e-2
TORCH_REF_STATE_ATOL = 5e-2
STATE_MODES = ("none", "bf16", "fp32")


@dataclass(frozen=True)
class PathSpec:
    """One public wrapper path measured by the runner."""

    name: str
    call: Callable[..., None]


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"invalid comma-separated integer list: {value!r}") from exc
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise ValueError(f"heads must be unique positive integers: {value!r}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(required_symbols: tuple[str, ...], source_files: tuple[Path, ...]) -> dict[str, object]:
    """Record the exact imported SO and caller-selected generated source files."""
    import flash_kda_C

    missing = [name for name in required_symbols if not hasattr(flash_kda_C, name)]
    if missing:
        raise RuntimeError(
            "loaded flash_kda_C lacks required ABI symbols "
            f"{missing}; do not compare extensions built from different trees"
        )
    extension_path = Path(flash_kda_C.__file__).resolve()
    if not extension_path.is_file():
        raise RuntimeError(f"loaded extension is not a regular file: {extension_path}")
    source_identity: list[dict[str, str]] = []
    for raw_path in source_files:
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--source is not a regular file: {path}")
        source_identity.append({"path": str(path), "sha256": _sha256(path)})
    runner_path = Path(__file__).resolve()
    return {
        "extension_path": str(extension_path),
        "extension_sha256": _sha256(extension_path),
        "extension_module": "flash_kda_C",
        "required_symbols": list(required_symbols),
        "runner_path": str(runner_path),
        "runner_sha256": _sha256(runner_path),
        "sources": source_identity,
    }


def _paths(variant: str, baseline: Callable[..., None]) -> tuple[PathSpec, ...]:
    if variant == "p2":
        return (
            PathSpec("baseline", baseline),
            PathSpec("p1_vshard", vshard.fwd),
            PathSpec("p2_prefetch2", prefetch2.fwd),
        )
    if variant == "vshard4":
        return (PathSpec("baseline", baseline), PathSpec("vshard4", vshard4.fwd))
    raise ValueError(f"unknown variant: {variant}")


def _run_path(path: PathSpec, x: common.Inputs, mode: str, state_seed: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_state, final_state = common.state_tensors(mode, x.q.shape[2], state_seed)
    return common.invoke(
        path.call,
        x,
        None if initial_state is None else initial_state.clone(),
        final_state,
    )


def _comparison(
    name: str,
    actual: tuple[torch.Tensor, torch.Tensor | None],
    expected: tuple[torch.Tensor, torch.Tensor | None],
) -> dict[str, object]:
    output, final_state = actual
    expected_output, expected_final_state = expected
    common.require_exact(f"{name}/output", output, expected_output)
    result: dict[str, object] = {
        "output_exact": True,
        "output_max_abs": common.max_abs(output, expected_output),
    }
    if final_state is not None or expected_final_state is not None:
        if final_state is None or expected_final_state is None:
            raise AssertionError(f"FAIL exact: {name}/final_state presence differs")
        common.require_exact(f"{name}/final_state", final_state, expected_final_state)
        result["final_state_exact"] = True
        result["final_state_max_abs"] = common.max_abs(final_state, expected_final_state)
    return result


def _torch_ref_comparison(
    name: str,
    actual: tuple[torch.Tensor, torch.Tensor | None],
    reference: tuple[torch.Tensor, torch.Tensor | None],
) -> dict[str, object]:
    output, final_state = actual
    ref_output, ref_final_state = reference
    common.require_close(
        f"{name}/output",
        output,
        ref_output,
        rtol=TORCH_REF_OUTPUT_RTOL,
        atol=TORCH_REF_OUTPUT_ATOL,
    )
    result: dict[str, object] = {
        "output_close": True,
        "output_bitwise_exact": torch.equal(output, ref_output),
        "output_max_abs": common.max_abs(output, ref_output),
        "output_tolerance": {"rtol": TORCH_REF_OUTPUT_RTOL, "atol": TORCH_REF_OUTPUT_ATOL},
    }
    if final_state is not None or ref_final_state is not None:
        if final_state is None or ref_final_state is None:
            raise AssertionError(f"FAIL close: {name}/final_state presence differs")
        common.require_close(
            f"{name}/final_state",
            final_state,
            ref_final_state,
            rtol=TORCH_REF_STATE_RTOL,
            atol=TORCH_REF_STATE_ATOL,
        )
        result["final_state_close"] = True
        result["final_state_bitwise_exact"] = torch.equal(final_state, ref_final_state)
        result["final_state_max_abs"] = common.max_abs(final_state, ref_final_state)
        result["final_state_tolerance"] = {"rtol": TORCH_REF_STATE_RTOL, "atol": TORCH_REF_STATE_ATOL}
    return result


def _exact_gate(
    paths: tuple[PathSpec, ...],
    x: common.Inputs,
    mode: str,
    state_seed: int,
    torch_ref: Callable[..., None] | None,
) -> dict[str, object]:
    """Run public wrappers independently, then compare their finalized outputs."""
    outputs = {path.name: _run_path(path, x, mode, state_seed) for path in paths}
    torch.cuda.synchronize()
    baseline = outputs["baseline"]
    exact: dict[str, object] = {}
    for path in paths[1:]:
        exact[f"{path.name}_vs_baseline"] = _comparison(
            f"{mode}/{path.name}_vs_baseline", outputs[path.name], baseline
        )
    if "p2_prefetch2" in outputs:
        exact["p2_prefetch2_vs_p1_vshard"] = _comparison(
            f"{mode}/p2_prefetch2_vs_p1_vshard", outputs["p2_prefetch2"], outputs["p1_vshard"]
        )

    result: dict[str, object] = {"candidate_exact": exact}
    if torch_ref is not None:
        ref = _run_path(PathSpec("torch_ref", torch_ref), x, mode, state_seed)
        torch.cuda.synchronize()
        result["torch_ref_tolerance"] = {
            path.name: _torch_ref_comparison(
                f"{mode}/{path.name}_vs_torch_ref", outputs[path.name], ref
            )
            for path in paths
        }
    return result


def _percentile_sorted(sorted_values: list[float], percentile: float) -> float:
    """Linear percentile, with p50 identical to statistics.median for finite samples."""
    if not sorted_values:
        raise ValueError("cannot summarize zero samples")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError(f"invalid percentile: {percentile}")
    position = (len(sorted_values) - 1) * percentile
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = position - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def _summary(values: list[float]) -> dict[str, object]:
    if not values or any(value < 0.0 for value in values):
        raise RuntimeError("CUDA event timing returned an invalid sample")
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": _percentile_sorted(ordered, 0.95),
        "p99_ms": _percentile_sorted(ordered, 0.99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "percentile_method": "linear interpolation at (n-1)*p",
    }


def _timed_call(path: PathSpec, x: common.Inputs, mode: str, state_seed: int) -> Callable[[], None]:
    initial_state, final_state = common.state_tensors(mode, x.q.shape[2], state_seed)
    out = torch.empty_like(x.v)

    def call() -> None:
        path.call(
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
            initial_state=initial_state,
            final_state=final_state,
        )

    return call


def _balanced_event_benchmark(
    paths: tuple[PathSpec, ...], x: common.Inputs, mode: str, warmup: int, samples: int
) -> dict[str, object]:
    """Measure every path once/sample, rotating ABC/BCA/CAB (or AB/BA)."""
    calls = {
        # Each wrapper receives the same fixed initial-state values.  The
        # tensors themselves remain separate because every call writes its own
        # final-state output buffer.
        path.name: _timed_call(path, x, mode, 17)
        for path in paths
    }
    names = tuple(path.name for path in paths)
    for index in range(warmup):
        offset = index % len(names)
        for name in names[offset:] + names[:offset]:
            calls[name]()
    torch.cuda.synchronize()

    values: dict[str, list[float]] = {name: [] for name in names}
    raw_samples: list[dict[str, object]] = []
    for index in range(samples):
        offset = index % len(names)
        order = names[offset:] + names[:offset]
        events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            calls[name]()
            end.record()
            events.append((name, start, end))
        torch.cuda.synchronize()
        record: dict[str, object] = {"sample": index, "order": list(order), "ms": {}}
        measured = record["ms"]
        assert isinstance(measured, dict)
        for name, start, end in events:
            elapsed = float(start.elapsed_time(end))
            values[name].append(elapsed)
            measured[name] = elapsed
        raw_samples.append(record)

    summaries = {name: _summary(path_values) for name, path_values in values.items()}
    baseline_p50 = float(summaries["baseline"]["p50_ms"])
    speedups = {
        name: baseline_p50 / float(summary["p50_ms"])
        for name, summary in summaries.items()
        if name != "baseline"
    }
    return {
        "event_contract": (
            "one CUDA event surrounds one full public Python wrapper call; wrapper workspace allocation is inside "
            "the event; host dispatch is excluded by CUDA-event timing"
        ),
        "ordering": "per-sample cyclic rotation: ABC/BCA/CAB for P2, AB/BA for vshard4",
        "warmup_calls_per_path": warmup,
        "raw_samples": raw_samples,
        "summary": summaries,
        "speedup_vs_same_h12_baseline_p50_x": speedups,
    }


def _require_cuda_and_fixed_shape(tokens: int, heads: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required; do not run this runner on a login node")
    if tokens <= 0 or heads <= 0 or tokens % 16:
        raise ValueError("fixed-length runner requires positive T/H and T divisible by CHUNK=16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("p2", "vshard4"), default="p2")
    parser.add_argument("--reference-root", type=Path, required=True, help="pinned FlashKDA tree containing tests/torch_ref.py")
    parser.add_argument("--source", type=Path, action="append", default=[], help="generated/source file to SHA-256-record; repeatable")
    parser.add_argument("--small-t", type=int, default=256)
    parser.add_argument("--small-heads", default="1,2,4,12")
    parser.add_argument("--official-t", type=int, default=8192)
    parser.add_argument("--official-h", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200, help="event samples per repeat")
    parser.add_argument("--repeats", type=int, default=5, help="total samples/path = iters * repeats")
    parser.add_argument("--small-only", action="store_true", help="run the small torch_ref matrix but skip the formal H12 gate/benchmark")
    parser.add_argument("--official-only", action="store_true", help="run formal H12 BF16 exact gate/benchmark without torch_ref")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.small_only and args.official_only:
        parser.error("--small-only and --official-only cannot be combined")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative; iters and repeats must be positive")
    return args


def main() -> None:
    args = parse_args()
    small_heads = _csv_ints(args.small_heads)
    _require_cuda_and_fixed_shape(args.small_t, max(small_heads))
    _require_cuda_and_fixed_shape(args.official_t, args.official_h)
    import flash_kda

    required = ("fwd", "get_workspace_size", "fwd_vshard", "fwd_vshard_p2") if args.variant == "p2" else (
        "fwd", "get_workspace_size", "fwd_vshard4"
    )
    identity = _identity(required, tuple(args.source))
    paths = _paths(args.variant, flash_kda.fwd)
    torch_ref = common._load_torch_ref(args.reference_root)
    result: dict[str, object] = {
        "schema_version": 1,
        "candidate_variant": args.variant,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "build_target": os.environ.get("C1_BUILD_TARGET", "unspecified"),
        "identity": identity,
        "paths": [path.name for path in paths],
        "seed": args.seed,
        "scope": {
            "fixed_length_only": True,
            "varlen_covered": False,
            "tail_tokens_covered": False,
            "official_state_mode": "bf16",
            "official_reference": "candidate-vs-baseline/P1 exact; torch_ref intentionally small-shape only",
        },
    }
    print(f"device={result['device']} capability={result['capability']} variant={args.variant}")
    print(f"extension_sha256={identity['extension_sha256']}")

    if not args.official_only:
        small_matrix: dict[str, object] = {}
        for head_index, heads in enumerate(small_heads):
            x = common.make_inputs(args.small_t, heads, args.seed + head_index * 1009)
            small_matrix[f"H{heads}"] = {
                mode: _exact_gate(paths, x, mode, args.seed + head_index * 1009 + mode_index * 101, torch_ref)
                for mode_index, mode in enumerate(STATE_MODES)
            }
        result["small_exact_matrix"] = {
            "shape": {"B": 1, "T": args.small_t, "H": list(small_heads), "K": 128, "V": 128},
            "states": list(STATE_MODES),
            "torch_ref_policy": {
                "output": {"rtol": TORCH_REF_OUTPUT_RTOL, "atol": TORCH_REF_OUTPUT_ATOL},
                "final_state": {"rtol": TORCH_REF_STATE_RTOL, "atol": TORCH_REF_STATE_ATOL},
            },
            "results": small_matrix,
            "pass": True,
        }
        print("PASS small exact/reference matrix")

    if not args.small_only:
        x = common.make_inputs(args.official_t, args.official_h, args.seed + 999_983)
        official_gate = _exact_gate(paths, x, "bf16", args.seed + 4_001, None)
        samples = args.iters * args.repeats
        official = {
            "shape": {"B": 1, "T": args.official_t, "H": args.official_h, "K": 128, "V": 128},
            "state_mode": "bf16",
            "candidate_exact_gate": official_gate["candidate_exact"],
            "candidate_exact_gate_pass": True,
            "torch_ref": "not run at formal T=8192; small matrix above records repository tolerance gate",
            "benchmark": _balanced_event_benchmark(paths, x, "bf16", args.warmup, samples),
        }
        result["official_h12"] = official
        print("PASS official H12 BF16 exact gate and benchmark")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
