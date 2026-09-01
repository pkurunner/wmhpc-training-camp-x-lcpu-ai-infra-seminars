#!/usr/bin/env python3
"""C1's deliberately narrow CHUNK/tcgen05 microbench.

Variables (the report-facing ledger is repeated here so machine output is not
detached from its notation): C is chunk/matrix size; g_i=-5 is the conservative
gate; r(C)=exp(-5*(C-1)); L is strictly lower triangular; P_j=L**(2**j);
N_C is the doubling Neumann product; Q is the independent matrix batch;
(M_t,N_t,K_t)=(128,64,64) is the existing M3 tcgen05 tile.

The script is intentionally a proxy.  It verifies BF16 conversion on the
actual CUDA device, times a batched FP16 version of the source's doubling
Neumann algebra, and reports tcgen05 tile geometry.  It does not claim to be a
compiled CHUNK=32/64 FlashKDA implementation or an end-to-end tcgen05 port.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only on bad runner env
    raise SystemExit(f"PyTorch is required: {exc}") from exc


CS = (16, 32, 64)
BF16_MIN_NORMAL = 2.0**-126
BF16_MIN_SUBNORMAL = 2.0**-133


def cuda_event_median_ms(fn: Any, *, warmup: int, iters: int, repeats: int) -> list[float]:
    """Return per-repeat CUDA-event averages; fn has no allocation in timed region."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) / iters)
    return samples


def classify_bf16_scalar(value: float) -> str:
    magnitude = abs(value)
    if magnitude == 0.0:
        return "zero"
    if magnitude < BF16_MIN_NORMAL:
        return "subnormal"
    return "normal"


def bf16_decay_probe(device: torch.device) -> dict[str, Any]:
    """Convert on-device values, not CPU emulation, then fetch classifications."""
    c_values = torch.arange(1, 65, device=device, dtype=torch.float32)
    fp32 = torch.exp(-5.0 * (c_values - 1.0))
    bf16_as_fp32 = fp32.to(torch.bfloat16).to(torch.float32).cpu().tolist()
    fp32_values = fp32.cpu().tolist()
    rows: list[dict[str, Any]] = []
    for c, before, after in zip(range(1, 65), fp32_values, bf16_as_fp32, strict=True):
        rows.append(
            {
                "C": c,
                "r_fp32": float(before),
                "r_bf16_as_fp32": float(after),
                "classification": classify_bf16_scalar(float(after)),
                "nonzero": bool(after != 0.0),
            }
        )
    last_nonzero = max(row["C"] for row in rows if row["nonzero"])
    first_zero = min(row["C"] for row in rows if not row["nonzero"])
    first_subnormal = min(row["C"] for row in rows if row["classification"] == "subnormal")
    selected = [row for row in rows if row["C"] in CS]
    # These assertions catch device/compiler FTZ behavior rather than assuming
    # the host's BF16 conversion semantics.  They follow RN BF16 expectations.
    assertions = {
        "C16_nonzero": selected[0]["nonzero"],
        "C32_zero": not selected[1]["nonzero"],
        "C64_zero": not selected[2]["nonzero"],
        "expected_boundary_last_nonzero_C19": last_nonzero == 19,
        "expected_boundary_first_zero_C20": first_zero == 20,
        "expected_first_subnormal_C19": first_subnormal == 19,
    }
    return {
        "formula": "r(C)=exp(-5*(C-1))",
        "bf16_min_normal": BF16_MIN_NORMAL,
        "bf16_min_subnormal": BF16_MIN_SUBNORMAL,
        "all_C_1_to_64": rows,
        "selected_C_16_32_64": selected,
        "observed_boundary": {
            "last_nonzero_C": last_nonzero,
            "first_zero_C": first_zero,
            "first_subnormal_C": first_subnormal,
        },
        "assertions": assertions,
    }


def exact_doubling_neumann(l: torch.Tensor) -> torch.Tensor:
    """Reference algebra, in FP32: prod(I+L^(2^j)) = (I-L)^-1 for nilpotent L."""
    c = l.shape[-1]
    if c & (c - 1):
        raise ValueError("C must be a power of two for this doubling probe")
    eye = torch.eye(c, dtype=l.dtype, device=l.device).expand(l.shape[0], -1, -1)
    inverse = eye + l
    power = l
    for _ in range(1, int(math.log2(c))):
        power = torch.bmm(power, power)
        inverse = inverse + torch.bmm(inverse, power)
    return inverse


def neumann_bench_for_c(
    c: int,
    *,
    target_matrix_bytes: int,
    warmup: int,
    iters: int,
    repeats: int,
    device: torch.device,
    q_override: int | None = None,
    measurement_basis: str = "equal_input_matrix_bytes",
) -> dict[str, Any]:
    """Time a batched FP16 Tensor Core proxy of upstream's doubling routine."""
    # Keep one FP16 [Q,C,C] matrix near the requested byte budget, bounded so
    # C=16 cannot create an impractically huge cuBLAS batch descriptor.
    auto_q = auto_q_for_c(c, target_matrix_bytes)
    q = auto_q if q_override is None else q_override
    if q <= 0:
        raise ValueError("Q must be positive")
    generator = torch.Generator(device=device).manual_seed(20260819 + c)
    l = torch.randn((q, c, c), device=device, dtype=torch.float16, generator=generator)
    l = torch.tril(l, diagonal=-1).mul_(0.002)

    # Algebra gate in FP32 on a small deterministic slice.  This confirms the
    # exact proxy formula; it is deliberately separate from performance timing.
    l_check = l[: min(q, 8)].float()
    got = exact_doubling_neumann(l_check)
    # `solve_triangular` backends need not accept a zero-stride expanded RHS,
    # so make this tiny correctness-only identity batch materialized.
    eye = torch.eye(c, dtype=torch.float32, device=device).repeat(l_check.shape[0], 1, 1)
    want = torch.linalg.solve_triangular(eye - l_check, eye, upper=False)
    max_abs_err = float((got - want).abs().max().item())
    algebra_pass = bool(torch.allclose(got, want, rtol=2e-5, atol=2e-6))

    # Allocate the identity once; no allocation, reference computation, or
    # synchronization is inside timed iterations.
    eye_h = torch.eye(c, dtype=torch.float16, device=device).expand(q, -1, -1)

    def launch() -> torch.Tensor:
        inverse = eye_h + l
        power = l
        for _ in range(1, int(math.log2(c))):
            power = torch.bmm(power, power)
            inverse = inverse + torch.bmm(inverse, power)
        return inverse

    # Materialize once before timing and retain only a scalar checksum so
    # PyTorch cannot accidentally make the benchmark look like an unused trace.
    out = launch()
    checksum = float(out[0, -1, 0].float().item())
    samples = cuda_event_median_ms(launch, warmup=warmup, iters=iters, repeats=repeats)
    samples_sorted = sorted(samples)
    median_ms = samples_sorted[len(samples_sorted) // 2]
    p = int(math.log2(c))
    dense_gemm_count = 2 * (p - 1)
    flop_per_matrix = 2 * dense_gemm_count * c**3
    total_flop = flop_per_matrix * q
    tflops = total_flop / (median_ms * 1e9)
    base_flop = 2 * (2 * (int(math.log2(16)) - 1)) * 16**3
    return {
        "C": c,
        "measurement_basis": measurement_basis,
        "proxy_definition": "FP16 batched dense doubling Neumann; not compiled FlashKDA CHUNK variant",
        "Q_independent_matrices": int(q),
        "auto_Q_for_equal_input_matrix_bytes": int(auto_q),
        "matrix_bytes_per_input": int(q * c * c * 2),
        "dense_gemm_count_per_matrix": dense_gemm_count,
        "gemm_fma_flop_per_matrix": int(flop_per_matrix),
        "total_gemm_fma_flop_per_proxy": int(total_flop),
        "gemm_fma_flop_ratio_vs_C16": float(flop_per_matrix / base_flop),
        "cuda_event_ms_per_proxy": median_ms,
        "cuda_event_ms_samples": samples,
        "proxy_tensorcore_tflops": tflops,
        "algebra_gate": {"max_abs_err_fp32": max_abs_err, "pass": algebra_pass},
        "checksum": checksum,
    }


def auto_q_for_c(c: int, target_matrix_bytes: int) -> int:
    """One FP16 [Q,C,C] input near the byte target, with a practical batch cap."""
    return max(1024, min(262_144, target_matrix_bytes // (c * c * 2)))


def add_time_ratios(rows: list[dict[str, Any]]) -> None:
    """In-place ratio fields are valid only when every row shares the same Q."""
    base = next(row for row in rows if row["C"] == 16)
    base_ms = float(base["cuda_event_ms_per_proxy"])
    for row in rows:
        row["time_ratio_vs_C16_same_Q"] = float(row["cuda_event_ms_per_proxy"] / base_ms)


def tcgen05_geometry(tcgen05_bin: str | None) -> dict[str, Any]:
    """Report static tile fit; optional existing M3 binary is a correctness gate."""
    mt, nt, kt = 128, 64, 64
    c = 16
    single_util = (c / mt) * (c / nt) * (c / kt)
    packed_independent = (mt // c) * (nt // c)
    packed_util = packed_independent * single_util
    result: dict[str, Any] = {
        "existing_m3_source": "assignment02/cuda/m3_tcgen05/02_single_tile.cu",
        "tcgen05_min_tile_MNK": [mt, nt, kt],
        "flashkda_neumann_subproblem_MNK": [c, c, c],
        "single_subproblem_dimension_utilization_MNK": [c / mt, c / nt, c / kt],
        "single_subproblem_useful_flop_fraction": single_util,
        "single_subproblem_useful_flop_percent": 100.0 * single_util,
        "independent_C16_gemms_packable_along_MN": packed_independent,
        "packed_MN_only_useful_flop_fraction": packed_util,
        "packed_MN_only_useful_flop_percent": 100.0 * packed_util,
        "tcgen05_tile_fma_flop": 2 * mt * nt * kt,
        "single_C16_fma_flop": 2 * c * c * c,
        "fma_ratio_tile_to_single_C16": (mt * nt * kt) // (c * c * c),
        "throughput_proxy_boundary": (
            "geometry-only useful-FLOP fraction; M3 executable has correctness output but no timing, "
            "so this field is not an observed tcgen05 throughput"
        ),
        "why_K_cannot_pack_independent_gemms": (
            "K is one output's reduction axis; packing independent products along K would sum them, "
            "not produce independent outputs"
        ),
    }
    if not tcgen05_bin:
        result["m3_correctness_gate"] = {"attempted": False, "reason": "no --tcgen05-bin supplied"}
        return result
    path = Path(tcgen05_bin)
    if not path.is_file() or not os.access(path, os.X_OK):
        result["m3_correctness_gate"] = {
            "attempted": True,
            "reason": f"not an executable file: {path}",
            "pass": False,
        }
        return result
    runs: list[dict[str, Any]] = []
    for seed in (1, 7, 42):
        completed = subprocess.run(
            [str(path), str(seed)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = completed.stdout[-4000:]
        runs.append(
            {
                "seed": seed,
                "returncode": completed.returncode,
                "pass_token": "PASS" in output and "FAIL" not in output,
                "output_tail": output,
            }
        )
    result["m3_correctness_gate"] = {
        "attempted": True,
        "binary": str(path),
        "runs": runs,
        "pass": all(run["returncode"] == 0 and run["pass_token"] for run in runs),
    }
    return result


def device_info(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": props.name,
        "compute_capability": [props.major, props.minor],
        "multiprocessors": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="output machine-readable JSON path")
    parser.add_argument("--target-matrix-mib", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tcgen05-bin", default=None, help="optional existing M3 tcgen05 executable")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this probe must run in the clean B300 allocation")
    if min(args.target_matrix_mib, args.warmup, args.iters, args.repeats) <= 0:
        raise SystemExit("target-matrix-mib, warmup, iters and repeats must be positive")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    started = time.time()
    decay = bf16_decay_probe(device)
    target_matrix_bytes = args.target_matrix_mib * 1024 * 1024
    equal_input_bytes = [
        neumann_bench_for_c(
            c,
            target_matrix_bytes=target_matrix_bytes,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            device=device,
            measurement_basis="equal_input_matrix_bytes",
        )
        for c in CS
    ]
    # All C values launch the same number of independent small matrices here.
    # This is the latency/work-scaling view; use the equal-input-byte table for
    # throughput efficiency instead of comparing its different-Q wall times.
    fixed_q = min(auto_q_for_c(c, target_matrix_bytes) for c in CS)
    fixed_q_same_parallelism = [
        neumann_bench_for_c(
            c,
            target_matrix_bytes=target_matrix_bytes,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            device=device,
            q_override=fixed_q,
            measurement_basis="fixed_Q_same_independent_matrices",
        )
        for c in CS
    ]
    add_time_ratios(fixed_q_same_parallelism)
    tcgen = tcgen05_geometry(args.tcgen05_bin)
    m3_gate = tcgen["m3_correctness_gate"]
    assertions: dict[str, bool] = dict(decay["assertions"])
    assertions.update(
        {
            f"neumann_equal_bytes_C{row['C']}_algebra": row["algebra_gate"]["pass"]
            for row in equal_input_bytes
        }
    )
    assertions.update(
        {
            f"neumann_fixed_Q_C{row['C']}_algebra": row["algebra_gate"]["pass"]
            for row in fixed_q_same_parallelism
        }
    )
    # An omitted existing binary is not a failure: geometry remains a transparent
    # static proxy.  A supplied binary must pass all strict M3 checks.
    if m3_gate["attempted"]:
        assertions["supplied_m3_tcgen05_strict_gate"] = bool(m3_gate["pass"])
    payload = {
        "schema": "c1_chunk_tcgen_microbench/v2",
        "timestamp_unix": started,
        "device": device_info(device),
        "variables": {
            "C": "KDA chunk and triangular matrix edge",
            "g_i": -5,
            "r(C)": "exp(-5*(C-1))",
            "L": "strict lower triangular FP16 proxy matrix",
            "P_j": "L^(2^j)",
            "N_C": "doubling Neumann product",
            "Q": "independent matrix batch",
            "M_t_N_t_K_t": [128, 64, 64],
        },
        "bf16_decay": decay,
        "neumann_doubling_proxy": {
            "comparison_rule": (
                "Only fixed_Q_same_parallelism has equal Q and valid wall-time ratios. "
                "equal_input_matrix_bytes changes Q by C and is throughput-efficiency evidence only."
            ),
            "fixed_Q": fixed_q,
            "equal_input_matrix_bytes": equal_input_bytes,
            "fixed_Q_same_parallelism": fixed_q_same_parallelism,
        },
        "tcgen05_geometry_and_proxy": tcgen,
        "assertions": assertions,
        "all_assertions_pass": all(assertions.values()),
        "elapsed_wall_seconds": time.time() - started,
    }
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {
        "json": str(out_path),
        "bf16_boundary": decay["observed_boundary"],
        "neumann_fixed_Q": [
            {
                "C": row["C"],
                "Q": row["Q_independent_matrices"],
                "total_flop": row["total_gemm_fma_flop_per_proxy"],
                "ms": row["cuda_event_ms_per_proxy"],
                "time_ratio_vs_C16": row["time_ratio_vs_C16_same_Q"],
                "tflops": row["proxy_tensorcore_tflops"],
            }
            for row in fixed_q_same_parallelism
        ],
        "tcgen05_single_useful_flop_percent": tcgen["single_subproblem_useful_flop_percent"],
        "all_assertions_pass": payload["all_assertions_pass"],
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True))
    return 0 if payload["all_assertions_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
