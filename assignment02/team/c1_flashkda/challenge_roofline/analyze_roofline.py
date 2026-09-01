#!/usr/bin/env python3
"""Reproduce the strict BF16 tensor-contraction roofline for H12 v4P2.

The FLOP convention deliberately counts only dense BF16->FP32 HMMA work.  It
does not pretend that scalar/SFU/control instructions have an unambiguous FLOP
equivalent.  NCU's measured DRAM traffic and per-kernel replay duration are
used for the byte and time axes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RAW = (
    HERE.parent
    / "challenge_vshard4_prefetch2"
    / "results"
    / "c1_h12_ncu_b300_sm103a_h12_r1_vshard4_p2_full_job10085_raw.csv"
)
DEFAULT_JSON = HERE / "results" / "c1_h12_bf16_tensor_roofline_r1.json"
DEFAULT_MD = HERE / "results" / "c1_h12_bf16_tensor_roofline_r1.md"

DURATION = "gpu__time_duration.sum"
DRAM_READ = "dram__bytes_read.sum"
DRAM_WRITE = "dram__bytes_write.sum"
DRAM_PEAK_PER_CYCLE = "dram__bytes.sum.peak_sustained"
DRAM_CLOCK = "dram__cycles_elapsed.avg.per_second"
HMMA_FLOPS = "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum"
HMMA_PEAK = HMMA_FLOPS + ".peak_sustained_elapsed.per_second"

EXPECTED_UNITS = {
    DURATION: "us",
    DRAM_READ: "Mbyte",
    DRAM_WRITE: "Mbyte",
    DRAM_PEAK_PER_CYCLE: "Kbyte/cycle",
    DRAM_CLOCK: "Ghz",
    HMMA_FLOPS: "",
    HMMA_PEAK: "",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    return parser.parse_args()


def read_ncu(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3:
        raise ValueError(f"expected NCU units row plus K1/K2 rows, got {len(rows)}")
    units, data = rows[0], rows[1:]
    if units.get("ID", ""):
        raise ValueError("first NCU row is not the units row")
    for metric, expected in EXPECTED_UNITS.items():
        actual = units.get(metric)
        if actual != expected:
            raise ValueError(f"unexpected unit for {metric}: {actual!r} != {expected!r}")
    return {key: units[key] for key in EXPECTED_UNITS}, data


def phase_name(row: dict[str, str]) -> str:
    kernel = row["Kernel Name"]
    if "_flash_kda_fwd_prepare" in kernel:
        return "K1_prepare"
    if "_flash_kda_fwd_recurrence" in kernel:
        return "K2_recurrence"
    raise ValueError(f"unrecognized kernel for row {row.get('ID')}: {kernel[:160]}")


def theoretical_flops(
    phase: str, *, batch: int, tokens: int, heads: int, chunk: int, dim: int
) -> dict[str, int]:
    if tokens % chunk:
        raise ValueError("the profiled fixed-length case must have tokens % chunk == 0")
    head_tiles = batch * heads * (tokens // chunk)
    if phase == "K1_prepare":
        # Two (C,D)x(D,C) contractions; an FMA is two FLOPs.
        per_head_tile = 4 * chunk * chunk * dim
    elif phase == "K2_recurrence":
        # Three D-wide contractions plus two C-wide contractions.
        per_head_tile = 6 * chunk * dim * dim + 4 * chunk * chunk * dim
    else:
        raise ValueError(phase)
    return {
        "head_tiles": head_tiles,
        "tensor_flops_per_head_tile": per_head_tile,
        "tensor_flops_total": head_tiles * per_head_tile,
    }


def analyze_row(
    row: dict[str, str], *, batch: int, tokens: int, heads: int, chunk: int, dim: int
) -> dict[str, Any]:
    phase = phase_name(row)
    theory = theoretical_flops(
        phase, batch=batch, tokens=tokens, heads=heads, chunk=chunk, dim=dim
    )
    measured_flops = int(row[HMMA_FLOPS])
    if measured_flops != theory["tensor_flops_total"]:
        raise AssertionError(
            f"{phase}: NCU HMMA FLOPs {measured_flops} != theory "
            f"{theory['tensor_flops_total']}"
        )

    duration_s = float(row[DURATION]) * 1e-6
    read_bytes = float(row[DRAM_READ]) * 1e6
    write_bytes = float(row[DRAM_WRITE]) * 1e6
    dram_bytes = read_bytes + write_bytes
    peak_bandwidth = float(row[DRAM_PEAK_PER_CYCLE]) * 1e3 * float(row[DRAM_CLOCK]) * 1e9
    peak_compute = float(row[HMMA_PEAK])
    intensity = measured_flops / dram_bytes
    achieved = measured_flops / duration_s
    ridge = peak_compute / peak_bandwidth
    memory_roof = intensity * peak_bandwidth
    roof = min(peak_compute, memory_roof)
    branch = "memory" if intensity < ridge else "compute"

    return {
        "phase": phase,
        "ncu_row_id": int(row["ID"]),
        "kernel_name_prefix": row["Kernel Name"].split("<", 1)[0].strip(),
        "theory": theory,
        "validation": {
            "ncu_hmma_flops": measured_flops,
            "theory_equals_ncu": True,
            "difference_flops": measured_flops - theory["tensor_flops_total"],
        },
        "measured": {
            "duration_us_ncu_replay": float(row[DURATION]),
            "dram_read_bytes": int(read_bytes),
            "dram_write_bytes": int(write_bytes),
            "dram_total_bytes": int(dram_bytes),
            "peak_tensor_flops_per_s": peak_compute,
            "peak_dram_bytes_per_s": peak_bandwidth,
        },
        "roofline": {
            "tensor_arithmetic_intensity_flops_per_byte": intensity,
            "achieved_tensor_flops_per_s": achieved,
            "ridge_point_flops_per_byte": ridge,
            "memory_roof_flops_per_s": memory_roof,
            "selected_roof_flops_per_s": roof,
            "selected_branch": branch,
            "roof_efficiency": achieved / roof,
            "compute_peak_utilization": achieved / peak_compute,
            "dram_payload_rate_bytes_per_s": dram_bytes / duration_s,
            "dram_peak_utilization": (dram_bytes / duration_s) / peak_bandwidth,
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    cfg = result["configuration"]
    phases = result["phases"]
    lines = [
        "# H12 v4P2 的严格 BF16 tensor-contraction roofline",
        "",
        "本页由 `analyze_roofline.py` 直接从同一版候选的 NCU Full raw CSV 生成。",
        "FLOP 口径只包含 NCU 计数的 dense BF16→FP32 HMMA；标量、SFU、控制和地址计算",
        "不被强行换算成 FLOP。因此这是可复核的 tensor-contraction roofline，而不是含糊的",
        "“所有操作总 FLOP”模型。duration 是 NCU replay duration，不替代 full-call event P50。",
        "",
        "## 变量表",
        "",
        "| 变量 | 含义 | 本次值 |",
        "|---|---|---:|",
        f"| $B$ | batch 数 | {cfg['batch']} |",
        f"| $T$ | 每个 batch 的 token 数 | {cfg['tokens']} |",
        f"| $H$ | 本卡 head 数 | {cfg['heads']} |",
        f"| $C$ | chunk 长度 | {cfg['chunk']} |",
        f"| $D$ | key/value head dimension | {cfg['dim']} |",
        "| $N$ | `(batch, head, chunk)` tile 数 | $B H T/C$ |",
        "| $F_{TC}$ | NCU dense BF16 HMMA tensor FLOP 数 | 分阶段见下 |",
        "| $Q_{DRAM}$ | NCU DRAM read+write bytes | 分阶段见下 |",
        "| $I_{TC}$ | tensor arithmetic intensity | $F_{TC}/Q_{DRAM}$ |",
        "| $P_{TC}$ | 实测 tensor FLOP/s | $F_{TC}/t$ |",
        "| $P_{peak}$ | NCU BF16 HMMA sustained peak | 分阶段回读 |",
        "| $BW_{peak}$ | NCU DRAM sustained peak | 分阶段回读 |",
        "| $I_{ridge}$ | roofline ridge point | $P_{peak}/BW_{peak}$ |",
        "",
        "## FLOP 闭合",
        "",
        "K1 有两个 `(C,D)×(D,C)` contraction；K2 有三个 D-wide contraction 和两个",
        "C-wide contraction。以 FMA=2 FLOP 计：",
        "",
        "$$N=B H T/C,$$",
        "",
        "$$F_{TC,K1}=N(4C^2D),$$",
        "",
        "$$F_{TC,K2}=N(6CD^2+4C^2D).$$",
        "",
        "两式的理论整数必须与 NCU `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_",
        "sparsity_off.sum` 完全相等，否则脚本直接失败。",
        "",
        "## 分阶段结果",
        "",
        "| phase | $F_{TC}$ | DRAM bytes | $I_{TC}$ (F/B) | achieved (TF/s) | ridge (F/B) | selected roof (TF/s) | efficiency | branch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for p in phases:
        roof = p["roofline"]
        measured = p["measured"]
        lines.append(
            "| {phase} | {flops:,} | {bytes:,} | {intensity:.6f} | {achieved:.6f} | "
            "{ridge:.6f} | {selected:.6f} | {eff:.4%} | {branch} |".format(
                phase=p["phase"],
                flops=p["validation"]["ncu_hmma_flops"],
                bytes=measured["dram_total_bytes"],
                intensity=roof["tensor_arithmetic_intensity_flops_per_byte"],
                achieved=roof["achieved_tensor_flops_per_s"] / 1e12,
                ridge=roof["ridge_point_flops_per_byte"],
                selected=roof["selected_roof_flops_per_s"] / 1e12,
                eff=roof["roof_efficiency"],
                branch=roof["selected_branch"],
            )
        )
    lines += [
        "",
        "K1 位于 memory-roof 分支。K2 的 intensity 越过 ridge、落在 compute-roof 分支，",
        "但这不等于“算力已饱和”：其 roof efficiency 很低，结合小 grid/低并行度 counter，",
        "应解释为延迟/并行度受限的 compute-side 点，而非接近峰值的 compute saturation。",
        "",
        "## 可复核性",
        "",
        f"- raw CSV SHA-256：`{result['input']['sha256']}`",
        f"- raw CSV：`{result['input']['path']}`",
        "- JSON 保留所有原始量、单位换算、理论整数和派生量。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    raw = args.raw_csv.resolve()
    units, rows = read_ncu(raw)
    phases = [
        analyze_row(
            row,
            batch=args.batch,
            tokens=args.tokens,
            heads=args.heads,
            chunk=args.chunk,
            dim=args.dim,
        )
        for row in rows
    ]
    phases.sort(key=lambda item: item["phase"])
    if [item["phase"] for item in phases] != ["K1_prepare", "K2_recurrence"]:
        raise AssertionError("expected exactly K1_prepare and K2_recurrence")

    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "strict_dense_bf16_to_fp32_hmma_tensor_contraction_roofline",
        "excludes": ["scalar_ops", "sfu_ops", "control_ops", "address_ops"],
        "configuration": {
            "batch": args.batch,
            "tokens": args.tokens,
            "heads": args.heads,
            "chunk": args.chunk,
            "dim": args.dim,
            "device": "NVIDIA B300",
            "compute_capability": "10.3",
            "candidate": "vshard4_prefetch2",
        },
        "input": {"path": str(raw), "sha256": sha256(raw), "ncu_units": units},
        "phases": phases,
        "interpretation_guard": (
            "A selected compute-roof branch identifies I_TC >= ridge only; it does not "
            "prove compute saturation when grid-level parallelism is low."
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.json_out}")
    print(f"wrote {args.markdown_out}")


if __name__ == "__main__":
    main()
