#!/usr/bin/env python3
"""Turn a raw NCU CSV into a conservative C1 K1/K2 evidence note.

This parser deliberately preserves the distinction between a *counter report*
and a roofline claim.  It groups K1/K2 metrics, writes the Tensor/SM,
DRAM/L2, occupancy/waves and warp-stall fields that NCU actually supplied, and
marks missing fields rather than inventing zeros.  A report author may only
state a roofline classification after the named source counters are present.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


# This is a ledger, not a request for a particular NCU version's exact metric
# spelling.  Section/metric names differ a little across NCU versions; the
# matching is intentionally broad and the raw CSV remains the authority.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "FLOP/instruction evidence": (
        "flop_count",
        "flop count",
        "inst_executed_pipe_tensor",
        "sass_thread_inst_executed_op_hmma",
        "sass_thread_inst_executed_op_ffma",
        "sass_thread_inst_executed_op_hfma",
    ),
    "DRAM byte evidence": (
        "dram__bytes",
        "dram__sectors",
    ),
    "Tensor/SM": (
        "compute (sm) throughput",
        "sm__throughput",
        "tensor",
        "pipe_tensor",
    ),
    "DRAM/L2": (
        "dram__throughput",
        "dram__bytes",
        "dram throughput",
        "l2 cache throughput",
        "l2 cache hit rate",
        "lts__throughput",
        "lts__t_",
        "memory throughput",
    ),
    "Occupancy/waves": (
        "occupancy",
        "warps_active",
        "warps active",
        "waves_per_multiprocessor",
        "launch__waves",
        "launch__grid",
        "launch__block",
        "grid size",
        "block size",
    ),
    "Warp stall": ("warp_issue_stalled", "warps_issue_stalled", "warp stall"),
}

MAX_METRICS_PER_CATEGORY = 48


def kernel_group(name: str) -> str | None:
    if "_flash_kda_fwd_prepare" in name:
        return "K1 prepare"
    if "_flash_kda_fwd_recurrence" in name:
        return "K2 recurrence"
    return None


def metric_categories(metric: str) -> list[str]:
    lowered = metric.lower()
    return [
        category
        for category, needles in CATEGORIES.items()
        if any(needle in lowered for needle in needles)
    ]


def clean(value: str) -> str:
    return " ".join(value.replace("\n", " ").split())


def render(csv_path: Path, out_path: Path) -> None:
    grouped: dict[str, dict[str, list[tuple[str, str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen_kernels: set[str] = set()
    total_rows = 0
    # Nsight Compute may prepend ==WARNING==/==PROF== lines before the CSV
    # header.  Locate the real header explicitly instead of letting
    # DictReader treat the first diagnostic line as a one-column schema.
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        raw_lines = f.readlines()
    header_index = next(
        (
            i
            for i, line in enumerate(raw_lines)
            if '"Kernel Name"' in line
            and ('"Metric Name"' in line or 'dram__' in line)
        ),
        None,
    )
    if header_index is None:
        raise ValueError("NCU CSV header with Kernel Name was not found")
    reader = csv.DictReader(raw_lines[header_index:])
    long_format = "Metric Name" in (reader.fieldnames or [])
    export_format = "long metric rows" if long_format else "wide raw metric columns"
    for row in reader:
        total_rows += 1
        group = kernel_group(row.get("Kernel Name", ""))
        if group is None:
            continue
        seen_kernels.add(group)
        if long_format:
            candidates = [
                (
                    clean(row.get("Metric Name", "")),
                    clean(row.get("Metric Value", "")),
                    clean(row.get("Metric Unit", "")),
                )
            ]
        else:
            candidates = [
                (clean(metric), clean(value or ""), "")
                for metric, value in row.items()
                if metric is not None
            ]
        for metric, value, unit in candidates:
            if not metric or value in {"", "no data"}:
                continue
            for category in metric_categories(metric):
                record = (metric, value, unit)
                if record not in grouped[group][category]:
                    grouped[group][category].append(record)

    lines = [
        "# C1 FlashKDA：NCU full counter / roofline 证据摘录",
        "",
        "## 变量表",
        "",
        "| 变量 | 含义 | 取值/来源 |",
        "| --- | --- | --- |",
        "| `K1` | workspace prepare kernel | `_flash_kda_fwd_prepare` |",
        "| `K2` | 有状态 recurrence kernel | `_flash_kda_fwd_recurrence` |",
        "| `F` | 目标 kernel 的实际有效 FLOP | 必须由 NCU roofline/FLOP 指标或独立指令计数给出 |",
        "| `B` | 目标 kernel 的实际 DRAM bytes | `dram__bytes*`；不是输入张量大小的猜测 |",
        "| `I=F/B` | operational intensity | 仅在 `F` 与 `B` 同一 kernel 同一 run 都可得时计算 |",
        "| `P_roof` | roofline 上界 | `min(P_peak, I*BW_peak)` |",
        "",
        "## 输入与完整性",
        "",
        f"- 原始 CSV：`{csv_path}`",
        f"- CSV 导出布局：{export_format}",
        f"- CSV 数据行：{total_rows}",
        f"- 识别到 kernel：{', '.join(sorted(seen_kernels)) or '无（FAIL）'}",
        "- 这份文件只摘录 NCU 实际输出；完整 CSV 和 `.ncu-rep` 才是原始证据。",
        "",
        "## 分 kernel 计数",
        "",
    ]
    all_present = seen_kernels == {"K1 prepare", "K2 recurrence"}
    for kernel in ("K1 prepare", "K2 recurrence"):
        lines.extend([f"### {kernel}", ""])
        for category in CATEGORIES:
            rows = grouped[kernel][category]
            if not rows:
                lines.append(f"- {category}: **MISSING**（NCU CSV 未找到匹配 metric；不能把它当作 0）。")
                all_present = False
                continue
            lines.append(f"- {category}:")
            for metric, value, unit in rows[:MAX_METRICS_PER_CATEGORY]:
                lines.append(f"  - `{metric}` = `{value}` {unit}".rstrip())
            if len(rows) > MAX_METRICS_PER_CATEGORY:
                lines.append(
                    f"  - 其余 {len(rows) - MAX_METRICS_PER_CATEGORY} 个匹配指标见原始 CSV；摘要不重复展开。"
                )
        lines.append("")

    lines.extend(
        [
            "## Roofline 判读纪律",
            "",
            "1. 对每个 kernel 单独取同一 profile 的 `F`、`B`、SM/Tensor、DRAM/L2、occupancy/waves/stall；不得把 K1 与 K2 相加后套一个瓶颈标签。",
            "2. 只有 raw report 含有可解释的实际 FLOP 与 DRAM bytes，才能算 `I=F/B`，再与该卡已记录的 `P_peak/BW_peak` 比较。缓存 hit 不能替代 DRAM bytes。",
            "3. 若 K2 的 waves/occupancy 偏低而 DRAM/SM 均非饱和，优先报告“并行度受限”；若 `I` 与 roofline 及 achieved bandwidth/compute 一致，才报告 memory-/compute-bound。",
            "",
            "### 自动完整性结论",
            "",
            ("- **COUNTER_SET_COMPLETE**：K1/K2 的六类计数均已找到，其中包含可换算的 FLOP/指令证据和 DRAM bytes/sectors 证据；仍需由 reviewer 从原始 report 核对换算与 `F/B`。"
             if all_present else
             "- **COUNTER_SET_INCOMPLETE**：至少一个 kernel 缺 FLOP/指令、DRAM bytes/sectors 或其余分类证据；报告不得声称完成 roofline 分类。"),
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"NCU CSV not found: {args.csv}")
    render(args.csv, args.out)
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
