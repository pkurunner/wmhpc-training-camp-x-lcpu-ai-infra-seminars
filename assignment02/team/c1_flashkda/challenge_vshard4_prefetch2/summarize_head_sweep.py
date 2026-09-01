#!/usr/bin/env python3
"""Validate per-head JSON evidence and emit a source-bound dispatch CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


VARIANTS = ("baseline", "vshard2_p2", "vshard4_p1", "vshard4_p2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--label", required=True, help="label between c1_vshard4_p2_ and _hN")
    parser.add_argument("--first-head", type=int, default=1)
    parser.add_argument("--last-head", type=int, default=96)
    parser.add_argument("--expected-T", type=int, default=8192)
    parser.add_argument("--expected-samples", type=int, default=500)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if (args.first_head <= 0 or args.last_head < args.first_head
            or args.expected_T <= 0 or args.expected_samples <= 0):
        parser.error("head bounds and expected sample count must be positive")

    rows: list[dict[str, object]] = []
    extension_sha: str | None = None
    for head in range(args.first_head, args.last_head + 1):
        source = args.results_dir / f"c1_vshard4_p2_{args.label}_h{head}_bf16_cyclic.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        shape = data.get("shape", {})
        expected_shape = {"B": 1, "T": args.expected_T, "H": head, "K": 128, "V": 128}
        if (not data.get("exact_gate_pass") or shape != expected_shape
                or data.get("states") != ["bf16"]
                or data.get("candidate") != "fwd_vshard4_p2"):
            raise RuntimeError(f"exact/shape gate failed for {source}")
        current_sha = data.get("extension", {}).get("extension_sha256")
        if not isinstance(current_sha, str):
            raise RuntimeError(f"missing extension SHA-256 in {source}")
        if extension_sha is None:
            extension_sha = current_sha
        elif current_sha != extension_sha:
            raise RuntimeError(f"mixed extension SHA-256 at H={head}")

        paths = data.get("benchmark", {}).get("paths", {})
        raw = data.get("benchmark", {}).get("raw_samples_ms", {})
        for variant in VARIANTS:
            if paths.get(variant, {}).get("samples") != args.expected_samples:
                raise RuntimeError(f"summary sample gate failed for H={head}/{variant}")
            if len(raw.get(variant, [])) != args.expected_samples:
                raise RuntimeError(f"raw sample gate failed for H={head}/{variant}")
        v2 = float(paths["vshard2_p2"]["p50_ms"])
        v4 = float(paths["vshard4_p2"]["p50_ms"])
        row: dict[str, object] = {
            "H": head,
            "samples_per_path": args.expected_samples,
            "baseline_p50_ms": paths["baseline"]["p50_ms"],
            "vshard2_p2_p50_ms": v2,
            "vshard4_p1_p50_ms": paths["vshard4_p1"]["p50_ms"],
            "vshard4_p2_p50_ms": v4,
            "vshard2_p2_over_vshard4_p2_p50_x": v2 / v4,
            "vshard4_p2_faster_than_vshard2_p2": v4 < v2,
            "vshard2_p2_p95_ms": paths["vshard2_p2"]["p95_ms"],
            "vshard4_p2_p95_ms": paths["vshard4_p2"]["p95_ms"],
            "vshard2_p2_p99_ms": paths["vshard2_p2"]["p99_ms"],
            "vshard4_p2_p99_ms": paths["vshard4_p2"]["p99_ms"],
            "extension_sha256": current_sha,
            "source_json": source.name,
            "source_json_sha256": sha256(source),
        }
        rows.append(row)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"validated {len(rows)} heads against one extension and wrote {args.csv}")


if __name__ == "__main__":
    main()
