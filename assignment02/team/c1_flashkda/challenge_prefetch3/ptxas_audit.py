#!/usr/bin/env python3
"""Extract P1/P2/P3 recurrence register and spill evidence from a ptxas log."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ENTRY = re.compile(
    r"ptxas info\s+: Compiling entry function '([^']+)' for '([^']+)'\n"
    r"ptxas info\s+: Function properties for [^\n]+\n"
    r"\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\n"
    r"ptxas info\s+: Used (\d+) registers, used (\d+) barriers"
)
FLAGS = re.compile(r"ELb([01])ELb([01])ELb([01])ELb([01])EEvT_T0")


def variant(name: str) -> str | None:
    if name.startswith("_Z35_flash_kda_fwd_recurrence_vshard_p3"):
        return "P3"
    if name.startswith("_Z35_flash_kda_fwd_recurrence_vshard_p2"):
        return "P2"
    if name.startswith("_Z32_flash_kda_fwd_recurrence_vshard"):
        return "P1"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    raw = args.log.read_bytes()
    entries: list[dict[str, object]] = []
    for match in ENTRY.finditer(raw.decode("utf-8", errors="replace")):
        name, arch, stack, stores, loads, registers, barriers = match.groups()
        label = variant(name)
        if label is None:
            continue
        flags = FLAGS.findall(name)
        if len(flags) != 1:
            raise RuntimeError(f"cannot decode state flags for {label}: found {len(flags)}")
        state_in, state_out, state_fp32, is_varlen = (bool(int(x)) for x in flags[0])
        entries.append({
            "variant": label, "arch": arch,
            "has_state_in": state_in, "has_state_out": state_out,
            "state_fp32": state_fp32, "is_varlen": is_varlen,
            "stack_bytes": int(stack), "spill_store_bytes": int(stores),
            "spill_load_bytes": int(loads), "registers": int(registers),
            "barriers": int(barriers),
        })
    result: dict[str, object] = {
        "log": str(args.log.resolve()), "log_sha256": hashlib.sha256(raw).hexdigest(), "entries": entries,
    }
    for label in ("P1", "P2", "P3"):
        selected = [entry for entry in entries if entry["variant"] == label]
        if len(selected) != 14:
            raise RuntimeError(f"expected 14 {label} entries, found {len(selected)}")
        result[label] = {
            "instances": len(selected),
            "registers_min": min(int(entry["registers"]) for entry in selected),
            "registers_max": max(int(entry["registers"]) for entry in selected),
            "spilled_instances": sum(
                int(entry["spill_store_bytes"]) > 0 or int(entry["spill_load_bytes"]) > 0
                for entry in selected
            ),
        }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
