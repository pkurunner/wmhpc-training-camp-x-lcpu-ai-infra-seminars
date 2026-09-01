#!/usr/bin/env python3
"""Extract resources and enforce zero spill for the formal vshard8-P2 path."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--require-formal-zero-spill", action="store_true")
    args = parser.parse_args()
    raw = args.log.read_bytes()
    entries: list[dict[str, object]] = []
    for match in ENTRY.finditer(raw.decode("utf-8", errors="replace")):
        name, arch, stack, stores, loads, registers, barriers = match.groups()
        if "_flash_kda_fwd_recurrence_vshard8_p2" not in name:
            continue
        flags = FLAGS.findall(name)
        if len(flags) != 1:
            raise RuntimeError(f"cannot decode vshard8 state flags: found {len(flags)}")
        state_in, state_out, state_fp32, is_varlen = (
            bool(int(value)) for value in flags[0]
        )
        entries.append(
            {
                "arch": arch,
                "has_state_in": state_in,
                "has_state_out": state_out,
                "state_fp32": state_fp32,
                "is_varlen": is_varlen,
                "stack_bytes": int(stack),
                "spill_store_bytes": int(stores),
                "spill_load_bytes": int(loads),
                "registers": int(registers),
                "barriers": int(barriers),
                "mangled_name": name,
            }
        )
    if not entries:
        raise RuntimeError("ptxas log contains no vshard8-P2 instances")
    formal = [
        entry
        for entry in entries
        if entry["has_state_in"]
        and entry["has_state_out"]
        and not entry["state_fp32"]
        and not entry["is_varlen"]
    ]
    if len(formal) != 1:
        raise RuntimeError(
            f"expected one fixed BF16-state vshard8-P2 instance, found {len(formal)}"
        )
    summary = {
        "log": str(args.log.resolve()),
        "log_sha256_at_parse": hashlib.sha256(raw).hexdigest(),
        "instances": len(entries),
        "registers_min": min(int(entry["registers"]) for entry in entries),
        "registers_max": max(int(entry["registers"]) for entry in entries),
        "spilled_instances": sum(
            bool(entry["spill_store_bytes"] or entry["spill_load_bytes"])
            for entry in entries
        ),
        "formal_bf16_fixed_state": formal[0],
        "entries": entries,
    }
    encoded = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if args.require_formal_zero_spill and (
        formal[0]["spill_store_bytes"] or formal[0]["spill_load_bytes"]
    ):
        raise RuntimeError("STOP: formal BF16 fixed-state vshard8-P2 instance spills")


if __name__ == "__main__":
    main()
