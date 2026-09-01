#!/usr/bin/env python3
"""Audit V8-P1/P2 resources and enforce the formal P1 zero-spill gate."""

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
    parser.add_argument("--require-p1-formal-zero-spill", action="store_true")
    args = parser.parse_args()
    raw = args.log.read_bytes()
    entries: list[dict[str, object]] = []
    for match in ENTRY.finditer(raw.decode("utf-8", errors="replace")):
        name, arch, stack, stores, loads, registers, barriers = match.groups()
        if "_flash_kda_fwd_recurrence_vshard8_p2" in name:
            variant = "vshard8_p2"
        elif "_flash_kda_fwd_recurrence_vshard8" in name:
            variant = "vshard8_p1"
        else:
            continue
        flags = FLAGS.findall(name)
        if len(flags) != 1:
            raise RuntimeError(f"cannot decode {variant} state flags")
        state_in, state_out, state_fp32, is_varlen = (
            bool(int(value)) for value in flags[0]
        )
        entries.append(
            {
                "variant": variant,
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
    for variant in ("vshard8_p1", "vshard8_p2"):
        if not any(entry["variant"] == variant for entry in entries):
            raise RuntimeError(f"missing {variant} ptxas instances")
    formal = {
        variant: [
            entry
            for entry in entries
            if entry["variant"] == variant
            and entry["has_state_in"]
            and entry["has_state_out"]
            and not entry["state_fp32"]
            and not entry["is_varlen"]
        ]
        for variant in ("vshard8_p1", "vshard8_p2")
    }
    if any(len(selected) != 1 for selected in formal.values()):
        raise RuntimeError("expected one formal BF16 fixed-state instance per V8 variant")
    summary = {
        "log": str(args.log.resolve()),
        "log_sha256_at_parse": hashlib.sha256(raw).hexdigest(),
        "formal_bf16_fixed_state": {
            variant: selected[0] for variant, selected in formal.items()
        },
        "entries": entries,
    }
    encoded = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    p1 = formal["vshard8_p1"][0]
    if args.require_p1_formal_zero_spill and (
        p1["spill_store_bytes"] or p1["spill_load_bytes"]
    ):
        raise RuntimeError("STOP: formal BF16 fixed-state vshard8-P1 instance spills")


if __name__ == "__main__":
    main()
