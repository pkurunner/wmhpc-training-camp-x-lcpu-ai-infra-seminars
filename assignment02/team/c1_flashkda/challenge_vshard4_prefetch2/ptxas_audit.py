#!/usr/bin/env python3
"""Extract resource evidence for the one-shot vshard2/vshard4 comparison SO."""

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
    # Mangled name lengths can vary by compiler; the isolated symbol stem cannot.
    if "_flash_kda_fwd_recurrence_vshard4_p2" in name:
        return "vshard4_p2"
    if "_flash_kda_fwd_recurrence_vshard4" in name:
        return "vshard4_p1"
    if "_flash_kda_fwd_recurrence_vshard_p2" in name:
        return "vshard2_p2"
    if "_flash_kda_fwd_recurrence_vshard" in name:
        return "vshard2_p1"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--require-vshard4-p2-bf16-fixed-zero-spill", action="store_true")
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
            raise RuntimeError(f"cannot decode state flags for {label}: found {len(flags)} in {name}")
        state_in, state_out, state_fp32, is_varlen = (bool(int(value)) for value in flags[0])
        entries.append({
            "variant": label, "arch": arch, "has_state_in": state_in, "has_state_out": state_out,
            "state_fp32": state_fp32, "is_varlen": is_varlen, "stack_bytes": int(stack),
            "spill_store_bytes": int(stores), "spill_load_bytes": int(loads),
            "registers": int(registers), "barriers": int(barriers), "mangled_name": name,
        })
    expected = ("vshard2_p1", "vshard2_p2", "vshard4_p1", "vshard4_p2")
    if not entries or any(not any(entry["variant"] == label for entry in entries) for label in expected):
        raise RuntimeError("ptxas log does not contain all vshard2/vshard4 P1/P2 variants")
    summary: dict[str, object] = {
        "log": str(args.log.resolve()),
        "log_sha256_at_parse": hashlib.sha256(raw).hexdigest(),
        "log_hash_scope": (
            "bytes visible when ptxas_audit read the log; a surrounding tee may append "
            "the emitted JSON and build epilogue afterward"
        ),
        "entries": entries,
    }
    for label in expected:
        selected = [entry for entry in entries if entry["variant"] == label]
        summary[label] = {
            "instances": len(selected),
            "registers_min": min(int(entry["registers"]) for entry in selected),
            "registers_max": max(int(entry["registers"]) for entry in selected),
            "spilled_instances": sum(int(entry["spill_store_bytes"]) > 0 or int(entry["spill_load_bytes"]) > 0 for entry in selected),
        }
    formal = [entry for entry in entries if entry["variant"] == "vshard4_p2" and entry["has_state_in"]
              and entry["has_state_out"] and not entry["state_fp32"] and not entry["is_varlen"]]
    if len(formal) != 1:
        raise RuntimeError(f"expected exactly one fixed-length BF16-state vshard4_p2 instance, found {len(formal)}")
    summary["vshard4_p2_formal_bf16_fixed_state"] = formal[0]
    encoded = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if args.require_vshard4_p2_bf16_fixed_zero_spill:
        entry = formal[0]
        if int(entry["spill_store_bytes"]) or int(entry["spill_load_bytes"]):
            raise RuntimeError("STOP: formal BF16 fixed-state vshard4_p2 instance spills")


if __name__ == "__main__":
    main()
