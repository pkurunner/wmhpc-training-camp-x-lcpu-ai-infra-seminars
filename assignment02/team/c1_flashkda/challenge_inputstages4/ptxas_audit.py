#!/usr/bin/env python3
"""Bind ptxas and CUBIN resource evidence to the isolated P2S4 candidate."""

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
CUBIN_SHARED = re.compile(
    r"(?:Function\s+)?([^\s\n]*_flash_kda_fwd_recurrence_vshard4_p2s4[^\s\n]*)"
    r"[\s\S]{0,400}?\b(?:SHARED|SMEM)\s*:\s*(\d+)",
    re.IGNORECASE,
)


def variant(name: str) -> str | None:
    if "_flash_kda_fwd_recurrence_vshard4_p2s4" in name:
        return "vshard4_p2s4"
    if "_flash_kda_fwd_recurrence_vshard4_p2" in name:
        return "vshard4_p2s3"
    if "_flash_kda_fwd_recurrence_vshard4" in name:
        return "vshard4_p1"
    if "_flash_kda_fwd_recurrence_vshard_p2" in name:
        return "vshard2_p2s3"
    if "_flash_kda_fwd_recurrence_vshard" in name:
        return "vshard2_p1"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="ptxas-preserving build log")
    parser.add_argument("--resource-log", type=Path, required=True, help="cuobjdump --dump-resource-usage output")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--require-p2s4-bf16-fixed-zero-spill", action="store_true")
    parser.add_argument("--require-p2s4-shared-memory-evidence", action="store_true")
    args = parser.parse_args()
    raw = args.log.read_bytes()
    resource_raw = args.resource_log.read_bytes()
    entries: list[dict[str, object]] = []
    for match in ENTRY.finditer(raw.decode("utf-8", errors="replace")):
        name, arch, stack, stores, loads, registers, barriers = match.groups()
        label = variant(name)
        if label is None:
            continue
        flags = FLAGS.findall(name)
        if len(flags) != 1:
            raise RuntimeError(f"cannot decode state flags for {label}: found {len(flags)}")
        state_in, state_out, state_fp32, is_varlen = (bool(int(value)) for value in flags[0])
        entries.append(
            {
                "variant": label,
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
    expected = ("vshard2_p1", "vshard2_p2s3", "vshard4_p1", "vshard4_p2s3", "vshard4_p2s4")
    if not entries or any(not any(entry["variant"] == label for entry in entries) for label in expected):
        raise RuntimeError("ptxas log does not contain every isolated P1/P2S3/P2S4 variant")
    formal = [
        entry
        for entry in entries
        if entry["variant"] == "vshard4_p2s4"
        and entry["has_state_in"]
        and entry["has_state_out"]
        and not entry["state_fp32"]
        and not entry["is_varlen"]
    ]
    if len(formal) != 1:
        raise RuntimeError(f"expected exactly one fixed BF16 both-state P2S4 instance, found {len(formal)}")
    resource_text = resource_raw.decode("utf-8", errors="replace")
    shared_records = [
        {"function": function, "shared_memory_bytes": int(shared)}
        for function, shared in CUBIN_SHARED.findall(resource_text)
    ]
    # The compiler's resource-usage report is generated from the exact CUBIN
    # that the fresh extension loads.  A concrete SHARED/SMEM record proves the
    # candidate has a legal launch allocation; we do not infer a byte count
    # from source-level type arithmetic.
    resource_evidence = {
        "resource_log": str(args.resource_log.resolve()),
        "resource_log_sha256": hashlib.sha256(resource_raw).hexdigest(),
        "candidate_shared_records": shared_records,
        "candidate_shared_memory_evidence": bool(shared_records),
    }
    summary: dict[str, object] = {
        "log": str(args.log.resolve()),
        "log_sha256_at_parse": hashlib.sha256(raw).hexdigest(),
        "log_hash_scope": "bytes visible when ptxas_audit read the build log; surrounding tee output may append later",
        "entries": entries,
        "p2s4_formal_bf16_fixed_both_state": formal[0],
        "resource_evidence": resource_evidence,
    }
    for label in expected:
        selected = [entry for entry in entries if entry["variant"] == label]
        summary[label] = {
            "instances": len(selected),
            "registers_min": min(int(entry["registers"]) for entry in selected),
            "registers_max": max(int(entry["registers"]) for entry in selected),
            "spilled_instances": sum(
                int(entry["spill_store_bytes"]) > 0 or int(entry["spill_load_bytes"]) > 0
                for entry in selected
            ),
        }
    encoded = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if args.require_p2s4_bf16_fixed_zero_spill:
        if int(formal[0]["spill_store_bytes"]) or int(formal[0]["spill_load_bytes"]):
            raise RuntimeError("STOP: formal BF16 fixed-state P2S4 instance spills")
    if args.require_p2s4_shared_memory_evidence and not shared_records:
        raise RuntimeError("STOP: cuobjdump did not expose candidate P2S4 SHARED/SMEM resource evidence")


if __name__ == "__main__":
    main()
