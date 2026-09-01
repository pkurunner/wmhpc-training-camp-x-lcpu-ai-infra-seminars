#!/usr/bin/env python3
"""Small valid-input matrix intended to run under compute-sanitizer memcheck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2  # noqa: E402
from assignment02.team.c1_flashkda.challenge_varlen_tail.run_varlen_tail import (  # noqa: E402
    Case,
    _compare,
    _invoke,
    _make_inputs,
    _states,
)
from assignment02.team.c1_flashkda.challenge_vshard4_prefetch2 import (  # noqa: E402
    vshard4_prefetch2,
)


CASES = (
    Case("san_fixed_t1_h1", 1, 1, heads=1),
    Case("san_fixed_t17_h1", 1, 17, heads=1),
    Case("san_batch_b2_t17_h1", 2, 17, heads=1),
    Case("san_varlen_15_17_h1", 1, 32, heads=1, lengths=(15, 17)),
    Case("san_varlen_1_15_17_31_h1", 1, 64, heads=1, lengths=(1, 15, 17, 31)),
)
CONTRACTS = ("none", "fp32_both")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    import flash_kda

    functions: dict[str, Callable[..., None]] = {
        "baseline": flash_kda.fwd,
        "vshard2_p2": prefetch2.fwd,
        "vshard4_p2": vshard4_prefetch2.fwd,
    }
    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "cases": [case.name for case in CASES],
        "contracts": list(CONTRACTS),
        "correctness": {},
        "exact_gate_pass": False,
        "intended_tool": "compute-sanitizer --tool memcheck --padding 32",
    }
    for case_index, case in enumerate(CASES):
        x = _make_inputs(case, args.seed + case_index * 1009)
        result["correctness"][case.name] = {}  # type: ignore[index]
        for contract_index, contract in enumerate(CONTRACTS):
            initial, final = _states(
                contract, case, args.seed + case_index * 1009 + contract_index * 101
            )
            outputs = {
                label: _invoke(
                    fn,
                    x,
                    None if initial is None else initial.clone(),
                    None if final is None else final.clone(),
                )
                for label, fn in functions.items()
            }
            baseline = outputs["baseline"]
            result["correctness"][case.name][contract] = {  # type: ignore[index]
                label: _compare(
                    f"{case.name}/{contract}/{label}_vs_baseline",
                    outputs[label],
                    baseline,
                )
                for label in ("vshard2_p2", "vshard4_p2")
            }
    result["exact_gate_pass"] = True
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
