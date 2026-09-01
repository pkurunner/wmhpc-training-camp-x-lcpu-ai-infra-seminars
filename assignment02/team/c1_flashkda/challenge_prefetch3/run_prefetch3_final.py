#!/usr/bin/env python3
"""H64 all-state exact gate and full-call P2S3/P3S3 AB/BA benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2  # noqa: E402
from assignment02.team.c1_flashkda.challenge_prefetch3 import prefetch3  # noqa: E402
from assignment02.team.c1_flashkda.harness import validate_and_bench as common  # noqa: E402


FROZEN_P1_B300_MS = 0.799616
ABSOLUTE_STRICT_MS = FROZEN_P1_B300_MS / 1.10
OUTPUT_TOL = {"rtol": 2e-2, "atol": 2e-2}
STATE_TOL = {"rtol": 5e-2, "atol": 5e-2}


def identity() -> dict[str, object]:
    import flash_kda_C

    required = ("fwd", "fwd_vshard", "fwd_vshard_p2", "fwd_vshard_p3", "get_workspace_size")
    missing = [name for name in required if not hasattr(flash_kda_C, name)]
    if missing:
        raise RuntimeError(f"loaded extension lacks required ABI: {missing}")
    path = Path(flash_kda_C.__file__).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "symbols": list(required)}


def exact_case(torch_ref: Callable[..., None], x: common.Inputs, mode: str, seed: int) -> dict[str, object]:
    initial, _ = common.state_tensors(mode, x.q.shape[2], seed)

    def run(fn: Callable[..., None]) -> tuple[torch.Tensor, torch.Tensor | None]:
        state_in = None if initial is None else initial.clone()
        state_out = None if initial is None else torch.zeros_like(initial)
        result = common.invoke(fn, x, state_in, state_out)
        torch.cuda.synchronize()
        return result

    out_p2, state_p2 = run(prefetch2.fwd)
    out_p3, state_p3 = run(prefetch3.fwd)
    out_ref, state_ref = run(torch_ref)
    common.require_exact(f"{mode}/p3_vs_p2/output", out_p3, out_p2)
    result: dict[str, object] = {
        "p3_vs_p2": {"output_exact": True, "output_max_abs": common.max_abs(out_p3, out_p2)},
    }
    if state_p3 is not None and state_p2 is not None:
        common.require_exact(f"{mode}/p3_vs_p2/final_state", state_p3, state_p2)
        result["p3_vs_p2"].update({"final_state_exact": True, "final_state_max_abs": common.max_abs(state_p3, state_p2)})
    for label, out, state in (("p2_vs_torch_ref", out_p2, state_p2), ("p3_vs_torch_ref", out_p3, state_p3)):
        common.require_close(f"{mode}/{label}/output", out, out_ref, **OUTPUT_TOL)
        item: dict[str, object] = {
            "output_close": True, "output_bitwise_exact": torch.equal(out, out_ref),
            "output_max_abs": common.max_abs(out, out_ref), "output_tolerance": OUTPUT_TOL,
        }
        if state is not None and state_ref is not None:
            common.require_close(f"{mode}/{label}/final_state", state, state_ref, **STATE_TOL)
            item.update({
                "final_state_close": True, "final_state_bitwise_exact": torch.equal(state, state_ref),
                "final_state_max_abs": common.max_abs(state, state_ref), "final_state_tolerance": STATE_TOL,
            })
        result[label] = item
    return result


def full_call_samples(x: common.Inputs, warmup: int, samples: int) -> dict[str, object]:
    initial, final_p2 = common.state_tensors("bf16", x.q.shape[2], 17)
    p3_initial = initial.clone()
    _, final_p3 = common.state_tensors("bf16", x.q.shape[2], 19)
    out_p2, out_p3 = torch.empty_like(x.v), torch.empty_like(x.v)

    def run_p2() -> None:
        prefetch2.fwd(x.q, x.k, x.v, x.g, x.beta, x.scale, out_p2, x.a_log, x.dt_bias, x.lower_bound,
                      initial_state=initial, final_state=final_p2)

    def run_p3() -> None:
        prefetch3.fwd(x.q, x.k, x.v, x.g, x.beta, x.scale, out_p3, x.a_log, x.dt_bias, x.lower_bound,
                      initial_state=p3_initial, final_state=final_p3)

    for index in range(warmup):
        first, second = (run_p2, run_p3) if index % 2 == 0 else (run_p3, run_p2)
        first(); second()
    torch.cuda.synchronize()
    p2_values: list[float] = []
    p3_values: list[float] = []
    order: list[str] = []
    for index in range(samples):
        p2_first = index % 2 == 0
        first, second = (run_p2, run_p3) if p2_first else (run_p3, run_p2)
        s1, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s2, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s1.record(); first(); e1.record()
        s2.record(); second(); e2.record()
        torch.cuda.synchronize()
        first_ms, second_ms = float(s1.elapsed_time(e1)), float(s2.elapsed_time(e2))
        if p2_first:
            p2_values.append(first_ms); p3_values.append(second_ms); order.append("P2,P3")
        else:
            p3_values.append(first_ms); p2_values.append(second_ms); order.append("P3,P2")

    def summary(values: list[float]) -> dict[str, object]:
        return {
            "mean_ms": statistics.fmean(values), "median_ms": statistics.median(values),
            "min_ms": min(values), "max_ms": max(values), "samples": len(values), "raw_event_ms": values,
        }

    p2, p3 = summary(p2_values), summary(p3_values)
    p2_median, p3_median = float(p2["median_ms"]), float(p3["median_ms"])
    return {
        "p2_current": p2, "p3": p3, "abba_order": order,
        "p2_over_p3_median_x": p2_median / p3_median,
        "same_allocation_1_10_pass": p2_median / p3_median >= 1.10,
        "absolute_strict_ms": ABSOLUTE_STRICT_MS,
        "absolute_strict_pass": p3_median <= ABSOLUTE_STRICT_MS,
        "full_strict_pass": p2_median / p3_median >= 1.10 and p3_median <= ABSOLUTE_STRICT_MS,
        "event_contract": "AB/BA alternating; one full public wrapper call per CUDA event; workspace allocation included",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("exact", "bench"), required=True)
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.T % 16 or args.H <= 0:
        raise RuntimeError("requires CUDA, positive H, and T divisible by 16")
    import flash_kda  # noqa: F401

    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
        "build_target": os.environ.get("C1_BUILD_TARGET", "unspecified"),
        "shape": {"B": 1, "T": args.T, "H": args.H, "K": 128, "V": 128},
        "extension": identity(), "phase": args.phase, "seed": args.seed,
    }
    x = common.make_inputs(args.T, args.H, args.seed)
    if args.phase == "exact":
        torch_ref = common._load_torch_ref(args.reference_root)
        result["correctness"] = {
            mode: exact_case(torch_ref, x, mode, args.seed + index * 101)
            for index, mode in enumerate(("none", "bf16", "fp32"))
        }
        result["exact_gate_pass"] = True
    else:
        result["benchmark"] = full_call_samples(x, args.warmup, args.samples)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result.get("benchmark", {"exact_gate_pass": True}), ensure_ascii=False, indent=2))
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
