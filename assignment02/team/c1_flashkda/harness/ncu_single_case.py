#!/usr/bin/env python3
"""Launch exactly one official BF16-state FlashKDA forward for NCU.

Unlike benchmarks/bench_fwd.py, this intentionally does not also run no-state,
FP32-state, FLA, or another benchmark.  A K1/K2 NCU report therefore has one
well-defined shape and state ABI instead of several variants mixed together.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as functional

import flash_kda


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--H", type=int, default=96)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    if min(args.T, args.H, args.D) <= 0:
        raise SystemExit("T, H and D must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for NCU collection")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    shape = (1, args.T, args.H, args.D)
    q = functional.normalize(torch.randn(shape, dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    k = functional.normalize(torch.randn(shape, dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device)
    g = torch.randn(shape, dtype=torch.bfloat16, device=device)
    beta = torch.randn((1, args.T, args.H), dtype=torch.bfloat16, device=device)
    a_log = torch.rand(args.H, dtype=torch.float32, device=device)
    dt_bias = torch.rand((args.H, args.D), dtype=torch.float32, device=device)
    initial_state = torch.arange(
        args.H * args.D * args.D, dtype=torch.float32, device=device
    ).reshape(1, args.H, args.D, args.D).to(torch.bfloat16)
    final_state = torch.zeros_like(initial_state)
    out = torch.zeros_like(q)
    torch.cuda.synchronize()
    print(
        "NCU_SINGLE_CASE "
        f"shape={list(shape)} state=bf16 seed={args.seed} "
        f"device={torch.cuda.get_device_name(device)}"
    )
    # One call only: NCU's kernel filter then captures the matching K1/K2 pair
    # for precisely this BF16-state ABI.
    flash_kda.fwd(
        q,
        k,
        v,
        g,
        beta,
        1.0 / math.sqrt(args.D),
        out,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        initial_state=initial_state,
        final_state=final_state,
    )
    torch.cuda.synchronize()
    print("NCU_SINGLE_CASE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
