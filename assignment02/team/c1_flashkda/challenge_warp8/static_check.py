#!/usr/bin/env python3
"""Read-only structural checks for the challenge_warp8 generator."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("challenge_warp8_patch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="read-only upstream FlashKDA checkout")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    patch = load_generator(root / "apply_warp8_patch.py")
    source = args.source.resolve()

    kernel = patch.rewrite_kernel2((source / "csrc" / "smxx" / "fwd_kernel2.cuh").read_text(encoding="utf-8"))
    launch = patch.rewrite_launch((source / "csrc" / "smxx" / "fwd_launch.cu").read_text(encoding="utf-8"))
    header = patch.rewrite_header((source / "csrc" / "fwd.h").read_text(encoding="utf-8"))
    binding = patch.rewrite_binding((source / "csrc" / "flash_kda.cpp").read_text(encoding="utf-8"))

    required = {
        "kernel": ("_flash_kda_fwd_recurrence_warp8", "constexpr int kComputeThreads = 256;", "SharedStorageK2Warp8", "K2LayoutsWarp8"),
        "launch": ("launch_fwd_warp8", "constexpr int kK2Threads = 32 * 2 + 256;", "_flash_kda_fwd_recurrence_warp8"),
        "header": ("launch_fwd_warp8",),
        "binding": ("void fwd_warp8(", "launch_fwd_warp8<", 'm.def("fwd_warp8"'),
    }
    generated = {"kernel": kernel, "launch": launch, "header": header, "binding": binding}
    for name, needles in required.items():
        for needle in needles:
            if needle not in generated[name]:
                raise RuntimeError(f"{name} missing required token: {needle}")
    for forbidden in ("u_acc[1]);", "out_acc[1]);", "tCrB_u_arr[1](_", "warp_id * 2"):
        if forbidden in kernel:
            raise RuntimeError(f"kernel retains unsafe two-block token: {forbidden}")
    print("static check passed: full V, 8 MMA warps + load/store = 320 threads")


if __name__ == "__main__":
    main()
