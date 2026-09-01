#!/usr/bin/env python3
"""Pinned mechanical V=32 / four-CTA-per-head K2 patch generator.

This front end reuses the audited 2-way generator at an exact SHA256, then
changes only its explicit value-shard cardinality and launch geometry.  It
never patches the vendored assignment snapshot; the target must be a clean
FlashKDA 1ce47ea worktree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path


BASE_SHA256 = "bca3248e1bf480ea51eb3bb3da0e79d8f477fb914ea17d320c0bf90679aaaf7c"


def die(message: str) -> None:
    raise RuntimeError(message)


def load_base():
    path = Path(__file__).resolve().parents[1] / "challenge_vshard" / "apply_vshard_patch.py"
    if not path.is_file():
        die(f"missing pinned 2-way generator: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BASE_SHA256:
        die(f"2-way generator digest changed: expected {BASE_SHA256}, got {digest}")
    spec = importlib.util.spec_from_file_location("c1_vshard2_pinned", path)
    if spec is None or spec.loader is None:
        die(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rewrite_kernel2(upstream: str) -> str:
    base = load_base()
    text = base.rewrite_kernel2(upstream).replace("vshard", "vshard4").replace("VShard", "VShard4")
    text = base.replace_once(
        text,
        'static_assert(KDim == 128 && VDim == 64, "vshard4 is deliberately specialized to K=128,Vshard=64");',
        'static_assert(KDim == 128 && VDim == 32, "vshard4 is deliberately specialized to K=128,Vshard=32");',
        "vshard4 dimensions",
    )
    text = base.replace_once(text, "constexpr int kComputeThreads = 128;", "constexpr int kComputeThreads = 64;", "vshard4 compute threads")
    # The base generator already made each compute warp own exactly one 16-V
    # block.  V=32 therefore needs precisely two MMA warps, plus the unchanged
    # load/store warps: NumThreads=64+32+32=128.
    if "u_acc[1]);" in text or "out_acc[1]);" in text or "tCrB_u_arr[1](_" in text:
        die("2-way rewrite did not eliminate second V-block accesses")
    if "value_shard = blockIdx.y / H" not in text:
        die("missing value-shard grid mapping")
    return text


def rewrite_launch(upstream: str) -> str:
    base = load_base()
    text = base.rewrite_launch(upstream)
    marker = "// ==================== launch_fwd_vshard (challenge only) ===================="
    start = text.index(marker)
    # Keep the upstream baseline launch untouched.  Only the appended vshard
    # variant contains the 2-way dimensions that must become 4-way.
    prefix, variant = text[:start], text[start:]
    variant = variant.replace("vshard", "vshard4").replace("VShard", "VShard4")
    variant = base.replace_once(
        variant,
        "using SharedStorageK2T = SharedStorageK2<K2L, kInputStages, kOutputStages>;",
        "using SharedStorageK2T = SharedStorageK2VShard4<K2L, kInputStages, kOutputStages>;",
        "vshard4 launch shared-storage type",
    )
    variant = base.replace_once(variant, "using K2L = K2VShard4Layouts<D, D / 2, CHUNK>;", "using K2L = K2VShard4Layouts<D, D / 4, CHUNK>;", "vshard4 V layout")
    variant = base.replace_once(variant, "CHUNK, D, D / 2, kInputStages, kOutputStages, kK2Threads,", "CHUNK, D, D / 4, kInputStages, kOutputStages, kK2Threads,", "vshard4 kernel V template")
    variant = base.replace_once(variant, "constexpr int kK2Threads = 32 * 2 + 128;", "constexpr int kK2Threads = 32 * 2 + 64;", "vshard4 launch threads")
    variant = base.replace_once(variant, "dim3 grid_k2(N, H * 2);", "dim3 grid_k2(N, H * 4);", "vshard4 launch grid")
    return prefix + variant


def rewrite_header(upstream: str) -> str:
    text = load_base().rewrite_header(upstream).replace("vshard", "vshard4")
    return text.replace(
        "K2 value dimension split into two independent CTA shards.",
        "K2 value dimension split into four independent CTA shards.",
    )


def rewrite_binding(upstream: str) -> str:
    text = load_base().rewrite_binding(upstream).replace("vshard", "vshard4")
    return text.replace(
        "FlashKDA Forward, 2-CTA/head value-shard challenge",
        "FlashKDA Forward, 4-CTA/head value-shard challenge",
    )


def static_check(source: Path) -> None:
    kernel = rewrite_kernel2((source / "csrc" / "smxx" / "fwd_kernel2.cuh").read_text(encoding="utf-8"))
    launch = rewrite_launch((source / "csrc" / "smxx" / "fwd_launch.cu").read_text(encoding="utf-8"))
    header = rewrite_header((source / "csrc" / "fwd.h").read_text(encoding="utf-8"))
    binding = rewrite_binding((source / "csrc" / "flash_kda.cpp").read_text(encoding="utf-8"))
    required = ("VDim == 32", "constexpr int kComputeThreads = 64;", "_flash_kda_fwd_recurrence_vshard4")
    if any(needle not in kernel for needle in required):
        die("kernel static check failed")
    required = ("using K2L = K2VShard4Layouts<D, D / 4, CHUNK>;", "SharedStorageK2VShard4<K2L, kInputStages, kOutputStages>", "CHUNK, D, D / 4, kInputStages", "kK2Threads = 32 * 2 + 64", "grid_k2(N, H * 4)", "launch_fwd_vshard4")
    if any(needle not in launch for needle in required) or 'm.def("fwd_vshard4"' not in binding:
        die("launch/binding static check failed")
    if "four independent CTA shards" not in header or "4-CTA/head value-shard challenge" not in binding:
        die("generated header/pybind CTA cardinality is stale")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    base = load_base()
    base.verify_source(source)
    static_check(source)
    if args.check_only:
        print("static check passed: V=32, 4 CTA/head, 2 MMA + load/store = 128 threads")
        return
    kernel = source / "csrc" / "smxx" / "fwd_kernel2.cuh"
    target_kernel = source / "csrc" / "smxx" / "fwd_kernel2_vshard4.cuh"
    launch = source / "csrc" / "smxx" / "fwd_launch.cu"
    header = source / "csrc" / "fwd.h"
    binding = source / "csrc" / "flash_kda.cpp"
    if target_kernel.exists():
        die(f"{target_kernel} already exists; refusing to patch twice")
    target_kernel.write_text(rewrite_kernel2(kernel.read_text(encoding="utf-8")), encoding="utf-8")
    launch_text = launch.read_text(encoding="utf-8")
    launch_text = base.replace_once(launch_text, '#include "fwd_kernel2.cuh"', '#include "fwd_kernel2.cuh"\n#include "fwd_kernel2_vshard4.cuh"', "vshard4 include")
    launch.write_text(rewrite_launch(launch_text), encoding="utf-8")
    header.write_text(rewrite_header(header.read_text(encoding="utf-8")), encoding="utf-8")
    binding.write_text(rewrite_binding(binding.read_text(encoding="utf-8")), encoding="utf-8")
    print("applied challenge-only vshard4 patch")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
