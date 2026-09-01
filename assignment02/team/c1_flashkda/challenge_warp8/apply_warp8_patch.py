#!/usr/bin/env python3
"""Install an isolated full-V / eight-compute-warp FlashKDA K2 candidate.

The source clone must be an untouched FlashKDA 1ce47ea worktree.  This script
adds a separate ``fwd_warp8`` pybind symbol and never changes the upstream
``fwd`` implementation.  It deliberately clones only the K2 launch path:
K1, workspace ABI, tensors, and all state dispatch cases stay upstream.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


UPSTREAM_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
CUTLASS_COMMIT = "5c149f52a436782210263fb2f19b354443a61c6a"


def die(message: str) -> None:
    raise RuntimeError(message)


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(source), *args), text=True).strip()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        die(f"{label}: expected one upstream match, found {count}")
    return text.replace(old, new, 1)


def rewrite_kernel2(upstream: str) -> str:
    """Make a name-isolated K2 clone with 8 one-column-block MMA warps.

    The original uses four MMA warps and lets each own two 16-column V blocks.
    This clone preserves every operation but assigns one V block to each of
    eight warps.  Its two extra warps remain the original TMA load/store roles,
    hence 8*32 + 2*32 = 320 threads.
    """
    text = upstream
    # Both the upstream and cloned headers are included by fwd_launch.cu.
    # `#pragma once` is per file, so the layout type itself also needs an
    # isolated spelling to avoid a C++ template redefinition.
    text = text.replace("K2Layouts", "K2LayoutsWarp8")
    text = replace_once(text, "struct SharedStorageK2 {", "struct SharedStorageK2Warp8 {", "warp8 shared-storage type")
    text = replace_once(text, "SharedStorageK2<Layouts, InputStages, OutputStages>", "SharedStorageK2Warp8<Layouts, InputStages, OutputStages>", "warp8 shared-storage use")
    text = replace_once(text, "_flash_kda_fwd_recurrence(", "_flash_kda_fwd_recurrence_warp8(", "warp8 kernel symbol")
    text = replace_once(text, "constexpr int kComputeThreads = 128;", "constexpr int kComputeThreads = 256;", "warp8 compute threads")
    text = replace_once(
        text,
        "// Each warp handles TWO 16x16 column blocks (N=128 / 4 warps = 32 = 2 x 16)",
        "// Eight MMA warps each own one 16x16 V block (N=128 / 8 = 16).",
        "warp8 phase-1 comment",
    )
    text = replace_once(
        text,
        "// Each warp handles columns [warp_id*32, (warp_id+1)*32] = 2 x 16x16 blocks",
        "// Each warp owns one 16-column block of the full V dimension.",
        "warp8 phase-6 comment",
    )

    # A single accumulator/output block is live per warp.  The transformations
    # below are intentionally limited to the K2 two-block loops; the 2-element
    # u_b_regs array is a fragment packing detail and must remain length two.
    for old, new, label in (
        ("AccFragT u_acc[2], out_acc[2];", "AccFragT u_acc[1], out_acc[1];", "warp8 accumulator array"),
        ("SFragT out_bf16[2];", "SFragT out_bf16[1];", "warp8 output fragment array"),
        ("SFragT v_bf16[2];", "SFragT v_bf16[1];", "warp8 V fragment array"),
        ("SFragT u_bf16[2];", "SFragT u_bf16[1];", "warp8 U fragment array"),
        ("BFragT_u tCrB_u_arr[2];", "BFragT_u tCrB_u_arr[1];", "warp8 U-B array"),
        ("SFragT ring_S_acc[2][PREFETCH];", "SFragT ring_S_acc[1][PREFETCH];", "warp8 state ring"),
    ):
        text = replace_once(text, old, new, label)

    i_loops = text.count("for (int i = 0; i < 2; ++i)")
    bi_loops = text.count("for (int bi = 0; bi < 2; ++bi)")
    if i_loops != 7 or bi_loops != 3:
        die(f"unexpected two-block loop count: i={i_loops}, bi={bi_loops}")
    text = text.replace("for (int i = 0; i < 2; ++i)", "for (int i = 0; i < 1; ++i)")
    text = text.replace("for (int bi = 0; bi < 2; ++bi)", "for (int bi = 0; bi < 1; ++bi)")

    # Map the eight warps onto V blocks 0..7.  Do this before deleting the old
    # second-block prefetch so no `warp_id * 2` indexing can survive silently.
    text = text.replace("warp_id * 2 + i", "warp_id")
    text = text.replace("warp_id * 2 + bi", "warp_id")
    text = text.replace("warp_id * 2", "warp_id")

    second_block_prefetch = """                copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                    local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id + 1, k))), tCrBi_view);
"""
    text = replace_once(text, second_block_prefetch, "                // warp8: one V block per MMA warp\n", "warp8 second-block prefetch")
    text = replace_once(
        text,
        """                cute::transform(tCrBi, tCrB, cute::identity{});

                if (k + 1 < K_BLOCKS) {""",
        """                if (k + 1 < K_BLOCKS) {""",
        "warp8 second B transform",
    )
    text = replace_once(
        text,
        "                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[1]);\n",
        "",
        "warp8 second U GEMM",
    )
    text = replace_once(
        text,
        "                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[1]);\n",
        "",
        "warp8 second output GEMM",
    )

    # Array extents are intentionally ``[1]``.  Reject only index-1 *uses*,
    # not the declaration itself.
    forbidden = ("u_acc[1]);", "out_acc[1]);", "tCrB_u_arr[1](_", "warp_id * 2")
    leftovers = [needle for needle in forbidden if needle in text]
    if leftovers:
        die("unsafe two-block references remain: " + ", ".join(leftovers))
    if "struct K2LayoutsWarp8" not in text or "constexpr int kComputeThreads = 256;" not in text:
        die("warp8 compute-thread rewrite disappeared")
    return text


def rewrite_launch(upstream: str) -> str:
    marker = "// ==================== launch_fwd ===================="
    start = upstream.index(marker)
    variant = upstream[start:]
    variant = variant.replace("launch_fwd(", "launch_fwd_warp8(")
    variant = variant.replace("launch_fwd<", "launch_fwd_warp8<")
    variant = variant.replace("INSTANTIATE_LAUNCH_FWD", "INSTANTIATE_WARP8_LAUNCH_FWD")
    variant = variant.replace("INSTANTIATE_STATE_VARIANTS", "INSTANTIATE_WARP8_STATE_VARIANTS")
    variant = replace_once(variant, "using SharedStorageK2T = SharedStorageK2<K2L, kInputStages, kOutputStages>;", "using SharedStorageK2T = SharedStorageK2Warp8<K2L, kInputStages, kOutputStages>;", "warp8 launch storage")
    variant = replace_once(variant, "constexpr int kK2Threads = 32 * 2 + 128;", "constexpr int kK2Threads = 32 * 2 + 256;", "warp8 launch threads")
    variant = replace_once(variant, "auto kernel2 = _flash_kda_fwd_recurrence<", "auto kernel2 = _flash_kda_fwd_recurrence_warp8<", "warp8 launch symbol")
    return upstream + "\n\n// ==================== launch_fwd_warp8 (challenge only) ====================\n" + variant


def rewrite_header(upstream: str) -> str:
    declaration = upstream[upstream.index("template <int D"):]
    return upstream + "\n// Full-V K2 with eight MMA warps; challenge-only entry.\n" + declaration.replace("launch_fwd(", "launch_fwd_warp8(")


def rewrite_binding(upstream: str) -> str:
    function_start = upstream.index("void fwd(")
    pybind_start = upstream.index("\nPYBIND11_MODULE", function_start)
    fwd_warp8 = upstream[function_start:pybind_start]
    fwd_warp8 = fwd_warp8.replace("void fwd(", "void fwd_warp8(", 1)
    fwd_warp8 = fwd_warp8.replace("launch_fwd<", "launch_fwd_warp8<")
    binding = """    m.def(\"fwd_warp8\", &fwd_warp8, \"FlashKDA Forward, full-V eight-MMA-warp K2 challenge\",
        py::arg(\"q\"), py::arg(\"k\"), py::arg(\"v\"), py::arg(\"g\"), py::arg(\"beta\"),
        py::arg(\"scale\"), py::arg(\"out\"), py::arg(\"workspace\"),
        py::arg(\"A_log\"), py::arg(\"dt_bias\"), py::arg(\"lower_bound\"),
        py::arg(\"initial_state\") = py::none(), py::arg(\"final_state\") = py::none(),
        py::arg(\"cu_seqlens\") = py::none());
"""
    result = upstream[:pybind_start] + "\n" + fwd_warp8 + upstream[pybind_start:]
    return replace_once(result, '    m.def("get_workspace_size",', binding + '    m.def("get_workspace_size",', "warp8 pybind entry")


def verify_source(source: Path) -> None:
    if git(source, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        die(f"expected FlashKDA commit {UPSTREAM_COMMIT}, got {git(source, 'rev-parse', 'HEAD')}")
    if git(source, "status", "--porcelain"):
        die("source tree is dirty; use a fresh dedicated worktree")
    if CUTLASS_COMMIT not in git(source, "submodule", "status", "cutlass"):
        die("cutlass submodule is not the pinned upstream revision")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="fresh FlashKDA 1ce47ea clone/worktree")
    args = parser.parse_args()
    source = args.source.resolve()
    verify_source(source)

    kernel = source / "csrc" / "smxx" / "fwd_kernel2.cuh"
    warp8_kernel = source / "csrc" / "smxx" / "fwd_kernel2_warp8.cuh"
    launch = source / "csrc" / "smxx" / "fwd_launch.cu"
    header = source / "csrc" / "fwd.h"
    binding = source / "csrc" / "flash_kda.cpp"
    if warp8_kernel.exists():
        die(f"{warp8_kernel} already exists; refusing to patch twice")

    warp8_kernel.write_text(rewrite_kernel2(kernel.read_text(encoding="utf-8")), encoding="utf-8")
    launch_text = launch.read_text(encoding="utf-8")
    launch_text = replace_once(launch_text, '#include "fwd_kernel2.cuh"', '#include "fwd_kernel2.cuh"\n#include "fwd_kernel2_warp8.cuh"', "warp8 include")
    launch.write_text(rewrite_launch(launch_text), encoding="utf-8")
    header.write_text(rewrite_header(header.read_text(encoding="utf-8")), encoding="utf-8")
    binding.write_text(rewrite_binding(binding.read_text(encoding="utf-8")), encoding="utf-8")
    print("applied challenge-only full-V warp8 patch")
    print("next: CXX=g++ FLASH_KDA_CUDA_ARCHS=103a NVCC_THREADS=8 python setup.py build_ext --inplace")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
