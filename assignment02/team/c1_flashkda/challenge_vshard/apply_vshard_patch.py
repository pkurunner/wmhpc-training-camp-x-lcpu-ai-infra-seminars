#!/usr/bin/env python3
"""Apply the isolated 2-CTA/head K2 value-shard patch to FlashKDA 1ce47ea.

The script intentionally refuses a dirty or mismatched source tree.  It adds a
separate pybind entry point (``fwd_vshard``) and leaves ``fwd`` bit-for-bit on
the upstream source path.  Run it only on a disposable clone/worktree, never
on the vendored assignment snapshot.
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
        die(f"{label}: expected exactly one matching upstream block, found {count}")
    return text.replace(old, new, 1)


VSHARD_LAYOUT = r'''template <int KDim, int VDim, int CHUNK = 16>
struct K2VShardLayouts {
    // K-side workspace remains [CHUNK, KDim]. Only V/output is sharded.
    // Upstream's raw state ABI is physically [V, K], so a CTA owns a
    // [VDim, KDim] state slice even though its mathematical view is [K, VDim].
    using MMALayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<KDim>{}),
        LayoutLeft{}
    ));
    using TransposedMMALayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<KDim>{}, Int<CHUNK>{}),
        LayoutRight{}
    ));
    using VOLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<VDim>{}),
        LayoutLeft{}
    ));
    using TransposedVOLayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<VDim>{}, Int<CHUNK>{}),
        LayoutRight{}
    ));
    using BetaSmemLayout = Layout<Shape<Int<32>>, Stride<Int<1>>>;
    using StateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<VDim>{}, Int<KDim>{}),
        LayoutLeft{}
    ));
    using TransposedStateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<KDim>{}, Int<VDim>{}),
        LayoutRight{}
    ));
    using GTotalLayout = Layout<Shape<Int<KDim>>, Stride<Int<1>>>;
    using LMLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<CHUNK>{}),
        LayoutLeft{}
    ));

    using TMAMMLayout = decltype(composition(
        MMALayout{}.layout_a(), MMALayout{}.offset(), prepend(MMALayout{}.layout_b())
    ));
    using TMABetaSmemLayout = BetaSmemLayout;
    using TMAVOLayout = decltype(composition(
        VOLayout{}.layout_a(), VOLayout{}.offset(), prepend(VOLayout{}.layout_b())
    ));
    using TMAStateSmemLayout = decltype(composition(
        StateSmemLayout{}.layout_a(), StateSmemLayout{}.offset(), prepend(StateSmemLayout{}.layout_b())
    ));
    using TMALMLayout = decltype(composition(
        LMLayout{}.layout_a(), LMLayout{}.offset(), prepend(LMLayout{}.layout_b())
    ));
    using TMAGTotalSmemLayout = decltype(prepend(GTotalLayout{}));
    using FP32StateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW32_Atom<float>{},
        make_shape(Int<VDim>{}, Int<KDim>{}),
        LayoutLeft{}
    ));
    using TMAFP32StateSmemLayout = decltype(composition(
        FP32StateSmemLayout{}.layout_a(), FP32StateSmemLayout{}.offset(),
        prepend(FP32StateSmemLayout{}.layout_b())
    ));
};

template <class FP32Layout, class BF16Layout, int KDim, int VDim, int NumThreads>
__device__ void vshard_cvt_fp32_to_bf16(float* fp32_smem, cutlass::bfloat16_t* bf16_smem, int tid) {
    using BF16 = cutlass::bfloat16_t;
    constexpr int kBlock = 8;
    constexpr int kRowBlocks = KDim / kBlock;
    constexpr int kColBlocks = VDim / kBlock;
    auto fp32_view = make_tensor(make_smem_ptr(fp32_smem), FP32Layout{});
    auto bf16_view = make_tensor(make_smem_ptr(bf16_smem), BF16Layout{});
    for (int blk = tid / 32; blk < kRowBlocks * kColBlocks; blk += NumThreads / 32) {
        int br = (blk / kColBlocks) * kBlock, bc = (blk % kColBlocks) * kBlock;
        int e0 = (tid % 32) * 2, e1 = e0 + 1;
        bf16_view(br + e0 / kBlock, bc + e0 % kBlock) = BF16(fp32_view(br + e0 / kBlock, bc + e0 % kBlock));
        bf16_view(br + e1 / kBlock, bc + e1 % kBlock) = BF16(fp32_view(br + e1 / kBlock, bc + e1 % kBlock));
    }
}

template <class BF16Layout, class FP32Layout, int KDim, int VDim, int NumThreads>
__device__ void vshard_cvt_bf16_to_fp32(cutlass::bfloat16_t* bf16_smem, float* fp32_smem, int tid) {
    constexpr int kBlock = 8;
    constexpr int kRowBlocks = KDim / kBlock;
    constexpr int kColBlocks = VDim / kBlock;
    auto bf16_view = make_tensor(make_smem_ptr(bf16_smem), BF16Layout{});
    auto fp32_view = make_tensor(make_smem_ptr(fp32_smem), FP32Layout{});
    for (int blk = tid / 32; blk < kRowBlocks * kColBlocks; blk += NumThreads / 32) {
        int br = (blk / kColBlocks) * kBlock, bc = (blk % kColBlocks) * kBlock;
        int e0 = (tid % 32) * 2, e1 = e0 + 1;
        fp32_view(br + e0 / kBlock, bc + e0 % kBlock) = bf16_to_f32(bf16_view(br + e0 / kBlock, bc + e0 % kBlock));
        fp32_view(br + e1 / kBlock, bc + e1 % kBlock) = bf16_to_f32(bf16_view(br + e1 / kBlock, bc + e1 % kBlock));
    }
}
'''


def rewrite_kernel2(upstream: str) -> str:
    begin = upstream.index("template <int D, int CHUNK = 16>\nstruct K2Layouts")
    end = upstream.index("template <class Layouts, int InputStages, int OutputStages>", begin)
    text = upstream[:begin] + VSHARD_LAYOUT + "\n" + upstream[end:]
    # This header is included alongside the upstream K2 header.  The original
    # storage helper is a namespace-level class template, so it must not retain
    # its upstream spelling in the challenge translation unit.
    text = replace_once(text, "struct SharedStorageK2 {", "struct SharedStorageK2VShard {", "vshard shared-storage type")
    text = replace_once(text, "SharedStorageK2<Layouts, InputStages, OutputStages>", "SharedStorageK2VShard<Layouts, InputStages, OutputStages>", "vshard shared-storage use")
    text = replace_once(text, """    int CHUNK,
    int D,
    int InputStages,""", """    int CHUNK,
    int KDim,
    int VDim,
    int InputStages,""", "kernel template dimensions")
    text = replace_once(text, "_flash_kda_fwd_recurrence(", "_flash_kda_fwd_recurrence_vshard(", "kernel symbol")
    text = replace_once(text, "using Layouts = K2Layouts<D, CHUNK>;", """using Layouts = K2VShardLayouts<KDim, VDim, CHUNK>;
    constexpr int D = KDim;
    static_assert(KDim == 128 && VDim == 64, "vshard is deliberately specialized to K=128,Vshard=64");""", "kernel layouts")
    text = replace_once(text, "using TMAVOLayout = typename Layouts::TMAVOLayout;", """using TMAMMLayout = typename Layouts::TMAMMLayout;
    using TMAVOLayout = typename Layouts::TMAVOLayout;""", "kernel TMA layouts")
    text = replace_once(text, """    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;""", """    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y % H;
    int value_shard = blockIdx.y / H;""", "kernel grid mapping")

    # The K-side workspace is full width.  The v/out/state views are 64-column slices.
    text = text.replace("make_shape(N * H, D, D)", "make_shape(N * H, KDim, KDim)")
    text = text.replace("make_shape(Int<1>{}, Int<D>{}, Int<D>{})", "make_shape(Int<1>{}, Int<VDim>{}, Int<KDim>{})")
    text = text.replace("seq_idx * H + head_idx, 0, 0", "seq_idx * H + head_idx, value_shard * VDim, 0")
    text = replace_once(text, """            auto v_off = g_v.layout()(head_idx, int(bos) + t * CHUNK, 0);
            Tensor g_v_tile = make_tensor(g_v.data() + v_off,
                make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_v.layout())));""", """            auto v_off = g_v.layout()(head_idx, int(bos) + t * CHUNK, value_shard * VDim);
            Tensor g_v_tile = make_tensor(g_v.data() + v_off,
                make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<VDim>{}), stride(g_v.layout())));""", "v TMA slice")
    text = text.replace("TMAVOLayout{});\n                cute::copy(tma_load_ws_", "TMAMMLayout{});\n                cute::copy(tma_load_ws_")
    text = text.replace("smem_cvt_fp32_to_bf16<FP32StateSmemLayout, StateSmemLayout, D, NumThreads>", "vshard_cvt_fp32_to_bf16<FP32StateSmemLayout, StateSmemLayout, VDim, KDim, NumThreads>")
    text = text.replace("smem_cvt_bf16_to_fp32<StateSmemLayout, FP32StateSmemLayout, D, NumThreads>", "vshard_cvt_bf16_to_fp32<StateSmemLayout, FP32StateSmemLayout, VDim, KDim, NumThreads>")

    # One 16-column block per MMA warp: four warps cover the 64-column shard.
    text = text.replace("AccFragT u_acc[2], out_acc[2];", "AccFragT u_acc[1], out_acc[1];")
    text = text.replace("SFragT out_bf16[2];", "SFragT out_bf16[1];")
    text = text.replace("SFragT v_bf16[2];", "SFragT v_bf16[1];")
    text = text.replace("SFragT u_bf16[2];", "SFragT u_bf16[1];")
    text = text.replace("BFragT_u tCrB_u_arr[2];", "BFragT_u tCrB_u_arr[1];")
    text = text.replace("SFragT ring_S_acc[2][PREFETCH];", "SFragT ring_S_acc[1][PREFETCH];")
    text = text.replace("for (int i = 0; i < 2; ++i)", "for (int i = 0; i < 1; ++i)")
    text = text.replace("for (int bi = 0; bi < 2; ++bi)", "for (int bi = 0; bi < 1; ++bi)")
    text = text.replace("warp_id * 2 + i", "warp_id")
    text = text.replace("warp_id * 2 + bi", "warp_id")
    text = text.replace("warp_id * 2, 0", "warp_id, 0")
    text = text.replace("warp_id * 2, k + 1", "warp_id, k + 1")

    second_column_load = """                copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                    local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id * 2 + 1, k))), tCrBi_view);
"""
    text = text.replace(second_column_load, "                // vshard: no second V block in this warp\n")
    # The generic coordinate rewrite above is intentionally not applied to the removed line.
    text = text.replace("""                cute::transform(tCrBi, tCrB, cute::identity{});

                if (k + 1 < K_BLOCKS) {""", """                if (k + 1 < K_BLOCKS) {""")
    text = replace_once(
        text,
        "                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[1]);\n",
        "",
        "second U GEMM",
    )
    text = replace_once(
        text,
        "                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[1]);\n",
        "",
        "second output GEMM",
    )

    text = replace_once(text, """                    int64_t global_base = (bos + t * CHUNK + row) * H * D + head_idx * D;
                    for (int col = 0; col < D; ++col) {""", """                    int64_t global_base = (bos + t * CHUNK + row) * H * D + head_idx * D + value_shard * VDim;
                    for (int col = 0; col < VDim; ++col) {""", "tail output slice")
    text = replace_once(text, """                auto out_off = g_out.layout()(head_idx, int(bos) + t * CHUNK, 0);
                Tensor g_out_tile = make_tensor(g_out.data() + out_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_out.layout())));""", """                auto out_off = g_out.layout()(head_idx, int(bos) + t * CHUNK, value_shard * VDim);
                Tensor g_out_tile = make_tensor(g_out.data() + out_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<VDim>{}), stride(g_out.layout())));""", "output TMA slice")

    leftovers = [needle for needle in ("u_acc[1]);", "out_acc[1]);", "tCrB_u_arr[1](_,") if needle in text]
    if leftovers:
        die("kernel rewrite left a second-column accumulator reference: " + ", ".join(leftovers))
    return text


def rewrite_launch(upstream: str) -> str:
    marker = "// ==================== launch_fwd ===================="
    start = upstream.index(marker)
    variant = upstream[start:]
    variant = variant.replace("launch_fwd(", "launch_fwd_vshard(")
    variant = variant.replace("launch_fwd<", "launch_fwd_vshard<")
    variant = variant.replace("INSTANTIATE_LAUNCH_FWD", "INSTANTIATE_VSHARD_LAUNCH_FWD")
    variant = variant.replace("INSTANTIATE_STATE_VARIANTS", "INSTANTIATE_VSHARD_STATE_VARIANTS")
    variant = replace_once(variant, "using K2L = K2Layouts<D, CHUNK>;", "using K2L = K2VShardLayouts<D, D / 2, CHUNK>;", "launch K2 layout")
    variant = replace_once(variant, "using TMAVOLayout = typename K1L::TMAVOLayout;", "using TMAK1VOLayout = typename K1L::TMAVOLayout;", "launch K1 V TMA layout")
    variant = replace_once(variant, "using TMAStateSmemLayout = typename K2L::TMAStateSmemLayout;", """using TMAMMLayout = typename K2L::TMAMMLayout;
    using TMAVOLayout = typename K2L::TMAVOLayout;
    using TMAStateSmemLayout = typename K2L::TMAStateSmemLayout;""", "launch K2 TMA layouts")
    for name in ("kd", "qd", "kr"):
        variant = variant.replace(f"tma_store_ws_{name}  = make_tma_copy(SM90_TMA_STORE{{}}, m_ws_{name}, TMAVOLayout{{}})", f"tma_store_ws_{name}  = make_tma_copy(SM90_TMA_STORE{{}}, m_ws_{name}, TMAK1VOLayout{{}})")
        variant = variant.replace(f"tma_load_ws_{name}  = make_tma_copy(SM90_TMA_LOAD{{}}, m_ws_{name}, TMAVOLayout{{}})", f"tma_load_ws_{name}  = make_tma_copy(SM90_TMA_LOAD{{}}, m_ws_{name}, TMAMMLayout{{}})")
    variant = replace_once(variant, """        auto kernel2 = _flash_kda_fwd_recurrence<""", """        auto kernel2 = _flash_kda_fwd_recurrence_vshard<""", "launch K2 symbol")
    variant = replace_once(variant, """            CHUNK, D, kInputStages, kOutputStages, kK2Threads,""", """            CHUNK, D, D / 2, kInputStages, kOutputStages, kK2Threads,""", "launch K2 template args")
    variant = replace_once(variant, "dim3 grid_k2(N, H);", "dim3 grid_k2(N, H * 2);", "launch K2 grid")
    return upstream + "\n\n// ==================== launch_fwd_vshard (challenge only) ====================\n" + variant


def rewrite_header(upstream: str) -> str:
    declaration = upstream[upstream.index("template <int D"):]
    return upstream + "\n// K2 value dimension split into two independent CTA shards.\n" + declaration.replace("launch_fwd(", "launch_fwd_vshard(")


def rewrite_binding(upstream: str) -> str:
    function_start = upstream.index("void fwd(")
    pybind_start = upstream.index("\nPYBIND11_MODULE", function_start)
    fwd_vshard = upstream[function_start:pybind_start]
    fwd_vshard = fwd_vshard.replace("void fwd(", "void fwd_vshard(", 1)
    fwd_vshard = fwd_vshard.replace("launch_fwd<", "launch_fwd_vshard<")
    binding = """    m.def("fwd_vshard", &fwd_vshard, "FlashKDA Forward, 2-CTA/head value-shard challenge",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"),
        py::arg("scale"), py::arg("out"), py::arg("workspace"),
        py::arg("A_log"), py::arg("dt_bias"), py::arg("lower_bound"),
        py::arg("initial_state") = py::none(), py::arg("final_state") = py::none(),
        py::arg("cu_seqlens") = py::none());
"""
    result = upstream[:pybind_start] + "\n" + fwd_vshard + upstream[pybind_start:]
    needle = '    m.def("get_workspace_size",'
    return replace_once(result, needle, binding + needle, "pybind vshard entry")


def verify_source(source: Path) -> None:
    if git(source, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        die(f"expected FlashKDA commit {UPSTREAM_COMMIT}, got {git(source, 'rev-parse', 'HEAD')}")
    status = git(source, "status", "--porcelain")
    if status:
        die("source tree is dirty; use a fresh dedicated worktree so the patch is reversible")
    submodule = git(source, "submodule", "status", "cutlass")
    if CUTLASS_COMMIT not in submodule:
        die(f"expected cutlass {CUTLASS_COMMIT}, got: {submodule}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="fresh FlashKDA 1ce47ea clone/worktree")
    args = parser.parse_args()
    source = args.source.resolve()
    verify_source(source)

    kernel = source / "csrc" / "smxx" / "fwd_kernel2.cuh"
    vshard_kernel = source / "csrc" / "smxx" / "fwd_kernel2_vshard.cuh"
    launch = source / "csrc" / "smxx" / "fwd_launch.cu"
    header = source / "csrc" / "fwd.h"
    binding = source / "csrc" / "flash_kda.cpp"
    if vshard_kernel.exists():
        die(f"{vshard_kernel} already exists; refusing to patch twice")

    vshard_kernel.write_text(rewrite_kernel2(kernel.read_text(encoding="utf-8")), encoding="utf-8")
    launch_text = launch.read_text(encoding="utf-8")
    launch_text = replace_once(launch_text, '#include "fwd_kernel2.cuh"', '#include "fwd_kernel2.cuh"\n#include "fwd_kernel2_vshard.cuh"', "vshard include")
    launch.write_text(rewrite_launch(launch_text), encoding="utf-8")
    header.write_text(rewrite_header(header.read_text(encoding="utf-8")), encoding="utf-8")
    binding.write_text(rewrite_binding(binding.read_text(encoding="utf-8")), encoding="utf-8")
    print("applied challenge-only vshard patch")
    print("next: FLASH_KDA_CUDA_ARCHS=103a NVCC_THREADS=8 python setup.py build_ext --inplace")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
