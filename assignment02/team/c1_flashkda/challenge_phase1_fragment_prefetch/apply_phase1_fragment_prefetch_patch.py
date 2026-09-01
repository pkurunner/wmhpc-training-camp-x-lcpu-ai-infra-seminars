#!/usr/bin/env python3
"""Build a one-shot, isolated Phase-1 two-slot fragment-prefetch candidate.

All added paths are regenerated in memory from one clean FlashKDA ``1ce47ea``
tree.  The candidate deliberately has a distinct header/type/pybind namespace
and may never be installed in the production dispatcher.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


PINNED_GENERATORS = {
    "vshard2": (
        "challenge_vshard/apply_vshard_patch.py",
        "bca3248e1bf480ea51eb3bb3da0e79d8f477fb914ea17d320c0bf90679aaaf7c",
    ),
    "vshard4": (
        "challenge_vshard4/apply_vshard4_patch.py",
        "46222004dfa3e8b00d6ff14a9f64305c8743e2598cd151410ec406d289444580",
    ),
    "prefetch2": (
        "challenge_prefetch2/apply_prefetch2_patch.py",
        "f83e3551907ec8f1a5c1f5c3421e94dc1e3d1941e9f35c845d1d982eef38ccb0",
    ),
}


def die(message: str) -> None:
    raise RuntimeError(message)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        die(f"{label}: expected one matching block, found {count}")
    return text.replace(old, new, 1)


def write_text(path: Path, text: str, newline: str) -> None:
    with path.open("w", encoding="utf-8", newline=newline) as handle:
        handle.write(text)


def load_pinned_modules() -> dict[str, ModuleType]:
    root = Path(__file__).resolve().parents[1]
    result: dict[str, ModuleType] = {}
    for label, (relative, expected_sha256) in PINNED_GENERATORS.items():
        path = root / relative
        if not path.is_file():
            die(f"missing pinned {label} generator: {path}")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            die(f"{label} generator digest changed: expected {expected_sha256}, got {actual_sha256}")
        spec = importlib.util.spec_from_file_location(f"c1_{label}_phase1pf", path)
        if spec is None or spec.loader is None:
            die(f"cannot load pinned {label} generator")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result[label] = module
    return result


def rename_vshard2_p2(text: str) -> str:
    return text.replace("VSHARD", "VSHARD_P2").replace("VShard", "VShardP2").replace("vshard", "vshard_p2")


def rename_vshard4_p2(text: str) -> str:
    return text.replace("VSHARD4", "VSHARD4_P2").replace("VShard4", "VShard4P2").replace("vshard4", "vshard4_p2")


def rename_phase1pf(text: str) -> str:
    """Give the candidate a namespace disjoint from measured P2S3."""
    return (
        text.replace("VSHARD4_P2", "VSHARD4_P2_PHASE1PF")
        .replace("VShard4P2", "VShard4P2Phase1PF")
        .replace("vshard4_p2", "vshard4_p2_phase1pf")
    )


def isolate_vshard4_launch_macros(text: str) -> str:
    return text.replace("INSTANTIATE_VSHARD_", "INSTANTIATE_VSHARD4_")


def with_prefetch2(p1_kernel: str, rename) -> str:
    text = rename(p1_kernel)
    text = replace_once(text, "            constexpr int PREFETCH = 1;", "            constexpr int PREFETCH = 2;", "Phase-6 prefetch depth")
    return replace_once(
        text,
        "            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;",
        "            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;\n"
        "            static_assert(PREFETCH > 0 && PREFETCH <= S_M_BLOCKS, \"invalid Phase-6 prefetch depth\");",
        "Phase-6 prefetch bound",
    )


def additions_from_upstream(upstream: str, rewritten: str, label: str) -> str:
    if not rewritten.startswith(upstream):
        die(f"{label} rewrite modified upstream prefix")
    return rewritten[len(upstream) :]


PHASE1_DECLARATIONS = """            Tensor tCrAi_k = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_k_view = smem_thr_copy_A.retile_D(tCrAi_k);
            auto tCrA_k = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrAi_q = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_q_view = smem_thr_copy_A.retile_D(tCrAi_q);
            auto tCrA_q = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrBi = make_fragment_like<BF16>(thr_mma.partition_fragment_B(B_ref));
            auto tCrBi_view = smem_thr_copy_B.retile_D(tCrBi);
            auto tCrB = thr_mma.partition_fragment_B(B_ref);"""

PHASE1_TWO_SLOT_DECLARATIONS = """            // Phase-1 has two independent register-fragment slots.  Slot k&1
            // is the active MMA operand; slot (k+1)&1 is only filled while
            // Q@s consumes the active operand.  No shared-memory location is
            // written here.
            Tensor tCrAi_k = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_k_view = smem_thr_copy_A.retile_D(tCrAi_k);
            Tensor tCrAi_k_alt = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_k_alt_view = smem_thr_copy_A.retile_D(tCrAi_k_alt);
            auto tCrA_k = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrAi_q = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_q_view = smem_thr_copy_A.retile_D(tCrAi_q);
            Tensor tCrAi_q_alt = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_q_alt_view = smem_thr_copy_A.retile_D(tCrAi_q_alt);
            auto tCrA_q = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrBi = make_fragment_like<BF16>(thr_mma.partition_fragment_B(B_ref));
            auto tCrBi_view = smem_thr_copy_B.retile_D(tCrBi);
            Tensor tCrBi_alt = make_fragment_like<BF16>(thr_mma.partition_fragment_B(B_ref));
            auto tCrBi_alt_view = smem_thr_copy_B.retile_D(tCrBi_alt);
            auto tCrB = thr_mma.partition_fragment_B(B_ref);"""

PHASE1_TWO_SLOT = """            // ======== Phase 1: K@s/Q@s, two-slot fragment prefetch ========
            // K and Q each retain their original increasing-k MMA accumulation
            // order.  The only schedule change is that k+1 loads target the
            // inactive fragment slot between the two independent GEMMs.
            constexpr int K_BLOCKS = decltype(cute::size<1>(k_decayed))::value / 16;
            constexpr int PHASE1_SLOTS = 2;
            static_assert(K_BLOCKS >= PHASE1_SLOTS, "Phase-1 two-slot candidate needs K=128");

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_k_view);
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_q_view);
            copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id, 0))), tCrBi_view);

            #pragma unroll
            for (int k = 0; k < K_BLOCKS; ++k) {
                if ((k & 1) == 0) {
                    cute::transform(tCrAi_k, tCrA_k, cute::identity{});
                    cute::transform(tCrAi_q, tCrA_q, cute::identity{});
                    cute::transform(tCrBi, tCrB, cute::identity{});
                } else {
                    cute::transform(tCrAi_k_alt, tCrA_k, cute::identity{});
                    cute::transform(tCrAi_q_alt, tCrA_q, cute::identity{});
                    cute::transform(tCrBi_alt, tCrB, cute::identity{});
                }

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[0]);

                if (k + 1 < K_BLOCKS) {
                    if (((k + 1) & 1) == 0) {
                        copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                            local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_k_view);
                        copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                            local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_q_view);
                        copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                            local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id, k + 1))), tCrBi_view);
                    } else {
                        copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                            local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_k_alt_view);
                        copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                            local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_q_alt_view);
                        copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                            local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id, k + 1))), tCrBi_alt_view);
                    }
                }

                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[0]);
            }

"""


def phase1pf_kernel_from_p2(p2: str) -> str:
    canonical = rename_phase1pf(p2)
    text = replace_once(canonical, PHASE1_DECLARATIONS, PHASE1_TWO_SLOT_DECLARATIONS, "Phase-1 fragment declarations")
    begin = text.index("            // ======== Phase 1: Dual GEMM k@s and q@s")
    end = text.index("            // ======== Phase 2:", begin)
    candidate = text[:begin] + PHASE1_TWO_SLOT + text[end:]
    # Make the isolation claim mechanically checkable: after undoing exactly
    # the declaration and Phase-1 schedule substitutions, the candidate is
    # byte-identical to a namespace-renamed current P2S3 kernel.
    candidate_begin = candidate.index("            // ======== Phase 1: K@s/Q@s, two-slot fragment prefetch")
    candidate_end = candidate.index("            // ======== Phase 2:", candidate_begin)
    canonical_begin = canonical.index("            // ======== Phase 1: Dual GEMM k@s and q@s")
    canonical_end = canonical.index("            // ======== Phase 2:", canonical_begin)
    restored = candidate[:candidate_begin] + canonical[canonical_begin:canonical_end] + candidate[candidate_end:]
    restored = replace_once(restored, PHASE1_TWO_SLOT_DECLARATIONS, PHASE1_DECLARATIONS, "Phase-1 restoration")
    if restored != canonical:
        die("candidate differs from namespace-renamed current P2S3 outside the two declared Phase-1 substitutions")
    return candidate


def rewrite_launch_all(upstream: str, modules: dict[str, ModuleType]) -> str:
    prefetch, vshard4 = modules["prefetch2"], modules["vshard4"]
    vshard2_p1, vshard2_p2 = prefetch.launch_additions(upstream, 3)
    vshard4_p1 = isolate_vshard4_launch_macros(additions_from_upstream(upstream, vshard4.rewrite_launch(upstream), "vshard4 launch"))
    vshard4_p2 = rename_vshard4_p2(vshard4_p1)
    return upstream + vshard2_p1 + vshard2_p2 + vshard4_p1 + vshard4_p2 + rename_phase1pf(vshard4_p2)


def rewrite_header_all(upstream: str, modules: dict[str, ModuleType]) -> str:
    vshard2 = modules["prefetch2"].rewrite_header_both(upstream)
    vshard4_p1 = additions_from_upstream(upstream, modules["vshard4"].rewrite_header(upstream), "vshard4 header")
    vshard4_p2 = rename_vshard4_p2(vshard4_p1)
    return vshard2 + vshard4_p1 + vshard4_p2 + rename_phase1pf(vshard4_p2)


def vshard4_binding_parts(upstream: str, module: ModuleType) -> tuple[str, str]:
    rewritten = module.rewrite_binding(upstream)
    function_start = rewritten.index("void fwd_vshard4(")
    pybind_start = rewritten.index("\nPYBIND11_MODULE", function_start)
    binding_start = rewritten.index('    m.def("fwd_vshard4",')
    binding_end = rewritten.index('    m.def("get_workspace_size",', binding_start)
    return rewritten[function_start:pybind_start], rewritten[binding_start:binding_end]


def rewrite_binding_all(upstream: str, modules: dict[str, ModuleType]) -> str:
    binding = modules["prefetch2"].rewrite_binding_both(upstream)
    function, definition = vshard4_binding_parts(upstream, modules["vshard4"])
    p2_function, p2_definition = rename_vshard4_p2(function), rename_vshard4_p2(definition)
    candidate_function = rename_phase1pf(p2_function)
    candidate_definition = rename_phase1pf(p2_definition).replace(
        "FlashKDA Forward, 4-CTA/head value-shard challenge",
        "FlashKDA Forward, 4-CTA/head value-shard P2S3 Phase-1 two-slot fragment-prefetch candidate",
    )
    pybind_start = binding.index("\nPYBIND11_MODULE")
    binding = binding[:pybind_start] + "\n" + function + "\n" + p2_function + "\n" + candidate_function + binding[pybind_start:]
    insertion = binding.index('    m.def("get_workspace_size",')
    return binding[:insertion] + definition + p2_definition + candidate_definition + binding[insertion:]


def generated(kernel: str, launch: str, header: str, binding: str) -> dict[str, str]:
    modules = load_pinned_modules()
    vshard2_p1 = modules["vshard2"].rewrite_kernel2(kernel)
    vshard2_p2 = with_prefetch2(vshard2_p1, rename_vshard2_p2)
    vshard4_p1 = modules["vshard4"].rewrite_kernel2(kernel)
    vshard4_p2 = with_prefetch2(vshard4_p1, rename_vshard4_p2)
    candidate = phase1pf_kernel_from_p2(vshard4_p2)
    return {
        "fwd_kernel2_vshard.cuh": vshard2_p1,
        "fwd_kernel2_vshard_p2.cuh": vshard2_p2,
        "fwd_kernel2_vshard4.cuh": vshard4_p1,
        "fwd_kernel2_vshard4_p2.cuh": vshard4_p2,
        "fwd_kernel2_vshard4_p2_phase1pf.cuh": candidate,
        "fwd_launch.cu": rewrite_launch_all(launch, modules),
        "fwd.h": rewrite_header_all(header, modules),
        "flash_kda.cpp": rewrite_binding_all(binding, modules),
    }


def with_includes(launch: str) -> str:
    return replace_once(
        launch,
        '#include "fwd_kernel2.cuh"',
        "\n".join((
            '#include "fwd_kernel2.cuh"', '#include "fwd_kernel2_vshard.cuh"',
            '#include "fwd_kernel2_vshard_p2.cuh"', '#include "fwd_kernel2_vshard4.cuh"',
            '#include "fwd_kernel2_vshard4_p2.cuh"', '#include "fwd_kernel2_vshard4_p2_phase1pf.cuh"',
        )),
        "all kernel includes",
    )


def static_check(source: Path) -> None:
    root = source / "csrc" / "smxx"
    outputs = generated(
        (root / "fwd_kernel2.cuh").read_text(encoding="utf-8"),
        (root / "fwd_launch.cu").read_text(encoding="utf-8"),
        (source / "csrc" / "fwd.h").read_text(encoding="utf-8"),
        (source / "csrc" / "flash_kda.cpp").read_text(encoding="utf-8"),
    )
    candidate = outputs["fwd_kernel2_vshard4_p2_phase1pf.cuh"]
    required = (
        "K2VShard4P2Phase1PFLayouts", "SharedStorageK2VShard4P2Phase1PF",
        "_flash_kda_fwd_recurrence_vshard4_p2_phase1pf", "constexpr int PREFETCH = 2;",
        "Phase 1: K@s/Q@s, two-slot fragment prefetch", "Tensor tCrAi_k_alt",
        "Tensor tCrAi_q_alt", "Tensor tCrBi_alt", "gemm(thr_mma, tCrA_k(_",
        "gemm(thr_mma, tCrA_q(_", "tCrAi_k_alt_view",
    )
    if any(needle not in candidate for needle in required):
        die("Phase-1 two-slot candidate static check failed")
    if "constexpr int PREFETCH = 1;" in candidate or "u_acc[1]);" in candidate or "out_acc[1]);" in candidate:
        die("candidate retained stale Phase-6 or second-V-block text")
    launch = with_includes(outputs["fwd_launch.cu"])
    if launch.count("constexpr int kInputStages = 3;") != 6 or "constexpr int kInputStages = 2;" in launch:
        die("expected six S=3 paths including the candidate")
    for macro in (
        "INSTANTIATE_VSHARD_LAUNCH_FWD", "INSTANTIATE_VSHARD_P2_LAUNCH_FWD",
        "INSTANTIATE_VSHARD4_LAUNCH_FWD", "INSTANTIATE_VSHARD4_P2_LAUNCH_FWD",
        "INSTANTIATE_VSHARD4_P2_PHASE1PF_LAUNCH_FWD",
    ):
        if launch.count(f"#define {macro}") != 1:
            die(f"launch macro isolation failed for {macro}")
    header, binding = outputs["fwd.h"], outputs["flash_kda.cpp"]
    for symbol in ("fwd_vshard", "fwd_vshard_p2", "fwd_vshard4", "fwd_vshard4_p2", "fwd_vshard4_p2_phase1pf"):
        if header.count(f"launch_{symbol}(") != 1 or binding.count(f"void {symbol}(") != 1 or binding.count(f'm.def("{symbol}"') != 1:
            die(f"ABI isolation failed for {symbol}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    modules = load_pinned_modules()
    modules["vshard2"].verify_source(source)
    root = source / "csrc" / "smxx"
    targets = tuple(root / name for name in (
        "fwd_kernel2_vshard.cuh", "fwd_kernel2_vshard_p2.cuh", "fwd_kernel2_vshard4.cuh",
        "fwd_kernel2_vshard4_p2.cuh", "fwd_kernel2_vshard4_p2_phase1pf.cuh",
    ))
    if any(path.exists() for path in targets):
        die("generated header already exists; refusing sequential patching")
    static_check(source)
    if args.check_only:
        print("static check passed: P2S3 preserved + isolated Phase-1 two-slot candidate")
        return
    newline = "\r\n" if b"\r\n" in (root / "fwd_kernel2.cuh").read_bytes() else "\n"
    outputs = generated(
        (root / "fwd_kernel2.cuh").read_text(encoding="utf-8"),
        (root / "fwd_launch.cu").read_text(encoding="utf-8"),
        (source / "csrc" / "fwd.h").read_text(encoding="utf-8"),
        (source / "csrc" / "flash_kda.cpp").read_text(encoding="utf-8"),
    )
    outputs["fwd_launch.cu"] = with_includes(outputs["fwd_launch.cu"])
    for path in targets:
        write_text(path, outputs[path.name], newline)
    write_text(root / "fwd_launch.cu", outputs["fwd_launch.cu"], newline)
    write_text(source / "csrc" / "fwd.h", outputs["fwd.h"], newline)
    write_text(source / "csrc" / "flash_kda.cpp", outputs["flash_kda.cpp"], newline)
    print("applied: baseline/vshard2-P2S3/vshard4-P2S3 + Phase-1 two-slot candidate")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
