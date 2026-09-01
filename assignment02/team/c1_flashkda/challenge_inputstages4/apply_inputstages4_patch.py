#!/usr/bin/env python3
"""Generate an isolated H12-only P2S4 comparison extension from clean FlashKDA.

This is deliberately a candidate-only generator.  It creates one extension
containing the untouched upstream baseline, the measured vshard2-P2S3 path,
the measured vshard4-P2S3 path, and a separately named vshard4-P2S4 path.
No generated text is read back as an input: every added entry is derived in
memory from one clean FlashKDA ``1ce47ea`` source snapshot.
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
        die(f"{label}: expected exactly one matching block, found {count}")
    return text.replace(old, new, 1)


def write_with_source_newlines(path: Path, text: str, newline: str) -> None:
    """Preserve the source newline convention in a Windows-hosted checkout."""
    with path.open("w", encoding="utf-8", newline=newline) as handle:
        handle.write(text)


def load_pinned_modules() -> dict[str, ModuleType]:
    root = Path(__file__).resolve().parents[1]
    loaded: dict[str, ModuleType] = {}
    for label, (relative, expected_sha256) in PINNED_GENERATORS.items():
        path = root / relative
        if not path.is_file():
            die(f"missing pinned {label} generator: {path}")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            die(
                f"{label} generator digest changed: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        spec = importlib.util.spec_from_file_location(
            f"c1_{label}_pinned_for_inputstages4", path
        )
        if spec is None or spec.loader is None:
            die(f"cannot load pinned {label} generator: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[label] = module
    return loaded


def rename_vshard2_p2(text: str) -> str:
    return (
        text.replace("VSHARD", "VSHARD_P2")
        .replace("VShard", "VShardP2")
        .replace("vshard", "vshard_p2")
    )


def rename_vshard4_p2s3(text: str) -> str:
    return (
        text.replace("VSHARD4", "VSHARD4_P2")
        .replace("VShard4", "VShard4P2")
        .replace("vshard4", "vshard4_p2")
    )


def rename_vshard4_p2s4(text: str) -> str:
    """Use a disjoint type/symbol namespace for the S=4 instantiations."""
    return (
        text.replace("VSHARD4_P2", "VSHARD4_P2S4")
        .replace("VShard4P2", "VShard4P2S4")
        .replace("vshard4_p2", "vshard4_p2s4")
    )


def isolate_vshard4_launch_macros(text: str) -> str:
    return text.replace("INSTANTIATE_VSHARD_", "INSTANTIATE_VSHARD4_")


def with_prefetch2(p1_kernel: str, rename) -> str:
    text = rename(p1_kernel)
    text = replace_once(
        text,
        "            constexpr int PREFETCH = 1;",
        "            constexpr int PREFETCH = 2;",
        "Phase-6 software prefetch depth",
    )
    return replace_once(
        text,
        "            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;",
        """            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;
            static_assert(PREFETCH > 0 && PREFETCH <= S_M_BLOCKS, "invalid Phase-6 prefetch depth");""",
        "Phase-6 prefetch bound",
    )


def additions_from_upstream(upstream: str, rewritten: str, label: str) -> str:
    if not rewritten.startswith(upstream):
        die(f"{label} rewrite unexpectedly modified the upstream prefix")
    return rewritten[len(upstream) :]


def p2s4_launch_from_p2s3(p2s3_launch: str) -> str:
    return replace_once(
        rename_vshard4_p2s4(p2s3_launch),
        "    constexpr int kInputStages = 3;",
        "    constexpr int kInputStages = 4;",
        "P2S4 K2 input-pipeline depth",
    )


def rewrite_launch_all(upstream: str, modules: dict[str, ModuleType]) -> str:
    prefetch = modules["prefetch2"]
    vshard4 = modules["vshard4"]
    vshard2_p1, vshard2_p2s3 = prefetch.launch_additions(upstream, 3)
    vshard4_p1 = isolate_vshard4_launch_macros(
        additions_from_upstream(upstream, vshard4.rewrite_launch(upstream), "vshard4 launch")
    )
    vshard4_p2s3 = rename_vshard4_p2s3(vshard4_p1)
    vshard4_p2s4 = p2s4_launch_from_p2s3(vshard4_p2s3)
    return (
        upstream
        + vshard2_p1
        + vshard2_p2s3
        + vshard4_p1
        + vshard4_p2s3
        + vshard4_p2s4
    )


def rewrite_header_all(upstream: str, modules: dict[str, ModuleType]) -> str:
    vshard2 = modules["prefetch2"].rewrite_header_both(upstream)
    vshard4_p1 = additions_from_upstream(
        upstream, modules["vshard4"].rewrite_header(upstream), "vshard4 header"
    )
    vshard4_p2s3 = rename_vshard4_p2s3(vshard4_p1)
    return vshard2 + vshard4_p1 + vshard4_p2s3 + rename_vshard4_p2s4(vshard4_p2s3)


def _vshard4_binding_parts(upstream: str, vshard4: ModuleType) -> tuple[str, str]:
    rewritten = vshard4.rewrite_binding(upstream)
    function_start = rewritten.index("void fwd_vshard4(")
    pybind_start = rewritten.index("\nPYBIND11_MODULE", function_start)
    binding_start = rewritten.index('    m.def("fwd_vshard4",')
    binding_end = rewritten.index('    m.def("get_workspace_size",', binding_start)
    return rewritten[function_start:pybind_start], rewritten[binding_start:binding_end]


def rewrite_binding_all(upstream: str, modules: dict[str, ModuleType]) -> str:
    binding = modules["prefetch2"].rewrite_binding_both(upstream)
    vshard4_function, vshard4_binding = _vshard4_binding_parts(upstream, modules["vshard4"])
    p2s3_function = rename_vshard4_p2s3(vshard4_function)
    p2s4_function = rename_vshard4_p2s4(p2s3_function)
    p2s3_binding = rename_vshard4_p2s3(vshard4_binding).replace(
        "FlashKDA Forward, 4-CTA/head value-shard challenge",
        "FlashKDA Forward, 4-CTA/head value-shard PREFETCH=2,S=3 comparison path",
    )
    p2s4_binding = rename_vshard4_p2s4(p2s3_binding).replace(
        "PREFETCH=2,S=3 comparison path",
        "PREFETCH=2,S=4 candidate-only path",
    )
    pybind_start = binding.index("\nPYBIND11_MODULE")
    binding = (
        binding[:pybind_start]
        + "\n"
        + vshard4_function
        + "\n"
        + p2s3_function
        + "\n"
        + p2s4_function
        + binding[pybind_start:]
    )
    insertion = binding.index('    m.def("get_workspace_size",')
    return (
        binding[:insertion]
        + vshard4_binding
        + p2s3_binding
        + p2s4_binding
        + binding[insertion:]
    )


def generated(
    upstream_kernel: str, upstream_launch: str, upstream_header: str, upstream_binding: str
) -> dict[str, str]:
    modules = load_pinned_modules()
    vshard2_p1 = modules["vshard2"].rewrite_kernel2(upstream_kernel)
    vshard2_p2s3 = with_prefetch2(vshard2_p1, rename_vshard2_p2)
    vshard4_p1 = modules["vshard4"].rewrite_kernel2(upstream_kernel)
    vshard4_p2s3 = with_prefetch2(vshard4_p1, rename_vshard4_p2s3)
    # The kernel text is P=2-identical; only its distinct S=4 launch
    # instantiation changes.  Its namespace remains separate so ptxas and ABI
    # evidence cannot be accidentally attributed to P2S3.
    vshard4_p2s4 = rename_vshard4_p2s4(vshard4_p2s3)
    return {
        "fwd_kernel2_vshard.cuh": vshard2_p1,
        "fwd_kernel2_vshard_p2.cuh": vshard2_p2s3,
        "fwd_kernel2_vshard4.cuh": vshard4_p1,
        "fwd_kernel2_vshard4_p2.cuh": vshard4_p2s3,
        "fwd_kernel2_vshard4_p2s4.cuh": vshard4_p2s4,
        "fwd_launch.cu": rewrite_launch_all(upstream_launch, modules),
        "fwd.h": rewrite_header_all(upstream_header, modules),
        "flash_kda.cpp": rewrite_binding_all(upstream_binding, modules),
    }


def _with_includes(launch: str) -> str:
    return replace_once(
        launch,
        '#include "fwd_kernel2.cuh"',
        "\n".join(
            (
                '#include "fwd_kernel2.cuh"',
                '#include "fwd_kernel2_vshard.cuh"',
                '#include "fwd_kernel2_vshard_p2.cuh"',
                '#include "fwd_kernel2_vshard4.cuh"',
                '#include "fwd_kernel2_vshard4_p2.cuh"',
                '#include "fwd_kernel2_vshard4_p2s4.cuh"',
            )
        ),
        "all generated kernel includes",
    )


def static_check(source: Path) -> None:
    kernel_root = source / "csrc" / "smxx"
    outputs = generated(
        (kernel_root / "fwd_kernel2.cuh").read_text(encoding="utf-8"),
        (kernel_root / "fwd_launch.cu").read_text(encoding="utf-8"),
        (source / "csrc" / "fwd.h").read_text(encoding="utf-8"),
        (source / "csrc" / "flash_kda.cpp").read_text(encoding="utf-8"),
    )
    outputs["fwd_launch.cu"] = _with_includes(outputs["fwd_launch.cu"])
    p2s4 = outputs["fwd_kernel2_vshard4_p2s4.cuh"]
    required_kernel = (
        "K2VShard4P2S4Layouts",
        "SharedStorageK2VShard4P2S4",
        "_flash_kda_fwd_recurrence_vshard4_p2s4",
        "VDim == 32",
        "constexpr int kComputeThreads = 64;",
        "constexpr int PREFETCH = 2;",
        "ring_S_acc[1][PREFETCH]",
        "static_assert(PREFETCH > 0 && PREFETCH <= S_M_BLOCKS",
    )
    if any(needle not in p2s4 for needle in required_kernel) or "constexpr int PREFETCH = 1;" in p2s4:
        die("vshard4 P2S4 kernel static check failed")
    if "u_acc[1]);" in p2s4 or "out_acc[1]);" in p2s4 or "tCrB_u_arr[1](" in p2s4:
        die("vshard4 P2S4 retained a second V-block access")
    launch = outputs["fwd_launch.cu"]
    required_launch = (
        "launch_fwd_vshard_p2(",
        "launch_fwd_vshard4_p2(",
        "launch_fwd_vshard4_p2s4(",
        "K2VShard4P2S4Layouts<D, D / 4, CHUNK>",
        "SharedStorageK2VShard4P2S4<K2L, kInputStages, kOutputStages>",
        "CHUNK, D, D / 4, kInputStages, kOutputStages, kK2Threads,",
        "constexpr int kK2Threads = 32 * 2 + 64;",
        "dim3 grid_k2(N, H * 4);",
    )
    if any(needle not in launch for needle in required_launch):
        die("vshard4 P2S4 launch static check failed")
    if launch.count("constexpr int kInputStages = 3;") != 5 or launch.count(
        "constexpr int kInputStages = 4;"
    ) != 1 or "constexpr int kInputStages = 2;" in launch:
        die("expected five preserved S=3 entries and exactly one candidate S=4 entry")
    for macro in (
        "INSTANTIATE_VSHARD_LAUNCH_FWD",
        "INSTANTIATE_VSHARD_P2_LAUNCH_FWD",
        "INSTANTIATE_VSHARD4_LAUNCH_FWD",
        "INSTANTIATE_VSHARD4_P2_LAUNCH_FWD",
        "INSTANTIATE_VSHARD4_P2S4_LAUNCH_FWD",
    ):
        if launch.count(f"#define {macro}") != 1:
            die(f"launch macro isolation failed for {macro}")
    header = outputs["fwd.h"]
    binding = outputs["flash_kda.cpp"]
    for symbol in (
        "fwd_vshard",
        "fwd_vshard_p2",
        "fwd_vshard4",
        "fwd_vshard4_p2",
        "fwd_vshard4_p2s4",
    ):
        if (
            header.count(f"launch_{symbol}(") != 1
            or binding.count(f"void {symbol}(") != 1
            or binding.count(f'm.def("{symbol}"') != 1
        ):
            die(f"header/binding symbol isolation failed for {symbol}")
    if "PREFETCH=2,S=4 candidate-only path" not in binding:
        die("P2S4 pybind description is stale")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="clean FlashKDA 1ce47ea worktree")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    modules = load_pinned_modules()
    modules["vshard2"].verify_source(source)
    kernel_root = source / "csrc" / "smxx"
    targets = (
        "fwd_kernel2_vshard.cuh",
        "fwd_kernel2_vshard_p2.cuh",
        "fwd_kernel2_vshard4.cuh",
        "fwd_kernel2_vshard4_p2.cuh",
        "fwd_kernel2_vshard4_p2s4.cuh",
    )
    if any((kernel_root / name).exists() for name in targets):
        die("generated header already exists; refusing sequential patch application")
    static_check(source)
    if args.check_only:
        print("static check passed: baseline/vshard2-P2S3/vshard4-P2S3 preserved + P2S4 candidate")
        return
    upstream_kernel = (kernel_root / "fwd_kernel2.cuh").read_text(encoding="utf-8")
    newline = "\r\n" if b"\r\n" in (kernel_root / "fwd_kernel2.cuh").read_bytes() else "\n"
    outputs = generated(
        upstream_kernel,
        (kernel_root / "fwd_launch.cu").read_text(encoding="utf-8"),
        (source / "csrc" / "fwd.h").read_text(encoding="utf-8"),
        (source / "csrc" / "flash_kda.cpp").read_text(encoding="utf-8"),
    )
    outputs["fwd_launch.cu"] = _with_includes(outputs["fwd_launch.cu"])
    for name in targets:
        write_with_source_newlines(kernel_root / name, outputs[name], newline)
    write_with_source_newlines(kernel_root / "fwd_launch.cu", outputs["fwd_launch.cu"], newline)
    write_with_source_newlines(source / "csrc" / "fwd.h", outputs["fwd.h"], newline)
    write_with_source_newlines(source / "csrc" / "flash_kda.cpp", outputs["flash_kda.cpp"], newline)
    print("applied one-shot candidate set: baseline/vshard2-P2S3/vshard4-P2S3 + vshard4-P2S4")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
