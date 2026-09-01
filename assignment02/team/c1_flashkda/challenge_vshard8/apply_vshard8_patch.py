#!/usr/bin/env python3
"""Add an isolated V=16 vshard8-P1 entry to the audited comparison SO."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


P2_RELATIVE = "challenge_vshard8_prefetch2/apply_vshard8_prefetch2_patch.py"
P2_SHA256 = "c0587eb3a220fdbd8acc276a2ee1c7cbbfa97822622ca565e181b5d9b2857a5d"


def die(message: str) -> None:
    raise RuntimeError(message)


def load_p2() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / P2_RELATIVE
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
    if actual != P2_SHA256:
        die(f"pinned vshard8-P2 generator changed: expected {P2_SHA256}, got {actual}")
    spec = importlib.util.spec_from_file_location("c1_vshard8_p2_pinned_for_p1", path)
    if spec is None or spec.loader is None:
        die(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rename_vshard8(text: str) -> str:
    return (
        text.replace("VSHARD4", "VSHARD8")
        .replace("VShard4", "VShard8")
        .replace("vshard4", "vshard8")
    )


def rewrite_kernel(upstream: str, p2: ModuleType, parent: ModuleType, modules: dict[str, ModuleType]) -> str:
    text = rename_vshard8(modules["vshard4"].rewrite_kernel2(upstream))
    text = parent.replace_once(
        text,
        'static_assert(KDim == 128 && VDim == 32, "vshard8 is deliberately specialized to K=128,Vshard=32");',
        'static_assert(KDim == 128 && VDim == 16, "vshard8 is deliberately specialized to K=128,Vshard=16");',
        "vshard8-P1 dimensions",
    )
    text = parent.replace_once(
        text,
        "constexpr int kComputeThreads = 64;",
        "constexpr int kComputeThreads = 32;",
        "vshard8-P1 compute threads",
    )
    required = (
        "K2VShard8Layouts",
        "SharedStorageK2VShard8",
        "_flash_kda_fwd_recurrence_vshard8",
        "VDim == 16",
        "constexpr int kComputeThreads = 32;",
        "constexpr int PREFETCH = 1;",
        "ring_S_acc[1][PREFETCH]",
    )
    if any(needle not in text for needle in required):
        die("vshard8-P1 kernel static check failed")
    if any(needle in text for needle in ("u_acc[1]);", "out_acc[1]);", "tCrB_u_arr[1](_")):
        die("vshard8-P1 retained a second V-block access")
    return text


def rewrite_launch(upstream: str, parent: ModuleType, modules: dict[str, ModuleType]) -> str:
    addition = parent.isolate_vshard4_launch_macros(
        parent.additions_from_upstream(
            upstream, modules["vshard4"].rewrite_launch(upstream), "vshard4 launch"
        )
    )
    text = rename_vshard8(addition)
    text = parent.replace_once(
        text,
        "using K2L = K2VShard8Layouts<D, D / 4, CHUNK>;",
        "using K2L = K2VShard8Layouts<D, D / 8, CHUNK>;",
        "vshard8-P1 V layout",
    )
    text = parent.replace_once(
        text,
        "CHUNK, D, D / 4, kInputStages, kOutputStages, kK2Threads,",
        "CHUNK, D, D / 8, kInputStages, kOutputStages, kK2Threads,",
        "vshard8-P1 template V",
    )
    text = parent.replace_once(
        text,
        "constexpr int kK2Threads = 32 * 2 + 64;",
        "constexpr int kK2Threads = 32 * 2 + 32;",
        "vshard8-P1 launch threads",
    )
    return parent.replace_once(
        text, "dim3 grid_k2(N, H * 4);", "dim3 grid_k2(N, H * 8);", "vshard8-P1 grid"
    )


def generated(source: Path) -> dict[str, str]:
    p2 = load_p2()
    parent = p2.load_parent()
    modules = parent.load_pinned_modules()
    kernel_root = source / "csrc" / "smxx"
    upstream_kernel = (kernel_root / "fwd_kernel2.cuh").read_text(encoding="utf-8")
    upstream_launch = (kernel_root / "fwd_launch.cu").read_text(encoding="utf-8")
    upstream_header = (source / "csrc" / "fwd.h").read_text(encoding="utf-8")
    upstream_binding = (source / "csrc" / "flash_kda.cpp").read_text(encoding="utf-8")
    outputs = p2.generated(
        upstream_kernel, upstream_launch, upstream_header, upstream_binding
    )
    outputs["fwd_kernel2_vshard8.cuh"] = rewrite_kernel(
        upstream_kernel, p2, parent, modules
    )
    outputs["fwd_launch.cu"] += rewrite_launch(upstream_launch, parent, modules)

    v4_header = parent.additions_from_upstream(
        upstream_header, modules["vshard4"].rewrite_header(upstream_header), "vshard4 header"
    )
    outputs["fwd.h"] += rename_vshard8(v4_header)

    v4_function, v4_binding = parent._vshard4_binding_parts(
        upstream_binding, modules["vshard4"]
    )
    function = rename_vshard8(v4_function)
    binding_addition = rename_vshard8(v4_binding).replace(
        "FlashKDA Forward, 4-CTA/head value-shard challenge",
        "FlashKDA Forward, 8-CTA/head V=16 PREFETCH=1,S=3 challenge",
    )
    binding = outputs["flash_kda.cpp"]
    pybind_start = binding.index("\nPYBIND11_MODULE")
    binding = binding[:pybind_start] + "\n" + function + binding[pybind_start:]
    insertion = binding.index('    m.def("get_workspace_size",')
    outputs["flash_kda.cpp"] = (
        binding[:insertion] + binding_addition + binding[insertion:]
    )
    outputs["fwd_launch.cu"] = p2.add_includes(outputs["fwd_launch.cu"], parent)
    outputs["fwd_launch.cu"] = parent.replace_once(
        outputs["fwd_launch.cu"],
        '#include "fwd_kernel2_vshard8_p2.cuh"',
        '#include "fwd_kernel2_vshard8_p2.cuh"\n#include "fwd_kernel2_vshard8.cuh"',
        "vshard8-P1 include",
    )
    return outputs


def static_check(source: Path) -> None:
    outputs = generated(source)
    launch = outputs["fwd_launch.cu"]
    required = (
        '#include "fwd_kernel2_vshard8.cuh"',
        "launch_fwd_vshard8(",
        "K2VShard8Layouts<D, D / 8, CHUNK>",
        "CHUNK, D, D / 8, kInputStages, kOutputStages, kK2Threads,",
        "constexpr int kK2Threads = 32 * 2 + 32;",
        "dim3 grid_k2(N, H * 8);",
    )
    if any(needle not in launch for needle in required):
        die("vshard8-P1 launch static check failed")
    if launch.count("constexpr int kInputStages = 3;") != 7:
        die("expected baseline plus six isolated launches")
    if outputs["fwd.h"].count("launch_fwd_vshard8(") != 1:
        die("vshard8-P1 header declaration is not isolated")
    binding = outputs["flash_kda.cpp"]
    if binding.count("void fwd_vshard8(") != 1 or binding.count('m.def("fwd_vshard8"') != 1:
        die("vshard8-P1 binding is not isolated")


def write_with_newlines(path: Path, text: str, newline: str) -> None:
    with path.open("w", encoding="utf-8", newline=newline) as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    static_check(source)
    if args.check_only:
        print("static check passed: existing comparison set + vshard8 P1/P2")
        return
    p2 = load_p2()
    modules = p2.load_parent().load_pinned_modules()
    modules["vshard2"].verify_source(source)
    kernel_root = source / "csrc" / "smxx"
    names = (
        "fwd_kernel2_vshard.cuh",
        "fwd_kernel2_vshard_p2.cuh",
        "fwd_kernel2_vshard4.cuh",
        "fwd_kernel2_vshard4_p2.cuh",
        "fwd_kernel2_vshard8_p2.cuh",
        "fwd_kernel2_vshard8.cuh",
    )
    targets = tuple(kernel_root / name for name in names)
    if any(path.exists() for path in targets):
        die("generated header already exists; use a fresh tree")
    newline = "\r\n" if b"\r\n" in (kernel_root / "fwd_kernel2.cuh").read_bytes() else "\n"
    outputs = generated(source)
    for path in targets:
        write_with_newlines(path, outputs[path.name], newline)
    write_with_newlines(kernel_root / "fwd_launch.cu", outputs["fwd_launch.cu"], newline)
    write_with_newlines(source / "csrc" / "fwd.h", outputs["fwd.h"], newline)
    write_with_newlines(source / "csrc" / "flash_kda.cpp", outputs["flash_kda.cpp"], newline)
    print("applied one-shot comparison set plus V=16 vshard8-P1/P2")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
