#!/usr/bin/env python3
"""Generate a one-SO V=16 / eight-CTA-per-head P2 comparison set.

The generator is intentionally layered on the exact audited vshard4-P2
generator.  It adds one isolated ``fwd_vshard8_p2`` entry while retaining the
baseline, vshard2-P1/P2 and vshard4-P1/P2 entries for same-SO comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


PARENT_RELATIVE = "challenge_vshard4_prefetch2/apply_vshard4_prefetch2_patch.py"
PARENT_SHA256 = "d4d9b2638c0b02c7fbb239684995554d94f3a581070818069e7b6450e9140813"


def die(message: str) -> None:
    raise RuntimeError(message)


def load_parent() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / PARENT_RELATIVE
    if not path.is_file():
        die(f"missing pinned parent generator: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != PARENT_SHA256:
        die(f"parent generator digest changed: expected {PARENT_SHA256}, got {actual}")
    spec = importlib.util.spec_from_file_location("c1_vshard4_p2_pinned_for_v8", path)
    if spec is None or spec.loader is None:
        die(f"cannot load parent generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rename_vshard8_p2(text: str) -> str:
    return (
        text.replace("VSHARD4", "VSHARD8_P2")
        .replace("VShard4", "VShard8P2")
        .replace("vshard4", "vshard8_p2")
    )


def rewrite_v8_kernel(upstream: str, parent: ModuleType, modules: dict[str, ModuleType]) -> str:
    v4_p1 = modules["vshard4"].rewrite_kernel2(upstream)
    text = parent.with_prefetch2(v4_p1, rename_vshard8_p2)
    text = parent.replace_once(
        text,
        'static_assert(KDim == 128 && VDim == 32, "vshard8_p2 is deliberately specialized to K=128,Vshard=32");',
        'static_assert(KDim == 128 && VDim == 16, "vshard8_p2 is deliberately specialized to K=128,Vshard=16");',
        "vshard8 dimensions",
    )
    text = parent.replace_once(
        text,
        "constexpr int kComputeThreads = 64;",
        "constexpr int kComputeThreads = 32;",
        "vshard8 compute threads",
    )
    required = (
        "K2VShard8P2Layouts",
        "SharedStorageK2VShard8P2",
        "_flash_kda_fwd_recurrence_vshard8_p2",
        "VDim == 16",
        "constexpr int kComputeThreads = 32;",
        "constexpr int PREFETCH = 2;",
        "ring_S_acc[1][PREFETCH]",
    )
    if any(needle not in text for needle in required):
        die("vshard8 kernel static check failed")
    if any(needle in text for needle in ("u_acc[1]);", "out_acc[1]);", "tCrB_u_arr[1](_")):
        die("vshard8 kernel retained a second V-block access")
    return text


def rewrite_v8_launch(upstream: str, parent: ModuleType, modules: dict[str, ModuleType]) -> str:
    v4_addition = parent.isolate_vshard4_launch_macros(
        parent.additions_from_upstream(
            upstream, modules["vshard4"].rewrite_launch(upstream), "vshard4 launch"
        )
    )
    text = rename_vshard8_p2(v4_addition)
    text = parent.replace_once(
        text,
        "using K2L = K2VShard8P2Layouts<D, D / 4, CHUNK>;",
        "using K2L = K2VShard8P2Layouts<D, D / 8, CHUNK>;",
        "vshard8 V layout",
    )
    text = parent.replace_once(
        text,
        "CHUNK, D, D / 4, kInputStages, kOutputStages, kK2Threads,",
        "CHUNK, D, D / 8, kInputStages, kOutputStages, kK2Threads,",
        "vshard8 kernel V template",
    )
    text = parent.replace_once(
        text,
        "constexpr int kK2Threads = 32 * 2 + 64;",
        "constexpr int kK2Threads = 32 * 2 + 32;",
        "vshard8 launch threads",
    )
    text = parent.replace_once(
        text,
        "dim3 grid_k2(N, H * 4);",
        "dim3 grid_k2(N, H * 8);",
        "vshard8 launch grid",
    )
    return text


def rewrite_header_all(upstream: str, parent: ModuleType, modules: dict[str, ModuleType]) -> str:
    base = parent.rewrite_header_all(upstream, modules)
    v4_addition = parent.additions_from_upstream(
        upstream, modules["vshard4"].rewrite_header(upstream), "vshard4 header"
    )
    return base + rename_vshard8_p2(v4_addition)


def rewrite_binding_all(upstream: str, parent: ModuleType, modules: dict[str, ModuleType]) -> str:
    binding = parent.rewrite_binding_all(upstream, modules)
    v4_function, v4_binding = parent._vshard4_binding_parts(upstream, modules["vshard4"])
    v8_function = rename_vshard8_p2(v4_function)
    v8_binding = rename_vshard8_p2(v4_binding).replace(
        "FlashKDA Forward, 4-CTA/head value-shard challenge",
        "FlashKDA Forward, 8-CTA/head V=16 PREFETCH=2,S=3 challenge",
    )
    pybind_start = binding.index("\nPYBIND11_MODULE")
    binding = binding[:pybind_start] + "\n" + v8_function + binding[pybind_start:]
    insertion = binding.index('    m.def("get_workspace_size",')
    return binding[:insertion] + v8_binding + binding[insertion:]


def generated(
    upstream_kernel: str,
    upstream_launch: str,
    upstream_header: str,
    upstream_binding: str,
) -> dict[str, str]:
    parent = load_parent()
    modules = parent.load_pinned_modules()
    outputs = parent.generated(
        upstream_kernel, upstream_launch, upstream_header, upstream_binding
    )
    outputs["fwd_kernel2_vshard8_p2.cuh"] = rewrite_v8_kernel(
        upstream_kernel, parent, modules
    )
    outputs["fwd_launch.cu"] += rewrite_v8_launch(upstream_launch, parent, modules)
    outputs["fwd.h"] = rewrite_header_all(upstream_header, parent, modules)
    outputs["flash_kda.cpp"] = rewrite_binding_all(upstream_binding, parent, modules)
    return outputs


def add_includes(text: str, parent: ModuleType) -> str:
    includes = "\n".join(
        (
            '#include "fwd_kernel2.cuh"',
            '#include "fwd_kernel2_vshard.cuh"',
            '#include "fwd_kernel2_vshard_p2.cuh"',
            '#include "fwd_kernel2_vshard4.cuh"',
            '#include "fwd_kernel2_vshard4_p2.cuh"',
            '#include "fwd_kernel2_vshard8_p2.cuh"',
        )
    )
    return parent.replace_once(
        text, '#include "fwd_kernel2.cuh"', includes, "all generated kernel includes"
    )


def static_check(source: Path) -> None:
    parent = load_parent()
    kernel_root = source / "csrc" / "smxx"
    outputs = generated(
        (kernel_root / "fwd_kernel2.cuh").read_text(encoding="utf-8"),
        (kernel_root / "fwd_launch.cu").read_text(encoding="utf-8"),
        (source / "csrc" / "fwd.h").read_text(encoding="utf-8"),
        (source / "csrc" / "flash_kda.cpp").read_text(encoding="utf-8"),
    )
    launch = add_includes(outputs["fwd_launch.cu"], parent)
    required_launch = (
        '#include "fwd_kernel2_vshard8_p2.cuh"',
        "launch_fwd_vshard8_p2(",
        "K2VShard8P2Layouts<D, D / 8, CHUNK>",
        "CHUNK, D, D / 8, kInputStages, kOutputStages, kK2Threads,",
        "constexpr int kK2Threads = 32 * 2 + 32;",
        "dim3 grid_k2(N, H * 8);",
    )
    if any(needle not in launch for needle in required_launch):
        die("vshard8 launch static check failed")
    if launch.count("constexpr int kInputStages = 3;") != 6:
        die("expected baseline + five isolated comparison launches")
    if outputs["fwd.h"].count("launch_fwd_vshard8_p2(") != 1:
        die("vshard8 header declaration is not isolated")
    binding = outputs["flash_kda.cpp"]
    if binding.count("void fwd_vshard8_p2(") != 1 or binding.count(
        'm.def("fwd_vshard8_p2"'
    ) != 1:
        die("vshard8 binding is not isolated")


def write_with_source_newlines(path: Path, text: str, newline: str) -> None:
    with path.open("w", encoding="utf-8", newline=newline) as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    parent = load_parent()
    modules = parent.load_pinned_modules()
    kernel_root = source / "csrc" / "smxx"
    static_check(source)
    if args.check_only:
        print("static check passed: one upstream tree -> existing comparison set + V=16 vshard8-P2")
        return
    modules["vshard2"].verify_source(source)
    targets = tuple(
        kernel_root / name
        for name in (
            "fwd_kernel2_vshard.cuh",
            "fwd_kernel2_vshard_p2.cuh",
            "fwd_kernel2_vshard4.cuh",
            "fwd_kernel2_vshard4_p2.cuh",
            "fwd_kernel2_vshard8_p2.cuh",
        )
    )
    if any(path.exists() for path in targets):
        die("a generated kernel header already exists; refusing repeated patch application")
    upstream_kernel = (kernel_root / "fwd_kernel2.cuh").read_text(encoding="utf-8")
    source_newline = (
        "\r\n" if b"\r\n" in (kernel_root / "fwd_kernel2.cuh").read_bytes() else "\n"
    )
    outputs = generated(
        upstream_kernel,
        (kernel_root / "fwd_launch.cu").read_text(encoding="utf-8"),
        (source / "csrc" / "fwd.h").read_text(encoding="utf-8"),
        (source / "csrc" / "flash_kda.cpp").read_text(encoding="utf-8"),
    )
    outputs["fwd_launch.cu"] = add_includes(outputs["fwd_launch.cu"], parent)
    for path in targets:
        write_with_source_newlines(path, outputs[path.name], source_newline)
    write_with_source_newlines(kernel_root / "fwd_launch.cu", outputs["fwd_launch.cu"], source_newline)
    write_with_source_newlines(source / "csrc" / "fwd.h", outputs["fwd.h"], source_newline)
    write_with_source_newlines(source / "csrc" / "flash_kda.cpp", outputs["flash_kda.cpp"], source_newline)
    print("applied one-shot comparison set plus V=16 vshard8-P2")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
