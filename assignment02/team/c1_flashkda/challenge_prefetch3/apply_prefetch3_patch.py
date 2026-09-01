#!/usr/bin/env python3
"""Generate baseline/P1/P2S3/P3S3 entries from pinned FlashKDA 1ce47ea."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path


P2_GENERATOR_SHA256 = "f83e3551907ec8f1a5c1f5c3421e94dc1e3d1941e9f35c845d1d982eef38ccb0"


def die(message: str) -> None:
    raise RuntimeError(message)


def load_p2():
    path = Path(__file__).resolve().with_name("pinned_prefetch2_generator.py")
    if not path.is_file():
        die(f"missing vendored current P2S3 generator: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != P2_GENERATOR_SHA256:
        die(f"P2S3 generator digest changed: expected {P2_GENERATOR_SHA256}, got {digest}")
    spec = importlib.util.spec_from_file_location("c1_prefetch2_pinned_for_p3", path)
    if spec is None or spec.loader is None:
        die(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rename_p2_to_p3(text: str) -> str:
    return text.replace("VSHARD_P2", "VSHARD_P3").replace("VShardP2", "VShardP3").replace("vshard_p2", "vshard_p3")


def rewrite_kernel3(upstream: str) -> str:
    p2 = load_p2()
    base = p2.load_base()
    text = rename_p2_to_p3(p2.rewrite_kernel2_p2(upstream))
    return base.replace_once(
        text,
        "            constexpr int PREFETCH = 2;",
        "            constexpr int PREFETCH = 3;",
        "P3 Phase-6 software prefetch depth",
    )


def rewrite_launch_all(upstream: str) -> str:
    p2 = load_p2()
    _, p2_addition = p2.launch_additions(upstream, 3)
    return p2.rewrite_launch_both(upstream, 3) + rename_p2_to_p3(p2_addition)


def rewrite_header_all(upstream: str) -> str:
    p2 = load_p2()
    base = p2.load_base()
    with_p1 = base.rewrite_header(upstream)
    with_p2 = p2.rewrite_header_both(upstream)
    p2_addition = with_p2[len(with_p1):]
    if "launch_fwd_vshard_p2" not in p2_addition:
        die("cannot isolate P2 header declaration")
    return with_p2 + rename_p2_to_p3(p2_addition)


def rewrite_binding_all(upstream: str) -> str:
    p2 = load_p2()
    with_p2 = p2.rewrite_binding_both(upstream)
    function_start = with_p2.index("void fwd_vshard_p2(")
    pybind_start = with_p2.index("\nPYBIND11_MODULE", function_start)
    p2_function = with_p2[function_start:pybind_start]
    result = with_p2[:pybind_start] + "\n" + rename_p2_to_p3(p2_function) + with_p2[pybind_start:]
    binding_start = result.index('    m.def("fwd_vshard_p2",')
    binding_end = result.index('    m.def("get_workspace_size",', binding_start)
    p2_binding = result[binding_start:binding_end]
    p3_binding = rename_p2_to_p3(p2_binding).replace("PREFETCH=2 challenge", "PREFETCH=3 challenge")
    return result[:binding_end] + p3_binding + result[binding_end:]


def static_check(source: Path) -> None:
    p2 = load_p2()
    base = p2.load_base()
    kernel_upstream = (source / "csrc" / "smxx" / "fwd_kernel2.cuh").read_text(encoding="utf-8")
    launch_upstream = (source / "csrc" / "smxx" / "fwd_launch.cu").read_text(encoding="utf-8")
    header_upstream = (source / "csrc" / "fwd.h").read_text(encoding="utf-8")
    binding_upstream = (source / "csrc" / "flash_kda.cpp").read_text(encoding="utf-8")
    p1_kernel = base.rewrite_kernel2(kernel_upstream)
    p2_kernel = p2.rewrite_kernel2_p2(kernel_upstream)
    p3_kernel = rewrite_kernel3(kernel_upstream)
    for label, text, depth, symbol in (
        ("P1", p1_kernel, 1, "_flash_kda_fwd_recurrence_vshard"),
        ("P2", p2_kernel, 2, "_flash_kda_fwd_recurrence_vshard_p2"),
        ("P3", p3_kernel, 3, "_flash_kda_fwd_recurrence_vshard_p3"),
    ):
        if text.count(f"constexpr int PREFETCH = {depth};") != 1 or symbol not in text:
            die(f"{label} kernel identity/depth check failed")
        if "ring_A_kr[PREFETCH]" not in text or "slot = m % PREFETCH" not in text or "m + PREFETCH < S_M_BLOCKS" not in text:
            die(f"{label} ring parameterization check failed")
    launch = rewrite_launch_all(launch_upstream)
    if launch.count("constexpr int kInputStages = 3;") != 4 or "constexpr int kInputStages = 2;" in launch:
        die("baseline/P1/P2/P3 must all be stage 3")
    for symbol in ("launch_fwd_vshard(", "launch_fwd_vshard_p2(", "launch_fwd_vshard_p3("):
        if symbol not in launch:
            die(f"missing launch symbol {symbol}")
    header = rewrite_header_all(header_upstream)
    binding = rewrite_binding_all(binding_upstream)
    for symbol in ("launch_fwd_vshard(", "launch_fwd_vshard_p2(", "launch_fwd_vshard_p3("):
        if symbol not in header:
            die(f"missing header symbol {symbol}")
    for symbol in ('m.def("fwd_vshard"', 'm.def("fwd_vshard_p2"', 'm.def("fwd_vshard_p3"'):
        if symbol not in binding:
            die(f"missing binding {symbol}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    p2 = load_p2()
    base = p2.load_base()
    base.verify_source(source)
    static_check(source)
    if args.check_only:
        print("static check passed: baseline/P1/P2/P3 stage3; PREFETCH=1/1/2/3")
        return

    kernel = source / "csrc" / "smxx" / "fwd_kernel2.cuh"
    p1_kernel = source / "csrc" / "smxx" / "fwd_kernel2_vshard.cuh"
    p2_kernel = source / "csrc" / "smxx" / "fwd_kernel2_vshard_p2.cuh"
    p3_kernel = source / "csrc" / "smxx" / "fwd_kernel2_vshard_p3.cuh"
    launch = source / "csrc" / "smxx" / "fwd_launch.cu"
    header = source / "csrc" / "fwd.h"
    binding = source / "csrc" / "flash_kda.cpp"
    if any(path.exists() for path in (p1_kernel, p2_kernel, p3_kernel)):
        die("generated header already exists; refusing to patch twice")
    upstream_kernel = kernel.read_text(encoding="utf-8")
    p1_kernel.write_text(base.rewrite_kernel2(upstream_kernel), encoding="utf-8")
    p2_kernel.write_text(p2.rewrite_kernel2_p2(upstream_kernel), encoding="utf-8")
    p3_kernel.write_text(rewrite_kernel3(upstream_kernel), encoding="utf-8")
    launch_text = launch.read_text(encoding="utf-8")
    launch_text = base.replace_once(
        launch_text,
        '#include "fwd_kernel2.cuh"',
        '#include "fwd_kernel2.cuh"\n#include "fwd_kernel2_vshard.cuh"\n#include "fwd_kernel2_vshard_p2.cuh"\n#include "fwd_kernel2_vshard_p3.cuh"',
        "P1/P2/P3 kernel includes",
    )
    launch.write_text(rewrite_launch_all(launch_text), encoding="utf-8")
    header.write_text(rewrite_header_all(header.read_text(encoding="utf-8")), encoding="utf-8")
    binding.write_text(rewrite_binding_all(binding.read_text(encoding="utf-8")), encoding="utf-8")
    print("applied isolated P1/P2S3/P3S3 patch")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
