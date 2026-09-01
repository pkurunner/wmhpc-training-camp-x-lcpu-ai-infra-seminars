#!/usr/bin/env python3
"""Generate isolated P1/P2 value-shard entries from pinned FlashKDA 1ce47ea.

The upstream ``fwd`` and the current two-CTA ``fwd_vshard`` remain available.
``fwd_vshard_p2`` always applies the K2 Phase-6 software ring-depth change
(``PREFETCH=1`` to 2).  Its K2 input-pipeline depth is selectable as 2 or 3;
the default is the measured-best P2S3 variant.  P1 remains bit-for-bit stage 3.
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
        die(f"missing pinned vshard2 generator: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BASE_SHA256:
        die(f"vshard2 generator digest changed: expected {BASE_SHA256}, got {digest}")
    spec = importlib.util.spec_from_file_location("c1_vshard2_pinned_for_p2", path)
    if spec is None or spec.loader is None:
        die(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rename_p1_to_p2(text: str) -> str:
    # The order matters: macro, CamelCase type, then lowercase symbol names.
    return text.replace("VSHARD", "VSHARD_P2").replace("VShard", "VShardP2").replace("vshard", "vshard_p2")


def rewrite_kernel2_p2(upstream: str) -> str:
    base = load_base()
    text = rename_p1_to_p2(base.rewrite_kernel2(upstream))
    text = base.replace_once(
        text,
        "            constexpr int PREFETCH = 1;",
        "            constexpr int PREFETCH = 2;",
        "Phase-6 software prefetch depth",
    )
    text = base.replace_once(
        text,
        "            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;",
        """            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;
            static_assert(PREFETCH > 0 && PREFETCH <= S_M_BLOCKS, "invalid Phase-6 prefetch depth");""",
        "Phase-6 prefetch bound",
    )
    return text


def launch_additions(upstream: str, p2_input_stages: int) -> tuple[str, str]:
    if p2_input_stages not in (2, 3):
        die(f"P2 input stages must be 2 or 3, got {p2_input_stages}")
    base = load_base()
    with_p1 = base.rewrite_launch(upstream)
    p1_addition = with_p1[len(upstream):]
    if "launch_fwd_vshard" not in p1_addition or "INSTANTIATE_VSHARD" not in p1_addition:
        die("cannot isolate pinned P1 launch addition")
    if p1_addition.count("    constexpr int kInputStages = 3;") != 1:
        die("pinned P1 launch must contain exactly one stage-3 K2 configuration")
    p2_addition = rename_p1_to_p2(p1_addition)
    if p2_input_stages == 2:
        p2_addition = base.replace_once(
            p2_addition,
            "    constexpr int kInputStages = 3;",
            "    constexpr int kInputStages = 2;",
            "P2 K2 input pipeline depth",
        )
    return p1_addition, p2_addition


def rewrite_launch_both(upstream: str, p2_input_stages: int = 3) -> str:
    p1_addition, p2_addition = launch_additions(upstream, p2_input_stages)
    with_p1 = upstream + p1_addition
    return with_p1 + p2_addition


def rewrite_header_both(upstream: str) -> str:
    base = load_base()
    with_p1 = base.rewrite_header(upstream)
    p1_addition = with_p1[len(upstream):]
    if "launch_fwd_vshard" not in p1_addition:
        die("cannot isolate pinned P1 header declaration")
    return with_p1 + rename_p1_to_p2(p1_addition)


def rewrite_binding_both(upstream: str) -> str:
    base = load_base()
    with_p1 = base.rewrite_binding(upstream)
    function_start = with_p1.index("void fwd_vshard(")
    pybind_start = with_p1.index("\nPYBIND11_MODULE", function_start)
    p1_function = with_p1[function_start:pybind_start]
    p2_function = rename_p1_to_p2(p1_function)
    result = with_p1[:pybind_start] + "\n" + p2_function + with_p1[pybind_start:]

    binding_start = result.index('    m.def("fwd_vshard",')
    binding_end = result.index('    m.def("get_workspace_size",', binding_start)
    p1_binding = result[binding_start:binding_end]
    p2_binding = rename_p1_to_p2(p1_binding).replace(
        "FlashKDA Forward, 2-CTA/head value-shard challenge",
        "FlashKDA Forward, 2-CTA/head value-shard PREFETCH=2 challenge",
    )
    return result[:binding_end] + p2_binding + result[binding_end:]


def static_check(source: Path, p2_input_stages: int) -> None:
    base = load_base()
    kernel_upstream = (source / "csrc" / "smxx" / "fwd_kernel2.cuh").read_text(encoding="utf-8")
    launch_upstream = (source / "csrc" / "smxx" / "fwd_launch.cu").read_text(encoding="utf-8")
    header_upstream = (source / "csrc" / "fwd.h").read_text(encoding="utf-8")
    binding_upstream = (source / "csrc" / "flash_kda.cpp").read_text(encoding="utf-8")
    p1 = base.rewrite_kernel2(kernel_upstream)
    p2 = rewrite_kernel2_p2(kernel_upstream)
    if p1.count("constexpr int PREFETCH = 1;") != 1 or "PREFETCH = 2" in p1:
        die("pinned P1 kernel changed")
    required_p2 = (
        "constexpr int PREFETCH = 2;",
        "ring_A_kr[PREFETCH]",
        "ring_S_acc[1][PREFETCH]",
        "slot = m % PREFETCH",
        "m + PREFETCH < S_M_BLOCKS",
        "static_assert(PREFETCH > 0 && PREFETCH <= S_M_BLOCKS",
        "_flash_kda_fwd_recurrence_vshard_p2",
    )
    if any(needle not in p2 for needle in required_p2) or "constexpr int PREFETCH = 1;" in p2:
        die("P2 kernel static check failed")
    p1_launch, p2_launch = launch_additions(launch_upstream, p2_input_stages)
    launch = launch_upstream + p1_launch + p2_launch
    header = rewrite_header_both(header_upstream)
    binding = rewrite_binding_both(binding_upstream)
    required_launch = ("launch_fwd_vshard(", "launch_fwd_vshard_p2(",
                       "_flash_kda_fwd_recurrence_vshard<", "_flash_kda_fwd_recurrence_vshard_p2<")
    if any(needle not in launch for needle in required_launch):
        die("P1/P2 launch static check failed")
    if launch_upstream.count("constexpr int kInputStages = 3;") != 1 or "constexpr int kInputStages = 2;" in launch_upstream:
        die("pinned public baseline launch is not exactly stage 3")
    if p1_launch.count("constexpr int kInputStages = 3;") != 1 or "constexpr int kInputStages = 2;" in p1_launch:
        die("P1 launch is not exactly stage 3")
    expected_p2 = f"constexpr int kInputStages = {p2_input_stages};"
    unexpected_p2 = f"constexpr int kInputStages = {5 - p2_input_stages};"
    if p2_launch.count(expected_p2) != 1 or unexpected_p2 in p2_launch:
        die(f"P2 launch is not exactly stage {p2_input_stages}")
    expected_stage3_count = 3 if p2_input_stages == 3 else 2
    expected_stage2_count = 0 if p2_input_stages == 3 else 1
    if (launch.count("constexpr int kInputStages = 3;") != expected_stage3_count or
            launch.count("constexpr int kInputStages = 2;") != expected_stage2_count):
        die("combined baseline/P1/P2 stage counts are inconsistent")
    if "launch_fwd_vshard_p2(" not in header:
        die("P2 header declaration missing")
    if 'm.def("fwd_vshard"' not in binding or 'm.def("fwd_vshard_p2"' not in binding:
        die("P1/P2 pybind entries missing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="fresh FlashKDA 1ce47ea clone/worktree")
    parser.add_argument(
        "--p2-input-stages", type=int, choices=(2, 3), default=3,
        help="K2 input-pipeline depth for P2 (default: 3, measured-best P2S3)",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    base = load_base()
    base.verify_source(source)
    static_check(source, args.p2_input_stages)
    if args.check_only:
        print(
            "static check passed: baseline/P1 stage3 retained; "
            f"P2 uses PREFETCH=2 plus input-stages={args.p2_input_stages}"
        )
        return

    kernel = source / "csrc" / "smxx" / "fwd_kernel2.cuh"
    p1_kernel = source / "csrc" / "smxx" / "fwd_kernel2_vshard.cuh"
    p2_kernel = source / "csrc" / "smxx" / "fwd_kernel2_vshard_p2.cuh"
    launch = source / "csrc" / "smxx" / "fwd_launch.cu"
    header = source / "csrc" / "fwd.h"
    binding = source / "csrc" / "flash_kda.cpp"
    if p1_kernel.exists() or p2_kernel.exists():
        die("P1/P2 generated header already exists; refusing to patch twice")

    upstream_kernel = kernel.read_text(encoding="utf-8")
    p1_kernel.write_text(base.rewrite_kernel2(upstream_kernel), encoding="utf-8")
    p2_kernel.write_text(rewrite_kernel2_p2(upstream_kernel), encoding="utf-8")
    launch_text = launch.read_text(encoding="utf-8")
    launch_text = base.replace_once(
        launch_text,
        '#include "fwd_kernel2.cuh"',
        '#include "fwd_kernel2.cuh"\n#include "fwd_kernel2_vshard.cuh"\n#include "fwd_kernel2_vshard_p2.cuh"',
        "P1/P2 kernel includes",
    )
    launch.write_text(rewrite_launch_both(launch_text, args.p2_input_stages), encoding="utf-8")
    header.write_text(rewrite_header_both(header.read_text(encoding="utf-8")), encoding="utf-8")
    binding.write_text(rewrite_binding_both(binding.read_text(encoding="utf-8")), encoding="utf-8")
    print(
        "applied isolated P1/P2 value-shard patch: "
        f"P2 PREFETCH=2, input-stages={args.p2_input_stages}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
