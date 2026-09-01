#!/usr/bin/env python3
"""Launch exactly one public fixed-length H=12 FlashKDA path for NCU.

This is deliberately not a benchmark or an exactness harness.  Its one job is
to give Nsight Compute a reproducible target process containing exactly one
call to the selected public wrapper.  The surrounding audit filters the
matching K2 recurrence kernel; K1/input-generation activity is intentionally
outside this runner's scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.harness import validate_and_bench as common


VariantCall = Callable[..., None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _variant_call(variant: str) -> tuple[str, VariantCall]:
    """Import only the selected public Python wrapper; never invoke it here."""
    if variant == "baseline":
        import flash_kda

        return "flash_kda.fwd", flash_kda.fwd
    if variant == "vshard2_p2":
        from assignment02.team.c1_flashkda.challenge_prefetch2 import prefetch2

        return "prefetch2.fwd -> flash_kda_C.fwd_vshard_p2", prefetch2.fwd
    if variant == "vshard4_p1":
        from assignment02.team.c1_flashkda.challenge_vshard4 import vshard4

        return "vshard4.fwd -> flash_kda_C.fwd_vshard4", vshard4.fwd
    if variant == "vshard4_p2":
        from assignment02.team.c1_flashkda.challenge_vshard4_prefetch2 import vshard4_prefetch2

        return "vshard4_prefetch2.fwd -> flash_kda_C.fwd_vshard4_p2", vshard4_prefetch2.fwd
    raise ValueError(f"unsupported variant: {variant}")


def _checksum(tensor: torch.Tensor) -> dict[str, float]:
    """Return stable, human-readable checksums after the profiled call completes."""
    values = tensor.float()
    return {
        "sum": float(values.sum().item()),
        "abs_sum": float(values.abs().sum().item()),
        "max_abs": float(values.abs().max().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("baseline", "vshard2_p2", "vshard4_p1", "vshard4_p2"),
    )
    parser.add_argument("--T", type=int, default=8192)
    parser.add_argument("--H", type=int, default=12)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    if args.T <= 0 or args.H <= 0 or args.T % 16 != 0:
        parser.error("T/H must be positive and T must be divisible by the fixed CHUNK=16")
    if args.D != 128:
        parser.error("the C1 public wrappers are proven only for D=128")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required; run only inside the approved GPU allocation")

    public_name, public_call = _variant_call(args.variant)
    import flash_kda_C

    extension = Path(flash_kda_C.__file__).resolve()
    if not extension.is_file():
        raise RuntimeError(f"flash_kda_C is not a regular shared object: {extension}")

    x = common.make_inputs(args.T, args.H, args.seed)
    initial_state, final_state = common.state_tensors("bf16", args.H, args.seed + 1)
    if initial_state is None or final_state is None:
        raise AssertionError("BF16 initial/final state allocation unexpectedly returned None")
    out = torch.zeros_like(x.v)

    # Input construction is intentionally complete before the sole wrapper
    # call.  No warm-up/repeat/exact reference is allowed in this NCU runner.
    torch.cuda.synchronize()
    public_call(
        x.q,
        x.k,
        x.v,
        x.g,
        x.beta,
        x.scale,
        out,
        A_log=x.a_log,
        dt_bias=x.dt_bias,
        lower_bound=x.lower_bound,
        initial_state=initial_state,
        final_state=final_state,
    )
    torch.cuda.synchronize()

    result = {
        "schema_version": 1,
        "variant": args.variant,
        "public_wrapper": public_name,
        "public_wrapper_calls": 1,
        "shape": {"B": 1, "T": args.T, "H": args.H, "D": args.D},
        "state": {"mode": "bf16", "initial_shape": list(initial_state.shape), "final_shape": list(final_state.shape)},
        "seed": args.seed,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "extension": str(extension),
        "extension_sha256": _sha256(extension),
        "output_checksum": _checksum(out),
        "final_state_checksum": _checksum(final_state),
        "scope": "one selected public wrapper call only; no benchmark and no exact/reference matrix",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
