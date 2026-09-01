#!/usr/bin/env python3
"""A one-process direct C2 target; no DSO may share this interpreter."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import torch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("c2_ncu_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("Contract", "_BATCH", "_QUERY_HEADS", "_HEAD_DIM", "_make_inputs", "_oracle", "_validate_args"):
        if not hasattr(module, name):
            raise RuntimeError(f"base harness lacks {name}")
    return module


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode("ascii"))
    h.update(repr(tuple(value.shape)).encode("ascii"))
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def visible_uuid() -> str:
    visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if len(visible) == 1 and (visible[0].isdigit() or visible[0].startswith("GPU-")):
        values = [line.strip() for line in subprocess.check_output(
            ["nvidia-smi", "--id", visible[0], "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines() if line.strip()]
        if len(values) == 1 and values[0].startswith("GPU-"):
            return values[0]
    value = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if isinstance(value, str) and value.startswith("GPU-"):
        return value
    raise RuntimeError("cannot establish allocated physical GPU UUID")


def call(op, output, inputs, contract):
    return op(output, *inputs, contract.scale, contract.q_scale, contract.k_scale, contract.v_scale)


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
    if torch.cuda.get_device_capability(0) != (10, 3):
        raise RuntimeError(f"B300 capability required, got {torch.cuda.get_device_capability(0)}")
    if not args.library.is_absolute() or not args.base_harness.is_absolute():
        raise ValueError("paths must be absolute")
    if digest(args.base_harness) != args.base_harness_sha256:
        raise RuntimeError("base harness checksum drift")
    base = load_base(args.base_harness)
    contract = base._validate_args(type("Args", (), {
        "num_physical_pages": 64, "max_logical_pages": 32,
        "scale": 1.0 / (128.0 ** 0.5), "q_scale": 0.25,
        "k_scale": 0.25, "v_scale": 0.5, "atol": 1.0e-4, "rtol": 1.0e-3,
    })())
    torch.ops.load_library(str(args.library))
    if not torch._C._dispatch_has_kernel_for_dispatch_key("_C::native_c2_msa_decode", "CUDA"):
        raise RuntimeError("native C2 CUDA dispatch is not registered")
    inputs = base._make_inputs(contract, args.seed)
    input_hashes = [tensor_digest(item) for item in inputs]
    input_manifest = hashlib.sha256("".join(input_hashes).encode("ascii")).hexdigest()
    output = torch.full((base._BATCH, base._QUERY_HEADS, base._HEAD_DIM), float("nan"), device="cuda", dtype=torch.bfloat16)
    pointer = output.data_ptr()
    returned = call(torch.ops._C.native_c2_msa_decode, output, inputs, contract)
    torch.cuda.synchronize()
    checks = {"return_is_none": returned is None, "pointer_unchanged": output.data_ptr() == pointer,
              "output_finite": bool(torch.isfinite(output).all().item())}
    record = {
        "schema": "c2-native-v6-ncu-target-v1", "mode": args.mode, "all_gates_pass": False,
        "operator_library": str(args.library.resolve()), "operator_library_sha256": digest(args.library),
        "base_harness": str(args.base_harness.resolve()), "base_harness_sha256": digest(args.base_harness),
        "seed": args.seed, "input_tensor_sha256": input_hashes, "input_manifest_sha256": input_manifest,
        "contract": {"batch": base._BATCH, "head_dim": base._HEAD_DIM, "kv_heads": 4, "q_heads": base._QUERY_HEADS,
                     "page_size": 128, "topk": 16, "num_physical_pages": 64, "max_logical_pages": 32,
                     "scale": contract.scale, "q_scale": contract.q_scale, "k_scale": contract.k_scale, "v_scale": contract.v_scale},
        "caller_output": checks, "device": {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
                     "gpu_uuid": visible_uuid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "dispatch": "direct_torch.ops._C.native_c2_msa_decode", "no_monkeypatch": True,
    }
    if args.mode == "ncu":
        record["all_gates_pass"] = all(checks.values())
        return record
    reference = base._oracle(*inputs, contract, torch.float64)
    actual = output.to(torch.float64)
    delta = (actual - reference).abs()
    denominator = reference.abs().clamp_min(torch.finfo(torch.float64).eps)
    correct = bool(torch.allclose(actual, reference, atol=1.0e-4, rtol=1.0e-3))
    profile_output = torch.full_like(output, float("nan"))
    profile_pointer = profile_output.data_ptr()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        profile_return = call(torch.ops._C.native_c2_msa_decode, profile_output, inputs, contract)
        torch.cuda.synchronize()
    events = profiler.key_averages()
    dispatches = [event for event in events if str(event.key) == "_C::native_c2_msa_decode"]
    kernels = [event for event in events if "native_c2_msa_decode_kernel" in str(event.key)]
    profile_checks = {"one_dispatch": len(dispatches) == 1 and int(getattr(dispatches[0], "count", 0)) == 1,
                      "one_native_kernel": len(kernels) == 1 and int(getattr(kernels[0], "count", 0)) == 1,
                      "profile_return_is_none": profile_return is None,
                      "profile_pointer_unchanged": profile_output.data_ptr() == profile_pointer}
    record["correctness"] = {"oracle_dtype": "float64", "atol": 1.0e-4, "rtol": 1.0e-3, "allclose": correct,
                              "finite_output": checks["output_finite"], "max_abs": float(delta.max().item()),
                              "mean_abs": float(delta.mean().item()), "max_rel": float((delta / denominator).max().item())}
    record["profiler"] = {"checks": profile_checks, "dispatch_count": len(dispatches), "native_kernel_count": len(kernels)}
    record["output_sha256"] = tensor_digest(output)
    record["all_gates_pass"] = bool(all(checks.values()) and correct and all(profile_checks.values()))
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "ncu"), required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--base-harness", type=Path, required=True)
    parser.add_argument("--base-harness-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        result = {"schema": "c2-native-v6-ncu-target-v1", "mode": args.mode, "all_gates_pass": False,
                  "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
