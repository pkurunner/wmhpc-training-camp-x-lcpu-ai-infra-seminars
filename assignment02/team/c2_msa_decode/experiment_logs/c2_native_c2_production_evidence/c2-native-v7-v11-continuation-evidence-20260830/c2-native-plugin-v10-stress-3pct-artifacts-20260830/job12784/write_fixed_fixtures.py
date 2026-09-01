#!/usr/bin/env python3
"""Create one immutable CPU fixture payload from the frozen base generator."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("c2_v10_fixture_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("Contract", "_make_inputs", "_validate_args"):
        if not hasattr(module, name):
            raise RuntimeError(f"base harness lacks {name}")
    return module


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(repr(tuple(cpu.shape)).encode("ascii"))
    digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-harness", type=Path, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(part) for part in args.seeds.split(",")]
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("expected exactly eight unique fixed seeds")
    base = load_base(args.base_harness)
    contract = base._validate_args(type("Args", (), {
        "num_physical_pages": 64, "max_logical_pages": 32,
        "scale": 1.0 / (128.0 ** 0.5), "q_scale": 0.25,
        "k_scale": 0.25, "v_scale": 0.5, "atol": 1.0e-4, "rtol": 1.0e-3,
    })())
    rows = []
    for seed in seeds:
        tensors = tuple(item.detach().cpu().contiguous() for item in base._make_inputs(contract, seed))
        rows.append({"seed": seed, "inputs": tensors,
                     "tensor_sha256": [tensor_sha256(item) for item in tensors]})
    payload = {
        "schema": "c2-native-v10-k-prefetch-fixed-fixtures-v1", "seeds": seeds,
        "contract": {"num_physical_pages": contract.num_physical_pages,
                     "max_logical_pages": contract.max_logical_pages,
                     "scale": contract.scale, "q_scale": contract.q_scale,
                     "k_scale": contract.k_scale, "v_scale": contract.v_scale},
        "rows": rows,
    }
    torch.save(payload, args.fixture)
    meta = {
        "schema": payload["schema"], "seeds": seeds,
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "base_harness_sha256": hashlib.sha256(args.base_harness.read_bytes()).hexdigest(),
        "contract": payload["contract"],
        "per_seed_tensor_sha256": {str(row["seed"]): row["tensor_sha256"] for row in rows},
    }
    args.meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
