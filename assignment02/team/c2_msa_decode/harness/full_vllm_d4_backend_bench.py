"""Exact d4da0c5 vLLM MiniMax-M3 MSA backend-layer B=16 FP8 audit.

| Symbol | Meaning | Audit value |
| --- | --- | --- |
| B | decode requests/query tokens | 16 |
| Hq/Hkv/D | Q heads/KV heads/head dimension | 64/4/128 |
| P/K | page tokens/selected pages per KV head | 128/16 |
| S_b | causal length of request b | random [2048, 4096] |
| q_s,k_s,v_s | FP8 dequantization scales | 0.25, 0.25, 0.5 |

Boundary: exact full-vLLM backend layer path; no model weights/server scheduler.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Callable
import zipfile

import torch

VLLM_COMMIT = "d4da0c55af3aa231b6209bf77871f3ed36eab0d2"
MSA_COMMIT = "087c161814d4d9c735b46c21212a09e5f8eb92fa"
CUTLASS_COMMIT = "eb61c911471867a5fd2466bfd8f29306cea6ebf8"
WHEEL_SHA256 = "91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06"
VERSION_FRAGMENT = "gd4da0c55a"
BATCH, Q_SCALE, RTOL, ATOL = 16, 0.25, 0.03, 0.03
VLLM_SOURCES = (
    "vllm/models/minimax_m3/nvidia/sparse_attention_msa.py",
    "vllm/models/minimax_m3/nvidia/msa_cutlass_sparse_decode.py",
    "vllm/models/minimax_m3/common/ops/sparse_attn.py",
)
FMHA_SOURCES = ("__init__.py", "api.py", "jit.py", "sparse.py", "sparse_fmha_adapter.py")


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def sha(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, text=True, capture_output=True
    ).stdout.strip()


def _tree_map(root: Path, *, skip_cutlass: bool = False) -> dict[str, str]:
    """CMake-source tree map, omitting VCS and import-cache artifacts."""
    result: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if (
            rel.startswith("__pycache__/")
            or "/__pycache__/" in rel
            or rel.endswith((".pyc", ".pyo"))
            or rel.endswith(".gitignore")
            or (skip_cutlass and rel.startswith("cutlass/"))
        ):
            continue
        result[rel] = sha(path)
    return result


def _map_digest(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _wheel_fmha_map(wheel: Path) -> dict[str, str]:
    prefix = "vllm/third_party/fmha_sm100/"
    result: dict[str, str] = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            rel = name.removeprefix(prefix)
            if rel.startswith("__pycache__/") or "/__pycache__/" in rel or rel.endswith((".pyc", ".pyo")):
                continue
            result[rel] = hashlib.sha256(archive.read(name)).hexdigest()
    if not result:
        raise RuntimeError(f"wheel has no packaged fmha_sm100 tree: {wheel}")
    return result


def source_audit(installed_root: Path, vllm_root: Path, msa_root: Path, wheel: Path) -> tuple[dict[str, Any], bool]:
    vllm: dict[str, Any] = {}
    fmha: dict[str, Any] = {}
    ok = True
    for rel in VLLM_SOURCES:
        installed = installed_root / rel.removeprefix("vllm/")
        checkout = vllm_root / rel
        installed_sha, checkout_sha = sha(installed), sha(checkout)
        match = installed_sha == checkout_sha
        vllm[rel] = {
            "installed_path": str(installed), "checkout_path": str(checkout),
            "installed_sha256": installed_sha, "checkout_sha256": checkout_sha,
            "match": match,
        }
        ok &= match
    for rel in FMHA_SOURCES:
        installed = installed_root / "third_party" / "fmha_sm100" / rel
        source = msa_root / "python" / "fmha_sm100" / rel
        installed_sha, source_sha = sha(installed), sha(source)
        match = installed_sha == source_sha
        fmha[rel] = {
            "installed_path": str(installed), "msa_source_path": str(source),
            "installed_sha256": installed_sha, "msa_source_sha256": source_sha,
            "match": match,
        }
        ok &= match
    installed_fmha_root = installed_root / "third_party" / "fmha_sm100"
    msa_fmha_root = msa_root / "python" / "fmha_sm100"
    installed_tree = _tree_map(installed_fmha_root)
    wheel_tree = _wheel_fmha_map(wheel)
    installed_non_cutlass = _tree_map(installed_fmha_root, skip_cutlass=True)
    msa_non_cutlass = _tree_map(msa_fmha_root, skip_cutlass=True)
    tree_checks = {
        "installed_matches_wheel_full_cmake_tree": installed_tree == wheel_tree,
        "installed_non_cutlass_matches_msa_full_source_tree": installed_non_cutlass == msa_non_cutlass,
        "cutlass_submodule_pin": git(msa_root, "ls-tree", "HEAD", "python/fmha_sm100/cutlass").split()[2] == CUTLASS_COMMIT,
    }
    ok &= all(tree_checks.values())
    trees = {
        "installed_fmha_full_tree": {"file_count": len(installed_tree), "sha256": _map_digest(installed_tree)},
        "wheel_fmha_full_tree": {"file_count": len(wheel_tree), "sha256": _map_digest(wheel_tree)},
        "installed_non_cutlass_tree": {"file_count": len(installed_non_cutlass), "sha256": _map_digest(installed_non_cutlass)},
        "msa_non_cutlass_tree": {"file_count": len(msa_non_cutlass), "sha256": _map_digest(msa_non_cutlass)},
        "checks": tree_checks,
    }
    return {"installed_vllm_key_sources": vllm, "installed_fmha_key_sources": fmha, "fmha_full_tree_audit": trees}, bool(ok)


def verify_pins(args: argparse.Namespace) -> dict[str, Any]:
    import vllm

    installed_root = Path(vllm.__file__).resolve().parent
    try:
        dist = importlib.metadata.distribution("vllm")
        installed_version = dist.version
        dist_root = Path(dist.locate_file("vllm")).resolve()
    except importlib.metadata.PackageNotFoundError:
        installed_version, dist_root = getattr(vllm, "__version__", "<missing>"), None
    hashes, source_match = source_audit(installed_root, args.vllm_root, args.msa_root, args.wheel)
    checks = {
        "vllm_git_head": git(args.vllm_root, "rev-parse", "HEAD") == VLLM_COMMIT,
        "msa_git_head": git(args.msa_root, "rev-parse", "HEAD") == MSA_COMMIT,
        "vllm_git_clean": not git(args.vllm_root, "status", "--porcelain"),
        "msa_git_clean": not git(args.msa_root, "status", "--porcelain"),
        "wheel_sha256": sha(args.wheel) == WHEEL_SHA256,
        "installed_version": VERSION_FRAGMENT in installed_version,
        "distribution_matches_import": dist_root == installed_root,
        "import_is_not_checkout": installed_root != (args.vllm_root / "vllm").resolve(),
        "installed_three_vllm_and_fmha_sources_match": source_match,
    }
    result = {
        "expected": {
            "vllm_commit": VLLM_COMMIT, "msa_commit": MSA_COMMIT,
            "cutlass_submodule_commit": CUTLASS_COMMIT,
            "wheel_sha256": WHEEL_SHA256, "version_fragment": VERSION_FRAGMENT,
        },
        "actual": {
            "vllm_root": str(args.vllm_root),
            "vllm_git_head": git(args.vllm_root, "rev-parse", "HEAD"),
            "msa_root": str(args.msa_root),
            "msa_git_head": git(args.msa_root, "rev-parse", "HEAD"),
            "wheel": str(args.wheel), "wheel_sha256": sha(args.wheel),
            "installed_vllm_root": str(installed_root),
            "installed_version": installed_version,
            "distribution_vllm_root": str(dist_root) if dist_root else None,
        },
        "source_hashes": hashes, "checks": checks, "pass": all(checks.values()),
    }
    if not result["pass"]:
        raise RuntimeError(f"strict pin/source checks failed: {checks}")
    return result


def config(backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(
            num_attention_heads=64, sparse_attention_config={"sparse_topk_blocks": 16}
        )),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, decode_context_parallel_size=1),
        cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"),
        attention_config=SimpleNamespace(minimax_m3_msa_decode_backend=backend),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=BATCH),
        speculative_config=None,
    )


def common_metadata(problem: Any, common_cls: type[Any]) -> Any:
    starts = torch.arange(BATCH + 1, device="cuda", dtype=torch.int32)
    return common_cls(
        query_start_loc=starts, query_start_loc_cpu=starts.cpu(),
        seq_lens=problem.seq_lens, num_reqs=BATCH, num_actual_tokens=BATCH,
        max_query_len=1, max_seq_len=4096, block_table_tensor=problem.block_table,
        slot_mapping=torch.full((BATCH,), -1, device="cuda", dtype=torch.int64),
        seq_lens_cpu_upper_bound=problem.seq_lens.cpu().contiguous(),
    )


def check_output(actual: torch.Tensor, oracle: torch.Tensor) -> dict[str, Any]:
    diff = (actual.float() - oracle.float()).abs()
    finite = bool(torch.isfinite(actual).all().item())
    close = bool(torch.isclose(actual.float(), oracle.float(), rtol=RTOL, atol=ATOL).all().item())
    return {
        "reference": "harness.reference.dense_sparse_attention_reference (independent FP32 selected-page causal oracle)",
        "rtol": RTOL, "atol": ATOL, "finite": finite,
        "max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item()),
        "pass": finite and close,
    }


def timing_abba(cutlass: Callable[[], torch.Tensor], triton: Callable[[], torch.Tensor], warmup: int, repetitions: int) -> dict[str, Any]:
    for _ in range(warmup):
        cutlass()
        triton()
    torch.cuda.synchronize()
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    calls = (("cutlass_a", cutlass), ("triton_b", triton), ("triton_b", triton), ("cutlass_a", cutlass))
    for _ in range(repetitions):
        for name, call in calls:
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            stop.record()
            events.append((name, start, stop))
    torch.cuda.synchronize()
    cycles: list[dict[str, float]] = []
    cutlass_samples: list[float] = []
    triton_samples: list[float] = []
    for cycle in range(repetitions):
        row: dict[str, float] = {}
        for position, (name, start, stop) in enumerate(events[cycle * 4 : cycle * 4 + 4]):
            elapsed = float(start.elapsed_time(stop))
            row[f"{name}_{position}"] = elapsed
            (cutlass_samples if name.startswith("cutlass") else triton_samples).append(elapsed)
        cycles.append(row)

    def summary(samples: list[float]) -> dict[str, Any]:
        ordered = sorted(samples)
        # Nearest-rank percentiles: rank=ceil(p*n), expressed as a 0-based index.
        lo = max(0, math.ceil(.10 * len(ordered)) - 1)
        hi = min(len(ordered) - 1, math.ceil(.90 * len(ordered)) - 1)
        median = float(statistics.median(ordered))
        return {
            "raw_ms": samples, "sample_count": len(samples),
            "p10_ms": ordered[lo], "median_ms": median, "p90_ms": ordered[hi],
            "p10_us": ordered[lo] * 1000., "median_us": median * 1000.,
            "p90_us": ordered[hi] * 1000.,
        }

    a, b = summary(cutlass_samples), summary(triton_samples)
    return {
        "protocol": "warmup excluded; ABBA=CUTLASS,TRITON,TRITON,CUTLASS; one default stream; CUDA events; no inter-call synchronization",
        "warmup": warmup, "repetitions": repetitions, "raw_cycles_ms": cycles,
        "cutlass": a, "triton": b,
        "triton_over_cutlass_median_speedup": b["median_ms"] / a["median_ms"],
        "pass": len(a["raw_ms"]) == 2 * repetitions and len(b["raw_ms"]) == 2 * repetitions
        and a["median_ms"] > 0 and b["median_ms"] > 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    cap = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name()
    if cap != (10, 3) or "B300" not in device_name.upper():
        raise RuntimeError(
            f"requires B300 with exact compute capability (10, 3); "
            f"got name={device_name!r}, capability={cap}"
        )
    pins = verify_pins(args)

    # These are all imports from the installed, pin-validated real vLLM package.
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.minimax_m3.nvidia import sparse_attention_msa as msa_module
    from vllm.models.minimax_m3.nvidia.sparse_attention_msa import MiniMaxM3SparseMSAImpl, MiniMaxM3SparseMSAMetadataBuilder
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode
    from harness.data import make_decode_problem
    from harness.reference import dense_sparse_attention_reference

    raw_problem = make_decode_problem(
        batch_size=BATCH, device="cuda", storage_dtype="fp8-scalar",
        seed=args.seed, decode_query_len=1, max_seq_len=4096,
    )
    problem = replace(raw_problem, topk_idx=torch.sort(raw_problem.topk_idx, dim=-1).values.contiguous())
    if not bool((problem.topk_idx[..., 1:] > problem.topk_idx[..., :-1]).all().item()):
        raise RuntimeError("top-k must be strict ascending after its one shared sort")
    if problem.k_scale is None or problem.v_scale is None or problem.k_scale.numel() != 1 or problem.v_scale.numel() != 1:
        raise RuntimeError("requires scalar FP8 K/V scales")

    spec = AttentionSpec(
        block_size=128, num_kv_heads=4, head_size=128, dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    common = common_metadata(problem, CommonAttentionMetadata)
    cutlass_builder = MiniMaxM3SparseMSAMetadataBuilder(spec, ["cutlass_layer"], config("cutlass"), torch.device("cuda"))
    triton_builder = MiniMaxM3SparseMSAMetadataBuilder(spec, ["triton_layer"], config("triton"), torch.device("cuda"))
    cutlass_first = cutlass_builder.build(0, common)
    cutlass_metadata = cutlass_builder.build(0, common)  # exact cache-reuse probe
    triton_metadata = triton_builder.build(0, common)
    first = cutlass_first.decode.msa_cutlass if cutlass_first.decode else None
    cutlass_md = cutlass_metadata.decode.msa_cutlass if cutlass_metadata.decode else None
    triton_md = triton_metadata.decode.msa_cutlass if triton_metadata.decode else None
    builder_checks = {
        "cutlass_metadata_nonempty": cutlass_md is not None,
        "cutlass_plan_cache_reused": cutlass_md is not None and first is not None
        and cutlass_md.plan is first.plan and len(cutlass_builder.msa_cutlass_plan_cache.plans) == 1,
        "triton_metadata_empty": triton_md is None,
        "triton_plan_cache_empty": len(triton_builder.msa_cutlass_plan_cache.plans) == 0,
    }
    if not all(builder_checks.values()):
        raise RuntimeError(f"real builder/build contract failed: {builder_checks}")

    query_fp8 = (problem.q.float() / Q_SCALE).to(torch.float8_e4m3fn).contiguous()
    token_major_topk = problem.topk_idx.permute(1, 0, 2).contiguous()
    k_scale, v_scale = float(problem.k_scale.item()), float(problem.v_scale.item())

    def layer(name: str) -> SimpleNamespace:
        return SimpleNamespace(
            layer_name=name, topk_indices_buffer=token_major_topk,
            _q_scale=torch.tensor(Q_SCALE, device="cuda"), _k_scale=problem.k_scale, _v_scale=problem.v_scale,
            _q_scale_float=Q_SCALE, _k_scale_float=k_scale, _v_scale_float=v_scale,
        )

    cutlass_layer, triton_layer = layer("cutlass_layer"), layer("triton_layer")
    cutlass_impl = MiniMaxM3SparseMSAImpl(
        64, 128, problem.sm_scale, 4, "fp8_e4m3", topk_blocks=16,
        sparse_block_size=128, msa_decode_backend="cutlass",
    )
    triton_impl = MiniMaxM3SparseMSAImpl(
        64, 128, problem.sm_scale, 4, "fp8_e4m3", topk_blocks=16,
        sparse_block_size=128, msa_decode_backend="triton",
    )
    cutlass_ctx = ForwardContext(no_compile_layers={}, attn_metadata={"cutlass_layer": cutlass_metadata}, slot_mapping={})
    triton_ctx = ForwardContext(no_compile_layers={}, attn_metadata={"triton_layer": triton_metadata}, slot_mapping={})
    cutlass_out, triton_out = torch.empty_like(problem.q), torch.empty_like(problem.q)
    cutlass_ptr, triton_ptr = cutlass_out.data_ptr(), triton_out.data_ptr()

    def call(impl: Any, one_layer: Any, context: Any, output: torch.Tensor) -> torch.Tensor:
        with override_forward_context(context):
            result = impl.forward(one_layer, problem.q, problem.kv_cache, output, query_fp8=query_fp8)
        if result.data_ptr() != output.data_ptr():
            raise RuntimeError("impl.forward replaced caller-owned output")
        return result

    cutlass_call = lambda: call(cutlass_impl, cutlass_layer, cutlass_ctx, cutlass_out)
    triton_call = lambda: call(triton_impl, triton_layer, triton_ctx, triton_out)

    # Oracle / quantization / metadata are complete before every event measurement.
    oracle = dense_sparse_attention_reference(problem)
    cutlass_actual, triton_actual = cutlass_call(), triton_call()
    torch.cuda.synchronize()
    correctness = {
        "cutlass": check_output(cutlass_actual, oracle),
        "triton": check_output(triton_actual, oracle),
        "caller_owned_output": {
            "cutlass": cutlass_actual.data_ptr() == cutlass_ptr,
            "triton": triton_actual.data_ptr() == triton_ptr,
        },
    }
    correctness["pass"] = bool(correctness["cutlass"]["pass"] and correctness["triton"]["pass"]
                               and all(correctness["caller_owned_output"].values()))

    # The temporary global wrappers prove the actual branch, then are restored.
    counts = {"cutlass": 0, "triton": 0}
    original_cutlass, original_triton = msa_module.msa_cutlass_sparse_decode, msa_module.minimax_m3_sparse_attn_decode

    def wrapped_cutlass(*call_args: Any, **kwargs: Any) -> Any:
        counts["cutlass"] += 1
        return original_cutlass(*call_args, **kwargs)

    def wrapped_triton(*call_args: Any, **kwargs: Any) -> Any:
        counts["triton"] += 1
        return original_triton(*call_args, **kwargs)

    try:
        msa_module.msa_cutlass_sparse_decode, msa_module.minimax_m3_sparse_attn_decode = wrapped_cutlass, wrapped_triton
        with override_forward_context(cutlass_ctx):
            cutlass_should = cutlass_impl.should_use_msa_decode("cutlass_layer")
        with override_forward_context(triton_ctx):
            triton_should = triton_impl.should_use_msa_decode("triton_layer")
        cutlass_call()
        cutlass_counts = dict(counts)
        counts = {"cutlass": 0, "triton": 0}
        triton_call()
        triton_counts = dict(counts)
    finally:
        msa_module.msa_cutlass_sparse_decode, msa_module.minimax_m3_sparse_attn_decode = original_cutlass, original_triton
    dispatch_checks = {
        "cutlass_impl_should_use_msa_decode": cutlass_should is True,
        "triton_impl_should_use_msa_decode": triton_should is False,
        "cutlass_smoke_calls_only_cutlass": cutlass_counts == {"cutlass": 1, "triton": 0},
        "triton_smoke_calls_only_triton": triton_counts == {"cutlass": 0, "triton": 1},
    }
    dispatch = {
        "cutlass_smoke_counts": cutlass_counts, "triton_smoke_counts": triton_counts,
        "checks": dispatch_checks, "pass": all(dispatch_checks.values()),
    }

    timing: dict[str, Any] = {"pass": False, "skipped": True, "reason": "correctness or dispatch failed"}
    if correctness["pass"] and dispatch["pass"]:
        timing = timing_abba(cutlass_call, triton_call, args.warmup, args.repetitions)
        timing["skipped"] = False

    gates = {
        "pin_and_source": pins["pass"], "real_builder_and_plan_reuse": all(builder_checks.values()),
        "correctness": correctness["pass"], "dispatch": dispatch["pass"], "timing": timing["pass"],
    }
    return {
        "schema": "c2-full-vllm-d4-backend-layer-abba-v1",
        "boundary": "exact full-vLLM backend layer path; no model weights/server scheduler",
        "all_gates_pass": all(gates.values()), "gates": gates, "source_pins": pins,
        "environment": {"torch": torch.__version__, "device": device_name, "compute_capability": list(cap)},
        "data_contract": {
            "generator": "make_decode_problem(B=16, storage_dtype='fp8-scalar', qlen=1, max_seq_len=4096)",
            "same_input_for_both_paths": True, "topk_sorted_once_and_shared": True,
            "q_dtype": "bfloat16", "query_fp8_dtype": "float8_e4m3fn",
            "q_scale": Q_SCALE, "k_scale": k_scale, "v_scale": v_scale,
            "page_size": 128, "topk": 16, "q_heads": 64, "kv_heads": 4, "head_dim": 128,
            "checksums": {
                "q_bf16": tensor_sha(problem.q), "query_fp8": tensor_sha(query_fp8),
                "kv_fp8": tensor_sha(problem.kv_cache), "block_table_i32": tensor_sha(problem.block_table),
                "topk_hkv_q_k_sorted_i32": tensor_sha(problem.topk_idx),
                "topk_q_hkv_k_i32": tensor_sha(token_major_topk), "seq_lens_i32": tensor_sha(problem.seq_lens),
            },
        },
        "real_vllm_api": {
            "builder": "MiniMaxM3SparseMSAMetadataBuilder.build",
            "metadata": "CommonAttentionMetadata",
            "kv_spec": "AttentionSpec(dtype=torch.uint8, kv_quant_mode=KVQuantMode.FP8_PER_TENSOR)",
            "forward_context": "ForwardContext + override_forward_context",
            "forward": "MiniMaxM3SparseMSAImpl.forward",
            "minimal_shell": "SimpleNamespace only for fields read by installed config/layer APIs",
        },
        "builder": {
            "checks": builder_checks,
            "cutlass_plan_cache_size": len(cutlass_builder.msa_cutlass_plan_cache.plans),
            "triton_plan_cache_size": len(triton_builder.msa_cutlass_plan_cache.plans),
            "pass": all(builder_checks.values()),
        },
        "correctness": correctness, "dispatch": dispatch, "timing": timing,
        "timing_excluded": ["metadata/build", "MSA plan", "FP32 oracle", "query FP8 quantization", "correctness checks"],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-root", required=True, type=Path)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--msa-root", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=positive, default=5)
    parser.add_argument("--repetitions", type=positive, default=50)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    for field in ("c2_root", "vllm_root", "msa_root", "wheel", "output"):
        setattr(args, field, getattr(args, field).resolve())
    if str(args.c2_root) not in sys.path:
        sys.path.insert(0, str(args.c2_root))
    try:
        result = run(args)
    except Exception as error:
        result = {
            "schema": "c2-full-vllm-d4-backend-layer-abba-v1",
            "boundary": "exact full-vLLM backend layer path; no model weights/server scheduler",
            "all_gates_pass": False,
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("all_gates_pass") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
