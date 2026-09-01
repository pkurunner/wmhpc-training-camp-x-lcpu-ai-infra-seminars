"""BF16 fair crossover: same paged data for vendored Triton and official MSA.

| Symbol | Meaning | Value |
| --- | --- | --- |
| B | decode batch | 1, 4, 8, 16 |
| Hq/Hkv/D | query/KV heads/head dimension | 64/4/128 |
| P/K | page tokens/selected logical pages | 128/16 |
| S_b | request b causal sequence length | random [2048,4096] |
| l[b,h,r] | sorted selected logical page | random, unique |
| p[b,l] | logical-to-physical page table | random permutation |

The sole input is make_decode_problem(..., BF16, seed+B).  Its random top-k
is sorted exactly once for the official ascending-index ABI, then that same
DecodeProblem is used by both paths. K/V bridge and MSA planning are outside
all event ranges. The source wrapper, the selected persistent-workspace C=1
optimization, and the official core each use caller-owned BF16 output and gate
against one independent FP32 selected-page causal oracle before timing.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any, Callable

import torch

from harness.data import DecodeProblem, PAGE_SIZE, make_decode_problem
from harness.triton_baseline import run_triton_baseline_into
from challenge.prepared_decode import PreparedSparseDecode

MSA_PIN = "80434d7f67877c6570ca19cac444b84bc9855dac"
CUTLASS_PIN = "eb61c911471867a5fd2466bfd8f29306cea6ebf8"
VLLM_PIN = "d4da0c5"
BATCHES = (1, 4, 8, 16)
RTOL = ATOL = 0.03


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _git(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_tensor(tensor: torch.Tensor) -> str:
    # CPU materialization is intentionally outside correctness/timing ranges.
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _pins(msa_root: Path, c2_root: Path) -> dict[str, Any]:
    msa = _git("git", "-C", str(msa_root), "rev-parse", "HEAD")
    if msa != MSA_PIN:
        raise RuntimeError(f"MSA pin mismatch: expected={MSA_PIN}, actual={msa}")
    submodules = _git("git", "-C", str(msa_root), "submodule", "status", "--recursive")
    if CUTLASS_PIN not in submodules:
        raise RuntimeError(f"CUTLASS pin {CUTLASS_PIN} absent from submodule status")
    if VLLM_PIN not in (c2_root / "TASK.md").read_text(encoding="utf-8"):
        raise RuntimeError(f"TASK.md does not declare vendored vLLM pin {VLLM_PIN}")
    return {
        "official_msa": {"expected_commit": MSA_PIN, "actual_commit": msa, "pin_verified": True},
        "cutlass": {"expected_commit": CUTLASS_PIN, "pin_verified": True, "submodule_status": submodules.splitlines()},
        "vendored_vllm": {"declared_snapshot_pin": VLLM_PIN, "verification": "TASK.md declaration plus source checksums; snapshot is not a git checkout"},
    }


def _source_hashes(c2_root: Path) -> dict[str, str]:
    h = c2_root / "harness"
    paths = {
        "fair_crossover_bench.py": h / "fair_crossover_bench.py",
        "data.py": h / "data.py", "triton_baseline.py": h / "triton_baseline.py",
        "reference.py": h / "reference.py",
        "challenge/prepared_decode.py": c2_root / "challenge" / "prepared_decode.py",
        "vllm_msa_ref/sparse_attn.py": c2_root / "vllm_msa_ref" / "sparse_attn.py",
        "vllm_msa_ref/msa_cutlass_sparse_decode.py": c2_root / "vllm_msa_ref" / "msa_cutlass_sparse_decode.py",
    }
    return {name: _sha_file(path) for name, path in paths.items()}


def _sort_topk(problem: DecodeProblem) -> DecodeProblem:
    topk = torch.sort(problem.topk_idx, dim=-1).values.contiguous()
    if topk.shape[-1] > 1 and not bool((topk[..., 1:] > topk[..., :-1]).all().item()):
        raise RuntimeError("topk must be strictly increasing for official MSA")
    return replace(problem, topk_idx=topk)


@torch.inference_mode()
def _fp32_oracle(problem: DecodeProblem) -> torch.Tensor:
    """Independent selected-page causal attention; it calls neither candidate."""
    if problem.storage_dtype != "bf16" or problem.k_scale is not None or problem.v_scale is not None:
        raise ValueError("only unscaled BF16 is valid for this benchmark")
    q, kv = problem.q.float(), problem.kv_cache.float()
    batch, max_blocks = problem.block_table.shape
    group, d = q.shape[1] // problem.num_kv_heads, q.shape[-1]
    out = torch.empty_like(q, dtype=torch.float32)
    offset = torch.arange(PAGE_SIZE, device=q.device)
    for b in range(batch):
        length = int(problem.seq_lens[b].item())
        for h in range(problem.num_kv_heads):
            logical = problem.topk_idx[h, b].long()
            if int(logical.max().item()) >= max_blocks:
                raise ValueError("logical topk index exceeds table")
            physical = problem.block_table[b].index_select(0, logical).long()
            pages = kv.index_select(0, physical)[:, h]
            key, value = pages[..., :d].reshape(-1, d), pages[..., d:].reshape(-1, d)
            positions = (logical[:, None] * PAGE_SIZE + offset[None, :]).reshape(-1)
            scores = (q[b, h * group : (h + 1) * group] @ key.T) * problem.sm_scale
            scores.masked_fill_(positions[None, :] >= length, float("-inf"))
            out[b, h * group : (h + 1) * group] = torch.softmax(scores, dim=-1) @ value
    return out


def _correctness(actual: torch.Tensor, oracle: torch.Tensor) -> dict[str, Any]:
    diff = (actual.float() - oracle).abs()
    finite = bool(torch.isfinite(actual.float()).all().item())
    passed = finite and bool(torch.isclose(actual.float(), oracle, rtol=RTOL, atol=ATOL).all().item())
    return {"reference": "independent_fp32_selected_page_causal_attention", "rtol": RTOL, "atol": ATOL,
            "max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item()), "finite": finite, "pass": passed}


def _timing(call: Callable[[], torch.Tensor], warmup: int, repetitions: int) -> dict[str, Any]:
    for _ in range(warmup): call()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repetitions)]
    for begin, end in events:
        begin.record(); call(); end.record()
    torch.cuda.synchronize()
    samples = sorted(float(begin.elapsed_time(end)) for begin, end in events)
    p10, p90 = max(0, int(.1 * repetitions) - 1), min(repetitions - 1, int(.9 * repetitions))
    median = float(statistics.median(samples))
    return {"protocol": "warmup then per-call CUDA event pairs on one stream; no inter-call synchronization",
            "warmup": warmup, "repetitions": repetitions, "p10_ms": samples[p10], "median_ms": median, "p90_ms": samples[p90],
            "p10_us": samples[p10] * 1000, "median_us": median * 1000, "p90_us": samples[p90] * 1000}


def _checksums(problem: DecodeProblem, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, topk: torch.Tensor) -> dict[str, str]:
    return {"q_bf16": _sha_tensor(problem.q), "kv_cache_interleaved_bf16": _sha_tensor(problem.kv_cache),
            "block_table_i32": _sha_tensor(problem.block_table), "topk_sorted_i32_hkv_q_k": _sha_tensor(problem.topk_idx),
            "seq_lens_i32": _sha_tensor(problem.seq_lens), "official_k_bf16": _sha_tensor(k), "official_v_bf16": _sha_tensor(v),
            "official_indices_i32": _sha_tensor(indices), "official_topk_i32_q_hkv_k": _sha_tensor(topk)}


def _run_batch(args: argparse.Namespace, batch: int, plan_fn: Any, msa_fn: Any) -> dict[str, Any]:
    # This is the common data contract; no path is allowed a separately-generated input.
    problem = _sort_topk(make_decode_problem(batch_size=batch, device="cuda", storage_dtype="bf16",
        seed=args.seed + batch, decode_query_len=1, max_seq_len=args.max_seq_len))
    k = problem.kv_cache[..., :problem.head_dim].contiguous()
    v = problem.kv_cache[..., problem.head_dim:].contiguous()
    # The official paged ABI expects a two-dimensional [B, table_stride]
    # logical-to-physical map.  K=4096 gives 32 entries/request, already a
    # multiple of the official four-entry padding requirement.
    indices = problem.block_table.contiguous()
    topk = problem.topk_idx.permute(1, 0, 2).contiguous()
    torch.cuda.synchronize()  # finish layout bridge before all correctness/event ranges
    qo_lens = torch.ones(batch, dtype=torch.int32)
    kv_lens = problem.seq_lens.cpu().to(torch.int32).contiguous()
    qo_offset = (kv_lens - 1).contiguous()
    plan_args = {"num_qo_heads": problem.num_q_heads, "num_kv_heads": problem.num_kv_heads, "page_size": PAGE_SIZE,
                 "kv_block_num": 16, "causal": True, "split_prefill_decode": False, "qo_lens": qo_lens.tolist(),
                 "kv_lens_same_as_seq_lens": kv_lens.tolist(), "qo_offset": qo_offset.tolist(),
                 "input_scale": "none (BF16 Q/K/V)", "output_maxscore": False, "output_o": True}
    with contextlib.redirect_stdout(io.StringIO()):
        plan = plan_fn(qo_lens, kv_lens, problem.num_q_heads, qo_offset=qo_offset, split_prefill_decode=False,
                       causal=True, page_size=PAGE_SIZE, kv_block_num=16,
                       num_kv_heads=problem.num_kv_heads, output_maxscore=False)
    torch.cuda.synchronize()  # plan creation is excluded
    out_triton = torch.empty_like(problem.q)
    out_prepared = torch.empty_like(problem.q)
    out_official = torch.empty_like(problem.q)
    prepared = PreparedSparseDecode(problem, out_prepared, num_topk_chunks=1)
    def triton() -> torch.Tensor:
        result = run_triton_baseline_into(problem, out_triton)
        if result.data_ptr() != out_triton.data_ptr(): raise RuntimeError("Triton ignored caller-owned output")
        return result
    def official() -> torch.Tensor:
        with contextlib.redirect_stdout(io.StringIO()):
            result, _ = msa_fn(problem.q, k, v, plan_info=plan, kv_indices=indices, kv_block_indexes=topk,
                               out=out_official, output_maxscore=False, output_o=True,
                               sm_scale=problem.sm_scale)
        if result is None or result.data_ptr() != out_official.data_ptr(): raise RuntimeError("official MSA ignored caller-owned output")
        return result
    def prepared_triton() -> torch.Tensor:
        result = prepared()
        if result.data_ptr() != out_prepared.data_ptr(): raise RuntimeError("prepared Triton ignored caller-owned output")
        return result
    oracle = _fp32_oracle(problem)
    triton_actual, prepared_actual, official_actual = triton(), prepared_triton(), official()
    torch.cuda.synchronize()
    gate_t = _correctness(triton_actual, oracle)
    gate_p = _correctness(prepared_actual, oracle)
    gate_o = _correctness(official_actual, oracle)
    valid = bool(gate_t["pass"] and gate_p["pass"] and gate_o["pass"])
    latency: dict[str, Any] = {"valid": False, "reason": "common FP32 gate failed"}
    if valid:
        latency = {"valid": True, "vendored_vllm_triton_source_wrapper": _timing(triton, args.warmup, args.repetitions),
                   "prepared_triton_selected_c1": _timing(prepared_triton, args.warmup, args.repetitions),
                   "official_msa": _timing(official, args.warmup, args.repetitions)}
        latency["official_over_source_wrapper_median_speedup"] = latency["vendored_vllm_triton_source_wrapper"]["median_ms"] / latency["official_msa"]["median_ms"]
        latency["official_over_prepared_median_speedup"] = latency["prepared_triton_selected_c1"]["median_ms"] / latency["official_msa"]["median_ms"]
    return {"batch": batch, "problem": {"generator": "make_decode_problem", "seed": args.seed + batch, "dtype": "bfloat16", "scale": "none",
            "q_heads": problem.num_q_heads, "kv_heads": problem.num_kv_heads, "head_dim": problem.head_dim, "page_size": PAGE_SIZE, "topk": 16,
            "random_global_block_table": True, "random_topk_sorted_for_both": True, "seq_lens": kv_lens.tolist(),
            "checksums": _checksums(problem, k, v, indices, topk)},
            "layout_bridge": {"timing_excluded": True, "vllm": "[physical,Hkv,P,2D] K|V", "official": "K,V each [physical,Hkv,P,D] contiguous",
                              "conversion": "K=kv[..., :D].contiguous; V=kv[..., D:].contiguous; table remains 2D; topk [Hkv,Q,K]->[Q,Hkv,K]"},
            "plan": {"timing_excluded": True, "args": plan_args},
            "caller_owned_output": {"source_wrapper": True, "prepared_triton": True, "official_msa": True, "shape": list(problem.q.shape), "dtype": "bfloat16"},
            "workspace_contract": {"source_wrapper": "allocates o_partial/lse inside each call", "prepared_triton": "persistent workspace, selected BF16 C=1", "official_msa": "plan/workspace prepared outside timing"},
            "correctness": {"common_oracle": "independent FP32", "source_wrapper": gate_t, "prepared_triton": gate_p, "official_msa": gate_o, "all_pass": valid}, "latency": latency}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msa-root", required=True, type=Path); parser.add_argument("--c2-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260819); parser.add_argument("--max-seq-len", type=_positive, default=4096)
    parser.add_argument("--warmup", type=_positive, default=20); parser.add_argument("--repetitions", type=_positive, default=100)
    parser.add_argument("--batches", type=int, choices=BATCHES, nargs="+", default=BATCHES)
    args = parser.parse_args()
    if args.max_seq_len % PAGE_SIZE: raise ValueError("max-seq-len must be page aligned")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    cap = torch.cuda.get_device_capability()
    if cap[0] != 10: raise RuntimeError(f"official fmha_sm100 requires SM100-family GPU, got {cap}")
    c2_root, msa_root = args.c2_root.resolve(), args.msa_root.resolve()
    pins = _pins(msa_root, c2_root)
    import fmha_sm100
    from fmha_sm100 import fmha_sm100 as msa_fn, fmha_sm100_plan as plan_fn
    module = Path(fmha_sm100.__file__).resolve()
    try: module.relative_to(msa_root)
    except ValueError as error: raise RuntimeError(f"fmha_sm100 is not imported from --msa-root: {module}") from error
    pins["official_msa"]["module_path"] = str(module)
    result: dict[str, Any] = {"schema": "c2-fair-bf16-crossover-v2-three-path", "source_pins": pins, "source_hashes": _source_hashes(c2_root),
        "environment": {"torch": str(torch.__version__), "device": torch.cuda.get_device_name(), "compute_capability": list(cap)},
        "fairness_contract": {"same_data": "same BF16 make_decode_problem data, seed, random table, sorted topk and seq_lens", "oracle": "same independent FP32 oracle",
                               "tolerance": {"rtol": RTOL, "atol": ATOL}, "caller_owned_output": True,
                               "primary_boundary": "source wrapper vs official core have different internal workspace lifetimes; prepared Triton C=1 is reported separately",
                               "excluded": ["K/V bridge", "MSA plan", "oracle", "correctness"]}, "results": []}
    for batch in args.batches: result["results"].append(_run_batch(args, batch, plan_fn, msa_fn))
    result["all_correctness_pass"] = all(entry["correctness"]["all_pass"] for entry in result["results"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_correctness_pass"] else 2


if __name__ == "__main__": raise SystemExit(main())
