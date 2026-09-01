"""Direct official MiniMax-AI/MSA SM100 sparse-decode correctness/latency adapter.

Variable table (the fixed acceptance shape is intentionally explicit):

| Symbol | Meaning | Value |
| --- | --- | --- |
| B | request batch size | ``--batch`` in {1,4,8,16}, default 16 |
| Q | decode query length per request | 1 |
| Hq / Hkv | query / KV heads | 64 / 4 |
| G=Hq/Hkv | GQA group size | 16 |
| D | head dimension | 128 |
| P | physical KV page size | 128 tokens |
| K | selected logical pages per (query, KV head) | 16 |
| L | full KV length per request | CLI ``--kv-len`` (default 8192) |

The independent oracle gathers the *logical* pages selected by
``kv_block_indexes`` and computes QK, causal masking, softmax, and PV in FP32.
It uses the already-quantized FP8 values, isolating kernel error from FP8
quantization error.  Pass requires elementwise ``rtol=0.02, atol=0.12`` and
global cosine similarity >= 0.999.  The 0.12 absolute bound follows the
official commit's FP8 sparse regression threshold (max-diff < 0.11) with a
small reporting margin.

Static API assumption, verified against commit 80434d7: the root
``fmha_sm100_plan`` selects the sparse decode schedule for short Q (Q<=32)
when ``kv_block_num`` is set.  This adapter deliberately calls the same root
API and arguments as ``benchmarks/bench_sparse_attention_ops.py::bench_sparse``
instead of importing vLLM shims.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess

import torch

import fmha_sm100
from fmha_sm100 import fmha_sm100 as run_fmha_sm100
from fmha_sm100 import fmha_sm100_plan


REPOSITORY = "MiniMax-AI/MSA"
EXPECTED_COMMIT = "80434d7f67877c6570ca19cac444b84bc9855dac"
DEFAULT_BATCH = 16
QUERY_LEN = 1
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 128
TOPK = 16
RTOL = 0.02
ATOL = 0.12
MIN_COSINE = 0.999


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _source_pin() -> dict[str, object]:
    module_path = Path(fmha_sm100.__file__).resolve()
    repo_root = next((p for p in module_path.parents if (p / ".git").exists()), None)
    actual = None
    if repo_root is not None:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        actual = completed.stdout.strip()
    verified = actual == EXPECTED_COMMIT
    if not verified:
        raise RuntimeError(
            f"official MSA source pin not verified: expected={EXPECTED_COMMIT}, actual={actual}"
        )
    return {
        "repository": REPOSITORY,
        "expected_commit": EXPECTED_COMMIT,
        "actual_commit": actual,
        "pin_verified": verified,
        "module_path": str(module_path),
        "repository_root": str(repo_root),
    }


def _make_inputs(kv_len: int, seed: int, batch: int) -> dict[str, object]:
    if kv_len % PAGE_SIZE:
        raise ValueError(f"kv_len must be divisible by page_size={PAGE_SIZE}")
    pages_per_request = kv_len // PAGE_SIZE
    if pages_per_request < TOPK:
        raise ValueError(f"kv_len must contain at least topk={TOPK} pages")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    fp8 = torch.float8_e4m3fn
    total_pages = batch * pages_per_request

    # Match the official benchmark: initialize in FP16, then quantize Q/K/V to FP8.
    q = torch.randn(
        batch * QUERY_LEN,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.float16,
        device=device,
    ).to(fp8)
    k_logical = torch.randn(
        total_pages,
        NUM_KV_HEADS,
        PAGE_SIZE,
        HEAD_DIM,
        dtype=torch.float16,
        device=device,
    ).to(fp8)
    v_logical = torch.randn_like(k_logical, dtype=torch.float16).to(fp8)

    # Logical page i is stored at physical page perm[i].  The padded 2-D page
    # table follows the official bench_sparse ABI exactly.
    permutation = torch.randperm(total_pages, dtype=torch.int64, device=device)
    k_physical = torch.empty_like(k_logical)
    v_physical = torch.empty_like(v_logical)
    k_physical[permutation] = k_logical
    v_physical[permutation] = v_logical
    page_table_stride = ((pages_per_request + 3) // 4) * 4
    kv_indices = torch.zeros(
        batch, page_table_stride, dtype=torch.int32, device=device
    )
    for request in range(batch):
        begin = request * pages_per_request
        end = begin + pages_per_request
        kv_indices[request, :pages_per_request] = permutation[begin:end].to(torch.int32)

    # Official sparse ABI requires ascending logical page IDs and -1 tail
    # padding.  Every row is independently random to exercise both indirections.
    kv_block_indexes = torch.full(
        (batch * QUERY_LEN, NUM_KV_HEADS, TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    selected_cpu: list[list[list[int]]] = []
    for request in range(batch):
        per_head: list[list[int]] = []
        for _ in range(NUM_KV_HEADS):
            selected = torch.randperm(
                pages_per_request, generator=cpu_generator
            )[:TOPK].sort().values.tolist()
            per_head.append(selected)
        selected_cpu.append(per_head)
    for request, per_head in enumerate(selected_cpu):
        for kv_head, selected in enumerate(per_head):
            kv_block_indexes[request, kv_head] = torch.tensor(
                selected, dtype=torch.int32, device=device
            )

    return {
        "q": q,
        "k_logical": k_logical,
        "v_logical": v_logical,
        "k_physical": k_physical,
        "v_physical": v_physical,
        "kv_indices": kv_indices,
        "kv_block_indexes": kv_block_indexes,
        "selected_cpu": selected_cpu,
        "pages_per_request": pages_per_request,
    }


@torch.inference_mode()
def _fp32_reference(
    inputs: dict[str, object], kv_len: int, batch: int
) -> torch.Tensor:
    q = inputs["q"]
    k_logical = inputs["k_logical"]
    v_logical = inputs["v_logical"]
    selected_cpu = inputs["selected_cpu"]
    pages_per_request = int(inputs["pages_per_request"])
    assert isinstance(q, torch.Tensor)
    assert isinstance(k_logical, torch.Tensor)
    assert isinstance(v_logical, torch.Tensor)
    assert isinstance(selected_cpu, list)

    output = torch.empty(
        batch * QUERY_LEN,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.float32,
        device=q.device,
    )
    gqa_group = NUM_Q_HEADS // NUM_KV_HEADS
    token_offset = torch.arange(PAGE_SIZE, device=q.device)
    query_position = kv_len - QUERY_LEN
    for request in range(batch):
        for kv_head in range(NUM_KV_HEADS):
            logical_blocks = torch.tensor(
                selected_cpu[request][kv_head], dtype=torch.int64, device=q.device
            )
            global_pages = request * pages_per_request + logical_blocks
            keys = k_logical[global_pages, kv_head].reshape(-1, HEAD_DIM).float()
            values = v_logical[global_pages, kv_head].reshape(-1, HEAD_DIM).float()
            positions = (
                logical_blocks[:, None] * PAGE_SIZE + token_offset[None, :]
            ).reshape(-1)
            visible = positions <= query_position
            head_begin = kv_head * gqa_group
            head_end = head_begin + gqa_group
            query = q[request, head_begin:head_end].float()
            scores = (query @ keys.T) * (HEAD_DIM**-0.5)
            scores.masked_fill_(~visible[None, :], float("-inf"))
            probability = torch.softmax(scores, dim=-1)
            output[request, head_begin:head_end] = probability @ values
    return output


def _median_cuda_ms(function, warmup: int, repetitions: int) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repetitions)
    ]
    for start, end in pairs:
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    samples = sorted(start.elapsed_time(end) for start, end in pairs)
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": samples[max(0, int(repetitions * 0.10) - 1)],
        "p90_ms": samples[min(repetitions - 1, int(repetitions * 0.90))],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(1, 4, 8, 16), default=DEFAULT_BATCH)
    parser.add_argument("--kv-len", type=_positive_int, default=8192)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--warmup", type=_positive_int, default=20)
    parser.add_argument("--repetitions", type=_positive_int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("official SM100 benchmark requires a CUDA GPU")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        raise RuntimeError(f"official fmha_sm100 requires SM100-family GPU, got {capability}")
    source = _source_pin()
    inputs = _make_inputs(args.kv_len, args.seed, args.batch)
    q = inputs["q"]
    k_physical = inputs["k_physical"]
    v_physical = inputs["v_physical"]
    kv_indices = inputs["kv_indices"]
    kv_block_indexes = inputs["kv_block_indexes"]
    assert all(isinstance(x, torch.Tensor) for x in (q, k_physical, v_physical, kv_indices, kv_block_indexes))

    qo_lens = torch.full((args.batch,), QUERY_LEN, dtype=torch.int32)
    kv_lens = torch.full((args.batch,), args.kv_len, dtype=torch.int32)
    qo_offset = torch.full(
        (args.batch,), args.kv_len - QUERY_LEN, dtype=torch.int32
    )
    with contextlib.redirect_stdout(io.StringIO()):
        plan = fmha_sm100_plan(
            qo_lens,
            kv_lens,
            NUM_Q_HEADS,
            qo_offset=qo_offset,
            split_prefill_decode=False,
            causal=True,
            page_size=PAGE_SIZE,
            kv_block_num=TOPK,
            num_kv_heads=NUM_KV_HEADS,
        )
    output = torch.empty(
        args.batch * QUERY_LEN,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )

    def execute() -> torch.Tensor:
        result, _ = run_fmha_sm100(
            q,
            k_physical,
            v_physical,
            plan_info=plan,
            kv_indices=kv_indices,
            kv_block_indexes=kv_block_indexes,
            out=output,
            sm_scale=HEAD_DIM**-0.5,
        )
        return result

    reference = _fp32_reference(inputs, args.kv_len, args.batch)
    with contextlib.redirect_stdout(io.StringIO()):
        actual = execute()
    torch.cuda.synchronize()
    if actual.data_ptr() != output.data_ptr():
        raise RuntimeError("official API did not return the caller-owned output")
    difference = (actual.float() - reference).abs()
    max_abs = float(difference.max())
    mean_abs = float(difference.mean())
    elementwise_pass = bool(
        torch.isclose(actual.float(), reference, rtol=RTOL, atol=ATOL).all()
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual.float().reshape(-1), reference.reshape(-1), dim=0
        )
    )
    passed = elementwise_pass and max_abs <= ATOL and cosine >= MIN_COSINE
    timing = None
    if passed:
        with contextlib.redirect_stdout(io.StringIO()):
            timing = _median_cuda_ms(execute, args.warmup, args.repetitions)

    payload = {
        "source": source,
        "environment": {
            "torch": str(torch.__version__),
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(capability),
        },
        "shape": {
            "batch": args.batch,
            "query_len": QUERY_LEN,
            "kv_len": args.kv_len,
            "q_heads": NUM_Q_HEADS,
            "kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "page_size": PAGE_SIZE,
            "topk": TOPK,
            "qkv_dtype": "float8_e4m3fn",
            "output_dtype": "bfloat16",
            "random_physical_page_table": True,
            "random_per_request_head_topk": True,
        },
        "correctness": {
            "reference": "independent_fp32_selected_logical_pages",
            "rtol": RTOL,
            "atol": ATOL,
            "minimum_cosine": MIN_COSINE,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "cosine": cosine,
            "pass": passed,
        },
        "timing": (
            {
                "protocol": "warmup_then_per_call_cuda_events_queued_on_one_stream",
                "warmup": args.warmup,
                "repetitions": args.repetitions,
                **timing,
            }
            if timing is not None
            else {"valid": False, "reason": "correctness gate failed"}
        ),
        "assumptions": [
            "commit 80434d7 root planner maps sparse Q<=32 to decode schedule",
            "kv_block_indexes are sorted logical page IDs; kv_indices maps logical to physical pages",
            "FP8 reference begins from the exact quantized Q/K/V tensors",
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
