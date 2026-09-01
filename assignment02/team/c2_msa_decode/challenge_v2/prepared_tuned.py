"""Prepared sparse decode with explicit Triton compilation/launch options.

This file intentionally lives outside ``challenge/``.  It does not modify the
pinned kernel body or the established prepared implementation; it only binds
the same two JIT functions to caller-owned output and persistent workspace,
then exposes the supported Triton options needed for an evidence-backed sweep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from challenge.prepared_decode import (
    _BASELINE,
    _DECODE_KERNEL,
    _FP8_DTYPES,
    _KV_SCALE_ARGS,
    _MERGE_KERNEL,
    _validate_chunks,
    _validate_problem_and_output,
    _validate_warps,
    baseline_num_topk_chunks,
)
from harness.data import DecodeProblem


PdlMode = Literal["auto", "on", "off"]


def _validate_stages(name: str, stages: int) -> None:
    # Triton's NVIDIA backend requires a positive software-pipeline depth.  A
    # deliberately small upper bound keeps accidental exhaustive sweeps sane.
    if stages not in (1, 2, 3, 4, 5, 6):
        raise ValueError(f"{name} must be one of 1, 2, 3, 4, 5, 6")


def _validate_maxnreg(name: str, value: int | None) -> None:
    if value is None:
        return
    # NVIDIA PTX .maxnreg accepts a positive integer; keeping multiples of 8
    # gives an interpretable register-pressure scan and rejects typos.
    if value < 32 or value > 256 or value % 8:
        raise ValueError(f"{name} must be None or a multiple of 8 in [32, 256]")


@dataclass(frozen=True)
class TuningConfig:
    """A single JIT configuration, recorded verbatim in sweep evidence."""

    num_topk_chunks: int | None = None
    decode_num_warps: int = 4
    merge_num_warps: int = 4
    decode_num_stages: int = 3
    merge_num_stages: int = 3
    pdl_mode: PdlMode = "auto"
    decode_maxnreg: int | None = None
    merge_maxnreg: int | None = None

    def validate(self, max_topk: int) -> None:
        if self.num_topk_chunks is not None:
            _validate_chunks(self.num_topk_chunks, max_topk)
        _validate_warps("decode_num_warps", self.decode_num_warps)
        _validate_warps("merge_num_warps", self.merge_num_warps)
        _validate_stages("decode_num_stages", self.decode_num_stages)
        _validate_stages("merge_num_stages", self.merge_num_stages)
        if self.pdl_mode not in ("auto", "on", "off"):
            raise ValueError("pdl_mode must be auto, on, or off")
        _validate_maxnreg("decode_maxnreg", self.decode_maxnreg)
        _validate_maxnreg("merge_maxnreg", self.merge_maxnreg)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TunedPreparedMetadata:
    num_topk_chunks: int
    merge_bypassed: bool
    caller_output_bytes: int
    unique_workspace_bytes: int
    pdl_mode_requested: str
    pdl_effective: bool
    decode_num_warps: int
    merge_num_warps: int
    decode_num_stages: int
    merge_num_stages: int
    decode_maxnreg: int | None
    merge_maxnreg: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class TunedPreparedSparseDecode:
    """Same prepared execution contract, with explicit Triton tuning knobs.

    One instance is bound to one stable input/output address set, is not
    re-entrant, and has no allocation in ``__call__``.  Thus a timing harness
    can legitimately put only its decode kernel launches between CUDA events.
    """

    def __init__(
        self, problem: DecodeProblem, output: torch.Tensor, *, config: TuningConfig
    ) -> None:
        _validate_problem_and_output(problem, output)
        config.validate(int(problem.topk_idx.shape[-1]))
        total_q, num_heads, head_dim = problem.q.shape
        chunks = (
            baseline_num_topk_chunks(total_q, problem.num_kv_heads, int(problem.topk_idx.shape[-1]))
            if config.num_topk_chunks is None
            else config.num_topk_chunks
        )
        assert chunks is not None
        self.problem, self.output, self.config, self.num_topk_chunks = (
            problem,
            output,
            config,
            chunks,
        )
        if chunks == 1:
            self.o_partial = output.unsqueeze(0)
        else:
            self.o_partial = torch.empty(
                chunks, total_q, num_heads, head_dim, dtype=problem.q.dtype, device=problem.q.device
            )
        self.lse_partial = torch.empty(
            chunks, total_q, num_heads, dtype=torch.float32, device=problem.q.device
        )
        self.use_fp8 = problem.kv_cache.dtype in _FP8_DTYPES
        (
            self.k_scale_arg,
            self.v_scale_arg,
            self.stride_ks_h,
            self.stride_ks_t,
            self.stride_vs_h,
            self.stride_vs_t,
            self.kv_scale_mode,
        ) = (
            _KV_SCALE_ARGS(output, problem.num_kv_heads, problem.k_scale, problem.v_scale)
            if self.use_fp8
            else (output, output, 0, 0, 0, 0, 0)
        )
        platform_support = bool(_BASELINE["current_platform"].is_arch_support_pdl())
        if config.pdl_mode == "auto":
            self.use_pdl = platform_support
        elif config.pdl_mode == "on":
            # Do not silently turn a requested test into the auto variant.
            # Unsupported launch errors are captured as a failed candidate by
            # the sweep harness and retained in its JSON evidence.
            self.use_pdl = True
        else:
            self.use_pdl = False
        self.platform_supports_pdl = platform_support

    @property
    def metadata(self) -> TunedPreparedMetadata:
        unique_partial_bytes = 0 if self.num_topk_chunks == 1 else self.o_partial.numel() * self.o_partial.element_size()
        return TunedPreparedMetadata(
            num_topk_chunks=self.num_topk_chunks,
            merge_bypassed=self.num_topk_chunks == 1,
            caller_output_bytes=self.output.numel() * self.output.element_size(),
            unique_workspace_bytes=unique_partial_bytes + self.lse_partial.numel() * self.lse_partial.element_size(),
            pdl_mode_requested=self.config.pdl_mode,
            pdl_effective=self.use_pdl,
            decode_num_warps=self.config.decode_num_warps,
            merge_num_warps=self.config.merge_num_warps,
            decode_num_stages=self.config.decode_num_stages,
            merge_num_stages=self.config.merge_num_stages,
            decode_maxnreg=self.config.decode_maxnreg,
            merge_maxnreg=self.config.merge_maxnreg,
        )

    @torch.inference_mode()
    def __call__(self) -> torch.Tensor:
        p, q = self.problem, self.problem.q
        total_q, num_heads, head_dim = q.shape
        pdl_launch: dict[str, bool] = {"launch_pdl": True} if self.use_pdl else {}
        decode_options: dict[str, Any] = {
            "num_warps": self.config.decode_num_warps,
            "num_stages": self.config.decode_num_stages,
        }
        if self.config.decode_maxnreg is not None:
            decode_options["maxnreg"] = self.config.decode_maxnreg
        _DECODE_KERNEL[(total_q * self.num_topk_chunks, p.num_kv_heads)](
            q, p.kv_cache, self.k_scale_arg, self.v_scale_arg, p.topk_idx,
            self.o_partial, self.lse_partial, p.block_table, p.seq_lens,
            total_q, num_heads // p.num_kv_heads, head_dim, p.topk_idx.shape[-1],
            p.sm_scale, p.decode_query_len,
            q.stride(0), q.stride(1), q.stride(2),
            p.kv_cache.stride(0), p.kv_cache.stride(1), p.kv_cache.stride(2), p.kv_cache.stride(3),
            self.stride_ks_h, self.stride_ks_t, self.stride_vs_h, self.stride_vs_t,
            p.topk_idx.stride(0), p.topk_idx.stride(1), p.topk_idx.stride(2),
            self.o_partial.stride(0), self.o_partial.stride(1), self.o_partial.stride(2), self.o_partial.stride(3),
            self.lse_partial.stride(0), self.lse_partial.stride(1), self.lse_partial.stride(2),
            p.block_table.stride(0),
            BLOCK_SIZE_K=128, NUM_TOPK_CHUNKS=self.num_topk_chunks,
            USE_FP8=self.use_fp8, KV_SCALE_MODE=self.kv_scale_mode, USE_PDL=self.use_pdl,
            **decode_options, **pdl_launch,
        )
        if self.num_topk_chunks == 1:
            return self.output
        merge_options: dict[str, Any] = {
            "num_warps": self.config.merge_num_warps,
            "num_stages": self.config.merge_num_stages,
        }
        if self.config.merge_maxnreg is not None:
            merge_options["maxnreg"] = self.config.merge_maxnreg
        _MERGE_KERNEL[(total_q, num_heads)](
            self.o_partial, self.lse_partial, self.output, head_dim,
            self.o_partial.stride(0), self.o_partial.stride(1), self.o_partial.stride(2), self.o_partial.stride(3),
            self.lse_partial.stride(0), self.lse_partial.stride(1), self.lse_partial.stride(2),
            self.output.stride(0), self.output.stride(1), self.output.stride(2),
            NUM_TOPK_CHUNKS=self.num_topk_chunks, USE_PDL=self.use_pdl,
            **merge_options, **pdl_launch,
        )
        return self.output
