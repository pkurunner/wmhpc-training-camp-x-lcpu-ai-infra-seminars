"""Fail-closed B300 dispatcher for the separately audited FlashKDA variants.

The dispatcher deliberately contains a *small* whitelist.  A miss is not an
error: it selects the upstream ``flash_kda.fwd`` implementation.  The only
accelerated contracts at present are the exact B300 measurements made for
``K=V=128``: the exhaustive ``B=1,T=8192`` BF16 head sweep, the exact B=1
H=12 length/state matrix, the separately released ``T=8191`` tail cells, two
measured length/head interaction slices, a small per-contract fixed
``B>1,H=12,T=2048`` table, and only the two released packed-varlen
skew-layout public contracts.  Every entry outside the original matrix is
kept in an exact-shape table backed by separately hashed release evidence.
Do not broaden this module from a plausible-looking shape alone; every new
entry needs its own correctness and clean-allocation latency evidence.

There is intentionally no ``try: launch ... except: launch baseline`` path.
Once a symbol was selected, a kernel failure must remain visible to the
caller.  The only fallbacks happen *before* a launch, while checking extension
symbols.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib
import math
from pathlib import Path
import threading
from typing import Any, Callable, Mapping

try:
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import varlen_metadata
except ModuleNotFoundError:  # pragma: no cover - installed/standalone paths
    try:
        from team.c1_flashkda.challenge_tp8_dispatch import varlen_metadata  # type: ignore[no-redef]
    except ModuleNotFoundError:
        try:
            from challenge_tp8_dispatch import varlen_metadata  # type: ignore[no-redef]
        except ModuleNotFoundError:
            import varlen_metadata  # type: ignore[no-redef]


_BF16 = "bfloat16"
_FP32 = "float32"
_BASELINE = "baseline"
_VSHARD2_P2 = "vshard2_p2"
_VSHARD4_P2 = "vshard4_p2"
_AUDITED_EXTENSION_SHA256 = frozenset(
    {
        # Fresh SM103a build used by the state-contract and FLA public-path
        # audits.  A symbol-compatible binary is not sufficient: every
        # accelerated launch must remain bound to an audited artifact.
        "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005",
    }
)
_H12_LENGTHS = frozenset((128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536))
_INTERACTION_LENGTHS = frozenset((2048, 32768))
_INTERACTION_VARIANTS = {
    1: _VSHARD4_P2,
    12: _VSHARD4_P2,
    37: _VSHARD4_P2,
    38: _VSHARD2_P2,
    64: _VSHARD2_P2,
    96: _VSHARD2_P2,
}
_FIXED_BATCH_PUBLIC_VARIANTS = {
    (2, "none"): _VSHARD4_P2,
    (2, "fp32_final_only"): _VSHARD4_P2,
    (2, "fp32_both"): _VSHARD4_P2,
    (3, "none"): _VSHARD4_P2,
    (3, "fp32_final_only"): _VSHARD4_P2,
    (3, "fp32_both"): _VSHARD4_P2,
    (4, "none"): _VSHARD2_P2,
    (4, "fp32_final_only"): _VSHARD2_P2,
    (5, "none"): _VSHARD2_P2,
    (5, "fp32_final_only"): _VSHARD2_P2,
    (5, "fp32_both"): _VSHARD2_P2,
    (6, "none"): _VSHARD2_P2,
    (6, "fp32_final_only"): _VSHARD2_P2,
}
_FIXED_SINGLE_BATCH_PUBLIC_VARIANTS = {
    # Tail-8191 release: two clean B300 allocations, each with two fresh
    # processes and two 1000-sample repeats, passed the >=2% P50/P95/P99
    # public-call gate.  Keep this table separate from _H12_LENGTHS so the
    # unmeasured fp32_both/bf16_both contracts remain baseline.
    (8191, "none"): _VSHARD4_P2,
    (8191, "fp32_final_only"): _VSHARD4_P2,
}
_VARLEN_LAYOUT_NAMES = {
    (0, 2048, 4096): "equal_n2_h12_t4096",
    (0, 2048, 4096, 6144, 8192): "equal_n4_h12_t8192",
    (0, 17, 528, 1552, 2852, 4901, 8192): "mixed_n6_h12_t8192",
    (0, 1, 2, 3, 4, 5, 12288): "skew_n6_h12_t12288",
}
_VARLEN_PUBLIC_VARIANTS = {
    # Public-FLA r4 release intersection plus the separately preregistered
    # fp32-both extension.  The latter passed two new clean B300 allocations,
    # each with two fresh processes and the >=2% P50/P95/P99 gate.  Keep the
    # exact offsets/state key: no neighbouring layout or contract inherits it.
    ((0, 1, 2, 3, 4, 5, 12288), "none"): _VSHARD2_P2,
    ((0, 1, 2, 3, 4, 5, 12288), "fp32_final_only"): _VSHARD2_P2,
    ((0, 1, 2, 3, 4, 5, 12288), "fp32_both"): _VSHARD4_P2,
}


@dataclass(frozen=True)
class DeviceMetadata:
    """The three device facts covered by the B300 measurement whitelist."""

    name: str
    capability: tuple[int, int]
    multiprocessor_count: int


@dataclass(frozen=True)
class DispatchDecision:
    """Pure policy result before optional extension-symbol fallback."""

    requested_variant: str
    chosen_variant: str
    reason: str


_UNINITIALIZED_DECISION: dict[str, object] = {
    "requested_variant": "uninitialized",
    "chosen_variant": "uninitialized",
    "reason": "no_dispatch_yet",
}
_DECISION_LOCAL = threading.local()


def get_last_decision() -> dict:
    """Return this thread's most recent policy/pre-launch fallback result."""

    return dict(getattr(_DECISION_LOCAL, "value", _UNINITIALIZED_DECISION))


def _record(decision: DispatchDecision, **extra: object) -> None:
    _DECISION_LOCAL.value = {**asdict(decision), **extra}


def _dtype_name(tensor: Any) -> str:
    """Normalize real torch dtypes and small CPU-test dtype doubles."""

    return str(getattr(tensor, "dtype", "")).lower().replace("torch.", "")


def _device_type(tensor: Any) -> str:
    device = getattr(tensor, "device", None)
    return str(getattr(device, "type", "")).lower()


def _device_index(tensor: Any) -> int | None:
    device = getattr(tensor, "device", None)
    index = getattr(device, "index", None)
    return None if index is None else int(index)


def _shape(tensor: Any) -> tuple[int, ...] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _is_contiguous(tensor: Any) -> bool:
    method = getattr(tensor, "is_contiguous", None)
    return bool(method()) if callable(method) else False


def _all_cuda_contiguous_same_device(tensors: tuple[Any, ...]) -> bool:
    if not tensors:
        return False
    first_index = _device_index(tensors[0])
    return all(
        _device_type(tensor) == "cuda"
        and _is_contiguous(tensor)
        and _device_index(tensor) == first_index
        for tensor in tensors
    )


def _state_tensor_matches(
    state: Any | None,
    sequences: int,
    heads: int,
    device_index: int | None,
    dtype: str,
) -> bool:
    if state is None:
        return False
    expected = (sequences, heads, 128, 128)
    return (
        _shape(state) == expected
        and _dtype_name(state) == dtype
        and _device_type(state) == "cuda"
        and _device_index(state) == device_index
        and _is_contiguous(state)
    )


def _state_whitelist_variant(
    initial_state: Any | None,
    final_state: Any | None,
    batch: int,
    tokens: int,
    heads: int,
    device_index: int | None,
) -> tuple[str | None, str]:
    """Return the measured state-contract choice, never an inferred one."""

    if batch > 1:
        if tokens != 2048 or heads != 12:
            return None, "fixed_batch_shape_not_whitelisted"
        if initial_state is None and final_state is None:
            contract = "none"
        elif initial_state is None and _state_tensor_matches(
            final_state, batch, heads, device_index, _FP32
        ):
            contract = "fp32_final_only"
        elif (
            _state_tensor_matches(initial_state, batch, heads, device_index, _FP32)
            and _state_tensor_matches(final_state, batch, heads, device_index, _FP32)
        ):
            contract = "fp32_both"
        else:
            return None, "fixed_batch_requires_exact_fla_public_state_contract"
        variant = _FIXED_BATCH_PUBLIC_VARIANTS.get((batch, contract))
        if variant is None:
            return None, f"fixed_batch_b{batch}_{contract}_not_whitelisted"
        return variant, f"fixed_batch_b{batch}_h12_t2048_{contract}_whitelist_hit"

    if initial_state is None and final_state is None:
        exact_variant = _FIXED_SINGLE_BATCH_PUBLIC_VARIANTS.get((tokens, "none"))
        if heads == 12 and exact_variant is not None:
            return exact_variant, f"fixed_single_batch_b1_h12_t{tokens}_none_whitelist_hit"
        if heads == 12 and tokens in _H12_LENGTHS:
            return _VSHARD4_P2, "state_contract_none_h12_length_whitelist_hit"
        if heads == 12:
            return None, "state_contract_none_h12_length_not_whitelisted"
        return None, "state_contract_none_only_h12_whitelisted"
    if initial_state is None and _state_tensor_matches(final_state, batch, heads, device_index, _FP32):
        exact_variant = _FIXED_SINGLE_BATCH_PUBLIC_VARIANTS.get((tokens, "fp32_final_only"))
        if heads == 12 and exact_variant is not None:
            return exact_variant, f"fixed_single_batch_b1_h12_t{tokens}_fp32_final_only_whitelist_hit"
        if heads == 12 and tokens in _H12_LENGTHS:
            return _VSHARD4_P2, "state_contract_fla_fp32_final_only_h12_length_whitelist_hit"
        if heads == 12:
            return None, "state_contract_fla_fp32_final_only_h12_length_not_whitelisted"
        return None, "state_contract_fla_fp32_final_only_only_h12_whitelisted"
    if (
        _state_tensor_matches(initial_state, batch, heads, device_index, _FP32)
        and _state_tensor_matches(final_state, batch, heads, device_index, _FP32)
    ):
        if heads == 12 and tokens in _H12_LENGTHS:
            return _VSHARD4_P2, "state_contract_fp32_both_h12_length_whitelist_hit"
        if heads == 12:
            return None, "state_contract_fp32_both_h12_length_not_whitelisted"
        return None, "state_contract_fp32_both_only_h12_whitelisted"
    if (
        _state_tensor_matches(initial_state, batch, heads, device_index, _BF16)
        and _state_tensor_matches(final_state, batch, heads, device_index, _BF16)
    ):
        if tokens == 8192:
            return (
                _VSHARD4_P2 if heads <= 37 else _VSHARD2_P2,
                "state_contract_bf16_both_t8192_full_head_whitelist_hit",
            )
        if heads == 12 and tokens in _H12_LENGTHS:
            return _VSHARD4_P2, "state_contract_bf16_both_h12_length_whitelist_hit"
        if tokens in _INTERACTION_LENGTHS and heads in _INTERACTION_VARIANTS:
            return (
                _INTERACTION_VARIANTS[heads],
                "state_contract_bf16_both_length_head_interaction_whitelist_hit",
            )
        return None, "state_contract_bf16_both_length_head_not_whitelisted"
    return None, "state_contract_requires_measured_bf16_both_or_h12_fla_contract"


def _varlen_public_state_contract(
    initial_state: Any | None,
    final_state: Any | None,
    sequences: int,
    heads: int,
    device_index: int | None,
) -> tuple[str | None, str]:
    """Classify only the three FLA public contracts used by varlen evidence."""

    if initial_state is None and final_state is None:
        return "none", "varlen_public_state_none"
    if initial_state is None and _state_tensor_matches(
        final_state, sequences, heads, device_index, _FP32
    ):
        return "fp32_final_only", "varlen_public_state_fp32_final_only"
    if (
        _state_tensor_matches(initial_state, sequences, heads, device_index, _FP32)
        and _state_tensor_matches(final_state, sequences, heads, device_index, _FP32)
    ):
        return "fp32_both", "varlen_public_state_fp32_both"
    return None, "varlen_requires_exact_fla_public_state_contract"


def varlen_public_variant(
    offsets: tuple[int, ...],
    heads: int,
    initial_state: Any | None,
    final_state: Any | None,
    device_index: int | None,
) -> tuple[str | None, str]:
    """Return an exact packed-layout choice; structural authentication is separate."""

    if heads != 12:
        return None, "varlen_only_h12_whitelisted"
    layout = _VARLEN_LAYOUT_NAMES.get(offsets)
    if layout is None:
        return None, "varlen_offsets_not_whitelisted"
    sequences = len(offsets) - 1
    contract, state_reason = _varlen_public_state_contract(
        initial_state, final_state, sequences, heads, device_index
    )
    if contract is None:
        return None, state_reason
    variant = _VARLEN_PUBLIC_VARIANTS.get((offsets, contract))
    if variant is None:
        return None, f"varlen_{layout}_{contract}_not_whitelisted"
    return variant, f"varlen_{layout}_{contract}_whitelist_hit"


def select_variant(
    metadata: DeviceMetadata,
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    out: Any,
    A_log: Any,
    dt_bias: Any,
    scale: float,
    lower_bound: float,
    initial_state: Any | None = None,
    final_state: Any | None = None,
    cu_seqlens: Any | None = None,
    certified_varlen_offsets: tuple[int, ...] | None = None,
) -> DispatchDecision:
    """Return the whitelist choice without importing CUDA or launching a kernel.

    This function is intentionally public and torch-free so the exact policy
    boundaries can be unit-tested with simple tensor metadata doubles.
    """

    if "B300" not in metadata.name.upper():
        return DispatchDecision(_BASELINE, _BASELINE, "device_name_not_b300")
    if metadata.capability != (10, 3):
        return DispatchDecision(_BASELINE, _BASELINE, "device_capability_not_sm103")
    if metadata.multiprocessor_count != 148:
        return DispatchDecision(_BASELINE, _BASELINE, "device_sm_count_not_148")

    q_shape = _shape(q)
    if q_shape is None or len(q_shape) != 4:
        return DispatchDecision(_BASELINE, _BASELINE, "q_must_be_rank4")
    batch, tokens, heads, key_dim = q_shape
    if batch < 1 or key_dim != 128:
        return DispatchDecision(_BASELINE, _BASELINE, "shape_requires_positive_batch_k128")
    if batch > 1 and (batch not in (2, 3, 4, 5, 6) or tokens != 2048 or heads != 12):
        return DispatchDecision(_BASELINE, _BASELINE, "fixed_batch_shape_not_whitelisted")
    if heads < 1 or heads > 96:
        return DispatchDecision(_BASELINE, _BASELINE, "head_count_outside_h1_to_h96")
    if any(_shape(tensor) != q_shape for tensor in (k, v, g, out)):
        return DispatchDecision(_BASELINE, _BASELINE, "q_k_v_g_out_shape_mismatch")
    if _shape(beta) != (batch, tokens, heads):
        return DispatchDecision(_BASELINE, _BASELINE, "beta_shape_mismatch")
    if _shape(A_log) != (heads,):
        return DispatchDecision(_BASELINE, _BASELINE, "A_log_shape_mismatch")
    if _shape(dt_bias) != (heads, key_dim):
        return DispatchDecision(_BASELINE, _BASELINE, "dt_bias_shape_mismatch")

    input_tensors = (q, k, v, g, beta, out)
    if any(_dtype_name(tensor) != _BF16 for tensor in input_tensors):
        return DispatchDecision(_BASELINE, _BASELINE, "q_k_v_g_beta_out_must_be_bf16")
    if any(_dtype_name(tensor) != _FP32 for tensor in (A_log, dt_bias)):
        return DispatchDecision(_BASELINE, _BASELINE, "A_log_and_dt_bias_must_be_fp32")
    all_tensors = input_tensors + (A_log, dt_bias)
    if not _all_cuda_contiguous_same_device(all_tensors):
        return DispatchDecision(_BASELINE, _BASELINE, "tensor_device_or_contiguity_mismatch")
    if not isinstance(scale, (float, int)) or not math.isfinite(float(scale)):
        return DispatchDecision(_BASELINE, _BASELINE, "scale_must_be_finite")
    if not isinstance(lower_bound, (float, int)) or not math.isfinite(float(lower_bound)):
        return DispatchDecision(_BASELINE, _BASELINE, "lower_bound_must_be_finite")
    if cu_seqlens is not None:
        if batch != 1:
            return DispatchDecision(_BASELINE, _BASELINE, "varlen_requires_q_batch_1")
        if certified_varlen_offsets is None:
            return DispatchDecision(_BASELINE, _BASELINE, "varlen_cpu_descriptor_not_certified")
        offsets = tuple(certified_varlen_offsets)
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != tokens
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
            or any(left >= right for left, right in zip(offsets, offsets[1:]))
        ):
            return DispatchDecision(_BASELINE, _BASELINE, "varlen_certified_offsets_invalid")
        variant, state_reason = varlen_public_variant(
            offsets, heads, initial_state, final_state, _device_index(q)
        )
        if variant is None:
            return DispatchDecision(_BASELINE, _BASELINE, state_reason)
        return DispatchDecision(variant, variant, state_reason)
    if certified_varlen_offsets is not None:
        return DispatchDecision(_BASELINE, _BASELINE, "varlen_certificate_without_cu_seqlens")
    variant, state_reason = _state_whitelist_variant(
        initial_state, final_state, batch, tokens, heads, _device_index(q)
    )
    if variant is None:
        return DispatchDecision(_BASELINE, _BASELINE, state_reason)
    return DispatchDecision(variant, variant, state_reason)


def _read_device_metadata(q: Any) -> DeviceMetadata:
    """Read metadata only for CUDA tensors; non-CUDA remains a clean miss."""

    if _device_type(q) != "cuda":
        return DeviceMetadata("", (-1, -1), 0)
    torch = importlib.import_module("torch")
    properties = torch.cuda.get_device_properties(q.device)
    return DeviceMetadata(
        name=str(properties.name),
        capability=(int(properties.major), int(properties.minor)),
        multiprocessor_count=int(properties.multi_processor_count),
    )


@lru_cache(maxsize=1)
def _load_extension_and_symbols(
) -> tuple[Any | None, frozenset[str], str | None, str | None]:
    """Bind symbols to the exact audited binary before a sharded launch.

    The digest is cached once per process.  Python extension modules cannot be
    safely replaced in-place after import, so re-reading a multi-megabyte SO on
    every inference call would add cost without making hot replacement safe.
    """

    try:
        extension = importlib.import_module("flash_kda_C")
    except ImportError:
        return None, frozenset(), None, "extension_import_failed"
    extension_file = getattr(extension, "__file__", None)
    if not extension_file:
        return None, frozenset(), None, "extension_path_unavailable"
    try:
        extension_path = Path(extension_file).resolve(strict=True)
        digest = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    except OSError:
        return None, frozenset(), None, "extension_identity_unreadable"
    if digest not in _AUDITED_EXTENSION_SHA256:
        return None, frozenset(), digest, "extension_sha256_not_allowlisted"
    symbols = frozenset(
        symbol
        for symbol in ("fwd_vshard4_p2", "fwd_vshard_p2")
        if callable(getattr(extension, symbol, None))
    )
    return extension, symbols, digest, None


def _choose_available_variant(
    decision: DispatchDecision,
) -> tuple[DispatchDecision, Any | None, str | None]:
    """Require the exact whitelisted symbol or fall back before launch.

    Variant selection is evidence-scoped: a cell released for v4 has not
    thereby released v2.  Substituting another accelerated kernel when the
    selected symbol is missing would silently escape the measured whitelist.
    """

    if decision.chosen_variant == _BASELINE:
        return decision, None, None

    extension, symbols, digest, identity_rejection = _load_extension_and_symbols()
    if identity_rejection is not None:
        reason = f"{decision.reason}; {identity_rejection}_prelaunch_fallback_to_baseline"
        return DispatchDecision(decision.requested_variant, _BASELINE, reason), None, digest
    if decision.chosen_variant == _VSHARD4_P2 and "fwd_vshard4_p2" in symbols:
        return decision, extension, digest
    if decision.chosen_variant == _VSHARD2_P2 and "fwd_vshard_p2" in symbols:
        return decision, extension, digest
    if decision.chosen_variant == _VSHARD4_P2:
        reason = f"{decision.reason}; fwd_vshard4_p2_missing_prelaunch_fallback_to_baseline"
    else:
        reason = f"{decision.reason}; fwd_vshard_p2_missing_prelaunch_fallback_to_baseline"
    return DispatchDecision(decision.requested_variant, _BASELINE, reason), None, digest


def _launch_baseline(**kwargs: Any) -> Any:
    flash_kda = importlib.import_module("flash_kda")
    return flash_kda.fwd(**kwargs)


def _launch_sharded(variant: str, extension: Any, **kwargs: Any) -> Any:
    """Launch an already-proven extension symbol; do not catch execution errors."""

    torch = importlib.import_module("torch")
    q = kwargs["q"]
    batch, tokens, heads, _ = q.shape
    cu_seqlens = kwargs["cu_seqlens"]
    sequences = batch if cu_seqlens is None else int(cu_seqlens.numel()) - 1
    if sequences < 1:
        raise RuntimeError("validated launch requires at least one sequence")
    workspace_bytes = extension.get_workspace_size(batch * tokens, heads, sequences)
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    symbol_name = "fwd_vshard4_p2" if variant == _VSHARD4_P2 else "fwd_vshard_p2"
    return getattr(extension, symbol_name)(
        kwargs["q"], kwargs["k"], kwargs["v"], kwargs["g"], kwargs["beta"],
        float(kwargs["scale"]), kwargs["out"], workspace, kwargs["A_log"],
        kwargs["dt_bias"], float(kwargs["lower_bound"]),
        initial_state=kwargs["initial_state"], final_state=kwargs["final_state"],
        cu_seqlens=kwargs["cu_seqlens"],
    )


def _launch_variant(variant: str, extension: Any | None, **kwargs: Any) -> Any:
    if variant == _BASELINE:
        return _launch_baseline(**kwargs)
    if extension is None:  # Defensive invariant: selection must have found a symbol.
        raise RuntimeError(f"{variant} was selected without a preflighted extension")
    return _launch_sharded(variant, extension, **kwargs)


def fwd(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    scale: float,
    out: Any,
    A_log: Any,
    dt_bias: Any,
    lower_bound: float,
    initial_state: Any | None = None,
    final_state: Any | None = None,
    cu_seqlens: Any | None = None,
    cu_seqlens_cpu: Any | None = None,
    _varlen_descriptor: Any | None = None,
) -> Any:
    """Dispatch one raw FlashKDA ABI call, preserving its in-place semantics."""

    launch_cu_seqlens = cu_seqlens
    certified_offsets: tuple[int, ...] | None = None
    cache_hit: bool | None = None
    if _varlen_descriptor is not None:
        if cu_seqlens is None or cu_seqlens_cpu is None:
            raise varlen_metadata.MetadataError(
                "packed_varlen_descriptor_requires_gpu_and_cpu_cu_seqlens"
            )
        cached = varlen_metadata.cached_gpu_offsets(
            q, cu_seqlens, cu_seqlens_cpu, _varlen_descriptor
        )
        launch_cu_seqlens = cached.tensor
        certified_offsets = cached.key.offsets
        cache_hit = bool(cached.cache_hit)

    decision = select_variant(
        _read_device_metadata(q), q, k, v, g, beta, out, A_log, dt_bias,
        scale, lower_bound, initial_state, final_state, launch_cu_seqlens,
        certified_offsets,
    )
    decision, extension, extension_sha256 = _choose_available_variant(decision)
    _record(
        decision,
        extension_sha256=extension_sha256,
        varlen_cpu_authoritative=_varlen_descriptor is not None,
        certified_varlen_offsets=(None if certified_offsets is None else list(certified_offsets)),
        canonical_cache_hit=cache_hit,
    )
    return _launch_variant(
        decision.chosen_variant,
        extension,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        out=out,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=launch_cu_seqlens,
    )


__all__ = [
    "DeviceMetadata",
    "DispatchDecision",
    "fwd",
    "get_last_decision",
    "select_variant",
    "varlen_public_variant",
]
