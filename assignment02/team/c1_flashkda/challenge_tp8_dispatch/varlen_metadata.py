"""CPU-authoritative packed-varlen metadata with a stream-safe CUDA cache.

This module deliberately contains no dispatch or performance policy.  A
descriptor proves that one explicitly opted-in CPU ``int64`` offsets tensor
was checked; it does *not* prove that any variant is eligible to run faster.
The later dispatcher may apply its own, narrower performance whitelist to the
returned ``offsets`` tuple.

There is intentionally no module-level :mod:`torch` import.  CPU policy tests
use small tensor metadata doubles; CUDA is imported only when the cache must
materialize a device copy.  In a real integration pass both FLA tensors:

``descriptor = issue_descriptor(q, cu_seqlens_cpu, opt_in=True)``
``cached = cached_gpu_offsets(q, cu_seqlens, cu_seqlens_cpu, descriptor)``

The GPU ``cu_seqlens`` is *only* validated structurally.  Its values are never
read: the cached tensor is copied from the authenticated CPU tuple instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import numbers
import threading
from typing import Any
from weakref import WeakKeyDictionary


INT32_MAX = (1 << 31) - 1


class MetadataError(ValueError):
    """Fail-closed rejection of a packed-varlen metadata contract."""


class CaptureCacheUseError(RuntimeError):
    """Base error for deliberately unsupported cache use during graph capture."""


class CaptureCacheMissError(CaptureCacheUseError):
    """Raised rather than allocating/copying offsets while CUDA graph captures."""


class CaptureCacheHitError(CaptureCacheUseError):
    """Raised rather than baking a cache-owned pointer into a CUDA graph."""


@dataclass(frozen=True)
class ValidatedDescriptor:
    """Authenticated CPU descriptor facts; never a performance authorization."""

    offsets: tuple[int, ...]
    sequence_count: int
    total_tokens: int


@dataclass(frozen=True)
class CacheKey:
    """The only cache identity: device index plus the canonical CPU tuple."""

    device_index: int
    offsets: tuple[int, ...]


@dataclass(frozen=True)
class CachedOffsets:
    """One current-stream-safe device offsets tensor returned by the cache."""

    tensor: Any
    key: CacheKey
    cache_hit: bool


@dataclass
class _IssuedRecord:
    issuer_token: object
    cpu_tensor: Any
    validated: ValidatedDescriptor


class VarlenDescriptor:
    """Opaque, process-local proof issued by :func:`issue_descriptor` only.

    The class has no public data attributes and every consuming API checks a
    private issuer token *and* original CPU tensor object identity in the
    private issuance ledger.  Python is not a cross-process security boundary,
    but normal construction, copying, pickling, and a bare ``object.__new__``
    object cannot forge a usable descriptor.
    """

    __slots__ = ("__weakref__",)

    def __init__(self, issuer_token: object) -> None:
        if issuer_token is not _ISSUER_TOKEN:
            raise TypeError("VarlenDescriptor instances must be issued by issue_descriptor()")

    def __reduce__(self) -> object:
        raise TypeError("VarlenDescriptor is process-local and cannot be pickled")


_ISSUER_TOKEN = object()
_ISSUED: WeakKeyDictionary[VarlenDescriptor, _IssuedRecord] = WeakKeyDictionary()
_ISSUED_LOCK = threading.RLock()


@dataclass
class _CacheEntry:
    gpu_tensor: Any
    completion_event: Any
    cpu_staging_tensor: Any


_CACHE: dict[CacheKey, _CacheEntry] = {}
_CACHE_LOCK = threading.RLock()
_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "capture_miss_rejections": 0,
    "capture_hit_rejections": 0,
}


def _dtype_name(tensor: Any) -> str:
    return str(getattr(tensor, "dtype", "")).lower().replace("torch.", "")


def _device_type(tensor: Any) -> str:
    return str(getattr(getattr(tensor, "device", None), "type", "")).lower()


def _device_index(tensor: Any) -> int | None:
    index = getattr(getattr(tensor, "device", None), "index", None)
    return None if index is None else int(index)


def _shape(tensor: Any) -> tuple[int, ...] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return None


def _numel(tensor: Any) -> int | None:
    method = getattr(tensor, "numel", None)
    if callable(method):
        try:
            return int(method())
        except (TypeError, ValueError):
            return None
    shape = _shape(tensor)
    if shape is None:
        return None
    product = 1
    for dimension in shape:
        product *= dimension
    return product


def _is_contiguous(tensor: Any) -> bool:
    method = getattr(tensor, "is_contiguous", None)
    return bool(method()) if callable(method) else False


def _is_int64(tensor: Any) -> bool:
    return _dtype_name(tensor) in {"int64", "long"}


def _q_batch_and_tokens(q: Any) -> tuple[int, int]:
    shape = _shape(q)
    if shape is None or len(shape) != 4:
        raise MetadataError("q_must_be_rank4")
    batch, tokens = shape[0], shape[1]
    if batch != 1:
        raise MetadataError("packed_varlen_requires_q_batch_1")
    if tokens < 1:
        raise MetadataError("q_tokens_must_be_positive")
    return batch, tokens


def _read_cpu_offsets(cu_seqlens_cpu: Any) -> tuple[int, ...]:
    """Read values exactly once, on the CPU authority path only."""
    if _device_type(cu_seqlens_cpu) != "cpu":
        raise MetadataError("cpu_descriptor_must_be_cpu")
    if not _is_int64(cu_seqlens_cpu):
        raise MetadataError("cpu_descriptor_must_be_int64")
    shape = _shape(cu_seqlens_cpu)
    if shape is None or len(shape) != 1:
        raise MetadataError("cpu_descriptor_must_be_rank1")
    if not _is_contiguous(cu_seqlens_cpu):
        raise MetadataError("cpu_descriptor_must_be_contiguous")
    tolist = getattr(cu_seqlens_cpu, "tolist", None)
    if not callable(tolist):
        raise MetadataError("cpu_descriptor_values_unreadable")
    values = tolist()
    if not isinstance(values, list):
        raise MetadataError("cpu_descriptor_values_must_be_a_list")
    if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) for value in values):
        raise MetadataError("cpu_descriptor_values_must_be_exact_integers")
    offsets = tuple(int(value) for value in values)
    if _numel(cu_seqlens_cpu) != len(offsets):
        raise MetadataError("cpu_descriptor_numel_mismatch_values")
    return offsets


def _validate_offsets(offsets: tuple[int, ...], tokens: int) -> ValidatedDescriptor:
    """Validate the complete canonical tuple before it can be issued."""
    if len(offsets) < 2:
        raise MetadataError("cpu_descriptor_requires_at_least_one_sequence")
    if any(value < 0 or value > INT32_MAX for value in offsets):
        raise MetadataError("cpu_descriptor_offsets_outside_int32_range")
    if offsets[0] != 0:
        raise MetadataError("cpu_descriptor_first_offset_must_be_zero")
    if offsets[-1] != tokens:
        raise MetadataError("cpu_descriptor_final_offset_must_equal_q_tokens")
    if any(left >= right for left, right in zip(offsets, offsets[1:])):
        raise MetadataError("cpu_descriptor_offsets_must_be_strictly_increasing")
    return ValidatedDescriptor(offsets, len(offsets) - 1, offsets[-1])


def validate_cpu_descriptor(q: Any, cu_seqlens_cpu: Any) -> ValidatedDescriptor:
    """Read and fully validate one current CPU offsets tuple.

    This is deliberately narrower than :func:`issue_descriptor`: it issues no
    capability and updates no ledger.  Consumers that need freshness after an
    earlier verifier pass use it to repeat the complete authority check on the
    mutable CPU tensor.  Exactly one ``tolist()`` call is made on every
    successful or value-validation path through this helper.
    """
    _, tokens = _q_batch_and_tokens(q)
    return _validate_offsets(_read_cpu_offsets(cu_seqlens_cpu), tokens)


def issue_descriptor(q: Any, cu_seqlens_cpu: Any, *, opt_in: bool = False) -> VarlenDescriptor:
    """Issue an opaque descriptor after explicit CPU-side opt-in and validation.

    ``opt_in`` is an authorization boundary only.  It does not consult any
    performance whitelist and does not permit a custom kernel by itself.
    """
    if opt_in is not True:
        raise MetadataError("packed_varlen_cpu_descriptor_requires_explicit_opt_in")
    validated = validate_cpu_descriptor(q, cu_seqlens_cpu)
    descriptor = VarlenDescriptor(_ISSUER_TOKEN)
    with _ISSUED_LOCK:
        _ISSUED[descriptor] = _IssuedRecord(_ISSUER_TOKEN, cu_seqlens_cpu, validated)
    return descriptor


def verify_descriptor(descriptor: Any, cu_seqlens_cpu: Any) -> ValidatedDescriptor:
    """Verify issuer token and original CPU tensor identity, then reveal facts.

    This intentionally checks object identity rather than rereading mutable
    CPU values.  The immutable tuple captured at issuance is authoritative.
    """
    if not isinstance(descriptor, VarlenDescriptor):
        raise MetadataError("packed_varlen_descriptor_type_invalid")
    with _ISSUED_LOCK:
        record = _ISSUED.get(descriptor)
        if record is None or record.issuer_token is not _ISSUER_TOKEN:
            raise MetadataError("packed_varlen_descriptor_not_issued_here")
        if record.cpu_tensor is not cu_seqlens_cpu:
            raise MetadataError("packed_varlen_descriptor_cpu_tensor_identity_mismatch")
        return record.validated


def _validate_gpu_structure(
    q: Any,
    cu_seqlens: Any,
    validated: ValidatedDescriptor,
) -> CacheKey:
    """Validate metadata only; this function must never read GPU offset values."""
    _, tokens = _q_batch_and_tokens(q)
    if tokens != validated.total_tokens:
        raise MetadataError("q_tokens_must_match_cpu_descriptor")
    if _device_type(q) != "cuda":
        raise MetadataError("q_must_be_cuda")
    q_device_index = _device_index(q)
    if q_device_index is None:
        raise MetadataError("q_cuda_device_index_required")
    if _device_type(cu_seqlens) != "cuda":
        raise MetadataError("gpu_cu_seqlens_must_be_cuda")
    if _device_index(cu_seqlens) != q_device_index:
        raise MetadataError("gpu_cu_seqlens_must_share_q_device")
    if not _is_int64(cu_seqlens):
        raise MetadataError("gpu_cu_seqlens_must_be_int64")
    shape = _shape(cu_seqlens)
    if shape is None or len(shape) != 1:
        raise MetadataError("gpu_cu_seqlens_must_be_rank1")
    if not _is_contiguous(cu_seqlens):
        raise MetadataError("gpu_cu_seqlens_must_be_contiguous")
    if _numel(cu_seqlens) != len(validated.offsets):
        raise MetadataError("gpu_cu_seqlens_numel_mismatch_cpu_descriptor")
    return CacheKey(q_device_index, validated.offsets)


def validate_gpu_descriptor(
    q: Any,
    cu_seqlens: Any,
    cu_seqlens_cpu: Any,
    descriptor: Any,
) -> CacheKey:
    """Authenticate CPU proof and structurally validate the untrusted GPU tensor.

    No indexing, conversion, comparison, synchronization, or value read is
    performed on ``cu_seqlens``.  The returned key is entirely CPU-authorized.
    """
    return _validate_gpu_structure(q, cu_seqlens, verify_descriptor(descriptor, cu_seqlens_cpu))


def _runtime(torch_module: Any | None) -> Any:
    return torch_module if torch_module is not None else importlib.import_module("torch")


def _current_stream(torch_module: Any, device: Any) -> Any:
    cuda = getattr(torch_module, "cuda", None)
    current_stream = getattr(cuda, "current_stream", None)
    if not callable(current_stream):
        raise RuntimeError("torch.cuda.current_stream is unavailable")
    return current_stream(device=device)


def _stream_is_capturing(torch_module: Any, stream: Any, device: Any) -> bool:
    """Return capture state; unknown state is a miss-time fail-closed error."""
    cuda = getattr(torch_module, "cuda", None)
    probe = getattr(cuda, "is_current_stream_capturing", None)
    if callable(probe):
        device_context = getattr(cuda, "device", None)
        if not callable(device_context):
            raise RuntimeError("torch.cuda.device is unavailable for capture probe")
        # PyTorch's no-argument probe inspects the current stream on the
        # *currently selected device*, which need not be q.device in a
        # multi-GPU process.  Select q.device around the probe so a capture on
        # that device can never be mistaken for an ordinary cache miss.
        with device_context(device):
            return bool(probe())
    stream_probe = getattr(stream, "is_capturing", None)
    if callable(stream_probe):
        return bool(stream_probe())
    raise RuntimeError("cannot determine CUDA graph capture state")


def _copy_tuple_to_current_stream(torch_module: Any, q: Any, offsets: tuple[int, ...], stream: Any) -> _CacheEntry:
    """Create CPU staging from the canonical tuple, enqueue H2D, then record event."""
    cuda = getattr(torch_module, "cuda", None)
    event_factory = getattr(cuda, "Event", None)
    tensor_factory = getattr(torch_module, "tensor", None)
    if not callable(event_factory) or not callable(tensor_factory):
        raise RuntimeError("torch CUDA tensor/event APIs are unavailable")
    # Source is the canonical tuple, never the caller-provided GPU tensor.
    cpu_staging = tensor_factory(offsets, dtype=torch_module.int64, device="cpu")
    to_device = getattr(cpu_staging, "to", None)
    if not callable(to_device):
        raise RuntimeError("CPU staging tensor lacks .to()")
    gpu_tensor = to_device(device=q.device, dtype=torch_module.int64, non_blocking=True)
    event = event_factory(enable_timing=False)
    event.record(stream)
    record_stream = getattr(gpu_tensor, "record_stream", None)
    if not callable(record_stream):
        raise RuntimeError("GPU offsets tensor lacks record_stream()")
    record_stream(stream)
    return _CacheEntry(gpu_tensor=gpu_tensor, completion_event=event, cpu_staging_tensor=cpu_staging)


def cached_gpu_offsets(
    q: Any,
    cu_seqlens: Any,
    cu_seqlens_cpu: Any,
    descriptor: Any,
    *,
    torch_module: Any | None = None,
) -> CachedOffsets:
    """Return an authenticated current-stream-safe GPU offsets tensor.

    On a cache miss the canonical tuple is copied CPU->GPU on the current
    stream, then a CUDA event is recorded.  CUDA graph capture is deliberately
    unsupported for both misses and hits: a hit would otherwise bake a pointer
    owned only by this clearable cache into a replayable graph.  Outside
    capture, a hit makes the current stream wait for the publication event and
    records the current stream before returning the tensor.
    """
    key = validate_gpu_descriptor(q, cu_seqlens, cu_seqlens_cpu, descriptor)
    torch_runtime = _runtime(torch_module)
    stream = _current_stream(torch_runtime, q.device)
    # Cache lookup, miss construction/event publication, hit synchronization,
    # accounting, and clear_cache() are one critical section.  Holding the
    # entry while calling wait_event/record_stream prevents clear_cache() from
    # removing the ledger entry between lookup and its stream-lifetime update.
    # RLock is intentional: a test or a future instrumentation callback can
    # inspect stats without deadlocking this thread.  The short CUDA calls may
    # block during a cache miss, but a per-key in-flight protocol would make
    # clear/stream lifetime semantics considerably less auditable.
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if _stream_is_capturing(torch_runtime, stream, q.device):
            if entry is None:
                _CACHE_STATS["capture_miss_rejections"] += 1
                raise CaptureCacheMissError("packed_varlen_cache_miss_during_cuda_graph_capture")
            _CACHE_STATS["capture_hit_rejections"] += 1
            raise CaptureCacheHitError("packed_varlen_cache_hit_during_cuda_graph_capture")
        if entry is None:
            entry = _copy_tuple_to_current_stream(torch_runtime, q, key.offsets, stream)
            _CACHE[key] = entry
            _CACHE_STATS["misses"] += 1
            return CachedOffsets(entry.gpu_tensor, key, cache_hit=False)
        wait_event = getattr(stream, "wait_event", None)
        record_stream = getattr(entry.gpu_tensor, "record_stream", None)
        if not callable(wait_event) or not callable(record_stream):
            raise RuntimeError("CUDA stream synchronization APIs are unavailable")
        wait_event(entry.completion_event)
        record_stream(stream)
        _CACHE_STATS["hits"] += 1
        return CachedOffsets(entry.gpu_tensor, key, cache_hit=True)


def clear_cache() -> None:
    """Drop cached GPU references and reset accounting; safe for CPU tests."""
    with _CACHE_LOCK:
        _CACHE.clear()
        for name in _CACHE_STATS:
            _CACHE_STATS[name] = 0


def cache_stats() -> dict[str, int]:
    """Return a snapshot for tests/audit logs, without exposing cache entries."""
    with _CACHE_LOCK:
        return {"entries": len(_CACHE), **{name: int(value) for name, value in _CACHE_STATS.items()}}


__all__ = [
    "CacheKey",
    "CachedOffsets",
    "CaptureCacheHitError",
    "CaptureCacheMissError",
    "CaptureCacheUseError",
    "INT32_MAX",
    "MetadataError",
    "ValidatedDescriptor",
    "VarlenDescriptor",
    "cache_stats",
    "cached_gpu_offsets",
    "clear_cache",
    "issue_descriptor",
    "validate_cpu_descriptor",
    "validate_gpu_descriptor",
    "verify_descriptor",
]
