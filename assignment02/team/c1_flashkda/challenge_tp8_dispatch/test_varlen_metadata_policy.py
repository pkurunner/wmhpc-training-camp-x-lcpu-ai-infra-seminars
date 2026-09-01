"""CPU-authoritative policy tests for :mod:`varlen_metadata`.

Run with ``python test_varlen_metadata_policy.py`` or pytest.  These tests use
only metadata doubles: they neither import torch nor require a GPU.
"""

from __future__ import annotations

from pathlib import Path
import sys
import threading
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_tp8_dispatch import varlen_metadata as metadata


class FakeDevice:
    def __init__(self, device_type: str, index: int | None = 0) -> None:
        self.type = device_type
        self.index = index


class FakeQ:
    def __init__(self, *, batch: int = 1, tokens: int = 8, device: FakeDevice | None = None) -> None:
        self.shape = (batch, tokens, 12, 128)
        self.device = device or FakeDevice("cuda", 0)


class FakeCpuOffsets:
    def __init__(
        self,
        values: list[Any],
        *,
        dtype: str = "torch.int64",
        shape: tuple[int, ...] | None = None,
        contiguous: bool = True,
    ) -> None:
        self._values = list(values)
        self.dtype = dtype
        self.shape = shape if shape is not None else (len(values),)
        self.device = FakeDevice("cpu", None)
        self._contiguous = contiguous
        self.tolist_calls = 0

    def is_contiguous(self) -> bool:
        return self._contiguous

    def tolist(self) -> list[Any]:
        self.tolist_calls += 1
        return list(self._values)


class FakeGpuOffsets:
    def __init__(
        self,
        numel: int,
        *,
        device: FakeDevice | None = None,
        dtype: str = "torch.int64",
        shape: tuple[int, ...] | None = None,
        contiguous: bool = True,
    ) -> None:
        self.dtype = dtype
        self.shape = shape if shape is not None else (numel,)
        self.device = device or FakeDevice("cuda", 0)
        self._contiguous = contiguous
        self.tolist_calls = 0

    def is_contiguous(self) -> bool:
        return self._contiguous

    def numel(self) -> int:
        product = 1
        for dimension in self.shape:
            product *= dimension
        return product

    def tolist(self) -> list[int]:
        self.tolist_calls += 1
        raise AssertionError("GPU offset values must never be read")


def _issued(tokens: int = 8, values: list[int] | None = None):
    q = FakeQ(tokens=tokens)
    cpu = FakeCpuOffsets(values or [0, 3, tokens])
    return q, cpu, metadata.issue_descriptor(q, cpu, opt_in=True)


def _assert_error(expected: str, callback) -> None:
    try:
        callback()
    except metadata.MetadataError as exc:
        assert str(exc) == expected
    else:  # pragma: no cover - assertion failure branch
        raise AssertionError(f"expected MetadataError({expected!r})")


def test_module_avoids_top_level_torch_import() -> None:
    source = Path(metadata.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source


def test_explicit_cpu_issue_canonicalizes_and_authenticates_identity() -> None:
    q, cpu, descriptor = _issued()
    facts = metadata.verify_descriptor(descriptor, cpu)
    assert facts.offsets == (0, 3, 8)
    assert facts.sequence_count == 2
    assert facts.total_tokens == 8
    assert cpu.tolist_calls == 1
    equivalent_but_distinct = FakeCpuOffsets([0, 3, 8])
    _assert_error(
        "packed_varlen_descriptor_cpu_tensor_identity_mismatch",
        lambda: metadata.verify_descriptor(descriptor, equivalent_but_distinct),
    )
    _assert_error(
        "packed_varlen_cpu_descriptor_requires_explicit_opt_in",
        lambda: metadata.issue_descriptor(q, cpu),
    )


def test_fresh_cpu_validation_reads_once_and_reuses_the_full_issue_contract() -> None:
    q = FakeQ(tokens=8)
    cpu = FakeCpuOffsets([0, 3, 8])
    fresh = metadata.validate_cpu_descriptor(q, cpu)
    assert fresh == metadata.ValidatedDescriptor((0, 3, 8), 2, 8)
    assert cpu.tolist_calls == 1

    # Each independent freshness pass is a complete current read, not a
    # comparison against a prior tuple or a partial shape-only shortcut.
    cpu._values = [0, 4, 8]
    fresh_again = metadata.validate_cpu_descriptor(q, cpu)
    assert fresh_again == metadata.ValidatedDescriptor((0, 4, 8), 2, 8)
    assert cpu.tolist_calls == 2

    invalid = FakeCpuOffsets([0, 4, 4])
    _assert_error(
        "cpu_descriptor_final_offset_must_equal_q_tokens",
        lambda: metadata.validate_cpu_descriptor(q, invalid),
    )
    assert invalid.tolist_calls == 1


def test_descriptor_cannot_be_normally_constructed_or_bare_object_forged() -> None:
    _, cpu, descriptor = _issued()
    _assert_error(
        "packed_varlen_descriptor_not_issued_here",
        lambda: metadata.verify_descriptor(object.__new__(metadata.VarlenDescriptor), cpu),
    )
    try:
        metadata.VarlenDescriptor(object())
    except TypeError as exc:
        assert "must be issued" in str(exc)
    else:  # pragma: no cover - assertion failure branch
        raise AssertionError("normal descriptor construction unexpectedly succeeded")
    try:
        descriptor.__reduce__()
    except TypeError as exc:
        assert "cannot be pickled" in str(exc)
    else:  # pragma: no cover - assertion failure branch
        raise AssertionError("descriptor unexpectedly allowed pickling")


def test_cpu_authority_rejects_all_tuple_and_q_boundaries() -> None:
    q = FakeQ(tokens=8)
    cases = (
        (False, FakeCpuOffsets([0, 8]), "packed_varlen_cpu_descriptor_requires_explicit_opt_in"),
        (True, FakeCpuOffsets([0]), "cpu_descriptor_requires_at_least_one_sequence"),
        (True, FakeCpuOffsets([1, 8]), "cpu_descriptor_first_offset_must_be_zero"),
        (True, FakeCpuOffsets([0, 3, 3, 8]), "cpu_descriptor_offsets_must_be_strictly_increasing"),
        (True, FakeCpuOffsets([0, 9]), "cpu_descriptor_final_offset_must_equal_q_tokens"),
        (True, FakeCpuOffsets([0, metadata.INT32_MAX + 1]), "cpu_descriptor_offsets_outside_int32_range"),
        (True, FakeCpuOffsets([0, 8], dtype="torch.int32"), "cpu_descriptor_must_be_int64"),
        (True, FakeCpuOffsets([0, 8], shape=(1, 2)), "cpu_descriptor_must_be_rank1"),
        (True, FakeCpuOffsets([0, 8], shape=(3,)), "cpu_descriptor_numel_mismatch_values"),
        (True, FakeCpuOffsets([0.0, 8.0]), "cpu_descriptor_values_must_be_exact_integers"),
        (True, FakeCpuOffsets([0, 8], contiguous=False), "cpu_descriptor_must_be_contiguous"),
    )
    for opt_in, cpu, expected in cases:
        _assert_error(expected, lambda opt_in=opt_in, cpu=cpu: metadata.issue_descriptor(q, cpu, opt_in=opt_in))
    _assert_error(
        "packed_varlen_requires_q_batch_1",
        lambda: metadata.issue_descriptor(FakeQ(batch=2, tokens=8), FakeCpuOffsets([0, 8]), opt_in=True),
    )


def test_gpu_validation_reads_no_gpu_values_and_requires_only_structure() -> None:
    q, cpu, descriptor = _issued()
    gpu = FakeGpuOffsets(3)
    key = metadata.validate_gpu_descriptor(q, gpu, cpu, descriptor)
    assert key == metadata.CacheKey(0, (0, 3, 8))
    assert gpu.tolist_calls == 0

    failures = (
        (FakeQ(device=FakeDevice("cpu", None)), gpu, "q_must_be_cuda"),
        (q, FakeGpuOffsets(3, device=FakeDevice("cpu", None)), "gpu_cu_seqlens_must_be_cuda"),
        (q, FakeGpuOffsets(3, device=FakeDevice("cuda", 1)), "gpu_cu_seqlens_must_share_q_device"),
        (q, FakeGpuOffsets(3, dtype="torch.int32"), "gpu_cu_seqlens_must_be_int64"),
        (q, FakeGpuOffsets(3, shape=(1, 3)), "gpu_cu_seqlens_must_be_rank1"),
        (q, FakeGpuOffsets(3, contiguous=False), "gpu_cu_seqlens_must_be_contiguous"),
        (q, FakeGpuOffsets(2), "gpu_cu_seqlens_numel_mismatch_cpu_descriptor"),
    )
    for bad_q, bad_gpu, expected in failures:
        _assert_error(expected, lambda bad_q=bad_q, bad_gpu=bad_gpu: metadata.validate_gpu_descriptor(bad_q, bad_gpu, cpu, descriptor))
    _assert_error(
        "q_tokens_must_match_cpu_descriptor",
        lambda: metadata.validate_gpu_descriptor(FakeQ(tokens=9), gpu, cpu, descriptor),
    )


class FakeStream:
    def __init__(self) -> None:
        self.waited_events: list[FakeEvent] = []

    def wait_event(self, event: "FakeEvent") -> None:
        self.waited_events.append(event)


class FakeEvent:
    def __init__(self) -> None:
        self.recorded_streams: list[FakeStream] = []

    def record(self, stream: FakeStream) -> None:
        self.recorded_streams.append(stream)


class FakeCudaDeviceContext:
    def __init__(self, cuda: "FakeCuda", device: FakeDevice) -> None:
        self.cuda = cuda
        self.device = device
        self.previous_device: FakeDevice | None = None

    def __enter__(self) -> None:
        self.previous_device = self.cuda.current_device
        self.cuda.current_device = self.device

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert self.previous_device is not None
        self.cuda.current_device = self.previous_device


class FakeCachedTensor:
    def __init__(self, device: FakeDevice) -> None:
        self.device = device
        self.recorded_streams: list[FakeStream] = []

    def record_stream(self, stream: FakeStream) -> None:
        self.recorded_streams.append(stream)


class FakeCpuStaging:
    def __init__(self, runtime: "FakeTorch") -> None:
        self.runtime = runtime

    def to(self, *, device: FakeDevice, dtype: str, non_blocking: bool) -> FakeCachedTensor:
        assert dtype == self.runtime.int64
        assert non_blocking is True
        self.runtime.h2d_calls += 1
        self.runtime.last_cached = FakeCachedTensor(device)
        return self.runtime.last_cached


class FakeCuda:
    def __init__(self, runtime: "FakeTorch") -> None:
        self.runtime = runtime
        self.capturing = False
        self.current_device = FakeDevice("cuda", 0)
        self.capture_by_device: dict[int, bool] = {}

    def current_stream(self, *, device: FakeDevice) -> FakeStream:
        assert device is self.runtime.expected_device
        return self.runtime.stream

    def is_current_stream_capturing(self) -> bool:
        return self.capture_by_device.get(int(self.current_device.index), self.capturing)

    def device(self, device: FakeDevice) -> FakeCudaDeviceContext:
        return FakeCudaDeviceContext(self, device)

    def Event(self, *, enable_timing: bool) -> FakeEvent:
        assert enable_timing is False
        event = FakeEvent()
        self.runtime.events.append(event)
        return event


class FakeTorch:
    int64 = "torch.int64"

    def __init__(self, expected_device: FakeDevice) -> None:
        self.expected_device = expected_device
        self.stream = FakeStream()
        self.cuda = FakeCuda(self)
        self.tuple_sources: list[tuple[int, ...]] = []
        self.h2d_calls = 0
        self.events: list[FakeEvent] = []
        self.last_cached: FakeCachedTensor | None = None

    def tensor(self, offsets: tuple[int, ...], *, dtype: str, device: str) -> FakeCpuStaging:
        assert dtype == self.int64
        assert device == "cpu"
        assert isinstance(offsets, tuple)
        self.tuple_sources.append(offsets)
        return FakeCpuStaging(self)


class BlockingCpuStaging(FakeCpuStaging):
    def to(self, *, device: FakeDevice, dtype: str, non_blocking: bool) -> FakeCachedTensor:
        self.runtime.copy_started.set()
        if not self.runtime.release_copy.wait(timeout=5):
            raise AssertionError("test did not release blocked H2D copy")
        return super().to(device=device, dtype=dtype, non_blocking=non_blocking)


class BlockingFakeTorch(FakeTorch):
    def __init__(self, expected_device: FakeDevice) -> None:
        super().__init__(expected_device)
        self.copy_started = threading.Event()
        self.release_copy = threading.Event()

    def tensor(self, offsets: tuple[int, ...], *, dtype: str, device: str) -> BlockingCpuStaging:
        assert dtype == self.int64
        assert device == "cpu"
        assert isinstance(offsets, tuple)
        self.tuple_sources.append(offsets)
        return BlockingCpuStaging(self)


def test_cache_miss_hit_stream_order_and_stats() -> None:
    metadata.clear_cache()
    q, cpu, descriptor = _issued()
    gpu = FakeGpuOffsets(3)
    runtime = FakeTorch(q.device)

    first = metadata.cached_gpu_offsets(q, gpu, cpu, descriptor, torch_module=runtime)
    assert first.cache_hit is False
    assert runtime.tuple_sources == [(0, 3, 8)]
    assert runtime.h2d_calls == 1
    assert len(runtime.events) == 1
    assert runtime.events[0].recorded_streams == [runtime.stream]
    assert first.tensor.recorded_streams == [runtime.stream]
    assert metadata.cache_stats() == {"entries": 1, "hits": 0, "misses": 1, "capture_miss_rejections": 0, "capture_hit_rejections": 0}

    hit_stream = FakeStream()
    runtime.stream = hit_stream
    second = metadata.cached_gpu_offsets(q, gpu, cpu, descriptor, torch_module=runtime)
    assert second.cache_hit is True
    assert second.tensor is first.tensor
    assert hit_stream.waited_events == runtime.events
    assert second.tensor.recorded_streams == [runtime.events[0].recorded_streams[0], hit_stream]
    assert metadata.cache_stats() == {"entries": 1, "hits": 1, "misses": 1, "capture_miss_rejections": 0, "capture_hit_rejections": 0}

    metadata.clear_cache()
    assert metadata.cache_stats() == {"entries": 0, "hits": 0, "misses": 0, "capture_miss_rejections": 0, "capture_hit_rejections": 0}


def test_capture_miss_is_fail_closed_before_h2d() -> None:
    metadata.clear_cache()
    q, cpu, descriptor = _issued()
    runtime = FakeTorch(q.device)
    runtime.cuda.capturing = True
    try:
        metadata.cached_gpu_offsets(q, FakeGpuOffsets(3), cpu, descriptor, torch_module=runtime)
    except metadata.CaptureCacheMissError as exc:
        assert str(exc) == "packed_varlen_cache_miss_during_cuda_graph_capture"
    else:  # pragma: no cover - assertion failure branch
        raise AssertionError("cache miss was incorrectly allowed during graph capture")
    assert runtime.tuple_sources == []
    assert runtime.h2d_calls == 0
    assert metadata.cache_stats() == {"entries": 0, "hits": 0, "misses": 0, "capture_miss_rejections": 1, "capture_hit_rejections": 0}


def test_capture_probe_selects_q_device_in_multi_gpu_process() -> None:
    metadata.clear_cache()
    q = FakeQ(device=FakeDevice("cuda", 1))
    cpu = FakeCpuOffsets([0, 3, 8])
    descriptor = metadata.issue_descriptor(q, cpu, opt_in=True)
    runtime = FakeTorch(q.device)
    assert runtime.cuda.current_device.index == 0
    runtime.cuda.capture_by_device = {0: False, 1: True}
    try:
        metadata.cached_gpu_offsets(
            q,
            FakeGpuOffsets(3, device=q.device),
            cpu,
            descriptor,
            torch_module=runtime,
        )
    except metadata.CaptureCacheMissError as exc:
        assert str(exc) == "packed_varlen_cache_miss_during_cuda_graph_capture"
    else:  # pragma: no cover - assertion failure branch
        raise AssertionError("q.device capture was missed because another CUDA device was current")
    assert runtime.cuda.current_device.index == 0
    assert runtime.h2d_calls == 0
    assert metadata.cache_stats() == {"entries": 0, "hits": 0, "misses": 0, "capture_miss_rejections": 1, "capture_hit_rejections": 0}


def test_capture_hit_is_fail_closed_before_wait_or_record_stream() -> None:
    metadata.clear_cache()
    q, cpu, descriptor = _issued()
    gpu = FakeGpuOffsets(3)
    runtime = FakeTorch(q.device)
    cached = metadata.cached_gpu_offsets(q, gpu, cpu, descriptor, torch_module=runtime)
    original_recorded_streams = list(cached.tensor.recorded_streams)
    capture_stream = FakeStream()
    runtime.stream = capture_stream
    runtime.cuda.capturing = True
    try:
        metadata.cached_gpu_offsets(q, gpu, cpu, descriptor, torch_module=runtime)
    except metadata.CaptureCacheHitError as exc:
        assert str(exc) == "packed_varlen_cache_hit_during_cuda_graph_capture"
    else:  # pragma: no cover - assertion failure branch
        raise AssertionError("cache hit was incorrectly allowed during graph capture")
    assert capture_stream.waited_events == []
    assert cached.tensor.recorded_streams == original_recorded_streams
    assert metadata.cache_stats() == {"entries": 1, "hits": 0, "misses": 1, "capture_miss_rejections": 0, "capture_hit_rejections": 1}
    metadata.clear_cache()


def test_concurrent_same_key_publishes_one_miss_without_deadlock() -> None:
    metadata.clear_cache()
    q, cpu, descriptor = _issued()
    gpu = FakeGpuOffsets(3)
    runtime = BlockingFakeTorch(q.device)
    results: list[metadata.CachedOffsets] = []
    failures: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(metadata.cached_gpu_offsets(q, gpu, cpu, descriptor, torch_module=runtime))
        except BaseException as exc:  # pragma: no cover - captured for main-thread assertion
            failures.append(exc)

    first = threading.Thread(target=worker, name="varlen-cache-first")
    second = threading.Thread(target=worker, name="varlen-cache-second")
    first.start()
    assert runtime.copy_started.wait(timeout=5)
    second.start()
    runtime.release_copy.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive(), "same-key cache call deadlocked"
    assert failures == []
    assert len(results) == 2
    assert sorted(result.cache_hit for result in results) == [False, True]
    assert runtime.h2d_calls == 1
    assert runtime.tuple_sources == [(0, 3, 8)]
    assert metadata.cache_stats() == {"entries": 1, "hits": 1, "misses": 1, "capture_miss_rejections": 0, "capture_hit_rejections": 0}


if __name__ == "__main__":
    tests = [object_ for name, object_ in sorted(globals().items()) if name.startswith("test_") and callable(object_)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} CPU-only varlen metadata tests")
