"""CPU-only policy tests for the C1 B300 FlashKDA dispatcher.

Run directly with ``python test_auto_dispatch_policy.py`` or with pytest.
No CUDA, torch, FLA installation, or extension is required.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
from typing import Any

# Support both ``python test_auto_dispatch_policy.py`` and repository-root pytest.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch
from assignment02.team.c1_flashkda.challenge_tp8_dispatch import fla_backend
from assignment02.team.c1_flashkda.challenge_tp8_dispatch import varlen_metadata


class FakeDevice:
    def __init__(self, device_type: str = "cuda", index: int = 0) -> None:
        self.type = device_type
        self.index = index


class FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: str,
        device: FakeDevice | None = None,
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device or FakeDevice()
        self._contiguous = contiguous

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self._contiguous

    def numel(self) -> int:
        product = 1
        for dimension in self.shape:
            product *= dimension
        return product

    def view(self, *shape: int) -> "FakeTensor":
        inferred = list(shape)
        if inferred.count(-1) > 1:
            raise ValueError("only one inferred dimension is supported")
        if -1 in inferred:
            known = 1
            for dimension in inferred:
                if dimension != -1:
                    known *= dimension
            inferred[inferred.index(-1)] = self.numel() // known
        return FakeTensor(tuple(inferred), self.dtype, self.device, self._contiguous)


class FakeCpuOffsets:
    def __init__(self, values: tuple[int, ...]) -> None:
        self._values = values
        self.shape = (len(values),)
        self.dtype = "torch.int64"
        self.device = SimpleNamespace(type="cpu", index=None)
        self.tolist_calls = 0

    def is_contiguous(self) -> bool:
        return True

    def numel(self) -> int:
        return len(self._values)

    def tolist(self) -> list[int]:
        self.tolist_calls += 1
        return list(self._values)


class NoReadGpuOffsets(FakeTensor):
    def __init__(self, count: int, device: FakeDevice | None = None) -> None:
        super().__init__((count,), "torch.int64", device=device)
        self.value_reads = 0

    def tolist(self) -> list[int]:
        self.value_reads += 1
        raise AssertionError("GPU offsets values must not be read")


VALID_DEVICE = auto_dispatch.DeviceMetadata("NVIDIA B300", (10, 3), 148)


def _inputs(
    heads: int = 12,
    *,
    batch: int = 1,
    tokens: int = 8192,
    initial: FakeTensor | None | object = ...,
    final: FakeTensor | None | object = ...,
    **overrides: Any,
) -> dict[str, Any]:
    q_shape = (batch, tokens, heads, 128)
    state_shape = (batch, heads, 128, 128)
    bf16 = lambda shape: FakeTensor(shape, "torch.bfloat16")
    fp32 = lambda shape: FakeTensor(shape, "torch.float32")
    if initial is ...:
        initial = bf16(state_shape)
    if final is ...:
        final = bf16(state_shape)
    values: dict[str, Any] = {
        "q": bf16(q_shape),
        "k": bf16(q_shape),
        "v": bf16(q_shape),
        "g": bf16(q_shape),
        "beta": bf16(q_shape[:-1]),
        "out": bf16(q_shape),
        "A_log": fp32((heads,)),
        "dt_bias": fp32((heads, 128)),
        "scale": 128 ** -0.5,
        "lower_bound": -5.0,
        "initial_state": initial,
        "final_state": final,
        "cu_seqlens": None,
    }
    values.update(overrides)
    return values


def _select(heads: int = 12, metadata: auto_dispatch.DeviceMetadata = VALID_DEVICE, **kwargs: Any):
    return auto_dispatch.select_variant(metadata, **_inputs(heads, **kwargs))


VARLEN_EXPECTED = {
    (0, 2048, 4096): ("baseline", "baseline", "baseline"),
    (0, 2048, 4096, 6144, 8192): ("baseline", "baseline", "baseline"),
    (0, 17, 528, 1552, 2852, 4901, 8192): ("baseline", "baseline", "baseline"),
    (0, 1, 2, 3, 4, 5, 12288): ("vshard2_p2", "vshard2_p2", "vshard4_p2"),
}
VARLEN_RELEASED = {
    ((0, 1, 2, 3, 4, 5, 12288), "none"): "vshard2_p2",
    ((0, 1, 2, 3, 4, 5, 12288), "fp32_final_only"): "vshard2_p2",
    ((0, 1, 2, 3, 4, 5, 12288), "fp32_both"): "vshard4_p2",
}


def _varlen_inputs(offsets: tuple[int, ...], contract: str) -> dict[str, Any]:
    sequences = len(offsets) - 1
    state = FakeTensor((sequences, 12, 128, 128), "torch.float32")
    initial = state if contract == "fp32_both" else None
    final = state if contract in ("fp32_final_only", "fp32_both") else None
    return _inputs(
        12,
        batch=1,
        tokens=offsets[-1],
        initial=initial,
        final=final,
        cu_seqlens=NoReadGpuOffsets(len(offsets)),
        certified_varlen_offsets=offsets,
    )


def test_positive_boundaries() -> None:
    assert _select(1).chosen_variant == "vshard4_p2"
    assert _select(37).chosen_variant == "vshard4_p2"
    assert _select(38).chosen_variant == "vshard2_p2"
    assert _select(96).chosen_variant == "vshard2_p2"


def test_real_fla_h12_state_contracts_are_explicitly_whitelisted() -> None:
    fp32_h12 = FakeTensor((1, 12, 128, 128), "torch.float32")
    assert _select(12, initial=None, final=None).chosen_variant == "vshard4_p2"
    assert _select(12, initial=fp32_h12, final=fp32_h12).chosen_variant == "vshard4_p2"
    assert _select(12, initial=None, final=fp32_h12).chosen_variant == "vshard4_p2"


def test_cross_length_entries_are_exactly_whitelisted() -> None:
    for tokens in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
        assert _select(12, tokens=tokens).chosen_variant == "vshard4_p2"
    for tokens in (2048, 32768):
        assert _select(1, tokens=tokens).chosen_variant == "vshard4_p2"
        assert _select(37, tokens=tokens).chosen_variant == "vshard4_p2"
        assert _select(38, tokens=tokens).chosen_variant == "vshard2_p2"
        assert _select(64, tokens=tokens).chosen_variant == "vshard2_p2"
        assert _select(96, tokens=tokens).chosen_variant == "vshard2_p2"
    assert _select(12, tokens=257).reason == "state_contract_bf16_both_length_head_not_whitelisted"
    assert _select(11, tokens=4096).reason == "state_contract_bf16_both_length_head_not_whitelisted"


def test_tail8191_release_is_exactly_state_scoped() -> None:
    fp32 = FakeTensor((1, 12, 128, 128), "torch.float32")
    none = _select(12, tokens=8191, initial=None, final=None)
    final_only = _select(12, tokens=8191, initial=None, final=fp32)
    assert none.chosen_variant == "vshard4_p2"
    assert none.reason == "fixed_single_batch_b1_h12_t8191_none_whitelist_hit"
    assert final_only.chosen_variant == "vshard4_p2"
    assert final_only.reason == "fixed_single_batch_b1_h12_t8191_fp32_final_only_whitelist_hit"

    # These two state contracts were explicit negative controls in both
    # release allocations and must not inherit the positive tail decision.
    fp32_both = _select(12, tokens=8191, initial=fp32, final=fp32)
    bf16_both = _select(12, tokens=8191)
    assert fp32_both.chosen_variant == "baseline"
    assert fp32_both.reason == "state_contract_fp32_both_h12_length_not_whitelisted"
    assert bf16_both.chosen_variant == "baseline"
    assert bf16_both.reason == "state_contract_bf16_both_length_head_not_whitelisted"

    # Shape neighbors remain governed by their own evidence, never by a
    # plausible-looking tail generalization.
    assert _select(11, tokens=8191, initial=None, final=None).chosen_variant == "baseline"
    assert _select(12, tokens=8190, initial=None, final=None).chosen_variant == "baseline"
    assert _select(batch=2, tokens=8191, initial=None, final=None).chosen_variant == "baseline"


def test_exact_fixed_batch_public_contract_entries_and_fallbacks() -> None:
    def fp32_state(batch: int) -> FakeTensor:
        return FakeTensor((batch, 12, 128, 128), "torch.float32")

    for batch in (2, 3):
        state = fp32_state(batch)
        assert _select(batch=batch, tokens=2048, initial=None, final=None).chosen_variant == "vshard4_p2"
        assert _select(batch=batch, tokens=2048, initial=None, final=state).chosen_variant == "vshard4_p2"
        assert _select(batch=batch, tokens=2048, initial=state, final=state).chosen_variant == "vshard4_p2"

    for batch in (4, 6):
        state = fp32_state(batch)
        assert _select(batch=batch, tokens=2048, initial=None, final=None).chosen_variant == "vshard2_p2"
        assert _select(batch=batch, tokens=2048, initial=None, final=state).chosen_variant == "vshard2_p2"
        both = _select(batch=batch, tokens=2048, initial=state, final=state)
        assert both.chosen_variant == "baseline"
        assert both.reason == f"fixed_batch_b{batch}_fp32_both_not_whitelisted"

    batch = 5
    state = fp32_state(batch)
    assert _select(batch=batch, tokens=2048, initial=None, final=None).chosen_variant == "vshard2_p2"
    assert _select(batch=batch, tokens=2048, initial=None, final=state).chosen_variant == "vshard2_p2"
    assert _select(batch=batch, tokens=2048, initial=state, final=state).chosen_variant == "vshard2_p2"

    assert _select(batch=7, tokens=2048, initial=None, final=None).reason == "fixed_batch_shape_not_whitelisted"
    assert _select(batch=8, tokens=2048, initial=None, final=None).reason == "fixed_batch_shape_not_whitelisted"
    assert _select(batch=2, tokens=8192, initial=None, final=None).reason == "fixed_batch_shape_not_whitelisted"
    assert _select(batch=2, tokens=2048).reason == "fixed_batch_requires_exact_fla_public_state_contract"
    bad_state = FakeTensor((1, 12, 128, 128), "torch.float32")
    assert _select(batch=2, tokens=2048, initial=None, final=bad_state).reason == "fixed_batch_requires_exact_fla_public_state_contract"


def test_all_device_whitelist_dimensions_fail_closed() -> None:
    assert _select(metadata=auto_dispatch.DeviceMetadata("NVIDIA H100", (10, 3), 148)).reason == "device_name_not_b300"
    assert _select(metadata=auto_dispatch.DeviceMetadata("B300", (10, 2), 148)).reason == "device_capability_not_sm103"
    assert _select(metadata=auto_dispatch.DeviceMetadata("B300", (10, 3), 147)).reason == "device_sm_count_not_148"
    assert _select(97).reason == "head_count_outside_h1_to_h96"
    assert _select(0).reason == "head_count_outside_h1_to_h96"


def test_all_tensor_contract_misses_fail_closed() -> None:
    assert _select(q=FakeTensor((2, 8192, 12, 128), "torch.bfloat16")).reason == "fixed_batch_shape_not_whitelisted"
    assert _select(q=FakeTensor((1, 8192, 12, 64), "torch.bfloat16")).reason == "shape_requires_positive_batch_k128"
    values = _inputs(12)
    values["v"] = FakeTensor((1, 8192, 12, 64), "torch.bfloat16")
    assert auto_dispatch.select_variant(VALID_DEVICE, **values).reason == "q_k_v_g_out_shape_mismatch"
    assert _select(beta=FakeTensor((1, 8192, 11), "torch.bfloat16")).reason == "beta_shape_mismatch"
    assert _select(A_log=FakeTensor((11,), "torch.float32")).reason == "A_log_shape_mismatch"
    assert _select(dt_bias=FakeTensor((12, 64), "torch.float32")).reason == "dt_bias_shape_mismatch"
    assert _select(v=FakeTensor((1, 8192, 12, 128), "torch.float32")).reason == "q_k_v_g_beta_out_must_be_bf16"
    assert _select(A_log=FakeTensor((12,), "torch.bfloat16")).reason == "A_log_and_dt_bias_must_be_fp32"
    assert _select(out=FakeTensor((1, 8192, 12, 128), "torch.bfloat16", contiguous=False)).reason == "tensor_device_or_contiguity_mismatch"
    assert _select(g=FakeTensor((1, 8192, 12, 128), "torch.bfloat16", device=FakeDevice("cpu"))).reason == "tensor_device_or_contiguity_mismatch"
    assert _select(cu_seqlens=object()).reason == "varlen_cpu_descriptor_not_certified"
    assert _select(scale=float("nan")).reason == "scale_must_be_finite"
    assert _select(lower_bound=float("inf")).reason == "lower_bound_must_be_finite"


def test_only_released_varlen_public_cells_are_whitelisted() -> None:
    contracts = ("none", "fp32_final_only", "fp32_both")
    assert auto_dispatch._VARLEN_PUBLIC_VARIANTS == VARLEN_RELEASED
    observed: set[tuple[tuple[int, ...], str]] = set()
    for offsets, expected_variants in VARLEN_EXPECTED.items():
        for contract, expected in zip(contracts, expected_variants):
            decision = auto_dispatch.select_variant(VALID_DEVICE, **_varlen_inputs(offsets, contract))
            assert decision.chosen_variant == expected, (offsets, contract, decision)
            key = (offsets, contract)
            if expected == "baseline":
                assert decision.requested_variant == "baseline"
                layout = auto_dispatch._VARLEN_LAYOUT_NAMES[offsets]
                assert decision.reason == f"varlen_{layout}_{contract}_not_whitelisted"
            else:
                assert key in VARLEN_RELEASED
                observed.add(key)
    assert observed == set(VARLEN_RELEASED)

    unlisted = (0, 1024, 4096)
    decision = auto_dispatch.select_variant(VALID_DEVICE, **_varlen_inputs(unlisted, "none"))
    assert decision.chosen_variant == "baseline"
    assert decision.reason == "varlen_offsets_not_whitelisted"

    values = _varlen_inputs((0, 2048, 4096), "none")
    values["certified_varlen_offsets"] = None
    assert auto_dispatch.select_variant(VALID_DEVICE, **values).reason == "varlen_cpu_descriptor_not_certified"
    values = _inputs(12, batch=1, tokens=4096, initial=None, final=None)
    values["certified_varlen_offsets"] = (0, 2048, 4096)
    assert auto_dispatch.select_variant(VALID_DEVICE, **values).reason == "varlen_certificate_without_cu_seqlens"


def test_varlen_workspace_uses_authenticated_sequence_count() -> None:
    calls: list[tuple[int, int, int]] = []
    launches: list[str] = []
    extension = SimpleNamespace()
    extension.get_workspace_size = lambda total, heads, sequences: (
        calls.append((total, heads, sequences)) or 64
    )
    extension.fwd_vshard_p2 = lambda *args, **kwargs: launches.append("v2")
    extension.fwd_vshard4_p2 = lambda *args, **kwargs: launches.append("v4")
    original_import = auto_dispatch.importlib.import_module
    try:
        auto_dispatch.importlib.import_module = lambda name: (
            SimpleNamespace(uint8="uint8", empty=lambda *args, **kwargs: object())
            if name == "torch"
            else original_import(name)
        )
        values = _varlen_inputs((0, 2048, 4096, 6144, 8192), "none")
        values.pop("certified_varlen_offsets")
        auto_dispatch._launch_sharded("vshard2_p2", extension, **values)
        assert calls == [(8192, 12, 4)]
        assert launches == ["v2"]
    finally:
        auto_dispatch.importlib.import_module = original_import


def _handoff_inputs() -> tuple[dict[str, Any], FakeCpuOffsets, Any, varlen_metadata.CacheKey, dict[str, Any]]:
    offsets = (0, 1, 2, 3, 4, 5, 12288)
    values = _varlen_inputs(offsets, "none")
    cpu = FakeCpuOffsets(offsets)
    descriptor = varlen_metadata.issue_descriptor(values["q"], cpu, opt_in=True)
    key = varlen_metadata.validate_gpu_descriptor(values["q"], values["cu_seqlens"], cpu, descriptor)
    call = {
        "q": values["q"],
        "k": values["k"],
        "v": values["v"],
        "g": values["g"],
        "beta": values["beta"],
        "scale": values["scale"],
        "initial_state": None,
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "allow_neg_eigval": False,
        "safe_gate": True,
        "lower_bound": values["lower_bound"],
        "disable_recompute": False,
        "return_intermediate_states": False,
        "state_v_first": True,
        "cu_seqlens": values["cu_seqlens"],
        "cu_seqlens_cpu": cpu,
        "cp_context": None,
        "A_log": values["A_log"],
        "dt_bias": values["dt_bias"],
        "extra_kwargs": {},
    }
    return values, cpu, descriptor, key, call


def _store_handoff(backend: fla_backend.C1B300FlashKDABackend, descriptor: Any, key: varlen_metadata.CacheKey, call: dict[str, Any]) -> bool:
    return backend._store_varlen_handoff(descriptor, key, **call)


def test_varlen_handoff_is_one_shot_thread_local_and_fresh() -> None:
    backend = fla_backend.C1B300FlashKDABackend()
    _, cpu, descriptor, key, call = _handoff_inputs()
    original_env = os.environ.get("C1_B300_VARLEN_CPU_DESCRIPTOR")
    try:
        os.environ["C1_B300_VARLEN_CPU_DESCRIPTOR"] = "1"
        assert _store_handoff(backend, descriptor, key, call) is True
        plan = backend._take_varlen_handoff(**call)
        assert plan is not None
        assert plan.descriptor is descriptor and plan.key is key
        # The CPU tuple is freshly reread exactly once at handoff consumption.
        assert cpu.tolist_calls == 2
        assert backend._take_varlen_handoff(**call) is None

        assert _store_handoff(backend, descriptor, key, call) is True
        observed: list[Any] = []
        worker = threading.Thread(
            target=lambda: observed.append(backend._take_varlen_handoff(**call)),
            name="c1-varlen-handoff-other-thread",
        )
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert observed == [None]
        # The other thread cannot consume the main thread's plan.
        assert backend._take_varlen_handoff(**call) is not None
    finally:
        if original_env is None:
            os.environ.pop("C1_B300_VARLEN_CPU_DESCRIPTOR", None)
        else:
            os.environ["C1_B300_VARLEN_CPU_DESCRIPTOR"] = original_env


def test_varlen_handoff_refuses_noncanonical_deferred_scalar_values() -> None:
    backend = fla_backend.C1B300FlashKDABackend()
    _, _, descriptor, key, call = _handoff_inputs()
    invalid_calls = (
        dict(call, scale=FakeTensor((1,), "torch.float32")),
        dict(call, lower_bound=FakeTensor((1,), "torch.float32")),
        dict(call, scale=True),
        dict(call, output_final_state=1),
        dict(call, use_gate_in_kernel=1),
    )
    for invalid in invalid_calls:
        assert _store_handoff(backend, descriptor, key, invalid) is False
        assert not hasattr(backend._handoff_local(), "plan")

    # None and exact builtin numerics are small, safe fingerprint values.  A
    # real preflight may reject a None lower bound separately; this test only
    # locks the handoff storage boundary.
    assert _store_handoff(backend, descriptor, key, dict(call, scale=None, lower_bound=None)) is True
    backend._clear_varlen_handoff()


def test_varlen_handoff_mismatch_or_cpu_mutation_forces_full_prepare_path() -> None:
    backend = fla_backend.C1B300FlashKDABackend()
    values, cpu, descriptor, key, call = _handoff_inputs()
    original_env = os.environ.get("C1_B300_VARLEN_CPU_DESCRIPTOR")
    try:
        os.environ["C1_B300_VARLEN_CPU_DESCRIPTOR"] = "1"
        assert _store_handoff(backend, descriptor, key, call) is True
        wrong_identity = dict(call, q=FakeTensor(values["q"].shape, "torch.bfloat16"))
        assert backend._take_varlen_handoff(**wrong_identity) is None

        assert _store_handoff(backend, descriptor, key, call) is True
        wrong_scalar = dict(call, scale=float(call["scale"]) * 2.0)
        assert backend._take_varlen_handoff(**wrong_scalar) is None

        assert _store_handoff(backend, descriptor, key, call) is True
        cpu._values = (0, 2, 3, 4, 5, 6, 12288)
        assert backend._take_varlen_handoff(**call) is None
        # A full prepare would issue a new descriptor from the currently valid
        # tuple, rather than reuse the verifier's old key.
        fresh = varlen_metadata.issue_descriptor(values["q"], cpu, opt_in=True)
        assert varlen_metadata.verify_descriptor(fresh, cpu).offsets == cpu._values

        cpu._values = (0, 2, 2, 4, 5, 6, 12288)
        assert _store_handoff(backend, fresh, varlen_metadata.CacheKey(0, (0, 2, 3, 4, 5, 6, 12288)), call) is True
        assert backend._take_varlen_handoff(**call) is None
        try:
            varlen_metadata.issue_descriptor(values["q"], cpu, opt_in=True)
        except varlen_metadata.MetadataError as exc:
            assert str(exc) == "cpu_descriptor_offsets_must_be_strictly_increasing"
        else:  # pragma: no cover - defensive failure branch
            raise AssertionError("invalid current CPU tuple unexpectedly passed full validation")
    finally:
        if original_env is None:
            os.environ.pop("C1_B300_VARLEN_CPU_DESCRIPTOR", None)
        else:
            os.environ["C1_B300_VARLEN_CPU_DESCRIPTOR"] = original_env


def test_varlen_handoff_body_reuses_only_the_descriptor_then_calls_auto_dispatch() -> None:
    backend = fla_backend.C1B300FlashKDABackend()
    values, _, descriptor, key, call = _handoff_inputs()
    original_env = os.environ.get("C1_B300_VARLEN_CPU_DESCRIPTOR")
    original_torch = sys.modules.get("torch")
    original_fwd = auto_dispatch.fwd
    original_prepare = backend._prepare_varlen
    out_buf = object()
    observed: dict[str, Any] = {}
    try:
        os.environ["C1_B300_VARLEN_CPU_DESCRIPTOR"] = "1"
        assert _store_handoff(backend, descriptor, key, call) is True
        sys.modules["torch"] = SimpleNamespace(empty_like=lambda value: out_buf)
        auto_dispatch.fwd = lambda *args, **kwargs: observed.update(kwargs)
        backend._prepare_varlen = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("matching handoff unexpectedly reran _prepare_varlen")
        )
        result = backend.chunk_kda(
            values["q"], values["k"], values["v"], values["g"], values["beta"],
            scale=call["scale"], initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, allow_neg_eigval=False, safe_gate=True,
            lower_bound=call["lower_bound"], disable_recompute=False,
            return_intermediate_states=False, state_v_first=True,
            cu_seqlens=values["cu_seqlens"], cu_seqlens_cpu=call["cu_seqlens_cpu"],
            A_log=values["A_log"], dt_bias=values["dt_bias"],
        )
        assert result == (out_buf, None)
        assert observed["_varlen_descriptor"] is descriptor
        assert observed["cu_seqlens"] is values["cu_seqlens"]
        # The handoff cannot select a variant itself; auto_dispatch remains
        # responsible for issuer/CPU identity, GPU structure, cache and ABI.
        assert "_varlen_key" not in observed
    finally:
        backend._prepare_varlen = original_prepare
        auto_dispatch.fwd = original_fwd
        if original_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original_torch
        if original_env is None:
            os.environ.pop("C1_B300_VARLEN_CPU_DESCRIPTOR", None)
        else:
            os.environ["C1_B300_VARLEN_CPU_DESCRIPTOR"] = original_env


def test_fixed_body_clears_handoff_without_calling_take() -> None:
    backend = fla_backend.C1B300FlashKDABackend()
    _, _, descriptor, key, handoff_call = _handoff_inputs()
    fixed = _inputs(12, initial=None, final=None)
    original_torch = sys.modules.get("torch")
    original_fwd = auto_dispatch.fwd
    original_take = backend._take_varlen_handoff
    original_clear = backend._clear_varlen_handoff
    calls: list[str] = []
    try:
        assert _store_handoff(backend, descriptor, key, handoff_call) is True
        backend._take_varlen_handoff = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("fixed path unexpectedly attempted handoff take")
        )

        def clear_spy() -> None:
            calls.append("clear")
            original_clear()

        backend._clear_varlen_handoff = clear_spy
        sys.modules["torch"] = SimpleNamespace(empty_like=lambda value: object())
        auto_dispatch.fwd = lambda *args, **kwargs: None
        backend.chunk_kda(
            fixed["q"], fixed["k"], fixed["v"], fixed["g"], fixed["beta"],
            scale=fixed["scale"], initial_state=None, output_final_state=False,
            lower_bound=fixed["lower_bound"], A_log=fixed["A_log"], dt_bias=fixed["dt_bias"],
        )
        assert calls == ["clear"]
        assert not hasattr(backend._handoff_local(), "plan")
    finally:
        backend._take_varlen_handoff = original_take
        backend._clear_varlen_handoff = original_clear
        auto_dispatch.fwd = original_fwd
        if original_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original_torch


def test_tail8191_fixed_public_body_preserves_production_dispatch_inputs() -> None:
    backend = fla_backend.C1B300FlashKDABackend()
    fixed = _inputs(12, tokens=8191, initial=None, final=None)
    original_torch = sys.modules.get("torch")
    original_fwd = auto_dispatch.fwd
    out_buf, final_buf = object(), object()
    observed: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    try:
        sys.modules["torch"] = SimpleNamespace(
            empty_like=lambda value: out_buf,
            empty=lambda *shape, **kwargs: final_buf,
            float32="torch.float32",
        )
        auto_dispatch.fwd = lambda *args, **kwargs: observed.append((args, kwargs))
        for output_final_state in (False, True):
            result = backend.chunk_kda(
                fixed["q"], fixed["k"], fixed["v"], fixed["g"], fixed["beta"],
                scale=fixed["scale"], initial_state=None,
                output_final_state=output_final_state,
                lower_bound=fixed["lower_bound"], A_log=fixed["A_log"],
                dt_bias=fixed["dt_bias"],
            )
            args, kwargs = observed[-1]
            assert args[:6] == (
                fixed["q"], fixed["k"], fixed["v"], fixed["g"],
                fixed["beta"], fixed["scale"],
            )
            assert args[6] is out_buf
            assert kwargs["A_log"] is fixed["A_log"]
            assert kwargs["dt_bias"] is fixed["dt_bias"]
            assert kwargs["lower_bound"] == fixed["lower_bound"]
            assert kwargs["initial_state"] is None
            assert kwargs["final_state"] is (final_buf if output_final_state else None)
            assert kwargs["cu_seqlens"] is None
            assert kwargs["cu_seqlens_cpu"] is None
            assert kwargs["_varlen_descriptor"] is None
            assert result == (out_buf, final_buf if output_final_state else None)
    finally:
        auto_dispatch.fwd = original_fwd
        if original_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original_torch


def test_verifier_entry_clears_a_previous_varlen_handoff() -> None:
    backend = fla_backend.C1B300FlashKDABackend()
    values, _, descriptor, key, call = _handoff_inputs()
    original_parent_verifier = fla_backend._PinnedFlashKDABackend.chunk_kda_verifier
    original_env = os.environ.get("C1_B300_FLASH_KDA")
    try:
        assert _store_handoff(backend, descriptor, key, call) is True
        fla_backend._PinnedFlashKDABackend.chunk_kda_verifier = lambda self, *args, **kwargs: (True, None)
        os.environ["C1_B300_FLASH_KDA"] = "1"
        # This non-varlen verifier call has no body, but it must still clear
        # a preceding same-thread varlen plan before returning.
        assert backend.chunk_kda_verifier(
            values["q"], values["k"], values["v"], values["g"], values["beta"]
        ) == (True, None)
        assert backend._take_varlen_handoff(**call) is None
    finally:
        fla_backend._PinnedFlashKDABackend.chunk_kda_verifier = original_parent_verifier
        if original_env is None:
            os.environ.pop("C1_B300_FLASH_KDA", None)
        else:
            os.environ["C1_B300_FLASH_KDA"] = original_env


def test_varlen_backend_preflight_is_explicit_and_never_reads_gpu_values() -> None:
    backend = fla_backend.C1B300FlashKDABackend()
    offsets = (0, 1, 2, 3, 4, 5, 12288)
    values = _varlen_inputs(offsets, "none")
    gpu = values["cu_seqlens"]
    cpu = FakeCpuOffsets(offsets)
    original_metadata = auto_dispatch._read_device_metadata
    original_parent_verifier = fla_backend._PinnedFlashKDABackend.chunk_kda_verifier
    original_general = os.environ.get("C1_B300_FLASH_KDA")
    original_varlen = os.environ.get("C1_B300_VARLEN_CPU_DESCRIPTOR")
    try:
        auto_dispatch._read_device_metadata = lambda q: VALID_DEVICE
        fla_backend._PinnedFlashKDABackend.chunk_kda_verifier = lambda self, *args, **kwargs: (True, None)
        os.environ["C1_B300_FLASH_KDA"] = "1"
        os.environ.pop("C1_B300_VARLEN_CPU_DESCRIPTOR", None)
        accepted, reason = backend.chunk_kda_verifier(
            q=values["q"], k=values["k"], v=values["v"], g=values["g"], beta=values["beta"],
            initial_state=None, output_final_state=False, cu_seqlens=gpu, cu_seqlens_cpu=cpu,
            lower_bound=values["lower_bound"], A_log=values["A_log"], dt_bias=values["dt_bias"],
        )
        assert accepted is False and "C1_B300_VARLEN_CPU_DESCRIPTOR" in str(reason)

        os.environ["C1_B300_VARLEN_CPU_DESCRIPTOR"] = "1"
        accepted, reason = backend.chunk_kda_verifier(
            q=values["q"], k=values["k"], v=values["v"], g=values["g"], beta=values["beta"],
            initial_state=None, output_final_state=False, cu_seqlens=gpu, cu_seqlens_cpu=cpu,
            lower_bound=values["lower_bound"], A_log=values["A_log"], dt_bias=values["dt_bias"],
        )
        assert (accepted, reason) == (True, None)
        assert gpu.value_reads == 0

        mixed = (0, 17, 528, 1552, 2852, 4901, 8192)
        mixed_values = _varlen_inputs(mixed, "fp32_both")
        accepted, reason = backend.chunk_kda_verifier(
            q=mixed_values["q"], k=mixed_values["k"], v=mixed_values["v"],
            g=mixed_values["g"], beta=mixed_values["beta"],
            initial_state=mixed_values["initial_state"], output_final_state=True,
            cu_seqlens=mixed_values["cu_seqlens"], cu_seqlens_cpu=FakeCpuOffsets(mixed),
            lower_bound=mixed_values["lower_bound"], A_log=mixed_values["A_log"],
            dt_bias=mixed_values["dt_bias"],
        )
        assert accepted is False and "not_whitelisted" in str(reason)
    finally:
        auto_dispatch._read_device_metadata = original_metadata
        fla_backend._PinnedFlashKDABackend.chunk_kda_verifier = original_parent_verifier
        if original_general is None:
            os.environ.pop("C1_B300_FLASH_KDA", None)
        else:
            os.environ["C1_B300_FLASH_KDA"] = original_general
        if original_varlen is None:
            os.environ.pop("C1_B300_VARLEN_CPU_DESCRIPTOR", None)
        else:
            os.environ["C1_B300_VARLEN_CPU_DESCRIPTOR"] = original_varlen


def test_unverified_state_contracts_select_baseline() -> None:
    assert _select(1, initial=None, final=None).reason == "state_contract_none_only_h12_whitelisted"
    assert _select(11, initial=None, final=None).reason == "state_contract_none_only_h12_whitelisted"
    assert _select(
        11,
        initial=None,
        final=FakeTensor((1, 11, 128, 128), "torch.float32"),
    ).reason == "state_contract_fla_fp32_final_only_only_h12_whitelisted"
    assert _select(
        11,
        initial=FakeTensor((1, 11, 128, 128), "torch.float32"),
        final=FakeTensor((1, 11, 128, 128), "torch.float32"),
    ).reason == "state_contract_fp32_both_only_h12_whitelisted"
    assert _select(
        initial=FakeTensor((1, 12, 128, 128), "torch.bfloat16"),
        final=FakeTensor((1, 12, 128, 64), "torch.bfloat16"),
    ).reason == "state_contract_requires_measured_bf16_both_or_h12_fla_contract"


def test_symbol_fallback_is_prelaunch_and_last_decision_is_observable() -> None:
    original_metadata = auto_dispatch._read_device_metadata
    original_symbols = auto_dispatch._load_extension_and_symbols
    original_launch = auto_dispatch._launch_variant
    calls: list[str] = []
    try:
        auto_dispatch._read_device_metadata = lambda q: VALID_DEVICE
        auto_dispatch._load_extension_and_symbols = lambda: (
            SimpleNamespace(), frozenset({"fwd_vshard_p2"}), "audited-test-sha", None
        )
        auto_dispatch._launch_variant = lambda variant, extension, **kwargs: calls.append(variant)
        tail_none = _inputs(12, tokens=8191, initial=None, final=None)
        assert auto_dispatch.fwd(**tail_none) is None
        assert calls == ["baseline"]
        decision = auto_dispatch.get_last_decision()
        assert decision["requested_variant"] == "vshard4_p2"
        assert decision["chosen_variant"] == "baseline"
        assert "fwd_vshard4_p2_missing_prelaunch_fallback_to_baseline" in str(decision["reason"])

        calls.clear()
        auto_dispatch._load_extension_and_symbols = lambda: (
            SimpleNamespace(), frozenset(), "audited-test-sha", None
        )
        fp32 = FakeTensor((1, 12, 128, 128), "torch.float32")
        auto_dispatch.fwd(**_inputs(12, tokens=8191, initial=None, final=fp32))
        assert calls == ["baseline"]
        decision = auto_dispatch.get_last_decision()
        assert decision["requested_variant"] == "vshard4_p2"
        assert decision["chosen_variant"] == "baseline"
        assert "missing_prelaunch_fallback_to_baseline" in str(decision["reason"])

        calls.clear()
        auto_dispatch.fwd(**_inputs(38))
        assert calls == ["baseline"]
        assert auto_dispatch.get_last_decision()["requested_variant"] == "vshard2_p2"
    finally:
        auto_dispatch._read_device_metadata = original_metadata
        auto_dispatch._load_extension_and_symbols = original_symbols
        auto_dispatch._launch_variant = original_launch


def test_kernel_exceptions_are_not_caught_or_replayed() -> None:
    original_metadata = auto_dispatch._read_device_metadata
    original_symbols = auto_dispatch._load_extension_and_symbols
    original_launch = auto_dispatch._launch_variant
    calls: list[str] = []
    try:
        auto_dispatch._read_device_metadata = lambda q: VALID_DEVICE
        auto_dispatch._load_extension_and_symbols = lambda: (
            SimpleNamespace(),
            frozenset({"fwd_vshard4_p2", "fwd_vshard_p2"}),
            "audited-test-sha",
            None,
        )

        def failing_launch(variant: str, extension: object, **kwargs: Any) -> None:
            calls.append(variant)
            raise RuntimeError("synthetic kernel failure")

        auto_dispatch._launch_variant = failing_launch
        try:
            auto_dispatch.fwd(**_inputs(12))
        except RuntimeError as exc:
            assert str(exc) == "synthetic kernel failure"
        else:  # pragma: no cover - defensive failure branch
            raise AssertionError("kernel exception was incorrectly swallowed")
        assert calls == ["vshard4_p2"]
        assert auto_dispatch.get_last_decision()["chosen_variant"] == "vshard4_p2"
    finally:
        auto_dispatch._read_device_metadata = original_metadata
        auto_dispatch._load_extension_and_symbols = original_symbols
        auto_dispatch._launch_variant = original_launch


def test_wrong_extension_sha_fails_closed_before_launch() -> None:
    original_import = auto_dispatch.importlib.import_module
    original_metadata = auto_dispatch._read_device_metadata
    original_launch = auto_dispatch._launch_variant
    calls: list[str] = []
    with tempfile.TemporaryDirectory() as directory:
        fake_so = Path(directory) / "flash_kda_C.so"
        fake_so.write_bytes(b"symbol-compatible but unaudited test binary")
        fake_extension = SimpleNamespace(
            __file__=str(fake_so),
            fwd_vshard4_p2=lambda: None,
            fwd_vshard_p2=lambda: None,
        )
        try:
            auto_dispatch._load_extension_and_symbols.cache_clear()
            auto_dispatch.importlib.import_module = (
                lambda name: fake_extension if name == "flash_kda_C" else original_import(name)
            )
            auto_dispatch._read_device_metadata = lambda q: VALID_DEVICE
            auto_dispatch._launch_variant = (
                lambda variant, extension, **kwargs: calls.append(variant)
            )
            assert auto_dispatch.fwd(**_inputs(12)) is None
            assert calls == ["baseline"]
            decision = auto_dispatch.get_last_decision()
            assert decision["requested_variant"] == "vshard4_p2"
            assert decision["chosen_variant"] == "baseline"
            assert "extension_sha256_not_allowlisted" in str(decision["reason"])
            assert decision["extension_sha256"] not in auto_dispatch._AUDITED_EXTENSION_SHA256
        finally:
            auto_dispatch.importlib.import_module = original_import
            auto_dispatch._read_device_metadata = original_metadata
            auto_dispatch._launch_variant = original_launch
            auto_dispatch._load_extension_and_symbols.cache_clear()


def test_opt_in_backend_registration_is_idempotent() -> None:
    original_loader = fla_backend._load_kda_registry
    original_registered = fla_backend._REGISTERED_BACKEND
    original_import_error = fla_backend._PINNED_BACKEND_IMPORT_ERROR
    original_env = os.environ.get("C1_B300_FLASH_KDA")
    registry = SimpleNamespace(backends=[])
    registry.register = registry.backends.append
    registry._get_sorted_backends = lambda: list(registry.backends)
    try:
        fla_backend._REGISTERED_BACKEND = None
        fla_backend._PINNED_BACKEND_IMPORT_ERROR = None
        fla_backend._load_kda_registry = lambda: registry
        os.environ.pop("C1_B300_FLASH_KDA", None)
        backend = fla_backend.C1B300FlashKDABackend()
        policy_values = _inputs(12)
        assert backend.priority < backend.__class__.__mro__[1].priority
        assert backend.chunk_kda_verifier(
            policy_values["q"], policy_values["k"], policy_values["v"],
            policy_values["g"], policy_values["beta"],
        )[0] is False
        os.environ["C1_B300_FLASH_KDA"] = "1"
        if backend.__class__.__mro__[1].__module__.startswith("fla."):
            # When pinned FLA is installed, exercise its real verifier rather
            # than the torch-free fallback double defined by fla_backend.py.
            import torch

            q = torch.empty((1, 1, 1, 128), dtype=torch.bfloat16)
            beta = torch.empty((1, 1, 1), dtype=torch.bfloat16)
            with torch.inference_mode():
                assert backend.chunk_kda_verifier(
                    q,
                    q,
                    q,
                    q,
                    beta,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True,
                    state_v_first=True,
                    safe_gate=True,
                )[0] is True
        else:
            assert backend.chunk_kda_verifier(
                policy_values["q"], policy_values["k"], policy_values["v"],
                policy_values["g"], policy_values["beta"],
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                state_v_first=True,
                safe_gate=True,
                lower_bound=policy_values["lower_bound"],
            )[0] is True
        first = fla_backend.register_backend()
        second = fla_backend.register_backend()
        assert first is second
        assert registry.backends == [first]
    finally:
        fla_backend._load_kda_registry = original_loader
        fla_backend._REGISTERED_BACKEND = original_registered
        fla_backend._PINNED_BACKEND_IMPORT_ERROR = original_import_error
        if original_env is None:
            os.environ.pop("C1_B300_FLASH_KDA", None)
        else:
            os.environ["C1_B300_FLASH_KDA"] = original_env


def test_registration_fails_closed_on_pinned_import_or_registry_conflict() -> None:
    original_loader = fla_backend._load_kda_registry
    original_registered = fla_backend._REGISTERED_BACKEND
    original_import_error = fla_backend._PINNED_BACKEND_IMPORT_ERROR
    loader_calls: list[str] = []
    try:
        fla_backend._REGISTERED_BACKEND = None
        fla_backend._PINNED_BACKEND_IMPORT_ERROR = ImportError("synthetic pinned dependency failure")
        fla_backend._load_kda_registry = lambda: loader_calls.append("called")
        try:
            fla_backend.register_backend()
        except RuntimeError as exc:
            assert "real pinned FLA FlashKDA backend failed to import" in str(exc)
        else:  # pragma: no cover - defensive failure branch
            raise AssertionError("registration fail-opened after pinned import failure")
        assert loader_calls == []

        fla_backend._PINNED_BACKEND_IMPORT_ERROR = None
        compatible = SimpleNamespace(
            backend_type="c1_b300_flash_kda",
            registration_compatibility_token=fla_backend.C1B300FlashKDABackend.registration_compatibility_token,
            priority=2,
            env_var="C1_B300_FLASH_KDA",
            varlen_env_var="C1_B300_VARLEN_CPU_DESCRIPTOR",
        )
        registry = SimpleNamespace(backends=[compatible])
        registry._get_sorted_backends = lambda: list(registry.backends)
        registry.register = registry.backends.append
        fla_backend._load_kda_registry = lambda: registry
        assert fla_backend.register_backend() is compatible
        assert registry.backends == [compatible]

        fla_backend._REGISTERED_BACKEND = None
        registry.backends = [SimpleNamespace(backend_type="c1_b300_flash_kda")]
        try:
            fla_backend.register_backend()
        except RuntimeError as exc:
            assert "incompatible C1 backend" in str(exc)
        else:  # pragma: no cover - defensive failure branch
            raise AssertionError("registration reused an incompatible reload-era backend")

        registry.backends = [compatible, compatible]
        try:
            fla_backend.register_backend()
        except RuntimeError as exc:
            assert "multiple C1 backend entries" in str(exc)
        else:  # pragma: no cover - defensive failure branch
            raise AssertionError("registration accepted duplicate C1 backend entries")
    finally:
        fla_backend._load_kda_registry = original_loader
        fla_backend._REGISTERED_BACKEND = original_registered
        fla_backend._PINNED_BACKEND_IMPORT_ERROR = original_import_error


def test_public_positional_order_and_allow_neg_eigval_guard() -> None:
    backend = fla_backend.C1B300FlashKDABackend()
    values = _inputs(12)
    original_parent_verifier = fla_backend._PinnedFlashKDABackend.chunk_kda_verifier
    original_env = os.environ.get("C1_B300_FLASH_KDA")
    observed: dict[str, Any] = {}
    try:
        def parent_verifier(self, **kwargs: Any) -> tuple[bool, str | None]:
            observed.update(kwargs)
            return True, None

        fla_backend._PinnedFlashKDABackend.chunk_kda_verifier = parent_verifier
        os.environ["C1_B300_FLASH_KDA"] = "1"
        public_positionals = (
            values["q"], values["k"], values["v"], values["g"], values["beta"],
            values["scale"], values["initial_state"], False,
            True, True, True, True, True, values["lower_bound"], False, False, True,
        )
        assert backend.chunk_kda_verifier(*public_positionals) == (True, None)
        assert observed["use_qk_l2norm_in_kernel"] is True
        assert observed["use_gate_in_kernel"] is True
        assert observed["use_beta_sigmoid_in_kernel"] is True
        assert observed["safe_gate"] is True
        assert observed["state_v_first"] is True
        try:
            backend.chunk_kda(*public_positionals)
        except ValueError as exc:
            assert "C1 safety guard blocked allow_neg_eigval=True" in str(exc)
        else:  # pragma: no cover - defensive failure branch
            raise AssertionError("allow_neg_eigval guard did not stop the unsupported raw kernel")
    finally:
        fla_backend._PinnedFlashKDABackend.chunk_kda_verifier = original_parent_verifier
        if original_env is None:
            os.environ.pop("C1_B300_FLASH_KDA", None)
        else:
            os.environ["C1_B300_FLASH_KDA"] = original_env


def test_verifier_uses_canonical_signature_after_instance_spy_install() -> None:
    backend = fla_backend.C1B300FlashKDABackend()
    values = _inputs(12)
    original_parent_verifier = fla_backend._PinnedFlashKDABackend.chunk_kda_verifier
    original_env = os.environ.get("C1_B300_FLASH_KDA")
    original_implementation = backend.chunk_kda
    try:
        fla_backend._PinnedFlashKDABackend.chunk_kda_verifier = (
            lambda self, *args, **kwargs: (True, None)
        )
        os.environ["C1_B300_FLASH_KDA"] = "1"

        def registry_spy(*args: Any, **kwargs: Any):
            return original_implementation(*args, **kwargs)

        backend.chunk_kda = registry_spy  # type: ignore[method-assign]
        accepted, reason = backend.chunk_kda_verifier(
            values["q"], values["k"], values["v"], values["g"], values["beta"],
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=True,
            safe_gate=True,
            lower_bound=values["lower_bound"],
        )
        assert (accepted, reason) == (True, None)
    finally:
        fla_backend._PinnedFlashKDABackend.chunk_kda_verifier = original_parent_verifier
        if original_env is None:
            os.environ.pop("C1_B300_FLASH_KDA", None)
        else:
            os.environ["C1_B300_FLASH_KDA"] = original_env


def test_public_pre_dispatch_errors_are_guarded_before_pinned_fallback() -> None:
    backend = fla_backend.C1B300FlashKDABackend()
    values = _inputs(12)
    original_parent_verifier = fla_backend._PinnedFlashKDABackend.chunk_kda_verifier
    original_env = os.environ.get("C1_B300_FLASH_KDA")
    try:
        fla_backend._PinnedFlashKDABackend.chunk_kda_verifier = (
            lambda self, *args, **kwargs: (True, None)
        )
        os.environ["C1_B300_FLASH_KDA"] = "1"
        common = dict(
            q=values["q"], k=values["k"], v=values["v"], g=values["g"], beta=values["beta"],
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, state_v_first=True, safe_gate=True,
        )
        for extra, expected in (
            ({"lower_bound": -5.01}, "safe range [-5, 0)"),
            ({"lower_bound": 0.0}, "safe range [-5, 0)"),
            ({"lower_bound": -5.0, "chunk_size": 17}, "chunk_size"),
            ({"lower_bound": -5.0, "transpose_state_layout": True}, "Cannot pass both"),
        ):
            call = {**common, **extra}
            assert backend.chunk_kda_verifier(**call) == (True, None)
            try:
                backend.chunk_kda(**call)
            except ValueError as exc:
                assert expected in str(exc)
            else:  # pragma: no cover - defensive failure branch
                raise AssertionError(f"public fail-stop guard did not reject {extra}")
    finally:
        fla_backend._PinnedFlashKDABackend.chunk_kda_verifier = original_parent_verifier
        if original_env is None:
            os.environ.pop("C1_B300_FLASH_KDA", None)
        else:
            os.environ["C1_B300_FLASH_KDA"] = original_env


def test_last_decision_is_thread_local_diagnostic_state() -> None:
    auto_dispatch._record(auto_dispatch.DispatchDecision("main", "baseline", "main"))
    barrier = threading.Barrier(3)
    observed: dict[str, dict[str, object]] = {}

    def worker(name: str) -> None:
        auto_dispatch._record(auto_dispatch.DispatchDecision(name, "baseline", name))
        barrier.wait(timeout=5)
        observed[name] = auto_dispatch.get_last_decision()
        barrier.wait(timeout=5)

    first = threading.Thread(target=worker, args=("first",))
    second = threading.Thread(target=worker, args=("second",))
    first.start()
    second.start()
    barrier.wait(timeout=5)
    assert auto_dispatch.get_last_decision()["requested_variant"] == "main"
    barrier.wait(timeout=5)
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert observed["first"]["requested_variant"] == "first"
    assert observed["second"]["requested_variant"] == "second"


if __name__ == "__main__":
    import inspect

    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} CPU-only dispatcher tests")
