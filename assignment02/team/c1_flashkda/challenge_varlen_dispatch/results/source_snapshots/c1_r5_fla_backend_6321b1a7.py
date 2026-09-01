"""Opt-in FLA backend that delegates its raw ABI call to :mod:`auto_dispatch`.

This module intentionally does not modify FLA's registry at import time.  A
caller must set ``C1_B300_FLASH_KDA=1`` and explicitly call
``register_backend()``.  That keeps the experiment isolated from normal FLA
users and makes fallback behavior auditable.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any

try:
    # Repository-root imports, used by tests and the documented harnesses.
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch, varlen_metadata
except ModuleNotFoundError:  # pragma: no cover - installed/standalone FLA path
    try:
        # ``A02_ROOT`` may itself be the ``assignment02`` directory on B300.
        from team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch, varlen_metadata  # type: ignore[no-redef]
    except ModuleNotFoundError:
        try:
            from challenge_tp8_dispatch import auto_dispatch, varlen_metadata  # type: ignore[no-redef]
        except ModuleNotFoundError:
            # A harness may instead add this experiment directory directly.
            import auto_dispatch  # type: ignore[no-redef]
            import varlen_metadata  # type: ignore[no-redef]

if TYPE_CHECKING:
    import torch
    from fla.ops.cp import FLACPContext


_PINNED_BACKEND_IMPORT_ERROR: ImportError | None = None
try:  # Production path: reuse the pinned FLA verifier and its input contract.
    from fla.ops.kda.backends.flash_kda import FlashKDABackend as _PinnedFlashKDABackend
except ImportError as exc:  # CPU policy tests intentionally run without torch/FLA installed.
    _PINNED_BACKEND_IMPORT_ERROR = exc

    class _PinnedFlashKDABackend:  # type: ignore[no-redef]
        backend_type = "flash_kda"
        package_name = "flash_kda"
        env_var = "FLA_FLASH_KDA"
        default_enable = True
        priority = 3

        def chunk_kda_verifier(self, *args: Any, **kwargs: Any) -> tuple[bool, str | None]:
            return True, None


class C1B300FlashKDABackend(_PinnedFlashKDABackend):
    """Pinned FlashKDA verifier plus an explicit C1 B300 dispatch experiment."""

    backend_type = "c1_b300_flash_kda"
    env_var = "C1_B300_FLASH_KDA"
    default_enable = False
    # FLA sorts lower numeric priorities first.  The pinned FlashKDA backend
    # uses priority 3, so the opt-in dispatcher must be tried before it.
    priority = 2
    varlen_env_var = "C1_B300_VARLEN_CPU_DESCRIPTOR"
    # A registry entry from a previous module instance is reusable only when
    # this exact semantic generation matches.  Change the token whenever the
    # verifier, implementation, public contract, or dispatch policy changes.
    registration_compatibility_token = "c1-b300-flash-kda-varlen-20260829-v1"

    class _PlannedFinalState:
        """Metadata-only stand-in used by the verifier; it never allocates CUDA."""

        dtype = "torch.float32"

        def __init__(self, shape: tuple[int, ...], device: Any) -> None:
            self.shape = shape
            self.device = device

        def is_contiguous(self) -> bool:
            return True

    @staticmethod
    def _public_fail_stop_reason(arguments: dict[str, Any]) -> str | None:
        """Mirror public pre-dispatch errors that the pinned backend omits."""

        if bool(arguments.get("allow_neg_eigval")):
            return (
                "C1 safety guard blocked allow_neg_eigval=True: the pinned/raw "
                "FlashKDA kernel implements sigmoid(beta), not 2*sigmoid(beta)"
            )
        if bool(arguments.get("safe_gate")) and bool(arguments.get("use_gate_in_kernel")):
            lower_bound = arguments.get("lower_bound")
            if lower_bound is None:
                return "`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`."
            if not isinstance(lower_bound, (int, float)) or not (-5 <= float(lower_bound) < 0):
                return f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}."
        chunk_size = arguments.get("chunk_size", 64)
        if chunk_size not in (32, 64):
            return f"`chunk_size` must be either 32 or 64 for KDA, got {chunk_size}."
        if "transpose_state_layout" in arguments and bool(arguments.get("state_v_first")):
            return "Cannot pass both `state_v_first` and the deprecated `transpose_state_layout`."
        return None

    def _prepare_varlen(
        self,
        *,
        q: Any,
        k: Any,
        v: Any,
        g: Any,
        beta: Any,
        scale: float | None,
        initial_state: Any | None,
        output_final_state: bool,
        cu_seqlens: Any,
        cu_seqlens_cpu: Any,
        lower_bound: float | None,
        A_log: Any | None,
        dt_bias: Any | None,
    ) -> tuple[Any, Any, auto_dispatch.DispatchDecision]:
        """Authenticate CPU authority and prove this exact public cell is custom-eligible."""

        if os.environ.get(self.varlen_env_var) != "1":
            raise varlen_metadata.MetadataError(
                f"set {self.varlen_env_var}=1 to opt into CPU-authoritative packed varlen"
            )
        descriptor = varlen_metadata.issue_descriptor(q, cu_seqlens_cpu, opt_in=True)
        key = varlen_metadata.validate_gpu_descriptor(
            q, cu_seqlens, cu_seqlens_cpu, descriptor
        )
        if scale is None:
            scale = q.shape[-1] ** -0.5
        if lower_bound is None:
            raise varlen_metadata.MetadataError("varlen_requires_explicit_lower_bound")
        if A_log is None or dt_bias is None:
            raise varlen_metadata.MetadataError("varlen_requires_A_log_and_dt_bias")
        if getattr(dt_bias, "ndim", len(getattr(dt_bias, "shape", ()))) == 1:
            dt_bias = dt_bias.view(v.shape[2], -1)
        final_state = None
        if output_final_state:
            final_state = self._PlannedFinalState(
                (len(key.offsets) - 1, v.shape[2], q.shape[-1], v.shape[-1]), q.device
            )
        decision = auto_dispatch.select_variant(
            auto_dispatch._read_device_metadata(q),
            q,
            k,
            v,
            g,
            beta,
            v,
            A_log,
            dt_bias,
            scale,
            lower_bound,
            initial_state,
            final_state,
            cu_seqlens,
            key.offsets,
        )
        if decision.chosen_variant == "baseline":
            raise varlen_metadata.MetadataError(decision.reason)
        return descriptor, key, decision

    def chunk_kda_verifier(
        self,
        q: "torch.Tensor",
        k: "torch.Tensor",
        v: "torch.Tensor",
        g: "torch.Tensor",
        beta: "torch.Tensor",
        scale: float | None = None,
        initial_state: "torch.Tensor | None" = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        state_v_first: bool = False,
        cu_seqlens: "torch.LongTensor | None" = None,
        cu_seqlens_cpu: "torch.LongTensor | None" = None,
        cp_context: "FLACPContext | None" = None,
        **kwargs: Any,
    ) -> tuple[bool, str | None]:
        """Use the public positional order and forward to pinned by name.

        FLA and the three source files are SHA-pinned by the clean integration
        runner.  An explicit signature avoids per-call reflection overhead and
        remains stable when registry instrumentation shadows the instance's
        ``chunk_kda`` implementation with a spy.
        """

        if os.environ.get(self.env_var) != "1":
            return False, f"set {self.env_var}=1 and call register_backend() to opt in"
        accepted, reason = super().chunk_kda_verifier(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_intermediate_states,
            cp_context=cp_context,
            **kwargs,
        )
        if not accepted:
            return accepted, reason
        arguments = {
            "q": q,
            "k": k,
            "v": v,
            "g": g,
            "beta": beta,
            "scale": scale,
            "initial_state": initial_state,
            "output_final_state": output_final_state,
            "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
            "use_gate_in_kernel": use_gate_in_kernel,
            "use_beta_sigmoid_in_kernel": use_beta_sigmoid_in_kernel,
            "allow_neg_eigval": allow_neg_eigval,
            "safe_gate": safe_gate,
            "lower_bound": lower_bound,
            "disable_recompute": disable_recompute,
            "return_intermediate_states": return_intermediate_states,
            "state_v_first": state_v_first,
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_cpu": cu_seqlens_cpu,
            "cp_context": cp_context,
            **kwargs,
        }
        # Intentionally accept public fail-stop cases as a guard: returning
        # False would let the same under-validating pinned backend take over at
        # the next registry priority. chunk_kda() raises before any work.
        if self._public_fail_stop_reason(arguments) is not None:
            return True, None
        if cu_seqlens is None:
            return True, None
        try:
            self._prepare_varlen(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=bool(output_final_state),
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                lower_bound=lower_bound,
                A_log=arguments.get("A_log"),
                dt_bias=arguments.get("dt_bias"),
            )
        except (varlen_metadata.MetadataError, AttributeError, TypeError, ValueError) as exc:
            return False, f"C1 packed-varlen preflight rejected: {exc}"
        return True, None

    def chunk_kda(
        self,
        q: "torch.Tensor",
        k: "torch.Tensor",
        v: "torch.Tensor",
        g: "torch.Tensor",
        beta: "torch.Tensor",
        scale: float | None = None,
        initial_state: "torch.Tensor | None" = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        state_v_first: bool = False,
        cu_seqlens: "torch.LongTensor | None" = None,
        cu_seqlens_cpu: "torch.LongTensor | None" = None,
        cp_context: "FLACPContext | None" = None,
        *,
        A_log: "torch.Tensor | None" = None,
        dt_bias: "torch.Tensor | None" = None,
        **kwargs: Any,
    ):
        """Preserve the pinned backend's tensor and return contract exactly."""

        public_arguments = {
            "allow_neg_eigval": allow_neg_eigval,
            "safe_gate": safe_gate,
            "use_gate_in_kernel": use_gate_in_kernel,
            "lower_bound": lower_bound,
            "state_v_first": state_v_first,
            **kwargs,
        }
        fail_stop_reason = self._public_fail_stop_reason(public_arguments)
        if fail_stop_reason is not None:
            raise ValueError(fail_stop_reason)

        import torch

        if scale is None:
            scale = q.shape[-1] ** -0.5
        if lower_bound is None:
            raise ValueError("FlashKDA backend requires an explicit lower_bound")
        if A_log is None or dt_bias is None:
            raise ValueError("FlashKDA backend requires A_log and dt_bias")

        key_dim = q.shape[-1]
        value_heads, value_dim = v.shape[2], v.shape[-1]

        if dt_bias.ndim == 1:
            dt_bias = dt_bias.view(value_heads, -1)

        out_buf = torch.empty_like(v)
        if initial_state is not None:
            initial_state = initial_state.contiguous()

        varlen_descriptor = None
        varlen_key = None
        if cu_seqlens is not None:
            varlen_descriptor, varlen_key, _ = self._prepare_varlen(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                lower_bound=lower_bound,
                A_log=A_log,
                dt_bias=dt_bias,
            )
        num_sequences = len(varlen_key.offsets) - 1 if varlen_key is not None else q.shape[0]

        final_state = None
        if output_final_state:
            final_state = torch.empty(
                num_sequences,
                value_heads,
                key_dim,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            )
        auto_dispatch.fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            out_buf,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            _varlen_descriptor=varlen_descriptor,
        )
        return out_buf, final_state


_REGISTERED_BACKEND: C1B300FlashKDABackend | None = None
_REGISTER_LOCK = threading.Lock()


def _load_kda_registry() -> Any:
    from fla.ops.kda.backends import kda_registry

    return kda_registry


def register_backend() -> C1B300FlashKDABackend:
    """Register this backend once and return the process-wide backend instance.

    The caller is responsible for setting ``C1_B300_FLASH_KDA=1`` before using
    it.  Registration itself is idempotent so notebook/module reload callers
    cannot accidentally add duplicate registry entries.
    """

    global _REGISTERED_BACKEND
    with _REGISTER_LOCK:
        if _PINNED_BACKEND_IMPORT_ERROR is not None:
            raise RuntimeError(
                "cannot register C1 backend because the real pinned FLA FlashKDA backend failed to import"
            ) from _PINNED_BACKEND_IMPORT_ERROR
        registry = _load_kda_registry()
        getter = getattr(registry, "_get_sorted_backends", None)
        if not callable(getter):
            raise RuntimeError("FLA KDA registry lacks auditable _get_sorted_backends()")
        matches = [
            backend
            for backend in getter()
            if getattr(backend, "backend_type", None) == C1B300FlashKDABackend.backend_type
        ]
        if len(matches) > 1:
            raise RuntimeError("FLA KDA registry already contains multiple C1 backend entries")
        if len(matches) == 1:
            existing = matches[0]
            compatible = (
                getattr(existing, "registration_compatibility_token", None)
                == C1B300FlashKDABackend.registration_compatibility_token
                and getattr(existing, "priority", None) == C1B300FlashKDABackend.priority
                and getattr(existing, "env_var", None) == C1B300FlashKDABackend.env_var
                and getattr(existing, "varlen_env_var", None)
                == C1B300FlashKDABackend.varlen_env_var
            )
            if not compatible:
                raise RuntimeError("FLA KDA registry contains an incompatible C1 backend entry")
            if _REGISTERED_BACKEND is not None and existing is not _REGISTERED_BACKEND:
                raise RuntimeError("module-local C1 backend identity disagrees with the FLA registry")
            _REGISTERED_BACKEND = existing
            return existing
        if _REGISTERED_BACKEND is not None:
            raise RuntimeError("module-local C1 backend exists but is absent from the FLA registry")
        backend = C1B300FlashKDABackend()
        registry.register(backend)
        _REGISTERED_BACKEND = backend
        return backend


__all__ = ["C1B300FlashKDABackend", "register_backend"]
