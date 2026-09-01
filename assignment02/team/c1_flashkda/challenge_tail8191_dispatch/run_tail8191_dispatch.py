#!/usr/bin/env python3
"""Two-allocation, fail-closed public FLA experiment for B=1,H=12,T=8191.

This file is intentionally an experiment harness, not a production dispatcher.
Inside this process only it replaces the *module attribute* consumed by the
already-audited C1 FLA backend with an exact-shape vshard4-P2 route.  It restores
that attribute before exit and never writes a production source file.  The two
public paths timed here are therefore the real pinned ``fla.ops.kda.chunk_kda``
API with the C1 backend disabled/enabled, rather than a raw-wrapper proxy.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DIM = 128
BATCH = 1
HEADS = 12
TOKENS = 8191
CONTRACTS = ("none", "fp32_final_only")
PERCENTILES = ("p50", "p95", "p99")
PATHS = ("pinned_public", "c1_test_route_public")
SAMPLES = 1000
REPEATS = 2
WARMUP = 100
MIN_MARGIN = 0.02
CLEAN_ENV = "C1_TAIL8191_DISPATCH_CLEAN_GPU"
EXPECTED_EXTENSION_SHA256 = "8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_PATCHED_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_REFERENCE_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT = "a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
EXPECTED_AUTO_DISPATCH_SHA256 = "f7ad41d6368e82dc75ed2a384542ee527f5487f38a001b054f25840855327b45"
EXPECTED_FLA_BACKEND_SHA256 = "3cd5ce30fb7869cca13131bc6255b6ec0cf2f9eaa86ac2a20d2fa7d9b0709342"
EXPECTED_HARNESS_SHA256 = "5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"
EXPECTED_FLASH_KDA_PYTHON_SHA256 = "9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
FLA_FILE_SHA256 = {
    "fla/__init__.py": "b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d",
    "fla/ops/backends/__init__.py": "a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635",
    "fla/ops/kda/__init__.py": "24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb",
    "fla/ops/kda/backends/__init__.py": "86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797",
    "fla/ops/kda/backends/flash_kda.py": "0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2",
    "fla/ops/kda/chunk.py": "a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lo = int(point)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def _summary(values: list[float]) -> dict[str, float | int]:
    if len(values) != SAMPLES or not all(math.isfinite(value) and value > 0.0 for value in values):
        raise AssertionError("CUDA-event sample count/values drift")
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _make_inputs(seed: int) -> Any:
    import torch
    import torch.nn.functional as functional

    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (BATCH, TOKENS, HEADS, DIM)
    # Match the pinned reference's normal input domain; all tensors are
    # contiguous and use independent generator draws.
    q = functional.normalize(torch.randn(shape, device="cuda", generator=generator), p=2, dim=-1).to(torch.bfloat16)
    k = functional.normalize(torch.randn(shape, device="cuda", generator=generator), p=2, dim=-1).to(torch.bfloat16)
    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator).contiguous(),
        "g": torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator).contiguous(),
        "beta": torch.randn((BATCH, TOKENS, HEADS), dtype=torch.bfloat16, device="cuda", generator=generator).contiguous(),
        "A_log": torch.rand(HEADS, dtype=torch.float32, device="cuda", generator=generator),
        "dt_bias": torch.rand((HEADS, DIM), dtype=torch.float32, device="cuda", generator=generator).contiguous(),
        "scale": 1.0 / math.sqrt(DIM),
        "lower_bound": -5.0,
    }


def _states(contract: str) -> tuple[Any | None, Any | None]:
    import torch
    if contract == "none":
        return None, None
    if contract != "fp32_final_only":
        raise ValueError(f"unsupported public contract: {contract}")
    return None, torch.zeros((BATCH, HEADS, DIM, DIM), dtype=torch.float32, device="cuda")


def _snapshot_inputs(x: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value.clone() if hasattr(value, "clone") else value for name, value in x.items()}


def _assert_inputs_unchanged(snapshot: Mapping[str, Any], x: Mapping[str, Any], label: str) -> dict[str, object]:
    import torch
    fields: dict[str, bool] = {}
    for name, before in snapshot.items():
        after = x[name]
        fields[name] = bool(torch.equal(before, after)) if hasattr(before, "shape") else before == after
    if not all(fields.values()):
        raise AssertionError(f"{label}: input mutation in {[name for name, ok in fields.items() if not ok]}")
    return {"input_immutability_exact": True, "input_immutability_fields": fields}


def _assert_initial_unchanged(before: Any | None, after: Any | None, label: str) -> dict[str, object]:
    import torch
    if before is None:
        if after is not None:
            raise AssertionError(f"{label}: unexpected initial state")
    elif not torch.equal(before, after):
        raise AssertionError(f"{label}: initial state mutation")
    return {"initial_state_immutability_exact": True}


def _invoke_raw(fn: Callable[..., Any], x: Mapping[str, Any], initial: Any | None, final: Any | None) -> tuple[Any, Any | None]:
    import torch
    out = torch.empty_like(x["v"])
    fn(
        x["q"], x["k"], x["v"], x["g"], x["beta"], x["scale"], out,
        A_log=x["A_log"], dt_bias=x["dt_bias"], lower_bound=x["lower_bound"],
        initial_state=initial, final_state=final, cu_seqlens=None,
    )
    torch.cuda.synchronize()
    return out, final


def _exact(label: str, actual: tuple[Any, Any | None], expected: tuple[Any, Any | None]) -> dict[str, object]:
    import torch
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common

    common.require_exact(f"{label}/output", actual[0], expected[0])
    evidence: dict[str, object] = {"output_exact": True, "output_max_abs": common.max_abs(actual[0], expected[0])}
    if actual[1] is None or expected[1] is None:
        if actual[1] is not None or expected[1] is not None:
            raise AssertionError(f"{label}: final-state presence mismatch")
        evidence["final_state_present"] = False
    else:
        if actual[1].dtype != torch.float32 or tuple(actual[1].shape) != (BATCH, HEADS, DIM, DIM) or not actual[1].is_contiguous():
            raise AssertionError(f"{label}: final-state ABI drift")
        common.require_exact(f"{label}/final_state", actual[1], expected[1])
        evidence.update({"final_state_present": True, "final_state_exact": True, "final_state_max_abs": common.max_abs(actual[1], expected[1])})
    return evidence


def _device_identity() -> dict[str, object]:
    import torch
    name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    if "B300" not in name.upper() or capability != (10, 3) or sm_count != 148:
        raise RuntimeError(f"B300-only run got name={name!r}, capability={capability}, SMs={sm_count}")
    return {"name": name, "capability": list(capability), "multiprocessor_count": sm_count, "gate_pass": True}


def _gpu_uuid() -> str:
    values = [line.strip() for line in subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], check=True, text=True, capture_output=True
    ).stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise RuntimeError(f"expected one visible GPU UUID, got {values!r}")
    return values[0]


def _identity(args: argparse.Namespace) -> dict[str, object]:
    import flash_kda
    import flash_kda_C
    import importlib
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch, fla_backend
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common

    runner = Path(__file__).resolve(strict=True)
    if _sha(runner) != args.expected_runner_sha256:
        raise RuntimeError("runner SHA mismatch")
    if _commit(args.patched_root) != EXPECTED_PATCHED_COMMIT or _commit(args.reference_root) != EXPECTED_REFERENCE_COMMIT or _commit(args.fla_root) != EXPECTED_FLA_COMMIT:
        raise RuntimeError("pinned git commit mismatch")
    ext_path = Path(flash_kda_C.__file__).resolve(strict=True)
    if not ext_path.is_relative_to(args.patched_root.resolve(strict=True)) or _sha(ext_path) != EXPECTED_EXTENSION_SHA256:
        raise RuntimeError("audited extension identity mismatch")
    required_symbols = ("fwd", "fwd_vshard4_p2", "get_workspace_size")
    if any(not callable(getattr(flash_kda_C, symbol, None)) for symbol in required_symbols):
        raise RuntimeError("extension lacks required tail8191 symbols")
    flash_path = Path(flash_kda.__file__).resolve(strict=True)
    if flash_path != (args.patched_root / "flash_kda" / "__init__.py").resolve(strict=True) or _sha(flash_path) != EXPECTED_FLASH_KDA_PYTHON_SHA256:
        raise RuntimeError("flash_kda Python identity mismatch")
    sources = {
        "auto_dispatch": REPO_ROOT / "assignment02/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py",
        "fla_backend": REPO_ROOT / "assignment02/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py",
        "harness": REPO_ROOT / "assignment02/team/c1_flashkda/harness/validate_and_bench.py",
    }
    expected = {"auto_dispatch": EXPECTED_AUTO_DISPATCH_SHA256, "fla_backend": EXPECTED_FLA_BACKEND_SHA256, "harness": EXPECTED_HARNESS_SHA256}
    source_shas = {name: _sha(path) for name, path in sources.items()}
    if source_shas != expected:
        raise RuntimeError(f"C1 source identity mismatch: {source_shas}")
    loaded_c1_modules = {
        "auto_dispatch": auto_dispatch,
        "fla_backend": fla_backend,
        "harness": common,
    }
    for name, module in loaded_c1_modules.items():
        loaded = Path(module.__file__).resolve(strict=True)
        if loaded != sources[name].resolve(strict=True):
            raise RuntimeError(f"loaded {name} module is outside the pinned source: {loaded}")
    fla_shas = {relative: _sha(args.fla_root / relative) for relative in FLA_FILE_SHA256}
    if fla_shas != FLA_FILE_SHA256:
        raise RuntimeError("pinned FLA source identity mismatch")
    fla_modules = {
        "fla": "fla/__init__.py",
        "fla.ops.backends": "fla/ops/backends/__init__.py",
        "fla.ops.kda": "fla/ops/kda/__init__.py",
        "fla.ops.kda.backends": "fla/ops/kda/backends/__init__.py",
        "fla.ops.kda.backends.flash_kda": "fla/ops/kda/backends/flash_kda.py",
        "fla.ops.kda.chunk": "fla/ops/kda/chunk.py",
    }
    for module_name, relative in fla_modules.items():
        loaded = Path(importlib.import_module(module_name).__file__).resolve(strict=True)
        expected_path = (args.fla_root / relative).resolve(strict=True)
        if loaded != expected_path:
            raise RuntimeError(f"loaded FLA module is outside FLA_ROOT: {module_name} -> {loaded}")
    reference = (args.reference_root / "tests" / "torch_ref.py").resolve(strict=True)
    return {
        "runner": {"path": str(runner), "sha256": _sha(runner), "gate_pass": True},
        "device": _device_identity(), "gpu_uuid": _gpu_uuid(),
        "extension": {"path": str(ext_path), "sha256": _sha(ext_path), "required_symbols": list(required_symbols), "gate_pass": True},
        "flash_kda_python": {"path": str(flash_path), "sha256": _sha(flash_path), "gate_pass": True},
        "commits": {"patched": EXPECTED_PATCHED_COMMIT, "reference": EXPECTED_REFERENCE_COMMIT, "fla": EXPECTED_FLA_COMMIT},
        "c1_sources": {name: {"path": str(sources[name]), "sha256": source_shas[name]} for name in sources},
        "fla_sources": fla_shas,
        "reference_torch_ref": {"path": str(reference), "sha256": _sha(reference)},
    }


def _route_contract(q: Any, k: Any, v: Any, g: Any, beta: Any, out: Any, A_log: Any, dt_bias: Any, initial: Any | None, final: Any | None, cu_seqlens: Any | None) -> str:
    import torch
    tensors = (q, k, v, g, out)
    if cu_seqlens is not None or any(tuple(t.shape) != (BATCH, TOKENS, HEADS, DIM) or t.dtype != torch.bfloat16 or not t.is_cuda or not t.is_contiguous() for t in tensors):
        raise RuntimeError("test-only route rejects non-exact tail8191 tensor contract")
    if tuple(beta.shape) != (BATCH, TOKENS, HEADS) or beta.dtype != torch.bfloat16 or not beta.is_cuda or not beta.is_contiguous():
        raise RuntimeError("test-only route rejects beta contract")
    if tuple(A_log.shape) != (HEADS,) or A_log.dtype != torch.float32 or tuple(dt_bias.shape) != (HEADS, DIM) or dt_bias.dtype != torch.float32:
        raise RuntimeError("test-only route rejects parameter contract")
    if initial is not None:
        raise RuntimeError("test-only route supports no initial state only")
    if final is None:
        return "none"
    if final.dtype != torch.float32 or tuple(final.shape) != (BATCH, HEADS, DIM, DIM) or not final.is_cuda or not final.is_contiguous():
        raise RuntimeError("test-only route rejects final-state contract")
    return "fp32_final_only"


@contextmanager
def _install_test_route() -> Iterator[Any]:
    """Install an in-memory exact route solely for the lifetime of this process."""
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch, fla_backend

    original = fla_backend.auto_dispatch.fwd
    if original is not auto_dispatch.fwd:
        raise RuntimeError("unexpected C1 auto_dispatch module identity")

    def route(q: Any, k: Any, v: Any, g: Any, beta: Any, scale: float, out: Any, A_log: Any, dt_bias: Any, lower_bound: float, initial_state: Any | None = None, final_state: Any | None = None, cu_seqlens: Any | None = None, cu_seqlens_cpu: Any | None = None, _varlen_descriptor: Any | None = None) -> Any:
        if cu_seqlens_cpu is not None or _varlen_descriptor is not None or not math.isfinite(float(scale)) or not math.isfinite(float(lower_bound)):
            raise RuntimeError("test-only route rejects descriptor/scalar drift")
        contract = _route_contract(q, k, v, g, beta, out, A_log, dt_bias, initial_state, final_state, cu_seqlens)
        extension, symbols, digest, rejection = auto_dispatch._load_extension_and_symbols()
        if rejection is not None or extension is None or "fwd_vshard4_p2" not in symbols or digest != EXPECTED_EXTENSION_SHA256:
            raise RuntimeError(f"test-only route refuses extension fallback: {rejection}, symbols={sorted(symbols)}")
        decision = auto_dispatch.DispatchDecision("vshard4_p2", "vshard4_p2", f"test_only_tail8191_h12_{contract}_exact_route")
        auto_dispatch._record(decision, extension_sha256=digest, test_only_route=True, production_source_mutated=False)
        return auto_dispatch._launch_sharded("vshard4_p2", extension, q=q, k=k, v=v, g=g, beta=beta, scale=scale, out=out, A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound, initial_state=initial_state, final_state=final_state, cu_seqlens=None)

    fla_backend.auto_dispatch.fwd = route
    try:
        yield auto_dispatch
    finally:
        fla_backend.auto_dispatch.fwd = original


def _public_kwargs(x: Mapping[str, Any], output_final_state: bool) -> dict[str, object]:
    return {
        "scale": x["scale"], "initial_state": None, "output_final_state": output_final_state,
        "use_qk_l2norm_in_kernel": True, "use_gate_in_kernel": True, "use_beta_sigmoid_in_kernel": True,
        "allow_neg_eigval": False, "state_v_first": True, "cu_seqlens": None, "cu_seqlens_cpu": None,
        "safe_gate": True, "lower_bound": x["lower_bound"], "disable_recompute": False,
        "return_intermediate_states": False, "cp_context": None, "A_log": x["A_log"], "dt_bias": x["dt_bias"],
    }


@contextmanager
def _c1_enabled(value: bool) -> Iterator[None]:
    prior = os.environ.get("C1_B300_FLASH_KDA")
    os.environ["C1_B300_FLASH_KDA"] = "1" if value else "0"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("C1_B300_FLASH_KDA", None)
        else:
            os.environ["C1_B300_FLASH_KDA"] = prior


def _registry_setup() -> tuple[Any, Any, Callable[..., Any], dict[str, int], dict[str, int], Callable[[], None]]:
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import fla_backend
    from fla.ops.kda.backends import kda_registry
    from fla.ops.kda import chunk_kda

    custom = fla_backend.register_backend()
    ordered = kda_registry._get_sorted_backends()
    c1 = [backend for backend in ordered if getattr(backend, "backend_type", None) == "c1_b300_flash_kda"]
    pinned = [backend for backend in ordered if getattr(backend, "backend_type", None) == "flash_kda"]
    if c1 != [custom] or len(pinned) != 1 or ordered.index(custom) >= ordered.index(pinned[0]):
        raise RuntimeError("FLA registry identity/priority gate failed")
    c1_counter, pinned_counter = {"calls": 0}, {"calls": 0}
    custom_original, pinned_original = custom.chunk_kda, pinned[0].chunk_kda
    def c1_spy(*args: object, **kwargs: object) -> object:
        c1_counter["calls"] += 1
        return custom_original(*args, **kwargs)
    def pinned_spy(*args: object, **kwargs: object) -> object:
        pinned_counter["calls"] += 1
        return pinned_original(*args, **kwargs)
    custom.chunk_kda, pinned[0].chunk_kda = c1_spy, pinned_spy
    def restore() -> None:
        custom.chunk_kda, pinned[0].chunk_kda = custom_original, pinned_original
    return custom, pinned[0], chunk_kda, c1_counter, pinned_counter, restore


def _public_call(public_fn: Callable[..., Any], x: Mapping[str, Any], contract: str, enable_c1: bool, c1_counter: Mapping[str, int], pinned_counter: Mapping[str, int], auto_dispatch: Any) -> tuple[tuple[Any, Any | None], dict[str, object]]:
    import torch
    c1_before, pinned_before = int(c1_counter["calls"]), int(pinned_counter["calls"])
    before_decision = auto_dispatch.get_last_decision()
    with _c1_enabled(enable_c1), torch.inference_mode():
        result = public_fn(x["q"], x["k"], x["v"], x["g"], x["beta"], **_public_kwargs(x, contract != "none"))
    torch.cuda.synchronize()
    c1_delta, pinned_delta = int(c1_counter["calls"]) - c1_before, int(pinned_counter["calls"]) - pinned_before
    if enable_c1:
        decision = auto_dispatch.get_last_decision()
        if c1_delta != 1 or pinned_delta != 0 or decision.get("chosen_variant") != "vshard4_p2" or decision.get("test_only_route") is not True:
            raise AssertionError(f"public C1 route proof failed: c1={c1_delta}, pinned={pinned_delta}, decision={decision}")
    else:
        decision = before_decision
        if c1_delta != 0 or pinned_delta != 1:
            raise AssertionError(f"public pinned route proof failed: c1={c1_delta}, pinned={pinned_delta}")
    return result, {"c1_spy_delta": c1_delta, "pinned_spy_delta": pinned_delta, "decision": decision, "passed": True}


def _raw_correctness(x: Mapping[str, Any], contract: str, torch_ref: Callable[..., Any]) -> dict[str, object]:
    import flash_kda
    import flash_kda_C

    snapshot = _snapshot_inputs(x)
    initial, final = _states(contract)
    initial_before = None if initial is None else initial.clone()
    reference = _invoke_raw(torch_ref, x, initial, final)
    reference_immutability = _assert_inputs_unchanged(snapshot, x, f"raw/{contract}/torch_reference")
    reference_immutability.update(_assert_initial_unchanged(initial_before, initial, f"raw/{contract}/torch_reference"))
    baseline_initial, baseline_final = _states(contract)
    baseline_initial_before = None if baseline_initial is None else baseline_initial.clone()
    baseline = _invoke_raw(flash_kda.fwd, x, baseline_initial, baseline_final)
    baseline_immutability = _assert_inputs_unchanged(snapshot, x, f"raw/{contract}/baseline")
    baseline_immutability.update(_assert_initial_unchanged(baseline_initial_before, baseline_initial, f"raw/{contract}/baseline"))
    def raw_v4(*args: Any, **kwargs: Any) -> None:
        import torch
        out = args[6]
        workspace = torch.empty(flash_kda_C.get_workspace_size(BATCH * TOKENS, HEADS, BATCH), dtype=torch.uint8, device=out.device)
        flash_kda_C.fwd_vshard4_p2(*args[:6], out, workspace, kwargs["A_log"], kwargs["dt_bias"], kwargs["lower_bound"], initial_state=kwargs.get("initial_state"), final_state=kwargs.get("final_state"), cu_seqlens=kwargs.get("cu_seqlens"))
    candidate_initial, candidate_final = _states(contract)
    candidate_initial_before = None if candidate_initial is None else candidate_initial.clone()
    candidate = _invoke_raw(raw_v4, x, candidate_initial, candidate_final)
    candidate_immutability = _assert_inputs_unchanged(snapshot, x, f"raw/{contract}/vshard4")
    candidate_immutability.update(_assert_initial_unchanged(candidate_initial_before, candidate_initial, f"raw/{contract}/vshard4"))
    return {
        "baseline_vs_pinned_torch_reference": _exact(f"raw/{contract}/baseline_vs_reference", baseline, reference),
        "vshard4_vs_pinned_torch_reference": _exact(f"raw/{contract}/vshard4_vs_reference", candidate, reference),
        "vshard4_vs_baseline": _exact(f"raw/{contract}/vshard4_vs_baseline", candidate, baseline),
        "immutability": {"reference": reference_immutability, "baseline": baseline_immutability, "vshard4": candidate_immutability},
        "passed": True,
    }


def _production_negative_controls() -> dict[str, object]:
    """Prove the source under test still rejects the two unmeasured contracts.

    This intentionally calls the *unmodified* production selector before the
    in-memory test route is installed.  It performs no launch and therefore
    establishes a pre-launch fail-closed boundary rather than a benchmark.
    """
    import torch
    from assignment02.team.c1_flashkda.challenge_tp8_dispatch import auto_dispatch

    x = _make_inputs(912_8191)
    out = torch.empty_like(x["v"])
    metadata = auto_dispatch._read_device_metadata(x["q"])
    cases: dict[str, object] = {}
    for contract in ("bf16_both", "fp32_both"):
        dtype = torch.bfloat16 if contract == "bf16_both" else torch.float32
        initial = torch.zeros((BATCH, HEADS, DIM, DIM), dtype=dtype, device="cuda")
        final = torch.zeros_like(initial)
        decision = auto_dispatch.select_variant(
            metadata, x["q"], x["k"], x["v"], x["g"], x["beta"], out, x["A_log"], x["dt_bias"],
            x["scale"], x["lower_bound"], initial, final, None, None,
        )
        if decision.requested_variant != "baseline" or decision.chosen_variant != "baseline":
            raise AssertionError(f"T8191 {contract} must remain production pre-launch baseline: {decision}")
        cases[contract] = {"requested_variant": decision.requested_variant, "chosen_variant": decision.chosen_variant, "reason": decision.reason, "passed": True}
    return {"production_source_unmodified": True, "negative_contracts": cases, "passed": True}


def _benchmark_repeat(process_index: int, repeat_index: int, x: Mapping[str, Any], contract: str, public_fn: Callable[..., Any], c1_counter: Mapping[str, int], pinned_counter: Mapping[str, int], auto_dispatch: Any) -> dict[str, object]:
    import torch
    snapshot = _snapshot_inputs(x)
    # The non-timed calls prove both real registry paths before sample collection.
    pinned, pinned_proof = _public_call(public_fn, x, contract, False, c1_counter, pinned_counter, auto_dispatch)
    c1, c1_proof = _public_call(public_fn, x, contract, True, c1_counter, pinned_counter, auto_dispatch)
    precheck = _exact(f"public/{process_index}/{repeat_index}/{contract}/c1_vs_pinned", c1, pinned)
    calls = {
        "pinned_public": lambda: _public_call(public_fn, x, contract, False, c1_counter, pinned_counter, auto_dispatch),
        "c1_test_route_public": lambda: _public_call(public_fn, x, contract, True, c1_counter, pinned_counter, auto_dispatch),
    }
    for warmup_index in range(WARMUP):
        order = PATHS[warmup_index % 2:] + PATHS[:warmup_index % 2]
        for path in order:
            calls[path]()
    torch.cuda.synchronize()
    raw = {path: [] for path in PATHS}
    first_counts = {path: 0 for path in PATHS}
    stream = torch.cuda.current_stream()
    for sample_index in range(SAMPLES):
        order = PATHS[sample_index % 2:] + PATHS[:sample_index % 2]
        first_counts[order[0]] += 1
        for path in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            calls[path]()
            end.record(stream)
            end.synchronize()
            raw[path].append(float(start.elapsed_time(end)))
    torch.cuda.synchronize()
    if first_counts != {"pinned_public": 500, "c1_test_route_public": 500}:
        raise AssertionError(f"cyclic schedule drift: {first_counts}")
    immutability = _assert_inputs_unchanged(snapshot, x, f"performance/{process_index}/{repeat_index}/{contract}")
    paths = {path: _summary(raw[path]) for path in PATHS}
    margins = {name: float(paths["pinned_public"][f"{name}_ms"]) / float(paths["c1_test_route_public"][f"{name}_ms"]) - 1.0 for name in PERCENTILES}
    if not all(value >= MIN_MARGIN for value in margins.values()):
        winner = "c1_test_route_public" if all(value > 0 for value in margins.values()) else "pinned_public"
    else:
        winner = "c1_test_route_public"
    return {
        "process_index": process_index, "repeat_index": repeat_index, "event_contract": "current-stream start event -> one real public chunk_kda call -> end event -> end.synchronize; synchronization is excluded from sample",
        "schedule": "two-path cyclic rotation; 100 warmups and 1000 timed samples per public path", "first_path_counts": first_counts,
        "public_precheck": {"pinned": pinned_proof, "c1_test_route": c1_proof, "exact": precheck},
        **immutability, "initial_state_immutability_exact": True,
        "raw_samples_ms": raw, "paths": paths, "c1_margin_over_pinned_by_percentile": margins,
        "winner_by_percentile": {name: "c1_test_route_public" if margins[name] > 0 else "pinned_public" for name in PERCENTILES},
        "repeat_gate_pass": winner == "c1_test_route_public" and all(value >= MIN_MARGIN for value in margins.values()), "passed": True,
    }


def _describe(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 1, "purpose": "two-allocation public FLA test-only tail8191 route; never production dispatch", "describe_only": True,
        "allocation_id": args.allocation, "shape": {"B": BATCH, "H": HEADS, "T": TOKENS, "K": DIM, "V": DIM},
        "contracts": list(CONTRACTS), "raw_abi_exact_paths": ["baseline_vs_reference", "vshard4_vs_reference"],
        "public_paths": list(PATHS), "fresh_pids_per_allocation": 2, "repeats_per_pid": REPEATS,
        "samples_per_path_repeat": SAMPLES, "required_percentiles": list(PERCENTILES), "minimum_c1_margin": MIN_MARGIN,
        "allocation_gate": "both contracts, 2 fresh PID x 2 repeats x P50/P95/P99 must make C1 public route >=2% faster",
        "freeze_gate": "both separately clean A1 and A2 allocations must pass; this runner itself has no release authority",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", choices=("A1", "A2"), required=True)
    parser.add_argument("--process-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--patched-root", type=Path)
    parser.add_argument("--fla-root", type=Path)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.describe:
        _write(args.json, _describe(args)); print(f"wrote tail8191 plan {args.json}"); return
    if os.environ.get(CLEAN_ENV) != "1" or args.reference_root is None or args.patched_root is None or args.fla_root is None:
        raise RuntimeError("refusing direct GPU run: use run_clean_tail8191_dispatch_audit.sh")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdecimal() or int(job_id) <= 0:
        raise RuntimeError("SLURM_JOB_ID must be a positive decimal identity")
    if not os.environ.get("FLA_FLASH_KDA"):
        raise RuntimeError("FLA_FLASH_KDA must be enabled for the real pinned public path")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from assignment02.team.c1_flashkda.harness import validate_and_bench as common
    identity = _identity(args)
    torch_ref = common._load_torch_ref(args.reference_root)
    result: dict[str, object] = {**_describe(args), "describe_only": False, "pid": os.getpid(), "slurm_job_id": job_id, "identity": identity, "raw_abi_correctness": {}, "public_benchmarks": {}, "negative_controls": _production_negative_controls(), "complete": False}
    _write(args.json, result)
    with _install_test_route() as auto_dispatch:
        custom = pinned = public_fn = c1_counter = pinned_counter = restore = None
        try:
            with _c1_enabled(True):
                custom, pinned, public_fn, c1_counter, pinned_counter, restore = _registry_setup()
            for contract_index, contract in enumerate(CONTRACTS):
                x = _make_inputs(args.seed + args.process_index * 1_000_003 + contract_index * 10_007)
                result["raw_abi_correctness"][contract] = _raw_correctness(x, contract, torch_ref)  # type: ignore[index]
                _write(args.json, result)
                repeats: list[dict[str, object]] = []
                result["public_benchmarks"][contract] = repeats  # type: ignore[index]
                for repeat_index in range(REPEATS):
                    repeat_x = _make_inputs(args.seed + args.process_index * 1_000_003 + contract_index * 10_007 + repeat_index * 101)
                    repeats.append(_benchmark_repeat(args.process_index, repeat_index, repeat_x, contract, public_fn, c1_counter, pinned_counter, auto_dispatch))
                    _write(args.json, result)
        finally:
            if restore is not None:
                restore()
    result["complete"] = True
    _write(args.json, result)
    print(f"wrote tail8191 public artifact {args.json}")


if __name__ == "__main__":
    main()
