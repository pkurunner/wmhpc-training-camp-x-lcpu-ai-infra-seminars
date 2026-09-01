#!/usr/bin/env bash
# One-kernel NCU mechanism diagnosis for the experimental v6 register-resident
# numerator.  This is intentionally not a benchmark: the only timing decision
# is the separately collected clean CUDA-event ABBA gate.  Here v5 and v6 are
# loaded in different Python processes because they register the same stable
# _C::native_c2_msa_decode schema and CUDA implementation.
#
# Submit only after reviewing this file and substituting its SHA-256:
#   sbatch --export=ALL,C2_EXPECTED_RUNTIME_SCRIPT_SHA=<sha256> \
#     profile_native_c2_v6_ncu_20260830.slurm

#SBATCH --job-name=c2-native-v6-ncu
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v6-ncu-artifacts-20260830/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONPATH=
export PYTHONDONTWRITEBYTECODE=1
export PATH=/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v6-ncu-artifacts-20260830
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID:?must run under Slurm}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
SUBMITTED_SCRIPT=/home/lcpu/85117379/profile_native_c2_v6_ncu_20260830.slurm
BASE_HARNESS=/home/lcpu/85117379/native_c2_operator_bench_20260829.py
FROZEN_V5_NCU_SUMMARY=/home/lcpu/85117379/c2-native-plugin-v5-ncu-artifacts-20260829/job11779/native-c2-v5-ncu-job11779.json

V5_LIBRARY=/home/lcpu/85117379/c2-native-plugin-v5-aot-artifacts-20260829/job11760/vllm/_native_c2_msa_decode_plugin.abi3.so
V5_SOURCE=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v5-warp-parallel-softmax-20260829-job11760/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
V5_RESOURCE=/home/lcpu/85117379/c2-native-plugin-v5-aot-artifacts-20260829/job11760/plugin-resource-usage-job11760.txt
V5_PROVENANCE=/home/lcpu/85117379/c2-native-plugin-v5-aot-artifacts-20260829/job11760/plugin-v5-provenance-job11760.sha256
V6_LIBRARY=/home/lcpu/85117379/c2-native-plugin-v6-aot-artifacts-20260830/job12278/vllm/_native_c2_msa_decode_plugin.abi3.so
V6_SOURCE=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v6-register-numerator-20260830-job12278/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
V6_RESOURCE=/home/lcpu/85117379/c2-native-plugin-v6-aot-artifacts-20260830/job12278/plugin-resource-gate-job12278.txt
V6_PROVENANCE=/home/lcpu/85117379/c2-native-plugin-v6-aot-artifacts-20260830/job12278/plugin-v6-provenance-job12278.sha256

EXPECTED_BASE_HARNESS_SHA=0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28
EXPECTED_FROZEN_V5_NCU_SUMMARY_SHA=5e8a025b1ae97160925f88d576599b9d60f274ecf8426c52dc88b381a3f710cb
EXPECTED_V5_PLUGIN_SHA=9ab755faf79ec7bc574a8f5afa09b22b9acedc8257d3ccfad6fc98cc3f1d2538
EXPECTED_V5_SOURCE_SHA=6409a54689ebbdd1f72fad76c2d72dfe588f79d3b8dd6c4dbaf1df5b630f0931
EXPECTED_V5_RESOURCE_SHA=ff55a3a07e35aa6d532fbaa36e24ebaea43c1fe40ad329186666a81e2b0c2f56
EXPECTED_V5_PROVENANCE_SHA=c7c2929297c7e52db8480be6c957a9751d004066f45c4c6e032010ffc916a63c
EXPECTED_V6_PLUGIN_SHA=0d5492bee11b2a785d5f16681614b0a1fc6c74ddb33351d44f50b124580e21a4
EXPECTED_V6_SOURCE_SHA=c7f05928fd2ba3a0e29d3154f2bf56849e9d7ad6fd3eddc24b0890722e117f02
EXPECTED_V6_RESOURCE_SHA=2f5d0898888446b79808b02c709dd2f0b33309679629cdc335c263482a965560
EXPECTED_V6_PROVENANCE_SHA=e2067f586f338139283fdf54b9cd6cc2b57ac5a2d189f04bfa3dfc732e0839d8

: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?reviewed NCU script SHA-256 required}"
[[ "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" =~ ^[0-9a-f]{64}$ ]]
for digest in "${EXPECTED_BASE_HARNESS_SHA}" "${EXPECTED_FROZEN_V5_NCU_SUMMARY_SHA}" \
  "${EXPECTED_V5_PLUGIN_SHA}" "${EXPECTED_V5_SOURCE_SHA}" "${EXPECTED_V5_RESOURCE_SHA}" \
  "${EXPECTED_V5_PROVENANCE_SHA}" "${EXPECTED_V6_PLUGIN_SHA}" "${EXPECTED_V6_SOURCE_SHA}" \
  "${EXPECTED_V6_RESOURCE_SHA}" "${EXPECTED_V6_PROVENANCE_SHA}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid pinned SHA-256: ${digest}" >&2; exit 2; }
done

# Keep the fresh comparison semantically identical to the successful v5
# job11779 collection, then add elapsed cycles and active-warp occupancy so
# the register-resident numerator has a fuller mechanism trace.
metrics=(
  gpu__time_duration.sum
  smsp__cycles_elapsed.sum
  dram__bytes_read.sum
  dram__bytes_write.sum
  lts__t_bytes.sum
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
  smsp__inst_executed_pipe_tensor.sum
  smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct
  smsp__average_warps_issue_stalled_wait_per_issue_active.pct
  smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
  smsp__warps_active.avg.per_cycle_active
  sm__warps_active.avg.pct_of_peak_sustained_active
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
)
metrics_csv="$(IFS=,; printf '%s' "${metrics[*]}")"

allocated_gpu_selector() {
  local selector=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}
  [[ "${selector}" != *,* ]] || {
    echo "expected one CUDA_VISIBLE_DEVICES entry, got: ${selector}" >&2
    return 1
  }
  [[ "${selector}" =~ ^[0-9]+$ || "${selector}" =~ ^GPU-[0-9A-Fa-f-]+$ ]] || {
    echo "unsupported CUDA_VISIBLE_DEVICES entry: ${selector}" >&2
    return 1
  }
  printf '%s\n' "${selector}"
}
visible_gpu_uuid() {
  local selector uuid
  local -a uuids
  selector=$(allocated_gpu_selector)
  mapfile -t uuids < <(nvidia-smi --id "${selector}" --query-gpu=uuid --format=csv,noheader,nounits)
  [[ ${#uuids[@]} -eq 1 ]] || {
    printf 'expected one allocated GPU UUID, got %s\n' "${#uuids[@]}" >&2
    return 1
  }
  uuid=${uuids[0]}
  [[ "${uuid}" =~ ^GPU- ]] || { echo "invalid GPU UUID: ${uuid}" >&2; return 1; }
  printf '%s\n' "${uuid}"
}
compute_apps() {
  local selector
  selector=$(allocated_gpu_selector)
  nvidia-smi --id "${selector}" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}
snapshot() {
  local selector
  selector=$(allocated_gpu_selector)
  printf '\n===== %s UTC=%s =====\n' "$1" "$(date -u +%FT%TZ)"
  nvidia-smi --id "${selector}" --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
    --format=csv,noheader,nounits || return 1
  printf '%s\n' '-- compute apps --'
  compute_apps || return 1
}
sha256_of() { sha256sum "$1" | awk '{print $1}'; }
finish() {
  local original_rc=$? final_rc finalizer_error=0 apps_after='' post_uuid=''
  local apps_rc uuid_rc manifest_rc
  local -a pipeline_rc
  final_rc=${original_rc}
  trap - EXIT
  set +e
  snapshot POST | tee "${ARTIFACT_DIR}/gpu-post-job${SLURM_JOB_ID}.txt"
  pipeline_rc=("${PIPESTATUS[@]}")
  [[ "${pipeline_rc[0]}" -eq 0 && "${pipeline_rc[1]}" -eq 0 ]] || finalizer_error=1
  apps_after=$(compute_apps)
  apps_rc=$?
  [[ "${apps_rc}" -eq 0 && -z "${apps_after}" ]] || finalizer_error=1
  post_uuid=$(visible_gpu_uuid)
  uuid_rc=$?
  [[ "${uuid_rc}" -eq 0 && -n "${EXPECTED_GPU_UUID:-}" && "${post_uuid}" == "${EXPECTED_GPU_UUID}" ]] || finalizer_error=1
  [[ "${finalizer_error}" -eq 0 || "${final_rc}" -ne 0 ]] || final_rc=2
  printf 'ORIGINAL_RC=%s POST_PIPELINE_RC=%s,%s APPS_QUERY_RC=%s APPS_EMPTY=%s UUID_QUERY_RC=%s EXPECTED_UUID=%s POST_UUID=%s\n' \
    "${original_rc}" "${pipeline_rc[0]:-missing}" "${pipeline_rc[1]:-missing}" "${apps_rc}" \
    "$([[ -z "${apps_after}" ]] && echo true || echo false)" "${uuid_rc}" \
    "${EXPECTED_GPU_UUID:-unset}" "${post_uuid:-unset}" \
    > "${ARTIFACT_DIR}/finalizer-diagnostics-job${SLURM_JOB_ID}.txt" || {
      [[ "${final_rc}" -ne 0 ]] || final_rc=2
    }
  manifest=${ARTIFACT_DIR}/outputs-manifest-job${SLURM_JOB_ID}.sha256
  manifest_tmp=${manifest}.tmp
  find "${ARTIFACT_DIR}" -maxdepth 1 -type f \
    ! -name 'outputs-manifest-job*.sha256' ! -name 'outputs-manifest-job*.sha256.tmp' \
    ! -name 'driver-job*.log' ! -name 'final-status-job*.txt' \
    -printf '%p\0' | sort -z | xargs -0r sha256sum > "${manifest_tmp}"
  pipeline_rc=("${PIPESTATUS[@]}")
  manifest_rc=0
  for value in "${pipeline_rc[@]}"; do [[ "${value}" -eq 0 ]] || manifest_rc=1; done
  if [[ "${manifest_rc}" -eq 0 ]]; then
    mv "${manifest_tmp}" "${manifest}" || manifest_rc=1
  fi
  if [[ "${manifest_rc}" -ne 0 ]]; then
    rm -f "${manifest_tmp}"
    [[ "${final_rc}" -ne 0 ]] || final_rc=2
  fi
  printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s MANIFEST_RC=%s UTC=%s\n' \
    "${final_rc}" "${original_rc}" "${finalizer_error}" "${manifest_rc}" "$(date -u +%FT%TZ)" \
    > "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt" || {
      [[ "${final_rc}" -ne 0 ]] || final_rc=2
    }
  printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s MANIFEST_RC=%s\n' \
    "${final_rc}" "${original_rc}" "${finalizer_error}" "${manifest_rc}"
  exit "${final_rc}"
}

[[ ! -e "${ARTIFACT_DIR}" ]] || { echo "refusing to reuse ${ARTIFACT_DIR}" >&2; exit 2; }
mkdir -p "${ARTIFACT_ROOT}" "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1
trap finish EXIT

# A queued sbatch job runs the immutable slurmd spool copy exposed through
# BASH_SOURCE.  Pin that actual copy and the reviewed submit-path copy without
# guessing a distribution-specific /var/{lib,spool} path.
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
[[ -f "${SCRIPT_PATH}" && -f "${SUBMITTED_SCRIPT}" ]]
[[ "$(sha256_of "${SCRIPT_PATH}")" == "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" ]]
[[ "$(sha256_of "${SUBMITTED_SCRIPT}")" == "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" ]]
cmp -s "${SCRIPT_PATH}" "${SUBMITTED_SCRIPT}"

snapshot PRE | tee "${ARTIFACT_DIR}/gpu-pre-job${SLURM_JOB_ID}.txt"
selector=$(allocated_gpu_selector)
gpu_line=$(nvidia-smi --id "${selector}" --query-gpu=name,compute_cap --format=csv,noheader,nounits)
EXPECTED_GPU_UUID=$(visible_gpu_uuid)
apps_before=$(compute_apps) || { echo 'unable to query pre-run compute apps' >&2; exit 2; }
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* ]]
[[ -z "${apps_before}" ]] || { echo 'GPU is not clean before NCU' >&2; exit 2; }
[[ -x "${PYTHON}" && -f "${BASE_HARNESS}" && -f "${FROZEN_V5_NCU_SUMMARY}" ]]
[[ -f "${V5_LIBRARY}" && -f "${V5_SOURCE}" && -f "${V5_RESOURCE}" && -f "${V5_PROVENANCE}" ]]
[[ -f "${V6_LIBRARY}" && -f "${V6_SOURCE}" && -f "${V6_RESOURCE}" && -f "${V6_PROVENANCE}" ]]
command -v ncu
ncu --version | tee "${ARTIFACT_DIR}/ncu-version-job${SLURM_JOB_ID}.txt"
"${PYTHON}" -c 'import platform, torch; assert __debug__; print({"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda})' \
  | tee "${ARTIFACT_DIR}/python-environment-job${SLURM_JOB_ID}.txt"
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${EXPECTED_BASE_HARNESS_SHA}" "${BASE_HARNESS}" \
  "${EXPECTED_FROZEN_V5_NCU_SUMMARY_SHA}" "${FROZEN_V5_NCU_SUMMARY}" \
  "${EXPECTED_V5_PLUGIN_SHA}" "${V5_LIBRARY}" \
  "${EXPECTED_V5_SOURCE_SHA}" "${V5_SOURCE}" \
  "${EXPECTED_V5_RESOURCE_SHA}" "${V5_RESOURCE}" \
  "${EXPECTED_V5_PROVENANCE_SHA}" "${V5_PROVENANCE}" \
  "${EXPECTED_V6_PLUGIN_SHA}" "${V6_LIBRARY}" \
  "${EXPECTED_V6_SOURCE_SHA}" "${V6_SOURCE}" \
  "${EXPECTED_V6_RESOURCE_SHA}" "${V6_RESOURCE}" \
  "${EXPECTED_V6_PROVENANCE_SHA}" "${V6_PROVENANCE}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SUBMITTED_SCRIPT}" \
  > "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
grep -F "${EXPECTED_V5_PLUGIN_SHA}  ${V5_LIBRARY}" "${V5_PROVENANCE}"
grep -F "${EXPECTED_V5_SOURCE_SHA}  ${V5_SOURCE}" "${V5_PROVENANCE}"
grep -F "${EXPECTED_V6_PLUGIN_SHA}  ${V6_LIBRARY}" "${V6_PROVENANCE}"
grep -F "${EXPECTED_V6_SOURCE_SHA}  ${V6_SOURCE}" "${V6_PROVENANCE}"
grep -Eq 'REG:64[[:space:]]+STACK:0[[:space:]]+SHARED:38752[[:space:]]+LOCAL:0' "${V5_RESOURCE}"
grep -qx 'REG=64' "${V6_RESOURCE}"
grep -qx 'STACK=0' "${V6_RESOURCE}"
grep -qx 'SHARED=30560' "${V6_RESOURCE}"
grep -qx 'LOCAL=0' "${V6_RESOURCE}"

METRIC_CATALOG=${ARTIFACT_DIR}/metric-catalog-job${SLURM_JOB_ID}.txt
grep_args=()
for metric in "${metrics[@]}"; do grep_args+=(--regexp="${metric}"); done
timeout --signal=TERM --kill-after=15s 180s ncu --query-metrics --query-metrics-mode all \
  | grep --fixed-strings "${grep_args[@]}" > "${METRIC_CATALOG}"
"${PYTHON}" - "${METRIC_CATALOG}" "${metrics[@]}" <<'PY'
import re
import sys
from pathlib import Path

catalog = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
available = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*(?:__[A-Za-z0-9_]+)+(?:\.[A-Za-z0-9_]+)+\b", catalog))
missing = [metric for metric in sys.argv[2:] if metric not in available]
if missing:
    raise SystemExit("missing exact NCU metrics: " + ", ".join(missing))
if len(sys.argv[2:]) != len(set(sys.argv[2:])):
    raise SystemExit("duplicate requested NCU metric")
PY

TARGET_RUNNER=${ARTIFACT_DIR}/ncu_target_runner.py
cat > "${TARGET_RUNNER}" <<'PY'
#!/usr/bin/env python3
"""A one-process direct C2 target; no DSO may share this interpreter."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import torch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("c2_ncu_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("Contract", "_BATCH", "_QUERY_HEADS", "_HEAD_DIM", "_make_inputs", "_oracle", "_validate_args"):
        if not hasattr(module, name):
            raise RuntimeError(f"base harness lacks {name}")
    return module


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode("ascii"))
    h.update(repr(tuple(value.shape)).encode("ascii"))
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def visible_uuid() -> str:
    visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if len(visible) == 1 and (visible[0].isdigit() or visible[0].startswith("GPU-")):
        values = [line.strip() for line in subprocess.check_output(
            ["nvidia-smi", "--id", visible[0], "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines() if line.strip()]
        if len(values) == 1 and values[0].startswith("GPU-"):
            return values[0]
    value = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if isinstance(value, str) and value.startswith("GPU-"):
        return value
    raise RuntimeError("cannot establish allocated physical GPU UUID")


def call(op, output, inputs, contract):
    return op(output, *inputs, contract.scale, contract.q_scale, contract.k_scale, contract.v_scale)


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
    if torch.cuda.get_device_capability(0) != (10, 3):
        raise RuntimeError(f"B300 capability required, got {torch.cuda.get_device_capability(0)}")
    if not args.library.is_absolute() or not args.base_harness.is_absolute():
        raise ValueError("paths must be absolute")
    if digest(args.base_harness) != args.base_harness_sha256:
        raise RuntimeError("base harness checksum drift")
    base = load_base(args.base_harness)
    contract = base._validate_args(type("Args", (), {
        "num_physical_pages": 64, "max_logical_pages": 32,
        "scale": 1.0 / (128.0 ** 0.5), "q_scale": 0.25,
        "k_scale": 0.25, "v_scale": 0.5, "atol": 1.0e-4, "rtol": 1.0e-3,
    })())
    torch.ops.load_library(str(args.library))
    if not torch._C._dispatch_has_kernel_for_dispatch_key("_C::native_c2_msa_decode", "CUDA"):
        raise RuntimeError("native C2 CUDA dispatch is not registered")
    inputs = base._make_inputs(contract, args.seed)
    input_hashes = [tensor_digest(item) for item in inputs]
    input_manifest = hashlib.sha256("".join(input_hashes).encode("ascii")).hexdigest()
    output = torch.full((base._BATCH, base._QUERY_HEADS, base._HEAD_DIM), float("nan"), device="cuda", dtype=torch.bfloat16)
    pointer = output.data_ptr()
    returned = call(torch.ops._C.native_c2_msa_decode, output, inputs, contract)
    torch.cuda.synchronize()
    checks = {"return_is_none": returned is None, "pointer_unchanged": output.data_ptr() == pointer,
              "output_finite": bool(torch.isfinite(output).all().item())}
    record = {
        "schema": "c2-native-v6-ncu-target-v1", "mode": args.mode, "all_gates_pass": False,
        "operator_library": str(args.library.resolve()), "operator_library_sha256": digest(args.library),
        "base_harness": str(args.base_harness.resolve()), "base_harness_sha256": digest(args.base_harness),
        "seed": args.seed, "input_tensor_sha256": input_hashes, "input_manifest_sha256": input_manifest,
        "contract": {"batch": base._BATCH, "head_dim": base._HEAD_DIM, "kv_heads": 4, "q_heads": base._QUERY_HEADS,
                     "page_size": 128, "topk": 16, "num_physical_pages": 64, "max_logical_pages": 32,
                     "scale": contract.scale, "q_scale": contract.q_scale, "k_scale": contract.k_scale, "v_scale": contract.v_scale},
        "caller_output": checks, "device": {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
                     "gpu_uuid": visible_uuid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "dispatch": "direct_torch.ops._C.native_c2_msa_decode", "no_monkeypatch": True,
    }
    if args.mode == "ncu":
        record["all_gates_pass"] = all(checks.values())
        return record
    reference = base._oracle(*inputs, contract, torch.float64)
    actual = output.to(torch.float64)
    delta = (actual - reference).abs()
    denominator = reference.abs().clamp_min(torch.finfo(torch.float64).eps)
    correct = bool(torch.allclose(actual, reference, atol=1.0e-4, rtol=1.0e-3))
    profile_output = torch.full_like(output, float("nan"))
    profile_pointer = profile_output.data_ptr()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        profile_return = call(torch.ops._C.native_c2_msa_decode, profile_output, inputs, contract)
        torch.cuda.synchronize()
    events = profiler.key_averages()
    dispatches = [event for event in events if str(event.key) == "_C::native_c2_msa_decode"]
    kernels = [event for event in events if "native_c2_msa_decode_kernel" in str(event.key)]
    profile_checks = {"one_dispatch": len(dispatches) == 1 and int(getattr(dispatches[0], "count", 0)) == 1,
                      "one_native_kernel": len(kernels) == 1 and int(getattr(kernels[0], "count", 0)) == 1,
                      "profile_return_is_none": profile_return is None,
                      "profile_pointer_unchanged": profile_output.data_ptr() == profile_pointer}
    record["correctness"] = {"oracle_dtype": "float64", "atol": 1.0e-4, "rtol": 1.0e-3, "allclose": correct,
                              "finite_output": checks["output_finite"], "max_abs": float(delta.max().item()),
                              "mean_abs": float(delta.mean().item()), "max_rel": float((delta / denominator).max().item())}
    record["profiler"] = {"checks": profile_checks, "dispatch_count": len(dispatches), "native_kernel_count": len(kernels)}
    record["output_sha256"] = tensor_digest(output)
    record["all_gates_pass"] = bool(all(checks.values()) and correct and all(profile_checks.values()))
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "ncu"), required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--base-harness", type=Path, required=True)
    parser.add_argument("--base-harness-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        result = {"schema": "c2-native-v6-ncu-target-v1", "mode": args.mode, "all_gates_pass": False,
                  "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
PY
chmod 0644 "${TARGET_RUNNER}"
"${PYTHON}" -m py_compile "${TARGET_RUNNER}"

V5_PREFLIGHT=${ARTIFACT_DIR}/v5-preflight-job${SLURM_JOB_ID}.json
V6_PREFLIGHT=${ARTIFACT_DIR}/v6-preflight-job${SLURM_JOB_ID}.json
run_target() {
  local mode=$1 library=$2 output=$3
  timeout --foreground --preserve-status --kill-after=15s 300s "${PYTHON}" "${TARGET_RUNNER}" \
    --mode "${mode}" --library "${library}" --base-harness "${BASE_HARNESS}" \
    --base-harness-sha256 "${EXPECTED_BASE_HARNESS_SHA}" --seed 20260829 > "${output}"
}
# These are target/correctness gates, outside NCU.  Separate processes are
# mandatory: each DSO owns the same Torch dispatcher slot.
run_target validate "${V5_LIBRARY}" "${V5_PREFLIGHT}"
run_target validate "${V6_LIBRARY}" "${V6_PREFLIGHT}"
"${PYTHON}" - "${V5_PREFLIGHT}" "${V6_PREFLIGHT}" "${V5_LIBRARY}" "${V6_LIBRARY}" \
  "${EXPECTED_V5_PLUGIN_SHA}" "${EXPECTED_V6_PLUGIN_SHA}" "${EXPECTED_GPU_UUID}" <<'PY'
import json
import sys
from pathlib import Path

v5_path, v6_path = map(Path, sys.argv[1:3])
v5_library, v6_library = map(Path, sys.argv[3:5])
v5_sha, v6_sha, expected_uuid = sys.argv[5:]
payloads = [json.loads(path.read_text(encoding="utf-8")) for path in (v5_path, v6_path)]
for label, payload, library, expected_sha in zip(("v5", "v6"), payloads, (v5_library, v6_library), (v5_sha, v6_sha), strict=True):
    if payload.get("schema") != "c2-native-v6-ncu-target-v1" or payload.get("mode") != "validate" or payload.get("all_gates_pass") is not True:
        raise SystemExit(f"{label}: target/correctness gate failed")
    if Path(payload.get("operator_library", "")).resolve() != library.resolve() or payload.get("operator_library_sha256") != expected_sha:
        raise SystemExit(f"{label}: DSO identity drift")
    if payload.get("correctness", {}).get("oracle_dtype") != "float64" or payload.get("correctness", {}).get("allclose") is not True:
        raise SystemExit(f"{label}: FP64 oracle gate")
    if payload.get("profiler", {}).get("checks", {}) != {"one_dispatch": True, "one_native_kernel": True, "profile_return_is_none": True, "profile_pointer_unchanged": True}:
        raise SystemExit(f"{label}: profiler target gate")
if payloads[0]["input_tensor_sha256"] != payloads[1]["input_tensor_sha256"] or payloads[0]["input_manifest_sha256"] != payloads[1]["input_manifest_sha256"]:
    raise SystemExit("v5/v6 preflight inputs differ")
if payloads[0]["contract"] != payloads[1]["contract"] or payloads[0]["base_harness_sha256"] != payloads[1]["base_harness_sha256"]:
    raise SystemExit("v5/v6 preflight harness/contract differs")
uuids = [payload["device"].get("gpu_uuid") for payload in payloads]
if uuids != [expected_uuid, expected_uuid]:
    raise SystemExit("v5/v6 preflight did not use one physical GPU UUID")
PY
apps_after_preflight=$(compute_apps) || { echo 'unable to query pre-NCU compute apps' >&2; exit 2; }
[[ -z "${apps_after_preflight}" ]] || { echo 'GPU is not clean after preflight' >&2; exit 2; }

V5_REPORT=${ARTIFACT_DIR}/native-c2-v5-job${SLURM_JOB_ID}.ncu-rep
V6_REPORT=${ARTIFACT_DIR}/native-c2-v6-job${SLURM_JOB_ID}.ncu-rep
V5_NCU_TARGET=${ARTIFACT_DIR}/v5-ncu-target-job${SLURM_JOB_ID}.log
V6_NCU_TARGET=${ARTIFACT_DIR}/v6-ncu-target-job${SLURM_JOB_ID}.log
profile_one() {
  local library=$1 report=$2 log=$3
  timeout --signal=TERM --kill-after=30s 300s ncu --target-processes all \
    --kernel-name-base function --kernel-name 'regex:.*native_c2_msa_decode_kernel.*' \
    --launch-skip 0 --launch-count 1 --clock-control none --metrics "${metrics_csv}" \
    --force-overwrite --export "${report}" \
    "${PYTHON}" "${TARGET_RUNNER}" --mode ncu --library "${library}" \
      --base-harness "${BASE_HARNESS}" --base-harness-sha256 "${EXPECTED_BASE_HARNESS_SHA}" --seed 20260829 > "${log}" 2>&1
  [[ -s "${report}" && -s "${log}" ]]
}
# NCU sees one direct call per target process and filters only the native
# kernel.  It is deliberately after the uninstrumented correctness gate.
profile_one "${V5_LIBRARY}" "${V5_REPORT}" "${V5_NCU_TARGET}"
apps_after_v5=$(compute_apps) || { echo 'unable to query post-v5 NCU compute apps' >&2; exit 2; }
[[ -z "${apps_after_v5}" ]] || { echo 'GPU is not clean between NCU profiles' >&2; exit 2; }
profile_one "${V6_LIBRARY}" "${V6_REPORT}" "${V6_NCU_TARGET}"
apps_after_v6=$(compute_apps) || { echo 'unable to query post-v6 NCU compute apps' >&2; exit 2; }
[[ -z "${apps_after_v6}" ]] || { echo 'GPU is not clean after v6 NCU' >&2; exit 2; }

V5_RAW=${ARTIFACT_DIR}/native-c2-v5-job${SLURM_JOB_ID}.raw.csv
V6_RAW=${ARTIFACT_DIR}/native-c2-v6-job${SLURM_JOB_ID}.raw.csv
timeout --signal=TERM --kill-after=15s 120s ncu --import "${V5_REPORT}" --csv --page raw --print-units base --print-fp > "${V5_RAW}"
timeout --signal=TERM --kill-after=15s 120s ncu --import "${V6_REPORT}" --csv --page raw --print-units base --print-fp > "${V6_RAW}"
[[ -s "${V5_RAW}" && -s "${V6_RAW}" ]]

V5_SUMMARY=${ARTIFACT_DIR}/native-c2-v5-ncu-job${SLURM_JOB_ID}.json
V6_SUMMARY=${ARTIFACT_DIR}/native-c2-v6-ncu-job${SLURM_JOB_ID}.json
"${PYTHON}" - "${V5_RAW}" "${V5_NCU_TARGET}" "${V5_SUMMARY}" v5 64 38752 "${V5_LIBRARY}" "${EXPECTED_V5_PLUGIN_SHA}" "${metrics[@]}" <<'PY'
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

raw_path, target_path, out_path, version, expected_reg, expected_shared, expected_library, expected_library_sha, *required = sys.argv[1:]
expected_reg, expected_shared = float(expected_reg), float(expected_shared)
target_records = []
for line in Path(target_path).read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(parsed, dict) and parsed.get("schema") == "c2-native-v6-ncu-target-v1":
        target_records.append(parsed)
if len(target_records) != 1:
    raise SystemExit(f"{version}: expected exactly one NCU-target JSON record, got {len(target_records)}")
target = target_records[0]
if target.get("mode") != "ncu" or target.get("all_gates_pass") is not True:
    raise SystemExit(f"{version}: NCU target call failed")
if Path(target.get("operator_library", "")).resolve() != Path(expected_library).resolve() or target.get("operator_library_sha256") != expected_library_sha:
    raise SystemExit(f"{version}: NCU target DSO identity drift")
rows = list(csv.reader(Path(raw_path).open(encoding="utf-8", errors="replace", newline="")))
header_index = next((i for i, row in enumerate(rows) if "ID" in row and "Kernel Name" in row and all(metric in row for metric in required)), None)
if header_index is None:
    raise SystemExit(f"{version}: raw CSV lacks a complete requested-metric header")
header = rows[header_index]
counts = Counter(header)
mandatory_launch = ("launch__cluster_size", "launch__cluster_max_potential_size", "launch__cluster_max_active",
                    "launch__occupancy_cluster_pct", "launch__occupancy_cluster_gpu_pct", "launch__registers_per_thread",
                    "launch__shared_mem_per_block", "launch__barrier_count")
for column in ("ID", "Kernel Name", *required, *mandatory_launch):
    if counts[column] != 1:
        raise SystemExit(f"{version}: required CSV column {column!r} count is {counts[column]}")
indices = {name: header.index(name) for name in ("ID", "Kernel Name", *required, *mandatory_launch)}
last = max(indices.values())
if header_index + 1 >= len(rows) or len(rows[header_index + 1]) <= last:
    raise SystemExit(f"{version}: CSV lacks units row")
units_row = rows[header_index + 1]
kernel_rows = [row for row in rows[header_index + 2:] if len(row) > last and "native_c2_msa_decode_kernel" in row[indices["Kernel Name"]]]
if len(kernel_rows) != 1:
    raise SystemExit(f"{version}: expected one filtered native kernel row, got {len(kernel_rows)}")
row = kernel_rows[0]
def number(name: str) -> float:
    raw = row[indices[name]].strip().replace(",", "")
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise SystemExit(f"{version}: invalid {name}: {raw!r}")
    return value
values = {metric: number(metric) for metric in required}
launch = {name: number(name) for name in mandatory_launch}
if values["gpu__time_duration.sum"] <= 0.0 or values["smsp__cycles_elapsed.sum"] <= 0.0 or values["smsp__inst_executed_pipe_tensor.sum"] <= 0.0:
    raise SystemExit(f"{version}: duration/cycles/tensor-pipe counter must be positive")
if launch["launch__registers_per_thread"] != expected_reg or launch["launch__shared_mem_per_block"] != expected_shared:
    raise SystemExit(f"{version}: launch resource drift: {launch}")
if launch["launch__cluster_size"] != 4.0 or launch["launch__barrier_count"] != 1.0:
    raise SystemExit(f"{version}: unexpected cluster/barrier launch: {launch}")
payload = {
    "schema": "c2-native-c2-v6-ncu-counters-v1", "version": version, "all_gates_pass": True,
    "kernel_name": row[indices["Kernel Name"]], "action_id": row[indices["ID"]].strip(), "matching_kernel_rows": 1,
    "target": target, "metrics": {metric: {"value": values[metric], "unit": units_row[indices[metric]].strip()} for metric in required},
    "launch_metadata": launch,
    "timing_valid_for_benchmark": False,
    "timing_note": "NCU replay/instrumentation timing is invalid for performance decisions; use the clean non-NCU ABBA result.",
}
Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
"${PYTHON}" - "${V6_RAW}" "${V6_NCU_TARGET}" "${V6_SUMMARY}" v6 64 30560 "${V6_LIBRARY}" "${EXPECTED_V6_PLUGIN_SHA}" "${metrics[@]}" <<'PY'
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

raw_path, target_path, out_path, version, expected_reg, expected_shared, expected_library, expected_library_sha, *required = sys.argv[1:]
expected_reg, expected_shared = float(expected_reg), float(expected_shared)
target_records = []
for line in Path(target_path).read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(parsed, dict) and parsed.get("schema") == "c2-native-v6-ncu-target-v1":
        target_records.append(parsed)
if len(target_records) != 1:
    raise SystemExit(f"{version}: expected exactly one NCU-target JSON record, got {len(target_records)}")
target = target_records[0]
if target.get("mode") != "ncu" or target.get("all_gates_pass") is not True:
    raise SystemExit(f"{version}: NCU target call failed")
if Path(target.get("operator_library", "")).resolve() != Path(expected_library).resolve() or target.get("operator_library_sha256") != expected_library_sha:
    raise SystemExit(f"{version}: NCU target DSO identity drift")
rows = list(csv.reader(Path(raw_path).open(encoding="utf-8", errors="replace", newline="")))
header_index = next((i for i, row in enumerate(rows) if "ID" in row and "Kernel Name" in row and all(metric in row for metric in required)), None)
if header_index is None:
    raise SystemExit(f"{version}: raw CSV lacks a complete requested-metric header")
header = rows[header_index]
counts = Counter(header)
mandatory_launch = ("launch__cluster_size", "launch__cluster_max_potential_size", "launch__cluster_max_active",
                    "launch__occupancy_cluster_pct", "launch__occupancy_cluster_gpu_pct", "launch__registers_per_thread",
                    "launch__shared_mem_per_block", "launch__barrier_count")
for column in ("ID", "Kernel Name", *required, *mandatory_launch):
    if counts[column] != 1:
        raise SystemExit(f"{version}: required CSV column {column!r} count is {counts[column]}")
indices = {name: header.index(name) for name in ("ID", "Kernel Name", *required, *mandatory_launch)}
last = max(indices.values())
if header_index + 1 >= len(rows) or len(rows[header_index + 1]) <= last:
    raise SystemExit(f"{version}: CSV lacks units row")
units_row = rows[header_index + 1]
kernel_rows = [row for row in rows[header_index + 2:] if len(row) > last and "native_c2_msa_decode_kernel" in row[indices["Kernel Name"]]]
if len(kernel_rows) != 1:
    raise SystemExit(f"{version}: expected one filtered native kernel row, got {len(kernel_rows)}")
row = kernel_rows[0]
def number(name: str) -> float:
    raw = row[indices[name]].strip().replace(",", "")
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise SystemExit(f"{version}: invalid {name}: {raw!r}")
    return value
values = {metric: number(metric) for metric in required}
launch = {name: number(name) for name in mandatory_launch}
if values["gpu__time_duration.sum"] <= 0.0 or values["smsp__cycles_elapsed.sum"] <= 0.0 or values["smsp__inst_executed_pipe_tensor.sum"] <= 0.0:
    raise SystemExit(f"{version}: duration/cycles/tensor-pipe counter must be positive")
if launch["launch__registers_per_thread"] != expected_reg or launch["launch__shared_mem_per_block"] != expected_shared:
    raise SystemExit(f"{version}: launch resource drift: {launch}")
if launch["launch__cluster_size"] != 4.0 or launch["launch__barrier_count"] != 1.0:
    raise SystemExit(f"{version}: unexpected cluster/barrier launch: {launch}")
payload = {
    "schema": "c2-native-c2-v6-ncu-counters-v1", "version": version, "all_gates_pass": True,
    "kernel_name": row[indices["Kernel Name"]], "action_id": row[indices["ID"]].strip(), "matching_kernel_rows": 1,
    "target": target, "metrics": {metric: {"value": values[metric], "unit": units_row[indices[metric]].strip()} for metric in required},
    "launch_metadata": launch,
    "timing_valid_for_benchmark": False,
    "timing_note": "NCU replay/instrumentation timing is invalid for performance decisions; use the clean non-NCU ABBA result.",
}
Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

MECHANISM=${ARTIFACT_DIR}/v6-v5-mechanism-job${SLURM_JOB_ID}.json
"${PYTHON}" - "${V5_SUMMARY}" "${V6_SUMMARY}" "${FROZEN_V5_NCU_SUMMARY}" "${MECHANISM}" \
  "${EXPECTED_FROZEN_V5_NCU_SUMMARY_SHA}" "${V5_LIBRARY}" "${V6_LIBRARY}" "${V5_SOURCE}" "${V6_SOURCE}" \
  "${EXPECTED_GPU_UUID}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

v5_path, v6_path, historical_path, out_path = map(Path, sys.argv[1:5])
expected_historical_sha = sys.argv[5]
v5_library, v6_library, v5_source, v6_source = map(Path, sys.argv[6:10])
expected_gpu_uuid = sys.argv[10]
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if digest(historical_path) != expected_historical_sha:
    raise SystemExit("frozen job11779 NCU result checksum drift")
v5, v6, historical = (json.loads(path.read_text(encoding="utf-8")) for path in (v5_path, v6_path, historical_path))
for label, payload in (("v5", v5), ("v6", v6)):
    if payload.get("schema") != "c2-native-c2-v6-ncu-counters-v1" or payload.get("all_gates_pass") is not True:
        raise SystemExit(f"invalid fresh {label} NCU summary")
    if payload.get("matching_kernel_rows") != 1 or payload.get("timing_valid_for_benchmark") is not False:
        raise SystemExit(f"invalid fresh {label} collection boundary")
if historical.get("schema") != "c2-native-c2-ncu-counters-v1" or historical.get("all_gates_pass") is not True:
    raise SystemExit("frozen job11779 template summary is invalid")
v5_metrics, v6_metrics = v5["metrics"], v6["metrics"]
if set(v5_metrics) != set(v6_metrics):
    raise SystemExit("fresh v5/v6 metric sets differ")
if not set(historical["metrics"]).issubset(v5_metrics):
    raise SystemExit("fresh collection no longer contains the frozen job11779 metric template")
for metric in v5_metrics:
    if v5_metrics[metric]["unit"] != v6_metrics[metric]["unit"]:
        raise SystemExit(f"unit mismatch: {metric}")
    if not all(math.isfinite(float(payload["metrics"][metric]["value"])) and float(payload["metrics"][metric]["value"]) >= 0.0 for payload in (v5, v6)):
        raise SystemExit(f"invalid counter: {metric}")
target_v5, target_v6 = v5["target"], v6["target"]
if target_v5["input_manifest_sha256"] != target_v6["input_manifest_sha256"] or target_v5["input_tensor_sha256"] != target_v6["input_tensor_sha256"]:
    raise SystemExit("NCU target input checksum differs")
if target_v5["base_harness_sha256"] != target_v6["base_harness_sha256"] or target_v5["contract"] != target_v6["contract"]:
    raise SystemExit("NCU target harness or contract differs")
gpu_uuids = [target_v5["device"].get("gpu_uuid"), target_v6["device"].get("gpu_uuid")]
if gpu_uuids != [expected_gpu_uuid, expected_gpu_uuid]:
    raise SystemExit("fresh NCU target runs did not share one physical GPU UUID")
pre_v5_path = Path(str(v5_path).replace("native-c2-v5-ncu", "v5-preflight"))
pre_v6_path = Path(str(v6_path).replace("native-c2-v6-ncu", "v6-preflight"))
pre_v5 = json.loads(pre_v5_path.read_text(encoding="utf-8"))
pre_v6 = json.loads(pre_v6_path.read_text(encoding="utf-8"))
# The summary target and preflight independently regenerate the exact fixed
# seed.  Require their input manifests to agree, not just v5-v6 equality.
if target_v5["input_manifest_sha256"] != pre_v5["input_manifest_sha256"] or target_v6["input_manifest_sha256"] != pre_v6["input_manifest_sha256"]:
    raise SystemExit("NCU target does not match its preflight fixture")
if pre_v5["device"].get("gpu_uuid") != expected_gpu_uuid or pre_v6["device"].get("gpu_uuid") != expected_gpu_uuid:
    raise SystemExit("preflight and NCU targets did not remain on one physical GPU UUID")
ratios, deltas = {}, {}
for metric in sorted(v5_metrics):
    old, new = float(v5_metrics[metric]["value"]), float(v6_metrics[metric]["value"])
    ratios[metric] = None if old == 0.0 else new / old
    deltas[metric] = new - old
def lower(metric):
    return ratios[metric] is not None and ratios[metric] < 1.0
payload = {
    "schema": "c2-native-c2-v6-v5-ncu-mechanism-comparison-v1", "all_collection_and_comparison_gates_pass": True,
    "reference": {"version": "v5", "aot_job": 11760, "summary": str(v5_path.resolve()), "summary_sha256": digest(v5_path),
                  "plugin": str(v5_library.resolve()), "plugin_sha256": target_v5["operator_library_sha256"],
                  "source": str(v5_source.resolve())},
    "candidate": {"version": "v6", "aot_job": 12278, "summary": str(v6_path.resolve()), "summary_sha256": digest(v6_path),
                  "plugin": str(v6_library.resolve()), "plugin_sha256": target_v6["operator_library_sha256"],
                  "source": str(v6_source.resolve())},
    "historical_template_only": {"v5_job": 11779, "summary": str(historical_path.resolve()), "summary_sha256": expected_historical_sha,
                                  "same_14_metric_template": True, "not_used_for_same-GPU_v6-v5_counter_delta": True},
    "same_collection_contract": {"seed": 20260829, "same_metric_set": True, "same_input_manifest_sha256": target_v5["input_manifest_sha256"],
                                 "same_harness_sha256": target_v5["base_harness_sha256"], "same_physical_gpu_uuid": gpu_uuids[0],
                                 "separate_python_processes": True, "reason": "each DSO registers the same stable _C schema/CUDA implementation",
                                 "single_native_kernel_action_each": True, "kernel_filter": "regex:.*native_c2_msa_decode_kernel.*"},
    "v6_minus_v5": deltas, "v6_over_v5": ratios,
    "resources": {"v5_aot": {"registers": 64, "stack": 0, "local": 0, "shared_bytes": 38752},
                  "v6_aot": {"registers": 64, "stack": 0, "local": 0, "shared_bytes": 30560},
                  "ncu_launch_v5": v5["launch_metadata"], "ncu_launch_v6": v6["launch_metadata"]},
    "mechanism_observations": {"shared_bytes_reduced_exactly": v5["launch_metadata"]["launch__shared_mem_per_block"] == 38752.0 and v6["launch_metadata"]["launch__shared_mem_per_block"] == 30560.0,
                               "registers_per_thread_unchanged": v5["launch_metadata"]["launch__registers_per_thread"] == v6["launch_metadata"]["launch__registers_per_thread"] == 64.0,
                               "shared_load_wavefronts_decreased": lower("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"),
                               "shared_store_wavefronts_decreased": lower("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"),
                               "shared_load_bank_conflicts_decreased": lower("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"),
                               "shared_store_bank_conflicts_decreased": lower("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"),
                               "barrier_stall_decreased": lower("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"),
                               "wait_stall_decreased": lower("smsp__average_warps_issue_stalled_wait_per_issue_active.pct"),
                               "scoreboard_stall_decreased": lower("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct"),
                               "cycles_metric": "smsp__cycles_elapsed.sum", "duration_metric": "gpu__time_duration.sum"},
    "decision_boundary": "NCU counters and NCU duration are mechanism evidence only. They neither promote nor reject v6; only independent clean non-NCU FP64-correct ABBA timing makes that performance decision.",
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

# Resolve the explicit preflight paths for the comparison parser's checks and
# then make an immutable final input/output manifest while all processes are
# gone.  `finish` writes POST plus an append-only final manifest.
sha256sum "${V5_PREFLIGHT}" "${V6_PREFLIGHT}" "${V5_REPORT}" "${V6_REPORT}" "${V5_RAW}" "${V6_RAW}" \
  "${V5_NCU_TARGET}" "${V6_NCU_TARGET}" "${V5_SUMMARY}" "${V6_SUMMARY}" "${MECHANISM}" \
  "${TARGET_RUNNER}" "${METRIC_CATALOG}" "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" \
  > "${ARTIFACT_DIR}/outputs-before-post-job${SLURM_JOB_ID}.sha256"
