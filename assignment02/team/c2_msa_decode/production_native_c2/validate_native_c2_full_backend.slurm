#!/usr/bin/env bash
#SBATCH --job-name=c2-native-full
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-full-backend-artifacts-20260829/slurm-%j.log

set -euo pipefail

ARTIFACT_DIR=${C2_FULL_ARTIFACT_DIR:-/home/lcpu/85117379/c2-native-full-backend-artifacts-20260829}
PYTHON=${C2_PYTHON:-/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python}
DERIVED=${C2_DERIVED:-/home/lcpu/85117379/vllm-d4-native-c2-20260829}
LIB=${C2_LIB:-/home/lcpu/85117379/vllm-d4-native-c2-build-20260829/_C_stable_libtorch.abi3.so}
LINK=${C2_LINK:-${DERIVED}/vllm/_C_stable_libtorch.abi3.so}
INSTALLED_VLLM=/home/lcpu/85117379/venv-vllm-d4da0c5/lib/python3.12/site-packages/vllm
FLASH_DIR=${C2_FLASH_DIR:-${DERIVED}/vllm/vllm_flash_attn}
C2_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/team/c2_msa_decode
HARNESS=/home/lcpu/85117379/native_c2_full_backend_bench_20260829.py
NATIVE_SOURCE=${C2_NATIVE_SOURCE:-${DERIVED}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu}
PATCH=/home/lcpu/85117379/c2-native-python-dispatch-20260829.patch
RESULT="${ARTIFACT_DIR}/full-backend-job${SLURM_JOB_ID}.json"
TRACE="${ARTIFACT_DIR}/full-backend-job${SLURM_JOB_ID}.chrome.json"

mkdir -p "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1
trap 'rc=$?; printf "FINAL_RC=%s UTC=%s\\n" "${rc}" "$(date -u +%FT%TZ)"' EXIT
export PATH="/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin"

snapshot() {
  local label=$1
  printf '\n===== %s UTC=%s =====\n' "${label}" "$(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap \
    --format=csv,noheader,nounits
  printf '%s\n' '-- compute apps --'
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true
}

snapshot PRE
gpu_line=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* ]]
[[ -d "${DERIVED}" && -d "${C2_ROOT}" ]]
[[ -f "${LIB}" && -f "${HARNESS}" && -f "${NATIVE_SOURCE}" && -f "${PATCH}" ]]
if [[ -n "${C2_EXPECTED_SOURCE_SHA:-}" ]]; then
  [[ "$(sha256sum "${NATIVE_SOURCE}" | awk '{print $1}')" == "${C2_EXPECTED_SOURCE_SHA}" ]]
fi
if [[ -n "${C2_EXPECTED_LIB_SHA:-}" ]]; then
  [[ "$(sha256sum "${LIB}" | awk '{print $1}')" == "${C2_EXPECTED_LIB_SHA}" ]]
fi

# These checkout-local links make a normal derived-source vLLM import load the
# built stable extension plus the immutable wheel's FA import prerequisites.
# Refuse to replace a regular file; the original venv is only read, never
# changed.  Keeping imports in the derived checkout is essential: that is
# where the native-C2 Python dispatch changes live.
ensure_symlink() {
  local source=$1
  local target=$2
  [[ -f "${source}" ]] || { echo "missing source ${source}" >&2; exit 1; }
  mkdir -p "$(dirname "${target}")"
  if [[ -e "${target}" || -L "${target}" ]]; then
    [[ -L "${target}" ]] || { echo "refusing to replace non-symlink ${target}" >&2; exit 1; }
    [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] || {
      echo "refusing unexpected symlink target ${target}" >&2
      exit 1
    }
  else
    ln -s "${source}" "${target}"
  fi
  [[ -L "${target}" && "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]]
}

ensure_symlink "${LIB}" "${LINK}"
ensure_symlink "${INSTALLED_VLLM}/vllm_flash_attn/_vllm_fa2_C.abi3.so" \
  "${FLASH_DIR}/_vllm_fa2_C.abi3.so"
ensure_symlink "${INSTALLED_VLLM}/vllm_flash_attn/_vllm_fa3_C.abi3.so" \
  "${FLASH_DIR}/_vllm_fa3_C.abi3.so"

git -C "${DERIVED}" rev-parse HEAD | tee "${ARTIFACT_DIR}/derived-head-job${SLURM_JOB_ID}.txt"
git -C "${DERIVED}" diff --check
git -C "${DERIVED}" diff --binary > "${ARTIFACT_DIR}/derived-diff-job${SLURM_JOB_ID}.patch"
sha256sum "${LIB}" "${LINK}" "${HARNESS}" "${NATIVE_SOURCE}" "${PATCH}" \
  "${FLASH_DIR}/_vllm_fa2_C.abi3.so" "${FLASH_DIR}/_vllm_fa3_C.abi3.so" \
  "${C2_ROOT}/harness/data.py" "${C2_ROOT}/harness/reference.py" \
  | tee "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
"${PYTHON}" - <<'PY'
import platform
import torch
print({"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda})
PY
ldd "${LIB}" | tee "${ARTIFACT_DIR}/ldd-job${SLURM_JOB_ID}.log"

PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${DERIVED}:${C2_ROOT}" \
timeout --foreground --preserve-status --kill-after=10s 900s \
  "${PYTHON}" "${HARNESS}" \
  --c2-root "${C2_ROOT}" \
  --vllm-root "${DERIVED}" \
  --output "${RESULT}" \
  --trace "${TRACE}" \
  --seed 20260829 \
  --timing-warmup 10 \
  --timing-repetitions 50

"${PYTHON}" - "${RESULT}" "${TRACE}" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["schema"] == "c2-native-c2-full-vllm-backend-v2"
assert payload["all_gates_pass"] is True
assert payload["warmup_calls"] == 1
assert payload["profiled_forward_calls"] == 1
assert all(payload["static_selection"].values())
assert all(payload["builder"]["checks"].values())
assert payload["builder"]["cutlass_plan_cache_size"] == 0
assert payload["caller_output"]["pointer_unchanged"] is True
assert payload["correctness"]["atol"] == 1e-4
assert payload["correctness"]["rtol"] == 1e-3
assert payload["correctness"]["pass"] is True
assert payload["triton_correctness"]["pass"] is True
for key in ("max_abs", "mean_abs"):
    assert math.isfinite(float(payload["correctness"][key]))
assert payload["profiling"]["pass"] is True
assert payload["profiling"]["checks"]["one_cpu_dispatcher_event"] is True
assert payload["profiling"]["checks"]["one_cuda_native_kernel_event"] is True
assert payload["timing"]["pass"] is True
assert "same BF16 query" in payload["timing"]["comparison_input_semantics"]
assert payload["timing"]["warmup"] >= 10
assert payload["timing"]["repetitions"] >= 50
assert payload["timing"]["native"]["sample_count"] == 2 * payload["timing"]["repetitions"]
assert payload["timing"]["triton"]["sample_count"] == 2 * payload["timing"]["repetitions"]
for name in ("native", "triton"):
    for key in ("p10_ms", "median_ms", "p90_ms"):
        assert math.isfinite(float(payload["timing"][name][key]))
        assert float(payload["timing"][name][key]) > 0.0
assert math.isfinite(float(payload["timing"]["triton_over_native_median_speedup"]))
assert math.isfinite(float(payload["timing"]["native_over_triton_median_latency_ratio"]))
assert float(payload["timing"]["native_over_triton_median_latency_ratio"]) > 0.0
trace = Path(sys.argv[2])
assert trace.is_file() and trace.stat().st_size > 0
PY

snapshot POST
sha256sum "${RESULT}" "${TRACE}" "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" \
  "${ARTIFACT_DIR}/derived-diff-job${SLURM_JOB_ID}.patch" \
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
