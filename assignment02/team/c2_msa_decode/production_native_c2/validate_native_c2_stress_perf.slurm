#!/usr/bin/env bash
#SBATCH --job-name=c2-native-stress
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-stress-artifacts-20260829/slurm-%j.log

set -euo pipefail

ARTIFACT_DIR=${C2_STRESS_ARTIFACT_DIR:-/home/lcpu/85117379/c2-native-stress-artifacts-20260829}
PYTHON=${C2_PYTHON:-/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python}
LIB=${C2_LIB:-/home/lcpu/85117379/vllm-d4-native-c2-build-20260829/_C_stable_libtorch.abi3.so}
BASE_HARNESS=${C2_BASE_HARNESS:-/home/lcpu/85117379/native_c2_operator_bench_20260829.py}
STRESS_HARNESS=${C2_STRESS_HARNESS:-/home/lcpu/85117379/native_c2_stress_perf_20260829.py}
SRC=${C2_NATIVE_SOURCE:-/home/lcpu/85117379/vllm-d4-native-c2-20260829/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu}
JSON_RESULT="${ARTIFACT_DIR}/stress-perf-job${SLURM_JOB_ID}.json"

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
[[ -f "${LIB}" && -f "${BASE_HARNESS}" && -f "${STRESS_HARNESS}" && -f "${SRC}" ]]
if [[ -n "${C2_EXPECTED_SOURCE_SHA:-}" ]]; then
  [[ "$(sha256sum "${SRC}" | awk '{print $1}')" == "${C2_EXPECTED_SOURCE_SHA}" ]]
fi
if [[ -n "${C2_EXPECTED_LIB_SHA:-}" ]]; then
  [[ "$(sha256sum "${LIB}" | awk '{print $1}')" == "${C2_EXPECTED_LIB_SHA}" ]]
fi
sha256sum "${LIB}" "${BASE_HARNESS}" "${STRESS_HARNESS}" "${SRC}" \
  | tee "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
"${PYTHON}" - <<'PY'
import platform
import torch
print({"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda})
PY
ldd "${LIB}" | tee "${ARTIFACT_DIR}/ldd-job${SLURM_JOB_ID}.log"

timeout --foreground --preserve-status --kill-after=10s 2400s \
  "${PYTHON}" "${STRESS_HARNESS}" \
  --library "${LIB}" \
  --base-harness "${BASE_HARNESS}" \
  --seeds 17,23,42,2024,314159,20260801,20260815,20260829 \
  --oracle-dtype float64 --atol 1e-4 --rtol 1e-3 \
  --warmup 30 --iters 200 --stability-repeats 4 \
  > "${JSON_RESULT}"

"${PYTHON}" - "${JSON_RESULT}" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["schema"] == "c2-native-c2-stress-perf-v1"
assert payload["all_gates_pass"] is True
assert payload["no_monkeypatch"] is True
assert payload["dispatch"] == "direct_torch.ops._C.native_c2_msa_decode"
assert payload["seed_count"] >= 8
assert payload["correctness_gate"] == {"atol": 1e-4, "oracle_dtype": "float64", "rtol": 1e-3}
for record in payload["seeds"]:
    assert record["all_gates_pass"] is True
    assert record["correctness"]["allclose"] is True
    assert record["correctness"]["finite_output"] is True
    assert record["caller_output"]["pointer_unchanged"] is True
    assert record["caller_output"]["timing_pointer_unchanged"] is True
    assert record["caller_output"]["return_is_none"] is True
    assert record["stability"]["bitwise_repeatable"] is True
    assert record["stability"]["repeat_pointers_unchanged"] is True
    assert record["stability"]["repeat_returns_none"] is True
    for key, value in record["latency"].items():
        if key != "method":
            assert math.isfinite(float(value)) and float(value) >= 0.0, (key, value)
    for value in record["throughput"].values():
        assert math.isfinite(float(value)) and float(value) > 0.0
PY

snapshot POST
sha256sum "${JSON_RESULT}" "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" \
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
