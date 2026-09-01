#!/usr/bin/env bash
# Run only inside the main session's already-granted *clean B300* allocation.
# It never calls salloc/srun/sbatch and therefore cannot reserve or occupy a GPU.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSIGNMENT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SRC="${SCRIPT_DIR}/hmma_tcgen05_fair.cu"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT_DIR="${OUT_DIR:-${ASSIGNMENT_DIR}/team/c1_flashkda/experiment_logs}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/c1_hmma_tcgen05_same_work_b300_job${JOB_TAG}.json}"
LOG_PATH="${LOG_PATH:-${OUT_DIR}/c1_hmma_tcgen05_same_work_b300_job${JOB_TAG}.log}"
SASS_PATH="${SASS_PATH:-${OUT_DIR}/c1_hmma_tcgen05_same_work_b300_job${JOB_TAG}.sass}"
ARCH="${ARCH:-103a}"
CUDA_BIN_DIR="${CUDA_BIN_DIR:-/usr/local/cuda/bin}"
export PATH="${CUDA_BIN_DIR}:$PATH"
NVCC_BIN="${NVCC_BIN:-${CUDA_BIN_DIR}/nvcc}"
CUOBJDUMP_BIN="${CUOBJDUMP_BIN:-${CUDA_BIN_DIR}/cuobjdump}"
CUDA_INCLUDE_DIR="${CUDA_INCLUDE_DIR:-/usr/local/cuda/targets/x86_64-linux/include}"
WORK_DIR="${WORK_DIR:-$(mktemp -d -t c1_hmma_tcgen05.XXXXXX)}"
BIN="${WORK_DIR}/c1_hmma_tcgen05_fair"

mkdir -p "${OUT_DIR}"
exec > >(tee "${LOG_PATH}") 2>&1

gpu_query() {
    nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total \
        --format=csv,noheader
}
app_query() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader
}
finish() {
    local rc=$?
    trap - EXIT
    echo "===== POST_AUDIT ====="
    date -Is
    gpu_query || true
    local apps
    apps="$(app_query || true)"
    echo "POST_COMPUTE_APPS_BEGIN"
    printf '%s\n' "${apps}"
    echo "POST_COMPUTE_APPS_END"
    if [[ -n "${apps//[[:space:]]/}" && "${rc}" -eq 0 ]]; then rc=91; fi
    printf 'C1_HMMA_TCGEN05_JSON=%s\n' "${OUT_JSON}"
    printf 'C1_HMMA_TCGEN05_SASS=%s\n' "${SASS_PATH}"
    printf 'C1_HMMA_TCGEN05_FINAL_RC=%d\n' "${rc}"
    exit "${rc}"
}
trap finish EXIT

echo "===== PRE_AUDIT ====="
date -Is
hostname
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s ARCH=%s\n' \
    "${SLURM_JOB_ID:-}" "${CUDA_VISIBLE_DEVICES:-}" "${ARCH}"
gpu_query
apps="$(app_query || true)"
echo "PRE_COMPUTE_APPS_BEGIN"
printf '%s\n' "${apps}"
echo "PRE_COMPUTE_APPS_END"
if [[ -n "${apps//[[:space:]]/}" ]]; then
    echo "REFUSE: allocated GPU has a compute application before C1 fair microbench" >&2
    exit 90
fi

echo "===== BUILD ====="
"${NVCC_BIN}" --version
printf 'ARCH=%s TILES=%s WARMUP=%s ITERS=%s REPEATS=%s\n' \
    "${ARCH}" "${TILES:-4096}" "${WARMUP:-20}" "${ITERS:-50}" "${REPEATS:-7}"
sha256sum "${SRC}" "${SCRIPT_DIR}/run_hmma_tcgen05_audit.sh"
"${NVCC_BIN}" -O3 -std=c++17 -I"${CUDA_INCLUDE_DIR}" -I"${ASSIGNMENT_DIR}/cuda" \
    -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
    -o "${BIN}" "${SRC}"

echo "===== SASS_AUDIT ====="
"${CUOBJDUMP_BIN}" --dump-sass "${BIN}" > "${SASS_PATH}"
printf 'SASS_HMMA=%s\n' "$(grep -Ec '(^|[[:space:]])HMMA[.]' "${SASS_PATH}" || true)"
printf 'SASS_UTCHMMA=%s\n' "$(grep -Ec 'UTCHMMA' "${SASS_PATH}" || true)"
printf 'SASS_LDTM=%s\n' "$(grep -Ec 'LDTM' "${SASS_PATH}" || true)"
grep -m 100 -E 'Function :|HMMA|UTCHMMA|LDTM' "${SASS_PATH}" || true
if ! grep -Eq '(^|[[:space:]])HMMA[.]' "${SASS_PATH}"; then
    echo "FAIL: the intended mma.sync reference did not lower to HMMA in SASS" >&2
    exit 20
fi
if ! grep -q 'UTCHMMA' "${SASS_PATH}"; then
    echo "FAIL: the BF16 tcgen05 path did not lower to UTCHMMA in SASS" >&2
    exit 21
fi

echo "===== SAME_LOGICAL_WORK_GATE_AND_TIMING ====="
"${BIN}" --json "${OUT_JSON}" \
    --tiles "${TILES:-4096}" --warmup "${WARMUP:-20}" \
    --iters "${ITERS:-50}" --repeats "${REPEATS:-7}"
python3 -m json.tool "${OUT_JSON}" > /dev/null
echo "JSON_STRICT_PARSE=PASS"
