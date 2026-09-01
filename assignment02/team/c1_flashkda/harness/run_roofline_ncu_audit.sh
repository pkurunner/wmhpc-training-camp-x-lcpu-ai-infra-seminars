#!/usr/bin/env bash
# Capture K1/K2's complete NCU evidence only inside an already granted clean
# allocation.  This script never submits, reserves, or holds a Slurm resource.
set -Eeuo pipefail

FLASH_ROOT="${1:?usage: run_roofline_ncu_audit.sh FLASH_ROOT FLA_ROOT PYTHON LABEL}"
FLA_ROOT="${2:?usage: run_roofline_ncu_audit.sh FLASH_ROOT FLA_ROOT PYTHON LABEL}"
PYTHON_BIN="${3:?usage: run_roofline_ncu_audit.sh FLASH_ROOT FLA_ROOT PYTHON LABEL}"
LABEL="${4:?usage: run_roofline_ncu_audit.sh FLASH_ROOT FLA_ROOT PYTHON LABEL}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A02_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_DIR="${LOG_DIR:-${A02_ROOT}/team/c1_flashkda/experiment_logs}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
REPORT="${LOG_DIR}/c1_ncu_full_${LABEL}_job${JOB_TAG}.ncu-rep"
CSV="${LOG_DIR}/c1_ncu_full_${LABEL}_job${JOB_TAG}.csv"
SUMMARY="${LOG_DIR}/c1_ncu_roofline_${LABEL}_job${JOB_TAG}.md"
LOG="${LOG_DIR}/c1_ncu_full_${LABEL}_job${JOB_TAG}.log"
NCU_BIN="${NCU_BIN:-ncu}"

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG}") 2>&1

gpu_query() {
    nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total,clocks.sm,power.draw \
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
    printf 'NCU_REPORT=%s\nNCU_CSV=%s\nNCU_SUMMARY=%s\nFINAL_RC=%d\n' \
        "${REPORT}" "${CSV}" "${SUMMARY}" "${rc}"
    exit "${rc}"
}
trap finish EXIT

echo "===== PRE_AUDIT ====="
date -Is
hostname
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' "${SLURM_JOB_ID:-}" "${CUDA_VISIBLE_DEVICES:-}"
gpu_query
apps="$(app_query || true)"
echo "PRE_COMPUTE_APPS_BEGIN"
printf '%s\n' "${apps}"
echo "PRE_COMPUTE_APPS_END"
if [[ -n "${apps//[[:space:]]/}" ]]; then
    echo "REFUSE: allocated GPU already has compute applications" >&2
    exit 90
fi

cd "${FLASH_ROOT}"
export PYTHONPATH="${FLASH_ROOT}:${FLA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -c 'import torch, flash_kda_C; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("extension=" + flash_kda_C.__file__)'
printf 'FLASH_COMMIT=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf snapshot)"
printf 'NCU_BIN=%s\n' "$(command -v "${NCU_BIN}")"
sha256sum "${SCRIPT_DIR}/ncu_single_case.py" "${SCRIPT_DIR}/summarize_ncu_roofline.py"

echo "===== NCU_FULL_K1_K2_FIXED_H96 ====="
"${NCU_BIN}" --set full --kernel-name-base function \
    -k 'regex:_flash_kda_fwd_(prepare|recurrence)' \
    --clock-control none --import-source yes --source-folders "${FLASH_ROOT}" \
    --export "${REPORT}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/ncu_single_case.py" \
        --T 8192 --H 96 --D 128

echo "===== EXPORT_RAW_CSV ====="
"${NCU_BIN}" --import "${REPORT}" --csv --page details > "${CSV}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_ncu_roofline.py" --csv "${CSV}" --out "${SUMMARY}"
if ! grep -q 'COUNTER_SET_COMPLETE' "${SUMMARY}"; then
    echo "FAIL: NCU full export lacks a complete K1/K2 counter set" >&2
    exit 30
fi
printf 'NCU_SUMMARY_HEAD\n'
sed -n '1,260p' "${SUMMARY}"
