#!/usr/bin/env bash
set -Eeuo pipefail

FLASH_ROOT="${1:?usage: run_instruction_audit.sh FLASH_ROOT FLA_ROOT PYTHON LOG_DIR}"
FLA_ROOT="${2:?usage: run_instruction_audit.sh FLASH_ROOT FLA_ROOT PYTHON LOG_DIR}"
PYTHON="${3:?usage: run_instruction_audit.sh FLASH_ROOT FLA_ROOT PYTHON LOG_DIR}"
LOG_DIR="${4:?usage: run_instruction_audit.sh FLASH_ROOT FLA_ROOT PYTHON LOG_DIR}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/c1_instruction_audit_b300_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() {
    nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader
}
app_query() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
}
finish() {
    rc=$?
    trap - EXIT
    echo "===== POST_AUDIT ====="
    date -Is
    gpu_query || true
    apps="$(app_query || true)"
    echo "POST_COMPUTE_APPS_BEGIN"
    printf '%s\n' "$apps"
    echo "POST_COMPUTE_APPS_END"
    if [[ -n "$apps" && "$rc" -eq 0 ]]; then rc=91; fi
    echo "FINAL_RC=$rc"
    exit "$rc"
}
trap finish EXIT

echo "===== PRE_AUDIT ====="
date -Is
hostname
gpu_query
apps="$(app_query || true)"
echo "PRE_COMPUTE_APPS_BEGIN"
printf '%s\n' "$apps"
echo "PRE_COMPUTE_APPS_END"
if [[ -n "$apps" ]]; then exit 90; fi

cd "$FLASH_ROOT"
SO_PATH="$(ls -1 flash_kda_C*.so | head -1)"
SASS_FILE="/tmp/c1_flashkda_job${SLURM_JOB_ID:-none}.sass"
/usr/local/cuda/bin/cuobjdump --dump-sass "$SO_PATH" > "$SASS_FILE"
echo "===== SASS_COUNTS ====="
printf 'HMMA=%s\n' "$(grep -Ec 'HMMA' "$SASS_FILE" || true)"
printf 'WGMMA=%s\n' "$(grep -Ec 'WGMMA' "$SASS_FILE" || true)"
printf 'UTCOMMA=%s\n' "$(grep -Ec 'UTCOMMA' "$SASS_FILE" || true)"
printf 'UTCHMMA=%s\n' "$(grep -Ec 'UTCHMMA' "$SASS_FILE" || true)"
printf 'UTC_ANY_MMA=%s\n' "$(grep -Eic 'UTC.*MMA' "$SASS_FILE" || true)"
echo "===== FIRST_HMMA_INSTRUCTIONS ====="
grep -m 80 -E 'Function :|HMMA' "$SASS_FILE"

export PYTHONPATH="$FLASH_ROOT:$FLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
NCU_CSV="$LOG_DIR/c1_ncu_basic_b300_job${SLURM_JOB_ID:-none}.csv"
echo "===== NCU_BASIC_FIXED_H96 ====="
/usr/local/cuda/bin/ncu --set basic --kernel-name-base function \
    -k 'regex:_flash_kda_fwd_(prepare|recurrence)' --clock-control none \
    --csv --log-file "$NCU_CSV" \
    "$PYTHON" benchmarks/bench_fwd.py --mode fixed --H 96 --D 128 \
    --warmup 0 --iters 1 --repeats 1
echo "NCU_CSV=$NCU_CSV"
grep -m 120 -E 'Kernel Name|_flash_kda_fwd_(prepare|recurrence)|SOL|DRAM|Tensor' "$NCU_CSV" || true
