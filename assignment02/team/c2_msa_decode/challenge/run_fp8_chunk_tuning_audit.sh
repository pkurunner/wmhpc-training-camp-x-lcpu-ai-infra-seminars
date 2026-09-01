#!/usr/bin/env bash
set -Eeuo pipefail

C2_ROOT="${1:?usage: run_fp8_chunk_tuning_audit.sh C2_ROOT LABEL}"
LABEL="${2:?usage: run_fp8_chunk_tuning_audit.sh C2_ROOT LABEL}"
PYTHON="${PYTHON:-python}"
LOG="$C2_ROOT/experiment_logs/c2_fp8_chunk_tuning_${LABEL}_job${SLURM_JOB_ID:-none}.log"
mkdir -p "$C2_ROOT/experiment_logs"
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

cd "$C2_ROOT"
for mode in fp8-scalar fp8-token; do
    echo "===== ${mode}_CHUNK_SWEEP ====="
    "$PYTHON" -m challenge.cli benchmark --all-batches \
        --storage-mode "$mode" --chunks 1 2 4 8 16 \
        --warmup 3 --samples 5 --inner 10 --no-cudagraph
done
