#!/usr/bin/env bash
set -Eeuo pipefail

C2_ROOT="${1:?usage: run_official_msa_q1_audit.sh C2_ROOT MSA_ROOT LABEL}"
MSA_ROOT="${2:?usage: run_official_msa_q1_audit.sh C2_ROOT MSA_ROOT LABEL}"
LABEL="${3:?usage: run_official_msa_q1_audit.sh C2_ROOT MSA_ROOT LABEL}"
PYTHON="${PYTHON:-python}"
LOG_DIR="$C2_ROOT/experiment_logs"
LOG="$LOG_DIR/c2_cutlass_q1_${LABEL}_job${SLURM_JOB_ID:-none}.log"
JSON="$LOG_DIR/c2_cutlass_q1_${LABEL}_job${SLURM_JOB_ID:-none}.json"
mkdir -p "$LOG_DIR"
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
KV_LEN="${KV_LEN:-8192}"
BATCH="${BATCH:-16}"
PYTHONPATH="$MSA_ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" harness/official_msa_cutlass_bench.py \
    --batch "$BATCH" --kv-len "$KV_LEN" \
    --warmup 20 --repetitions 100 | tee "$JSON"
echo "CUTLASS_Q1_JSON=$JSON"
