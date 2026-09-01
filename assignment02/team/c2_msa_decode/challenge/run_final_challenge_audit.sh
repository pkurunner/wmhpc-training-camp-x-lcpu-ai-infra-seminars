#!/usr/bin/env bash
set -Eeuo pipefail

C2_ROOT="${1:?usage: run_final_challenge_audit.sh C2_ROOT LABEL}"
LABEL="${2:?usage: run_final_challenge_audit.sh C2_ROOT LABEL}"
PYTHON="${PYTHON:-python}"
LOG_DIR="$C2_ROOT/experiment_logs"
MANIFEST="$LOG_DIR/c2_challenge_${LABEL}_gate.json"
LOG="$LOG_DIR/c2_challenge_${LABEL}_job${SLURM_JOB_ID:-none}.log"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG") 2>&1

gpu_query() {
    nvidia-smi \
        --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total,clocks.sm,power.draw \
        --format=csv,noheader
}

app_query() {
    nvidia-smi \
        --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader
}

post_audit() {
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
trap post_audit EXIT

echo "===== PRE_AUDIT ====="
date -Is
hostname
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "${SLURM_JOB_ID:-}" "${CUDA_VISIBLE_DEVICES:-}"
gpu_query
apps="$(app_query || true)"
echo "PRE_COMPUTE_APPS_BEGIN"
printf '%s\n' "$apps"
echo "PRE_COMPUTE_APPS_END"
if [[ -n "$apps" ]]; then exit 90; fi

cd "$C2_ROOT"
"$PYTHON" -c 'import torch,triton; print("torch=" + torch.__version__); print("triton=" + triton.__version__); print("cuda=" + str(torch.version.cuda))'

echo "===== FINAL_GATE_B1_B4_B8_B16_ALL_STORAGE_MODES ====="
"$PYTHON" -m challenge.cli final-gate --chunks selected --manifest "$MANIFEST"

for mode in bf16 fp8-scalar fp8-token; do
    echo "===== FINAL_FAIR_BENCHMARK_${mode} ====="
    "$PYTHON" -m challenge.cli final-benchmark --chunks selected \
        --manifest "$MANIFEST" --storage-mode "$mode" \
        --warmup 5 --samples 9 --inner 10 --no-cudagraph
done

echo "===== FINAL_CUDAGRAPH_BF16 ====="
"$PYTHON" -m challenge.cli final-benchmark --chunks selected \
    --manifest "$MANIFEST" --storage-mode bf16 \
    --warmup 5 --samples 9 --inner 10
