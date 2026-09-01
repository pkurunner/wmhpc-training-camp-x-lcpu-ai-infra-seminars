#!/usr/bin/env bash
set -Eeuo pipefail

C2_ROOT="${1:?usage: run_challenge_audit.sh C2_ROOT LABEL}"
LABEL="${2:?usage: run_challenge_audit.sh C2_ROOT LABEL}"
PYTHON="${PYTHON:-python}"
LOG_DIR="$C2_ROOT/experiment_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/c2_challenge_${LABEL}_job${SLURM_JOB_ID:-none}.log"
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
    if [[ -n "$apps" && "$rc" -eq 0 ]]; then
        rc=91
    fi
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
if [[ -n "$apps" ]]; then
    exit 90
fi

cd "$C2_ROOT"
"$PYTHON" -c 'import torch,triton; print("torch=" + torch.__version__); print("triton=" + triton.__version__); print("cuda=" + str(torch.version.cuda))'

echo "===== CORRECTNESS_AUTO_ALL_BATCHES_ALL_MODES ====="
"$PYTHON" -m challenge.cli correctness --all-batches --all-modes --chunks auto

echo "===== CORRECTNESS_ALL_CHUNKS_BF16 ====="
"$PYTHON" -m challenge.cli correctness --all-batches --storage-mode bf16 \
    --chunks 1 2 4 8 16

echo "===== BENCHMARK_CHUNK_SWEEP_BF16 ====="
"$PYTHON" -m challenge.cli benchmark --all-batches --storage-mode bf16 \
    --chunks auto 1 2 4 8 16 --warmup 20 --samples 21 --inner 20
