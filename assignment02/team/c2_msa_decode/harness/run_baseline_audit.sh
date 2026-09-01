#!/usr/bin/env bash
set -Eeuo pipefail

C2ROOT="${1:?usage: run_baseline_audit.sh /absolute/path/to/c2_msa_decode [label]}"
LABEL="${2:-gpu}"
PYTHON="${PYTHON:-python}"
LOG_DIR="$C2ROOT/experiment_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/c2_baseline_${LABEL}_job${SLURM_JOB_ID:-none}.log"
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
    post_apps="$(app_query || true)"
    echo "POST_COMPUTE_APPS_BEGIN"
    printf '%s\n' "$post_apps"
    echo "POST_COMPUTE_APPS_END"
    if [[ -n "$post_apps" && "$rc" -eq 0 ]]; then
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
for wait_second in $(seq 0 120); do
    entry_apps="$(app_query || true)"
    [[ -z "$entry_apps" ]] && break
    printf 'wait_second=%d apps=%s\n' "$wait_second" "$entry_apps"
    sleep 1
done
if [[ -n "${entry_apps:-}" ]]; then
    echo "PRE_AUDIT_FAIL: allocated GPU remained occupied"
    exit 90
fi
sleep 2
gpu_query
pre_apps="$(app_query || true)"
echo "PRE_COMPUTE_APPS_BEGIN"
printf '%s\n' "$pre_apps"
echo "PRE_COMPUTE_APPS_END"
if [[ -n "$pre_apps" ]]; then
    exit 90
fi

cd "$C2ROOT"
"$PYTHON" -c 'import torch,triton; print("torch=" + torch.__version__); print("triton=" + triton.__version__); print("cuda=" + torch.version.cuda)'

# The task explicitly requires profiling the existing Triton baseline before
# any challenge kernel is designed.  Keep these four steps first in the log.
for batch in 1 4 8 16; do
    echo "===== PROFILE_BASELINE_B${batch} ====="
    "$PYTHON" -m harness.cli profile --batch "$batch" --storage-mode bf16 \
        --warmup 20 --profile-steps 10 --row-limit 30 \
        --trace "$LOG_DIR/c2_baseline_${LABEL}_b${batch}_trace.json"
done

echo "===== CORRECTNESS_ALL_MODES ====="
"$PYTHON" -m harness.cli correctness --all-batches \
    --storage-modes bf16 fp8-scalar fp8-token

for mode in bf16 fp8-scalar fp8-token; do
    echo "===== BENCHMARK_${mode} ====="
    "$PYTHON" -m harness.cli benchmark --all-batches --storage-mode "$mode" \
        --warmup 20 --repetitions 100
done

