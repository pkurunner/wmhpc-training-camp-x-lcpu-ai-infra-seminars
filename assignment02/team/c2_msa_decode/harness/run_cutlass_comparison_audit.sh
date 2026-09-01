#!/usr/bin/env bash
set -Eeuo pipefail

C2_ROOT="${1:?usage: run_cutlass_comparison_audit.sh C2_ROOT MSA_ROOT LABEL}"
MSA_ROOT="${2:?usage: run_cutlass_comparison_audit.sh C2_ROOT MSA_ROOT LABEL}"
LABEL="${3:?usage: run_cutlass_comparison_audit.sh C2_ROOT MSA_ROOT LABEL}"
PYTHON="${PYTHON:-python}"
LOG_DIR="$C2_ROOT/experiment_logs"
LOG="$LOG_DIR/c2_cutlass_compare_${LABEL}_job${SLURM_JOB_ID:-none}.log"
CSV="$LOG_DIR/c2_cutlass_msa_${LABEL}_job${SLURM_JOB_ID:-none}.csv"
MANIFEST="$LOG_DIR/c2_challenge_b300_mode_gate.json"
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

echo "===== SOURCE_PINS ====="
git -C "$MSA_ROOT" rev-parse HEAD
git -C "$MSA_ROOT" submodule status
"$PYTHON" -c 'import torch; import fmha_sm100; print("torch=" + torch.__version__); print("fmha_sm100=" + fmha_sm100.__file__)'

echo "===== OFFICIAL_MINIMAX_MSA_CUTLASS_SPARSE_DECODE_B16 ====="
PYTHONPATH="$MSA_ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$MSA_ROOT/benchmarks/bench_sparse_attention_ops.py" \
    --sections sparse_decode --dtype fp8 --decode-k 4096 --decode-b 16 \
    --topk 16 --head-dim 128 --blk-kv 128 --dry-run-ms 50 --repeat-ms 200 \
    --output "$CSV"

echo "===== OUR_TRITON_SELECTED_POLICY_B16_FP8_SCALAR ====="
cd "$C2_ROOT"
"$PYTHON" -m challenge.cli final-benchmark --chunks selected \
    --manifest "$MANIFEST" --storage-mode fp8-scalar \
    --warmup 10 --samples 21 --inner 20 --no-cudagraph

echo "CUTLASS_CSV=$CSV"
