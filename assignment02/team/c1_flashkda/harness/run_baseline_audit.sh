#!/usr/bin/env bash
set -Eeuo pipefail

FLASH_ROOT="${1:?usage: run_baseline_audit.sh FLASH_ROOT FLA_ROOT LABEL}"
FLA_ROOT="${2:?usage: run_baseline_audit.sh FLASH_ROOT FLA_ROOT LABEL}"
LABEL="${3:?usage: run_baseline_audit.sh FLASH_ROOT FLA_ROOT LABEL}"
PYTHON="${PYTHON:-python}"
WARMUP="${WARMUP:-30}"
ITERS="${ITERS:-200}"
REPEATS="${REPEATS:-5}"
RUN_VARLEN="${RUN_VARLEN:-1}"
LOG_DIR="${LOG_DIR:-$FLASH_ROOT/assignment02_logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/c1_baseline_${LABEL}_job${SLURM_JOB_ID:-none}.log"
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

cd "$FLASH_ROOT"
export PYTHONPATH="$FLASH_ROOT:$FLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
printf 'FLASH_COMMIT=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf snapshot)"
printf 'CUTLASS_PIN=5c149f52a436782210263fb2f19b354443a61c6a\n'
printf 'FLA_PIN=a3edffc\n'
"$PYTHON" -c 'import torch, flash_kda_C; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("device=" + torch.cuda.get_device_name(0)); print("extension=" + flash_kda_C.__file__)'

for heads in 96 64; do
    echo "===== OFFICIAL_FIXED_H${heads} ====="
    "$PYTHON" benchmarks/bench_fwd.py --mode fixed --H "$heads" --D 128 \
        --warmup "$WARMUP" --iters "$ITERS" --repeats "$REPEATS"
done

if [[ "$RUN_VARLEN" == 1 ]]; then
    echo "===== OFFICIAL_VARLEN_H96 ====="
    "$PYTHON" benchmarks/bench_fwd.py --mode varlen --H 96 --D 128 \
        --warmup "$WARMUP" --iters "$ITERS" --repeats "$REPEATS"
fi
