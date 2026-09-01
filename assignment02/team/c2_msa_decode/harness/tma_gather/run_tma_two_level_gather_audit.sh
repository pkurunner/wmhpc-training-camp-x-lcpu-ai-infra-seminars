#!/usr/bin/env bash
# Usage (inside a clean B300 allocation):
#   bash harness/tma_gather/run_tma_two_level_gather_audit.sh "$PWD" b300
# The script deliberately refuses a GPU with another compute process.  It
# builds a real SM100f TMA binary, records source hashes and CUDA event results,
# then verifies the GPU is clean again before reporting FINAL_RC=0.
set -Eeuo pipefail

C2_ROOT="${1:?usage: run_tma_two_level_gather_audit.sh C2_ROOT LABEL}"
LABEL="${2:?usage: run_tma_two_level_gather_audit.sh C2_ROOT LABEL}"
CUDA_BIN_DIR="${CUDA_BIN_DIR:-/usr/local/cuda/bin}"
export PATH="${CUDA_BIN_DIR}:$PATH"
NVCC="${NVCC:-${CUDA_BIN_DIR}/nvcc}"
ARCH="${ARCH:-100f}"
CUDA_INCLUDE_DIR="${CUDA_INCLUDE_DIR:-/usr/local/cuda/targets/x86_64-linux/include}"
SRC="$C2_ROOT/harness/tma_gather/tma_two_level_gather.cu"
SCRIPT="$C2_ROOT/harness/tma_gather/run_tma_two_level_gather_audit.sh"
BIN_DIR="$C2_ROOT/harness/tma_gather/bin"
LOG_DIR="$C2_ROOT/experiment_logs"
BIN="$BIN_DIR/tma_two_level_gather_${LABEL}"
LOG="$LOG_DIR/c2_tma_two_level_gather_${LABEL}_job${SLURM_JOB_ID:-none}.log"
JSON="$LOG_DIR/c2_tma_two_level_gather_${LABEL}_job${SLURM_JOB_ID:-none}.json"

mkdir -p "$BIN_DIR" "$LOG_DIR"
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
if [[ -n "$apps" ]]; then
    echo "REFUSE_SHARED_GPU=1"
    exit 90
fi

echo "===== TOOLCHAIN_AND_SOURCE ====="
"$NVCC" --version
sha256sum "$SRC" "$SCRIPT"
echo "ARCH=$ARCH"

echo "===== BUILD_REAL_TMA_BINARY ====="
"$NVCC" -O3 -std=c++17 \
    -I"$CUDA_INCLUDE_DIR" \
    -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
    -lcuda -o "$BIN" "$SRC"

echo "===== STATIC_TMA_SOURCE_CHECK ====="
grep -nE 'CUtensorMap|cuTensorMapEncodeTiled|cp\.async\.bulk\.tensor\.2d|fence\.proxy\.async' "$SRC"
echo "===== SASS_DUMP_NON_GATING_EVIDENCE ====="
if command -v cuobjdump >/dev/null 2>&1; then
    cuobjdump --dump-sass "$BIN" | grep -Ei 'UTCCP|TMA|CP\.ASYNC' || true
else
    echo "cuobjdump unavailable; runtime correctness and CUDA-event timing still run"
fi

echo "===== RUN_CORRECTNESS_AND_CUDA_EVENT_TIMING ====="
"$BIN" --batch 4 --topk 16 --logical-pages 64 --physical-pages 128 \
    --page-elems 1024 --warmup 20 --iters 100 --json "$JSON"
echo "JSON=$JSON"
echo "LOG=$LOG"
