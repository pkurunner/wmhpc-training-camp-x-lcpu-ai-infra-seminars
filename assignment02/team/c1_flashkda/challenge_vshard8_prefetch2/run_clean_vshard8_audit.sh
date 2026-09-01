#!/usr/bin/env bash
# Strict clean-allocation exact/resource/performance audit for vshard8-P2.
set -Eeuo pipefail

if [[ "${C1_VSHARD8_P2_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_VSHARD8_P2_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT}"
: "${REFERENCE_ROOT:?set REFERENCE_ROOT}"
: "${LABEL:?set LABEL}"
: "${PYTHON_INCLUDE:?set PYTHON_INCLUDE}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_vshard8_prefetch2"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_vshard8_p2_${LABEL}_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
app_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
require_clean() {
    local stage="$1" apps used
    apps="$(app_query)" || { echo "$stage: compute-app query failed" >&2; return 92; }
    used="$(memory_query)" || { echo "$stage: memory query failed" >&2; return 92; }
    echo "${stage}_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "${stage}_COMPUTE_APPS_END"
    printf '%s_MEMORY_USED_MIB=%s\n' "$stage" "$used"
    [[ -z "$apps" && "$used" =~ ^[[:space:]]*0[[:space:]]*$ ]]
}
finish() {
    local rc=$?; trap - EXIT
    echo "===== POST_AUDIT ====="; date -Is; gpu_query || rc=92
    require_clean POST || rc=91
    echo "FINAL_RC=$rc"; exit "$rc"
}
trap finish EXIT

echo "===== ENVIRONMENT_GATE ====="
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 89
command -v "$PYTHON_BIN"
command -v nvcc
[[ -f "$CUDA_HOME/include/cuda_runtime.h" && -f "$PYTHON_INCLUDE/Python.h" ]] || exit 89
echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_clean PRE || exit 90
echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short; printf 'PATCHED_STATUS_END\n'
sha256sum "$OWNED/apply_vshard8_prefetch2_patch.py" "$OWNED/vshard8_prefetch2.py" \
  "$OWNED/run_vshard8_final.py" "$OWNED/run_clean_vshard8_audit.sh" "$OWNED/ptxas_audit.py" \
  "$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so \
  "$PATCHED_ROOT/csrc/flash_kda.cpp" "$PATCHED_ROOT/csrc/fwd.h" \
  "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu" "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard8_p2.cuh"

export CUDA_HOME
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"
RUNNER="$OWNED/run_vshard8_final.py"
echo "===== SMALL_ALL_CONTRACT_TORCH_REF_EXACT ====="
"$PYTHON_BIN" "$RUNNER" --reference-root "$REFERENCE_ROOT" --T 256 --heads 1,2,4 \
  --candidate vshard8_p2 \
  --contracts none,bf16_both,fp32_both,fp32_final_only --torch-ref --no-bench \
  --json "$RESULTS_DIR/c1_vshard8_p2_${LABEL}_small_all_contracts.json"
require_clean BETWEEN_SMALL_AND_H12 || exit 93
echo "===== H12_ALL_CONTRACT_EXACT_AND_CYCLIC ====="
"$PYTHON_BIN" "$RUNNER" --reference-root "$REFERENCE_ROOT" --T 8192 --heads 12 \
  --candidate vshard8_p2 \
  --contracts none,bf16_both,fp32_both,fp32_final_only --warmup 30 --samples 1000 \
  --json "$RESULTS_DIR/c1_vshard8_p2_${LABEL}_h12_all_contracts.json"
require_clean AFTER_H12 || exit 94
