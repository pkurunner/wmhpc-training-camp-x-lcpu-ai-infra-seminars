#!/usr/bin/env bash
# Strict clean-allocation B300 audit for the one-SO vshard4 P2S3 comparison.
set -Eeuo pipefail

if [[ "${C1_VSHARD4_P2_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_VSHARD4_P2_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?usage: A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
: "${PATCHED_ROOT:?usage: A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
: "${REFERENCE_ROOT:?usage: A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
: "${LABEL:?usage: A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
: "${PYTHON_INCLUDE:?set PYTHON_INCLUDE to the selected Python include directory}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_vshard4_p2_${LABEL}_job${SLURM_JOB_ID:-none}.log"
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
command -v "$PYTHON_BIN"
command -v nvcc
[[ -f "$CUDA_HOME/include/cuda_runtime.h" && -f "$PYTHON_INCLUDE/Python.h" ]] || exit 89
echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_clean PRE || exit 90
echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'C1_BUILD_TARGET=%s\n' "${C1_BUILD_TARGET:-unspecified}"
printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short; printf 'PATCHED_STATUS_END\n'
sha256sum "$OWNED/apply_vshard4_prefetch2_patch.py" "$OWNED/vshard4_prefetch2.py" \
  "$OWNED/run_vshard4_prefetch2_final.py" "$OWNED/run_clean_vshard4_prefetch2_audit.sh" \
  "$OWNED/ptxas_audit.py" "$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so \
  "$PATCHED_ROOT/csrc/flash_kda.cpp" "$PATCHED_ROOT/csrc/fwd.h" "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu" \
  "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard.cuh" "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard_p2.cuh" \
  "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard4.cuh" "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard4_p2.cuh"

export CUDA_HOME
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"
RUNNER="$OWNED/run_vshard4_prefetch2_final.py"
echo "===== SMALL_EXACT_MATRIX ====="
"$PYTHON_BIN" "$RUNNER" --reference-root "$REFERENCE_ROOT" --T 256 --small-heads 1,2,4 --states all --no-bench \
  --json "$RESULTS_DIR/c1_vshard4_p2_${LABEL}_small_matrix.json"
require_clean BETWEEN_SMALL_AND_H12 || exit 93
echo "===== H12_BF16_EXACT_AND_FOUR_PATH_CYCLIC ====="
"$PYTHON_BIN" "$RUNNER" --reference-root "$REFERENCE_ROOT" --T 8192 --H 12 --states bf16 --warmup 30 --samples 1000 \
  --json "$RESULTS_DIR/c1_vshard4_p2_${LABEL}_h12_bf16_cyclic.json"
require_clean AFTER_H12 || exit 94
