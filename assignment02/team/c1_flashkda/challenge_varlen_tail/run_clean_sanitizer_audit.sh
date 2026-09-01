#!/usr/bin/env bash
# Clean B300 compute-sanitizer gate for tail/batch/varlen candidates.
set -Eeuo pipefail

if [[ "${C1_VARLEN_SANITIZER_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_VARLEN_SANITIZER_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT}"
: "${LABEL:?set LABEL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
: "${PYTHON_INCLUDE:?set PYTHON_INCLUDE to the selected Python include directory}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_varlen_tail"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
SANITIZER="$CUDA_HOME/bin/compute-sanitizer"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_varlen_sanitizer_${LABEL}_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
app_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
require_clean() {
    local stage="$1" apps used
    apps="$(app_query)" || return 92
    used="$(memory_query)" || return 92
    echo "${stage}_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "${stage}_COMPUTE_APPS_END"
    printf '%s_MEMORY_USED_MIB=%s\n' "$stage" "$used"
    [[ -z "$apps" ]] || return 1
    [[ "$(printf '%s\n' "$used" | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || return 1
    while IFS= read -r line; do [[ "$line" =~ ^[[:space:]]*0[[:space:]]*$ ]] || return 1; done <<<"$used"
}
finish() {
    local rc=$?; trap - EXIT
    echo "===== POST_AUDIT ====="; date -Is; gpu_query || rc=92
    require_clean POST || rc=91
    echo "FINAL_RC=$rc"; exit "$rc"
}
trap finish EXIT

echo "===== ENVIRONMENT_GATE ====="
[[ -x "$SANITIZER" && -x "$(dirname "$PYTHON_BIN")/ninja" \
    && -f "$CUDA_HOME/include/cuda_runtime.h" \
    && -f "$PYTHON_INCLUDE/Python.h" \
    && -f "$PATCHED_ROOT/flash_kda_C.cpython-312-x86_64-linux-gnu.so" ]] || exit 89
[[ "$(gpu_query | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || exit 88
echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_clean PRE || exit 90
echo "===== SOURCE_IDENTITY ====="
sha256sum \
    "$OWNED/run_sanitizer_smoke.py" \
    "$OWNED/run_varlen_tail.py" \
    "$OWNED/run_clean_sanitizer_audit.sh" \
    "$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so \
    "$PATCHED_ROOT/csrc/smxx/fwd_kernel1.cuh" \
    "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard_p2.cuh" \
    "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard4_p2.cuh"

export CUDA_HOME
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"
echo "===== COMPUTE_SANITIZER_MEMCHECK ====="
"$SANITIZER" --tool memcheck --padding 32 --error-exitcode 86 --target-processes all \
    "$PYTHON_BIN" "$OWNED/run_sanitizer_smoke.py" \
    --json "$RESULTS_DIR/c1_varlen_sanitizer_${LABEL}.json"
require_clean AFTER_SANITIZER || exit 93
