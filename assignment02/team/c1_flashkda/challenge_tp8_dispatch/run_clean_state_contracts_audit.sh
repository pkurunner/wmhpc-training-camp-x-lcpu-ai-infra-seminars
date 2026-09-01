#!/usr/bin/env bash
# Clean B300 prerequisite gate for the exact state contracts used by FLA.
set -Eeuo pipefail

if [[ "${C1_TP8_DISPATCH_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_TP8_DISPATCH_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT}"
: "${REFERENCE_ROOT:?set REFERENCE_ROOT}"
: "${LABEL:?set LABEL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
: "${PYTHON_INCLUDE:?set PYTHON_INCLUDE to the selected Python include directory}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_tp8_dispatch_${LABEL}_state_contracts_job${SLURM_JOB_ID:-none}.log"
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
    [[ -z "$apps" ]] || return 1
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
command -v "$PYTHON_BIN"
[[ -x "$(dirname "$PYTHON_BIN")/ninja" && -f "$CUDA_HOME/include/cuda_runtime.h" \
    && -f "$PYTHON_INCLUDE/Python.h" \
    && -f "$PATCHED_ROOT/flash_kda_C.cpython-312-x86_64-linux-gnu.so" ]] || exit 89
echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_clean PRE || exit 90
echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
sha256sum "$OWNED/run_state_contracts.py" "$OWNED/run_clean_state_contracts_audit.sh" \
    "$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so \
    "$PATCHED_ROOT/csrc/flash_kda.cpp" "$PATCHED_ROOT/csrc/fwd.h"

export CUDA_HOME
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"
echo "===== H12_REAL_FLA_STATE_CONTRACTS ====="
"$PYTHON_BIN" "$OWNED/run_state_contracts.py" \
    --reference-root "$REFERENCE_ROOT" --T 8192 --H 12 \
    --contracts none,bf16_both,fp32_both,fp32_final_only --warmup 30 --samples 500 \
    --json "$RESULTS_DIR/c1_tp8_dispatch_${LABEL}_h12_state_contracts.json"
require_clean AFTER_STATE_CONTRACTS || exit 93
