#!/usr/bin/env bash
# Clean eight-B300 audit of the opt-in FLA dispatcher and TP critical path.
set -Eeuo pipefail

if [[ "${C1_TP8_DISPATCH_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_TP8_DISPATCH_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT}"
: "${FLA_ROOT:?set FLA_ROOT}"
: "${LABEL:?set LABEL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
: "${PYTHON_INCLUDE:?set PYTHON_INCLUDE to the selected Python include directory}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
TORCHRUN_BIN="$(dirname "$PYTHON_BIN")/torchrun"
AUDIT_WORLD_SIZE="${AUDIT_WORLD_SIZE:-8}"
TARGET_TP_DEGREE="${TARGET_TP_DEGREE:-8}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_tp8_dispatch_${LABEL}_tp8_fla_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
app_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
require_expected_b300() {
    local rows
    rows="$(gpu_query)" || return 92
    [[ "$(printf '%s\n' "$rows" | sed '/^[[:space:]]*$/d' | wc -l)" -eq "$AUDIT_WORLD_SIZE" ]] || return 1
    while IFS= read -r line; do
        [[ "$line" == *"B300"* && "$line" == *"10.3"* ]] || return 1
    done <<<"$rows"
}
require_clean() {
    local stage="$1" apps used
    apps="$(app_query)" || { echo "$stage: compute-app query failed" >&2; return 92; }
    used="$(memory_query)" || { echo "$stage: memory query failed" >&2; return 92; }
    echo "${stage}_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "${stage}_COMPUTE_APPS_END"
    printf '%s_MEMORY_USED_MIB=\n%s\n' "$stage" "$used"
    [[ -z "$apps" ]] || return 1
    [[ "$(printf '%s\n' "$used" | sed '/^[[:space:]]*$/d' | wc -l)" -eq "$AUDIT_WORLD_SIZE" ]] || return 1
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
[[ -x "$TORCHRUN_BIN" && -x "$(dirname "$PYTHON_BIN")/ninja" \
    && -f "$CUDA_HOME/include/cuda_runtime.h" \
    && -f "$PYTHON_INCLUDE/Python.h" \
    && -f "$PATCHED_ROOT/flash_kda_C.cpython-312-x86_64-linux-gnu.so" \
    && -f "$FLA_ROOT/fla/ops/backends/__init__.py" ]] || exit 89
[[ "$AUDIT_WORLD_SIZE" =~ ^[1-8]$ && "$TARGET_TP_DEGREE" -ge "$AUDIT_WORLD_SIZE" ]] || exit 87
require_expected_b300 || {
    echo "expected exactly $AUDIT_WORLD_SIZE visible B300 SM103 GPUs" >&2
    exit 88
}
echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_clean PRE || exit 90

echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'FLA_COMMIT=%s\n' "$(git -C "$FLA_ROOT" rev-parse HEAD)"
sha256sum \
    "$OWNED/auto_dispatch.py" \
    "$OWNED/fla_backend.py" \
    "$OWNED/test_auto_dispatch_policy.py" \
    "$OWNED/run_state_contracts.py" \
    "$OWNED/run_tp8_fla.py" \
    "$OWNED/run_clean_tp8_fla_audit.sh" \
    "$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so \
    "$FLA_ROOT/fla/ops/backends/__init__.py" \
    "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" \
    "$FLA_ROOT/fla/ops/kda/chunk.py"

export CUDA_HOME
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT:$FLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_B300_FLASH_KDA=1
export NCCL_ASYNC_ERROR_HANDLING=1
cd "$PATCHED_ROOT"

echo "===== CPU_POLICY_GATE ====="
"$PYTHON_BIN" "$OWNED/test_auto_dispatch_policy.py"
echo "===== TP8_REAL_FLA ====="
"$TORCHRUN_BIN" --standalone --nproc_per_node="$AUDIT_WORLD_SIZE" \
    "$OWNED/run_tp8_fla.py" --T 8192 --H 12 --warmup 30 --samples 300 \
    --expected-world-size "$AUDIT_WORLD_SIZE" --target-tp-degree "$TARGET_TP_DEGREE" \
    --json "$RESULTS_DIR/c1_tp8_dispatch_${LABEL}_tp8_fla.json"
require_clean AFTER_TP8_FLA || exit 93
