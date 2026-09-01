#!/usr/bin/env bash
# Clean one-GPU B300 sequence-count/head boundary audit.  This script never rebuilds FlashKDA.
set -Eeuo pipefail

if [[ "${C1_SEQCOUNT_DISPATCH_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_SEQCOUNT_DISPATCH_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT to the already-built one-SO comparison worktree}"
: "${REFERENCE_ROOT:?set REFERENCE_ROOT}"
: "${LABEL:?set LABEL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_seqcount_dispatch_${LABEL}_job${SLURM_JOB_ID:-none}.log"
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
command -v "$PYTHON_BIN"
[[ -f "$PATCHED_ROOT/flash_kda_C.cpython-312-x86_64-linux-gnu.so" \
    && -f "$REFERENCE_ROOT/tests/torch_ref.py" ]] || exit 89
[[ "$(gpu_query | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || exit 88
echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_clean PRE || exit 90

echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
sha256sum \
    "$OWNED/run_seqcount_dispatch.py" \
    "$OWNED/run_clean_seqcount_dispatch_audit.sh" \
    "$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so \
    "$PATCHED_ROOT/csrc/flash_kda.cpp" \
    "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu"

echo "===== PYTHON_COMPILE_GATE ====="
"$PYTHON_BIN" -m py_compile "$OWNED/run_seqcount_dispatch.py"

# The callable extension must already exist.  Do not invoke setup.py, nvcc, or a patch generator here.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"
echo "===== SEQUENCE_COUNT_HEAD_MATRIX ====="
"$PYTHON_BIN" "$OWNED/run_seqcount_dispatch.py" \
    --reference-root "$REFERENCE_ROOT" --warmup 100 --samples 1000 \
    --json "$RESULTS_DIR/c1_seqcount_dispatch_${LABEL}.json"
require_clean AFTER_MATRIX || exit 93
