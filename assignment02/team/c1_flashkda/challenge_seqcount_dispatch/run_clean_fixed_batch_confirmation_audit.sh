#!/usr/bin/env bash
# Clean one-GPU B300 confirmation for the pre-registered fixed B=2/3/4/6/8 matrix.
# This script only runs an already-built audited extension; it never rebuilds it.
set -Eeuo pipefail

if [[ "${C1_FIXED_BATCH_CONFIRMATION_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_FIXED_BATCH_CONFIRMATION_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT to the already-built audited comparison worktree}"
: "${LABEL:?set LABEL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
EXPECTED_SO_SHA256="8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_fixed_batch_confirmation_${LABEL}_job${SLURM_JOB_ID:-none}.log"
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
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
[[ "${#SO_PATHS[@]}" -eq 1 && -f "${SO_PATHS[0]}" ]] || {
    echo "expected exactly one prebuilt flash_kda_C.cpython-*-linux-gnu.so in PATCHED_ROOT" >&2
    exit 89
}
[[ "$(sha256sum "${SO_PATHS[0]}" | awk '{print $1}')" == "$EXPECTED_SO_SHA256" ]] || {
    echo "audited extension SHA256 gate failed" >&2
    exit 87
}
[[ "$(gpu_query | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || exit 88

echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_clean PRE || exit 90

echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short; printf 'PATCHED_STATUS_END\n'
sha256sum \
    "$OWNED/run_fixed_batch_confirmation.py" \
    "$OWNED/run_clean_fixed_batch_confirmation_audit.sh" \
    "$OWNED/run_seqcount_dispatch.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/prefetch2.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2/vshard4_prefetch2.py" \
    "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" \
    "${SO_PATHS[0]}" \
    "$PATCHED_ROOT/flash_kda/__init__.py" \
    "$PATCHED_ROOT/csrc/flash_kda.cpp" \
    "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu"

echo "===== PYTHON_COMPILE_GATE ====="
"$PYTHON_BIN" -m py_compile "$OWNED/run_fixed_batch_confirmation.py"

# No setup.py, NVCC, patch-generator, or source mutation is permitted below.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_FIXED_BATCH_CLEAN_GPU_GATES=1
cd "$PATCHED_ROOT"
echo "===== FIXED_BATCH_CONFIRMATION ====="
"$PYTHON_BIN" "$OWNED/run_fixed_batch_confirmation.py" \
    --json "$RESULTS_DIR/c1_fixed_batch_confirmation_${LABEL}.json"
require_clean AFTER_CONFIRMATION || exit 93
