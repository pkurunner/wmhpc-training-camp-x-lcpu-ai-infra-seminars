#!/usr/bin/env bash
# Clean-allocation audit wrapper for C1 V=32 / four-CTA vshard4.
set -Eeuo pipefail

A02_ROOT="${1:?usage: run_clean_vshard4_audit.sh A02_ROOT PATCHED_ROOT LABEL}"
PATCHED_ROOT="${2:?usage: run_clean_vshard4_audit.sh A02_ROOT PATCHED_ROOT LABEL}"
LABEL="${3:?usage: run_clean_vshard4_audit.sh A02_ROOT PATCHED_ROOT LABEL}"
PYTHON="${PYTHON:-python}"
LOG_DIR="$A02_ROOT/team/c1_flashkda/challenge_vshard4/results"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/c1_vshard4_${LABEL}_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
app_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_used_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
finish() {
    rc=$?; trap - EXIT
    echo "===== POST_AUDIT ====="; date -Is; gpu_query || rc=92
    if ! apps="$(app_query)"; then apps="QUERY_FAILED"; rc=92; fi
    if ! used="$(memory_used_query)"; then used="QUERY_FAILED"; rc=92; fi
    echo "POST_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "POST_COMPUTE_APPS_END"
    printf 'POST_MEMORY_USED_MIB=%s\n' "$used"
    if [[ -n "$apps" || ! "$used" =~ ^[[:space:]]*0[[:space:]]*$ ]]; then rc=91; fi
    echo "FINAL_RC=$rc"; exit "$rc"
}
trap finish EXIT

echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
apps="$(app_query)"; used="$(memory_used_query)"
echo "PRE_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "PRE_COMPUTE_APPS_END"
printf 'PRE_MEMORY_USED_MIB=%s\n' "$used"
if [[ -n "$apps" || ! "$used" =~ ^[[:space:]]*0[[:space:]]*$ ]]; then exit 90; fi

echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'C1_BUILD_TARGET=%s\n' "${C1_BUILD_TARGET:-unspecified}"
printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short; printf 'PATCHED_STATUS_END\n'
sha256sum "$A02_ROOT/team/c1_flashkda/challenge_vshard4/apply_vshard4_patch.py" \
  "$A02_ROOT/team/c1_flashkda/challenge_vshard4/vshard4.py" \
  "$A02_ROOT/team/c1_flashkda/challenge_vshard4/run_vshard4_final.py" \
  "$PATCHED_ROOT/flash_kda_C.cpython-312-x86_64-linux-gnu.so" \
  "$PATCHED_ROOT/csrc/flash_kda.cpp" "$PATCHED_ROOT/csrc/fwd.h" \
  "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu" "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard4.cuh"

export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"
RUNNER="$A02_ROOT/team/c1_flashkda/challenge_vshard4/run_vshard4_final.py"
echo "===== SMALL_EXACT_MATRIX ====="
"$PYTHON" "$RUNNER" --T 256 --small-heads 1,2,4 --states all --no-bench \
  --json "$LOG_DIR/c1_vshard4_${LABEL}_small_matrix.json"
for heads in 64 96; do
  echo "===== FULL_CALL_H${heads}_BF16 ====="
  "$PYTHON" "$RUNNER" --T 8192 --H "$heads" --states bf16 --warmup 30 --iters 200 --repeats 5 \
    --json "$LOG_DIR/c1_vshard4_${LABEL}_h${heads}.json"
done
