#!/usr/bin/env bash
# Clean-allocation head-count sweep for the one-SO vshard4+P2 comparison.
#
# This is intentionally a shell-level process harness: every head count runs
# in a fresh Python process, so the audit can prove that CUDA allocations from
# one shape are gone before the next shape starts.
set -Eeuo pipefail

if [[ "${C1_VSHARD4_P2_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_VSHARD4_P2_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift
[[ $# -eq 0 ]] || { echo "unexpected arguments; configure the sweep through environment variables" >&2; exit 64; }

: "${A02_ROOT:?usage: A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
: "${PATCHED_ROOT:?usage: A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
: "${REFERENCE_ROOT:?usage: A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
: "${LABEL:?usage: A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
: "${PYTHON_INCLUDE:?set PYTHON_INCLUDE to the selected Python include directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
HEADS="${HEADS:-1,2,4,8,12,16,24,32,48,64,96}"
T="${T:-8192}"
STATES="${STATES:-bf16}"
WARMUP="${WARMUP:-30}"
SAMPLES="${SAMPLES:-500}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_vshard4_p2_${LABEL}_head_sweep_job${SLURM_JOB_ID:-none}.log"
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

[[ "$T" == 8192 ]] || { echo "this audit is fixed to T=8192; got T=$T" >&2; exit 65; }
[[ "$STATES" == bf16 ]] || { echo "this audit is fixed to BF16 state; got STATES=$STATES" >&2; exit 65; }
[[ "$WARMUP" =~ ^[0-9]+$ && "$SAMPLES" =~ ^[1-9][0-9]*$ ]] || {
    echo "WARMUP must be nonnegative and SAMPLES positive integers" >&2; exit 65;
}
IFS=',' read -r -a HEAD_ARRAY <<< "$HEADS"
[[ ${#HEAD_ARRAY[@]} -gt 0 ]] || { echo "HEADS must contain at least one positive head count" >&2; exit 65; }
for h in "${HEAD_ARRAY[@]}"; do
    [[ "$h" =~ ^[1-9][0-9]*$ ]] || { echo "invalid head count in HEADS: $h" >&2; exit 65; }
done

echo "===== ENVIRONMENT_GATE ====="
command -v "$PYTHON_BIN"
command -v nvcc
[[ -f "$CUDA_HOME/include/cuda_runtime.h" && -f "$PYTHON_INCLUDE/Python.h" ]] || exit 89
echo "T=$T STATES=$STATES WARMUP=$WARMUP SAMPLES=$SAMPLES HEADS=$HEADS"
echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_clean PRE || exit 90
echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'C1_BUILD_TARGET=%s\n' "${C1_BUILD_TARGET:-unspecified}"
printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short; printf 'PATCHED_STATUS_END\n'
sha256sum "$OWNED/apply_vshard4_prefetch2_patch.py" "$OWNED/vshard4_prefetch2.py" \
  "$OWNED/run_vshard4_prefetch2_final.py" "$OWNED/run_clean_head_sweep_audit.sh" \
  "$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so \
  "$PATCHED_ROOT/csrc/flash_kda.cpp" "$PATCHED_ROOT/csrc/fwd.h" "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu" \
  "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard.cuh" "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard_p2.cuh" \
  "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard4.cuh" "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard4_p2.cuh"

export CUDA_HOME
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"
RUNNER="$OWNED/run_vshard4_prefetch2_final.py"

for h in "${HEAD_ARRAY[@]}"; do
    echo "===== H${h}_BF16_EXACT_AND_FOUR_PATH_CYCLIC ====="
    require_clean "PRE_H${h}" || exit 93
    # The runner is a child process for exactly one shape.  It must exit before
    # this post-shape allocation gate can allow the next head count to begin.
    "$PYTHON_BIN" "$RUNNER" --reference-root "$REFERENCE_ROOT" --T "$T" --H "$h" --states "$STATES" \
      --warmup "$WARMUP" --samples "$SAMPLES" \
      --json "$RESULTS_DIR/c1_vshard4_p2_${LABEL}_h${h}_bf16_cyclic.json"
    require_clean "POST_H${h}" || exit 94
done
