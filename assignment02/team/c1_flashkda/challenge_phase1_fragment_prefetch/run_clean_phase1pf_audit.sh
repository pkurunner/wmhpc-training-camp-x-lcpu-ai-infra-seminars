#!/usr/bin/env bash
# One clean allocation: exact H12 four-contract test plus two 1000-sample repeats.
set -Eeuo pipefail

if [[ "${C1_PHASE1PF_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_PHASE1PF_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT to assignment02 checkout}"
: "${PATCHED_ROOT:?set PATCHED_ROOT to freshly built candidate worktree}"
: "${PYTHON_INCLUDE:?set PYTHON_INCLUDE to selected Python headers}"
: "${PTXAS_JSON:?set PTXAS_JSON to successful candidate ptxas JSON}"
: "${LABEL:?set LABEL to unique allocation label}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_phase1_fragment_prefetch"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
mkdir -p "$RESULTS_DIR"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "must run inside Slurm GPU allocation" >&2; exit 65; }
for required in "$PATCHED_ROOT/flash_kda_C.cpython-"*-linux-gnu.so "$OWNED/run_phase1_fragment_prefetch.py" \
                "$OWNED/analyze_phase1_fragment_prefetch.py" "$PTXAS_JSON" "$PYTHON_INCLUDE/Python.h"; do
    compgen -G "$required" >/dev/null || { echo "missing required path/glob: $required" >&2; exit 66; }
done
LOG="$RESULTS_DIR/c1_phase1pf_${LABEL}_job${SLURM_JOB_ID}.log"
RESULT_JSON="$RESULTS_DIR/c1_phase1pf_${LABEL}_h12_all_contracts_two_repeats.json"
GATE_JSON="$RESULTS_DIR/c1_phase1pf_${LABEL}_one_allocation_gate.json"
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
    [[ -z "$apps" && "$used" =~ ^[[:space:]]*0[[:space:]]*$ ]]
}
finish() {
    local rc=$?; trap - EXIT
    echo "===== POST_AUDIT ====="; date -Is; gpu_query || rc=92; require_clean POST || rc=91
    echo "FINAL_RC=$rc"; exit "$rc"
}
trap finish EXIT

echo "===== ENVIRONMENT_GATE ====="; command -v "$PYTHON_BIN"; command -v nvcc
[[ -f "$CUDA_HOME/include/cuda_runtime.h" && -f "$PYTHON_INCLUDE/Python.h" ]] || exit 89
echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query; require_clean PRE || exit 90
echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'C1_BUILD_TARGET=%s\n' "${C1_BUILD_TARGET:-unspecified}"
printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short; printf 'PATCHED_STATUS_END\n'
sha256sum "$OWNED/apply_phase1_fragment_prefetch_patch.py" "$OWNED/phase1_fragment_prefetch.py" \
  "$OWNED/run_phase1_fragment_prefetch.py" "$OWNED/analyze_phase1_fragment_prefetch.py" \
  "$OWNED/run_clean_phase1pf_audit.sh" "$OWNED/ptxas_audit.py" "$PTXAS_JSON" \
  "$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so "$PATCHED_ROOT/csrc/flash_kda.cpp" \
  "$PATCHED_ROOT/csrc/fwd.h" "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu" \
  "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard4_p2_phase1pf.cuh"

export CUDA_HOME
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"
echo "===== H12_ALL_CONTRACTS_EXACT_TWO_REPEATS ====="
"$PYTHON_BIN" "$OWNED/run_phase1_fragment_prefetch.py" --T 8192 --H 12 \
  --contracts none,bf16_both,fp32_both,fp32_final_only --warmup 30 --samples 1000 --repeats 2 --json "$RESULT_JSON"
require_clean AFTER_H12 || exit 93
echo "===== ONE_ALLOCATION_PRE_REGISTERED_GATE ====="
"$PYTHON_BIN" "$OWNED/analyze_phase1_fragment_prefetch.py" --ptxas "$PTXAS_JSON" --result "$RESULT_JSON" --json "$GATE_JSON" --require-eligible
echo "ONE_ALLOCATION_ONLY=true"
echo "INDEPENDENT_CONFIRMATION_REQUIRED_BEFORE_ANY_PRODUCTION_DISCUSSION=true"
