#!/usr/bin/env bash
# One authorized B300 H96 confirmation for the frozen P2S3 candidate.
set -Eeuo pipefail

if [[ "${C1_PREFETCH2_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_PREFETCH2_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift
A02_ROOT="${1:?usage: ... A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
PATCHED_ROOT="${2:?usage: ... A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
REFERENCE_ROOT="${3:?usage: ... A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
LABEL="${4:?usage: ... A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
PYTHON="${PYTHON:-python}"
PYTHON_BIN_DIR="$(dirname "$PYTHON")"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PYTHON_DEV_INCLUDE="${C1_PYTHON_DEV_INCLUDE:-/home/lcpu/85117379/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/include/python3.12}"
export CUDA_HOME
export PATH="$PYTHON_BIN_DIR:$CUDA_HOME/bin:$PATH"
export CPATH="$PYTHON_DEV_INCLUDE${CPATH:+:$CPATH}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_prefetch2"
LOG_DIR="$OWNED/results"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/c1_prefetch2_${LABEL}_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
app_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_used_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
require_clean() {
    local stage="$1" apps used
    if ! apps="$(app_query)"; then
        echo "${stage}: compute-app query failed" >&2
        return 92
    fi
    if ! used="$(memory_used_query)"; then
        echo "${stage}: memory-used query failed" >&2
        return 92
    fi
    echo "${stage}_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "${stage}_COMPUTE_APPS_END"
    printf '%s_MEMORY_USED_MIB=%s\n' "$stage" "$used"
    [[ -z "$apps" && "$used" =~ ^[[:space:]]*0[[:space:]]*$ ]]
}
finish() {
    rc=$?; trap - EXIT
    echo "===== POST_AUDIT ====="; date -Is; gpu_query || rc=92
    require_clean POST || rc=91
    echo "FINAL_RC=$rc"; exit "$rc"
}
trap finish EXIT

echo "===== ENVIRONMENT_GATE ====="
command -v ninja
command -v nvcc
[[ -x "$PYTHON" && -x "$PYTHON_BIN_DIR/ninja" && -f "$CUDA_HOME/include/cuda_runtime.h" && -f "$PYTHON_DEV_INCLUDE/Python.h" ]] || exit 89
echo "===== PRE_AUDIT ====="; date -Is; hostname; printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"; gpu_query
require_clean PRE || exit 90

echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'C1_BUILD_TARGET=%s\n' "${C1_BUILD_TARGET:-unspecified}"
printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short; printf 'PATCHED_STATUS_END\n'
sha256sum \
  "$OWNED/apply_prefetch2_patch.py" "$OWNED/prefetch2.py" \
  "$OWNED/run_prefetch2_final.py" "$OWNED/run_clean_prefetch2_h96_audit.sh" \
  "$A02_ROOT/team/c1_flashkda/challenge_vshard/apply_vshard_patch.py" \
  "$A02_ROOT/team/c1_flashkda/challenge_vshard/vshard.py" \
  "$PATCHED_ROOT/flash_kda_C.cpython-"*-linux-gnu.so \
  "$PATCHED_ROOT/csrc/flash_kda.cpp" "$PATCHED_ROOT/csrc/fwd.h" \
  "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu" \
  "$PATCHED_ROOT/csrc/smxx/fwd_kernel2.cuh" \
  "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard.cuh" \
  "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard_p2.cuh"

export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"
RUNNER="$OWNED/run_prefetch2_final.py"
echo "===== H96_T8192_ALLSTATE_EXACT ====="
"$PYTHON" "$RUNNER" --reference-root "$REFERENCE_ROOT" --T 8192 --H 96 \
  --states all --no-bench --json "$LOG_DIR/c1_prefetch2_${LABEL}_h96_allstate_exact.json"
require_clean BETWEEN_EXACT_AND_BENCH || exit 93

echo "===== H96_T8192_BF16_P1_P2_ABBA ====="
"$PYTHON" "$RUNNER" --reference-root "$REFERENCE_ROOT" --T 8192 --H 96 --states bf16 \
  --warmup 30 --iters 200 --repeats 5 \
  --json "$LOG_DIR/c1_prefetch2_${LABEL}_h96_bf16.json"
require_clean AFTER_BENCH || exit 94
