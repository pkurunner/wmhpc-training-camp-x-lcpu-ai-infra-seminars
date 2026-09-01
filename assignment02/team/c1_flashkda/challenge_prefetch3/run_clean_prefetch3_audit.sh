#!/usr/bin/env bash
# Authorized clean B300 H64 exact + P2S3/P3S3 ABBA audit.
set -Eeuo pipefail

if [[ "${C1_PREFETCH3_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run" >&2; exit 64
fi
shift
A02_ROOT="${1:?usage: A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
PATCHED_ROOT="${2:?usage: A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
REFERENCE_ROOT="${3:?usage: A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
LABEL="${4:?usage: A02_ROOT PATCHED_ROOT REFERENCE_ROOT LABEL}"
PYTHON="${PYTHON:-python}"
PYTHON_BIN_DIR="$(dirname "$PYTHON")"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PYTHON_DEV_INCLUDE="${C1_PYTHON_DEV_INCLUDE:-/home/lcpu/85117379/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/include/python3.12}"
export CUDA_HOME PATH="$PYTHON_BIN_DIR:$CUDA_HOME/bin:$PATH" CPATH="$PYTHON_DEV_INCLUDE${CPATH:+:$CPATH}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_prefetch3"
LOG_DIR="$OWNED/results"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/c1_prefetch3_${LABEL}_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
app_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
require_clean() {
    local stage="$1" apps used
    if ! apps="$(app_query)"; then echo "$stage compute-app query failed" >&2; return 92; fi
    if ! used="$(memory_query)"; then echo "$stage memory query failed" >&2; return 92; fi
    echo "${stage}_APPS_BEGIN"; printf '%s\n' "$apps"; echo "${stage}_APPS_END"
    echo "${stage}_MEMORY_USED_MIB=$used"
    [[ -z "$apps" && "$used" =~ ^[[:space:]]*0[[:space:]]*$ ]]
}
finish() {
    rc=$?; trap - EXIT
    echo "===== POST ====="; date -Is; gpu_query || rc=92
    require_clean POST || rc=91
    echo "FINAL_RC=$rc"; exit "$rc"
}
trap finish EXIT

echo "===== ENV_GATE ====="
command -v ninja; command -v nvcc
[[ -x "$PYTHON" && -x "$PYTHON_BIN_DIR/ninja" && -f "$CUDA_HOME/include/cuda_runtime.h" && -f "$PYTHON_DEV_INCLUDE/Python.h" ]] || exit 89
echo "===== PRE ====="; date -Is; hostname; gpu_query; require_clean PRE || exit 90
echo "===== SOURCE ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'C1_BUILD_TARGET=%s\n' "${C1_BUILD_TARGET:-unspecified}"
git -C "$PATCHED_ROOT" status --short
sha256sum "$OWNED/apply_prefetch3_patch.py" "$OWNED/pinned_prefetch2_generator.py" \
  "$OWNED/prefetch3.py" "$OWNED/run_prefetch3_final.py" "$OWNED/run_clean_prefetch3_audit.sh" \
  "$PATCHED_ROOT/flash_kda_C.cpython-"*-linux-gnu.so "$PATCHED_ROOT/csrc/flash_kda.cpp" \
  "$PATCHED_ROOT/csrc/fwd.h" "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu" \
  "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard.cuh" \
  "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard_p2.cuh" \
  "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard_p3.cuh"

export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"
RUNNER="$OWNED/run_prefetch3_final.py"
echo "===== H64_ALLSTATE_EXACT ====="
"$PYTHON" "$RUNNER" --reference-root "$REFERENCE_ROOT" --phase exact --T 8192 --H 64 \
  --json "$LOG_DIR/c1_prefetch3_${LABEL}_h64_allstate_exact.json"
require_clean BETWEEN || exit 93
echo "===== H64_BF16_P2_P3_ABBA ====="
"$PYTHON" "$RUNNER" --reference-root "$REFERENCE_ROOT" --phase bench --T 8192 --H 64 \
  --warmup 30 --samples 1000 --json "$LOG_DIR/c1_prefetch3_${LABEL}_h64_bf16_abba.json"
require_clean AFTER || exit 94
