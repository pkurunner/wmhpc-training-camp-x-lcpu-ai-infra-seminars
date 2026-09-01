#!/usr/bin/env bash
# Run the upstream FLA chunk-vs-gold tests in an already allocated one-B300 job.
# This script does not submit Slurm work, rebuild FlashKDA, or modify either
# source checkout.
set -Eeuo pipefail

if [[ "${C1_FLA_CHUNK_VALIDATION_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_FLA_CHUNK_VALIDATION_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

# A job id is required so this entry point cannot accidentally be used from a
# login node.  It intentionally never invokes sbatch, salloc, or srun.
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "refusing GPU run outside an existing Slurm allocation" >&2; exit 65; }

: "${A02_ROOT:?set A02_ROOT to the assignment02 checkout}"
: "${PATCHED_ROOT:?set PATCHED_ROOT to the already-built candidate FlashKDA checkout}"
: "${FLA_ROOT:?set FLA_ROOT to the pinned FLA checkout}"
: "${LABEL:?set LABEL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
: "${PYTHON_INCLUDE:?set PYTHON_INCLUDE to the selected Python include directory}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_fla_chunk_validation"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
EXPECTED_PATCHED_COMMIT="${EXPECTED_PATCHED_COMMIT:-1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b}"
EXPECTED_FLA_COMMIT="${EXPECTED_FLA_COMMIT:-a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d}"
TEST_FILE="$PATCHED_ROOT/tests/test_fwd.py"
RUNNER="$OWNED/run_fla_chunk_validation.py"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_fla_chunk_validation_${LABEL}_job${SLURM_JOB_ID}.log"
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
    local rc=$?
    trap - EXIT
    echo "===== POST_AUDIT ====="; date -Is; gpu_query || rc=92
    require_clean POST || rc=91
    echo "FINAL_RC=$rc"
    exit "$rc"
}
trap finish EXIT

require_commit() {
    local root="$1" expected="$2" label="$3"
    [[ "$(git -C "$root" rev-parse HEAD)" == "$expected" ]] || {
        echo "$label commit gate failed: expected $expected" >&2
        return 85
    }
}

echo "===== ENVIRONMENT_GATE ====="
command -v "$PYTHON_BIN"
[[ -d "$A02_ROOT" && -d "$PATCHED_ROOT" && -d "$FLA_ROOT" && -f "$RUNNER" && -f "$TEST_FILE" ]] || exit 89
[[ -x "$(dirname "$PYTHON_BIN")/ninja" && -f "$CUDA_HOME/include/cuda_runtime.h" && -f "$PYTHON_INCLUDE/Python.h" ]] || {
    echo "official torch_ref sigmoid_ext JIT prerequisites are incomplete" >&2
    exit 89
}
require_commit "$PATCHED_ROOT" "$EXPECTED_PATCHED_COMMIT" "PATCHED_ROOT"
require_commit "$FLA_ROOT" "$EXPECTED_FLA_COMMIT" "FLA_ROOT"
[[ -z "$(git -C "$FLA_ROOT" status --short --untracked-files=no)" ]] || { echo "FLA_ROOT has tracked/staged changes" >&2; exit 84; }
git -C "$PATCHED_ROOT" diff --quiet -- tests/test_fwd.py || { echo "official test_fwd.py is locally modified" >&2; exit 84; }
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
[[ "${#SO_PATHS[@]}" -eq 1 && -f "${SO_PATHS[0]}" \
    && -f "$PATCHED_ROOT/flash_kda/__init__.py" \
    && -f "$PATCHED_ROOT/tests/torch_ref.py" \
    && -f "$FLA_ROOT/fla/ops/backends/__init__.py" \
    && -f "$FLA_ROOT/fla/ops/kda/__init__.py" \
    && -f "$FLA_ROOT/fla/ops/kda/chunk.py" \
    && -f "$FLA_ROOT/fla/ops/kda/fused_recurrent.py" ]] || exit 89
GPU_ROWS="$(gpu_query)"
[[ "$(printf '%s\n' "$GPU_ROWS" | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || { echo "expected exactly one visible GPU" >&2; exit 88; }
[[ "$GPU_ROWS" == *"B300"* && "$GPU_ROWS" == *"10.3"* ]] || { echo "expected B300 SM10.3" >&2; exit 88; }

echo "===== PRE_AUDIT ====="; date -Is; hostname; printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"; printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"; gpu_query
require_clean PRE || exit 90

echo "===== SOURCE_IDENTITY ====="
printf 'A02_ROOT=%s\nPATCHED_ROOT=%s\nFLA_ROOT=%s\n' "$A02_ROOT" "$PATCHED_ROOT" "$FLA_ROOT"
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'FLA_COMMIT=%s\n' "$(git -C "$FLA_ROOT" rev-parse HEAD)"
printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short --untracked-files=no; printf 'PATCHED_STATUS_END\n'
printf 'FLA_STATUS_BEGIN\n'; git -C "$FLA_ROOT" status --short --untracked-files=no; printf 'FLA_STATUS_END\n'
sha256sum \
    "$RUNNER" \
    "$OWNED/run_clean_fla_chunk_validation.sh" \
    "$TEST_FILE" \
    "$PATCHED_ROOT/tests/torch_ref.py" \
    "$PATCHED_ROOT/flash_kda/__init__.py" \
    "${SO_PATHS[0]}" \
    "$FLA_ROOT/fla/ops/backends/__init__.py" \
    "$FLA_ROOT/fla/ops/kda/__init__.py" \
    "$FLA_ROOT/fla/ops/kda/chunk.py" \
    "$FLA_ROOT/fla/ops/kda/chunk_fwd.py" \
    "$FLA_ROOT/fla/ops/kda/fused_recurrent.py"

echo "===== PYTHON_COMPILE_GATE ====="
"$PYTHON_BIN" -m py_compile "$RUNNER"

# These values must be present before the runner imports FLA.  Unsetting both
# opt-in knobs documents that this run has no C1/pinned FLA backend intent;
# dispatch bypass, not a verifier fallback, is the isolation mechanism.
export CUDA_HOME
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
export PYTHONPATH="$PATCHED_ROOT:$FLA_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export FLA_DISABLE_BACKEND_DISPATCH=1
unset C1_B300_FLASH_KDA
unset FLA_FLASH_KDA
cd "$RESULTS_DIR"
ARTIFACT="$RESULTS_DIR/c1_fla_chunk_validation_${LABEL}.json"
echo "===== OFFICIAL_FLA_CHUNK_TESTS ====="
"$PYTHON_BIN" "$RUNNER" --test-file "$TEST_FILE" --patched-root "$PATCHED_ROOT" --fla-root "$FLA_ROOT" --json "$ARTIFACT"
require_clean AFTER_OFFICIAL_TESTS || exit 93
