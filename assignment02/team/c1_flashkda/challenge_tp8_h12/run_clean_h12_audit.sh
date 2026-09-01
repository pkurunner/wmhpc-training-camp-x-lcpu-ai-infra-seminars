#!/usr/bin/env bash
# Clean-allocation audit wrapper for the C1 TP8 / H=12 fixed-length experiment.
#
# All remote locations are explicit environment variables so the same script can
# run from a staged workspace without editing tracked paths.
set -Eeuo pipefail

if [[ "${C1_H12_AUTHORIZED:-0}" != "1" || "${1:-}" != "--authorized-by-user" ]]; then
    echo "refusing GPU run: set C1_H12_AUTHORIZED=1 and pass --authorized-by-user" >&2
    exit 64
fi
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${C1_H12_WORKSPACE_ROOT:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd)}"
PATCHED_ROOT="${C1_H12_PATCHED_ROOT:?set C1_H12_PATCHED_ROOT to the rebuilt FlashKDA worktree}"
REFERENCE_ROOT="${C1_H12_REFERENCE_ROOT:?set C1_H12_REFERENCE_ROOT to pinned FlashKDA source with tests/torch_ref.py}"
VARIANT="${C1_H12_VARIANT:-p2}"
LABEL="${C1_H12_LABEL:-tp8_h12}"
OUTPUT_DIR="${C1_H12_OUTPUT_DIR:-$SCRIPT_DIR/results}"
PYTHON="${C1_H12_PYTHON:-python}"
RUNNER="${C1_H12_RUNNER:-$SCRIPT_DIR/run_h12.py}"
WARMUP="${C1_H12_WARMUP:-30}"
ITERS="${C1_H12_ITERS:-200}"
REPEATS="${C1_H12_REPEATS:-5}"
SMALL_HEADS="${C1_H12_SMALL_HEADS:-1,2,4,12}"
SMALL_T="${C1_H12_SMALL_T:-256}"
OFFICIAL_T="${C1_H12_OFFICIAL_T:-8192}"
OFFICIAL_H="${C1_H12_OFFICIAL_H:-12}"

case "$VARIANT" in
    p2|vshard4) ;;
    *) echo "C1_H12_VARIANT must be p2 or vshard4, got $VARIANT" >&2; exit 65 ;;
esac
[[ -f "$RUNNER" && -d "$PATCHED_ROOT" && -d "$REFERENCE_ROOT" ]] || {
    echo "runner, patched root, or reference root is unavailable" >&2
    exit 66
}
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/c1_${LABEL}_${VARIANT}_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
app_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_used_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
require_clean() {
    local stage="$1" apps used
    apps="$(app_query)" || { echo "$stage: compute-app query failed" >&2; return 92; }
    used="$(memory_used_query)" || { echo "$stage: memory query failed" >&2; return 92; }
    echo "${stage}_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "${stage}_COMPUTE_APPS_END"
    printf '%s_MEMORY_USED_MIB=%s\n' "$stage" "$used"
    [[ -z "$apps" && "$used" =~ ^[[:space:]]*0[[:space:]]*$ ]]
}
finish() {
    local rc=$?
    trap - EXIT
    echo "===== POST_AUDIT ====="; date -Is
    gpu_query || rc=92
    require_clean POST || rc=91
    echo "FINAL_RC=$rc"
    exit "$rc"
}
trap finish EXIT

echo "===== PRE_AUDIT ====="; date -Is; hostname
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
gpu_query
require_clean PRE || exit 90

declare -a SOURCE_FILES SOURCE_ARGS
SOURCE_FILES=(
    "$PATCHED_ROOT/csrc/flash_kda.cpp"
    "$PATCHED_ROOT/csrc/fwd.h"
    "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu"
)
if [[ "$VARIANT" == "p2" ]]; then
    SOURCE_FILES+=(
        "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard.cuh"
        "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard_p2.cuh"
        "$WORKSPACE_ROOT/assignment02/team/c1_flashkda/challenge_vshard/apply_vshard_patch.py"
        "$WORKSPACE_ROOT/assignment02/team/c1_flashkda/challenge_vshard/vshard.py"
        "$WORKSPACE_ROOT/assignment02/team/c1_flashkda/challenge_prefetch2/apply_prefetch2_patch.py"
        "$WORKSPACE_ROOT/assignment02/team/c1_flashkda/challenge_prefetch2/prefetch2.py"
    )
else
    SOURCE_FILES+=(
        "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard4.cuh"
        "$WORKSPACE_ROOT/assignment02/team/c1_flashkda/challenge_vshard4/apply_vshard4_patch.py"
        "$WORKSPACE_ROOT/assignment02/team/c1_flashkda/challenge_vshard4/vshard4.py"
    )
fi
for source_file in "${SOURCE_FILES[@]}"; do
    [[ -f "$source_file" ]] || { echo "missing audit source: $source_file" >&2; exit 67; }
    SOURCE_ARGS+=(--source "$source_file")
done

echo "===== IDENTITY ====="
printf 'C1_BUILD_TARGET=%s\n' "${C1_BUILD_TARGET:-unspecified}"
printf 'VARIANT=%s\n' "$VARIANT"
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short; printf 'PATCHED_STATUS_END\n'
sha256sum "$RUNNER" "$SCRIPT_DIR/run_clean_h12_audit.sh" "${SOURCE_FILES[@]}"

export PYTHONPATH="$PATCHED_ROOT:$WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"
SMALL_JSON="$OUTPUT_DIR/c1_${LABEL}_${VARIANT}_small_matrix.json"
OFFICIAL_JSON="$OUTPUT_DIR/c1_${LABEL}_${VARIANT}_h${OFFICIAL_H}_bf16.json"

echo "===== SMALL_EXACT_MATRIX ====="
"$PYTHON" "$RUNNER" --variant "$VARIANT" --reference-root "$REFERENCE_ROOT" "${SOURCE_ARGS[@]}" \
    --small-t "$SMALL_T" --small-heads "$SMALL_HEADS" --small-only --json "$SMALL_JSON"
require_clean BETWEEN_SMALL_AND_OFFICIAL || exit 93

echo "===== H${OFFICIAL_H}_BF16_FORMAL_GATE_AND_BENCH ====="
"$PYTHON" "$RUNNER" --variant "$VARIANT" --reference-root "$REFERENCE_ROOT" "${SOURCE_ARGS[@]}" \
    --official-t "$OFFICIAL_T" --official-h "$OFFICIAL_H" --official-only \
    --warmup "$WARMUP" --iters "$ITERS" --repeats "$REPEATS" --json "$OFFICIAL_JSON"
require_clean AFTER_OFFICIAL || exit 94
