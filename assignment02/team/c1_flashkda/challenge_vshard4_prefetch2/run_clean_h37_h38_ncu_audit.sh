#!/usr/bin/env bash
# K2-only NCU Basic capture at the observed H=37/H=38 dispatch boundary.
set -Eeuo pipefail

if [[ "${C1_BOUNDARY_NCU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing NCU run: set C1_BOUNDARY_NCU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift
[[ $# -eq 0 && -n "${SLURM_JOB_ID:-}" ]] || {
    echo "this audit takes no extra arguments and must run inside Slurm" >&2
    exit 64
}

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT}"
: "${LABEL:?set LABEL}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NCU_BIN="${NCU_BIN:-ncu}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2"
RUNNER="$OWNED/ncu_single_variant.py"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
JOB_TAG="$SLURM_JOB_ID"
LOG="$RESULTS_DIR/c1_h37_h38_ncu_${LABEL}_job${JOB_TAG}.log"
MANIFEST="$RESULTS_DIR/c1_h37_h38_ncu_${LABEL}_job${JOB_TAG}.sha256"
mkdir -p "$RESULTS_DIR"
exec > >(tee "$LOG") 2>&1

gpu_query() {
    nvidia-smi --query-gpu=index,uuid,name,compute_cap,memory.used,memory.total --format=csv,noheader
}
app_query() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
}
memory_query() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
}
require_clean() {
    local stage="$1" apps used
    apps="$(app_query)" || return 92
    used="$(memory_query)" || return 92
    echo "${stage}_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "${stage}_COMPUTE_APPS_END"
    echo "${stage}_MEMORY_USED_MIB=$used"
    [[ -z "${apps//[[:space:]]/}" && "$used" =~ ^[[:space:]]*0[[:space:]]*$ ]]
}
require_b300() {
    local row
    row="$(gpu_query)" || return 92
    [[ "$row" == *B300* && "$row" == *", 10.3,"* ]]
}
finish() {
    local rc=$?
    trap - EXIT
    echo "===== POST_AUDIT ====="; date -Is
    gpu_query || rc=92
    require_b300 || rc=75
    require_clean POST || rc=91
    echo "LOG=$LOG"
    echo "MANIFEST=$MANIFEST"
    echo "FINAL_RC=$rc"
    exit "$rc"
}
trap finish EXIT

if [[ "$PYTHON_BIN" == */* ]]; then
    [[ -x "$PYTHON_BIN" ]] || exit 65
else
    PYTHON_BIN="$(command -v "$PYTHON_BIN")" || exit 65
fi
command -v "$NCU_BIN" >/dev/null || exit 65
shopt -s nullglob
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
(( ${#SO_PATHS[@]} == 1 )) || exit 65
SO_PATH="${SO_PATHS[0]}"
[[ -f "$RUNNER" ]] || exit 65

echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_b300 || exit 75
require_clean PRE || exit 90
sha256sum "$RUNNER" "$OWNED/run_clean_h37_h38_ncu_audit.sh" "$SO_PATH" | tee "$MANIFEST"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"

for heads in 37 38; do
    report="$RESULTS_DIR/c1_h37_h38_ncu_${LABEL}_h${heads}_vshard4_p2_basic_job${JOB_TAG}.ncu-rep"
    csv="${report%.ncu-rep}.csv"
    echo "===== H${heads}_VSHARD4_P2_NCU_BASIC ====="
    require_clean "BEFORE_H${heads}" || exit 93
    "$NCU_BIN" --set basic --kernel-name-base function \
      --kernel-name 'regex:.*_flash_kda_fwd_recurrence_vshard4_p2.*' \
      --clock-control none --force-overwrite --export "$report" \
      "$PYTHON_BIN" "$RUNNER" --variant vshard4_p2 --T 8192 --H "$heads" --D 128 --seed 20260828
    "$NCU_BIN" --import "$report" --csv --page details > "$csv"
    [[ -s "$report" && -s "$csv" ]] || exit 76
    grep -q '_flash_kda_fwd_recurrence_vshard4_p2' "$csv" || exit 76
    sha256sum "$report" "$csv" | tee -a "$MANIFEST"
    require_clean "AFTER_H${heads}" || exit 94
done
