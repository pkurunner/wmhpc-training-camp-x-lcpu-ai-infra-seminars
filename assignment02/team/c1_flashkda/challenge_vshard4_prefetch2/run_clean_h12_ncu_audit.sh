#!/usr/bin/env bash
# Profile the H=12 one-SO variants without mixing public calls in one process.
# This script never allocates Slurm resources; the coordinator must grant a
# clean B300 GPU and explicitly authorize this expensive NCU capture.
set -Eeuo pipefail

if [[ "${C1_H12_NCU_AUTHORIZED:-0}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing NCU run: set C1_H12_NCU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "refusing NCU run outside a Slurm allocation" >&2
    exit 64
fi

: "${A02_ROOT:?set A02_ROOT to the assignment02 root}"
: "${PATCHED_ROOT:?set PATCHED_ROOT to the fresh one-SO comparison worktree}"
: "${LABEL:?set LABEL to an evidence label}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NCU_BIN="${NCU_BIN:-ncu}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2"
RUNNER="$OWNED/ncu_single_variant.py"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
JOB_TAG="${SLURM_JOB_ID}"
SEED="${C1_H12_NCU_SEED:-20260828}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_h12_ncu_${LABEL}_job${JOB_TAG}.log"
MANIFEST="$RESULTS_DIR/c1_h12_ncu_${LABEL}_job${JOB_TAG}.sha256"
exec > >(tee "$LOG") 2>&1

gpu_query() {
    nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total \
        --format=csv,noheader
}

app_query() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
}

require_b300() {
    local row name capability saw_gpu=0
    while IFS= read -r row; do
        [[ -n "${row//[[:space:]]/}" ]] || continue
        saw_gpu=1
        name="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print $4}' <<<"$row")"
        capability="$(awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5); print $5}' <<<"$row")"
        [[ "$name" == *B300* && "$capability" == "10.3" ]] || {
            echo "expected B300 compute capability 10.3; found $name / $capability" >&2
            return 75
        }
    done < <(gpu_query)
    [[ "$saw_gpu" == 1 ]] || {
        echo "nvidia-smi did not return a visible GPU" >&2
        return 75
    }
}

require_clean() {
    local stage="$1" apps used
    apps="$(app_query)" || { echo "$stage: compute-app query failed" >&2; return 92; }
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" || {
        echo "$stage: memory query failed" >&2; return 92;
    }
    echo "${stage}_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "${stage}_COMPUTE_APPS_END"
    echo "${stage}_MEMORY_USED_MIB=$used"
    [[ -z "${apps//[[:space:]]/}" ]] || return 91
    while IFS= read -r value; do
        value="${value//[[:space:]]/}"
        [[ "$value" =~ ^0$ ]] || return 91
    done <<<"$used"
}

finish() {
    local rc=$? check_rc
    trap - EXIT
    echo "===== POST_AUDIT ====="; date -Is
    gpu_query || { check_rc=92; [[ "$rc" -ne 0 ]] || rc="$check_rc"; }
    require_b300 || { check_rc=75; [[ "$rc" -ne 0 ]] || rc="$check_rc"; }
    require_clean POST || { check_rc=91; [[ "$rc" -ne 0 ]] || rc="$check_rc"; }
    echo "LOG=$LOG"
    echo "MANIFEST=$MANIFEST"
    echo "FINAL_RC=$rc"
    exit "$rc"
}
trap finish EXIT

shopt -s nullglob
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
(( ${#SO_PATHS[@]} == 1 )) || {
    echo "expected exactly one flash_kda_C shared object under $PATCHED_ROOT" >&2
    exit 65
}
SO_PATH="${SO_PATHS[0]}"
SOURCE_FILES=(
    "$OWNED/ncu_single_variant.py"
    "$OWNED/run_clean_h12_ncu_audit.sh"
    "$OWNED/vshard4_prefetch2.py"
    "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/prefetch2.py"
    "$A02_ROOT/team/c1_flashkda/challenge_vshard4/vshard4.py"
    "$PATCHED_ROOT/csrc/flash_kda.cpp"
    "$PATCHED_ROOT/csrc/fwd.h"
    "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu"
    "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard.cuh"
    "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard_p2.cuh"
    "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard4.cuh"
    "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard4_p2.cuh"
    "$SO_PATH"
)
for source_file in "${SOURCE_FILES[@]}"; do
    [[ -f "$source_file" ]] || { echo "missing source identity file: $source_file" >&2; exit 65; }
done
if [[ "$PYTHON_BIN" == */* ]]; then
    [[ -x "$PYTHON_BIN" ]] || { echo "missing executable Python: $PYTHON_BIN" >&2; exit 65; }
else
    PYTHON_BIN="$(command -v "$PYTHON_BIN")" || { echo "missing Python on PATH" >&2; exit 65; }
fi
command -v "$NCU_BIN" >/dev/null || { echo "missing NCU: $NCU_BIN" >&2; exit 65; }
[[ -d "$PATCHED_ROOT" && -f "$RUNNER" ]] || { echo "patched root or NCU runner unavailable" >&2; exit 65; }

echo "===== ENVIRONMENT_GATE ====="
printf 'SLURM_JOB_ID=%s CUDA_VISIBLE_DEVICES=%s\n' "$SLURM_JOB_ID" "${CUDA_VISIBLE_DEVICES:-}"
printf 'PYTHON_BIN=%s NCU_BIN=%s\n' "$PYTHON_BIN" "$(command -v "$NCU_BIN")"
"$NCU_BIN" --version | tail -n 1
echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_b300
require_clean PRE || exit 90
echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'PATCHED_STATUS_BEGIN\n'; git -C "$PATCHED_ROOT" status --short; printf 'PATCHED_STATUS_END\n'
sha256sum "${SOURCE_FILES[@]}" | tee "$MANIFEST"

export CUDA_HOME
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"

profile_basic() {
    local variant="$1" kernel_regex="$2" expected_fragment="$3"
    local report="$RESULTS_DIR/c1_h12_ncu_${LABEL}_${variant}_basic_job${JOB_TAG}.ncu-rep"
    local csv="${report%.ncu-rep}.csv"
    echo "===== NCU_BASIC_${variant} ====="
    echo "K2_FILTER=$kernel_regex"
    require_clean "BEFORE_BASIC_${variant}"
    "$NCU_BIN" --set basic --kernel-name-base function --kernel-name "$kernel_regex" \
        --clock-control none --force-overwrite --export "$report" \
        "$PYTHON_BIN" "$RUNNER" --variant "$variant" --T 8192 --H 12 --D 128 --seed "$SEED"
    [[ -s "$report" ]] || { echo "missing NCU basic report: $report" >&2; return 76; }
    "$NCU_BIN" --import "$report" --csv --page details > "$csv"
    [[ -s "$csv" ]] && grep -q "$expected_fragment" "$csv" || {
        echo "basic report did not contain only requested K2 fragment: $expected_fragment" >&2
        return 76
    }
    sha256sum "$report" "$csv" | tee -a "$MANIFEST"
    require_clean "AFTER_BASIC_${variant}"
}

profile_full_winner() {
    local report="$RESULTS_DIR/c1_h12_ncu_${LABEL}_vshard4_p2_full_job${JOB_TAG}.ncu-rep"
    local csv="${report%.ncu-rep}.csv"
    local kernel_regex='regex:.*_flash_kda_fwd_(prepare|recurrence_vshard4_p2).*'
    echo "===== NCU_FULL_vshard4_p2_K1_K2 ====="
    echo "K1_K2_FILTER=$kernel_regex"
    require_clean BEFORE_FULL_vshard4_p2
    "$NCU_BIN" --set full --kernel-name-base function --kernel-name "$kernel_regex" \
        --clock-control none --import-source yes --source-folders "$PATCHED_ROOT" \
        --force-overwrite --export "$report" \
        "$PYTHON_BIN" "$RUNNER" --variant vshard4_p2 --T 8192 --H 12 --D 128 --seed "$SEED"
    [[ -s "$report" ]] || { echo "missing NCU full report: $report" >&2; return 76; }
    "$NCU_BIN" --import "$report" --csv --page details > "$csv"
    [[ -s "$csv" ]] && grep -q '_flash_kda_fwd_prepare' "$csv" && \
        grep -q '_flash_kda_fwd_recurrence_vshard4_p2' "$csv" || {
        echo "full report did not capture both vshard4_p2 K1 and K2" >&2
        return 76
    }
    sha256sum "$report" "$csv" | tee -a "$MANIFEST"
    require_clean AFTER_FULL_vshard4_p2
}

# Each target process below selects one public wrapper and calls it once.  K2
# regexes are unique to that process's selected path; no cross-variant launch
# appears in an application's CUDA stream.
# The first two patterns explicitly reject the underscore that begins a
# generated alternative symbol, so they cannot also match a later variant if
# the runner is ever extended.
profile_basic baseline 'regex:.*_flash_kda_fwd_recurrence([^_].*|$)' '_flash_kda_fwd_recurrence'
profile_basic vshard2_p2 'regex:.*_flash_kda_fwd_recurrence_vshard_p2.*' '_flash_kda_fwd_recurrence_vshard_p2'
profile_basic vshard4_p1 'regex:.*_flash_kda_fwd_recurrence_vshard4([^_].*|$)' '_flash_kda_fwd_recurrence_vshard4'
profile_basic vshard4_p2 'regex:.*_flash_kda_fwd_recurrence_vshard4_p2.*' '_flash_kda_fwd_recurrence_vshard4_p2'
profile_full_winner

echo "===== NCU_CAPTURE_COMPLETE ====="
