#!/usr/bin/env bash
# One-allocation, discovery-only B=7 raw-wrapper experiment on a clean B300.
# This script intentionally never submits a second allocation and never builds
# or modifies a dispatcher, map, FLA implementation, or extension source.
set -Eeuo pipefail

if [[ "${C1_FIXED_BATCH_B7_DISCOVERY_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_FIXED_BATCH_B7_DISCOVERY_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT to the prebuilt audited worktree}"
: "${REFERENCE_ROOT:?set REFERENCE_ROOT to the pinned upstream reference worktree}"
: "${FLA_ROOT:?set FLA_ROOT to the pinned FLA worktree}"
: "${C1_PINNED_REFERENCE_HELPER_PATH:?set C1_PINNED_REFERENCE_HELPER_PATH}"
: "${LABEL:?set LABEL}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_fixed_batch_b7"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
EXPECTED_PATCHED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_REFERENCE_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT="a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
EXPECTED_SO_SHA256="8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_HELPER_SHA256="8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
EXPECTED_RUNNER_SHA256="d36c22917eeecfa8ec23a9abda8d42fd0b87587e07852653343127e302609981"
EXPECTED_ANALYZER_SHA256="a96e4cb9ba1954f59854512bb8808691d6ca5cb3b3e164445963daa3a8f32430"
EXPECTED_SHARED_SHA256="4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f"
SHARED_HELPER="$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py"
EXPECTED_VSHARD2_SHA256="752126488487ac317a7ee167b660b0895562e4877aedbc4ce2a599c1f59a10d0"
EXPECTED_VSHARD4_SHA256="445c90815919fb8c1db3bc79289d10029ea77b1d9accb393c326c120ba1f8385"
EXPECTED_HARNESS_SHA256="5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"
EXPECTED_FLASH_KDA_PYTHON_SHA256="9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
EXPECTED_PINNED_LOADER_SHA256="9445d94a38cb8acc0ab6b15352df01a8d9c93a8b2b6daf0d9adb39a7315c740b"
GPU_TIMING_FIELDS="index,uuid,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu"

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_fixed_batch_b7_discovery_${LABEL}_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
gpu_timing_query() { nvidia-smi --query-gpu="$GPU_TIMING_FIELDS" --format=csv,noheader; }
print_timing_state() {
    local stage="$1"
    echo "${stage}_GPU_TIMING_BEGIN"
    gpu_timing_query
    echo "${stage}_GPU_TIMING_END"
}
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
require_sha() {
    local path="$1" expected="$2" label="$3"
    [[ -f "$path" ]] || { echo "missing $label: $path" >&2; return 86; }
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || {
        echo "$label SHA256 gate failed" >&2; return 87;
    }
}
require_commit() {
    local path="$1" expected="$2" label="$3"
    [[ "$(git -C "$path" rev-parse HEAD)" == "$expected" ]] || {
        echo "$label commit gate failed" >&2; return 85;
    }
}
finish() {
    local rc=$?
    trap - EXIT
    echo "===== POST_AUDIT ====="; date -Is
    gpu_query || rc=92
    print_timing_state POST || rc=92
    require_clean POST || rc=91
    echo "FINAL_RC=$rc"
    exit "$rc"
}
trap finish EXIT

echo "===== ENVIRONMENT_GATE ====="
command -v "$PYTHON_BIN"
require_commit "$PATCHED_ROOT" "$EXPECTED_PATCHED_COMMIT" "patched"
require_commit "$REFERENCE_ROOT" "$EXPECTED_REFERENCE_COMMIT" "reference"
require_commit "$FLA_ROOT" "$EXPECTED_FLA_COMMIT" "FLA"
[[ -z "$(git -C "$REFERENCE_ROOT" status --short --untracked-files=no)" ]] || {
    echo "reference tracked/staged worktree is not clean" >&2; exit 84;
}
[[ -z "$(git -C "$FLA_ROOT" status --short --untracked-files=no)" ]] || {
    echo "FLA tracked/staged worktree is not clean" >&2; exit 84;
}
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
[[ "${#SO_PATHS[@]}" -eq 1 && -f "${SO_PATHS[0]}" ]] || {
    echo "expected exactly one prebuilt flash_kda_C.cpython-*-linux-gnu.so" >&2; exit 89;
}
require_sha "${SO_PATHS[0]}" "$EXPECTED_SO_SHA256" "extension"
require_sha "$C1_PINNED_REFERENCE_HELPER_PATH" "$EXPECTED_HELPER_SHA256" "pinned reference helper"
require_sha "$OWNED/run_fixed_batch_b7_discovery.py" "$EXPECTED_RUNNER_SHA256" "B=7 runner"
require_sha "$OWNED/analyze_fixed_batch_b7_discovery.py" "$EXPECTED_ANALYZER_SHA256" "B=7 analyzer"
require_sha "$SHARED_HELPER" "$EXPECTED_SHARED_SHA256" "shared input/state helper"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/prefetch2.py" "$EXPECTED_VSHARD2_SHA256" "vshard2 wrapper"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2/vshard4_prefetch2.py" "$EXPECTED_VSHARD4_SHA256" "vshard4 wrapper"
require_sha "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" "$EXPECTED_HARNESS_SHA256" "validation harness"
require_sha "$PATCHED_ROOT/flash_kda/__init__.py" "$EXPECTED_FLASH_KDA_PYTHON_SHA256" "loaded flash_kda Python wrapper"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_varlen_dispatch/run_varlen_dispatch_confirmation.py" "$EXPECTED_PINNED_LOADER_SHA256" "pinned-reference loader"
[[ "$(gpu_query | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || {
    echo "expected exactly one visible B300 GPU" >&2; exit 88;
}

echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
print_timing_state PRE
require_clean PRE || exit 90

echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'REFERENCE_COMMIT=%s\n' "$(git -C "$REFERENCE_ROOT" rev-parse HEAD)"
printf 'FLA_COMMIT=%s\n' "$(git -C "$FLA_ROOT" rev-parse HEAD)"
sha256sum \
    "$OWNED/run_fixed_batch_b7_discovery.py" \
    "$OWNED/analyze_fixed_batch_b7_discovery.py" \
    "$SHARED_HELPER" \
    "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/prefetch2.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2/vshard4_prefetch2.py" \
    "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" \
    "$PATCHED_ROOT/flash_kda/__init__.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_varlen_dispatch/run_varlen_dispatch_confirmation.py" \
    "$C1_PINNED_REFERENCE_HELPER_PATH" \
    "${SO_PATHS[0]}"

echo "===== DESCRIBE_AND_PYTHON_COMPILE_GATE ====="
"$PYTHON_BIN" "$OWNED/run_fixed_batch_b7_discovery.py" \
    --describe --process-index 0 --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" \
    --json "$RESULTS_DIR/c1_fixed_batch_b7_discovery_${LABEL}.plan.json"
"$PYTHON_BIN" -m py_compile \
    "$OWNED/run_fixed_batch_b7_discovery.py" \
    "$OWNED/analyze_fixed_batch_b7_discovery.py"

# No setup.py, NVCC, build, patch generator, git mutation, or source mutation
# is present below.  Both runner invocations are fresh Python processes.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$FLA_ROOT:$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_FIXED_BATCH_B7_DISCOVERY_CLEAN_GPU=1
export C1_PINNED_REFERENCE_HELPER_SHA256="$EXPECTED_HELPER_SHA256"
cd "$PATCHED_ROOT"

run_main() {
    local index="$1" artifact="$RESULTS_DIR/c1_fixed_batch_b7_discovery_${LABEL}_main${1}.json"
    echo "===== B7_MAIN_${index} ====="; date -Is
    print_timing_state "MAIN_${index}_PRE"
    "$PYTHON_BIN" "$OWNED/run_fixed_batch_b7_discovery.py" \
        --process-index "$index" \
        --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" \
        --reference-root "$REFERENCE_ROOT" \
        --patched-root "$PATCHED_ROOT" \
        --fla-root "$FLA_ROOT" \
        --json "$artifact"
    print_timing_state "MAIN_${index}_POST"
}

run_main 0
require_clean BETWEEN_MAIN_0_AND_1 || exit 93
run_main 1
require_clean AFTER_MAIN_1 || exit 93

main0="$RESULTS_DIR/c1_fixed_batch_b7_discovery_${LABEL}_main0.json"
main1="$RESULTS_DIR/c1_fixed_batch_b7_discovery_${LABEL}_main1.json"
AUDIT="$RESULTS_DIR/c1_fixed_batch_b7_discovery_${LABEL}.independent_audit.json"
echo "===== INDEPENDENT_STDLIB_AUDIT ====="
require_clean BEFORE_INDEPENDENT_AUDIT || exit 94
"$PYTHON_BIN" "$OWNED/analyze_fixed_batch_b7_discovery.py" "$main0" "$main1" \
    --expected-main-sha256 "$(sha256sum "$main0" | awk '{print $1}')" "$(sha256sum "$main1" | awk '{print $1}')" \
    --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" \
    --expected-analyzer-sha256 "$EXPECTED_ANALYZER_SHA256" \
    --json "$AUDIT"
require_clean AFTER_INDEPENDENT_AUDIT || exit 94
eligibility="$($PYTHON_BIN - "$AUDIT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)["second_allocation_decision"]["eligible"]
if type(value) is not bool:
    raise SystemExit("second-allocation eligibility is not an exact bool")
print(str(value).lower())
PY
)"
echo "DISCOVERY_ONLY=1"
echo "SECOND_ALLOCATION_ELIGIBLE=$eligibility"
echo "SECOND_ALLOCATION=not_submitted"
