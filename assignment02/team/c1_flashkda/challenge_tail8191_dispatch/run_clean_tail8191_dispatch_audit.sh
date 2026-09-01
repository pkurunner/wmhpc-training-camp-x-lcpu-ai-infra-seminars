#!/usr/bin/env bash
# One clean B300 allocation for the pre-registered tail8191 public-route gate.
# Invoke this script once with ALLOCATION_ID=A1 and once, in a new Slurm job,
# with ALLOCATION_ID=A2.  It neither builds nor patches any source tree.
set -Eeuo pipefail

if [[ "${C1_TAIL8191_DISPATCH_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_TAIL8191_DISPATCH_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT to the assignment02 checkout}"
: "${PATCHED_ROOT:?set PATCHED_ROOT to the prebuilt audited comparison worktree}"
: "${REFERENCE_ROOT:?set REFERENCE_ROOT to the pinned upstream worktree}"
: "${FLA_ROOT:?set FLA_ROOT to the pinned FLA worktree}"
: "${LABEL:?set LABEL}"
: "${ALLOCATION_ID:?set ALLOCATION_ID=A1 or A2}"
: "${SLURM_JOB_ID:?this protocol requires a Slurm allocation identity}"
[[ "$ALLOCATION_ID" == A1 || "$ALLOCATION_ID" == A2 ]] || { echo "ALLOCATION_ID must be A1 or A2" >&2; exit 65; }
[[ "$SLURM_JOB_ID" =~ ^[1-9][0-9]*$ ]] || { echo "SLURM_JOB_ID must be a positive decimal" >&2; exit 66; }

PYTHON_BIN="${PYTHON_BIN:-python}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_tail8191_dispatch"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
EXPECTED_SO_SHA256="8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_RUNNER_SHA256="ecddc9250c19f899616ae85fb6ff0e4a7047b414f8144b5836b0a65975f07f64"
EXPECTED_ANALYZER_SHA256="40297138e2a9fd6c0b58c159bd8801750e0842c2c30f6b6a708e29bfb779594f"
EXPECTED_AUTO_DISPATCH_SHA256="f7ad41d6368e82dc75ed2a384542ee527f5487f38a001b054f25840855327b45"
EXPECTED_FLA_BACKEND_SHA256="3cd5ce30fb7869cca13131bc6255b6ec0cf2f9eaa86ac2a20d2fa7d9b0709342"
EXPECTED_HARNESS_SHA256="5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"
EXPECTED_FLASH_KDA_PYTHON_SHA256="9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
EXPECTED_PATCHED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_REFERENCE_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT="a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_tail8191_dispatch_${LABEL}_${ALLOCATION_ID}_job${SLURM_JOB_ID}.log"
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
require_sha() {
    local path="$1" expected="$2" label="$3"
    [[ -f "$path" ]] || { echo "missing $label: $path" >&2; return 86; }
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || { echo "$label SHA256 gate failed" >&2; return 87; }
}
require_commit() {
    local path="$1" expected="$2" label="$3"
    [[ "$(git -C "$path" rev-parse HEAD)" == "$expected" ]] || { echo "$label commit gate failed" >&2; return 85; }
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

echo "===== ENVIRONMENT_GATE ====="
command -v "$PYTHON_BIN"
require_commit "$PATCHED_ROOT" "$EXPECTED_PATCHED_COMMIT" patched
require_commit "$REFERENCE_ROOT" "$EXPECTED_REFERENCE_COMMIT" reference
require_commit "$FLA_ROOT" "$EXPECTED_FLA_COMMIT" FLA
[[ -z "$(git -C "$REFERENCE_ROOT" status --short --untracked-files=no)" ]] || { echo "reference tracked tree is dirty" >&2; exit 84; }
[[ -z "$(git -C "$FLA_ROOT" status --short --untracked-files=no)" ]] || { echo "FLA tracked tree is dirty" >&2; exit 84; }
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
[[ "${#SO_PATHS[@]}" -eq 1 && -f "${SO_PATHS[0]}" ]] || { echo "expected exactly one audited extension" >&2; exit 89; }
require_sha "${SO_PATHS[0]}" "$EXPECTED_SO_SHA256" extension
require_sha "$OWNED/run_tail8191_dispatch.py" "$EXPECTED_RUNNER_SHA256" runner
require_sha "$OWNED/analyze_tail8191_dispatch.py" "$EXPECTED_ANALYZER_SHA256" analyzer
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py" "$EXPECTED_AUTO_DISPATCH_SHA256" auto_dispatch
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py" "$EXPECTED_FLA_BACKEND_SHA256" fla_backend
require_sha "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" "$EXPECTED_HARNESS_SHA256" harness
require_sha "$PATCHED_ROOT/flash_kda/__init__.py" "$EXPECTED_FLASH_KDA_PYTHON_SHA256" flash_kda_python
[[ "$(gpu_query | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || { echo "expected exactly one visible GPU" >&2; exit 88; }

echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_clean PRE || exit 90
echo "===== SOURCE_IDENTITY ====="
sha256sum "$OWNED/run_tail8191_dispatch.py" "$OWNED/analyze_tail8191_dispatch.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py" \
    "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" \
    "$PATCHED_ROOT/flash_kda/__init__.py" "${SO_PATHS[0]}" \
    "$FLA_ROOT/fla/__init__.py" "$FLA_ROOT/fla/ops/backends/__init__.py" \
    "$FLA_ROOT/fla/ops/kda/__init__.py" "$FLA_ROOT/fla/ops/kda/backends/__init__.py" \
    "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" "$FLA_ROOT/fla/ops/kda/chunk.py" \
    "$REFERENCE_ROOT/tests/torch_ref.py"

echo "===== COMPILE_AND_PLAN_GATE ====="
"$PYTHON_BIN" -m py_compile "$OWNED/run_tail8191_dispatch.py" "$OWNED/analyze_tail8191_dispatch.py"
"$PYTHON_BIN" "$OWNED/analyze_tail8191_dispatch.py" --self-test
"$PYTHON_BIN" "$OWNED/run_tail8191_dispatch.py" --allocation "$ALLOCATION_ID" --process-index 0 \
    --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" --describe \
    --json "$RESULTS_DIR/c1_tail8191_dispatch_${LABEL}_${ALLOCATION_ID}.plan.json"

# There is deliberately no setup.py/NVCC/patch generator/source mutation below.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PATCHED_ROOT:$FLA_ROOT:$(dirname "$A02_ROOT"):$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_TAIL8191_DISPATCH_CLEAN_GPU=1
export FLA_FLASH_KDA=1
export C1_B300_FLASH_KDA=1
cd "$PATCHED_ROOT"
run_main() {
    local index="$1" artifact="$RESULTS_DIR/c1_tail8191_dispatch_${LABEL}_${ALLOCATION_ID}_main${1}.json"
    echo "===== ${ALLOCATION_ID}_MAIN_${index} ====="; date -Is
    "$PYTHON_BIN" "$OWNED/run_tail8191_dispatch.py" --allocation "$ALLOCATION_ID" --process-index "$index" \
        --reference-root "$REFERENCE_ROOT" --patched-root "$PATCHED_ROOT" --fla-root "$FLA_ROOT" \
        --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" --json "$artifact"
}
run_main 0
require_clean BETWEEN_MAIN_0_AND_1 || exit 93
run_main 1
require_clean AFTER_MAIN_1 || exit 93

main0="$RESULTS_DIR/c1_tail8191_dispatch_${LABEL}_${ALLOCATION_ID}_main0.json"
main1="$RESULTS_DIR/c1_tail8191_dispatch_${LABEL}_${ALLOCATION_ID}_main1.json"
audit="$RESULTS_DIR/c1_tail8191_dispatch_${LABEL}_${ALLOCATION_ID}.allocation_audit.json"
echo "===== INDEPENDENT_STDLIB_ALLOCATION_AUDIT ====="
"$PYTHON_BIN" "$OWNED/analyze_tail8191_dispatch.py" --allocation "$ALLOCATION_ID" \
    --main-json "$main0" "$main1" --expected-main-sha256 "$(sha256sum "$main0" | awk '{print $1}')" "$(sha256sum "$main1" | awk '{print $1}')" \
    --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" --expected-analyzer-sha256 "$EXPECTED_ANALYZER_SHA256" --json "$audit"
require_clean AFTER_INDEPENDENT_AUDIT || exit 94
eligible="$($PYTHON_BIN - "$audit" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)["allocation_gate"]["eligible"]
if type(value) is not bool:
    raise SystemExit("allocation eligibility must be an exact bool")
print(str(value).lower())
PY
)"
printf 'ALLOCATION_ID=%s\nALLOCATION_ELIGIBLE=%s\n' "$ALLOCATION_ID" "$eligible"
if [[ "$eligible" != true ]]; then
    echo "STOP: allocation failed the preregistered all-contract/all-repeat/all-percentile gate" >&2
    exit 95
fi
