#!/usr/bin/env bash
# One fresh, clean B300 allocation for B=7,H=12,T=2048,none only.
# A1 and A2 must be submitted as distinct Slurm allocations by the caller.
set -Eeuo pipefail

if [[ "${C1_FIXED_BATCH_B7_NONE_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_FIXED_BATCH_B7_NONE_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT}"
: "${REFERENCE_ROOT:?set REFERENCE_ROOT}"
: "${FLA_ROOT:?set FLA_ROOT}"
: "${LABEL:?set LABEL}"
: "${ALLOCATION_ID:?set ALLOCATION_ID=A1 or A2}"
: "${SLURM_JOB_ID:?this protocol requires a Slurm job ID}"
: "${EXPECTED_PROTOCOL_SHELL_SHA256:?external sbatch entry must set the frozen protocol-shell SHA256}"
PINNED_REFERENCE_HELPER_PATH="/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so"
EXPECTED_PINNED_REFERENCE_HELPER_SHA256="8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
: "${C1_PINNED_REFERENCE_HELPER_PATH:?set the exact cached sigmoid_ext helper path}"
: "${C1_PINNED_REFERENCE_HELPER_SHA256:?set the cached sigmoid_ext helper SHA256}"
[[ "$ALLOCATION_ID" == A1 || "$ALLOCATION_ID" == A2 ]] || { echo "ALLOCATION_ID must be A1 or A2" >&2; exit 65; }
[[ "$SLURM_JOB_ID" =~ ^[1-9][0-9]*$ ]] || { echo "SLURM_JOB_ID must be a positive decimal" >&2; exit 66; }
[[ "$EXPECTED_PROTOCOL_SHELL_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "EXPECTED_PROTOCOL_SHELL_SHA256 must be lowercase SHA256" >&2; exit 67; }
[[ "$C1_PINNED_REFERENCE_HELPER_PATH" == "$PINNED_REFERENCE_HELPER_PATH" ]] || { echo "C1_PINNED_REFERENCE_HELPER_PATH must equal the pinned helper path" >&2; exit 67; }
[[ "$C1_PINNED_REFERENCE_HELPER_SHA256" == "$EXPECTED_PINNED_REFERENCE_HELPER_SHA256" ]] || { echo "C1_PINNED_REFERENCE_HELPER_SHA256 must equal the pinned helper SHA256" >&2; exit 67; }
[[ "$(readlink -f "$C1_PINNED_REFERENCE_HELPER_PATH")" == "$PINNED_REFERENCE_HELPER_PATH" ]] || { echo "pinned helper canonical path gate failed" >&2; exit 67; }
export C1_PINNED_REFERENCE_HELPER_PATH C1_PINNED_REFERENCE_HELPER_SHA256
PROTOCOL_SHELL_PATH="$(readlink -f "$0")"
ACTUAL_PROTOCOL_SHELL_SHA256="$(sha256sum "$0" | awk '{print $1}')"
[[ "$ACTUAL_PROTOCOL_SHELL_SHA256" == "$EXPECTED_PROTOCOL_SHELL_SHA256" ]] || { echo "external protocol-shell SHA256 gate failed" >&2; exit 67; }
export C1_FIXED_BATCH_B7_NONE_PROTOCOL_SHELL_PATH="$PROTOCOL_SHELL_PATH"
export C1_FIXED_BATCH_B7_NONE_PROTOCOL_SHELL_SHA256="$ACTUAL_PROTOCOL_SHELL_SHA256"
if [[ "$ALLOCATION_ID" == A1 && ( -n "${A1_AUDIT+x}" || -n "${EXPECTED_A1_AUDIT_SHA256+x}" ) ]]; then
    echo "A1 rejects A1_AUDIT / EXPECTED_A1_AUDIT_SHA256; they are an A2-only precondition" >&2
    exit 67
fi
if [[ "$ALLOCATION_ID" == A2 ]]; then
    : "${A1_AUDIT:?A2 requires the exact eligible A1 allocation audit}"
    : "${EXPECTED_A1_AUDIT_SHA256:?A2 requires the SHA256 of A1_AUDIT}"
    [[ "$EXPECTED_A1_AUDIT_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "EXPECTED_A1_AUDIT_SHA256 must be lowercase SHA256" >&2; exit 67; }
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_fixed_batch_b7_none"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
EXPECTED_SO_SHA256="8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_RUNNER_SHA256="481462766589ee3ec23c7ab0454a923f2f28aa506826413433fda0450030f534"
EXPECTED_ANALYZER_SHA256="2fd71aecd563dc9c5c314de78f52050dddd76539266fcc54d9d93982c2892705"
EXPECTED_AUTO_DISPATCH_SHA256="9cdd460058254016af58723875bdf99ebe74f8e016a4c6027eb7fb38c8e9a88c"
EXPECTED_FLA_BACKEND_SHA256="206e448abcd3d64826f87a20e7d57c790fef6adacd91e26edcb10a3711b9b656"
EXPECTED_HARNESS_SHA256="5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"
EXPECTED_FLASH_KDA_PYTHON_SHA256="9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
EXPECTED_PATCHED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_REFERENCE_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT="a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
PATCHED_DIRTY_CPP_SHA256="38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4"
PATCHED_DIRTY_FWD_H_SHA256="613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083"
PATCHED_DIRTY_LAUNCH_SHA256="a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_fixed_batch_b7_none_${LABEL}_${ALLOCATION_ID}_job${SLURM_JOB_ID}.log"
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
require_patched_dirty_overlay() {
    local expected actual
    expected=$' M csrc/flash_kda.cpp\n M csrc/fwd.h\n M csrc/smxx/fwd_launch.cu'
    actual="$(git -C "$PATCHED_ROOT" status --porcelain=v1 --untracked-files=no)" || { echo "patched tracked-status query failed" >&2; return 85; }
    [[ "$actual" == "$expected" ]] || { echo "patched tracked dirty set gate failed: $actual" >&2; return 84; }
    require_sha "$PATCHED_ROOT/csrc/flash_kda.cpp" "$PATCHED_DIRTY_CPP_SHA256" patched_dirty_flash_kda_cpp
    require_sha "$PATCHED_ROOT/csrc/fwd.h" "$PATCHED_DIRTY_FWD_H_SHA256" patched_dirty_fwd_h
    require_sha "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu" "$PATCHED_DIRTY_LAUNCH_SHA256" patched_dirty_fwd_launch_cu
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
require_patched_dirty_overlay
require_commit "$REFERENCE_ROOT" "$EXPECTED_REFERENCE_COMMIT" reference
require_commit "$FLA_ROOT" "$EXPECTED_FLA_COMMIT" FLA
[[ -z "$(git -C "$REFERENCE_ROOT" status --short --untracked-files=no)" ]] || { echo "reference tracked tree is dirty" >&2; exit 84; }
[[ -z "$(git -C "$FLA_ROOT" status --short --untracked-files=no)" ]] || { echo "FLA tracked tree is dirty" >&2; exit 84; }
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
[[ "${#SO_PATHS[@]}" -eq 1 && -f "${SO_PATHS[0]}" ]] || { echo "expected exactly one audited extension" >&2; exit 89; }
require_sha "${SO_PATHS[0]}" "$EXPECTED_SO_SHA256" extension
require_sha "$OWNED/run_fixed_batch_b7_none.py" "$EXPECTED_RUNNER_SHA256" runner
require_sha "$OWNED/analyze_fixed_batch_b7_none.py" "$EXPECTED_ANALYZER_SHA256" analyzer
require_sha "$PROTOCOL_SHELL_PATH" "$EXPECTED_PROTOCOL_SHELL_SHA256" protocol_shell_external_attestation
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py" "$EXPECTED_AUTO_DISPATCH_SHA256" auto_dispatch
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py" "$EXPECTED_FLA_BACKEND_SHA256" fla_backend
require_sha "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" "$EXPECTED_HARNESS_SHA256" harness
require_sha "$PATCHED_ROOT/flash_kda/__init__.py" "$EXPECTED_FLASH_KDA_PYTHON_SHA256" flash_kda_python
require_sha "$C1_PINNED_REFERENCE_HELPER_PATH" "$EXPECTED_PINNED_REFERENCE_HELPER_SHA256" pinned_reference_helper
[[ "$(gpu_query | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || { echo "expected one visible GPU" >&2; exit 88; }

echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
require_clean PRE || exit 90
echo "===== SOURCE_IDENTITY ====="
printf 'PROTOCOL_SHELL_PATH=%s\nPROTOCOL_SHELL_SHA256=%s\nEXPECTED_PROTOCOL_SHELL_SHA256=%s\nPINNED_REFERENCE_HELPER_PATH=%s\nPINNED_REFERENCE_HELPER_SHA256=%s\n' "$PROTOCOL_SHELL_PATH" "$ACTUAL_PROTOCOL_SHELL_SHA256" "$EXPECTED_PROTOCOL_SHELL_SHA256" "$C1_PINNED_REFERENCE_HELPER_PATH" "$C1_PINNED_REFERENCE_HELPER_SHA256"
sha256sum "$0"
echo "===== PATCHED_TRACKED_DIRTY_OVERLAY ====="
git -C "$PATCHED_ROOT" status --porcelain=v1 --untracked-files=no
sha256sum "$PATCHED_ROOT/csrc/flash_kda.cpp" "$PATCHED_ROOT/csrc/fwd.h" "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu"
sha256sum "$OWNED/run_fixed_batch_b7_none.py" "$OWNED/analyze_fixed_batch_b7_none.py" \
    "$PROTOCOL_SHELL_PATH" \
    "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py" \
    "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" \
    "$PATCHED_ROOT/flash_kda/__init__.py" "$C1_PINNED_REFERENCE_HELPER_PATH" "${SO_PATHS[0]}" \
    "$FLA_ROOT/fla/__init__.py" "$FLA_ROOT/fla/ops/backends/__init__.py" \
    "$FLA_ROOT/fla/ops/kda/__init__.py" "$FLA_ROOT/fla/ops/kda/backends/__init__.py" \
    "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" "$FLA_ROOT/fla/ops/kda/chunk.py" \
    "$REFERENCE_ROOT/tests/torch_ref.py"
echo "===== COMPILE_AND_PLAN_GATE ====="
"$PYTHON_BIN" -m py_compile "$OWNED/run_fixed_batch_b7_none.py" "$OWNED/analyze_fixed_batch_b7_none.py"
"$PYTHON_BIN" "$OWNED/run_fixed_batch_b7_none.py" --self-test
"$PYTHON_BIN" "$OWNED/analyze_fixed_batch_b7_none.py" --self-test
if [[ "$ALLOCATION_ID" == A2 ]]; then
    echo "===== A2_A1_PRECONDITION (CPU-only, before any CUDA workload) ====="
    "$PYTHON_BIN" "$OWNED/analyze_fixed_batch_b7_none.py" --precondition-a1 \
        --a1-audit "$A1_AUDIT" --expected-a1-sha256 "$EXPECTED_A1_AUDIT_SHA256" \
        --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" --expected-analyzer-sha256 "$EXPECTED_ANALYZER_SHA256" \
        --expected-protocol-shell-sha256 "$EXPECTED_PROTOCOL_SHELL_SHA256" --current-slurm-job-id "$SLURM_JOB_ID"
else
    echo "===== A1_A2_PRECONDITION=not_applicable ====="
fi
"$PYTHON_BIN" "$OWNED/run_fixed_batch_b7_none.py" --allocation "$ALLOCATION_ID" --process-index 0 \
    --reference-root "$REFERENCE_ROOT" --patched-root "$PATCHED_ROOT" --fla-root "$FLA_ROOT" \
    --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" --expected-analyzer-sha256 "$EXPECTED_ANALYZER_SHA256" \
    --expected-protocol-shell-sha256 "$EXPECTED_PROTOCOL_SHELL_SHA256" --describe \
    --json "$RESULTS_DIR/c1_fixed_batch_b7_none_${LABEL}_${ALLOCATION_ID}.plan.json"

# No setup.py/NVCC/patch generator/source mutation appears below.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PATCHED_ROOT:$FLA_ROOT:$(dirname "$A02_ROOT"):$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_FIXED_BATCH_B7_NONE_CLEAN_GPU=1
export FLA_FLASH_KDA=1
export C1_B300_FLASH_KDA=1
cd "$PATCHED_ROOT"
run_main() {
    local index="$1" artifact="$RESULTS_DIR/c1_fixed_batch_b7_none_${LABEL}_${ALLOCATION_ID}_main${1}.json"
    echo "===== ${ALLOCATION_ID}_MAIN_${index} ====="; date -Is
    "$PYTHON_BIN" "$OWNED/run_fixed_batch_b7_none.py" --allocation "$ALLOCATION_ID" --process-index "$index" \
        --reference-root "$REFERENCE_ROOT" --patched-root "$PATCHED_ROOT" --fla-root "$FLA_ROOT" \
        --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" --expected-analyzer-sha256 "$EXPECTED_ANALYZER_SHA256" \
        --expected-protocol-shell-sha256 "$EXPECTED_PROTOCOL_SHELL_SHA256" --json "$artifact"
}
run_main 0
require_clean BETWEEN_MAIN_0_AND_1 || exit 93
run_main 1
require_clean AFTER_MAIN_1 || exit 93
main0="$RESULTS_DIR/c1_fixed_batch_b7_none_${LABEL}_${ALLOCATION_ID}_main0.json"
main1="$RESULTS_DIR/c1_fixed_batch_b7_none_${LABEL}_${ALLOCATION_ID}_main1.json"
audit="$RESULTS_DIR/c1_fixed_batch_b7_none_${LABEL}_${ALLOCATION_ID}.allocation_audit.json"
echo "===== INDEPENDENT_STDLIB_ALLOCATION_AUDIT ====="
a1_binding_args=()
if [[ "$ALLOCATION_ID" == A2 ]]; then
    # Persist the same exact prerequisite that was re-opened before any CUDA
    # workload.  The allocation auditor independently reopens it again and
    # binds its canonical path/SHA/job/full source identity into the A2 JSON.
    a1_binding_args=(--a1-audit "$A1_AUDIT" --expected-a1-sha256 "$EXPECTED_A1_AUDIT_SHA256" --current-slurm-job-id "$SLURM_JOB_ID")
fi
"$PYTHON_BIN" "$OWNED/analyze_fixed_batch_b7_none.py" --allocation "$ALLOCATION_ID" \
    --main-json "$main0" "$main1" --expected-main-sha256 "$(sha256sum "$main0" | awk '{print $1}')" "$(sha256sum "$main1" | awk '{print $1}')" \
    --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" --expected-analyzer-sha256 "$EXPECTED_ANALYZER_SHA256" \
    --expected-protocol-shell-sha256 "$EXPECTED_PROTOCOL_SHELL_SHA256" "${a1_binding_args[@]}" --json "$audit"
require_clean AFTER_INDEPENDENT_AUDIT || exit 94
eligible="$($PYTHON_BIN - "$audit" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)["allocation_gate"]["eligible"]
if type(value) is not bool:
    raise SystemExit("allocation eligibility must be exact bool")
print(str(value).lower())
PY
)"
printf 'ALLOCATION_ID=%s\nALLOCATION_ELIGIBLE=%s\n' "$ALLOCATION_ID" "$eligible"
if [[ "$eligible" != true ]]; then
    echo "STOP: B7 none did not meet every preregistered repeat/percentile gate" >&2
    exit 95
fi
