#!/usr/bin/env bash
# One clean B300 allocation for the *real production* T=8191 freeze protocol.
# A1 and A2 must be separate Slurm jobs.  This script builds nothing and never
# writes auto_dispatch.py, fla_backend.py, or the comparison worktrees.
set -Eeuo pipefail

if [[ "${C1_TAIL8191_PRODUCTION_FREEZE_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != --authorized-by-parent ]]; then
    echo "refusing GPU run: set C1_TAIL8191_PRODUCTION_FREEZE_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift
: "${A02_ROOT:?set A02_ROOT to the assignment02 checkout}"
: "${PATCHED_ROOT:?set PATCHED_ROOT to audited patched worktree}"
: "${REFERENCE_ROOT:?set REFERENCE_ROOT to pinned reference worktree}"
: "${FLA_ROOT:?set FLA_ROOT to pinned FLA worktree}"
: "${LABEL:?set LABEL}"
: "${ALLOCATION_ID:?set ALLOCATION_ID=A1 or A2}"
: "${SLURM_JOB_ID:?this protocol requires a Slurm allocation identity}"
: "${EXPECTED_PROTOCOL_SHELL_SHA256:?parent must externally pin this protocol shell SHA256}"
[[ "$ALLOCATION_ID" == A1 || "$ALLOCATION_ID" == A2 ]] || { echo "ALLOCATION_ID must be A1 or A2" >&2; exit 65; }
[[ "$SLURM_JOB_ID" =~ ^[1-9][0-9]*$ ]] || { echo "SLURM_JOB_ID must be a positive decimal" >&2; exit 66; }
[[ "$EXPECTED_PROTOCOL_SHELL_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid external protocol shell SHA256" >&2; exit 67; }
if [[ "$ALLOCATION_ID" == A1 ]] && { [[ -n "${A1_AUDIT+x}" ]] || [[ -n "${A1_AUDIT_SHA256+x}" ]]; }; then
    echo "A1 must not receive A1_AUDIT or A1_AUDIT_SHA256; refusing ignored prerequisite arguments" >&2
    exit 70
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_tail8191_production_freeze"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
EXPECTED_SHELL_PATH="$(realpath -e "$OWNED/run_clean_tail8191_production_freeze.sh")"
EXECUTED_SHELL_PATH="$(realpath -e "$0")"
[[ "$EXECUTED_SHELL_PATH" == "$EXPECTED_SHELL_PATH" ]] || { echo "executed protocol shell canonical path drift" >&2; exit 68; }
[[ "$(sha256sum "$EXECUTED_SHELL_PATH" | awk '{print $1}')" == "$EXPECTED_PROTOCOL_SHELL_SHA256" ]] || { echo "executed protocol shell external SHA256 gate failed" >&2; exit 69; }
EXPECTED_SO_SHA256="8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_FLASH_KDA_PYTHON_SHA256="9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
EXPECTED_AUTO_DISPATCH_SHA256="9633179c6a5e8ce61b1857a102f81074c6bdfdc63163e13980f40c24d6583c29"
EXPECTED_FLA_BACKEND_SHA256="152ac4ba897c1887963553d592456b7f988c89105e6f6cb507a44757058d5fd1"
EXPECTED_RUNNER_SHA256="f4144f5fbdd61396ff907c6290b767b5570e04d19087f8332f9db10e56e7b1dc"
EXPECTED_ANALYZER_SHA256="0e42ff13dce296f83dff8cac8359eebb7ca459caaaef926fd6f6affb284b91dc"
EXPECTED_HARNESS_SHA256="5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"
EXPECTED_REFERENCE_TORCH_REF_SHA256="bb037c8b74bb7caa9bdddd144a0f221309c72116456fe7cb2fbd95dbd9b406a5"
EXPECTED_PINNED_REFERENCE_HELPER_PATH="/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so"
EXPECTED_PINNED_REFERENCE_HELPER_SHA256="8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
EXPECTED_PATCHED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_REFERENCE_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT="a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
EXPECTED_PATCHED_FLASH_KDA_CPP_SHA256="38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4"
EXPECTED_PATCHED_FWD_H_SHA256="613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083"
EXPECTED_PATCHED_FWD_LAUNCH_SHA256="a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928"
EXPECTED_FLA_INIT_SHA256="b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d"
EXPECTED_FLA_BACKENDS_SHA256="a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635"
EXPECTED_FLA_KDA_INIT_SHA256="24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb"
EXPECTED_FLA_KDA_BACKENDS_SHA256="86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797"
EXPECTED_FLA_FLASH_KDA_SHA256="0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2"
EXPECTED_FLA_CHUNK_SHA256="a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_tail8191_production_freeze_${LABEL}_${ALLOCATION_ID}_job${SLURM_JOB_ID}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
apps_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
require_clean() {
    local stage="$1" apps used
    apps="$(apps_query)" || { echo "$stage: compute-app query failed" >&2; return 92; }
    used="$(memory_query)" || { echo "$stage: memory query failed" >&2; return 92; }
    echo "${stage}_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "${stage}_COMPUTE_APPS_END"; printf '%s_MEMORY_USED_MIB=%s\n' "$stage" "$used"
    [[ -z "$apps" ]] || return 1
    [[ "$(printf '%s\n' "$used" | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || return 1
    [[ "$(printf '%s' "$used" | tr -d '[:space:]')" == 0 ]] || return 1
}
require_sha() { [[ -f "$1" ]] || { echo "missing $3: $1" >&2; return 86; }; [[ "$(sha256sum "$1" | awk '{print $1}')" == "$2" ]] || { echo "$3 SHA256 gate failed" >&2; return 87; }; }
require_commit() { [[ "$(git -C "$1" rev-parse HEAD)" == "$2" ]] || { echo "$3 commit gate failed" >&2; return 85; }; }
source_snapshot() {
    sha256sum "$OWNED/run_clean_tail8191_production_freeze.sh" \
      "$OWNED/run_tail8191_production_freeze.py" "$OWNED/analyze_tail8191_production_freeze.py" \
      "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py" \
      "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py" \
      "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" \
      "${SO_PATHS[0]}" "$PATCHED_ROOT/flash_kda/__init__.py" \
      "$PATCHED_ROOT/csrc/flash_kda.cpp" "$PATCHED_ROOT/csrc/fwd.h" "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu" \
      "$REFERENCE_ROOT/tests/torch_ref.py" \
      "$PINNED_REFERENCE_HELPER_CANONICAL_PATH" \
      "$FLA_ROOT/fla/__init__.py" "$FLA_ROOT/fla/ops/backends/__init__.py" \
      "$FLA_ROOT/fla/ops/kda/__init__.py" "$FLA_ROOT/fla/ops/kda/backends/__init__.py" \
      "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" "$FLA_ROOT/fla/ops/kda/chunk.py"
}
finish() {
    local rc=$?
    trap - EXIT
    echo "===== POST_AUDIT ====="; date -Is; gpu_query || rc=92
    require_clean POST || rc=91
    if [[ -n "${PRE_SOURCE_SNAPSHOT:-}" ]]; then
        [[ "$PRE_SOURCE_SNAPSHOT" == "$(source_snapshot)" ]] || { echo "source/SO identity changed during allocation" >&2; rc=96; }
    fi
    echo "FINAL_RC=$rc"
    exit "$rc"
}
trap finish EXIT

echo "===== ENVIRONMENT_GATE ====="; command -v "$PYTHON_BIN"
require_commit "$PATCHED_ROOT" "$EXPECTED_PATCHED_COMMIT" patched
require_commit "$REFERENCE_ROOT" "$EXPECTED_REFERENCE_COMMIT" reference
require_commit "$FLA_ROOT" "$EXPECTED_FLA_COMMIT" FLA
EXPECTED_PATCHED_STATUS=$' M csrc/flash_kda.cpp\n M csrc/fwd.h\n M csrc/smxx/fwd_launch.cu'
[[ "$(git -C "$PATCHED_ROOT" status --porcelain=v1 --untracked-files=no)" == "$EXPECTED_PATCHED_STATUS" ]] || { echo "patched tracked status manifest drift" >&2; exit 84; }
[[ -z "$(git -C "$REFERENCE_ROOT" status --porcelain=v1 --untracked-files=no)" && -z "$(git -C "$FLA_ROOT" status --porcelain=v1 --untracked-files=no)" ]] || { echo "reference/FLA tracked tree dirty" >&2; exit 84; }
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
[[ ${#SO_PATHS[@]} -eq 1 && -f "${SO_PATHS[0]}" ]] || { echo "expected exactly one extension SO" >&2; exit 89; }
require_sha "${SO_PATHS[0]}" "$EXPECTED_SO_SHA256" extension
require_sha "$OWNED/run_clean_tail8191_production_freeze.sh" "$EXPECTED_PROTOCOL_SHELL_SHA256" "externally pinned protocol shell"
require_sha "$OWNED/run_tail8191_production_freeze.py" "$EXPECTED_RUNNER_SHA256" runner
require_sha "$OWNED/analyze_tail8191_production_freeze.py" "$EXPECTED_ANALYZER_SHA256" analyzer
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py" "$EXPECTED_AUTO_DISPATCH_SHA256" auto_dispatch
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py" "$EXPECTED_FLA_BACKEND_SHA256" fla_backend
require_sha "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" "$EXPECTED_HARNESS_SHA256" harness
require_sha "$PATCHED_ROOT/flash_kda/__init__.py" "$EXPECTED_FLASH_KDA_PYTHON_SHA256" "flash_kda Python wrapper"
require_sha "$PATCHED_ROOT/csrc/flash_kda.cpp" "$EXPECTED_PATCHED_FLASH_KDA_CPP_SHA256" "patched csrc/flash_kda.cpp"
require_sha "$PATCHED_ROOT/csrc/fwd.h" "$EXPECTED_PATCHED_FWD_H_SHA256" "patched csrc/fwd.h"
require_sha "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu" "$EXPECTED_PATCHED_FWD_LAUNCH_SHA256" "patched csrc/smxx/fwd_launch.cu"
require_sha "$REFERENCE_ROOT/tests/torch_ref.py" "$EXPECTED_REFERENCE_TORCH_REF_SHA256" "reference torch_ref"
: "${C1_PINNED_REFERENCE_HELPER_PATH:?set C1_PINNED_REFERENCE_HELPER_PATH to the exact audited helper}"
: "${C1_PINNED_REFERENCE_HELPER_SHA256:?set C1_PINNED_REFERENCE_HELPER_SHA256 to the exact audited helper SHA256}"
[[ "$C1_PINNED_REFERENCE_HELPER_PATH" == "$EXPECTED_PINNED_REFERENCE_HELPER_PATH" ]] || { echo "pinned reference helper path environment drift" >&2; exit 97; }
[[ "$C1_PINNED_REFERENCE_HELPER_SHA256" == "$EXPECTED_PINNED_REFERENCE_HELPER_SHA256" ]] || { echo "pinned reference helper SHA environment drift" >&2; exit 97; }
PINNED_REFERENCE_HELPER_CANONICAL_PATH="$(realpath -e "$C1_PINNED_REFERENCE_HELPER_PATH")"
[[ "$PINNED_REFERENCE_HELPER_CANONICAL_PATH" == "$EXPECTED_PINNED_REFERENCE_HELPER_PATH" && -f "$PINNED_REFERENCE_HELPER_CANONICAL_PATH" ]] || { echo "pinned reference helper canonical path drift" >&2; exit 97; }
require_sha "$PINNED_REFERENCE_HELPER_CANONICAL_PATH" "$EXPECTED_PINNED_REFERENCE_HELPER_SHA256" "pinned reference helper"
require_sha "$FLA_ROOT/fla/__init__.py" "$EXPECTED_FLA_INIT_SHA256" "FLA init"
require_sha "$FLA_ROOT/fla/ops/backends/__init__.py" "$EXPECTED_FLA_BACKENDS_SHA256" "FLA backends init"
require_sha "$FLA_ROOT/fla/ops/kda/__init__.py" "$EXPECTED_FLA_KDA_INIT_SHA256" "FLA KDA init"
require_sha "$FLA_ROOT/fla/ops/kda/backends/__init__.py" "$EXPECTED_FLA_KDA_BACKENDS_SHA256" "FLA KDA backends init"
require_sha "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" "$EXPECTED_FLA_FLASH_KDA_SHA256" "FLA flash_kda backend"
require_sha "$FLA_ROOT/fla/ops/kda/chunk.py" "$EXPECTED_FLA_CHUNK_SHA256" "FLA public chunk API"
[[ "$(gpu_query | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || { echo "exactly one visible GPU required" >&2; exit 88; }
PRE_SOURCE_SNAPSHOT="$(source_snapshot)"

if [[ "$ALLOCATION_ID" == A2 ]]; then
    : "${A1_AUDIT:?A2 is blocked until A1_AUDIT is supplied}"
    : "${A1_AUDIT_SHA256:?A2 is blocked until A1_AUDIT_SHA256 is supplied}"
    echo "===== A1_PREREQUISITE ====="
    "$PYTHON_BIN" "$OWNED/analyze_tail8191_production_freeze.py" --verify-allocation --audit "$A1_AUDIT" --expected-audit-sha256 "$A1_AUDIT_SHA256" --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" --expected-analyzer-sha256 "$EXPECTED_ANALYZER_SHA256" --expected-protocol-shell-sha256 "$EXPECTED_PROTOCOL_SHELL_SHA256" --current-slurm-job-id "$SLURM_JOB_ID" --require-pass
fi

echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query; require_clean PRE || exit 90
echo "===== COMPILE_AND_PLAN_GATE ====="
"$PYTHON_BIN" -m py_compile "$OWNED/run_tail8191_production_freeze.py" "$OWNED/analyze_tail8191_production_freeze.py"
"$PYTHON_BIN" "$OWNED/analyze_tail8191_production_freeze.py" --self-test
"$PYTHON_BIN" "$OWNED/run_tail8191_production_freeze.py" --self-test
"$PYTHON_BIN" "$OWNED/run_tail8191_production_freeze.py" --allocation "$ALLOCATION_ID" --process-index 0 --protocol-shell-path "$OWNED/run_clean_tail8191_production_freeze.sh" --expected-protocol-shell-sha256 "$EXPECTED_PROTOCOL_SHELL_SHA256" --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" --expected-analyzer-sha256 "$EXPECTED_ANALYZER_SHA256" --describe --json "$RESULTS_DIR/c1_tail8191_production_freeze_${LABEL}_${ALLOCATION_ID}.plan.json"

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PATCHED_ROOT:$FLA_ROOT:$(dirname "$A02_ROOT"):$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_TAIL8191_PRODUCTION_FREEZE_CLEAN_GPU=1 FLA_FLASH_KDA=1 C1_B300_FLASH_KDA=1
cd "$PATCHED_ROOT"
run_main() {
    local index="$1" artifact="$RESULTS_DIR/c1_tail8191_production_freeze_${LABEL}_${ALLOCATION_ID}_main${1}.json"
    echo "===== ${ALLOCATION_ID}_MAIN_${index} ====="; date -Is
    "$PYTHON_BIN" "$OWNED/run_tail8191_production_freeze.py" --allocation "$ALLOCATION_ID" --process-index "$index" --reference-root "$REFERENCE_ROOT" --patched-root "$PATCHED_ROOT" --fla-root "$FLA_ROOT" --protocol-shell-path "$OWNED/run_clean_tail8191_production_freeze.sh" --expected-protocol-shell-sha256 "$EXPECTED_PROTOCOL_SHELL_SHA256" --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" --expected-analyzer-sha256 "$EXPECTED_ANALYZER_SHA256" --json "$artifact"
}
run_main 0; require_clean BETWEEN_MAIN_0_AND_1 || exit 93
run_main 1; require_clean AFTER_MAIN_1 || exit 93
main0="$RESULTS_DIR/c1_tail8191_production_freeze_${LABEL}_${ALLOCATION_ID}_main0.json"
main1="$RESULTS_DIR/c1_tail8191_production_freeze_${LABEL}_${ALLOCATION_ID}_main1.json"
audit="$RESULTS_DIR/c1_tail8191_production_freeze_${LABEL}_${ALLOCATION_ID}.allocation_audit.json"
echo "===== INDEPENDENT_STDLIB_ALLOCATION_AUDIT ====="
a1_bind_args=()
a2_current_job_args=()
if [[ "$ALLOCATION_ID" == A2 ]]; then
    a1_bind_args=(--a1-audit "$A1_AUDIT" --expected-a1-sha256 "$A1_AUDIT_SHA256")
    a2_current_job_args=(--current-slurm-job-id "$SLURM_JOB_ID")
fi
"$PYTHON_BIN" "$OWNED/analyze_tail8191_production_freeze.py" --allocation "$ALLOCATION_ID" --main-json "$main0" "$main1" --expected-main-sha256 "$(sha256sum "$main0" | awk '{print $1}')" "$(sha256sum "$main1" | awk '{print $1}')" --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" --expected-analyzer-sha256 "$EXPECTED_ANALYZER_SHA256" --expected-protocol-shell-sha256 "$EXPECTED_PROTOCOL_SHELL_SHA256" "${a1_bind_args[@]}" "${a2_current_job_args[@]}" --json "$audit" --require-pass
require_clean AFTER_INDEPENDENT_AUDIT || exit 94
echo "ALLOCATION_ID=$ALLOCATION_ID"; echo "ALLOCATION_ELIGIBLE=true"
