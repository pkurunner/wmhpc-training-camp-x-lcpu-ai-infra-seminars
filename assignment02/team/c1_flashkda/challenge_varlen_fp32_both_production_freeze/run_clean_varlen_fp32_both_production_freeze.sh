#!/usr/bin/env bash
# External-gated A1/A2 entrypoint for the production skew FP32-both freeze.
set -Eeuo pipefail

[[ "${1:-}" == "--authorized-by-parent" ]] || { echo "parent authorization argument required" >&2; exit 64; }
[[ "${C1_SKEW_PRODUCTION_FREEZE_GPU_AUTHORIZED:-}" == 1 ]] || { echo "GPU authorization environment gate required" >&2; exit 64; }
: "${A02_ROOT:?A02_ROOT required}"
: "${PATCHED_ROOT:?PATCHED_ROOT required}"
: "${REFERENCE_ROOT:?REFERENCE_ROOT required}"
: "${FLA_ROOT:?FLA_ROOT required}"
: "${PYTHON_BIN:?PYTHON_BIN required}"
: "${ALLOCATION_ID:?ALLOCATION_ID required}"
: "${LABEL:?LABEL required}"
: "${SLURM_JOB_ID:?must run inside a Slurm allocation}"
: "${C1_PINNED_REFERENCE_HELPER_PATH:?pinned reference helper path required}"
: "${C1_PINNED_REFERENCE_HELPER_SHA256:?pinned reference helper SHA required}"
: "${C1_SKEW_PRODUCTION_FREEZE_RUNNER_SHA256:?external runner SHA required}"
: "${C1_SKEW_PRODUCTION_FREEZE_ANALYZER_SHA256:?external analyzer SHA required}"
: "${EXPECTED_PROTOCOL_SHELL_SHA256:?external shell SHA required}"
: "${C1_SKEW_PRODUCTION_AUTO_DISPATCH_SHA256:?external final auto_dispatch SHA required}"
: "${C1_SKEW_PRODUCTION_FLA_BACKEND_SHA256:?external final fla_backend SHA required}"

case "$ALLOCATION_ID" in A1|A2) ;; *) echo "ALLOCATION_ID must be A1 or A2" >&2; exit 64;; esac
[[ "$SLURM_JOB_ID" =~ ^[1-9][0-9]*$ ]] || { echo "SLURM_JOB_ID must be positive ASCII decimal" >&2; exit 64; }
for digest in "$C1_SKEW_PRODUCTION_FREEZE_RUNNER_SHA256" "$C1_SKEW_PRODUCTION_FREEZE_ANALYZER_SHA256" "$EXPECTED_PROTOCOL_SHELL_SHA256" "$C1_SKEW_PRODUCTION_AUTO_DISPATCH_SHA256" "$C1_SKEW_PRODUCTION_FLA_BACKEND_SHA256" "$C1_PINNED_REFERENCE_HELPER_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || { echo "all SHA gates require lowercase SHA-256" >&2; exit 64; }
done

ROOT="$A02_ROOT/team/c1_flashkda/challenge_varlen_fp32_both_production_freeze"
RUNNER="$ROOT/run_varlen_fp32_both_production_freeze.py"
ANALYZER="$ROOT/analyze_varlen_fp32_both_production_freeze.py"
EXPECTED_SELF="$(readlink -f -- "$ROOT/run_clean_varlen_fp32_both_production_freeze.sh")" || { echo "cannot resolve canonical protocol shell" >&2; exit 66; }
EXECUTED_SELF="$(readlink -f -- "${BASH_SOURCE[0]}")" || { echo "cannot resolve actually executed protocol shell" >&2; exit 66; }
[[ "$EXECUTED_SELF" == "$EXPECTED_SELF" ]] || { echo "must execute canonical protocol shell directly, not a copied/spooled artifact" >&2; exit 67; }
SELF="$EXECUTED_SELF"
RESULTS="$ROOT/results"
PREFIX="$RESULTS/c1_varlen_fp32_both_production_${ALLOCATION_ID}_${LABEL}"
MAIN0="${PREFIX}_pid0.json"
MAIN1="${PREFIX}_pid1.json"
MANIFEST="${PREFIX}_allocation.json"
AUTO="$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py"
FLA_BACKEND="$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py"
CANDIDATE="$A02_ROOT/team/c1_flashkda/challenge_varlen_dispatch/run_varlen_fla_handoff_candidate.py"
VARLEN_METADATA="$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/varlen_metadata.py"
EXPECTED_CANDIDATE_SHA="e07481e7e66551f1f9eaafa7f1fb9db9ef193f1694ff831ee0a07802c8ffbe14"
EXPECTED_METADATA_SHA="f89a97ba284c7a24a3df54efca7bb60eb70f9cbda659200bd6cb3e8dfdaf4ccd"
EXPECTED_WRAPPER_SHA="9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
EXPECTED_PATCHED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLASH_KDA_CPP_SHA="38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4"
EXPECTED_FWD_H_SHA="613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083"
EXPECTED_FWD_LAUNCH_SHA="a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928"
EXPECTED_FLA_COMMIT="a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"

sha_file() { sha256sum "$1" | awk '{print $1}'; }
require_clean() {
  local stage="$1" row lines used apps
  row="$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)" || return 90
  lines="$(printf '%s\n' "$row" | sed '/^[[:space:]]*$/d' | wc -l)"
  [[ "$lines" == 1 ]] || { echo "$stage requires exactly one visible GPU" >&2; return 90; }
  used="$(printf '%s\n' "$row" | awk -F, '{gsub(/ /,"",$3); print $3}')"
  [[ "$used" == 0 ]] || { echo "$stage requires 0 MiB, got $used" >&2; return 90; }
  if ! apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"; then
    echo "$stage compute-app query failed" >&2
    return 90
  fi
  [[ -z "$apps" ]] || { echo "$stage has compute apps: $apps" >&2; return 90; }
}
require_patched_tree() {
  [[ "$(git -C "$PATCHED_ROOT" rev-parse HEAD)" == "$EXPECTED_PATCHED_COMMIT" ]] || return 91
  local expected actual
  expected=$' M csrc/flash_kda.cpp\n M csrc/fwd.h\n M csrc/smxx/fwd_launch.cu'
  actual="$(git -C "$PATCHED_ROOT" status --porcelain=v1 --untracked-files=no)"
  [[ "$actual" == "$expected" ]] || { echo "patched exact dirty-set drift" >&2; return 91; }
  [[ "$(sha_file "$PATCHED_ROOT/csrc/flash_kda.cpp")" == "$EXPECTED_FLASH_KDA_CPP_SHA" ]] || return 91
  [[ "$(sha_file "$PATCHED_ROOT/csrc/fwd.h")" == "$EXPECTED_FWD_H_SHA" ]] || return 91
  [[ "$(sha_file "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu")" == "$EXPECTED_FWD_LAUNCH_SHA" ]] || return 91
  [[ "$(sha_file "$PATCHED_ROOT/flash_kda/__init__.py")" == "$EXPECTED_WRAPPER_SHA" ]] || return 91
}
require_reference_and_fla_trees() {
  [[ "$(git -C "$REFERENCE_ROOT" rev-parse HEAD)" == "$EXPECTED_PATCHED_COMMIT" ]] || return 91
  [[ -z "$(git -C "$REFERENCE_ROOT" status --porcelain=v1 --untracked-files=no)" ]] || return 91
  [[ "$(git -C "$FLA_ROOT" rev-parse HEAD)" == "$EXPECTED_FLA_COMMIT" ]] || return 91
  [[ -z "$(git -C "$FLA_ROOT" status --porcelain=v1 --untracked-files=no)" ]] || return 91
}
source_snapshot() {
  local stage="$1"
  echo "===== ${stage}_SOURCE_SNAPSHOT ====="
  sha256sum "$RUNNER" "$ANALYZER" "$SELF" "$AUTO" "$FLA_BACKEND" "$CANDIDATE" "$VARLEN_METADATA" "$C1_PINNED_REFERENCE_HELPER_PATH" "$PATCHED_ROOT/flash_kda/__init__.py"
  git -C "$PATCHED_ROOT" status --porcelain=v1 --untracked-files=no
}
finish() {
  local rc=$?
  trap - EXIT INT TERM
  echo "===== POST_AUDIT ====="
  require_clean POST || rc=90
  require_patched_tree || rc=91
  require_reference_and_fla_trees || rc=91
  source_snapshot POST || rc=91
  echo "FINAL_RC=$rc"
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for path in "$RUNNER" "$ANALYZER" "$SELF" "$AUTO" "$FLA_BACKEND" "$CANDIDATE" "$VARLEN_METADATA" "$C1_PINNED_REFERENCE_HELPER_PATH" "$PATCHED_ROOT" "$REFERENCE_ROOT" "$FLA_ROOT"; do [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 66; }; done
[[ "$(sha_file "$RUNNER")" == "$C1_SKEW_PRODUCTION_FREEZE_RUNNER_SHA256" ]] || { echo "runner SHA mismatch" >&2; exit 67; }
[[ "$(sha_file "$ANALYZER")" == "$C1_SKEW_PRODUCTION_FREEZE_ANALYZER_SHA256" ]] || { echo "analyzer SHA mismatch" >&2; exit 67; }
[[ "$(sha_file "$SELF")" == "$EXPECTED_PROTOCOL_SHELL_SHA256" ]] || { echo "shell SHA mismatch" >&2; exit 67; }
[[ "$(sha_file "$AUTO")" == "$C1_SKEW_PRODUCTION_AUTO_DISPATCH_SHA256" ]] || { echo "auto_dispatch SHA mismatch" >&2; exit 67; }
[[ "$(sha_file "$FLA_BACKEND")" == "$C1_SKEW_PRODUCTION_FLA_BACKEND_SHA256" ]] || { echo "fla_backend SHA mismatch" >&2; exit 67; }
[[ "$(sha_file "$CANDIDATE")" == "$EXPECTED_CANDIDATE_SHA" && "$(sha_file "$VARLEN_METADATA")" == "$EXPECTED_METADATA_SHA" ]] || { echo "stable source ledger mismatch" >&2; exit 67; }
[[ "$(sha_file "$C1_PINNED_REFERENCE_HELPER_PATH")" == "$C1_PINNED_REFERENCE_HELPER_SHA256" && "$C1_PINNED_REFERENCE_HELPER_SHA256" == "8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f" ]] || { echo "pinned helper SHA mismatch" >&2; exit 67; }
require_patched_tree || exit 91
require_reference_and_fla_trees || exit 91
"$PYTHON_BIN" -m py_compile "$RUNNER" "$ANALYZER"
source_snapshot PRE

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$FLA_ROOT:$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_SKEW_PRODUCTION_FREEZE_CLEAN_GPU=1 C1_B300_FLASH_KDA=1 C1_B300_VARLEN_CPU_DESCRIPTOR=1 FLA_FLASH_KDA=1
export C1_SKEW_PRODUCTION_FREEZE_SHELL_PATH="$SELF"
if [[ "$ALLOCATION_ID" == A2 ]]; then
  : "${A1_ALLOCATION_MANIFEST:?A2 requires A1_ALLOCATION_MANIFEST}"
  : "${A1_ALLOCATION_MANIFEST_SHA256:?A2 requires A1_ALLOCATION_MANIFEST_SHA256}"
  "$PYTHON_BIN" "$ANALYZER" verify-allocation --allocation-manifest "$A1_ALLOCATION_MANIFEST" --expected-allocation-sha256 "$A1_ALLOCATION_MANIFEST_SHA256" --current-slurm-job-id "$SLURM_JOB_ID"
  a1_args=(--a1-allocation-manifest "$A1_ALLOCATION_MANIFEST" --expected-a1-allocation-manifest-sha256 "$A1_ALLOCATION_MANIFEST_SHA256")
else
  a1_args=()
fi
echo "===== PRE_AUDIT ====="
hostname
require_clean PRE || exit 90
cd "$PATCHED_ROOT"
run_pid() {
  local index="$1" artifact="$2"
  require_clean "PID_${index}_PRE" || return 90
  "$PYTHON_BIN" "$RUNNER" --allocation-id "$ALLOCATION_ID" --process-index "$index" --reference-root "$REFERENCE_ROOT" --json "$artifact"
  require_clean "PID_${index}_POST" || return 93
}
run_pid 0 "$MAIN0"
require_clean BETWEEN || exit 93
run_pid 1 "$MAIN1"
"$PYTHON_BIN" "$ANALYZER" allocation --allocation-id "$ALLOCATION_ID" --runner-json "$MAIN0" "$MAIN1" --expected-runner-sha256s "$(sha_file "$MAIN0")" "$(sha_file "$MAIN1")" "${a1_args[@]}" --json "$MANIFEST" --require-pass
echo "ALLOCATION_MANIFEST=$MANIFEST"
echo "NO_PRODUCTION_CHANGE=1"
