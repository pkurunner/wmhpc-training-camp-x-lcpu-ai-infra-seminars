#!/usr/bin/env bash
# Run exactly one independent A1 or A2 allocation; never changes production.
set -Eeuo pipefail

# This hash is deliberately supplied by the submitter, rather than embedded
# here: embedding a script's own final hash would create an edit/hash cycle.
: "${EXPECTED_PROTOCOL_SHELL_SHA256:?submitter must set EXPECTED_PROTOCOL_SHELL_SHA256}"
[[ "$EXPECTED_PROTOCOL_SHELL_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "EXPECTED_PROTOCOL_SHELL_SHA256 must be lowercase SHA-256" >&2; exit 63; }
[[ "$(sha256sum "$0" | awk '{print $1}')" == "$EXPECTED_PROTOCOL_SHELL_SHA256" ]] || { echo "external protocol-shell SHA mismatch" >&2; exit 63; }
export C1_VARLEN_FP32_BOTH_PROTOCOL_SHELL_PATH="$(realpath "$0")"

if [[ "${C1_VARLEN_FP32_BOTH_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_VARLEN_FP32_BOTH_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT}"
: "${REFERENCE_ROOT:?set REFERENCE_ROOT}"
: "${FLA_ROOT:?set FLA_ROOT}"
: "${ALLOCATION_ID:?set ALLOCATION_ID to A1 or A2}"
: "${LABEL:?set LABEL}"
: "${C1_PINNED_REFERENCE_HELPER_PATH:?set C1_PINNED_REFERENCE_HELPER_PATH}"
[[ "$ALLOCATION_ID" == A1 || "$ALLOCATION_ID" == A2 ]] || { echo "ALLOCATION_ID must be A1 or A2" >&2; exit 65; }
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || { echo "SLURM_JOB_ID must be a strictly positive decimal allocation ID" >&2; exit 65; }
PYTHON_BIN="${PYTHON_BIN:-python}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_varlen_fp32_both"
PATCHED_WRAPPER="$PATCHED_ROOT/flash_kda/__init__.py"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
mkdir -p "$RESULTS_DIR"
RUNNER="$OWNED/run_varlen_fp32_both_release.py"
ANALYZER="$OWNED/analyze_varlen_fp32_both_release.py"
CANDIDATE_HELPER="$A02_ROOT/team/c1_flashkda/challenge_varlen_dispatch/run_varlen_fla_handoff_candidate.py"
EXPECTED_RUNNER_SHA="9c6ea7a76a5fdc613996d583f6717be16bd0646fa1bd5df922cc7d026ad668ec"
EXPECTED_ANALYZER_SHA="9f2daedf5d84b436d935bd7884b42332b5158388603725b82bbf060852c4f14b"
EXPECTED_CANDIDATE_HELPER_SHA="e07481e7e66551f1f9eaafa7f1fb9db9ef193f1694ff831ee0a07802c8ffbe14"
EXPECTED_PRODUCTION_WRAPPER_SHA="9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
EXPECTED_PATCHED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_PATCHED_FLASH_KDA_CPP_SHA="38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4"
EXPECTED_PATCHED_FWD_H_SHA="613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083"
EXPECTED_PATCHED_FWD_LAUNCH_SHA="a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928"
RUNNER_SHA="$EXPECTED_RUNNER_SHA"
ANALYZER_SHA="$EXPECTED_ANALYZER_SHA"
TELEMETRY="$RESULTS_DIR/c1_varlen_fp32_both_${ALLOCATION_ID}_${LABEL}_telemetry.csv"
LOG="$RESULTS_DIR/c1_varlen_fp32_both_${ALLOCATION_ID}_${LABEL}_job${SLURM_JOB_ID}.log"
MAIN0="$RESULTS_DIR/c1_varlen_fp32_both_${ALLOCATION_ID}_${LABEL}_pid0.json"
MAIN1="$RESULTS_DIR/c1_varlen_fp32_both_${ALLOCATION_ID}_${LABEL}_pid1.json"
MANIFEST="$RESULTS_DIR/c1_varlen_fp32_both_${ALLOCATION_ID}_${LABEL}_allocation.json"
TELEMETRY_WINDOW="$RESULTS_DIR/c1_varlen_fp32_both_${ALLOCATION_ID}_${LABEL}_telemetry_window.json"
telemetry_pid=''
exec > >(tee "$LOG") 2>&1

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
app_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
require_clean() {
    local stage="$1" apps used
    apps="$(app_query)" || return 92; used="$(memory_query)" || return 92
    echo "${stage}_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "${stage}_COMPUTE_APPS_END"
    printf '%s_MEMORY_USED_MIB=%s\n' "$stage" "$used"
    [[ -z "$apps" && "$used" =~ ^[[:space:]]*0[[:space:]]*$ ]]
}
require_patched_exact_dirty_set() {
    local actual expected
    [[ "$(git -C "$PATCHED_ROOT" rev-parse HEAD)" == "$EXPECTED_PATCHED_COMMIT" ]] || { echo "PATCHED_ROOT commit drift" >&2; return 91; }
    expected=$' M csrc/flash_kda.cpp\n M csrc/fwd.h\n M csrc/smxx/fwd_launch.cu'
    actual="$(git -C "$PATCHED_ROOT" status --porcelain=v1 --untracked-files=no)"
    [[ "$actual" == "$expected" ]] || { echo "PATCHED_ROOT exact permitted dirty-set drift" >&2; printf 'actual status:\n%s\n' "$actual" >&2; return 91; }
    [[ "$(sha256sum "$PATCHED_ROOT/csrc/flash_kda.cpp" | awk '{print $1}')" == "$EXPECTED_PATCHED_FLASH_KDA_CPP_SHA" ]] || return 91
    [[ "$(sha256sum "$PATCHED_ROOT/csrc/fwd.h" | awk '{print $1}')" == "$EXPECTED_PATCHED_FWD_H_SHA" ]] || return 91
    [[ "$(sha256sum "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu" | awk '{print $1}')" == "$EXPECTED_PATCHED_FWD_LAUNCH_SHA" ]] || return 91
}
source_snapshot() {
    local stage="$1"
    echo "${stage}_PATCHED_WRAPPER_BEGIN"
    sha256sum "$PATCHED_WRAPPER"
    echo "${stage}_PATCHED_WRAPPER_END"
    [[ "$(sha256sum "$PATCHED_WRAPPER" | awk '{print $1}')" == "$EXPECTED_PRODUCTION_WRAPPER_SHA" ]]
    echo "${stage}_PATCHED_EXACT_DIRTY_SET_BEGIN"
    git -C "$PATCHED_ROOT" status --porcelain=v1 --untracked-files=no
    sha256sum "$PATCHED_ROOT/csrc/flash_kda.cpp" "$PATCHED_ROOT/csrc/fwd.h" "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu"
    echo "${stage}_PATCHED_EXACT_DIRTY_SET_END"
}
stop_telemetry() { if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; telemetry_pid=''; fi; }
require_telemetry_alive() { [[ -n "$telemetry_pid" ]] && kill -0 "$telemetry_pid" 2>/dev/null || { echo "telemetry loop unexpectedly exited" >&2; return 94; }; }
finish() { local rc=$?; trap - EXIT INT TERM; stop_telemetry; echo "===== POST_AUDIT ====="; date -Is; gpu_query || rc=92; require_clean POST || rc=91; require_patched_exact_dirty_set || rc=91; source_snapshot POST || rc=91; echo "FINAL_RC=$rc"; exit "$rc"; }
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
telemetry_loop() {
    while true; do
        local stamp row
        stamp="$(date +%s%N)"
        row="$(nvidia-smi --query-gpu=index,uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,power.limit --format=csv,noheader 2>/dev/null || true)"
        [[ "$(printf '%s\n' "$row" | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] && printf '%s,%s\n' "$stamp" "$row"
        sleep 0.2
    done
}

echo "===== ENVIRONMENT_GATE ====="; command -v "$PYTHON_BIN"
for path in "$RUNNER" "$ANALYZER" "$CANDIDATE_HELPER" "$C1_PINNED_REFERENCE_HELPER_PATH" "$PATCHED_ROOT" "$REFERENCE_ROOT" "$FLA_ROOT" "$PATCHED_WRAPPER"; do [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 66; }; done
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$EXPECTED_RUNNER_SHA" ]] || { echo "runner SHA mismatch" >&2; exit 67; }
[[ "$(sha256sum "$ANALYZER" | awk '{print $1}')" == "$EXPECTED_ANALYZER_SHA" ]] || { echo "analyzer SHA mismatch" >&2; exit 67; }
[[ "$(sha256sum "$CANDIDATE_HELPER" | awk '{print $1}')" == "$EXPECTED_CANDIDATE_HELPER_SHA" ]] || { echo "candidate handoff helper SHA mismatch" >&2; exit 67; }
[[ "$(sha256sum "$PATCHED_WRAPPER" | awk '{print $1}')" == "$EXPECTED_PRODUCTION_WRAPPER_SHA" ]] || { echo "production wrapper SHA mismatch" >&2; exit 67; }
[[ "$(sha256sum "$C1_PINNED_REFERENCE_HELPER_PATH" | awk '{print $1}')" == "${C1_PINNED_REFERENCE_HELPER_SHA256:-}" ]] || { echo "pinned helper SHA environment/file mismatch" >&2; exit 67; }
require_patched_exact_dirty_set || exit 91
source_snapshot PRE || exit 91
export C1_VARLEN_FP32_BOTH_RUNNER_SHA256="$RUNNER_SHA"
export C1_VARLEN_FP32_BOTH_ANALYZER_SHA256="$ANALYZER_SHA"
if [[ "$ALLOCATION_ID" == A2 ]]; then
    : "${A1_ALLOCATION_MANIFEST:?A2 requires A1_ALLOCATION_MANIFEST}"
    : "${A1_ALLOCATION_MANIFEST_SHA256:?A2 requires A1_ALLOCATION_MANIFEST_SHA256}"
    [[ "$A1_ALLOCATION_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "A1 manifest SHA must be lowercase SHA-256" >&2; exit 68; }
    "$PYTHON_BIN" "$ANALYZER" verify-allocation --allocation-manifest "$A1_ALLOCATION_MANIFEST" \
      --expected-allocation-sha256 "$A1_ALLOCATION_MANIFEST_SHA256" --expected-allocation-id A1 \
      --current-slurm-job-id "$SLURM_JOB_ID" --require-independent-current-job
    a1_allocation_args=(--a1-allocation-manifest "$A1_ALLOCATION_MANIFEST" \
      --expected-a1-allocation-manifest-sha256 "$A1_ALLOCATION_MANIFEST_SHA256" \
      --current-slurm-job-id "$SLURM_JOB_ID")
else
    a1_allocation_args=()
fi
echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query; require_clean PRE || exit 90
echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'REFERENCE_COMMIT=%s\n' "$(git -C "$REFERENCE_ROOT" rev-parse HEAD)"
printf 'FLA_COMMIT=%s\n' "$(git -C "$FLA_ROOT" rev-parse HEAD)"
sha256sum "$RUNNER" "$ANALYZER" "$CANDIDATE_HELPER" "$0" "$C1_PINNED_REFERENCE_HELPER_PATH" "$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so
"$PYTHON_BIN" -m py_compile "$RUNNER" "$ANALYZER"

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$FLA_ROOT:$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_VARLEN_FP32_BOTH_CLEAN_GPU=1 C1_B300_FLASH_KDA=1 C1_B300_VARLEN_CPU_DESCRIPTOR=1 FLA_FLASH_KDA=1
cd "$PATCHED_ROOT"
echo "===== START_TELEMETRY ====="; telemetry_loop >"$TELEMETRY" & telemetry_pid=$!; sleep 0.25; require_telemetry_alive
run_pid() {
    local index="$1" artifact="$2"
    echo "===== FRESH_PID_${index} ====="; date -Is; require_clean "PID_${index}_PRE" || return 90
    "$PYTHON_BIN" "$RUNNER" --allocation-id "$ALLOCATION_ID" --process-index "$index" --reference-root "$REFERENCE_ROOT" --json "$artifact"
    require_clean "PID_${index}_POST" || return 93
}
TELEMETRY_MAIN0_START_NS="$(date +%s%N)"
run_pid 0 "$MAIN0"
require_telemetry_alive
echo "===== BETWEEN_FRESH_PIDS ====="; require_clean BETWEEN || exit 93
run_pid 1 "$MAIN1"
require_telemetry_alive
TELEMETRY_MAIN1_END_NS="$(date +%s%N)"
stop_telemetry
[[ -s "$TELEMETRY" ]] || { echo "telemetry sidecar empty" >&2; exit 94; }
printf '{"main0_start_ns":%s,"main1_end_ns":%s,"telemetry_pid_was_alive":true}\n' "$TELEMETRY_MAIN0_START_NS" "$TELEMETRY_MAIN1_END_NS" >"$TELEMETRY_WINDOW"
[[ -s "$TELEMETRY_WINDOW" ]] || { echo "telemetry window sidecar empty" >&2; exit 94; }
echo "===== ALLOCATION_ANALYZER ====="
"$PYTHON_BIN" "$ANALYZER" allocation --allocation-id "$ALLOCATION_ID" --runner-json "$MAIN0" "$MAIN1" \
  --expected-runner-sha256s "$(sha256sum "$MAIN0" | awk '{print $1}')" "$(sha256sum "$MAIN1" | awk '{print $1}')" \
  --expected-runner-sha256 "$RUNNER_SHA" --telemetry-csv "$TELEMETRY" --expected-telemetry-sha256 "$(sha256sum "$TELEMETRY" | awk '{print $1}')" \
  --telemetry-window-sidecar "$TELEMETRY_WINDOW" --expected-telemetry-window-sidecar-sha256 "$(sha256sum "$TELEMETRY_WINDOW" | awk '{print $1}')" \
  --telemetry-window-start-ns "$TELEMETRY_MAIN0_START_NS" --telemetry-window-end-ns "$TELEMETRY_MAIN1_END_NS" \
  "${a1_allocation_args[@]}" --json "$MANIFEST" --require-pass
echo "ALLOCATION_MANIFEST=$MANIFEST"
echo "NO_PRODUCTION_CHANGE=1"
