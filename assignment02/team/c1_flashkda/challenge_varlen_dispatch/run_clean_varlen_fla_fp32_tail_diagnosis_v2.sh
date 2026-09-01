#!/usr/bin/env bash
# Corrected v2 diagnostic only.  No build and no production-map mutation occur here.
set -Eeuo pipefail

if [[ "${C1_VARLEN_FLA_FP32_TAIL_DIAG_V2_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_VARLEN_FLA_FP32_TAIL_DIAG_V2_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT}"
: "${REFERENCE_ROOT:?set REFERENCE_ROOT}"
: "${FLA_ROOT:?set FLA_ROOT}"
: "${LABEL:?set LABEL}"
: "${C1_PINNED_REFERENCE_HELPER_PATH:?set C1_PINNED_REFERENCE_HELPER_PATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_varlen_dispatch"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
EXPECTED_PATCHED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT="a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
EXPECTED_RUNNER_SHA256="3b0342af8aca96e85bef75228ae71b1a1e401484373dc42aedc204c2ed533fb0"
EXPECTED_ANALYZER_SHA256="0adb9f93e879d80d5287ff460b9b559935dd04a44853f0dff13363782449c100"
EXPECTED_CANDIDATE_SHA256="e07481e7e66551f1f9eaafa7f1fb9db9ef193f1694ff831ee0a07802c8ffbe14"
EXPECTED_AUTO_SHA256="2b817adb7d21d1f223e8df4616eeccd74e34a5b1944492211f0f0254147ba883"
EXPECTED_BACKEND_SHA256="8555995c04ecd666a580ddee02eae1d34820ef1a601cbad5d10f9c6b8505974b"
EXPECTED_METADATA_SHA256="f89a97ba284c7a24a3df54efca7bb60eb70f9cbda659200bd6cb3e8dfdaf4ccd"
EXPECTED_SO_SHA256="8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_HELPER_SHA256="8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f"
GPU_TIMING_FIELDS="index,uuid,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu"
SIDECAR_FIELDS="timestamp,index,uuid,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,power.draw,utilization.gpu,utilization.memory,temperature.gpu"

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_varlen_fla_fp32_tail_diagnosis_v2_${LABEL}_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1
telemetry_pid=''

gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
gpu_timing_query() { nvidia-smi --query-gpu="$GPU_TIMING_FIELDS" --format=csv,noheader; }
app_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
require_clean() {
    local stage="$1" apps used
    apps="$(app_query)" || return 92
    used="$(memory_query)" || return 92
    echo "${stage}_COMPUTE_APPS_BEGIN"; printf '%s\n' "$apps"; echo "${stage}_COMPUTE_APPS_END"
    printf '%s_MEMORY_USED_MIB=%s\n' "$stage" "$used"
    [[ -z "$apps" ]] || return 1
    [[ "$(printf '%s\n' "$used" | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || return 1
    while IFS= read -r line; do [[ "$line" =~ ^[[:space:]]*0[[:space:]]*$ ]] || return 1; done <<<"$used"
}
print_timing_state() {
    local stage="$1"
    echo "${stage}_GPU_TIMING_BEGIN"
    gpu_timing_query
    echo "${stage}_GPU_TIMING_END"
}
stop_telemetry() {
    if [[ -n "${telemetry_pid:-}" ]]; then
        kill "$telemetry_pid" 2>/dev/null || true
        wait "$telemetry_pid" 2>/dev/null || true
        telemetry_pid=''
    fi
}
finish() {
    local rc=$?
    trap - EXIT INT TERM
    stop_telemetry
    echo "===== POST_AUDIT ====="; date -Is
    gpu_query || rc=92
    print_timing_state POST || rc=92
    require_clean POST || rc=91
    echo "FINAL_RC=$rc"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
require_sha() {
    local path="$1" expected="$2" label="$3"
    [[ -f "$path" ]] || { echo "missing $label: $path" >&2; return 86; }
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected" ]] || { echo "$label SHA256 gate failed" >&2; return 87; }
}
require_commit() { [[ "$(git -C "$1" rev-parse HEAD)" == "$2" ]] || { echo "commit gate failed for $1" >&2; return 85; }; }

echo "===== ENVIRONMENT_GATE ====="
command -v "$PYTHON_BIN"
require_commit "$PATCHED_ROOT" "$EXPECTED_PATCHED_COMMIT"
require_commit "$REFERENCE_ROOT" "$EXPECTED_PATCHED_COMMIT"
require_commit "$FLA_ROOT" "$EXPECTED_FLA_COMMIT"
[[ -z "$(git -C "$REFERENCE_ROOT" status --short --untracked-files=no)" ]] || { echo "reference tracked/staged diff is not clean" >&2; exit 84; }
[[ -z "$(git -C "$FLA_ROOT" status --short --untracked-files=no)" ]] || { echo "FLA tracked/staged diff is not clean" >&2; exit 84; }
[[ "${C1_PINNED_REFERENCE_HELPER_SHA256:-}" == "$EXPECTED_HELPER_SHA256" ]] || { echo "helper SHA environment drift" >&2; exit 87; }
export C1_PINNED_REFERENCE_HELPER_SHA256="$EXPECTED_HELPER_SHA256"
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
[[ "${#SO_PATHS[@]}" -eq 1 && -f "${SO_PATHS[0]}" ]] || { echo "expected exactly one audited extension SO" >&2; exit 89; }
require_sha "${SO_PATHS[0]}" "$EXPECTED_SO_SHA256" "extension SO"
require_sha "$C1_PINNED_REFERENCE_HELPER_PATH" "$EXPECTED_HELPER_SHA256" "pinned helper"
require_sha "$OWNED/run_varlen_fla_fp32_tail_diagnosis_v2.py" "$EXPECTED_RUNNER_SHA256" "v2 runner"
require_sha "$OWNED/analyze_varlen_fla_fp32_tail_diagnosis_v2.py" "$EXPECTED_ANALYZER_SHA256" "v2 analyzer"
require_sha "$OWNED/run_varlen_fla_handoff_candidate.py" "$EXPECTED_CANDIDATE_SHA256" "candidate helper"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py" "$EXPECTED_AUTO_SHA256" "auto dispatcher"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py" "$EXPECTED_BACKEND_SHA256" "C1 FLA backend"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/varlen_metadata.py" "$EXPECTED_METADATA_SHA256" "varlen metadata"

echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query; print_timing_state PRE; require_clean PRE || exit 90
echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'REFERENCE_COMMIT=%s\n' "$(git -C "$REFERENCE_ROOT" rev-parse HEAD)"
printf 'FLA_COMMIT=%s\n' "$(git -C "$FLA_ROOT" rev-parse HEAD)"
sha256sum "$OWNED/run_varlen_fla_fp32_tail_diagnosis_v2.py" "$OWNED/analyze_varlen_fla_fp32_tail_diagnosis_v2.py" "$0" "$OWNED/run_varlen_fla_handoff_candidate.py" "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/auto_dispatch.py" "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/fla_backend.py" "$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch/varlen_metadata.py" "$C1_PINNED_REFERENCE_HELPER_PATH" "${SO_PATHS[0]}"

# Intentionally no build command: only the frozen, pre-existing extension/helper are allowed.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$FLA_ROOT:$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_VARLEN_FLA_FP32_TAIL_DIAG_V2_CLEAN_GPU=1 C1_B300_FLASH_KDA=1 C1_B300_VARLEN_CPU_DESCRIPTOR=1 FLA_FLASH_KDA=1
export C1_VARLEN_FLA_FP32_TAIL_DIAG_V2_RUNNER_SHA256="$EXPECTED_RUNNER_SHA256"
export C1_VARLEN_FLA_FP32_TAIL_DIAG_V2_ANALYZER_SHA256="$EXPECTED_ANALYZER_SHA256"
export C1_VARLEN_FLA_FP32_TAIL_DIAG_V2_CANDIDATE_SHA256="$EXPECTED_CANDIDATE_SHA256"
cd "$PATCHED_ROOT"
echo "===== CPU_DESCRIBE_AND_CONSTRUCTION ====="
"$PYTHON_BIN" "$OWNED/run_varlen_fla_fp32_tail_diagnosis_v2.py" --mode stability_main --process-index 0 --describe --json "$RESULTS_DIR/c1_varlen_fla_fp32_tail_diagnosis_v2_${LABEL}.plan.json"
"$PYTHON_BIN" "$OWNED/run_varlen_fla_fp32_tail_diagnosis_v2.py" --mode stability_main --process-index 0 --cpu-construction-check --json "$RESULTS_DIR/c1_varlen_fla_fp32_tail_diagnosis_v2_${LABEL}.cpu.json"
"$PYTHON_BIN" -m py_compile "$OWNED/run_varlen_fla_fp32_tail_diagnosis_v2.py" "$OWNED/analyze_varlen_fla_fp32_tail_diagnosis_v2.py"

run_main() {
    local index="$1" artifact
    artifact="$RESULTS_DIR/c1_varlen_fla_fp32_tail_diagnosis_v2_${LABEL}_main${index}.json"
    echo "===== STABILITY_MAIN_${index} ====="; date -Is
    print_timing_state "MAIN_${index}_PRE"; require_clean "MAIN_${index}_PRE" || return 90
    "$PYTHON_BIN" "$OWNED/run_varlen_fla_fp32_tail_diagnosis_v2.py" --reference-root "$REFERENCE_ROOT" --mode stability_main --process-index "$index" --json "$artifact"
    print_timing_state "MAIN_${index}_POST"; require_clean "MAIN_${index}_POST" || return 93
}
run_main 0
echo "===== BETWEEN_MAIN_0_1 ====="; print_timing_state BETWEEN_MAIN_0_1; require_clean BETWEEN_MAIN_0_1 || exit 93
run_main 1
echo "===== BETWEEN_MAIN_1_2 ====="; print_timing_state BETWEEN_MAIN_1_2; require_clean BETWEEN_MAIN_1_2 || exit 93
run_main 2
echo "===== BETWEEN_MAIN_2_TELEMETRY ====="; print_timing_state BETWEEN_MAIN_2_TELEMETRY; require_clean BETWEEN_MAIN_2_TELEMETRY || exit 93

TELEMETRY_JSON="$RESULTS_DIR/c1_varlen_fla_fp32_tail_diagnosis_v2_${LABEL}_telemetry.json"
TELEMETRY_CSV="$RESULTS_DIR/c1_varlen_fla_fp32_tail_diagnosis_v2_${LABEL}_telemetry.csv"
echo "===== NON_GATING_TELEMETRY ====="; date -Is
print_timing_state TELEMETRY_PRE; require_clean TELEMETRY_PRE || exit 90
if nvidia-smi --query-gpu="$SIDECAR_FIELDS" --format=csv,noheader >/dev/null 2>&1; then
    nvidia-smi --query-gpu="$SIDECAR_FIELDS" --format=csv,noheader --loop-ms=200 >"$TELEMETRY_CSV" 2>&1 &
    telemetry_pid=$!
else
    printf 'UNAVAILABLE: nvidia-smi did not support the explanatory sidecar fields\n' >"$TELEMETRY_CSV"
fi
set +e
"$PYTHON_BIN" "$OWNED/run_varlen_fla_fp32_tail_diagnosis_v2.py" --reference-root "$REFERENCE_ROOT" --mode non_gating_telemetry --process-index 3 --json "$TELEMETRY_JSON"
telemetry_rc=$?
set -e
stop_telemetry
[[ "$telemetry_rc" -eq 0 ]] || exit "$telemetry_rc"
print_timing_state TELEMETRY_POST; require_clean TELEMETRY_POST || exit 93

main0="$RESULTS_DIR/c1_varlen_fla_fp32_tail_diagnosis_v2_${LABEL}_main0.json"
main1="$RESULTS_DIR/c1_varlen_fla_fp32_tail_diagnosis_v2_${LABEL}_main1.json"
main2="$RESULTS_DIR/c1_varlen_fla_fp32_tail_diagnosis_v2_${LABEL}_main2.json"
AUDIT="$RESULTS_DIR/c1_varlen_fla_fp32_tail_diagnosis_v2_${LABEL}.independent_audit.json"
echo "===== INDEPENDENT_DIAGNOSTIC_AUDIT ====="
"$PYTHON_BIN" "$OWNED/analyze_varlen_fla_fp32_tail_diagnosis_v2.py" "$main0" "$main1" "$main2" \
    --expected-main-sha256 "$(sha256sum "$main0" | awk '{print $1}')" "$(sha256sum "$main1" | awk '{print $1}')" "$(sha256sum "$main2" | awk '{print $1}')" \
    --telemetry-json "$TELEMETRY_JSON" --expected-telemetry-json-sha256 "$(sha256sum "$TELEMETRY_JSON" | awk '{print $1}')" \
    --telemetry-csv "$TELEMETRY_CSV" --expected-telemetry-csv-sha256 "$(sha256sum "$TELEMETRY_CSV" | awk '{print $1}')" \
    --json "$AUDIT"
"$PYTHON_BIN" - "$AUDIT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)["second_allocation_decision"]["eligible"]
if type(value) is not bool:
    raise SystemExit("audit eligible field is not bool")
print("SECOND_ALLOCATION_ELIGIBLE=" + str(value).lower())
PY
echo "===== AFTER_INDEPENDENT_ANALYZER ====="; print_timing_state AFTER_INDEPENDENT_ANALYZER; require_clean AFTER_INDEPENDENT_ANALYZER || exit 94
echo "DIAGNOSTIC_ONLY=1"
echo "WHITELIST_ACTION=unchanged"
echo "SECOND_ALLOCATION_DECISION=inspect second_allocation_decision.eligible in $AUDIT; this script does not submit or promote"
