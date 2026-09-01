#!/usr/bin/env bash
# Fresh, authorized one-GPU production-freeze audit for public fixed B=5.
# It runs only already-built artifacts and does not submit work, rebuild code,
# mutate git state, or change any dispatcher/registry source.
set -Eeuo pipefail

if [[ "${C1_FIXED_BATCH_B5_PRODUCTION_FREEZE_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: set C1_FIXED_BATCH_B5_PRODUCTION_FREEZE_GPU_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${A02_ROOT:?set A02_ROOT}"
: "${PATCHED_ROOT:?set PATCHED_ROOT to the prebuilt audited worktree}"
: "${FLA_ROOT:?set FLA_ROOT to the pinned FLA worktree}"
: "${LABEL:?set LABEL}"
: "${SLURM_JOB_ID:?run this production freeze only inside its own Slurm allocation}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_tp8_dispatch"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
HISTORY_JSON="${HISTORY_JSON:-$OWNED/results/c1_fixed_batch_fla_integration_b300_sm103a_b5_public_r1.json}"
HISTORY_LOG="${HISTORY_LOG:-$OWNED/results/c1_fixed_batch_fla_integration_b300_sm103a_b5_public_r1_job11786.log}"

EXPECTED_PATCHED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_FLA_COMMIT="a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d"
EXPECTED_SO_SHA256="8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005"
EXPECTED_FLASH_KDA_PYTHON_SHA256="9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84"
EXPECTED_AUTO_DISPATCH_SHA256="f7ad41d6368e82dc75ed2a384542ee527f5487f38a001b054f25840855327b45"
EXPECTED_FLA_BACKEND_SHA256="3cd5ce30fb7869cca13131bc6255b6ec0cf2f9eaa86ac2a20d2fa7d9b0709342"
EXPECTED_RUNNER_SHA256="97cbed25e748539159c7f15d6f0e12c639f9809dc877f4164b3dcf88e7d6d385"
EXPECTED_POLICY_SHA256="954558233af89f0b578bbe98606ce2db6f18fe4157f39052d6cac17db6333918"
EXPECTED_ANALYZER_SHA256="f6ae4192b2e9a1d6b93f280fec284bb29219c4232b6f77cbe5d1965261c04347"
EXPECTED_SHARED_SHA256="4ba4b26241c97c59ca7ffe8b4d8f4a965bc259bb7e40aa4a93d1271b06cdb83f"
EXPECTED_HARNESS_SHA256="5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52"
EXPECTED_FLA_INIT_SHA256="b7e0d26abd3162884ce94f37d0210a25344025055383d377937f300a3bf5f45d"
EXPECTED_FLA_BACKENDS_SHA256="a6934ff1ae38c412bf7cbc7cc5009522c103931c55ea06db51088c50a6e6e635"
EXPECTED_FLA_KDA_INIT_SHA256="24564f0101f87a26056e9061ed771caf0d8f6d9a00b4dbded701ea7525b45acb"
EXPECTED_FLA_KDA_BACKENDS_SHA256="86a2e1c1313dcfc6c64a3ba906dd824448edca36c31a961595c250cfaf8dd797"
EXPECTED_FLA_FLASH_KDA_SHA256="0d35a0bb8532135528e9c53a4cb0d16f282a55c974a84faab350ef72203bd5a2"
EXPECTED_FLA_CHUNK_SHA256="a15aa6ac257af5c8f7a1d6afc21a417391568e90a8898d4e228ddeb48cb0e9b8"
EXPECTED_HISTORY_JSON_SHA256="7867854fee67d8632ab09d08d1b0f1b8f0bee2f632f9c12943c4a5da9ba18a1f"
EXPECTED_HISTORY_LOG_SHA256="1767cdea2355bd6884f9d555fd0b4812c89f07899e1dc349cf612ea251316e94"
EXPECTED_HISTORY_SLURM_JOB_ID="11786"
GPU_TIMING_FIELDS="index,uuid,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu"

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_fixed_batch_b5_production_freeze_${LABEL}_job${SLURM_JOB_ID}.log"
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
    gpu_query || { [[ "$rc" -eq 0 ]] && rc=92; }
    print_timing_state POST || { [[ "$rc" -eq 0 ]] && rc=92; }
    require_clean POST || { [[ "$rc" -eq 0 ]] && rc=91; }
    echo "FINAL_RC=$rc"
    exit "$rc"
}
trap finish EXIT

echo "===== ENVIRONMENT_GATE ====="
command -v "$PYTHON_BIN"
require_commit "$PATCHED_ROOT" "$EXPECTED_PATCHED_COMMIT" "patched"
require_commit "$FLA_ROOT" "$EXPECTED_FLA_COMMIT" "FLA"
[[ -z "$(git -C "$FLA_ROOT" status --short --untracked-files=no)" ]] || {
    echo "FLA tracked/staged worktree is not clean" >&2; exit 84;
}
SO_PATHS=("$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so)
[[ "${#SO_PATHS[@]}" -eq 1 && -f "${SO_PATHS[0]}" ]] || {
    echo "expected exactly one prebuilt flash_kda_C.cpython-*-linux-gnu.so" >&2; exit 89;
}
require_sha "${SO_PATHS[0]}" "$EXPECTED_SO_SHA256" "extension"
require_sha "$PATCHED_ROOT/flash_kda/__init__.py" "$EXPECTED_FLASH_KDA_PYTHON_SHA256" "loaded flash_kda Python wrapper"
require_sha "$OWNED/auto_dispatch.py" "$EXPECTED_AUTO_DISPATCH_SHA256" "auto_dispatch"
require_sha "$OWNED/fla_backend.py" "$EXPECTED_FLA_BACKEND_SHA256" "fla_backend"
require_sha "$OWNED/run_fixed_batch_fla_integration.py" "$EXPECTED_RUNNER_SHA256" "public integration runner"
require_sha "$OWNED/test_auto_dispatch_policy.py" "$EXPECTED_POLICY_SHA256" "dispatcher policy tests"
require_sha "$OWNED/analyze_fixed_batch_b5_production_freeze.py" "$EXPECTED_ANALYZER_SHA256" "production-freeze analyzer"
require_sha "$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py" "$EXPECTED_SHARED_SHA256" "shared input/state helper"
require_sha "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" "$EXPECTED_HARNESS_SHA256" "validation harness"
require_sha "$FLA_ROOT/fla/__init__.py" "$EXPECTED_FLA_INIT_SHA256" "FLA init"
require_sha "$FLA_ROOT/fla/ops/backends/__init__.py" "$EXPECTED_FLA_BACKENDS_SHA256" "FLA backends init"
require_sha "$FLA_ROOT/fla/ops/kda/__init__.py" "$EXPECTED_FLA_KDA_INIT_SHA256" "FLA KDA init"
require_sha "$FLA_ROOT/fla/ops/kda/backends/__init__.py" "$EXPECTED_FLA_KDA_BACKENDS_SHA256" "FLA KDA backends init"
require_sha "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" "$EXPECTED_FLA_FLASH_KDA_SHA256" "pinned FLA flash_kda backend"
require_sha "$FLA_ROOT/fla/ops/kda/chunk.py" "$EXPECTED_FLA_CHUNK_SHA256" "pinned FLA public chunk API"
require_sha "$HISTORY_JSON" "$EXPECTED_HISTORY_JSON_SHA256" "frozen historical public JSON"
require_sha "$HISTORY_LOG" "$EXPECTED_HISTORY_LOG_SHA256" "frozen historical public Slurm log"
[[ "${FLA_DISABLE_BACKEND_DISPATCH:-0}" != 1 ]] || { echo "FLA backend dispatch is disabled" >&2; exit 83; }
[[ "$(gpu_query | sed '/^[[:space:]]*$/d' | wc -l)" -eq 1 ]] || { echo "expected exactly one visible B300 GPU" >&2; exit 82; }

echo "===== PRE_AUDIT ====="; date -Is; hostname; gpu_query
print_timing_state PRE
require_clean PRE || exit 90

echo "===== SOURCE_IDENTITY ====="
printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD)"
printf 'FLA_COMMIT=%s\n' "$(git -C "$FLA_ROOT" rev-parse HEAD)"
sha256sum \
    "$OWNED/run_clean_fixed_batch_b5_production_freeze.sh" \
    "$OWNED/analyze_fixed_batch_b5_production_freeze.py" \
    "$OWNED/run_fixed_batch_fla_integration.py" \
    "$OWNED/auto_dispatch.py" \
    "$OWNED/fla_backend.py" \
    "$OWNED/test_auto_dispatch_policy.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py" \
    "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" \
    "$PATCHED_ROOT/flash_kda/__init__.py" \
    "${SO_PATHS[0]}" \
    "$FLA_ROOT/fla/__init__.py" \
    "$FLA_ROOT/fla/ops/backends/__init__.py" \
    "$FLA_ROOT/fla/ops/kda/__init__.py" \
    "$FLA_ROOT/fla/ops/kda/backends/__init__.py" \
    "$FLA_ROOT/fla/ops/kda/backends/flash_kda.py" \
    "$FLA_ROOT/fla/ops/kda/chunk.py" \
    "$HISTORY_JSON" \
    "$HISTORY_LOG"

echo "===== DESCRIBE_COMPILE_AND_CPU_POLICY_GATES ====="
"$PYTHON_BIN" "$OWNED/run_fixed_batch_fla_integration.py" \
    --describe --seed 20260831 \
    --json "$RESULTS_DIR/c1_fixed_batch_b5_production_freeze_${LABEL}.plan.json"
"$PYTHON_BIN" -m py_compile \
    "$OWNED/auto_dispatch.py" \
    "$OWNED/fla_backend.py" \
    "$OWNED/run_fixed_batch_fla_integration.py" \
    "$OWNED/analyze_fixed_batch_b5_production_freeze.py"
"$PYTHON_BIN" "$OWNED/test_auto_dispatch_policy.py"
"$PYTHON_BIN" "$OWNED/test_varlen_metadata_policy.py"

# No setup.py, compiler, build command, git mutation, patch generator, or
# sbatch occurs below.  The public runner is a new process in this allocation.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT:$FLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export C1_B300_FLASH_KDA=1
export FLA_FLASH_KDA=1
export C1_FIXED_BATCH_FLA_INTEGRATION_CLEAN_GPU=1
cd "$PATCHED_ROOT"

CURRENT_JSON="$RESULTS_DIR/c1_fixed_batch_b5_production_freeze_${LABEL}.json"
echo "===== FRESH_PUBLIC_FLA_18_CELL_RUNNER ====="; date -Is
print_timing_state RUNNER_PRE
require_clean RUNNER_PRE || exit 93
"$PYTHON_BIN" "$OWNED/run_fixed_batch_fla_integration.py" --seed 20260831 --json "$CURRENT_JSON"
print_timing_state RUNNER_POST
require_clean AFTER_FRESH_PUBLIC_RUNNER || exit 93

FREEZE_JSON="$RESULTS_DIR/c1_fixed_batch_b5_production_freeze_${LABEL}.production_freeze.json"
echo "===== TWO_ALLOCATION_PRODUCTION_FREEZE_GATE ====="
require_clean BEFORE_PRODUCTION_FREEZE_ANALYSIS || exit 94
"$PYTHON_BIN" "$OWNED/analyze_fixed_batch_b5_production_freeze.py" "$HISTORY_JSON" "$CURRENT_JSON" \
    --expected-history-json-sha256 "$EXPECTED_HISTORY_JSON_SHA256" \
    --expected-current-json-sha256 "$(sha256sum "$CURRENT_JSON" | awk '{print $1}')" \
    --expected-auto-dispatch-sha256 "$EXPECTED_AUTO_DISPATCH_SHA256" \
    --expected-fla-backend-sha256 "$EXPECTED_FLA_BACKEND_SHA256" \
    --expected-runner-sha256 "$EXPECTED_RUNNER_SHA256" \
    --expected-policy-sha256 "$EXPECTED_POLICY_SHA256" \
    --expected-analyzer-sha256 "$EXPECTED_ANALYZER_SHA256" \
    --history-slurm-log "$HISTORY_LOG" \
    --expected-history-slurm-log-sha256 "$EXPECTED_HISTORY_LOG_SHA256" \
    --history-slurm-job-id "$EXPECTED_HISTORY_SLURM_JOB_ID" \
    --current-slurm-log "$LOG" \
    --current-slurm-job-id "$SLURM_JOB_ID" \
    --json "$FREEZE_JSON"
require_clean AFTER_PRODUCTION_FREEZE_ANALYSIS || exit 94
freeze_value="$("$PYTHON_BIN" - "$FREEZE_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)["production_freeze_passed"]
if type(value) is not bool or value is not True:
    raise SystemExit("production_freeze_passed is not exact true")
print(str(value).lower())
PY
)"
[[ "$freeze_value" == true ]] || { echo "production freeze output did not pass" >&2; exit 95; }
echo "PRODUCTION_FREEZE_PASSED=true"
