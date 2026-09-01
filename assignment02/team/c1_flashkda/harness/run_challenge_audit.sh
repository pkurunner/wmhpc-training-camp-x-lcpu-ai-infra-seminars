#!/usr/bin/env bash
set -Eeuo pipefail

A02_ROOT="${1:?usage: run_challenge_audit.sh A02_ROOT PATCHED_ROOT BASELINE_ROOT LABEL}"
PATCHED_ROOT="${2:?usage: run_challenge_audit.sh A02_ROOT PATCHED_ROOT BASELINE_ROOT LABEL}"
BASELINE_ROOT="${3:?usage: run_challenge_audit.sh A02_ROOT PATCHED_ROOT BASELINE_ROOT LABEL}"
LABEL="${4:?usage: run_challenge_audit.sh A02_ROOT PATCHED_ROOT BASELINE_ROOT LABEL}"
PYTHON="${PYTHON:-python}"
export PATH="$(dirname "$PYTHON"):$PATH"
HARNESS="$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py"
FLA_REF="$A02_ROOT/team/c1_flashkda/fla_kda_ref"
LOG_DIR="$A02_ROOT/team/c1_flashkda/experiment_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/c1_vshard_${LABEL}_job${SLURM_JOB_ID:-none}.log"
exec > >(tee "$LOG") 2>&1

gpu_query() {
    nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader
}
app_query() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
}
finish() {
    rc=$?
    trap - EXIT
    echo "===== POST_AUDIT ====="
    date -Is
    gpu_query || true
    apps="$(app_query || true)"
    echo "POST_COMPUTE_APPS_BEGIN"
    printf '%s\n' "$apps"
    echo "POST_COMPUTE_APPS_END"
    if [[ -n "$apps" && "$rc" -eq 0 ]]; then rc=91; fi
    echo "FINAL_RC=$rc"
    exit "$rc"
}
trap finish EXIT

echo "===== PRE_AUDIT ====="
date -Is
hostname
gpu_query
apps="$(app_query || true)"
echo "PRE_COMPUTE_APPS_BEGIN"
printf '%s\n' "$apps"
echo "PRE_COMPUTE_APPS_END"
if [[ -n "$apps" ]]; then exit 90; fi

echo "===== SOURCE_IDENTITY ====="
printf 'PATCHED_COMMIT=%s\n' "$(git -C "$PATCHED_ROOT" rev-parse HEAD 2>/dev/null || printf snapshot)"
printf 'BASELINE_COMMIT=%s\n' "$(git -C "$BASELINE_ROOT" rev-parse HEAD 2>/dev/null || printf snapshot)"
printf 'PATCHED_STATUS_BEGIN\n'
git -C "$PATCHED_ROOT" status --short 2>/dev/null || true
printf 'PATCHED_STATUS_END\n'
printf 'BASELINE_STATUS_BEGIN\n'
git -C "$BASELINE_ROOT" status --short 2>/dev/null || true
printf 'BASELINE_STATUS_END\n'
for source in \
    "$A02_ROOT/team/c1_flashkda/challenge_vshard/apply_vshard_patch.py" \
    "$A02_ROOT/team/c1_flashkda/challenge_vshard/vshard.py" \
    "$A02_ROOT/team/c1_flashkda/harness/validate_and_bench.py" \
    "$PATCHED_ROOT/csrc/flash_kda.cpp" \
    "$PATCHED_ROOT/csrc/fwd.h" \
    "$PATCHED_ROOT/csrc/smxx/fwd_launch.cu" \
    "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard.cuh"; do
    if [[ -f "$source" ]]; then
        sha256sum "$source"
    else
        printf 'MISSING_SOURCE=%s\n' "$source"
        exit 92
    fi
done

export PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"
"$PYTHON" -c 'import hashlib,pathlib,torch,flash_kda_C; p=pathlib.Path(flash_kda_C.__file__); print("device=" + torch.cuda.get_device_name(0)); print("extension=" + str(p)); print("extension_sha256=" + hashlib.sha256(p.read_bytes()).hexdigest()); print("has_vshard=" + str(hasattr(flash_kda_C,"fwd_vshard")))'

echo "===== SMALL_TRIPLE_REFERENCE_GATE ====="
small_gate_args=()
if [[ "${C1_REUSE_PRIOR_REFERENCE_GATE:-0}" == "1" ]]; then
    prior_gate="$LOG_DIR/c1_vshard_b300_small_gate.json"
    prior_log="$LOG_DIR/c1_vshard_b300_job4306.log"
    if [[ ! -f "$prior_gate" || ! -f "$prior_log" ]]; then
        echo "MISSING_PRIOR_REFERENCE_GATE=$prior_gate or $prior_log" >&2
        exit 93
    fi
    echo "REUSE_PRIOR_TRIPLE_REFERENCE_GATE=1"
    sha256sum "$prior_gate" "$prior_log"
    small_gate_args+=(--skip-torch-ref --skip-fla-ref)
fi
"$PYTHON" "$HARNESS" --reference-root "$BASELINE_ROOT" --fla-root "$FLA_REF" \
    --T 256 --H 2 --states all --no-bench "${small_gate_args[@]}" \
    --json "$LOG_DIR/c1_vshard_${LABEL}_small_gate.json"

for heads in 96 64; do
    echo "===== OFFICIAL_TIMING_H${heads} ====="
    "$PYTHON" "$HARNESS" --reference-root "$BASELINE_ROOT" --fla-root "$FLA_REF" \
        --T 8192 --H "$heads" --states bf16 --skip-torch-ref --skip-fla-ref \
        --warmup 30 --iters 200 --repeats 5 \
        --json "$LOG_DIR/c1_vshard_${LABEL}_h${heads}.json"
done
