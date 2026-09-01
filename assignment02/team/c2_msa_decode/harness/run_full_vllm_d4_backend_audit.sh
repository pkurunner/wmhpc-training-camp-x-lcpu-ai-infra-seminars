#!/usr/bin/env bash
# Run only inside an already-exclusive B300 allocation; this script submits no job.
# Usage:
#   PYTHON=/path/to/python PYTHONPATH=/path/to/vllm-runtime:/path/to/deps \
#   bash harness/run_full_vllm_d4_backend_audit.sh C2_ROOT VLLM_ROOT MSA_ROOT WHEEL LABEL
set -Eeuo pipefail

if [[ "$#" -ne 5 ]]; then
    echo "usage: run_full_vllm_d4_backend_audit.sh C2_ROOT VLLM_ROOT MSA_ROOT WHEEL LABEL" >&2
    exit 64
fi
C2_ROOT="$1"
VLLM_ROOT="$2"
MSA_ROOT="$3"
WHEEL="$4"
LABEL="$5"
PYTHON="${PYTHON:-python}"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "REFUSE_NO_SLURM_ALLOCATION=1" >&2
    exit 65
fi
JOB_ID="$SLURM_JOB_ID"
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$C2_ROOT:$PYTHONPATH"
else
    export PYTHONPATH="$C2_ROOT"
fi

SCRIPT="$C2_ROOT/harness/full_vllm_d4_backend_bench.py"
AUDIT_SCRIPT="$C2_ROOT/harness/run_full_vllm_d4_backend_audit.sh"
LOG_DIR="$C2_ROOT/experiment_logs/full_vllm_d4_backend"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOG_DIR/c2_full_vllm_d4_backend_$LABEL"_"job$JOB_ID"_"$STAMP.log"
JSON="$LOG_DIR/c2_full_vllm_d4_backend_$LABEL"_"job$JOB_ID"_"$STAMP.json"
mkdir -p "$LOG_DIR"
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
    if [[ -n "$apps" && "$rc" -eq 0 ]]; then
        echo "POST_SHARED_GPU_DETECTED=1"
        rc=91
    fi
    echo "JSON=$JSON"
    echo "LOG=$LOG"
    echo "FINAL_RC=$rc"
    exit "$rc"
}
trap finish EXIT

echo "===== PRE_AUDIT ====="
date -Is
hostname
gpu_lines="$(gpu_query)"
printf '%s\n' "$gpu_lines"
while IFS= read -r gpu_line; do
    if [[ ! "$gpu_line" =~ B300 ]] || [[ ! "$gpu_line" =~ 10\.3 ]]; then
        echo "REFUSE_NON_B300_SM103=1 gpu=$gpu_line"
        exit 89
    fi
done <<< "$gpu_lines"
apps="$(app_query || true)"
echo "PRE_COMPUTE_APPS_BEGIN"
printf '%s\n' "$apps"
echo "PRE_COMPUTE_APPS_END"
if [[ -n "$apps" ]]; then
    echo "REFUSE_SHARED_GPU=1"
    exit 90
fi

echo "===== ENVIRONMENT_AND_PROVENANCE ====="
"$PYTHON" --version
"$PYTHON" - <<'PY'
import importlib.metadata
import torch
import triton
import vllm
print("torch=" + torch.__version__)
print("triton=" + triton.__version__)
print("vllm_version=" + importlib.metadata.version("vllm"))
print("vllm_module=" + vllm.__file__)
PY
git -C "$VLLM_ROOT" rev-parse HEAD
git -C "$MSA_ROOT" rev-parse HEAD
git -C "$VLLM_ROOT" status --porcelain
git -C "$MSA_ROOT" status --porcelain
sha256sum "$WHEEL" "$SCRIPT" "$AUDIT_SCRIPT" "$C2_ROOT/harness/data.py" "$C2_ROOT/harness/reference.py"

echo "===== EXACT_D4_BACKEND_ABBA ====="
"$PYTHON" "$SCRIPT" \
    --c2-root "$C2_ROOT" --vllm-root "$VLLM_ROOT" --msa-root "$MSA_ROOT" --wheel "$WHEEL" \
    --output "$JSON" --warmup 5 --repetitions 50 --seed 20260828

"$PYTHON" - "$JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
if result.get("schema") != "c2-full-vllm-d4-backend-layer-abba-v1":
    raise SystemExit("unexpected JSON schema")
if result.get("boundary") != "exact full-vLLM backend layer path; no model weights/server scheduler":
    raise SystemExit("unexpected benchmark boundary")
if result.get("all_gates_pass") is not True:
    raise SystemExit("all_gates_pass is not true")
for name in ("pin_and_source", "real_builder_and_plan_reuse", "correctness", "dispatch", "timing"):
    if result.get("gates", {}).get(name) is not True:
        raise SystemExit("required gate is not true: " + name)
timing = result.get("timing", {})
if timing.get("warmup") != 5 or timing.get("repetitions") != 50:
    raise SystemExit("unexpected timing protocol")
if len(timing.get("raw_cycles_ms", [])) != 50:
    raise SystemExit("expected 50 ABBA cycles")
for path in ("cutlass", "triton"):
    if timing.get(path, {}).get("sample_count") != 100:
        raise SystemExit("expected 100 raw samples for " + path)
print("JSON_STRICT_GATE=PASS B=16 FP8-E4M3 exact-d4-backend-layer ABBA")
PY
