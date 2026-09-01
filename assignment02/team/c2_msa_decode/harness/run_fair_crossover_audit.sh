#!/usr/bin/env bash
# Usage in an exclusive B300 allocation:
# bash harness/run_fair_crossover_audit.sh "$PWD" /home/lcpu/85117379/msa-official b300
set -Eeuo pipefail

C2_ROOT="${1:?usage: run_fair_crossover_audit.sh C2_ROOT MSA_ROOT LABEL}"
MSA_ROOT="${2:?usage: run_fair_crossover_audit.sh C2_ROOT MSA_ROOT LABEL}"
LABEL="${3:?usage: run_fair_crossover_audit.sh C2_ROOT MSA_ROOT LABEL}"
PYTHON="${PYTHON:-python}"
export PATH="$(dirname "$PYTHON"):$PATH"
SCRIPT="$C2_ROOT/harness/fair_crossover_bench.py"
AUDIT_SCRIPT="$C2_ROOT/harness/run_fair_crossover_audit.sh"
LOG_DIR="$C2_ROOT/experiment_logs"
LOG="$LOG_DIR/c2_fair_bf16_crossover_${LABEL}_job${SLURM_JOB_ID:-none}.log"
JSON="$LOG_DIR/c2_fair_bf16_crossover_${LABEL}_job${SLURM_JOB_ID:-none}.json"
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
if [[ -n "$apps" ]]; then
    echo "REFUSE_SHARED_GPU=1"
    exit 90
fi

echo "===== PIN_AND_SOURCE_AUDIT ====="
"$PYTHON" --version
"$PYTHON" -c 'import torch, triton, fmha_sm100; print("torch=" + torch.__version__); print("triton=" + triton.__version__); print("fmha_sm100=" + fmha_sm100.__file__)'
git -C "$MSA_ROOT" rev-parse HEAD
git -C "$MSA_ROOT" submodule status --recursive
sha256sum "$SCRIPT" "$AUDIT_SCRIPT" "$C2_ROOT/harness/data.py" \
    "$C2_ROOT/harness/triton_baseline.py" "$C2_ROOT/harness/reference.py" \
    "$C2_ROOT/challenge/prepared_decode.py" \
    "$C2_ROOT/vllm_msa_ref/sparse_attn.py" "$C2_ROOT/vllm_msa_ref/msa_cutlass_sparse_decode.py"

echo "===== SAME_DATA_BF16_FAIR_CROSSOVER ====="
PYTHONPATH="$MSA_ROOT/python:$C2_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$SCRIPT" --c2-root "$C2_ROOT" --msa-root "$MSA_ROOT" \
    --seed 20260819 --max-seq-len 4096 --warmup 20 --repetitions 100 \
    --batches 1 4 8 16 | tee "$JSON"
"$PYTHON" - "$JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
if result.get("schema") != "c2-fair-bf16-crossover-v2-three-path":
    raise SystemExit("unexpected fair crossover schema")
if result.get("all_correctness_pass") is not True:
    raise SystemExit("fair crossover correctness gate is not true")
rows = result.get("results", [])
if [row.get("batch") for row in rows] != [1, 4, 8, 16]:
    raise SystemExit("fair crossover JSON does not contain exact B=1/4/8/16 rows")
if not all(row.get("latency", {}).get("valid") is True for row in rows):
    raise SystemExit("fair crossover contains an invalid timing row")
print("JSON_STRICT_GATE=PASS batches=1,4,8,16 paths=source-wrapper,prepared-c1,official")
PY
echo "JSON=$JSON"
echo "LOG=$LOG"
