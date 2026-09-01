#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "${C1_VSHARD8_P1_GPU_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing GPU run: explicit V8-P1 parent authorization required" >&2; exit 64
fi
shift
: "${A02_ROOT:?}" "${PATCHED_ROOT:?}" "${REFERENCE_ROOT:?}" "${LABEL:?}" "${PYTHON_INCLUDE:?}"
PYTHON_BIN="${PYTHON_BIN:-python}"; CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_vshard8"
P2="$A02_ROOT/team/c1_flashkda/challenge_vshard8_prefetch2"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"; mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/c1_vshard8_p1_${LABEL}_job${SLURM_JOB_ID:-none}.log"; exec > >(tee "$LOG") 2>&1
gpu_query() { nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,compute_cap,driver_version,memory.used,memory.total --format=csv,noheader; }
app_query() { nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader; }
memory_query() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
require_clean() { local s="$1" a u; a="$(app_query)" || return 92; u="$(memory_query)" || return 92; echo "${s}_COMPUTE_APPS_BEGIN"; printf '%s\n' "$a"; echo "${s}_COMPUTE_APPS_END"; echo "${s}_MEMORY_USED_MIB=$u"; [[ -z "$a" && "$u" =~ ^[[:space:]]*0[[:space:]]*$ ]]; }
finish() { local rc=$?; trap - EXIT; echo '===== POST_AUDIT ====='; date -Is; gpu_query || rc=92; require_clean POST || rc=91; echo "FINAL_RC=$rc"; exit "$rc"; }
trap finish EXIT
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 89
echo '===== PRE_AUDIT ====='; date -Is; hostname; gpu_query; require_clean PRE || exit 90
echo '===== SOURCE_IDENTITY ====='; git -C "$PATCHED_ROOT" rev-parse HEAD; git -C "$PATCHED_ROOT" status --short
sha256sum "$OWNED/apply_vshard8_patch.py" "$OWNED/vshard8.py" "$OWNED/ptxas_audit.py" \
  "$P2/run_vshard8_final.py" "$PATCHED_ROOT"/flash_kda_C.cpython-*-linux-gnu.so \
  "$PATCHED_ROOT/csrc/smxx/fwd_kernel2_vshard8.cuh"
export CUDA_HOME PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}" PYTHONPATH="$PATCHED_ROOT:$A02_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PATCHED_ROOT"; RUNNER="$P2/run_vshard8_final.py"
echo '===== SMALL_ALL_CONTRACT_TORCH_REF_EXACT ====='
"$PYTHON_BIN" "$RUNNER" --candidate vshard8_p1 --reference-root "$REFERENCE_ROOT" --T 256 \
  --heads 1,2,4 --contracts none,bf16_both,fp32_both,fp32_final_only --torch-ref --no-bench \
  --json "$RESULTS_DIR/c1_vshard8_p1_${LABEL}_small_all_contracts.json"
require_clean BETWEEN_SMALL_AND_H12 || exit 93
echo '===== H12_ALL_CONTRACT_EXACT_AND_CYCLIC ====='
"$PYTHON_BIN" "$RUNNER" --candidate vshard8_p1 --reference-root "$REFERENCE_ROOT" --T 8192 \
  --heads 12 --contracts none,bf16_both,fp32_both,fp32_final_only --warmup 30 --samples 1000 \
  --json "$RESULTS_DIR/c1_vshard8_p1_${LABEL}_h12_all_contracts.json"
require_clean AFTER_H12 || exit 94
