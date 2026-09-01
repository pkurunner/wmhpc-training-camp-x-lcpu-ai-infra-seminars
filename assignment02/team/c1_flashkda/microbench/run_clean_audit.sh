#!/usr/bin/env bash
# Run only inside the parent's already-granted clean B300 allocation.
# This script does not submit, reserve, or hold any Slurm resource itself.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSIGNMENT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_JSON="${OUT_JSON:-${ASSIGNMENT_DIR}/team/c1_flashkda/experiment_logs/c1_chunk_tcgen_microbench_b300.json}"
TCGEN05_BIN="${TCGEN05_BIN:-}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
LOG_PATH="${LOG_PATH:-${ASSIGNMENT_DIR}/team/c1_flashkda/experiment_logs/c1_chunk_tcgen_microbench_b300_job${JOB_TAG}.log}"

mkdir -p "$(dirname "${OUT_JSON}")" "$(dirname "${LOG_PATH}")"
# Keep the exact PRE/POST audit and terminal status beside the JSON.  The
# process substitution is Bash-only by design: B300 runs this script under
# Bash, whereas the development host may be Windows without a local Bash.
exec > >(tee "${LOG_PATH}") 2>&1

if [[ -z "${TCGEN05_BIN}" ]]; then
  CANDIDATE="${ASSIGNMENT_DIR}/cuda/bin/m3_tcgen05/02_single_tile"
  if [[ -x "${CANDIDATE}" ]]; then
    TCGEN05_BIN="${CANDIDATE}"
  fi
fi

audit_gpu() {
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used \
    --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
}

finish() {
  local final_rc=$?
  local post_apps
  # Never let an audit command itself hide the benchmark's original error.
  set +e
  printf 'C1_MICROBENCH_POST\n'
  audit_gpu
  post_apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true)"
  if [[ -n "${post_apps//[[:space:]]/}" ]]; then
    printf 'FAIL: GPU has compute applications after C1 microbench:\n%s\n' "${post_apps}" >&2
    final_rc=4
  fi
  printf 'C1_MICROBENCH_FINAL_RC=%d\n' "${final_rc}"
  printf 'C1_MICROBENCH_JSON=%s\n' "${OUT_JSON}"
  printf 'C1_MICROBENCH_LOG=%s\n' "${LOG_PATH}"
  trap - EXIT
  exit "${final_rc}"
}
trap finish EXIT

before_apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true)"
if [[ -n "${before_apps//[[:space:]]/}" ]]; then
  printf 'REFUSE: GPU already has compute applications before C1 microbench:\n%s\n' "${before_apps}" >&2
  exit 3
fi

printf 'C1_MICROBENCH_PRE\n'
audit_gpu

args=(
  "${SCRIPT_DIR}/c1_chunk_tcgen_microbench.py"
  --json "${OUT_JSON}"
  --target-matrix-mib "${TARGET_MATRIX_MIB:-128}"
  --warmup "${WARMUP:-10}"
  --iters "${ITERS:-20}"
  --repeats "${REPEATS:-5}"
)
if [[ -n "${TCGEN05_BIN}" ]]; then
  args+=(--tcgen05-bin "${TCGEN05_BIN}")
fi
"${PYTHON_BIN}" "${args[@]}"
