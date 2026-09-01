#!/usr/bin/env bash
# One authorized Slurm job: fresh build, ptxas gate, clean GPU exact/perf audit.
set -Eeuo pipefail

if [[ "${C1_VSHARD8_FULL_AUTHORIZED:-0}" != 1 ]]; then
    echo "refusing combined run: set C1_VSHARD8_FULL_AUTHORIZED=1" >&2
    exit 64
fi
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "must run in Slurm" >&2; exit 65; }

ACCOUNT_ROOT="${ACCOUNT_ROOT:-/home/lcpu/85117379}"
A02_ROOT="${A02_ROOT:-$ACCOUNT_ROOT/codex-a02-20260819-main/assignment02}"
SOURCE_BASE="${SOURCE_BASE:-$ACCOUNT_ROOT/flashkda-1ce47ea}"
SOURCE_FRESH="${SOURCE_FRESH:-$ACCOUNT_ROOT/flashkda-vshard8-prefetch2-1ce47ea-b300-r1}"
REFERENCE_ROOT="${REFERENCE_ROOT:-$SOURCE_BASE}"
VENV="${VENV:-$A02_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV/bin/python}"
PYTHON_INCLUDE="${PYTHON_INCLUDE:-$ACCOUNT_ROOT/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/include/python3.12}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_vshard8_prefetch2"
RESULTS_DIR="${RESULTS_DIR:-$OWNED/results}"
BUILD_LOG="$RESULTS_DIR/c1_vshard8_p2_build_b300_r1_job${SLURM_JOB_ID}.log"
PTXAS_JSON="$RESULTS_DIR/c1_vshard8_p2_ptxas_b300_r1_job${SLURM_JOB_ID}.json"
SUPPORT_BIN="/tmp/c1-vshard8-build-${SLURM_JOB_ID}/bin"

mkdir -p "$RESULTS_DIR"
export SOURCE_BASE SOURCE_FRESH A02_ROOT VENV PYTHON_BIN PYTHON_INCLUDE BUILD_LOG PTXAS_JSON SUPPORT_BIN
export C1_VSHARD8_P2_BUILD_AUTHORIZED=1
bash "$OWNED/build_fresh_b300_sm103a.sh" --authorized-by-parent

export PATCHED_ROOT="$SOURCE_FRESH" REFERENCE_ROOT RESULTS_DIR
export LABEL="b300_sm103a_h12_r1"
export C1_VSHARD8_P2_GPU_AUTHORIZED=1
bash "$OWNED/run_clean_vshard8_audit.sh" --authorized-by-parent
