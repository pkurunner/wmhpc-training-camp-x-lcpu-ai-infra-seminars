#!/usr/bin/env bash
# Fresh, candidate-only SM103a build. It never imports the extension or calls CUDA.
set -Eeuo pipefail

if [[ "${C1_INPUTSTAGES4_BUILD_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing build: set C1_INPUTSTAGES4_BUILD_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${SOURCE_BASE:?set SOURCE_BASE to clean FlashKDA 1ce47ea source}"
: "${SOURCE_FRESH:?set SOURCE_FRESH to a new non-existent candidate destination}"
: "${A02_ROOT:?set A02_ROOT to the assignment02 checkout}"
: "${VENV:?set VENV to the Python virtualenv}"
: "${PYTHON_INCLUDE:?set PYTHON_INCLUDE to the selected Python include directory}"
: "${BUILD_LOG:?set BUILD_LOG to a writable ptxas-preserving build log}"
: "${RESOURCE_LOG:?set RESOURCE_LOG to a writable cuobjdump resource log}"
: "${PTXAS_JSON:?set PTXAS_JSON to the resource-audit JSON output}"
SUPPORT_BIN="${SUPPORT_BIN:?set SUPPORT_BIN to a build-helper directory}"
PYTHON_BIN="${PYTHON_BIN:-$VENV/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PINNED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_inputstages4"
GENERATOR="$OWNED/apply_inputstages4_patch.py"

for required in "$SOURCE_BASE/.git" "$SOURCE_BASE/cutlass/include/cutlass/cutlass.h" "$GENERATOR" \
                "$OWNED/provisioning_git_shim.sh" "$OWNED/ptxas_audit.py" "$PYTHON_BIN" \
                "$PYTHON_INCLUDE/Python.h" "$CUDA_HOME/bin/cuobjdump"; do
    [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 66; }
done
[[ ! -e "$SOURCE_FRESH" ]] || { echo "fresh target already exists: $SOURCE_FRESH" >&2; exit 67; }
mkdir -p "$SUPPORT_BIN" "$(dirname "$BUILD_LOG")" "$(dirname "$RESOURCE_LOG")" "$(dirname "$PTXAS_JSON")"
cp "$OWNED/provisioning_git_shim.sh" "$SUPPORT_BIN/git"
chmod 755 "$SUPPORT_BIN/git"
exec > >(tee "$BUILD_LOG") 2>&1

date -Is
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-none}"
sha256sum "$GENERATOR" "$OWNED/ptxas_audit.py" "$OWNED/build_fresh_b300_sm103a.sh" \
  "$A02_ROOT/team/c1_flashkda/challenge_vshard/apply_vshard_patch.py" \
  "$A02_ROOT/team/c1_flashkda/challenge_vshard4/apply_vshard4_patch.py" \
  "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/apply_prefetch2_patch.py"
git -C "$SOURCE_BASE" rev-parse HEAD
git -C "$SOURCE_BASE" submodule status cutlass
git clone --no-hardlinks "$SOURCE_BASE" "$SOURCE_FRESH"
git -C "$SOURCE_FRESH" checkout --detach "$PINNED_COMMIT"
git -C "$SOURCE_FRESH" status --short
git -C "$SOURCE_FRESH" submodule status cutlass
"$PYTHON_BIN" "$GENERATOR" --source "$SOURCE_FRESH" --check-only
"$PYTHON_BIN" "$GENERATOR" --source "$SOURCE_FRESH"
rsync -a --exclude=.git "$SOURCE_BASE/cutlass/" "$SOURCE_FRESH/cutlass/"

cd "$SOURCE_FRESH"
export PATH="$SUPPORT_BIN:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
export FLASH_KDA_CUDA_ARCHS=103a
export NVCC_THREADS="${NVCC_THREADS:-8}"
export MAX_JOBS="${MAX_JOBS:-4}"
"$PYTHON_BIN" setup.py build_ext --inplace
shopt -s nullglob
extensions=(flash_kda_C.cpython-*-linux-gnu.so)
(( ${#extensions[@]} == 1 )) || { echo "expected exactly one extension, found ${#extensions[@]}" >&2; exit 68; }
"$CUDA_HOME/bin/cuobjdump" --dump-resource-usage "${extensions[0]}" | tee "$RESOURCE_LOG"
sha256sum "${extensions[0]}" csrc/flash_kda.cpp csrc/fwd.h csrc/smxx/fwd_launch.cu \
  csrc/smxx/fwd_kernel2_vshard.cuh csrc/smxx/fwd_kernel2_vshard_p2.cuh \
  csrc/smxx/fwd_kernel2_vshard4.cuh csrc/smxx/fwd_kernel2_vshard4_p2.cuh \
  csrc/smxx/fwd_kernel2_vshard4_p2s4.cuh "$RESOURCE_LOG"
"$PYTHON_BIN" "$OWNED/ptxas_audit.py" "$BUILD_LOG" --resource-log "$RESOURCE_LOG" --json "$PTXAS_JSON" \
  --require-p2s4-bf16-fixed-zero-spill --require-p2s4-shared-memory-evidence
git status --short
date -Is
