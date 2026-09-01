#!/usr/bin/env bash
# Fresh CPU-side SM103a build; requires an explicit parent-authorized allocation.
set -Eeuo pipefail

if [[ "${C1_VSHARD8_P2_BUILD_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing build: set C1_VSHARD8_P2_BUILD_AUTHORIZED=1 and pass --authorized-by-parent" >&2
    exit 64
fi
shift

: "${SOURCE_BASE:?set SOURCE_BASE to clean-source clone origin}"
: "${SOURCE_FRESH:?set SOURCE_FRESH to a new non-existent destination}"
: "${A02_ROOT:?set A02_ROOT to assignment02 checkout}"
: "${VENV:?set VENV to Python virtualenv}"
: "${PYTHON_INCLUDE:?set PYTHON_INCLUDE}"
: "${BUILD_LOG:?set BUILD_LOG}"
: "${PTXAS_JSON:?set PTXAS_JSON}"
SUPPORT_BIN="${SUPPORT_BIN:?set SUPPORT_BIN}"
PYTHON_BIN="${PYTHON_BIN:-$VENV/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PINNED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_vshard8_prefetch2"
PARENT="$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2"
GENERATOR="$OWNED/apply_vshard8_prefetch2_patch.py"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "must run inside an existing Slurm allocation" >&2; exit 65; }
for required in "$SOURCE_BASE/.git" "$SOURCE_BASE/cutlass/include/cutlass/cutlass.h" \
                "$GENERATOR" "$PARENT/provisioning_git_shim.sh" "$OWNED/ptxas_audit.py" \
                "$PYTHON_BIN" "$PYTHON_INCLUDE/Python.h"; do
    [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 66; }
done
[[ ! -e "$SOURCE_FRESH" ]] || { echo "fresh target already exists: $SOURCE_FRESH" >&2; exit 67; }
mkdir -p "$SUPPORT_BIN" "$(dirname "$BUILD_LOG")" "$(dirname "$PTXAS_JSON")"
cp "$PARENT/provisioning_git_shim.sh" "$SUPPORT_BIN/git"
chmod 755 "$SUPPORT_BIN/git"
exec > >(tee "$BUILD_LOG") 2>&1

date -Is
hostname
printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
sha256sum "$GENERATOR" "$PARENT/apply_vshard4_prefetch2_patch.py" \
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
sha256sum "${extensions[0]}" csrc/flash_kda.cpp csrc/fwd.h csrc/smxx/fwd_launch.cu \
  csrc/smxx/fwd_kernel2_vshard.cuh csrc/smxx/fwd_kernel2_vshard_p2.cuh \
  csrc/smxx/fwd_kernel2_vshard4.cuh csrc/smxx/fwd_kernel2_vshard4_p2.cuh \
  csrc/smxx/fwd_kernel2_vshard8_p2.cuh
"$PYTHON_BIN" "$OWNED/ptxas_audit.py" "$BUILD_LOG" --json "$PTXAS_JSON" \
  --require-formal-zero-spill
git status --short
date -Is
