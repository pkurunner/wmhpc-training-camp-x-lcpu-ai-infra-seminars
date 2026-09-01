#!/usr/bin/env bash
# CPU-side fresh build inside the already allocated B300 job; never imports/runs the extension.
set -Eeuo pipefail

SOURCE_BASE="${SOURCE_BASE:-/home/lcpu/85117379/flashkda-1ce47ea}"
SOURCE_FRESH="${SOURCE_FRESH:-/home/lcpu/85117379/flashkda-prefetch2-1ce47ea-b300-r1}"
A02_ROOT="${A02_ROOT:-/home/lcpu/85117379/codex-a02-20260819-main/assignment02}"
VENV="${VENV:-$A02_ROOT/.venv}"
PYTHON_INCLUDE="${PYTHON_INCLUDE:-/home/lcpu/85117379/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/include/python3.12}"
SUPPORT_BIN="${SUPPORT_BIN:-/home/lcpu/85117379/c1-prefetch2-support-b300-r1/bin}"
BUILD_LOG="${BUILD_LOG:-/home/lcpu/85117379/c1_prefetch2_build_b300_r1.log}"
PINNED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "must run inside an existing Slurm allocation" >&2; exit 65; }
for required in "$SOURCE_BASE/.git" "$SOURCE_BASE/cutlass/include/cutlass/cutlass.h" \
                "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/apply_prefetch2_patch.py" \
                "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/provisioning_git_shim.sh" \
                "$VENV/bin/python" "$PYTHON_INCLUDE/Python.h"; do
    [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 66; }
done
[[ ! -e "$SOURCE_FRESH" ]] || { echo "fresh target already exists: $SOURCE_FRESH" >&2; exit 67; }

mkdir -p "$SUPPORT_BIN"
cp "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/provisioning_git_shim.sh" "$SUPPORT_BIN/git"
chmod 755 "$SUPPORT_BIN/git"
exec > >(tee "$BUILD_LOG") 2>&1
date -Is
hostname
printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
git -C "$SOURCE_BASE/cutlass" rev-parse HEAD
sha256sum "$SOURCE_BASE/cutlass/include/cutlass/cutlass.h" \
  "$A02_ROOT/team/c1_flashkda/challenge_prefetch2/apply_prefetch2_patch.py"

git clone --no-hardlinks "$SOURCE_BASE" "$SOURCE_FRESH"
git -C "$SOURCE_FRESH" checkout --detach "$PINNED_COMMIT"
git -C "$SOURCE_FRESH" status --short
git -C "$SOURCE_FRESH" submodule status cutlass
GENERATOR="$A02_ROOT/team/c1_flashkda/challenge_prefetch2/apply_prefetch2_patch.py"
"$VENV/bin/python" "$GENERATOR" --source "$SOURCE_FRESH" --check-only
"$VENV/bin/python" "$GENERATOR" --source "$SOURCE_FRESH"
mkdir -p "$SOURCE_FRESH/cutlass"
rsync -a --exclude=.git "$SOURCE_BASE/cutlass/" "$SOURCE_FRESH/cutlass/"

cd "$SOURCE_FRESH"
export PATH="$SUPPORT_BIN:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
export FLASH_KDA_CUDA_ARCHS=103a
export NVCC_THREADS=8
export MAX_JOBS=4
"$VENV/bin/python" setup.py build_ext --inplace
test -f flash_kda_C.cpython-312-x86_64-linux-gnu.so
sha256sum flash_kda_C.cpython-312-x86_64-linux-gnu.so \
  csrc/smxx/fwd_kernel2_vshard.cuh csrc/smxx/fwd_kernel2_vshard_p2.cuh
git status --short
date -Is
