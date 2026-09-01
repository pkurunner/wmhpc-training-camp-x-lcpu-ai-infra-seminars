#!/usr/bin/env bash
# Reproducible CPU-side fresh build for the authorized 5090 SM120a fallback.
set -Eeuo pipefail

SOURCE_BASE="${SOURCE_BASE:-/home/lcpu/85117379/flashkda-1ce47ea}"
SOURCE_FRESH="${SOURCE_FRESH:-/home/lcpu/85117379/flashkda-prefetch2-1ce47ea-5090-r1}"
SUPPORT_ROOT="${SUPPORT_ROOT:-/home/lcpu/85117379/c1-prefetch2-support-5090-r1}"
CUTLASS_ARCHIVE="${CUTLASS_ARCHIVE:-/home/lcpu/85117379/cutlass-5c149f5-local.tar.gz}"
VENV="${VENV:-/home/lcpu/85117379/c1env-cu130}"
BUILD_LOG="${BUILD_LOG:-/home/lcpu/85117379/c1_prefetch2_build_5090_r1.log}"
PINNED_COMMIT="1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"

for required in "$SOURCE_BASE/.git" "$SUPPORT_ROOT/assignment02/team/c1_flashkda/challenge_prefetch2/apply_prefetch2_patch.py" \
                "$SUPPORT_ROOT/bin/git" "$CUTLASS_ARCHIVE" "$VENV/bin/python"; do
    [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 66; }
done
[[ ! -e "$SOURCE_FRESH" ]] || { echo "fresh target already exists: $SOURCE_FRESH" >&2; exit 67; }

exec > >(tee "$BUILD_LOG") 2>&1
date -Is
hostname
sha256sum "$CUTLASS_ARCHIVE" "$SUPPORT_ROOT/assignment02/team/c1_flashkda/challenge_prefetch2/apply_prefetch2_patch.py"
git clone --no-hardlinks "$SOURCE_BASE" "$SOURCE_FRESH"
git -C "$SOURCE_FRESH" checkout --detach "$PINNED_COMMIT"
git -C "$SOURCE_FRESH" status --short
git -C "$SOURCE_FRESH" submodule status cutlass

GENERATOR="$SUPPORT_ROOT/assignment02/team/c1_flashkda/challenge_prefetch2/apply_prefetch2_patch.py"
"$VENV/bin/python" "$GENERATOR" --source "$SOURCE_FRESH" --check-only
"$VENV/bin/python" "$GENERATOR" --source "$SOURCE_FRESH"
mkdir -p "$SOURCE_FRESH/cutlass"
tar -xzf "$CUTLASS_ARCHIVE" --strip-components=1 -C "$SOURCE_FRESH/cutlass"

cd "$SOURCE_FRESH"
export PATH="$SUPPORT_ROOT/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FLASH_KDA_CUDA_ARCHS=120a
export NVCC_THREADS=8
export MAX_JOBS=4
"$VENV/bin/python" setup.py build_ext --inplace
test -f flash_kda_C.cpython-312-x86_64-linux-gnu.so
sha256sum flash_kda_C.cpython-312-x86_64-linux-gnu.so \
  csrc/smxx/fwd_kernel2_vshard.cuh csrc/smxx/fwd_kernel2_vshard_p2.cuh
git status --short
date -Is
