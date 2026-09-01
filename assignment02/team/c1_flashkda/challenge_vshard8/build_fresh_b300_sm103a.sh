#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${C1_VSHARD8_P1_BUILD_AUTHORIZED:-0}" != 1 || "${1:-}" != "--authorized-by-parent" ]]; then
    echo "refusing build: explicit V8-P1 parent authorization required" >&2; exit 64
fi
shift
: "${SOURCE_BASE:?}" "${SOURCE_FRESH:?}" "${A02_ROOT:?}" "${VENV:?}" \
  "${PYTHON_INCLUDE:?}" "${BUILD_LOG:?}" "${PTXAS_JSON:?}" "${SUPPORT_BIN:?}"
PYTHON_BIN="${PYTHON_BIN:-$VENV/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
OWNED="$A02_ROOT/team/c1_flashkda/challenge_vshard8"
P2="$A02_ROOT/team/c1_flashkda/challenge_vshard8_prefetch2"
PARENT="$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2"
GENERATOR="$OWNED/apply_vshard8_patch.py"
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 65
for path in "$SOURCE_BASE/.git" "$SOURCE_BASE/cutlass/include/cutlass/cutlass.h" "$GENERATOR" \
            "$PARENT/provisioning_git_shim.sh" "$OWNED/ptxas_audit.py" "$PYTHON_BIN" \
            "$PYTHON_INCLUDE/Python.h"; do [[ -e "$path" ]] || { echo "missing $path"; exit 66; }; done
[[ ! -e "$SOURCE_FRESH" ]] || { echo "fresh target exists: $SOURCE_FRESH"; exit 67; }
mkdir -p "$SUPPORT_BIN" "$(dirname "$BUILD_LOG")" "$(dirname "$PTXAS_JSON")"
cp "$PARENT/provisioning_git_shim.sh" "$SUPPORT_BIN/git"; chmod 755 "$SUPPORT_BIN/git"
exec > >(tee "$BUILD_LOG") 2>&1
date -Is; hostname; printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
sha256sum "$GENERATOR" "$P2/apply_vshard8_prefetch2_patch.py" \
  "$PARENT/apply_vshard4_prefetch2_patch.py"
git clone --no-hardlinks "$SOURCE_BASE" "$SOURCE_FRESH"
git -C "$SOURCE_FRESH" checkout --detach 1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b
"$PYTHON_BIN" "$GENERATOR" --source "$SOURCE_FRESH" --check-only
"$PYTHON_BIN" "$GENERATOR" --source "$SOURCE_FRESH"
rsync -a --exclude=.git "$SOURCE_BASE/cutlass/" "$SOURCE_FRESH/cutlass/"
cd "$SOURCE_FRESH"
export PATH="$SUPPORT_BIN:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin" CC=/usr/bin/gcc CXX=/usr/bin/g++
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}" FLASH_KDA_CUDA_ARCHS=103a
export NVCC_THREADS="${NVCC_THREADS:-8}" MAX_JOBS="${MAX_JOBS:-4}"
"$PYTHON_BIN" setup.py build_ext --inplace
shopt -s nullglob; extensions=(flash_kda_C.cpython-*-linux-gnu.so)
(( ${#extensions[@]} == 1 )) || exit 68
sha256sum "${extensions[0]}" csrc/flash_kda.cpp csrc/fwd.h csrc/smxx/fwd_launch.cu \
  csrc/smxx/fwd_kernel2_vshard8.cuh csrc/smxx/fwd_kernel2_vshard8_p2.cuh
"$PYTHON_BIN" "$OWNED/ptxas_audit.py" "$BUILD_LOG" --json "$PTXAS_JSON" \
  --require-p1-formal-zero-spill
git status --short; date -Is
