#!/usr/bin/env bash
#SBATCH --job-name=c2-native-aot
#SBATCH --partition=cpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-aot-artifacts-20260829/slurm-%j.log

set -euo pipefail

SRC=/home/lcpu/85117379/vllm-d4-native-c2-20260829
BUILD_DIR=/home/lcpu/85117379/vllm-d4-native-c2-build-20260829
ARTIFACT_DIR=/home/lcpu/85117379/c2-native-aot-artifacts-20260829
TOOLS=/home/lcpu/85117379/c2-native-tools-20260829
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
CUTLASS=/home/lcpu/85117379/cutlass-v4.4.2-c2-20260829
MSA=/home/lcpu/85117379/msa-official
DEPS=/home/lcpu/85117379/c2-native-aot-deps-20260829

mkdir -p "${BUILD_DIR}" "${ARTIFACT_DIR}" "${DEPS}"
exec > >(tee "${ARTIFACT_DIR}/build-driver-job${SLURM_JOB_ID}.log") 2>&1

export PATH="${TOOLS}/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin"
export TORCH_CUDA_ARCH_LIST=10.3f
export VLLM_CUTLASS_SRC_DIR="${CUTLASS}"
export FMHA_SM100_SRC_DIR="${MSA}"

printf 'UTC=%s\n' "$(date -u +%FT%TZ)"
printf 'SLURM_JOB_ID=%s HOST=%s CPUS=%s\n' \
  "${SLURM_JOB_ID}" "$(hostname)" "${SLURM_CPUS_PER_TASK}"
printf 'SRC=%s\nBUILD_DIR=%s\nCUTLASS=%s\nMSA=%s\n' \
  "${SRC}" "${BUILD_DIR}" "${CUTLASS}" "${MSA}"
git -C "${SRC}" rev-parse HEAD
git -C "${SRC}" status --short
git -C "${CUTLASS}" rev-parse HEAD
git -C "${MSA}" rev-parse HEAD
git -C "${MSA}/python/fmha_sm100/cutlass" rev-parse HEAD
"${PYTHON}" -c 'import torch; print("TORCH", torch.__version__, torch.version.cuda)'
cmake --version
ninja --version
nvcc --version

cmake -S "${SRC}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE="${PYTHON}" \
  -DFETCHCONTENT_BASE_DIR="${DEPS}" \
  -DCMAKE_INSTALL_PREFIX="${ARTIFACT_DIR}/install" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DNVCC_THREADS=2 \
  -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
  -DCMAKE_JOB_POOLS:STRING=compile=32 \
  2>&1 | tee "${ARTIFACT_DIR}/configure-job${SLURM_JOB_ID}.log"

cmake --build "${BUILD_DIR}" --target _C_stable_libtorch -- -j64 \
  2>&1 | tee "${ARTIFACT_DIR}/compile-job${SLURM_JOB_ID}.log"

find "${BUILD_DIR}" -type f \
  \( -name '*_C_stable_libtorch*.so' -o -name '*_C_stable_libtorch*.o' \) \
  -print -exec sha256sum {} \; | tee "${ARTIFACT_DIR}/products-job${SLURM_JOB_ID}.log"
git -C "${SRC}" diff --check
git -C "${SRC}" diff --binary > "${ARTIFACT_DIR}/native-c2-job${SLURM_JOB_ID}.patch"
sha256sum "${ARTIFACT_DIR}"/*job"${SLURM_JOB_ID}"* \
  > "${ARTIFACT_DIR}/build-job${SLURM_JOB_ID}.sha256"
printf 'FINAL_RC=0 UTC=%s\n' "$(date -u +%FT%TZ)"
