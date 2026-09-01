#!/usr/bin/env bash
# Reuse the strict same-BF16-query production-vLLM backend gate against the
# reviewed four-producer AOT artifact.  The backend base script keeps its
# profiler cardinality and output-pointer checks unchanged.
#SBATCH --job-name=c2-native-v2-4p-full
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-v2-four-producer-full-backend-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail

: "${C2_EXPECTED_LIB_SHA:?submit with the reviewed v2+4P AOT SHA-256}"
[[ "${C2_EXPECTED_LIB_SHA}" =~ ^[0-9a-f]{64}$ ]]
export C2_EXPECTED_SOURCE_SHA=a46830564aa62e30e26de67dc98882043a239985f6d1c7e1b2dc2a03630fd8a1
export C2_FULL_ARTIFACT_DIR=/home/lcpu/85117379/c2-native-v2-four-producer-full-backend-artifacts-20260829
export C2_NATIVE_SOURCE=/home/lcpu/85117379/c2-native-v2-four-producer-build-artifacts-20260829/native_c2_decode.v2-four-producer.cu

# The derived checkout's import symlink intentionally continues to reference
# BUILD_DIR.  The four-producer build writes that path, and the independently
# copied source above plus C2_EXPECTED_LIB_SHA pin the tested build identity.
export C2_LIB=${C2_LIB:-/home/lcpu/85117379/vllm-d4-native-c2-build-20260829/_C_stable_libtorch.abi3.so}
[[ -f "${C2_LIB}" && -f "${C2_NATIVE_SOURCE}" ]]
[[ "$(sha256sum "${C2_LIB}" | awk '{print $1}')" == "${C2_EXPECTED_LIB_SHA}" ]]

exec bash /home/lcpu/85117379/validate_native_c2_full_backend_20260829.sh
