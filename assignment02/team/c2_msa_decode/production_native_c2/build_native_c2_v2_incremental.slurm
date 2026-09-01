#!/usr/bin/env bash
# Incremental, provenance-gated build of the warp-owned PV candidate.  Submit
# only after the v1 overlay wheel and isolated installed-wheel validation have
# completed; the v1 binary is copied to the independent artifact directory
# before the derived checkout/build tree advances to v2.
#SBATCH --job-name=c2-native-v2-build
#SBATCH --partition=cpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --output=/home/lcpu/85117379/c2-native-v2-build-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail

SRC=/home/lcpu/85117379/vllm-d4-native-c2-20260829
SOURCE=${SRC}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
BUILD_DIR=/home/lcpu/85117379/vllm-d4-native-c2-build-20260829
LIB=${BUILD_DIR}/_C_stable_libtorch.abi3.so
PATCH=/home/lcpu/85117379/native_c2_v2_warp_owned_pv_20260829.patch
ARTIFACT_DIR=/home/lcpu/85117379/c2-native-v2-build-artifacts-20260829
TOOLS=/home/lcpu/85117379/c2-native-tools-20260829
EXPECTED_HEAD=d4da0c55af3aa231b6209bf77871f3ed36eab0d2
EXPECTED_V1_SOURCE_SHA=f67495586759b3b8f718dbbf60627c499f917842f2ed6ecf89d2344f1c82bc4b
EXPECTED_V1_LIB_SHA=beed8c557858da9bdd9e5f4c3681a04c29c8435dd92994f43c9935ce715ed064
EXPECTED_PATCH_SHA=2fa34736ba80a50cd6d4a40377ad419709941349eff5c26e745541cfd8862816
EXPECTED_V2_SOURCE_SHA=6379cb0c71aa79610dd807b40982a6653637d926c1e3dab78155f1f3a5a4f6ce
EXPECTED_V1_TRACKED_DIFF_SHA=${C2_EXPECTED_V1_TRACKED_DIFF_SHA:?submit with reviewed v1 tracked-diff SHA-256}
EXPECTED_V1_STATUS_SHA=${C2_EXPECTED_V1_STATUS_SHA:?submit with reviewed v1 porcelain-status SHA-256}
FROZEN_PATCH=${ARTIFACT_DIR}/native_c2_v2.input-job${SLURM_JOB_ID}.patch

mkdir -p "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/build-driver-job${SLURM_JOB_ID}.log") 2>&1
trap 'rc=$?; printf "FINAL_RC=%s UTC=%s\n" "${rc}" "$(date -u +%FT%TZ)"' EXIT
export PATH="${TOOLS}/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin"

printf 'UTC=%s JOB=%s HOST=%s\n' "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}" "$(hostname)"
[[ "$(git -C "${SRC}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]]
[[ -f "${SOURCE}" && -f "${LIB}" && -f "${PATCH}" ]]
command -v patch
[[ "$(sha256sum "${SOURCE}" | awk '{print $1}')" == "${EXPECTED_V1_SOURCE_SHA}" ]]
[[ "$(sha256sum "${LIB}" | awk '{print $1}')" == "${EXPECTED_V1_LIB_SHA}" ]]
[[ "$(sha256sum "${PATCH}" | awk '{print $1}')" == "${EXPECTED_PATCH_SHA}" ]]
git -C "${SRC}" diff --binary --no-ext-diff --no-textconv HEAD \
  > "${ARTIFACT_DIR}/v1-tracked-diff-job${SLURM_JOB_ID}.patch"
git -C "${SRC}" status --porcelain=v1 --untracked-files=all -z \
  > "${ARTIFACT_DIR}/v1-status-job${SLURM_JOB_ID}.bin"
[[ "$(sha256sum "${ARTIFACT_DIR}/v1-tracked-diff-job${SLURM_JOB_ID}.patch" | awk '{print $1}')" == "${EXPECTED_V1_TRACKED_DIFF_SHA}" ]]
[[ "$(sha256sum "${ARTIFACT_DIR}/v1-status-job${SLURM_JOB_ID}.bin" | awk '{print $1}')" == "${EXPECTED_V1_STATUS_SHA}" ]]
cp --preserve=mode,timestamps "${PATCH}" "${FROZEN_PATCH}"
[[ "$(sha256sum "${FROZEN_PATCH}" | awk '{print $1}')" == "${EXPECTED_PATCH_SHA}" ]]
patch --dry-run --batch --forward -p1 -d "${SRC}" < "${FROZEN_PATCH}" \
  | tee "${ARTIFACT_DIR}/patch-dry-run-job${SLURM_JOB_ID}.log"

cp --reflink=auto --preserve=mode,timestamps "${LIB}" \
  "${ARTIFACT_DIR}/_C_stable_libtorch.v1.abi3.so"
cp --preserve=mode,timestamps "${SOURCE}" \
  "${ARTIFACT_DIR}/native_c2_decode.v1.cu"
sha256sum "${SOURCE}" "${LIB}" "${PATCH}" \
  | tee "${ARTIFACT_DIR}/v1-inputs-job${SLURM_JOB_ID}.sha256"

patch --batch --forward -p1 -d "${SRC}" < "${FROZEN_PATCH}" \
  | tee "${ARTIFACT_DIR}/patch-apply-job${SLURM_JOB_ID}.log"
git -C "${SRC}" diff --check
[[ "$(sha256sum "${SOURCE}" | awk '{print $1}')" == "${EXPECTED_V2_SOURCE_SHA}" ]]
sha256sum "${SOURCE}" \
  | tee "${ARTIFACT_DIR}/v2-source-job${SLURM_JOB_ID}.sha256"
cp --preserve=mode,timestamps "${SOURCE}" \
  "${ARTIFACT_DIR}/native_c2_decode.v2-warp-owned-pv.cu"
if git diff --no-index --binary -- \
    "${ARTIFACT_DIR}/native_c2_decode.v1.cu" \
    "${ARTIFACT_DIR}/native_c2_decode.v2-warp-owned-pv.cu" \
    > "${ARTIFACT_DIR}/native-c2-v1-to-v2-job${SLURM_JOB_ID}.patch"; then
  echo "v1 and v2 native sources unexpectedly identical" >&2
  exit 1
else
  rc=$?
  [[ "${rc}" -eq 1 ]]
fi
git -C "${SRC}" diff --binary --no-ext-diff --no-textconv HEAD \
  > "${ARTIFACT_DIR}/derived-v2-job${SLURM_JOB_ID}.patch"

cmake --build "${BUILD_DIR}" --target _C_stable_libtorch -- -j64 \
  2>&1 | tee "${ARTIFACT_DIR}/compile-v2-job${SLURM_JOB_ID}.log"

cp --reflink=auto --preserve=mode,timestamps "${LIB}" \
  "${ARTIFACT_DIR}/_C_stable_libtorch.v2-warp-owned-pv.abi3.so"
sha256sum "${ARTIFACT_DIR}/_C_stable_libtorch.v1.abi3.so" \
  "${ARTIFACT_DIR}/_C_stable_libtorch.v2-warp-owned-pv.abi3.so" \
  "${SOURCE}" "${PATCH}" \
  | tee "${ARTIFACT_DIR}/products-job${SLURM_JOB_ID}.sha256"
[[ "$(sha256sum "${ARTIFACT_DIR}/_C_stable_libtorch.v1.abi3.so" | awk '{print $1}')" == "${EXPECTED_V1_LIB_SHA}" ]]
V2_LIB_SHA=$(sha256sum \
  "${ARTIFACT_DIR}/_C_stable_libtorch.v2-warp-owned-pv.abi3.so" \
  | awk '{print $1}')
[[ "${V2_LIB_SHA}" != "${EXPECTED_V1_LIB_SHA}" ]]
printf 'V2_LIB_SHA=%s\n' "${V2_LIB_SHA}"

sha256sum \
  "${ARTIFACT_DIR}/v1-inputs-job${SLURM_JOB_ID}.sha256" \
  "${ARTIFACT_DIR}/v2-source-job${SLURM_JOB_ID}.sha256" \
  "${ARTIFACT_DIR}/patch-dry-run-job${SLURM_JOB_ID}.log" \
  "${ARTIFACT_DIR}/patch-apply-job${SLURM_JOB_ID}.log" \
  "${ARTIFACT_DIR}/native_c2_v2.input-job${SLURM_JOB_ID}.patch" \
  "${ARTIFACT_DIR}/v1-tracked-diff-job${SLURM_JOB_ID}.patch" \
  "${ARTIFACT_DIR}/v1-status-job${SLURM_JOB_ID}.bin" \
  "${ARTIFACT_DIR}/native-c2-v1-to-v2-job${SLURM_JOB_ID}.patch" \
  "${ARTIFACT_DIR}/derived-v2-job${SLURM_JOB_ID}.patch" \
  "${ARTIFACT_DIR}/compile-v2-job${SLURM_JOB_ID}.log" \
  "${ARTIFACT_DIR}/products-job${SLURM_JOB_ID}.sha256" \
  "${ARTIFACT_DIR}/native_c2_decode.v1.cu" \
  "${ARTIFACT_DIR}/native_c2_decode.v2-warp-owned-pv.cu" \
  > "${ARTIFACT_DIR}/build-job${SLURM_JOB_ID}.sha256"
