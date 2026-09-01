#!/usr/bin/env bash
# Provenance-gated incremental build for the four-producer candidate.  This is
# deliberately sequenced after v2 has passed its standalone acceptance gate:
# it mutates the shared derived checkout/build tree only after saving immutable
# v2 copies to a separate artifact directory.
#SBATCH --job-name=c2-native-v2-4p-build
#SBATCH --partition=cpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --output=/home/lcpu/85117379/c2-native-v2-four-producer-build-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail

SRC=/home/lcpu/85117379/vllm-d4-native-c2-20260829
SOURCE=${SRC}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
BUILD_DIR=/home/lcpu/85117379/vllm-d4-native-c2-build-20260829
LIB=${BUILD_DIR}/_C_stable_libtorch.abi3.so
PATCH=/home/lcpu/85117379/native_c2_four_producer_20260829.patch
ARTIFACT_DIR=/home/lcpu/85117379/c2-native-v2-four-producer-build-artifacts-20260829
TOOLS=/home/lcpu/85117379/c2-native-tools-20260829
EXPECTED_HEAD=d4da0c55af3aa231b6209bf77871f3ed36eab0d2
EXPECTED_V2_SOURCE_SHA=6379cb0c71aa79610dd807b40982a6653637d926c1e3dab78155f1f3a5a4f6ce
EXPECTED_PATCH_SHA=4547dc1be68f92a98acc960589f7a0d3d9efb89edb067d6b2f73a445cbefab17
EXPECTED_V2_FOUR_PRODUCER_SOURCE_SHA=a46830564aa62e30e26de67dc98882043a239985f6d1c7e1b2dc2a03630fd8a1

# The v2 AOT digest is only known after the v2 build.  Requiring it here
# prevents accidentally deriving four-producer from v1 or an unreviewed v2
# library.  The new digest is recorded below; if a rebuild is desired, it can
# additionally be locked with C2_EXPECTED_V2_FOUR_PRODUCER_LIB_SHA.
: "${C2_EXPECTED_V2_LIB_SHA:?submit with the reviewed v2 AOT SHA-256}"
C2_EXPECTED_V2_FOUR_PRODUCER_LIB_SHA=${C2_EXPECTED_V2_FOUR_PRODUCER_LIB_SHA:-}
[[ "${C2_EXPECTED_V2_LIB_SHA}" =~ ^[0-9a-f]{64}$ ]]
[[ -z "${C2_EXPECTED_V2_FOUR_PRODUCER_LIB_SHA}" || "${C2_EXPECTED_V2_FOUR_PRODUCER_LIB_SHA}" =~ ^[0-9a-f]{64}$ ]]

mkdir -p "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/build-driver-job${SLURM_JOB_ID}.log") 2>&1
trap 'rc=$?; printf "FINAL_RC=%s UTC=%s\n" "${rc}" "$(date -u +%FT%TZ)"' EXIT
export PATH="${TOOLS}/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin"

printf 'UTC=%s JOB=%s HOST=%s\n' "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}" "$(hostname)"
[[ "$(git -C "${SRC}" rev-parse HEAD)" == "${EXPECTED_HEAD}" ]]
[[ -f "${SOURCE}" && -f "${LIB}" && -f "${PATCH}" ]]
command -v patch
[[ "$(sha256sum "${SOURCE}" | awk '{print $1}')" == "${EXPECTED_V2_SOURCE_SHA}" ]]
[[ "$(sha256sum "${LIB}" | awk '{print $1}')" == "${C2_EXPECTED_V2_LIB_SHA}" ]]
[[ "$(sha256sum "${PATCH}" | awk '{print $1}')" == "${EXPECTED_PATCH_SHA}" ]]
patch --dry-run --batch --forward -p1 -d "${SRC}" < "${PATCH}" \
  | tee "${ARTIFACT_DIR}/patch-dry-run-job${SLURM_JOB_ID}.log"

cp --reflink=auto --preserve=mode,timestamps "${LIB}" \
  "${ARTIFACT_DIR}/_C_stable_libtorch.v2.abi3.so"
cp --preserve=mode,timestamps "${SOURCE}" \
  "${ARTIFACT_DIR}/native_c2_decode.v2.cu"
cp --preserve=mode,timestamps "${PATCH}" \
  "${ARTIFACT_DIR}/native_c2_four_producer.patch"
sha256sum "${SOURCE}" "${LIB}" "${PATCH}" \
  | tee "${ARTIFACT_DIR}/v2-inputs-job${SLURM_JOB_ID}.sha256"

patch --batch --forward -p1 -d "${SRC}" < "${PATCH}" \
  | tee "${ARTIFACT_DIR}/patch-apply-job${SLURM_JOB_ID}.log"
git -C "${SRC}" diff --check
[[ "$(sha256sum "${SOURCE}" | awk '{print $1}')" == "${EXPECTED_V2_FOUR_PRODUCER_SOURCE_SHA}" ]]
cp --preserve=mode,timestamps "${SOURCE}" \
  "${ARTIFACT_DIR}/native_c2_decode.v2-four-producer.cu"
sha256sum "${SOURCE}" \
  | tee "${ARTIFACT_DIR}/v2-four-producer-source-job${SLURM_JOB_ID}.sha256"

# native_c2_decode.cu is an overlay/untracked file in this derived checkout;
# git diff cannot prove its transformation through the repository index.  Keep
# a direct binary-safe no-index comparison instead, and require exactly one
# normal "different" exit.
set +e
git diff --no-index --binary -- \
  "${ARTIFACT_DIR}/native_c2_decode.v2.cu" "${SOURCE}" \
  > "${ARTIFACT_DIR}/v2-to-v2-four-producer-job${SLURM_JOB_ID}.diff"
diff_rc=$?
set -e
[[ "${diff_rc}" -eq 1 ]]

cmake --build "${BUILD_DIR}" --target _C_stable_libtorch -- -j64 \
  2>&1 | tee "${ARTIFACT_DIR}/compile-v2-four-producer-job${SLURM_JOB_ID}.log"

NEW_LIB_SHA=$(sha256sum "${LIB}" | awk '{print $1}')
[[ "${NEW_LIB_SHA}" != "${C2_EXPECTED_V2_LIB_SHA}" ]]
if [[ -n "${C2_EXPECTED_V2_FOUR_PRODUCER_LIB_SHA}" ]]; then
  [[ "${NEW_LIB_SHA}" == "${C2_EXPECTED_V2_FOUR_PRODUCER_LIB_SHA}" ]]
fi
cp --reflink=auto --preserve=mode,timestamps "${LIB}" \
  "${ARTIFACT_DIR}/_C_stable_libtorch.v2-four-producer.abi3.so"
sha256sum \
  "${ARTIFACT_DIR}/_C_stable_libtorch.v2.abi3.so" \
  "${ARTIFACT_DIR}/_C_stable_libtorch.v2-four-producer.abi3.so" \
  "${SOURCE}" "${PATCH}" \
  | tee "${ARTIFACT_DIR}/products-job${SLURM_JOB_ID}.sha256"
[[ "$(sha256sum "${ARTIFACT_DIR}/_C_stable_libtorch.v2.abi3.so" | awk '{print $1}')" == "${C2_EXPECTED_V2_LIB_SHA}" ]]
[[ "$(sha256sum "${ARTIFACT_DIR}/_C_stable_libtorch.v2-four-producer.abi3.so" | awk '{print $1}')" == "${NEW_LIB_SHA}" ]]
printf 'C2_EXPECTED_V2_FOUR_PRODUCER_LIB_SHA=%s\n' "${NEW_LIB_SHA}" \
  | tee "${ARTIFACT_DIR}/next-submit-job${SLURM_JOB_ID}.env"

sha256sum \
  "${ARTIFACT_DIR}/v2-inputs-job${SLURM_JOB_ID}.sha256" \
  "${ARTIFACT_DIR}/patch-dry-run-job${SLURM_JOB_ID}.log" \
  "${ARTIFACT_DIR}/patch-apply-job${SLURM_JOB_ID}.log" \
  "${ARTIFACT_DIR}/v2-four-producer-source-job${SLURM_JOB_ID}.sha256" \
  "${ARTIFACT_DIR}/v2-to-v2-four-producer-job${SLURM_JOB_ID}.diff" \
  "${ARTIFACT_DIR}/compile-v2-four-producer-job${SLURM_JOB_ID}.log" \
  "${ARTIFACT_DIR}/products-job${SLURM_JOB_ID}.sha256" \
  "${ARTIFACT_DIR}/native_c2_decode.v2.cu" \
  "${ARTIFACT_DIR}/native_c2_decode.v2-four-producer.cu" \
  "${ARTIFACT_DIR}/native_c2_four_producer.patch" \
  > "${ARTIFACT_DIR}/build-job${SLURM_JOB_ID}.sha256"
