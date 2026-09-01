#!/usr/bin/env bash
# Build the NCU-motivated warp-owned-PV v2 as a *separate* C2 plugin DSO.
# It never builds, writes, links, or packages _C_stable_libtorch.abi3.so.
#
# Required immutable pre-uploaded inputs:
#   /home/lcpu/85117379/native_c2_decode.plugin-v1.cu
#   /home/lcpu/85117379/native_c2_plugin_schema_20260829.patch
#   /home/lcpu/85117379/exact_d4_native_c2_plugin_cmake_20260829.patch
#   /home/lcpu/85117379/exact_d4_python_dispatch_20260829.patch
#   /home/lcpu/85117379/exact_d4_native_c2_plugin_python_loader_20260829.patch
#   /home/lcpu/85117379/native_c2_v2_warp_owned_pv_20260829.patch
#   mkdir -p /home/lcpu/85117379/c2-native-plugin-v2-aot-artifacts-20260829
#
# This job deliberately starts from a fresh no-hardlinks clone rather than
# modifying the v1 plugin checkout.  A failed candidate consequently leaves
# both the v1 DSO and its evidence lineage intact.

#SBATCH --job-name=c2-native-plugin-v2-aot
#SBATCH --partition=cpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v2-aot-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail

BASE=/home/lcpu/85117379/vllm-d4da0c5
SRC=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v2-warp-owned-pv-20260829-job${SLURM_JOB_ID}
BUILD_DIR=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v2-warp-owned-pv-build-20260829-job${SLURM_JOB_ID}
ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v2-aot-artifacts-20260829
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID}
TOOLS=/home/lcpu/85117379/c2-native-tools-20260829
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
CUTLASS=/home/lcpu/85117379/cutlass-v4.4.2-c2-20260829
MSA=/home/lcpu/85117379/msa-official
DEPS=/home/lcpu/85117379/c2-native-plugin-v2-aot-deps-20260829-job${SLURM_JOB_ID}

NATIVE_V1_SOURCE=/home/lcpu/85117379/native_c2_decode.plugin-v1.cu
SCHEMA_PATCH=/home/lcpu/85117379/native_c2_plugin_schema_20260829.patch
CMAKE_PATCH=/home/lcpu/85117379/exact_d4_native_c2_plugin_cmake_20260829.patch
DISPATCH_PATCH=/home/lcpu/85117379/exact_d4_python_dispatch_20260829.patch
LOADER_PATCH=/home/lcpu/85117379/exact_d4_native_c2_plugin_python_loader_20260829.patch
V2_PATCH=/home/lcpu/85117379/native_c2_v2_warp_owned_pv_20260829.patch

EXPECTED_BASE_COMMIT=d4da0c55af3aa231b6209bf77871f3ed36eab0d2
EXPECTED_NATIVE_V1_SOURCE_SHA=f67495586759b3b8f718dbbf60627c499f917842f2ed6ecf89d2344f1c82bc4b
EXPECTED_SCHEMA_PATCH_SHA=53e7cd4b09a8999010442934c863427c30b019fc549d7f0cf0f3d16ccc1e2e6e
EXPECTED_CMAKE_PATCH_SHA=5b145f36ce5f8b12dae71ce02e77d65545a29eefcb5a6045f591348bc325ac35
EXPECTED_DISPATCH_PATCH_SHA=1ca224ec4f94adf8ee97580a0e08321552224555570784c308c0edcda34bcd91
EXPECTED_LOADER_PATCH_SHA=9cb3a22216048083d7a4802ba700fa07fd892436151871dc6e909c77239adb33
EXPECTED_V2_PATCH_SHA=2fa34736ba80a50cd6d4a40377ad419709941349eff5c26e745541cfd8862816
# Current frozen v1 contract, followed by the v2 source-only candidate patch.
EXPECTED_DISPATCH_CUSTOM_OPS_SHA=b57dfa819424b89bed8bf39924d5a4a3d2ebf327b39f3ae7c2e9e65b61c2398d
EXPECTED_DISPATCH_CONFIG_SHA=caf68c29d8d36324d252a66f0811a25b26d11da9860732359076fbb76625b5ae
EXPECTED_DISPATCH_ADAPTER_PRELOADER_SHA=2e350ba3b339a42140c1d252677f55c374bdad873ba14781b3976c2053774dd6
EXPECTED_DISPATCH_SPARSE_SHA=07d00372ca1995aa96159a0b9f8949b1b40f817a63f49224e88aafac1b722200
EXPECTED_DISPATCH_REGISTRY_SHA=cd539941417fd5faf6672676de0f8ae15c9adb2a915b47cdd508e154c34faea4
EXPECTED_DISPATCH_CUSTOM_OPS_BLOB=5088efa8388ead2054c29d34dd9eb423f39e4cd7
EXPECTED_DISPATCH_CONFIG_BLOB=9e96be7a844a94c002573e6a349d26c2693b38b9
EXPECTED_DISPATCH_ADAPTER_PRELOADER_BLOB=d1c4725259a5fd4fba125bb613ec745dd1189c8f
EXPECTED_DISPATCH_SPARSE_BLOB=c19acdb08b99575d59a5875231b15f4176d971df
EXPECTED_DISPATCH_REGISTRY_BLOB=a6a79382e3ff761b5d201f1818960c9b50b6e3c4
EXPECTED_PLUGIN_V1_SOURCE_SHA=8fee086f8ba4f5b2710347025211f9733ce64c61b04e44e710016d59b336f406
EXPECTED_V2_SOURCE_SHA=f2aae9c9c62e646b3cd38b136f224009d400298c21a99b9b1cf5a402f944acc4
EXPECTED_PLUGIN_CMAKE_SHA=6662b0400eb04c590b56c15346848b4a26c45d86e796a69e1bcb4a0d91523734
EXPECTED_PLUGIN_ADAPTER_SHA=6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412

for input in "${NATIVE_V1_SOURCE}" "${SCHEMA_PATCH}" "${CMAKE_PATCH}" \
             "${DISPATCH_PATCH}" "${LOADER_PATCH}" "${V2_PATCH}"; do
  [[ -f "${input}" ]] || { echo "missing required input: ${input}" >&2; exit 2; }
done
[[ -d "${ARTIFACT_ROOT}" ]] || {
  echo "missing pre-created artifact root: ${ARTIFACT_ROOT}" >&2; exit 2;
}
[[ ! -e "${SRC}" ]] || { echo "refusing to reuse/overwrite ${SRC}" >&2; exit 2; }
[[ ! -e "${BUILD_DIR}" ]] || { echo "refusing to reuse/overwrite ${BUILD_DIR}" >&2; exit 2; }
[[ ! -e "${ARTIFACT_DIR}" ]] || { echo "refusing to reuse/overwrite ${ARTIFACT_DIR}" >&2; exit 2; }
mkdir -p "${ARTIFACT_DIR}" "${DEPS}"
exec > >(tee "${ARTIFACT_DIR}/build-driver-job${SLURM_JOB_ID}.log") 2>&1

finalize() {
  local rc=$?
  trap - EXIT
  set +e
  printf 'FINAL_RC=%s UTC=%s\n' "${rc}" "$(date -u +%FT%TZ)" \
    | tee "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt"
  exit "${rc}"
}
trap finalize EXIT

export PATH="${TOOLS}/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin"
export TORCH_CUDA_ARCH_LIST=10.3f
export VLLM_CUTLASS_SRC_DIR="${CUTLASS}"
export FMHA_SM100_SRC_DIR="${MSA}"

sha256_of() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(sha256_of "${path}")
  printf '%s SHA256=%s path=%s\n' "${label}" "${actual}" "${path}"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} hash mismatch: expected ${expected}, got ${actual}" >&2; exit 2;
  }
}

require_git_blob() {
  local relative_path=$1 expected=$2 label=$3 actual
  actual=$(git -C "${SRC}" hash-object "${relative_path}")
  printf '%s GIT_BLOB=%s path=%s\n' "${label}" "${actual}" "${relative_path}"
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} blob mismatch: expected ${expected}, got ${actual}" >&2; exit 2;
  }
}

printf 'UTC=%s\nSLURM_JOB_ID=%s HOST=%s CPUS=%s\n' \
  "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}" "$(hostname)" "${SLURM_CPUS_PER_TASK}"
printf 'BASE=%s\nSRC=%s\nBUILD_DIR=%s\nARTIFACT_DIR=%s\n' \
  "${BASE}" "${SRC}" "${BUILD_DIR}" "${ARTIFACT_DIR}"
git -C "${BASE}" rev-parse HEAD | tee "${ARTIFACT_DIR}/base-commit-job${SLURM_JOB_ID}.txt"
[[ "$(git -C "${BASE}" rev-parse HEAD)" == "${EXPECTED_BASE_COMMIT}" ]]
[[ -z "$(git -C "${BASE}" status --porcelain)" ]] || {
  echo "base checkout is dirty; refusing to clone mutable source" >&2; exit 2;
}
require_hash "${NATIVE_V1_SOURCE}" "${EXPECTED_NATIVE_V1_SOURCE_SHA}" native-v1-source
require_hash "${SCHEMA_PATCH}" "${EXPECTED_SCHEMA_PATCH_SHA}" schema-patch
require_hash "${CMAKE_PATCH}" "${EXPECTED_CMAKE_PATCH_SHA}" cmake-patch
require_hash "${DISPATCH_PATCH}" "${EXPECTED_DISPATCH_PATCH_SHA}" dispatch-patch
require_hash "${LOADER_PATCH}" "${EXPECTED_LOADER_PATCH_SHA}" loader-patch
require_hash "${V2_PATCH}" "${EXPECTED_V2_PATCH_SHA}" warp-owned-pv-v2-patch
git -C "${CUTLASS}" rev-parse HEAD
git -C "${MSA}" rev-parse HEAD
git -C "${MSA}/python/fmha_sm100/cutlass" rev-parse HEAD
"${PYTHON}" -c 'import torch; print("TORCH", torch.__version__, torch.version.cuda)'
cmake --version
ninja --version
nvcc --version
patch --version > "${ARTIFACT_DIR}/patch-version-job${SLURM_JOB_ID}.txt"
sed -n '1p' "${ARTIFACT_DIR}/patch-version-job${SLURM_JOB_ID}.txt"

# No hardlinks: the new checkout cannot mutate base or the frozen v1 checkout.
git clone --no-hardlinks "${BASE}" "${SRC}"
git -C "${SRC}" checkout --detach "${EXPECTED_BASE_COMMIT}"
[[ -z "$(git -C "${SRC}" status --porcelain)" ]]
TARGET_SOURCE=${SRC}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
install -D -m 0644 "${NATIVE_V1_SOURCE}" "${TARGET_SOURCE}"
require_hash "${TARGET_SOURCE}" "${EXPECTED_NATIVE_V1_SOURCE_SHA}" copied-native-v1-source

for patch_file in "${SCHEMA_PATCH}" "${CMAKE_PATCH}" "${DISPATCH_PATCH}"; do
  patch --dry-run --batch --fuzz=0 -d "${SRC}" -p1 < "${patch_file}"
  patch --batch --fuzz=0 -d "${SRC}" -p1 < "${patch_file}"
done
require_hash "${SRC}/vllm/_custom_ops.py" "${EXPECTED_DISPATCH_CUSTOM_OPS_SHA}" post-dispatch-custom-ops
require_hash "${SRC}/vllm/config/attention.py" "${EXPECTED_DISPATCH_CONFIG_SHA}" post-dispatch-config
require_hash "${SRC}/vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py" \
  "${EXPECTED_DISPATCH_ADAPTER_PRELOADER_SHA}" post-dispatch-adapter-preloader
require_hash "${SRC}/vllm/models/minimax_m3/nvidia/sparse_attention_msa.py" \
  "${EXPECTED_DISPATCH_SPARSE_SHA}" post-dispatch-sparse
require_hash "${SRC}/vllm/v1/attention/backends/registry.py" \
  "${EXPECTED_DISPATCH_REGISTRY_SHA}" post-dispatch-registry
require_git_blob vllm/_custom_ops.py "${EXPECTED_DISPATCH_CUSTOM_OPS_BLOB}" post-dispatch-custom-ops
require_git_blob vllm/config/attention.py "${EXPECTED_DISPATCH_CONFIG_BLOB}" post-dispatch-config
require_git_blob vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py \
  "${EXPECTED_DISPATCH_ADAPTER_PRELOADER_BLOB}" post-dispatch-adapter-preloader
require_git_blob vllm/models/minimax_m3/nvidia/sparse_attention_msa.py \
  "${EXPECTED_DISPATCH_SPARSE_BLOB}" post-dispatch-sparse
require_git_blob vllm/v1/attention/backends/registry.py \
  "${EXPECTED_DISPATCH_REGISTRY_BLOB}" post-dispatch-registry
sha256sum "${SRC}/vllm/_custom_ops.py" "${SRC}/vllm/config/attention.py" \
  "${SRC}/vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py" \
  "${SRC}/vllm/models/minimax_m3/nvidia/sparse_attention_msa.py" \
  "${SRC}/vllm/v1/attention/backends/registry.py" \
  | tee "${ARTIFACT_DIR}/post-dispatch-members-job${SLURM_JOB_ID}.sha256"
patch --dry-run --batch --fuzz=0 -d "${SRC}" -p1 < "${LOADER_PATCH}"
patch --batch --fuzz=0 -d "${SRC}" -p1 < "${LOADER_PATCH}"
require_hash "${TARGET_SOURCE}" "${EXPECTED_PLUGIN_V1_SOURCE_SHA}" plugin-v1-source-before-v2
require_hash "${SRC}/CMakeLists.txt" "${EXPECTED_PLUGIN_CMAKE_SHA}" post-plugin-cmake
require_hash "${SRC}/vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py" \
  "${EXPECTED_PLUGIN_ADAPTER_SHA}" post-plugin-adapter
"${PYTHON}" -m py_compile "${SRC}/vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py"

patch --dry-run --batch --fuzz=0 -d "${SRC}" -p1 < "${V2_PATCH}" \
  | tee "${ARTIFACT_DIR}/v2-patch-dry-run-job${SLURM_JOB_ID}.log"
patch --batch --fuzz=0 -d "${SRC}" -p1 < "${V2_PATCH}" \
  | tee "${ARTIFACT_DIR}/v2-patch-apply-job${SLURM_JOB_ID}.log"
require_hash "${TARGET_SOURCE}" "${EXPECTED_V2_SOURCE_SHA}" post-warp-owned-pv-v2-source
git -C "${SRC}" add -N csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu \
  vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py
git -C "${SRC}" diff --check
git -C "${SRC}" status --short | tee "${ARTIFACT_DIR}/derived-status-job${SLURM_JOB_ID}.txt"
git -C "${SRC}" diff --binary > "${ARTIFACT_DIR}/derived-plugin-v2-job${SLURM_JOB_ID}.patch"
cp --preserve=mode,timestamps "${TARGET_SOURCE}" \
  "${ARTIFACT_DIR}/native_c2_decode.v2-warp-owned-pv.cu"

# Requested B300 10.3f is intersected with exact-d4 CUDA-13's supported 10.0
# family baseline; the later GPU job proves that sm_100 cubin runs on B300.
cmake -S "${SRC}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE="${PYTHON}" \
  -DFETCHCONTENT_BASE_DIR="${DEPS}" \
  -DCMAKE_INSTALL_PREFIX="${ARTIFACT_DIR}/install" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DNVCC_THREADS=2 \
  -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
  -DCMAKE_JOB_POOLS:STRING=compile=32 \
  2>&1 | tee "${ARTIFACT_DIR}/configure-job${SLURM_JOB_ID}.log"
grep -F 'CUDA target architectures: 10.3f' \
  "${ARTIFACT_DIR}/configure-job${SLURM_JOB_ID}.log"
grep -F 'CUDA supported target architectures: 10.0' \
  "${ARTIFACT_DIR}/configure-job${SLURM_JOB_ID}.log"
grep -F 'Building standalone native C2 plugin for archs: 10.0' \
  "${ARTIFACT_DIR}/configure-job${SLURM_JOB_ID}.log"
grep -F 'native_c2_decode.cu' "${BUILD_DIR}/compile_commands.json" \
  | tee "${ARTIFACT_DIR}/native-c2-compile-command-job${SLURM_JOB_ID}.json"
grep -F -- '-gencode arch=compute_100,code=sm_100' \
  "${ARTIFACT_DIR}/native-c2-compile-command-job${SLURM_JOB_ID}.json"

cmake --build "${BUILD_DIR}" --target _native_c2_msa_decode_plugin -- -j64 -v \
  2>&1 | tee "${ARTIFACT_DIR}/compile-job${SLURM_JOB_ID}.log"
mapfile -t PLUGIN_CANDIDATES < <(find "${BUILD_DIR}" -type f \
  -name '_native_c2_msa_decode_plugin.abi3.so' -print)
[[ ${#PLUGIN_CANDIDATES[@]} -eq 1 ]] || {
  printf 'expected exactly one plugin DSO, found %s\n' "${#PLUGIN_CANDIDATES[@]}" >&2
  printf '%s\n' "${PLUGIN_CANDIDATES[@]:-}" >&2; exit 2;
}
PLUGIN=${PLUGIN_CANDIDATES[0]}
install -D -m 0755 "${PLUGIN}" "${ARTIFACT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so"
grep -F -- '-gencode arch=compute_100,code=sm_100' \
  "${ARTIFACT_DIR}/compile-job${SLURM_JOB_ID}.log"
nm -D --defined-only "${PLUGIN}" | tee "${ARTIFACT_DIR}/plugin-symbols-job${SLURM_JOB_ID}.txt"
! grep -q ' PyInit_' "${ARTIFACT_DIR}/plugin-symbols-job${SLURM_JOB_ID}.txt"
cuobjdump --list-elf "${PLUGIN}" | tee "${ARTIFACT_DIR}/plugin-cuobjdump-elf-job${SLURM_JOB_ID}.txt"
mapfile -t CUBIN_LINES < <(grep -E '^ELF file[[:space:]]+[0-9]+:.*\.cubin$' \
  "${ARTIFACT_DIR}/plugin-cuobjdump-elf-job${SLURM_JOB_ID}.txt")
[[ ${#CUBIN_LINES[@]} -eq 1 ]] || {
  printf 'expected exactly one plugin cubin, found %s\n' "${#CUBIN_LINES[@]}" >&2
  printf '%s\n' "${CUBIN_LINES[@]:-}" >&2
  exit 2
}
[[ "${CUBIN_LINES[0]}" == *'.sm_100.cubin' ]] || {
  printf 'plugin cubin is not the required sm_100 code object: %s\n' "${CUBIN_LINES[0]}" >&2
  exit 2
}
! grep -Eqi 'sm_10(3|3f)|compute_10(3|3f)|\.ptx($|[^[:alnum:]_])' \
  "${ARTIFACT_DIR}/plugin-cuobjdump-elf-job${SLURM_JOB_ID}.txt"
cuobjdump --list-ptx "${PLUGIN}" \
  | tee "${ARTIFACT_DIR}/plugin-cuobjdump-ptx-job${SLURM_JOB_ID}.txt"
[[ ! -s "${ARTIFACT_DIR}/plugin-cuobjdump-ptx-job${SLURM_JOB_ID}.txt" ]] || {
  echo "plugin must contain exactly one sm_100 cubin and no PTX fallback" >&2
  exit 2
}
cuobjdump --dump-sass "${PLUGIN}" | tee "${ARTIFACT_DIR}/plugin-sass-job${SLURM_JOB_ID}.sass"
grep -m 80 -E 'native_c2_msa_decode_kernel|HMMA|MMA' \
  "${ARTIFACT_DIR}/plugin-sass-job${SLURM_JOB_ID}.sass" \
  | tee "${ARTIFACT_DIR}/plugin-sass-proof-job${SLURM_JOB_ID}.txt"
grep -q 'native_c2_msa_decode_kernel' "${ARTIFACT_DIR}/plugin-sass-proof-job${SLURM_JOB_ID}.txt"
grep -Eq 'HMMA|MMA' "${ARTIFACT_DIR}/plugin-sass-proof-job${SLURM_JOB_ID}.txt"

sha256sum "${NATIVE_V1_SOURCE}" "${SCHEMA_PATCH}" "${CMAKE_PATCH}" \
  "${DISPATCH_PATCH}" "${LOADER_PATCH}" "${V2_PATCH}" "${TARGET_SOURCE}" "${PLUGIN}" \
  "${ARTIFACT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so" \
  > "${ARTIFACT_DIR}/plugin-v2-provenance-job${SLURM_JOB_ID}.sha256"
ldd "${PLUGIN}" | tee "${ARTIFACT_DIR}/plugin-ldd-job${SLURM_JOB_ID}.txt"
sha256sum "${ARTIFACT_DIR}"/*.log "${ARTIFACT_DIR}"/*.txt \
  "${ARTIFACT_DIR}"/*.json "${ARTIFACT_DIR}"/*.patch \
  "${ARTIFACT_DIR}"/*.sha256 "${ARTIFACT_DIR}"/*.cu \
  > "${ARTIFACT_DIR}/build-job${SLURM_JOB_ID}.sha256"
