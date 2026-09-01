#!/usr/bin/env bash
# Build v3 only as a new standalone C2-plugin DSO.  The controlled source
# delta over v2 is pv_contribution's float row leading dimension 16 -> 20.
# No wheel or _C_stable_libtorch artifact is produced or modified.

#SBATCH --job-name=c2-native-plugin-v3-aot
#SBATCH --partition=cpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v3-aot-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail

BASE=/home/lcpu/85117379/vllm-d4da0c5
SRC=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v3-padded-pv-20260829-job${SLURM_JOB_ID}
BUILD_DIR=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v3-padded-pv-build-20260829-job${SLURM_JOB_ID}
DEPS=/home/lcpu/85117379/c2-native-plugin-v3-aot-deps-20260829-job${SLURM_JOB_ID}
ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v3-aot-artifacts-20260829
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID}
TOOLS=/home/lcpu/85117379/c2-native-tools-20260829
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
CUTLASS=/home/lcpu/85117379/cutlass-v4.4.2-c2-20260829
MSA=/home/lcpu/85117379/msa-official

NATIVE_V1_SOURCE=/home/lcpu/85117379/native_c2_decode.plugin-v1.cu
SCHEMA_PATCH=/home/lcpu/85117379/native_c2_plugin_schema_20260829.patch
CMAKE_PATCH=/home/lcpu/85117379/exact_d4_native_c2_plugin_cmake_20260829.patch
DISPATCH_PATCH=/home/lcpu/85117379/exact_d4_python_dispatch_20260829.patch
LOADER_PATCH=/home/lcpu/85117379/exact_d4_native_c2_plugin_python_loader_20260829.patch
V2_PATCH=/home/lcpu/85117379/native_c2_v2_warp_owned_pv_20260829.patch
V3_PATCH=/home/lcpu/85117379/native_c2_v3_padded_pv_20260829.patch

EXPECTED_BASE_COMMIT=d4da0c55af3aa231b6209bf77871f3ed36eab0d2
EXPECTED_NATIVE_V1_SOURCE_SHA=f67495586759b3b8f718dbbf60627c499f917842f2ed6ecf89d2344f1c82bc4b
EXPECTED_SCHEMA_PATCH_SHA=53e7cd4b09a8999010442934c863427c30b019fc549d7f0cf0f3d16ccc1e2e6e
EXPECTED_CMAKE_PATCH_SHA=5b145f36ce5f8b12dae71ce02e77d65545a29eefcb5a6045f591348bc325ac35
EXPECTED_DISPATCH_PATCH_SHA=1ca224ec4f94adf8ee97580a0e08321552224555570784c308c0edcda34bcd91
EXPECTED_LOADER_PATCH_SHA=9cb3a22216048083d7a4802ba700fa07fd892436151871dc6e909c77239adb33
EXPECTED_V2_PATCH_SHA=2fa34736ba80a50cd6d4a40377ad419709941349eff5c26e745541cfd8862816
EXPECTED_V3_PATCH_SHA=12fa59d5ffeb213e7f476bef06fdc68347bdf28c64136de58b2d48dcb70b1a22
EXPECTED_PLUGIN_V1_SOURCE_SHA=8fee086f8ba4f5b2710347025211f9733ce64c61b04e44e710016d59b336f406
EXPECTED_V2_SOURCE_SHA=f2aae9c9c62e646b3cd38b136f224009d400298c21a99b9b1cf5a402f944acc4
EXPECTED_V3_SOURCE_SHA=3e01404e8dfd2c82f818bc8ad2e90434f384992dc71adfd9308a6f5baa5eeff2
EXPECTED_PLUGIN_CMAKE_SHA=6662b0400eb04c590b56c15346848b4a26c45d86e796a69e1bcb4a0d91523734
EXPECTED_PLUGIN_ADAPTER_SHA=6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412

for input in "${NATIVE_V1_SOURCE}" "${SCHEMA_PATCH}" "${CMAKE_PATCH}" \
  "${DISPATCH_PATCH}" "${LOADER_PATCH}" "${V2_PATCH}" "${V3_PATCH}"; do
  [[ -f "${input}" ]] || { echo "missing input: ${input}" >&2; exit 2; }
done
[[ -d "${ARTIFACT_ROOT}" && ! -e "${SRC}" && ! -e "${BUILD_DIR}" && ! -e "${ARTIFACT_DIR}" ]] || {
  echo 'artifact root missing or job-scoped output already exists' >&2; exit 2;
}
mkdir -p "${ARTIFACT_DIR}" "${DEPS}"
exec > >(tee "${ARTIFACT_DIR}/build-driver-job${SLURM_JOB_ID}.log") 2>&1
finalize() { local rc=$?; trap - EXIT; set +e; printf 'FINAL_RC=%s UTC=%s\n' "${rc}" "$(date -u +%FT%TZ)" | tee "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt"; exit "${rc}"; }
trap finalize EXIT

export PATH="${TOOLS}/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin"
export TORCH_CUDA_ARCH_LIST=10.3f
export VLLM_CUTLASS_SRC_DIR="${CUTLASS}"
export FMHA_SM100_SRC_DIR="${MSA}"
sha256_of() { sha256sum "$1" | awk '{print $1}'; }
require_hash() { local actual; actual=$(sha256_of "$1"); printf '%s SHA256=%s path=%s\n' "$3" "${actual}" "$1"; [[ "${actual}" == "$2" ]] || { echo "$3 hash mismatch" >&2; exit 2; }; }

printf 'UTC=%s JOB=%s HOST=%s CPUS=%s\n' "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}" "$(hostname)" "${SLURM_CPUS_PER_TASK}"
[[ "$(git -C "${BASE}" rev-parse HEAD)" == "${EXPECTED_BASE_COMMIT}" && -z "$(git -C "${BASE}" status --porcelain)" ]] || { echo 'exact d4 base not clean' >&2; exit 2; }
printf '%s\n' "${EXPECTED_BASE_COMMIT}" > "${ARTIFACT_DIR}/base-commit-job${SLURM_JOB_ID}.txt"
require_hash "${NATIVE_V1_SOURCE}" "${EXPECTED_NATIVE_V1_SOURCE_SHA}" native-v1-source
require_hash "${SCHEMA_PATCH}" "${EXPECTED_SCHEMA_PATCH_SHA}" schema-patch
require_hash "${CMAKE_PATCH}" "${EXPECTED_CMAKE_PATCH_SHA}" cmake-patch
require_hash "${DISPATCH_PATCH}" "${EXPECTED_DISPATCH_PATCH_SHA}" dispatch-patch
require_hash "${LOADER_PATCH}" "${EXPECTED_LOADER_PATCH_SHA}" loader-patch
require_hash "${V2_PATCH}" "${EXPECTED_V2_PATCH_SHA}" warp-owned-pv-v2-patch
require_hash "${V3_PATCH}" "${EXPECTED_V3_PATCH_SHA}" padded-pv-v3-patch
git -C "${CUTLASS}" rev-parse HEAD; git -C "${MSA}" rev-parse HEAD; git -C "${MSA}/python/fmha_sm100/cutlass" rev-parse HEAD
"${PYTHON}" -c 'import torch; print("TORCH", torch.__version__, torch.version.cuda)'
cmake --version; ninja --version; nvcc --version

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
patch --dry-run --batch --fuzz=0 -d "${SRC}" -p1 < "${LOADER_PATCH}"
patch --batch --fuzz=0 -d "${SRC}" -p1 < "${LOADER_PATCH}"
require_hash "${TARGET_SOURCE}" "${EXPECTED_PLUGIN_V1_SOURCE_SHA}" plugin-v1-source-before-v2
require_hash "${SRC}/CMakeLists.txt" "${EXPECTED_PLUGIN_CMAKE_SHA}" plugin-cmake
require_hash "${SRC}/vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py" "${EXPECTED_PLUGIN_ADAPTER_SHA}" plugin-adapter
"${PYTHON}" -m py_compile "${SRC}/vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py"

patch --dry-run --batch --fuzz=0 -d "${SRC}" -p1 < "${V2_PATCH}" | tee "${ARTIFACT_DIR}/v2-patch-dry-run-job${SLURM_JOB_ID}.log"
patch --batch --fuzz=0 -d "${SRC}" -p1 < "${V2_PATCH}" | tee "${ARTIFACT_DIR}/v2-patch-apply-job${SLURM_JOB_ID}.log"
require_hash "${TARGET_SOURCE}" "${EXPECTED_V2_SOURCE_SHA}" post-v2-warp-owned-pv-source
patch --dry-run --batch --fuzz=0 -d "${SRC}" -p1 < "${V3_PATCH}" | tee "${ARTIFACT_DIR}/v3-patch-dry-run-job${SLURM_JOB_ID}.log"
patch --batch --fuzz=0 -d "${SRC}" -p1 < "${V3_PATCH}" | tee "${ARTIFACT_DIR}/v3-patch-apply-job${SLURM_JOB_ID}.log"
require_hash "${TARGET_SOURCE}" "${EXPECTED_V3_SOURCE_SHA}" post-v3-padded-pv-source
grep -F 'float pv_contribution[kWarps][kTile][20];' "${TARGET_SOURCE}"
grep -F '                                20, wmma::mem_row_major);' "${TARGET_SOURCE}"
git -C "${SRC}" add -N csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
git -C "${SRC}" diff --check
git -C "${SRC}" diff --binary > "${ARTIFACT_DIR}/derived-plugin-v3-job${SLURM_JOB_ID}.patch"
cp --preserve=mode,timestamps "${TARGET_SOURCE}" "${ARTIFACT_DIR}/native_c2_decode.v3-padded-pv.cu"

# exact-d4 CUDA 13 accepts requested B300 10.3f, then its 10.0 family
# intersection emits a single native sm_100 cubin; it must contain no PTX.
cmake -S "${SRC}" -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DVLLM_TARGET_DEVICE=cuda -DVLLM_PYTHON_EXECUTABLE="${PYTHON}" \
  -DFETCHCONTENT_BASE_DIR="${DEPS}" -DCMAKE_INSTALL_PREFIX="${ARTIFACT_DIR}/install" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DNVCC_THREADS=2 \
  -DCMAKE_JOB_POOL_COMPILE:STRING=compile -DCMAKE_JOB_POOLS:STRING=compile=32 \
  2>&1 | tee "${ARTIFACT_DIR}/configure-job${SLURM_JOB_ID}.log"
grep -F 'CUDA target architectures: 10.3f' "${ARTIFACT_DIR}/configure-job${SLURM_JOB_ID}.log"
grep -F 'CUDA supported target architectures: 10.0' "${ARTIFACT_DIR}/configure-job${SLURM_JOB_ID}.log"
grep -F 'Building standalone native C2 plugin for archs: 10.0' "${ARTIFACT_DIR}/configure-job${SLURM_JOB_ID}.log"
grep -F native_c2_decode.cu "${BUILD_DIR}/compile_commands.json" | tee "${ARTIFACT_DIR}/native-c2-compile-command-job${SLURM_JOB_ID}.json"
grep -F -- '-gencode arch=compute_100,code=sm_100' "${ARTIFACT_DIR}/native-c2-compile-command-job${SLURM_JOB_ID}.json"
cmake --build "${BUILD_DIR}" --target _native_c2_msa_decode_plugin -- -j64 -v 2>&1 | tee "${ARTIFACT_DIR}/compile-job${SLURM_JOB_ID}.log"
mapfile -t PLUGINS < <(find "${BUILD_DIR}" -type f -name '_native_c2_msa_decode_plugin.abi3.so' -print)
[[ ${#PLUGINS[@]} -eq 1 ]] || { printf 'expected one DSO, got %s\n' "${#PLUGINS[@]}" >&2; exit 2; }
PLUGIN=${PLUGINS[0]}
install -D -m 0755 "${PLUGIN}" "${ARTIFACT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so"
grep -F -- '-gencode arch=compute_100,code=sm_100' "${ARTIFACT_DIR}/compile-job${SLURM_JOB_ID}.log"
nm -D --defined-only "${PLUGIN}" | tee "${ARTIFACT_DIR}/plugin-symbols-job${SLURM_JOB_ID}.txt"
! grep -q ' PyInit_' "${ARTIFACT_DIR}/plugin-symbols-job${SLURM_JOB_ID}.txt"
cuobjdump --list-elf "${PLUGIN}" | tee "${ARTIFACT_DIR}/plugin-cuobjdump-elf-job${SLURM_JOB_ID}.txt"
mapfile -t CUBINS < <(grep -E '^ELF file[[:space:]]+[0-9]+:.*\.cubin$' "${ARTIFACT_DIR}/plugin-cuobjdump-elf-job${SLURM_JOB_ID}.txt")
[[ ${#CUBINS[@]} -eq 1 && "${CUBINS[0]}" == *'.sm_100.cubin' ]] || { printf 'bad cubin list: %s\n' "${CUBINS[*]:-}" >&2; exit 2; }
! grep -Eqi 'sm_10(3|3f)|compute_10(3|3f)|\.ptx($|[^[:alnum:]_])' "${ARTIFACT_DIR}/plugin-cuobjdump-elf-job${SLURM_JOB_ID}.txt"
cuobjdump --list-ptx "${PLUGIN}" | tee "${ARTIFACT_DIR}/plugin-cuobjdump-ptx-job${SLURM_JOB_ID}.txt"
[[ ! -s "${ARTIFACT_DIR}/plugin-cuobjdump-ptx-job${SLURM_JOB_ID}.txt" ]] || { echo 'PTX fallback forbidden' >&2; exit 2; }
cuobjdump --dump-sass "${PLUGIN}" | tee "${ARTIFACT_DIR}/plugin-sass-job${SLURM_JOB_ID}.sass"
grep -m 80 -E 'native_c2_msa_decode_kernel|HMMA|MMA' "${ARTIFACT_DIR}/plugin-sass-job${SLURM_JOB_ID}.sass" | tee "${ARTIFACT_DIR}/plugin-sass-proof-job${SLURM_JOB_ID}.txt"
grep -q native_c2_msa_decode_kernel "${ARTIFACT_DIR}/plugin-sass-proof-job${SLURM_JOB_ID}.txt"
grep -Eq 'HMMA|MMA' "${ARTIFACT_DIR}/plugin-sass-proof-job${SLURM_JOB_ID}.txt"

sha256sum "${NATIVE_V1_SOURCE}" "${SCHEMA_PATCH}" "${CMAKE_PATCH}" "${DISPATCH_PATCH}" "${LOADER_PATCH}" "${V2_PATCH}" "${V3_PATCH}" "${TARGET_SOURCE}" "${PLUGIN}" "${ARTIFACT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so" > "${ARTIFACT_DIR}/plugin-v3-provenance-job${SLURM_JOB_ID}.sha256"
ldd "${PLUGIN}" | tee "${ARTIFACT_DIR}/plugin-ldd-job${SLURM_JOB_ID}.txt"
sha256sum "${ARTIFACT_DIR}"/*.log "${ARTIFACT_DIR}"/*.txt "${ARTIFACT_DIR}"/*.json "${ARTIFACT_DIR}"/*.patch "${ARTIFACT_DIR}"/*.sha256 "${ARTIFACT_DIR}"/*.cu > "${ARTIFACT_DIR}/build-job${SLURM_JOB_ID}.sha256"
