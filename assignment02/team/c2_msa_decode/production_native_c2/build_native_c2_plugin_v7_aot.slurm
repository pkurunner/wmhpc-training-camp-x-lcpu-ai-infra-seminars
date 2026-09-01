#!/usr/bin/env bash
# Build v7 only as a new standalone C2-plugin DSO.  It begins from the
# archived exact v6 source and prefetches each lane's eight original FP8 V
# elements before softmax.  The ABI, four-producer cluster topology, softmax,
# BF16 staging layout/conversion, PV WMMA, and merge protocol remain frozen.

#SBATCH --job-name=c2-native-plugin-v7-aot
#SBATCH --partition=cpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v7-aot-artifacts-20260830/slurm-%j.log

set -Eeuo pipefail

BASE=/home/lcpu/85117379/vllm-d4da0c5
SRC=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v7-v-prefetch-20260830-job${SLURM_JOB_ID}
BUILD_DIR=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v7-v-prefetch-build-20260830-job${SLURM_JOB_ID}
DEPS=/home/lcpu/85117379/c2-native-plugin-v7-aot-deps-20260830-job${SLURM_JOB_ID}
ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v7-aot-artifacts-20260830
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
V4_PATCH=/home/lcpu/85117379/native_c2_v4_four_producer_20260829.patch
V5_PATCH=/home/lcpu/85117379/native_c2_v5_warp_parallel_softmax_20260829.patch
V6_PATCH=/home/lcpu/85117379/native_c2_v6_register_numerator_20260830.patch
V6_ARCHIVED_SOURCE=/home/lcpu/85117379/c2-native-plugin-v6-aot-artifacts-20260830/job12278/native_c2_decode.v6-register-numerator.cu
V7_PATCH=/home/lcpu/85117379/native_c2_v7_v_prefetch_20260830.patch

EXPECTED_BASE_COMMIT=d4da0c55af3aa231b6209bf77871f3ed36eab0d2
EXPECTED_CUTLASS_COMMIT=da5e086dab31d63815acafdac9a9c5893b1c69e2
EXPECTED_MSA_COMMIT=80434d7f67877c6570ca19cac444b84bc9855dac
EXPECTED_MSA_CUTLASS_COMMIT=eb61c911471867a5fd2466bfd8f29306cea6ebf8
EXPECTED_NATIVE_V1_SOURCE_SHA=f67495586759b3b8f718dbbf60627c499f917842f2ed6ecf89d2344f1c82bc4b
EXPECTED_SCHEMA_PATCH_SHA=53e7cd4b09a8999010442934c863427c30b019fc549d7f0cf0f3d16ccc1e2e6e
EXPECTED_CMAKE_PATCH_SHA=5b145f36ce5f8b12dae71ce02e77d65545a29eefcb5a6045f591348bc325ac35
EXPECTED_DISPATCH_PATCH_SHA=1ca224ec4f94adf8ee97580a0e08321552224555570784c308c0edcda34bcd91
EXPECTED_LOADER_PATCH_SHA=9cb3a22216048083d7a4802ba700fa07fd892436151871dc6e909c77239adb33
EXPECTED_V2_PATCH_SHA=2fa34736ba80a50cd6d4a40377ad419709941349eff5c26e745541cfd8862816
EXPECTED_V4_PATCH_SHA=4547dc1be68f92a98acc960589f7a0d3d9efb89edb067d6b2f73a445cbefab17
EXPECTED_V5_PATCH_SHA=98d8a458f28d7aa69aae8e865de0651aba8394276fe7b0450fd11b2f4e7f8710
EXPECTED_V6_PATCH_SHA=1e0b1740f48307307f7a450f07226cc82f3a61f4147aca14ec8f0c9784eb8fd5
EXPECTED_V7_PATCH_SHA=b0c02002a3577a0245d48fbc29ef167da0f1ad9960c2ed1b987a2249305c66ee
EXPECTED_PLUGIN_V1_SOURCE_SHA=8fee086f8ba4f5b2710347025211f9733ce64c61b04e44e710016d59b336f406
EXPECTED_V2_SOURCE_SHA=f2aae9c9c62e646b3cd38b136f224009d400298c21a99b9b1cf5a402f944acc4
EXPECTED_V4_SOURCE_SHA=96c0af14ed68261ada954688c1379d4a9d8a541f84997ef39eb12f373ea7f4c9
EXPECTED_V5_SOURCE_SHA=6409a54689ebbdd1f72fad76c2d72dfe588f79d3b8dd6c4dbaf1df5b630f0931
EXPECTED_V6_SOURCE_SHA=c7f05928fd2ba3a0e29d3154f2bf56849e9d7ad6fd3eddc24b0890722e117f02
EXPECTED_V7_SOURCE_SHA=1d638ce66c3b17b2b83298fa98f44770b14b94154c0f56cf5ff84d508a12e9a3
EXPECTED_PLUGIN_CMAKE_SHA=6662b0400eb04c590b56c15346848b4a26c45d86e796a69e1bcb4a0d91523734
EXPECTED_PLUGIN_ADAPTER_SHA=6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412

for input in "${NATIVE_V1_SOURCE}" "${SCHEMA_PATCH}" "${CMAKE_PATCH}" \
  "${DISPATCH_PATCH}" "${LOADER_PATCH}" "${V2_PATCH}" "${V4_PATCH}" \
  "${V5_PATCH}" "${V6_PATCH}" "${V6_ARCHIVED_SOURCE}" "${V7_PATCH}"; do
  [[ -f "${input}" ]] || { echo "missing input: ${input}" >&2; exit 2; }
done
[[ -d "${ARTIFACT_ROOT}" && ! -e "${SRC}" && ! -e "${BUILD_DIR}" && \
   ! -e "${DEPS}" && ! -e "${ARTIFACT_DIR}" ]] || {
  echo 'artifact root missing or job-scoped output already exists' >&2; exit 2;
}
mkdir -p "${ARTIFACT_DIR}" "${DEPS}"
DRIVER_LOG=${ARTIFACT_DIR}/build-driver-job${SLURM_JOB_ID}.log
OUTPUTS_MANIFEST=${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256
FINAL_STATUS=${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt
FINAL_STATUS_SIDECAR=${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.sha256
exec 3>&1 4>&2
exec > >(tee "${DRIVER_LOG}") 2>&1
DRIVER_TEE_PID=$!
finalize() {
  local body_rc=$?
  trap - EXIT
  set +e
  printf 'BODY_RC=%s UTC=%s FINALIZING=1\n' "${body_rc}" "$(date -u +%FT%TZ)"

  # Close and wait for the process-substitution tee before hashing its log.
  # The final status has its own sidecar, avoiding a self-referential manifest.
  exec 1>&3 2>&4
  exec 3>&- 4>&-
  wait "${DRIVER_TEE_PID}"
  local tee_rc=$?
  local manifest_rc=0
  local -a outputs=()
  mapfile -d '' -t outputs < <(
    find "${ARTIFACT_DIR}" -type f \
      ! -name "$(basename "${OUTPUTS_MANIFEST}")" \
      ! -name "$(basename "${FINAL_STATUS}")" \
      ! -name "$(basename "${FINAL_STATUS_SIDECAR}")" \
      -print0 | sort -z
  )
  if (( ${#outputs[@]} == 0 )); then
    manifest_rc=1
  else
    sha256sum "${outputs[@]}" > "${OUTPUTS_MANIFEST}" || manifest_rc=$?
    if (( manifest_rc == 0 )); then
      sha256sum -c "${OUTPUTS_MANIFEST}" >/dev/null || manifest_rc=$?
    fi
  fi

  local final_rc=${body_rc}
  if (( final_rc == 0 && (tee_rc != 0 || manifest_rc != 0) )); then
    final_rc=1
  fi
  printf 'BODY_RC=%s TEE_RC=%s MANIFEST_RC=%s FINAL_RC=%s UTC=%s\n' \
    "${body_rc}" "${tee_rc}" "${manifest_rc}" "${final_rc}" \
    "$(date -u +%FT%TZ)" > "${FINAL_STATUS}"
  sha256sum "${FINAL_STATUS}" > "${FINAL_STATUS_SIDECAR}" || {
    (( final_rc == 0 )) && final_rc=1
  }
  exit "${final_rc}"
}
trap finalize EXIT

export PATH="${TOOLS}/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin"
export TORCH_CUDA_ARCH_LIST=10.3f
export VLLM_CUTLASS_SRC_DIR="${CUTLASS}"
export FMHA_SM100_SRC_DIR="${MSA}"
sha256_of() { sha256sum "$1" | awk '{print $1}'; }
require_hash() { local actual; actual=$(sha256_of "$1"); printf '%s SHA256=%s path=%s\n' "$3" "${actual}" "$1"; [[ "${actual}" == "$2" ]] || { echo "$3 hash mismatch" >&2; exit 2; }; }

printf 'UTC=%s JOB=%s HOST=%s CPUS=%s\n' "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}" "$(hostname)" "${SLURM_CPUS_PER_TASK}"
[[ "$(git -C "${BASE}" rev-parse HEAD)" == "${EXPECTED_BASE_COMMIT}" && -z "$(git -C "${BASE}" status --porcelain)" ]] || { echo 'exact d4 base not clean' >&2; exit 2; }
[[ "$(git -C "${CUTLASS}" rev-parse HEAD)" == "${EXPECTED_CUTLASS_COMMIT}" && -z "$(git -C "${CUTLASS}" status --porcelain)" ]] || { echo 'CUTLASS dependency identity/cleanliness mismatch' >&2; exit 2; }
[[ "$(git -C "${MSA}" rev-parse HEAD)" == "${EXPECTED_MSA_COMMIT}" && -z "$(git -C "${MSA}" status --porcelain)" ]] || { echo 'MSA dependency identity/cleanliness mismatch' >&2; exit 2; }
[[ "$(git -C "${MSA}/python/fmha_sm100/cutlass" rev-parse HEAD)" == "${EXPECTED_MSA_CUTLASS_COMMIT}" && -z "$(git -C "${MSA}/python/fmha_sm100/cutlass" status --porcelain)" ]] || { echo 'nested MSA CUTLASS identity/cleanliness mismatch' >&2; exit 2; }
printf '%s\n' "${EXPECTED_BASE_COMMIT}" > "${ARTIFACT_DIR}/base-commit-job${SLURM_JOB_ID}.txt"
require_hash "${NATIVE_V1_SOURCE}" "${EXPECTED_NATIVE_V1_SOURCE_SHA}" native-v1-source
require_hash "${SCHEMA_PATCH}" "${EXPECTED_SCHEMA_PATCH_SHA}" schema-patch
require_hash "${CMAKE_PATCH}" "${EXPECTED_CMAKE_PATCH_SHA}" cmake-patch
require_hash "${DISPATCH_PATCH}" "${EXPECTED_DISPATCH_PATCH_SHA}" dispatch-patch
require_hash "${LOADER_PATCH}" "${EXPECTED_LOADER_PATCH_SHA}" loader-patch
require_hash "${V2_PATCH}" "${EXPECTED_V2_PATCH_SHA}" warp-owned-pv-v2-patch
require_hash "${V4_PATCH}" "${EXPECTED_V4_PATCH_SHA}" four-producer-v4-patch
require_hash "${V5_PATCH}" "${EXPECTED_V5_PATCH_SHA}" warp-parallel-softmax-v5-patch
require_hash "${V6_PATCH}" "${EXPECTED_V6_PATCH_SHA}" register-numerator-v6-patch
require_hash "${V6_ARCHIVED_SOURCE}" "${EXPECTED_V6_SOURCE_SHA}" archived-v6-register-numerator-source
require_hash "${V7_PATCH}" "${EXPECTED_V7_PATCH_SHA}" v-prefetch-v7-patch
{
  printf 'BASE_COMMIT=%s\n' "${EXPECTED_BASE_COMMIT}"
  git -C "${CUTLASS}" rev-parse HEAD | sed 's/^/CUTLASS_COMMIT=/'
  git -C "${MSA}" rev-parse HEAD | sed 's/^/MSA_COMMIT=/'
  git -C "${MSA}/python/fmha_sm100/cutlass" rev-parse HEAD | sed 's/^/MSA_CUTLASS_COMMIT=/'
  sha256sum "${NATIVE_V1_SOURCE}" "${SCHEMA_PATCH}" "${CMAKE_PATCH}" \
    "${DISPATCH_PATCH}" "${LOADER_PATCH}" "${V2_PATCH}" "${V4_PATCH}" \
    "${V5_PATCH}" "${V6_PATCH}" "${V6_ARCHIVED_SOURCE}" "${V7_PATCH}"
} > "${ARTIFACT_DIR}/input-provenance-job${SLURM_JOB_ID}.txt"
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
patch --dry-run --batch --fuzz=0 -d "${SRC}" -p1 < "${V4_PATCH}" | tee "${ARTIFACT_DIR}/v4-patch-dry-run-job${SLURM_JOB_ID}.log"
patch --batch --fuzz=0 -d "${SRC}" -p1 < "${V4_PATCH}" | tee "${ARTIFACT_DIR}/v4-patch-apply-job${SLURM_JOB_ID}.log"
require_hash "${TARGET_SOURCE}" "${EXPECTED_V4_SOURCE_SHA}" post-v4-four-producer-source
patch --dry-run --batch --fuzz=0 -d "${SRC}" -p1 < "${V5_PATCH}" | tee "${ARTIFACT_DIR}/v5-patch-dry-run-job${SLURM_JOB_ID}.log"
patch --batch --fuzz=0 -d "${SRC}" -p1 < "${V5_PATCH}" | tee "${ARTIFACT_DIR}/v5-patch-apply-job${SLURM_JOB_ID}.log"
require_hash "${TARGET_SOURCE}" "${EXPECTED_V5_SOURCE_SHA}" post-v5-warp-parallel-softmax-source
patch --dry-run --batch --fuzz=0 -d "${SRC}" -p1 < "${V6_PATCH}" | tee "${ARTIFACT_DIR}/v6-patch-dry-run-job${SLURM_JOB_ID}.log"
patch --batch --fuzz=0 -d "${SRC}" -p1 < "${V6_PATCH}" | tee "${ARTIFACT_DIR}/v6-patch-apply-job${SLURM_JOB_ID}.log"
require_hash "${TARGET_SOURCE}" "${EXPECTED_V6_SOURCE_SHA}" post-v6-register-numerator-source
cmp -s "${TARGET_SOURCE}" "${V6_ARCHIVED_SOURCE}" || { echo 'reconstructed v6 source differs from archived job12278 source' >&2; exit 2; }
patch --dry-run --batch --fuzz=0 -d "${SRC}" -p1 < "${V7_PATCH}" | tee "${ARTIFACT_DIR}/v7-patch-dry-run-job${SLURM_JOB_ID}.log"
patch --batch --fuzz=0 -d "${SRC}" -p1 < "${V7_PATCH}" | tee "${ARTIFACT_DIR}/v7-patch-apply-job${SLURM_JOB_ID}.log"
require_hash "${TARGET_SOURCE}" "${EXPECTED_V7_SOURCE_SHA}" post-v7-v-prefetch-source
grep -F 'constexpr int kProducerCtas = 4;' "${TARGET_SOURCE}"
grep -F 'constexpr int kMergeCta = 0;' "${TARGET_SOURCE}"
grep -F 'constexpr int kMbarrierArrivals = kProducerCtas;' "${TARGET_SOURCE}"
grep -F 'remote_partial3' "${TARGET_SOURCE}"
grep -F 'float numerator_accumulator[kTile / 2] = {};' "${TARGET_SOURCE}"
grep -F '__nv_fp8_e4m3 value_prefetch[kTile / 2];' "${TARGET_SOURCE}"
grep -F 'static_cast<float>(value_prefetch[element / kWarpSize]) *' "${TARGET_SOURCE}"
! grep -F '__shared__ __align__(32) float numerator[' "${TARGET_SOURCE}"
git -C "${SRC}" add -N csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
git -C "${SRC}" diff --check
git -C "${SRC}" diff --binary > "${ARTIFACT_DIR}/derived-plugin-v7-job${SLURM_JOB_ID}.patch"
cp --preserve=mode,timestamps "${TARGET_SOURCE}" "${ARTIFACT_DIR}/native_c2_decode.v7-v-prefetch.cu"

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
"${PYTHON}" - "${BUILD_DIR}/compile_commands.json" \
  "${ARTIFACT_DIR}/native-c2-compile-command-job${SLURM_JOB_ID}.json" \
  "${TARGET_SOURCE}" <<'PY'
import json
import sys
from pathlib import Path

database_path, output_path, source_path = map(Path, sys.argv[1:])
database = json.loads(database_path.read_text(encoding="utf-8"))
source = source_path.resolve()
matches = [entry for entry in database if Path(entry["file"]).resolve() == source]
assert len(matches) == 1, (source, len(matches))
output_path.write_text(
    json.dumps(matches[0], indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
grep -F -- '-gencode arch=compute_100,code=sm_100' "${ARTIFACT_DIR}/native-c2-compile-command-job${SLURM_JOB_ID}.json"
cmake --build "${BUILD_DIR}" --target _native_c2_msa_decode_plugin -- -j64 -v 2>&1 | tee "${ARTIFACT_DIR}/compile-job${SLURM_JOB_ID}.log"
mapfile -t PLUGINS < <(find "${BUILD_DIR}" -type f -name '_native_c2_msa_decode_plugin.abi3.so' -print)
[[ ${#PLUGINS[@]} -eq 1 ]] || { printf 'expected one DSO, got %s\n' "${#PLUGINS[@]}" >&2; exit 2; }
PLUGIN=${PLUGINS[0]}
install -D -m 0755 "${PLUGIN}" "${ARTIFACT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so"
grep -F -- '-gencode arch=compute_100,code=sm_100' "${ARTIFACT_DIR}/compile-job${SLURM_JOB_ID}.log"
cuobjdump --dump-resource-usage "${PLUGIN}" | tee "${ARTIFACT_DIR}/plugin-resource-usage-job${SLURM_JOB_ID}.txt"
grep -A1 'native_c2_msa_decode_kernel' "${ARTIFACT_DIR}/plugin-resource-usage-job${SLURM_JOB_ID}.txt" | tee "${ARTIFACT_DIR}/plugin-resource-proof-job${SLURM_JOB_ID}.txt"
RESOURCE_LINE=$(grep -m 1 -E 'REG:[[:space:]]*[0-9]+.*STACK:[[:space:]]*[0-9]+.*SHARED:[[:space:]]*[0-9]+.*LOCAL:[[:space:]]*[0-9]+' "${ARTIFACT_DIR}/plugin-resource-proof-job${SLURM_JOB_ID}.txt")
[[ -n "${RESOURCE_LINE}" ]] || { echo 'native kernel resource line missing' >&2; exit 2; }
read_resource() {
  local label=$1
  sed -nE "s/.*${label}:[[:space:]]*([0-9]+).*/\\1/p" <<< "${RESOURCE_LINE}"
}
REG=$(read_resource REG)
STACK=$(read_resource STACK)
SHARED=$(read_resource SHARED)
LOCAL=$(read_resource LOCAL)
[[ "${REG}" =~ ^[0-9]+$ && "${STACK}" =~ ^[0-9]+$ && "${SHARED}" =~ ^[0-9]+$ && "${LOCAL}" =~ ^[0-9]+$ ]] || { echo "unparseable resource line: ${RESOURCE_LINE}" >&2; exit 2; }
printf 'RESOURCE_LINE=%s\nREG=%s\nSTACK=%s\nSHARED=%s\nLOCAL=%s\n' "${RESOURCE_LINE}" "${REG}" "${STACK}" "${SHARED}" "${LOCAL}" | tee "${ARTIFACT_DIR}/plugin-resource-gate-job${SLURM_JOB_ID}.txt"
(( REG <= 72 && STACK == 0 && SHARED == 30560 && LOCAL == 0 )) || { echo "v7 resource gate failed: REG=${REG} STACK=${STACK} SHARED=${SHARED} LOCAL=${LOCAL}" >&2; exit 2; }
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

sha256sum "${NATIVE_V1_SOURCE}" "${SCHEMA_PATCH}" "${CMAKE_PATCH}" "${DISPATCH_PATCH}" "${LOADER_PATCH}" "${V2_PATCH}" "${V4_PATCH}" "${V5_PATCH}" "${V6_PATCH}" "${V6_ARCHIVED_SOURCE}" "${V7_PATCH}" "${TARGET_SOURCE}" "${PLUGIN}" "${ARTIFACT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so" > "${ARTIFACT_DIR}/plugin-v7-provenance-job${SLURM_JOB_ID}.sha256"
ldd "${PLUGIN}" | tee "${ARTIFACT_DIR}/plugin-ldd-job${SLURM_JOB_ID}.txt"
