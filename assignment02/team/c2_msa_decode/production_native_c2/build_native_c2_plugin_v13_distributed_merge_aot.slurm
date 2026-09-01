#!/usr/bin/env bash
# Build the v13 distributed-merge candidate as a standalone C2-plugin DSO.
# It restores the accepted v12 Q-row-padding source by its immutable SHA-256
# and applies only the reviewed v13 distributed-merge delta with fuzz=0.

#SBATCH --job-name=c2-native-plugin-v13-aot
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/slurm-c2-native-plugin-v13-aot-%j.log

set -Eeuo pipefail

: "${C2_EXPECTED_SCRIPT_SHA:?submit with the exact reviewed v13 AOT script SHA-256}"
[[ "${C2_EXPECTED_SCRIPT_SHA}" =~ ^[0-9a-f]{64}$ ]]

BASE=/home/lcpu/85117379/vllm-d4da0c5
# Clone, FetchContent payloads, and the build tree are disposable and large.
# Keep them on node-local scratch so a dependency clone cannot consume the
# durable evidence quota and break the tee/finalizer pipe.
RUNTIME=/tmp/c2-native-plugin-v13-distributed-merge-aot-job${SLURM_JOB_ID}
SRC=${RUNTIME}/src
BUILD_DIR=${RUNTIME}/build
DEPS=${RUNTIME}/deps
ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v13-distributed-merge-aot-r3-artifacts-20260831
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID}
TOOLS=/home/lcpu/85117379/c2-native-tools-20260829
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
CUTLASS=/home/lcpu/85117379/cutlass-v4.4.2-c2-20260829
MSA=/home/lcpu/85117379/msa-official
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")

NATIVE_V1_SOURCE=/home/lcpu/85117379/native_c2_decode.plugin-v1.cu
SCHEMA_PATCH=/home/lcpu/85117379/native_c2_plugin_schema_20260829.patch
CMAKE_PATCH=/home/lcpu/85117379/exact_d4_native_c2_plugin_cmake_20260829.patch
DISPATCH_PATCH=/home/lcpu/85117379/exact_d4_python_dispatch_20260829.patch
LOADER_PATCH=/home/lcpu/85117379/exact_d4_native_c2_plugin_python_loader_20260829.patch
V12_BASELINE_SOURCE=/home/lcpu/85117379/c2-native-plugin-v12-aot-artifacts-20260830/job12983/native_c2_decode.v12-q-row-padding.cu
V13_PATCH=/home/lcpu/85117379/native_c2_v13_distributed_merge_20260831.patch

EXPECTED_BASE_COMMIT=d4da0c55af3aa231b6209bf77871f3ed36eab0d2
EXPECTED_CUTLASS_COMMIT=da5e086dab31d63815acafdac9a9c5893b1c69e2
EXPECTED_MSA_COMMIT=80434d7f67877c6570ca19cac444b84bc9855dac
EXPECTED_MSA_CUTLASS_COMMIT=eb61c911471867a5fd2466bfd8f29306cea6ebf8
EXPECTED_NATIVE_V1_SOURCE_SHA=f67495586759b3b8f718dbbf60627c499f917842f2ed6ecf89d2344f1c82bc4b
EXPECTED_SCHEMA_PATCH_SHA=53e7cd4b09a8999010442934c863427c30b019fc549d7f0cf0f3d16ccc1e2e6e
EXPECTED_CMAKE_PATCH_SHA=5b145f36ce5f8b12dae71ce02e77d65545a29eefcb5a6045f591348bc325ac35
EXPECTED_DISPATCH_PATCH_SHA=1ca224ec4f94adf8ee97580a0e08321552224555570784c308c0edcda34bcd91
EXPECTED_LOADER_PATCH_SHA=9cb3a22216048083d7a4802ba700fa07fd892436151871dc6e909c77239adb33
EXPECTED_PLUGIN_V1_SOURCE_SHA=8fee086f8ba4f5b2710347025211f9733ce64c61b04e44e710016d59b336f406
EXPECTED_PLUGIN_CMAKE_SHA=6662b0400eb04c590b56c15346848b4a26c45d86e796a69e1bcb4a0d91523734
EXPECTED_PLUGIN_ADAPTER_SHA=6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412
EXPECTED_V12_BASELINE_SOURCE_SHA=535d90b856ed1062aa7b8a105eb2c5f236c450826e65496e646d0d5a27eb8aaf
EXPECTED_V13_PATCH_SHA=6efdebfdfde202cac6af66fd008afb32028b31291e276cf32e5566f57ed36016
EXPECTED_V13_SOURCE_SHA=def42508c28eef50995c33b70ebddba31a82e932764840c6e4ff1714b9c7f063

[[ -d "${ARTIFACT_ROOT}" && ! -L "${ARTIFACT_ROOT}" && \
   "$(readlink -f -- "${ARTIFACT_ROOT}")" == "${ARTIFACT_ROOT}" && \
   "$(stat -c %u -- "${ARTIFACT_ROOT}")" == "$(id -u)" ]] || {
  echo 'artifact root missing, linked, noncanonical, or not owned by this user' >&2; exit 2;
}
[[ -z "$(find "${ARTIFACT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
  echo 'artifact root must be empty before v13 AOT submission' >&2; exit 2;
}
[[ "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]
[[ "${RUNTIME}" == "/tmp/c2-native-plugin-v13-distributed-merge-aot-job${SLURM_JOB_ID}" ]]
[[ ! -e "${RUNTIME}" && ! -L "${RUNTIME}" && ! -e "${ARTIFACT_DIR}" ]] || {
  echo 'job-scoped output already exists' >&2; exit 2;
}
mkdir -m 0700 -- "${ARTIFACT_DIR}"
DRIVER_LOG=${ARTIFACT_DIR}/build-driver-job${SLURM_JOB_ID}.log
OUTPUTS_MANIFEST=${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256
FINAL_STATUS=${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt
FINAL_STATUS_SIDECAR=${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.sha256
SUBMITTED_SCRIPT_COPY=${ARTIFACT_DIR}/submitted-script-job${SLURM_JOB_ID}.slurm
cleanup_runtime() {
  local runtime=$1
  [[ "${runtime}" == "/tmp/c2-native-plugin-v13-distributed-merge-aot-job${SLURM_JOB_ID}" ]] || return 1
  if [[ ! -e "${runtime}" && ! -L "${runtime}" ]]; then
    return 0
  fi
  [[ -d "${runtime}" && ! -L "${runtime}" && \
     "$(readlink -f -- "${runtime}")" == "${runtime}" && \
     "$(stat -c %u -- "${runtime}")" == "$(id -u)" ]] || return 1
  rm -rf -- "${runtime}" || return 1
  [[ ! -e "${runtime}" && ! -L "${runtime}" ]]
}
setup_finalize() {
  local body_rc=$?
  local cleanup_rc=0 manifest_rc=0
  local setup_diag=${ARTIFACT_DIR}/setup-finalizer-diagnostics-job${SLURM_JOB_ID}.txt
  trap - EXIT
  set +e
  cleanup_runtime "${RUNTIME}" || cleanup_rc=$?
  printf 'SETUP_FINALIZER=1 BODY_RC=%s CLEANUP_RC=%s UTC=%s\n' \
    "${body_rc}" "${cleanup_rc}" "$(date -u +%FT%TZ)" > "${setup_diag}"
  local -a setup_outputs=()
  [[ -f "${SUBMITTED_SCRIPT_COPY}" ]] && setup_outputs+=("${SUBMITTED_SCRIPT_COPY}")
  [[ -f "${setup_diag}" ]] && setup_outputs+=("${setup_diag}")
  if [[ "${#setup_outputs[@]}" -gt 0 ]]; then
    sha256sum "${setup_outputs[@]}" > "${OUTPUTS_MANIFEST}" && sha256sum -c "${OUTPUTS_MANIFEST}" >/dev/null || manifest_rc=1
  else
    manifest_rc=1
  fi
  printf 'BODY_RC=%s TEE_RC=0 MANIFEST_RC=%s CLEANUP_RC=%s SETUP_FAILURE=1 FINAL_RC=2 UTC=%s\n' \
    "${body_rc}" "${manifest_rc}" "${cleanup_rc}" "$(date -u +%FT%TZ)" > "${FINAL_STATUS}"
  sha256sum "${FINAL_STATUS}" > "${FINAL_STATUS_SIDECAR}"
  exit 2
}
trap setup_finalize EXIT

for input in "${SCRIPT_PATH}" "${NATIVE_V1_SOURCE}" "${SCHEMA_PATCH}" \
  "${CMAKE_PATCH}" "${DISPATCH_PATCH}" "${LOADER_PATCH}" \
  "${V12_BASELINE_SOURCE}" "${V13_PATCH}"; do
  [[ -f "${input}" && ! -L "${input}" ]] || { echo "missing or linked input: ${input}" >&2; exit 2; }
done
mkdir -m 0700 -- "${RUNTIME}"
[[ -d "${RUNTIME}" && ! -L "${RUNTIME}" && \
   "$(readlink -f -- "${RUNTIME}")" == "${RUNTIME}" && \
   "$(stat -c %u -- "${RUNTIME}")" == "$(id -u)" ]]
mkdir -m 0700 -- "${DEPS}"
printf '%s  %s\n' "${C2_EXPECTED_SCRIPT_SHA}" "${SCRIPT_PATH}" | sha256sum -c - || exit 2
cp -- "${SCRIPT_PATH}" "${SUBMITTED_SCRIPT_COPY}" || exit 2
chmod a-w "${SUBMITTED_SCRIPT_COPY}" || exit 2
printf '%s  %s\n' "${C2_EXPECTED_SCRIPT_SHA}" "${SUBMITTED_SCRIPT_COPY}" | sha256sum -c - || exit 2
exec 3>&1 4>&2
exec > >(tee "${DRIVER_LOG}") 2>&1
DRIVER_TEE_PID=$!
finalize() {
  local body_rc=$?
  trap - EXIT
  set +e
  printf 'BODY_RC=%s UTC=%s FINALIZING=1\n' "${body_rc}" "$(date -u +%FT%TZ)"

  # Hash the completed tee log, then construct a NUL-safe recursive manifest.
  # Status and sidecar stay outside that manifest to avoid self-reference.
  exec 1>&3 2>&4
  exec 3>&- 4>&-
  wait "${DRIVER_TEE_PID}"
  local tee_rc=$?
  local cleanup_rc=0 manifest_rc=0
  cleanup_runtime "${RUNTIME}" || cleanup_rc=$?
  local -a outputs=()
  local output_list="${ARTIFACT_DIR}/outputs-list-job${SLURM_JOB_ID}.nul"
  if ! (set -o pipefail; find "${ARTIFACT_DIR}" -type f \
      ! -name "$(basename "${OUTPUTS_MANIFEST}")" \
      ! -name "$(basename "${FINAL_STATUS}")" \
      ! -name "$(basename "${FINAL_STATUS_SIDECAR}")" \
      ! -name "$(basename "${output_list}")" \
      -print0 | sort -z > "${output_list}"); then
    manifest_rc=1
  elif ! mapfile -d '' -t outputs < "${output_list}"; then
    manifest_rc=1
  fi
  if ! rm -f -- "${output_list}" || [[ -e "${output_list}" ]]; then
    manifest_rc=1
  fi
  if (( manifest_rc == 0 && ${#outputs[@]} == 0 )); then
    manifest_rc=1
  fi
  if (( manifest_rc == 0 )); then
    sha256sum "${outputs[@]}" > "${OUTPUTS_MANIFEST}" || manifest_rc=$?
    if (( manifest_rc == 0 )); then
      sha256sum -c "${OUTPUTS_MANIFEST}" >/dev/null || manifest_rc=$?
    fi
  fi
  local final_rc=${body_rc}
  if (( final_rc == 0 && (tee_rc != 0 || cleanup_rc != 0 || manifest_rc != 0) )); then
    final_rc=1
  fi
  write_final_status() {
    local tmp_status="${FINAL_STATUS}.tmp.$$"
    if ! printf 'BODY_RC=%s TEE_RC=%s CLEANUP_RC=%s MANIFEST_RC=%s FINAL_RC=%s UTC=%s\n' \
      "${body_rc}" "${tee_rc}" "${cleanup_rc}" "${manifest_rc}" "${final_rc}" \
      "$(date -u +%FT%TZ)" > "${tmp_status}"; then
      rm -f -- "${tmp_status}"
      return 1
    fi
    mv -f -- "${tmp_status}" "${FINAL_STATUS}"
  }
  write_final_status_sidecar() {
    local tmp_sidecar="${FINAL_STATUS_SIDECAR}.tmp.$$"
    if ! sha256sum "${FINAL_STATUS}" > "${tmp_sidecar}"; then
      rm -f -- "${tmp_sidecar}"
      return 1
    fi
    mv -f -- "${tmp_sidecar}" "${FINAL_STATUS_SIDECAR}"
  }
  if ! write_final_status; then
    final_rc=1
    if ! write_final_status; then
      rm -f -- "${FINAL_STATUS}" "${FINAL_STATUS_SIDECAR}"
      exit "${final_rc}"
    fi
  fi
  if ! write_final_status_sidecar; then
    final_rc=1
    if ! write_final_status || ! write_final_status_sidecar; then
      rm -f -- "${FINAL_STATUS}" "${FINAL_STATUS_SIDECAR}"
    fi
  fi
  exit "${final_rc}"
}
trap finalize EXIT

export PATH="${TOOLS}/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin"
export TORCH_CUDA_ARCH_LIST=10.3f
export VLLM_CUTLASS_SRC_DIR="${CUTLASS}"
export FMHA_SM100_SRC_DIR="${MSA}"
sha256_of() { sha256sum "$1" | awk '{print $1}'; }
require_hash() {
  local actual
  actual=$(sha256_of "$1")
  printf '%s SHA256=%s path=%s\n' "$3" "${actual}" "$1"
  [[ "${actual}" == "$2" ]] || { echo "$3 hash mismatch" >&2; exit 2; }
}

printf 'UTC=%s JOB=%s HOST=%s CPUS=%s\n' "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}" "$(hostname)" "${SLURM_CPUS_PER_TASK}"
[[ "$(git -C "${BASE}" rev-parse HEAD)" == "${EXPECTED_BASE_COMMIT}" && -z "$(git -C "${BASE}" status --porcelain)" ]] || { echo 'exact d4 base not clean' >&2; exit 2; }
[[ "$(git -C "${CUTLASS}" rev-parse HEAD)" == "${EXPECTED_CUTLASS_COMMIT}" && -z "$(git -C "${CUTLASS}" status --porcelain)" ]] || { echo 'CUTLASS dependency identity/cleanliness mismatch' >&2; exit 2; }
[[ "$(git -C "${MSA}" rev-parse HEAD)" == "${EXPECTED_MSA_COMMIT}" && -z "$(git -C "${MSA}" status --porcelain)" ]] || { echo 'MSA dependency identity/cleanliness mismatch' >&2; exit 2; }
[[ "$(git -C "${MSA}/python/fmha_sm100/cutlass" rev-parse HEAD)" == "${EXPECTED_MSA_CUTLASS_COMMIT}" && -z "$(git -C "${MSA}/python/fmha_sm100/cutlass" status --porcelain)" ]] || { echo 'nested MSA CUTLASS identity/cleanliness mismatch' >&2; exit 2; }
printf '%s\n' "${EXPECTED_BASE_COMMIT}" > "${ARTIFACT_DIR}/base-commit-job${SLURM_JOB_ID}.txt"
require_hash "${SCRIPT_PATH}" "${C2_EXPECTED_SCRIPT_SHA}" v13-aot-submission-script
require_hash "${SUBMITTED_SCRIPT_COPY}" "${C2_EXPECTED_SCRIPT_SHA}" v13-aot-durable-submitted-script
require_hash "${NATIVE_V1_SOURCE}" "${EXPECTED_NATIVE_V1_SOURCE_SHA}" native-v1-source
require_hash "${SCHEMA_PATCH}" "${EXPECTED_SCHEMA_PATCH_SHA}" schema-patch
require_hash "${CMAKE_PATCH}" "${EXPECTED_CMAKE_PATCH_SHA}" cmake-patch
require_hash "${DISPATCH_PATCH}" "${EXPECTED_DISPATCH_PATCH_SHA}" dispatch-patch
require_hash "${LOADER_PATCH}" "${EXPECTED_LOADER_PATCH_SHA}" loader-patch
require_hash "${V12_BASELINE_SOURCE}" "${EXPECTED_V12_BASELINE_SOURCE_SHA}" accepted-v12-q-row-padding-source
require_hash "${V13_PATCH}" "${EXPECTED_V13_PATCH_SHA}" v13-distributed-merge-patch
{
  printf 'BASE_COMMIT=%s\n' "${EXPECTED_BASE_COMMIT}"
  printf 'AOT_SCRIPT_EXPECTED_SHA256=%s\n' "${C2_EXPECTED_SCRIPT_SHA}"
  git -C "${CUTLASS}" rev-parse HEAD | sed 's/^/CUTLASS_COMMIT=/'
  git -C "${MSA}" rev-parse HEAD | sed 's/^/MSA_COMMIT=/'
  git -C "${MSA}/python/fmha_sm100/cutlass" rev-parse HEAD | sed 's/^/MSA_CUTLASS_COMMIT=/'
  sha256sum "${SUBMITTED_SCRIPT_COPY}" "${NATIVE_V1_SOURCE}" "${SCHEMA_PATCH}" \
    "${CMAKE_PATCH}" "${DISPATCH_PATCH}" "${LOADER_PATCH}" \
    "${V12_BASELINE_SOURCE}" "${V13_PATCH}"
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
require_hash "${TARGET_SOURCE}" "${EXPECTED_PLUGIN_V1_SOURCE_SHA}" plugin-v1-source-before-frozen-v12
require_hash "${SRC}/CMakeLists.txt" "${EXPECTED_PLUGIN_CMAKE_SHA}" plugin-cmake
require_hash "${SRC}/vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py" "${EXPECTED_PLUGIN_ADAPTER_SHA}" plugin-adapter
"${PYTHON}" -m py_compile "${SRC}/vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py"

# The accepted v12 source is authoritative.  Historical v9/v11/v12 patch
# chains are deliberately not inputs to this job: the sole kernel baseline is
# the immutable v12 source, followed by exactly one v13 patch.
install -m 0644 "${V12_BASELINE_SOURCE}" "${TARGET_SOURCE}"
require_hash "${TARGET_SOURCE}" "${EXPECTED_V12_BASELINE_SOURCE_SHA}" restored-accepted-v12-q-row-padding-source
patch --dry-run --batch --fuzz=0 -d "${SRC}" -p1 < "${V13_PATCH}" | tee "${ARTIFACT_DIR}/v13-patch-dry-run-job${SLURM_JOB_ID}.log"
patch --batch --fuzz=0 -d "${SRC}" -p1 < "${V13_PATCH}" | tee "${ARTIFACT_DIR}/v13-patch-apply-job${SLURM_JOB_ID}.log"
require_hash "${TARGET_SOURCE}" "${EXPECTED_V13_SOURCE_SHA}" post-v13-distributed-merge-source

# Structural drift alarms.  Directed correctness is still the semantic gate.
grep -F 'constexpr int kProducerCtas = 4;' "${TARGET_SOURCE}"
grep -F 'constexpr int kMbarrierArrivals = kProducerCtas;' "${TARGET_SOURCE}"
grep -F 'constexpr int kMergeHeadsPerCta = kGqaGroup / kClusterCtas;' "${TARGET_SOURCE}"
grep -F 'constexpr int kMergeElementsPerCta =' "${TARGET_SOURCE}"
grep -F 'static_assert((kGqaGroup * kHeadDim) % kClusterCtas == 0);' "${TARGET_SOURCE}"
grep -F 'static_assert(kMergeHeadsPerCta * kClusterCtas == kGqaGroup);' "${TARGET_SOURCE}"
grep -F 'static_assert(kMergeElementsPerCta == kMergeHeadsPerCta * kHeadDim);' "${TARGET_SOURCE}"
grep -F 'static_assert(kMergeElementsPerCta % kThreads == 0);' "${TARGET_SOURCE}"
grep -F '__shared__ float merge_weights[kMergeHeadsPerCta][kProducerCtas];' "${TARGET_SOURCE}"
grep -F '__shared__ float merge_denominator[kMergeHeadsPerCta];' "${TARGET_SOURCE}"
grep -F 'role == kMergeCta' "${TARGET_SOURCE}"
grep -F 'cluster.map_shared_rank(&producer_ready_barrier, kMergeCta);' "${TARGET_SOURCE}"
grep -F 'for (int element = role * kMergeElementsPerCta + thread;' "${TARGET_SOURCE}"
grep -F 'int const local_head = head - role * kMergeHeadsPerCta;' "${TARGET_SOURCE}"
grep -F 'role == 0 ? local_partial : cluster.map_shared_rank(local_partial, 0);' "${TARGET_SOURCE}"
grep -F 'role == 3 ? local_partial : cluster.map_shared_rank(local_partial, 3);' "${TARGET_SOURCE}"
grep -F 'constexpr int kQTileStride = kHeadDim + 8;' "${TARGET_SOURCE}"
[[ $(grep -Ec 'wmma::load_matrix_sync\(q_fragment_[0-7],.*kQTileStride\);' "${TARGET_SOURCE}") == 8 ]] || {
  echo 'v13 must retain all eight padded-Q WMMA loads' >&2; exit 2;
}
! grep -F '__shared__ __align__(32) float numerator[' "${TARGET_SOURCE}"
git -C "${SRC}" add -N csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
git -C "${SRC}" diff --check
git -C "${SRC}" diff --binary > "${ARTIFACT_DIR}/derived-plugin-v13-job${SLURM_JOB_ID}.patch"
DURABLE_CANDIDATE_SOURCE=${ARTIFACT_DIR}/native_c2_decode.v13-distributed-merge.cu
cp --preserve=mode,timestamps "${TARGET_SOURCE}" "${DURABLE_CANDIDATE_SOURCE}"
require_hash "${DURABLE_CANDIDATE_SOURCE}" "${EXPECTED_V13_SOURCE_SHA}" durable-v13-candidate-source
# Retain the build-source basename expected by the hardened directed
# provenance parser after node-local scratch is removed.
DURABLE_BUILD_SOURCE=${ARTIFACT_DIR}/build-source/native_c2_decode.cu
install -D -m 0444 "${TARGET_SOURCE}" "${DURABLE_BUILD_SOURCE}"
require_hash "${DURABLE_BUILD_SOURCE}" "${EXPECTED_V13_SOURCE_SHA}" durable-v13-build-source

# exact-d4 CUDA 13 accepts requested B300 10.3f, then its 10.0 family
# intersection emits one native sm_100 cubin and must contain no PTX.
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
output_path.write_text(json.dumps(matches[0], indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
PY
grep -F -- '-gencode arch=compute_100,code=sm_100' "${ARTIFACT_DIR}/native-c2-compile-command-job${SLURM_JOB_ID}.json"
cmake --build "${BUILD_DIR}" --target _native_c2_msa_decode_plugin -- -j64 -v 2>&1 | tee "${ARTIFACT_DIR}/compile-job${SLURM_JOB_ID}.log"
mapfile -t PLUGINS < <(find "${BUILD_DIR}" -type f -name '_native_c2_msa_decode_plugin.abi3.so' -print)
[[ ${#PLUGINS[@]} -eq 1 ]] || { printf 'expected one DSO, got %s\n' "${#PLUGINS[@]}" >&2; exit 2; }
PLUGIN=${PLUGINS[0]}
install -D -m 0755 "${PLUGIN}" "${ARTIFACT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so"
[[ "$(sha256_of "${PLUGIN}")" == "$(sha256_of "${ARTIFACT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so")" ]] || {
  echo 'durable candidate plugin differs from build output' >&2; exit 2;
}
grep -F -- '-gencode arch=compute_100,code=sm_100' "${ARTIFACT_DIR}/compile-job${SLURM_JOB_ID}.log"
cuobjdump --dump-resource-usage "${PLUGIN}" | tee "${ARTIFACT_DIR}/plugin-resource-usage-job${SLURM_JOB_ID}.txt"
grep -A1 'native_c2_msa_decode_kernel' "${ARTIFACT_DIR}/plugin-resource-usage-job${SLURM_JOB_ID}.txt" | tee "${ARTIFACT_DIR}/plugin-resource-proof-job${SLURM_JOB_ID}.txt"
RESOURCE_LINE=$(grep -m 1 -E 'REG:[[:space:]]*[0-9]+.*STACK:[[:space:]]*[0-9]+.*SHARED:[[:space:]]*[0-9]+.*LOCAL:[[:space:]]*[0-9]+' "${ARTIFACT_DIR}/plugin-resource-proof-job${SLURM_JOB_ID}.txt")
[[ -n "${RESOURCE_LINE}" ]] || { echo 'native kernel resource line missing' >&2; exit 2; }
read_resource() { local label=$1; sed -nE "s/.*${label}:[[:space:]]*([0-9]+).*/\\1/p" <<< "${RESOURCE_LINE}"; }
REG=$(read_resource REG); STACK=$(read_resource STACK); SHARED=$(read_resource SHARED); LOCAL=$(read_resource LOCAL)
[[ "${REG}" =~ ^[0-9]+$ && "${STACK}" =~ ^[0-9]+$ && "${SHARED}" =~ ^[0-9]+$ && "${LOCAL}" =~ ^[0-9]+$ ]] || { echo "unparseable resource line: ${RESOURCE_LINE}" >&2; exit 2; }
printf 'RESOURCE_LINE=%s\nREG=%s\nSTACK=%s\nSHARED=%s\nLOCAL=%s\n' "${RESOURCE_LINE}" "${REG}" "${STACK}" "${SHARED}" "${LOCAL}" | tee "${ARTIFACT_DIR}/plugin-resource-gate-job${SLURM_JOB_ID}.txt"
# v13 retains v12's padded Q tile but shrinks rank-local merge state from
# 320 B to 80 B.  Its smaller state also removes 16 B of compiler alignment
# padding before the 32-B-aligned Q tile, so the frozen job13419 observation is
# 31,136 - 240 - 16 = 30,880 B.  Stack/local spills remain forbidden; REG is
# recorded rather than guessed.
(( STACK == 0 && LOCAL == 0 && SHARED == 30880 )) || {
  echo "v13 resource gate failed: REG=${REG} STACK=${STACK} SHARED=${SHARED} LOCAL=${LOCAL}" >&2; exit 2;
}
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

sha256sum "${SUBMITTED_SCRIPT_COPY}" "${NATIVE_V1_SOURCE}" "${SCHEMA_PATCH}" "${CMAKE_PATCH}" \
  "${DISPATCH_PATCH}" "${LOADER_PATCH}" "${V12_BASELINE_SOURCE}" "${V13_PATCH}" \
  "${DURABLE_CANDIDATE_SOURCE}" "${DURABLE_BUILD_SOURCE}" \
  "${ARTIFACT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so" \
  > "${ARTIFACT_DIR}/plugin-v13-provenance-job${SLURM_JOB_ID}.sha256"
ldd "${PLUGIN}" | tee "${ARTIFACT_DIR}/plugin-ldd-job${SLURM_JOB_ID}.txt"
