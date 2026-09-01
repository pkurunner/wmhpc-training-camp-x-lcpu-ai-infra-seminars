#!/usr/bin/env bash
# Build a local, experimental overlay wheel only after the accepted v16
# AOT -> directed -> same-GPU ABBA chain.  This is packaging/inventory work,
# not an upstream change, release artifact, or a new performance measurement.
#
# The v16 AOT checkout/build directory is deliberately not an input: it was
# transient and has been removed. Candidate identity is the durable AOT plugin
# and source. A fresh, job-local checkout is reconstructed only to supply the
# reviewed Python overlay surface required by the packager; it is never built.

#SBATCH --job-name=c2-native-plugin-v16-wheel
#SBATCH --partition=cpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/lcpu/85117379/slurm-c2-native-plugin-v16-wheel-%j.log

set -Eeuo pipefail
umask 077
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONPATH= PYTHONDONTWRITEBYTECODE=1

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v16-k-chunk-lookahead-overlay-wheel-artifacts-20260831
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID:?must run under Slurm}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
PACKAGER=/home/lcpu/85117379/build_native_c2_plugin_overlay_wheel_20260829.py
BASE_WHEEL=/home/lcpu/85117379/vllm-d4-wheel/vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl
SUBMITTED_SCRIPT=/home/lcpu/85117379/build_native_c2_plugin_v16_k_chunk_lookahead_overlay_wheel_20260831.slurm
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
BASE=/home/lcpu/85117379/vllm-d4da0c5
RUNTIME=/tmp/c2-native-plugin-v16-k-chunk-lookahead-overlay-wheel-job${SLURM_JOB_ID:?must run under Slurm}
STAGED_CHECKOUT=${RUNTIME}/src
TARGET_SOURCE=${STAGED_CHECKOUT}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
NATIVE_V1_SOURCE=/home/lcpu/85117379/native_c2_decode.plugin-v1.cu
SCHEMA_PATCH=/home/lcpu/85117379/native_c2_plugin_schema_20260829.patch
CMAKE_PATCH=/home/lcpu/85117379/exact_d4_native_c2_plugin_cmake_20260829.patch
DISPATCH_PATCH=/home/lcpu/85117379/exact_d4_python_dispatch_20260829.patch
LOADER_PATCH=/home/lcpu/85117379/exact_d4_native_c2_plugin_python_loader_20260829.patch
V12_BASELINE_SOURCE=/home/lcpu/85117379/c2-native-plugin-v12-aot-artifacts-20260830/job12983/native_c2_decode.v12-q-row-padding.cu
V16_PATCH=/home/lcpu/85117379/native_c2_v16_k_chunk_lookahead_20260831.patch

EXPECTED_D4_COMMIT=d4da0c55af3aa231b6209bf77871f3ed36eab0d2
EXPECTED_BASELINE_SHA=91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06
EXPECTED_STABLE_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_PACKAGER_SHA=84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e
EXPECTED_PLUGIN_SHA=1da5f731da796656759f0e673e3479392b6b8337c054a61aa4ca1fd0afb4edd4
EXPECTED_SOURCE_SHA=a01b78c89205a8ded07d65c82cdcf9d2eb5b39df4fc45a2fd992d92200b84180
EXPECTED_ADAPTER_SHA=6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412
EXPECTED_NATIVE_V1_SOURCE_SHA=f67495586759b3b8f718dbbf60627c499f917842f2ed6ecf89d2344f1c82bc4b
EXPECTED_SCHEMA_PATCH_SHA=53e7cd4b09a8999010442934c863427c30b019fc549d7f0cf0f3d16ccc1e2e6e
EXPECTED_CMAKE_PATCH_SHA=5b145f36ce5f8b12dae71ce02e77d65545a29eefcb5a6045f591348bc325ac35
EXPECTED_DISPATCH_PATCH_SHA=1ca224ec4f94adf8ee97580a0e08321552224555570784c308c0edcda34bcd91
EXPECTED_LOADER_PATCH_SHA=9cb3a22216048083d7a4802ba700fa07fd892436151871dc6e909c77239adb33
EXPECTED_PLUGIN_V1_SOURCE_SHA=8fee086f8ba4f5b2710347025211f9733ce64c61b04e44e710016d59b336f406
EXPECTED_PLUGIN_CMAKE_SHA=6662b0400eb04c590b56c15346848b4a26c45d86e796a69e1bcb4a0d91523734
EXPECTED_V12_BASELINE_SOURCE_SHA=535d90b856ed1062aa7b8a105eb2c5f236c450826e65496e646d0d5a27eb8aaf
EXPECTED_V16_PATCH_SHA=586d15cfcac5c9a1bbe1efa94e89e9031f090eb19a376ec817f10599ba8e11be

# The packager requires a git checkout in order to attest every overlay Python
# member. This path is created afresh below from the clean d4 base plus exactly
# the same patch ordering used by the v16 AOT job; it is not a build input.
DISPATCH_CHECKOUT=${STAGED_CHECKOUT}
ADAPTER_MEMBER=vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py
DISPATCH_CUSTOM_OPS=${DISPATCH_CHECKOUT}/vllm/_custom_ops.py
DISPATCH_ATTENTION=${DISPATCH_CHECKOUT}/vllm/config/attention.py
DISPATCH_REGISTRY=${DISPATCH_CHECKOUT}/vllm/v1/attention/backends/registry.py
DISPATCH_ADAPTER=${DISPATCH_CHECKOUT}/${ADAPTER_MEMBER}
DISPATCH_SPARSE=${DISPATCH_CHECKOUT}/vllm/models/minimax_m3/nvidia/sparse_attention_msa.py
EXPECTED_DISPATCH_CUSTOM_OPS_SHA=b57dfa819424b89bed8bf39924d5a4a3d2ebf327b39f3ae7c2e9e65b61c2398d
EXPECTED_DISPATCH_ATTENTION_SHA=caf68c29d8d36324d252a66f0811a25b26d11da9860732359076fbb76625b5ae
EXPECTED_DISPATCH_REGISTRY_SHA=cd539941417fd5faf6672676de0f8ae15c9adb2a915b47cdd508e154c34faea4
EXPECTED_DISPATCH_SPARSE_SHA=07d00372ca1995aa96159a0b9f8949b1b40f817a63f49224e88aafac1b722200

AOT_DIR=/home/lcpu/85117379/c2-native-plugin-v16-k-chunk-lookahead-aot-artifacts-20260831/job13773
AOT_PLUGIN=${AOT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so
AOT_SOURCE=${AOT_DIR}/native_c2_decode.v16-k-chunk-lookahead.cu
AOT_RESOURCE=${AOT_DIR}/plugin-resource-gate-job13773.txt
AOT_PROVENANCE=${AOT_DIR}/plugin-v16-provenance-job13773.sha256
AOT_FINAL=${AOT_DIR}/final-status-job13773.txt
AOT_FINAL_SIDECAR=${AOT_DIR}/final-status-job13773.sha256
AOT_OUTPUTS=${AOT_DIR}/outputs-job13773.sha256
EXPECTED_AOT_RESOURCE_SHA=806183180358cde9c25334c50ff2478546e40a2374ddcf64da7fd0c4f4b1a3a6
EXPECTED_AOT_PROVENANCE_SHA=3c640900ab67f25f53eee07f5e94106bb3f849d258b6d3863aab6f8c1b5068eb
EXPECTED_AOT_FINAL_SHA=649066423b484d5269661d4644f5dd5cbd1e89f79b33cd7e0d863dc319dbeae6
EXPECTED_AOT_FINAL_SIDECAR_SHA=6e66751ee1c635c308f027cae10880328fbd116d93257298567e1fef84af66a9
EXPECTED_AOT_OUTPUTS_SHA=4e25f4aeb2f3a56bd55e05b240db930601e050bc2a9afd5a0efc06e69b7b6dc4

DIRECTED_DIR=/home/lcpu/85117379/c2-native-plugin-v16-k-chunk-lookahead-directed-artifacts-20260831/job13786
DIRECTED_RESULT=${DIRECTED_DIR}/v16-k-chunk-lookahead-directed-job13786.json
DIRECTED_FINAL=${DIRECTED_DIR}/final-status-job13786.txt
DIRECTED_FINAL_SIDECAR=${DIRECTED_DIR}/final-status-job13786.sha256
DIRECTED_OUTPUTS=${DIRECTED_DIR}/outputs-job13786.sha256
DIRECTED_MANIFEST=${DIRECTED_DIR}/run-manifest-job13786.json
EXPECTED_DIRECTED_RESULT_SHA=cff227a42cd243bc49a33eaa6a7fe1ee2d332a19eedcb64f2a77c27f2f1e9944
EXPECTED_DIRECTED_FINAL_SHA=2709c051960ddfac837b5435e6e2f79bd330e557da6ffbc9f83c73b13281d23d
EXPECTED_DIRECTED_FINAL_SIDECAR_SHA=044950b5874ac4cf0c0b020905ba57a01acf82bcf07bbfa26dd27cd493d86b85
EXPECTED_DIRECTED_OUTPUTS_SHA=697a9533dcf273596a14445f1c3e66978d77fc9fb1ce2bf979108a042d7a896f
EXPECTED_DIRECTED_MANIFEST_SHA=490a2966c6861bb66f4d8b680a72c8704b91164ed44446a5fe52ecc939a1631e

STRESS_DIR=/home/lcpu/85117379/c2-native-plugin-v16-k-chunk-lookahead-stress-3pct-artifacts-20260831/job13789
STRESS_DECISION=${STRESS_DIR}/decision-job13789.json
STRESS_AGGREGATE=${STRESS_DIR}/aggregate-job13789.json
STRESS_FINAL=${STRESS_DIR}/final-status-job13789.txt
STRESS_FINAL_SIDECAR=${STRESS_DIR}/final-status-job13789.sha256
STRESS_OUTPUTS=${STRESS_DIR}/outputs-job13789.sha256
EXPECTED_STRESS_DECISION_SHA=3763ce31dbda5482896aa00c0192af1aca0ba8e322a4e555b52a6c7ad491bed9
EXPECTED_STRESS_AGGREGATE_SHA=95c456509add154bf11747a78b3b5203cc6580ad08bccf31cc562b3815172984
EXPECTED_STRESS_FINAL_SHA=996fe73ff2e20556751a6e92cd5d74eab9488985979c9b449afb0dfc1e65fb63
EXPECTED_STRESS_FINAL_SIDECAR_SHA=4dc631671c51e335a49a51009effc4c15ba38b5afd696efc83572d6ea1b464af
EXPECTED_STRESS_OUTPUTS_SHA=507bb58ff618e21b6520a0772d25f9f4ba1487012207e378c856ba694267ec5e

: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?reviewed v16 wheel-script SHA-256 required}"
for digest in \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${EXPECTED_BASELINE_SHA}" "${EXPECTED_STABLE_SHA}" \
  "${EXPECTED_PACKAGER_SHA}" "${EXPECTED_PLUGIN_SHA}" "${EXPECTED_SOURCE_SHA}" \
  "${EXPECTED_ADAPTER_SHA}" "${EXPECTED_NATIVE_V1_SOURCE_SHA}" \
  "${EXPECTED_SCHEMA_PATCH_SHA}" "${EXPECTED_CMAKE_PATCH_SHA}" \
  "${EXPECTED_DISPATCH_PATCH_SHA}" "${EXPECTED_LOADER_PATCH_SHA}" \
  "${EXPECTED_PLUGIN_V1_SOURCE_SHA}" "${EXPECTED_PLUGIN_CMAKE_SHA}" \
  "${EXPECTED_V12_BASELINE_SOURCE_SHA}" "${EXPECTED_V16_PATCH_SHA}" \
  "${EXPECTED_DISPATCH_CUSTOM_OPS_SHA}" \
  "${EXPECTED_DISPATCH_ATTENTION_SHA}" "${EXPECTED_DISPATCH_REGISTRY_SHA}" \
  "${EXPECTED_DISPATCH_SPARSE_SHA}" "${EXPECTED_AOT_RESOURCE_SHA}" \
  "${EXPECTED_AOT_PROVENANCE_SHA}" "${EXPECTED_AOT_FINAL_SHA}" \
  "${EXPECTED_AOT_FINAL_SIDECAR_SHA}" "${EXPECTED_AOT_OUTPUTS_SHA}" \
  "${EXPECTED_DIRECTED_RESULT_SHA}" "${EXPECTED_DIRECTED_FINAL_SHA}" \
  "${EXPECTED_DIRECTED_FINAL_SIDECAR_SHA}" "${EXPECTED_DIRECTED_OUTPUTS_SHA}" \
  "${EXPECTED_DIRECTED_MANIFEST_SHA}" "${EXPECTED_STRESS_DECISION_SHA}" \
  "${EXPECTED_STRESS_AGGREGATE_SHA}" "${EXPECTED_STRESS_FINAL_SHA}" \
  "${EXPECTED_STRESS_FINAL_SIDECAR_SHA}" "${EXPECTED_STRESS_OUTPUTS_SHA}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]
done

[[ -d "${ARTIFACT_ROOT}" && ! -L "${ARTIFACT_ROOT}" &&
   "$(readlink -f -- "${ARTIFACT_ROOT}")" == "${ARTIFACT_ROOT}" &&
   "$(stat -c %u -- "${ARTIFACT_ROOT}")" == "$(id -u)" ]] || {
  echo 'artifact root must be an owned, canonical pre-created directory' >&2; exit 2;
}
[[ -z "$(find "${ARTIFACT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
  echo 'artifact root must be empty before v16 overlay-wheel submission' >&2; exit 2;
}
[[ ! -e "${ARTIFACT_DIR}" && ! -L "${ARTIFACT_DIR}" ]] || {
  echo "refusing to reuse ${ARTIFACT_DIR}" >&2; exit 2;
}
[[ "${RUNTIME}" == "/tmp/c2-native-plugin-v16-k-chunk-lookahead-overlay-wheel-job${SLURM_JOB_ID}" &&
   ! -e "${RUNTIME}" && ! -L "${RUNTIME}" ]] || {
  echo "refusing unsafe or reused runtime path: ${RUNTIME}" >&2; exit 2;
}
[[ -f "${SCRIPT_PATH}" && -f "${SUBMITTED_SCRIPT}" ]] || {
  echo 'missing executed or reviewed submitted wheel script' >&2; exit 2;
}
cmp -s "${SCRIPT_PATH}" "${SUBMITTED_SCRIPT}" || {
  echo 'Slurm spool script differs from reviewed submitted copy' >&2; exit 2;
}
mkdir -m 0700 -- "${ARTIFACT_DIR}"
mkdir -m 0700 -- "${RUNTIME}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1

cleanup_runtime() {
  local runtime=$1
  [[ "${runtime}" == "/tmp/c2-native-plugin-v16-k-chunk-lookahead-overlay-wheel-job${SLURM_JOB_ID}" ]] || return 1
  [[ ! -e "${runtime}" && ! -L "${runtime}" ]] && return 0
  [[ -d "${runtime}" && ! -L "${runtime}" &&
     "$(readlink -f -- "${runtime}")" == "${runtime}" &&
     "$(stat -c %u -- "${runtime}")" == "$(id -u)" ]] || return 1
  rm -rf -- "${runtime}" || return 1
  [[ ! -e "${runtime}" && ! -L "${runtime}" ]]
}

finalize() {
  local rc=$? manifest_rc=0 cleanup_rc=0
  local status=${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt
  local manifest=${ARTIFACT_DIR}/final-output-manifest-job${SLURM_JOB_ID}.sha256
  trap - EXIT
  set +e
  cleanup_runtime "${RUNTIME}" || cleanup_rc=$?
  (( cleanup_rc == 0 || rc != 0 )) || rc=2
  printf 'FINAL_RC=%s CLEANUP_RC=%s UTC=%s\n' "${rc}" "${cleanup_rc}" "$(date -u +%FT%TZ)" > "${status}"
  find "${ARTIFACT_DIR}" -maxdepth 1 -type f \
    ! -name 'driver-job*.log' ! -name 'final-output-manifest-job*.sha256' \
    -printf '%p\0' | sort -z | xargs -0r sha256sum > "${manifest}" || manifest_rc=1
  if (( manifest_rc != 0 && rc == 0 )); then
    rc=2
    printf 'FINAL_RC=%s CLEANUP_RC=%s UTC=%s\n' "${rc}" "${cleanup_rc}" "$(date -u +%FT%TZ)" > "${status}"
    find "${ARTIFACT_DIR}" -maxdepth 1 -type f \
      ! -name 'driver-job*.log' ! -name 'final-output-manifest-job*.sha256' \
      -printf '%p\0' | sort -z | xargs -0r sha256sum > "${manifest}" || true
  fi
  exit "${rc}"
}
trap finalize EXIT

for path in \
  "${BASE_WHEEL}" "${PACKAGER}" "${AOT_PLUGIN}" "${AOT_SOURCE}" \
  "${AOT_RESOURCE}" "${AOT_PROVENANCE}" "${AOT_FINAL}" "${AOT_FINAL_SIDECAR}" \
  "${AOT_OUTPUTS}" "${DIRECTED_RESULT}" "${DIRECTED_FINAL}" \
  "${DIRECTED_FINAL_SIDECAR}" "${DIRECTED_OUTPUTS}" "${DIRECTED_MANIFEST}" \
  "${STRESS_DECISION}" "${STRESS_AGGREGATE}" "${STRESS_FINAL}" \
  "${STRESS_FINAL_SIDECAR}" "${STRESS_OUTPUTS}" "${NATIVE_V1_SOURCE}" \
  "${SCHEMA_PATCH}" "${CMAKE_PATCH}" "${DISPATCH_PATCH}" "${LOADER_PATCH}" \
  "${V12_BASELINE_SOURCE}" "${V16_PATCH}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || { echo "missing or linked input: ${path}" >&2; exit 2; }
done
[[ -x "${PYTHON}" && -d "${BASE}" && ! -L "${BASE}" ]]
"${PYTHON}" -c 'raise SystemExit(0 if __debug__ else 1)'
PYTHONPYCACHEPREFIX="${RUNTIME}/pycache" "${PYTHON}" -m py_compile "${PACKAGER}"
grep -qxE 'BODY_RC=0 TEE_RC=0 CLEANUP_RC=0 MANIFEST_RC=0 FINAL_RC=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${AOT_FINAL}"
grep -qxE 'FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${DIRECTED_FINAL}"
grep -qxE 'FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${STRESS_FINAL}"

# The AOT artifact itself is the sole v16 candidate identity.  In particular,
# do not read a v16 transient checkout or build output here.
printf '%s  %s\n' \
  "${EXPECTED_BASELINE_SHA}" "${BASE_WHEEL}" \
  "${EXPECTED_PACKAGER_SHA}" "${PACKAGER}" \
  "${EXPECTED_PLUGIN_SHA}" "${AOT_PLUGIN}" \
  "${EXPECTED_SOURCE_SHA}" "${AOT_SOURCE}" \
  "${EXPECTED_NATIVE_V1_SOURCE_SHA}" "${NATIVE_V1_SOURCE}" \
  "${EXPECTED_SCHEMA_PATCH_SHA}" "${SCHEMA_PATCH}" \
  "${EXPECTED_CMAKE_PATCH_SHA}" "${CMAKE_PATCH}" \
  "${EXPECTED_DISPATCH_PATCH_SHA}" "${DISPATCH_PATCH}" \
  "${EXPECTED_LOADER_PATCH_SHA}" "${LOADER_PATCH}" \
  "${EXPECTED_V12_BASELINE_SOURCE_SHA}" "${V12_BASELINE_SOURCE}" \
  "${EXPECTED_V16_PATCH_SHA}" "${V16_PATCH}" \
  "${EXPECTED_AOT_RESOURCE_SHA}" "${AOT_RESOURCE}" \
  "${EXPECTED_AOT_PROVENANCE_SHA}" "${AOT_PROVENANCE}" \
  "${EXPECTED_AOT_FINAL_SHA}" "${AOT_FINAL}" \
  "${EXPECTED_AOT_FINAL_SIDECAR_SHA}" "${AOT_FINAL_SIDECAR}" \
  "${EXPECTED_AOT_OUTPUTS_SHA}" "${AOT_OUTPUTS}" \
  "${EXPECTED_DIRECTED_RESULT_SHA}" "${DIRECTED_RESULT}" \
  "${EXPECTED_DIRECTED_FINAL_SHA}" "${DIRECTED_FINAL}" \
  "${EXPECTED_DIRECTED_FINAL_SIDECAR_SHA}" "${DIRECTED_FINAL_SIDECAR}" \
  "${EXPECTED_DIRECTED_OUTPUTS_SHA}" "${DIRECTED_OUTPUTS}" \
  "${EXPECTED_DIRECTED_MANIFEST_SHA}" "${DIRECTED_MANIFEST}" \
  "${EXPECTED_STRESS_DECISION_SHA}" "${STRESS_DECISION}" \
  "${EXPECTED_STRESS_AGGREGATE_SHA}" "${STRESS_AGGREGATE}" \
  "${EXPECTED_STRESS_FINAL_SHA}" "${STRESS_FINAL}" \
  "${EXPECTED_STRESS_FINAL_SIDECAR_SHA}" "${STRESS_FINAL_SIDECAR}" \
  "${EXPECTED_STRESS_OUTPUTS_SHA}" "${STRESS_OUTPUTS}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SUBMITTED_SCRIPT}" \
  > "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
sha256sum -c "${AOT_FINAL_SIDECAR}"
sha256sum -c "${AOT_OUTPUTS}"
sha256sum -c "${AOT_PROVENANCE}"
sha256sum -c "${DIRECTED_FINAL_SIDECAR}"
sha256sum -c "${DIRECTED_OUTPUTS}"
sha256sum -c "${STRESS_FINAL_SIDECAR}"
sha256sum -c "${STRESS_OUTPUTS}"

[[ "$(git -C "${BASE}" rev-parse HEAD)" == "${EXPECTED_D4_COMMIT}" &&
   -z "$(git -C "${BASE}" status --porcelain)" ]] || {
  echo 'exact d4 base is not clean at the required commit' >&2; exit 2;
}
git clone --no-hardlinks "${BASE}" "${STAGED_CHECKOUT}"
git -C "${STAGED_CHECKOUT}" checkout --detach "${EXPECTED_D4_COMMIT}"
[[ -z "$(git -C "${STAGED_CHECKOUT}" status --porcelain)" ]]
require_hash() {
  local actual
  actual=$(sha256sum "$1" | awk '{print $1}')
  [[ "${actual}" == "$2" ]] || { echo "hash mismatch for $3: ${actual}" >&2; exit 2; }
}
install -D -m 0644 "${NATIVE_V1_SOURCE}" "${TARGET_SOURCE}"
require_hash "${TARGET_SOURCE}" "${EXPECTED_NATIVE_V1_SOURCE_SHA}" copied-native-v1-source
for patch_file in "${SCHEMA_PATCH}" "${CMAKE_PATCH}" "${DISPATCH_PATCH}"; do
  patch --dry-run --batch --fuzz=0 -d "${STAGED_CHECKOUT}" -p1 < "${patch_file}"
  patch --batch --fuzz=0 -d "${STAGED_CHECKOUT}" -p1 < "${patch_file}"
done
patch --dry-run --batch --fuzz=0 -d "${STAGED_CHECKOUT}" -p1 < "${LOADER_PATCH}"
patch --batch --fuzz=0 -d "${STAGED_CHECKOUT}" -p1 < "${LOADER_PATCH}"
require_hash "${TARGET_SOURCE}" "${EXPECTED_PLUGIN_V1_SOURCE_SHA}" plugin-v1-source-before-v16
require_hash "${STAGED_CHECKOUT}/CMakeLists.txt" "${EXPECTED_PLUGIN_CMAKE_SHA}" plugin-cmake
require_hash "${DISPATCH_CUSTOM_OPS}" "${EXPECTED_DISPATCH_CUSTOM_OPS_SHA}" staged-custom-ops
require_hash "${DISPATCH_ATTENTION}" "${EXPECTED_DISPATCH_ATTENTION_SHA}" staged-attention
require_hash "${DISPATCH_REGISTRY}" "${EXPECTED_DISPATCH_REGISTRY_SHA}" staged-registry
require_hash "${DISPATCH_ADAPTER}" "${EXPECTED_ADAPTER_SHA}" staged-adapter
require_hash "${DISPATCH_SPARSE}" "${EXPECTED_DISPATCH_SPARSE_SHA}" staged-sparse-attention
install -m 0644 "${V12_BASELINE_SOURCE}" "${TARGET_SOURCE}"
require_hash "${TARGET_SOURCE}" "${EXPECTED_V12_BASELINE_SOURCE_SHA}" restored-accepted-v12-source
patch --dry-run --batch --fuzz=0 -d "${STAGED_CHECKOUT}" -p1 < "${V16_PATCH}"
patch --batch --fuzz=0 -d "${STAGED_CHECKOUT}" -p1 < "${V16_PATCH}"
require_hash "${TARGET_SOURCE}" "${EXPECTED_SOURCE_SHA}" staged-v16-source
git -C "${STAGED_CHECKOUT}" diff --check
"${PYTHON}" - "${DISPATCH_CUSTOM_OPS}" "${DISPATCH_ATTENTION}" "${DISPATCH_REGISTRY}" \
  "${DISPATCH_ADAPTER}" "${DISPATCH_SPARSE}" <<'PY'
import ast, sys
from pathlib import Path
for raw in sys.argv[1:]:
    path = Path(raw)
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
PY
{
  printf 'BASE_COMMIT=%s\n' "${EXPECTED_D4_COMMIT}"
  printf 'STAGED_CHECKOUT=%s\n' "${STAGED_CHECKOUT}"
  sha256sum "${TARGET_SOURCE}" "${STAGED_CHECKOUT}/CMakeLists.txt" \
    "${DISPATCH_CUSTOM_OPS}" "${DISPATCH_ATTENTION}" "${DISPATCH_REGISTRY}" \
    "${DISPATCH_ADAPTER}" "${DISPATCH_SPARSE}"
} > "${ARTIFACT_DIR}/staged-checkout-derivation-job${SLURM_JOB_ID}.txt"
grep -Fx "${EXPECTED_SOURCE_SHA}  ${TARGET_SOURCE}" "${ARTIFACT_DIR}/staged-checkout-derivation-job${SLURM_JOB_ID}.txt"
grep -Fx "${EXPECTED_ADAPTER_SHA}  ${DISPATCH_ADAPTER}" "${ARTIFACT_DIR}/staged-checkout-derivation-job${SLURM_JOB_ID}.txt"

"${PYTHON}" - "${ARTIFACT_DIR}/aot-attestation-job${SLURM_JOB_ID}.json" \
  "${AOT_PLUGIN}" "${AOT_SOURCE}" "${AOT_RESOURCE}" "${AOT_PROVENANCE}" \
  "${AOT_FINAL}" "${AOT_FINAL_SIDECAR}" "${AOT_OUTPUTS}" <<'PY'
import hashlib, json, re, sys
from pathlib import Path

out, plugin, source, resource, provenance, final, final_sidecar, outputs = map(Path, sys.argv[1:])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(plugin) == '1da5f731da796656759f0e673e3479392b6b8337c054a61aa4ca1fd0afb4edd4'
assert digest(source) == 'a01b78c89205a8ded07d65c82cdcf9d2eb5b39df4fc45a2fd992d92200b84180'
status = dict(part.split('=', 1) for part in final.read_text(encoding='utf-8').split() if '=' in part)
assert all(status.get(key) == '0' for key in ('BODY_RC', 'TEE_RC', 'CLEANUP_RC', 'MANIFEST_RC', 'FINAL_RC'))
line = next(item for item in resource.read_text(encoding='utf-8').splitlines() if item.startswith('RESOURCE_LINE='))
numbers = {key: int(re.search(rf'{key}:\s*([0-9]+)', line).group(1)) for key in ('REG', 'STACK', 'SHARED', 'LOCAL')}
assert numbers['REG'] <= 128 and numbers['STACK'] == 0 and numbers['SHARED'] == 31136 and numbers['LOCAL'] == 0
def manifest(path):
    rows = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        value, sep, name = raw.partition('  ')
        item = Path(name).resolve()
        assert sep and re.fullmatch(r'[0-9a-f]{64}', value) and item.is_file() and digest(item) == value and item not in rows
        rows[item] = value
    assert rows
    return rows
assert final.resolve() in manifest(final_sidecar)
output_rows = manifest(outputs)
assert all(item.resolve() in output_rows for item in (plugin, source, resource, provenance))
provenance_rows = manifest(provenance)
assert provenance_rows.get(plugin.resolve()) == digest(plugin)
assert any(item.name == 'native_c2_decode.cu' and value == digest(source) for item, value in provenance_rows.items())
out.write_text(json.dumps({
    'schema': 'c2-native-plugin-v16-k-chunk-lookahead-aot-attestation-v1',
    'aot_job_id': 13773,
    'aot_final_rc': 0,
    'resource_gate': numbers,
    'v16_plugin_sha256': digest(plugin),
    'v16_source_sha256': digest(source),
    'aot_files_sha256': {
        'resource_gate': digest(resource), 'provenance': digest(provenance),
        'final_status': digest(final), 'outputs': digest(outputs),
    },
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

"${PYTHON}" - "${ARTIFACT_DIR}/promotion-prerequisites-job${SLURM_JOB_ID}.json" \
  "${DIRECTED_RESULT}" "${DIRECTED_MANIFEST}" "${STRESS_DECISION}" \
  "${STRESS_AGGREGATE}" "${AOT_PLUGIN}" <<'PY'
import hashlib, json, sys
from pathlib import Path

out, directed_path, directed_manifest_path, decision_path, aggregate_path, plugin_path = map(Path, sys.argv[1:])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
directed = json.loads(directed_path.read_text(encoding='utf-8'))
run_manifest = json.loads(directed_manifest_path.read_text(encoding='utf-8'))
decision = json.loads(decision_path.read_text(encoding='utf-8'))
aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
assert directed['schema'] == 'c2-native-c2-v16-k-chunk-lookahead-directed-v1'
assert directed['all_gates_pass'] is True and directed['no_monkeypatch'] is True
assert directed['v6_directed_complete_gate_passed'] is True
v6_gate = directed['v6_directed_complete_gate']
assert v6_gate['profiler']['native_dispatcher_hit'] is True
assert v6_gate['profiler']['exactly_one_native_kernel_event'] is True
assert directed['k_chunk_directed_case_count'] == 3
assert all(directed[name]['all_gates_pass'] is True and directed[name]['stale_or_shifted_sensitivity']['passed'] is True
           for name in ('k_chunk_transition', 'k_chunk_last_chunk', 'k_chunk_stale_or_shifted'))
assert run_manifest['schema'] == 'c2-native-c2-v16-k-chunk-lookahead-directed-input-manifest-v1'
entries = {item['label']: item for item in run_manifest['inputs']}
candidate = entries['v16_k_chunk_lookahead_plugin']
assert Path(candidate['path']).resolve() == plugin_path.resolve()
assert candidate['sha256'] == digest(plugin_path)
assert decision['schema'] == 'c2-native-plugin-v16-k-chunk-lookahead-vs-v12-3pct-incremental-decision-v1'
assert decision['accepted'] is True and decision['all_correctness_and_resource_identity_gates_pass'] is True
assert decision['aggregate_sha256'] == digest(aggregate_path)
performance = decision['performance']
assert performance['paired_median_improvement_fraction'] > performance['required_improvement_fraction'] == .03
assert performance['bootstrap_95pct_lower_bound_fraction'] > performance['required_bootstrap_lower_bound_fraction'] == 0
assert aggregate['schema'] == 'c2-native-plugin-v16-k-chunk-lookahead-vs-v12-3pct-incremental-aggregate-v1'
assert len(aggregate['seeds']) == 8 and aggregate['allocated_gpu_uuid'].startswith('GPU-')
out.write_text(json.dumps({
    'schema': 'c2-native-plugin-v16-k-chunk-lookahead-wheel-promotion-prerequisites-v1',
    'aot_job_id': 13773, 'directed_job_id': 13786, 'stress_job_id': 13789,
    'directed_all_gates_pass': True, 'stress_accepted': True,
    'paired_median_improvement_fraction': performance['paired_median_improvement_fraction'],
    'bootstrap_95pct_lower_bound_fraction': performance['bootstrap_95pct_lower_bound_fraction'],
    'one_physical_gpu_uuid': aggregate['allocated_gpu_uuid'],
    'inputs_sha256': {
        'directed_result': digest(directed_path), 'directed_manifest': digest(directed_manifest_path),
        'stress_decision': digest(decision_path), 'stress_aggregate': digest(aggregate_path),
    },
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

"${PYTHON}" "${PACKAGER}" \
  --base-wheel "${BASE_WHEEL}" \
  --plugin-checkout "${DISPATCH_CHECKOUT}" \
  --plugin-library "${AOT_PLUGIN}" \
  --plugin-archive-member vllm/_native_c2_msa_decode_plugin.abi3.so \
  --expected-plugin-sha256 "${EXPECTED_PLUGIN_SHA}" \
  --artifact-dir "${ARTIFACT_DIR}" | tee "${ARTIFACT_DIR}/result-job${SLURM_JOB_ID}.json"

# Independently validate the derived archive: all RECORD entries, exact member
# delta, every unchanged baseline member, and the immutable stable DSO.
"${PYTHON}" - "${ARTIFACT_DIR}" "${BASE_WHEEL}" "${AOT_PLUGIN}" "${AOT_SOURCE}" \
  "${ARTIFACT_DIR}/wheel-driver-verification-job${SLURM_JOB_ID}.json" <<'PY'
import base64, csv, hashlib, io, json, sys, zipfile
from pathlib import Path

artifact_dir, baseline, plugin, source, out = map(Path, sys.argv[1:])
digest = lambda data: hashlib.sha256(data).hexdigest()
plugin_member = 'vllm/_native_c2_msa_decode_plugin.abi3.so'
adapter_member = 'vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py'
stable_member = 'vllm/_C_stable_libtorch.abi3.so'
dispatch = {
    'vllm/_custom_ops.py': 'b57dfa819424b89bed8bf39924d5a4a3d2ebf327b39f3ae7c2e9e65b61c2398d',
    'vllm/config/attention.py': 'caf68c29d8d36324d252a66f0811a25b26d11da9860732359076fbb76625b5ae',
    'vllm/v1/attention/backends/registry.py': 'cd539941417fd5faf6672676de0f8ae15c9adb2a915b47cdd508e154c34faea4',
    adapter_member: '6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412',
    'vllm/models/minimax_m3/nvidia/sparse_attention_msa.py': '07d00372ca1995aa96159a0b9f8949b1b40f817a63f49224e88aafac1b722200',
}
def record_check(archive, names, record):
    rows = list(csv.reader(io.StringIO(archive.read(record).decode('utf-8'))))
    index = {row[0]: row for row in rows}
    payload = {name for name in names if not name.endswith('/')} - {record}
    assert rows and len(index) == len(rows) and set(index) - {record} == payload
    assert index[record] == [record, '', '']
    for name in payload:
        data = archive.read(name)
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
        assert index[name] == [name, f'sha256={encoded}', str(len(data))]
    return {'record_member': record, 'payload_count': len(payload), 'all_payload_entries_validated': True}
wheel = artifact_dir / baseline.name
provenance = json.loads((artifact_dir / 'c2-native-plugin-overlay-provenance.json').read_text(encoding='utf-8'))
manifest = json.loads((artifact_dir / 'c2-native-plugin-overlay-manifest.json').read_text(encoding='utf-8'))
assert digest(wheel.read_bytes()) == provenance['derived_wheel']['sha256'] == manifest['derived_wheel_sha256']
assert provenance['baseline_wheel']['sha256'] == '91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06'
assert provenance['packager']['sha256'] == manifest['packager_sha256'] == '84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e'
assert provenance['member_set']['new_members'] == [plugin_member, adapter_member]
assert provenance['member_set']['removed_members'] == []
assert provenance['baseline_stable_member']['byte_identical'] is True
assert provenance['baseline_stable_member']['baseline_sha256'] == provenance['baseline_stable_member']['derived_sha256'] == 'cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
members = {item['archive_member']: item for item in provenance['overlay_members']}
assert members[plugin_member]['source_sha256'] == '1da5f731da796656759f0e673e3479392b6b8337c054a61aa4ca1fd0afb4edd4'
with zipfile.ZipFile(baseline) as base, zipfile.ZipFile(wheel) as derived:
    assert base.testzip() is None and derived.testzip() is None
    base_names, derived_names = base.namelist(), derived.namelist()
    assert len(base_names) == len(set(base_names)) and len(derived_names) == len(set(derived_names))
    base_records = [name for name in base_names if name.endswith('.dist-info/RECORD')]
    derived_records = [name for name in derived_names if name.endswith('.dist-info/RECORD')]
    assert len(base_records) == len(derived_records) == 1
    base_record = record_check(base, base_names, base_records[0])
    derived_record = record_check(derived, derived_names, derived_records[0])
    assert set(derived_names) == set(base_names) | {plugin_member, adapter_member}
    unchanged = []
    for name in base_names:
        if name.endswith('/') or name in set(dispatch) | {base_records[0]}:
            continue
        assert base.read(name) == derived.read(name), name
        unchanged.append(name)
    assert base.read(stable_member) == derived.read(stable_member)
    assert digest(derived.read(stable_member)) == 'cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
    assert derived.read(plugin_member) == plugin.read_bytes()
    assert digest(derived.read(plugin_member)) == '1da5f731da796656759f0e673e3479392b6b8337c054a61aa4ca1fd0afb4edd4'
    for name, expected in dispatch.items():
        assert digest(derived.read(name)) == expected
out.write_text(json.dumps({
    'schema': 'c2-native-plugin-v16-k-chunk-lookahead-overlay-wheel-driver-verification-v1',
    'wheel': str(wheel), 'wheel_sha256': digest(wheel.read_bytes()),
    'plugin_sha256': digest(plugin.read_bytes()), 'v16_candidate_source_sha256': digest(source.read_bytes()),
    'stable_member_sha256': 'cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442',
    'stable_member_byte_identical': True,
    'only_expected_new_members': [plugin_member, adapter_member],
    'only_existing_dispatch_replacements': sorted(dispatch),
    'record': {'baseline': base_record, 'derived': derived_record},
    'unchanged_baseline_payload_count': len(unchanged), 'all_gates_pass': True,
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY

wheel=$("${PYTHON}" - "${ARTIFACT_DIR}/c2-native-plugin-overlay-provenance.json" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['derived_wheel']['filename'])
PY
)
sha256sum \
  "${ARTIFACT_DIR}/c2-native-plugin-overlay-provenance.json" \
  "${ARTIFACT_DIR}/c2-native-plugin-overlay-manifest.json" \
  "${ARTIFACT_DIR}/${wheel}" \
  "${ARTIFACT_DIR}/staged-checkout-derivation-job${SLURM_JOB_ID}.txt" \
  "${ARTIFACT_DIR}/aot-attestation-job${SLURM_JOB_ID}.json" \
  "${ARTIFACT_DIR}/promotion-prerequisites-job${SLURM_JOB_ID}.json" \
  "${ARTIFACT_DIR}/wheel-driver-verification-job${SLURM_JOB_ID}.json" \
  "${ARTIFACT_DIR}/result-job${SLURM_JOB_ID}.json" \
  > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
