#!/usr/bin/env bash
# Fail-closed installed-wheel lifecycle validation for accepted v12 Q-row
# padding.  The wheel is installed only into node-local scratch, then one clean
# B300 allocation exercises the real MiniMaxM3SparseMSAImpl.forward lifecycle.

#SBATCH --job-name=c2-v12-lifecycle
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/slurm-c2-native-plugin-v12-lifecycle-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=
export PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin

# jobs12993-job12996 remain immutable failure evidence.  This execution writes
# only to a fresh root and separates the importable wheel payload from installer
# caches so that any symlink decision has an exact ownership boundary.
ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v12-lifecycle-r4-artifacts-20260831
RUN_ID=${C2_RUN_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is required unless C2_RUN_ID is supplied}}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe C2 run identifier: ${RUN_ID}" >&2; exit 2; }
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${RUN_ID}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
UV=/usr/local/bin/uv
C2_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/team/c2_msa_decode
BASE_HARNESS=/home/lcpu/85117379/native_c2_full_backend_bench_20260829.py
HARNESS=/home/lcpu/85117379/native_c2_v12_lifecycle_20260830.py
CORE_SOURCE=/home/lcpu/85117379/native_c2_lifecycle_core_20260830.py
V12_PATCH=/home/lcpu/85117379/native_c2_v12_q_row_padding_20260830.patch
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")

WHEEL_DIR=/home/lcpu/85117379/c2-native-plugin-v12-wheel-artifacts-20260830/job12987
WHEEL=${WHEEL_DIR}/vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl
PROVENANCE=${WHEEL_DIR}/c2-native-plugin-overlay-provenance.json
WHEEL_VERIFICATION=${WHEEL_DIR}/wheel-driver-verification-job12987.json
WHEEL_PREREQUISITES=${WHEEL_DIR}/promotion-prerequisites-job12987.json
WHEEL_OUTPUTS=${WHEEL_DIR}/outputs-job12987.sha256
WHEEL_FINAL=${WHEEL_DIR}/final-status-job12987.txt

AOT_DIR=/home/lcpu/85117379/c2-native-plugin-v12-aot-artifacts-20260830/job12983
AOT_PLUGIN=${AOT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so
AOT_SOURCE=${AOT_DIR}/native_c2_decode.v12-q-row-padding.cu
AOT_RESOURCE_GATE=${AOT_DIR}/plugin-resource-gate-job12983.txt
AOT_PROVENANCE=${AOT_DIR}/plugin-v12-provenance-job12983.sha256
AOT_OUTPUTS=${AOT_DIR}/outputs-job12983.sha256
AOT_FINAL=${AOT_DIR}/final-status-job12983.txt
AOT_FINAL_SIDECAR=${AOT_DIR}/final-status-job12983.sha256

DIRECTED_DIR=/home/lcpu/85117379/c2-native-plugin-v12-directed-artifacts-20260830/job12984
DIRECTED_RESULT=${DIRECTED_DIR}/v12-q-row-padding-directed-job12984.json
DIRECTED_OUTPUTS=${DIRECTED_DIR}/outputs-job12984.sha256
DIRECTED_FINAL=${DIRECTED_DIR}/final-status-job12984.txt
DIRECTED_FINAL_SIDECAR=${DIRECTED_DIR}/final-status-job12984.sha256
DIRECTED_RUN_MANIFEST=${DIRECTED_DIR}/run-manifest-job12984.json

STRESS_DIR=/home/lcpu/85117379/c2-native-plugin-v12-stress-3pct-artifacts-20260830/job12985
STRESS_DECISION=${STRESS_DIR}/v12-q-row-padding-vs-v11-3pct-incremental-decision-job12985.json
STRESS_AGGREGATE=${STRESS_DIR}/v12-q-row-padding-vs-v11-3pct-incremental-aggregate-job12985.json
STRESS_OUTPUTS=${STRESS_DIR}/outputs-job12985.sha256
STRESS_FINAL=${STRESS_DIR}/final-status-job12985.txt
STRESS_FINAL_SIDECAR=${STRESS_DIR}/final-status-job12985.sha256

EXPECTED_WHEEL_SHA=b730e5610bf1135bf8042627143b7bb4b27c98421900bf2f1817fe3e8a7cc67b
EXPECTED_PROVENANCE_SHA=ed3ada6f5485a3d57e877e7012f36423c7c8d47b69f32d4130606c5f9ffad1bf
EXPECTED_WHEEL_VERIFICATION_SHA=af9d286b5e5186542a57a0e3c3d459e4cf8a92d9d9bcae59775d9f3dd4631e80
EXPECTED_WHEEL_PREREQUISITES_SHA=7549806cddbee1d0ae8d483081ef985c0e8fe9e8a3bd3ebe61b6c5a1d5c96838
EXPECTED_WHEEL_OUTPUTS_SHA=e28cec08fbf0ad9ce93a295066f9875d6d2fa917e031adbe2e350ff78f685877
EXPECTED_WHEEL_FINAL_SHA=cf7fe04ba75d7b59c4bb55f845bb51d0e553a326552fa06d12c94b6fb8d38090
EXPECTED_PLUGIN_SHA=064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6
EXPECTED_SOURCE_SHA=535d90b856ed1062aa7b8a105eb2c5f236c450826e65496e646d0d5a27eb8aaf
EXPECTED_V12_PATCH_SHA=52da58b2e531399dafd412fcfbcd6fda77de188c5c3080fc33628e2565b3bf83
EXPECTED_AOT_RESOURCE_GATE_SHA=c4a6cd69617ce82e99459fcd7a975a37fbfad497e51b36a6ec09945ffa97d6c0
EXPECTED_AOT_PROVENANCE_SHA=65bb886cece4a49b7ef7d7d4cab00588e4cc1f9238d2aa45dd5cf36f6952d2a2
EXPECTED_AOT_OUTPUTS_SHA=ca1f37f00f9ce78c0d88bb3a90e8288bbffd83e94315a378b36a9c7998d99419
EXPECTED_AOT_FINAL_SHA=b5fd3d82083165415ba20c37cf574e898ef0209d1a7844866c7491cc438b0f18
EXPECTED_AOT_FINAL_SIDECAR_SHA=912b28b5f81610b7cc04457c54a26165d4e321cfe709bdc79489a3b102041f67
EXPECTED_DIRECTED_RESULT_SHA=240b683c1b6921981cc814084697ff945a7714de1bace5956f74a09ea38765c8
EXPECTED_DIRECTED_OUTPUTS_SHA=0b1b512ef6bf0c8a5b461561f6fd4b5b768e4922f75c0285ce8bac37b793bf7a
EXPECTED_DIRECTED_FINAL_SHA=16026037d3916cf0dc741317e35ca258536ab3a8012f2671a0886e2c49ee2758
EXPECTED_DIRECTED_FINAL_SIDECAR_SHA=9da4e355841c9448683cd82d3281112753e2accdf519b7fdad05b60f8a3bbc5a
EXPECTED_DIRECTED_RUN_MANIFEST_SHA=d07f6028ed6ff0815cb0019a3df30235edc3a3b918cded23b413feaaf94bacde
EXPECTED_STRESS_DECISION_SHA=829505fce32c737f596b372819add7f7701dc793c87d58f429e61ba7806862d9
EXPECTED_STRESS_AGGREGATE_SHA=0e9b4da6ea78c812c66feab0d4c669e68d0dc9a65630d13fbbc37c2e66295142
EXPECTED_STRESS_OUTPUTS_SHA=799ee63d1dd815d381fcc866687da97bf52177994a490df7ab8b5f07e5bf78cc
EXPECTED_STRESS_FINAL_SHA=4e8c40bb6784b111dced540a2b977ff383d2a5cb41796a65711ef609bf409ca1
EXPECTED_STRESS_FINAL_SIDECAR_SHA=042449479aa483c2b3f1b345e04c89d0c711322b2ee4b82014efd4a40a54468e
EXPECTED_BASE_HARNESS_SHA=7883edc25df48e3b69a9f4948775a1baafd3ba0b1bd17edcfeac6644aa7b4762
EXPECTED_LIFECYCLE_CORE_SHA=5ca35d63101de99ab5827470e37e275f7019a7f68dcbf7a6dcaec2ef892a0788
EXPECTED_STABLE_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_PYTHON_RESOLVED=/home/lcpu/85117379/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12
EXPECTED_PYTHON_VERSION='Python 3.12.11'
EXPECTED_PYTHON_BINARY_SHA=ccc93c55c24bb92866cc2f9eff61b124abf5d1ee6c8da0ef9b5eed4d78a21f07
EXPECTED_UV_BINARY_SHA=92face6b1f0462ad911857957bd168cd4ae45515e2a2cb3fcc3ecbda3d4d82b1
EXPECTED_UV_VERSION='uv 0.12.1 (x86_64-unknown-linux-gnu)'

: "${C2_EXPECTED_LIFECYCLE_HARNESS_SHA:?submit with exact reviewed v12 lifecycle harness SHA-256}"
: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?submit with exact reviewed v12 lifecycle Slurm SHA-256}"
for digest in "${C2_EXPECTED_LIFECYCLE_HARNESS_SHA}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid SHA-256: ${digest}" >&2; exit 2; }
done

[[ ! -e "${ARTIFACT_ROOT}" || ( -d "${ARTIFACT_ROOT}" && ! -L "${ARTIFACT_ROOT}" && -z "$(find "${ARTIFACT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ) ]] || {
  echo 'v12 lifecycle artifact root must be absent or an empty regular directory before submission' >&2; exit 2;
}
[[ ! -e "${ARTIFACT_DIR}" ]] || { echo 'refusing to reuse v12 lifecycle artifact path' >&2; exit 2; }
mkdir -p "${ARTIFACT_ROOT}" "${ARTIFACT_DIR}"
DRIVER_LOG=${ARTIFACT_DIR}/driver-job${RUN_ID}.log
FINAL_STATUS=${ARTIFACT_DIR}/final-status-job${RUN_ID}.txt
FINAL_STATUS_SIDECAR=${ARTIFACT_DIR}/final-status-job${RUN_ID}.sha256
OUTPUTS_MANIFEST=${ARTIFACT_DIR}/outputs-job${RUN_ID}.sha256
SCRATCH_AUDIT=${ARTIFACT_DIR}/scratch-cleanup-job${RUN_ID}.json
RESULT=${ARTIFACT_DIR}/native-c2-v12-lifecycle-job${RUN_ID}.json
INSTALL_AUDIT=${ARTIFACT_DIR}/installed-wheel-audit-job${RUN_ID}.json
EVIDENCE_ATTESTATION=${ARTIFACT_DIR}/accepted-evidence-attestation-job${RUN_ID}.json
SYMLINK_AUDIT=${ARTIFACT_DIR}/runtime-symlink-audit-job${RUN_ID}.json
HARNESS_COPY=${ARTIFACT_DIR}/native_c2_v12_lifecycle.py

RUNTIME_PARENT=${SLURM_TMPDIR:-/tmp}
RUNTIME=
setup_finalize() {
  local original_rc=$?
  local cleanup_rc=0 finalizer_error=0 manifest_rc=0
  local setup_diag=${ARTIFACT_DIR}/setup-finalizer-diagnostics-job${RUN_ID}.txt
  trap - EXIT
  set +e
  if [[ -n "${RUNTIME}" ]]; then
    if [[ "${RUNTIME}" == "${RUNTIME_PARENT%/}/c2-v12-lifecycle-${RUN_ID}."* && -d "${RUNTIME}" && ! -L "${RUNTIME}" ]]; then
      chmod -R u+w -- "${RUNTIME}" || cleanup_rc=$?
      if [[ "${cleanup_rc}" -eq 0 ]]; then rm -rf -- "${RUNTIME}" || cleanup_rc=$?; fi
      [[ ! -e "${RUNTIME}" ]] || cleanup_rc=1
    else
      cleanup_rc=1
    fi
  fi
  [[ "${cleanup_rc}" -eq 0 ]] || finalizer_error=1
  "${PYTHON}" - "${SCRATCH_AUDIT}" "${RUNTIME}" "${RUNTIME_PARENT}" "${cleanup_rc}" <<'PY' || finalizer_error=1
import json
import sys
from pathlib import Path
output, runtime, parent, cleanup_rc = sys.argv[1:]
Path(output).write_text(json.dumps({
    "schema": "c2-native-plugin-v12-lifecycle-scratch-cleanup-v1",
    "runtime_path": runtime,
    "runtime_parent": parent,
    "node_local_install": True,
    "cleanup_rc": int(cleanup_rc),
    "runtime_absent_after_cleanup": not runtime or not Path(runtime).exists(),
    "phase": "setup",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  printf 'SETUP_FINALIZER=1 ORIGINAL_RC=%s CLEANUP_RC=%s UTC=%s\n' \
    "${original_rc}" "${cleanup_rc}" "$(date -u +%FT%TZ)" > "${setup_diag}" || finalizer_error=1
  local -a setup_outputs=()
  [[ -f "${SCRATCH_AUDIT}" ]] && setup_outputs+=("${SCRATCH_AUDIT}")
  [[ -f "${setup_diag}" ]] && setup_outputs+=("${setup_diag}")
  if [[ "${#setup_outputs[@]}" -gt 0 ]]; then
    sha256sum "${setup_outputs[@]}" > "${OUTPUTS_MANIFEST}" && sha256sum -c "${OUTPUTS_MANIFEST}" >/dev/null || manifest_rc=1
  else
    manifest_rc=1
  fi
  [[ "${manifest_rc}" -eq 0 ]] || finalizer_error=1
  printf 'FINAL_RC=2 ORIGINAL_RC=%s FINALIZER_ERROR=%s MANIFEST_RC=%s CLEANUP_RC=%s PHASE=SETUP UTC=%s\n' \
    "${original_rc}" "${finalizer_error}" "${manifest_rc}" "${cleanup_rc}" "$(date -u +%FT%TZ)" > "${FINAL_STATUS}" || finalizer_error=1
  sha256sum "${FINAL_STATUS}" > "${FINAL_STATUS_SIDECAR}" || finalizer_error=1
  exit 2
}
trap setup_finalize EXIT
[[ "${RUNTIME_PARENT}" == /* && -d "${RUNTIME_PARENT}" && ! -L "${RUNTIME_PARENT}" ]] || {
  echo 'SLURM_TMPDIR (or /tmp fallback) must be an absolute regular directory' >&2; exit 2;
}
RUNTIME=$(mktemp -d "${RUNTIME_PARENT%/}/c2-v12-lifecycle-${RUN_ID}.XXXXXX")
[[ "${RUNTIME}" == "${RUNTIME_PARENT%/}/c2-v12-lifecycle-${RUN_ID}."* && -d "${RUNTIME}" && ! -L "${RUNTIME}" ]] || {
  echo 'failed to create a confined node-local runtime directory' >&2; exit 2;
}
INSTALL_ROOT=${RUNTIME}/site
STATE_ROOT=${RUNTIME}/state
SUPPORT=${RUNTIME}/c2_support
CORE_COPY=${RUNTIME}/c2_lifecycle_core.py
export UV_CACHE_DIR=${STATE_ROOT}/uv-cache PIP_CACHE_DIR=${STATE_ROOT}/pip-cache XDG_CACHE_HOME=${STATE_ROOT}/xdg-cache TMPDIR=${STATE_ROOT}/tmp PYTHONPYCACHEPREFIX=${STATE_ROOT}/pycache
export UV_LINK_MODE=copy
mkdir -p "${INSTALL_ROOT}" "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${PYTHONPYCACHEPREFIX}"

exec 3>&1 4>&2
exec > >(tee "${DRIVER_LOG}") 2>&1
DRIVER_TEE_PID=$!
EXPECTED_GPU_UUID=
selector() { local value=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}; [[ "${value}" != *,* && ( "${value}" =~ ^[0-9]+$ || "${value}" =~ ^GPU-[0-9A-Fa-f-]+$ ) ]] || return 1; printf '%s\n' "${value}"; }
gpu_uuid() { nvidia-smi --id "$(selector)" --query-gpu=uuid --format=csv,noheader,nounits | awk 'NR==1 {print $1} END {if (NR != 1) exit 1}'; }
compute_apps() { nvidia-smi --id "$(selector)" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
snapshot() {
  local tag=$1 out=$2 apps=$3 metadata_rc=0 apps_rc=0
  printf '\n===== %s UTC=%s =====\n' "${tag}" "$(date -u +%FT%TZ)"
  if ! nvidia-smi --id "$(selector)" --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader,nounits | tee "${out}"; then metadata_rc=1; fi
  if ! compute_apps | tee "${apps}"; then apps_rc=1; fi
  [[ "${metadata_rc}" -eq 0 && "${apps_rc}" -eq 0 ]]
}

finalize() {
  local original_rc=$?
  local final_rc=${original_rc}
  local finalizer_error=0 tee_rc cleanup_rc=0 manifest_rc=0
  local post_uuid='' apps_after='' apps_rc=0
  local post_gpu=${ARTIFACT_DIR}/gpu-post-job${RUN_ID}.csv
  local post_apps=${ARTIFACT_DIR}/gpu-post-compute-apps-job${RUN_ID}.csv
  trap - EXIT
  set +e
  if snapshot FINAL_POST "${post_gpu}" "${post_apps}"; then
    if [[ -f "${post_apps}" ]]; then apps_after=$(<"${post_apps}"); else apps_rc=1; fi
  else
    finalizer_error=1
    apps_rc=1
    if [[ -f "${post_apps}" ]]; then apps_after=$(<"${post_apps}"); fi
  fi
  [[ "${apps_rc}" -eq 0 && -z "${apps_after}" ]] || finalizer_error=1
  post_uuid=$(gpu_uuid)
  [[ -n "${EXPECTED_GPU_UUID}" && "${post_uuid}" == "${EXPECTED_GPU_UUID}" ]] || finalizer_error=1
  # Support/harness inputs were made read-only during the run.  Restore the
  # job owner's write bit before deletion; failure is retained in the audit
  # and upgrades FINAL_RC, rather than silently leaking node-local state.
  if [[ "${RUNTIME}" == "${RUNTIME_PARENT%/}/c2-v12-lifecycle-${RUN_ID}."* && -d "${RUNTIME}" && ! -L "${RUNTIME}" ]]; then
    chmod -R u+w -- "${RUNTIME}" || cleanup_rc=$?
    if [[ "${cleanup_rc}" -eq 0 ]]; then rm -rf -- "${RUNTIME}" || cleanup_rc=$?; fi
    [[ ! -e "${RUNTIME}" ]] || cleanup_rc=1
  else
    cleanup_rc=1
  fi
  [[ "${cleanup_rc}" -eq 0 ]] || finalizer_error=1
  "${PYTHON}" - "${SCRATCH_AUDIT}" "${RUNTIME}" "${RUNTIME_PARENT}" "${cleanup_rc}" <<'PY' || finalizer_error=1
import json
import sys
from pathlib import Path
output, runtime, parent, cleanup_rc = sys.argv[1:]
Path(output).write_text(json.dumps({
    "schema": "c2-native-plugin-v12-lifecycle-scratch-cleanup-v1",
    "runtime_path": runtime,
    "runtime_parent": parent,
    "node_local_install": True,
    "cleanup_rc": int(cleanup_rc),
    "runtime_absent_after_cleanup": not Path(runtime).exists(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  exec 1>&3 2>&4
  exec 3>&- 4>&-
  wait "${DRIVER_TEE_PID}"; tee_rc=$?
  [[ "${tee_rc}" -eq 0 ]] || finalizer_error=1
  local -a outputs=()
  local output_list="${ARTIFACT_DIR}/outputs-list-job${RUN_ID}.nul"
  if ! (set -o pipefail; find "${ARTIFACT_DIR}" -type f ! -name "$(basename "${OUTPUTS_MANIFEST}")" ! -name "$(basename "${FINAL_STATUS}")" ! -name "$(basename "${FINAL_STATUS_SIDECAR}")" ! -name "$(basename "${output_list}")" -print0 | sort -z > "${output_list}"); then
    manifest_rc=1
  elif ! mapfile -d '' -t outputs < "${output_list}"; then
    manifest_rc=1
  fi
  if ! rm -f -- "${output_list}" || [[ -e "${output_list}" ]]; then manifest_rc=1; fi
  [[ "${#outputs[@]}" -gt 0 ]] || manifest_rc=1
  if [[ "${manifest_rc}" -eq 0 ]]; then sha256sum "${outputs[@]}" > "${OUTPUTS_MANIFEST}" && sha256sum -c "${OUTPUTS_MANIFEST}" >/dev/null || manifest_rc=1; fi
  [[ "${manifest_rc}" -eq 0 ]] || finalizer_error=1
  [[ "${finalizer_error}" -eq 0 && "${original_rc}" -eq 0 ]] || final_rc=2
  printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s TEE_RC=%s MANIFEST_RC=%s CLEANUP_RC=%s POST_APPS_EMPTY=%s EXPECTED_UUID=%s POST_UUID=%s UTC=%s\n' \
    "${final_rc}" "${original_rc}" "${finalizer_error}" "${tee_rc}" "${manifest_rc}" "${cleanup_rc}" \
    "$([[ -z "${apps_after}" ]] && echo true || echo false)" "${EXPECTED_GPU_UUID:-unset}" "${post_uuid:-unset}" "$(date -u +%FT%TZ)" > "${FINAL_STATUS}" || final_rc=2
  sha256sum "${FINAL_STATUS}" > "${FINAL_STATUS_SIDECAR}" || final_rc=2
  exit "${final_rc}"
}
trap finalize EXIT

GPU_PRE=${ARTIFACT_DIR}/gpu-pre-job${RUN_ID}.csv
GPU_PRE_APPS=${ARTIFACT_DIR}/gpu-pre-compute-apps-job${RUN_ID}.csv
snapshot PRE "${GPU_PRE}" "${GPU_PRE_APPS}"
gpu_line=$(<"${GPU_PRE}")
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* && -z "$(<"${GPU_PRE_APPS}")" ]] || { echo 'requires an idle B300 GPU' >&2; exit 1; }
EXPECTED_GPU_UUID=$(gpu_uuid); [[ "${EXPECTED_GPU_UUID}" =~ ^GPU- ]]

for path in "${BASE_HARNESS}" "${HARNESS}" "${CORE_SOURCE}" "${V12_PATCH}" "${SCRIPT_PATH}" \
  "${WHEEL}" "${PROVENANCE}" "${WHEEL_VERIFICATION}" "${WHEEL_PREREQUISITES}" "${WHEEL_OUTPUTS}" "${WHEEL_FINAL}" \
  "${AOT_PLUGIN}" "${AOT_SOURCE}" "${AOT_RESOURCE_GATE}" "${AOT_PROVENANCE}" "${AOT_OUTPUTS}" "${AOT_FINAL}" "${AOT_FINAL_SIDECAR}" \
  "${DIRECTED_RESULT}" "${DIRECTED_OUTPUTS}" "${DIRECTED_FINAL}" "${DIRECTED_FINAL_SIDECAR}" "${DIRECTED_RUN_MANIFEST}" \
  "${STRESS_DECISION}" "${STRESS_AGGREGATE}" "${STRESS_OUTPUTS}" "${STRESS_FINAL}" "${STRESS_FINAL_SIDECAR}"; do [[ -f "${path}" && ! -L "${path}" ]] || { echo "missing or linked input: ${path}" >&2; exit 2; }; done
[[ -x "${PYTHON}" && -x "${UV}" && -d "${C2_ROOT}/harness" ]]
PYTHON_RESOLVED=$(readlink -f "${PYTHON}")
[[ "${PYTHON_RESOLVED}" == "${EXPECTED_PYTHON_RESOLVED}" && -f "${PYTHON_RESOLVED}" && ! -L "${PYTHON_RESOLVED}" && -x "${PYTHON_RESOLVED}" ]]
[[ "$(sha256sum "${PYTHON_RESOLVED}" | awk '{print $1}')" == "${EXPECTED_PYTHON_BINARY_SHA}" && "$("${PYTHON}" -V 2>&1)" == "${EXPECTED_PYTHON_VERSION}" ]]
[[ "$(sha256sum "${UV}" | awk '{print $1}')" == "${EXPECTED_UV_BINARY_SHA}" && "$("${UV}" --version)" == "${EXPECTED_UV_VERSION}" ]]
[[ "${C2_EXPECTED_LIFECYCLE_HARNESS_SHA}" == "$(sha256sum "${HARNESS}" | awk '{print $1}')" ]]
[[ "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" == "$(sha256sum "${SCRIPT_PATH}" | awk '{print $1}')" ]]

printf '%s  %s\n' \
  "${EXPECTED_WHEEL_SHA}" "${WHEEL}" "${EXPECTED_PROVENANCE_SHA}" "${PROVENANCE}" "${EXPECTED_WHEEL_VERIFICATION_SHA}" "${WHEEL_VERIFICATION}" "${EXPECTED_WHEEL_PREREQUISITES_SHA}" "${WHEEL_PREREQUISITES}" "${EXPECTED_WHEEL_OUTPUTS_SHA}" "${WHEEL_OUTPUTS}" "${EXPECTED_WHEEL_FINAL_SHA}" "${WHEEL_FINAL}" \
  "${EXPECTED_PLUGIN_SHA}" "${AOT_PLUGIN}" "${EXPECTED_SOURCE_SHA}" "${AOT_SOURCE}" "${EXPECTED_V12_PATCH_SHA}" "${V12_PATCH}" "${EXPECTED_AOT_RESOURCE_GATE_SHA}" "${AOT_RESOURCE_GATE}" "${EXPECTED_AOT_PROVENANCE_SHA}" "${AOT_PROVENANCE}" "${EXPECTED_AOT_OUTPUTS_SHA}" "${AOT_OUTPUTS}" "${EXPECTED_AOT_FINAL_SHA}" "${AOT_FINAL}" "${EXPECTED_AOT_FINAL_SIDECAR_SHA}" "${AOT_FINAL_SIDECAR}" \
  "${EXPECTED_DIRECTED_RESULT_SHA}" "${DIRECTED_RESULT}" "${EXPECTED_DIRECTED_OUTPUTS_SHA}" "${DIRECTED_OUTPUTS}" "${EXPECTED_DIRECTED_FINAL_SHA}" "${DIRECTED_FINAL}" "${EXPECTED_DIRECTED_FINAL_SIDECAR_SHA}" "${DIRECTED_FINAL_SIDECAR}" "${EXPECTED_DIRECTED_RUN_MANIFEST_SHA}" "${DIRECTED_RUN_MANIFEST}" \
  "${EXPECTED_STRESS_DECISION_SHA}" "${STRESS_DECISION}" "${EXPECTED_STRESS_AGGREGATE_SHA}" "${STRESS_AGGREGATE}" "${EXPECTED_STRESS_OUTPUTS_SHA}" "${STRESS_OUTPUTS}" "${EXPECTED_STRESS_FINAL_SHA}" "${STRESS_FINAL}" "${EXPECTED_STRESS_FINAL_SIDECAR_SHA}" "${STRESS_FINAL_SIDECAR}" \
  "${EXPECTED_BASE_HARNESS_SHA}" "${BASE_HARNESS}" "${EXPECTED_LIFECYCLE_CORE_SHA}" "${CORE_SOURCE}" "${C2_EXPECTED_LIFECYCLE_HARNESS_SHA}" "${HARNESS}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" > "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"
sha256sum -c "${WHEEL_OUTPUTS}"; sha256sum -c "${AOT_OUTPUTS}"; sha256sum -c "${DIRECTED_OUTPUTS}"; sha256sum -c "${STRESS_OUTPUTS}"
grep -qxE 'FINAL_RC=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${WHEEL_FINAL}"
grep -q 'FINAL_RC=0' "${AOT_FINAL}"; grep -q 'FINAL_RC=0' "${DIRECTED_FINAL}"; grep -q 'FINAL_RC=0' "${STRESS_FINAL}"

"${PYTHON}" - "${WHEEL}" "${PROVENANCE}" "${WHEEL_VERIFICATION}" "${WHEEL_PREREQUISITES}" "${AOT_PLUGIN}" "${AOT_SOURCE}" "${AOT_RESOURCE_GATE}" "${AOT_PROVENANCE}" "${DIRECTED_RESULT}" "${STRESS_DECISION}" "${STRESS_AGGREGATE}" "${EVIDENCE_ATTESTATION}" <<'PY'
import hashlib, json, sys
from pathlib import Path
paths = list(map(Path, sys.argv[1:]))
wheel, provenance, wheel_check, prereq, plugin, source, resource, aot_provenance, directed, decision, aggregate, output = paths
H = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
wheel_sha = 'b730e5610bf1135bf8042627143b7bb4b27c98421900bf2f1817fe3e8a7cc67b'
plugin_sha = '064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6'
assert H(wheel) == wheel_sha
p = json.loads(provenance.read_text()); w = json.loads(wheel_check.read_text()); r = json.loads(prereq.read_text())
d = json.loads(directed.read_text()); s = json.loads(decision.read_text()); a = json.loads(aggregate.read_text())
assert p['schema'] == 'c2-native-plugin-overlay-provenance-v1'
assert p['derived_wheel']['sha256'] == wheel_sha and p['plugin_expected_sha256'] == plugin_sha
assert p['baseline_stable_member']['byte_identical'] is True
assert w['schema'] == 'c2-native-plugin-v12-overlay-wheel-driver-verification-v1' and w['all_gates_pass'] is True and w['plugin_sha256'] == plugin_sha
assert r['schema'] == 'c2-native-plugin-v12-wheel-promotion-prerequisites-v1' and r['directed_all_gates_pass'] is True and r['stress_accepted'] is True
assert H(plugin) == plugin_sha and H(source) == '535d90b856ed1062aa7b8a105eb2c5f236c450826e65496e646d0d5a27eb8aaf'
assert H(resource) == 'c4a6cd69617ce82e99459fcd7a975a37fbfad497e51b36a6ec09945ffa97d6c0'
assert H(aot_provenance) == '65bb886cece4a49b7ef7d7d4cab00588e4cc1f9238d2aa45dd5cf36f6952d2a2'
assert d['all_gates_pass'] is True
assert s['schema'] == 'c2-native-plugin-v12-q-row-padding-vs-v11-3pct-incremental-decision-v1' and s['accepted'] is True
assert s['all_correctness_and_resource_identity_gates_pass'] is True and s['performance']['paired_median_improvement_fraction'] > .03 and s['performance']['bootstrap_95pct_lower_bound_fraction'] > 0
assert s['aggregate_sha256'] == H(aggregate) and a['schema'].startswith('c2-native-plugin-v12-q-row-padding')
output.write_text(json.dumps({'schema':'c2-native-plugin-v12-lifecycle-accepted-evidence-attestation-v1','wheel_sha256':wheel_sha,'plugin_sha256':plugin_sha,'wheel_verification_sha256':H(wheel_check),'wheel_prerequisites_sha256':H(prereq),'aot_source_sha256':H(source),'aot_provenance_sha256':H(aot_provenance),'directed_result_sha256':H(directed),'stress_decision_sha256':H(decision),'stress_aggregate_sha256':H(aggregate),'all_gates_pass':True}, indent=2, sort_keys=True) + '\n')
PY

"${UV}" pip install --python "${PYTHON}" --target "${INSTALL_ROOT}" --no-deps "${WHEEL}"
install -D -m 0444 "${HARNESS}" "${HARNESS_COPY}"
install -D -m 0444 "${CORE_SOURCE}" "${CORE_COPY}"
printf '%s  %s\n' \
  'df9dd0b01c3e91afeb1a490785fab1a0554a16f001e118005192f55f891358a8' 'harness/__init__.py' \
  '38faefb89b38e3bc26b135b762a6bbbd813dcc695423982b423a806fe41c11d6' 'harness/data.py' \
  '712c1632367700137392d2c43bd5c6962130d945f01f1f585cc6338788ce53cc' 'harness/reference.py' \
  '6b0a40463b4f1c05ca53649d6689377b4f755d1efb58b10700dad05705b53f71' 'harness/triton_baseline.py' \
  '5db5d5e692044e54bb57418cbb7dcf5c9cc94f789eaccf128eec6fa8df86d967' 'vllm_msa_ref/sparse_attn.py' > "${ARTIFACT_DIR}/support-source-job${RUN_ID}.sha256"
(cd "${C2_ROOT}" && sha256sum -c "${ARTIFACT_DIR}/support-source-job${RUN_ID}.sha256")
while read -r digest member; do source_path=${C2_ROOT}/${member}; [[ -f "${source_path}" && ! -L "${source_path}" ]]; install -D -m 0444 "${source_path}" "${SUPPORT}/${member}"; done < "${ARTIFACT_DIR}/support-source-job${RUN_ID}.sha256"
cp "${ARTIFACT_DIR}/support-source-job${RUN_ID}.sha256" "${ARTIFACT_DIR}/support-installed-job${RUN_ID}.sha256"
(cd "${SUPPORT}" && sha256sum -c "${ARTIFACT_DIR}/support-installed-job${RUN_ID}.sha256")
chmod -R a-w "${SUPPORT}" "${HARNESS_COPY}" "${CORE_COPY}"

"${PYTHON}" - "${RUNTIME}" "${INSTALL_ROOT}" "${STATE_ROOT}" "${SUPPORT}" "${CORE_COPY}" "${HARNESS_COPY}" "${WHEEL}" "${SYMLINK_AUDIT}" <<'PY'
import hashlib
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

runtime, install_root, state_root, support, core_copy, harness_copy, wheel, output = map(Path, sys.argv[1:])
known_state_names = {'uv-cache', 'pip-cache', 'xdg-cache', 'tmp', 'pycache'}
errors = []
links = []
identity_cache = {}

def is_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def raise_walk_error(error):
    raise error

def directory_manifest_sha256(root):
    key = str(root)
    if key in identity_cache:
        return identity_cache[key]
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=raise_walk_error, followlinks=False):
        dirnames.sort()
        filenames.sort()
        parent = Path(dirpath)
        for name in [*dirnames, *filenames]:
            path = parent / name
            rel = path.relative_to(root).as_posix()
            metadata = path.lstat()
            mode = metadata.st_mode
            if stat.S_ISLNK(mode):
                record = f'L\0{rel}\0{os.readlink(path)}\0'
            elif stat.S_ISREG(mode):
                record = f'F\0{rel}\0{metadata.st_size}\0{file_sha256(path)}\0'
            elif stat.S_ISDIR(mode):
                record = f'D\0{rel}\0'
            else:
                record = f'O\0{rel}\0{stat.S_IFMT(mode)}\0{metadata.st_size}\0'
            digest.update(record.encode('utf-8', errors='surrogateescape'))
    value = digest.hexdigest()
    identity_cache[key] = value
    return value

def resolve_required(path, label):
    try:
        return path.resolve(strict=True)
    except Exception as exc:
        errors.append(f'{label} resolution failed: {path}: {type(exc).__name__}: {exc}')
        return None

runtime_resolved = resolve_required(runtime, 'runtime')
install_resolved = resolve_required(install_root, 'install')
state_resolved = resolve_required(state_root, 'state')
support_resolved = resolve_required(support, 'support')
core_parent_resolved = resolve_required(core_copy.parent, 'core parent')
state_component_roots = {}

for required, label in ((runtime, 'runtime'), (install_root, 'install'), (state_root, 'state'), (support, 'support')):
    if required.is_symlink() or not required.is_dir():
        errors.append(f'{label} root is missing, linked, or not a directory: {required}')
for required, label in ((core_copy, 'core'), (harness_copy, 'harness')):
    if required.is_symlink() or not required.is_file():
        errors.append(f'{label} is missing, linked, or not a regular file: {required}')
for resolved, label in ((install_resolved, 'install'), (state_resolved, 'state'), (support_resolved, 'support'), (core_parent_resolved, 'core parent')):
    if resolved is not None and runtime_resolved is not None and not is_within(resolved, runtime_resolved):
        errors.append(f'{label} escapes the runtime root: {resolved}')
for name in sorted(known_state_names):
    root = state_root / name
    if root.is_symlink() or not root.is_dir():
        errors.append(f'state area root is missing, linked, or not a directory: {root}')
        continue
    try:
        resolved = root.resolve(strict=True)
        if state_resolved is None or not is_within(resolved, state_resolved):
            errors.append(f'state area root escapes state root: {root} -> {resolved}')
            continue
        state_component_roots[name] = resolved
    except Exception as exc:
        errors.append(f'state area root resolution failed: {root}: {type(exc).__name__}: {exc}')

zip_links = []
try:
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                zip_links.append({
                    'member': info.filename,
                    'mode': oct(mode),
                    'raw_target': archive.read(info).decode('utf-8', errors='replace'),
                })
except Exception as exc:
    errors.append(f'wheel ZIP symlink inspection failed: {type(exc).__name__}: {exc}')
if zip_links:
    errors.append('wheel ZIP contains Unix symlink members')

def record_runtime_walk_error(error):
    errors.append(f'runtime traversal failed: {type(error).__name__}: {error}')

for dirpath, dirnames, filenames in os.walk(runtime, topdown=True, onerror=record_runtime_walk_error, followlinks=False):
    dirnames.sort()
    filenames.sort()
    parent = Path(dirpath)
    for name in [*dirnames, *filenames]:
        path = parent / name
        if not path.is_symlink():
            continue
        rel = path.relative_to(runtime).as_posix()
        raw_target = os.readlink(path)
        if is_within(path, install_root):
            area = 'install'
            state_component = None
        elif is_within(path, support):
            area = 'support'
            state_component = None
        elif path == core_copy:
            area = 'core'
            state_component = None
        elif is_within(path, state_root):
            state_rel = path.relative_to(state_root)
            state_component = state_rel.parts[0] if state_rel.parts else ''
            area = f'state/{state_component}' if state_component else 'state/unknown'
        else:
            area = 'unknown'
            state_component = None

        entry = {
            'path': str(path),
            'relative_path': rel,
            'raw_target': raw_target,
            'area': area,
            'resolved_target': None,
            'target_exists': False,
            'target_kind': None,
            'target_size': None,
            'target_sha256': None,
            'target_directory_manifest_sha256': None,
            'resolved_within_runtime': False,
            'resolved_within_same_state_area': False,
            'policy_pass': False,
            'error': None,
        }
        try:
            resolved = path.resolve(strict=True)
            entry['resolved_target'] = str(resolved)
            entry['target_exists'] = True
            entry['resolved_within_runtime'] = runtime_resolved is not None and is_within(resolved, runtime_resolved)
            metadata = resolved.stat()
            entry['target_size'] = metadata.st_size
            if resolved.is_file():
                entry['target_kind'] = 'regular_file'
                if entry['resolved_within_runtime']:
                    entry['target_sha256'] = file_sha256(resolved)
            elif resolved.is_dir():
                entry['target_kind'] = 'directory'
                if entry['resolved_within_runtime']:
                    entry['target_directory_manifest_sha256'] = directory_manifest_sha256(resolved)
            else:
                entry['target_kind'] = 'other'

            if area.startswith('state/') and state_component in state_component_roots:
                component_root = state_component_roots[state_component]
                entry['resolved_within_same_state_area'] = is_within(resolved, component_root)
                entry['policy_pass'] = (
                    entry['resolved_within_same_state_area']
                    and entry['target_kind'] in {'regular_file', 'directory'}
                    and (entry['target_sha256'] is not None or entry['target_directory_manifest_sha256'] is not None)
                )
        except Exception as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
        if not entry['policy_pass']:
            errors.append(f'disallowed or unauditable symlink: {rel}')
        links.append(entry)

payload = {
    'schema': 'c2-native-plugin-v12-runtime-symlink-audit-v1',
    'runtime': str(runtime),
    'install_root': str(install_root),
    'state_root': str(state_root),
    'support_root': str(support),
    'core_copy': str(core_copy),
    'harness_copy': str(harness_copy),
    'wheel': str(wheel),
    'policy': {
        'wheel_zip_symlinks_forbidden': True,
        'install_support_core_unknown_symlinks_forbidden': True,
        'state_symlinks_require_same_area_confinement_and_content_identity': True,
        'known_state_areas': sorted(known_state_names),
    },
    'wheel_zip_symlinks': zip_links,
    'symlinks': links,
    'symlink_count': len(links),
    'errors': errors,
    'all_gates_pass': not errors,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
if errors:
    raise SystemExit('; '.join(errors))
PY

"${PYTHON}" - "${INSTALL_ROOT}" "${WHEEL}" "${PROVENANCE}" "${INSTALL_AUDIT}" <<'PY'
import hashlib, json, sys, zipfile
from pathlib import Path
runtime, wheel, provenance, output = map(Path, sys.argv[1:])
p = json.loads(provenance.read_text()); root = runtime / 'vllm'
stable = root / '_C_stable_libtorch.abi3.so'; plugin = root / '_native_c2_msa_decode_plugin.abi3.so'; adapter = root / 'models/minimax_m3/nvidia/msa_native_c2_decode.py'
assert all(path.is_file() and not path.is_symlink() for path in (stable, plugin, adapter))
with zipfile.ZipFile(wheel) as archive:
    assert stable.read_bytes() == archive.read('vllm/_C_stable_libtorch.abi3.so')
    assert plugin.read_bytes() == archive.read('vllm/_native_c2_msa_decode_plugin.abi3.so')
    assert adapter.read_bytes() == archive.read('vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py')
assert hashlib.sha256(stable.read_bytes()).hexdigest() == 'cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
assert hashlib.sha256(plugin.read_bytes()).hexdigest() == p['plugin_expected_sha256'] == '064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6'
output.write_text(json.dumps({'schema':'c2-native-plugin-v12-lifecycle-installed-wheel-v1','runtime':str(runtime),'node_local_install':True,'stable_sha256':hashlib.sha256(stable.read_bytes()).hexdigest(),'plugin_sha256':hashlib.sha256(plugin.read_bytes()).hexdigest(),'adapter_sha256':hashlib.sha256(adapter.read_bytes()).hexdigest(),'all_gates_pass':True}, indent=2, sort_keys=True) + '\n')
PY

PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${INSTALL_ROOT}:${SUPPORT}" \
  timeout --foreground --preserve-status --kill-after=10s 2400s \
  "${PYTHON}" "${HARNESS_COPY}" --c2-root "${SUPPORT}" --vllm-root "${INSTALL_ROOT}" --base-harness "${BASE_HARNESS}" \
  --lifecycle-core "${CORE_COPY}" --expected-lifecycle-core-sha "${EXPECTED_LIFECYCLE_CORE_SHA}" --output "${RESULT}" --seed 20260830 \
  --steady-iterations 1000 --graph-replays 100 --memory-bound-bytes 8388608

"${PYTHON}" - "${RESULT}" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
required = {'fresh_adapter_eight_thread_first_load','real_forward_reaches_dispatcher_and_native_cuda_kernel','steady_1000_real_forward_lifecycle','cuda_graph_capture_replay_and_static_query_update','native_support_predicate_matrix','dynamic_native_rejection_without_fallback_claim','native_forward_sequence_boundaries','dispatcher_surface_persists_through_lifecycle'}
assert p['schema'] == 'c2-native-c2-v12-production-lifecycle-v1' and p['all_gates_pass'] is True
assert set(p['gates']) == required and all(p['gates'].values())
assert p['first_load']['concurrent_loader']['thread_count'] == 8 and p['first_load']['concurrent_loader']['load_library_call_count'] == 1
assert p['steady_lifecycle']['iterations'] == 1000 and p['steady_lifecycle']['cuda_memory_counters']['pass'] is True
for label in ('original_query', 'updated_query'):
    replay = p['cuda_graph'][label]['replays']; assert replay['replays'] == 100 and replay['cuda_memory_counters']['pass'] is True
assert p['dynamic_rejection_gate']['mode'] == 'selection_only' and all(row['pass'] for row in p['dynamic_rejection_gate']['rows'])
assert [row['sequence_length'] for row in p['sequence_length_boundaries']['rows']] == [2048,2049,4095,4096] and all(row['pass'] for row in p['sequence_length_boundaries']['rows'])
assert p['lifecycle_core']['schema'] == 'c2-native-c2-v12-lifecycle-core-binding-v1' and p['lifecycle_core']['sha256'] == '5ca35d63101de99ab5827470e37e275f7019a7f68dcbf7a6dcaec2ef892a0788'
PY
exit 0
