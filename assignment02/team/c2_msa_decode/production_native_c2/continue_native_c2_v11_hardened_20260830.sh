#!/usr/bin/env bash
# One-connection recovery launcher for the hardened v11 evidence chain.
set -Eeuo pipefail

HOME_ROOT=/home/lcpu/85117379
RECOVERY_LOCK=${HOME_ROOT}/.c2-native-v11-hardened-recovery-20260830.lock
AOT_SCRIPT=${HOME_ROOT}/build_native_c2_plugin_v11_q_fragment_reuse_aot.slurm
DIRECTED_SCRIPT=${HOME_ROOT}/validate_native_c2_plugin_v11_q_fragment_reuse_directed.slurm
STRESS_SCRIPT=${HOME_ROOT}/validate_native_c2_plugin_v11_q_fragment_reuse_stress_perf_3pct.slurm
PATCH=${HOME_ROOT}/native_c2_v11_q_fragment_reuse_20260830.patch

AOT_SCRIPT_SHA=f85fa3c65470755a0647f3be34bcfb511548d577f095ce84ee99bcce212d61b3
DIRECTED_SCRIPT_SHA=8cb6fb8265c9558484eadfacb09e2137c8bf521ba050e5d8e2814c4f74d6a2e5
STRESS_SCRIPT_SHA=62f21263d545b1cd50619e48d555883e1ae68f008994c67a588be2db2073a6c1
PATCH_SHA=c024220e16b4b36185a4ca2b53baa047fb7f376b96235983a25e58945bfe640b
SOURCE_SHA=0e82b278b7aa44a034a96d2ddd19946a27928d95a9f9d03e8e1fe9f30680c5b4
V9_PLUGIN=${HOME_ROOT}/c2-native-plugin-v9-aot-artifacts-20260830/job12701/vllm/_native_c2_msa_decode_plugin.abi3.so
V9_PLUGIN_SHA=a98a7bee3a5983cce7e824d8539381fc0c2f8e82cdb6f65c9ebfea18284fc592

command -v flock >/dev/null
exec 9>"${RECOVERY_LOCK}"
flock -n 9 || { echo 'another v11 hardened recovery is already active' >&2; exit 2; }

printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${AOT_SCRIPT_SHA}" "${AOT_SCRIPT}" \
  "${DIRECTED_SCRIPT_SHA}" "${DIRECTED_SCRIPT}" \
  "${STRESS_SCRIPT_SHA}" "${STRESS_SCRIPT}" \
  "${PATCH_SHA}" "${PATCH}" | sha256sum -c -

wait_for_job() {
  local job=$1 status=$2 label=$3
  local attempt
  for attempt in $(seq 1 240); do
    if [[ -f "${status}" ]] && [[ -z "$(squeue -h -j "${job}" -o %T)" ]]; then
      printf '%s job%s finished after %s polls\n' "${label}" "${job}" "${attempt}"
      return 0
    fi
    sleep 5
  done
  printf '%s job%s timed out waiting for durable status\n' "${label}" "${job}" >&2
  return 1
}

publish_job_receipt() {
  local receipt=$1 job=$2 tmp
  [[ "${job}" =~ ^[0-9]+$ && ! -e "${receipt}" ]]
  tmp=$(mktemp "${receipt}.tmp.XXXXXX")
  printf '%s\n' "${job}" > "${tmp}"
  chmod a-w "${tmp}"
  mv -T -- "${tmp}" "${receipt}"
}

queued_job_ids() {
  local job_name=$1
  squeue -h -u 85117379 -n "${job_name}" -o %i | \
    sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; /^[[:space:]]*$/d' | sort -u
}

AOT_ROOT=${HOME_ROOT}/c2-native-plugin-v11-aot-artifacts-20260830
PREHARDENED=${HOME_ROOT}/c2-native-plugin-v11-aot-prehardening-job12793-artifacts-20260830
AOT_RECEIPT=${HOME_ROOT}/c2-native-plugin-v11-aot-job-id-20260830.txt
AOT_JOB_NAME=c2-v11-aot-20260830
if [[ -d "${AOT_ROOT}/job12793" ]]; then
  [[ ! -e "${PREHARDENED}" ]]
  grep -Eq '^BODY_RC=0 TEE_RC=0 MANIFEST_RC=0 FINAL_RC=0 ' \
    "${AOT_ROOT}/job12793/final-status-job12793.txt"
  mv -- "${AOT_ROOT}" "${PREHARDENED}"
  mkdir -- "${AOT_ROOT}"
fi
[[ -d "${AOT_ROOT}" ]]
aot_queue_ids=$(queued_job_ids "${AOT_JOB_NAME}")
if [[ -d "${AOT_ROOT}/job12825" ]]; then
  AOT_JOB=12825
  [[ -z "${aot_queue_ids}" || "${aot_queue_ids}" == "${AOT_JOB}" ]]
  if [[ -f "${AOT_RECEIPT}" ]]; then
    grep -Fx "${AOT_JOB}" "${AOT_RECEIPT}"
  else
    publish_job_receipt "${AOT_RECEIPT}" "${AOT_JOB}"
  fi
  printf 'V11_HARDENED_AOT_RESUME=%s\n' "${AOT_JOB}"
elif [[ -f "${AOT_RECEIPT}" ]]; then
  grep -Eq '^[0-9]+$' "${AOT_RECEIPT}"
  AOT_JOB=$(<"${AOT_RECEIPT}")
  [[ -z "${aot_queue_ids}" || "${aot_queue_ids}" == "${AOT_JOB}" ]]
  printf 'V11_HARDENED_AOT_RECEIPT_RESUME=%s\n' "${AOT_JOB}"
elif [[ -n "${aot_queue_ids}" ]]; then
  [[ "${aot_queue_ids}" =~ ^[0-9]+$ ]]
  AOT_JOB=${aot_queue_ids}
  publish_job_receipt "${AOT_RECEIPT}" "${AOT_JOB}"
  printf 'V11_HARDENED_AOT_QUEUE_RESUME=%s\n' "${AOT_JOB}"
else
  [[ -z "$(find "${AOT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
  AOT_JOB=$(sbatch --parsable --job-name="${AOT_JOB_NAME}" --cpus-per-task=16 --mem=64G \
    --export=ALL,C2_EXPECTED_SCRIPT_SHA=${AOT_SCRIPT_SHA} "${AOT_SCRIPT}")
  publish_job_receipt "${AOT_RECEIPT}" "${AOT_JOB}"
  printf 'V11_HARDENED_AOT=%s\n' "${AOT_JOB}"
fi
AOT_DIR=${AOT_ROOT}/job${AOT_JOB}
wait_for_job "${AOT_JOB}" "${AOT_DIR}/final-status-job${AOT_JOB}.txt" AOT

AOT_STATUS=${AOT_DIR}/final-status-job${AOT_JOB}.txt
AOT_SIDECAR=${AOT_DIR}/final-status-job${AOT_JOB}.sha256
AOT_OUTPUTS=${AOT_DIR}/outputs-job${AOT_JOB}.sha256
PLUGIN=${AOT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so
SOURCE=${AOT_DIR}/native_c2_decode.v11-q-fragment-reuse.cu
RESOURCE=${AOT_DIR}/plugin-resource-gate-job${AOT_JOB}.txt
PROVENANCE=${AOT_DIR}/plugin-v11-provenance-job${AOT_JOB}.sha256
grep -Eq '^BODY_RC=0 TEE_RC=0 MANIFEST_RC=0 FINAL_RC=0 ' "${AOT_STATUS}"
sha256sum -c "${AOT_SIDECAR}"
sha256sum -c "${AOT_OUTPUTS}"
[[ "$(sha256sum "${SOURCE}" | awk '{print $1}')" == "${SOURCE_SHA}" ]]
grep -Eq 'STACK:[[:space:]]*0.*SHARED:[[:space:]]*30880.*LOCAL:[[:space:]]*0' "${RESOURCE}"

PLUGIN_SHA=$(sha256sum "${PLUGIN}" | awk '{print $1}')
RESOURCE_SHA=$(sha256sum "${RESOURCE}" | awk '{print $1}')
PROVENANCE_SHA=$(sha256sum "${PROVENANCE}" | awk '{print $1}')
AOT_STATUS_SHA=$(sha256sum "${AOT_STATUS}" | awk '{print $1}')
AOT_SIDECAR_SHA=$(sha256sum "${AOT_SIDECAR}" | awk '{print $1}')
AOT_OUTPUTS_SHA=$(sha256sum "${AOT_OUTPUTS}" | awk '{print $1}')

DIRECTED_ROOT=${HOME_ROOT}/c2-native-plugin-v11-directed-artifacts-20260830
DIRECTED_FAILURE_ROOT=${HOME_ROOT}/c2-native-plugin-v11-directed-failure-audit-job12829-artifacts-20260830
DIRECTED_RECEIPT=${HOME_ROOT}/c2-native-plugin-v11-directed-job-id-20260830.txt
DIRECTED_FAILURE_RECEIPT=${HOME_ROOT}/c2-native-plugin-v11-directed-failure-job12829-id-20260830.txt
DIRECTED_JOB_NAME=c2-v11-directed-20260830
if [[ -d "${DIRECTED_ROOT}/job12829" ]]; then
  [[ ! -e "${DIRECTED_FAILURE_ROOT}" ]]
  [[ "$(find "${DIRECTED_ROOT}" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)" == job12829 ]]
  grep -Eq '^FINAL_RC=2 ORIGINAL_RC=2 FINALIZER_ERROR=0 POST_CHECKED=0 ' \
    "${DIRECTED_ROOT}/job12829/final-status-job12829.txt"
  (cd "${DIRECTED_ROOT}/job12829" && sha256sum -c final-status-job12829.sha256)
  if [[ -e "${DIRECTED_RECEIPT}" ]]; then
    [[ ! -e "${DIRECTED_FAILURE_RECEIPT}" ]]
    grep -Fx '12829' "${DIRECTED_RECEIPT}"
    mv -- "${DIRECTED_RECEIPT}" "${DIRECTED_FAILURE_RECEIPT}"
  fi
  mv -- "${DIRECTED_ROOT}" "${DIRECTED_FAILURE_ROOT}"
fi
mkdir -p -- "${DIRECTED_ROOT}"
directed_job_name=$(find "${DIRECTED_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'job*' -printf '%f\n' | sort)
directed_queue_ids=$(queued_job_ids "${DIRECTED_JOB_NAME}")
if [[ -f "${DIRECTED_RECEIPT}" ]]; then
  grep -Eq '^[0-9]+$' "${DIRECTED_RECEIPT}"
  DIRECTED_JOB=$(<"${DIRECTED_RECEIPT}")
  [[ -z "${directed_job_name}" || "${directed_job_name}" == "job${DIRECTED_JOB}" ]]
  [[ -z "${directed_queue_ids}" || "${directed_queue_ids}" == "${DIRECTED_JOB}" ]]
  printf 'V11_DIRECTED_RECEIPT_RESUME=%s\n' "${DIRECTED_JOB}"
elif [[ -n "${directed_job_name}" ]]; then
  [[ "${directed_job_name}" =~ ^job([0-9]+)$ ]]
  DIRECTED_JOB=${BASH_REMATCH[1]}
  [[ -z "${directed_queue_ids}" || "${directed_queue_ids}" == "${DIRECTED_JOB}" ]]
  publish_job_receipt "${DIRECTED_RECEIPT}" "${DIRECTED_JOB}"
  printf 'V11_DIRECTED_DISCOVERED_RESUME=%s\n' "${DIRECTED_JOB}"
elif [[ -n "${directed_queue_ids}" ]]; then
  [[ "${directed_queue_ids}" =~ ^[0-9]+$ ]]
  DIRECTED_JOB=${directed_queue_ids}
  publish_job_receipt "${DIRECTED_RECEIPT}" "${DIRECTED_JOB}"
  printf 'V11_DIRECTED_QUEUE_RESUME=%s\n' "${DIRECTED_JOB}"
else
  [[ -z "$(find "${DIRECTED_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
  DIRECTED_JOB=$(sbatch --parsable --job-name="${DIRECTED_JOB_NAME}" --export=ALL,\
C2_V11_CANDIDATE_PLUGIN=${PLUGIN},\
C2_EXPECTED_V11_CANDIDATE_SHA=${PLUGIN_SHA},\
C2_V11_CANDIDATE_SOURCE=${SOURCE},\
C2_EXPECTED_V11_SOURCE_SHA=${SOURCE_SHA},\
C2_V11_CANDIDATE_RESOURCE_GATE=${RESOURCE},\
C2_EXPECTED_V11_RESOURCE_GATE_SHA=${RESOURCE_SHA},\
C2_V11_CANDIDATE_PROVENANCE=${PROVENANCE},\
C2_EXPECTED_V11_PROVENANCE_SHA=${PROVENANCE_SHA},\
C2_V11_AOT_FINAL_STATUS=${AOT_STATUS},\
C2_EXPECTED_V11_AOT_FINAL_STATUS_SHA=${AOT_STATUS_SHA},\
C2_V11_AOT_FINAL_STATUS_SIDECAR=${AOT_SIDECAR},\
C2_EXPECTED_V11_AOT_FINAL_STATUS_SIDECAR_SHA=${AOT_SIDECAR_SHA},\
C2_V11_AOT_OUTPUTS_MANIFEST=${AOT_OUTPUTS},\
C2_EXPECTED_V11_AOT_OUTPUTS_MANIFEST_SHA=${AOT_OUTPUTS_SHA},\
C2_EXPECTED_V6_DIRECTED_HARNESS_SHA=c38ec0bc7ffb9d85567d3cbbf4c2991077eb6cb3a52874778ddac01d107577a5,\
C2_EXPECTED_V11_DIRECTED_SCRIPT_SHA=${DIRECTED_SCRIPT_SHA} "${DIRECTED_SCRIPT}")
  publish_job_receipt "${DIRECTED_RECEIPT}" "${DIRECTED_JOB}"
  printf 'V11_DIRECTED=%s\n' "${DIRECTED_JOB}"
fi
DIRECTED_DIR=${DIRECTED_ROOT}/job${DIRECTED_JOB}
wait_for_job "${DIRECTED_JOB}" "${DIRECTED_DIR}/final-status-job${DIRECTED_JOB}.txt" DIRECTED

DIRECTED_RESULT=${DIRECTED_DIR}/v11-q-fragment-reuse-directed-job${DIRECTED_JOB}.json
DIRECTED_STATUS=${DIRECTED_DIR}/final-status-job${DIRECTED_JOB}.txt
DIRECTED_SIDECAR=${DIRECTED_DIR}/final-status-job${DIRECTED_JOB}.sha256
DIRECTED_OUTPUTS=${DIRECTED_DIR}/outputs-job${DIRECTED_JOB}.sha256
DIRECTED_RUN_MANIFEST=${DIRECTED_DIR}/run-manifest-job${DIRECTED_JOB}.json
grep -Eq '^FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 POST_CHECKED=1 ' "${DIRECTED_STATUS}"
sha256sum -c "${DIRECTED_SIDECAR}"
sha256sum -c "${DIRECTED_OUTPUTS}"

DIRECTED_RESULT_SHA=$(sha256sum "${DIRECTED_RESULT}" | awk '{print $1}')
DIRECTED_STATUS_SHA=$(sha256sum "${DIRECTED_STATUS}" | awk '{print $1}')
DIRECTED_SIDECAR_SHA=$(sha256sum "${DIRECTED_SIDECAR}" | awk '{print $1}')
DIRECTED_OUTPUTS_SHA=$(sha256sum "${DIRECTED_OUTPUTS}" | awk '{print $1}')
DIRECTED_RUN_MANIFEST_SHA=$(sha256sum "${DIRECTED_RUN_MANIFEST}" | awk '{print $1}')

STRESS_ROOT=${HOME_ROOT}/c2-native-plugin-v11-stress-3pct-artifacts-20260830
STRESS_RECEIPT=${HOME_ROOT}/c2-native-plugin-v11-stress-3pct-job-id-20260830.txt
STRESS_JOB_NAME=c2-v11-stress-3pct-20260830
mkdir -p -- "${STRESS_ROOT}"
stress_job_name=$(find "${STRESS_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'job*' -printf '%f\n' | sort)
stress_queue_ids=$(queued_job_ids "${STRESS_JOB_NAME}")
if [[ -f "${STRESS_RECEIPT}" ]]; then
  grep -Eq '^[0-9]+$' "${STRESS_RECEIPT}"
  STRESS_JOB=$(<"${STRESS_RECEIPT}")
  [[ -z "${stress_job_name}" || "${stress_job_name}" == "job${STRESS_JOB}" ]]
  [[ -z "${stress_queue_ids}" || "${stress_queue_ids}" == "${STRESS_JOB}" ]]
  printf 'V11_STRESS_RECEIPT_RESUME=%s\n' "${STRESS_JOB}"
elif [[ -n "${stress_job_name}" ]]; then
  [[ "${stress_job_name}" =~ ^job([0-9]+)$ ]]
  STRESS_JOB=${BASH_REMATCH[1]}
  [[ -z "${stress_queue_ids}" || "${stress_queue_ids}" == "${STRESS_JOB}" ]]
  publish_job_receipt "${STRESS_RECEIPT}" "${STRESS_JOB}"
  printf 'V11_STRESS_DISCOVERED_RESUME=%s\n' "${STRESS_JOB}"
elif [[ -n "${stress_queue_ids}" ]]; then
  [[ "${stress_queue_ids}" =~ ^[0-9]+$ ]]
  STRESS_JOB=${stress_queue_ids}
  publish_job_receipt "${STRESS_RECEIPT}" "${STRESS_JOB}"
  printf 'V11_STRESS_QUEUE_RESUME=%s\n' "${STRESS_JOB}"
else
  [[ -z "$(find "${STRESS_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
  STRESS_JOB=$(sbatch --parsable --job-name="${STRESS_JOB_NAME}" --export=ALL,\
C2_REFERENCE_PLUGIN=${V9_PLUGIN},\
C2_EXPECTED_REFERENCE_SHA=${V9_PLUGIN_SHA},\
C2_CANDIDATE_PLUGIN=${PLUGIN},\
C2_CANDIDATE_SOURCE=${SOURCE},\
C2_EXPECTED_CANDIDATE_SHA=${PLUGIN_SHA},\
C2_EXPECTED_CANDIDATE_SOURCE_SHA=${SOURCE_SHA},\
C2_CANDIDATE_RESOURCE_GATE=${RESOURCE},\
C2_EXPECTED_CANDIDATE_RESOURCE_GATE_SHA=${RESOURCE_SHA},\
C2_CANDIDATE_PROVENANCE=${PROVENANCE},\
C2_EXPECTED_CANDIDATE_PROVENANCE_SHA=${PROVENANCE_SHA},\
C2_DIRECTED_RESULT=${DIRECTED_RESULT},\
C2_EXPECTED_DIRECTED_RESULT_SHA=${DIRECTED_RESULT_SHA},\
C2_DIRECTED_FINAL_STATUS=${DIRECTED_STATUS},\
C2_EXPECTED_DIRECTED_FINAL_STATUS_SHA=${DIRECTED_STATUS_SHA},\
C2_DIRECTED_FINAL_STATUS_SIDECAR=${DIRECTED_SIDECAR},\
C2_EXPECTED_DIRECTED_FINAL_STATUS_SIDECAR_SHA=${DIRECTED_SIDECAR_SHA},\
C2_DIRECTED_OUTPUTS_MANIFEST=${DIRECTED_OUTPUTS},\
C2_EXPECTED_DIRECTED_OUTPUTS_MANIFEST_SHA=${DIRECTED_OUTPUTS_SHA},\
C2_DIRECTED_RUN_MANIFEST=${DIRECTED_RUN_MANIFEST},\
C2_EXPECTED_DIRECTED_RUN_MANIFEST_SHA=${DIRECTED_RUN_MANIFEST_SHA},\
C2_EXPECTED_RUNTIME_SCRIPT_SHA=${STRESS_SCRIPT_SHA} "${STRESS_SCRIPT}")
  publish_job_receipt "${STRESS_RECEIPT}" "${STRESS_JOB}"
  printf 'V11_STRESS=%s\n' "${STRESS_JOB}"
fi
STRESS_DIR=${STRESS_ROOT}/job${STRESS_JOB}
wait_for_job "${STRESS_JOB}" "${STRESS_DIR}/final-status-job${STRESS_JOB}.txt" STRESS

cat "${STRESS_DIR}/final-status-job${STRESS_JOB}.txt"
cat "${STRESS_DIR}/v11-q-fragment-reuse-vs-v9-3pct-incremental-decision-job${STRESS_JOB}.json"
