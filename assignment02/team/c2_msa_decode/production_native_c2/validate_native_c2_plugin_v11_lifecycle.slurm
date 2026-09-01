#!/usr/bin/env bash
# Fail-closed lifecycle validation for the accepted job12957 v11 overlay wheel.
# This is an installed-wheel, fresh-interpreter integration gate.  It binds the
# wheel's construction evidence and the accepted job12960 fresh-backend result
# before exercising the production lifecycle boundary on one clean B300.

#SBATCH --job-name=c2-v11-lifecycle
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v11-lifecycle-artifacts-20260830/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=
export PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v11-lifecycle-artifacts-20260830
RUN_ID=${C2_RUN_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is required unless C2_RUN_ID is supplied}}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe C2 run identifier: ${RUN_ID}" >&2; exit 2; }
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${RUN_ID}
RUNTIME_ROOT=/home/lcpu/85117379/c2-native-plugin-v11-lifecycle-runtime-20260830
RUNTIME=${RUNTIME_ROOT}/job${RUN_ID}
SUPPORT=${RUNTIME}/c2_support
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
UV=/usr/local/bin/uv
C2_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/team/c2_msa_decode
BASE_HARNESS=/home/lcpu/85117379/native_c2_full_backend_bench_20260829.py
HARNESS=/home/lcpu/85117379/native_c2_v11_lifecycle_20260830.py
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
HARNESS_COPY=${ARTIFACT_DIR}/native_c2_v11_lifecycle.py
CORE_SOURCE=/home/lcpu/85117379/native_c2_lifecycle_core_20260830.py
CORE_COPY=${RUNTIME}/c2_lifecycle_core.py
RESULT=${ARTIFACT_DIR}/native-c2-v11-lifecycle-job${RUN_ID}.json
INSTALL_AUDIT=${ARTIFACT_DIR}/installed-wheel-audit-job${RUN_ID}.json
EVIDENCE_ATTESTATION=${ARTIFACT_DIR}/accepted-evidence-attestation-job${RUN_ID}.json

EXPECTED_WHEEL_SHA=278c01a9c99490993f807ce64b7e52e96d9513f0d22bd542cf1d60c3221e7f34
EXPECTED_PROVENANCE_SHA=b8fb03e0337c5099d4c49f419171f79c55291f62122beef5501b96aacaae9c0b
EXPECTED_PLUGIN_SHA=501b9c56a29a81f779a35a34b2808b4122af7dcb23e8108d8c590306be43d957
EXPECTED_WHEEL_VERIFICATION_SHA=cfe2e5ae4002331d1eb75e3787ffc94de54a1b97e39aab73395b8bc39c26c227
EXPECTED_WHEEL_PREREQUISITES_SHA=9f831eb0021fd4de67deaf3b930ae97ea719b7078c0dee7b00ebb8629605839e
EXPECTED_WHEEL_OUTPUTS_SHA=c4e05dec0bbdff88fca14930c7d26ea0ba5baf4f9a40a85bccbef293168a2e4d
EXPECTED_WHEEL_FINAL_SHA=c76938c217806be788c574e8ed2bb841cfa3cfcf62212590e6757600ec8eaf2c
EXPECTED_FRESH_DECISION_SHA=209c01d7e620b11eaf09aed1639f90c9778bbef94ba683818f1234b565c01e65
EXPECTED_FRESH_SUMMARY_SHA=647cd7958f51bf1d3291fc464ba2f5d8342c953bc015ac567b55c69952478228
EXPECTED_FRESH_OUTPUTS_SHA=1a598358db85f6e5e4fe0ead5bef5030abb7081641133b7bf85949d88210ad94
EXPECTED_FRESH_FINAL_SHA=2259405aad00b9424414af98549ef1b7eb7288d424adf39c9f006faa3647bc08
EXPECTED_BASE_HARNESS_SHA=7883edc25df48e3b69a9f4948775a1baafd3ba0b1bd17edcfeac6644aa7b4762
EXPECTED_LIFECYCLE_CORE_SHA=5ca35d63101de99ab5827470e37e275f7019a7f68dcbf7a6dcaec2ef892a0788
EXPECTED_STABLE_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_PACKAGER_SHA=84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e
EXPECTED_WHEEL_FILENAME=vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl

: "${C2_WHEEL_ABSPATH:?absolute accepted v11 wheel required}"
: "${C2_PROVENANCE_ABSPATH:?absolute accepted v11 provenance required}"
: "${C2_WHEEL_BUILD_VERIFICATION_ABSPATH:?job12957 verification JSON required}"
: "${C2_WHEEL_PROMOTION_PREREQUISITES_ABSPATH:?job12957 prerequisites JSON required}"
: "${C2_WHEEL_OUTPUTS_MANIFEST_ABSPATH:?job12957 outputs manifest required}"
: "${C2_WHEEL_FINAL_STATUS_ABSPATH:?job12957 final status required}"
: "${C2_FRESH_DECISION_ABSPATH:?accepted job12960 decision required}"
: "${C2_FRESH_SUMMARY_ABSPATH:?accepted job12960 summary required}"
: "${C2_FRESH_OUTPUTS_MANIFEST_ABSPATH:?accepted job12960 outputs manifest required}"
: "${C2_FRESH_FINAL_STATUS_ABSPATH:?accepted job12960 final status required}"
: "${C2_EXPECTED_WHEEL_SHA:?reviewed wheel SHA-256 required}"
: "${C2_EXPECTED_PROVENANCE_SHA:?reviewed provenance SHA-256 required}"
: "${C2_EXPECTED_WHEEL_BUILD_VERIFICATION_SHA:?reviewed verification SHA-256 required}"
: "${C2_EXPECTED_WHEEL_PROMOTION_PREREQUISITES_SHA:?reviewed prerequisites SHA-256 required}"
: "${C2_EXPECTED_WHEEL_OUTPUTS_MANIFEST_SHA:?reviewed wheel outputs SHA-256 required}"
: "${C2_EXPECTED_WHEEL_FINAL_STATUS_SHA:?reviewed wheel final-status SHA-256 required}"
: "${C2_EXPECTED_FRESH_DECISION_SHA:?reviewed fresh decision SHA-256 required}"
: "${C2_EXPECTED_FRESH_SUMMARY_SHA:?reviewed fresh summary SHA-256 required}"
: "${C2_EXPECTED_FRESH_OUTPUTS_MANIFEST_SHA:?reviewed fresh outputs SHA-256 required}"
: "${C2_EXPECTED_FRESH_FINAL_STATUS_SHA:?reviewed fresh final-status SHA-256 required}"
: "${C2_EXPECTED_PLUGIN_SHA:?reviewed v11 plugin SHA-256 required}"
: "${C2_EXPECTED_PACKAGER_SHA:?reviewed overlay packager SHA-256 required}"
: "${C2_EXPECTED_BASE_HARNESS_SHA:?reviewed base-harness SHA-256 required}"
: "${C2_EXPECTED_LIFECYCLE_CORE_SHA:?reviewed staged lifecycle-core SHA-256 required}"
: "${C2_EXPECTED_LIFECYCLE_HARNESS_SHA:?reviewed lifecycle harness SHA-256 required}"
: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?reviewed lifecycle Slurm SHA-256 required}"

for digest in "${C2_EXPECTED_WHEEL_SHA}" "${C2_EXPECTED_PROVENANCE_SHA}" \
  "${C2_EXPECTED_WHEEL_BUILD_VERIFICATION_SHA}" "${C2_EXPECTED_WHEEL_PROMOTION_PREREQUISITES_SHA}" \
  "${C2_EXPECTED_WHEEL_OUTPUTS_MANIFEST_SHA}" "${C2_EXPECTED_WHEEL_FINAL_STATUS_SHA}" \
  "${C2_EXPECTED_FRESH_DECISION_SHA}" "${C2_EXPECTED_FRESH_SUMMARY_SHA}" \
  "${C2_EXPECTED_FRESH_OUTPUTS_MANIFEST_SHA}" "${C2_EXPECTED_FRESH_FINAL_STATUS_SHA}" \
  "${C2_EXPECTED_PLUGIN_SHA}" "${C2_EXPECTED_PACKAGER_SHA}" "${C2_EXPECTED_BASE_HARNESS_SHA}" \
  "${C2_EXPECTED_LIFECYCLE_CORE_SHA}" "${C2_EXPECTED_LIFECYCLE_HARNESS_SHA}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid SHA-256: ${digest}" >&2; exit 2; }
done
[[ "${C2_EXPECTED_WHEEL_SHA}" == "${EXPECTED_WHEEL_SHA}" && "${C2_EXPECTED_PROVENANCE_SHA}" == "${EXPECTED_PROVENANCE_SHA}" ]]
[[ "${C2_EXPECTED_WHEEL_BUILD_VERIFICATION_SHA}" == "${EXPECTED_WHEEL_VERIFICATION_SHA}" && "${C2_EXPECTED_WHEEL_PROMOTION_PREREQUISITES_SHA}" == "${EXPECTED_WHEEL_PREREQUISITES_SHA}" ]]
[[ "${C2_EXPECTED_WHEEL_OUTPUTS_MANIFEST_SHA}" == "${EXPECTED_WHEEL_OUTPUTS_SHA}" && "${C2_EXPECTED_WHEEL_FINAL_STATUS_SHA}" == "${EXPECTED_WHEEL_FINAL_SHA}" ]]
[[ "${C2_EXPECTED_FRESH_DECISION_SHA}" == "${EXPECTED_FRESH_DECISION_SHA}" && "${C2_EXPECTED_FRESH_SUMMARY_SHA}" == "${EXPECTED_FRESH_SUMMARY_SHA}" ]]
[[ "${C2_EXPECTED_FRESH_OUTPUTS_MANIFEST_SHA}" == "${EXPECTED_FRESH_OUTPUTS_SHA}" && "${C2_EXPECTED_FRESH_FINAL_STATUS_SHA}" == "${EXPECTED_FRESH_FINAL_SHA}" ]]
[[ "${C2_EXPECTED_PLUGIN_SHA}" == "${EXPECTED_PLUGIN_SHA}" && "${C2_EXPECTED_PACKAGER_SHA}" == "${EXPECTED_PACKAGER_SHA}" ]]
[[ "${C2_EXPECTED_BASE_HARNESS_SHA}" == "${EXPECTED_BASE_HARNESS_SHA}" && "${C2_EXPECTED_LIFECYCLE_CORE_SHA}" == "${EXPECTED_LIFECYCLE_CORE_SHA}" ]]
for path in "${C2_WHEEL_ABSPATH}" "${C2_PROVENANCE_ABSPATH}" "${C2_WHEEL_BUILD_VERIFICATION_ABSPATH}" "${C2_WHEEL_PROMOTION_PREREQUISITES_ABSPATH}" "${C2_WHEEL_OUTPUTS_MANIFEST_ABSPATH}" "${C2_WHEEL_FINAL_STATUS_ABSPATH}" "${C2_FRESH_DECISION_ABSPATH}" "${C2_FRESH_SUMMARY_ABSPATH}" "${C2_FRESH_OUTPUTS_MANIFEST_ABSPATH}" "${C2_FRESH_FINAL_STATUS_ABSPATH}"; do [[ "${path}" == /* ]]; done
WHEEL=$(readlink -f "${C2_WHEEL_ABSPATH}")
PROVENANCE=$(readlink -f "${C2_PROVENANCE_ABSPATH}")
WHEEL_VERIFICATION=$(readlink -f "${C2_WHEEL_BUILD_VERIFICATION_ABSPATH}")
WHEEL_PREREQUISITES=$(readlink -f "${C2_WHEEL_PROMOTION_PREREQUISITES_ABSPATH}")
WHEEL_OUTPUTS=$(readlink -f "${C2_WHEEL_OUTPUTS_MANIFEST_ABSPATH}")
WHEEL_FINAL=$(readlink -f "${C2_WHEEL_FINAL_STATUS_ABSPATH}")
FRESH_DECISION=$(readlink -f "${C2_FRESH_DECISION_ABSPATH}")
FRESH_SUMMARY=$(readlink -f "${C2_FRESH_SUMMARY_ABSPATH}")
FRESH_OUTPUTS=$(readlink -f "${C2_FRESH_OUTPUTS_MANIFEST_ABSPATH}")
FRESH_FINAL=$(readlink -f "${C2_FRESH_FINAL_STATUS_ABSPATH}")
[[ "$(basename "${WHEEL}")" == "${EXPECTED_WHEEL_FILENAME}" && "$(basename "${PROVENANCE}")" == c2-native-plugin-overlay-provenance.json ]]
[[ "$(dirname "${WHEEL}")" == "$(dirname "${PROVENANCE}")" ]]
[[ "$(dirname "${WHEEL}")" == /home/lcpu/85117379/c2-native-plugin-v11-wheel-artifacts-20260830/job12957 ]]
[[ "$(dirname "${WHEEL_VERIFICATION}")" == "$(dirname "${WHEEL}")" && "$(dirname "${WHEEL_PREREQUISITES}")" == "$(dirname "${WHEEL}")" && "$(dirname "${WHEEL_OUTPUTS}")" == "$(dirname "${WHEEL}")" && "$(dirname "${WHEEL_FINAL}")" == "$(dirname "${WHEEL}")" ]]
[[ "$(dirname "${FRESH_DECISION}")" == /home/lcpu/85117379/c2-native-plugin-v11-fresh-backend-artifacts-20260830/job12960 ]]
[[ "$(dirname "${FRESH_SUMMARY}")" == "$(dirname "${FRESH_DECISION}")" && "$(dirname "${FRESH_OUTPUTS}")" == "$(dirname "${FRESH_DECISION}")" && "$(dirname "${FRESH_FINAL}")" == "$(dirname "${FRESH_DECISION}")" ]]
[[ ! -e "${ARTIFACT_DIR}" && ! -e "${RUNTIME}" ]] || { echo "refusing to reuse v11 lifecycle artifact/runtime path" >&2; exit 2; }
mkdir -p "${ARTIFACT_ROOT}" "${RUNTIME_ROOT}" "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${RUN_ID}.log") 2>&1
finalize() { local rc=$?; trap - EXIT; set +e; printf 'FINAL_RC=%s UTC=%s RUN_ID=%s\n' "${rc}" "$(date -u +%FT%TZ)" "${RUN_ID}" | tee "${ARTIFACT_DIR}/final-status-job${RUN_ID}.txt"; exit "${rc}"; }
trap finalize EXIT

selector() { local value=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}; [[ "${value}" != *,* && ( "${value}" =~ ^[0-9]+$ || "${value}" =~ ^GPU-[0-9A-Fa-f-]+$ ) ]] || return 1; printf '%s\n' "${value}"; }
gpu_uuid() { nvidia-smi --id "$(selector)" --query-gpu=uuid --format=csv,noheader,nounits | awk 'NR==1 {print $1} END {if (NR != 1) exit 1}'; }
compute_apps() { nvidia-smi --id "$(selector)" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
snapshot() { local tag=$1 gpu=$2 apps=$3; printf '\n===== %s UTC=%s =====\n' "${tag}" "$(date -u +%FT%TZ)"; nvidia-smi --id "$(selector)" --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader,nounits | tee "${gpu}"; compute_apps | tee "${apps}"; }
GPU_PRE=${ARTIFACT_DIR}/gpu-pre-job${RUN_ID}.csv
GPU_POST=${ARTIFACT_DIR}/gpu-post-job${RUN_ID}.csv
GPU_PRE_APPS=${ARTIFACT_DIR}/gpu-pre-compute-apps-job${RUN_ID}.csv
GPU_POST_APPS=${ARTIFACT_DIR}/gpu-post-compute-apps-job${RUN_ID}.csv

snapshot PRE "${GPU_PRE}" "${GPU_PRE_APPS}"
gpu_line=$(<"${GPU_PRE}")
[[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* && -z "$(<"${GPU_PRE_APPS}")" ]] || { echo 'requires an idle B300 GPU' >&2; exit 1; }
EXPECTED_GPU_UUID=$(gpu_uuid); [[ "${EXPECTED_GPU_UUID}" =~ ^GPU- ]]
[[ -x "${PYTHON}" && -x "${UV}" && -f "${BASE_HARNESS}" && -f "${HARNESS}" && -f "${CORE_SOURCE}" && -f "${SCRIPT_PATH}" && -d "${C2_ROOT}/harness" ]]
[[ -f "${WHEEL}" && -f "${PROVENANCE}" && -f "${WHEEL_VERIFICATION}" && -f "${WHEEL_PREREQUISITES}" && -f "${WHEEL_OUTPUTS}" && -f "${WHEEL_FINAL}" && -f "${FRESH_DECISION}" && -f "${FRESH_SUMMARY}" && -f "${FRESH_OUTPUTS}" && -f "${FRESH_FINAL}" ]]
[[ ! -L "${CORE_SOURCE}" ]] || { echo 'lifecycle core must be a staged regular file' >&2; exit 2; }
CORE_SOURCE=$(readlink -f "${CORE_SOURCE}")

printf '%s  %s\n' \
  "${C2_EXPECTED_WHEEL_SHA}" "${WHEEL}" "${C2_EXPECTED_PROVENANCE_SHA}" "${PROVENANCE}" \
  "${C2_EXPECTED_WHEEL_BUILD_VERIFICATION_SHA}" "${WHEEL_VERIFICATION}" "${C2_EXPECTED_WHEEL_PROMOTION_PREREQUISITES_SHA}" "${WHEEL_PREREQUISITES}" "${C2_EXPECTED_WHEEL_OUTPUTS_MANIFEST_SHA}" "${WHEEL_OUTPUTS}" "${C2_EXPECTED_WHEEL_FINAL_STATUS_SHA}" "${WHEEL_FINAL}" \
  "${C2_EXPECTED_FRESH_DECISION_SHA}" "${FRESH_DECISION}" "${C2_EXPECTED_FRESH_SUMMARY_SHA}" "${FRESH_SUMMARY}" "${C2_EXPECTED_FRESH_OUTPUTS_MANIFEST_SHA}" "${FRESH_OUTPUTS}" "${C2_EXPECTED_FRESH_FINAL_STATUS_SHA}" "${FRESH_FINAL}" \
  "${C2_EXPECTED_BASE_HARNESS_SHA}" "${BASE_HARNESS}" "${C2_EXPECTED_LIFECYCLE_CORE_SHA}" "${CORE_SOURCE}" "${C2_EXPECTED_LIFECYCLE_HARNESS_SHA}" "${HARNESS}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" \
  > "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"
sha256sum -c "${WHEEL_OUTPUTS}"; sha256sum -c "${FRESH_OUTPUTS}"
grep -qxE 'FINAL_RC=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${WHEEL_FINAL}"
grep -qxE 'FINAL_RC=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z RUN_ID=12960' "${FRESH_FINAL}"

"${PYTHON}" - "${WHEEL}" "${PROVENANCE}" "${WHEEL_VERIFICATION}" "${WHEEL_PREREQUISITES}" "${FRESH_DECISION}" "${FRESH_SUMMARY}" "${EVIDENCE_ATTESTATION}" <<'PY'
import hashlib, json, sys
from pathlib import Path
wheel, prov, check, prereq, decision, summary, output = map(Path, sys.argv[1:])
H = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plugin = '501b9c56a29a81f779a35a34b2808b4122af7dcb23e8108d8c590306be43d957'
wheel_sha = '278c01a9c99490993f807ce64b7e52e96d9513f0d22bd542cf1d60c3221e7f34'
p = json.loads(prov.read_text()); c = json.loads(check.read_text()); r = json.loads(prereq.read_text()); d = json.loads(decision.read_text()); s = json.loads(summary.read_text())
assert H(wheel) == wheel_sha == p['derived_wheel']['sha256'] == c['wheel_sha256']
assert p['schema'] == 'c2-native-plugin-overlay-provenance-v1' and p['plugin_expected_sha256'] == plugin
assert p['baseline_stable_member']['byte_identical'] is True and p['baseline_stable_member']['derived_sha256'] == 'cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
assert c['schema'] == 'c2-native-plugin-v11-overlay-wheel-driver-verification-v1' and c['all_gates_pass'] is True
assert c['script_spool_same_bytes'] is True and c['stable_member_byte_identical'] is True and c['plugin_sha256'] == plugin
assert r['schema'] == 'c2-native-plugin-v11-wheel-promotion-prerequisites-v1' and r['directed_all_gates_pass'] is True and r['stress_accepted'] is True
assert d['schema'] == 'c2-native-plugin-v11-fresh-backend-decision-v1' and d['accepted_for_fresh_integration'] is True and d['final_rc'] == 0
assert all(d['hard_gates'].values()) and d['performance']['triton_parity_required'] is False
assert s['schema'] == 'c2-native-plugin-v11-fresh-backend-summary-v1' and s['all_integration_and_correctness_gates_pass'] is True and s['seed_count'] == 8 and s['fresh_process_per_seed'] is True and s['v11_dso_sha256'] == plugin
output.write_text(json.dumps({'schema':'c2-native-plugin-v11-lifecycle-accepted-evidence-attestation-v1','wheel_sha256':wheel_sha,'plugin_sha256':plugin,'wheel_verification_sha256':H(check),'wheel_prerequisites_sha256':H(prereq),'fresh_decision_sha256':H(decision),'fresh_summary_sha256':H(summary),'all_gates_pass':True}, indent=2, sort_keys=True) + '\n')
PY

"${UV}" pip install --python "${PYTHON}" --target "${RUNTIME}" --no-deps "${WHEEL}"
[[ ! -L "${RUNTIME}" && -z "$(find "${RUNTIME}" -type l -print -quit)" ]]
install -D -m 0444 "${HARNESS}" "${HARNESS_COPY}"
install -D -m 0444 "${CORE_SOURCE}" "${CORE_COPY}"
printf '%s  %s\n' \
  'df9dd0b01c3e91afeb1a490785fab1a0554a16f001e118005192f55f891358a8' 'harness/__init__.py' \
  '38faefb89b38e3bc26b135b762a6bbbd813dcc695423982b423a806fe41c11d6' 'harness/data.py' \
  '712c1632367700137392d2c43bd5c6962130d945f01f1f585cc6338788ce53cc' 'harness/reference.py' \
  '6b0a40463b4f1c05ca53649d6689377b4f755d1efb58b10700dad05705b53f71' 'harness/triton_baseline.py' \
  '5db5d5e692044e54bb57418cbb7dcf5c9cc94f789eaccf128eec6fa8df86d967' 'vllm_msa_ref/sparse_attn.py' > "${ARTIFACT_DIR}/support-source-job${RUN_ID}.sha256"
(cd "${C2_ROOT}" && sha256sum -c "${ARTIFACT_DIR}/support-source-job${RUN_ID}.sha256")
while read -r digest member; do source=${C2_ROOT}/${member}; [[ -f "${source}" && ! -L "${source}" ]]; install -D -m 0444 "${source}" "${SUPPORT}/${member}"; done < "${ARTIFACT_DIR}/support-source-job${RUN_ID}.sha256"
cp "${ARTIFACT_DIR}/support-source-job${RUN_ID}.sha256" "${ARTIFACT_DIR}/support-installed-job${RUN_ID}.sha256"
(cd "${SUPPORT}" && sha256sum -c "${ARTIFACT_DIR}/support-installed-job${RUN_ID}.sha256")
chmod -R a-w "${SUPPORT}" "${HARNESS_COPY}" "${CORE_COPY}"

"${PYTHON}" - "${RUNTIME}" "${WHEEL}" "${PROVENANCE}" "${INSTALL_AUDIT}" <<'PY'
import hashlib, json, sys, zipfile
from pathlib import Path
runtime, wheel, provenance, output = map(Path, sys.argv[1:])
p = json.loads(provenance.read_text()); root = runtime / 'vllm'
stable = root / '_C_stable_libtorch.abi3.so'; plugin = root / '_native_c2_msa_decode_plugin.abi3.so'; adapter = root / 'models/minimax_m3/nvidia/msa_native_c2_decode.py'
assert all(x.is_file() for x in (stable, plugin, adapter))
with zipfile.ZipFile(wheel) as z:
    assert stable.read_bytes() == z.read('vllm/_C_stable_libtorch.abi3.so')
    assert plugin.read_bytes() == z.read('vllm/_native_c2_msa_decode_plugin.abi3.so')
    assert adapter.read_bytes() == z.read('vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py')
assert hashlib.sha256(stable.read_bytes()).hexdigest() == 'cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
assert hashlib.sha256(plugin.read_bytes()).hexdigest() == p['plugin_expected_sha256'] == '501b9c56a29a81f779a35a34b2808b4122af7dcb23e8108d8c590306be43d957'
output.write_text(json.dumps({'schema':'c2-native-plugin-v11-lifecycle-installed-wheel-v1','runtime':str(runtime),'stable_sha256':hashlib.sha256(stable.read_bytes()).hexdigest(),'plugin_sha256':hashlib.sha256(plugin.read_bytes()).hexdigest(),'adapter_sha256':hashlib.sha256(adapter.read_bytes()).hexdigest(),'all_gates_pass':True}, indent=2, sort_keys=True) + '\n')
PY

PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}:${SUPPORT}" \
  timeout --foreground --preserve-status --kill-after=10s 2400s \
  "${PYTHON}" "${HARNESS_COPY}" --c2-root "${SUPPORT}" --vllm-root "${RUNTIME}" --base-harness "${BASE_HARNESS}" \
  --lifecycle-core "${CORE_COPY}" --expected-lifecycle-core-sha "${EXPECTED_LIFECYCLE_CORE_SHA}" --output "${RESULT}" --seed 20260830 \
  --steady-iterations 1000 --graph-replays 100 --memory-bound-bytes 8388608

"${PYTHON}" - "${RESULT}" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
assert p['schema'] == 'c2-native-c2-v11-production-lifecycle-v1' and p['all_gates_pass'] is True
required = {'fresh_adapter_eight_thread_first_load','real_forward_reaches_dispatcher_and_native_cuda_kernel','steady_1000_real_forward_lifecycle','cuda_graph_capture_replay_and_static_query_update','native_support_predicate_matrix','dynamic_native_rejection_without_fallback_claim','native_forward_sequence_boundaries','dispatcher_surface_persists_through_lifecycle'}
assert set(p['gates']) == required and all(p['gates'].values())
assert p['first_load']['concurrent_loader']['thread_count'] == 8 and p['first_load']['concurrent_loader']['load_library_call_count'] == 1
assert p['steady_lifecycle']['iterations'] == 1000
assert p['cuda_graph']['original_query']['replays']['replays'] == 100 and p['cuda_graph']['updated_query']['replays']['replays'] == 100
assert p['dynamic_rejection_gate']['mode'] == 'selection_only' and all(x['pass'] for x in p['dynamic_rejection_gate']['rows'])
assert [x['sequence_length'] for x in p['sequence_length_boundaries']['rows']] == [2048, 2049, 4095, 4096] and all(x['pass'] for x in p['sequence_length_boundaries']['rows'])
assert p['lifecycle_core']['schema'] == 'c2-native-c2-v11-lifecycle-core-binding-v1' and p['lifecycle_core']['sha256'] == '5ca35d63101de99ab5827470e37e275f7019a7f68dcbf7a6dcaec2ef892a0788'
PY

snapshot POST "${GPU_POST}" "${GPU_POST_APPS}"
[[ "$(gpu_uuid)" == "${EXPECTED_GPU_UUID}" && -z "$(<"${GPU_POST_APPS}")" ]] || { echo 'GPU is not clean after v11 lifecycle validation' >&2; exit 1; }
sha256sum "${RESULT}" "${INSTALL_AUDIT}" "${EVIDENCE_ATTESTATION}" "${HARNESS_COPY}" "${CORE_COPY}" \
  "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256" "${ARTIFACT_DIR}/support-source-job${RUN_ID}.sha256" "${ARTIFACT_DIR}/support-installed-job${RUN_ID}.sha256" \
  "${GPU_PRE}" "${GPU_POST}" "${GPU_PRE_APPS}" "${GPU_POST_APPS}" > "${ARTIFACT_DIR}/outputs-job${RUN_ID}.sha256"
exit 0
