#!/usr/bin/env bash
# Fail-closed fresh-target integration gate for the accepted v11 q-fragment-
# reuse overlay wheel.  Each of the eight real-backend samples runs in a new
# Python process which imports only the wheel installation and a tiny reviewed
# support package.  Native and Triton are ABBA-measured only inside that fresh
# process.  Their ratio is an observation, not a v11-v9 promotion claim: the
# accepted v9->v11 direct A/B is job12905 and its two DSOs must stay isolated.

#SBATCH --job-name=c2-native-plugin-v11-fresh
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v11-fresh-backend-artifacts-20260830/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=
export PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v11-fresh-backend-artifacts-20260830
RUN_ID=${C2_RUN_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is required unless C2_RUN_ID is supplied}}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe C2 run identifier: ${RUN_ID}" >&2; exit 2; }
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${RUN_ID}
RUNTIME_ROOT=/home/lcpu/85117379/c2-native-plugin-v11-fresh-runtime-20260830
RUNTIME=${RUNTIME_ROOT}/job${RUN_ID}
SUPPORT=${RUNTIME}/c2_support
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
UV=/usr/local/bin/uv
BASELINE_VLLM=/home/lcpu/85117379/venv-vllm-d4da0c5/lib/python3.12/site-packages/vllm
C2_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/team/c2_msa_decode
HARNESS=/home/lcpu/85117379/native_c2_full_backend_bench_20260829.py
HARNESS_COPY=${ARTIFACT_DIR}/native_c2_full_backend_bench.py
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")

EXPECTED_HARNESS_SHA=7883edc25df48e3b69a9f4948775a1baafd3ba0b1bd17edcfeac6644aa7b4762
EXPECTED_BASELINE_WHEEL_SHA=91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06
EXPECTED_STABLE_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_PLUGIN_SHA=501b9c56a29a81f779a35a34b2808b4122af7dcb23e8108d8c590306be43d957
EXPECTED_PACKAGER_SHA=84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e
EXPECTED_WHEEL_FILENAME=vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl
EXPECTED_PLUGIN_MEMBER=vllm/_native_c2_msa_decode_plugin.abi3.so
EXPECTED_ADAPTER_MEMBER=vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py
EXPECTED_LOADER_MODULE=vllm.models.minimax_m3.nvidia.msa_native_c2_decode
EXPECTED_LOADER_CALLABLE=_load_native_c2_plugin_once
SEEDS=(17 23 42 2024 314159 20260801 20260815 20260829)

# Immutable AOT and promotion evidence.  None of these paths is supplied by
# the submitter, so a fresh wheel cannot be attached to an unrelated DSO.
AOT_DIR=/home/lcpu/85117379/c2-native-plugin-v11-aot-artifacts-20260830/job12825
AOT_DSO=${AOT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so
AOT_SOURCE=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v11-q-fragment-reuse-20260830-job12825/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
AOT_FINAL=${AOT_DIR}/final-status-job12825.txt
AOT_RESOURCE=${AOT_DIR}/plugin-resource-gate-job12825.txt
AOT_PROVENANCE=${AOT_DIR}/plugin-v11-provenance-job12825.sha256
DIRECTED_DIR=/home/lcpu/85117379/c2-native-plugin-v11-directed-artifacts-20260830/job12904
DIRECTED_RESULT=${DIRECTED_DIR}/v11-q-fragment-reuse-directed-job12904.json
DIRECTED_FINAL=${DIRECTED_DIR}/final-status-job12904.txt
DIRECTED_FINAL_SIDECAR=${DIRECTED_DIR}/final-status-job12904.sha256
DIRECTED_OUTPUTS=${DIRECTED_DIR}/outputs-job12904.sha256
DIRECTED_RUN_MANIFEST=${DIRECTED_DIR}/run-manifest-job12904.json
STRESS_DIR=/home/lcpu/85117379/c2-native-plugin-v11-stress-3pct-artifacts-20260830/job12905
STRESS_DECISION=${STRESS_DIR}/v11-q-fragment-reuse-vs-v9-3pct-incremental-decision-job12905.json
STRESS_AGGREGATE=${STRESS_DIR}/v11-q-fragment-reuse-vs-v9-3pct-incremental-aggregate-job12905.json
STRESS_B1=${STRESS_DIR}/candidate-v11-q-fragment-reuse-b1-round-job12905.json
STRESS_B2=${STRESS_DIR}/candidate-v11-q-fragment-reuse-b2-round-job12905.json
STRESS_FINAL=${STRESS_DIR}/final-status-job12905.txt
STRESS_FINAL_SIDECAR=${STRESS_DIR}/final-status-job12905.sha256
STRESS_OUTPUTS=${STRESS_DIR}/outputs-job12905.sha256

EXPECTED_AOT_SOURCE_SHA=0e82b278b7aa44a034a96d2ddd19946a27928d95a9f9d03e8e1fe9f30680c5b4
EXPECTED_AOT_FINAL_SHA=69bd4be581ea5a6d519753c6bf746083ad1b393e6add3fcaf6e93c0e9b2653d3
EXPECTED_AOT_RESOURCE_SHA=f82d6890d14be7cfb1df54ec12fdff0b2de3096204591b554c46eb02862d5ab4
EXPECTED_AOT_PROVENANCE_SHA=6d96a7437616c672c8640592ac99da0b722d0d6f4e7d8f78b18a3a6652b59128
EXPECTED_DIRECTED_RESULT_SHA=7567626bc80950303bde34bf3e9e5062be2338bb927c19a4184b4245dde58f93
EXPECTED_DIRECTED_FINAL_SHA=a8d67a71b4f74b5dbc236e0cbeb0322a39ad9d7f8ed2219bfd0f637faeb3e521
EXPECTED_DIRECTED_FINAL_SIDECAR_SHA=f7665b3e4814bfd581e6b58a6ab6f3431f4fa3d1667ce22bfc10973ab436c13e
EXPECTED_DIRECTED_OUTPUTS_SHA=9b3d7fb69065ba5923131d5b00cb0bb1357fc8d0376cb1e85d1b2363793c6edf
EXPECTED_DIRECTED_RUN_MANIFEST_SHA=1af137f33adcb3ab52a2fce09f5926afc5d8bccc2c008af473b2d6cdcad640c6
EXPECTED_STRESS_DECISION_SHA=ae34d33010be4f71bc9bfbbd2a0f8857e04c6071099314f538c6b406735b1569
EXPECTED_STRESS_AGGREGATE_SHA=85b9489cc36155a18c239729f4fcea7b746c67c7c3bed8b10944c73ddd7635b9
EXPECTED_STRESS_B1_SHA=f080548881afb87032cc180c6a6881604390fff308cf13255100ea6352644793
EXPECTED_STRESS_B2_SHA=fdc559b5dfd97df2e71c4a420559352f9cccca0c48a832ee857d9cb0391519bb
EXPECTED_STRESS_FINAL_SHA=003b99492f9ff2a1a1f61fba9b142f05145519a916f1978a344d143638a109bc
EXPECTED_STRESS_FINAL_SIDECAR_SHA=b39c5100667149b259e4fb227872591b95482fda8a21c77b53540f0a92a7fa62
EXPECTED_STRESS_OUTPUTS_SHA=34366fad9ca87f3b034c899ca7ed421534a4e25b15a987dd9d851c6d5b88fd98

: "${C2_WHEEL_ABSPATH:?absolute job-specific v11 overlay wheel is required}"
: "${C2_PROVENANCE_ABSPATH:?matching v11 overlay provenance is required}"
: "${C2_WHEEL_BUILD_VERIFICATION_ABSPATH:?wheel-driver verification JSON is required}"
: "${C2_WHEEL_PROMOTION_PREREQUISITES_ABSPATH:?wheel promotion-prerequisites JSON is required}"
: "${C2_WHEEL_OUTPUTS_MANIFEST_ABSPATH:?wheel outputs SHA-256 manifest is required}"
: "${C2_EXPECTED_WHEEL_SHA:?reviewed v11 wheel SHA-256 is required}"
: "${C2_EXPECTED_PROVENANCE_SHA:?reviewed v11 provenance SHA-256 is required}"
: "${C2_EXPECTED_WHEEL_BUILD_VERIFICATION_SHA:?reviewed wheel verification SHA-256 is required}"
: "${C2_EXPECTED_WHEEL_PROMOTION_PREREQUISITES_SHA:?reviewed wheel prerequisites SHA-256 is required}"
: "${C2_EXPECTED_WHEEL_OUTPUTS_MANIFEST_SHA:?reviewed wheel outputs manifest SHA-256 is required}"
: "${C2_EXPECTED_PLUGIN_SHA:?reviewed v11 plugin SHA-256 is required}"
: "${C2_EXPECTED_PACKAGER_SHA:?reviewed overlay packager SHA-256 is required}"
: "${C2_EXPECTED_HARNESS_SHA:?reviewed real-backend harness SHA-256 is required}"
: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?reviewed fresh runtime-script SHA-256 is required}"
for digest in "${C2_EXPECTED_WHEEL_SHA}" "${C2_EXPECTED_PROVENANCE_SHA}" \
  "${C2_EXPECTED_WHEEL_BUILD_VERIFICATION_SHA}" "${C2_EXPECTED_WHEEL_PROMOTION_PREREQUISITES_SHA}" \
  "${C2_EXPECTED_WHEEL_OUTPUTS_MANIFEST_SHA}" "${C2_EXPECTED_PLUGIN_SHA}" \
  "${C2_EXPECTED_PACKAGER_SHA}" "${C2_EXPECTED_HARNESS_SHA}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid SHA-256: ${digest}" >&2; exit 2; }
done
[[ "${C2_EXPECTED_PLUGIN_SHA}" == "${EXPECTED_PLUGIN_SHA}" ]]
[[ "${C2_EXPECTED_PACKAGER_SHA}" == "${EXPECTED_PACKAGER_SHA}" ]]
[[ "${C2_EXPECTED_HARNESS_SHA}" == "${EXPECTED_HARNESS_SHA}" ]]
for path in "${C2_WHEEL_ABSPATH}" "${C2_PROVENANCE_ABSPATH}" \
  "${C2_WHEEL_BUILD_VERIFICATION_ABSPATH}" "${C2_WHEEL_PROMOTION_PREREQUISITES_ABSPATH}" \
  "${C2_WHEEL_OUTPUTS_MANIFEST_ABSPATH}"; do [[ "${path}" == /* ]]; done
WHEEL=$(readlink -f "${C2_WHEEL_ABSPATH}")
PROVENANCE=$(readlink -f "${C2_PROVENANCE_ABSPATH}")
WHEEL_VERIFICATION=$(readlink -f "${C2_WHEEL_BUILD_VERIFICATION_ABSPATH}")
WHEEL_PREREQUISITES=$(readlink -f "${C2_WHEEL_PROMOTION_PREREQUISITES_ABSPATH}")
WHEEL_OUTPUTS=$(readlink -f "${C2_WHEEL_OUTPUTS_MANIFEST_ABSPATH}")
[[ "$(basename "${WHEEL}")" == "${EXPECTED_WHEEL_FILENAME}" ]]
[[ "$(basename "${PROVENANCE}")" == c2-native-plugin-overlay-provenance.json ]]
[[ "$(dirname "${WHEEL}")" == "$(dirname "${PROVENANCE}")" ]]
[[ ! -e "${ARTIFACT_DIR}" && ! -e "${RUNTIME}" ]] || { echo "refusing to reuse v11 runtime/artifact path: ${RUN_ID}" >&2; exit 2; }
[[ "${RUNTIME}" != "${C2_ROOT}" && "${RUNTIME}" != "${BASELINE_VLLM}" ]]
mkdir -p "${ARTIFACT_ROOT}" "${RUNTIME_ROOT}" "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${RUN_ID}.log") 2>&1
finalize() { local rc=$?; trap - EXIT; set +e; printf 'FINAL_RC=%s UTC=%s RUN_ID=%s\n' "${rc}" "$(date -u +%FT%TZ)" "${RUN_ID}" | tee "${ARTIFACT_DIR}/final-status-job${RUN_ID}.txt"; exit "${rc}"; }
trap finalize EXIT

selector() { local value=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}; [[ "${value}" != *,* && ( "${value}" =~ ^[0-9]+$ || "${value}" =~ ^GPU-[0-9A-Fa-f-]+$ ) ]] || return 1; printf '%s\n' "${value}"; }
gpu_uuid() { nvidia-smi --id "$(selector)" --query-gpu=uuid --format=csv,noheader,nounits | awk 'NR==1 {print $1} END {if (NR != 1) exit 1}'; }
compute_apps() { nvidia-smi --id "$(selector)" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
snapshot() { local tag=$1 gpu=$2 apps=$3; printf '\n===== %s UTC=%s =====\n' "${tag}" "$(date -u +%FT%TZ)"; nvidia-smi --id "$(selector)" --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader,nounits | tee "${gpu}"; compute_apps | tee "${apps}"; }
GPU_PRE=${ARTIFACT_DIR}/gpu-pre-job${RUN_ID}.csv; GPU_POST=${ARTIFACT_DIR}/gpu-post-job${RUN_ID}.csv
GPU_PRE_APPS=${ARTIFACT_DIR}/gpu-pre-compute-apps-job${RUN_ID}.csv; GPU_POST_APPS=${ARTIFACT_DIR}/gpu-post-compute-apps-job${RUN_ID}.csv
BASELINE_OPS=${ARTIFACT_DIR}/baseline-stable-ops-job${RUN_ID}.json; INSTALLED_PRE=${ARTIFACT_DIR}/installed-pre-plugin-ops-job${RUN_ID}.json; INSTALLED_POST=${ARTIFACT_DIR}/installed-post-plugin-ops-job${RUN_ID}.json
INSTALLED_MEMBERS=${ARTIFACT_DIR}/installed-members-job${RUN_ID}.json; SUMMARY=${ARTIFACT_DIR}/plugin-v11-fresh-backend-summary-job${RUN_ID}.json; DECISION=${ARTIFACT_DIR}/plugin-v11-fresh-backend-decision-job${RUN_ID}.json
SUPPORT_SOURCE_MANIFEST=${ARTIFACT_DIR}/support-source-job${RUN_ID}.sha256; SUPPORT_INSTALLED_MANIFEST=${ARTIFACT_DIR}/support-installed-job${RUN_ID}.sha256; UUID_MANIFEST=${ARTIFACT_DIR}/seed-gpu-uuids-job${RUN_ID}.csv

snapshot PRE "${GPU_PRE}" "${GPU_PRE_APPS}"
gpu_line=$(<"${GPU_PRE}"); [[ "${gpu_line}" == *B300* && "${gpu_line}" == *10.3* ]]
EXPECTED_GPU_UUID=$(gpu_uuid); [[ "${EXPECTED_GPU_UUID}" =~ ^GPU- ]]
[[ -z "$(<"${GPU_PRE_APPS}")" ]] || { echo 'GPU is not clean before fresh validation' >&2; exit 1; }
[[ -f "${WHEEL}" && -f "${PROVENANCE}" && -f "${WHEEL_VERIFICATION}" && -f "${WHEEL_PREREQUISITES}" && -f "${WHEEL_OUTPUTS}" && -f "${HARNESS}" && -f "${SCRIPT_PATH}" ]]
[[ -f "${AOT_DSO}" && -f "${AOT_SOURCE}" && -f "${AOT_FINAL}" && -f "${AOT_RESOURCE}" && -f "${AOT_PROVENANCE}" ]]
[[ -f "${DIRECTED_RESULT}" && -f "${DIRECTED_FINAL}" && -f "${DIRECTED_FINAL_SIDECAR}" && -f "${DIRECTED_OUTPUTS}" && -f "${DIRECTED_RUN_MANIFEST}" ]]
[[ -f "${STRESS_DECISION}" && -f "${STRESS_AGGREGATE}" && -f "${STRESS_B1}" && -f "${STRESS_B2}" && -f "${STRESS_FINAL}" && -f "${STRESS_FINAL_SIDECAR}" && -f "${STRESS_OUTPUTS}" ]]
[[ -d "${C2_ROOT}/harness" && ! -e "${C2_ROOT}/vllm" && -d "${C2_ROOT}/vllm_msa_ref" && -d "${BASELINE_VLLM}" && -x "${UV}" && -x "${PYTHON}" ]]
"${PYTHON}" -c 'raise SystemExit(0 if __debug__ else 1)'
[[ "$(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" == 1 ]]

printf '%s  %s\n' \
  "${C2_EXPECTED_WHEEL_SHA}" "${WHEEL}" "${C2_EXPECTED_PROVENANCE_SHA}" "${PROVENANCE}" \
  "${C2_EXPECTED_WHEEL_BUILD_VERIFICATION_SHA}" "${WHEEL_VERIFICATION}" \
  "${C2_EXPECTED_WHEEL_PROMOTION_PREREQUISITES_SHA}" "${WHEEL_PREREQUISITES}" \
  "${C2_EXPECTED_WHEEL_OUTPUTS_MANIFEST_SHA}" "${WHEEL_OUTPUTS}" \
  "${C2_EXPECTED_PLUGIN_SHA}" "${AOT_DSO}" "${C2_EXPECTED_PACKAGER_SHA}" /home/lcpu/85117379/build_native_c2_plugin_overlay_wheel_20260829.py \
  "${C2_EXPECTED_HARNESS_SHA}" "${HARNESS}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" \
  "${EXPECTED_AOT_SOURCE_SHA}" "${AOT_SOURCE}" "${EXPECTED_AOT_FINAL_SHA}" "${AOT_FINAL}" "${EXPECTED_AOT_RESOURCE_SHA}" "${AOT_RESOURCE}" "${EXPECTED_AOT_PROVENANCE_SHA}" "${AOT_PROVENANCE}" \
  "${EXPECTED_DIRECTED_RESULT_SHA}" "${DIRECTED_RESULT}" "${EXPECTED_DIRECTED_FINAL_SHA}" "${DIRECTED_FINAL}" "${EXPECTED_DIRECTED_FINAL_SIDECAR_SHA}" "${DIRECTED_FINAL_SIDECAR}" "${EXPECTED_DIRECTED_OUTPUTS_SHA}" "${DIRECTED_OUTPUTS}" "${EXPECTED_DIRECTED_RUN_MANIFEST_SHA}" "${DIRECTED_RUN_MANIFEST}" \
  "${EXPECTED_STRESS_DECISION_SHA}" "${STRESS_DECISION}" "${EXPECTED_STRESS_AGGREGATE_SHA}" "${STRESS_AGGREGATE}" "${EXPECTED_STRESS_B1_SHA}" "${STRESS_B1}" "${EXPECTED_STRESS_B2_SHA}" "${STRESS_B2}" "${EXPECTED_STRESS_FINAL_SHA}" "${STRESS_FINAL}" "${EXPECTED_STRESS_FINAL_SIDECAR_SHA}" "${STRESS_FINAL_SIDECAR}" "${EXPECTED_STRESS_OUTPUTS_SHA}" "${STRESS_OUTPUTS}" \
  > "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"
sha256sum -c "${DIRECTED_FINAL_SIDECAR}"; sha256sum -c "${DIRECTED_OUTPUTS}"; sha256sum -c "${STRESS_FINAL_SIDECAR}"; sha256sum -c "${STRESS_OUTPUTS}"; sha256sum -c "${WHEEL_OUTPUTS}"
grep -qxE 'BODY_RC=0 TEE_RC=0 MANIFEST_RC=0 FINAL_RC=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${AOT_FINAL}"
grep -qx 'REG=127' "${AOT_RESOURCE}"; grep -qx 'STACK=0' "${AOT_RESOURCE}"; grep -qx 'SHARED=30880' "${AOT_RESOURCE}"; grep -qx 'LOCAL=0' "${AOT_RESOURCE}"
grep -qxE 'FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 POST_CHECKED=1 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${DIRECTED_FINAL}"
grep -qxE 'FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${STRESS_FINAL}"

# Validate the pre-wheel AOT/direct evidence, and bind the derived wheel to it
# before installing a single payload byte.
"${PYTHON}" - "${AOT_PROVENANCE}" "${DIRECTED_RESULT}" "${DIRECTED_RUN_MANIFEST}" "${STRESS_DECISION}" "${STRESS_AGGREGATE}" "${STRESS_B1}" "${STRESS_B2}" "${WHEEL_VERIFICATION}" "${WHEEL_PREREQUISITES}" "${PROVENANCE}" "${WHEEL}" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
aot, directed, directed_manifest, decision, aggregate, b1, b2, wheel_check, wheel_prereq, provenance, wheel = map(Path, sys.argv[1:])
H=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected_plugin='501b9c56a29a81f779a35a34b2808b4122af7dcb23e8108d8c590306be43d957'
expected_source='0e82b278b7aa44a034a96d2ddd19946a27928d95a9f9d03e8e1fe9f30680c5b4'
records={line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0] for line in aot.read_text().splitlines()}
assert expected_plugin in records.values() and expected_source in records.values()
d=json.loads(directed.read_text()); m=json.loads(directed_manifest.read_text())
assert d['all_gates_pass'] is True and d['no_monkeypatch'] is True
assert d['profiler']['native_dispatcher_hit'] is True and d['profiler']['exactly_one_native_kernel_event'] is True
assert m['schema'] and H(directed)== '7567626bc80950303bde34bf3e9e5062be2338bb927c19a4184b4245dde58f93'
q=json.loads(decision.read_text()); a=json.loads(aggregate.read_text())
assert q['schema']=='c2-native-plugin-v11-q-fragment-reuse-vs-v9-3pct-incremental-decision-v1'
assert q['accepted'] is True and q['issues']==[] and q['all_correctness_and_resource_identity_gates_pass'] is True
assert q['aggregate_sha256']==H(aggregate) and q['input_checksums']['candidate_v11_plugin']==expected_plugin
perf=q['performance']; assert perf['paired_median_improvement_fraction']>perf['required_improvement_fraction']==.03 and perf['bootstrap_95pct_lower_bound_fraction']>0
assert a['schema']=='c2-native-plugin-v11-q-fragment-reuse-vs-v9-3pct-incremental-aggregate-v1'
assert a['all_correctness_and_comparability_gates_pass'] is True and a['issues']==[]
assert len(a['candidate_seed_mean_ms'])==len(a['reference_seed_mean_ms'])==8 and len(set(a['gpu_uuids']))==1
assert all(math.isfinite(float(x)) and float(x)>0 for x in a['candidate_seed_mean_ms']+a['reference_seed_mean_ms'])
for worker, expected_hash, role in ((b1, 'f080548881afb87032cc180c6a6881604390fff308cf13255100ea6352644793', 'candidate_v11_q_fragment_reuse_b1'), (b2, 'fdc559b5dfd97df2e71c4a420559352f9cccca0c48a832ee857d9cb0391519bb', 'candidate_v11_q_fragment_reuse_b2')):
    w=json.loads(worker.read_text()); assert H(worker)==expected_hash and w['schema']=='c2-native-plugin-v11-q-fragment-reuse-fixed-fixture-worker-v1' and w['role']==role and w['operator_library_sha256']==expected_plugin and w['all_gates_pass'] is True and len(w['seeds'])==8
    for record in w['seeds']:
        assert record['all_gates_pass'] is True and record['correctness']['allclose'] is True and record['caller_output']['pointer_unchanged'] is True and record['caller_output']['timing_pointer_unchanged'] is True and record['stability']['bitwise_repeatable'] is True and record['profiling']['pass'] is True
wc=json.loads(wheel_check.read_text()); wp=json.loads(wheel_prereq.read_text()); p=json.loads(provenance.read_text())
assert wc['schema']=='c2-native-plugin-v11-overlay-wheel-driver-verification-v1' and wc['all_gates_pass'] is True
assert wc['script_spool_same_bytes'] is True and wc['stable_member_byte_identical'] is True
assert wc['only_expected_new_members']==['vllm/_native_c2_msa_decode_plugin.abi3.so','vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py']
assert wc['wheel_sha256']==H(wheel)==p['derived_wheel']['sha256'] and wc['plugin_sha256']==expected_plugin
assert wp['schema']=='c2-native-plugin-v11-wheel-promotion-prerequisites-v1' and wp['directed_all_gates_pass'] is True and wp['stress_accepted'] is True
assert wp['inputs_sha256']['directed_result']==H(directed) and wp['inputs_sha256']['stress_decision']==H(decision) and wp['inputs_sha256']['stress_aggregate']==H(aggregate)
assert p['schema']=='c2-native-plugin-overlay-provenance-v1' and p['plugin_expected_sha256']==expected_plugin
assert p['baseline_wheel']['sha256']=='91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06'
assert p['packager']['sha256']=='84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e'
assert p['baseline_stable_member']['byte_identical'] is True and p['baseline_stable_member']['derived_sha256']=='cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
PY

"${UV}" pip install --python "${PYTHON}" --target "${RUNTIME}" --no-deps "${WHEEL}"
[[ ! -L "${RUNTIME}" && -z "$(find "${RUNTIME}" -type l -print -quit)" ]]
install -D -m 0644 "${HARNESS}" "${HARNESS_COPY}"

# Independently check wheel RECORD, all installed payloads, the stable DSO and
# every allowed dispatch replacement.  uv may rewrite RECORD/add five records;
# everything else must equal the wheel byte-for-byte.
"${PYTHON}" - "${RUNTIME}" "${WHEEL}" "${PROVENANCE}" "${INSTALLED_MEMBERS}" <<'PY'
import base64,csv,hashlib,io,json,sys,zipfile
from pathlib import Path
runtime,wheel,prov,out=map(Path,sys.argv[1:]); H=lambda b:hashlib.sha256(b).hexdigest()
plugin='vllm/_native_c2_msa_decode_plugin.abi3.so'; adapter='vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py'; stable='vllm/_C_stable_libtorch.abi3.so'
dispatch={'vllm/_custom_ops.py':'b57dfa819424b89bed8bf39924d5a4a3d2ebf327b39f3ae7c2e9e65b61c2398d','vllm/config/attention.py':'caf68c29d8d36324d252a66f0811a25b26d11da9860732359076fbb76625b5ae',adapter:'6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412','vllm/models/minimax_m3/nvidia/sparse_attention_msa.py':'07d00372ca1995aa96159a0b9f8949b1b40f817a63f49224e88aafac1b722200','vllm/v1/attention/backends/registry.py':'cd539941417fd5faf6672676de0f8ae15c9adb2a915b47cdd508e154c34faea4'}
p=json.loads(prov.read_text()); assert p['plugin_expected_sha256']=='501b9c56a29a81f779a35a34b2808b4122af7dcb23e8108d8c590306be43d957' and p['expected_dispatch_member_sha256']==dispatch
with zipfile.ZipFile(wheel) as z:
  names=[n for n in z.namelist() if not n.endswith('/')]; rec=[n for n in names if n.endswith('.dist-info/RECORD')]; assert len(rec)==1; rec=rec[0]
  rows=list(csv.reader(io.StringIO(z.read(rec).decode()))); assert {r[0] for r in rows}==set(names) and len(rows)==len({r[0] for r in rows})
  for n,e,s in rows:
    if n==rec: assert e==s==''; continue
    encoded=base64.urlsafe_b64encode(hashlib.sha256(z.read(n)).digest()).decode().rstrip('='); assert e==f'sha256={encoded}' and int(s)==len(z.read(n))
  assert H(z.read(plugin))==p['plugin_expected_sha256']; assert H(z.read(stable))=='cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
  for n,h in dispatch.items(): assert H(z.read(n))==h
  for n in names:
    f=runtime/n; assert f.is_file()
    if n!=rec: assert f.read_bytes()==z.read(n),n
  installed=list(csv.reader((runtime/rec).read_text().splitlines())); added={'bin/vllm',f'{rec.rsplit("/",1)[0]}/INSTALLER',f'{rec.rsplit("/",1)[0]}/REQUESTED',f'{rec.rsplit("/",1)[0]}/direct_url.json',f'{rec.rsplit("/",1)[0]}/uv_cache.json'}
  assert {r[0] for r in installed}==set(names)|added and len(installed)==len({r[0] for r in installed})
  for n,e,s in installed:
    f=(runtime/n).resolve(); assert f.is_relative_to(runtime.resolve()) and f.is_file()
    if n==rec: assert e==s==''; continue
    encoded=base64.urlsafe_b64encode(hashlib.sha256(f.read_bytes()).digest()).decode().rstrip('='); assert e==f'sha256={encoded}' and int(s)==f.stat().st_size
actual={str(f.relative_to(runtime)) for f in runtime.rglob('*') if f.is_file()}; assert actual=={r[0] for r in installed}|{'.lock'}
out.write_text(json.dumps({'schema':'c2-native-plugin-v11-installed-members-v1','wheel_record_full_verification':True,'installed_record_full_verification':True,'wheel_payload_except_installer_rewritten_record_byte_identical':True,'plugin_sha256':p['plugin_expected_sha256'],'stable_member_sha256':'cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442','all_gates_pass':True},indent=2,sort_keys=True)+'\n')
PY

printf '%s  %s\n' \
 'df9dd0b01c3e91afeb1a490785fab1a0554a16f001e118005192f55f891358a8' 'harness/__init__.py' \
 '38faefb89b38e3bc26b135b762a6bbbd813dcc695423982b423a806fe41c11d6' 'harness/data.py' \
 '712c1632367700137392d2c43bd5c6962130d945f01f1f585cc6338788ce53cc' 'harness/reference.py' \
 '6b0a40463b4f1c05ca53649d6689377b4f755d1efb58b10700dad05705b53f71' 'harness/triton_baseline.py' \
 '5db5d5e692044e54bb57418cbb7dcf5c9cc94f789eaccf128eec6fa8df86d967' 'vllm_msa_ref/sparse_attn.py' > "${SUPPORT_SOURCE_MANIFEST}"
(cd "${C2_ROOT}" && sha256sum -c "${SUPPORT_SOURCE_MANIFEST}")
mkdir -p "${SUPPORT}"
while read -r digest member; do source=${C2_ROOT}/${member}; [[ -f "${source}" && ! -L "${source}" ]]; install -D -m 0444 "${source}" "${SUPPORT}/${member}"; done < "${SUPPORT_SOURCE_MANIFEST}"
cp "${SUPPORT_SOURCE_MANIFEST}" "${SUPPORT_INSTALLED_MANIFEST}"; (cd "${SUPPORT}" && sha256sum -c "${SUPPORT_INSTALLED_MANIFEST}")
[[ -z "$(find "${SUPPORT}" -type l -print -quit)" && ! -e "${SUPPORT}/vllm" ]]; chmod -R a-w "${SUPPORT}"

# The pre-loader process sees the immutable vLLM base only.  The fresh target
# process imports the wheel adapter but snapshots before and after its loader.
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "${PYTHON}" - "${BASELINE_VLLM}" "${BASELINE_OPS}" <<'PY'
import hashlib,json,sys
from pathlib import Path
import torch,vllm
import vllm._C_stable_libtorch as stable
from vllm import _custom_ops
from vllm.config.attention import MiniMaxM3MSADecodeBackend
from vllm.platforms import current_platform
root,out=map(Path,sys.argv[1:]); assert Path(vllm.__file__).resolve().parent==root.resolve(); assert hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest()=='cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
def snap():
 ops=sorted(x for x in torch._C._dispatch_get_all_op_names() if x.startswith('_C::')); schemas={}
 for s in torch._C._jit_get_all_schemas():
  if s.name.startswith('_C::'): schemas.setdefault(f'{s.name}.{s.overload_name}' if s.overload_name else s.name,[]).append(str(s))
 dispatch={}
 for op in ops:
  rows={}
  for line in torch._C._dispatch_dump_table(op).splitlines():
   name,sep,registration=line.partition(': ')
   if sep:
    left=registration.rfind('['); assert left>=0 and registration.endswith(']'); rows[name]=registration[left:]
  assert rows; dispatch[op]=rows
 return ops,{k:sorted(v) for k,v in sorted(schemas.items())},dispatch
ops,schemas,dispatch=snap(); assert '_C::native_c2_msa_decode' not in ops; out.write_text(json.dumps({'schema':'c2-native-plugin-v11-baseline-dispatch-surface-v1','ops':ops,'schemas':schemas,'dispatch_key_registration_kinds':dispatch,'stable_sha256':'cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'},indent=2,sort_keys=True)+'\n')
PY
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}" "${PYTHON}" - "${RUNTIME}" "${WHEEL}" "${BASELINE_OPS}" "${INSTALLED_PRE}" "${INSTALLED_POST}" <<'PY'
import hashlib,importlib,importlib.metadata,json,sys,zipfile
from pathlib import Path
import torch,vllm
import vllm._C_stable_libtorch as stable
runtime,wheel,baseline,pre,post=map(Path,sys.argv[1:]); root=Path(vllm.__file__).resolve().parent; assert root==runtime.resolve()/'vllm'; assert importlib.metadata.distribution('vllm').locate_file('vllm').resolve()==root
assert hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest()=='cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
def snap():
 ops=sorted(x for x in torch._C._dispatch_get_all_op_names() if x.startswith('_C::')); schemas={}
 for s in torch._C._jit_get_all_schemas():
  if s.name.startswith('_C::'): schemas.setdefault(f'{s.name}.{s.overload_name}' if s.overload_name else s.name,[]).append(str(s))
 dispatch={}
 for op in ops:
  rows={}
  for line in torch._C._dispatch_dump_table(op).splitlines():
   name,sep,registration=line.partition(': ')
   if sep:
    left=registration.rfind('['); assert left>=0 and registration.endswith(']'); rows[name]=registration[left:]
  assert rows; dispatch[op]=rows
 return ops,{k:sorted(v) for k,v in sorted(schemas.items())},dispatch
b=json.loads(baseline.read_text()); mod=importlib.import_module('vllm.models.minimax_m3.nvidia.msa_native_c2_decode'); assert Path(mod.__file__).resolve()==root/'models/minimax_m3/nvidia/msa_native_c2_decode.py'
with zipfile.ZipFile(wheel) as z: assert Path(mod.__file__).read_bytes()==z.read('vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py')
ops,schemas,dispatch=snap(); assert ops==b['ops'] and schemas==b['schemas'] and dispatch==b['dispatch_key_registration_kinds'] and '_C::native_c2_msa_decode' not in ops; pre.write_text(json.dumps({'schema':'c2-native-plugin-v11-installed-preloader-surface-v1','matches_immutable_baseline':True,'ops':ops,'schemas':schemas,'dispatch_key_registration_kinds':dispatch},indent=2,sort_keys=True)+'\n')
loader=getattr(mod,'_load_native_c2_plugin_once'); assert callable(loader) and loader() is True; ops,schemas,dispatch=snap(); assert sorted(set(ops)-set(b['ops']))==['_C::native_c2_msa_decode'] and torch._C._dispatch_has_kernel_for_dispatch_key('_C::native_c2_msa_decode','CUDA'); assert all(schemas[k]==b['schemas'][k] and dispatch[k]==b['dispatch_key_registration_kinds'][k] for k in b['schemas']); assert loader() is True and snap()==(ops,schemas,dispatch)
plugin=root/'_native_c2_msa_decode_plugin.abi3.so'; assert hashlib.sha256(plugin.read_bytes()).hexdigest()=='501b9c56a29a81f779a35a34b2808b4122af7dcb23e8108d8c590306be43d957'; post.write_text(json.dumps({'schema':'c2-native-plugin-v11-installed-postloader-surface-v1','plugin_sha256':hashlib.sha256(plugin.read_bytes()).hexdigest(),'added_ops':['_C::native_c2_msa_decode'],'native_c2_has_cuda_kernel':True,'all_baseline_op_names_schemas_and_dispatch_surfaces_preserved':all(schemas[k]==b['schemas'][k] and dispatch[k]==b['dispatch_key_registration_kinds'][k] for k in b['schemas'])},indent=2,sort_keys=True)+'\n')
PY

printf 'seed,before_uuid,after_uuid,result,trace\n' > "${UUID_MANIFEST}"
for seed in "${SEEDS[@]}"; do
  result=${ARTIFACT_DIR}/plugin-v11-wheel-full-backend-seed${seed}-job${RUN_ID}.json; trace=${ARTIFACT_DIR}/plugin-v11-wheel-full-backend-seed${seed}-job${RUN_ID}.chrome.json
  before=$(gpu_uuid); [[ "${before}" == "${EXPECTED_GPU_UUID}" ]]
  PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}:${SUPPORT}" timeout --foreground --preserve-status --kill-after=10s 360s "${PYTHON}" "${HARNESS_COPY}" --c2-root "${SUPPORT}" --vllm-root "${RUNTIME}" --output "${result}" --trace "${trace}" --seed "${seed}" --timing-warmup 10 --timing-repetitions 50
  after=$(gpu_uuid); [[ "${after}" == "${EXPECTED_GPU_UUID}" ]]; printf '%s,%s,%s,%s,%s\n' "${seed}" "${before}" "${after}" "${result}" "${trace}" >> "${UUID_MANIFEST}"
done

"${PYTHON}" - "${SUMMARY}" "${DECISION}" "${UUID_MANIFEST}" "${RUNTIME}" "${ARTIFACT_DIR}" "${EXPECTED_GPU_UUID}" "${EXPECTED_PLUGIN_SHA}" "${SEEDS[@]}" <<'PY'
import csv,hashlib,json,math,statistics,sys
from pathlib import Path
summary,decision,uuids,runtime,art,uuid,dso,*ss=sys.argv[1:]; summary,decision,uuids,runtime,art=map(Path,(summary,decision,uuids,runtime,art)); seeds=list(map(int,ss)); assert len(seeds)==len(set(seeds))==8
rows=list(csv.DictReader(uuids.open())); assert [int(x['seed']) for x in rows]==seeds and all(x['before_uuid']==x['after_uuid']==uuid for x in rows)
results=[]; native=[]; triton=[]; ratios=[]; shape=None
for seed in seeds:
 pth=art/f'plugin-v11-wheel-full-backend-seed{seed}-job{art.name.removeprefix("job")}.json'; trace=art/f'plugin-v11-wheel-full-backend-seed{seed}-job{art.name.removeprefix("job")}.chrome.json'; p=json.loads(pth.read_text())
 assert p['schema']=='c2-native-c2-full-vllm-backend-v2' and p['seed']==seed and p['all_gates_pass'] is True and p['boundary']=='real MiniMaxM3SparseMSAImpl.forward backend layer; no model weights or server scheduler'
 assert p['correctness']['pass'] is True and 'independent FP32' in p['correctness']['oracle'] and p['triton_correctness']['pass'] is True
 assert p['caller_output']['pointer_unchanged'] is True and p['profiling']['checks']['one_cpu_dispatcher_event'] is True and p['profiling']['checks']['one_cuda_native_kernel_event'] is True and trace.is_file() and trace.stat().st_size>0
 t=p['timing']; assert t['pass'] is True and 'ABBA=native,triton,triton,native' in t['protocol'] and t['warmup']==10 and t['repetitions']==50 and t['native']['sample_count']==t['triton']['sample_count']==100
 n=float(t['native']['median_ms']); tr=float(t['triton']['median_ms']); assert math.isfinite(n) and math.isfinite(tr) and n>0 and tr>0 and Path(p['environment']['vllm_import_root']).resolve()==runtime/'vllm'
 here={k:p['data_contract'][k] for k in ('batch','query_heads','kv_heads','head_dim','page_size','topk')}; shape=here if shape is None else shape; assert here==shape
 native.append(n); triton.append(tr); ratios.append(n/tr); results.append({'seed':seed,'result':str(pth),'result_sha256':hashlib.sha256(pth.read_bytes()).hexdigest(),'trace':str(trace),'native_median_ms':n,'triton_median_ms':tr,'native_over_triton_latency_ratio':n/tr})
payload={'schema':'c2-native-plugin-v11-fresh-backend-summary-v1','all_integration_and_correctness_gates_pass':True,'seeds':results,'seed_count':8,'one_physical_gpu_uuid':uuid,'fresh_process_per_seed':True,'native_median_of_seed_medians_ms':statistics.median(native),'triton_median_of_seed_medians_ms':statistics.median(triton),'native_over_triton_ratio_of_seed_medians':statistics.median(ratios),'triton_parity_achieved':max(ratios)<=1.0,'triton_parity_is_hard_gate':False,'v11_dso_sha256':dso,'v11_vs_v9_fresh_wheel_comparator':{'available':False,'comparison_kind':'none','reason':'no v9 fresh-wheel comparator was run; accepted v9->v11 promotion remains the separate direct job12905 same-GPU ABBA result','fresh_native_over_triton_is_not_v11_vs_v9_promotion':True}}
summary.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); decision.write_text(json.dumps({'schema':'c2-native-plugin-v11-fresh-backend-decision-v1','accepted_for_fresh_integration':True,'final_rc':0,'hard_gates':{'wheel_identity':True,'record_and_install':True,'stable_dso_preserved':True,'loader_surface':True,'eight_seed_real_backend':True,'direct_v11_bitwise_and_promotion_evidence':True,'clean_b300_before_after':True},'performance':{'reported':True,'triton_parity_required':False,'triton_parity_achieved':payload['triton_parity_achieved'],'v11_vs_v9_fresh_wheel_comparator':'not available; not a promotion decision'},'v11_vs_v9_fresh_wheel_boundary':payload['v11_vs_v9_fresh_wheel_comparator']},indent=2,sort_keys=True)+'\n')
PY

snapshot POST "${GPU_POST}" "${GPU_POST_APPS}"; [[ "$(gpu_uuid)" == "${EXPECTED_GPU_UUID}" ]]; [[ -z "$(<"${GPU_POST_APPS}")" ]] || { echo 'GPU is not clean after fresh validation' >&2; exit 1; }
sha256sum "${SUMMARY}" "${DECISION}" "${INSTALLED_MEMBERS}" "${BASELINE_OPS}" "${INSTALLED_PRE}" "${INSTALLED_POST}" "${UUID_MANIFEST}" "${GPU_PRE}" "${GPU_POST}" "${GPU_PRE_APPS}" "${GPU_POST_APPS}" "${SUPPORT_SOURCE_MANIFEST}" "${SUPPORT_INSTALLED_MANIFEST}" "${HARNESS_COPY}" "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256" ${ARTIFACT_DIR}/plugin-v11-wheel-full-backend-seed*-job${RUN_ID}.json ${ARTIFACT_DIR}/plugin-v11-wheel-full-backend-seed*-job${RUN_ID}.chrome.json > "${ARTIFACT_DIR}/outputs-job${RUN_ID}.sha256"
exit 0
