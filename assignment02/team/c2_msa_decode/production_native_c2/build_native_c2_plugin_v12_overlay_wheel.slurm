#!/usr/bin/env bash
# Build only a local experimental overlay wheel for the accepted v12
# Q-row-padding DSO.  This is a fail-closed packaging/inventory gate, not an
# upstream change, release artifact, or new performance measurement.

#SBATCH --job-name=c2-native-plugin-v12-wheel
#SBATCH --partition=cpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v12-wheel-artifacts-20260830/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONPATH=

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v12-wheel-artifacts-20260830
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID:?must run under Slurm}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
PACKAGER=/home/lcpu/85117379/build_native_c2_plugin_overlay_wheel_20260829.py
BASE_WHEEL=/home/lcpu/85117379/vllm-d4-wheel/vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl
CHECKOUT=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v12-q-row-padding-20260830-job12983
BUILD_DSO=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v12-q-row-padding-build-20260830-job12983/_native_c2_msa_decode_plugin.abi3.so
AOT_DIR=/home/lcpu/85117379/c2-native-plugin-v12-aot-artifacts-20260830/job12983
DIRECTED_DIR=/home/lcpu/85117379/c2-native-plugin-v12-directed-artifacts-20260830/job12984
STRESS_DIR=/home/lcpu/85117379/c2-native-plugin-v12-stress-3pct-artifacts-20260830/job12985
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SUBMITTED_SCRIPT=/home/lcpu/85117379/build_native_c2_plugin_v12_overlay_wheel_20260830.slurm

EXPECTED_D4_COMMIT=d4da0c55af3aa231b6209bf77871f3ed36eab0d2
EXPECTED_BASELINE_SHA=91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06
EXPECTED_STABLE_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_PACKAGER_SHA=84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e
EXPECTED_ADAPTER_SHA=6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412
EXPECTED_SOURCE_SHA=535d90b856ed1062aa7b8a105eb2c5f236c450826e65496e646d0d5a27eb8aaf
EXPECTED_PLUGIN_SHA=064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6

AOT_DSO=${AOT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so
AOT_SOURCE=${AOT_DIR}/native_c2_decode.v12-q-row-padding.cu
AOT_FINAL=${AOT_DIR}/final-status-job12983.txt
AOT_FINAL_SIDECAR=${AOT_DIR}/final-status-job12983.sha256
AOT_OUTPUTS=${AOT_DIR}/outputs-job12983.sha256
AOT_RESOURCE=${AOT_DIR}/plugin-resource-gate-job12983.txt
AOT_PROVENANCE=${AOT_DIR}/plugin-v12-provenance-job12983.sha256
DIRECTED_RESULT=${DIRECTED_DIR}/v12-q-row-padding-directed-job12984.json
DIRECTED_FINAL=${DIRECTED_DIR}/final-status-job12984.txt
DIRECTED_FINAL_SIDECAR=${DIRECTED_DIR}/final-status-job12984.sha256
DIRECTED_OUTPUTS=${DIRECTED_DIR}/outputs-job12984.sha256
DIRECTED_MANIFEST=${DIRECTED_DIR}/run-manifest-job12984.json
STRESS_DECISION=${STRESS_DIR}/v12-q-row-padding-vs-v11-3pct-incremental-decision-job12985.json
STRESS_AGGREGATE=${STRESS_DIR}/v12-q-row-padding-vs-v11-3pct-incremental-aggregate-job12985.json
STRESS_FINAL=${STRESS_DIR}/final-status-job12985.txt
STRESS_FINAL_SIDECAR=${STRESS_DIR}/final-status-job12985.sha256
STRESS_OUTPUTS=${STRESS_DIR}/outputs-job12985.sha256

EXPECTED_AOT_FINAL_SHA=b5fd3d82083165415ba20c37cf574e898ef0209d1a7844866c7491cc438b0f18
EXPECTED_AOT_FINAL_SIDECAR_SHA=912b28b5f81610b7cc04457c54a26165d4e321cfe709bdc79489a3b102041f67
EXPECTED_AOT_OUTPUTS_SHA=ca1f37f00f9ce78c0d88bb3a90e8288bbffd83e94315a378b36a9c7998d99419
EXPECTED_AOT_RESOURCE_SHA=c4a6cd69617ce82e99459fcd7a975a37fbfad497e51b36a6ec09945ffa97d6c0
EXPECTED_AOT_PROVENANCE_SHA=65bb886cece4a49b7ef7d7d4cab00588e4cc1f9238d2aa45dd5cf36f6952d2a2
EXPECTED_DIRECTED_RESULT_SHA=240b683c1b6921981cc814084697ff945a7714de1bace5956f74a09ea38765c8
EXPECTED_DIRECTED_FINAL_SHA=16026037d3916cf0dc741317e35ca258536ab3a8012f2671a0886e2c49ee2758
EXPECTED_DIRECTED_FINAL_SIDECAR_SHA=9da4e355841c9448683cd82d3281112753e2accdf519b7fdad05b60f8a3bbc5a
EXPECTED_DIRECTED_OUTPUTS_SHA=0b1b512ef6bf0c8a5b461561f6fd4b5b768e4922f75c0285ce8bac37b793bf7a
EXPECTED_DIRECTED_MANIFEST_SHA=d07f6028ed6ff0815cb0019a3df30235edc3a3b918cded23b413feaaf94bacde
EXPECTED_STRESS_DECISION_SHA=829505fce32c737f596b372819add7f7701dc793c87d58f429e61ba7806862d9
EXPECTED_STRESS_AGGREGATE_SHA=0e9b4da6ea78c812c66feab0d4c669e68d0dc9a65630d13fbbc37c2e66295142
EXPECTED_STRESS_FINAL_SHA=4e8c40bb6784b111dced540a2b977ff383d2a5cb41796a65711ef609bf409ca1
EXPECTED_STRESS_FINAL_SIDECAR_SHA=042449479aa483c2b3f1b345e04c89d0c711322b2ee4b82014efd4a40a54468e
EXPECTED_STRESS_OUTPUTS_SHA=799ee63d1dd815d381fcc866687da97bf52177994a490df7ab8b5f07e5bf78cc

: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?reviewed v12 wheel-script SHA-256 required}"
for digest in "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${EXPECTED_BASELINE_SHA}" "${EXPECTED_STABLE_SHA}" "${EXPECTED_PACKAGER_SHA}" "${EXPECTED_ADAPTER_SHA}" "${EXPECTED_SOURCE_SHA}" "${EXPECTED_PLUGIN_SHA}" "${EXPECTED_AOT_FINAL_SHA}" "${EXPECTED_AOT_FINAL_SIDECAR_SHA}" "${EXPECTED_AOT_OUTPUTS_SHA}" "${EXPECTED_AOT_RESOURCE_SHA}" "${EXPECTED_AOT_PROVENANCE_SHA}" "${EXPECTED_DIRECTED_RESULT_SHA}" "${EXPECTED_DIRECTED_FINAL_SHA}" "${EXPECTED_DIRECTED_FINAL_SIDECAR_SHA}" "${EXPECTED_DIRECTED_OUTPUTS_SHA}" "${EXPECTED_DIRECTED_MANIFEST_SHA}" "${EXPECTED_STRESS_DECISION_SHA}" "${EXPECTED_STRESS_AGGREGATE_SHA}" "${EXPECTED_STRESS_FINAL_SHA}" "${EXPECTED_STRESS_FINAL_SIDECAR_SHA}" "${EXPECTED_STRESS_OUTPUTS_SHA}"; do [[ "${digest}" =~ ^[0-9a-f]{64}$ ]]; done
[[ -f "${SCRIPT_PATH}" && -f "${SUBMITTED_SCRIPT}" ]] || { echo 'missing executed/submitted wheel script' >&2; exit 2; }
cmp -s "${SCRIPT_PATH}" "${SUBMITTED_SCRIPT}" || { echo 'spooled script differs from reviewed submitted copy' >&2; exit 2; }
[[ ! -e "${ARTIFACT_DIR}" ]] || { echo "refusing to reuse ${ARTIFACT_DIR}" >&2; exit 2; }
mkdir -p "${ARTIFACT_ROOT}" "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1
finalize() {
  local rc=$?; trap - EXIT; set +e
  printf 'FINAL_RC=%s UTC=%s\n' "${rc}" "$(date -u +%FT%TZ)" | tee "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt"
  find "${ARTIFACT_DIR}" -maxdepth 1 -type f ! -name 'driver-job*.log' ! -name 'final-output-manifest-job*.sha256' -printf '%p\0' | sort -z | xargs -0r sha256sum > "${ARTIFACT_DIR}/final-output-manifest-job${SLURM_JOB_ID}.sha256" || rc=2
  exit "${rc}"
}
trap finalize EXIT

for path in "${BASE_WHEEL}" "${PACKAGER}" "${AOT_DSO}" "${BUILD_DSO}" "${AOT_SOURCE}" "${AOT_FINAL}" "${AOT_FINAL_SIDECAR}" "${AOT_OUTPUTS}" "${AOT_RESOURCE}" "${AOT_PROVENANCE}" "${DIRECTED_RESULT}" "${DIRECTED_FINAL}" "${DIRECTED_FINAL_SIDECAR}" "${DIRECTED_OUTPUTS}" "${DIRECTED_MANIFEST}" "${STRESS_DECISION}" "${STRESS_AGGREGATE}" "${STRESS_FINAL}" "${STRESS_FINAL_SIDECAR}" "${STRESS_OUTPUTS}"; do [[ -f "${path}" ]]; done
[[ -x "${PYTHON}" && -d "${CHECKOUT}" ]] ; "${PYTHON}" -c 'raise SystemExit(0 if __debug__ else 1)'; "${PYTHON}" -m py_compile "${PACKAGER}"
[[ "$(git -C "${CHECKOUT}" rev-parse HEAD)" == "${EXPECTED_D4_COMMIT}" ]]; git -C "${CHECKOUT}" diff --check
[[ "$(sha256sum "${CHECKOUT}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu" | awk '{print $1}')" == "${EXPECTED_SOURCE_SHA}" ]]
grep -qxE 'BODY_RC=0 TEE_RC=0 MANIFEST_RC=0 FINAL_RC=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${AOT_FINAL}"
grep -qx 'REG=127' "${AOT_RESOURCE}"; grep -qx 'STACK=0' "${AOT_RESOURCE}"; grep -qx 'SHARED=31136' "${AOT_RESOURCE}"; grep -qx 'LOCAL=0' "${AOT_RESOURCE}"
grep -qxE 'FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 POST_CHECKED=1 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${DIRECTED_FINAL}"
grep -qxE 'FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${STRESS_FINAL}"

printf '%s  %s\n' "${EXPECTED_BASELINE_SHA}" "${BASE_WHEEL}" "${EXPECTED_PACKAGER_SHA}" "${PACKAGER}" "${EXPECTED_ADAPTER_SHA}" "${CHECKOUT}/vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py" "${EXPECTED_SOURCE_SHA}" "${AOT_SOURCE}" "${EXPECTED_SOURCE_SHA}" "${CHECKOUT}/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu" "${EXPECTED_PLUGIN_SHA}" "${AOT_DSO}" "${EXPECTED_PLUGIN_SHA}" "${BUILD_DSO}" "${EXPECTED_AOT_FINAL_SHA}" "${AOT_FINAL}" "${EXPECTED_AOT_FINAL_SIDECAR_SHA}" "${AOT_FINAL_SIDECAR}" "${EXPECTED_AOT_OUTPUTS_SHA}" "${AOT_OUTPUTS}" "${EXPECTED_AOT_RESOURCE_SHA}" "${AOT_RESOURCE}" "${EXPECTED_AOT_PROVENANCE_SHA}" "${AOT_PROVENANCE}" "${EXPECTED_DIRECTED_RESULT_SHA}" "${DIRECTED_RESULT}" "${EXPECTED_DIRECTED_FINAL_SHA}" "${DIRECTED_FINAL}" "${EXPECTED_DIRECTED_FINAL_SIDECAR_SHA}" "${DIRECTED_FINAL_SIDECAR}" "${EXPECTED_DIRECTED_OUTPUTS_SHA}" "${DIRECTED_OUTPUTS}" "${EXPECTED_DIRECTED_MANIFEST_SHA}" "${DIRECTED_MANIFEST}" "${EXPECTED_STRESS_DECISION_SHA}" "${STRESS_DECISION}" "${EXPECTED_STRESS_AGGREGATE_SHA}" "${STRESS_AGGREGATE}" "${EXPECTED_STRESS_FINAL_SHA}" "${STRESS_FINAL}" "${EXPECTED_STRESS_FINAL_SIDECAR_SHA}" "${STRESS_FINAL_SIDECAR}" "${EXPECTED_STRESS_OUTPUTS_SHA}" "${STRESS_OUTPUTS}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SUBMITTED_SCRIPT}" > "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
sha256sum -c "${AOT_FINAL_SIDECAR}"; sha256sum -c "${AOT_OUTPUTS}"; sha256sum -c "${AOT_PROVENANCE}"; sha256sum -c "${DIRECTED_FINAL_SIDECAR}"; sha256sum -c "${DIRECTED_OUTPUTS}"; sha256sum -c "${STRESS_FINAL_SIDECAR}"; sha256sum -c "${STRESS_OUTPUTS}"

"${PYTHON}" - "${ARTIFACT_DIR}/promotion-prerequisites-job${SLURM_JOB_ID}.json" "${DIRECTED_RESULT}" "${STRESS_DECISION}" "${STRESS_AGGREGATE}" "${AOT_PROVENANCE}" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, directed_path, decision_path, aggregate_path, provenance_path = map(Path, sys.argv[1:])
H=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads(directed_path.read_text()); q=json.loads(decision_path.read_text()); a=json.loads(aggregate_path.read_text())
assert d['all_gates_pass'] is True and d['no_monkeypatch'] is True
assert d['profiler']['native_dispatcher_hit'] is True and d['profiler']['exactly_one_native_kernel_event'] is True
assert q['schema']=='c2-native-plugin-v12-q-row-padding-vs-v11-3pct-incremental-decision-v1'
assert q['accepted'] is True and q['issues']==[] and q['all_correctness_and_resource_identity_gates_pass'] is True
assert q['aggregate_sha256']==H(aggregate_path) and q['input_checksums']['candidate_v12_plugin']=='064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6'
p=q['performance']; assert p['paired_median_improvement_fraction']>p['required_improvement_fraction']==.03 and p['bootstrap_95pct_lower_bound_fraction']>p['required_bootstrap_lower_bound_fraction']==0
assert a['all_correctness_and_comparability_gates_pass'] is True and a['issues']==[] and len(set(a['gpu_uuids']))==1 and a['allocated_gpu_uuid']==a['gpu_uuids'][0]
records={line.split(maxsplit=1)[1]:line.split(maxsplit=1)[0] for line in provenance_path.read_text().splitlines()}
assert '064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6' in records.values()
out.write_text(json.dumps({'schema':'c2-native-plugin-v12-wheel-promotion-prerequisites-v1','aot_job_id':12983,'directed_job_id':12984,'stress_job_id':12985,'directed_all_gates_pass':True,'stress_accepted':True,'paired_median_improvement_fraction':p['paired_median_improvement_fraction'],'bootstrap_95pct_lower_bound_fraction':p['bootstrap_95pct_lower_bound_fraction'],'one_physical_gpu_uuid':a['allocated_gpu_uuid'],'inputs_sha256':{'directed_result':H(directed_path),'stress_decision':H(decision_path),'stress_aggregate':H(aggregate_path),'aot_provenance':H(provenance_path)}},indent=2,sort_keys=True)+'\n')
PY

"${PYTHON}" "${PACKAGER}" --base-wheel "${BASE_WHEEL}" --plugin-checkout "${CHECKOUT}" --plugin-library "${AOT_DSO}" --plugin-archive-member vllm/_native_c2_msa_decode_plugin.abi3.so --expected-plugin-sha256 "${EXPECTED_PLUGIN_SHA}" --artifact-dir "${ARTIFACT_DIR}" | tee "${ARTIFACT_DIR}/result-job${SLURM_JOB_ID}.json"

"${PYTHON}" - "${ARTIFACT_DIR}" "${BASE_WHEEL}" "${AOT_DSO}" "${ARTIFACT_DIR}/wheel-driver-verification-job${SLURM_JOB_ID}.json" <<'PY'
import base64,csv,hashlib,io,json,sys,zipfile
from pathlib import Path
art,base,plugin,out=map(Path,sys.argv[1:]); H=lambda b:hashlib.sha256(b).hexdigest()
plugin_member='vllm/_native_c2_msa_decode_plugin.abi3.so'; adapter='vllm/models/minimax_m3/nvidia/msa_native_c2_decode.py'; stable='vllm/_C_stable_libtorch.abi3.so'
dispatch={'vllm/_custom_ops.py':'b57dfa819424b89bed8bf39924d5a4a3d2ebf327b39f3ae7c2e9e65b61c2398d','vllm/config/attention.py':'caf68c29d8d36324d252a66f0811a25b26d11da9860732359076fbb76625b5ae','vllm/v1/attention/backends/registry.py':'cd539941417fd5faf6672676de0f8ae15c9adb2a915b47cdd508e154c34faea4',adapter:'6d75e6eec2d6e7d3006fdc1edf65be96f634a2a8bdcc1ba878393988e5fc8412','vllm/models/minimax_m3/nvidia/sparse_attention_msa.py':'07d00372ca1995aa96159a0b9f8949b1b40f817a63f49224e88aafac1b722200'}
def record(z,names,rec):
 rows=list(csv.reader(io.StringIO(z.read(rec).decode()))); idx={x[0]:x for x in rows}; payload={n for n in names if not n.endswith('/')}-{rec}; assert len(idx)==len(rows) and set(idx)-{rec}==payload and idx[rec]==[rec,'','']
 for n in payload:
  data=z.read(n); encoded=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('='); assert idx[n]==[n,f'sha256={encoded}',str(len(data))]
 return {'record_member':rec,'payload_count':len(payload),'all_payload_entries_validated':True}
wheel=art/'vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl'; p=json.loads((art/'c2-native-plugin-overlay-provenance.json').read_text()); m=json.loads((art/'c2-native-plugin-overlay-manifest.json').read_text())
assert H(wheel.read_bytes())==p['derived_wheel']['sha256']==m['derived_wheel_sha256'] and p['baseline_wheel']['sha256']=='91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06' and p['packager']['sha256']=='84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e'
with zipfile.ZipFile(base) as b, zipfile.ZipFile(wheel) as d:
 assert b.testzip() is None and d.testzip() is None; bn=b.namelist(); dn=d.namelist(); assert len(bn)==len(set(bn)) and len(dn)==len(set(dn)); br=[n for n in bn if n.endswith('.dist-info/RECORD')]; dr=[n for n in dn if n.endswith('.dist-info/RECORD')]; assert len(br)==len(dr)==1
 rb=record(b,bn,br[0]); rd=record(d,dn,dr[0]); assert set(dn)==set(bn)|{plugin_member,adapter}; approved=set(dispatch)|{br[0]}; unchanged=[]
 for n in bn:
  if n.endswith('/') or n in approved: continue
  assert b.read(n)==d.read(n),n; unchanged.append(n)
 assert b.read(stable)==d.read(stable) and H(d.read(stable))=='cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'; assert d.read(plugin_member)==plugin.read_bytes() and H(d.read(plugin_member))=='064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6'
 for n,h in dispatch.items(): assert H(d.read(n))==h
out.write_text(json.dumps({'schema':'c2-native-plugin-v12-overlay-wheel-driver-verification-v1','wheel':str(wheel),'wheel_sha256':H(wheel.read_bytes()),'plugin_sha256':'064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6','stable_member_sha256':'cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442','stable_member_byte_identical':True,'only_expected_new_members':[plugin_member,adapter],'only_existing_dispatch_replacements':sorted(dispatch),'record':{'baseline':rb,'derived':rd},'unchanged_baseline_payload_count':len(unchanged),'all_gates_pass':True},indent=2,sort_keys=True)+'\n')
PY

wheel=$("${PYTHON}" - "${ARTIFACT_DIR}/c2-native-plugin-overlay-provenance.json" <<'PY'
import json,sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())['derived_wheel']['filename'])
PY
)
sha256sum "${ARTIFACT_DIR}/c2-native-plugin-overlay-provenance.json" "${ARTIFACT_DIR}/c2-native-plugin-overlay-manifest.json" "${ARTIFACT_DIR}/${wheel}" "${ARTIFACT_DIR}/promotion-prerequisites-job${SLURM_JOB_ID}.json" "${ARTIFACT_DIR}/wheel-driver-verification-job${SLURM_JOB_ID}.json" "${ARTIFACT_DIR}/result-job${SLURM_JOB_ID}.json" > "${ARTIFACT_DIR}/outputs-job${SLURM_JOB_ID}.sha256"
