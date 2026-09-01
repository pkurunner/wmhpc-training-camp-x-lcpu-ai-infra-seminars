#!/usr/bin/env bash
# Fresh-process, real-backend integration gate for the accepted v12 Q-row-
# padding overlay wheel.  It deliberately reports native-vs-Triton only as an
# observation: no v11 fresh-wheel comparator is present, so it cannot create
# or replace the v12-v11 direct ABBA promotion decision from job12985.

#SBATCH --job-name=c2-native-plugin-v12-fresh
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v12-fresh-backend-artifacts-20260830/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=
export PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v12-fresh-backend-artifacts-20260830
# A full wheel target consumes more than the remaining per-account home
# quota.  Keep the small, auditable artifacts in the job-scoped home root,
# but materialize the isolated import target on node-local temporary storage.
# Nothing under this temporary directory is evidence; the EXIT trap removes it.
TMP_PARENT=$(readlink -f "${SLURM_TMPDIR:-/tmp}")
[[ -d "${TMP_PARENT}" && -w "${TMP_PARENT}" ]] || { echo "unusable node-local temporary directory: ${TMP_PARENT}" >&2; exit 2; }
RUN_ID=${C2_RUN_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is required unless C2_RUN_ID is supplied}}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe C2 run identifier: ${RUN_ID}" >&2; exit 2; }
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${RUN_ID}; RUNTIME=; SUPPORT=
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python; UV=/usr/local/bin/uv
BASELINE_VLLM=/home/lcpu/85117379/venv-vllm-d4da0c5/lib/python3.12/site-packages/vllm
C2_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/team/c2_msa_decode
HARNESS=/home/lcpu/85117379/native_c2_full_backend_bench_20260829.py; HARNESS_COPY=${ARTIFACT_DIR}/native_c2_full_backend_bench.py
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}"); SUBMITTED_SCRIPT=/home/lcpu/85117379/validate_native_c2_plugin_v12_fresh_backend_20260830.slurm
EXPECTED_BASELINE_WHEEL_SHA=91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06
EXPECTED_STABLE_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_PLUGIN_SHA=064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6
EXPECTED_PACKAGER_SHA=84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e
EXPECTED_HARNESS_SHA=7883edc25df48e3b69a9f4948775a1baafd3ba0b1bd17edcfeac6644aa7b4762
EXPECTED_WHEEL_FILENAME=vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl
SEEDS=(17 23 42 2024 314159 20260801 20260815 20260829)

# The original accepted evidence is immutable and is re-checked independently
# of the wheel package supplied to this run.
AOT_DIR=/home/lcpu/85117379/c2-native-plugin-v12-aot-artifacts-20260830/job12983
DIRECTED_DIR=/home/lcpu/85117379/c2-native-plugin-v12-directed-artifacts-20260830/job12984
STRESS_DIR=/home/lcpu/85117379/c2-native-plugin-v12-stress-3pct-artifacts-20260830/job12985
AOT_DSO=${AOT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so; AOT_SOURCE=${AOT_DIR}/native_c2_decode.v12-q-row-padding.cu
AOT_FINAL=${AOT_DIR}/final-status-job12983.txt; AOT_RESOURCE=${AOT_DIR}/plugin-resource-gate-job12983.txt; AOT_PROVENANCE=${AOT_DIR}/plugin-v12-provenance-job12983.sha256
DIRECTED_RESULT=${DIRECTED_DIR}/v12-q-row-padding-directed-job12984.json; DIRECTED_FINAL=${DIRECTED_DIR}/final-status-job12984.txt; DIRECTED_FINAL_SIDECAR=${DIRECTED_DIR}/final-status-job12984.sha256; DIRECTED_OUTPUTS=${DIRECTED_DIR}/outputs-job12984.sha256; DIRECTED_MANIFEST=${DIRECTED_DIR}/run-manifest-job12984.json
STRESS_DECISION=${STRESS_DIR}/v12-q-row-padding-vs-v11-3pct-incremental-decision-job12985.json; STRESS_AGGREGATE=${STRESS_DIR}/v12-q-row-padding-vs-v11-3pct-incremental-aggregate-job12985.json; STRESS_FINAL=${STRESS_DIR}/final-status-job12985.txt; STRESS_FINAL_SIDECAR=${STRESS_DIR}/final-status-job12985.sha256; STRESS_OUTPUTS=${STRESS_DIR}/outputs-job12985.sha256

: "${C2_WHEEL_ABSPATH:?absolute v12 overlay wheel is required}"; : "${C2_PROVENANCE_ABSPATH:?matching v12 provenance is required}"; : "${C2_WHEEL_BUILD_VERIFICATION_ABSPATH:?matching v12 wheel verification is required}"; : "${C2_WHEEL_PROMOTION_PREREQUISITES_ABSPATH:?matching v12 prerequisites is required}"; : "${C2_WHEEL_OUTPUTS_MANIFEST_ABSPATH:?matching v12 outputs manifest is required}"; : "${C2_WHEEL_FINAL_STATUS_ABSPATH:?matching v12 final status is required}"
: "${C2_EXPECTED_WHEEL_SHA:?reviewed wheel SHA-256 required}"; : "${C2_EXPECTED_PROVENANCE_SHA:?reviewed provenance SHA-256 required}"; : "${C2_EXPECTED_WHEEL_BUILD_VERIFICATION_SHA:?reviewed wheel verification SHA-256 required}"; : "${C2_EXPECTED_WHEEL_PROMOTION_PREREQUISITES_SHA:?reviewed wheel prerequisites SHA-256 required}"; : "${C2_EXPECTED_WHEEL_OUTPUTS_MANIFEST_SHA:?reviewed wheel outputs SHA-256 required}"; : "${C2_EXPECTED_WHEEL_FINAL_STATUS_SHA:?reviewed wheel final-status SHA-256 required}"; : "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?reviewed fresh-script SHA-256 required}"
for digest in "${C2_EXPECTED_WHEEL_SHA}" "${C2_EXPECTED_PROVENANCE_SHA}" "${C2_EXPECTED_WHEEL_BUILD_VERIFICATION_SHA}" "${C2_EXPECTED_WHEEL_PROMOTION_PREREQUISITES_SHA}" "${C2_EXPECTED_WHEEL_OUTPUTS_MANIFEST_SHA}" "${C2_EXPECTED_WHEEL_FINAL_STATUS_SHA}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}"; do [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || exit 2; done
for path in "${C2_WHEEL_ABSPATH}" "${C2_PROVENANCE_ABSPATH}" "${C2_WHEEL_BUILD_VERIFICATION_ABSPATH}" "${C2_WHEEL_PROMOTION_PREREQUISITES_ABSPATH}" "${C2_WHEEL_OUTPUTS_MANIFEST_ABSPATH}" "${C2_WHEEL_FINAL_STATUS_ABSPATH}"; do [[ "${path}" == /* ]]; done
WHEEL=$(readlink -f "${C2_WHEEL_ABSPATH}"); PROVENANCE=$(readlink -f "${C2_PROVENANCE_ABSPATH}"); WHEEL_CHECK=$(readlink -f "${C2_WHEEL_BUILD_VERIFICATION_ABSPATH}"); WHEEL_PREREQ=$(readlink -f "${C2_WHEEL_PROMOTION_PREREQUISITES_ABSPATH}"); WHEEL_OUTPUTS=$(readlink -f "${C2_WHEEL_OUTPUTS_MANIFEST_ABSPATH}"); WHEEL_FINAL=$(readlink -f "${C2_WHEEL_FINAL_STATUS_ABSPATH}")
WHEEL_DIR=$(dirname "${WHEEL}"); case "${WHEEL_DIR}" in /home/lcpu/85117379/c2-native-plugin-v12-wheel-artifacts-20260830/job*) ;; *) echo 'wheel path outside v12 wheel artifact root' >&2; exit 2;; esac
[[ "$(dirname "${PROVENANCE}")" == "${WHEEL_DIR}" && "$(dirname "${WHEEL_CHECK}")" == "${WHEEL_DIR}" && "$(dirname "${WHEEL_PREREQ}")" == "${WHEEL_DIR}" && "$(dirname "${WHEEL_OUTPUTS}")" == "${WHEEL_DIR}" && "$(dirname "${WHEEL_FINAL}")" == "${WHEEL_DIR}" && "$(basename "${WHEEL}")" == "${EXPECTED_WHEEL_FILENAME}" ]]
[[ -f "${SCRIPT_PATH}" && -f "${SUBMITTED_SCRIPT}" ]] || { echo 'missing executed/submitted fresh script' >&2; exit 2; }; cmp -s "${SCRIPT_PATH}" "${SUBMITTED_SCRIPT}" || { echo 'spooled script differs from submitted copy' >&2; exit 2; }
[[ ! -e "${ARTIFACT_DIR}" ]] || { echo "refusing to reuse ${RUN_ID}" >&2; exit 2; }
mkdir -p "${ARTIFACT_ROOT}" "${ARTIFACT_DIR}"; exec > >(tee "${ARTIFACT_DIR}/driver-job${RUN_ID}.log") 2>&1
finalize() {
  local rc=$? cleanup_rc=0
  trap - EXIT
  set +e
  # The guard makes rm target only the mktemp-created child of the selected
  # node-local parent, never the artifact root or a caller-controlled path.
  if [[ -n "${RUNTIME:-}" ]]; then
    if [[ -d "${RUNTIME}" && ! -L "${RUNTIME}" && "$(dirname -- "${RUNTIME}")" == "${TMP_PARENT}" && "$(basename -- "${RUNTIME}")" =~ ^c2-native-plugin-v12-fresh-runtime-${RUN_ID}\.[A-Za-z0-9]+$ ]]; then
      # The support fixture is made read-only while the real backend runs.
      # Restore owner write permission only after every gate/output manifest
      # has been produced, then remove this exact mktemp child.
      chmod -R u+rwX -- "${RUNTIME}" || cleanup_rc=1
      [[ "${cleanup_rc}" -eq 0 ]] && rm -rf -- "${RUNTIME}" || cleanup_rc=1
    else
      cleanup_rc=1
    fi
  fi
  if [[ "${cleanup_rc}" -ne 0 && "${rc}" -eq 0 ]]; then rc=2; fi
  printf 'FINAL_RC=%s UTC=%s RUN_ID=%s RUNTIME_CLEANUP_RC=%s\n' "${rc}" "$(date -u +%FT%TZ)" "${RUN_ID}" "${cleanup_rc}" | tee "${ARTIFACT_DIR}/final-status-job${RUN_ID}.txt"
  exit "${rc}"
}
trap finalize EXIT

for path in "${WHEEL}" "${PROVENANCE}" "${WHEEL_CHECK}" "${WHEEL_PREREQ}" "${WHEEL_OUTPUTS}" "${WHEEL_FINAL}" "${AOT_DSO}" "${AOT_SOURCE}" "${AOT_FINAL}" "${AOT_RESOURCE}" "${AOT_PROVENANCE}" "${DIRECTED_RESULT}" "${DIRECTED_FINAL}" "${DIRECTED_FINAL_SIDECAR}" "${DIRECTED_OUTPUTS}" "${DIRECTED_MANIFEST}" "${STRESS_DECISION}" "${STRESS_AGGREGATE}" "${STRESS_FINAL}" "${STRESS_FINAL_SIDECAR}" "${STRESS_OUTPUTS}" "${HARNESS}"; do [[ -f "${path}" ]]; done
printf '%s  %s\n' "${C2_EXPECTED_WHEEL_SHA}" "${WHEEL}" "${C2_EXPECTED_PROVENANCE_SHA}" "${PROVENANCE}" "${C2_EXPECTED_WHEEL_BUILD_VERIFICATION_SHA}" "${WHEEL_CHECK}" "${C2_EXPECTED_WHEEL_PROMOTION_PREREQUISITES_SHA}" "${WHEEL_PREREQ}" "${C2_EXPECTED_WHEEL_OUTPUTS_MANIFEST_SHA}" "${WHEEL_OUTPUTS}" "${C2_EXPECTED_WHEEL_FINAL_STATUS_SHA}" "${WHEEL_FINAL}" "${EXPECTED_HARNESS_SHA}" "${HARNESS}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SUBMITTED_SCRIPT}" > "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"; sha256sum -c "${WHEEL_OUTPUTS}"; sha256sum -c "${AOT_PROVENANCE}"; sha256sum -c "${DIRECTED_FINAL_SIDECAR}"; sha256sum -c "${DIRECTED_OUTPUTS}"; sha256sum -c "${STRESS_FINAL_SIDECAR}"; sha256sum -c "${STRESS_OUTPUTS}"
grep -qxE 'FINAL_RC=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${WHEEL_FINAL}"; grep -qxE 'BODY_RC=0 TEE_RC=0 MANIFEST_RC=0 FINAL_RC=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${AOT_FINAL}"; grep -qx 'REG=127' "${AOT_RESOURCE}"; grep -qx 'STACK=0' "${AOT_RESOURCE}"; grep -qx 'SHARED=31136' "${AOT_RESOURCE}"; grep -qx 'LOCAL=0' "${AOT_RESOURCE}"; grep -qxE 'FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 POST_CHECKED=1 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${DIRECTED_FINAL}"; grep -qxE 'FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${STRESS_FINAL}"

"${PYTHON}" - "${PROVENANCE}" "${WHEEL_CHECK}" "${WHEEL_PREREQ}" "${WHEEL}" "${DIRECTED_RESULT}" "${STRESS_DECISION}" "${STRESS_AGGREGATE}" "${AOT_PROVENANCE}" <<'PY'
import hashlib,json,sys
from pathlib import Path
prov,wcheck,prereq,wheel,directed,decision,aggregate,aot=map(Path,sys.argv[1:]); H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads(prov.read_text()); c=json.loads(wcheck.read_text()); r=json.loads(prereq.read_text()); d=json.loads(directed.read_text()); q=json.loads(decision.read_text()); a=json.loads(aggregate.read_text())
assert p['schema']=='c2-native-plugin-overlay-provenance-v1' and p['plugin_expected_sha256']=='064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6' and p['derived_wheel']['sha256']==H(wheel)
assert p['baseline_wheel']['sha256']=='91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06' and p['packager']['sha256']=='84f607ec46f4535a5d5e3506631d4eeb9a044f9e8d48f4aa0eec57984ca2866e'
assert c['schema']=='c2-native-plugin-v12-overlay-wheel-driver-verification-v1' and c['all_gates_pass'] is True and c['wheel_sha256']==H(wheel) and c['stable_member_byte_identical'] is True
assert r['schema']=='c2-native-plugin-v12-wheel-promotion-prerequisites-v1' and r['stress_accepted'] is True and r['directed_all_gates_pass'] is True
assert d['all_gates_pass'] is True and d['no_monkeypatch'] is True and d['profiler']['native_dispatcher_hit'] is True
assert q['accepted'] is True and q['issues']==[] and q['aggregate_sha256']==H(aggregate) and q['input_checksums']['candidate_v12_plugin']=='064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6'
assert a['all_correctness_and_comparability_gates_pass'] is True and a['issues']==[] and len(set(a['gpu_uuids']))==1
assert '064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6' in aot.read_text()
PY

RUNTIME=$(mktemp -d -p "${TMP_PARENT}" "c2-native-plugin-v12-fresh-runtime-${RUN_ID}.XXXXXXXX")
[[ -d "${RUNTIME}" && ! -L "${RUNTIME}" && "$(dirname -- "${RUNTIME}")" == "${TMP_PARENT}" ]] || { echo 'failed to create isolated node-local runtime' >&2; exit 2; }
SUPPORT=${RUNTIME}/c2_support
printf 'runtime_parent=%s\nruntime_is_node_local_tmp=true\nruntime_is_ephemeral=true\n' "${TMP_PARENT}" > "${ARTIFACT_DIR}/runtime-location-job${RUN_ID}.txt"
"${UV}" pip install --python "${PYTHON}" --target "${RUNTIME}" --no-deps "${WHEEL}"
[[ ! -L "${RUNTIME}" && -z "$(find "${RUNTIME}" -type l -print -quit)" ]]; install -D -m 0644 "${HARNESS}" "${HARNESS_COPY}"
INSTALLED_MEMBERS=${ARTIFACT_DIR}/installed-members-job${RUN_ID}.json
"${PYTHON}" - "${RUNTIME}" "${WHEEL}" "${PROVENANCE}" "${INSTALLED_MEMBERS}" <<'PY'
import base64,csv,hashlib,io,json,sys,zipfile
from pathlib import Path
runtime,wheel,prov,out=map(Path,sys.argv[1:]); H=lambda b:hashlib.sha256(b).hexdigest(); p=json.loads(prov.read_text())
plugin='vllm/_native_c2_msa_decode_plugin.abi3.so'; stable='vllm/_C_stable_libtorch.abi3.so'
with zipfile.ZipFile(wheel) as z:
 names=[n for n in z.namelist() if not n.endswith('/')]; rec=[n for n in names if n.endswith('.dist-info/RECORD')]; assert len(rec)==1; rec=rec[0]; rows=list(csv.reader(io.StringIO(z.read(rec).decode()))); assert len(rows)==len({r[0] for r in rows}) and {r[0] for r in rows}==set(names)
 for n,e,s in rows:
  if n==rec: assert e==s==''; continue
  enc=base64.urlsafe_b64encode(hashlib.sha256(z.read(n)).digest()).decode().rstrip('='); assert e==f'sha256={enc}' and int(s)==len(z.read(n)); assert (runtime/n).is_file() and (runtime/n).read_bytes()==z.read(n)
 assert H(z.read(plugin))==p['plugin_expected_sha256']=='064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6'; assert H(z.read(stable))=='cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
 installed=list(csv.reader((runtime/rec).read_text().splitlines())); added={'bin/vllm',f'{rec.rsplit("/",1)[0]}/INSTALLER',f'{rec.rsplit("/",1)[0]}/REQUESTED',f'{rec.rsplit("/",1)[0]}/direct_url.json',f'{rec.rsplit("/",1)[0]}/uv_cache.json'}; assert {r[0] for r in installed}==set(names)|added
 for n,e,s in installed:
  f=(runtime/n).resolve(); assert f.is_relative_to(runtime.resolve()) and f.is_file()
  if n==rec: assert e==s==''; continue
  enc=base64.urlsafe_b64encode(hashlib.sha256(f.read_bytes()).digest()).decode().rstrip('='); assert e==f'sha256={enc}' and int(s)==f.stat().st_size
 actual={str(f.relative_to(runtime)) for f in runtime.rglob('*') if f.is_file()}; assert actual=={r[0] for r in installed}|{'.lock'}
out.write_text(json.dumps({'schema':'c2-native-plugin-v12-installed-members-v1','wheel_record_full_verification':True,'installed_record_full_verification':True,'wheel_payload_except_installer_rewritten_record_byte_identical':True,'plugin_sha256':p['plugin_expected_sha256'],'stable_member_sha256':'cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442','all_gates_pass':True},indent=2,sort_keys=True)+'\n')
PY

SUPPORT_SOURCE_MANIFEST=${ARTIFACT_DIR}/support-source-job${RUN_ID}.sha256; SUPPORT_INSTALLED_MANIFEST=${ARTIFACT_DIR}/support-installed-job${RUN_ID}.sha256
printf '%s  %s\n' 'df9dd0b01c3e91afeb1a490785fab1a0554a16f001e118005192f55f891358a8' 'harness/__init__.py' '38faefb89b38e3bc26b135b762a6bbbd813dcc695423982b423a806fe41c11d6' 'harness/data.py' '712c1632367700137392d2c43bd5c6962130d945f01f1f585cc6338788ce53cc' 'harness/reference.py' '6b0a40463b4f1c05ca53649d6689377b4f755d1efb58b10700dad05705b53f71' 'harness/triton_baseline.py' '5db5d5e692044e54bb57418cbb7dcf5c9cc94f789eaccf128eec6fa8df86d967' 'vllm_msa_ref/sparse_attn.py' > "${SUPPORT_SOURCE_MANIFEST}"
(cd "${C2_ROOT}" && sha256sum -c "${SUPPORT_SOURCE_MANIFEST}"); mkdir -p "${SUPPORT}"; while read -r digest member; do source=${C2_ROOT}/${member}; [[ -f "${source}" && ! -L "${source}" ]]; install -D -m 0444 "${source}" "${SUPPORT}/${member}"; done < "${SUPPORT_SOURCE_MANIFEST}"; cp "${SUPPORT_SOURCE_MANIFEST}" "${SUPPORT_INSTALLED_MANIFEST}"; (cd "${SUPPORT}" && sha256sum -c "${SUPPORT_INSTALLED_MANIFEST}"); [[ -z "$(find "${SUPPORT}" -type l -print -quit)" && ! -e "${SUPPORT}/vllm" ]]; chmod -R a-w "${SUPPORT}"

BASELINE_OPS=${ARTIFACT_DIR}/baseline-dispatch-surface-job${RUN_ID}.json; INSTALLED_PRE=${ARTIFACT_DIR}/installed-preloader-surface-job${RUN_ID}.json; INSTALLED_POST=${ARTIFACT_DIR}/installed-postloader-surface-job${RUN_ID}.json
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "${PYTHON}" - "${BASELINE_VLLM}" "${BASELINE_OPS}" <<'PY'
import hashlib,json,sys
from pathlib import Path
import torch,vllm; import vllm._C_stable_libtorch as stable
root,out=map(Path,sys.argv[1:]); assert Path(vllm.__file__).resolve().parent==root.resolve(); assert hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest()=='cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
ops=sorted(x for x in torch._C._dispatch_get_all_op_names() if x.startswith('_C::')); assert '_C::native_c2_msa_decode' not in ops; out.write_text(json.dumps({'schema':'c2-native-plugin-v12-baseline-dispatch-surface-v1','ops':ops,'stable_sha256':'cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'},indent=2,sort_keys=True)+'\n')
PY
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}" "${PYTHON}" - "${RUNTIME}" "${WHEEL}" "${BASELINE_OPS}" "${INSTALLED_PRE}" "${INSTALLED_POST}" <<'PY'
import hashlib,importlib,importlib.metadata,json,sys,zipfile
from pathlib import Path
import torch,vllm; import vllm._C_stable_libtorch as stable
runtime,wheel,baseline,pre,post=map(Path,sys.argv[1:]); root=Path(vllm.__file__).resolve().parent; assert root==runtime.resolve()/'vllm' and importlib.metadata.distribution('vllm').locate_file('vllm').resolve()==root; assert hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest()=='cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
b=json.loads(baseline.read_text()); mod=importlib.import_module('vllm.models.minimax_m3.nvidia.msa_native_c2_decode'); assert Path(mod.__file__).resolve()==root/'models/minimax_m3/nvidia/msa_native_c2_decode.py'; before=sorted(x for x in torch._C._dispatch_get_all_op_names() if x.startswith('_C::')); assert before==b['ops'] and '_C::native_c2_msa_decode' not in before; pre.write_text(json.dumps({'schema':'c2-native-plugin-v12-installed-preloader-surface-v1','matches_immutable_baseline':True,'ops':before},indent=2,sort_keys=True)+'\n')
loader=getattr(mod,'_load_native_c2_plugin_once'); assert callable(loader) and loader() is True; after=sorted(x for x in torch._C._dispatch_get_all_op_names() if x.startswith('_C::')); assert sorted(set(after)-set(before))==['_C::native_c2_msa_decode'] and torch._C._dispatch_has_kernel_for_dispatch_key('_C::native_c2_msa_decode','CUDA'); assert loader() is True and sorted(x for x in torch._C._dispatch_get_all_op_names() if x.startswith('_C::'))==after; plugin=root/'_native_c2_msa_decode_plugin.abi3.so'; assert hashlib.sha256(plugin.read_bytes()).hexdigest()=='064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6'; post.write_text(json.dumps({'schema':'c2-native-plugin-v12-installed-postloader-surface-v1','plugin_sha256':'064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6','added_ops':['_C::native_c2_msa_decode'],'native_c2_has_cuda_kernel':True,'all_baseline_op_names_preserved':set(before).issubset(after)},indent=2,sort_keys=True)+'\n')
PY

selector() { local value=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}; [[ "${value}" != *,* && ( "${value}" =~ ^[0-9]+$ || "${value}" =~ ^GPU-[0-9A-Fa-f-]+$ ) ]] || return 1; printf '%s\n' "${value}"; }
gpu_uuid() { nvidia-smi --id "$(selector)" --query-gpu=uuid --format=csv,noheader,nounits | awk 'NR==1 {print $1} END {if (NR != 1) exit 1}'; }
compute_apps() { nvidia-smi --id "$(selector)" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
snapshot() { local tag=$1 gpu=$2 apps=$3; printf '===== %s UTC=%s =====\n' "${tag}" "$(date -u +%FT%TZ)" > "${gpu}"; nvidia-smi --id "$(selector)" --query-gpu=uuid,name,driver_version,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader >> "${gpu}"; compute_apps > "${apps}"; }
GPU_PRE=${ARTIFACT_DIR}/gpu-pre-job${RUN_ID}.txt; GPU_POST=${ARTIFACT_DIR}/gpu-post-job${RUN_ID}.txt; GPU_PRE_APPS=${ARTIFACT_DIR}/gpu-pre-apps-job${RUN_ID}.txt; GPU_POST_APPS=${ARTIFACT_DIR}/gpu-post-apps-job${RUN_ID}.txt; EXPECTED_GPU_UUID=$(gpu_uuid); snapshot PRE "${GPU_PRE}" "${GPU_PRE_APPS}"; [[ -z "$(<"${GPU_PRE_APPS}")" ]] || { echo 'GPU is not clean before fresh validation' >&2; exit 1; }

UUID_MANIFEST=${ARTIFACT_DIR}/seed-gpu-uuid-job${RUN_ID}.csv; printf 'seed,before_uuid,after_uuid,result,trace\n' > "${UUID_MANIFEST}"
for seed in "${SEEDS[@]}"; do
  result=${ARTIFACT_DIR}/plugin-v12-wheel-full-backend-seed${seed}-job${RUN_ID}.json; trace=${ARTIFACT_DIR}/plugin-v12-wheel-full-backend-seed${seed}-job${RUN_ID}.chrome.json; before=$(gpu_uuid); [[ "${before}" == "${EXPECTED_GPU_UUID}" ]]; PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}:${SUPPORT}" timeout --foreground --preserve-status --kill-after=10s 360s "${PYTHON}" "${HARNESS_COPY}" --c2-root "${SUPPORT}" --vllm-root "${RUNTIME}" --output "${result}" --trace "${trace}" --seed "${seed}" --timing-warmup 10 --timing-repetitions 50; after=$(gpu_uuid); [[ "${after}" == "${EXPECTED_GPU_UUID}" ]]; printf '%s,%s,%s,%s,%s\n' "${seed}" "${before}" "${after}" "${result}" "${trace}" >> "${UUID_MANIFEST}"
done

SUMMARY=${ARTIFACT_DIR}/plugin-v12-fresh-backend-summary-job${RUN_ID}.json; DECISION=${ARTIFACT_DIR}/plugin-v12-fresh-backend-decision-job${RUN_ID}.json
"${PYTHON}" - "${SUMMARY}" "${DECISION}" "${UUID_MANIFEST}" "${ARTIFACT_DIR}" "${RUNTIME}" "${EXPECTED_GPU_UUID}" "${SEEDS[@]}" <<'PY'
import csv,hashlib,json,math,statistics,sys
from pathlib import Path
summary,decision,uuids,art,runtime,uuid,*ss=sys.argv[1:]; summary,decision,uuids,art,runtime=map(Path,(summary,decision,uuids,art,runtime)); seeds=list(map(int,ss)); rows=list(csv.DictReader(uuids.open())); assert [int(x['seed']) for x in rows]==seeds and all(x['before_uuid']==x['after_uuid']==uuid for x in rows)
native=[]; triton=[]; ratios=[]; results=[]
for seed in seeds:
 p=art/f'plugin-v12-wheel-full-backend-seed{seed}-job{art.name.removeprefix("job")}.json'; trace=art/f'plugin-v12-wheel-full-backend-seed{seed}-job{art.name.removeprefix("job")}.chrome.json'; x=json.loads(p.read_text()); assert x['schema']=='c2-native-c2-full-vllm-backend-v2' and x['seed']==seed and x['all_gates_pass'] is True and x['correctness']['pass'] is True and x['triton_correctness']['pass'] is True and x['caller_output']['pointer_unchanged'] is True and x['profiling']['checks']['one_cpu_dispatcher_event'] is True and x['profiling']['checks']['one_cuda_native_kernel_event'] is True and trace.is_file() and trace.stat().st_size>0
 t=x['timing']; assert t['pass'] is True and 'ABBA=native,triton,triton,native' in t['protocol'] and t['warmup']==10 and t['repetitions']==50 and t['native']['sample_count']==t['triton']['sample_count']==100; n=float(t['native']['median_ms']); tr=float(t['triton']['median_ms']); assert math.isfinite(n) and math.isfinite(tr) and n>0 and tr>0 and Path(x['environment']['vllm_import_root']).resolve()==runtime/'vllm'; native.append(n); triton.append(tr); ratios.append(n/tr); results.append({'seed':seed,'result':str(p),'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'trace':str(trace),'native_median_ms':n,'triton_median_ms':tr,'native_over_triton_latency_ratio':n/tr})
payload={'schema':'c2-native-plugin-v12-fresh-backend-summary-v1','all_integration_and_correctness_gates_pass':True,'seeds':results,'seed_count':8,'one_physical_gpu_uuid':uuid,'fresh_process_per_seed':True,'native_median_of_seed_medians_ms':statistics.median(native),'triton_median_of_seed_medians_ms':statistics.median(triton),'native_over_triton_ratio_of_seed_medians':statistics.median(ratios),'triton_parity_achieved':max(ratios)<=1.0,'triton_parity_is_hard_gate':False,'v12_dso_sha256':'064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6','v12_vs_v11_fresh_wheel_comparator':{'available':False,'comparison_kind':'none','reason':'no v11 fresh-wheel comparator was run; accepted v12-v11 promotion remains the separate direct job12985 same-GPU ABBA result','fresh_native_over_triton_is_not_v12_vs_v11_promotion':True}}
summary.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); decision.write_text(json.dumps({'schema':'c2-native-plugin-v12-fresh-backend-decision-v1','accepted_for_fresh_integration':True,'final_rc':0,'hard_gates':{'wheel_identity':True,'record_and_install':True,'stable_dso_preserved':True,'loader_surface':True,'eight_seed_real_backend':True,'direct_v12_vs_v11_bitwise_and_promotion_evidence':True,'clean_b300_before_after':True},'performance':{'reported':True,'triton_parity_required':False,'triton_parity_achieved':payload['triton_parity_achieved'],'v12_vs_v11_fresh_wheel_comparator':'not available; not a promotion decision'},'v12_vs_v11_fresh_wheel_boundary':payload['v12_vs_v11_fresh_wheel_comparator']},indent=2,sort_keys=True)+'\n')
PY
snapshot POST "${GPU_POST}" "${GPU_POST_APPS}"; [[ "$(gpu_uuid)" == "${EXPECTED_GPU_UUID}" && -z "$(<"${GPU_POST_APPS}")" ]] || { echo 'GPU is not clean after fresh validation' >&2; exit 1; }
sha256sum "${SUMMARY}" "${DECISION}" "${INSTALLED_MEMBERS}" "${BASELINE_OPS}" "${INSTALLED_PRE}" "${INSTALLED_POST}" "${UUID_MANIFEST}" "${GPU_PRE}" "${GPU_POST}" "${GPU_PRE_APPS}" "${GPU_POST_APPS}" "${SUPPORT_SOURCE_MANIFEST}" "${SUPPORT_INSTALLED_MANIFEST}" "${HARNESS_COPY}" "${ARTIFACT_DIR}/runtime-location-job${RUN_ID}.txt" "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256" ${ARTIFACT_DIR}/plugin-v12-wheel-full-backend-seed*-job${RUN_ID}.json ${ARTIFACT_DIR}/plugin-v12-wheel-full-backend-seed*-job${RUN_ID}.chrome.json > "${ARTIFACT_DIR}/outputs-job${RUN_ID}.sha256"
