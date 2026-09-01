#!/usr/bin/env bash
# Independent fresh-process / real-MiniMax-backend integration and parity
# observation for the accepted v16 K-chunk-lookahead overlay wheel.  This is
# deliberately not a v16-v12 promotion experiment: it measures only the
# installed v16 wheel's native-versus-Triton ratio in fresh processes.

#SBATCH --job-name=c2-v16-fresh-backend
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v16-k-chunk-lookahead-fresh-backend-artifacts-20260831/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=
export PATH=/usr/local/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v16-k-chunk-lookahead-fresh-backend-artifacts-20260831
RUN_ID=${C2_RUN_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID is required unless C2_RUN_ID is supplied}}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo 'unsafe C2 run identifier' >&2; exit 2; }
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${RUN_ID}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
UV=/usr/local/bin/uv
BASELINE_VLLM=/home/lcpu/85117379/venv-vllm-d4da0c5/lib/python3.12/site-packages/vllm
C2_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/team/c2_msa_decode
HARNESS=/home/lcpu/85117379/native_c2_full_backend_bench_20260829.py
SUBMITTED_SCRIPT=/home/lcpu/85117379/validate_native_c2_plugin_v16_k_chunk_lookahead_fresh_backend_8seeds_20260831.slurm
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
EXPECTED_HARNESS_SHA=7883edc25df48e3b69a9f4948775a1baafd3ba0b1bd17edcfeac6644aa7b4762
EXPECTED_PLUGIN_SHA=1da5f731da796656759f0e673e3479392b6b8337c054a61aa4ca1fd0afb4edd4
EXPECTED_STABLE_SHA=cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442
EXPECTED_WHEEL_SHA=3947fab41739c98a30a8fd5486b867347b932f3419def3bfbd846db458ba90a9
EXPECTED_WHEEL_PROVENANCE_SHA=6228ea8d84d2e15408ed365a3ae341809913e02c3ed1c731b5e186abaae49467
EXPECTED_WHEEL_VERIFICATION_SHA=f3d74b3932307b3b754f2e06da4265ffc8825c014d905457c765cd5436cd77c9
EXPECTED_WHEEL_PREREQUISITES_SHA=279906ea6e4ef2375f92c65d8588c16668f3738379da6e0549671e4b4bbde573
EXPECTED_WHEEL_OUTPUTS_SHA=af8cdc0ec460a3b04e91dba0fcb3e1b325edc9ea7d7ff6257d245b9593de3d02
EXPECTED_WHEEL_FINAL_SHA=77ee692757cb613eefe773b9927588789dfddf94c57f7181e122bffaa3c4363b
EXPECTED_AOT_PROVENANCE_SHA=3c640900ab67f25f53eee07f5e94106bb3f849d258b6d3863aab6f8c1b5068eb
EXPECTED_DIRECTED_RESULT_SHA=cff227a42cd243bc49a33eaa6a7fe1ee2d332a19eedcb64f2a77c27f2f1e9944
EXPECTED_STRESS_DECISION_SHA=3763ce31dbda5482896aa00c0192af1aca0ba8e322a4e555b52a6c7ad491bed9
EXPECTED_STRESS_AGGREGATE_SHA=95c456509add154bf11747a78b3b5203cc6580ad08bccf31cc562b3815172984
EXPECTED_LIFECYCLE_RESULT_SHA=b3c45988ad20daf352f598f74a8be287f20e417b12b1e8e6b18aee62ad991ade
EXPECTED_LIFECYCLE_ATTESTATION_SHA=f8860613c63950ef741c5ccb2e0f6ad413d798fb2ba3d65f2f92f685ed3020c9
EXPECTED_LIFECYCLE_OUTPUTS_SHA=5cc19ed7732bc1eeb16f5fc62dae0ca8500717e71b3693b7ffee16cf7a6588b9
EXPECTED_LIFECYCLE_FINAL_SHA=ed69b4477aa54d20c595515c190c78b68aad85c6efd413f9db02da297ac6e884
EXPECTED_LIFECYCLE_FINAL_SIDECAR_SHA=3f72104cbc6d544536d5403fb3d801e3e61e3e1938db633fde241381dbd1077d
EXPECTED_WHEEL_FILENAME=vllm-0.26.1rc1.dev370+gd4da0c55a-cp38-abi3-manylinux_2_28_x86_64.whl
SEEDS=(17 23 42 2024 314159 20260801 20260815 20260829)

# The two accepted direct gates are immutable prerequisites.  They establish
# v16 identity and v16-v12 promotion separately from this fresh-wheel test.
AOT_DIR=/home/lcpu/85117379/c2-native-plugin-v16-k-chunk-lookahead-aot-artifacts-20260831/job13773
DIRECTED_DIR=/home/lcpu/85117379/c2-native-plugin-v16-k-chunk-lookahead-directed-artifacts-20260831/job13786
STRESS_DIR=/home/lcpu/85117379/c2-native-plugin-v16-k-chunk-lookahead-stress-3pct-artifacts-20260831/job13789
LIFECYCLE_DIR=/home/lcpu/85117379/c2-native-plugin-v16-k-chunk-lookahead-lifecycle-artifacts-20260831/job13845
AOT_PLUGIN=${AOT_DIR}/vllm/_native_c2_msa_decode_plugin.abi3.so
AOT_PROVENANCE=${AOT_DIR}/plugin-v16-provenance-job13773.sha256
DIRECTED_RESULT=${DIRECTED_DIR}/v16-k-chunk-lookahead-directed-job13786.json
STRESS_DECISION=${STRESS_DIR}/decision-job13789.json
STRESS_AGGREGATE=${STRESS_DIR}/aggregate-job13789.json
LIFECYCLE_RESULT=${LIFECYCLE_DIR}/native-c2-v16-k-chunk-lookahead-lifecycle-job13845.json
LIFECYCLE_ATTESTATION=${LIFECYCLE_DIR}/accepted-evidence-attestation-job13845.json
LIFECYCLE_OUTPUTS=${LIFECYCLE_DIR}/outputs-job13845.sha256
LIFECYCLE_FINAL=${LIFECYCLE_DIR}/final-status-job13845.txt
LIFECYCLE_FINAL_SIDECAR=${LIFECYCLE_DIR}/final-status-job13845.sha256

: "${C2_V16_WHEEL_DIR:?canonical v16 overlay-wheel job directory required}"
: "${C2_V16_WHEEL:?canonical v16 overlay wheel required}"
: "${C2_V16_WHEEL_PROVENANCE:?canonical v16 wheel provenance required}"
: "${C2_V16_WHEEL_VERIFICATION:?canonical v16 wheel verification required}"
: "${C2_V16_WHEEL_PREREQUISITES:?canonical v16 promotion prerequisites required}"
: "${C2_V16_WHEEL_OUTPUTS:?canonical v16 wheel outputs required}"
: "${C2_V16_WHEEL_FINAL:?canonical v16 wheel final-status required}"
: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?reviewed exact fresh-script SHA-256 required}"
WHEEL_DIR=${C2_V16_WHEEL_DIR}; WHEEL=${C2_V16_WHEEL}; WHEEL_PROVENANCE=${C2_V16_WHEEL_PROVENANCE}
WHEEL_VERIFICATION=${C2_V16_WHEEL_VERIFICATION}; WHEEL_PREREQUISITES=${C2_V16_WHEEL_PREREQUISITES}
WHEEL_OUTPUTS=${C2_V16_WHEEL_OUTPUTS}; WHEEL_FINAL=${C2_V16_WHEEL_FINAL}
[[ "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" =~ ^[0-9a-f]{64}$ ]] || { echo 'invalid fresh script SHA-256' >&2; exit 2; }
[[ "${WHEEL_DIR}" =~ ^/home/lcpu/85117379/c2-native-plugin-v16-k-chunk-lookahead-overlay-wheel-artifacts-20260831/job[0-9]+$ ]] || { echo 'wheel directory is not canonical v16 overlay output' >&2; exit 2; }
WHEEL_JOB_ID=${WHEEL_DIR##*/job}
[[ "${WHEEL}" == "${WHEEL_DIR}/${EXPECTED_WHEEL_FILENAME}" &&
   "${WHEEL_PROVENANCE}" == "${WHEEL_DIR}/c2-native-plugin-overlay-provenance.json" &&
   "${WHEEL_VERIFICATION}" == "${WHEEL_DIR}/wheel-driver-verification-job${WHEEL_JOB_ID}.json" &&
   "${WHEEL_PREREQUISITES}" == "${WHEEL_DIR}/promotion-prerequisites-job${WHEEL_JOB_ID}.json" &&
   "${WHEEL_OUTPUTS}" == "${WHEEL_DIR}/outputs-job${WHEEL_JOB_ID}.sha256" &&
   "${WHEEL_FINAL}" == "${WHEEL_DIR}/final-status-job${WHEEL_JOB_ID}.txt" ]] || { echo 'wheel evidence names do not match canonical job directory' >&2; exit 2; }
# Slurm opens its #SBATCH output before the script body, so this root must be
# pre-created and may contain only that exact regular Slurm log at entry.
[[ -d "${ARTIFACT_ROOT}" && ! -L "${ARTIFACT_ROOT}" && "$(readlink -f -- "${ARTIFACT_ROOT}")" == "${ARTIFACT_ROOT}" && "$(stat -c %u -- "${ARTIFACT_ROOT}")" == "$(id -u)" ]] || { echo 'fresh artifact root must be a canonical owner-controlled directory' >&2; exit 2; }
while IFS= read -r -d '' entry; do
  [[ "${entry}" == "${ARTIFACT_ROOT}/slurm-${SLURM_JOB_ID:?SLURM_JOB_ID is required}.log" && -f "${entry}" && ! -L "${entry}" ]] || { echo "unexpected pre-existing fresh-root entry: ${entry}" >&2; exit 2; }
done < <(find "${ARTIFACT_ROOT}" -mindepth 1 -maxdepth 1 -print0)
[[ ! -e "${ARTIFACT_DIR}" && -f "${SCRIPT_PATH}" && -f "${SUBMITTED_SCRIPT}" ]] || { echo 'reused run id or submitted script missing' >&2; exit 2; }
cmp -s "${SCRIPT_PATH}" "${SUBMITTED_SCRIPT}" || { echo 'spooled script differs from submitted source' >&2; exit 2; }
mkdir -p "${ARTIFACT_ROOT}" "${ARTIFACT_DIR}"

TMP_PARENT=$(readlink -f "${SLURM_TMPDIR:-/tmp}")
[[ -d "${TMP_PARENT}" && -w "${TMP_PARENT}" && ! -L "${TMP_PARENT}" ]] || { echo 'unusable node-local scratch parent' >&2; exit 2; }
RUNTIME= SUPPORT= EXPECTED_GPU_UUID= DRIVER_TEE_PID=
DRIVER_LOG=${ARTIFACT_DIR}/driver-job${RUN_ID}.log
FINAL_STATUS=${ARTIFACT_DIR}/final-status-job${RUN_ID}.txt
FINAL_SIDECAR=${ARTIFACT_DIR}/final-status-job${RUN_ID}.sha256
OUTPUTS_MANIFEST=${ARTIFACT_DIR}/outputs-job${RUN_ID}.sha256
exec 3>&1 4>&2
exec > >(tee "${DRIVER_LOG}") 2>&1
DRIVER_TEE_PID=$!

selector() { local value=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}; [[ "${value}" != *,* && ( "${value}" =~ ^[0-9]+$ || "${value}" =~ ^GPU-[0-9A-Fa-f-]+$ ) ]] || return 1; printf '%s\n' "${value}"; }
gpu_uuid() { nvidia-smi --id "$(selector)" --query-gpu=uuid --format=csv,noheader,nounits | awk 'NR==1 {print $1} END {if (NR != 1) exit 1}'; }
compute_apps() { nvidia-smi --id "$(selector)" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
snapshot() { local tag=$1 out=$2 apps=$3; printf '===== %s UTC=%s =====\n' "${tag}" "$(date -u +%FT%TZ)" > "${out}"; nvidia-smi --id "$(selector)" --query-gpu=uuid,name,driver_version,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader >> "${out}"; compute_apps > "${apps}"; }
cleanup_runtime() {
  [[ -z "${RUNTIME}" ]] && return 0
  [[ -d "${RUNTIME}" && ! -L "${RUNTIME}" && "$(dirname -- "${RUNTIME}")" == "${TMP_PARENT}" && "$(basename -- "${RUNTIME}")" =~ ^c2-v16-fresh-runtime-${RUN_ID}\.[A-Za-z0-9]+$ ]] || return 1
  chmod -R u+rwX -- "${RUNTIME}" && rm -rf -- "${RUNTIME}" && [[ ! -e "${RUNTIME}" && ! -L "${RUNTIME}" ]]
}
finalize() {
  local original_rc=$? cleanup_rc=0 tee_rc=0 manifest_rc=0 finalizer_error=0 final_rc post_uuid='' post_apps=''
  local -a files=()
  local post_gpu=${ARTIFACT_DIR}/gpu-final-post-job${RUN_ID}.txt post_apps_file=${ARTIFACT_DIR}/gpu-final-post-apps-job${RUN_ID}.txt list=${ARTIFACT_DIR}/outputs-list-job${RUN_ID}.nul
  trap - EXIT
  set +e
  if [[ -n "${EXPECTED_GPU_UUID}" ]]; then
    snapshot FINAL_POST "${post_gpu}" "${post_apps_file}" || finalizer_error=1
    post_uuid=$(gpu_uuid) || finalizer_error=1
    [[ "${post_uuid}" == "${EXPECTED_GPU_UUID}" ]] || finalizer_error=1
    post_apps=$(<"${post_apps_file}") || finalizer_error=1
    [[ -z "${post_apps}" ]] || finalizer_error=1
  else
    finalizer_error=1
  fi
  cleanup_runtime || cleanup_rc=$?
  [[ "${cleanup_rc}" -eq 0 ]] || finalizer_error=1
  exec 1>&3 2>&4
  exec 3>&- 4>&-
  wait "${DRIVER_TEE_PID}" || tee_rc=$?
  [[ "${tee_rc}" -eq 0 ]] || finalizer_error=1
  if find "${ARTIFACT_DIR}" -type f ! -name "$(basename "${DRIVER_LOG}")" ! -name "$(basename "${OUTPUTS_MANIFEST}")" ! -name "$(basename "${FINAL_STATUS}")" ! -name "$(basename "${FINAL_SIDECAR}")" ! -name "$(basename "${list}")" -print0 | sort -z > "${list}"; then
    mapfile -d '' -t files < "${list}" || manifest_rc=1
    rm -f -- "${list}" || manifest_rc=1
    [[ "${#files[@]}" -gt 0 ]] && sha256sum "${files[@]}" > "${OUTPUTS_MANIFEST}" && sha256sum -c "${OUTPUTS_MANIFEST}" >/dev/null || manifest_rc=1
  else
    manifest_rc=1
  fi
  [[ "${manifest_rc}" -eq 0 ]] || finalizer_error=1
  final_rc=${original_rc}; [[ "${final_rc}" -eq 0 && "${finalizer_error}" -eq 0 ]] || final_rc=2
  printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s TEE_RC=%s MANIFEST_RC=%s CLEANUP_RC=%s POST_UUID=%s UTC=%s\n' "${final_rc}" "${original_rc}" "${finalizer_error}" "${tee_rc}" "${manifest_rc}" "${cleanup_rc}" "${post_uuid:-unset}" "$(date -u +%FT%TZ)" > "${FINAL_STATUS}"
  sha256sum "${FINAL_STATUS}" > "${FINAL_SIDECAR}" || final_rc=2
  exit "${final_rc}"
}
trap finalize EXIT

for path in "${WHEEL}" "${WHEEL_PROVENANCE}" "${WHEEL_VERIFICATION}" "${WHEEL_PREREQUISITES}" "${WHEEL_OUTPUTS}" "${WHEEL_FINAL}" "${AOT_PLUGIN}" "${AOT_PROVENANCE}" "${DIRECTED_RESULT}" "${STRESS_DECISION}" "${STRESS_AGGREGATE}" "${LIFECYCLE_RESULT}" "${LIFECYCLE_ATTESTATION}" "${LIFECYCLE_OUTPUTS}" "${LIFECYCLE_FINAL}" "${LIFECYCLE_FINAL_SIDECAR}" "${HARNESS}"; do [[ -f "${path}" && ! -L "${path}" ]] || { echo "missing or linked input: ${path}" >&2; exit 2; }; done
for item in "${WHEEL_DIR}" "${AOT_DIR}" "${DIRECTED_DIR}" "${STRESS_DIR}" "${LIFECYCLE_DIR}"; do [[ -d "${item}" && ! -L "${item}" && "$(readlink -f -- "${item}")" == "${item}" ]] || { echo "noncanonical evidence directory: ${item}" >&2; exit 2; }; done
[[ -x "${PYTHON}" && -x "${UV}" && -d "${BASELINE_VLLM}" && -d "${C2_ROOT}/harness" ]] || { echo 'required Python, installer, baseline, or support root is unavailable' >&2; exit 2; }
printf '%s  %s\n' \
  "${EXPECTED_WHEEL_SHA}" "${WHEEL}" \
  "${EXPECTED_WHEEL_PROVENANCE_SHA}" "${WHEEL_PROVENANCE}" \
  "${EXPECTED_WHEEL_VERIFICATION_SHA}" "${WHEEL_VERIFICATION}" \
  "${EXPECTED_WHEEL_PREREQUISITES_SHA}" "${WHEEL_PREREQUISITES}" \
  "${EXPECTED_WHEEL_OUTPUTS_SHA}" "${WHEEL_OUTPUTS}" \
  "${EXPECTED_WHEEL_FINAL_SHA}" "${WHEEL_FINAL}" \
  "${EXPECTED_PLUGIN_SHA}" "${AOT_PLUGIN}" \
  "${EXPECTED_AOT_PROVENANCE_SHA}" "${AOT_PROVENANCE}" \
  "${EXPECTED_DIRECTED_RESULT_SHA}" "${DIRECTED_RESULT}" \
  "${EXPECTED_STRESS_DECISION_SHA}" "${STRESS_DECISION}" \
  "${EXPECTED_STRESS_AGGREGATE_SHA}" "${STRESS_AGGREGATE}" \
  "${EXPECTED_LIFECYCLE_RESULT_SHA}" "${LIFECYCLE_RESULT}" \
  "${EXPECTED_LIFECYCLE_ATTESTATION_SHA}" "${LIFECYCLE_ATTESTATION}" \
  "${EXPECTED_LIFECYCLE_OUTPUTS_SHA}" "${LIFECYCLE_OUTPUTS}" \
  "${EXPECTED_LIFECYCLE_FINAL_SHA}" "${LIFECYCLE_FINAL}" \
  "${EXPECTED_LIFECYCLE_FINAL_SIDECAR_SHA}" "${LIFECYCLE_FINAL_SIDECAR}" \
  "${EXPECTED_HARNESS_SHA}" "${HARNESS}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SUBMITTED_SCRIPT}" > "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${RUN_ID}.sha256"
sha256sum -c "${WHEEL_OUTPUTS}"; sha256sum -c "${AOT_PROVENANCE}"; sha256sum -c "${LIFECYCLE_OUTPUTS}"; sha256sum -c "${LIFECYCLE_FINAL_SIDECAR}"
grep -qxE 'FINAL_RC=0 CLEANUP_RC=0 UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${WHEEL_FINAL}"
grep -qxE 'FINAL_RC=0 ORIGINAL_RC=0 FINALIZER_ERROR=0 TEE_RC=0 MANIFEST_RC=0 CLEANUP_RC=0 POST_APPS_EMPTY=true EXPECTED_UUID=GPU-[0-9a-f-]+ POST_UUID=GPU-[0-9a-f-]+ UTC=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]+Z' "${LIFECYCLE_FINAL}"

"${PYTHON}" - "${WHEEL}" "${WHEEL_PROVENANCE}" "${WHEEL_VERIFICATION}" "${WHEEL_PREREQUISITES}" "${DIRECTED_RESULT}" "${STRESS_DECISION}" "${STRESS_AGGREGATE}" "${LIFECYCLE_RESULT}" "${LIFECYCLE_ATTESTATION}" "${AOT_PROVENANCE}" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
wheel, prov, verify, prereq, directed, decision, aggregate, lifecycle, attest, aot = map(Path, sys.argv[1:])
H=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads(prov.read_text()); v=json.loads(verify.read_text()); r=json.loads(prereq.read_text())
d=json.loads(directed.read_text()); q=json.loads(decision.read_text()); a=json.loads(aggregate.read_text())
l=json.loads(lifecycle.read_text()); t=json.loads(attest.read_text())
plugin='1da5f731da796656759f0e673e3479392b6b8337c054a61aa4ca1fd0afb4edd4'
assert H(wheel)=='3947fab41739c98a30a8fd5486b867347b932f3419def3bfbd846db458ba90a9'
assert p['schema']=='c2-native-plugin-overlay-provenance-v1' and p['derived_wheel']['sha256']==H(wheel) and p['plugin_expected_sha256']==plugin
assert v['schema']=='c2-native-plugin-v16-k-chunk-lookahead-overlay-wheel-driver-verification-v1' and v['all_gates_pass'] is True and v['wheel_sha256']==H(wheel) and v['stable_member_byte_identical'] is True and v['plugin_sha256']==plugin
assert r['schema']=='c2-native-plugin-v16-k-chunk-lookahead-wheel-promotion-prerequisites-v1' and r['stress_accepted'] is True and r['directed_all_gates_pass'] is True
assert d['all_gates_pass'] is True and d['no_monkeypatch'] is True and d['v6_directed_complete_gate_passed'] is True
assert q['accepted'] is True and q['all_correctness_and_resource_identity_gates_pass'] is True and q['aggregate_sha256']==H(aggregate)
assert q['performance']['paired_median_improvement_fraction'] > .03 and q['performance']['bootstrap_95pct_lower_bound_fraction'] > 0
assert a['schema']=='c2-native-plugin-v16-k-chunk-lookahead-vs-v12-3pct-incremental-aggregate-v1' and a['allocated_gpu_uuid'].startswith('GPU-')
assert len(a['seeds'])==len(set(a['seeds']))==8 and all(isinstance(seed, int) for seed in a['seeds'])
for vector in ('candidate_seed_mean_ms','reference_seed_mean_ms','paired_seed_improvement_fraction'):
    assert len(a[vector])==8 and all(math.isfinite(float(value)) for value in a[vector])
assert a['process_isolation']['same_python_process'] is False
assert a['paired_median_improvement_fraction']==q['performance']['paired_median_improvement_fraction']
assert a['bootstrap']['lower_bound_fraction']==q['performance']['bootstrap_95pct_lower_bound_fraction']
assert l['schema']=='c2-native-c2-v16-k-chunk-lookahead-production-lifecycle-v1' and l['all_gates_pass'] is True
assert t['schema']=='c2-native-plugin-v16-k-chunk-lookahead-lifecycle-accepted-evidence-attestation-v1' and t['all_gates_pass'] is True and t['plugin_sha256']==plugin and t['stress_decision_sha256']==H(decision) and t['stress_aggregate_sha256']==H(aggregate)
assert plugin in aot.read_text()
PY

RUNTIME=$(mktemp -d -p "${TMP_PARENT}" "c2-v16-fresh-runtime-${RUN_ID}.XXXXXXXX")
[[ -d "${RUNTIME}" && ! -L "${RUNTIME}" && "$(dirname -- "${RUNTIME}")" == "${TMP_PARENT}" ]] || { echo 'failed to create confined fresh runtime' >&2; exit 2; }
SUPPORT=${RUNTIME}/c2_support
printf 'runtime_parent=%s\nnode_local_install=true\nephemeral=true\n' "${TMP_PARENT}" > "${ARTIFACT_DIR}/runtime-location-job${RUN_ID}.txt"
"${UV}" pip install --python "${PYTHON}" --target "${RUNTIME}" --no-deps "${WHEEL}"
[[ ! -L "${RUNTIME}" && -z "$(find "${RUNTIME}" -type l -print -quit)" ]]
install -D -m 0644 "${HARNESS}" "${ARTIFACT_DIR}/native_c2_full_backend_bench.py"

"${PYTHON}" - "${RUNTIME}" "${WHEEL}" "${WHEEL_PROVENANCE}" "${ARTIFACT_DIR}/installed-members-job${RUN_ID}.json" <<'PY'
import base64,csv,hashlib,io,json,sys,zipfile
from pathlib import Path
runtime,wheel,prov,out=map(Path,sys.argv[1:]); H=lambda b:hashlib.sha256(b).hexdigest(); p=json.loads(prov.read_text())
plugin='vllm/_native_c2_msa_decode_plugin.abi3.so'; stable='vllm/_C_stable_libtorch.abi3.so'
with zipfile.ZipFile(wheel) as z:
 names=[n for n in z.namelist() if not n.endswith('/')]; rec=[n for n in names if n.endswith('.dist-info/RECORD')]; assert len(rec)==1; rec=rec[0]
 rows=list(csv.reader(io.StringIO(z.read(rec).decode()))); assert len(rows)==len({r[0] for r in rows}) and {r[0] for r in rows}==set(names)
 for n,e,s in rows:
  if n==rec: assert e==s==''; continue
  enc=base64.urlsafe_b64encode(hashlib.sha256(z.read(n)).digest()).decode().rstrip('='); assert e==f'sha256={enc}' and int(s)==len(z.read(n)); assert (runtime/n).is_file() and (runtime/n).read_bytes()==z.read(n)
 assert H(z.read(plugin))==p['plugin_expected_sha256']=='1da5f731da796656759f0e673e3479392b6b8337c054a61aa4ca1fd0afb4edd4'
 assert H(z.read(stable))=='cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
 installed=list(csv.reader((runtime/rec).read_text().splitlines())); added={'bin/vllm',f'{rec.rsplit("/",1)[0]}/INSTALLER',f'{rec.rsplit("/",1)[0]}/REQUESTED',f'{rec.rsplit("/",1)[0]}/direct_url.json',f'{rec.rsplit("/",1)[0]}/uv_cache.json'}; assert {r[0] for r in installed}==set(names)|added
 for n,e,s in installed:
  f=(runtime/n).resolve(); assert f.is_relative_to(runtime.resolve()) and f.is_file()
  if n==rec: assert e==s==''; continue
  enc=base64.urlsafe_b64encode(hashlib.sha256(f.read_bytes()).digest()).decode().rstrip('='); assert e==f'sha256={enc}' and int(s)==f.stat().st_size
 actual={str(f.relative_to(runtime)) for f in runtime.rglob('*') if f.is_file()}; assert actual=={r[0] for r in installed}|{'.lock'}
out.write_text(json.dumps({'schema':'c2-native-plugin-v16-k-chunk-lookahead-fresh-installed-members-v1','wheel_record_full_verification':True,'installed_record_full_verification':True,'wheel_payload_except_installer_rewritten_record_byte_identical':True,'plugin_sha256':p['plugin_expected_sha256'],'stable_member_sha256':'cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442','all_gates_pass':True},indent=2,sort_keys=True)+'\n')
PY

SUPPORT_SOURCE_MANIFEST=${ARTIFACT_DIR}/support-source-job${RUN_ID}.sha256
SUPPORT_INSTALLED_MANIFEST=${ARTIFACT_DIR}/support-installed-job${RUN_ID}.sha256
printf '%s  %s\n' 'df9dd0b01c3e91afeb1a490785fab1a0554a16f001e118005192f55f891358a8' 'harness/__init__.py' '38faefb89b38e3bc26b135b762a6bbbd813dcc695423982b423a806fe41c11d6' 'harness/data.py' '712c1632367700137392d2c43bd5c6962130d945f01f1f585cc6338788ce53cc' 'harness/reference.py' '6b0a40463b4f1c05ca53649d6689377b4f755d1efb58b10700dad05705b53f71' 'harness/triton_baseline.py' '5db5d5e692044e54bb57418cbb7dcf5c9cc94f789eaccf128eec6fa8df86d967' 'vllm_msa_ref/sparse_attn.py' > "${SUPPORT_SOURCE_MANIFEST}"
(cd "${C2_ROOT}" && sha256sum -c "${SUPPORT_SOURCE_MANIFEST}")
mkdir -p "${SUPPORT}"
while read -r digest member; do source=${C2_ROOT}/${member}; [[ -f "${source}" && ! -L "${source}" ]]; install -D -m 0444 "${source}" "${SUPPORT}/${member}"; done < "${SUPPORT_SOURCE_MANIFEST}"
cp "${SUPPORT_SOURCE_MANIFEST}" "${SUPPORT_INSTALLED_MANIFEST}"
(cd "${SUPPORT}" && sha256sum -c "${SUPPORT_INSTALLED_MANIFEST}")
[[ -z "$(find "${SUPPORT}" -type l -print -quit)" && ! -e "${SUPPORT}/vllm" ]]; chmod -R a-w "${SUPPORT}"

BASELINE_OPS=${ARTIFACT_DIR}/baseline-dispatch-surface-job${RUN_ID}.json
INSTALLED_PRE=${ARTIFACT_DIR}/installed-preloader-surface-job${RUN_ID}.json
INSTALLED_POST=${ARTIFACT_DIR}/installed-postloader-surface-job${RUN_ID}.json
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH= "${PYTHON}" - "${BASELINE_VLLM}" "${BASELINE_OPS}" <<'PY'
import hashlib,json,sys
from pathlib import Path
import torch,vllm; import vllm._C_stable_libtorch as stable
root,out=map(Path,sys.argv[1:]); assert Path(vllm.__file__).resolve().parent==root.resolve(); assert hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest()=='cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
ops=sorted(x for x in torch._C._dispatch_get_all_op_names() if x.startswith('_C::')); assert '_C::native_c2_msa_decode' not in ops
out.write_text(json.dumps({'schema':'c2-native-plugin-v16-k-chunk-lookahead-fresh-baseline-dispatch-surface-v1','ops':ops,'stable_sha256':'cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'},indent=2,sort_keys=True)+'\n')
PY
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}" "${PYTHON}" - "${RUNTIME}" "${BASELINE_OPS}" "${INSTALLED_PRE}" "${INSTALLED_POST}" <<'PY'
import hashlib,importlib,importlib.metadata,json,sys
from pathlib import Path
import torch,vllm; import vllm._C_stable_libtorch as stable
runtime,baseline,pre,post=map(Path,sys.argv[1:]); root=Path(vllm.__file__).resolve().parent; assert root==runtime.resolve()/'vllm' and importlib.metadata.distribution('vllm').locate_file('vllm').resolve()==root; assert hashlib.sha256(Path(stable.__file__).read_bytes()).hexdigest()=='cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442'
b=json.loads(baseline.read_text()); mod=importlib.import_module('vllm.models.minimax_m3.nvidia.msa_native_c2_decode'); assert Path(mod.__file__).resolve()==root/'models/minimax_m3/nvidia/msa_native_c2_decode.py'
before=sorted(x for x in torch._C._dispatch_get_all_op_names() if x.startswith('_C::')); assert before==b['ops'] and '_C::native_c2_msa_decode' not in before
pre.write_text(json.dumps({'schema':'c2-native-plugin-v16-k-chunk-lookahead-fresh-installed-preloader-surface-v1','matches_immutable_baseline':True,'ops':before},indent=2,sort_keys=True)+'\n')
loader=getattr(mod,'_load_native_c2_plugin_once'); assert callable(loader) and loader() is True
after=sorted(x for x in torch._C._dispatch_get_all_op_names() if x.startswith('_C::')); assert sorted(set(after)-set(before))==['_C::native_c2_msa_decode'] and torch._C._dispatch_has_kernel_for_dispatch_key('_C::native_c2_msa_decode','CUDA'); assert loader() is True and sorted(x for x in torch._C._dispatch_get_all_op_names() if x.startswith('_C::'))==after
plugin=root/'_native_c2_msa_decode_plugin.abi3.so'; assert hashlib.sha256(plugin.read_bytes()).hexdigest()=='1da5f731da796656759f0e673e3479392b6b8337c054a61aa4ca1fd0afb4edd4'
post.write_text(json.dumps({'schema':'c2-native-plugin-v16-k-chunk-lookahead-fresh-installed-postloader-surface-v1','plugin_sha256':'1da5f731da796656759f0e673e3479392b6b8337c054a61aa4ca1fd0afb4edd4','added_ops':['_C::native_c2_msa_decode'],'native_c2_has_cuda_kernel':True,'all_baseline_op_names_preserved':set(before).issubset(after)},indent=2,sort_keys=True)+'\n')
PY

GPU_PRE=${ARTIFACT_DIR}/gpu-pre-job${RUN_ID}.txt; GPU_PRE_APPS=${ARTIFACT_DIR}/gpu-pre-apps-job${RUN_ID}.txt
GPU_POST=${ARTIFACT_DIR}/gpu-post-job${RUN_ID}.txt; GPU_POST_APPS=${ARTIFACT_DIR}/gpu-post-apps-job${RUN_ID}.txt
EXPECTED_GPU_UUID=$(gpu_uuid); snapshot PRE "${GPU_PRE}" "${GPU_PRE_APPS}"
[[ -z "$(<"${GPU_PRE_APPS}")" ]] || { echo 'GPU is not clean before fresh validation' >&2; exit 1; }
grep -q 'B300' "${GPU_PRE}" || { echo 'requires B300 allocation' >&2; exit 1; }

UUID_MANIFEST=${ARTIFACT_DIR}/seed-gpu-uuid-job${RUN_ID}.csv
printf 'seed,before_uuid,after_uuid,result,trace\n' > "${UUID_MANIFEST}"
for seed in "${SEEDS[@]}"; do
  result=${ARTIFACT_DIR}/plugin-v16-wheel-full-backend-seed${seed}-job${RUN_ID}.json
  trace=${ARTIFACT_DIR}/plugin-v16-wheel-full-backend-seed${seed}-job${RUN_ID}.chrome.json
  before=$(gpu_uuid); [[ "${before}" == "${EXPECTED_GPU_UUID}" ]]
  PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH="${RUNTIME}:${SUPPORT}" timeout --foreground --preserve-status --kill-after=10s 360s "${PYTHON}" "${ARTIFACT_DIR}/native_c2_full_backend_bench.py" --c2-root "${SUPPORT}" --vllm-root "${RUNTIME}" --output "${result}" --trace "${trace}" --seed "${seed}" --timing-warmup 10 --timing-repetitions 50
  after=$(gpu_uuid); [[ "${after}" == "${EXPECTED_GPU_UUID}" ]]
  printf '%s,%s,%s,%s,%s\n' "${seed}" "${before}" "${after}" "${result}" "${trace}" >> "${UUID_MANIFEST}"
done

SUMMARY=${ARTIFACT_DIR}/plugin-v16-fresh-backend-summary-job${RUN_ID}.json
DECISION=${ARTIFACT_DIR}/plugin-v16-fresh-backend-decision-job${RUN_ID}.json
"${PYTHON}" - "${SUMMARY}" "${DECISION}" "${UUID_MANIFEST}" "${ARTIFACT_DIR}" "${RUNTIME}" "${EXPECTED_GPU_UUID}" "${SEEDS[@]}" <<'PY'
import csv,hashlib,json,math,statistics,sys
from pathlib import Path
summary,decision,uuids,art,runtime,uuid,*ss=sys.argv[1:]; summary,decision,uuids,art,runtime=map(Path,(summary,decision,uuids,art,runtime)); seeds=list(map(int,ss)); rows=list(csv.DictReader(uuids.open())); assert [int(x['seed']) for x in rows]==seeds and all(x['before_uuid']==x['after_uuid']==uuid for x in rows)
native=[]; triton=[]; ratios=[]; results=[]
for seed in seeds:
 p=art/f'plugin-v16-wheel-full-backend-seed{seed}-job{art.name.removeprefix("job")}.json'; trace=art/f'plugin-v16-wheel-full-backend-seed{seed}-job{art.name.removeprefix("job")}.chrome.json'; x=json.loads(p.read_text())
 assert x['schema']=='c2-native-c2-full-vllm-backend-v2' and x['seed']==seed and x['all_gates_pass'] is True and x['correctness']['pass'] is True and x['triton_correctness']['pass'] is True and x['correctness']['oracle'].startswith('independent FP32 ') and x['triton_correctness']['oracle'].startswith('independent FP32 ') and x['caller_output']['pointer_unchanged'] is True and x['profiling']['checks']['one_cpu_dispatcher_event'] is True and x['profiling']['checks']['one_cuda_native_kernel_event'] is True and trace.is_file() and trace.stat().st_size>0
 t=x['timing']; assert t['pass'] is True and 'ABBA=native,triton,triton,native' in t['protocol'] and t['warmup']==10 and t['repetitions']==50 and t['native']['sample_count']==t['triton']['sample_count']==100 and all(t['pre_timing_checks'][k] is True for k in ('distinct_output_buffers','native_hard_gates','triton_builder','triton_correctness','triton_static_selection'))
 n=float(t['native']['median_ms']); tr=float(t['triton']['median_ms']); assert math.isfinite(n) and math.isfinite(tr) and n>0 and tr>0 and Path(x['environment']['vllm_import_root']).resolve()==runtime/'vllm'
 native.append(n); triton.append(tr); ratios.append(n/tr); results.append({'seed':seed,'result':str(p),'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'trace':str(trace),'native_median_ms':n,'triton_median_ms':tr,'native_over_triton_latency_ratio':n/tr})
payload={'schema':'c2-native-plugin-v16-k-chunk-lookahead-fresh-backend-summary-v1','all_integration_and_correctness_gates_pass':True,'seeds':results,'seed_count':8,'one_physical_gpu_uuid':uuid,'fresh_process_per_seed':True,'native_median_of_seed_medians_ms':statistics.median(native),'triton_median_of_seed_medians_ms':statistics.median(triton),'native_over_triton_ratio_of_seed_medians':statistics.median(ratios),'triton_parity_achieved':max(ratios)<=1.0,'triton_parity_is_hard_gate':False,'v16_dso_sha256':'1da5f731da796656759f0e673e3479392b6b8337c054a61aa4ca1fd0afb4edd4','v16_vs_v12_fresh_wheel_comparator':{'available':False,'comparison_kind':'none','reason':'no v12 fresh-wheel comparator is installed in this job; accepted v16-v12 promotion remains the separate direct job13789 same-GPU ABBA result','fresh_native_over_triton_is_not_v16_vs_v12_promotion':True}}
summary.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
decision.write_text(json.dumps({'schema':'c2-native-plugin-v16-k-chunk-lookahead-fresh-backend-decision-v1','accepted_for_fresh_integration':True,'final_rc':0,'hard_gates':{'wheel_identity':True,'record_and_install':True,'stable_dso_preserved':True,'loader_surface':True,'eight_seed_real_backend':True,'direct_v16_vs_v12_bitwise_and_promotion_evidence':True,'lifecycle_evidence':True,'clean_b300_before_after':True},'performance':{'reported':True,'triton_parity_required':False,'triton_parity_achieved':payload['triton_parity_achieved'],'v16_vs_v12_fresh_wheel_comparator':'not available; not a promotion decision'},'v16_vs_v12_fresh_wheel_boundary':payload['v16_vs_v12_fresh_wheel_comparator']},indent=2,sort_keys=True)+'\n')
PY
snapshot POST "${GPU_POST}" "${GPU_POST_APPS}"
[[ "$(gpu_uuid)" == "${EXPECTED_GPU_UUID}" && -z "$(<"${GPU_POST_APPS}")" ]] || { echo 'GPU is not clean after fresh validation' >&2; exit 1; }
printf 'PRE_UUID=%s\nPOST_UUID=%s\n' "${EXPECTED_GPU_UUID}" "$(gpu_uuid)" > "${ARTIFACT_DIR}/gpu-identity-job${RUN_ID}.txt"
