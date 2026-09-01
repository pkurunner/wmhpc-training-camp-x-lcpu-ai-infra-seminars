#!/usr/bin/env bash
# Paired one-kernel NCU mechanism diagnosis for v12 Q-row padding against
# v11 Q-row padding. This is explicitly not a benchmark: NCU replay timing
# cannot alter the clean FP64-correct CUDA-event ABBA acceptance in job12985.
# Separate processes are mandatory because both DSOs register _C::native_c2_msa_decode.
# Submit only after review:
# sbatch --export=ALL,C2_EXPECTED_RUNTIME_SCRIPT_SHA=<sha256> profile_native_c2_v12_vs_v11_ncu_20260830.slurm

#SBATCH --job-name=c2-native-v12-v11-ncu
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/home/lcpu/85117379/c2-native-plugin-v12-vs-v11-ncu-artifacts-20260830/slurm-%j.log

set -Eeuo pipefail
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONPATH=
export PYTHONDONTWRITEBYTECODE=1
export PATH=/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin

ARTIFACT_ROOT=/home/lcpu/85117379/c2-native-plugin-v12-vs-v11-ncu-artifacts-20260830
ARTIFACT_DIR=${ARTIFACT_ROOT}/job${SLURM_JOB_ID:?must run under Slurm}
PYTHON=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
SUBMITTED_SCRIPT=/home/lcpu/85117379/profile_native_c2_v12_vs_v11_ncu_20260830.slurm
BASE_HARNESS=/home/lcpu/85117379/native_c2_operator_bench_20260829.py
STRESS_DECISION=/home/lcpu/85117379/c2-native-plugin-v12-stress-3pct-artifacts-20260830/job12985/v12-q-row-padding-vs-v11-3pct-incremental-decision-job12985.json
STRESS_AGGREGATE=/home/lcpu/85117379/c2-native-plugin-v12-stress-3pct-artifacts-20260830/job12985/v12-q-row-padding-vs-v11-3pct-incremental-aggregate-job12985.json

V11_LIBRARY=/home/lcpu/85117379/c2-native-plugin-v11-aot-artifacts-20260830/job12825/vllm/_native_c2_msa_decode_plugin.abi3.so
V11_SOURCE=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v11-q-fragment-reuse-20260830-job12825/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
V11_RESOURCE=/home/lcpu/85117379/c2-native-plugin-v11-aot-artifacts-20260830/job12825/plugin-resource-gate-job12825.txt
V11_PROVENANCE=/home/lcpu/85117379/c2-native-plugin-v11-aot-artifacts-20260830/job12825/plugin-v11-provenance-job12825.sha256
V12_LIBRARY=/home/lcpu/85117379/c2-native-plugin-v12-aot-artifacts-20260830/job12983/vllm/_native_c2_msa_decode_plugin.abi3.so
V12_SOURCE=/home/lcpu/85117379/vllm-d4-native-c2-plugin-v12-q-row-padding-20260830-job12983/csrc/libtorch_stable/attention/minimax_m3/native_c2_decode.cu
V12_RESOURCE=/home/lcpu/85117379/c2-native-plugin-v12-aot-artifacts-20260830/job12983/plugin-resource-gate-job12983.txt
V12_PROVENANCE=/home/lcpu/85117379/c2-native-plugin-v12-aot-artifacts-20260830/job12983/plugin-v12-provenance-job12983.sha256

EXPECTED_BASE_HARNESS_SHA=0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28
EXPECTED_STRESS_DECISION_SHA=829505fce32c737f596b372819add7f7701dc793c87d58f429e61ba7806862d9
EXPECTED_STRESS_AGGREGATE_SHA=0e9b4da6ea78c812c66feab0d4c669e68d0dc9a65630d13fbbc37c2e66295142
EXPECTED_V11_PLUGIN_SHA=501b9c56a29a81f779a35a34b2808b4122af7dcb23e8108d8c590306be43d957
EXPECTED_V11_SOURCE_SHA=0e82b278b7aa44a034a96d2ddd19946a27928d95a9f9d03e8e1fe9f30680c5b4
EXPECTED_V11_RESOURCE_SHA=f82d6890d14be7cfb1df54ec12fdff0b2de3096204591b554c46eb02862d5ab4
EXPECTED_V11_PROVENANCE_SHA=6d96a7437616c672c8640592ac99da0b722d0d6f4e7d8f78b18a3a6652b59128
EXPECTED_V12_PLUGIN_SHA=064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6
EXPECTED_V12_SOURCE_SHA=535d90b856ed1062aa7b8a105eb2c5f236c450826e65496e646d0d5a27eb8aaf
EXPECTED_V12_RESOURCE_SHA=c4a6cd69617ce82e99459fcd7a975a37fbfad497e51b36a6ec09945ffa97d6c0
EXPECTED_V12_PROVENANCE_SHA=65bb886cece4a49b7ef7d7d4cab00588e4cc1f9238d2aa45dd5cf36f6952d2a2

: "${C2_EXPECTED_RUNTIME_SCRIPT_SHA:?reviewed NCU script SHA-256 required}"
[[ ${C2_EXPECTED_RUNTIME_SCRIPT_SHA} =~ ^[0-9a-f]{64}$ ]]
for digest in "${EXPECTED_BASE_HARNESS_SHA}" "${EXPECTED_STRESS_DECISION_SHA}" "${EXPECTED_STRESS_AGGREGATE_SHA}" \
  "${EXPECTED_V11_PLUGIN_SHA}" "${EXPECTED_V11_SOURCE_SHA}" "${EXPECTED_V11_RESOURCE_SHA}" "${EXPECTED_V11_PROVENANCE_SHA}" \
  "${EXPECTED_V12_PLUGIN_SHA}" "${EXPECTED_V12_SOURCE_SHA}" "${EXPECTED_V12_RESOURCE_SHA}" "${EXPECTED_V12_PROVENANCE_SHA}"; do
  [[ ${digest} =~ ^[0-9a-f]{64}$ ]] || { echo "invalid pinned SHA-256: ${digest}" >&2; exit 2; }
done

# These counters are queried first and required exactly in each raw report.
metrics=(
  gpu__time_duration.sum smsp__cycles_elapsed.sum dram__bytes_read.sum dram__bytes_write.sum lts__t_bytes.sum
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active smsp__inst_executed_pipe_tensor.sum
  smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct
  smsp__average_warps_issue_stalled_wait_per_issue_active.pct
  smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
  smsp__warps_active.avg.per_cycle_active sm__warps_active.avg.pct_of_peak_sustained_active
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
)
metrics_csv="$(IFS=,; printf '%s' "${metrics[*]}")"
sha256_of() { sha256sum "$1" | awk '{print $1}'; }
allocated_gpu_selector() {
  local selector=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}
  [[ ${selector} != *,* && ( ${selector} =~ ^[0-9]+$ || ${selector} =~ ^GPU-[0-9A-Fa-f-]+$ ) ]] || {
    echo "expected exactly one numeric/GPU UUID CUDA_VISIBLE_DEVICES, got: ${selector}" >&2; return 1; }
  printf '%s\n' "${selector}"
}
visible_gpu_uuid() {
  local selector uuid; selector=$(allocated_gpu_selector)
  mapfile -t uuids < <(nvidia-smi --id "${selector}" --query-gpu=uuid --format=csv,noheader,nounits)
  [[ ${#uuids[@]} -eq 1 && ${uuids[0]} =~ ^GPU- ]] || { echo 'unable to identify one physical GPU UUID' >&2; return 1; }
  uuid=${uuids[0]}; printf '%s\n' "${uuid}"
}
compute_apps() { nvidia-smi --id "$(allocated_gpu_selector)" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
snapshot() {
  printf '\n===== %s UTC=%s =====\n' "$1" "$(date -u +%FT%TZ)"
  nvidia-smi --id "$(allocated_gpu_selector)" --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader,nounits
  echo '-- compute apps --'; compute_apps
}
finish() {
  local original_rc=$?
  local final_rc=${original_rc} finalizer_error=0 apps_after post_uuid manifest tmp
  trap - EXIT; set +e
  snapshot POST | tee "${ARTIFACT_DIR}/gpu-post-job${SLURM_JOB_ID}.txt"
  [[ ${PIPESTATUS[0]} -eq 0 && ${PIPESTATUS[1]} -eq 0 ]] || finalizer_error=1
  apps_after=$(compute_apps); [[ $? -eq 0 && -z ${apps_after} ]] || finalizer_error=1
  post_uuid=$(visible_gpu_uuid); [[ $? -eq 0 && ${post_uuid} == "${EXPECTED_GPU_UUID:-}" ]] || finalizer_error=1
  [[ ${finalizer_error} -eq 0 || ${final_rc} -ne 0 ]] || final_rc=2
  printf 'ORIGINAL_RC=%s FINALIZER_ERROR=%s EXPECTED_UUID=%s POST_UUID=%s APPS_EMPTY=%s\n' "${original_rc}" "${finalizer_error}" "${EXPECTED_GPU_UUID:-unset}" "${post_uuid:-unset}" "$([[ -z ${apps_after} ]] && echo true || echo false)" > "${ARTIFACT_DIR}/finalizer-diagnostics-job${SLURM_JOB_ID}.txt" || final_rc=2
  manifest=${ARTIFACT_DIR}/outputs-manifest-job${SLURM_JOB_ID}.sha256; tmp=${manifest}.tmp
  find "${ARTIFACT_DIR}" -maxdepth 1 -type f ! -name 'outputs-manifest-job*.sha256*' ! -name 'driver-job*.log' ! -name 'final-status-job*.txt' -printf '%p\0' | sort -z | xargs -0r sha256sum > "${tmp}" || final_rc=2
  [[ ${final_rc} -ne 0 ]] || mv "${tmp}" "${manifest}" || final_rc=2
  rm -f "${tmp}"
  printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s UTC=%s\n' "${final_rc}" "${original_rc}" "${finalizer_error}" "$(date -u +%FT%TZ)" > "${ARTIFACT_DIR}/final-status-job${SLURM_JOB_ID}.txt" || final_rc=2
  exit "${final_rc}"
}

[[ ! -e ${ARTIFACT_DIR} ]] || { echo "refusing to reuse ${ARTIFACT_DIR}" >&2; exit 2; }
mkdir -p "${ARTIFACT_ROOT}" "${ARTIFACT_DIR}"
exec > >(tee "${ARTIFACT_DIR}/driver-job${SLURM_JOB_ID}.log") 2>&1
trap finish EXIT
SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
[[ -f ${SCRIPT_PATH} && -f ${SUBMITTED_SCRIPT} ]]
[[ $(sha256_of "${SCRIPT_PATH}") == "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" ]]
[[ $(sha256_of "${SUBMITTED_SCRIPT}") == "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" ]]
cmp -s "${SCRIPT_PATH}" "${SUBMITTED_SCRIPT}"

snapshot PRE | tee "${ARTIFACT_DIR}/gpu-pre-job${SLURM_JOB_ID}.txt"
gpu_line=$(nvidia-smi --id "$(allocated_gpu_selector)" --query-gpu=name,compute_cap --format=csv,noheader,nounits)
EXPECTED_GPU_UUID=$(visible_gpu_uuid)
apps_before=$(compute_apps) || { echo 'unable to query pre-run compute apps' >&2; exit 2; }
[[ ${gpu_line} == *B300* && ${gpu_line} == *10.3* && -z ${apps_before} ]] || { echo 'requires one clean B300 compute-capability 10.3 allocation' >&2; exit 2; }
[[ -x ${PYTHON} && -f ${BASE_HARNESS} && -f ${STRESS_DECISION} && -f ${STRESS_AGGREGATE} ]]
[[ -f ${V11_LIBRARY} && -f ${V11_SOURCE} && -f ${V11_RESOURCE} && -f ${V11_PROVENANCE} ]]
[[ -f ${V12_LIBRARY} && -f ${V12_SOURCE} && -f ${V12_RESOURCE} && -f ${V12_PROVENANCE} ]]
command -v ncu; ncu --version | tee "${ARTIFACT_DIR}/ncu-version-job${SLURM_JOB_ID}.txt"
"${PYTHON}" -c 'import platform, torch; assert __debug__; print({"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda})' | tee "${ARTIFACT_DIR}/python-environment-job${SLURM_JOB_ID}.txt"
printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${EXPECTED_BASE_HARNESS_SHA}" "${BASE_HARNESS}" "${EXPECTED_STRESS_DECISION_SHA}" "${STRESS_DECISION}" "${EXPECTED_STRESS_AGGREGATE_SHA}" "${STRESS_AGGREGATE}" \
  "${EXPECTED_V11_PLUGIN_SHA}" "${V11_LIBRARY}" "${EXPECTED_V11_SOURCE_SHA}" "${V11_SOURCE}" "${EXPECTED_V11_RESOURCE_SHA}" "${V11_RESOURCE}" "${EXPECTED_V11_PROVENANCE_SHA}" "${V11_PROVENANCE}" \
  "${EXPECTED_V12_PLUGIN_SHA}" "${V12_LIBRARY}" "${EXPECTED_V12_SOURCE_SHA}" "${V12_SOURCE}" "${EXPECTED_V12_RESOURCE_SHA}" "${V12_RESOURCE}" "${EXPECTED_V12_PROVENANCE_SHA}" "${V12_PROVENANCE}" \
  "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SCRIPT_PATH}" "${C2_EXPECTED_RUNTIME_SCRIPT_SHA}" "${SUBMITTED_SCRIPT}" > "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
sha256sum -c "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256"
grep -F "${EXPECTED_V11_PLUGIN_SHA}  ${V11_LIBRARY}" "${V11_PROVENANCE}"; grep -F "${EXPECTED_V11_SOURCE_SHA}  ${V11_SOURCE}" "${V11_PROVENANCE}"
grep -F "${EXPECTED_V12_PLUGIN_SHA}  ${V12_LIBRARY}" "${V12_PROVENANCE}"; grep -F "${EXPECTED_V12_SOURCE_SHA}  ${V12_SOURCE}" "${V12_PROVENANCE}"
for item in 'V11_RESOURCE:REG=127' 'V11_RESOURCE:STACK=0' 'V11_RESOURCE:SHARED=30880' 'V11_RESOURCE:LOCAL=0' 'V12_RESOURCE:REG=127' 'V12_RESOURCE:STACK=0' 'V12_RESOURCE:SHARED=31136' 'V12_RESOURCE:LOCAL=0'; do
  variable=${item%%:*}; expected=${item#*:}; grep -qx "${expected}" "${!variable}" || { echo "resource gate failed: ${item}" >&2; exit 2; }
done
"${PYTHON}" - "${STRESS_DECISION}" "${STRESS_AGGREGATE}" "${EXPECTED_STRESS_DECISION_SHA}" "${EXPECTED_STRESS_AGGREGATE_SHA}" <<'PY'
import hashlib, json, sys
from pathlib import Path
decision, aggregate = map(Path, sys.argv[1:3]); expected_decision, expected_aggregate = sys.argv[3:]
if decision.name != 'v12-q-row-padding-vs-v11-3pct-incremental-decision-job12985.json' or aggregate.name != 'v12-q-row-padding-vs-v11-3pct-incremental-aggregate-job12985.json':
    raise SystemExit('unexpected job12985 evidence filenames')
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if digest(decision) != expected_decision or digest(aggregate) != expected_aggregate: raise SystemExit('job12985 evidence hash drift')
d = json.loads(decision.read_text(encoding='utf-8'))
if d.get('schema') != 'c2-native-plugin-v12-q-row-padding-vs-v11-3pct-incremental-decision-v1': raise SystemExit('unexpected job12985 decision schema')
if d.get('accepted') is not True or d.get('issues') != [] or d.get('all_correctness_and_resource_identity_gates_pass') is not True: raise SystemExit('job12985 was not accepted cleanly')
p = d.get('performance', {})
if not (float(p.get('paired_median_improvement_fraction', 0)) > .03 and float(p.get('bootstrap_95pct_lower_bound_fraction', 0)) > 0): raise SystemExit('job12985 performance gates are not the accepted result')
if d.get('aggregate_sha256') != expected_aggregate: raise SystemExit('job12985 aggregate linkage drift')
PY

METRIC_CATALOG=${ARTIFACT_DIR}/metric-catalog-job${SLURM_JOB_ID}.txt
grep_args=(); for metric in "${metrics[@]}"; do grep_args+=(--regexp="${metric}"); done
timeout --signal=TERM --kill-after=15s 180s ncu --query-metrics --query-metrics-mode all | grep --fixed-strings "${grep_args[@]}" > "${METRIC_CATALOG}"
"${PYTHON}" - "${METRIC_CATALOG}" "${metrics[@]}" <<'PY'
import re, sys
catalog = open(sys.argv[1], encoding='utf-8', errors='replace').read()
available = set(re.findall(r'\b[A-Za-z][A-Za-z0-9_]*(?:__[A-Za-z0-9_]+)+(?:\.[A-Za-z0-9_]+)+\b', catalog))
missing = [m for m in sys.argv[2:] if m not in available]
if missing or len(set(sys.argv[2:])) != len(sys.argv[2:]): raise SystemExit('missing/duplicate exact NCU metrics: ' + ', '.join(missing))
PY

TARGET_RUNNER=${ARTIFACT_DIR}/ncu_target_runner.py
cat > "${TARGET_RUNNER}" <<'PY'
#!/usr/bin/env python3
import argparse, hashlib, importlib.util, json, os, platform, subprocess, sys, traceback
from pathlib import Path
import torch
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_base(path):
    spec = importlib.util.spec_from_file_location('c2_ncu_base', path)
    if spec is None or spec.loader is None: raise RuntimeError(f'cannot import {path}')
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    for name in ('_BATCH','_QUERY_HEADS','_HEAD_DIM','_make_inputs','_oracle','_validate_args'):
        if not hasattr(mod, name): raise RuntimeError(f'base harness lacks {name}')
    return mod
def tensor_digest(t):
    t=t.detach().contiguous().cpu(); h=hashlib.sha256(); h.update(str(t.dtype).encode()); h.update(repr(tuple(t.shape)).encode()); h.update(t.view(torch.uint8).numpy().tobytes()); return h.hexdigest()
def uuid():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(visible)!=1 or not visible[0] or not (visible[0].isdigit() or visible[0].startswith('GPU-')): raise RuntimeError('one CUDA_VISIBLE_DEVICES required')
    rows=[x.strip() for x in subprocess.check_output(['nvidia-smi','--id',visible[0],'--query-gpu=uuid','--format=csv,noheader,nounits'], text=True).splitlines() if x.strip()]
    if len(rows)!=1 or not rows[0].startswith('GPU-'): raise RuntimeError('cannot establish physical GPU UUID')
    return rows[0]
def invoke(op,out,inputs,c): return op(out,*inputs,c.scale,c.q_scale,c.k_scale,c.v_scale)
def main(a):
    record={'schema':'c2-native-v12-vs-v11-ncu-target-v1','mode':a.mode,'version':a.version,'all_gates_pass':False}
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(10,3): raise RuntimeError('one B300 CUDA device required')
        if not Path(a.library).is_absolute() or digest(a.library)!=a.library_sha256 or digest(a.base_harness)!=a.base_harness_sha256: raise RuntimeError('pinned input checksum drift')
        base=load_base(Path(a.base_harness)); c=base._validate_args(type('Args',(),{'num_physical_pages':64,'max_logical_pages':32,'scale':1/(128**.5),'q_scale':.25,'k_scale':.25,'v_scale':.5,'atol':1e-4,'rtol':1e-3})())
        torch.ops.load_library(a.library)
        if not torch._C._dispatch_has_kernel_for_dispatch_key('_C::native_c2_msa_decode','CUDA'): raise RuntimeError('native CUDA dispatch is not registered')
        inputs=base._make_inputs(c,a.seed); ih=[tensor_digest(x) for x in inputs]
        out=torch.full((base._BATCH,base._QUERY_HEADS,base._HEAD_DIM),float('nan'),device='cuda',dtype=torch.bfloat16); ptr=out.data_ptr(); ret=invoke(torch.ops._C.native_c2_msa_decode,out,inputs,c); torch.cuda.synchronize()
        checks={'return_is_none':ret is None,'pointer_unchanged':out.data_ptr()==ptr,'output_finite':bool(torch.isfinite(out).all().item())}
        record.update({'operator_library':str(Path(a.library).resolve()),'operator_library_sha256':digest(a.library),'base_harness':str(Path(a.base_harness).resolve()),'base_harness_sha256':digest(a.base_harness),'seed':a.seed,'input_tensor_sha256':ih,'input_manifest_sha256':hashlib.sha256(''.join(ih).encode()).hexdigest(),'contract':{'batch':base._BATCH,'head_dim':base._HEAD_DIM,'kv_heads':4,'q_heads':base._QUERY_HEADS,'page_size':128,'topk':16,'num_physical_pages':64,'max_logical_pages':32,'scale':c.scale,'q_scale':c.q_scale,'k_scale':c.k_scale,'v_scale':c.v_scale},'caller_output':checks,'device':{'name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),'gpu_uuid':uuid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','')},'environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda},'dispatch':'direct_torch.ops._C.native_c2_msa_decode','no_monkeypatch':True})
        if a.mode=='ncu': record['all_gates_pass']=all(checks.values()); print(json.dumps(record,sort_keys=True)); return
        ref=base._oracle(*inputs,c,torch.float64); actual=out.to(torch.float64); delta=(actual-ref).abs(); denom=ref.abs().clamp_min(torch.finfo(torch.float64).eps)
        p_out=torch.full_like(out,float('nan')); p_ptr=p_out.data_ptr()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            p_ret=invoke(torch.ops._C.native_c2_msa_decode,p_out,inputs,c); torch.cuda.synchronize()
        events=prof.key_averages(); dispatch=[x for x in events if str(x.key)=='_C::native_c2_msa_decode']; kernels=[x for x in events if 'native_c2_msa_decode_kernel' in str(x.key)]
        pchecks={'one_dispatch':len(dispatch)==1 and int(getattr(dispatch[0],'count',0))==1,'one_native_kernel':len(kernels)==1 and int(getattr(kernels[0],'count',0))==1,'profile_return_is_none':p_ret is None,'profile_pointer_unchanged':p_out.data_ptr()==p_ptr}
        record['correctness']={'oracle_dtype':'float64','atol':1e-4,'rtol':1e-3,'allclose':bool(torch.allclose(actual,ref,atol=1e-4,rtol=1e-3)),'finite_output':checks['output_finite'],'max_abs':float(delta.max().item()),'max_rel':float((delta/denom).max().item())}; record['profile']=pchecks; record['all_gates_pass']=all(checks.values()) and all(pchecks.values()) and record['correctness']['allclose']; print(json.dumps(record,sort_keys=True))
    except Exception as e:
        record['error']=str(e); record['traceback']=traceback.format_exc(); print(json.dumps(record,sort_keys=True)); raise
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=('validate','ncu'),required=True); p.add_argument('--version',choices=('v11','v12'),required=True); p.add_argument('--library',type=Path,required=True); p.add_argument('--library-sha256',required=True); p.add_argument('--base-harness',type=Path,required=True); p.add_argument('--base-harness-sha256',required=True); p.add_argument('--seed',type=int,required=True); main(p.parse_args())
PY
chmod 0644 "${TARGET_RUNNER}"; "${PYTHON}" -m py_compile "${TARGET_RUNNER}"
run_target() { timeout --foreground --preserve-status --kill-after=15s 300s "${PYTHON}" "${TARGET_RUNNER}" --mode validate --version "$1" --library "$2" --library-sha256 "$3" --base-harness "${BASE_HARNESS}" --base-harness-sha256 "${EXPECTED_BASE_HARNESS_SHA}" --seed 20260829 > "$4" 2>&1; }
V11_PREFLIGHT=${ARTIFACT_DIR}/v11-preflight-job${SLURM_JOB_ID}.json; V12_PREFLIGHT=${ARTIFACT_DIR}/v12-preflight-job${SLURM_JOB_ID}.json
run_target v11 "${V11_LIBRARY}" "${EXPECTED_V11_PLUGIN_SHA}" "${V11_PREFLIGHT}"; run_target v12 "${V12_LIBRARY}" "${EXPECTED_V12_PLUGIN_SHA}" "${V12_PREFLIGHT}"
"${PYTHON}" - "${V11_PREFLIGHT}" "${V12_PREFLIGHT}" "${V11_LIBRARY}" "${V12_LIBRARY}" "${EXPECTED_V11_PLUGIN_SHA}" "${EXPECTED_V12_PLUGIN_SHA}" "${EXPECTED_GPU_UUID}" <<'PY'
import json, sys
from pathlib import Path
paths=list(map(Path,sys.argv[1:3])); libs=list(map(Path,sys.argv[3:5])); hashes=sys.argv[5:7]; expected=sys.argv[7]
def target_record(path):
    records=[]
    for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
        try: value=json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(value,dict) and value.get('schema')=='c2-native-v12-vs-v11-ncu-target-v1': records.append(value)
    if len(records)!=1: raise SystemExit(f'{path}: expected one target JSON record, got {len(records)}')
    return records[0]
rows=[target_record(p) for p in paths]
for version,row,lib,digest in zip(('v11','v12'),rows,libs,hashes):
    if row.get('schema')!='c2-native-v12-vs-v11-ncu-target-v1' or row.get('version')!=version or row.get('mode')!='validate' or row.get('all_gates_pass') is not True: raise SystemExit(f'{version}: invalid preflight')
    if Path(row.get('operator_library','')).resolve()!=lib.resolve() or row.get('operator_library_sha256')!=digest: raise SystemExit(f'{version}: DSO drift')
if rows[0]['input_tensor_sha256']!=rows[1]['input_tensor_sha256'] or rows[0]['input_manifest_sha256']!=rows[1]['input_manifest_sha256'] or rows[0]['contract']!=rows[1]['contract'] or rows[0]['base_harness_sha256']!=rows[1]['base_harness_sha256']: raise SystemExit('v11/v12 preflight fixture or harness differs')
if [x['device'].get('gpu_uuid') for x in rows] != [expected,expected]: raise SystemExit('v11/v12 preflight lacks same physical GPU UUID')
PY
apps_after_preflight=$(compute_apps) || exit 2; [[ -z ${apps_after_preflight} ]] || { echo 'GPU not clean after preflight' >&2; exit 2; }

V11_REPORT=${ARTIFACT_DIR}/native-c2-v11-job${SLURM_JOB_ID}.ncu-rep; V12_REPORT=${ARTIFACT_DIR}/native-c2-v12-job${SLURM_JOB_ID}.ncu-rep
V11_TARGET=${ARTIFACT_DIR}/v11-ncu-target-job${SLURM_JOB_ID}.log; V12_TARGET=${ARTIFACT_DIR}/v12-ncu-target-job${SLURM_JOB_ID}.log
profile_one() { timeout --signal=TERM --kill-after=30s 300s ncu --target-processes all --kernel-name-base function --kernel-name 'regex:.*native_c2_msa_decode_kernel.*' --launch-skip 0 --launch-count 1 --clock-control none --metrics "${metrics_csv}" --force-overwrite --export "$3" "${PYTHON}" "${TARGET_RUNNER}" --mode ncu --version "$1" --library "$2" --library-sha256 "$4" --base-harness "${BASE_HARNESS}" --base-harness-sha256 "${EXPECTED_BASE_HARNESS_SHA}" --seed 20260829 > "$5" 2>&1; [[ -s $3 && -s $5 ]]; }
profile_one v11 "${V11_LIBRARY}" "${V11_REPORT}" "${EXPECTED_V11_PLUGIN_SHA}" "${V11_TARGET}"
apps_after_v11=$(compute_apps) || exit 2; [[ -z ${apps_after_v11} ]] || { echo 'GPU not clean between NCU profiles' >&2; exit 2; }
profile_one v12 "${V12_LIBRARY}" "${V12_REPORT}" "${EXPECTED_V12_PLUGIN_SHA}" "${V12_TARGET}"
apps_after_v12=$(compute_apps) || exit 2; [[ -z ${apps_after_v12} ]] || { echo 'GPU not clean after v12 NCU' >&2; exit 2; }
V11_RAW=${ARTIFACT_DIR}/native-c2-v11-job${SLURM_JOB_ID}.raw.csv; V12_RAW=${ARTIFACT_DIR}/native-c2-v12-job${SLURM_JOB_ID}.raw.csv
timeout --signal=TERM --kill-after=15s 120s ncu --import "${V11_REPORT}" --csv --page raw --print-units base --print-fp > "${V11_RAW}"
timeout --signal=TERM --kill-after=15s 120s ncu --import "${V12_REPORT}" --csv --page raw --print-units base --print-fp > "${V12_RAW}"
[[ -s ${V11_RAW} && -s ${V12_RAW} ]]

parse_one() {
  "${PYTHON}" - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "${metrics[@]}" <<'PY'
import csv,json,math,sys
from collections import Counter
from pathlib import Path
raw,target,out,version,ereg,eshared,library,*required=sys.argv[1:]; ereg=float(ereg); eshared=float(eshared)
records=[]
for line in Path(target).read_text(encoding='utf-8',errors='replace').splitlines():
    try: value=json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(value,dict) and value.get('schema')=='c2-native-v12-vs-v11-ncu-target-v1': records.append(value)
if len(records)!=1 or records[0].get('mode')!='ncu' or records[0].get('version')!=version or records[0].get('all_gates_pass') is not True: raise SystemExit(f'{version}: invalid NCU target')
if Path(records[0].get('operator_library','')).resolve()!=Path(library).resolve(): raise SystemExit(f'{version}: target DSO drift')
rows=list(csv.reader(Path(raw).open(encoding='utf-8',errors='replace',newline=''))); launch=('launch__cluster_size','launch__cluster_max_potential_size','launch__cluster_max_active','launch__occupancy_cluster_pct','launch__occupancy_cluster_gpu_pct','launch__registers_per_thread','launch__shared_mem_per_block','launch__barrier_count')
i=next((j for j,row in enumerate(rows) if 'ID' in row and 'Kernel Name' in row and all(m in row for m in required)),None)
if i is None: raise SystemExit(f'{version}: raw CSV lacks requested metric header')
header=rows[i]; counts=Counter(header)
if any(counts[c]!=1 for c in ('ID','Kernel Name',*required,*launch)): raise SystemExit(f'{version}: incomplete/duplicated CSV column')
idx={x:header.index(x) for x in ('ID','Kernel Name',*required,*launch)}; last=max(idx.values())
if i+1>=len(rows) or len(rows[i+1])<=last: raise SystemExit(f'{version}: raw CSV lacks units row')
matches=[r for r in rows[i+2:] if len(r)>last and 'native_c2_msa_decode_kernel' in r[idx['Kernel Name']]]
if len(matches)!=1: raise SystemExit(f'{version}: expected exactly one filtered native kernel row, got {len(matches)}')
row=matches[0]
def num(name):
    x=float(row[idx[name]].strip().replace(',',''))
    if not math.isfinite(x) or x<0: raise SystemExit(f'{version}: invalid {name}')
    return x
values={m:num(m) for m in required}; meta={m:num(m) for m in launch}
if any(values[m]<=0 for m in ('gpu__time_duration.sum','smsp__cycles_elapsed.sum','smsp__inst_executed_pipe_tensor.sum')): raise SystemExit(f'{version}: required positive counters absent')
if meta['launch__registers_per_thread']!=ereg or meta['launch__shared_mem_per_block']!=eshared or meta['launch__cluster_size']!=4 or meta['launch__barrier_count']!=1: raise SystemExit(f'{version}: launch resource drift {meta}')
payload={'schema':'c2-native-v12-vs-v11-ncu-counters-v1','version':version,'all_gates_pass':True,'kernel_name':row[idx['Kernel Name']],'action_id':row[idx['ID']].strip(),'matching_kernel_rows':1,'target':records[0],'metrics':{m:{'value':values[m],'unit':rows[i+1][idx[m]].strip()} for m in required},'launch_metadata':meta,'timing_valid_for_benchmark':False,'timing_note':'NCU replay/instrumentation timing is invalid for performance decisions; job12985 clean non-NCU ABBA is the only performance decision.'}
Path(out).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); print(json.dumps(payload,sort_keys=True))
PY
}
V11_SUMMARY=${ARTIFACT_DIR}/native-c2-v11-ncu-job${SLURM_JOB_ID}.json; V12_SUMMARY=${ARTIFACT_DIR}/native-c2-v12-ncu-job${SLURM_JOB_ID}.json
parse_one "${V11_RAW}" "${V11_TARGET}" "${V11_SUMMARY}" v11 127 30880 "${V11_LIBRARY}"
parse_one "${V12_RAW}" "${V12_TARGET}" "${V12_SUMMARY}" v12 127 31136 "${V12_LIBRARY}"

MECHANISM=${ARTIFACT_DIR}/v12-v11-mechanism-job${SLURM_JOB_ID}.json
"${PYTHON}" - "${V11_SUMMARY}" "${V12_SUMMARY}" "${V11_PREFLIGHT}" "${V12_PREFLIGHT}" "${STRESS_DECISION}" "${STRESS_AGGREGATE}" "${MECHANISM}" "${EXPECTED_GPU_UUID}" "${EXPECTED_STRESS_DECISION_SHA}" "${EXPECTED_STRESS_AGGREGATE_SHA}" "${V11_LIBRARY}" "${V12_LIBRARY}" "${V11_SOURCE}" "${V12_SOURCE}" <<'PY'
import hashlib,json,math,sys
from pathlib import Path
v11p,v12p,pre9p,pre11p,decp,aggp,out=map(Path,sys.argv[1:8]); expected_uuid,expected_decision,expected_aggregate=sys.argv[8:11]; v11lib,v12lib,v11src,v12src=map(Path,sys.argv[11:15])
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if digest(decp)!=expected_decision or digest(aggp)!=expected_aggregate: raise SystemExit('job12985 input hash drift')
v11=json.loads(v11p.read_text()); v12=json.loads(v12p.read_text()); decision=json.loads(decp.read_text())
def target_record(path):
    records=[]
    for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
        try: value=json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(value,dict) and value.get('schema')=='c2-native-v12-vs-v11-ncu-target-v1': records.append(value)
    if len(records)!=1: raise SystemExit(f'{path}: expected one preflight target JSON record, got {len(records)}')
    return records[0]
pre9=target_record(pre9p); pre11=target_record(pre11p)
for label,p in (('v11',v11),('v12',v12)):
    if p.get('schema')!='c2-native-v12-vs-v11-ncu-counters-v1' or p.get('all_gates_pass') is not True or p.get('matching_kernel_rows')!=1 or p.get('timing_valid_for_benchmark') is not False: raise SystemExit(f'{label}: invalid summary boundary')
if set(v11['metrics'])!=set(v12['metrics']): raise SystemExit('metric set differs')
for key in v11['metrics']:
    if v11['metrics'][key]['unit']!=v12['metrics'][key]['unit'] or not all(math.isfinite(float(p['metrics'][key]['value'])) and float(p['metrics'][key]['value'])>=0 for p in (v11,v12)): raise SystemExit(f'bad metric {key}')
targets=[v11['target'],v12['target']]
if targets[0]['input_tensor_sha256']!=targets[1]['input_tensor_sha256'] or targets[0]['input_manifest_sha256']!=targets[1]['input_manifest_sha256'] or targets[0]['contract']!=targets[1]['contract'] or targets[0]['base_harness_sha256']!=targets[1]['base_harness_sha256']: raise SystemExit('NCU fixture/harness differs')
if [x['device'].get('gpu_uuid') for x in targets] != [expected_uuid,expected_uuid] or [x['device'].get('gpu_uuid') for x in (pre9,pre11)] != [expected_uuid,expected_uuid]: raise SystemExit('not same allocation/UUID')
if targets[0]['input_manifest_sha256']!=pre9['input_manifest_sha256'] or targets[1]['input_manifest_sha256']!=pre11['input_manifest_sha256']: raise SystemExit('NCU target/preflight fixture drift')
if decision.get('accepted') is not True or decision.get('issues')!=[] or decision.get('aggregate_sha256')!=expected_aggregate: raise SystemExit('job12985 acceptance linkage invalid')
ratio={}; delta={}
for k in sorted(v11['metrics']):
    old,new=float(v11['metrics'][k]['value']),float(v12['metrics'][k]['value']); ratio[k]=None if old==0 else new/old; delta[k]=new-old
lower=lambda key:ratio[key] is not None and ratio[key]<1
payload={'schema':'c2-native-v12-v11-ncu-mechanism-comparison-v1','all_collection_and_comparison_gates_pass':True,'reference':{'version':'v11','aot_job':12825,'summary':str(v11p.resolve()),'summary_sha256':digest(v11p),'plugin':str(v11lib.resolve()),'plugin_sha256':targets[0]['operator_library_sha256'],'source':str(v11src.resolve())},'candidate':{'version':'v12','aot_job':12983,'summary':str(v12p.resolve()),'summary_sha256':digest(v12p),'plugin':str(v12lib.resolve()),'plugin_sha256':targets[1]['operator_library_sha256'],'source':str(v12src.resolve())},'same_collection_contract':{'seed':20260829,'same_metric_set':True,'same_input_manifest_sha256':targets[0]['input_manifest_sha256'],'same_harness_sha256':targets[0]['base_harness_sha256'],'same_physical_gpu_uuid':expected_uuid,'separate_python_processes':True,'reason':'each DSO registers the same stable _C schema/CUDA implementation','single_native_kernel_action_each':True,'kernel_filter':'regex:.*native_c2_msa_decode_kernel.*'},'resources':{'v11_aot':{'registers':127,'stack':0,'local':0,'shared_bytes':30880},'v12_aot':{'registers':127,'stack':0,'local':0,'shared_bytes':31136},'ncu_launch_v11':v11['launch_metadata'],'ncu_launch_v12':v12['launch_metadata']},'v12_minus_v11':delta,'v12_over_v11':ratio,'mechanism_observations':{'hypothesis':'v12 changes only the Q shared-tile stride from 128 to 136 BF16 elements; counters test shared-memory conflict and stall consequences, not a new performance criterion.','shared_bytes_increased_exactly':v11['launch_metadata']['launch__shared_mem_per_block']==30880 and v12['launch_metadata']['launch__shared_mem_per_block']==31136,'registers_per_thread_unchanged':v11['launch_metadata']['launch__registers_per_thread']==127 and v12['launch_metadata']['launch__registers_per_thread']==127,'shared_load_wavefronts_decreased':lower('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'),'shared_store_wavefronts_decreased':lower('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum'),'shared_load_bank_conflicts_decreased':lower('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'),'shared_store_bank_conflicts_decreased':lower('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum'),'long_scoreboard_stall_decreased':lower('smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct'),'active_warps_ratio':ratio['smsp__warps_active.avg.per_cycle_active'],'occupancy_ratio':ratio['sm__warps_active.avg.pct_of_peak_sustained_active'],'tensor_instructions_ratio':ratio['smsp__inst_executed_pipe_tensor.sum'],'tensor_cycles_active_ratio':ratio['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active'],'elapsed_cycles_ratio':ratio['smsp__cycles_elapsed.sum']},'explained_performance_evidence':{'job':12985,'accepted_point_improvement_fraction':0.036491182231437325,'accepted_bootstrap_95pct_lower_bound_fraction':0.034053072960836483,'decision_sha256':expected_decision,'aggregate_sha256':expected_aggregate},'decision_boundary':'NCU counters and NCU duration are mechanism evidence only. They do not promote, reject, or recompute the +3.6491182231437325% job12985 ABBA decision.'}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); print(json.dumps(payload,sort_keys=True))
PY
sha256sum "${V11_PREFLIGHT}" "${V12_PREFLIGHT}" "${V11_REPORT}" "${V12_REPORT}" "${V11_RAW}" "${V12_RAW}" "${V11_TARGET}" "${V12_TARGET}" "${V11_SUMMARY}" "${V12_SUMMARY}" "${MECHANISM}" "${TARGET_RUNNER}" "${METRIC_CATALOG}" "${ARTIFACT_DIR}/inputs-job${SLURM_JOB_ID}.sha256" > "${ARTIFACT_DIR}/outputs-before-post-job${SLURM_JOB_ID}.sha256"

