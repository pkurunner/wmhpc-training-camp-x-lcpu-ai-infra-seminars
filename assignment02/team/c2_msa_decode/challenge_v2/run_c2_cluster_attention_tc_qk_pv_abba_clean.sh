#!/usr/bin/env bash
# Slurm-only, double-authorized B=1/C=2 WMMA-QK control versus WMMA-QK+PV
# candidate audit.  This script never submits; its parent supplies one empty
# B300 allocation.  All compilation, static instruction evidence, correctness,
# ABBA timing, provenance, and JSON validation remain on that allocation.
set -Eeuo pipefail

if [[ "${C2_CLUSTER_ATTENTION_TC_QK_PV_ABBA_AUTHORIZED:-}" != 1 || "${1:-}" != --authorized-by-parent ]]; then
  printf '%s\n' 'Refusing TC-QK+PV AB/BA benchmark without both authorization tokens.' >&2; exit 64
fi
[[ -n "${SLURM_JOB_ID:-}" ]] || { printf '%s\n' 'Refusing TC-QK+PV AB/BA benchmark outside Slurm.' >&2; exit 64; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
c2_root="$(cd "${script_dir}/.." && pwd)"; assignment_root="$(cd "${c2_root}/../.." && pwd)"
python_bin="${C2_PYTHON_BIN:-${assignment_root}/.venv/bin/python}"
nvcc_candidate="${C2_NVCC_BIN:-$(command -v nvcc || true)}"; nvcc_bin=""
[[ -n "${nvcc_candidate}" ]] && nvcc_bin="$(readlink -f "${nvcc_candidate}")"
cuobjdump_bin="$(dirname "${nvcc_bin:-/missing/nvcc}")/cuobjdump"
source_path="${script_dir}/c2_cluster_attention_tc_qk_pv_abba.cu"
tc_qk_import_path="${script_dir}/c2_cluster_attention_tc_qk_abba.cu"
warp_import_path="${script_dir}/c2_cluster_attention_warp_producer_abba.cu"
scalar_import_path="${script_dir}/c2_cluster_attention_mbarrier_smoke.cu"
runner_path="${script_dir}/run_c2_cluster_attention_tc_qk_pv_abba_clean.sh"
tc_qk_sha_expected=523347312f07345487a2591d6e52ef231c94ba8b6fe00e36e1fc16a02bb53431
warp_sha_expected=24938b464a5b179a7c0e6f2450dd72b231635c73e7b46ea6c5a3fac85357444a
scalar_sha_expected=6d69cecbda9baf93b19bfba3162a4c33e42298bee43a9235de98e2acd369372f
out_dir="${C2_CLUSTER_ATTENTION_TC_QK_PV_ABBA_OUT_DIR:-${c2_root}/experiment_logs/c2_cluster_attention_tc_qk_pv_abba}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)_job${SLURM_JOB_ID}"; mkdir -p "${out_dir}"
audit_log="${out_dir}/c2_tc_qk_pv_abba_clean_${stamp}.log"; compile_log="${out_dir}/c2_tc_qk_pv_abba_compile_${stamp}.log"
run_log="${out_dir}/c2_tc_qk_pv_abba_run_${stamp}.log"; raw_json="${out_dir}/c2_tc_qk_pv_abba_raw_${stamp}.json"
final_json="${out_dir}/c2_tc_qk_pv_abba_clean_${stamp}.json"; instruction_json="${out_dir}/c2_tc_qk_pv_abba_instruction_${stamp}.json"
binary_path="${out_dir}/c2_tc_qk_pv_abba_${stamp}"; ptx_path="${out_dir}/c2_tc_qk_pv_abba_${stamp}.ptx"; sass_path="${out_dir}/c2_tc_qk_pv_abba_${stamp}.sass"
exec > >(tee -a "${audit_log}") 2>&1

compute_apps() { nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
require_b300_empty() {
  local label="$1" rows name cap mem apps
  rows="$(nvidia-smi --query-gpu=name,compute_cap,memory.used --format=csv,noheader,nounits)" || return 74
  [[ -n "${rows//[[:space:]]/}" ]] || return 74
  while IFS=',' read -r name cap mem; do
    name="${name## }"; cap="${cap//[[:space:]]/}"; mem="${mem//[[:space:]]/}"
    [[ "${name}" == *B300* && "${cap}" == 10.3 && "${mem}" =~ ^0$ ]] || { printf 'ABORT %s: expected empty B300, got %q/%q/%q\n' "${label}" "${name}" "${cap}" "${mem}" >&2; return 75; }
  done <<<"${rows}"
  apps="$(compute_apps)" || return 74; [[ -z "${apps}" ]] || { printf 'ABORT %s: compute apps: %s\n' "${label}" "${apps}" >&2; return 73; }
}
gpu_uuid() { local -a a=(); mapfile -t a < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'); [[ ${#a[@]} == 1 && ${a[0]} == GPU-* ]] || return 75; printf '%s\n' "${a[0]}"; }
snapshot() { printf '\n===== %s UTC %s (Slurm %s) =====\n' "$1" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}"; nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader,nounits || true; printf '%s\n' '-- compute apps --'; compute_apps || true; sha256sum "${source_path}" "${tc_qk_import_path}" "${warp_import_path}" "${scalar_import_path}" "${runner_path}" || true; }
post_done=0
on_exit() { local rc=$? post_rc=0; trap - EXIT; set +e; [[ ${post_done} == 1 ]] || { snapshot POST_ON_EXIT; require_b300_empty POST_ON_EXIT || post_rc=$?; gpu_uuid >/dev/null || post_rc=$?; }; [[ ${rc} != 0 || ${post_rc} == 0 ]] || rc=${post_rc}; printf '\n===== FINAL_RC=%s =====\n' "${rc}"; exit "${rc}"; }
trap on_exit EXIT

[[ -x "${python_bin}" && -x "${nvcc_bin}" && -x "${cuobjdump_bin}" ]] || { printf '%s\n' 'Missing Python, nvcc, or cuobjdump.' >&2; exit 65; }
[[ -f "${source_path}" && -f "${tc_qk_import_path}" && -f "${warp_import_path}" && -f "${scalar_import_path}" ]] || { printf '%s\n' 'Missing source/import.' >&2; exit 65; }
command -v timeout >/dev/null || { printf '%s\n' 'Missing timeout.' >&2; exit 65; }
cuda_root="$(cd "$(dirname "${nvcc_bin}")/.." && pwd)"; cuda_include_dir="${cuda_root}/targets/x86_64-linux/include"; cuda_cccl_dir="${cuda_include_dir}/cccl"
[[ -f "${cuda_include_dir}/cuda_runtime.h" && -f "${cuda_cccl_dir}/cuda/std/type_traits" ]] || { printf '%s\n' 'Missing CUDA headers.' >&2; exit 65; }
[[ -x "${cuda_root}/nvvm/bin/cicc" ]] && export PATH="${cuda_root}/nvvm/bin:${PATH}"
export PATH="$(dirname "${nvcc_bin}"):${PATH}" PYTHONOPTIMIZE=0
[[ "$("${python_bin}" -c 'print(int(__debug__))')" == 1 ]] || { printf '%s\n' 'Python assertions disabled.' >&2; exit 65; }
grep -q sm_103a <<<"$("${nvcc_bin}" --help)" || { printf '%s\n' 'nvcc lacks sm_103a.' >&2; exit 65; }

source_sha_pre="$(sha256sum "${source_path}"|awk '{print $1}')"; tc_qk_sha_pre="$(sha256sum "${tc_qk_import_path}"|awk '{print $1}')"; warp_sha_pre="$(sha256sum "${warp_import_path}"|awk '{print $1}')"; scalar_sha_pre="$(sha256sum "${scalar_import_path}"|awk '{print $1}')"; runner_sha_pre="$(sha256sum "${runner_path}"|awk '{print $1}')"
[[ ${tc_qk_sha_pre} == ${tc_qk_sha_expected} && ${warp_sha_pre} == ${warp_sha_expected} && ${scalar_sha_pre} == ${scalar_sha_expected} ]] || { printf '%s\n' 'Audited import SHA mismatch.' >&2; exit 66; }
nvcc_version="$("${nvcc_bin}" --version)"; compile_flags="-std=c++17 -O3 -arch=sm_103a -I${cuda_include_dir} -I${cuda_cccl_dir}"
snapshot PRE; require_b300_empty PRE; uuid="$(gpu_uuid)"

"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a -I"${cuda_include_dir}" -I"${cuda_cccl_dir}" "${source_path}" -o "${binary_path}" >"${compile_log}" 2>&1
"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a -I"${cuda_include_dir}" -I"${cuda_cccl_dir}" -ptx "${source_path}" -o "${ptx_path}" >>"${compile_log}" 2>&1
"${cuobjdump_bin}" --dump-sass "${binary_path}" >"${sass_path}"

"${python_bin}" - "${ptx_path}" "${sass_path}" "${instruction_json}" <<'PY'
import json, re, sys
from pathlib import Path
ptx, sass, out = map(Path, sys.argv[1:])
def extract(lines, mark, needle):
    hit=[i for i,l in enumerate(lines) if mark in l and needle in l]
    assert len(hit)==1,(needle,hit)
    i=hit[0]; j=next((k for k in range(i+1,len(lines)) if mark in lines[k]),len(lines)); return ''.join(lines[i:j])
p=ptx.read_text().splitlines(keepends=True); s=sass.read_text().splitlines(keepends=True)
def one(name):
    a=extract(p,'.entry ',name); b=extract(s,'Function : ',name)
    r={'ptx_mbarrier_init':a.count('mbarrier.init.shared.b64'),'ptx_mbarrier_release_arrive':a.count('mbarrier.arrive.release.cluster.shared::cluster.b64'),'ptx_mbarrier_acquire_wait':a.count('mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64'),'ptx_cluster_arrive':a.count('barrier.cluster.arrive'),'ptx_cluster_wait':a.count('barrier.cluster.wait'),'sass_mbarrier_init':b.count('SYNCS.EXCH.64'),'sass_mbarrier_release_arrive':b.count('SYNCS.ARRIVE.TRANS64.RED.A1T0'),'sass_mbarrier_acquire_wait':b.count('SYNCS.PHASECHK.TRANS64.TRYWAIT'),'sass_cluster_arrive':b.count('UCGABAR_ARV'),'sass_cluster_wait':b.count('UCGABAR_WAIT'),'ptx_bf16_wmma_mma':len(re.findall(r'\bwmma\.mma\.sync(?:\.aligned)?\.[^\n]*\.bf16\.bf16(?:\.|\s)',a)),'sass_hmma_16816_f32_bf16':b.count('HMMA.16816.F32.BF16')}
    for k in ('ptx_mbarrier_init','ptx_mbarrier_release_arrive','ptx_mbarrier_acquire_wait','sass_mbarrier_init','sass_mbarrier_release_arrive','sass_mbarrier_acquire_wait'): assert r[k]==1,(k,r)
    for k in ('ptx_cluster_arrive','ptx_cluster_wait','sass_cluster_arrive','sass_cluster_wait'): assert r[k]==2,(k,r)
    return r
control=one('cluster_attention_mbarrier_warp_producer_tc_qk_kernel'); candidate=one('cluster_attention_mbarrier_warp_producer_tc_qk_pv_kernel')
assert control['ptx_bf16_wmma_mma']>0 and candidate['ptx_bf16_wmma_mma']>control['ptx_bf16_wmma_mma'],(control,candidate)
assert control['sass_hmma_16816_f32_bf16']>0 and candidate['sass_hmma_16816_f32_bf16']>control['sass_hmma_16816_f32_bf16'],(control,candidate)
Path(out).write_text(json.dumps({'schema':'c2-tc-qk-pv-instruction-v1','status':'pass','control':control,'candidate':candidate},sort_keys=True,indent=2)+'\n')
PY

set +e; timeout --preserve-status --kill-after=5s 240s "${binary_path}" >"${raw_json}" 2>"${run_log}"; rc=$?; set -e
[[ ${rc} == 0 ]] || { printf 'TC-QK+PV AB/BA failed/timeout rc=%s\n' "${rc}" >&2; exit "${rc}"; }
for p in "${source_path}" "${tc_qk_import_path}" "${warp_import_path}" "${scalar_import_path}" "${runner_path}"; do [[ -n "$(sha256sum "${p}")" ]] || exit 66; done
[[ ${source_sha_pre} == "$(sha256sum "${source_path}"|awk '{print $1}')" && ${tc_qk_sha_pre} == "$(sha256sum "${tc_qk_import_path}"|awk '{print $1}')" && ${warp_sha_pre} == "$(sha256sum "${warp_import_path}"|awk '{print $1}')" && ${scalar_sha_pre} == "$(sha256sum "${scalar_import_path}"|awk '{print $1}')" && ${runner_sha_pre} == "$(sha256sum "${runner_path}"|awk '{print $1}')" ]] || { printf '%s\n' 'Source/import/runner changed during audit.' >&2; exit 66; }

"${python_bin}" - "${raw_json}" "${instruction_json}" "${final_json}" "${source_sha_pre}" "${tc_qk_sha_pre}" "${warp_sha_pre}" "${scalar_sha_pre}" "${runner_sha_pre}" "$(sha256sum "${binary_path}"|awk '{print $1}')" "$(sha256sum "${ptx_path}"|awk '{print $1}')" "$(sha256sum "${sass_path}"|awk '{print $1}')" "${nvcc_version}" "${compile_flags}" "${SLURM_JOB_ID}" "${uuid}" <<'PY'
import copy, json, math, struct, sys
from pathlib import Path
assert sys.flags.optimize == 0
raw, inst, out, *meta=sys.argv[1:]
p=json.loads(Path(raw).read_text()); i=json.loads(Path(inst).read_text())
def finite_number(value, context):
    # `bool` subclasses `int`; accept exactly JSON int/float evidence only.
    assert type(value) in (int,float), ('non-numeric',context,value)
    value=float(value); assert math.isfinite(value), ('non-finite',context,value)
    return value
def strict_int(value, context):
    assert type(value) is int, ('non-integer',context,value)
    return value
def require_bool(value, expected, context):
    assert type(value) is bool, ('non-bool',context,value)
    if expected is not None: assert value is expected, ('wrong-bool',context,value,expected)
def f32(value): return struct.unpack('<f',struct.pack('<f',finite_number(value,'f32')))[0]
def summary(values):
    values=sorted(f32(v) for v in values); assert values and all(v>0 for v in values); n=len(values)
    # Match summarize_us(): event samples are float32, while the even median
    # promotes each sorted float to double before the addition/division.
    median=float(values[n//2]) if n%2 else (float(values[n//2-1])+float(values[n//2]))/2.0
    return {'p10_us':values[max(0,math.ceil(.1*n)-1)],'median_us':median,'p90_us':values[min(n-1,math.ceil(.9*n)-1)]}
def assert_cpp9(actual, expected, context):
    # Raw samples and C++ Statistics travel through std::setprecision(9) with
    # defaultfloat.  Compare their canonical nine-significant-digit decimal
    # transmission, rather than incorrectly re-rounding a C++ double statistic
    # to float32.
    actual=finite_number(actual,context); assert format(actual,'.9g')==format(float(expected),'.9g'), (context,actual,expected,format(actual,'.9g'),format(float(expected),'.9g'))
def validate(payload, instruction):
    assert set(payload)=={'schema','status','boundary','timing_seed','shape','cluster_layout','producer_contract','synchronization','dtype_contract','environment','resource_model','correctness','post_timing_correctness','timing'}
    assert payload['schema']=='c2-cluster-attention-tc-qk-pv-abba-v1' and payload['status']=='pass' and isinstance(payload['boundary'],str)
    assert strict_int(payload['timing_seed'],'timing_seed')==2026
    shape_expected={'B':1,'Hkv':4,'Hq':64,'G':16,'D':128,'page_size':128,'selected_pages':16,'logical_pages':32}
    assert set(payload['shape'])==set(shape_expected)
    for key, expected in shape_expected.items(): assert strict_int(payload['shape'][key],('shape',key))==expected
    layout_expected={'num_ctas':4,'clusters':4,'selected_pages_per_producer':8,'threads_per_block':256}
    assert set(payload['cluster_layout'])==set(layout_expected)
    for key, expected in layout_expected.items(): assert strict_int(payload['cluster_layout'][key],('layout',key))==expected
    contract=payload['producer_contract']; expected_contract={'control','candidate','changed_field','same_wmma_qk','same_remote_dsm_mbarrier_protocol','same_rank2_merge_output_abi_and_lifetime_sync','same_launch_shape','same_real_selected_causal_attention','persistent_device_buffers_outside_timing','caller_owned_independent_outputs','single_kernel_launch_per_cuda_event_sample','ABBA_interleaved','initialization_copies_and_oracle_outside_timing','post_timing_fresh_sentinel_reset_and_relaunch','candidate_extra_shared_and_cta_barriers_included','no_global_score_or_weight_workspace','no_second_kernel','no_scalar_residual_correction','cross_arm_bitwise'}
    assert set(contract)==expected_contract and all(isinstance(contract[k],str) for k in ('control','candidate','changed_field','cross_arm_bitwise'))
    for key in expected_contract-{'control','candidate','changed_field','cross_arm_bitwise'}: require_bool(contract[key],True,('contract',key))
    sync=payload['synchronization']; assert set(sync)=={'mbarrier_expected_arrivals','mbarrier_wait_parity','mbarrier_max_polls','producer_ready','cluster_sync','candidate_cta_barriers'}
    assert strict_int(sync['mbarrier_expected_arrivals'],'sync.arrivals')==2 and strict_int(sync['mbarrier_wait_parity'],'sync.parity')==0 and strict_int(sync['mbarrier_max_polls'],'sync.polls')==1<<24
    assert all(isinstance(sync[k],str) for k in ('producer_ready','cluster_sync','candidate_cta_barriers'))
    dtype=payload['dtype_contract']; assert set(dtype)=={'producer_partial','caller_output','qk','pv','online_state','oracle_accumulator','oracle','tolerance'}
    assert dtype['producer_partial']==dtype['caller_output']=='bfloat16' and dtype['oracle_accumulator']=='float64' and all(isinstance(dtype[k],str) for k in ('qk','pv','online_state','oracle'))
    # C++ emits these float constants through setprecision(9), so accept only
    # their bounded decimal transport error—not a changed numerical tolerance.
    assert set(dtype['tolerance'])=={'rtol','atol'} and math.isclose(finite_number(dtype['tolerance']['rtol'],'rtol'),.005,rel_tol=0,abs_tol=1e-9) and math.isclose(finite_number(dtype['tolerance']['atol'],'atol'),.0005,rel_tol=0,abs_tol=1e-9)
    env=payload['environment']; assert set(env)=={'device','capability','cuda_runtime','cuda_driver','cluster_launch_supported'} and isinstance(env['device'],str) and 'B300' in env['device']
    assert isinstance(env['capability'],list) and [strict_int(v,('capability',n)) for n,v in enumerate(env['capability'])]==[10,3]
    strict_int(env['cuda_runtime'],'cuda_runtime'); strict_int(env['cuda_driver'],'cuda_driver'); require_bool(env['cluster_launch_supported'],True,'cluster_launch')
    resources=payload['resource_model']; assert set(resources)=={'control','candidate'}
    for arm in ('control','candidate'):
        assert set(resources[arm])=={'static_shared_bytes','num_regs','local_bytes'}
        assert strict_int(resources[arm]['static_shared_bytes'],(arm,'shared'))>0 and strict_int(resources[arm]['num_regs'],(arm,'regs'))>0 and strict_int(resources[arm]['local_bytes'],(arm,'local'))>=0
    assert strict_int(resources['candidate']['local_bytes'],'candidate local')==0
    assert set(instruction)=={'schema','status','control','candidate'} and instruction['schema']=='c2-tc-qk-pv-instruction-v1' and instruction['status']=='pass'
    instruction_keys={'ptx_mbarrier_init','ptx_mbarrier_release_arrive','ptx_mbarrier_acquire_wait','ptx_cluster_arrive','ptx_cluster_wait','sass_mbarrier_init','sass_mbarrier_release_arrive','sass_mbarrier_acquire_wait','sass_cluster_arrive','sass_cluster_wait','ptx_bf16_wmma_mma','sass_hmma_16816_f32_bf16'}
    for arm in ('control','candidate'):
        row=instruction[arm]; assert set(row)==instruction_keys
        for k in ('ptx_mbarrier_init','ptx_mbarrier_release_arrive','ptx_mbarrier_acquire_wait','sass_mbarrier_init','sass_mbarrier_release_arrive','sass_mbarrier_acquire_wait'): assert strict_int(row[k],(arm,k))==1
        for k in ('ptx_cluster_arrive','ptx_cluster_wait','sass_cluster_arrive','sass_cluster_wait'): assert strict_int(row[k],(arm,k))==2
        assert strict_int(row['ptx_bf16_wmma_mma'],(arm,'ptx mma'))>0 and strict_int(row['sass_hmma_16816_f32_bf16'],(arm,'sass hmma'))>0
    assert strict_int(instruction['candidate']['ptx_bf16_wmma_mma'],'candidate ptx mma')>strict_int(instruction['control']['ptx_bf16_wmma_mma'],'control ptx mma')
    assert strict_int(instruction['candidate']['sass_hmma_16816_f32_bf16'],'candidate sass hmma')>strict_int(instruction['control']['sass_hmma_16816_f32_bf16'],'control sass hmma')
    assert isinstance(payload['correctness'],list) and len(payload['correctness'])==2 and {strict_int(x['seed'],'seed') for x in payload['correctness']}=={17,2026}
    assert strict_int(payload['post_timing_correctness']['seed'],'post seed')==2026
    rows=[*payload['correctness'],payload['post_timing_correctness']]
    for n,row in enumerate(rows):
        expected={'seed','hierarchy_valid','control','candidate','cross_arm_diagnostic'}
        if n<2: expected|={'sequence_length','adversarial_unselected_visible_pages','adversarial_masked_tokens'}
        assert set(row)==expected and strict_int(row['seed'],('row',n,'seed')) in {17,2026}; require_bool(row['hierarchy_valid'],True,('row',n,'hierarchy'))
        if n<2:
            assert strict_int(row['sequence_length'],('row',n,'length'))>0 and strict_int(row['adversarial_unselected_visible_pages'],('row',n,'poison'))>0 and strict_int(row['adversarial_masked_tokens'],('row',n,'masked'))==4*127
        for arm in ('control','candidate'):
            a=row[arm]; assert set(a)=={'max_abs','max_rel','oracle_finite','finite','sentinel_clean','allclose'}
            finite_number(a['max_abs'],(n,arm,'abs')); finite_number(a['max_rel'],(n,arm,'rel'))
            for k in ('oracle_finite','finite','sentinel_clean','allclose'): require_bool(a[k],True,(n,arm,k))
        cross=row['cross_arm_diagnostic']; assert set(cross)=={'max_abs','max_rel','bfloat16_bitwise_equal'}; finite_number(cross['max_abs'],(n,'cross abs')); finite_number(cross['max_rel'],(n,'cross rel')); require_bool(cross['bfloat16_bitwise_equal'],None,(n,'cross bits'))
    timing=payload['timing']; assert set(timing)=={'protocol','warmup_each','abba_pairs','samples_per_arm','raw_samples_us','control','candidate','speedup_control_over_candidate','speedup_by_partition','promotion_gate'} and timing['protocol']=='warmup_each_then_101_control_candidate_candidate_control_ABBA_pairs'
    assert strict_int(timing['warmup_each'],'warmup')==20 and strict_int(timing['abba_pairs'],'pairs')==101 and strict_int(timing['samples_per_arm'],'samples')==202
    assert set(timing['raw_samples_us'])=={'control','candidate'}; computed={}
    for arm in ('control','candidate'):
        assert set(timing['raw_samples_us'][arm])=={'AB','BA'} and set(timing[arm])=={'all','when_launch_order_is_AB','when_launch_order_is_BA'}; computed[arm]={}
        ab_raw=timing['raw_samples_us'][arm]['AB']; ba_raw=timing['raw_samples_us'][arm]['BA']; assert isinstance(ab_raw,list) and isinstance(ba_raw,list) and len(ab_raw)==len(ba_raw)==101
        for part,values in (('all',[*ab_raw,*ba_raw]),('when_launch_order_is_AB',ab_raw),('when_launch_order_is_BA',ba_raw)):
            expected=summary(values); actual=timing[arm][part]; assert set(actual)=={'p10_us','median_us','p90_us'}
            for stat,value in expected.items(): assert_cpp9(actual[stat],value,(arm,part,stat))
            computed[arm][part]=expected
    speed=computed['control']['all']['median_us']/computed['candidate']['all']['median_us']; ab=computed['control']['when_launch_order_is_AB']['median_us']/computed['candidate']['when_launch_order_is_AB']['median_us']; ba=computed['control']['when_launch_order_is_BA']['median_us']/computed['candidate']['when_launch_order_is_BA']['median_us']
    assert_cpp9(timing['speedup_control_over_candidate'],speed,'speed'); assert set(timing['speedup_by_partition'])=={'AB','BA'}; assert_cpp9(timing['speedup_by_partition']['AB'],ab,'ab'); assert_cpp9(timing['speedup_by_partition']['BA'],ba,'ba')
    gate=timing['promotion_gate']; assert set(gate)=={'combined_control_over_candidate_at_least_1_10','AB_control_over_candidate_greater_than_1_05','BA_control_over_candidate_greater_than_1_05','candidate_local_size_bytes_zero','all_correct','promoted'}
    require_bool(gate['combined_control_over_candidate_at_least_1_10'],speed>=1.10,'gate combined'); require_bool(gate['AB_control_over_candidate_greater_than_1_05'],ab>1.05,'gate AB'); require_bool(gate['BA_control_over_candidate_greater_than_1_05'],ba>1.05,'gate BA'); require_bool(gate['candidate_local_size_bytes_zero'],True,'gate local'); require_bool(gate['all_correct'],True,'gate correct'); require_bool(gate['promoted'],speed>=1.10 and ab>1.05 and ba>1.05,'gate promoted')
validate(p,i)
# Guard the validator itself with representative raw/type adversarial mutations.
for mutate in (lambda q:q['timing']['raw_samples_us']['control']['AB'].__setitem__(0,True), lambda q:q['timing']['raw_samples_us']['control']['AB'].__setitem__(slice(None),[v*1.25 for v in q['timing']['raw_samples_us']['control']['AB']]), lambda q:q['timing']['promotion_gate'].__setitem__('promoted',1), lambda q:q['shape'].__setitem__('B',1.0), lambda q:q['dtype_contract']['tolerance'].__setitem__('rtol',.00500001)):
    bad=copy.deepcopy(p); mutate(bad)
    try: validate(bad,i)
    except AssertionError: pass
    else: raise AssertionError('negative JSON/raw tamper accepted')
record={'schema':'c2-cluster-attention-tc-qk-pv-clean-v1','status':'pass','slurm_job_id':int(meta[-2]),'gpu_uuid':meta[-1],'raw':p,'instruction_evidence':i,'artifacts':{'source_sha256':meta[0],'tc_qk_control_sha256':meta[1],'warp_import_sha256':meta[2],'scalar_import_sha256':meta[3],'runner_sha256':meta[4],'binary_sha256':meta[5],'ptx_sha256':meta[6],'sass_sha256':meta[7],'nvcc_version':meta[8],'compile_flags':meta[9]}}
Path(out).write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
PY
snapshot POST; require_b300_empty POST; [[ "$(gpu_uuid)" == "${uuid}" ]] || { printf '%s\n' 'GPU identity changed.' >&2; exit 75; }; post_done=1
printf '%s\n' "${final_json}"
