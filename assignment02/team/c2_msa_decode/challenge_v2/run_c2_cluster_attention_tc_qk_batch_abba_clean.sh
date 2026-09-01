#!/usr/bin/env bash
# Slurm-only runner for batched native C=2 warp-QK versus WMMA-QK AB/BA.
# It does not submit jobs; a parent must provide both authorization tokens.
set -Eeuo pipefail

if [[ "${C2_CLUSTER_ATTENTION_TC_QK_BATCH_ABBA_AUTHORIZED:-}" != "1" || "${1:-}" != "--authorized-by-parent" ]]; then
  printf '%s\n' 'Refusing batched native C=2 AB/BA experiment without both authorization tokens.' >&2; exit 64
fi
[[ -n "${SLURM_JOB_ID:-}" ]] || { printf '%s\n' 'Refusing batched native C=2 experiment outside Slurm.' >&2; exit 64; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
c2_root="$(cd "${script_dir}/.." && pwd)"
assignment_root="$(cd "${c2_root}/../.." && pwd)"
python_bin="${C2_PYTHON_BIN:-${assignment_root}/.venv/bin/python}"
nvcc_candidate="${C2_NVCC_BIN:-$(command -v /usr/local/cuda/bin/nvcc || command -v nvcc || true)}"
nvcc_bin=""; [[ -n "${nvcc_candidate}" ]] && nvcc_bin="$(readlink -f "${nvcc_candidate}")"
cuobjdump_bin="$(dirname "${nvcc_bin:-/missing/nvcc}")/cuobjdump"
source_path="${script_dir}/c2_cluster_attention_tc_qk_batch_abba.cu"
warp_reference_path="${script_dir}/c2_cluster_attention_warp_producer_abba.cu"
tc_qk_reference_path="${script_dir}/c2_cluster_attention_tc_qk_abba.cu"
scalar_import_path="${script_dir}/c2_cluster_attention_mbarrier_smoke.cu"
runner_path="${script_dir}/run_c2_cluster_attention_tc_qk_batch_abba_clean.sh"
audited_warp_sha256='24938b464a5b179a7c0e6f2450dd72b231635c73e7b46ea6c5a3fac85357444a'
audited_tc_qk_sha256='523347312f07345487a2591d6e52ef231c94ba8b6fe00e36e1fc16a02bb53431'
audited_scalar_sha256='6d69cecbda9baf93b19bfba3162a4c33e42298bee43a9235de98e2acd369372f'
out_dir="${C2_CLUSTER_ATTENTION_TC_QK_BATCH_ABBA_OUT_DIR:-${c2_root}/experiment_logs/c2_cluster_attention_tc_qk_batch_abba}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)_job${SLURM_JOB_ID}"
audit_log="${out_dir}/c2_cluster_attention_tc_qk_batch_abba_clean_${stamp}.log"
compile_log="${out_dir}/c2_cluster_attention_tc_qk_batch_abba_compile_${stamp}.log"
run_log="${out_dir}/c2_cluster_attention_tc_qk_batch_abba_run_${stamp}.log"
raw_json="${out_dir}/c2_cluster_attention_tc_qk_batch_abba_raw_${stamp}.json"
final_json="${out_dir}/c2_cluster_attention_tc_qk_batch_abba_clean_${stamp}.json"
binary_path="${out_dir}/c2_cluster_attention_tc_qk_batch_abba_${stamp}"
ptx_path="${out_dir}/c2_cluster_attention_tc_qk_batch_abba_${stamp}.ptx"
sass_path="${out_dir}/c2_cluster_attention_tc_qk_batch_abba_${stamp}.sass"
control_ptx_path="${out_dir}/c2_cluster_attention_tc_qk_batch_abba_warp_control_${stamp}.ptx"
candidate_ptx_path="${out_dir}/c2_cluster_attention_tc_qk_batch_abba_tc_qk_candidate_${stamp}.ptx"
control_sass_path="${out_dir}/c2_cluster_attention_tc_qk_batch_abba_warp_control_${stamp}.sass"
candidate_sass_path="${out_dir}/c2_cluster_attention_tc_qk_batch_abba_tc_qk_candidate_${stamp}.sass"
mkdir -p "${out_dir}"
exec > >(tee -a "${audit_log}") 2>&1

apps() { nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
one_uuid() { mapfile -t a < <(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'); [[ "${#a[@]}" -eq 1 && "${a[0]}" == GPU-* ]] || return 75; printf '%s\n' "${a[0]}"; }
require_b300() {
  local rows name capability
  rows="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)" || return 74
  [[ -n "${rows//[[:space:]]/}" ]] || return 74
  while IFS=',' read -r name capability; do
    name="${name#"${name%%[![:space:]]*}"}"; name="${name%"${name##*[![:space:]]}"}"; capability="${capability//[[:space:]]/}"
    [[ "${name}" == *B300* && "${capability}" == 10.3 ]] || { printf 'Expected B300 CC10.3, got %q %q\n' "${name}" "${capability}" >&2; return 75; }
  done <<<"${rows}"
}
require_empty() {
  local running rows memory
  running="$(apps)" || return 74; [[ -z "${running}" ]] || { printf 'Visible compute apps: %s\n' "${running}" >&2; return 73; }
  rows="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" || return 74
  [[ -n "${rows//[[:space:]]/}" ]] || { printf '%s\n' 'GPU memory query returned no rows.' >&2; return 74; }
  while IFS= read -r memory; do
    memory="${memory//[[:space:]]/}"
    [[ "${memory}" =~ ^[0-9]+$ && "${memory}" -eq 0 ]] || { printf 'GPU memory is not empty: %s MiB\n' "${memory}" >&2; return 73; }
  done <<<"${rows}"
}
snapshot() {
  printf '\n===== %s UTC %s job=%s =====\n' "$1" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID}"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader,nounits || true
  printf '%s\n' '-- compute apps --'; apps || true
  printf '%s\n' '-- source/audited-reference/runner SHA256 --'; sha256sum "${source_path}" "${warp_reference_path}" "${tc_qk_reference_path}" "${scalar_import_path}" "${runner_path}" || true
}
post_done=0
on_exit() {
  local rc=$? post_rc=0
  trap - EXIT; set +e
  if [[ "${post_done}" -eq 0 ]]; then snapshot POST_ON_EXIT; require_b300 || post_rc=$?; require_empty || post_rc=$?; one_uuid >/dev/null || post_rc=$?; [[ "${rc}" -ne 0 || "${post_rc}" -eq 0 ]] || rc="${post_rc}"; fi
  printf '\n===== FINAL_RC=%s =====\n' "${rc}"; exit "${rc}"
}
trap on_exit EXIT

[[ -x "${python_bin}" && -n "${nvcc_bin}" && -x "${nvcc_bin}" && -x "${cuobjdump_bin}" ]] || { printf '%s\n' 'Missing Python or paired nvcc/cuobjdump.' >&2; exit 65; }
[[ -f "${source_path}" && -f "${warp_reference_path}" && -f "${tc_qk_reference_path}" && -f "${scalar_import_path}" ]] || { printf '%s\n' 'Missing source or audited reference.' >&2; exit 65; }
command -v timeout >/dev/null; export PYTHONOPTIMIZE=0
[[ "$("${python_bin}" -c 'print(int(__debug__))')" == 1 ]] || { printf '%s\n' 'Python assertions disabled.' >&2; exit 65; }
grep -q sm_103a <<<"$("${nvcc_bin}" --help)" || { printf '%s\n' 'nvcc lacks sm_103a.' >&2; exit 65; }
source_sha_pre="$(sha256sum "${source_path}" | awk '{print $1}')"; warp_sha_pre="$(sha256sum "${warp_reference_path}" | awk '{print $1}')"; tc_qk_sha_pre="$(sha256sum "${tc_qk_reference_path}" | awk '{print $1}')"; scalar_sha_pre="$(sha256sum "${scalar_import_path}" | awk '{print $1}')"; runner_sha_pre="$(sha256sum "${runner_path}" | awk '{print $1}')"
[[ "${warp_sha_pre}" == "${audited_warp_sha256}" && "${tc_qk_sha_pre}" == "${audited_tc_qk_sha256}" && "${scalar_sha_pre}" == "${audited_scalar_sha256}" ]] || { printf '%s\n' 'Audited scalar, warp-control, or TC-QK reference SHA mismatch.' >&2; exit 66; }
compile_flags='-std=c++17 -O3 -arch=sm_103a'
nvcc_version="$("${nvcc_bin}" --version)"
snapshot PRE; require_b300; require_empty; gpu_uuid="$(one_uuid)"

"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a "${source_path}" -o "${binary_path}" >"${compile_log}" 2>&1
"${nvcc_bin}" -std=c++17 -O3 -arch=sm_103a -ptx "${source_path}" -o "${ptx_path}" >>"${compile_log}" 2>&1
"${cuobjdump_bin}" --dump-sass "${binary_path}" >"${sass_path}"

instruction_json="${out_dir}/c2_cluster_attention_tc_qk_batch_abba_instruction_gate_${stamp}.json"
"${python_bin}" - "${ptx_path}" "${sass_path}" "${control_ptx_path}" "${candidate_ptx_path}" "${control_sass_path}" "${candidate_sass_path}" "${instruction_json}" <<'PY'
import json, re, sys
from pathlib import Path
ptx_path, sass_path, control_ptx_path, candidate_ptx_path, control_sass_path, candidate_sass_path, instruction_json = sys.argv[1:]
ptx = Path(ptx_path).read_text().splitlines(keepends=True)
sass = Path(sass_path).read_text().splitlines(keepends=True)
def extract(lines, marker, needle):
    starts = [i for i, line in enumerate(lines) if marker in line and needle in line]
    assert len(starts) == 1, (marker, needle, starts)
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if marker in lines[i]), len(lines))
    return ''.join(lines[start:end])
control_ptx = extract(ptx, '.entry ', 'c2_batch_warp_control_mbarrier_kernel')
candidate_ptx = extract(ptx, '.entry ', 'c2_batch_tc_qk_mbarrier_kernel')
control_sass = extract(sass, 'Function : ', 'c2_batch_warp_control_mbarrier_kernel')
candidate_sass = extract(sass, 'Function : ', 'c2_batch_tc_qk_mbarrier_kernel')
for path, text in ((control_ptx_path, control_ptx), (candidate_ptx_path, candidate_ptx),
                   (control_sass_path, control_sass), (candidate_sass_path, candidate_sass)):
    Path(path).write_text(text)
def proof(ptx_text, sass_text, require_shuffle, require_tc):
    out = {
      'ptx_mbarrier_init': ptx_text.count('mbarrier.init.shared.b64'),
      'ptx_mbarrier_release_arrive': ptx_text.count('mbarrier.arrive.release.cluster.shared::cluster.b64'),
      'ptx_mbarrier_acquire_wait': ptx_text.count('mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64'),
      'ptx_cluster_arrive': ptx_text.count('barrier.cluster.arrive'),
      'ptx_cluster_wait': ptx_text.count('barrier.cluster.wait'),
      'sass_mbarrier_init': sass_text.count('SYNCS.EXCH.64'),
      'sass_mbarrier_release_arrive': sass_text.count('SYNCS.ARRIVE.TRANS64.RED.A1T0'),
      'sass_mbarrier_acquire_wait': sass_text.count('SYNCS.PHASECHK.TRANS64.TRYWAIT'),
      'sass_cluster_arrive': sass_text.count('UCGABAR_ARV'),
      'sass_cluster_wait': sass_text.count('UCGABAR_WAIT'),
      'ptx_shuffle_down': ptx_text.count('shfl.sync.down.b32'),
      'ptx_shuffle_index': ptx_text.count('shfl.sync.idx.b32'),
      'sass_shuffle_down': sass_text.count('SHFL.DOWN'),
      'sass_shuffle_index': sass_text.count('SHFL.IDX'),
      'ptx_bf16_mma_sync': len(re.findall(r'\b(?:wmma\.)?mma\.sync(?:\.aligned)?\.row\.col\.m16n16k16\.f32\.bf16\.bf16\.f32\b', ptx_text)),
      'sass_hmma_16816_f32_bf16': len(re.findall(r'\bHMMA\.16816\.F32\.BF16\b', sass_text)),
    }
    for key in ('ptx_mbarrier_init','ptx_mbarrier_release_arrive','ptx_mbarrier_acquire_wait',
                'sass_mbarrier_init','sass_mbarrier_release_arrive','sass_mbarrier_acquire_wait'):
        assert out[key] == 1, (key, out)
    for key in ('ptx_cluster_arrive','ptx_cluster_wait','sass_cluster_arrive','sass_cluster_wait'):
        assert out[key] == 2, (key, out)
    if require_shuffle:
        for key in ('ptx_shuffle_down','ptx_shuffle_index','sass_shuffle_down','sass_shuffle_index'):
            assert out[key] >= 1, (key, out)
    if require_tc:
        assert out['ptx_bf16_mma_sync'] >= 1 and out['sass_hmma_16816_f32_bf16'] >= 1, out
    else:
        assert out['ptx_bf16_mma_sync'] == 0 and out['sass_hmma_16816_f32_bf16'] == 0, out
    return out
out = {'symbol_scoped_instruction_gate':'pass',
       'evidence':{'warp_control':proof(control_ptx, control_sass, True, False),
                   'tc_qk_candidate':proof(candidate_ptx, candidate_sass, False, True)}}
Path(instruction_json).write_text(json.dumps(out, sort_keys=True, indent=2)+'\n')
print(json.dumps(out, sort_keys=True))
PY
set +e; timeout --preserve-status --kill-after=10s 240s "${binary_path}" >"${raw_json}" 2>"${run_log}"; run_rc=$?; set -e
[[ "${run_rc}" -eq 0 ]] || { printf 'Batched AB/BA failed/timed out rc=%s raw=%s stderr=%s\n' "${run_rc}" "${raw_json}" "${run_log}" >&2; exit "${run_rc}"; }
source_sha_post="$(sha256sum "${source_path}" | awk '{print $1}')"; warp_sha_post="$(sha256sum "${warp_reference_path}" | awk '{print $1}')"; tc_qk_sha_post="$(sha256sum "${tc_qk_reference_path}" | awk '{print $1}')"; scalar_sha_post="$(sha256sum "${scalar_import_path}" | awk '{print $1}')"; runner_sha_post="$(sha256sum "${runner_path}" | awk '{print $1}')"
[[ "${source_sha_pre}" == "${source_sha_post}" && "${warp_sha_pre}" == "${warp_sha_post}" && "${tc_qk_sha_pre}" == "${tc_qk_sha_post}" && "${scalar_sha_pre}" == "${scalar_sha_post}" && "${runner_sha_pre}" == "${runner_sha_post}" ]] || { printf '%s\n' 'Source, audited reference, or runner changed during audit.' >&2; exit 66; }
binary_sha="$(sha256sum "${binary_path}" | awk '{print $1}')"; ptx_sha="$(sha256sum "${ptx_path}" | awk '{print $1}')"; sass_sha="$(sha256sum "${sass_path}" | awk '{print $1}')"

"${python_bin}" - "${raw_json}" "${final_json}" "${instruction_json}" "${source_sha_pre}" "${warp_sha_pre}" "${tc_qk_sha_pre}" "${scalar_sha_pre}" "${runner_sha_pre}" "${binary_sha}" "${ptx_sha}" "${sass_sha}" "${nvcc_version}" "${compile_flags}" "${SLURM_JOB_ID}" "${gpu_uuid}" <<'PY'
import json, math, struct, sys
from pathlib import Path
assert sys.flags.optimize == 0
(raw, final, instruction_path, source_sha, warp_sha, tc_qk_sha, scalar_sha, runner_sha,
 binary_sha, ptx_sha, sass_sha, nvcc_version, compile_flags, job, uuid) = sys.argv[1:]
p = json.loads(Path(raw).read_text())
instruction = json.loads(Path(instruction_path).read_text())
def finite_number(value, context):
    # JSON bool is a Python int and float(...) accepts strings: neither may
    # stand in for numeric evidence.
    assert isinstance(value, (int, float)) and not isinstance(value, bool), ('non-numeric-field', context, value)
    value = float(value); assert math.isfinite(value), ('non-finite-field', context, value)
    return value
def integer_number(value, context):
    value = finite_number(value, context); assert value.is_integer(), ('non-integral-field', context, value)
    return int(value)
def require_bool(value, expected, context):
    assert isinstance(value, bool), ('non-bool-field', context, value)
    if expected is not None: assert value is expected, ('unexpected-bool-field', context, value, expected)
    return value
assert p['schema'] == 'c2-cluster-attention-tc-qk-batch-abba-v1' and p['status'] == 'pass', p
assert set(p) == {'schema','status','boundary','timing_seed','batch_cases','shape','abi','provenance','cluster_layout','producer_contract','synchronization','dtype_contract','environment','resource_model','cases'}
assert isinstance(p['boundary'], str)
assert isinstance(p['batch_cases'], list) and [integer_number(value,('batch_cases',index)) for index,value in enumerate(p['batch_cases'])] == [1,4,8,16]
assert integer_number(p['timing_seed'],'timing_seed') == 2026
expected_shape={'Hkv':4,'Hq':64,'G':16,'D':128,'page_size':128,'selected_pages':16,'logical_pages':32,'physical_pages_per_batch':32}
assert set(p['shape']) == set(expected_shape)
for key, expected in expected_shape.items(): assert integer_number(p['shape'][key],('shape',key)) == expected
assert set(p['abi']) == {'query','output','seq_lens','block_table','topk','cache','cluster_mapping','disjoint_physical_page_pool_per_batch','topk_row_order_differs_from_first','topk_row_order_scope'}
assert p['abi']['query'] == p['abi']['output'] == '[B,Hq,D]'
assert p['abi']['seq_lens'] == '[B]' and p['abi']['block_table'] == '[B,max_blocks]' and p['abi']['topk'] == '[B,Hkv,Ktop]'
assert p['abi']['cache'] == '[physical_page,Hkv,P,D]' and p['abi']['cluster_mapping'] == 'batch=(blockIdx.x/4)/Hkv; kv_head=(blockIdx.x/4)%Hkv'
require_bool(p['abi']['disjoint_physical_page_pool_per_batch'], True, ('abi','disjoint_physical_page_pool_per_batch'))
require_bool(p['abi']['topk_row_order_differs_from_first'], True, ('abi','topk_row_order_differs_from_first'))
assert p['abi']['topk_row_order_scope'] == 'each nonzero (batch,kv_head) ordered row differs from (0,0); paired sorted-set signatures provide the stronger two-seed row-coverage gate'
assert p['provenance'] == {
 'scalar_protocol':'c2_cluster_attention_mbarrier_smoke.cu is preprocessor-included and SHA-pinned',
 'warp_mapping_reference':'c2_cluster_attention_warp_producer_abba.cu is SHA-pinned audited reference only; its mapping is derived here and it is not preprocessor-included',
 'tc_qk_design_reference':'c2_cluster_attention_tc_qk_abba.cu is SHA-pinned by the runner; its QK mapping is adapted here to the batch ABI'}
layout=p['cluster_layout']; assert set(layout) == {'num_ctas','grid','selected_pages_per_producer','threads_per_block'} and layout['grid'] == 'B*Hkv*4'
for key, expected in {'num_ctas':4,'selected_pages_per_producer':8,'threads_per_block':256}.items(): assert integer_number(layout[key],('cluster_layout',key)) == expected
contract=p['producer_contract']
assert set(contract) == {'same_remote_dsm_mbarrier_protocol','same_output_abi','same_launch_shape','same_real_selected_causal_attention','persistent_device_buffers_outside_timing','caller_owned_independent_outputs','single_kernel_launch_per_cuda_event_sample','ABBA_interleaved','initialization_copies_and_oracle_outside_timing','changed_field','candidate_extra_shared_and_cta_barriers_included','timed_launch_validation_scope'}
for key in ('same_remote_dsm_mbarrier_protocol','same_output_abi','same_launch_shape','same_real_selected_causal_attention','persistent_device_buffers_outside_timing','caller_owned_independent_outputs','single_kernel_launch_per_cuda_event_sample','ABBA_interleaved','initialization_copies_and_oracle_outside_timing','candidate_extra_shared_and_cta_barriers_included'):
    require_bool(contract[key], True, ('producer_contract',key))
assert contract['changed_field'] == 'rank-0/1 producer QK mapping only: warp shuffle control versus BF16 WMMA QK candidate'
assert contract['timed_launch_validation_scope'] == 'pre-timing two-seed checks plus post-timing fresh sentinel-reset control/candidate launches and oracle recheck; intermediate timed outputs not inspected'
sync=p['synchronization']; assert set(sync) == {'mbarrier_expected_arrivals','mbarrier_wait_parity','mbarrier_max_polls'}
for key, expected in {'mbarrier_expected_arrivals':2,'mbarrier_wait_parity':0,'mbarrier_max_polls':1<<24}.items(): assert integer_number(sync[key],('synchronization',key)) == expected
environment=p['environment']; assert set(environment) == {'device','capability','cluster_launch_supported'} and isinstance(environment['device'],str) and 'B300' in environment['device']
assert isinstance(environment['capability'],list) and [integer_number(value,('environment','capability',index)) for index,value in enumerate(environment['capability'])] == [10,3]
require_bool(environment['cluster_launch_supported'], True, ('environment','cluster_launch_supported'))
dtype=p['dtype_contract']; assert set(dtype) == {'producer_partial','caller_output','tc_qk_accumulator','oracle_accumulator','oracle_softmax','tolerance'}
assert dtype['producer_partial'] == dtype['caller_output'] == 'bfloat16' and dtype['tc_qk_accumulator'] == 'float32' and dtype['oracle_accumulator'] == 'float64' and dtype['oracle_softmax'] == 'independent two-pass natural-exp direct selected-page causal attention'
assert set(dtype['tolerance']) == {'rtol','atol'}
assert math.isclose(finite_number(dtype['tolerance']['atol'],('dtype_contract','tolerance','atol')),5e-4,rel_tol=0,abs_tol=1e-9) and math.isclose(finite_number(dtype['tolerance']['rtol'],('dtype_contract','tolerance','rtol')),5e-3,rel_tol=0,abs_tol=1e-9)
resources=p['resource_model']; assert set(resources) == {'shared_equal','warp_control','tc_qk_candidate'}
require_bool(resources['shared_equal'], False, ('resource_model','shared_equal'))
for arm in ('warp_control','tc_qk_candidate'):
    assert set(resources[arm]) == {'static_shared_bytes','num_regs','local_bytes'}
    assert integer_number(resources[arm]['static_shared_bytes'],('resource_model',arm,'static_shared_bytes')) > 0
    assert integer_number(resources[arm]['num_regs'],('resource_model',arm,'num_regs')) > 0
    assert integer_number(resources[arm]['local_bytes'],('resource_model',arm,'local_bytes')) >= 0
assert integer_number(resources['tc_qk_candidate']['local_bytes'],('resource_model','tc_qk_candidate','local_bytes')) == 0
assert set(instruction) == {'symbol_scoped_instruction_gate','evidence'} and instruction['symbol_scoped_instruction_gate'] == 'pass'
assert set(instruction['evidence']) == {'warp_control','tc_qk_candidate'}
control_i=instruction['evidence']['warp_control']; tc_i=instruction['evidence']['tc_qk_candidate']
for row in (control_i,tc_i):
    assert set(row) == {'ptx_mbarrier_init','ptx_mbarrier_release_arrive','ptx_mbarrier_acquire_wait','ptx_cluster_arrive','ptx_cluster_wait','sass_mbarrier_init','sass_mbarrier_release_arrive','sass_mbarrier_acquire_wait','sass_cluster_arrive','sass_cluster_wait','ptx_shuffle_down','ptx_shuffle_index','sass_shuffle_down','sass_shuffle_index','ptx_bf16_mma_sync','sass_hmma_16816_f32_bf16'}
    for key in ('ptx_mbarrier_init','ptx_mbarrier_release_arrive','ptx_mbarrier_acquire_wait','sass_mbarrier_init','sass_mbarrier_release_arrive','sass_mbarrier_acquire_wait'):
        assert integer_number(row[key],('instruction',key)) == 1, (key,row)
    for key in ('ptx_cluster_arrive','ptx_cluster_wait','sass_cluster_arrive','sass_cluster_wait'):
        assert integer_number(row[key],('instruction',key)) == 2, (key,row)
assert integer_number(control_i['ptx_bf16_mma_sync'],('instruction','control','ptx_bf16_mma_sync')) == 0
assert integer_number(control_i['sass_hmma_16816_f32_bf16'],('instruction','control','sass_hmma_16816_f32_bf16')) == 0
assert integer_number(tc_i['ptx_bf16_mma_sync'],('instruction','candidate','ptx_bf16_mma_sync')) >= 1
assert integer_number(tc_i['sass_hmma_16816_f32_bf16'],('instruction','candidate','sass_hmma_16816_f32_bf16')) >= 1
for key in ('ptx_shuffle_down','ptx_shuffle_index','sass_shuffle_down','sass_shuffle_index'):
    assert integer_number(control_i[key],('instruction','control',key)) >= 1, (key,control_i)
def f32(value):
    return struct.unpack('<f',struct.pack('<f',finite_number(value, 'float32 timing field')))[0]
def summary(values):
    values=sorted(f32(v) for v in values); assert values and all(v>0 for v in values); n=len(values)
    median=values[n//2] if n%2 else f32(f32(values[n//2-1]+values[n//2])*f32(.5))
    return {'p10_us':values[max(0,math.ceil(.1*n)-1)],'median_us':median,'p90_us':values[min(n-1,math.ceil(.9*n)-1)]}
def assert_f32(actual, expected, context):
    assert f32(actual)==expected, (context,actual,expected,f32(actual))
assert isinstance(p['cases'],list) and [integer_number(case['B'],('cases',index,'B')) for index,case in enumerate(p['cases'])] == [1,4,8,16]
for case in p['cases']:
    assert set(case) == {'B','paired_seed_selected_set_signatures_unique','correctness','post_timing_correctness','timing'}
    B=integer_number(case['B'],('case','B')); require_bool(case['paired_seed_selected_set_signatures_unique'],True,(B,'paired_seed_selected_set_signatures_unique'))
    assert isinstance(case['correctness'],list) and len(case['correctness']) == 2 and {integer_number(row['seed'],(B,'correctness','seed')) for row in case['correctness']} == {17,2026}
    assert integer_number(case['post_timing_correctness']['seed'],(B,'post_timing_correctness','seed')) == 2026
    for row in [*case['correctness'],case['post_timing_correctness']]:
        assert set(row) == {'seed','batch','hierarchy_valid','adversarial_unselected_visible_pages','adversarial_masked_tokens','warp_control','tc_qk_candidate','cross_arm'}
        assert integer_number(row['batch'],(B,'correctness','batch')) == B
        require_bool(row['hierarchy_valid'],True,(B,'correctness','hierarchy_valid'))
        assert integer_number(row['adversarial_unselected_visible_pages'],(B,'correctness','adversarial_unselected_visible_pages')) > 0
        assert integer_number(row['adversarial_masked_tokens'],(B,'correctness','adversarial_masked_tokens')) == B*4*127
        for arm in ('warp_control','tc_qk_candidate'):
            a=row[arm]; assert set(a) == {'max_abs','max_rel','oracle_finite','finite','sentinel_clean','allclose'}
            for key in ('oracle_finite','finite','sentinel_clean','allclose'): require_bool(a[key],True,(B,'correctness',arm,key))
            finite_number(a['max_abs'],(B,'correctness',arm,'max_abs')); finite_number(a['max_rel'],(B,'correctness',arm,'max_rel'))
        cross=row['cross_arm']; assert set(cross) == {'max_abs','max_rel','bfloat16_bitwise_equal'}
        finite_number(cross['max_abs'],(B,'correctness','cross_arm','max_abs')); finite_number(cross['max_rel'],(B,'correctness','cross_arm','max_rel')); require_bool(cross['bfloat16_bitwise_equal'],None,(B,'correctness','cross_arm','bfloat16_bitwise_equal'))
    timing=case['timing']; assert set(timing) == {'protocol','warmup_each','abba_pairs','samples_per_arm','raw_samples_us','warp_control','tc_qk_candidate','speedup_warp_control_over_tc_qk','speedup_by_partition','promotion_gate'} and timing['protocol']=='warmup_each_then_51_warp_control_tc_qk_tc_qk_warp_control_ABBA_pairs'
    for key, expected in {'warmup_each':10,'abba_pairs':51,'samples_per_arm':102}.items(): assert integer_number(timing[key],(B,'timing',key)) == expected
    assert set(timing['raw_samples_us']) == {'warp_control','tc_qk_candidate'}
    computed={}
    for arm in ('warp_control','tc_qk_candidate'):
        computed[arm]={}
        assert set(timing['raw_samples_us'][arm]) == {'AB','BA'} and set(timing[arm]) == {'all','when_launch_order_is_AB','when_launch_order_is_BA'}
        ab=timing['raw_samples_us'][arm]['AB']; ba=timing['raw_samples_us'][arm]['BA']; assert isinstance(ab,list) and isinstance(ba,list) and len(ab)==len(ba)==51
        for key,values in (('all',[*ab,*ba]),('when_launch_order_is_AB',ab),('when_launch_order_is_BA',ba)):
            expect=summary(values); actual=timing[arm][key]; assert set(actual)=={'p10_us','median_us','p90_us'},(B,arm,key,actual)
            for stat,value in expect.items(): assert_f32(actual[stat],value,(B,arm,key,stat))
            computed[arm][key]=expect
    speed=f32(computed['warp_control']['all']['median_us']/computed['tc_qk_candidate']['all']['median_us'])
    ab=f32(computed['warp_control']['when_launch_order_is_AB']['median_us']/computed['tc_qk_candidate']['when_launch_order_is_AB']['median_us'])
    ba=f32(computed['warp_control']['when_launch_order_is_BA']['median_us']/computed['tc_qk_candidate']['when_launch_order_is_BA']['median_us'])
    assert set(timing['speedup_by_partition']) == {'AB','BA'}
    assert_f32(timing['speedup_warp_control_over_tc_qk'],speed,(B,'speedup_warp_control_over_tc_qk'))
    assert_f32(timing['speedup_by_partition']['AB'],ab,(B,'speedup_by_partition','AB'))
    assert_f32(timing['speedup_by_partition']['BA'],ba,(B,'speedup_by_partition','BA'))
    gate=timing['promotion_gate']
    assert set(gate) == {'combined_warp_control_over_tc_qk_at_least_1_10','AB_warp_control_over_tc_qk_greater_than_1_05','BA_warp_control_over_tc_qk_greater_than_1_05','all_correct','promoted'}
    require_bool(gate['combined_warp_control_over_tc_qk_at_least_1_10'],speed>=1.10,(B,'promotion_gate','combined_warp_control_over_tc_qk_at_least_1_10'))
    require_bool(gate['AB_warp_control_over_tc_qk_greater_than_1_05'],ab>1.05,(B,'promotion_gate','AB_warp_control_over_tc_qk_greater_than_1_05'))
    require_bool(gate['BA_warp_control_over_tc_qk_greater_than_1_05'],ba>1.05,(B,'promotion_gate','BA_warp_control_over_tc_qk_greater_than_1_05'))
    require_bool(gate['all_correct'],True,(B,'promotion_gate','all_correct'))
    require_bool(gate['promoted'],speed>=1.10 and ab>1.05 and ba>1.05,(B,'promotion_gate','promoted'))
assert warp_sha=='24938b464a5b179a7c0e6f2450dd72b231635c73e7b46ea6c5a3fac85357444a'
assert tc_qk_sha=='523347312f07345487a2591d6e52ef231c94ba8b6fe00e36e1fc16a02bb53431'
assert scalar_sha=='6d69cecbda9baf93b19bfba3162a4c33e42298bee43a9235de98e2acd369372f'
p['build']={'nvcc_version':nvcc_version,'compile_flags':compile_flags,'binary_sha256':binary_sha,'ptx_sha256':ptx_sha,'sass_sha256':sass_sha,'instruction_gate_json':instruction_path,'symbol_scoped_instruction_gate':'warp control: exact one mbarrier init/release/wait and two cluster-barrier pairs with no BF16 MMA/HMMA; TC-QK candidate: exact protocol counts plus BF16 MMA/HMMA'}
p['execution']={'slurm_job_id':job,'gpu_uuid':uuid}
p['source_sha256']={
 'challenge_v2/c2_cluster_attention_tc_qk_batch_abba.cu':source_sha,
 'challenge_v2/c2_cluster_attention_warp_producer_abba.cu':warp_sha,
 'challenge_v2/c2_cluster_attention_tc_qk_abba.cu':tc_qk_sha,
 'challenge_v2/c2_cluster_attention_mbarrier_smoke.cu':scalar_sha,
 'challenge_v2/run_c2_cluster_attention_tc_qk_batch_abba_clean.sh':runner_sha}
Path(final).write_text(json.dumps(p,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'secondary_gate':'pass','json':final,'per_batch_promotion':{str(c['B']):c['timing']['promotion_gate']['promoted'] for c in p['cases']}},sort_keys=True))
PY
require_b300; require_empty; post_uuid="$(one_uuid)"; [[ "${post_uuid}" == "${gpu_uuid}" ]] || { printf 'GPU UUID changed %s -> %s\n' "${gpu_uuid}" "${post_uuid}" >&2; exit 75; }
post_done=1; snapshot POST
printf 'Native batched ABI warp-control/TC-QK ABBA completed: %s\n' "${final_json}"
