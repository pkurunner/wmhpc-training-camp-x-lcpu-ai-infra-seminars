#!/usr/bin/env bash
# v14 K/V-stage-padding directed gate.  Frozen v6 is oracle-only, not v14 metadata.
#SBATCH --job-name=c2-native-plugin-v14-directed
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/lcpu/85117379/slurm-c2-native-plugin-v14-directed-%j.log
set -Eeuo pipefail
preflight(){ r=$?; trap - EXIT; case "$r" in 0|2)exit "$r";;*)exit 2;;esac; }; trap preflight EXIT
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONPATH= PYTHONDONTWRITEBYTECODE=1
ROOT=/home/lcpu/85117379/c2-native-plugin-v14-kv-stage-padding-directed-artifacts-20260831
PY=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
BASE=/home/lcpu/85117379/native_c2_operator_bench_20260829.py
V5=/home/lcpu/85117379/native_c2_v5_softmax_directed_20260829.py
FROZEN=/home/lcpu/85117379/native_c2_v6_register_numerator_directed_20260830.py
BASE_SHA=0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28
V5_SHA=43ad54c5ad72cc40bf8b8a6e17592fa16cacc1d1e62f5f8c42ac2a4d3206cdd1
: "$SLURM_JOB_ID"; : "$C2_V14_CANDIDATE_PLUGIN"; : "$C2_EXPECTED_V14_CANDIDATE_SHA"; : "$C2_V14_CANDIDATE_SOURCE"; : "$C2_EXPECTED_V14_SOURCE_SHA"
: "$C2_V14_CANDIDATE_RESOURCE_GATE"; : "$C2_EXPECTED_V14_RESOURCE_GATE_SHA"; : "$C2_V14_CANDIDATE_PROVENANCE"; : "$C2_EXPECTED_V14_PROVENANCE_SHA"
: "$C2_V14_AOT_FINAL_STATUS"; : "$C2_EXPECTED_V14_AOT_FINAL_STATUS_SHA"; : "$C2_V14_AOT_FINAL_STATUS_SIDECAR"; : "$C2_EXPECTED_V14_AOT_FINAL_STATUS_SIDECAR_SHA"
: "$C2_V14_AOT_OUTPUTS_MANIFEST"; : "$C2_EXPECTED_V14_AOT_OUTPUTS_MANIFEST_SHA"; : "$C2_EXPECTED_V6_DIRECTED_HARNESS_SHA"
: "$C2_V14_EXPECTED_REG"; : "$C2_V14_EXPECTED_STACK"; : "$C2_V14_EXPECTED_SHARED"; : "$C2_V14_EXPECTED_LOCAL"; : "$C2_EXPECTED_V14_DIRECTED_SCRIPT_SHA"
for x in "$C2_EXPECTED_V14_CANDIDATE_SHA" "$C2_EXPECTED_V14_SOURCE_SHA" "$C2_EXPECTED_V14_RESOURCE_GATE_SHA" "$C2_EXPECTED_V14_PROVENANCE_SHA" "$C2_EXPECTED_V14_AOT_FINAL_STATUS_SHA" "$C2_EXPECTED_V14_AOT_FINAL_STATUS_SIDECAR_SHA" "$C2_EXPECTED_V14_AOT_OUTPUTS_MANIFEST_SHA" "$C2_EXPECTED_V6_DIRECTED_HARNESS_SHA" "$C2_EXPECTED_V14_DIRECTED_SCRIPT_SHA"; do [[ "$x" =~ ^[0-9a-f]{64}$ ]]||exit 2; done
for x in "$C2_V14_EXPECTED_REG" "$C2_V14_EXPECTED_STACK" "$C2_V14_EXPECTED_SHARED" "$C2_V14_EXPECTED_LOCAL"; do [[ "$x" =~ ^[0-9]+$ ]]||exit 2; done
((C2_V14_EXPECTED_STACK==0&&C2_V14_EXPECTED_SHARED==33184&&C2_V14_EXPECTED_LOCAL==0))||exit 2
for x in "$C2_V14_CANDIDATE_PLUGIN" "$C2_V14_CANDIDATE_SOURCE" "$C2_V14_CANDIDATE_RESOURCE_GATE" "$C2_V14_CANDIDATE_PROVENANCE" "$C2_V14_AOT_FINAL_STATUS" "$C2_V14_AOT_FINAL_STATUS_SIDECAR" "$C2_V14_AOT_OUTPUTS_MANIFEST"; do [[ "$x" == /*&&-f "$x" ]]||exit 2;done
[[ -f "$BASE"&&-f "$V5"&&-f "$FROZEN"&&-d "$ROOT" ]]||exit 2
DIR=$ROOT/job$SLURM_JOB_ID; [[ -z "$(find "$ROOT" -mindepth 1 -maxdepth 1 -print -quit)"&& ! -e "$DIR" ]]||exit 2; mkdir -p "$DIR"
SCRIPT=$(readlink -f "$0"); COPY=$DIR/submitted-script-job$SLURM_JOB_ID.slurm; STATUS=$DIR/final-status-job$SLURM_JOB_ID.txt; SIDECAR=$DIR/final-status-job$SLURM_JOB_ID.sha256
export PATH=/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin PYTHONPYCACHEPREFIX=$DIR/pycache
UUID=
sel(){ d=$CUDA_VISIBLE_DEVICES; [[ "$d" != *,* && ( "$d" =~ ^[0-9]+$ || "$d" =~ ^GPU-[0-9A-Fa-f-]+$ ) ]]||return 1;printf '%s\n' "$d"; }
gpu(){ nvidia-smi --id "$(sel)" --query-gpu=uuid --format=csv,noheader,nounits|sed -n '1p'; }
apps(){ nvidia-smi --id "$(sel)" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null|sed '/^[[:space:]]*$/d'; }
snap(){ tag=$1;out=$DIR/gpu-$tag-job$SLURM_JOB_ID.txt;{ printf '===== %s UTC=%s =====\n' "$tag" "$(date -u +%FT%TZ)";nvidia-smi --id "$(sel)" --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader,nounits;printf 'COMPUTE_APPS\n';apps;}>"$out";cat "$out"; }
end(){ original=$?;rc=$original;err=0;trap - EXIT;set +e;case "$original" in 0|2);;*)rc=2;err=1;;esac;snap final-post||err=1;[[ -z "$(apps)"&&-n "$UUID"&&"$(gpu)"=="$UUID" ]]||err=1;[[ $err -eq 0 ]]||rc=2;printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s UTC=%s\n' "$rc" "$original" "$err" "$(date -u +%FT%TZ)">"$STATUS";sha256sum "$STATUS" "$DIR/gpu-final-post-job$SLURM_JOB_ID.txt">"$SIDECAR"||rc=2;exit "$rc";};trap end EXIT
exec > >(tee "$DIR/driver-job$SLURM_JOB_ID.log") 2>&1
printf '%s  %s\n' "$C2_EXPECTED_V14_CANDIDATE_SHA" "$C2_V14_CANDIDATE_PLUGIN" "$C2_EXPECTED_V14_SOURCE_SHA" "$C2_V14_CANDIDATE_SOURCE" "$C2_EXPECTED_V14_RESOURCE_GATE_SHA" "$C2_V14_CANDIDATE_RESOURCE_GATE" "$C2_EXPECTED_V14_PROVENANCE_SHA" "$C2_V14_CANDIDATE_PROVENANCE" "$C2_EXPECTED_V14_AOT_FINAL_STATUS_SHA" "$C2_V14_AOT_FINAL_STATUS" "$C2_EXPECTED_V14_AOT_FINAL_STATUS_SIDECAR_SHA" "$C2_V14_AOT_FINAL_STATUS_SIDECAR" "$C2_EXPECTED_V14_AOT_OUTPUTS_MANIFEST_SHA" "$C2_V14_AOT_OUTPUTS_MANIFEST" "$BASE_SHA" "$BASE" "$V5_SHA" "$V5" "$C2_EXPECTED_V6_DIRECTED_HARNESS_SHA" "$FROZEN" "$C2_EXPECTED_V14_DIRECTED_SCRIPT_SHA" "$SCRIPT"|sha256sum -c -||exit 2
cp -- "$SCRIPT" "$COPY";chmod a-w "$COPY";printf '%s  %s\n' "$C2_EXPECTED_V14_DIRECTED_SCRIPT_SHA" "$COPY"|sha256sum -c -||exit 2
snap pre||exit 2;line=$(nvidia-smi --id "$(sel)" --query-gpu=name,compute_cap --format=csv,noheader,nounits);[[ "$line" == *B300*&&"$line" == *10.3*&&-z "$(apps)" ]]||exit 2;UUID=$(gpu);[[ "$UUID" =~ ^GPU- ]]||exit 2
RES=$(grep -m1 -E 'REG:[[:space:]]*[0-9]+.*STACK:[[:space:]]*[0-9]+.*SHARED:[[:space:]]*[0-9]+.*LOCAL:[[:space:]]*[0-9]+' "$C2_V14_CANDIDATE_RESOURCE_GATE")||exit 2
readres(){ sed -nE "s/.*$1:[[:space:]]*([0-9]+).*/\\1/p"<<<"$RES";};REG=$(readres REG);STACK=$(readres STACK);SHARED=$(readres SHARED);LOCAL=$(readres LOCAL)
((REG==C2_V14_EXPECTED_REG&&STACK==0&&STACK==C2_V14_EXPECTED_STACK&&SHARED==33184&&SHARED==C2_V14_EXPECTED_SHARED&&LOCAL==0&&LOCAL==C2_V14_EXPECTED_LOCAL))||exit 2
grep -Fx "$C2_EXPECTED_V14_CANDIDATE_SHA  $C2_V14_CANDIDATE_PLUGIN" "$C2_V14_CANDIDATE_PROVENANCE"||exit 2
"$PY" - "$C2_V14_CANDIDATE_PLUGIN" "$C2_V14_CANDIDATE_SOURCE" "$C2_V14_CANDIDATE_RESOURCE_GATE" "$C2_V14_CANDIDATE_PROVENANCE" "$C2_V14_AOT_FINAL_STATUS" "$C2_V14_AOT_FINAL_STATUS_SIDECAR" "$C2_V14_AOT_OUTPUTS_MANIFEST"<<'PY'||exit 2
import hashlib,re,sys
from pathlib import Path
p,s,r,v,f,z,o=(Path(x).resolve() for x in sys.argv[1:])
def d(x):return hashlib.sha256(x.read_bytes()).hexdigest()
def m(x):
 a={}
 for q in x.read_text().splitlines():
  h,sep,n=q.partition('  ');n=Path(n).resolve();assert sep and re.fullmatch('[0-9a-f]{64}',h)and n.is_file()and d(n)==h and n not in a;a[n]=h
 assert a;return a
g=re.fullmatch(r'final-status-job([0-9]+)\.txt',f.name);assert g;j=g.group(1);root=f.parent
assert root.name==f'job{j}'and z==root/f'final-status-job{j}.sha256'and o==root/f'outputs-job{j}.sha256'and p==root/'vllm/_native_c2_msa_decode_plugin.abi3.so'and s==root/'native_c2_decode.v14-kv-stage-padding.cu'and r.parent==root and v.parent==root
st=dict(q.split('=',1)for q in f.read_text().split()if'='in q);assert all(st.get(k)=='0'for k in('BODY_RC','TEE_RC','MANIFEST_RC','FINAL_RC'));assert f in m(z);om=m(o);assert all(q in om for q in(p,s,r,v));vm=m(v);assert vm.get(p)==d(p)and any(q.name=='native_c2_decode.cu'and h==d(s)for q,h in vm.items())
PY
"$PY" -m py_compile "$BASE" "$V5" "$FROZEN"||exit 2
"$PY" - "$FROZEN"<<'PY'||exit 2
import ast,sys
ast.parse(open(sys.argv[1],encoding='utf-8').read(),filename=sys.argv[1])
PY
RUN=$DIR/run-manifest-job$SLURM_JOB_ID.json
"$PY" - "$RUN" "$C2_V14_CANDIDATE_PLUGIN" "$C2_V14_CANDIDATE_SOURCE" "$C2_V14_CANDIDATE_RESOURCE_GATE" "$C2_V14_CANDIDATE_PROVENANCE" "$C2_V14_AOT_FINAL_STATUS" "$C2_V14_AOT_FINAL_STATUS_SIDECAR" "$C2_V14_AOT_OUTPUTS_MANIFEST" "$BASE" "$V5" "$FROZEN" "$COPY"<<'PY'||exit 2
import hashlib,json,sys
from pathlib import Path
ls=('v14_kv_stage_padding_plugin','v14_kv_stage_padding_source','v14_kv_stage_padding_resource_gate','v14_kv_stage_padding_provenance','v14_aot_final_status','v14_aot_final_status_sidecar','v14_aot_outputs_manifest','base_harness','v5_directed_harness','frozen_v6_directed_harness','v14_kv_stage_padding_directed_script');ps=[Path(q).resolve()for q in sys.argv[2:]];assert len(ls)==len(ps)and all(q.is_file()for q in ps)
Path(sys.argv[1]).write_text(json.dumps({'schema':'c2-native-c2-v14-kv-stage-padding-directed-input-manifest-v1','candidate':'v14-kv-stage-padding','frozen_harness_schema':'c2-native-c2-v6-register-numerator-directed-v1','inputs':[{'label':a,'path':str(b),'sha256':hashlib.sha256(b.read_bytes()).hexdigest(),'size_bytes':b.stat().st_size}for a,b in zip(ls,ps,strict=True)]},sort_keys=True,indent=2)+'\n')
PY
IN=$DIR/inputs-job$SLURM_JOB_ID.sha256;sha256sum "$C2_V14_CANDIDATE_PLUGIN" "$C2_V14_CANDIDATE_SOURCE" "$C2_V14_CANDIDATE_RESOURCE_GATE" "$C2_V14_CANDIDATE_PROVENANCE" "$C2_V14_AOT_FINAL_STATUS" "$C2_V14_AOT_FINAL_STATUS_SIDECAR" "$C2_V14_AOT_OUTPUTS_MANIFEST" "$BASE" "$V5" "$FROZEN" "$COPY" "$RUN" "$DIR/gpu-pre-job$SLURM_JOB_ID.txt">"$IN"
RESULT=$DIR/v14-kv-stage-padding-directed-job$SLURM_JOB_ID.json
timeout --foreground --preserve-status --kill-after=10s 600s "$PY" "$FROZEN" --library "$C2_V14_CANDIDATE_PLUGIN" --base-harness "$BASE" --v5-directed-harness "$V5" --sync-repeats 4 --max-iteration-seconds 15 --atol 1e-4 --rtol 1e-3>"$RESULT"||exit 2
"$PY" - "$RESULT" "$C2_V14_CANDIDATE_PLUGIN"<<'PY'||exit 2
import json,math,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text());assert x['schema']=='c2-native-c2-v6-register-numerator-directed-v1'and x['all_gates_pass']is True and x['no_monkeypatch']is True and Path(x['operator_library']).resolve()==Path(sys.argv[2]).resolve()and x['dispatch']=='direct_torch.ops._C.native_c2_msa_decode';assert x['contract']['producer_ctas']==4 and x['contract']['pages_per_producer']==4 and x['contract']['selected_pages']==16 and x['contract']['page_size']==128
def c(q):assert q['all_gates_pass']is True and q['caller_output']['pointer_unchanged']is True and q['caller_output']['return_is_none']is True and q['stability']['bitwise_repeatable']is True and q['oracle']['finite_output']is True and q['oracle']['reference_fp64_fp32_agree']is True and q['oracle']['fp64']['allclose']is True and q['oracle']['fp32']['allclose']is True and len(q['sync_watchdog']['elapsed_seconds'])==4 and all(math.isfinite(float(v))and 0<=float(v)<=15 for v in q['sync_watchdog']['elapsed_seconds'])
c(x['head_dim_encoding']);e=x['head_dim_encoding']['head_dim_encoding'];assert e['dim_code_count']==128 and e['head_signature_count']==64 and len(e['head_signature_offsets'])==64 and len(set(e['head_signature_offsets']))==64 and len(e['target_tokens'])==64 and e['exact_location_match']is True and e['nonzero_mismatch_count']==0 and e['max_abs_to_exact_expected']==0.0
assert x['rank_directed_case_count']==4
for i,q in enumerate(x['rank_directed']):c(q);assert q['producer_rank']==i and q['selected_page_indices']==list(range(4*i,4*i+4))and q['rank_expected_value']['allclose']is True
assert x['softmax_directed_case_count']==2
for q,(n,l,t)in zip(x['softmax_directed'],(('all-valid-online-rescale',2048,False),('mixed-causal-tail-37',37,True)),strict=True):c(q);assert q['case']==n and q['sequence_length']==l and q['mixed_causal_subgroup']is t
c(x['all_invalid']);assert x['all_invalid']['case']=='all-invalid'and x['all_invalid']['all_zero']is True and x['all_invalid']['oracle']['fp64']['max_abs']==0.0 and x['all_invalid']['oracle']['fp32']['max_abs']==0.0
p=x['profiler'];assert p['native_dispatcher_hit']is True and p['native_kernel_hit']is True and p['exactly_one_dispatcher_event']is True and p['exactly_one_native_kernel_event']is True and p['dispatcher_events']==[{'key':'_C::native_c2_msa_decode','count':1}]and len(p['kernel_events'])==1 and p['kernel_events'][0]['count']==1 and p['caller_output_pointer_unchanged']is True and p['return_is_none']is True and p['finite_output']is True
PY
snap post||exit 2;[[ -z "$(apps)" ]]||exit 2;sha256sum -c "$IN"||exit 2;OUT=$DIR/outputs-job$SLURM_JOB_ID.sha256;sha256sum "$RESULT" "$RUN" "$IN" "$DIR/gpu-pre-job$SLURM_JOB_ID.txt" "$DIR/gpu-post-job$SLURM_JOB_ID.txt">"$OUT";sha256sum -c "$OUT"
