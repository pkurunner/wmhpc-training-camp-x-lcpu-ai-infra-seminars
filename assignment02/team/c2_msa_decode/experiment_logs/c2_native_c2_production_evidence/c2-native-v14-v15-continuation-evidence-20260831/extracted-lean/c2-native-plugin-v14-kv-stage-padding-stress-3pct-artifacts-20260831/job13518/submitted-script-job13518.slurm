#!/usr/bin/env bash
# v14 K/V-stage-padding strict v12(job12983)->v14 ABBA performance gate.
# A/B workers are independent Python processes because both DSOs use the same dispatcher schema.
#SBATCH --job-name=c2-native-v14-3pct-confirm
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/slurm-c2-native-plugin-v14-stress-3pct-%j.log
set -Eeuo pipefail
preflight(){ r=$?;trap - EXIT;case "$r" in 0|2|3)exit "$r";;*)exit 2;;esac;};trap preflight EXIT
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONOPTIMIZE=0 PYTHONPATH= PYTHONDONTWRITEBYTECODE=1
ROOT=/home/lcpu/85117379/c2-native-plugin-v14-kv-stage-padding-stress-3pct-artifacts-20260831
PY=/home/lcpu/85117379/venv-vllm-d4da0c5/bin/python
BASE=/home/lcpu/85117379/native_c2_operator_bench_20260829.py
BASE_SHA=0b51fede1e6258eea0b535acba2a86e371b284e9655352156cdeb3d310d90c28
REF=/home/lcpu/85117379/c2-native-plugin-v12-aot-artifacts-20260830/job12983/vllm/_native_c2_msa_decode_plugin.abi3.so
REF_SHA=064f967ade97ad79420f5a24be006733a6271c3743c0e729e116d7236831d7b6
REF_SRC=/home/lcpu/85117379/c2-native-plugin-v12-aot-artifacts-20260830/job12983/native_c2_decode.v12-q-row-padding.cu
REF_SRC_SHA=535d90b856ed1062aa7b8a105eb2c5f236c450826e65496e646d0d5a27eb8aaf
SEEDS=17,23,42,2024,314159,20260801,20260815,20260829
: "$SLURM_JOB_ID";: "$C2_REFERENCE_PLUGIN";: "$C2_EXPECTED_REFERENCE_SHA";: "$C2_CANDIDATE_PLUGIN";: "$C2_EXPECTED_CANDIDATE_SHA";: "$C2_CANDIDATE_SOURCE";: "$C2_EXPECTED_CANDIDATE_SOURCE_SHA";: "$C2_CANDIDATE_RESOURCE_GATE";: "$C2_EXPECTED_CANDIDATE_RESOURCE_GATE_SHA";: "$C2_CANDIDATE_PROVENANCE";: "$C2_EXPECTED_CANDIDATE_PROVENANCE_SHA"
: "$C2_DIRECTED_RESULT";: "$C2_EXPECTED_DIRECTED_RESULT_SHA";: "$C2_DIRECTED_FINAL_STATUS";: "$C2_EXPECTED_DIRECTED_FINAL_STATUS_SHA";: "$C2_DIRECTED_FINAL_STATUS_SIDECAR";: "$C2_EXPECTED_DIRECTED_FINAL_STATUS_SIDECAR_SHA";: "$C2_DIRECTED_OUTPUTS_MANIFEST";: "$C2_EXPECTED_DIRECTED_OUTPUTS_MANIFEST_SHA";: "$C2_DIRECTED_RUN_MANIFEST";: "$C2_EXPECTED_DIRECTED_RUN_MANIFEST_SHA"
: "$C2_V14_AOT_FINAL_STATUS";: "$C2_EXPECTED_V14_AOT_FINAL_STATUS_SHA";: "$C2_V14_AOT_FINAL_STATUS_SIDECAR";: "$C2_EXPECTED_V14_AOT_FINAL_STATUS_SIDECAR_SHA";: "$C2_V14_AOT_OUTPUTS_MANIFEST";: "$C2_EXPECTED_V14_AOT_OUTPUTS_MANIFEST_SHA"
: "$C2_V14_EXPECTED_REG";: "$C2_V14_EXPECTED_STACK";: "$C2_V14_EXPECTED_SHARED";: "$C2_V14_EXPECTED_LOCAL";: "$C2_EXPECTED_RUNTIME_SCRIPT_SHA"
for x in "$C2_EXPECTED_REFERENCE_SHA" "$C2_EXPECTED_CANDIDATE_SHA" "$C2_EXPECTED_CANDIDATE_SOURCE_SHA" "$C2_EXPECTED_CANDIDATE_RESOURCE_GATE_SHA" "$C2_EXPECTED_CANDIDATE_PROVENANCE_SHA" "$C2_EXPECTED_DIRECTED_RESULT_SHA" "$C2_EXPECTED_DIRECTED_FINAL_STATUS_SHA" "$C2_EXPECTED_DIRECTED_FINAL_STATUS_SIDECAR_SHA" "$C2_EXPECTED_DIRECTED_OUTPUTS_MANIFEST_SHA" "$C2_EXPECTED_DIRECTED_RUN_MANIFEST_SHA" "$C2_EXPECTED_V14_AOT_FINAL_STATUS_SHA" "$C2_EXPECTED_V14_AOT_FINAL_STATUS_SIDECAR_SHA" "$C2_EXPECTED_V14_AOT_OUTPUTS_MANIFEST_SHA" "$C2_EXPECTED_RUNTIME_SCRIPT_SHA";do [[ "$x" =~ ^[0-9a-f]{64}$ ]]||exit 2;done
for x in "$C2_V14_EXPECTED_REG" "$C2_V14_EXPECTED_STACK" "$C2_V14_EXPECTED_SHARED" "$C2_V14_EXPECTED_LOCAL";do [[ "$x" =~ ^[0-9]+$ ]]||exit 2;done
((C2_V14_EXPECTED_STACK==0&&C2_V14_EXPECTED_SHARED==33184&&C2_V14_EXPECTED_LOCAL==0))||exit 2
[[ "$C2_REFERENCE_PLUGIN" == "$REF"&&"$C2_EXPECTED_REFERENCE_SHA" == "$REF_SHA"&&-f "$REF"&&-f "$REF_SRC"&&-f "$BASE"&&-d "$ROOT"&&-z "$(find "$ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]||exit 2
for x in "$C2_CANDIDATE_PLUGIN" "$C2_CANDIDATE_SOURCE" "$C2_CANDIDATE_RESOURCE_GATE" "$C2_CANDIDATE_PROVENANCE" "$C2_V14_AOT_FINAL_STATUS" "$C2_V14_AOT_FINAL_STATUS_SIDECAR" "$C2_V14_AOT_OUTPUTS_MANIFEST" "$C2_DIRECTED_RESULT" "$C2_DIRECTED_FINAL_STATUS" "$C2_DIRECTED_FINAL_STATUS_SIDECAR" "$C2_DIRECTED_OUTPUTS_MANIFEST" "$C2_DIRECTED_RUN_MANIFEST";do [[ "$x" == /*&&-f "$x" ]]||exit 2;done
DIR=$ROOT/job$SLURM_JOB_ID;[[ ! -e "$DIR" ]]||exit 2;mkdir -p "$DIR";SCRIPT=$(readlink -f "$0");COPY=$DIR/submitted-script-job$SLURM_JOB_ID.slurm;STATUS=$DIR/final-status-job$SLURM_JOB_ID.txt;SIDE=$DIR/final-status-job$SLURM_JOB_ID.sha256
export PATH=/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin:/usr/bin:/bin PYTHONPYCACHEPREFIX=$DIR/pycache
UUID=
sel(){ d=$CUDA_VISIBLE_DEVICES;[[ "$d" != *,* && ( "$d" =~ ^[0-9]+$ || "$d" =~ ^GPU-[0-9A-Fa-f-]+$ ) ]]||return 1;printf '%s\n' "$d";}
gpu(){ nvidia-smi --id "$(sel)" --query-gpu=uuid --format=csv,noheader,nounits|sed -n '1p';}
apps(){ nvidia-smi --id "$(sel)" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null|sed '/^[[:space:]]*$/d';}
snap(){ tag=$1;out=$DIR/gpu-$tag-job$SLURM_JOB_ID.txt;{ printf '===== %s UTC=%s =====\n' "$tag" "$(date -u +%FT%TZ)";nvidia-smi --id "$(sel)" --query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,compute_cap --format=csv,noheader,nounits;printf 'COMPUTE_APPS\n';apps;}>"$out";cat "$out";}
end(){ original=$?;rc=$original;err=0;trap - EXIT;set +e;case "$original" in 0|2|3);;*)rc=2;err=1;;esac;snap final-post||err=1;[[ -z "$(apps)"&&-n "$UUID"&&"$(gpu)"=="$UUID" ]]||err=1;[[ $err -eq 0 ]]||rc=2;printf 'FINAL_RC=%s ORIGINAL_RC=%s FINALIZER_ERROR=%s UTC=%s\n' "$rc" "$original" "$err" "$(date -u +%FT%TZ)">"$STATUS";sha256sum "$STATUS" "$DIR/gpu-final-post-job$SLURM_JOB_ID.txt">"$SIDE"||rc=2;exit "$rc";};trap end EXIT
exec > >(tee "$DIR/driver-job$SLURM_JOB_ID.log") 2>&1
printf '%s  %s\n' "$REF_SHA" "$REF" "$REF_SRC_SHA" "$REF_SRC" "$C2_EXPECTED_CANDIDATE_SHA" "$C2_CANDIDATE_PLUGIN" "$C2_EXPECTED_CANDIDATE_SOURCE_SHA" "$C2_CANDIDATE_SOURCE" "$C2_EXPECTED_CANDIDATE_RESOURCE_GATE_SHA" "$C2_CANDIDATE_RESOURCE_GATE" "$C2_EXPECTED_CANDIDATE_PROVENANCE_SHA" "$C2_CANDIDATE_PROVENANCE" "$C2_EXPECTED_V14_AOT_FINAL_STATUS_SHA" "$C2_V14_AOT_FINAL_STATUS" "$C2_EXPECTED_V14_AOT_FINAL_STATUS_SIDECAR_SHA" "$C2_V14_AOT_FINAL_STATUS_SIDECAR" "$C2_EXPECTED_V14_AOT_OUTPUTS_MANIFEST_SHA" "$C2_V14_AOT_OUTPUTS_MANIFEST" "$C2_EXPECTED_DIRECTED_RESULT_SHA" "$C2_DIRECTED_RESULT" "$C2_EXPECTED_DIRECTED_FINAL_STATUS_SHA" "$C2_DIRECTED_FINAL_STATUS" "$C2_EXPECTED_DIRECTED_FINAL_STATUS_SIDECAR_SHA" "$C2_DIRECTED_FINAL_STATUS_SIDECAR" "$C2_EXPECTED_DIRECTED_OUTPUTS_MANIFEST_SHA" "$C2_DIRECTED_OUTPUTS_MANIFEST" "$C2_EXPECTED_DIRECTED_RUN_MANIFEST_SHA" "$C2_DIRECTED_RUN_MANIFEST" "$BASE_SHA" "$BASE" "$C2_EXPECTED_RUNTIME_SCRIPT_SHA" "$SCRIPT"|sha256sum -c -||exit 2
cp -- "$SCRIPT" "$COPY";chmod a-w "$COPY";printf '%s  %s\n' "$C2_EXPECTED_RUNTIME_SCRIPT_SHA" "$COPY"|sha256sum -c -||exit 2
snap pre||exit 2;line=$(nvidia-smi --id "$(sel)" --query-gpu=name,compute_cap --format=csv,noheader,nounits);[[ "$line" == *B300*&&"$line" == *10.3*&&-z "$(apps)" ]]||exit 2;UUID=$(gpu);[[ "$UUID" =~ ^GPU- ]]||exit 2
RES=$(grep -m1 -E 'REG:[[:space:]]*[0-9]+.*STACK:[[:space:]]*[0-9]+.*SHARED:[[:space:]]*[0-9]+.*LOCAL:[[:space:]]*[0-9]+' "$C2_CANDIDATE_RESOURCE_GATE")||exit 2
rr(){ sed -nE "s/.*$1:[[:space:]]*([0-9]+).*/\\1/p"<<<"$RES";};REG=$(rr REG);STACK=$(rr STACK);SHARED=$(rr SHARED);LOCAL=$(rr LOCAL);((REG==C2_V14_EXPECTED_REG&&STACK==0&&STACK==C2_V14_EXPECTED_STACK&&SHARED==33184&&SHARED==C2_V14_EXPECTED_SHARED&&LOCAL==0&&LOCAL==C2_V14_EXPECTED_LOCAL))||exit 2
grep -Fx "$C2_EXPECTED_CANDIDATE_SHA  $C2_CANDIDATE_PLUGIN" "$C2_CANDIDATE_PROVENANCE"||exit 2
"$PY" - "$C2_CANDIDATE_PLUGIN" "$C2_CANDIDATE_SOURCE" "$C2_CANDIDATE_RESOURCE_GATE" "$C2_CANDIDATE_PROVENANCE" "$C2_V14_AOT_FINAL_STATUS" "$C2_V14_AOT_FINAL_STATUS_SIDECAR" "$C2_V14_AOT_OUTPUTS_MANIFEST"<<'PY'||exit 2
import hashlib,re,sys
from pathlib import Path
p,s,r,v,f,z,o=(Path(x).resolve()for x in sys.argv[1:])
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
"$PY" - "$C2_DIRECTED_RESULT" "$C2_DIRECTED_FINAL_STATUS" "$C2_DIRECTED_FINAL_STATUS_SIDECAR" "$C2_DIRECTED_OUTPUTS_MANIFEST" "$C2_DIRECTED_RUN_MANIFEST" "$C2_CANDIDATE_PLUGIN" "$C2_EXPECTED_CANDIDATE_SHA"<<'PY'||exit 2
import hashlib,json,re,sys
from pathlib import Path
r,s,z,o,m,p,h=(Path(x).resolve()for x in sys.argv[1:7]);h=sys.argv[7]
def d(x):return hashlib.sha256(x.read_bytes()).hexdigest()
def mm(x):
 a={}
 for q in x.read_text().splitlines():
  v,sep,n=q.partition('  ');n=Path(n).resolve();assert sep and re.fullmatch('[0-9a-f]{64}',v)and n.is_file()and d(n)==v;a[n]=v
 return a
g=re.fullmatch(r'v14-kv-stage-padding-directed-job([0-9]+)\.json',r.name);assert g;j=g.group(1);root=r.parent;assert root.name==f'job{j}'and s==root/f'final-status-job{j}.txt'and z==root/f'final-status-job{j}.sha256'and o==root/f'outputs-job{j}.sha256'and m==root/f'run-manifest-job{j}.json'
st=dict(q.split('=',1)for q in s.read_text().split()if'='in q);assert st.get('FINAL_RC')=='0'and st.get('ORIGINAL_RC')=='0'and st.get('FINALIZER_ERROR')=='0'and s in mm(z);om=mm(o);assert all(q in om for q in(r,m,root/f'inputs-job{j}.sha256',root/f'gpu-pre-job{j}.txt',root/f'gpu-post-job{j}.txt'))
rm=json.loads(m.read_text());assert rm['schema']=='c2-native-c2-v14-kv-stage-padding-directed-input-manifest-v1'and rm['candidate']=='v14-kv-stage-padding';e={q['label']:q for q in rm['inputs']};assert Path(e['v14_kv_stage_padding_plugin']['path']).resolve()==p and e['v14_kv_stage_padding_plugin']['sha256']==h
x=json.loads(r.read_text());assert x['schema']=='c2-native-c2-v6-register-numerator-directed-v1'and x['all_gates_pass']is True and x['rank_directed_case_count']==4 and x['softmax_directed_case_count']==2 and x['all_invalid']['all_zero']is True and x['profiler']['exactly_one_dispatcher_event']is True and x['profiler']['exactly_one_native_kernel_event']is True
PY
WR=$DIR/write-fixtures.py;WK=$DIR/worker.py;FIX=$DIR/fixtures-8-seeds.pt;META=$DIR/fixtures-8-seeds-meta.json
cat >"$WR"<<'PY'
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--base',type=Path);p.add_argument('--seeds');p.add_argument('--fixture',type=Path);p.add_argument('--meta',type=Path);a=p.parse_args();sp=importlib.util.spec_from_file_location('v14_fix',a.base);b=importlib.util.module_from_spec(sp);sys.modules[sp.name]=b;sp.loader.exec_module(b)
x=type('A',(),{'num_physical_pages':64,'max_logical_pages':32,'scale':1/(128**.5),'q_scale':.25,'k_scale':.25,'v_scale':.5,'atol':1e-4,'rtol':1e-3})();c=b._validate_args(x);ss=[int(q)for q in a.seeds.split(',')];assert len(ss)==8 and len(set(ss))==8;obj={'schema':'c2-native-c2-v14-kv-stage-padding-fixed-fixtures-v1','seeds':ss,'contract':c.__dict__,'rows':[{'seed':q,'inputs':tuple(v.cpu().contiguous()for v in b._make_inputs(c,q))}for q in ss]};torch.save(obj,a.fixture);a.meta.write_text(json.dumps({'schema':obj['schema'],'seeds':ss,'fixture_sha256':hashlib.sha256(a.fixture.read_bytes()).hexdigest(),'base_harness_sha256':hashlib.sha256(a.base.read_bytes()).hexdigest(),'contract':obj['contract']},sort_keys=True,indent=2)+'\n')
PY
cat >"$WK"<<'PY'
import argparse,hashlib,importlib.util,json,os,statistics,subprocess,sys,torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--label');p.add_argument('--library',type=Path);p.add_argument('--base',type=Path);p.add_argument('--fixture',type=Path);p.add_argument('--meta',type=Path);a=p.parse_args();d=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();sp=importlib.util.spec_from_file_location('v14_w',a.base);b=importlib.util.module_from_spec(sp);sys.modules[sp.name]=b;sp.loader.exec_module(b);meta=json.loads(a.meta.read_text());f=torch.load(a.fixture,map_location='cpu');assert d(a.fixture)==meta['fixture_sha256']and d(a.base)==meta['base_harness_sha256']and torch.cuda.get_device_capability()==(10,3);u=subprocess.check_output(['nvidia-smi','--id',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=uuid','--format=csv,noheader,nounits'],text=True).strip();torch.ops.load_library(str(a.library));op=torch.ops._C.native_c2_msa_decode;c=b.Contract(**f['contract']);out=[]
for row in f['rows']:
 i=tuple(q.cuda().contiguous()for q in row['inputs']);z=torch.full((b._BATCH,b._QUERY_HEADS,b._HEAD_DIM),float('nan'),device='cuda',dtype=torch.bfloat16);ptr=z.data_ptr();ret=op(z,*i,c.scale,c.q_scale,c.k_scale,c.v_scale);torch.cuda.synchronize();ok=bool(torch.isfinite(z).all())and bool(torch.allclose(z.float(),b._oracle(*i,c,torch.float32),atol=1e-4,rtol=1e-3))and ptr==z.data_ptr()and ret is None;copy=z.clone()
 for _ in range(3):q=torch.empty_like(z);ok=ok and op(q,*i,c.scale,c.q_scale,c.k_scale,c.v_scale)is None;torch.cuda.synchronize();ok=ok and torch.equal(copy,q)
 for _ in range(30):assert op(z,*i,c.scale,c.q_scale,c.k_scale,c.v_scale)is None
 torch.cuda.synchronize();ev=[]
 for _ in range(200):s,e=torch.cuda.Event(True),torch.cuda.Event(True);s.record();assert op(z,*i,c.scale,c.q_scale,c.k_scale,c.v_scale)is None;e.record();ev.append((s,e))
 torch.cuda.synchronize();out.append({'seed':row['seed'],'all_gates_pass':ok and bool(torch.isfinite(z).all()),'mean_ms':statistics.fmean(s.elapsed_time(e)for s,e in ev)})
print(json.dumps({'schema':'c2-native-c2-v14-kv-stage-padding-fixture-worker-v1','label':a.label,'library_sha256':d(a.library),'fixture_sha256':d(a.fixture),'fixture_meta_sha256':d(a.meta),'base_harness_sha256':d(a.base),'gpu_uuid':u,'all_gates_pass':all(q['all_gates_pass']for q in out),'seeds':out,'measurement':{'warmup':30,'iterations':200}},sort_keys=True))
PY
"$PY" "$WR" --base "$BASE" --seeds "$SEEDS" --fixture "$FIX" --meta "$META";for tag in A B1 B2 A2;do case "$tag" in A|A2)lib=$REF;;*)lib=$C2_CANDIDATE_PLUGIN;;esac;timeout --foreground --preserve-status --kill-after=10s 1200s "$PY" "$WK" --label "$tag" --library "$lib" --base "$BASE" --fixture "$FIX" --meta "$META">"$DIR/$tag-job$SLURM_JOB_ID.json"||exit 2;done
AGG=$DIR/aggregate-job$SLURM_JOB_ID.json;DEC=$DIR/decision-job$SLURM_JOB_ID.json;set +e
"$PY" - "$DIR" "$SLURM_JOB_ID" "$REF_SHA" "$C2_EXPECTED_CANDIDATE_SHA" "$FIX" "$META" "$AGG" "$DEC"<<'PY'
import hashlib,json,math,random,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);j=sys.argv[2];rh,ch=sys.argv[3:5];fix,meta,agg,dec=map(Path,sys.argv[5:]);d=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();x={n:json.loads((root/f'{n}-job{j}.json').read_text())for n in('A','B1','B2','A2')}
assert all(q['all_gates_pass']is True and len(q['seeds'])==8 and all(v['all_gates_pass']and math.isfinite(float(v['mean_ms']))and float(v['mean_ms'])>0 for v in q['seeds'])and q['fixture_sha256']==d(fix)and q['fixture_meta_sha256']==d(meta)for q in x.values())and x['A']['library_sha256']==rh==x['A2']['library_sha256']and x['B1']['library_sha256']==ch==x['B2']['library_sha256']and len({q['gpu_uuid']for q in x.values()})==1
seeds=[q['seed']for q in x['A']['seeds']];assert all([q['seed']for q in v['seeds']]==seeds for v in x.values());a=[statistics.fmean((x['A']['seeds'][i]['mean_ms'],x['A2']['seeds'][i]['mean_ms']))for i in range(8)];b=[statistics.fmean((x['B1']['seeds'][i]['mean_ms'],x['B2']['seeds'][i]['mean_ms']))for i in range(8)];im=[1-y/z for z,y in zip(a,b,strict=True)];point=statistics.median(im);rng=random.Random(20260831);boot=sorted(statistics.median([im[rng.randrange(8)]for _ in range(8)])for _ in range(10000));lcb=boot[249];ok=point>.03 and lcb>0
g={'schema':'c2-native-plugin-v14-kv-stage-padding-vs-v12-3pct-incremental-aggregate-v1','round_order':['A','B1','B2','A2'],'process_isolation':{'same_slurm_step':True,'same_python_process':False},'allocated_gpu_uuid':x['A']['gpu_uuid'],'fixture_sha256':d(fix),'fixture_meta_sha256':d(meta),'seeds':seeds,'reference_seed_mean_ms':a,'candidate_seed_mean_ms':b,'paired_seed_improvement_fraction':im,'paired_median_improvement_fraction':point,'bootstrap':{'method':'10,000 deterministic paired-seed resamples; statistic=median(1-candidate/reference)','seed':20260831,'confidence':.95,'lower_bound_fraction':lcb,'required_lower_bound_fraction':0.0},'threshold_provenance':'Strict >3% point improvement plus bootstrap LCB >0, single-change v12-to-v14 gate.'};Path(agg).write_text(json.dumps(g,sort_keys=True,indent=2)+'\n');q={'schema':'c2-native-plugin-v14-kv-stage-padding-vs-v12-3pct-incremental-decision-v1','accepted':ok,'all_correctness_and_resource_identity_gates_pass':True,'performance':{'paired_median_improvement_fraction':point,'required_improvement_fraction':.03,'required_improvement_relation':'strictly greater than','bootstrap_95pct_lower_bound_fraction':lcb,'required_bootstrap_lower_bound_fraction':0.0,'required_bootstrap_lower_bound_relation':'strictly greater than'},'aggregate_sha256':d(Path(agg)),'rejection_policy':'Return 3 if valid but non-significant; retain v12. Return 2 if identity/correctness/comparability fails.','cross_gpu_claim':'none; same physical GPU only.'};Path(dec).write_text(json.dumps(q,sort_keys=True,indent=2)+'\n');raise SystemExit(0 if ok else 3)
PY
RC=$?;set -e;snap post||exit 2;[[ -z "$(apps)" ]]||exit 2
IN=$DIR/inputs-job$SLURM_JOB_ID.sha256;sha256sum "$REF" "$REF_SRC" "$C2_CANDIDATE_PLUGIN" "$C2_CANDIDATE_SOURCE" "$C2_CANDIDATE_RESOURCE_GATE" "$C2_CANDIDATE_PROVENANCE" "$C2_V14_AOT_FINAL_STATUS" "$C2_V14_AOT_FINAL_STATUS_SIDECAR" "$C2_V14_AOT_OUTPUTS_MANIFEST" "$C2_DIRECTED_RESULT" "$C2_DIRECTED_FINAL_STATUS" "$C2_DIRECTED_FINAL_STATUS_SIDECAR" "$C2_DIRECTED_OUTPUTS_MANIFEST" "$C2_DIRECTED_RUN_MANIFEST" "$BASE" "$COPY" "$WR" "$WK">$IN;sha256sum -c "$IN"||exit 2
OUT=$DIR/outputs-job$SLURM_JOB_ID.sha256;sha256sum "$DIR/A-job$SLURM_JOB_ID.json" "$DIR/B1-job$SLURM_JOB_ID.json" "$DIR/B2-job$SLURM_JOB_ID.json" "$DIR/A2-job$SLURM_JOB_ID.json" "$AGG" "$DEC" "$FIX" "$META" "$WR" "$WK" "$IN" "$DIR/gpu-pre-job$SLURM_JOB_ID.txt" "$DIR/gpu-post-job$SLURM_JOB_ID.txt">$OUT;sha256sum -c "$OUT";exit "$RC"
