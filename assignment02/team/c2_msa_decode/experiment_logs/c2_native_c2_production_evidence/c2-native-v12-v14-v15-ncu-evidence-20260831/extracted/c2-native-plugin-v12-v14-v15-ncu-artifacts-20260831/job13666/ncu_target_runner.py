import argparse,hashlib,importlib.util,json,os,platform,subprocess,sys,traceback
from pathlib import Path
import torch
def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def base(p):
 s=importlib.util.spec_from_file_location('c2_ncu_harness',p); assert s and s.loader; m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m)
 for n in ('_BATCH','_QUERY_HEADS','_HEAD_DIM','_make_inputs','_oracle','_validate_args'): assert hasattr(m,n),n
 return m
def th(t):
 t=t.detach().contiguous().cpu(); x=hashlib.sha256(); x.update(str(t.dtype).encode()); x.update(repr(tuple(t.shape)).encode()); x.update(t.view(torch.uint8).numpy().tobytes()); return x.hexdigest()
def uuid():
 v=os.environ.get('CUDA_VISIBLE_DEVICES','').split(','); assert len(v)==1 and v[0]
 r=subprocess.check_output(['nvidia-smi','--id',v[0],'--query-gpu=uuid','--format=csv,noheader,nounits'],text=True).strip(); assert r.startswith('GPU-'); return r
def call(op,out,ins,c): return op(out,*ins,c.scale,c.q_scale,c.k_scale,c.v_scale)
def record(a):
 out={'schema':'c2-native-v12-v14-v15-ncu-target-v1','version':a.version,'mode':a.mode,'all_gates_pass':False}
 try:
  assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_capability(0)==(10,3)
  assert Path(a.library).is_absolute() and h(a.library)==a.library_sha256 and h(a.base_harness)==a.base_harness_sha256
  b=base(a.base_harness); c=b._validate_args(type('Args',(),{'num_physical_pages':64,'max_logical_pages':32,'scale':1/(128**.5),'q_scale':.25,'k_scale':.25,'v_scale':.5,'atol':1e-4,'rtol':1e-3})())
  torch.ops.load_library(a.library); assert torch._C._dispatch_has_kernel_for_dispatch_key('_C::native_c2_msa_decode','CUDA')
  ins=b._make_inputs(c,a.seed); y=torch.full((b._BATCH,b._QUERY_HEADS,b._HEAD_DIM),float('nan'),device='cuda',dtype=torch.bfloat16); p=y.data_ptr(); ret=call(torch.ops._C.native_c2_msa_decode,y,ins,c); torch.cuda.synchronize()
  checks={'return_is_none':ret is None,'pointer_unchanged':y.data_ptr()==p,'finite_output':bool(torch.isfinite(y).all().item())}
  out.update({'operator_library':str(Path(a.library).resolve()),'operator_library_sha256':h(a.library),'base_harness_sha256':h(a.base_harness),'seed':a.seed,'input_tensor_sha256':[th(x) for x in ins],'contract':{'batch':b._BATCH,'head_dim':b._HEAD_DIM,'kv_heads':4,'q_heads':b._QUERY_HEADS,'page_size':128,'topk':16,'num_physical_pages':64,'max_logical_pages':32,'scale':c.scale,'q_scale':c.q_scale,'k_scale':c.k_scale,'v_scale':c.v_scale},'caller_output':checks,'device':{'name':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),'gpu_uuid':uuid()},'environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda},'dispatch':'direct_torch.ops._C.native_c2_msa_decode','no_monkeypatch':True})
  out['input_manifest_sha256']=hashlib.sha256(''.join(out['input_tensor_sha256']).encode()).hexdigest()
  if a.mode=='ncu': out['logical_dispatch_actions']=1; out['all_gates_pass']=all(checks.values()); print(json.dumps(out,sort_keys=True)); return
  ref=b._oracle(*ins,c,torch.float64); actual=y.to(torch.float64); z=torch.full_like(y,float('nan')); q=z.data_ptr()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof: rr=call(torch.ops._C.native_c2_msa_decode,z,ins,c); torch.cuda.synchronize()
  ev=prof.key_averages(); ds=[e for e in ev if str(e.key)=='_C::native_c2_msa_decode']; ks=[e for e in ev if 'native_c2_msa_decode_kernel' in str(e.key)]
  out['correctness']={'allclose':bool(torch.allclose(actual,ref,atol=1e-4,rtol=1e-3)),'oracle_dtype':'float64'}; out['profiler']={'exactly_one_dispatcher_event':len(ds)==1 and ds[0].count==1,'exactly_one_native_kernel_event':len(ks)==1 and ks[0].count==1,'return_is_none':rr is None,'pointer_unchanged':z.data_ptr()==q}; out['all_gates_pass']=all(checks.values()) and out['correctness']['allclose'] and all(out['profiler'].values()); print(json.dumps(out,sort_keys=True))
 except Exception as e: out['error']=str(e); out['traceback']=traceback.format_exc(); print(json.dumps(out,sort_keys=True)); raise
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--mode',choices=('validate','ncu'),required=True); p.add_argument('--version',choices=('v12','v14','v15'),required=True); p.add_argument('--library',type=Path,required=True); p.add_argument('--library-sha256',required=True); p.add_argument('--base-harness',type=Path,required=True); p.add_argument('--base-harness-sha256',required=True); p.add_argument('--seed',type=int,required=True); record(p.parse_args())
