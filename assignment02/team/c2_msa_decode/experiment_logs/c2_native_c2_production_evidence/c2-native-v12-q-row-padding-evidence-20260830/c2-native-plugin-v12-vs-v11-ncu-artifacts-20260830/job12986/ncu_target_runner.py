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
