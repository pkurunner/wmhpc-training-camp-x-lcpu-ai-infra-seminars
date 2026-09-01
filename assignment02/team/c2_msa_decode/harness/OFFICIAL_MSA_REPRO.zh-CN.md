# 官方 MiniMax MSA / CUTLASS 对照复现

本实验没有把 Triton 替身命名为 CUTLASS。B300 job 4308 使用
[MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) 的真实
`fmha_sm100_plan` / `fmha_sm100` API，固定版本如下：

| 项目 | 固定值 |
| --- | --- |
| MSA commit | `80434d7f67877c6570ca19cac444b84bc9855dac` |
| CUTLASS submodule | `eb61c911471867a5fd2466bfd8f29306cea6ebf8` |
| GPU | B300 SXM6 AC，CC 10.3 |
| Torch / CUDA | `2.13.0+cu130` / 13.0 |
| `fmha_sm100` | 0.1.1 editable install |

服务器安装步骤：

```bash
git clone --recursive https://github.com/MiniMax-AI/MSA.git \
  /home/lcpu/85117379/msa-official
git -C /home/lcpu/85117379/msa-official checkout \
  80434d7f67877c6570ca19cac444b84bc9855dac
git -C /home/lcpu/85117379/msa-official submodule update --init --recursive

uv pip install --python assignment02/.venv/bin/python \
  nvidia-cutlass-dsl quack-kernels apache-tvm-ffi pybind11 cuda-python
uv pip install --no-deps --python assignment02/.venv/bin/python \
  -e /home/lcpu/85117379/msa-official
```

严格 q=1 crossover 使用
[`official_msa_cutlass_bench.py`](official_msa_cutlass_bench.py)；它先以独立
FP32 selected-page oracle 验证随机物理 page table 与随机 per-head top-k，
正确性失败时不输出有效 timing。清卡包装为
[`run_official_msa_q1_audit.sh`](run_official_msa_q1_audit.sh)。例如：

```bash
cd assignment02/team/c2_msa_decode
source ../../.venv/bin/activate
export PYTHONPATH=/home/lcpu/85117379/msa-official/python
export BATCH=16 KV_LEN=4096
bash harness/run_official_msa_q1_audit.sh \
  "$PWD" /home/lcpu/85117379/msa-official b300_b16
```

另以官方未修改的 `bench_sparse_attention_ops.py` 跑了 q=8、B=16 的原生
`sparse_decode` section，原始 CSV/日志也一并保存。

证据边界：任务快照来自 vLLM `d4da0c5`，其中 batch≥16 是当时完整服务集成的
静态门槛；这里可安装并实测的是更新后的 MiniMax 官方 MSA `80434d7`。因此本实验
可检验“当前官方 core kernel 是否仍在 16 才 crossover”，但不能冒充相同 vLLM
revision 的 A/B。版本差异、plan/metadata 更新与服务调度都必须纳入最终解释。
