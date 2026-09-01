# B300 跨长度与 head/state 交互挑战

## 目的

在同一份 B300 `sm_103a` extension 上，比较 pinned baseline、vshard2-P2 和
vshard4-P2。实验把长度、head 数和 state contract 分开扫，用于回答“winner 是否
能从一个 shape 外推到所有长度/head/state”的问题；结果只用于保守的精确 policy，
不自动发布未测组合。

## 输入与状态契约

runner 是 [`run_cross_length.py`](run_cross_length.py)，输入由 harness 生成，固定
`B=1,K=128,V=128`。H12 长度轴为
`T=128,256,512,1024,2048,4096,8192,16384,32768,65536`；每个长度测试
`none`、`bf16_both`、`fp32_both`、`fp32_final_only` 四种 state contract。
另外只对 `bf16_both` 测试 `T=2048/32768` 与
`H=1/12/37/38/64/96` 的交互点。计时为三条路径循环轮换，每个 CUDA event 包含
一次 wrapper call 和 workspace allocation。

## B300 运行前提与命令

需要一张空闲 B300，CUDA、选定 Python 的 `ninja`/`Python.h`、已构建的
`flash_kda_C` extension，以及 A02、patched root。clean 环境审计和授权门由
[`run_clean_cross_length_audit.sh`](run_clean_cross_length_audit.sh) 执行；脚本拒绝
缺少显式授权的 GPU 运行：

```bash
export A02_ROOT=/path/to/assignment02
export PATCHED_ROOT=/path/to/patched/flashkda
export PYTHON_BIN=/path/to/python
export PYTHON_INCLUDE=/path/to/python/include
export CUDA_HOME=/usr/local/cuda
export LABEL=rerun
export C1_CROSS_LENGTH_GPU_AUTHORIZED=1
bash challenge_cross_length/run_clean_cross_length_audit.sh --authorized-by-parent
```

脚本默认 `--warmup 30 --samples 300`，通过 `--json` 写入结果；运行前、扫完后和
退出时检查 compute apps 为空且显存为零。

## 已有结果与停止门

[`c1_cross_length_b300_sm103a_r1.json`](results/c1_cross_length_b300_sm103a_r1.json)
记录设备为 NVIDIA B300 SXM6 AC、capability `10.3`，`warmup=30`、`samples=300`，
`exact_gate_pass=true`、`complete=true`。H12 的十个长度及两组交互点均完成；所有
被测 state contract 的 output/final state 与 baseline 逐位一致。H12 长度轴上
vshard4-P2 为 winner；交互点中 H1/12/37 选 vshard4-P2，H38/64/96 选
vshard2-P2。

对应的 clean B300 日志是
[`c1_cross_length_b300_sm103a_r1_job10654.log`](results/c1_cross_length_b300_sm103a_r1_job10654.log)，
其中 `FINAL_RC=0`，且 `POST_COMPUTE_APPS` 为空、`POST_MEMORY_USED_MIB=0`。
停止条件是 exact gate、完整写出 JSON、以及 post audit 全部通过；任一条件失败时
不得把该 sweep 写入 policy。

## 边界

这是一张 B300 单卡、固定 `B=1,K=128,V=128` 的实测表，不是任意长度/head 的
性能模型，也不覆盖其他 GPU 架构。`T=257`、H11/T4096 等未测组合必须回退；
本目录没有独立 sanitizer 或多 GPU TP8 证据，相关结论应引用各自 challenge 目录。

## 证据索引

- runner：[`run_cross_length.py`](run_cross_length.py)
- clean audit：[`run_clean_cross_length_audit.sh`](run_clean_cross_length_audit.sh)
- 原始结果：[`c1_cross_length_b300_sm103a_r1.json`](results/c1_cross_length_b300_sm103a_r1.json)
- clean 日志：[`c1_cross_length_b300_sm103a_r1_job10654.log`](results/c1_cross_length_b300_sm103a_r1_job10654.log)
- 报告对应章节：[`REPORT.zh-CN.md`](../REPORT.zh-CN.md#4-跨长度表只把实测格点写进-policy)
