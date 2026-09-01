# tail、batch 与 packed-varlen 挑战

## 目的

在同一份 B300 `sm_103a` extension 上覆盖固定长度尾块、batch 和 packed-varlen，
检查 vshard2-P2/vshard4-P2 在非 16 对齐长度、不同 batch、不同 state contract 下
的 correctness 和性能。该 challenge 同时提供一个小规模 compute-sanitizer memcheck，
用于发现 tail/batch/varlen 的越界或非法访问；它不替代完整任意 `cu_seqlens` 穷举。

## 输入与状态契约

主 runner 是 [`run_varlen_tail.py`](run_varlen_tail.py)，固定 `H=12,K=128,V=128`
和 `chunk=16`，覆盖 11 个 shape：fixed `T=1/15/17/31/127/8191`，
`B=2,T=17`，`B=4,T=127/2048`，以及 lengths 为 `[1,15,17,31]` 的短 packed
case 和 `[17,511,1024,1300,2049,3291]` 的 mixed `total T=8192` case。每个
shape 测试 `none`、`bf16_both`、`fp32_both`、`fp32_final_only`；部分非对齐/varlen
case 还与 torch reference 比较。主计时使用 warmup 30、samples 300。

## B300 运行前提与命令

需要一张空闲 B300、CUDA/Python headers、`ninja`、已构建的 `flash_kda_C` extension
和 reference root。clean 主审计由 [`run_clean_varlen_tail_audit.sh`](run_clean_varlen_tail_audit.sh)
执行，GPU 运行需要显式授权：

```bash
export A02_ROOT=/path/to/assignment02
export PATCHED_ROOT=/path/to/patched/flashkda
export REFERENCE_ROOT=/path/to/reference/assignment02
export PYTHON_BIN=/path/to/python
export PYTHON_INCLUDE=/path/to/python/include
export CUDA_HOME=/usr/local/cuda
export LABEL=rerun
export C1_VARLEN_TAIL_GPU_AUTHORIZED=1
bash challenge_varlen_tail/run_clean_varlen_tail_audit.sh --authorized-by-parent
```

sanitizer 审计由 [`run_clean_sanitizer_audit.sh`](run_clean_sanitizer_audit.sh) 执行，
调用 [`run_sanitizer_smoke.py`](run_sanitizer_smoke.py)，使用
`compute-sanitizer --tool memcheck --padding 32 --error-exitcode 86`。两条脚本均在
退出时检查 compute apps 为空和显存归零。

## 已有结果与停止门

主结果 [`c1_varlen_tail_b300_sm103a_r1.json`](results/c1_varlen_tail_b300_sm103a_r1.json)
记录 B300 capability `10.3`、`exact_gate_pass=true`；11 个 shape 的 output/final
state 在 baseline、vshard2-P2、vshard4-P2 间逐位一致，Torch reference 子集也通过。
主 clean 日志 [`c1_varlen_tail_b300_sm103a_r1_job10629.log`](results/c1_varlen_tail_b300_sm103a_r1_job10629.log)
以 `FINAL_RC=0` 结束。

性能 winner 依赖输入：fixed `T=8191` 的 none/FP32-final-only 选 vshard4-P2；
`B=4,T=2048` 和 mixed varlen `total T=8192` 的 none/FP32-final-only 选
vshard2-P2。因此不能把 vshard4 扩成无条件默认。runtime dispatch 的非白名单
varlen 调用 output exact 且选择 baseline，原因是
`varlen_cu_seqlens_not_whitelisted`。

sanitizer 结果 [`c1_varlen_sanitizer_b300_sm103a_r1.json`](results/c1_varlen_sanitizer_b300_sm103a_r1.json)
的 `exact_gate_pass=true`，覆盖 5 个 fixed/padded-tail/batch/varlen smoke case；
日志 [`c1_varlen_sanitizer_b300_sm103a_r1_job10636.log`](results/c1_varlen_sanitizer_b300_sm103a_r1_job10636.log)
以 `FINAL_RC=0` 结束且无 memcheck error。停止门是 exact、reference 子集、JSON
完整、sanitizer 无 error、以及 post audit 全部通过；任一失败不得发布候选。

## 当前边界

这些证据闭合的是列出的 11 个 shape 和 5 个 sanitizer smoke case，不是任意
`cu_seqlens` 的生产级性能模型。未列出的 packed lengths、未列入 fixed 精确表的
`B>1` 组合和其他架构仍保守回 baseline；不能从少数 case 外推 winner，也不能把
单卡结果称为 TP8 或真实模型质量。

## 证据索引

- 主 runner/audit：[`run_varlen_tail.py`](run_varlen_tail.py)、[`run_clean_varlen_tail_audit.sh`](run_clean_varlen_tail_audit.sh)
- sanitizer runner/audit：[`run_sanitizer_smoke.py`](run_sanitizer_smoke.py)、[`run_clean_sanitizer_audit.sh`](run_clean_sanitizer_audit.sh)
- 主 JSON：[`c1_varlen_tail_b300_sm103a_r1.json`](results/c1_varlen_tail_b300_sm103a_r1.json)
- 主日志：[`c1_varlen_tail_b300_sm103a_r1_job10629.log`](results/c1_varlen_tail_b300_sm103a_r1_job10629.log)
- sanitizer JSON/日志：[`c1_varlen_sanitizer_b300_sm103a_r1.json`](results/c1_varlen_sanitizer_b300_sm103a_r1.json)、[`job10636`](results/c1_varlen_sanitizer_b300_sm103a_r1_job10636.log)
- 报告对应章节：[`REPORT.zh-CN.md`](../REPORT.zh-CN.md#2-tailbatch-与-varlen正确性通过winner-不可一刀切)
