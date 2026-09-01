# FLA public dispatcher 与 TP8 挑战

## 目的

验证 opt-in 的 C1 backend 是否能通过 FLA public `chunk_kda` 接口工作，并检查
state contract、backend priority、非白名单回退和 TP8 critical-path 计时。KDA
本身按 head 独立，因此每个 rank 使用一个 `H=12` shard；只有八个 rank 同时运行
时，结果才可称为完整 TP8。

## 输入与状态契约

TP8 runner 是 [`run_tp8_fla.py`](run_tp8_fla.py)，固定 `T=8192,H=12,K=128,V=128`
和 `B=1`，比较 pinned FlashKDA、C1 auto backend、FLA public `chunk_kda` 三条
路径。三种 public contract 为 `none`、`fp32_final_only`、`fp32_both`；每次样本
在 NCCL barrier 后计时，critical path 取观测 rank 的逐样本最大值。`T=257` 的
故意非白名单调用必须由 C1 backend 接管后，在 launch 前选择 baseline。

前置 state-contract runner 是 [`run_state_contracts.py`](run_state_contracts.py)，
固定 H12、T8192，另覆盖 `bf16_both`，并与 torch reference、三种 kernel path
比较。dispatcher 的规则和 opt-in 开关分别在
[`auto_dispatch.py`](auto_dispatch.py) 与 [`fla_backend.py`](fla_backend.py)。

## 2026-08-30 v5 production policy

当前 source SHA-256 为 `auto_dispatch.py`
`9633179c6a5e8ce61b1857a102f81074c6bdfdc63163e13980f40c24d6583c29`、
`fla_backend.py` `152ac4ba897c1887963553d592456b7f988c89105e6f6cb507a44757058d5fd1`；
registry compatibility token 是
`c1-b300-flash-kda-skew-fp32-both-20260830-v5`。27 个 dispatcher policy tests 与
11 个 varlen-metadata tests 均通过。

在既有白名单之外，本轮重点冻结以下窄点：

- fixed `B=1,H=12,T=8191,K=V=128` 的 `none`、`fp32_final_only` 精确选择
  `vshard4_p2`；真实 production A1/A2/chain 已通过，见
  [`challenge_tail8191_production_freeze`](../challenge_tail8191_production_freeze/README.zh-CN.md)；
- packed offsets `(0,1,2,3,4,5,12288)`、`H=12,T_total=12288` 的 `none`、
  `fp32_final_only` 分别选择 `vshard2_p2`，`fp32_both` 当前 source 选择
  `vshard4_p2`；第三格先过 test-only A1/A2，随后在 fresh job12770/12771 完成真实
  production A1/A2/freeze，见
  [`challenge_varlen_fp32_both_production_freeze`](../challenge_varlen_fp32_both_production_freeze/README.zh-CN.md)；
- B7 none-only 新协议已得到有效负结果，因此 B7 全部保持 baseline。

variant 选择现在是 exact-symbol fail-closed：如果 policy 指定的 v4 symbol 不存在，launch
前直接回 baseline，绝不静默替换为 v2。未列 shape/state/offsets、未审计 SO 或错误设备身份
也都在 launch 前回退。

## B300 运行前提与命令

需要 FLA checkout、A02 checkout、已构建的 B300 `flash_kda_C` extension、CUDA/
Python headers、`torchrun`，以及按目标 world size 分配的空闲 B300 GPU。CPU policy
测试和 TP8 clean 审计由 [`run_clean_tp8_fla_audit.sh`](run_clean_tp8_fla_audit.sh)
执行；GPU 运行必须显式授权：

```bash
export A02_ROOT=/path/to/assignment02
export PATCHED_ROOT=/path/to/patched/flashkda
export FLA_ROOT=/path/to/fla
export PYTHON_BIN=/path/to/python
export PYTHON_INCLUDE=/path/to/python/include
export CUDA_HOME=/usr/local/cuda
export LABEL=rerun
export AUDIT_WORLD_SIZE=8
export TARGET_TP_DEGREE=8
export C1_TP8_DISPATCH_GPU_AUTHORIZED=1
bash challenge_tp8_dispatch/run_clean_tp8_fla_audit.sh --authorized-by-parent
```

脚本固定 `torchrun --nproc_per_node=8`、warmup 30、samples 300，并要求恰好
观测到目标数量的 B300 SM103 GPU。state-contract 前置审计使用
[`run_clean_state_contracts_audit.sh`](run_clean_state_contracts_audit.sh)，其命令
和结果路径由脚本固定。

## 已有结果与停止门

H12 state-contract 结果
[`c1_tp8_dispatch_b300_sm103a_h12_realstate_r1_h12_state_contracts.json`](results/c1_tp8_dispatch_b300_sm103a_h12_realstate_r1_h12_state_contracts.json)
的 `exact_gate_pass=true`；四种 contract 均完成 baseline、vshard2-P2、
vshard4-P2 与 torch reference 的 exact 比较，且 vshard4-P2 在该前置测量中胜出。

FLA 单 shard 的两次结果中，推荐引用较新的
[`c1_tp8_dispatch_b300_sm103a_tp8_shard_r2_tp8_fla.json`](results/c1_tp8_dispatch_b300_sm103a_tp8_shard_r2_tp8_fla.json)
及其 [`job10667 audit`](results/c1_tp8_dispatch_b300_sm103a_tp8_shard_r2_tp8_fla_job10667.log)：
`exact_gate_pass=true`，CPU policy 9 项通过，三个 public contract 逐位一致，
`T=257` 正确回退，且进程退出 `FINAL_RC=0`。但是 JSON 明确记录
`observed_concurrent_ranks=1`、`tp8_concurrent_gate_pass=false`，coverage 是
`local_tp_shard_only_due_to_scheduler_quota`。

完整 TP8 的停止门为：world size=8、target TP degree=8、恰好八张 B300 同时可见，
所有 rank exact、critical-path JSON 完整、POST audit 清洁。未满足时只能报告
“FLA-level 单 shard 集成”，不能报告八 rank TP8 性能或端到端模型结果。

## 当前阻塞与边界

2026-08-30 的
[`c1_tp8_quota_reprobe_20260830.txt`](results/c1_tp8_quota_reprobe_20260830.txt) 记录
2/4/8-GPU `sbatch --test-only` 均被 `QOSMaxGRESPerUser` 拒绝，只有 1 GPU 可申请；因此
现有结果不等价于 8-rank 并发。[模型资产探针](../challenge_long_context_quality/results/c1_real_model_asset_probe_20260830.txt)
还在可访问根下找到 0 个大权重候选和 0 个 C1 模型评测 launcher，所以不能由此 challenge
声称真实模型质量或完整 TP8；探针不推断不可访问外部存储。禁用 opt-in 的行为由 CPU policy
test 覆盖，不把它写成 GPU public-call 证据。

## 证据索引

- dispatcher：[`auto_dispatch.py`](auto_dispatch.py)、[`fla_backend.py`](fla_backend.py)
- CPU/FLA policy tests：[`test_auto_dispatch_policy.py`](test_auto_dispatch_policy.py)
- state-contract runner/audit：[`run_state_contracts.py`](run_state_contracts.py)、[`run_clean_state_contracts_audit.sh`](run_clean_state_contracts_audit.sh)
- TP8 runner/audit：[`run_tp8_fla.py`](run_tp8_fla.py)、[`run_clean_tp8_fla_audit.sh`](run_clean_tp8_fla_audit.sh)
- state-contract JSON：[`h12_state_contracts.json`](results/c1_tp8_dispatch_b300_sm103a_h12_realstate_r1_h12_state_contracts.json)
- 单 shard JSON/日志：[`r2 JSON`](results/c1_tp8_dispatch_b300_sm103a_tp8_shard_r2_tp8_fla.json)、[`job10667`](results/c1_tp8_dispatch_b300_sm103a_tp8_shard_r2_tp8_fla_job10667.log)
- 最新配额证据：[`2026-08-30 quota reprobe`](results/c1_tp8_quota_reprobe_20260830.txt)
- 模型资产边界：[`2026-08-30 asset probe`](../challenge_long_context_quality/results/c1_real_model_asset_probe_20260830.txt)
- 报告对应章节：[`REPORT.zh-CN.md`](../REPORT.zh-CN.md#1-fla-public-调用链与保守自动-dispatch)
