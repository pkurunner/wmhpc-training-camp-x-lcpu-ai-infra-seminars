# C2 prepared 同合同 Triton 调参扫描（独立实验目录）

本目录不改动 `challenge/`、基线 wrapper 或既有报告。它以相同的
`DecodeProblem`、固定种子、调用方预分配输出、计时前构造的 persistent
workspace，扫描现有 vLLM Triton JIT kernel 的三个受支持选项：
`num_stages`、`launch_pdl`/`USE_PDL`、`maxnreg`。

| 项 | 含义 |
| --- | --- |
| `B` | decode batch，候选值 `1,4,8,16` |
| `C` | `num_topk_chunks`；`selected` 复用当前 mode-aware policy |
| `s` | Triton `num_stages`，编译期 software-pipeline 深度 |
| `r` | Triton `maxnreg`；`none` 表示不传该选项，保留编译器默认值 |
| PDL | CUDA Programmatic Dependent Launch；`auto/on/off` 分别为平台判断、强制尝试、禁用 |
| `t` | 一次 prepared decode 的 CUDA-event 延迟（微秒） |

## 静态 API 检查

在 C2 根目录执行：

```bash
PYTHONPATH=. .venv/bin/python -m challenge_v2.cli static-check
```

目标环境的 Triton 3.7.1 中，`triton.runtime.autotuner.Config` 和 NVIDIA
`CUDAOptions` 都声明 `maxnreg`，后者也声明 `launch_pdl`；该命令只验证 API。
真正的编译/运行兼容性由下述每个候选的 correctness gate 判定。

## 一键 gate + compact sweep（建议先跑）

```bash
PYTHONPATH=. .venv/bin/python -m challenge_v2.cli sweep \
  --storage-mode bf16 --all-batches --chunks selected --sweep compact \
  --warmup 20 --repetitions 41 --seed 20260819 \
  --output team/c2_msa_decode/experiment_logs/c2_tuned_bf16_compact.json
```

`compact` 包含当前配置（`s=3, PDL=auto, r=none`）并逐一改变 stage、PDL
或 register ceiling，控制 JIT 编译数量。每个候选在计时前运行独立 FP32
selected-page causal-attention oracle；失败/不受支持项会以 `rejected` 保留在
JSON 中而不计时。每个样本都只有一次 runner 调用夹在一对 CUDA event 中；
warmup、JIT 编译、输出/workspace 分配和 oracle 全在事件范围外。

`summary.winner_speedup` 是相对于同一 JSON 内 **当前 prepared 等价候选**
（`s=3, PDL=auto, r=none`）的比值，
`strict_10_percent_target_met` 只有在该值至少为 `1.10` 时为真。
这避免与不同机器、不同 event 合同或旧日志直接比较。

## 全网格复验或 FP8

compact 找到潜在 winner 后，可用目标组合做小网格复验，例如：

```bash
PYTHONPATH=. .venv/bin/python -m challenge_v2.cli sweep \
  --storage-mode bf16 --all-batches --chunks selected --sweep grid \
  --stages 2,3,4 --pdl-modes auto,on,off --maxnregs none,64,96,128 \
  --warmup 30 --repetitions 101 --seed 20260819 \
  --output team/c2_msa_decode/experiment_logs/c2_tuned_bf16_grid.json
```

若 compact 在某个 batch 只有很小收益，可先在该 batch 做有界的针对性
Cartesian（30 个 JIT 配置），它专门覆盖低 stage、关闭 PDL 和中等 register
ceiling 的交互：

```bash
PYTHONPATH=. .venv/bin/python -m challenge_v2.cli sweep \
  --storage-mode bf16 --batch 4 --chunks selected --sweep grid \
  --stages 1,2,3,4,5 --pdl-modes off,on --maxnregs none,128,160 \
  --warmup 20 --repetitions 41 --seed 20260819 \
  --output team/c2_msa_decode/experiment_logs/c2_tuned_b4_targeted.json
```

默认 `decode/merge_num_warps=4`，以保持与当前 prepared 的同合同控制组。
需要在胜出 stage/PDL/register 组合附近再试线程数时，添加
`--decode-warps 2,4,8`（C=1 不走 merge；多 chunk 时可再添加
`--merge-warps 2,4,8`）。warp 候选也被逐个 oracle gate，而不能直接以
不通过正确性的低 event 时间取胜。

FP8 如需研究，使用同一命令替换 `--storage-mode fp8-scalar` 或
`fp8-token`，或加入 `--all-modes`。这些是独立探索证据；只有固定 winning
config 后，重新运行完整 gate 并与同合同 baseline 复测，才可以写入最终报告。

## 解释边界

该扫描没有改变问题、selected-page 语义、输出所有权或 workspace 生命周期；
它不包含 host 端 metadata 准备、tensor 分配、JIT 编译和 oracle。`on` 模式
不会伪造成功：若目标 GPU/驱动不接受 PDL，候选将显式 `rejected`。同理
`maxnreg` 的静态支持不等于每个 ceiling 可以通过编译或更快，必须以 JSON
中的 correctness 和 event 数据为准。

## GQA head-shard（C=1 的 B=4/8 并行度候选）

`head_shard.py` 复制的是 decode kernel 的计算语义，而非输入或输出 ABI。
原 kernel 的一个 CTA 负责同一个 KV head 所属的 16 个 Q heads；新 kernel 用
`S=2/4` 把它们切成每 CTA `16/S=8/4` 个 head，网格从
`(total_q, Hkv)` 扩为 `(total_q, Hkv*S)`。令 `y` 为第二维 program id，则
`kv_head=y//S`、`shard=y%S`、`q_head_start=kv_head*16+shard*(16/S)`；KV/top-k
仍按 `kv_head` 读取，而 output/LSE 写入各自的 `q_head_start` 区间，因此写入
不相交且 C=1 不需要 merge。

```bash
PYTHONPATH=. .venv/bin/python -m challenge_v2.head_shard_cli \
  --batch 4 --storage-mode bf16 --shards 2,4 --stages 2,3,4 \
  --pdl-modes off --maxnregs none --warps 4 --warmup 20 --repetitions 41 \
  --seed 20260819 --output team/c2_msa_decode/experiment_logs/c2_head_shard_b4.json
```

该命令包含一个原 prepared 的 C=1 同合同控制组；每个 shard 候选先经过同一
独立 FP32 oracle，再按一调用一 event 计时。小 M 维 `tl.dot` 的编译可行性
取决于目标 Triton/SM100 后端，不能事先假定；编译失败将完整记录为
`rejected`，不构成性能结论。

## C=1 online softmax：删除无消费者的 LSE

对 C=1，decode partial 已是全局输出，merge 不会启动。因此一般 split-K kernel
中每个 selected page 的 `log2(exp2(...)+l)` 更新、末尾 LSE store 和整块
`[1,Q,H]` LSE 工作区都没有消费者。`c1_no_lse.py` 只保留稳定 online state：

| 变量 | 含义 |
| --- | --- |
| `m_i` | 当前已处理 token 的行最大 score（base-2 域） |
| `l_i` | 以 `m_i` 为基准的 softmax 分母 |
| `acc_i` | 以同一基准缩放的 value 加权和 |
| `m_new` | 加入一页 token 后的新行最大值 |
| `alpha` | 重缩放因子 `exp2(m_i-m_new)` |

每页更新 `acc_i=alpha*acc_i+P@V`、`l_i=alpha*l_i+sum(P)`，最后直接写
`out=acc_i/l_i`。没有 LSE allocation 或 store；这与一般 kernel 的 C=1 输出
数学等价，但仍必须用独立 FP32 oracle 验证。

```bash
PYTHONPATH=. .venv/bin/python -m challenge_v2.c1_no_lse_cli \
  --batch 4 --storage-mode bf16 --shards 1,2,4 --stages 2,3,4 \
  --pdl-modes off --maxnregs none --warps 4 --warmup 20 --repetitions 41 \
  --seed 20260819 --output team/c2_msa_decode/experiment_logs/c2_c1_no_lse_b4.json
```

其中 `shards=1` 是不分 GQA head 的 no-LSE 控制；`2/4` 是其输出不相交的
head-shard 扩展。JSON 同时包含当前 prepared C=1 控制项，所有比较仅在同一
seed、同一 CUDA-event 合同内报告。

### 下一轮 clean targeted 复验（B=4/8）

此前 B=4/8 的探索轮受外来训练进程影响，**只保留其中的正确性 gate 和候选
方向，不使用任何性能数字，也不写入最终性能表**。gate 表明 no-LSE 的输出能
通过独立 FP32 oracle；因此 clean 轮只在以下两组小范围组合中搜索，而不是
展开耗时的大笛卡尔积：

| 组 | 固定结构 | 扫描项 |
| --- | --- | --- |
| A | `shards=1, stages=5, PDL=on` | `warps={2,4}` × `maxnreg={none,64,96,128,160}` |
| B | `shards=2, stages=4, PDL=off` | `warps={2,4}` × `maxnreg={none,64,96,128,160}` |

每组只有 10 个 no-LSE 候选，且每次运行都会另建当前 prepared C=1 的同合同
控制项。以下命令的 `B` 替换为 `4` 或 `8`，并分别更换输出文件名：

```bash
PYTHONPATH=. .venv/bin/python -m challenge_v2.c1_no_lse_cli \
  --batch B --storage-mode bf16 --shards 1 --stages 5 --pdl-modes on \
  --maxnregs none,64,96,128,160 --warps 2,4 --warmup 30 --repetitions 101 \
  --seed 20260819 --output team/c2_msa_decode/experiment_logs/optimization_v2/c2_nolse_bB_s1_s5_pdlon_regwarp_clean.json

PYTHONPATH=. .venv/bin/python -m challenge_v2.c1_no_lse_cli \
  --batch B --storage-mode bf16 --shards 2 --stages 4 --pdl-modes off \
  --maxnregs none,64,96,128,160 --warps 2,4 --warmup 30 --repetitions 101 \
  --seed 20260819 --output team/c2_msa_decode/experiment_logs/optimization_v2/c2_nolse_bB_s2_s4_pdloff_regwarp_clean.json
```

clean 轮开始前和结束后都需记录 `nvidia-smi` compute-apps；若出现本实验之外的
PID，则该轮只能作为正确性/候选探索，不得以 event 延迟申报加速。最终只接受
同一输入 seed、同一 caller-owned output、同一 persistent-workspace 生命周期、
相同单调用 CUDA-event 协议下，correctness PASS 且相对同轮控制达到目标的结果。

## 已冻结终验：RTX 5090 / SM120 的 BF16 C=1 no-LSE

最终候选没有从扫参最小值中直接申报，而是在一轮独立的
[B=4 AB/BA freeze gate](results/5090_sm120_nolse_s1_freeze_gate_job7001) 前冻结为
`G=1, warps=4, stages=3, PDL=off, maxnreg=none`。这里 `G=1` 表示一个 KV
head 的 16 个 Q heads 留在一个 CTA；这避免 `G=2` 时同一 KV 页被两个 CTA
重复读取。相对控制组仍是当前 prepared 的 BF16 `C=1`（`warps=4, stages=3,
PDL=auto, maxnreg=none`）。候选只在 `C=1` 删除无消费者的 LSE/workspace 路径，
不用于 FP8，也不把 5090 数字外推到 B300。

为保留 job 7001/7002 的核心源码逐 byte 身份，
`c1_no_lse_abba_cli.py` 内部的无参数 `DEFAULT_CANDIDATE` 仍是更早的
`G=2,warps=2,stages=4` 探索配置；它不是本节终验默认。最终
`run_nolse_abba_clean_audit.sh` 的默认值以及历史 clean 命令都显式传入上面的
`G=1,warps=4,stages=3,PDL=off` 冻结配置，并由 JSON validator 逐字段检查。
因此直接运行 CLI 而不传候选参数只能视为探索，不能复现本节表格。

| 符号 | 含义 | 终验取值 |
| --- | --- | --- |
| `B` | decode batch | `1,4,8,16` |
| `C` | selected-page split 数 | `1` |
| `G` | 每 KV head 的 Q-head CTA 分片数 | `1` |
| `s` | Triton software-pipeline stages | `3` |
| `t_A,t_B` | control/candidate 的单调用 CUDA-event 延迟 | 每路径 202 个样本 |

每个 `B` 使用独立固定 seed、caller-owned output、计时外的 persistent workspace，
先通过独立 FP32 selected-page causal-attention oracle（`rtol=atol=0.03`），再进行
101 个 `ABBA` 对：`control→candidate→candidate→control`。因此每个路径有 202
个 event 样本，既报告合并中位数也记录两种相邻启动顺序。完整 PRE/POST、GPU UUID、
源码 SHA256 与四份机器可读 JSON 已归档在
[`results/5090_sm120_nolse_s1_final_20260820`](results/5090_sm120_nolse_s1_final_20260820)。

| B | control / candidate 合并中位数 (us) | 合并 speedup | AB speedup | BA speedup | FP32 gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 84.560 / 69.392 | **1.218584x** | 1.2050x | 1.2290x | PASS，max abs `6.10e-5` |
| 4 | 116.160 / 90.368 | **1.285411x** | 1.2585x | 1.3216x | PASS，max abs `3.05e-5` |
| 8 | 83.744 / 67.392 | **1.242640x** | 1.2218x | 1.2640x | PASS，max abs `3.05e-5` |
| 16 | 83.808 / 67.456 | **1.242410x** | 1.2372x | 1.2483x | PASS，max abs `6.10e-5` |

该 clean job 的 PRE/POST 均为同一 RTX 5090（SM120）UUID、显存 0 MiB、
`compute-apps` 为空，Slurm `FINAL_RC=0`。修正后的 wrapper 还在四份既有 JSON
上离线严格重算了 config、event 样本数、速度比与 strict gate；四个 context 均为
`true`。早先 `G=2` 的网格 min-of-grid 曾显示很高数字，但固定 AB/BA 只有约
1.04--1.05x，故该数字被保留为探索负例，绝不作为最终结论。

上述 5090 表本身不能替代 B300 的同源码、同 AB/BA 合同清洁审计；它没有声称
5090 的绝对延迟、speedup 或 PDL 结论可迁移到 B300，也没有对 FP8 scalar/token
做 C=1 直接比较。随后完成的 B300 独立复验在下一节。

## B300 / SM103 clean 复验：no-LSE 不替换 current prepared

同一 source-bound no-LSE、相同 `B,C,G,s` 冻结配置和相同 101 个 ABBA 对已在 clean
B300 job 4446 上独立运行。每个 B 仍使用独立固定 seed、caller-owned output、计时外
persistent workspace，并先通过相同的独立 FP32 oracle；控制与候选各有 202 个 event
样本。完整 JSON、PRE/POST 和 source hash 位于
[`results/b300_sm103_nolse_s1_job4446`](results/b300_sm103_nolse_s1_job4446)：

| B | current prepared / no-LSE 合并中位数 (us) | 合并 speedup | FP32 gate | B300 决策 |
| ---: | ---: | ---: | --- | --- |
| 1 | 30.560 / 28.448 | **1.074241x** | PASS | 正，但未达 strict 10% |
| 4 | 28.448 / 30.496 | **0.932844x** | PASS | 保持 current prepared |
| 8 | 28.480 / 30.528 | **0.932914x** | PASS | 保持 current prepared |
| 16 | 30.464 / 32.544 | **0.936087x** | PASS | 保持 current prepared |

该 runner 的 `RC=3` 是 strict 10% gate 对四个 context 的预期失败码，而不是 benchmark
未执行或 correctness 失败。结论因此按架构分开：5090/SM120 可选择此冻结 no-LSE 的
BF16 C=1 专用路径，B300/SM103 在当前证据下保持 current prepared；两者都不外推到 FP8。


## B300 BF16 prepared C=1：stage-5 专用续轮

| 符号 | 下标 / 上标含义 | 取值 |
| --- | --- | --- |
| `B` | batch；无下标 | 冻结终验 `1,16` |
| `C` | selected-page chunk 数；无下标 | `1` |
| `s_3,s_5` | 下标 `3/5` 为 decode pipeline stage | control `3`，candidate `5` |
| `t_3,t_5` | 下标同上 | 每路径 202 个 AB/BA event 样本的合并中位数 |
| `R_{5/3}` | stage-5 相对 stage-3 | `t_3/t_5` |
| `D_s,L_s` | 下标 `s∈{3,5}` | DRAM bytes 与 L2 traffic bytes |
| `I_s^{TC},A_s^{TC}` | 上标 `TC` 为 Tensor pipe；下标为 stage | Tensor 指令数与 Tensor-active |
| `S_s^{long}` | 上标 `long` 为 long-scoreboard stall | 同一 NCU set 内的归一化 stall 指标 |

[prepared_stage_abba_cli.py](prepared_stage_abba_cli.py) 和
[run_prepared_stage5_abba_clean.sh](run_prepared_stage5_abba_clean.sh) 固定 BF16、
`C=1`、warps 4、PDL auto、maxnreg none，只比较 decode stage `3→5`。
每个 context 先用独立 FP32 oracle 做 correctness gate，再执行 30 次 warmup 和
101 个 ABBA pair；JSON 保存全部 202+202 个 raw event 样本。

B300 clean 终验中，`B=1` 在两个不同 GPU UUID 上分别为
`30.560→27.360 us (1.116959x)` 与 `30.848→27.296 us (1.130129x)`，
均通过 strict 10%；`B=16` 为 `30.560→28.512 us (1.071829x)`，正确但未过
10% 门槛。因此这里只允许 `B=1` BF16 C=1 的 opt-in stage-5 specialization，
不改变 B=4/8/16 或 FP8 的默认 stage。

[run_prepared_stage5_ncu_b1_clean.sh](run_prepared_stage5_ncu_b1_clean.sh) 采集 section
级方向性数据；[run_prepared_stage5_ncu_counters_b1_clean.sh](run_prepared_stage5_ncu_counters_b1_clean.sh)
再固定九个 raw counters，并强制 B300、Slurm、PRE/POST 空卡、matching launch、
base units、单一 action、每指标恰好一个 finite 值及 SHA256 manifest。NCU driver
JSON 一律写入 `timing_valid_for_benchmark=false`，不得用 profiler 内延迟替代
非 profiler AB/BA。

最终证据位于
[ABBA 目录](../experiment_logs/prepared_stage5_abba)、
[section NCU 目录](../experiment_logs/prepared_stage5_ncu) 和
[exact-counter 目录](../experiment_logs/prepared_stage5_ncu_counters_v1)；
[机器可读摘要](../experiment_logs/prepared_stage5_followup_summary_20260828.json)
记录了结论与证据 hash。完整讨论、机制解释和仍未完成挑战见
[主报告](../REPORT.zh-CN.md) 文末续轮。
