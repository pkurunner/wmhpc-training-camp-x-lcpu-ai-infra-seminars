# C1 B300：vshard4-P2 的 InputStages=4 候选（否决）

S=4 的独立 candidate 已在 B300 完成 fresh build 和一次 clean GPU allocation（job
12216）。它满足资源与逐位正确性门，但没有满足预注册的全格性能门；因此**禁止第二次
allocation**，不登记 dispatch，生产 `vshard4-P2S3` 保持不变。

| 变量 | 含义 | 本次实测取值 |
| --- | --- | --- |
| `B,T,H` | batch、每序列 token 数、每卡 head 数 | 正式 `1,8192,12`；小矩阵 `T=256,H=1/2/4` |
| `K,V,V_s` | key/value 宽度、单 CTA value shard 宽度 | `128,128,32` |
| `P` | Phase-6 软件预取环深度 | `2` |
| `S` | K2 input-pipeline stage 数 | 对照 `S=3`，candidate `S=4` |
| `q` | 延迟分位数 | P50、P95、P99；每路径保留 1000 个 CUDA-event 样本 |
| `r_q` | `vshard4-P2S3` latency 除以 P2S4 latency | 每个 contract、每个 `q` 的发布门为 `r_q≥1.02` |

## 构建与正确性

job 12216 从干净 FlashKDA `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b` 创建 fresh
SM103a worktree `flashkda-inputstages4-1ce47ea-b300-r1`。one-shot generator 在同一
extension 中保留 baseline、vshard2-P2S3、vshard4-P2S3，并将 candidate 隔离为
`fwd_vshard4_p2s4`；所以 S=4 的资源与测时不会混入 S=3。

P2S4 有 14 个 ptxas 实例，寄存器范围 56–59，全部零 spill；formal fixed
BF16 initial+final-state 实例是 59 registers、9 barriers、0 stack、0 spill。CUBIN
resource audit 对 14 个 P2S4 实例均记录 1024 B **静态** shared-memory record，构建资源门
通过；动态 shared memory 仍由 launch 配置，这个记录不能解释为 kernel 的总 shared memory。

在同一 extension（SHA-256
`55b8078acbb5536cc927af5a878a43a5f30bd2bb83e8e8468a31d6639b8c6d21`）中，
`H=1/2/4` 小矩阵的四种 contract，以及 H12 的 `none`、BF16-both、FP32-both、
FP32-final-only，P2S4 output 和适用的 final state 均相对 baseline 逐位一致。GPU
audit 的 PRE/POST 都是 0 MiB、无 compute app，`FINAL_RC=0`。

## H12 full-call 性能与停止判定

每个格子以四路径 cyclic rotation 计时：baseline、vshard2-P2S3、vshard4-P2S3、
P2S4；每个 CUDA event 包含一次完整 public wrapper 调用和 workspace allocation。下表
是 `r_q = P2S3 / P2S4`，大于 1 才表示 S=4 更快：

| H12 contract | P50 `r_q` | P95 `r_q` | P99 `r_q` | 是否全为 `≥1.02` |
| --- | ---: | ---: | ---: | --- |
| none | 0.997762 | 0.998689 | 1.001421 | 否 |
| BF16-both | 1.003970 | 1.003327 | 1.000347 | 否 |
| FP32-both | 1.004289 | 1.005954 | 1.006730 | 否 |
| FP32-final-only | 0.998438 | 0.997336 | 0.994234 | 否 |

第一 allocation 的跨 12 个性能格最小值分别为 P50 **0.997762x**、P95
**0.997336x**、P99 **0.994234x**，均低于 1.02x。故 analyzer 的
`performance_pass=false`、`publication_eligible=false`；不是“还差一次 repeat”，而是
已经命中预注册停止条件，第二 allocation 不应再消耗 B300 时间。candidate 保持
non-production，既有 P2S3 production/dispatch 不改变。

## 可审计工件

| 工件 | SHA-256 |
| --- | --- |
| [fresh build log](results/c1_inputstages4_build_b300_sm103a_r1.log) | `2d76fd4eba0a600a4a2fc27fe572f6621ce5413ea4dabf1887cc91d5e9cd13bd` |
| [CUBIN resource dump](results/c1_inputstages4_build_b300_sm103a_r1.cuobjdump.txt) | `bceef078a4b7c4b5a3ed1ffab19dd990aa5896df4c162ac0e6fc3e8c17506500` |
| [ptxas resource ledger](results/c1_inputstages4_build_b300_sm103a_r1.ptxas.json) | `b36601360c296ae58f7f9e33d39a20b29330ca0ed5550c6d58bb18d819d8ad88` |
| [small H1/H2/H4 all-contract exact](results/c1_inputstages4_b300_sm103a_h12_r1_small_matrix.json) | `64120a6b9586e60337f766c6f16f03869d4d477d257e00fc8eb29612b0172880` |
| [H12 all-contract exact and raw samples](results/c1_inputstages4_b300_sm103a_h12_r1_h12_all_contracts.json) | `99e578cbe27dee43a339e30ec32dba84733343b87776cb435067831844e074ef` |
| [clean GPU audit log, job 12216](results/c1_inputstages4_b300_sm103a_h12_r1_job12216.log) | `bbb95a34807a81823d407f92b949fdc0d269c8e43ac12b466dbc3627e3d9109c` |
| [first-allocation pre-registered gate](results/c1_inputstages4_b300_sm103a_h12_r1_one_allocation_gate.json) | `d12c30cb606d1791a5c52702668df676c16b032b683b0a3c8664847ad897400f` |

七个工件都来自 job 12216 的 fresh build/clean audit 链；JSON 账本记录 candidate、
extension、contract、allocation 和 raw timing，日志记录 source identity、PTXAS、清卡
前后状态及最终返回码。
