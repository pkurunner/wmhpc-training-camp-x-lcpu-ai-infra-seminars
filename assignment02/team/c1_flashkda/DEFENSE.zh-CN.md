# C1 答辩稿：B300 上的 FlashKDA 复现、边界与保守发布（约 10 分钟）

## 变量速记（0:00–0:25）

| 符号 | 含义 | 本答辩中的取值 / 下标 |
| --- | --- | --- |
| `B,T,H` | batch、每条序列 token、当前 GPU head 数 | 基线 `B=1,T=8192,H=96`；TP8 单 shard `H=12` |
| `D=K=V` | query/key/value 维度 | 128 |
| `C,c` | chunk 长度与 chunk 时间下标 | `C=16`，`0≤c<ceil(T/C)` |
| `S_c` | chunk `c` 结束后的递推 state | 逻辑 `[K,V]`，API 物理 ABI `[V,K]` |
| `V_s,s` | 单 CTA value-shard 宽度与分片号 | v2: `V_s=64,s∈{0,1}`；v4: `V_s=32,s∈{0,1,2,3}` |
| `M=N_seq×H` | K2 的 sequence-head 工作项数 | 只能作为特征，不能单独推导 winner |
| `o_i` | CPU-authoritative packed offset | 当前 skew 为 `(0,1,2,3,4,5,12288)` |
| `A,p,j` | 资格/production 门的 allocation、fresh PID、repeat 下标 | `A∈{A1,A2}`，每 A 两 PID、每 PID 两轮 |
| `J_A,u_A` | allocation 的 Slurm job 与 GPU UUID | cross-map 为 job12958/12959；两次 UUID 相同 |
| `q,r^{gate}_{A,p,j,q}` | 分位数与资格/production 门的对照/C1 延迟比 | `q∈{P50,P95,P99}`；`r>1` 表示 C1 更快 |
| `δ^{gate}_{A,p,j,q}` | 资格/production 门中 C1 相对对照的裕量 | `r^{gate}-1`；逐格要求至少 2% |
| `c_{map}` | cross-map 预注册的正向 cell | 四个正向 cell；每个 PID/cell 一组 100-sample 对拍，不设 repeat `j` |
| `r^{cm}_{A,p,c_{map},q}` | cross-map 非发布 sentry 的 `pinned/C1` | 仅要求逐格 `r^{cm}>1`，不是 2% 发布门 |

## Slide 1（0:25–1:10）：问题、结论和完成度

题目有三层：B300 复现与指令/NCU 证据、六项讨论的量化结论、以及任选 SM100 路线的可复核挑战。我的结论是：已完成可复核的窄部署子集与多条负结论；没有把单 shard、微基准或候选性能写成通用发布。

| 题目层 | 状态 | 本次答辩的窄结论 |
| --- | --- | --- |
| 1. 复现、official benchmark、NCU/SASS | 完成 | B300 官方 fixed/varlen 复现；主路径是 SM80 HMMA，不是 WGMMA/tcgen05；K2 低 grid 是关键瓶颈。 |
| 2. 六项讨论 | 完成分析；真实模型质量有外部缺口 | 每项都有纸面量化或微基准/NCU/长序列证据；第 5 项的真实模型任务层不在仓库资产内。 |
| 3. SM100 挑战 | 核心 value-shard 路线、停止树与已集成格点冻结完成 | T8191 真实 production chain 已过；skew/FP32-both 也以 fresh job12770/12771 完成真实 production A1/A2/freeze；B7/phase1/V8/S4 负结果保留。 |

总报告是 [REPORT.zh-CN.md](REPORT.zh-CN.md)，任务原文是 [TASK.md](TASK.md)。

## Slide 2（1:10–2:05）：复现和硬件事实

- 固定 FlashKDA `1ce47ea`、CUTLASS `5c149f5`、FLA `a3edffc`，并对 B300 job 的 PRE/POST 空卡、源码与二进制身份做审计。
- 官方 `B=1,T=8192,H=96,D=128` 的 B300 BF16-state 为 **1.0306 ms**；H64 对照为 **0.9411 ms**。fixed 外还实际通过两组 packed-varlen。
- SASS 全库计数是 HMMA=1544、WGMMA=0、UTCOMMA=0、UTCHMMA=0；抽样为 `HMMA.16816.F32.BF16`。所以不能仅因 GPU 是 SM100 就宣称 kernel 已使用新指令。
- full NCU：K1 grid `(512,96,1)`、41.51 waves/SM；K2 grid `(1,96,1)`、0.32 waves/SM、active warps 9.37%。K2 的计算和 DRAM 都未饱和，首先是可并行 CTA 不足/依赖延迟问题。

对应的 baseline、SASS、NCU 原始证据均在 [报告环境与 NCU 章节](REPORT.zh-CN.md#可复核环境与复现)。

## Slide 3（2:05–3:05）：讨论 1–2：为何保持 `CHUNK=16`，为何不把 tcgen05 当发布路线

| 讨论点 | 量化结果 | 裁决 |
| --- | --- | --- |
| 1. `CHUNK=16` | 最坏 gate 衰减 `exp(-5(C-1))`：C16 为 `2.68e-33`，C32/C64 远低于 BF16 可表示范围；实测 C20 首次归零。Neumann 代理 C32/C64 的稠密工作量为 C16 的 8x/64x。 | C16 同时满足数值边界、16×16 运算结构和工作区约束。C32/64 的代理不是重编译后的端到端 FlashKDA。 |
| 2. tcgen05 | 最小 `128×64×64` tile 对一个 `16×16×16` 问题仅 0.78125% 有效 FMA。公平完整 CTA 微基准在 C16/32/64 都是 HMMA 更快，tcgen05 有 48/32/0 个 zero-K。 | tcgen05 指令可运行且 microbench exact，但不等于 tcgen05-KDA 吞吐；在当前结构停止盲目替换。 |

数值范围、Neumann 代理和完整 CTA 的 SASS/计时均可复核于 [讨论 1–2](REPORT.zh-CN.md#六个讨论点)。

## Slide 4（3:05–4:05）：讨论 3–4：状态递推的并行度与 roofline

- K2 在 `c` 方向必须串行推进 `S_c`；可尝试的轴是 head、persistent CTA 或 state 的 value 列。实际挑战选择不切归约轴的 value-shard：每个 CTA 只写 disjoint 的 state value 行。
- 2 CTA/head 在 H64 可加速，但 H96 仅 1.0010x；这说明“更多 CTA”不是通用结论。TP8 `H=12` 才是该方向的有效工作点。
- H12 的严格 tensor-contraction roofline 把理论 HMMA FLOP 与 NCU counter 逐整数绑定：K1 强度 7.3732、memory-roof 分支、效率 34.37%；K2 强度 82.5405 已越过 ridge，但仅 compute roof 的 **3.84%**。它仍是低 grid/依赖延迟受限，而非算力饱和。

并行候选与反例见 [讨论 3](REPORT.zh-CN.md#3-有状态递推下还能从哪里取并行度)，严格 H12 roofline 见 [部署续轮第 5 节](REPORT.zh-CN.md#5-严格-bf16-tensor-contraction-roofline)。

## Slide 5（4:05–5:00）：讨论 5–6：精度证据的层次与发布边界

| 讨论点 | 已完成 | 明确没有宣称 |
| --- | --- | --- |
| 5. BF16 state 精度 | candidate/upstream 与 Torch reference 的 exact；FP32 recurrence oracle 对 H1/H12、random/stress、多 seed、最长 262,144 tokens 的逐段数值误差。 | 没有真实模型、权重、数据、perplexity 或下游任务 harness，故**真实模型质量未完成**；合成 recurrence 不能替代它。 |
| 6. sm100a v2 发布 | 仅 B300 SM10.3、经 extension SHA、shape/state/public API 审计的白名单可 opt-in；未测 shape/state/架构 launch 前回 baseline。 | 不将 sm100a 路径说成跨架构默认，也不把失败候选加入 dispatch。 |

该证据边界与外部阻塞所需资产见 [讨论 5–6](REPORT.zh-CN.md#5-bf16-state-的精度验证) 和 [未完成挑战表](REPORT.zh-CN.md#仍未完成的挑战与后续方向)。

## Slide 6（5:00–6:10）：挑战主线与当前 H12 production dispatch

挑战实现为 value-shard：v2 将 `V=128` 分成两条 64 列路径，v4 分成四条 32 列路径；每条都先过 output/final-state exact，再看完整 public-call P50/P95/P99。

- H12 条件性 current：v4-P2 在两次独立 clean B300 allocation 都是 **0.529472 ms**，相对同轮 v2-P2 的 0.595440/0.595136 ms 为 **1.124592x/1.124018x**。
- 为避免单点外推，`H=1…96` 做了逐整数 exact/reference 与三分位 sweep：H1–37 选 v4-P2，H38–96 选 v2-P2。边界 H37→38 正好使 `4H` CTA grid 从 148 到 152，并观测 K2 duration 跳变。
- fixed `T=8191` 的 none/FP32-final-only 已从 test-only A1/A2 升到 job12592/12593 真实 production A1/A2，最终 chain 为 `production_freeze_passed=true`。
- skew/FP32-both 新资格门的两个 allocation 最差 P99 速度比为 1.053236x/1.051927x；集成 v5 source 后，job12598 因负控隔离缺陷不计 A1，修复版 job12770/12771 再以四个 fresh PID 闭合真实 production route/exact/control/freeze，最终 `eligible_for_production_freeze=true`。
- v5 cross-map 非发布 sentry 又在 job12958/12959 同时复验四个正向 cell 与三个负控；48 个分位比全局最小为 1.046089x，map 不变、chain 通过。两次 job 落在同一 GPU UUID，因此不称跨 GPU 复现，也不据此扩表。
- 生产 policy 因此是带 B300 身份、精确 `H/T/state/offsets` 白名单的 fail-closed 选择；v4 exact symbol 缺失会直接 baseline，不能静默降 v2。它不是 H64/H96 的旧 “+10%” 目标改写，也不是 TP8 全并发结论。

实现与精确白名单在 [auto_dispatch.py](challenge_tp8_dispatch/auto_dispatch.py)，H12 证据在 [报告 H12 续轮](REPORT.zh-CN.md#tp8--h12-续轮vshard4--p2s3-的条件性-current)。

## Slide 7（6:10–7:05）：fixed-batch B5 已发布，B7 只得到负结论

- B5 是此前 B4/B6 间的空洞。B5 的三个 public contract（none、FP32-final-only、FP32-both）先通过两 allocation、raw samples 的 discovery/confirmation，再进入真实 public FLA registry 审计。
- 最终 corrected job11788 的 production freeze 对 **18 个 cell** 给出 13 个正例、5 个负控 baseline，`production_freeze_passed=true`；B5 三个精确格点均发布为 v2-P2。旧类型门错误的 freeze JSON 已明确废弃，不能引用。
- 旧 B7 discovery 的四 ABI exact，`none` 在三路径内部看似胜出，但全契约门因两个 FP32 contract 失败而停止。随后不追认旧样本的新 none-only schema-3 job12570 再测真实 public pinned/C1：四轮的 12 个分位 margin 全负，最差 `-4.14279%`，得到有效 `STOP_keep_production_baseline`。没有 A2、chain 或 B7 map 条目。

B5 release 冻结见 [strict freeze JSON](challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r2_strict.production_freeze.json)；B7 历史 discovery 见 [旧 audit](challenge_fixed_batch_b7/results/c1_fixed_batch_b7_discovery_b300_sm103a_b7_r1.independent_audit.json)，最终 none-only 负结论见 [schema-3 A1 audit](challenge_fixed_batch_b7_none/results/c1_fixed_batch_b7_none_b300_sm103a_b7_none_r3_helper_nobuild_A1.allocation_audit.json)。

## Slide 8（7:05–7:50）：题面点名的 FLA Triton `chunk_kda` 已直接对拍

job12216/r3 显式关闭 backend dispatch，直接执行官方 `test_fwd_vs_fla` 和 `test_fwd_varlen_vs_fla`。candidate FlashKDA、FP64 fused-recurrent gold、未被截获的 FLA Triton `chunk_kda` 各真实调用 20 次（fixed/varlen 各 10 次）；每次 output/final state 都存在，coverage 通过，PRE/AFTER/POST 是 0 MiB，`FINAL_RC=0`。

判定强度严格沿用 upstream：candidate output 对 gold 是 hard tolerance；candidate/chunk state 是 warning，chunk output 是 error-ratio/绘图。因此这里证明 direct Triton 路径执行并量化其误差，不把所有 tensor 都说成 hard assert。

这是题面所需的官方 Triton 对拍/正确性证据，不是 Triton 性能对比、自动 dispatch 发布、完整 TP8 或模型质量。证据为 [r3 JSON](challenge_fla_chunk_validation/results/c1_fla_chunk_validation_b300_sm103a_fla_chunk_r3.json)、[job12216 clean log](challenge_fla_chunk_validation/results/c1_fla_chunk_validation_b300_sm103a_fla_chunk_r3_job12216.log) 和 [挑战说明](challenge_fla_chunk_validation/README.zh-CN.md)。

## Slide 9（7:50–8:35）：InputStages=4 的正确性通过，但性能门失败

S=4 候选保留为 non-production：fresh build 的正式 BF16 instance 零 spill，小形状及 H12 四个 raw contract 都 exact。性能却必须逐 contract、逐分位相对 S=3 达到至少 1.02x，且需两个不同 Slurm job。

当前 job12216 单 allocation 的 S3/S4 速度比分别为：none 0.9978/0.9987/1.0014，BF16 1.0040/1.0033/1.0003，FP32-both 1.0043/1.0060/1.0067，FP32-final-only 0.9984/0.9973/0.9942（P50/P95/P99）。最低仅 **0.9942x**，所以 `performance_pass=false`、`publication_eligible=false`；不为凑第二次 allocation而继续。

同样，phase-1 fragment prefetch 虽 exact、zero-spill 且 shared-memory 证据通过，P50/P95/P99 速度比却低于 1 达约 1.68–1.81 个百分点，也按首轮负门停止。这两项都是完成的负结论，不是未做。

完整 gate 在 [one-allocation audit](challenge_inputstages4/results/c1_inputstages4_b300_sm103a_h12_r1_one_allocation_gate.json)，精确性结果在 [H12 all-contract JSON](challenge_inputstages4/results/c1_inputstages4_b300_sm103a_h12_r1_h12_all_contracts.json)。

## Slide 10（8:35–9:25）：六项讨论完成矩阵

| # | 问题 | 结论 / 证据状态 |
| ---: | --- | --- |
| 1 | CHUNK=16 的数值、Neumann、MMA 形状 | 已量化；C32/64 的数值/工作区/代理代价均不支持当前结构扩大。 |
| 2 | tcgen05 最小 tile 与 C16 | 已量化并有公平 CTA microbench；存在严重 padding，不能推成 KDA 端到端加速。 |
| 3 | stateful recurrence 的并行度 | 已实现并测 v2/v4/V8；H12 有条件性正例，V8/P3/persistent/multi-head 在现结构停止。 |
| 4 | compute 或 memory bound | K1/K2 分开 NCU 与严格 H12 roofline 已完成；K2 不是峰值 compute-bound，而是低 grid/依赖限制。 |
| 5 | BF16 state 精度 | kernel 与合成长序列数值层已完成；真实模型任务质量因资产/launcher 缺失仍外部阻塞。 |
| 6 | 是否发布 sm100a v2 | 已实现 B300-only exact 白名单；policy 27/27、metadata 11/11。T8191 与 skew/FP32-both 的 production A1/A2 均已过；r5 cross-map sentry 以非发布方式闭合且 map 未变；B7/S4/V8/phase1 负例不发布。 |

## Slide 11（9:25–10:00）：诚实的未完成项和交付结论

核心讨论层仍有两项当前权限/资产外部阻塞，以及按证据门**有意不发布**的边界外扩展：

1. 8-rank TP8：QOS `QOSMaxGRESPerUser` 拒绝同时 8 GPU；现有 JSON 只有单 shard，绝不称 full TP8。
2. 真实 Kimi 模型端到端/任务质量：仓库没有模型、权重、数据或完整 TP launcher；不能把 long recurrence 误差替换成 perplexity 或下游任务。
3. B7-none 已由新协议得到有效负结果；其他未白名单 varlen、架构/shape/state 都保持 pre-launch baseline，后续必须各自重过 exact、clean allocation、public API 和稳定分位门。

所以当前交付是：一套可复核的 B300 复现与六项讨论证据、一个 value-shard 挑战、精确 H12/B5/T8191/skew 发布子集、一个 v5 cross-map 非发布 sentry，以及每个失败候选的停止证据；没有把这些窄白名单外推为任意 varlen、full TP8 或跨架构结论。完整外部阻塞和下一成功门在 [TP8 README](challenge_tp8_dispatch/README.zh-CN.md#当前阻塞与边界) 与 [报告未完成表](REPORT.zh-CN.md#仍未完成的挑战与后续方向)。

## 预备问答

**问：为什么 B7 none 变快还不发布？** 那只是旧三路径 discovery 的相对排序，且其全契约门已经失败。更关键的是不追认旧样本的新 none-only job12570：真实 public pinned/C1 的 12 个分位 margin 全负，最差 `-4.14279%`，所以新协议也在 A1 有效否决。

**问：FLA Triton 对拍通过是否代表真实模型质量？** 不代表。它证明 pinned FLA 的 fixed/varlen 官方测试中三条计算路径都真实执行并产出 output/state；按 upstream 原口径，candidate output 通过 hard tolerance，candidate/chunk state 是 warning，chunk output 是 error-ratio/绘图而非 hard assert。模型质量仍需要真实权重、数据和任务指标。

**问：S=4 既然 exact，为什么不试第二次 allocation？** 它在唯一 allocation 的性能门已经失败，最小速度比 0.9942x；预注册停止树禁止把“正确”偷换成“值得发布”。

**问：为什么不是 tcgen05？** 公平完整 CTA 对照中，C16/32/64 都输给 HMMA，且 C16 有 75% 的 `K=64` padding；需要全新 K2 数据流/同步重排的端到端实现才是另一项任务，不能从一条指令名外推。
