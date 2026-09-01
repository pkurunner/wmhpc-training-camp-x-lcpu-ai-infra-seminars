# C1：FlashKDA 在 B300 上的复现、边界与 value-shard 挑战

## 结论与证据范围

FlashKDA `1ce47ea` 在 B300 上能够干净复现，主计算路径的静态指令是
`HMMA.16816.F32.BF16`，而不是 WGMMA 或 tcgen05。官方形状
`[B=1,T=8192,H=96,D=128]` 的 B300 BF16-state 基线均值为 **1.0306 ms**；
在同一固定输入的 `H=64` 对照上为 **0.9411 ms**。重新绑定源码并在 clean B300
窗口完成的 value-shard 终测中，`2 CTA/head` 在 `H=64` 达到 **1.1766x**
中位数加速，而在官方 `H=96` 仅 **1.0010x**；随后同一 H64 路径的 P2 software-prefetch
在 clean B300 上将旧 current 0.799616 ms 更新为 **0.737600 ms**（1.084078x）。它们都只
适用于低每卡 head 数的条件性优化，不能包装成通用收益；对旧 current 再快 10% 的原严格门槛
0.726924 ms 尚未达到。

本轮补上了此前最大的部署证据空白：TP8 理论每卡 `H=12`。同一份新 extension 内的
`vshard4 + P2S3` 在两次独立 clean B300 allocation 中均为 **0.529472 ms**，相对同轮
vshard2-P2 的 0.595440/0.595136 ms 达到 **1.124592x/1.124018x**，且所有 output/
final-state exact gate 通过。随后在同一 B300、同一 extension 上逐整数穷举 `H=1–96`：
96/96 shape 均通过 exact/reference gate，且 P50/P95/P99 三种分位数一致地显示
`H=1–37` 选择 vshard4-P2、`H=38–96` 选择 vshard2-P2。首个 dispatch 阶跃位于
`H=37→38`：B300 有 148 SM，`4H` grid 在 H37 为 148 CTA，而 H38 的 152 CTA 使 K2
NCU duration 从 470.43 μs 跳到 593.63 μs。故这是固定 B300/T/D/state 轴上已穷举的
TP8/低-head 条件性 current，不是对 H64/H96 原严格门槛的改写，也不能跨 shape 或架构
无条件设为默认。

本文只把链接的清卡日志、机器可读 JSON、仓库快照和源码当作事实依据。新增的
[C1 microbench v2](experiment_logs/c1_chunk_tcgen_microbench_b300.json) 以同一
B300 清卡作业验证了 `C=16/32/64` 的 BF16 衰减边界，并以固定并行度测量了稠密
Neumann doubling **代理**；它不是重新编译后的 FlashKDA `CHUNK=32/64` kernel。
同样，tcgen05 的结论不是 tcgen05-KDA 吞吐：job 4340 已完成同逻辑工作、同 CTA
路径的 HMMA/tcgen05 事件对照，并把 exact gate、SASS 和 JSON 一起归档；它量化的
是该固定 tile 的 padding 与完整 CTA 数据/同步路径成本。后续续轮已用 job 10085 的
同一份 K1/K2 NCU raw counter 闭合严格的 **BF16 tensor-contraction** FLOP 口径：K1
落在 memory-roof 分支，K2 的 arithmetic intensity 越过 ridge，但只达到 compute roof
的 3.84%，因此仍是低 grid/依赖延迟受限，而不是接近峰值的 compute saturation。

部署续轮还完成了保守自动 dispatch、FLA public `chunk_kda` 单 TP shard、11 组
tail/batch/varlen 四状态 exact、最长 262,144 token 的独立 FP32 recurrence 对照、
10 个 H12 长度与两个 length/head 切片，以及 V8-P2/P1 的资源与性能停止门。真正的
8-rank TP8 因 Slurm 用户 GPU 配额只能保留为外部阻塞项；单 rank 结果不会被写成 full
TP8。其后的四小时补轮又以 54-case sequence-count 矩阵否决了仅按 `N_seq×H` 推导
winner 的通用规则，并经过 discovery、独立确认、逐 cell 发布门和真实 public FLA
调用链，先将 10 个精确 `B>1,H=12,T=2048` state contract 加入保守白名单；本次八小时
续轮又补齐遗漏的 B5，经过两 allocation raw 门和 18-cell public audit 后扩为 13 个。具体证据边界
和仍未完成项集中列在文末。packed-varlen 的**历史 r6** 更窄：raw 门的 10 个候选在完整 public
`chunk_kda` 计时后只剩 skew `(0,1,2,3,4,5,12288)` 的 `none` 与
FP32-final-only 两个 cell；其余 10 个 layout/state 组合当时均保持 pinned baseline。该轮还完成了
verifier→body 的 one-shot descriptor handoff：真实 public C1 的 `_prepare_varlen` 从每次两次
降为一次，但 body 仍重读 CPU tuple 做 freshness 校验；两项 r6 production freeze 由 job 11590
独立审计通过。随后 job11767 的旧 FP32-both 诊断虽有 6/6 相对门通过，却因该 allocation 的
1.20 ms 绝对尾门全败而按原规则停止；这段历史没有被事后追认为 release 证据。

2026-08-30 的当前 v5 补轮重新从独立协议起步。固定 `T=8191` 的 `none` 与
`fp32_final_only` 先在 test-only A1/A2 的 48 个 repeat×分位格全部通过（每 allocation 24 个；全局最小裕量
35.775%），再由真实 production registry 的 job12592/12593 两个 allocation 与 chain
复验，`production_freeze_passed=true`。skew/FP32-both 也在新的 test-only schema-3
A1/A2 上逐位 exact，两个 allocation 的最差 P50/P95/P99 速度比分别为
`1.105265/1.090924/1.053236` 与 `1.104953/1.093693/1.051927`；它已按 exact tuple/state
集成到最终 production map。真实 production 预试 job12598 的目标 route/exact 本身通过，
随后却因 runner 的邻接-offset 负控复用了 stale one-shot handoff 而失败，故该 job 不计 A1；
隔离修复后，fresh job12770/12771 已以四个新 PID 完成真实 production A1/A2/freeze，最终
artifact 为 `eligible_for_production_freeze=true`、`complete=true`。
为检查 v5 多张白名单之间是否发生集成回归，后续又用只读、非发布的 cross-map sentry 在
job12958/12959 两个不同 Slurm job 上同时复验 4 个正向 cell 与 3 个负控；48 个正向
PID×cell×分位比的全局最小 `pinned/C1` 为 `1.046089`，map digest 前后不变，最终 chain
`production_freeze_passed=true`。两次 allocation 落在同一 GPU UUID，故这不是跨 GPU 证据，也不触发扩表。
fixed-batch 方向则补齐 B5：job11781/11782 的两-allocation raw 门、job11786 的 18-cell
public 集成和修正版 job11788 的 fresh-seed production freeze 均已完成，最终 fixed public 正例由
10 个增至 13 个；严格发布审计只采信修复类型门后的 analyzer，不采信被 reviewer 否决的旧
freeze JSON。

最终完成度补轮又闭合了三个容易混淆的边界。第一，job12216/r3 关闭 backend dispatch 后，
直接强制 pinned FLA Triton `fla/ops/kda/chunk.py`，官方 fixed/varlen 两个测试均通过；candidate、
FP64 fused-recurrent gold 与 Triton chunk 各被实际调用 20 次。第二，遗漏的 fixed B7 先做
2 fresh PID × 2 repeats × 1000 samples 的 discovery：`none` 单元稳定选择 vshard2-P2，但两个
FP32 public contract 都选择 baseline，因此按预注册的全契约门停止。随后另起、不追认 discovery
的 none-only schema-3 协议；job12570 的四 raw ABI、真实 public route 与 helper no-build 均 exact，
但四个 repeat 的 P50/P95/P99 裕量全部为负（最差 `-4.14279%`），得到有效
`STOP_keep_production_baseline`，按门不申请 A2、也不改 production map。
第三，独立 `InputStages=4` 候选 fresh build 为零 spill，small/H12 四契约均 exact，但相对当前
P2S3 的四契约 P50/P95/P99 只有 0.9942x–1.0067x，未过 1.02x 门，故同样停止且不改生产路径。
2026-08-30 的新探针还确认 2/4/8 GPU 均被用户 QOS 拒绝、1 GPU 可申请；可访问根下也没有
大模型权重或 C1 模型评测 launcher。phase-1 fragment-prefetch 也在唯一 allocation 上虽通过
exact/zero-spill/shared-memory 门，但 P50/P95/P99 速度比低于 1 达约 1.68–1.81 个百分点，按预注册规则止于 A1。
因而核心三层任务、六项讨论的可执行证据与负候选停止树已经闭合；skew/FP32-both 的真实
production A1/A2/freeze 也已在修复版协议闭合。外部尚余 full 8-rank TP8 与真实模型质量，
绝不以 test-only、单卡或合成数据冒充。

## 2026-08-31 已闭环新增成果索引

本次整理只收录已有完整机器工件、且结论门已经终止的实验；中途运行、单份 raw 或尚未形成
有效 allocation/chain 的尝试不进入下表，也不据此修改发布结论。

| 方向 | 已闭环证据 | 结论 | 发布边界 |
| --- | --- | --- | --- |
| phase-1 fragment prefetch | [job12401 A1 gate](challenge_phase1_fragment_prefetch/results/c1_phase1pf_phase1pf_a1r3_one_allocation_gate.json)，SHA256 `bc2471d4…92685` | 四契约 exact、zero-spill、shared-memory evidence 均通过；P50/P95/P99 速度比为 `0.981869/0.982007/0.983188` | 完成的负结论；按预注册门止于 A1，不申请 A2、不接 dispatcher |
| fixed B7 none-only | [job12570 A1 audit](challenge_fixed_batch_b7_none/results/c1_fixed_batch_b7_none_b300_sm103a_b7_none_r3_helper_nobuild_A1.allocation_audit.json)，SHA256 `6312e183…e866` | raw/public 均 exact；12 个 repeat×分位 margin 全负，最差 `-4.14279%`；`complete=true` | `STOP_keep_production_baseline`；不申请 A2、无 chain、不新增 B7 map |
| fixed T8191 | [test-only chain](challenge_tail8191_dispatch/results/c1_tail8191_dispatch_b300_sm103a_A1_A2.chain.json) 与 [production chain](challenge_tail8191_production_freeze/results/c1_tail8191_production_freeze_b300_sm103a_t8191_prod_v5_helper_r1_A1_A2.chain.json)，后者 SHA256 `e5b2981e…1c33` | test-only job12406/12415 全局最小裕量 `35.775%`；production job12592/12593 全局最小裕量 `35.4923%`，`production_freeze_passed=true` | 仅发布 `B=1,H=12,T=8191,K=V=128` 的 `none`、`fp32_final_only` 精确格点 |
| skew/FP32-both | [schema-3 test-only freeze](challenge_varlen_fp32_both/results/c1_varlen_fp32_both_b300_sm103a_A1_A2_r5c.freeze.json) 与 [production freeze](challenge_varlen_fp32_both_production_freeze/results/c1_varlen_fp32_both_production_b300_sm103a_skew_production_r2_isolation_A1_A2.freeze.json)，后者 SHA256 `bafa65f8…ad12` | test-only job12555/12556 两 allocation 均过门；production job12770/12771 为四个 fresh PID，最终 `eligible_for_production_freeze=true`、`complete=true` | 仅发布 offsets `(0,1,2,3,4,5,12288)`、`H=12`、`fp32_both` → `vshard4_p2`，不外推任意 varlen |
| v5 cross-map sentry | [job12958/12959 chain](challenge_v5_crossmap_regression/results/c1_v5_crossmap_b300_sm103a_v5_crossmap_r5_a2fix_A1_A2.chain.json)，SHA256 `33ff98ed…e55f17` | 四个已发布正向 cell 与三个负控同时复验；48 个速度比全局最小 `1.046089`，map digest 不变，chain 通过 | 只读、非发布回归证据；两个 job 落在同一 GPU UUID，不称跨 GPU 复现、不扩表 |

## 变量与记号

| 变量 | 含义 | 本实验中的取值或下标 |
| --- | --- | --- |
| `B` | 输入张量的 batch 维 | 主路径为 1；multi-batch 发布正例为 B2–B6（13 cells），另有 B1/T8191 窄点；B7/B8 及未列 cell 均回退；varlen 输入为 1 |
| `T` | 每个 fixed batch 项的 token 数；varlen 时为拼接总长度 | 主路径为 8192；fixed B2–B6 发布点为 2048，B1 另有 T8191 窄发布点 |
| `H` | 当前 GPU 上的 KDA head 数 | 官方 96；附加对照 64；TP8 时理论上为 12 |
| `N_seq` | 当前 launch 的独立 sequence 数 | fixed 时等于 `B`；varlen 时为 `len(cu_seqlens)-1` |
| `L_i` | 第 `i` 条 packed sequence 的长度 | `i=0,…,N_seq-1`；winner 不能只由长度总和推出 |
| `o_j` | CPU-authoritative packed offset，`o_j=sum_{i<j}L_i` | `o_0=0`、`o_{N_seq}=T`，严格递增；精确 tuple 才能进入 varlen 白名单 |
| `d_varlen` | 由原始 CPU `int64` offsets tensor 签发的不透明 descriptor | 进程内、不可 pickle；证明 metadata，不直接授权性能路径 |
| `h_cache` | canonical offsets 的 device-cache 命中标志 | miss 负责 H2D+event 发布，hit 在当前 stream 等待发布 event |
| `e_pub` | cache entry 的 CUDA publication event | 防止跨 stream 使用尚未完成的 offsets copy；graph capture 下 hit/miss 均拒绝 |
| `A` | 相互独立的 Slurm allocation | 新补轮用 `A∈{A1,A2}`；首轮负门则不申请 A2 |
| `p,j` | 一个 allocation 内的 fresh PID 与该 PID 的 repeat 下标 | 通常 `p∈{0,1}`、`j∈{0,1}` |
| `q` | 延迟分位下标 | `q∈{P50,P95,P99}` |
| `r_{A,p,j,q}` | 同一格对照路径与 C1 路径的延迟比 | `t_{control,A,p,j,q}/t_{C1,A,p,j,q}`；`r>1` 表示 C1 更快 |
| `δ_{A,p,j,q}` | 同格 C1 相对对照的裕量 | `r_{A,p,j,q}-1`；发布协议通常要求每格 `δ≥2%` |
| `M=N_seq×H` | K2 的逻辑 `(sequence,head)` 工作项数 | sequence-count 补轮检验其能否单独决定 winner |
| `D=K=V` | query/key/value 及状态的两个矩阵维度 | 128 |
| `C` | KDA chunk 长度 | 官方实现 `C=16` |
| `c` | chunk 编号 | `0 <= c < ceil(T/C)` |
| `S_c` | 完成 chunk `c` 后的递推状态 | 逻辑 `[K,V]`，API 物理布局 `[V,K]` |
| `g_i` | 第 `i` 个 token 的 gate 对数衰减 | 安全下界为 `-5` |
| `s,V_s` | value 分片号与单 CTA value 宽度 | vshard2 为 `s∈{0,1},V_s=64`；vshard4 为 `s∈{0,1,2,3},V_s=32` |
| `M_G,N_G,K_G` | 公平对照的逻辑 GEMM 维度；下标 `G` 表示 GEMM | `128,64,C` |
| `M_TC,N_TC,K_TC` | tcgen05 的物理 tile 维度；下标 `TC` 表示该 Tensor Core 指令 tile | `128,64,64`；与后文 sequence-count 的 `M` 无关 |
| `Q` | 一次公平对照 event 的独立 CTA tile 数 | 4096；两条路径相同 |
| `F,Q_DRAM,I` | 同一 kernel 的有效 FLOP、实际 DRAM bytes、强度 `I=F/Q_DRAM` | 只在同一 profile 的计数可无歧义换算时计算 |

## 可复核环境与复现

| 项目 | B300 | RTX 5090 |
| --- | --- | --- |
| GPU / compute capability | NVIDIA B300 SXM6 AC / 10.3 | NVIDIA GeForce RTX 5090 / 12.0 |
| 驱动、CUDA、Torch | 580.126.09、13.0、2.13.0+cu130 | 580.142、13.0、2.13.0+cu130 |
| FlashKDA / CUTLASS / FLA pin | `1ce47ea` / `5c149f5` / `a3edffc` | 相同 |
| 清卡纪律 | PRE/POST 均显示 0 MiB 且 compute-apps 为空 | 同左 |

完整的 PRE/POST 审计、版本串和命令输出在
[B300 基线日志](experiment_logs/c1_baseline_b300_job4306.log)、
[5090 基线日志](experiment_logs/c1_baseline_5090-smoke_job6848.log)、
[B300 指令审计日志](experiment_logs/c1_instruction_audit_b300_job4306.log) 与
[B300 挑战日志](experiment_logs/c1_vshard_b300_job4306.log)。
基线/挑战均是 CUDA event 计时；B300 基线使用 warmup 30、200 iterations、
5 repeats，5090 冒烟复现为 warmup 5、20 iterations、3 repeats，故跨卡数值是
架构对照而非严格同方差竞赛。

### 官方基线结果

固定长度 `T=8192,D=128` 的均值（ms）：

| GPU | H | FlashKDA no-state | FlashKDA BF16-state | FlashKDA FP32-state | FLA `chunk_kda` | gated delta rule |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B300 | 96 | 1.0286 | 1.0306 | 0.9994 | 2.3503 | 1.2958 |
| B300 | 64 | 0.9393 | 0.9411 | 0.9102 | 1.6279 | 0.8898 |
| RTX 5090 | 96 | 2.6232 | 2.6293 | 2.6332 | 5.4374 | 3.0829 |
| RTX 5090 | 64 | 2.2101 | 2.2128 | 2.2183 | 3.5290 | 2.0000 |

B300 还覆盖了两组变长序列（总 `T=8192,H=96,D=128`）：

| `seq_lens` | BF16-state | no-state | FP32-state | `chunk_kda` | gated delta rule |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[1300,547,2048,963,271,3063]` | 0.8604 | 0.8581 | 0.8718 | 2.3744 | 1.3025 |
| `8 × 1024` | 0.7150 | 0.7149 | 0.7352 | 2.3278 | 1.2541 |

这说明固定长度之外的 `cu_seqlens` 路径也实际跑通；它不等价于对任意长度分布
的吞吐承诺。

### 指令与 NCU 审计

对正式 B300 扩展 `.so` 做 `cuobjdump --dump-sass`，日志中的全库计数为
`HMMA=1544`、`WGMMA=0`、`UTCOMMA=0`、`UTCHMMA=0`；抽样指令为
`HMMA.16816.F32.BF16`。这与源码中 K2 的
`MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>` 一致，足以确认主矩阵路径
停在 SM80 MMA。

NCU `--set basic` 还抓到了 K1 prepare 与 K2 recurrence。K2 的正式 `H=96`
launch grid 是 `(1,96,1)`，报告指出只有约 0.3 full waves；这比“换一条
新 MMA 指令”更直接地暴露了 K2 的并行度上限。该 basic report 对 prepare 记录的
L1/TEX、L2、Compute(SM) 吞吐分别为 68.82%、72.46%、70.21%，但它不是完整
roofline 采样，不能据此断言全端到端必然 compute-bound 或 memory-bound。

## 六个讨论点

### 1. 为什么 `CHUNK=16`，以及 32/64 的代价

| 变量 | 含义 | 本节约束 |
| --- | --- | --- |
| `C` | 一个 K2 chunk 的长度 | `16,32,64` |
| `g_i` | chunk 内第 `i` 项的 gate 下界 | `g_i=-5` |
| `r(C)` | 相对首项的最小衰减上界 | `exp(-5(C-1))` |

在 gate 处于最保守下界 `g_i=-5` 时，一个 chunk 内相对首项的最小衰减上界为

`r(C)=exp(-5(C-1))`。

| C | `r(C)`（近似） | 相对 C=16 的稠密 `C×C` 工作/存储 |
| ---: | ---: | ---: |
| 16 | `exp(-75)=2.68e-33` | 1x / 1x |
| 32 | `exp(-155)≈4.8345e-68` | 8x / 4x |
| 64 | `exp(-315)=1.57e-137` | 64x / 16x |

BF16 的最小 normal 约为 `1.18e-38`，所以上述保守边界下 16 尚在 normal
范围内，32 已在其下很远；即使允许 subnormal，32/64 也会迅速下溢。这里的
“先破”是数值范围。清卡 B300 v2 JSON 实测 FP32 值 cast 为 BF16 后：`C=19`
仍为非零 subnormal（`8.265e-40`），`C=20` 首次为零；`C=32/64` 均为零。
另一方面，严格下三角 `L` 的 Neumann 逆
`(I+L)^-1 = I-L+...+(-L)^(C-1)` 最多需要 `C` 项；以常规稠密 GEMM 的
`O(C^3)` 量级估计，32 与 64 的代价分别至少放大 8 和 64 倍，并且 workspace
面积按 4 和 16 倍增长。源码也把 `CHUNK=16` 固化在
[`fwd_launch.cu`](FlashKDA/csrc/smxx/fwd_launch.cu)。

同一 [job 4308 清卡日志](experiment_logs/c1_chunk_tcgen_microbench_b300_job4308.log)
还在固定 `Q=16384` 个独立矩阵、相同并行度下做 FP16 batched dense doubling
Neumann 代理。下表的 FLOP 是该代理实际执行的 GEMM FMA FLOP，误差列均对 FP32
代数参考通过；因此 wall-time ratio 可横比，但不能冒充 FlashKDA 端到端时间。

| C | 代理 CUDA event (ms) | 相对时间 | 相对代理 FLOP | 代理 TFLOP/s | 代数 gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 0.29324 | 1.000 | 1.000 | 2.746 | PASS (`9.31e-10`) |
| 32 | 0.48510 | 1.654 | 10.667 | 17.708 | PASS (`1.40e-9`) |
| 64 | 1.07644 | 3.671 | 106.667 | 79.799 | PASS (`1.86e-9`) |

时间并未按 FLOP 倍数增长，不表示 `C=64` 在 KDA 中“便宜”：小 `16×16` 稠密
batch 的 Tensor Core 效率很低且 launch/固定开销占比高；更大的矩阵提高利用率，
故代理吞吐从 2.746 升至 79.799 TFLOP/s。这正说明只能把该表用于解释小矩阵效率，
不可替代编译 FlashKDA 变体的端到端 benchmark。

### 2. tcgen05 与 `CHUNK=16` 是否形状匹配

直接 tcgen05 程序 [`02_single_tile.cu`](../../cuda/m3_tcgen05/02_single_tile.cu)
的最小 tile 为 `M_TC=128,N_TC=64,K_TC=64`；KDA Neumann 子问题是 `16×16×16`。
因此单个子问题在 `M_TC/N_TC/K_TC` 三维的占用为 `1/8,1/4,1/4`，有用 FMA 仅
`16³/(M_TC×N_TC×K_TC)=0.78125%`。即使沿 `M_TC/N_TC` 打包 32 个独立 C16 GEMM，
有用比例也仅 25%；不能沿 `K_TC` 打包，因为逻辑 `K_G` 是一个输出的归约轴，会把独立积
错误相加。这是 tile 几何论证，不是吞吐计时。

同一 v2 JSON 记录该独立 M3 可执行程序在 seed `1/7/42` 都返回 `PASS`，证明
该 tcgen05 单 tile 路径可正确运行。它仍不能证明 KDA BF16 端到端更快：K2 还
交织状态递推、TMA、寄存器 transpose、TMEM 和 async-proxy fence/completion；
把 HMMA 文本替换为 tcgen05 不是等价改写。结论是有明确的低利用率挑战和正确性
证据，而**没有**被伪造的 tcgen05-KDA 吞吐结论。

为避免只用几何百分比替代实际同工作量比较，新增
[`microbench/hmma_tcgen05_fair.cu`](microbench/hmma_tcgen05_fair.cu)。它将两条
路径固定为同一个逻辑 `M_G=128,N_G=64,K_G=C` GEMM、相同 BF16 A/B（tcgen05 的
`K_TC=64`，尾部全为 bit-zero）、FP32 accumulator/output、相同 `Q` grid 与
128-thread CTA；逐元素 bitwise gate 后才以 CUDA event 计时。tcgen05 必然执行 padding，
因此该对照会同时报告有效和物理 TFLOPS，而不是偏袒某一条路径：

| `C` | 有效 FLOP/CTA | HMMA 预期 `m16n8k16` 动态次数 | tcgen05 `mma` 命令 | tcgen05 zero-K | 有效/物理 FLOP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 262,144 | 64 | 4 | 48 | 25.00% |
| 32 | 524,288 | 128 | 4 | 32 | 50.00% |
| 64 | 1,048,576 | 256 | 4 | 0 | 100.00% |

HMMA 一侧是 32 个 `16x16` accumulator fragment、每逻辑 WMMA tile 对应两个
`m16n8k16`；tcgen05 一侧固定为 32 KiB TMEM 输出与 32 个 warp-level
`tcgen05.ld.x8` collectives。`run_hmma_tcgen05_audit.sh` 会拒绝没有 `HMMA` 或
`UTCHMMA` SASS 的编译物。该实测比较的是相同逻辑 GEMM、CTA 数和 grid 下的完整
CTA 路径及 `K` padding 代价：HMMA 路径直接加载 WMMA fragment，而 tcgen05 路径还
包含 global→swizzled shared、TMEM alloc/ld/dealloc，因此结果不能孤立归因于某一条
指令的吞吐。它不是只发射一条指令的微基准，也不是 FlashKDA 端到端移植；比较范围是
这个完整 CTA path。job 4340 的 PRE/POST 均为 0 MiB、compute-apps 为空，exact gate
三组均为 PASS（每组 `0/16384` mismatch）；静态 SASS gate 为 **HMMA=14、
UTCHMMA=4、LDTM=8**。原始 [JSON](experiment_logs/c1_hmma_tcgen05_same_work_b300_job4340.json)、
[运行日志](experiment_logs/c1_hmma_tcgen05_same_work_b300_job4340.log) 和
[SASS](experiment_logs/c1_hmma_tcgen05_same_work_b300_job4340.sass) 可相互核对。

| `C` | HMMA median (ms) / 逻辑 TFLOP/s | tcgen05 median (ms) / 有效 TFLOP/s | tcgen05 物理 TFLOP/s | HMMA/tcgen05 时延比 | exact |
| ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 0.056799360 / 18.904118 | 0.242131189 / 4.434546 | 17.738183 | 0.234581 | PASS |
| 32 | 0.088442236 / 24.281200 | 0.242168307 / 8.867732 | 17.735464 | 0.365210 | PASS |
| 64 | 0.152294397 / 28.201742 | 0.242264956 / 17.728389 | 17.728389 | 0.628627 | PASS |

`C=16/32` 的差距正好暴露 `48/32` 个 zero-K 的实际成本：tcgen05 的物理吞吐约
17.74 TFLOP/s 基本不变，但有效吞吐随有效 K 比例从 25% 到 50% 上升；即使 `C=64`
没有 padding，HMMA 仍在此完整 CTA 对照中更快。该事实不推出 FlashKDA 端到端一定
如此，因其还包含 K1、状态递推、布局及跨 kernel 调度。

### 3. 有状态递推下还能从哪里取并行度

源码与 deep-dive 将 K1 切为 token-parallel grid `N×H×num_chunks`，K2 则必须
按 chunk 顺序推进状态，天然主要是 `N×H`。可行候选及反例如下：

| 候选 | 可并行部分 | 主要反例/风险 |
| --- | --- | --- |
| 多 head 合入 CTA | 独立 head | 每 head 的状态与 TMA tile 仍独立，寄存器/shared 增长可能降低 residency |
| persistent K2 | 同 CTA 复用状态与 descriptor | 不能消除 chunk 顺序依赖；head 较少时仍可能填不满全卡 |
| 2 CTA/head value-shard | 状态的 value 列独立 | K1 workspace 读取、TMA 与同步增加，`H=96` 可能已饱和 |

本提交实际实现第三项；其数学安全性和 ABI 限制在
[value-shard README](challenge_vshard/README.zh-CN.md) 中给出。它只在
`K=V=128,V_s=64` 启用，并让两 CTA 写 disjoint 的 state value 行；不会把
递推归约轴错误地切开。

### 4. compute-bound 还是 memory-bound，应该看什么

job 4339 已在 clean B300 窗口对正式 `fixed,H=96,D=128` 分开完成 K1/K2 的
NCU `--set full`。PRE/POST 均为 0 MiB、compute-apps 为空、脚本返回 0；原始
[NCU 日志](experiment_logs/c1_ncu_full_b300_job4339.log)、
[raw CSV](experiment_logs/c1_ncu_full_b300_job4339.csv)、
[`.ncu-rep`](experiment_logs/c1_ncu_full_b300_job4339.ncu-rep) 与
[中文摘录](experiment_logs/c1_ncu_roofline_b300_job4339.md) 同时保留。K1/K2 绝不
合并成一个标签；`.ncu-rep` 的 SHA-256 为
`c9724636c3c83988c4b3a162ad5ddc49e00980e37e70c15612c74a07d408d8aa`。实际指标
与保守结论如下。

| kernel | 实际 compute / memory metric（elapsed） | 实际 parallelism metric | 分类 |
| --- | --- | --- | --- |
| K1 prepare | HMMA `5.633783%`；DRAM read/write `28.779275%/29.829528%`，L2 sectors `30.967491%`，L2 hit `45.384291%` | grid `(512,96,1)`，`41.51` waves/SM，active warps `96.625428%` | **非并行度受限**；compute 与 DRAM 都未饱和，不能仅凭这些百分比贴 compute-/memory-bound 标签。barrier/long-scoreboard/short-scoreboard stall ratio 分别为 `8.308038/4.034395/2.551617`，说明仍需按指令与同步路径继续拆解。 |
| K2 recurrence | HMMA `19.390659%`；DRAM read/write `15.394061%/3.243208%`，L2 sectors `12.481916%`，L2 hit `53.078298%` | grid `(1,96,1)`，仅 `0.32` waves/SM，active warps `9.365785%` | **并行度/小 grid 受限**；此时 compute 和 DRAM 也均未饱和，不能说是 compute-bound 或 memory-bound。short-scoreboard/long-scoreboard ratio 为 `0.769999/0.413727`，是进一步优化数据依赖和加载的候选，而非 roofline 定论。 |

`dram__bytes_*` 已存在于同一次 profile，但当时的 H96 摘要没有定义能无歧义映射为
kernel tensor contraction 的 FLOP 口径；不能把 HMMA 利用率或输入张量大小偷换成 `F`。
故本节对 job 4339 的 `I=F/B` 和 `P_roof=min(P_peak,I×BW_peak)` 明确标为**未计算**，
这里只使用“是否饱和”与 wave/occupancy 的直接证据。续轮随后针对 H12 current 用显式
contraction 计数并要求理论整数与 NCU HMMA counter 完全相等，严格 roofline 见文末；
两处 shape 与证据口径不混用。

### 5. BF16 state 的精度验证

验证设计按证据目标分三层，而不是把三个 reference 错算成三层：

1. kernel 回归：同一随机输入与 initial state 下，candidate 对 upstream
   `flash_kda.fwd`、官方 `tests/torch_ref.py` 与经 `[V,K]↔[K,V]` ABI 转换的
   `fla_kda_ref/naive.py` 同时比较 output/final state；
2. 长期数值：用独立 FP32 recurrence oracle，覆盖多 seed、random/stress、H1/H12 与逐段
   BF16 persistence，报告随 token/segment 的误差；
3. 模型质量：真实多层激活、logits、perplexity 与下游任务 gate。

第一层已完成：小形状 `B=1,T=256,H=2` 的无 state、BF16 state、FP32 state 对 upstream/
Torch reference 均 exact；FLA naive 的最大 output 绝对误差为 `4.882812e-4`，有 state 时最大
state 误差为 `2.838939e-3`，低于 output `rtol=atol=0.02`、state `0.05` 的冻结阈值。第二层
也已由文末最长 262,144-token 的 synthetic recurrence 补齐。第三层仍需外部模型权重、数据
与 launcher，不能只引用官方 “internal testing”，也不能由前两层推导为通过。

### 6. 是否发布 sm100a v2

结论是：保留可移植的 SM80 基线，**可以**把 value-shard 做成 sm100a 的受限
opt-in 分支，但不应把它设为无条件默认。支持专版的一边是 B300 `H=64` 的
1.1766x，以及 K2 `0.32 waves/SM` 的 NCU 证据；反对的一边是官方 `H=96` 只有
1.0010x，且 tcgen05 几何利用率仅 0.78125%、尚无 KDA 端到端吞吐，专用代码还增加了构建、测试和
架构 dispatch 负担。合理的 v2 gate 是：保持 exact reference 回归，按每卡
head 数/shape profiling 选择，若代表性 P50/P99 没有稳定收益就回退 `fwd`。

## 挑战：K2 value 维二分

挑战代码位于 [challenge_vshard](challenge_vshard/)，不修改 pinned 的官方
快照。对于

| 变量 | 含义 | 本节取值/关系 |
| --- | --- | --- |
| `c` | K2 chunk 时间下标 | 沿序列递推 |
| `O_c` | 第 `c` 个 chunk 的输出块 | 写入 value 列 |
| `S_c` | 第 `c` 个 chunk 后的状态矩阵 | 物理 ABI 为 `[V,K]` |
| `Q'_c,K'_c,M_c,U_c,g_c` | K2 递推的输入/中间矩阵与 gate | 按 pinned FlashKDA 定义 |
| `V,V_s` | 完整 value 宽度与单 CTA value shard 宽度 | `128,64`（vshard2） |

`O_c = Q'_c S_(c-1) + M_c U_c`，

`S_c = diag(exp(g_c)) S_(c-1) + K'_c^T U_c`，

value 列没有跨列归约，故 `S_c=[S_c^(0)|S_c^(1)]`。每个 `(sequence,head,shard)`
发射一个 CTA，`V=128` 切为两个 `V_s=64`。K1 仍只运行一次；两 CTA 仅共享
只读 workspace。

小形状三个 reference 全部通过后，正式形状只计 BF16-state（大形状 reference
因顺序朴素实现太慢而未与计时混跑）：

| B300 形状 | baseline median (ms) | vshard median (ms) | speedup | 结论 |
| --- | ---: | ---: | ---: | --- |
| `T=8192,H=96,D=128` | 1.030880 | 1.029888 | 1.000963x | 近似持平，不能称优化 |
| `T=8192,H=64,D=128` | 0.940800 | 0.799616 | 1.176565x | 正收益 |

这不是把新数字随意接到旧二进制上：终测
[日志](experiment_logs/c1_vshard_b300_p1final_job4340.log) 在同一份 PRE/POST 清卡审计
中把 baseline 与 patched tree 固定到 `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`。日志中的
SHA256 绑定如下；它们是本表 timing 对应的实际脚本、patched source 与已载入扩展。

| 对象 | SHA256 |
| --- | --- |
| `apply_vshard_patch.py` | `bca3248e1bf480ea51eb3bb3da0e79d8f477fb914ea17d320c0bf90679aaaf7c` |
| `vshard.py` | `7f508c60003439de27b184d70c130ca615439fcaa61e9970be4dc961ffcccaec` |
| `validate_and_bench.py` | `5c92ac532525827b72ae5a714303eed896ef1f4742db17e0c970bd8055287d52` |
| patched `flash_kda.cpp` / `fwd.h` | `4d0909b74d92c6cde3171bab432af4463f7303a89c2fff0592200a93f5d5fdb3` / `4944a2d2f2ea3757cd8d5421507556a3e6ae860d303a26c86ca67accbaaff3d4` |
| patched `fwd_launch.cu` / `fwd_kernel2_vshard.cuh` | `8c9a4ffbbf086901505be40494293ef8f4038fad9c118b40cab361c8944218c1` / `af588b32f6ad4a8043874eb82f6b997a34eca98736a8dc7fe61543c7d684ae69` |
| loaded `flash_kda_C*.so` | `ed28f470cb32560675a51ef4936ffebea1c5f003b7cfb5f0518acb25a47cb838` |

job 4306 在同一 r4 extension 上运行了 none/BF16/FP32 的三层小形状门：
baseline-vs-vshard 与 upstream Torch ref 为 exact PASS，FLA naive 在冻结容差内
PASS。job 4340 明确复用并记录了该旧 gate 的 JSON/log hash；其当前
`p1final_small_gate` 只重跑 baseline-vs-vshard exact（有 state 时也含 final state），
同时重新绑定当前 patch/script/generated-source/extension SHA。正式 H96/H64 BF16
计时也在计时前通过 output/final-state exact PASS。机器可读终值为
[H=96 JSON](experiment_logs/c1_vshard_b300_p1final_h96.json)、
[H=64 JSON](experiment_logs/c1_vshard_b300_p1final_h64.json) 和
[small-gate JSON](experiment_logs/c1_vshard_b300_p1final_small_gate.json)。挑战尚未在
5090 上复测，因而不把 B300 的 sm100a 结论外推给 5090。

## 优化续轮：4 CTA/head 的 vshard4 终态

为检验“更多 value 分片”是否能继续超过当前 vshard2，另一个候选把 `V=128` 切为四个
`V_s=32` 切片。其变量与计时对照如下：

| 变量 | 含义 | 终验取值 |
| --- | --- | --- |
| `H` | 每卡 head 数 | `64,96` |
| `V_s` | vshard4 单 CTA 负责的 value 列数 | `32` |
| `t_0,t_4,t_2` | 官方 baseline、vshard4、冻结 current vshard2 的完整调用中位数 | ms |

clean B300 job 4446 的 small matrix 在 none/BF16/FP32 均 exact，正式 H64/H96 的
BF16 output/final-state 也 exact；各路径采集 1000 个单调用 CUDA-event 样本，并交替
AB/BA。性能如下：

| shape | `t_0` 官方 baseline | `t_4` vshard4 | `t_0/t_4` | `t_2` current vshard2 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| B300 `T=8192,H=64` | 0.943168 | 0.814592 | **1.157841x** | 0.799616 | 相对官方正，但低于 current |
| B300 `T=8192,H=96` | 1.033472 | 1.149568 | **0.899009x** | 1.029888 | 负优化 |

原始 [H64 JSON](challenge_vshard4/results/c1_vshard4_b300_job4446_h64.json)、
[H96 JSON](challenge_vshard4/results/c1_vshard4_b300_job4446_h96.json) 和同作业
[审计日志](challenge_vshard4/results/c1_vshard4_b300_job4446_job4446.log) 均绑定源码/SO
hash，且 PRE/POST 为 0 MiB、compute-apps 为空。严格“在 current vshard2 基础上再
快 10%”的门槛为 H64≤0.726924 ms、H96≤0.936262 ms；vshard4 未达到，不能替换 current。

RTX 5090 job 6999 的独立 clean repeat 亦为正确但负优化：H64 为 2.227344→3.276448 ms
（0.679804x），H96 为 2.628704→4.906656 ms（0.535742x）。因此不跨架构混用数值。

## 优化续轮：P2 software-prefetch 的 B300 新 current

P2 只改变既有 vshard2 K2 phase-6 的软件预取距离；其变量与比较边界如下：

| 变量 | 含义 | 终验取值 |
| --- | --- | --- |
| `P` | K2 phase-6 的软件预取距离 | P1=`1`，P2=`2`，P3=`3` |
| `S` | K2 `InputStages` | 主候选 `3`；唯一消融 `2` |
| `t_1,t_2,t_3,t_f` | 同 allocation P1/P2/P3 与原冻结 vshard2 的 full-call 中位数 | ms |

clean B300 job 4467 的 P2、`S=3` 在 all-state gate 为 bitwise exact。固定
`B=1,T=8192,H=64,K=V=128` 的 AB/BA full-call CUDA-event 中，P1/P2 分别为
`t_1=0.8063519895`、`t_2=0.7375999987` ms，即 `t_1/t_2=1.0932103998x`；相对
原冻结 `t_f=0.799616` ms，则 `t_f/t_2=1.084078093x`。原始
[B300 P2 S3 JSON](challenge_prefetch2/results/c1_prefetch2_b300_sm103a_fresh_r2_envfix2_h64_bf16.json)、
[small all-state JSON](challenge_prefetch2/results/c1_prefetch2_b300_sm103a_fresh_r2_envfix2_small_matrix.json)
与 [clean 审计](challenge_prefetch2/results/c1_prefetch2_b300_sm103a_fresh_r2_envfix2_job4467.log)
可复核。

因此 B300 H64 的 current 更新为 P2 `S=3` **0.737600 ms**，相对旧 current 正优化
**+8.41%**。但本轮起点要求的原严格目标为 `t≤0.726924` ms，P2 仍高约 **1.47%**，
故“在原 current 基础上再 +10%”最终未达。

H96 的同一 P2 `S=3` 在 clean B300 job 4467 的 none/BF16/FP32 all-state gate 也为
bitwise exact。其 P1/P2 为 1.035279989→1.003152013 ms，`t_1/t_2=1.032027x`；相对
冻结 H96 best 1.029888 ms 为 **1.026651x**。这同样是正收益，但未到原 H96 strict
门槛 0.936262 ms。因此两个代表性 shape 均没有达到“在原 current 基础上再 +10%”。
原始 [H96 all-state JSON](challenge_prefetch2/results/c1_prefetch2_b300_sm103a_p2stage3_h96_r2_h96_allstate_exact.json)、
[H96 BF16 JSON](challenge_prefetch2/results/c1_prefetch2_b300_sm103a_p2stage3_h96_r2_h96_bf16.json)
及 [H96 clean 审计](challenge_prefetch2/results/c1_prefetch2_b300_sm103a_p2stage3_h96_r2_job4467.log)
可复核。

唯一扩展 P2+`S=2` 同样 all-state exact，但 P1/P2 为 0.805312→0.776032 ms，
仅 1.037730x，相对 frozen 为 1.030390x，较 P2 `S=3` 差；其
[JSON](challenge_prefetch2/results/c1_prefetch2_b300_sm103a_p2stage2_r3_h64_bf16.json)
和 [审计](challenge_prefetch2/results/c1_prefetch2_b300_sm103a_p2stage2_r3_job4467.log) 已归档。
RTX 5090 的 P2 亦为 2.036384→2.036832 ms（0.999780x，持平/负），见
[5090 JSON](challenge_prefetch2/results/c1_prefetch2_5090_sm120a_r2_h64_bf16.json)。
这两项没有超过 B300 P2 `S=3`；按预先停止树，只再验证最后一个 P3 `S=3` 消融。

clean B300 job 4617 的 PRE、all-state 后、benchmark 后与 POST 四次审计均为同一
UUID `GPU-dadf9…`、0 MiB、compute-apps 为空。P3 与同一 SO 内的 P2 在
none/BF16/FP32 output/final-state 上 bitwise exact，对 torch reference 也全部 close、
max-abs=0；fixed BF16 P3 的 ptxas 账本为 60 registers、0 spill。

同 allocation 的 BF16 AB/BA 中，`t_2=0.734975994`、`t_3=0.793200016` ms，
`t_2/t_3=0.926596x`，即 P3 慢约 **7.92%**，并且没有达到 absolute strict
`≤0.726923636` ms。完整 [P3 说明](challenge_prefetch3/README.md)、
[all-state JSON](challenge_prefetch3/results/c1_prefetch3_b300_sm103a_p3s3_r1_h64_allstate_exact.json)、
[raw ABBA JSON](challenge_prefetch3/results/c1_prefetch3_b300_sm103a_p3s3_r1_h64_bf16_abba.json)
与 [clean audit](challenge_prefetch3/results/c1_prefetch3_b300_sm103a_p3s3_r1_job4617.log)
均已归档。

job 4617 的 P2 0.734975994 ms 只用于同轮 P2/P3 因果对照；实验纪律上不把它跨 allocation
替换 job 4467 冻结的 P2 current **0.737600 ms**。P3 显著倒退，故按停止树不做 P4，
P2 `S=3` 保持 current；两个代表性 shape 的原 strict +10% 结论仍为最终未达。

## TP8 / H12 续轮：vshard4 + P2S3 的条件性 current

| 变量 | 含义 | 本轮取值/关系 |
| --- | --- | --- |
| `H` | 当前 B300 上的 KDA head 数 | TP8 目标为 `12`；正式 dispatch sweep 逐整数覆盖 `1–96`，另有 H37/H38 独立复测 |
| `N_SM` | B300 的物理 SM 数 | 本轮 NCU 回读为 `148` |
| `V_s` | vshard4 每 CTA 负责的 value 宽度 | `32`，故 K2 grid 为 `G_4=4H` |
| `G_2,G_4` | vshard2/vshard4 的 K2 CTA grid 大小 | `G_2=2H`、`G_4=4H` |
| `P,S` | phase-6 预取距离与 input stages | 组合候选固定为 `P=2,S=3` |
| `t_0,t_2,t_{4,1},t_{4,2}` | baseline、vshard2-P2、vshard4-P1、vshard4-P2 的完整 public-wrapper 时延 | 同一 SO、同一 allocation 的 CUDA-event P50/P95/P99 |

本轮没有把两个旧 patch 顺序叠加，而是从干净 `1ce47ea` 一次性生成四个独立 entry；
代码和复现脚本位于 [challenge_vshard4_prefetch2](challenge_vshard4_prefetch2/)。同一
extension 同时导出 baseline、vshard2-P2、vshard4-P1 和 vshard4-P2，实际加载 SO 的
SHA-256 为 `8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005`。
[fresh build log](challenge_vshard4_prefetch2/results/c1_vshard4_p2_build_b300_r1.log) 与
[ptxas 账本](challenge_vshard4_prefetch2/results/c1_vshard4_p2_ptxas_b300_r1.json) 显示，
正式 fixed BF16 initial+final-state 的 vshard4-P2 实例为 59 registers、9 barriers、
0 stack、0 spill；14 个 vshard4-P2 实例也全部无 spill。

正确性先于性能。`H=1/2/4 × none/BF16/FP32` small matrix 中，四个候选的 output/
final state 均与 baseline bitwise exact，对 pinned Torch reference 也全部通过且
max-abs 为 0；正式 `T=8192,H=12,BF16` 同样 exact。两次正式作业分别落在不同 B300
UUID，PRE、中间与 POST 均为 0 MiB、compute-apps 为空。每条路径保留 1000 个按四路
循环轮换的原始样本：

| clean run | `t_0` P50 | `t_2` P50 | `t_{4,1}` P50 | `t_{4,2}` P50 | `t_2/t_{4,2}` | `t_0/t_{4,2}` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| job 10005 / r1 | 0.792096 | 0.595440 | 0.593184 | **0.529472** | **1.124592x** | **1.496011x** |
| job 10008 / r2 | 0.799968 | 0.595136 | 0.591744 | **0.529472** | **1.124018x** | **1.510879x** |

r1 的 vshard2-P2 / vshard4-P2 P95 为 0.599267/0.533538 ms，P99 为
0.607790/0.538541 ms；r2 分别为 0.600117/0.534403 ms 与 0.605645/0.538796 ms，
故收益不只存在于 P50。原始 [r1 JSON](challenge_vshard4_prefetch2/results/c1_vshard4_p2_b300_sm103a_h12_r1_h12_bf16_cyclic.json)、
[r2 JSON](challenge_vshard4_prefetch2/results/c1_vshard4_p2_b300_sm103a_h12_r2_h12_bf16_cyclic.json)、
[r1 audit](challenge_vshard4_prefetch2/results/c1_vshard4_p2_b300_sm103a_h12_r1_job10005.log) 与
[r2 audit](challenge_vshard4_prefetch2/results/c1_vshard4_p2_b300_sm103a_h12_r2_job10008.log)
可逐样本复核。这里的成功门是同一 H12 SO 中超过 current；H12 没有复用 H64 的
0.726924 ms 绝对阈值。

为避免把 H12 单点误写成通用收益，job 10173 在同一 B300 UUID 与同一 extension 上
逐整数穷举 `H=1–96`。每个 H 使用独立 Python 进程，四条路径各保留 500 个循环轮换的
CUDA-event raw samples；每个 shape 的 BF16 output/final-state 都与 baseline bitwise
exact，对 pinned Torch reference 的 max-abs 也均为 0。source-bound 汇总器逐文件验证
shape、state、SO SHA、exact gate、500 条 summary/raw samples 后给出的关键 P50 如下：

| `H` | `t_2` vshard2-P2 | `t_{4,2}` vshard4-P2 | `t_2/t_{4,2}` | 判读 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.549856 | 0.486336 | 1.130609x | vshard4-P2 胜 |
| 12 | 0.609344 | 0.542848 | 1.122495x | TP8 目标胜 |
| 32 | 0.652576 | 0.584832 | 1.115835x | vshard4-P2 胜 |
| 37 | 0.666080 | 0.598032 | **1.113787x** | 正区间上沿 |
| 38 | 0.669248 | 0.724288 | **0.924008x** | 首个阶跃，vshard2-P2 胜 |
| 48 | 0.696576 | 0.752992 | 0.925078x | vshard2-P2 胜 |
| 64 | 0.741536 | 0.801536 | 0.925144x | vshard2-P2 胜 |
| 74 | 0.770752 | 0.828032 | 0.930824x | vshard2-P2 胜 |
| 75 | 0.944864 | 1.092048 | 0.865222x | 第二个共同阶跃 |
| 96 | 1.007488 | 1.152352 | 0.874288x | vshard2-P2 胜 |

不只 P50：P95 和 P99 的胜负符号也逐 H 得到同一个干净分界，即 H1–37 全正、H38–96
全负。P50 的正区间 speedup 为 1.113149–1.130609x，负区间比值为
0.865222–0.930824x。H74 时 `2H=148`、`4H=296=2×148`；H75 同时越过两条路径的
整数 CTA/SM 波次，CUDA-event P50 随之从 0.770752/0.828032 ms 跳到
0.944864/1.092048 ms。这个第二阶跃与 CTA wave/tail 解释一致，但没有单独做 H74/H75
NCU，故只作为 event 证据，不冒充 profiler 因果证明。

[全整数 sweep audit](challenge_vshard4_prefetch2/results/c1_vshard4_p2_b300_sm103a_hs_full_r1_head_sweep_job10173.log)、
[source-bound summary](challenge_vshard4_prefetch2/results/c1_vshard4_p2_b300_sm103a_hs_full_r1_summary_job10173.csv) 与
[原始证据包](challenge_vshard4_prefetch2/results/c1_vshard4_p2_hs_full_r1_job10173.tgz)
绑定同一源码/SO；证据包的
[SHA-256 sidecar](challenge_vshard4_prefetch2/results/c1_vshard4_p2_hs_full_r1_job10173.tgz.sha256)
记录 `c848b1444cca70188c4f51404781dcf0eedc073eb58ea3836d3ab62fef8c2205`。audit 在每个 H
的进程退出后检查 0 MiB。关键 raw JSON 可直接见
[H37](challenge_vshard4_prefetch2/results/c1_vshard4_p2_b300_sm103a_hs_full_r1_h37_bf16_cyclic.json) 与
[H38](challenge_vshard4_prefetch2/results/c1_vshard4_p2_b300_sm103a_hs_full_r1_h38_bf16_cyclic.json)。较早的
[代表点 scouting](challenge_vshard4_prefetch2/results/c1_vshard4_p2_b300_sm103a_hs_r1_head_sweep_job10044.log)、
[H33–47 连续扫描](challenge_vshard4_prefetch2/results/c1_vshard4_p2_b300_sm103a_hs_boundary_r1_head_sweep_job10053.log) 和
[H37/H38 独立 1000-sample 复测](challenge_vshard4_prefetch2/results/c1_vshard4_p2_b300_sm103a_hs_boundary_r2_head_sweep_job10079.log)
给出相同方向；后者的 H37/H38 比值为 1.116044x/0.924324x。

H37/H38 的 K2-only NCU Basic 直接验证了边界，不只是按 SM 数猜测：

| `H` | vshard4-P2 K2 grid | K2 duration | Compute (SM) | DRAM | L2 | achieved occupancy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 37 | 148 CTA | 470.43 μs | 19.63% | 11.32% | 43.09% | 6.25% |
| 38 | 152 CTA | 593.63 μs | 15.86% | 9.55% | 34.84% | 6.45% |

两个 K2 的模板、每 CTA 工作量与资源配置相同：均为 128 threads、59 registers/thread、
NCU 显示 55.30 Kbyte/block dynamic shared memory（约 54 KiB）；全局 K2 grid 仅由
148 增至 152 CTA。观测到的 26.19% duration
阶跃支持“跨过一 CTA/SM 后由少数承载额外 CTA 的 SM 决定尾延迟/资源竞争”的解释；
NCU 自身的 formal `Waves Per SM` 是 0.25/0.26，因此不把它误称为 occupancy 意义上的
第二个完整 wave。原始 [boundary NCU audit](challenge_vshard4_prefetch2/results/c1_h37_h38_ncu_b300_sm103a_boundary_r1_job10105.log)、
[H37 CSV](challenge_vshard4_prefetch2/results/c1_h37_h38_ncu_b300_sm103a_boundary_r1_h37_vshard4_p2_basic_job10105.csv)
和 [H38 CSV](challenge_vshard4_prefetch2/results/c1_h37_h38_ncu_b300_sm103a_boundary_r1_h38_vshard4_p2_basic_job10105.csv)
均已归档。

H12 四路 K2 Basic 则解释了组合内的 P2 收益：

| K2 路径 | grid / block | NCU duration | Compute (SM) | DRAM | L2 | achieved occupancy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 12 / 192 | 742.05 μs | 2.49% | 2.15% | 3.35% | 9.37% |
| vshard2-P2 | 24 / 192 | 535.07 μs | 4.51% | 3.05% | 7.18% | 9.36% |
| vshard4-P1 | 48 / 128 | 533.22 μs | 5.69% | 3.10% | 11.71% | 6.25% |
| vshard4-P2 | 48 / 128 | **470.59 μs** | 6.37% | 3.52% | 13.65% | 6.25% |

P1/P2 的 grid、block 和 shared memory 相同，P2 只多 1 register/thread，却把 K2 NCU
duration 降低 11.75%，并提高同工作量的 SM/L2 利用；这与软件预取覆盖依赖延迟一致。
但 Full NCU 的 vshard4-P2 K2 仍只有 6.36% SM、3.51% DRAM、6.26% achieved occupancy；
scheduler 有 80.07% 周期没有 eligible warp，active/eligible warps per scheduler 仅
1.00/0.199。raw counter 给出 DRAM read/write 110.724096/16.110336 MB、L2 hit
73.73%，主要 stall ratio 为 wait 1.288240、sleeping 0.862676、short-scoreboard
0.736282、long-scoreboard 0.696244。故它仍是低 grid/依赖延迟受限，而不是带宽饱和。
[NCU audit](challenge_vshard4_prefetch2/results/c1_h12_ncu_b300_sm103a_h12_r1_job10085.log)、
[Full details CSV](challenge_vshard4_prefetch2/results/c1_h12_ncu_b300_sm103a_h12_r1_vshard4_p2_full_job10085.csv)、
[raw CSV](challenge_vshard4_prefetch2/results/c1_h12_ncu_b300_sm103a_h12_r1_vshard4_p2_full_job10085_raw.csv)
与无损压缩的 [Full report](challenge_vshard4_prefetch2/results/c1_h12_ncu_vshard4_p2_full_job10085.ncu-rep.gz)
可复核。NCU replay duration 只用于分 kernel 机理对照，不替代 CUDA-event full-call P50。

因此当前最窄且有证据的发布结论是：在 B300 fixed `T=8192,D=128,BF16-state`、本轮
逐整数穷举的 `H=1–96` 轴上，dispatch 选择 `H≤37` 的 vshard4-P2、`H≥38` 的
vshard2-P2；TP8 的 `H=12` 明确落在正区间。这不是跨 `T`、state、batch 或架构的证明，
真实发布仍应把这些条件编码为白名单，任何未测组合继续逐级回退。

## 部署续轮：从部署证据到停止树闭合

### 续轮变量与判据

| 变量 | 含义 | 本轮取值或关系 |
| --- | --- | --- |
| `R` | 同时运行的 NCCL rank 数 | 目标 TP8 为 8；受配额实际观测为 1 |
| `N_seq` | 当前 launch 中相互独立的 sequence 数 | fixed 时为 batch；varlen 时为 `len(cu_seqlens)-1` |
| `M=N_seq×H` | K2 的逻辑 `(sequence,head)` 工作项数 | 54-case 补轮检验其能否单独决定 winner |
| `p` | 延迟分位数下标 | `p∈{50,95,99}` |
| `δ_p` | winner 相对次优路径的裕量 `(t_{runner-up,p}/t_{winner,p})-1` | 固定 batch 发布门要求每次 repeat 的三个 `δ_p≥2%` |
| `S_v` | 每 head 的 value shard 数 | vshard2/4/8 分别为 2/4/8 |
| `G_K2` | K2 CTA 总数 | `N_seq × H × S_v` |
| `P` | phase-6 software-prefetch 深度 | V8-P2 为 2，V8-P1 为 1 |
| `N_tile` | fixed roofline 中的 `(batch,head,chunk)` tile 数 | `BHT/C` |
| `F_TC` | NCU dense BF16→FP32 HMMA tensor FLOP 数 | 只计 tensor contraction，FMA 计 2 FLOP |
| `Q_DRAM` | NCU DRAM read+write bytes | K1/K2 分阶段回读 |
| `I_TC` | tensor arithmetic intensity | `F_TC/Q_DRAM` |
| `ε_o,ε_s` | output/state 相对 L2 误差 | BF16 persistence 与独立 FP32 recurrence 对照 |

本轮先做可改变发布结论的 P0/P1，再做能闭合归因的 P2，最后只给 P3 一次有明确
停止门的低成本实现机会。结果不是“所有设想都实现”，而是把方向分成：通过、有限
shape 通过、负证据停止、外部阻塞四类。

| 原优先级方向 | 本轮状态 | 决策 |
| --- | --- | --- |
| P0 FLA/TP8 自动 dispatch | **单 shard 与 public API 已完成；full TP8 外部阻塞** | 保留 opt-in 白名单；不把单 rank 写成 TP8 |
| P1 tail/batch/varlen | **11 shape 四状态 exact、5 组 memcheck；历史 public 10→2；新 FP32-both 资格与真实 production 双 allocation 均通过** | 当前 v5 source 为 skew exact tuple 的 3 个 state cell；第三格 job12770/12771 production freeze 已闭合，r5 cross-map 非发布 sentry 也已闭合且未改 map；其他 varlen 精确回退 |
| P1 sequence-count/fixed batch | **54-case 否决 M-only；13 个精确 fixed public contract 已完成集成** | 只发布逐 cell 白名单；不从 `M` 外推 |
| P1 长上下文 BF16 质量 | **数值误差实验完成；模型任务质量外部阻塞** | 报告 recurrence 误差，不冒充 perplexity/任务质量 |
| P2 跨长度 dispatch | **B300 长度轴与两个交互切片完成** | 只编码实测点，其他长度/架构回退 |
| P2 严格 roofline | **完成** | 用 HMMA exact counter 闭合 tensor-contraction 口径 |
| P3 tcgen/chunk/persistent/multi-head | **现结构下负证据停止** | 不投入端到端重写；先试更直接的 V8 低-grid 候选 |
| V8-P2/P1 | **资源门/性能门分别失败** | 八分片方向终止，不接 dispatcher |

### 1. FLA public 调用链与保守自动 dispatch

新增 [auto_dispatch.py](challenge_tp8_dispatch/auto_dispatch.py) 与
[fla_backend.py](challenge_tp8_dispatch/fla_backend.py)。backend 只有在调用方显式设置
`C1_B300_FLASH_KDA=1` 并注册后才生效；实测 registry 中自定义 backend priority 2，
官方 FlashKDA priority 3。policy 同时核对 B300 名称、SM103/148 SM、CUDA contiguous
dtype/layout、`K=V=128`、shape 与 state contract；最终 v5 采用 exact-symbol fail-closed：
policy 选中的专用 symbol 缺失时在 launch 前直接回 baseline，尤其不允许把已审计的 v4 cell
静默降成未为该 cell 审计的 v2。运行时还会对 `flash_kda_C.__file__` 计算
SHA-256；只有与 state/FLA 审计共同绑定的 `8f8cb970…e005` 才允许专用 launch，同名同
symbol 但未审计的 SO 会在 launch 前回 baseline。fixed 与 exact packed-varlen policy 的
CPU tests（含错 SO、伪造 descriptor、CPU tensor identity、capture hit/miss 反例）最终分别为
27/27 与 11/11 通过；backend compatibility token 为
`c1-b300-flash-kda-skew-fp32-both-20260830-v5`。

真实 state ABI runner 在 `B=1,T=8192,H=12` 对四种 raw contract 重新对拍；其中
none、FP32-final-only、FP32-both 是 FLA public contract，BF16-both 仅是原生 FlashKDA
ABI 覆盖，不能写成 FLA public state：

| state contract | baseline P50 (ms) | vshard2-P2 | vshard4-P2 | v2/v4 | exact |
| --- | ---: | ---: | ---: | ---: | --- |
| none | 0.798400 | 0.591040 | **0.525776** | 1.12413x | baseline + Torch reference |
| BF16 both（raw ABI） | 0.800160 | 0.595152 | **0.527520** | 1.12821x | baseline + Torch reference |
| FP32 both | 0.768144 | 0.628128 | **0.531392** | 1.18204x | baseline + Torch reference |
| FP32 final only | 0.758768 | 0.597248 | **0.521344** | 1.14559x | baseline + Torch reference |

[state-contract JSON](challenge_tp8_dispatch/results/c1_tp8_dispatch_b300_sm103a_h12_realstate_r1_h12_state_contracts.json)
保留完整样本。FLA public `chunk_kda` 随后在 B300 单个 TP shard 上完成两次 clean run：
三种 public state contract 全部 exact；白名单 `T=8192,H=12` 命中 vshard4-P2。故意设置
的 `T=257` 仍由已注册的自定义 backend 接管，但其内部 auto-dispatch 在 launch 前选择
upstream baseline；这不等于 registry 改选官方 backend。r2 实测了 custom priority 2、
官方 priority 3 的 registry 顺序。禁用 opt-in 的拒绝行为只由 CPU registration test
覆盖，没有伪称为 r2 的 GPU public-call 证据。详见
[r2 JSON](challenge_tp8_dispatch/results/c1_tp8_dispatch_b300_sm103a_tp8_shard_r2_tp8_fla.json)
和 [audit](challenge_tp8_dispatch/results/c1_tp8_dispatch_b300_sm103a_tp8_shard_r2_tp8_fla_job10667.log)。

这里必须保留两个边界：JSON 明确记录 `observed_concurrent_ranks=1`、
`tp8_concurrent_gate_pass=false`。2026-08-30 的
[新配额探针](challenge_tp8_dispatch/results/c1_tp8_quota_reprobe_20260830.txt) 证明同一用户的
2/4/8-GPU `sbatch --test-only` 均被 `QOSMaxGRESPerUser` 拒绝，只有 1 GPU 可申请，且未创建
任何测试作业；[模型资产探针](challenge_long_context_quality/results/c1_real_model_asset_probe_20260830.txt)
在可访问的 `/home/lcpu/85117379` 与 `/mnt` 下找到 0 个大权重候选和 0 个 C1 模型评测 launcher。
因此结果是“FLA-level 单 shard 集成”，不是 8-rank 或模型端到端 TP8；资产探针也不声称覆盖
不可访问的外部存储。

### 2. tail、batch 与 varlen：正确性通过，winner 不可一刀切

[challenge_varlen_tail](challenge_varlen_tail/) 在同一 B300 extension 上覆盖 11 个 shape：
fixed `T=1/15/17/31/127/8191`、`B=2,T=17`、`B=4,T=127/2048`，以及两组短/混合
`cu_seqlens`。每个 shape 都覆盖 none、BF16-both、FP32-both、FP32-final-only；baseline、
vshard2-P2、vshard4-P2 的 output/final state 全部逐位一致，Torch reference 子集也通过。
另一个 compute-sanitizer memcheck 作业覆盖 5 个 padded tail/batch/varlen case，0 error。

性能却说明不能把 vshard4 扩成无条件默认：

| case / contract | baseline P50 | vshard2-P2 | vshard4-P2 | winner |
| --- | ---: | ---: | ---: | --- |
| fixed `T=8191`, none | 0.804288 | 0.593984 | **0.527808** | vshard4-P2 |
| fixed `T=8191`, FP32 final | 0.765376 | 0.599376 | **0.521696** | vshard4-P2 |
| `B=4,T=2048`, none | 0.247200 | **0.195072** | 0.209056 | vshard2-P2 |
| `B=4,T=2048`, FP32 final | 0.240864 | **0.197344** | 0.210944 | vshard2-P2 |
| mixed varlen total `T=8192`, none | 0.365792 | **0.281024** | 0.294896 | vshard2-P2 |
| mixed varlen total `T=8192`, FP32 final | 0.354784 | **0.284224** | 0.299104 | vshard2-P2 |

因此“这些测试 shape 的 kernel 正确性”已完成，但生产级任意 `cu_seqlens` 的选择规则
尚未穷举。当前 dispatcher 只把下一小节要求的 CPU descriptor + exact offset/state
白名单作为 varlen 例外；其他 `cu_seqlens != None` 和未列入 fixed 精确表的 `B>1`
组合仍保守回 baseline，而不是从几个 case 外推通用性能模型。原始证据为
[result JSON](challenge_varlen_tail/results/c1_varlen_tail_b300_sm103a_r1.json) 与
[sanitizer JSON](challenge_varlen_tail/results/c1_varlen_sanitizer_b300_sm103a_r1.json)。

#### 2.1 packed-varlen 的 CPU-authoritative descriptor 与精确发布

sequence-count 反例已经证明，`N_seq×H`、总 token 和 K1 tile 数都相同时，长度分布仍可
改变 winner。因此这里没有再拟合通用 varlen 规则，而是增加一条窄的生产路径：调用方
必须显式设置 `C1_B300_VARLEN_CPU_DESCRIPTOR=1`，并提供 CPU contiguous `int64`
offsets。`varlen_metadata.py` 的每次 validation attempt 只做一次 CPU `tolist()`，验证首项、
末项、严格递增、int32 范围和 `q` token 数，再签发进程内、不可 pickle 的 `d_varlen`。
原 public 路径的 verifier 与 body 会各自执行一次完整 `_prepare_varlen`；续轮加入 one-shot
handoff 后，verifier 签发 descriptor，body 消费前先清除 TLS plan，再重读当前 CPU tuple 并
要求与 verifier key 相等，因此保留 freshness 检查而不再重复第二次完整 prepare/descriptor
构造。descriptor 同时绑定签发器 token、原始 CPU tensor 对象身份和不可变 offset tuple；
它只认证 metadata，最终是否走 custom kernel 仍由下面的 exact tuple/state 白名单决定。

GPU `cu_seqlens` 在热路径只做 device/dtype/rank/contiguous/numel 结构校验，**不读取其值**。
真正传给 kernel 的 device offsets 来自已认证 CPU tuple：cache miss 在当前 stream 做 H2D、
记录 `e_pub` 和 `record_stream`；cache hit 先 `wait_event(e_pub)` 再更新 stream lifetime。
CUDA graph capture 时 cold miss 与 hot hit 都在进入 backend/kernel 前拒绝，避免把 cache-owned
pointer 固化进可重放 graph。三阶段求交后的生产表为：

| canonical CPU offsets `o` | `N_seq` | `T_total` | none | FP32 final only | FP32 both |
| --- | ---: | ---: | --- | --- | --- |
| `(0,2048,4096)` | 2 | 4096 | **baseline（public pinned 更快）** | **baseline（public pinned 更快）** | **baseline（public pinned 更快）** |
| `(0,2048,4096,6144,8192)` | 4 | 8192 | **baseline（public pinned 更快）** | **baseline（public pinned 更快）** | **baseline（raw release 未过 2% 门）** |
| `(0,17,528,1552,2852,4901,8192)` | 6 | 8192 | **baseline（public pinned 更快）** | **baseline（public pinned 更快）** | **record-only baseline** |
| `(0,1,2,3,4,5,12288)` | 6 | 12288 | **v2** | **v2** | **v4（v5 source + job12770/12771 production freeze）** |

最后一格先经过 2026-08-30 新双-allocation test-only 资格门，再进入当前 source，并由
job12770/12771 的真实 production A1/A2/freeze 闭合。后续从 job11212 到 job11767 的叙述是它被
加入 v5 之前的历史证据，不能把当时的 baseline 结论误读为当前 map；test-only 与 production
两层 artifact 也仍不可互换。

第一阶段 confirmation job 11212 已在同一 B300/SO 上完成：11 个 promotion cell 各做两次
warmup 100、每路径 1000-sample 的 cyclic CUDA-event repeat；四个 layout × 四种 raw ABI
contract 的 v2/v4 output/final state 都逐位等于 baseline，三种 public contract 的 baseline
又逐位等于 pinned Torch reference，全部输入保持不变。11/11 cell 在 P50/P95/P99 上都由
预注册 winner 胜出且每个 margin 至少 2%；record-only cell 只保存样本，不获得发布资格。
其 JSON SHA256 为 `447d7f49…a64e51c`，PRE/AFTER/POST 均为 0 MiB，`FINAL_RC=0`。

为避免 confirmation 与 policy 同源自证，raw release 11393 使用新 seed/new allocation
再做两次 repeat，并把 frozen discovery、confirmation 两次 repeat 与本轮两次 repeat 按
cell 独立求交。作业 `FINAL_RC=0`，PRE/AFTER/POST 均为 0 MiB；原始 JSON
SHA256 为 `338d0b27…f5838`，其 stdlib-only analyzer 又从四轮原始样本独立得到 **10 release
+ 1 fallback**（audit SHA256 `9f6174f1…754a1`）。唯一剔除项是 equal-N4/FP32-both：
v4 虽仍在 P50/P95/P99 胜出，但两次新 repeat 的最小 margin 只有 **0.91%/1.23%**，没有
达到预注册 2%，因此精确回 baseline；其余 10 个 cell 的新轮最小 margin 至少 4.14%。

public-FLA 门通过真实 registry 分别计数 C1/pinned route，覆盖 exact output/final ABI、输入
不变性、CPU authority、错 GPU metadata、语义回退、双 stream、capture cold/hot 拒绝、
hot-return no-sync 和完整 public-call 时延。为保留失败史，job 11395 先因 Dynamo 装饰器的
raw source 身份误判 fail-close；import probe 证明 `inspect.unwrap` 后仍是 pinned
`fla.ops.kda.chunk_kda`。job 11454 又在 kernel 前暴露 runner 未把 verifier 放进
`torch.inference_mode()`；job 11461 已持久化 10/10 正例与 2/2 policy fallback 后，继续在
负控 helper 暴露缺失的局部 `import torch`。这些都是测试驱动错误，PRE/POST 保持 0 MiB，
且均未被当作发布证据。

修正并经过全文件未绑定名审计后，clean job **11466** 完整 `FINAL_RC=0`：schema 2、10 个
public C1 正例全部与 pinned/reference 逐位一致并保持输入不变；两个预注册 policy cell 精确走
pinned；其余 negative、CPU-authority、双 stream、capture、hot-sync 和 fixed-control 全部通过。
原始 JSON SHA256 为 `a608bb83…4119d`，stdlib-only analyzer 从每 cell 两轮 × 两路径 ×
1000 个原始 event sample 重算后 `independent_audit_pass=true`（audit SHA256
`644007e1…16e4`）。该 **pre-v5 历史阶段**的端到端性能只发布：

| public release cell | variant | r4 最小 `m` | r5 最小 `m` | handoff r6 最小 `m` | 裁决 |
| --- | --- | ---: | ---: | ---: | --- |
| skew N6/T12288, none | vshard2-P2 | 8.257% | 12.469% | 21.499% | 发布 |
| skew N6/T12288, FP32 final only | vshard2-P2 | 12.503% | 5.719% | 13.163% | 发布 |
| skew N6/T12288, FP32 both | vshard4-P2 | 1.9765%（repeat 0 P99） | pinned fallback | candidate r3 的 repeat 0 P95/P99 由 pinned 胜出 | baseline |

另外七个 raw 候选在完整 public 调用上均由 pinned 更快，说明 descriptor/cache/policy 的
端到端成本足以反转短 case 的 raw-wrapper 收益；不能用 raw kernel 数字代替生产调用时延。
当时的 production dispatcher 据此精确收缩为上表前两项。最终 clean freeze job **11479** 又以
schema 3 证明 2/2 C1 route 与 10/10 pinned fallback：后者按 8 个 public-release-failed、
1 个 raw-release-failed、1 个 record-only 分类，每项都要求精确 C1 reject、pinned
verifier、registry spy、direct-pinned/public/Torch-reference 三角逐位一致和三路径输入不变。
17 个 negative、三类 cache 观察、fixed-control 均通过；两项性能再做共 8000 个 timed
public calls 后均发布，`FINAL_RC=0`，PRE/AFTER/POST 为 0 MiB。r5 JSON SHA256 为
`bebd2565…abb41`，独立审计 SHA256 为 `2eb65274…fd65`，且没有 performance-failed cell。
证据见
[public r4 JSON](challenge_varlen_dispatch/results/c1_varlen_fla_integration_b300_sm103a_public_r4.json)
与 [r4 独立审计](challenge_varlen_dispatch/results/c1_varlen_fla_integration_b300_sm103a_public_r4.independent_audit.json)，
以及 [public r5 JSON](challenge_varlen_dispatch/results/c1_varlen_fla_integration_b300_sm103a_public_r5.json)
与 [r5 独立审计](challenge_varlen_dispatch/results/c1_varlen_fla_integration_b300_sm103a_public_r5.independent_audit.json)。

##### public preflight 开销与 one-shot handoff

为区分 kernel 胜负与 public dispatch 开销，诊断使用以下四条实测路径；下标 `P/D` 分别
表示 public registry/direct wrapper，下标 `C1/Pin` 分别表示 C1/pinned backend。

| 变量 | 含义 |
| --- | --- |
| `t_PC1` | 完整 public registry 选择 C1 的单次延迟 |
| `t_DC1` | 直接调用 C1 wrapper 的单次延迟 |
| `t_PPin` | 完整 public registry 选择 pinned backend 的单次延迟 |
| `t_DPin` | 直接调用 pinned wrapper 的单次延迟 |
| `Δ_prep` | 去除 public registry 共性开销后的 C1 额外 preflight 差分 |

对同下标原始样本独立计算
`Δ_prep=(t_PC1-t_DC1)-(t_PPin-t_DPin)`。diagnostic job **11493** 覆盖 equal、mixed、
skew 三个 `none` layout、两轮、四路径各 1000 样本，共 24,000 个 timed calls；spy 实测
`_prepare_varlen` 次数依次为 public-C1/direct-C1/public-pinned/direct-pinned = **2/1/0/0**。
两轮 `Δ_prep` 均值分别为 equal **30.27/30.41 μs**、mixed **32.36/32.35 μs**、
skew **32.12/31.94 μs**。原始 JSON SHA256 为 `012c587c…854aa`，独立审计 SHA256 为
`823321e3…0e65`，clean log SHA256 为 `6c346ec9…30c`；见
[overhead JSON](challenge_varlen_dispatch/results/c1_varlen_public_overhead_b300_sm103a_diag_r1.json)、
[overhead 独立审计](challenge_varlen_dispatch/results/c1_varlen_public_overhead_b300_sm103a_diag_r1.independent_audit.json)
与 [overhead clean 日志](challenge_varlen_dispatch/results/c1_varlen_public_overhead_b300_sm103a_diag_r1_job11493.log)。
该首版诊断的 spy restore 后来被发现会遗留 instance bound-method shadow，因此这些微秒数
只作为方向证据，不作为精确因果估计或发布资格；后续 candidate/r6 runner 已增加 no-shadow
门并在计时前恢复正常 class descriptor binding。

据此实现的 handoff 只在同线程 verifier 成功后暂存一次 plan：tensor 参数用 weak identity
绑定，标量/flag 只接受精确 builtin 类型，未知 kwargs 关闭 handoff；body 先清除 plan，再
重读当前 CPU tuple，身份、内容或参数任一不匹配都回完整 `_prepare_varlen`。复用的只有
descriptor/key；cache、capture、GPU structure、device policy、extension SHA/symbol 与最终
variant selection 仍由 `auto_dispatch.fwd` 完整复核。实现 SHA256 为 `fla_backend.py`
`8555995c…974b`、`varlen_metadata.py` `f89a97ba…4ccd`，该历史阶段的 production dispatcher 仍为原两项 SHA256
`2b817adb…883`。

candidate 的失败史也保持 fail-close：job 11514 的 GPU runner 虽完成，但 analyzer 少解一层
map-restoration wrapper，故 `FINAL_RC=2`；job 11529 又在复核发现 JSON `bool/int` 类型绕过后
于 34 秒主动取消。二者都没有被当作接受证据。最终 runner/analyzer/shell SHA256 固定为
`e07481e7…be14`、`10e3ebd2…2481`、`6729eb73…6ea9` 后才提交下一次 clean allocation。

clean candidate job **11538** 用 process-local 10-cell 候选表做 40,000 个 timed public calls，
但在 `finally` 恢复同一个生产 map 对象；10 个候选的 output/final/reference/输入不变性与所有
负控均通过，PRE/AFTER/POST 为 0 MiB、`FINAL_RC=0`。独立审计只保留现有 `none` 与
FP32-final-only：两轮最小 `m` 分别为 **16.024%/16.924%**、**10.814%/2.764%**。
`FP32-both` 的 repeat 1 最小 `m` 为 5.464%，但 repeat 0 的 C1 P95/P99 为
1.232312/1.251348 ms，反而慢于 pinned 的 1.228218/1.239558 ms，故不能用另一轮或先前
diagnostic 的正结果覆盖该失败。candidate JSON SHA256 为 `2d50d219…23158`，独立审计
SHA256 为 `61c352b7…078fd`，clean log SHA256 为 `2da68def…4d71`；见
[candidate r3 JSON](challenge_varlen_dispatch/results/c1_varlen_fla_handoff_candidate_b300_sm103a_r3.json)、
[candidate r3 独立审计](challenge_varlen_dispatch/results/c1_varlen_fla_handoff_candidate_b300_sm103a_r3.independent_audit.json)
与 [candidate r3 clean 日志](challenge_varlen_dispatch/results/c1_varlen_fla_handoff_candidate_b300_sm103a_r3_job11538.log)。

pre-v5 production freeze job **11590** 固定 schema 4、精确检查当时 production map 仍为 2 项，并验证
两个 public C1 正例的 `_prepare_varlen` delta 都为 **1**、pinned 都为 **0**；2/2 C1、
10/10 fallback、17 个 negative、cache/capture/two-stream/hot-sync/fixed control 全通过。
两项共 8000 个 timed calls 的两轮最小 `m` 为 none **21.499%/22.003%**、FP32-final-only
**13.163%/15.616%**；独立 analyzer 从 raw samples 重算得到 exact 2 release、0 failed。
PRE、AFTER-runner、AFTER-analyzer、POST 均为 0 MiB，`FINAL_RC=0`。JSON SHA256 为
`e8fc8a09…ced9f8`、独立审计 SHA256 为 `dd8bef39…f2317`、日志 SHA256 为
`fff6d60d…b4742`；runner/analyzer/clean-shell SHA256 分别为 `71a01630…ac058`、
`9cb9c626…28cb0b`、`90a2eed9…f335c`。证据见 [r6 JSON](challenge_varlen_dispatch/results/c1_varlen_fla_integration_r6_b300_sm103a_public_r6.json)、
[r6 独立审计](challenge_varlen_dispatch/results/c1_varlen_fla_integration_r6_b300_sm103a_public_r6.independent_audit.json)
与 [r6 clean 日志](challenge_varlen_dispatch/results/c1_varlen_fla_integration_r6_b300_sm103a_public_r6_job11590.log)。
因此 handoff 在 r6 已进入受审生产路径；该版白名单保持两项，当前 v5 的第三项边界见后文补轮。

##### FP32-both 历史尾部的离线定位

| 变量 | 含义 |
| --- | --- |
| `i` | 同一 repeat 内、按原始记录顺序配对的样本下标，范围 0–999 |
| `C_i`,`P_i` | 同下标 C1、pinned public-call 的 CUDA-event 时延 |
| `d_i=C_i-P_i` | 同下标配对差；负值表示 C1 更快 |
| `B_j` | 第 `j` 个连续 100-sample block |
| `m_p=P_p/C_p-1` | 分位 `p∈{50,95,99}` 上 C1 相对 pinned 的裕量 |
| `J` | 同一 repeat 中同时满足 `C_i>P_i` 且 `C_i>1.20 ms` 的样本数 |

为避免把一次 P95/P99 失败笼统写成“噪声”，新增 stdlib-only
[历史尾部重算器](challenge_varlen_dispatch/analyze_varlen_fla_fp32_tail_history.py)；它固定校验
r1/r3 输入 SHA、目标 cell、两轮与每路径 1000 个有限正样本，并从 `raw_samples_ms`
独立计算 `d_i`、parity、block 和 1.20 ms 阈值命中。脚本 SHA256 为
`530eeb70…41b4c`，可复现 [历史尾部 JSON](challenge_varlen_dispatch/results/c1_varlen_fla_fp32_tail_history_r1_r3.json)
SHA256 `a3dc8c50…8c85ad`。

| 输入 / repeat | C1 P95/P99 (ms) | pinned P95/P99 (ms) | `C1>pinned / C1>1.20 / pinned>1.20` |
| --- | ---: | ---: | ---: |
| r1 / 0 | 1.069955 / 1.078753 | 1.175520 / 1.182564 | 0 / 0 / 2 |
| r1 / 1 | 1.070656 / 1.080386 | 1.175360 / 1.180425 | 0 / 0 / 0 |
| r3 / 0 | **1.232312 / 1.251348** | **1.228218 / 1.239558** | **73 / 108 / 132** |
| r3 / 1 | 1.071870 / 1.124538 | 1.176800 / 1.185986 | 2 / 1 / 2 |

r3 repeat 0 的 108 个 C1 `>1.20 ms` 样本集中于下标 359–514，而不是单个离群点。
`B_3/B_4/B_5` 的 C1 `>1.20 ms` 计数为 33/64/11；其中 `B_4` 的 C1/pinned
均值为 1.203489/1.216262 ms，pinned `>1.20 ms` 反而有 74 个，说明宽簇同时抬高两条
路径。even/odd 两种先发顺序也都进入高尾，只存在弱 order 相关，不能推出 launch-order
因果。r1 两轮与 r3 repeat 1 没有复现同样宽簇。因此现有 raw 可以证实“r3 repeat 0
存在共享时间段宽尾”，却不能把原因归到 handoff、CPU descriptor 或 C1 kernel；这正是后续
fresh-process、分块与独立遥测诊断要回答的边界。

##### fresh-process 尾部诊断：修正计时契约后仍按停止门关闭

首个 fresh-process 诊断 job **11679** 完成了 3 个 main PID × 2 repeat、独立 telemetry PID、
48 项 exact、map 同对象恢复和全程 0 MiB；但复核发现 runner 把 route-count dictionary、
decision 检查和 spy 包装放在 CUDA start/end event 之间，而 candidate r3 的 event 内只有真实
public `_call`。因此 v1 产物只能证明执行完整性和“仪器化契约下 C1 相对 pinned 更快”，不能
作为 release-equivalent 绝对尾部或白名单证据。其原始 main0/1/2 SHA256 为
`cb701eda…f0b4`/`3a2cf62b…d63`/`cb247d67…7c44`，独立审计 SHA256 为
`cfd730d3…8f16`；见 [v1 审计](challenge_varlen_dispatch/results/c1_varlen_fla_fp32_tail_diagnosis_b300_sm103a_r1.independent_audit.json)
与 [v1 clean 日志](challenge_varlen_dispatch/results/c1_varlen_fla_fp32_tail_diagnosis_b300_sm103a_r1_job11679.log)。

随后冻结的 v2 runner 将 path 选择、route 快照、event 构造和事后 accounting 全部移到
计时区间外；start event 已消费后，区间内唯一调用是 candidate 的真实 `_call`。runner、
stdlib analyzer、clean shell SHA256 分别为 `3b0342af…fb0`、`0adb9f93…100`、
`f339373a…402`，并经独立源码复核后提交 job **11767**。六轮 raw 重算如下（时延单位 ms，
`m` 为百分数）：

| PID / repeat | C1 P50/P95/P99 | pinned P50/P95/P99 | `m_50/m_95/m_99` | `d` P99 | `C_i>P_i / J` |
| --- | --- | --- | --- | ---: | ---: |
| 265224 / 0 | 1.731072/1.744578/1.750675 | 2.004304/2.017608/2.022496 | 15.784/15.650/15.527 | -0.245184 | 1 / 1 |
| 265224 / 1 | 1.730768/1.741643/1.746977 | 2.003904/2.018230/2.022946 | 15.781/15.881/15.797 | -0.246880 | 0 / 0 |
| 265538 / 0 | 1.729712/1.742688/1.748946 | 2.004816/2.018309/2.022209 | 15.905/15.816/15.624 | -0.247481 | 1 / 1 |
| 265538 / 1 | 1.728304/1.740290/1.750506 | 2.003648/2.016706/2.021281 | 15.931/15.883/15.468 | -0.246486 | 0 / 0 |
| 265972 / 0 | 1.728512/1.741253/1.746887 | 2.004704/2.017954/2.022081 | 15.979/15.891/15.753 | -0.245440 | 1 / 1 |
| 265972 / 1 | 1.728544/1.739648/1.747056 | 2.004384/2.017408/2.022369 | 15.958/15.966/15.759 | -0.250516 | 0 / 0 |

相对门 6/6 全过，最小 `m` 为 **15.468%**，六个 paired `d` P99 也都小于 0；但
预注册绝对门要求每轮 C1 P99 `<=1.20 ms`，而本 allocation 的 6000 个 C1 与 6000 个
pinned 样本全部超过该阈值。三个 PID 在 repeat 0 又都只于下标 221 出现一次 `C_i>P_i`
且 `C_i>1.20 ms`，repeat 1 没有。审计因此分类为
`absolute_or_shared_scale_failure`，严格布尔字段
`second_allocation_decision.eligible=false`，没有提交第二 allocation，也没有扩张白名单。

该作业使用另一块 B300 `GPU-778768b4…48ae`，main 前后与 telemetry sidecar 均记录
P0、graphics/SM **1095 MHz**、memory 3996 MHz；历史 candidate r3 来自不同 UUID 且未
时间对齐记录时钟。低时钟与两条路径整体抬高相符，但只是设备状态解释，不能写成尾部因果，
更不能事后放宽 1.20 ms 门。job 11767 `FINAL_RC=0`，PRE/BETWEEN/POST 均为 0 MiB；
main0/1/2 SHA256 为 `1e3848d0…1e83`/`00c7ee53…1022`/`7ff9ab3a…86b5`，远端审计
SHA256 `a8f2e21e…506a`，本地从 raw 独立复算除路径字符串外逐字段相同。证据见
[v2 审计](challenge_varlen_dispatch/results/c1_varlen_fla_fp32_tail_diagnosis_v2_b300_sm103a_v2_r1.independent_audit.json)、
[本地复算](challenge_varlen_dispatch/results/c1_varlen_fla_fp32_tail_diagnosis_v2_b300_sm103a_v2_r1.local_recompute.json)、
[telemetry sidecar](challenge_varlen_dispatch/results/c1_varlen_fla_fp32_tail_diagnosis_v2_b300_sm103a_v2_r1_telemetry.csv)
与 [v2 clean 日志](challenge_varlen_dispatch/results/c1_varlen_fla_fp32_tail_diagnosis_v2_b300_sm103a_v2_r1_job11767.log)。

#### 2.2 sequence-count 反例与 fixed-batch 精确发布

[sequence-count runner](challenge_seqcount_dispatch/run_seqcount_dispatch.py) 在同一 B300/SO
上覆盖 54 个 fixed、balanced-varlen、skewed-varlen case；每个
`M∈{24,36,37,38,39,40,48,72,75,76,96}` 至少有两种不同 `(N_seq,H)` 分解。四种 raw
state contract 的 v2/v4 output 与 final state 均逐位等于 baseline，23 个预注册
case-contract（37 个 output/state tensor）也逐位等于 pinned Torch reference；162 个性能
cell 的三路径各采样 1000 次。只按 `M` 的策略明确失败：在 `M=38,T=257` 必要控制族中，
fixed 与 balanced-varlen 的 `none` 在 P50/P95/P99 均选 v2，skewed-varlen 却均选 v4；
三者具有相同 `N_seq`、`H`、总 token 和实际 K1 tile 数。序列长度分布是不可由 `M` 消去
的变量，因此通用 M-only dispatch 到此停止。

fixed `H=12,T=2048` 则可按精确 batch/state cell 独立裁决。discovery job 10740、两次新
repeat 的 confirmation job 10771，以及重新读取历史 raw samples 后再做两次新 repeat 的
release job 10784，均要求 P50/P95/P99 winner 一致且相对次优路径至少快 2%。B8 `none`
曾因一次 P99 仅 1.81% 回退；B6 FP32-both 又因一次新 P99 仅 1.85% 回退。最终候选表为：

| fixed public shape | none | FP32 final only | FP32 both |
| --- | --- | --- | --- |
| `B=2,H=12,T=2048` | vshard4-P2 | vshard4-P2 | vshard4-P2 |
| `B=3,H=12,T=2048` | vshard4-P2 | vshard4-P2 | vshard4-P2 |
| `B=4,H=12,T=2048` | vshard2-P2 | vshard2-P2 | baseline |
| `B=5,H=12,T=2048` | vshard2-P2 | vshard2-P2 | vshard2-P2 |
| `B=6,H=12,T=2048` | vshard2-P2 | vshard2-P2 | baseline |
| `B=8,H=12,T=2048` | baseline | baseline | baseline |

在 B5 补测前，表中原有五个 batch 的裁决被精确编码进 [auto_dispatch.py](challenge_tp8_dispatch/auto_dispatch.py)，并由
[public integration runner](challenge_tp8_dispatch/run_fixed_batch_fla_integration.py) 通过
真实 pinned `FlashKDABackend`、direct custom backend 和公开 `fla.ops.kda.chunk_kda`
三路复核。最终 clean job 10810 的 15/15 cell 全部逐位一致；10 个正向 cell 命中上表
v2/v4，5 个负控命中 baseline；每个 public call 的 registered-custom-backend spy 都恰好
增加 1，排除了读取旧 decision 的假阳性。FP32 final state 还逐项验证为 contiguous
`[B,12,128,128]`。runner 固定 FLA commit/六份文件 SHA，并证明实际加载的 6 个 `fla`
模块都位于该 checkout；PRE/AFTER/POST 均为 0 MiB，`FINAL_RC=0`。因此上表已经完成
public FLA 集成，不再只是 raw-wrapper candidate。任何其他 `B>1` shape/state、设备、SO
或 varlen 仍在 launch 前回 baseline；BF16-both 仅属 raw ABI，不是 public FLA 输入合同，
故不作 public fallback 承诺。既有 `B=1` 白名单不受影响。

原始证据为 [sequence-count JSON](challenge_seqcount_dispatch/results/c1_seqcount_dispatch_b300_sm103a_r2.json)、
[confirmation JSON](challenge_seqcount_dispatch/results/c1_fixed_batch_confirmation_b300_sm103a_r1.json)、
[release JSON](challenge_seqcount_dispatch/results/c1_fixed_batch_release_gate_b300_sm103a_r1.json) 与
[public FLA JSON](challenge_tp8_dispatch/results/c1_fixed_batch_fla_integration_b300_sm103a_r3.json)
（最终 JSON SHA256 `b3b2fb61...ff8657a`）；对应 clean logs 为
[10740](challenge_seqcount_dispatch/results/c1_seqcount_dispatch_b300_sm103a_r2_job10740.log)、
[10771](challenge_seqcount_dispatch/results/c1_fixed_batch_confirmation_b300_sm103a_r1_job10771.log)、
[10784](challenge_seqcount_dispatch/results/c1_fixed_batch_release_gate_b300_sm103a_r1_job10784.log) 和
[10810](challenge_tp8_dispatch/results/c1_fixed_batch_fla_integration_b300_sm103a_r3_job10810.log)。

job11781 启动时，原固定批量矩阵仍遗漏夹在 B4/B6 之间的 `B=5,H=12,T=2048`，而上面的 M-only 反例说明
不能从相邻 batch 外推。续轮因此只新增 discovery-only runner，不修改 dispatcher 或 public
FLA 映射。clean job 11781 在两个新进程中各做两次 1000-sample repeat；baseline、vshard2-P2、
vshard4-P2 的 4 个 raw ABI 和 3 个 pinned reference 对拍在两进程中均逐位通过。3 个 public
state contract 的 36 个 `(repeat,percentile)` 比较全部由 vshard2-P2 胜出：

| B5 discovery contract | 共同 winner | 4 repeats × P50/P95/P99 最小相对次优裕量 | discovery 裁决 |
| --- | --- | ---: | --- |
| none | vshard2-P2 | 6.202% | 可进入独立 allocation 确认 |
| FP32 final only | vshard2-P2 | 5.934% | 可进入独立 allocation 确认 |
| FP32 both | vshard2-P2 | 2.292% | 可进入独立 allocation 确认 |

其中 FP32-both 只略高于预注册的 2% 门，故不能直接发布。独立审计还固定 runner/analyzer/
shell SHA256 为 `45f54fd2…1eae`/`7f93cb81…35d`/`3cf303a2…3a3a`，验证两个 fresh PID、
同一物理 GPU UUID、同一已审计 SO 与 pinned Torch reference；PRE/BETWEEN/POST 均为 0 MiB，
`FINAL_RC=0`。其 `second_allocation_decision` 仅为 true，不是生产资格；证据见
[B5 discovery 审计](challenge_seqcount_dispatch/results/c1_fixed_batch_b5_discovery_b300_sm103a_b5_r1.independent_audit.json)、
[本地 raw 复算](challenge_seqcount_dispatch/results/c1_fixed_batch_b5_discovery_b300_sm103a_b5_r1.local_recompute.json)
与 [job 11781 clean 日志](challenge_seqcount_dispatch/results/c1_fixed_batch_b5_discovery_b300_sm103a_b5_r1_job11781.log)。

独立 confirmation job 11782 在新 Slurm allocation、`seed=20260830` 和两个新 PID 上重复同一
冻结测量协议；chain analyzer 直接重读两份 raw main、核对其 SHA 与完整 seed 公式，并要求
历史/当前 raw SHA 集合、job ID、日志路径均不相交。当前 allocation 的 36 个分位比较仍全部
选择 vshard2-P2，none/FP32-final-only/FP32-both 的最小裕量分别为 **6.416%/5.540%/2.227%**；
结合历史 4 repeats 后，三个 contract 的 8 repeats × P50/P95/P99 全部通过 2% 门。job11782
为 `COMPLETED/0:0`，与 discovery 使用同一物理 GPU；main0/main1 POST 与 main1 PRE 均为
2032 MHz。所有 correctness/immutability gate 通过，
PRE/main 间/POST 均为 0 MiB，日志只出现一次 `PUBLIC_INTEGRATION_ELIGIBLE=true` 和一次
`FINAL_RC=0`。这仍只授予 public-integration review，不自动修改 dispatcher；证据见
[confirmation chain](challenge_seqcount_dispatch/results/c1_fixed_batch_b5_confirmation_b300_sm103a_b5_confirm_r1.confirmation_chain.json)、
[本地 chain 复算](challenge_seqcount_dispatch/results/c1_fixed_batch_b5_confirmation_b300_sm103a_b5_confirm_r1.confirmation_chain.local_recompute.json)、
[当前 measurement audit](challenge_seqcount_dispatch/results/c1_fixed_batch_b5_confirmation_b300_sm103a_b5_confirm_r1.measurement_audit.json)
与 [job11782 clean 日志](challenge_seqcount_dispatch/results/c1_fixed_batch_b5_confirmation_b300_sm103a_b5_confirm_r1_job11782.log)。
对应 SHA256 为 chain `5f817df8…f963`、本地 chain `faddd30f…3f79`、measurement audit
`b589cd95…fe4b`、本地 measurement audit `489f9535…0aa4`、完整日志 `fd6c7a2a…54d9`。

通过 confirmation 后才修改真实 production candidate：fixed shape gate 只新增 B5，精确表只
新增上述三个 vshard2-P2 cell，并更新 FLA backend 的 registry compatibility token；B4/B6
FP32-both、B7/B8、错 shape/state 和原 varlen 表均保持 fail closed。独立静态复核与本地/远端
当时冻结 source 的 CPU 门为 dispatcher 25/25、metadata 11/11，describe 矩阵恰为
18 cell = 13 正 + 5 负；当前 v5 的 27/27 见后文补轮。
clean public job11786 随后通过真实 pinned backend、direct custom backend 与
`fla.ops.kda.chunk_kda` 复核完整 18-cell 矩阵：18/18 output、所有应有 final state 都逐位
等于 pinned；每个 public call 的 registered-backend spy 恰 `+1`，紧邻读取的 decision 与表
一致。B5 三格全部命中 vshard2-P2，旧 15 格无回归。作业为 `COMPLETED/0:0`，PRE/AFTER/POST
均为 0 MiB，日志固定本轮 dispatcher/backend/runner/test SHA，且只有一个 `FINAL_RC=0`。
证据见 [18-cell public JSON](challenge_tp8_dispatch/results/c1_fixed_batch_fla_integration_b300_sm103a_b5_public_r1.json)
（SHA256 `7867854f…18a1f`）、[describe plan](challenge_tp8_dispatch/results/c1_fixed_batch_fla_integration_b300_sm103a_b5_public_r1.plan.json)
与 [job11786 clean 日志](challenge_tp8_dispatch/results/c1_fixed_batch_fla_integration_b300_sm103a_b5_public_r1_job11786.log)
（SHA256 `1767cdea…6e94`）。

最后用不再改动的四份 production source 做 fresh-allocation freeze。job11787 以新
`seed=20260831` 再跑完整 18-cell public runner，18/18 逐位通过；raw JSON SHA256
`feb68401…3850` 与 job11786 的 `7867854f…18a1f` 不同，job ID/日志路径也不同。作业
`COMPLETED/0:0`、POST 0 MiB、唯一 `FINAL_RC=0`，完整日志 SHA256 `1ffd36c8…b259`。
初版 freeze analyzer 随后被 reviewer 用 JSON float shape `5.0` 击穿类型门；它在 job11787
生成的 `...b5_prod_freeze_r1.production_freeze.json`（SHA256 `f7f74a7b…960e`）
**明确废弃，不作为证据**。[修正版 analyzer](challenge_tp8_dispatch/analyze_fixed_batch_b5_production_freeze.py)
把 initial、expected-final、direct/public actual/pinned 六类 state shape 的每一维都强制为
exact JSON integer，SHA256 为 `f6ae4192…4347`；调用它的
[freeze shell](challenge_tp8_dispatch/run_clean_fixed_batch_b5_production_freeze.sh) SHA256 为
`df8c5672…8279`。12 个定向 float/bool 篡改，以及 333 个 int、328 个 bool、60 个 float、
293 个 object、72 个 array 的系统变异均零漏过。修正版先对 job11786+11787 冻结输入离线重算
得到 exact `production_freeze_passed=true`；对应交叉复核为
[远端 strict audit](challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r1.production_freeze.strict_recompute.json)
（SHA256 `c912f6d3…c691`）与
[本地 strict audit](challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r1.production_freeze.strict_local_recompute.json)
（SHA256 `d1a68866…4672`），以及
[job11787 raw public JSON](challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r1.json)
和 [clean 日志](challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r1_job11787.log)。
为让严格 analyzer 与 GPU run 同属一个受审作业，修正版 shell 又在 fresh job11788 内完整
重跑：18/18 public cell 通过，随后严格 analyzer 原位写出 `production_freeze_passed=true`；
作业 `COMPLETED/0:0`、POST 0 MiB、唯一 `FINAL_RC=0`。最终主 freeze 证据为
[job11788 freeze JSON](challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r2_strict.production_freeze.json)
（SHA256 `38bc8bba…b7f5`）、[raw public JSON](challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r2_strict.json)
（SHA256 `feb68401…3850`）和 [clean 日志](challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r2_strict_job11788.log)
（SHA256 `8f037aa9…2294`）。

### 3. 长上下文：分离 FP32 state ABI 与 FP32 recurrence

原生 kernel 的 FP32 state ABI 只是把 FP32 state 载入后转为 BF16 计算，再把 BF16 state
转回 FP32 存储；它不是 FP32 recurrence。新 runner 先把 FLA fused recurrent oracle 与
naive FP32 实现校准到 output/state 相对 L2 `1.29e-7/5.53e-8`，再比较真实 BF16
persistence。

正式作业包括 random 与 retention-stress 两种 regime：`H=1` 用 32 段 × 8192 token、
seed 0–3，总长 262,144；`H=12` 用 16 段 × 8192、seed 0–1，总长 131,072。12 组主实验
均完成。random 的 BF16-persisted output 对 FP32 recurrence 相对 L2 约为 0.005；stress
约为 0.009。专门比较“段间持久化 BF16 state”和“每段从 FP32 oracle 重置”的累积项：
stress output 最大约 0.0042、state 最大约 0.0029，曲线没有出现随段数失控增长或
NaN/Inf。完整逐段误差见
[full JSON](challenge_long_context_quality/results/c1_long_context_b300_sm103a_full_r1.json)。

这完成的是合成 recurrence 的多 seed/多段数值研究。仓库没有真实模型权重、数据集、
perplexity 或下游任务 harness，因此不能把上述误差翻译成模型质量通过；该部分仍是外部
依赖，而不是继续增加随机 seed 就能闭合的问题。

### 4. 跨长度表：只把实测格点写进 policy

[cross-length JSON](challenge_cross_length/results/c1_cross_length_b300_sm103a_r1.json)
覆盖 H12 的 `T=128/256/512/1024/2048/4096/8192/16384/32768/65536`。10 个长度 × 4
state contract 共 40 个 cell 全部 exact，vshard4-P2 在 P50/P95/P99 三个分位数上均胜
vshard2-P2。BF16-both 的两个额外交互切片 `T=2048/32768 × H=1/12/37/38/64/96`
也在三个分位数上给出同一边界：H1/12/37 选 vshard4-P2，H38/64/96 选 vshard2-P2。

所以 policy 对 H12 四种实测 state 使用上述十个长度；BF16-both 另保留 `T=8192` 的
H1–96 全轴和 `T=2048/32768` 的六个交互点。`T=257`、H11/T4096 等未测组合明确回退。
这闭合了 B300 的代表性长度轴，但没有将结果外推到其他 GPU 架构。

### 5. 严格 BF16 tensor-contraction roofline

只对 NCU 能逐指令计数的 dense BF16→FP32 HMMA 定义 FLOP；不把标量、SFU、地址和
控制操作强行折算。固定 H12 profile 中：

`N_tile = BHT/C`

`F_TC,K1 = N_tile × 4C²D`

`F_TC,K2 = N_tile × (6CD² + 4C²D)`

理论整数分别为 805,306,368 与 10,468,982,784，必须与 NCU tensor-op counter 完全
相等；[分析脚本](challenge_roofline/analyze_roofline.py) 不相等即失败。

| phase | `F_TC` | DRAM bytes | `I_TC` (F/B) | achieved | ridge | selected roof | roof efficiency | branch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| K1 prepare | 805,306,368 | 109,220,864 | 7.3732 | 19.4181 TF/s | 75.3134 | 56.4896 TF/s | 34.37% | memory |
| K2 recurrence | 10,468,982,784 | 126,834,432 | 82.5405 | 22.2404 TF/s | 75.5125 | 579.3050 TF/s | 3.84% | compute |

K1 是 memory-roof 分支。K2 虽在 ridge 右侧，却离 compute roof 很远；结合前文 48-CTA
grid、低 eligible warp/SM throughput，应解释为 compute-side 的延迟/并行度受限点，
不能改写成“算力饱和”。机器可读结果和口径说明见
[roofline JSON](challenge_roofline/results/c1_h12_bf16_tensor_roofline_r1.json) 与
[中文摘要](challenge_roofline/results/c1_h12_bf16_tensor_roofline_r1.md)。

### 6. P3 可行性裁决与 V8 最终实验

只读调用链分析确认：当前 K2 的每个 CTA 已经沿一个 sequence/head/value-shard 持有
state 并连续处理所有 time tiles，因此它本身就是该任务粒度上的 persistent kernel；再建
persistent worker 不会消除 launch，multi-head CTA 反而减少独立 CTA 数。跨 CTA 切时间
需要每个 tile 的全局状态交接/同步，不属于低风险优化。tcgen05 则需重排 shared/TMEM、
barrier 和现有多阶段 fragment 数据流；已有同逻辑 CTA microbench 在 `C=16/64` 都输给
HMMA。`CHUNK=64` 的 K2 shared-memory 保守估算约 314 KiB，超过当前 200,704 B 配置；
`CHUNK=32` 虽约 165 KiB，但 normalization、beta storage、inverse 和多个 16×16 fragment
均硬编码，且已有 BF16 decay 下溢边界。它们在本预算和当前结构下是有负证据的停止项，
不是尚待盲做的常量替换。

唯一值得实际实现的低-grid 分支是把 V=32 四分片继续拆为 V=16 八分片，使 H12 K2
grid 从 48 增至 96 CTA。结果分两步：

- [V8-P2](challenge_vshard8_prefetch2/) clean SM103a build 成功，但正式 BF16-state
  实例为 56 registers、9 barriers、8-byte stack、12-byte spill stores、8-byte spill
  loads；job 10704 按预设资源门在任何 GPU kernel 前停止，故没有 P2 correctness/性能。
- [V8-P1](challenge_vshard8/) 去掉双缓冲后为 58 registers、9 barriers、0 stack、0
  spill。small `T=256,H=1/2/4` 的四 contract 对 baseline/Torch reference 全部逐位一致；
  H12 四 contract 对 baseline 也全部逐位一致。

V8-P1 的 H12 性能门仍明确失败，每路径 1000 个四路循环样本：

| contract | vshard4-P2 P50/P95/P99 (ms) | V8-P1 P50/P95/P99 (ms) | V8-P1 P50 相对慢 | 三分位胜出 |
| --- | --- | --- | ---: | --- |
| none | 0.528640 / 0.531168 / 0.534467 | 0.599200 / 0.601541 / 0.604835 | 13.35% | 否/否/否 |
| BF16 both | 0.529856 / 0.531872 / 0.534977 | 0.578528 / 0.580098 / 0.582145 | 9.19% | 否/否/否 |
| FP32 both | 0.532192 / 0.535462 / 0.538561 | 0.595520 / 0.597635 / 0.601217 | 11.90% | 否/否/否 |
| FP32 final only | 0.523600 / 0.525312 / 0.532132 | 0.583424 / 0.586080 / 0.591432 | 11.43% | 否/否/否 |

因此不再做 H1–18 sweep 或 NCU，八分片不进入 dispatcher。P2 的 spill 和 P1 的稳定
负性能共同说明：当前 H12 已不是“继续增加 value shards”就能改善的方向。

## 最终完成度补轮：直接 FLA 溯源、B7 与 `InputStages=4`

| 变量 | 含义 | 本节口径 |
| --- | --- | --- |
| `B_f` | fixed launch 的 batch 数 | 补测点为 `B_f=7` |
| `S` | K2 software-pipeline 的 `InputStages` | current 为 `S=3`，候选为 `S=4` |
| `j` | 同一 fresh PID 内的 repeat 下标 | `j∈{0,1}` |
| `q` | 延迟分位下标 | `q∈{P50,P95,P99}`，每路径 1000 samples |
| `δ_{j,q}` | 第 `j` 次 repeat 中 winner 相对 runner-up 的裕量 | `(t_{runner-up,j,q}/t_{winner,j,q})-1`；B7 门为每格至少 2% |
| `ρ_{c,q}` | contract `c` 上 S4 相对 current 的加速 | `t_{P2S3,c,q}/t_{S4,c,q}`；S4 门为每格至少 1.02x |
| `A` | 相互独立的 clean Slurm allocation 数 | 候选必须先过首轮才允许申请 `A=2` |

### 1. 强制走 pinned FLA Triton `chunk.py`

此前 public FLA 集成证明了 registry/dispatch，却可能由已注册 backend 截获，不能单独证明
Triton 参照确实执行。job12216/r3 因而在 import 前设置 `FLA_DISABLE_BACKEND_DISPATCH=1`，
验证 dispatch 已禁用并固定 FLA `a3edffc` 与 candidate SO。官方 `test_fwd_vs_fla` 和
`test_fwd_varlen_vs_fla` 均报告 `Assert results: Success`；fixed 与 varlen 各 10 个 case，
candidate FlashKDA、FP64 fused-recurrent gold、FLA Triton chunk 各有 20 次非空 output/final-state
调用。这里沿用 upstream 的实际判定强度：candidate output 对 gold 是 hard tolerance；candidate/
chunk final state 是 warning，chunk output 只报告 error ratio/绘图。Triton chunk 对 gold 的最大
output/state error ratio 为 fixed `5.409708e-3/6.832830e-3`、varlen
`6.088918e-3/3.617549e-3`，不被事后改写为 hard gate。PRE/AFTER/POST 均为 0 MiB、
`FINAL_RC=0`。主证据为
[r3 JSON](challenge_fla_chunk_validation/results/c1_fla_chunk_validation_b300_sm103a_fla_chunk_r3.json)
（SHA256 `4564b7e3…5cd2`）与
[clean log](challenge_fla_chunk_validation/results/c1_fla_chunk_validation_b300_sm103a_fla_chunk_r3_job12216.log)
（SHA256 `cec980cb…e75`）；[自包含复现说明](challenge_fla_chunk_validation/README.zh-CN.md)
明确区分这个 direct-source gate 与 production dispatch gate。r1/r2 只是 provenance wrapper/
Python header 的 fail-closed 配置失败，不作为结果。

### 2. `B_f=7`：单 contract 有信号，但全契约门否决发布

[B7 runner](challenge_fixed_batch_b7/run_fixed_batch_b7_discovery.py) 对
`B=7,H=12,T=2048,D=128` 的四种 raw ABI 完成 baseline/vshard2-P2/vshard4-P2 exact，
并在适用处对 pinned Torch reference exact。随后两个 fresh PID 各做两次 repeat，每次三路径
各 1000 samples；[独立审计](challenge_fixed_batch_b7/results/c1_fixed_batch_b7_discovery_b300_sm103a_b7_r1.independent_audit.json)
直接重读 [main0](challenge_fixed_batch_b7/results/c1_fixed_batch_b7_discovery_b300_sm103a_b7_r1_main0.json)
与 [main1](challenge_fixed_batch_b7/results/c1_fixed_batch_b7_discovery_b300_sm103a_b7_r1_main1.json)：

| public contract | 4 repeats × 三分位共同 winner | 裕量范围 | 单格门 | 全契约决策 |
| --- | --- | ---: | --- | --- |
| none | vshard2-P2 | 2.285%–3.526% | 通过 | 不单独晋升 |
| FP32 final only | baseline | 1.254%–2.209% | 失败 | baseline |
| FP32 both | baseline | 1.715%–3.024% | 失败 | baseline |

预注册规则要求三个 public contract 作为一组都由 optimized path 胜出且所有 `δ_{r,p}≥2%`，
所以 `second_allocation_decision=baseline_stop_no_second_allocation`；没有事后把 `none` 拆出来，
也没有修改 production mapping。完整 [plan](challenge_fixed_batch_b7/results/c1_fixed_batch_b7_discovery_b300_sm103a_b7_r1.plan.json)
与 [job12216 log](challenge_fixed_batch_b7/results/c1_fixed_batch_b7_discovery_b300_sm103a_b7_r1_job12216.log)
同时归档。若未来重启，`none` 只能作为**新预注册、单 contract、跨 allocation**的候选，不能
把本轮 discovery 追认成 release gate。

### 3. `InputStages=4`：资源与 exact 通过，性能近似持平

独立候选从 clean FlashKDA `1ce47ea` fresh build，保留 baseline、vshard2-P2S3、
vshard4-P2S3，并只新增非生产 ABI `fwd_vshard4_p2s4`。14 个 S4 实例使用 56–59 registers；
formal BF16 fixed both-state 实例为 59 registers、9 barriers、zero stack/spill。证据为
[build log](challenge_inputstages4/results/c1_inputstages4_build_b300_sm103a_r1.log)、
[ptxas JSON](challenge_inputstages4/results/c1_inputstages4_build_b300_sm103a_r1.ptxas.json)
（SHA256 `b3660136…ad88`）和
[CUBIN resource dump](challenge_inputstages4/results/c1_inputstages4_build_b300_sm103a_r1.cuobjdump.txt)；
dump 中的 1024-byte 记录是静态 shared 证据，不能误写成 launch 的总动态 shared memory。

[small matrix](challenge_inputstages4/results/c1_inputstages4_b300_sm103a_h12_r1_small_matrix.json)
在 `H=1/2/4` 的四 contract 全 exact；
[H12 raw JSON](challenge_inputstages4/results/c1_inputstages4_b300_sm103a_h12_r1_h12_all_contracts.json)
也在四 contract 的 output/final state 全 exact。相对 current vshard4-P2S3，首个 clean allocation
的 `ρ_{c,p}` 为：

| contract | `ρ_{c,50}` | `ρ_{c,95}` | `ρ_{c,99}` |
| --- | ---: | ---: | ---: |
| none | 0.997762x | 0.998689x | 1.001421x |
| BF16 both | 1.003970x | 1.003327x | 1.000347x |
| FP32 both | 1.004289x | 1.005954x | 1.006730x |
| FP32 final only | 0.998438x | 0.997336x | 0.994234x |

[one-allocation gate](challenge_inputstages4/results/c1_inputstages4_b300_sm103a_h12_r1_one_allocation_gate.json)
给出 `performance_pass=false`、`publication_eligible=false`；首轮没有一个 contract 的三分位都
达到 1.02x，因此按门禁止第二 allocation。clean [job12216 log](challenge_inputstages4/results/c1_inputstages4_b300_sm103a_h12_r1_job12216.log)
为 `FINAL_RC=0` 且 PRE/POST 0 MiB。S4 不接 dispatcher，production P2S3 保持不变；完整哈希
与复现边界见 [S4 README](challenge_inputstages4/README.zh-CN.md)。

### 4. 当前最高优先级补轮：负门、T8191 与 skew v5

| 变量 | 含义 | 本节固定口径 |
| --- | --- | --- |
| `A` | 独立 Slurm allocation | `A1`、`A2`；首轮负门不申请 A2 |
| `p,j` | 资格/production 门的 fresh PID 与 repeat 下标 | 每个 allocation 各两个 PID，每 PID 两轮；cross-map 不使用 `j` |
| `q` | 延迟分位 | `P50`、`P95`、`P99` |
| `r^{gate}_{A,p,j,q}` | 资格/production 门的对照路径延迟除以 C1 延迟 | `r>1` 表示 C1 更快 |
| `δ^{gate}_{A,p,j,q}` | 资格/production 门中 C1 相对对照的裕量 | `r^{gate}-1`；逐格要求至少 2% |
| `c_{map}` | cross-map 预注册的正向 cell | 四个正向 cell；每个 PID/cell 只有一组 100-sample 对拍，不设 repeat `j` |
| `r^{cm}_{A,p,c_{map},q}` | cross-map 非发布 sentry 的 `pinned/C1` | 仅要求逐格 `r^{cm}>1`，不是 2% 发布门 |
| `o_i` | 第 `i` 个 CPU-authoritative packed offset | skew 为 `(0,1,2,3,4,5,12288)` |

#### 4.1 phase-1 fragment prefetch：正确但更慢，首轮停止

[job12401 gate](challenge_phase1_fragment_prefetch/results/c1_phase1pf_phase1pf_a1r3_one_allocation_gate.json)
固定候选 `fwd_vshard4_p2_phase1pf`；四个 output/final contract 都 exact，正式 BF16 实例
zero-spill，shared-memory evidence 也通过。可是相对 frozen current 的全局最小速度比只有
P50 `0.981869`、P95 `0.982007`、P99 `0.983188`，即速度比分别低于 1 达 1.813、1.799、1.681 个百分点。
因此 `performance_pass=false`；按预注册规则不做 A2、不接 dispatcher。这是完整负结果，
不是“尚未尝试”的方向。

#### 4.2 B7 none-only：新协议推翻旧 discovery 的表面正信号

旧 `challenge_fixed_batch_b7` 的 none cell 只是在三路径 discovery 内胜出，且其全契约门已经
禁止发布。新的 [none-only 协议](challenge_fixed_batch_b7_none/README.zh-CN.md) 不读取、不合并
旧样本；schema-3 job12570 直接加载固定 helper、`no_build=true`，四个 raw ABI 与真实 public
output/final-state 均 exact。其四轮 `δ_{A1,p,j,q}` 为：

| PID / repeat | P50 | P95 | P99 |
| --- | ---: | ---: | ---: |
| 2283616 / 0 | -1.03127% | -1.78295% | -2.73296% |
| 2283616 / 1 | -0.93992% | -1.46609% | -2.52822% |
| 2284814 / 0 | -0.95521% | -3.47353% | -4.14279% |
| 2284814 / 1 | -0.85771% | -1.39858% | -2.28954% |

[A1 audit](challenge_fixed_batch_b7_none/results/c1_fixed_batch_b7_none_b300_sm103a_b7_none_r3_helper_nobuild_A1.allocation_audit.json)
由 raw samples 重算得到 `eligible=false`、`STOP_keep_production_baseline`；shell 的
`FINAL_RC=95` 是预期门拒绝。故不申请 A2、无 chain、production B7 继续 baseline，且不能
再把历史 discovery 追认为第一轮 release evidence。

#### 4.3 T8191：test-only 资格与真实 production freeze 已分层闭合

固定 `B=1,H=12,T=8191,K=V=128` 的 `none`、`fp32_final_only` 先由
[test-only 协议](challenge_tail8191_dispatch/README.zh-CN.md)在 job12406/12415 两个
allocation 通过；两 allocation 合计 48 个 repeat×分位格（每 allocation 24 个）的全局最小裕量为 35.775%。这只授权进入 production
复验，不能单独称发布。最终 v5 source 不再改动后，真实 public registry 的
[production 协议](challenge_tail8191_production_freeze/README.zh-CN.md)又在 job12592/12593
完成 A1/A2：四份 schema-4 main 均为 fresh PID、helper no-build、真实 v4 route、raw/public/
Torch-reference exact；本地从 16 个 raw timing 格重算得到生产链全局最小 `pinned/C1=1.3549231148`
（35.4923% 裕量），所有性能格过 2%。[A1→A2 chain](challenge_tail8191_production_freeze/results/c1_tail8191_production_freeze_b300_sm103a_t8191_prod_v5_helper_r1_A1_A2.chain.json)
SHA256 为 `e5b2981e…1c33`，写出 `production_freeze_passed=true`。因此 T8191 不再属于
production pending；历史 test-only JSON 与当前 chain 仍保持证据角色分离。

#### 4.4 skew/FP32-both：资格、source 集成与真实 production freeze 已分层闭合

[schema-3 test-only freeze](challenge_varlen_fp32_both/results/c1_varlen_fp32_both_b300_sm103a_A1_A2_r5c.freeze.json)
绑定 `o=(0,1,2,3,4,5,12288)`、`H=12,T_total=12288,fp32_both`。job12555/12556
四个 fresh PID 的 output/final 均 exact；A1 最差 P50/P95/P99 为
`1.105265/1.090924/1.053236`，A2 为 `1.104953/1.093693/1.051927`，telemetry 高频占比
`0.9032/0.8629` 均有效。它只给出 public-freeze eligibility。当前 production source 已新增
该 tuple/state → `vshard4_p2`，compatibility token 升为 v5，并将“选中 v4 但 symbol 缺失”
改为 pre-launch baseline，禁止静默降 v2；27 个 policy 与 11 个 metadata CPU tests 全过。

真实 production 预试 job12598 在目标 public route、direct/pinned/reference exact 后，因 runner
负控复用 stale handoff 而失败；该 job 不计 A1。修复版在每个控制前清理 handoff/cache，以
verifier/issuer spy 证明当前 CPU/GPU offsets，并对邻接 offsets 精确 reject；独立复核 P0/P1/P2
均无。随后 fresh A1 job12770 与不同 allocation 的 A2 job12771 各运行两个 fresh PID：四份 raw
均通过目标真实 public `vshard4_p2`、相邻 state 的 `vshard2_p2`、相邻 offsets 的 fail-closed
baseline、v4-symbol-missing baseline、逐 bit exact、source/map 不变与清卡门。A1/A2 allocation
manifest SHA256 分别为 `7fc7d86c…a3dd4`、`baf2126d…2f51`；最终
[production freeze](challenge_varlen_fp32_both_production_freeze/results/c1_varlen_fp32_both_production_b300_sm103a_skew_production_r2_isolation_A1_A2.freeze.json)
SHA256 为 `bafa65f8…ad12`，写出 `eligible_for_production_freeze=true`、`complete=true`。因此
job12598 继续只作为协议缺陷记录，不污染 fresh A1/A2；该精确 tuple/state 已不再属于 production
pending，也不据此外推任意 packed layout。

#### 4.5 v5 public-registry 跨映射回归 sentry（非发布）

最终 v5 source 不再变化后，[cross-map 协议](challenge_v5_crossmap_regression/README.zh-CN.md)
把四个已经有独立发布证据的精确 cell 放进同一只读回归矩阵：fixed-B2/FP32-both→v4、
fixed-B5/FP32-both→v2、fixed-T8191/none→v4，以及 skew/FP32-both→v4；同时保留
B7-none、T8191/FP32-both 与邻接 offsets `(0,1,2,3,4,6,12288)` 三个 baseline/fail-closed
负控。fresh A1 job12958 与 A2 job12959 各有两个 PID，四个正向 cell 的 public/direct/
pinned/reference 全 exact，负控全部按预注册路径回退；每个正向 cell、每个 PID 的
P50/P95/P99 均满足非发布门 `pinned/C1>1`。48 个比值的全局最小为 `1.046089209043`，
出现在 A1/PID1 的 fixed-B5、P99。

[最终 chain](challenge_v5_crossmap_regression/results/c1_v5_crossmap_b300_sm103a_v5_crossmap_r5_a2fix_A1_A2.chain.json)
SHA256 为 `33ff98ed…e55f17`，重开两份 allocation audit 与四份 raw 后写出
`production_freeze_passed=true`；production map digest 始终为 `a4fb43fb…2117b`，没有 source/map
改动。A1/A2 是不同 Slurm job，但恰好使用同一 GPU UUID，因此这里只是 allocation/job 隔离，
不能称跨 GPU 重复。job12828/12882（重型 import 早于 clean gate）、job12911（缺
`shared._make_inputs` runtime binding）与 job12929（旧 analyzer 的 A2 typed-label 缺陷）都在
有效 audit/chain 前失败，明确不计 A1/A2、性能或 correctness/freeze 证据。这个 sentry 只降低
“已发布精确格点互相回归”的风险，不是 `>=2%` 发布门，不新增任意 varlen、shape、state 或架构结论。

### 5. 任务、讨论与交付闭合矩阵

| TASK 要求 | 状态 | 最终证据与边界 |
| --- | --- | --- |
| 任务 1：B300 复现、官方 benchmark、NCU/SASS | **完成** | clean B300 benchmark、官方形状计时、NCU raw counter 与 SASS HMMA 计数均已归档 |
| 任务 2：六项讨论逐项“结论 + 证据” | **完成讨论口径** | 下表逐项闭合；量化项均有推算与 microbench/profile，不把代理或边界外数据冒充结果 |
| 任务 3：任选 SM100 切面挑战 | **完成** | value-shard/P2S3 已实现、fresh build、exact、性能、dispatch 与负候选停止树均齐全；题目允许无正收益，本实现另有 H12 正收益 |
| 交付：代码 + 报告 + 答辩 | **核心交付与已集成格点的生产冻结完成** | challenge 源码与工件、本文，以及 [DEFENSE.zh-CN.md](DEFENSE.zh-CN.md) 均为自包含入口；T8191 与 skew/FP32-both 均保留 test-only→source→真实 production 的证据分层 |

| 讨论点 | 状态 | 结论边界 |
| --- | --- | --- |
| D1 `CHUNK=16` | **完成** | BF16 衰减、Neumann 工作量与硬件 tile 三条证据；32/64 为代理 microbench，不伪称重编译 kernel |
| D2 tcgen05 | **完成** | 同逻辑/同 CTA 公平对照、exact/SASS/事件齐全；结论是现 tile 低利用率，不伪称 tcgen05-KDA 吞吐 |
| D3 有状态递推并行度 | **完成** | sequence/head/value-shard 三轴分析；2/4/8 CTA、H1–96 与边界 NCU 给出正负证据 |
| D4 compute/memory | **完成** | 同一 NCU profile 的 BF16 HMMA FLOP 与 DRAM bytes 闭合；K1 memory-roof、K2 越 ridge 但仅 3.84% compute roof |
| D5 BF16 state 精度 | **层 1/2 完成；层 3 外部阻塞** | kernel exact 与最长 262,144-token synthetic recurrence 已完成；真实模型 logits/perplexity/任务质量因权重、数据、launcher 缺失而不能执行 |
| D6 是否发布 sm100a v2 | **production policy/source 与已集成格点冻结完成** | 发布受限 B300 白名单 current，保留 SM80 fallback；T8191 与 skew/FP32-both 均有不同 job 的真实 production A1/A2；未测 shape/架构与 full TP8 绝不外推 |

## 仍未完成的挑战与后续方向

| 后续优先级 | 仍未完成 | 当前阻塞/边界 | 真正的下一成功门 |
| --- | --- | --- | --- |
| P0 | 8-rank 并发 FLA/TP8 | [2026-08-30 探针](challenge_tp8_dispatch/results/c1_tp8_quota_reprobe_20260830.txt) 显示 2/4/8 GPU 均触发 `QOSMaxGRESPerUser`，1 GPU 可申请；当前 JSON 只观测 1 rank | 获得 8 张同时可见 B300；8 rank 三个 FLA public contract 全 exact、白名单/回退正确，并报告 rank-max P50/P95/P99 |
| P0 | 真实 Kimi 模型端到端与任务质量 | [资产探针](challenge_long_context_quality/results/c1_real_model_asset_probe_20260830.txt) 在可访问根下按列出的后缀与 `>100 MiB` 阈值找到 0 个大权重候选，并找到 0 个 C1 模型评测 launcher；仓库也没有数据集 | 提供固定 commit 的模型权重、tokenizer、评测数据和 TP launcher；在真实调用链验证 dispatch，并给 logits/perplexity/下游任务 gate |
| P1 | 任意 layout 的通用 varlen 生产 dispatch | descriptor handoff 与 metadata 可信链已经完成；当前 exact skew tuple 有三个 state cell，但 54-case 已证明 `N_seq×H` 不足以外推 winner | 在真实调用方证明 CPU offsets 生命周期，并让每个新 distribution/state 独立通过 raw 与 public 门；未列 cell 继续 pre-launch fallback |
| P2 | 当前组合候选的跨架构表 | 本轮新证据只属于 B300 SM103a/148 SM；旧 5090 负例不能代替当前组合复测 | 每个目标架构 fresh build、ptxas/SASS、all-state exact、full-call P50/P95/P99；未测架构继续 baseline |
| P2 | 旧 H64/H96 “再快 10%”严格门 | H12 条件性 current 不改写旧目标；P3、V8 和继续加 shard 均已给出负证据 | 需要新的 K2 排程/算法级候选先过零 spill与 exact，再相对 frozen current 达到原绝对阈值 |

严格 roofline、代表性跨长度、测试域内 tail/batch/varlen、synthetic 长上下文 recurrence、
强制 FLA Triton 对拍和 V8/S4 停止树已不再列为“未做”。tcgen05/CHUNK32/64、
persistent/multi-head、继续增加 stage/shard 也不应在没有新算法证据前重复投入；它们是本预算
下完成的可行性否决，不是成功实现。

## 交付自检与证据边界

- [x] B300 官方 fixed/varlen 复现、5090 fixed 对照、清卡审计。
- [x] SASS 证明官方 SM80 HMMA（HMMA=1544，WGMMA/UTCOMMA/UTCHMMA=0）；公平完整 CTA 对照另有 HMMA=14、UTCHMMA=4。
- [x] 2/4 CTA-per-head 与 P2S3：baseline/Torch exact、双 clean H12 repeat、H1–96 sweep、H37/H38 boundary NCU。
- [x] FLA public backend 与 opt-in 自动 dispatch：dispatcher policy tests 为 27/27、metadata tests 为 11/11；one-shot descriptor handoff 将真实 public C1 `_prepare_varlen` 降为一次；v5 exact map 增加 skew/FP32-both，选中 v4 但缺 symbol 时直接 baseline，不静默降 v2。
- [x] pinned FLA Triton `chunk.py` direct-source gate：禁用 backend dispatch 后，官方 fixed/varlen 测试均成功，三条真实实现各调用 20 次，clean job12216/r3 归档。
- [x] skew/FP32-both 性能资格：历史失败边界保留；新 schema-3 test-only 以四 fresh PID、A1/A2、telemetry 与本地独立复算通过，source 已按 v5 集成。
- [x] skew/FP32-both 真实 production re-freeze：job12598 因协议负控隔离失败不计 A1；修复后 job12770/12771 的四份 fresh raw、两份 allocation manifest 与最终 freeze 均通过，`eligible_for_production_freeze=true`。
- [x] fixed T8191 production：test-only 资格与 job12592/12593 真实 registry A1/A2 分层完成，chain `production_freeze_passed=true`。
- [x] v5 cross-map 非发布 sentry：job12958/12959 的四正例、三负控、48 个分位比与 A1→A2 chain 均通过；同 GPU UUID、map 不变，未外推为跨 GPU 或新发布格点。
- [x] tail/batch/varlen：11 shape × 4 state contract exact，5 个 sanitizer case 0 error；性能正负例均保留。
- [x] sequence-count/fixed batch：54-case M-only 反例；原三轮 raw-sample 门与 B5 两-allocation 补轮；13 个精确 fixed public cell 经 18-cell FLA registry 审计及 fresh-seed production freeze 发布，5 个负控回退。
- [x] fixed B7：历史全契约 discovery 停止；新 none-only schema-3 job12570 四 raw/public exact，但所有 12 个 repeat×分位 margin 为负，A1 有效否决、无 A2、未改生产表。
- [x] phase-1 fragment prefetch：exact/zero-spill/shared-memory 通过，但三分位速度比低于 1 达约 1.68–1.81 个百分点，首轮按门停止。
- [x] 长上下文：FP32 oracle 校准，H1/H12、random/stress、多 seed、最长 262,144 token 的逐段误差 JSON。
- [x] 跨长度：H12 十长度四 contract 共 40 cell，以及 T2048/T32768 六个 H 交互点，P50/P95/P99 winner 一致。
- [x] 严格 tensor-contraction roofline：理论 HMMA FLOP 整数与 NCU counter exact，raw CSV SHA 绑定。
- [x] V8：P2 spill 在 GPU 前停止；P1 zero-spill/all-state exact，但四 contract 的 P50/P95/P99 全负，按门终止。
- [x] InputStages=4：fresh build zero-spill、small/H12 all-state exact；首轮 0.9942x–1.0067x 未过 1.02x 门，按门终止且不改 P2S3 current。
- [ ] full 8-rank TP8、真实模型任务质量、任意 layout 的通用 varlen dispatch 与跨架构当前组合，均在上表保留边界；T8191、skew/FP32-both 与 B7-none 已分别由生产正门或有效负门闭合，不再列作待跑。

**最终证据口径：**本文的“exact”始终说明候选与 pinned baseline/reference 在指定输入上
一致；它不自动等于模型质量。tensor roofline 只计 dense BF16 HMMA，不声称覆盖所有标量
工作。vshard dispatch 只对白名单 B300 shape/state cell 生效；full TP8、未测输入和未测架构均不从
单 shard/单架构数字外推。V8-P2 没有启动 GPU，故绝不宣称其端到端正确性或性能。
