# C1:FlashKDA——官方 kernel 停在 SM80 MMA

## 背景

Kimi K3 的线性注意力部分是 KDA(Kimi Delta Attention)。Moonshot 开源的
FlashKDA(本目录 `FlashKDA/` 快照)是其高性能 forward kernel,在 GB200 上
对 fla 的 Triton `chunk_kda` 有 1.7-3.3 倍加速(`BENCHMARK_GB200.md`)。
值得注意的是:这份 kernel 的矩阵乘用的是 SM80 世代的 `mma.sync`,而不是
wgmma(SM90)或 tcgen05(SM100)——在 GB200/B300 上运行时同样如此。
作者在 `docs/20260420-flashkda-v1-deep-dive.md` 里解释了设计,但"为什么
停在 SM80 指令"这个决策的量化论证是留白的。这道题就是把这份论证做出来
——或者推翻它。

## 任务(三层)

1. 复现:在 B300 上装起 FlashKDA,跑通官方 benchmark(`benchmarks/`,
   形状对照 `BENCHMARK_GB200.md`),用 ncu/SASS 确认计算主路径确实是
   SM80 MMA(`benchmarks/ncu.sh` 是官方的 ncu 模板)。
2. 分析:下面的讨论点逐个给出"结论 + 证据"。量化类的先纸面推算,
   再用 microbench 验证。
3. 挑战:选 SM100 路线的任一切面动手——只换指令不动算法、大 CHUNK +
   rescale、并行度重构,任选其一。正确性对 `fla_kda_ref/` 的实现对拍
   (`naive.py` 是纯 PyTorch 朴素参考,`chunk.py` 是 Triton 参照),
   性能对 FlashKDA 本体。做不出正收益也算完成:把"官方停在 SM80 是
   对的"论证扎实,就是讨论点 6 的另一半答案。（状态同步：已完成 SM100-family/B300
   的并行度切面；vshard2、vshard4-P2 与 V8-P2/P1 均有独立 correctness/性能或
   资源停止门证据，正负结论见 `REPORT.zh-CN.md` 的“P3 可行性裁决与 V8 最终实验”。）

   2026-08-30 状态同步：phase-1 fragment prefetch 与 B7-none 新协议均已得到可复核负门，
   不再列作待跑；fixed T8191 已将 test-only 资格与真实 production A1/A2/chain 分层闭合；
   skew/FP32-both 已通过 test-only A1/A2 并集成 v5 production source；job12598 因随后
   协议负控隔离失败仍不计 A1，但修复版 job12770/12771 已以不同 allocation、四个 fresh PID
   完成真实 production A1/A2/freeze。test-only 与 production 证据继续分层，完整边界见
   `REPORT.zh-CN.md` 的“当前最高优先级补轮”。

交付:代码 + 报告 + 答辩。

### 变量速记

| 变量 | 含义 | 本文常用取值或下标 |
| --- | --- | --- |
| `B,T,H,D` | batch、每条序列 token 数、当前 GPU head 数、head dimension | 官方形状 `1,8192,96,128`；TP8 单 shard `H=12` |
| `K,V` | key/value channel width | 均为 128 |
| `C` | chunk length | official 为 16；讨论对照 32/64 |
| `r(C)` | chunk 内最早项相对末项的保守最小衰减 | `exp[-5(C-1)]` |
| `I_C,L` | `C×C` 单位矩阵、严格下三角 chunk 内递推矩阵 | Neumann 式中的 `I_C+L` |
| `S_c` | 第 `c` 个 chunk 后的递推 state | `c` 是 chunk 时间下标 |
| `N_seq` | 同一 launch 的独立 sequence 数 | fixed 时等于 `B`；varlen 时由 offsets 决定 |
| `M=N_seq×H` | K2 的 sequence-head 工作项数 | 只能作为 dispatch 特征，不能单独决定 winner |
| `V_s,s` | 单 CTA 的 value-shard 宽度及 shard 下标 | v2 为 64，v4 为 32 |
| `F_TC,Q_DRAM,I_TC` | HMMA tensor FLOP、DRAM bytes、tensor arithmetic intensity | `I_TC=F_TC/Q_DRAM` |
| `p,δ_p` | 延迟分位数及 winner 相对 runner-up 裕量 | `p∈{50,95,99}` |

## 讨论点

1. CHUNK=16 的三个理由——bf16 数值范围、16×16 Neumann 级数求逆、
   SM80 MMA 形状匹配——各自量化:CHUNK=32/64 时哪个先破,代价多大?

   A：数值范围先破

   ① bf16数值范围：

   chunk 内最早项相对最后项的最小衰减可保守写为：

   \[ r(C)=\exp[-5(C-1)]. \]

   | `C`  | `r(C)` 近似值         | BF16 结果              |
   | ---- | --------------------- | ---------------------- |
   | 16   | `exp(-75)=2.68e-33`   | 仍在 BF16 normal 范围  |
   | 32   | `exp(-155)=4.83e-68`  | 远低于 BF16 可表示范围 |
   | 64   | `exp(-315)=1.57e-137` | 必然下溢               |

   B300 实测 FP32 值 cast 到 BF16 后：

   - `C=19` 仍是非零 subnormal；
   - `C=20` 首次归零；
   - `C=32/64` 均为零。

   所以从 16 扩到 32 时，最先出现的是累计衰减下溢。

   ② 16×16 Neumann 级数求逆：

   矩阵规模扩大后，工作量近似按三次方增长，存储按平方增长。

   严格下三角矩阵 `L` 满足：

   \[ (I_C+L)^{-1}=I_C-L+L^2-\cdots+(-L)^{C-1}. \]

   按稠密矩阵工作量估计：

   | `C`  | 相对 C16 的计算量 | 相对 workspace |
   | ---- | ----------------- | -------------- |
   | 16   | 1x                | 1x             |
   | 32   | 约 8x             | 4x             |
   | 64   | 约 64x            | 16x            |

   B300 固定并行度的 Neumann doubling 代理结果为：

   | `C`  | event 时间 | 相对时间 | 代理实际 FLOP 比 |
   | ---- | ---------- | -------- | ---------------- |
   | 16   | 0.29324 ms | 1.000x   | 1.000x           |
   | 32   | 0.48510 ms | 1.654x   | 10.667x          |
   | 64   | 1.07644 ms | 3.671x   | 106.667x         |

   时间没有按 FLOP 比增长，是因为 C16 小矩阵的 Tensor Core 效率很低，launch 和固定开销占比较高；矩阵变大后吞吐利用率提高。

   ③ SM80形状：（状态同步：上文已给出 C16/C32/C64 的 B300 BF16 下溢与 Neumann
   代理定量数据；HMMA/tcgen05 的 SASS/精确门及时间数据见
   `experiment_logs/c1_chunk_tcgen_microbench_b300.json`。）

   SM80 使用 `m16n8k16`，C16 正好对应一个 16 元素归约步。

   一个逻辑 `16×16×16` 结果可以用两个 `m16n8k16` 覆盖 N 方向。C32/C64 仍然可以分块执行，但分别需要更多 K-step、fragment 和中间 workspace。

   因此：

   - BF16 数值范围是首先出现的硬问题；
   - Neumann/存储代价随后快速增长；
   - MMA 形状不是 C32/C64 的正确性障碍，但 C16 的映射最紧凑。

2. tcgen05 的最小 tile 与 CHUNK=16 的形状匹配吗?不动 CHUNK 只换指令
   有没有收益——先纸上算,再 microbench 验证。

   ① 最小tile不匹配。当前可执行直接路径是 `m128n64k64`，远大于 `16×16×16`。

   单个 C16 子问题的有用工作比例为：

   \[ \frac{16\times16\times16}{128\times64\times64} =0.78125\%. \]

   即使沿 M/N 打包 32 个独立 C16 GEMM，有效比例也只有 25%，因为 K 方向仍然只有 `16/64` 有效。

   不能沿 K 打包独立问题，因为 K 是归约轴，这样会把不同问题的结果错误地累加在一起。

   ② microbench 的结果？

   相同逻辑工作和 CTA 数下，HMMA 在 C16/32/64 都更快。

   | `C`  | HMMA        | tcgen05     | tcgen05 相对慢 |
   | ---- | ----------- | ----------- | -------------- |
   | 16   | 0.056799 ms | 0.242131 ms | 约 4.26x       |
   | 32   | 0.088442 ms | 0.242168 ms | 约 2.74x       |
   | 64   | 0.152294 ms | 0.242265 ms | 约 1.59x       |

   三组结果均通过 bitwise exact gate；SASS 中同时确认 `HMMA=14`、`UTCHMMA=4`、`LDTM=8`。

3. 递推在 chunk 间有状态依赖,并行度还能从哪来(多 head 进一个 CTA /
   persistent kernel / 2-CTA)?列出候选方案,互相找反例。

   三类候选方案

   | 方案                   | 可利用的并行轴                            | 主要反例或风险                                           |
   | ---------------------- | ----------------------------------------- | -------------------------------------------------------- |
   | 多 head 合入一个 CTA   | head 之间独立                             | 单 CTA 的寄存器/shared/state 增大；grid 还可能进一步缩小 |
   | persistent K2          | CTA 内复用 descriptor/state，减少调度开销 | 不能消除 chunk 时间依赖；head 太少时仍填不满全卡         |
   | 2 CTA/head value-shard | value 列之间独立                          | 重复读取 K1 workspace，TMA 和同步数量增加                |
   | 4 CTA/head value-shard | 进一步增加 grid                           | 单 CTA 工作过小，重复加载和 launch 开销超过收益          |

   实验结果：2 CTA/head 是当前较有效的方案（此处 H64/H96 是历史对照，不替换
   后续的 H12 条件性裁决）。

   - vshard2 : H64：为 `1.176565x`； H96：仅 `1.000963x`
   - vshard4 在 H64 虽比官方快，但慢于 vshard2；H96 为负优化
   - warp8 在 H64/H96 都退化约 2%

   后续 B300 证据补齐了 4/8 分片：H12 的 vshard4-P2 在两次独立 clean allocation
   中为 `0.529472 ms`，相对同轮 vshard2-P2 的 `0.595440/0.595136 ms` 为
   `1.124592x/1.124018x`，并通过 output/final-state exact；V8-P2 因 spill 在 GPU
   前按资源门停止，V8-P1 虽 zero-spill 且 exact，但四个 state contract 的
   P50/P95/P99 都输给 vshard4-P2（P50 慢 `9.19%–13.35%`），故八分片方向已停止。
   证据分别为 `challenge_vshard4_prefetch2/results/c1_vshard4_p2_b300_sm103a_h12_r1_h12_bf16_cyclic.json`、
   `challenge_vshard8_prefetch2/README.zh-CN.md` 与 `challenge_vshard8/README.zh-CN.md`。

   这说明：并行度过少时，拆分 CTA 有价值；原 grid 已经足够大时，重复加载和同步会抵消收益。

   H12 已有上述单 B300/per-shard 的条件性证据；**它不是 full 8-rank TP8**。真实
   8-rank 并发仍受用户 QOS/GPU 配额阻塞，现有 single-rank 结果不能写成 TP8
   端到端性能，见 `challenge_tp8_dispatch/results/c1_tp8_quota_reprobe_20260830.txt`。

4. 这个负载在你们的卡上是 compute-bound 还是 memory-bound?用哪几个
   ncu metric 回答?(assignment 4.5 的瘦 GEMM 表是现成的参照系,
   `in_proj_qkvgfab` 就是 KDA 的输入投影。)

   ​	**至少要同时看四组指标。**

   1. Tensor Core：
      - `sm__inst_executed_pipe_tensor_subpipe_hmma.avg.pct_of_peak_sustained_elapsed`
   2. DRAM/L2：
      - `dram__bytes_read.sum`
      - `dram__bytes_write.sum`
      - 对应的 `pct_of_peak_sustained_elapsed`
      - `lts__t_sectors.sum`
      - `lts__t_sector_hit_rate.pct`
   3. 并行度：
      - `launch__grid_size`
      - `launch__waves_per_multiprocessor`
      - `sm__warps_active.avg.pct_of_peak_sustained_active`
   4. stall：
      - barrier
      - long scoreboard
      - short scoreboard
      - wait / not-selected

   ------

   **问：K1 的结论是什么？**

   **答：K1 并行度充分，但计算和 DRAM 都没有达到饱和，不能简单分类。**

   - grid：`(512,96,1)`
   - waves/SM：`41.51`
   - active warps：`96.63%`
   - HMMA：`5.63%`
   - DRAM read/write：`28.78%/29.83%`
   - L2 sectors：`30.97%`
   - barrier stall ratio：`8.31`
   - long scoreboard：`4.03`

   所以 K1 不是小 grid 问题，更可能需要继续拆解同步、供数和指令路径；当前证据不足以说它纯 compute-bound 或纯 memory-bound。

   ------

   **问：K2 的结论是什么？**

   **答：K2 首先是并行度受限。**

   - grid：`(1,96,1)`
   - waves/SM：`0.32`
   - active warps：`9.37%`
   - HMMA：`19.39%`
   - DRAM read/write：`15.39%/3.24%`

   计算和显存带宽都未饱和，而全卡只有 96 个长生命周期 CTA。因此最直接的分类是：**K2 为小 grid/并行度受限，而不是典型 compute-bound 或 memory-bound。**

   ------

   **问：与 assignment 4.5 的 `in_proj_qkvgfab` 如何对应？**

   **答：它们说明了同一个原则：Tensor Core 能否发挥作用首先取决于形状和并行度。**

   4.5 中：

   - 小 `M≤16` 时，launch、尾 tile 和并行 block 数量主导；
   - 大 M 时，`in_proj_qkvgfab` 可达到约 `1.65–1.77 PFLOPS`，进入 Tensor Core/调度/占用限制的平台。

   KDA 的 K2 虽然不是同一个 GEMM，但其问题与小 M skinny GEMM 类似：**不是理论峰值不够，而是没有足够独立工作填满 GPU。**

   状态同步：已在同一 H12 NCU profile 完成严格 BF16 tensor-contraction roofline：
   K1 位于 memory-roof 分支（`I_TC=7.3732`、效率 `34.37%`），K2 位于 compute-roof
   分支但效率仅 `3.84%`，结合低 grid/并行度仍解释为延迟/并行度受限，而非算力饱和。
   证据为 `challenge_roofline/results/c1_h12_bf16_tensor_roofline_r1.md`；该口径只计
   可由 NCU 逐指令核对的 BF16→FP32 HMMA，不冒充所有操作的 FLOP。



5. 状态存 bf16 的精度验证怎么设计?官方只说内部测试通过,拿出你们的
   验证方案和数据。

   1. **验证方案应该分哪几层？**

      **答：应区分 kernel 正确性、长期数值误差和模型质量。**

      第一层，kernel 回归：

      - 固定同一随机输入和 initial state；
      - 覆盖 none/BF16/FP32 state；
      - candidate 对 upstream、torch reference 和 FLA naive；
      - 同时比较 output 和 final state。

      第二层，长期数值误差：

      - 增加多个随机 seed；
      - 扫描长序列；
      - 每个 chunk 记录 BF16-state 对 FP32-state 的误差；
      - 报告 max-abs、相对 L2、P50/P95/P99 和随时间的漂移；
      - 增加 gate 接近 0 和接近 `-5` 的压力输入。

      第三层，模型质量：

      - 多层递推；
      - 真实激活分布；
      - 比较任务指标或 logits，而不只比较单 kernel 张量。

      ------

      **问：目前已经得到哪些数据？**

      **答：第一层与第二层已完成；第三层的真实模型质量仍是外部资产阻塞。**

      小形状中：

      - candidate 对 baseline output/state exact；
      - candidate 对官方 torch reference exact；
      - 对 FLA naive：
        - 最大 output 绝对误差：`4.882812e-4`
        - 最大 state 绝对误差：`2.838939e-3`

      长期数值层已补齐：多 seed、random/stress、H1/H12、最长 262,144 token 的合成
      recurrence 对 FP32 oracle 分段误差已归档在
      `challenge_long_context_quality/results/c1_long_context_b300_sm103a_full_r1.json`。
      但真实 Kimi logits、perplexity 或下游任务质量仍未完成：资产审计未找到可访问的
      模型权重、数据集或 C1 model-eval/TP launcher（
      `challenge_long_context_quality/results/c1_real_model_asset_probe_20260830.txt`）。
      因而不能把合成 recurrence 或 kernel exact 写成真实模型质量通过。

6. 假设你们是作者:v2 出不出 sm100a 专版?把可移植性与峰值两边的论据
   都写全,给出你们的结论。

   1. **问：支持发布专版的论据是什么？**

      **答：B300 上确实存在架构相关的条件性收益。**

      - K2 只有 `0.32 waves/SM`，并行度问题明确。
      - vshard2 在 H64 获得 `1.176565x`。
      - P2 软件预取将 H64 current 更新至 `0.737600 ms`。
      - H96 的最新 P2 也有约 `1.026651x` 的小幅收益。
      - Blackwell 更大的资源和异步机制仍可能支持更深的专用优化。

      ------

      **问：反对无条件专版的论据是什么？**

      **答：收益不稳定，且最新指令并没有自动转化为收益。**

      - H96 的 vshard2 基本持平。
      - tcgen05 与 C16 的 tile 匹配很差。
      - 公平 microbench 中 tcgen05 在 C16/32/64 都慢于 HMMA。
      - vshard4、warp8 和 P3 都提供了负例。
      - 5090 上 P2 持平或略负，说明不能跨架构外推。
      - 专版会增加编译目标、dispatch、reference 回归和维护成本。
      - H12、固定 batch/精确 varlen tuple、tail 与合成长期 BF16 recurrence 已有各自
        有界证据；但 full 8-rank TP8、任意 varlen layout、跨架构与真实模型质量仍未闭合。

      ------

      **问：最终决策是什么？**

      **答：出一个受限的 B300-only opt-in 路径，但不替换通用 SM80 默认路径。**

      建议结构为：

      - SM80 HMMA 路径继续作为 portable baseline；
      - B300 实际对应 SM103a，在满足已验证 shape/head 条件时启用 value-shard + P2；
      - dispatch 必须基于架构、每卡 head 数、序列形状和 state dtype；
      - 任一正确性门或性能门未通过时回退 upstream `fwd`；
      - H12 与精确 varlen 子集已经验证；在 full 8-rank TP8、任意 varlen layout 和真实
        workload/模型质量完成前，不设为无条件默认。

      因此我们的最终判断是：

      > 官方停在 SM80 MMA 作为通用默认是合理的；B300 专版的价值主要来自并行度与软件流水，而不是简单更换为 tcgen05。


## 材料

- `FlashKDA/`:官方仓库快照,pin commit `1ce47ea`(2026-07-29)。
  cutlass 子模块未含在快照里,构建时用
  `git clone --recurse-submodules https://github.com/MoonshotAI/FlashKDA`
  后 `git checkout 1ce47ea`(cutlass pin `5c149f5`)。
  重点文件:`docs/20260420-flashkda-v1-deep-dive.md`(设计文档)、
  `BENCHMARK_GB200.md`(官方数据表)、`csrc/smxx/`(kernel 本体)、
  `benchmarks/`(bench 与 ncu 脚本)、README(chunk_kda 调用约定与
  dispatch 调试方法)。
- `fla_kda_ref/`:fla-org/flash-linear-attention 的 `fla/ops/kda/` 快照,
  pin commit `a3edffc`。`naive.py` 纯 PyTorch 参考;`chunk.py` 及其依赖
  是 Triton 参照;`backends/flash_kda.py` 是 fla 调用 FlashKDA 的适配层
  (两边张量约定的对照表)。
- 形状:K3 的 KDA 配置是 96 头 × head_dim 128(93 层中 69 层 KDA、
  24 层全注意力);官方 benchmark 的 `T=8192, H=96, D=128` 即此,
  H=64 组只是附加对照形状。注意 TP 部署下每卡头数是 96/TP(TP8 为
  12)——讨论并行度时用每卡数;GEMM 侧形状见 assignment 4.5。
