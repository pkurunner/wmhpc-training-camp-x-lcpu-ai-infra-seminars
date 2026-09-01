# C2 答辩提纲（10 分钟）

## 变量速记（不计时）

| 符号 | 含义 |
| --- | --- |
| `B` | 同时 decode 的 request 数，取 1/4/8/16 |
| `G,H_q,H_kv,D` | GQA group、query/KV head 数与 head 维；取 `16,64,4,128` |
| `P,K_top,C` | page 长、选中 page 数、split-K chunk 数；取 `128,16` 与 mode-aware selected 值 |
| `t_8,h,j,s_buf` | v8 token tile、GQA head、token subgroup lane 与双缓冲槽；`s_buf=t_8 mod 2`，不等于 attention scale |
| `M^{(8)}_{t_8},B^{(8)}_{t_8}` | v8 当前 tile 的 weights/alpha/active metadata 与其写后 CTA publication barrier |
| `I_{x←y}` | 候选 `x` 相对冻结基线 `y` 的 8-seed paired-median 改善；本续轮要求严格 `I>3%` |
| `LCB_{x←y}` | 对相同 8 个固定 seed 做 10,000 次 deterministic paired bootstrap 的 95% 下界，只描述该 fixture 的重采样稳定性 |
| `q_i` | v11 预先装入并跨 4 个 selected page 复用的第 `i` 个 Q WMMA `matrix_a` fragment，`i=0,…,7` |
| `S_Q` | Q shared-memory tile 的 leading stride（以 BF16 元素计）；v12/v16 为 `136`，v15 独立为 `136→144` |
| `S_{KV}` | v14 K/V staging tile 的 leading stride（以 BF16 元素计）；由 `16` 改为 `24`，live 列仍为 16 |
| `W,T,e` | v14 shared 算术中的 CTA warp 数、每 warp WMMA tile 行数、BF16 元素字节数；`8,16,2 B` |
| `R_{12},R_{14},R_{15},R_{16}` | 下标为版本的 v12/v14/v15/v16 AOT static shared memory；`31136/33184/31392/31136 B` |
| `K_c` | 下标 `c` 表示 v16 的页内 raw-FP8 K lookahead chunk；lane-private，不改变跨页语义、PV、merge 或 ABI |
| `I_{v16←v12}` / `LCB_{v16←v12}` | v16 相对 v12 的 8-seed paired-median 改善 / deterministic paired bootstrap 95% 下界；本轮接受门为 `I>3%` 且 `LCB>0` |
| `\rho_{x/12}^{m}` | 同 UUID、同 input/harness 下候选 `x∈{14,15}` 相对 v12 的 NCU counter `m` 比率；只解释机制，不是性能倍率 |
| `C_{ld},W_{ld},S_{long},S_{wait},N_{TC}` | shared-load bank conflict / wavefront / long-scoreboard / wait / tensor instruction；job13666 每臂采 15 项 counter |

## Slide 1（0:00–0:45）：问题与不夸大的结论

- MiniMax M3 的小 batch decode 使用 16 个已选 KV page、128 token/page、64/4 GQA、D=128。
- 先 profile，后设计：decode 是主项，B300 merge 仍占 26.8%–38.5%。
- current mode-aware v2 的 12/12 gate PASS；BF16 用 C=1，FP8 按各自最优 split，三种 mode 都有公平计时正收益。

## Slide 2（0:45–1:25）：冻结验收协议

- 固定 B={1,4,8,16}、随机 block_table、独立 FP32 sparse reference。
- 12 个 B×storage 组合，`rtol=atol=0.03`；FP8 参照同一量化 cache 的反量化值。
- 正确性失败即不计性能，避免“快但页表/scale 错”的假优化。

## Slide 3（1:25–2:15）：双架构清卡基线

- B300 与 5090 都有 PRE/POST 0 MiB、compute-apps 为空、四个 trace。
- B300 B1/B4/B8/B16 decode=3.958/5.234/7.329/11.501 us。
- 5090 同项=3.504/8.397/13.171/21.780 us；重点是 profile 结构，非跨卡营销。

## Slide 4（2:15–3:15）：算术强度

- 每 `(request,kv_head,page)`：QK+PV=`4GPD=1,048,576 FLOP`。
- BF16 K/V 最小读 65,536 B，AI 至多 16 FLOP/B；FP8 理想上界 32，scale 会降低。
- `tl.dot` 有 Tensor Core 矩阵路径，但随机页、softmax、低 B 和索引使内存/调度影响很大。

## Slide 5（3:15–4:10）：为什么不是直接 TMA

- `topk_idx -> logical block -> block_table -> physical page` 是两级 data-dependent gather。
- TMA tensor map 描述固定仿射 layout，不能一次表达这种运行时索引链。
- B300 job 4340 已以真实 `CUtensorMap`/`UTMALDG.2D` 跑完：相同 `B=4,K=16,E=1024`、两条等字节搬运腿，三路均 bit-exact 0 mismatch、PRE/POST 清卡且 `FINAL_RC=0`。
- CUDA event：software→software=`0.007272` ms（144.200 GB/s）、software-gather→真实 TMA=`0.008216` ms（127.621 GB/s）、连续→真实 TMA=`0.007186` ms（145.921 GB/s）。
- 结论是“先 scalar/vector load index 并 software gather，连续 buffer 才可用 TMA”，不是“当前 TMA gather 更快”；实际两级 gather 的 TMA 第二腿在此尺寸较慢。

## Slide 6（4:10–5:15）：split-K 与 merge 的选择

- C>1 保持 CTA 并行度，却需 partial `O_c,lse_c` 和第二个 merge kernel。
- cluster/mbarrier 可以作为后续候选，但并不消除随机 KV 读，且有 phase/资源风险。
- C=1 时 softmax 已全量：output alias caller `O`，merge 可以代数上正确地省掉。

## Slide 7（5:15–6:20）：prepared 实现

- 调用方 output 在计时外；persistent 对象复用真实 production workspace。
- 对 C、warps 和三种 storage 分别验收；仅 BF16 的 C=1 alias output 并跳过 merge，FP8 保留可用 split/merge。
- 不将 workspace 复用包装为 Tensor Core FLOP 加速。

## Slide 8（6:20–7:30）：公平 B300 结果

- schema `c2-final-gate-v2-mode-aware`、current CLI hash 的 12/12 PASS；selected C 分别是 BF16 `[1,1,1,1]`、scalar `[16,4,8,4]`、token `[4,16,16,4]`。
- steady CUDA：BF16=1.502/1.631/1.609/1.453x；scalar=1.305/1.296/1.306/1.295x；token=1.330/1.341/1.355/1.313x。
- host 单步也为 BF16 1.04–1.09x、scalar 1.21–1.33x、token 1.22–1.27x；CUDA graph 是另一种运行模式，单列报告。

## Slide 9（7:30–8:40）：同 BF16 数据的三路径与跨 pin 边界

- job 4339 对每个 B 让 source wrapper、prepared BF16 `C=1`、official MSA 都使用同一 BF16 Q/K/V、随机页表、排序 top-k 和独立 FP32 oracle；四个 B 的三路径全 PASS。
- per-call CUDA median（B=1/4/8/16）：source=48.272/46.896/48.192/46.976 us；prepared=31.280/29.152/29.152/31.200 us；official=27.216/26.160/27.120/26.960 us。
- official 相对 source 为 1.773662/1.792661/1.776991/1.742433x，相对 prepared 为 1.149324/1.114373/1.074926/1.157270x；p10/p90 与输入 checksum 均在 JSON 中。
- MSA `80434d7` / CUTLASS `eb61c91` 和 vendored `d4da0c5` 不是同 pin；K/V physical ABI bridge、official plan/workspace 在计时外，prepared 的 persistent workspace 与 selected `C=1` 也改变工作量。因此它是等数据 core-level 对照，绝不称 full-vLLM 同 pin crossover。

## Slide 10（8:40–10:00）：答辩结论与边界

- current v2 gate、真实 TMA 两级 gather、三路径 BF16 core-level 对照都已可复核。
- 结论是 mode-aware prepared policy 三种 storage 都较快；不是“所有 mode 都使用 C=1”。
- 保留两条边界：需要统一 NCU roofline；没有同 pin full-vLLM runtime/plan/extension 的端到端 CUTLASS-vs-Triton scoreboard，TMA 微基准也不代表完整 decode。

## 预备问答

**问：为什么 merge 不直接塞进 decode kernel？** C>1 的 producer CTA 必须完成并发布
partial LSE/output；强行融合要做跨 CTA 的同步和共享存储。C=1 是唯一无需归并的
严格情形，故先验证这个低风险候选。

**问：为什么 FP8 现在也有收益？** 旧 C=1 试验在 FP8 上是负例；current v2 没把
C=1 外推，而是保留 scalar/token 各自验收过的 split 数并复用 workspace，所以得到
1.21–1.36x CUDA 收益。它不表示所有 FP8 或所有 shape 都快。

**问：公平 B=16 的官方数字在哪里？** 同 BF16 数据三路径表的 official B16 median
是 26.960 us、独立 FP32 gate PASS；source/prepared 分别为 46.976/31.200 us。另有
旧 FP8 direct-core adapter 的 B16=26.080 us，但它不是这张公平表，不能混用。两者都
不是 d4da0c5 full-vLLM 同 pipeline，后者的 B>=16 门槛还包含 plan metadata/integration。

**问：TMA 是不是已经直接做了 page gather？** 不是。job 4340 的真实 TMA 指令和三路
bit-exact 结果验证了连续第二腿；`topk -> block_table` 仍需软件读 index/gather，且本
尺寸的 `gather_then_tma` 为 0.008216 ms，比 software staged 的 0.007272 ms 慢。

**问：现在的结果是最终 revision 吗？** mode-aware v2 的 12/12 gate（job4306）、真实
TMA（job4340）和等 BF16 三路径对照（job4339）均为当前清卡证据。它们仍不等同于未运行的
同 pin full-vLLM integration 服务端到端数字。

## 优化续轮答辩补充：5090 SM120 的 C=1 no-LSE

| 变量 | 含义 | 固定值 |
| --- | --- | --- |
| `B` | decode batch | `1,4,8,16` |
| `C` | split 数 | `1`（仅 BF16） |
| `G` | KV head 的 Q-head CTA 分片数 | `1` |
| `s` | Triton stage | `3` |

- 候选在 `C=1` 用 online `(m,l,acc)` 直接输出 `acc/l`，删除无消费者的 LSE 更新、
  store 和 workspace；`G=1` 避免 `G=2` 将相同 K/V page 交给两个 CTA。
- 配置在[独立 B=4 freeze gate](challenge_v2/results/5090_sm120_nolse_s1_freeze_gate_job7001)
  **之前**冻结为 `warps=4, stages=3, PDL=off, maxnreg=none`；
  对照是相同输入/seed/输出生命周期下的 current prepared BF16 C=1。
- 每个 B 先过独立 FP32 oracle，再以 101 个 ABBA 对交错发射，控制和候选各 202 个
  single-call CUDA-event；PRE/POST 是同一 RTX 5090 SM120、0 MiB、compute-apps 空，
  `FINAL_RC=0`。原始 JSON/audit 在
  [`challenge_v2/results/5090_sm120_nolse_s1_final_20260820`](challenge_v2/results/5090_sm120_nolse_s1_final_20260820)。
- 合并中位数 speedup（B=1/4/8/16）为 **1.218584/1.285411/1.242640/1.242410x**；每一项的
  AB 与 BA 子中位数都为正收益，最大 FP32 oracle 绝对误差不超过 `6.10e-5`。
- 这只是一组 5090/SM120、BF16 C=1 的 source-bound 结果。它本身不外推 B300、FP8 或
  service latency；下面给出随后在清卡窗口以同一 ABBA 合同完成的 B300 独立复验。

## 优化续轮答辩补充：B300 复验改变部署决策

- clean B300 job 4446 已用相同源码/ABBA 合同复验，PRE/POST 为 0 MiB、compute-apps 空，
  每个 B 也先过独立 oracle，再采集 control/candidate 各 202 个 event 样本。
- B=1/4/8/16 的 no-LSE speedup 是 **1.074241/0.932844/0.932914/0.936087x**。
  因而 B1 只有未达门槛的正收益，B4/B8/B16 是负收益，B300 保持 current prepared。
- runner `RC=3` 仅表示四个 context 未全部达到 strict 10% speedup gate；它不是运行、
  清卡或 FP32 oracle 的错误。原始 JSON 在
  [`challenge_v2/results/b300_sm103_nolse_s1_job4446`](challenge_v2/results/b300_sm103_nolse_s1_job4446)。
- 结论：此冻结 no-LSE 配置是 5090/SM120 的架构特化结果；当前证据不能把它包装成
  B300 或 FP8 的通用优化。

**问：为什么不报此前更高的 `G=2` 网格数字？** 因为固定配置后 AB/BA 只剩约
1.04--1.05x；这说明 min-of-grid 受启动顺序/频率影响，不能作为最终结果。最终表只用
在冻结配置后的交错复验得到的数值。

## production-native 补充：部署前的证据闭环（B300）

- **v16 是最新的 lifecycle-closed 内部 production candidate。** 历史三个严格 `>3%` 方向
  （v7、v9、v11）及 v12 的续轮证据保持原裁决；新增的方向 4 是 v16 相对 v12 的独立同卡
  direct-plugin 比较。v16 已完成 overlay wheel 与 job13845 单进程 lifecycle，但仍不是 upstream、
  release、full-model、server、multi-process 或 service E2E 认证，也不表示 Triton parity。
- **v13/v14/v15 均不晋级。** v13 AOT 成功后，directed 首调用 illegal instruction，按止损未跑
  stress；v14 只把 K/V stage stride `S_{KV}:16→24`，有效 ABBA 为 `+2.40452555%`、LCB
  `+2.29466423%`，未过 `>3%`；v15 独立把 Q-stage stride `136→144`，AOT/directed 通过但有效
  ABBA 为 `-0.09713217%`、LCB `-0.11928154%`。v14/v15 均以 RC3 纯性能拒绝并保留 v12。
- job **12295** 完成 v5 生命周期门禁：8 个并发首加载者只发生一次真实
  `torch.ops.load_library`；随后通过 dispatcher/kernel、1000 次实调用、CUDA Graph
  replay、原地 query 更新、页边界、拒绝路径和有界显存计数检查。它验证的是本插件的
  load-once 与已声明 C2 契约，不是多进程服务端到端认证。
- job **12278** 把 v6 的 register-numerator 改动 AOT 编译为仅含 `sm_100` cubin 的
  固定 DSO；资源记录为 64 registers、0 stack、30,560 B shared、0 local。job
  **12322** 再以 64 个 head-offset、128 个 dim code、4 个 producer rank、长序列及
  非法输入验证该 DSO，并用 profiler 精确计到 1 次 dispatcher 与 1 个 native kernel。
- job **12314** 在同一固定 C2 fixture 上做 8 seed、隔离进程的 v5/v6 ABBA：v6 中位数
  `0.107585 ms`，v5 `0.116682 ms`，点估计改善 **+7.7958%**；10,000 次配对 bootstrap 的
  95% LCB 为 **+7.7492%**。预设门是点估计至少 5% 且 LCB 至少 0；两项均通过。该 LCB
  只描述 8 个固定 seed 的重采样稳定性，不外推未采样 workload/GPU 或声明总体置信结论。
- job **12331** 将上述 DSO 与脚本内容固定为 overlay wheel 及 provenance；job
  **12385** 从新的安装目标起 8 seed 运行真实 MiniMax full backend。native 为
  `0.108112 ms`，Triton 为 `0.029296 ms`；8 个逐-seed native/Triton ratio 的中位数为
  **3.68566**（不是前两个 latency 中位数之商）。该实验未获得
  parity，因此它是“真实 backend 下的性能与加载证据”，不是可宣称的 Triton 等价替代。
- job **12396** 的 NCU 只用于机制解释：v6 shared 从 38,752 B 降至 30,560 B，load/store
  wavefront 比分别为 `0.7912` / `0.8418`。它解释 register-numerator 映射为何可能减轻
  资源/访存开销，不能单独证明端到端吞吐或服务延迟收益。
- v7 只把原始 FP8 V load 提前到 softmax 前。jobs **12484/12501** 的 AOT 与定向合同全过；
  job **12513** 同卡 8-seed `v6→v7→v7→v6` 的点改善为 **+4.8278%**、bootstrap 95% LCB
  为 **+4.7563%**。8 个已观察 seed 均为正，但点估计未过预先固定的 5% 门；该重采样不作
  未采样 workload/GPU 的总体推断，所以这是有效候选的纯
  性能拒绝；不构建 v7 wheel/fresh/NCU，仍保留 v6。
- v8 不叠加 v7，只把 softmax metadata 做成双槽并删去每个 tile 的 tail CTA barrier。
  `B^{(8)}_{t_8}` 发布槽 `s_buf`；页内下一轮 barrier 等齐所有线程，故 `t_8+2` 覆盖同槽前，
  `t_8` 的所有读者已结束；tile 7 的跨页边界则由下一页 QK/score CTA barrier 等齐。job
  **12534** 的资源为 56 registers、0 stack、31,200 B shared、
  0 local；job **12548** 的 rank/tail/all-invalid/head-dim/FP64/pointer/profiler 门全过。
  但 job **12557** 同卡 8-seed `v6→v8→v8→v6` 得到 v6/v8
  `0.107806/0.115449 ms`，改善 **-7.0892%**、bootstrap LCB **-7.3237%**。所有身份、正确性、
  资源、可比性门仍全过，因此按性能门拒绝 v8，也不构建 wheel/fresh/NCU。

### 严格 `>3%` 续轮的三个显著方向

冻结判定先要求全部硬门通过，再要求 paired-median 点改善严格大于 `3%`、固定 8-seed 的
deterministic paired-bootstrap 95% LCB 大于 `0`；LCB 本身不另设 `>3%` 门。

- **方向 1：raw-FP8 V 预取。** 旧 job12513 必须继续保留为冻结 5% 门下的合法拒绝；新问题
  另以 3% 门重放同一冻结 v6/v7 DSO。job **12599** 的 `v6→v7→v7→v6` 全部身份、正确性、
  资源和单 UUID 门通过，`I_{v7←v6}=+4.933390%`，`LCB_{v7←v6}=+4.717097%`，故仅对本轮
  `>3%` 问题接受 v7。
- **方向 2：最终四路归并并行化。** v9 先让 16 个线程各算一个 head 的四个权重与分母，
  CTA barrier 后由 256 个线程分片写 2,048 个输出元素。jobs **12701/12767** 的 AOT 与
  directed 门通过；job12772 在计时前因 source path 不等于 provenance 固定路径而
  fail-closed。精确路径重跑 job **12776** 得到
  `I_{v9←v7}=+28.101799%`、`LCB_{v9←v7}=+27.922466%`，全门通过。
- **方向 3：Q WMMA fragment 跨 selected-page 复用。** v11 把 8 个不随 page 改变的
  `q_i` 在四页循环前各装一次，保留 K/V、softmax、barrier 与 v9 merge；AOT job **12825**
  为 `REG=127, STACK=0, SHARED=30880, LOCAL=0`。初次 directed job12829 暴露“artifact
  source path 与 build-worktree provenance path 被错误要求相等”的脚本矛盾，在 harness 前
  `RC=2`、无性能数据；改为分别固定路径并以唯一同 SHA 源记录绑定后，job **12904** 的
  directed 门全过。正式 job **12905** 在 UUID
  `GPU-778768b4-6c9e-e483-890e-0812760948ae` 上完成 `v9→v11→v11→v9`，得到
  `I_{v11←v9}=+12.814628%`、`LCB_{v11←v9}=+12.782150%`，`issues=[]`，严格通过 3% 门。
- 三段百分比的基线不同，**不能相乘后宣称 Triton parity、full-model 或 service 收益**。
  job12905 是 v11 唯一的性能晋升裁决；其余 v11 实验只验证可部署性或解释机制，不能重算
  该 ABBA 结论。
- **overlay wheel。** job **12957** `FINAL_RC=0`，生成 SHA-256 为
  `278c01a9c99490993f807ce64b7e52e96d9513f0d22bd542cf1d60c3221e7f34` 的 wheel。稳定 DSO
  保持 byte-identical；baseline/derived `RECORD` 分别验证 4,746/4,748 个 payload，4,742 个
  baseline payload 不变，新增成员仅为 native plugin 与 adapter。这是同 distribution-version
  overlay，不是官方 release wheel。
- **fresh real backend。** job **12960** `FINAL_RC=0`，每个 seed 从独立 fresh 进程进入真实
  `MiniMaxM3SparseMSAImpl.forward`，8/8 硬门通过。native 与 Triton 的 seed-median 再取中位数
  分别为 `0.1110240035 ms` 与 `0.0476640016 ms`，逐 seed ratio 的中位数为 `2.3306479`；
  Triton parity 为 false，且 fresh 实验没有 v11/v9 wheel comparator，故不构成新的性能晋升。
- **NCU 机制解释。** job **12965** `FINAL_RC=0`，在同 fixture、同一第二块 B300、分别的
  Python 进程中各采一个 native kernel：v11/v9 的 shared-load wavefront 为 `0.6530`
  （-34.70%）、load bank conflict 为 `0.4759`（-52.41%）、long-scoreboard 为 `0.6830`
  （-31.70%），tensor instruction 不变，寄存器为 `64→127`。NCU duration/cycle 只作机制
  观察，不是性能裁决。
- **lifecycle。** job **12977** 全门通过：8 线程首加载恰好一次真实 load、真实 forward、
  1,000 次 forward、每个静态 query 100 次 CUDA Graph replay 与 query 原地变更、选择/拒绝、
  `2048/2049/4095/4096` 序列边界、有界显存和清卡均通过。它覆盖的是 fresh adapter 的
  单进程生命周期，不是多进程服务认证。
- job12905 的性能 UUID 是 `GPU-778768b4-6c9e-e483-890e-0812760948ae`；jobs12960/12965/12977
  的 UUID 是 `GPU-3924bd78-8b2f-8e85-3f20-8b0687d0bb0a`。后者增加第二块 B300 上的
  integration/mechanism/lifecycle 证据，但没有在该 UUID 重放 v9/v11 ABBA，因此**不是严格的
  跨 GPU 性能复现**。

### 续轮后续：v12 Q shared 行间距

- job **12983** 将 Q shared stride 从 `128` 改为 `136` BF16 元素并完成 AOT；job **12984**
  的定向合同通过。该变动不改 Q 以外的 layout、softmax、barrier、merge 或 ABI。
- job **12985** 是 v12 唯一的性能裁决：在同卡 8-seed `v11→v12→v12→v11` ABBA 中，
  `I_{v12←v11}=+3.649118%`、`LCB_{v12←v11}=+3.405307%`，因此按本轮严格 `>3%` 门接受。
- job **12986** 只解释机制：Q-load bank-conflict ratio 为 `0.634685`、wavefront ratio 为
  `0.845731`、long-scoreboard ratio 为 `0.973661`；store 略升。NCU timing 不是性能证据，
  不会重算或替代 job12985 的 ABBA 决策。
- job **12987** 的同 distribution-version overlay wheel 通过 `RECORD` 与 identity 验证；
  job **12991** 的 8-seed fresh real backend 通过，native/Triton 的 seed-median 再取中位数为
  `0.064600/0.029248 ms`，ratio 为 `2.209910`。它仍未获 Triton parity，且没有 v11 fresh-wheel
  comparator，所以不是 v12/v11 的新性能晋升。
- v12 lifecycle 已由 job **13334** 闭环。保留 jobs12992/12993/12994/12996 的 fail-closed
  审计后，修复版本只 allowlist 一个 `state/uv-cache/wheels-v6/url` symlink；该项仍在受控 state area 内，
  resolved target 目录 manifest 已固定。8-thread 首加载恰一次真实 load、real forward/dispatcher/native kernel、1000 次
  steady、Graph `100+100` replay+mutation、support、dynamic rejection、sequence boundary、dispatcher
  persistence 八门全过，`FINAL_RC=0`；v12 当时取代 v11 成为 lifecycle-closed candidate，随后由
  v16 接替为当前最新版本。
- job12985 的性能 UUID 是 `GPU-dadf9f3b-df58-d3fa-07b0-5fe223423db1`，job12986 NCU 的 UUID
  是 `GPU-3924bd78-8b2f-8e85-3f20-8b0687d0bb0a`；NCU、fresh 及其他辅助实验只增加
  机制/集成证据，**不构成严格跨 GPU 性能复现**。jobs **12988**（quota）与 **12990**
  （cleanup）仅列入失败审计。
- **v13：** job **13428** AOT 资源为 `REG=121, STACK=0, SHARED=30880, LOCAL=0` 且只有
  `sm_100` cubin；job **13439** 的第一个 head-dim encoding 同步调用遇到 illegal instruction，
  `FINAL_RC=2`，因此不跑 stress、没有性能数据。
- **v14：** 变量已在页首表中定义。仅 K/V `S_{KV}:16→24`，静态算术
  `ΔR=W×T×(24−16)×e=2048 B`，故 `R_{14}=33184 B`；job **13487** AOT 实测
  `REG=118, STACK=0, SHARED=33184, LOCAL=0`，job **13513** directed 全过。job **13518** 在
  预测量 harness preflight 因 argument-unpack 失败而无测量；重跑 job **13539** 的
  `I_{v14←v12}=+2.40452555%`、`LCB=+2.29466423%`，但点估计 `<3%`，以 RC3 拒绝，不能累加到
  历史收益或描述为 v12 晋级。
- **v15：** 不叠加 v14，只把已接受 v12 的 Q-stage stride `S_Q:136→144`。job **13564** AOT
  实测 `REG=127, STACK=0, SHARED=31392, LOCAL=0`；job **13575** directed 全门通过。job
  **13576** 的同卡四隔离进程、8-seed `v12→v15→v15→v12` ABBA 全部硬门通过，
  `I_{v15←v12}=-0.09713217%`、`LCB_{v15←v12}=-0.11928154%`，以 RC3 纯性能拒绝；不构建
  v15 wheel/fresh/lifecycle，也不改写 v12。v14 与 v15 都是相对 v12 的独立 pair，不能相加或连乘。
- **NCU 机制已闭环，不重算性能。** job **13662** 在首 snapshot 前因 Bash `set -u` 下同一 local
  声明引用未赋值 `tag` 而 `FAILED/1:0`、约 1 秒；无 GPU/NCU/final-status，仅保留失败审计。修复
  profile SHA `8314a341…` 后，job **13666** 即时观测为 `COMPLETED/0:0`、43 秒，六个 finalizer
  rc 全为 0，在 `GPU-0c223cf1-4325-822f-1e38-43ae57897edd` 上以同 input/harness、15 counters、
  三个独立 Python 进程与每臂一个 logical kernel action 采集；`timing_valid_for_benchmark=false`。

  | `m` | `\rho_{14/12}^{m}` | `\rho_{15/12}^{m}` |
  | --- | ---: | ---: |
  | `C_{ld}` | `0.33780898`（`-66.2191%`） | `1.08235542`（`+8.2355%`） |
  | `W_{ld}` | `0.79008057`（`-20.9919%`） | `1.02610728`（`+2.6107%`） |
  | `S_{long}` | `0.99237394`（`-0.7626%`） | `1.00092039`（`+0.0920%`） |
  | `S_{wait}` | `0.96137716`（`-3.8623%`） | `0.99966180`（`-0.0338%`） |
  | `N_{TC}` | `1.0` | `1.0` |

  v14 的 padding 确实消除了多数 load bank conflict，但与 long-scoreboard 小幅变化及 ABBA 仅
  `+2.40452555%` 合看，只能说明本次降幅不足以跨过性能门，不能由一次 NCU 采集认定唯一或主要
  剩余瓶颈；v15 的 counter 反向/无益，与 `-0.09713217%` 一致。NCU replay、cycle 与 counter
  绝不进入 performance scoreboard。

### 方向 4：页内 K-chunk raw-FP8 lane-private lookahead（v16）

先固定本方向变量和判定边界；以下性能数只属于 v16/v12 的冻结 direct-plugin fixture，绝不与前三个
方向或 v12 的历史百分比累加、连乘。

| 变量 | 答辩时的含义 | 冻结值 / 边界 |
| --- | --- | --- |
| `K_c` | 页内 raw-FP8 K lookahead chunk；下标 `c` 区分 chunk，且 load 为 lane-private | v16 唯一调度改动；不改跨页、PV、softmax、merge 或 ABI |
| `R_{16}` | v16 AOT 的 static shared memory | `31136 B`；实测 `REG=128, STACK=0, LOCAL=0` |
| `I_{v16←v12}` | 8-seed paired-median 中 v16 相对 v12 的改善 | 必须严格 `>3%` |
| `LCB_{v16←v12}` | 同一 8 个固定 seed 的 deterministic paired-bootstrap 95% 下界 | 必须严格 `>0`；不外推未采样 workload 或 GPU |

- **单变量、AOT 与定向正确性。** v16 从冻结 v12 只拆分页内 raw-FP8 K lookahead 为 lane-private
  `K_c`。job **13773** AOT 通过，资源为 `REG=128, STACK=0, SHARED=31136, LOCAL=0`，唯一
  `sm_100` cubin、无 PTX；job **13786** 通过 v12 全合同以及 transition、last、stale-shifted 新 case，
  无 monkeypatch，PRE/POST/final-post 为同一 UUID 且 compute-apps 为空。
- **正式性能裁决。** job **13789** 在
  `GPU-dadf9f3b-df58-d3fa-07b0-5fe223423db1` 以四个隔离 Python 进程
  `v12→v16→v16→v12`、8 seed、每臂 30 warmup/200 单调用 CUDA-event 完成；所有 paired seed 改善为正，
  correctness/identity/resource/comparability 全门通过。得到
  `I_{v16←v12}=+6.2838716258%`、`LCB_{v16←v12}=+6.1226889551%`，故 v16 严格接受。
- **wheel 与 lifecycle。** job **13832** 生成 SHA-256
  `3947fab41739c98a30a8fd5486b867347b932f3419def3bfbd846db458ba90a9` 的 overlay wheel，
  `FINAL_RC=0, CLEANUP_RC=0`；job **13845** 随后完成 8 loader、1000 次真实 forward、两组
  `100` 次 CUDA Graph replay、8 MiB 有界内存、选择/拒绝与边界合同。8 个 lifecycle gate、同 UUID
  空 compute-apps 的 PRE/POST/final-post 和 scratch cleanup 均通过，最终状态字段均为 0。因此当前最新
  lifecycle-closed candidate 是 v16（取代此前的 v12）；它仍不等于正式 release 或多进程/长期服务认证。
- **归档与失败边界。** 本地 [v16 lean tar](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-k-chunk-lookahead-evidence-20260831/c2-native-plugin-v16-k-chunk-lookahead-lean-evidence-20260831.tar.gz)、
  [122-record manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-k-chunk-lookahead-evidence-20260831/c2-native-plugin-v16-k-chunk-lookahead-lean-evidence-20260831.manifest.sha256)
  与 [sidecar](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-k-chunk-lookahead-evidence-20260831/c2-native-plugin-v16-k-chunk-lookahead-lean-evidence-20260831.tar.gz.sha256)
  已本地 `122/122` 复核；tar / manifest / sidecar SHA-256 分别为
  `a9ce039b9ca9cbf151c49a82f5cdbf5800ff84fa00182c03a1fe59ab80afccd8` /
  `be213b6427b173e77930c7881b041e4d2de86750b30949edcdfb5fd356984a59` /
  `1b3040b34b59128e786119567b932fc6257928fa7b272928916be223b8b46895`。lean 包有意排除 68,219,593 B
  stress fixture；其 SHA-256 为 `2c571f37c94c744492bed673741930be8a4738b1d35d4d547aa71caab6f1d4a7`，完整远端 replay tar 仍保留它并固定为
  `784a0d8de98b6ffba3532f55c11e65a3c7c35bb2528b72994f5c4f1496bd104f`，不把后者误称为本地已下载。
  job13767 是编译前 Git/submodule 网络失败，不是候选失败；job13783 原 Slurm log 为 0 字节，日期
  harness 路径问题只由确定性 operator preflight replay 定位，不能声称原日志自带该原因。
- **matching NCU 已闭环。** job13868 在同一 `GPU-3924…bb0a` 上以隔离进程、同 input/harness 和每臂
  一个 matching native action 采 15 个 counter：v16/v12 elapsed-cycles ratio `0.92172466`、tensor-active
  ratio `1.08181818`，tensor instructions 与 DRAM read 均为 `1.0`。这支持“等工作量下 instrumented cycles
  更少”的有限解释；long-scoreboard ratio 反而为 `1.11319760`，故不能声称 long-scoreboard 下降或 K-load
  latency hiding 是唯一原因。NCU timing 不进入 performance scoreboard，也不是第二 UUID 的性能复现。
- fresh 统计变量先统一如下：

  | 变量 | 含义 |
  | --- | --- |
  | `L_{N,s}` / `L_{T,s}` | seed `s` 的 native / Triton latency median |
  | `R_s=L_{N,s}/L_{T,s}` | seed `s` 的 native/Triton latency ratio |
  | `median_s(R_s)` | 8 个逐 seed ratio 的中位数，不等同于跨作业比较 |

- **fresh wheel / 真实后端也已闭环。** job13900 的 8-seed
  `median_s(R_s)=2.081999763`，`median_s(L_{N,s})=0.1009760015 ms`、
  `median_s(L_{T,s})=0.04848000035 ms`。8 个独立 fresh Python、真实 MiniMax forward、独立 FP32
  oracle、dispatcher/kernel trace、wheel/DSO 身份与 clean-GPU/finalizer 门全过，但所有 `R_s>1`，所以
  integration 成功而 Triton parity 未达到。该 job 没有 v12 fresh-wheel comparator，不能用它重算
  job13789 的 v16/v12 晋升。本地 [fresh archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-fresh-backend-evidence-20260831/c2-native-v16-k-chunk-lookahead-fresh-backend-job13900-lean-evidence-20260831.tar.gz)、
  [42-record manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-fresh-backend-evidence-20260831/c2-native-v16-k-chunk-lookahead-fresh-backend-job13900-lean-evidence-20260831.manifest.sha256)
  与 [sidecar](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-fresh-backend-evidence-20260831/c2-native-v16-k-chunk-lookahead-fresh-backend-job13900-lean-evidence-20260831.tar.gz.sha256)
  SHA 分别为 `4e93c690…9743`、`a03a0295…3914`、`901a8adc…78cd`；42/42 safe regular member
  已在本地逐字节复核。
- **下一优先级。** 当前最高可行动项已经变为：沿 matching native/Triton profile 与 SASS 依赖链，尝试
  单一可归因 kernel 变量，并继续执行独立 oracle/fresh process/`>3%` 止损门。严格跨 GPU v16/v12 ABBA
  仍重要但受第二 UUID 调度不可控限制；job13942 的一次预注册 replay 又被分配原 UUID，并在任何性能
  action 前按计划 RC3 停止，故不择机重跑。full-model 继续受 checkpoint/config 资产缺失阻塞。不能把现有
  one-row native NCU parser 强套到 Triton：真实 forward 包含 decode + merge 两个 Triton kernel，且 fresh
  harness 内有 Kineto profiler；后续必须用专门的 NCU-only runner 分别采三项 action。

- 两个止损分支不计入三项成功：v10 raw-K lookahead 的 jobs12702/12783 通过，但 job12784
  因一臂 profiler 漏采 native event 而硬门失败，只能把约 `0.194 ms` 当方向性诊断；v9b
  jobs12790/12794 通过，job12795 因 launcher 缺少 directed-sidecar 环境变量在建 job 目录前
  preflight 失败。二者都没有合法性能点估计。

性能 decision 与四臂原始数据见本地 [hardened lean archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-v7-v11-continuation-hardened-lean-evidence-20260830.tar.gz)
及解包后的 [job12905 decision](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-stress-3pct-artifacts-20260830/job12905/v11-q-fragment-reuse-vs-v9-3pct-incremental-decision-job12905.json)。
v11 的 [wheel](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-wheel-artifacts-20260830/job12957/wheel-driver-verification-job12957.json)、
[fresh backend](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-fresh-backend-artifacts-20260830/job12960/plugin-v11-fresh-backend-decision-job12960.json)、
[NCU mechanism](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-vs-v9-ncu-artifacts-20260830/job12965/v11-v9-mechanism-job12965.json)
与 [lifecycle](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-lifecycle-artifacts-20260830/job12977/native-c2-v11-lifecycle-job12977.json)
亦已归档；机器可读的摘要、失败审计与未完成边界见
[continuation closure](experiment_logs/c2_native_c2_production_closure_20260830.json)及其 sidecar manifest。
v12 的 AOT、directed、ABBA、NCU、wheel 与旧 lifecycle failure audit 见
[v12 lean evidence](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-q-row-padding-evidence-20260830/v12-q-row-padding-lean-evidence-20260830.tar.gz)。
成功重放另见 [job13334 lifecycle archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-lifecycle-closure-evidence-20260831/c2-v12-lifecycle-job13334-evidence-20260831.tar.gz)；
v13 的 AOT 成功与 directed stop-loss 见 [v13 archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v13-distributed-merge-stoploss-evidence-20260831/c2-v13-distributed-merge-stoploss-evidence-20260831.tar.gz)。
v14/v15 归档已落地并本地验证，根为
[continuation evidence root](experiment_logs/c2_native_c2_production_evidence/c2-native-v14-v15-continuation-evidence-20260831/)，其中包括
[lean tar](experiment_logs/c2_native_c2_production_evidence/c2-native-v14-v15-continuation-evidence-20260831/c2-native-v14-v15-continuation-lean-evidence-20260831.tar.gz)、
[lean manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v14-v15-continuation-evidence-20260831/c2-native-v14-v15-continuation-lean-evidence-20260831.manifest.sha256)、
[excluded-fixture list](experiment_logs/c2_native_c2_production_evidence/c2-native-v14-v15-continuation-evidence-20260831/c2-native-v14-v15-continuation-excluded-fixtures-20260831.sha256)、
[full manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v14-v15-continuation-evidence-20260831/c2-native-v14-v15-continuation-evidence-20260831.manifest.sha256)
与 [full sidecar](experiment_logs/c2_native_c2_production_evidence/c2-native-v14-v15-continuation-evidence-20260831/c2-native-v14-v15-continuation-evidence-20260831.tar.gz.sha256)。lean SHA-256 为
`5ae30832ed5345b4d51cb69b378f472774d7792355a6e5ab93c107e7a9671846`，161 条 manifest/162 个安全 tar member
均通过；仅排除两份 fixture payload，其 SHA 由 excluded-fixture list/full manifest 固定。完整远端包 SHA-256
为 `0a5260556ad189ec0b0b9c405fa0b5f10aaa0f84d98ab4ab9fa0ca0d21301458`，未被表述为已下载的本地完整 tar。

job13666 的 [NCU archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-v14-v15-ncu-evidence-20260831/c2-native-v12-v14-v15-ncu-job13666-evidence-20260831.tar.gz)、
[30-record manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-v14-v15-ncu-evidence-20260831/c2-native-v12-v14-v15-ncu-job13666-evidence-20260831.manifest.sha256)
与 [operator audit](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-v14-v15-ncu-evidence-20260831/ncu-jobs13662-13666-operator-audit-20260831.json)
在本地通过：tar SHA `8ccfb059…`、30 条 manifest/31 个 member 与 ratio 重算均 PASS。controller 随后 purge
job13666 且 slurmdbd 不可用，因此 archive 不能独立证明 scheduler 状态；`COMPLETED/0:0` 仅是完成后的即时
operator observation。

**失败记录不能计入成功证据。** jobs **12366/12393/12493/12500/12772/12784/12795/12829** 是
此前的 fail-closed 门禁修复过程；v11 新增的 job **12956** 只可确认在 wheel 输入/身份门
fail-closed（归档不足以证明更具体的断言），jobs **12961/12963** 是 NCU JSON parser 失败，
job **12976** 在 artifact 创建前因提交的三个预期哈希错误而失败，既无 final-status 也无有效
运行日志。重跑的 jobs **12957/12965/12977** 分别成功完成相应门。它们均保留在失败审计中，
  不进入成功 scoreboard；jobs12513/12557 仍只支持旧 5% 问题下的性能拒绝，而
  jobs12599/12776/12905/12985 才支持本轮新 `>3%` 问题下的 direct-plugin 接受。job13334 单独
  覆盖 v12 lifecycle；job13439 是 v13 stop-loss，job13518 是预测量 argument-unpack 失败、无测量，
  job13539 与 job13576 分别是 v14/v15 的 RC3 性能拒绝，job13662 是无 GPU/NCU/final-status 的 NCU
  pre-snapshot fail-closed。v16 的 job13767 是编译前网络基础设施失败、job13783 是零字节原日志的
  directed preflight（原因只由后续确定性 replay 定位）；二者不计入候选失败或成功。job13773/13786/
  13789/13832/13845 才是 v16 的有效 AOT/directed/ABBA/wheel/lifecycle 链。两套门槛与边界不能追溯
  改写或混为 wheel 晋升。job13942 是第二 UUID allocation gate 的受控 RC3：它分到 job13789 原 UUID，
  因而在 fixture/A-B/计时前停止；[operator audit](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-second-uuid-attempt-evidence-20260831/job13942-operator-audit.json)
  与 [manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-second-uuid-attempt-evidence-20260831/c2-native-v16-second-uuid-attempt-evidence-20260831.sha256)
  只证明当前调度不可择 UUID，不是性能或候选裁决。

**仍未完成的边界。** 还没有官方 upstream/release/CI 认证；缺少携带真实权重与服务
调度的 full-model/server 端到端实验；没有严格完成“同一冻结 v16 wheel 的 v12/v16 ABBA 在第二个
  GPU UUID”的跨 GPU 性能复现；也没有覆盖契约外的广泛 shape/模式。v16 fresh backend 已闭环但
  `2.081999763x` 未获 Triton parity；matching v12/v16 NCU 已闭环但只给出有限机制结论，不能重算 ABBA。
  跨 GPU 性能复现仍是边界；job13942 未产生任何性能值且不关闭该边界。当前最高可行动技术项是
  native/Triton kernel gap；full-model 在
  checkpoint/config 资产未提供前继续停止重复搜索。不得把 NCU timing、fresh 跨作业比率或其他 UUID
  的非性能门预写成严格性能复现。机器可读边界由
  [v16 successor closure](experiment_logs/c2_native_c2_production_closure_20260831_v16.json)与
  [v16 sidecar](experiment_logs/c2_native_c2_production_closure_20260831_v16.sha256)固定；历史
  [v12–v15 successor closure](experiment_logs/c2_native_c2_production_closure_20260831.json)、
  [历史 sidecar](experiment_logs/c2_native_c2_production_closure_20260831.sha256)及
  [53-record aggregate](experiment_logs/c2_native_c2_v12_v15_continuation_evidence_aggregate_20260831.sha256)
  保持不变且不被追溯改写。
  native 的选择性拒绝不等同于 Triton fallback，更不应被描述为任意输入上的通用替代品。
