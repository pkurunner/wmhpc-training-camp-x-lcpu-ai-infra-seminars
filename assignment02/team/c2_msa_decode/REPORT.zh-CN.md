# C2：MiniMax M3 MSA 小 batch decode 的测量、分析与 prepared 挑战

## 结论与证据状态

在冻结的 MiniMax M3 形状下，基线是 vLLM `d4da0c5` 的 Triton split-K
decode 加独立 LSE merge；B300 和 RTX 5090 都先完成 profile，之后才实现
挑战。B300 profile 显示 decode kernel 是主项（B=1/4/8/16 分别为
3.958/5.234/7.329/11.501 us），但 merge 仍占 26.8%–38.5%；这给了
“复用 workspace，且只在可验证时减少 merge”的动机。

原始挑战的终态证据是清卡 B300 的 [mode-aware v2 gate JSON](experiment_logs/c2_challenge_b300_mode_gate.json)
和 [job 4306 log](experiment_logs/c2_challenge_b300_mode_job4306.log)：schema 为
`c2-final-gate-v2-mode-aware`，当前 `challenge/cli.py` SHA-256 是
`ae6aabbb86a32d10f321d0319c801d201b6403779b05cb14daf83b83ba683768`，12 个
`B×storage` 全部 PASS。最终 selected policy 是 BF16 `[1,1,1,1]`、FP8 scalar
`[16,4,8,4]`、FP8 token `[4,16,16,4]`（次序均为 B=1/4/8/16），不是把旧版
BF16 `C=1` 结论强行外推给 FP8。公平 benchmark 也在同一 log 中，三种 mode
都有正确性 gated 的正收益，详见“挑战结果”。

另有官方 MiniMax-AI/MSA `80434d7`、其 CUTLASS submodule `eb61c91` 的直接
SM100 核心路径测量；Q=1、K=4096 的 B=1/4/8/16 都经独立 FP32 selected-page
参考验证。它补齐“官方核心 kernel 的 B>=16 对照”，但不伪装成同一
`d4da0c5` full-vLLM integration 的端到端对比；两者的版本、plan metadata、
wrapper 和集成路径不同。

P1 闭环新增两份清卡 B300 证据。其一是 [job 4340 的真实 TMA 两级
gather JSON](experiment_logs/c2_tma_two_level_gather_b300_job4340.json) 与
[完整日志](experiment_logs/c2_tma_two_level_gather_b300_job4340.log)：三条路径都
bit-exact，且 SASS 中可见 `UTMALDG.2D`。其二是 [job 4339 的三路径 BF16
同语义对照 JSON](experiment_logs/c2_fair_bf16_crossover_b300_job4339.json) 与
[完整日志](experiment_logs/c2_fair_bf16_crossover_b300_job4339.log)：同一输入、
同一独立 FP32 oracle 下，比较 vendored vLLM source wrapper、persistent-workspace
prepared Triton（selected `C=1`）和官方 MSA core。后者是跨 pin 的核心级对照，
不是 full-vLLM 同 pin 的端到端 scoreboard；其物理 ABI bridge、官方 plan 与工作区
均在计时外，且已逐项记录。

2026-08-28 起的续轮按优先级连续关闭了精确 `d4da0c5` 真实 backend-layer
CUTLASS-vs-Triton、两张 B300 的 kernel roofline、BF16/FP8 stage policy、`C=2`
真实 attention 的 DSM/cluster-scope mbarrier 正确性与故障活性；随后完成同步/topology
归因、warp producer、真实 BF16 Tensor Core QK、batched WMMA/TC-PV 与
production-native AOT/overlay-wheel/backend-layer 闭环。版本链先晋升 four-producer v4、
warp-parallel-softmax v5，最终由 job 12278/12314/12322/12331/12385/12396 把
register-resident-numerator v6 晋升为当前**实验性 production-native 保留版本**。随后两条
基于冻结 v6 的两条单变量候选先按当时预设的 5% 门完成 fail-closed 验收：v7 raw-FP8 V
prefetch 在 job 12513 正确性与可比性全过但只改善 `4.8278% < 5%`；v8
softmax-metadata 双缓冲在 jobs 12534/12548 通过 AOT 与定向门，却在 job 12557 同卡
ABBA 中慢 `7.0892%`。二者均按原门槛拒绝，未构建 wheel/fresh/NCU，v6 wheel 不变。
此后用户为新的四小时优化续轮单独定义“严格改善 `>3%`”标准；它不追溯改写旧 5% 裁决。
在该新标准下，job 12599 对冻结 v7 做了独立同卡确认，改善 `4.9334%`、bootstrap 95% LCB
`4.7171%`，因此 v7 成为本轮直接 DSO 基线；再由 jobs 12701/12767/12776 将最终归并并行化
v9 关闭，v9 相对 v7 改善 `28.1018%`、LCB `27.9225%`，全部身份、资源、正确性与可比性门
通过。第三个方向 v11 仅在 v9 上复用跨 selected-page 不变的 Q WMMA fragments；其 hardened
AOT job 12825、修复 provenance 矛盾后的 directed job 12904 与同卡 ABBA job 12905 均通过，
v11 相对 v9 改善 `12.8146%`、LCB `12.7822%`。因此 v7、v9、v11 是本轮三个分别超过严格
`>3%` 门的单变量方向。它们仍只是 sequential direct-plugin 比较，不能相乘外推到 Triton 或
full-model。raw-FP8 K 跨页预取 v10 的 AOT/directed 通过，但 job 12784 因一次 profiler kernel
event 漏采以 `RC=2` 关闭，且两轮候选诊断延迟约 `0.194 ms`，没有形成合法性能裁决。

随后完成的是 v11 的部署前晋级，而不是新的性能百分比。job 12957 构建并逐 RECORD 验证了
同 distribution-version overlay wheel；稳定 DSO 保持逐字节不变，只加入冻结 v11 plugin 与
adapter。job 12960 从新安装目标起以 8 个 fresh Python 进程运行真实
`MiniMaxM3SparseMSAImpl.forward`，全部硬门通过；native/Triton seed-median 的中位数分别为
`0.111024/0.047664 ms`，逐 seed ratio 中位数为 `2.330648`，parity 未达成，且该试验没有
v9 fresh-wheel comparator，不能改写 job12905 的 v11-v9 性能裁决。job 12965 只提供机制补证：
v11 相对 v9 的 shared-load wavefront、shared-load bank-conflict 和 long-scoreboard 计数分别下降
`34.70%`、`52.41%`、`31.70%`，tensor 指令数不变；NCU duration/cycle 不进入性能 scoreboard。
job 12977 又在安装 wheel 上关闭 8 线程首次加载恰好一次、真实 forward、1000 次稳态调用、
CUDA Graph `100+100` replay 与 query mutation、选择/拒绝、`2048/2049/4095/4096` 序列边界、
有界内存计数及清卡门。故 v11 当时已替代 v6，成为本项目最新的**完整通过本地受控 lifecycle 的
实验性 production candidate**；它仍不是 upstream/release wheel，也没有 full-model/server
E2E、Triton parity 或多进程长期服务认证。job12905 与 jobs12960/12965/12977 落在不同 B300
UUID，这增加了第二张 B300 上的集成、机制与生命周期证据，但不是同一冻结 workload 的严格
跨 GPU 性能复现。

继续沿 profile 最高优先级推进后，v12 仅把 Q shared stride 从 `128` 改为 `136`，job12985
相对 v11 改善 `3.649118%`、LCB `3.405307%`；jobs12986/12987/12991 又分别关闭有限 NCU
机制、overlay-wheel 身份和 8-seed fresh real-backend integration。修复 symlink 审计与
finalizer 后，job13334 又通过 v12 全部单进程 lifecycle 门；allowlist 中唯一
`state/uv-cache/wheels-v6/url` symlink 仍在受控 state area 内，resolved target 目录 manifest 已固定。
它是 v16 晋级前的 lifecycle-closed 基线，而非当前最新版本。
随后 v13 的分布式 merge AOT 成功，但 directed 首调用遇到 illegal instruction，按止损不跑 stress。
v14 只将 K/V stage stride `16→24`；AOT 与 directed 均通过，valid ABBA 的改善
`2.40452555%`、LCB `2.29466423%` 未达严格 `>3%`，以 `RC=3` 拒绝并保留 v12。v15 独立只将
Q-stage stride `136→144`；AOT job13564、directed job13575 与 valid ABBA job13576 均完成，但
改善 `-0.09713217%`、LCB `-0.11928154%`，同以 `RC=3` 拒绝。job13666 的同卡三臂 NCU 进一步显示
v14 的 load bank conflict 降 `66.2191%`，而 v15 反升 `8.2355%`；它支持/否定 shared-layout 机制，
但不重算任何 ABBA。两者都是独立 v12 比较，不能与历史百分比相加或连乘。随后 v16 将页内
raw-FP8 K 的 lookahead 限定在 lane-private K chunk，AOT job13773、directed job13786 与同 UUID
8-seed clean ABBA job13789 全过；其相对 v12 的 paired-median 改善为 **`+6.2838716258%`**，bootstrap
95% LCB 为 **`+6.1226889551%`**，严格通过本轮 `>3%` 门。wheel job13832 与 lifecycle job13845
也均闭环，故 v16 已取代 v12 成为最新 lifecycle-closed candidate。matching NCU job13868 支持
“同工作量下 instrumented cycles 更少”但不支持 long-scoreboard 下降；fresh real-backend job13900
的 8-seed native/Triton ratio 为 `2.081999763x`，集成全过但 parity 未达。完整边界、失败审计与
未完成优先级集中在文末。

### 2026-09-01 GitHub 发布摘要：只列已闭环新增成果

本次发布只整理已经通过既定正确性、身份、清卡、统计或 lifecycle 门的结果。尚未形成合法性能
裁决的方向不进入下表，也不据此增加完成度；它们在后文只作为声明边界保留。

| 已闭环成果 | 冻结证据 | 可发布结论 |
| --- | --- | --- |
| v16 单变量性能晋级 | job13789，8-seed 四进程 `v12→v16→v16→v12` clean ABBA | v16 相对 v12 paired-median 改善 `+6.2838716258%`，bootstrap 95% LCB `+6.1226889551%`，严格通过 `>3%` 门 |
| v16 wheel 与受控 lifecycle | jobs13832/13845 | exact overlay wheel、安装身份、8 线程首次加载、真实 forward、1000 次稳态、CUDA Graph、边界/拒绝矩阵、内存与清卡门全部闭环；v16 成为最新 lifecycle-closed 实验候选 |
| v12/v16 matching NCU 机制补证 | job13868，同输入、同 harness、隔离进程、每臂一个 native action | v16/v12 elapsed-cycles ratio `0.92172466`、tensor-active ratio `1.08181818`，支持“同工作量下 instrumented cycles 更少”；long-scoreboard ratio `1.11319760`，因此不声称其下降 |
| fresh wheel / 真实 MiniMax backend | job13900，8 seeds、每 seed 独立 fresh Python 进程 | 8/8 正确性与集成门通过；native/Triton seed-ratio 中位数 `2.081999763x`。这是已闭环的集成与差距观测，不是 Triton parity 或 v16/v12 晋级结果 |
| 可复核发布包 | v16、matching NCU、fresh-backend 的 lean archive/manifest/sidecar 与 successor closure | 本次发布使用轻量归档和逐项 SHA-256 清单；大体积 wheel payload 不进入 Git，wheel 与插件身份仍由 manifest、RECORD 和固定哈希约束 |


## 变量与冻结形状

| 变量 | 含义 | 值/下标 |
| --- | --- | --- |
| `B` | 同时 decode 的 request 数 | `{1,4,8,16}` |
| `H_q` / `H_kv` | query/KV head 数 | `64` / `4` |
| `G=H_q/H_kv` | 每个 KV head 服务的 query head 数 | `16` |
| `D` | head dimension | `128` |
| `P` | KV page（也是 sparse block）长度 | `128` tokens |
| `K_top` | 每个 KV head 选取的 page 数 | `16` |
| `L=K_top P` | 每个 query head 参与 softmax 的 token 数 | `2048` |
| `C` | split-K 的 chunk 数 | baseline auto=`16/16/8/4`；selected：BF16=`1/1/1/1`，scalar=`16/4/8/4`，token=`4/16/16/4` |
| `O_c, lse_c` | 第 `c` 个 partial output / base-2 LSE | split-K 的临时工作区 |
| `page` | 物理 KV page | `page=block_table[request, topk_idx[kv_head,q,c]]` |
| `s` | attention 缩放 | `1/sqrt(D)` |
| `F_page` | 下标 `page` 表示单个 `(request,kv_head,page)` | 该边界内 QK+PV 的 FLOP 数 |
| `M_page` | 下标 `page` 同上 | 该边界内 K+V 的最小读字节数 |
| `AI_page=F_page/M_page` | 下标 `page` 同上 | 忽略额外流量时的纸面算术强度上界 |
| `S_Q` / `S_{KV}` | Q tile / K/V staging tile 的 leading stride，单位 BF16 元素 | v12/v16 Q=`136`；v14 K/V=`16→24`；v15 Q=`136→144` |
| `W,T,e` | K/V staging 的 warp 数、每 warp WMMA tile 行数、BF16 元素字节数 | `8,16,2 B` |
| `R_{12},R_{14},R_{15},R_{16}` | v12/v14/v15/v16 AOT static shared memory（下标为候选版本） | `31136/33184/31392/31136 B` |
| `K_c` | 下标 `c` 表示页内的 raw-FP8 K lookahead chunk；只在 v16 的 lane-private 预取路径中使用 | v16 的冻结单变量；不改变跨页次序、PV、merge 或 ABI |
| `I_{v16←v12}` / `LCB_{v16←v12}` | v16 相对 v12 的同卡 8-seed paired-median 改善 / deterministic paired bootstrap 95% 下界 | 严格接受门：`I>3%` 且 `LCB>0` |
| `\rho_{x/12}^{m}` | 同 UUID、同 input/harness 下候选 `x` 相对 v12 的 NCU counter `m` 比率 | `x∈{14,15,16}`；只解释机制，不是性能倍率 |
| `C_{ld},W_{ld},S_{long},S_{wait},N_{TC}` | `m` 的五个选中计数：shared-load bank conflict / wavefront / long-scoreboard / wait / tensor instruction | job13666 每臂均采 15 个 counter；本表只报告这五项 |

数据生成、独立参考、精度和 seed 已在
[验收协议](harness/ACCEPTANCE_PLAN.zh-CN.md) 冻结：随机且一一映射的
`block_table` 防止逻辑/物理页恰好同号，FP32 参考只聚合已选 16 页，最终 BF16
输出用 `rtol=atol=0.03` 验收。这样测到的是 sparse 语义，不是错误的 dense
attention 替代品。

## 复现、清卡与基线测量

两张卡的完整 PRE/POST 审计均显示独占 GPU、compute-apps 为空且 `FINAL_RC=0`：

| 平台 | GPU / CC | 软件 | 原始证据 |
| --- | --- | --- | --- |
| B300 | NVIDIA B300 SXM6 AC / 10.3 | CUDA 13.0、Torch 2.13.0+cu130、Triton 3.7.1 | [clean log](experiment_logs/c2_baseline_b300_job4306.log) 与 4 份 trace |
| RTX 5090 | NVIDIA GeForce RTX 5090 / 12.0 | CUDA 13.0、Torch 2.13.0+cu130、Triton 3.7.1 | [clean log](experiment_logs/c2_baseline_5090_job6845.log) 与 4 份 trace |

两次基线都在挑战实现前按 B=1、4、8、16 profile。每个 profile 取 10 次调用的
CUDA 平均（us）：

| B | B300 decode | B300 merge | merge 占比 | 5090 decode | 5090 merge | merge 占比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.958 | 2.474 | 38.46% | 3.504 | 1.443 | 29.17% |
| 4 | 5.234 | 3.139 | 37.49% | 8.397 | 2.006 | 19.29% |
| 8 | 7.329 | 4.332 | 37.15% | 13.171 | 2.720 | 17.12% |
| 16 | 11.501 | 4.211 | 26.80% | 21.780 | 2.586 | 10.61% |

所以两卡都由 `_gqa_sparse_decode_kernel` 主导；B300 的 merge 比例更高，尤其是
小 B。不同设备的数字不是同一种端到端 service latency：profile 的 CUDA 核时间
用于找瓶颈；另有 host benchmark 用来观察 wrapper/launch 影响。基线 12 个
正确性组合都通过，最大绝对误差不超过 `6.103515625e-05`，远小于 0.03 gate。

原始基线的逐次同步 host median（ms）如下，作为两架构对照；它包含旧 convenience
wrapper 的输出分配，不能与后面的 final fair challenge 作加速比：

| 平台 / storage | B=1 | B=4 | B=8 | B=16 |
| --- | ---: | ---: | ---: | ---: |
| B300 BF16 | 0.04722 | 0.04707 | 0.04848 | 0.04861 |
| B300 FP8 scalar | 0.04764 | 0.04728 | 0.04930 | 0.05882 |
| B300 FP8 token | 0.04920 | 0.04864 | 0.05059 | 0.05897 |
| RTX 5090 BF16 | 0.12978 | 0.12867 | 0.13134 | 0.13033 |
| RTX 5090 FP8 scalar | 0.13231 | 0.12920 | 0.13010 | 0.13276 |
| RTX 5090 FP8 token | 0.20071 | 0.13838 | 0.13265 | 0.13255 |

## 六个讨论点

### 1. 算术强度与 Tensor Core

对一个 `(request,kv_head,page)`，忽略 Q、输出、index 与 scale 的读写，QK 和 PV
各是一个 `G×D` 与 `D×P`（或 `P×D`）的矩阵乘：

`F_page = 2GPD + 2GPD = 4GPD = 1,048,576 FLOP`。

BF16 K 和 V 的最小读流量为

`M_page = 2PD × 2 bytes = 65,536 bytes`，

所以 optimistic arithmetic intensity 上界为 `AI_page=F_page/M_page=16 FLOP/B`。
FP8 KV 若不计 scale 则是 32 FLOP/B 上界；per-token scale 的加载会把它再拉低。

| 量 | BF16 KV（忽略额外流量） | FP8 KV（忽略 scale） |
| --- | ---: | ---: |
| 每 page FLOP | 1,048,576 | 1,048,576 |
| K+V 最小 bytes | 65,536 | 32,768 |
| `AI_page` 上界 | 16 FLOP/B | 32 FLOP/B |

`tl.dot(q,k)` 与 `tl.dot(p,v)` 确实能使用 Tensor Core 型矩阵路径，因此答案不是
“完全没有用武之地”；但随机 page、两级间接、低 B、softmax 与额外 scale 都使有效
AI 不高。截至本节原始轮次，profile 没有记录 DRAM bytes/peak bandwidth，故没有把
这个纸面上界伪报成达到峰值的 roofline 百分比。文末 2026-08-28 续轮已用两张
B300、同一 metric set 补齐 DRAM/L2 bytes、Tensor Core active、stall counters、
可审计 FLOP 合同与机器 balance；完整的**kernel-boundary roofline**现已闭环。

### 2. partial + merge 要不要融合到 cluster/mbarrier

基线源码 [sparse_attn.py](vllm_msa_ref/sparse_attn.py) 对 split-K 分别发射
`_gqa_sparse_decode_kernel` 与 `_merge_topk_attn_out_kernel`；profile 的 merge
比例证明它不是可以忽略的固定项。`C=1` 时 partial 已是全量 softmax，因此挑战
将 partial alias 到调用方 `O` 并跳过 merge，这不是近似融合，而是代数上不需要
归并。

对 `C>1`，把多个 CTA 放进 cluster、以 mbarrier 通知 merge CTA 在理论上可行，
但仍要解决三件事：partial 的 LSE/out 要在 cluster shared memory 有确定的存储
与 phase；所有 producer 完成前 merge 不可读取；随机 KV page 的全局读不会因为
共置而消失。cluster 还占用协作 CTA 资源，低 B 时可能进一步降低 resident
clusters。原始挑战只用 baseline 的 PDL dependency，并在 `C=1` 消除第二次 launch；
文末续轮已在 B300 上完成真实 attention 的 DSM/cluster control 和 cluster-scope
release/acquire `mbarrier` 正确性原型，但它仍不是性能化或 production dispatch。

### 3. 两级间接寻址与 TMA

物理地址依赖两次运行时数据读取：先从 `topk_idx` 得到逻辑 `blk`，再从
`block_table[request,blk]` 得到物理 `page`，最后访问 `kv_cache[page,...]`。
这在源码中就是连续的 `tl.load(t_ptr)`、`tl.load(bt_row+blk)` 和普通
`tl.load(kv_cache_ptr+page*stride...)`。TMA tensor map 描述仿射坐标到内存的
固定 layout，不能在一次 descriptor load 中表达这个 data-dependent gather；
需要先由线程/warp scalar-load index，再做普通向量化 load（或先做软件 gather）。

为使该结论可由 API 和硬件共同复核，新增了独立的
[TMA 两级 gather 微基准](harness/tma_gather/README.zh-CN.md)。其专用变量如下；
这些变量只用于搬运实验，不替换上文注意力的 `P,D` 定义。

| 变量 | 含义 | 受控默认值 |
| --- | --- | ---: |
| `B` | request 数 | 4 |
| `K` | 每 request selected logical page 数 | 16 |
| `L` | 每 request 逻辑 page table 长度 | 64 |
| `R` | 物理 page 数 | 128 |
| `E` | 每个 payload page 的 `uint32` 元素数 | 1024 |
| `s=bK+r` | `(request b, selected-rank r)` 的连续输出槽位 | `0..BK-1` |
| `ell_s` | 运行时读取的 logical page，`topk[b,r]` | 随机且无重复 |
| `p_s` | 运行时读取的 physical page，`block_table[b,ell_s]` | 随机置换 |
| `X[p,e]` / `Y[s,e]` | 物理 page payload / 正确的连续输出 | `Y[s,e]=X[p_s,e]` |

该程序比较三个两腿、等端点字节量的路径：`software_staged` 以普通 CUDA 先
two-level gather、再普通连续 copy；`gather_then_tma` 的第一腿完全相同、第二腿以
真实 `cp.async.bulk.tensor.2d.shared::cluster.global.tile` 搬运；
`contiguous_then_tma` 使用已按 `s` 排列的同一 payload，仍保持两腿和相同
`4*sizeof(Y)` 的估计读写字节量。三条路径都对同一独立 host `uint32` reference
逐元素严格比较，错配时不进入 CUDA-event 计时。这样 1 vs 2 只比较第二腿的
软件 copy/TMA，2 vs 3 只比较第一腿是否需要动态页表寻址，而不把一次额外 materialize
隐瞒为“零成本 TMA gather”。

连续对照的 TMA map 明确只含 base、shape、stride、box，且每 CTA 发出的坐标固定为
`{0,s}`，所以其地址是 `base+s*E*sizeof(uint32_t)`。真实 paged 地址则为
`base+E*block_table[floor(s/K),topk[floor(s/K),s mod K]]+e`；后者需要两次运行时
内存读取，且 `CUtensorMap` 没有可放这两个 index-buffer 指针的字段。因此单个
tensor map 不能表达动态两级 gather；实际可行路径是线程/warp 读 index 并普通 gather，
然后才对连续 buffer 使用 TMA。

这一点的文档依据是 [CUDA 13.0 Programming Guide 的 TMA 小节](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma)：multi-dimensional bulk-tensor copy 使用 host 端
`cuTensorMapEncode*` 创建、以 `const __grid_constant__` 传入的 layout descriptor；
公开参数为 base/size/stride/box。由 descriptor 字段不包含 data-dependent index
load 得出“无法表达本访问图”是本报告的明确**推论**，不是把文档中未出现的 API 能力
冒充为直接引文；微基准正用于以实际 `CUtensorMap`/PTX 和硬件正确性门交叉验证这个
推论。

**B300 硬件结果（job 4340，`FINAL_RC=0`）：**PRE/POST 均为 0 MiB 且
compute-apps 为空；日志同时固定了 `tma_two_level_gather.cu` 的 SHA-256、实际
`nvcc` 命令和 `UTMALDG.2D` SASS 证据。三条路径先对同一独立 host `uint32`
reference 逐元素比较，三列错配均为 0，之后才允许 CUDA-event 计时：
[外层 allocation 审计](../../experiment_logs/b300_team_p1_closure_job4340.log) 也以
`TEAM_P1_FINAL_RC=0` 结束。

| B300 实测项（`B=4,K=16,L=64,R=128,E=1024`） | `software_staged` | `gather_then_tma` | `contiguous_then_tma` |
| --- | ---: | ---: | ---: |
| bit-exact mismatches | 0 | 0 | 0 |
| CUDA event 平均（ms） | 0.007272 | 0.008216 | 0.007186 |
| 估计有效带宽（GB/s；两腿读+写） | 144.200 | 127.621 | 145.921 |

因此这里的硬件结论是**语义与 API 约束都成立，而不是 TMA 魔法加速**：把第二腿由
software copy 换为真实 TMA 后，受控 `gather_then_tma` 为 0.008216 ms，未比
`software_staged` 的 0.007272 ms 更快；而已连续化输入的 TMA 对照是 0.007186 ms。
`CUtensorMap`/`UTMALDG.2D` 已真实执行，但 TMA descriptor 的 base/shape/stride/box
与固定坐标仍不能编码 `topk -> block_table -> physical_page` 的两次运行时 load。
实际 decode 仍须由线程/warp 做 index load 与 software gather，连续 staging 后才可
选择 TMA；本微基准不外推为整个 MSA decode 一定更快。

### 4. FP8 KV scale 放在哪里

基线定义三种 ABI：无 scale、K/V scalar、`[kv_head,max_kv_tokens]` per-token/head。
`KV_SCALE_MODE=2` 在加载 K/V 后使用 `page*128+off_n` 索引 scale，关键是 `page`
已经是**物理页**，故不会把随机 block table 错当为逻辑连续页。上游
[FP8 scale test](vllm_msa_ref/test_sparse_attn_fp8_scale.py) 的口径是先从同一
FP8 cache 按相同 scale 反量化出参考，再以 `2e-2` 比较，并明确检查未缩放结果
不可错误通过。

本 harness 采用同一思想，独立 FP32 sparse 参考并把 gate 放宽到 0.03 只为容纳
BF16 split-K/merge 舍入；12 个 BF16/FP8-scalar/FP8-token 组合均硬通过。因此
scale 应在 K/V 载入后、`tl.dot` 前应用，不能在最终输出层统一补乘。

### 5. profile 后的瓶颈结论与设计

先 profile 再设计的时间顺序可由两个基线 clean log 的 section 顺序与随后挑战
日志的时间戳复核。decode 是最大 CUDA 项，但 B300 merge 仍为 26.8%–38.5%，
于是挑战不猜测更大的 tile，而是：

1. 调用方在计时外拥有 `O`；
2. `PreparedSparseDecode` 持久复用 `O_c,lse_c`；
3. 搜索 `C` 与 warp；仅 BF16 selected `C=1` 时 alias `O_c=O`、跳过 merge，FP8 保留各自选中的 split/merge；
4. 对 BF16、FP8 scalar、FP8 token 分别验收。

这把潜在收益限定为 workspace/launch/merge 优化，而没有伪装成 FLOP 吞吐革命。

### 6. 验收方案

挑战前已经公开 [验收协议](harness/ACCEPTANCE_PLAN.zh-CN.md)。参照是独立的
`dense_sparse_attention_reference`（FP32 QK、softmax、PV，且只取 top-k 页），不是
Triton 输出自己比自己。每种 B/storage 报告 `max_abs,mean_abs` 并调用
`torch.testing.assert_close(rtol=atol=0.03)`；一项失败就不允许计时。当前
[mode-aware v2 final gate JSON](experiment_logs/c2_challenge_b300_mode_gate.json) 记录
12/12 PASS、所有 source hash 与每个 mode 的 selected chunk，故它是当前 revision
的硬门，并可由其记录的 current source hash 直接复核。

## 挑战结果：当前 revision 的公平计时

当前 B300 clean final run 的两侧均使用 caller-owned、计时外预分配的 output；每个
样本“一次调用 + 一次 device synchronize”获得 host 单步延迟，另以 CUDA event 的
20 次成组调用报告 steady-state（各 21 个样本）。这两个指标不可混称。下表由
[job 4306 的三个 `FINAL_FAIR_BENCHMARK_*` JSON 段](experiment_logs/c2_challenge_b300_mode_job4306.log)
逐项解析，数值为 eager steady-state CUDA median（us）；所有行先通过当前 gate。

| storage | B | selected C | baseline (us) | prepared (us) | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1 | 1 | 42.410 | 28.240 | 1.502x |
| BF16 | 4 | 1 | 43.206 | 26.486 | 1.631x |
| BF16 | 8 | 1 | 42.378 | 26.333 | 1.609x |
| BF16 | 16 | 1 | 41.987 | 28.902 | 1.453x |
| FP8 scalar | 1 | 16 | 44.096 | 33.805 | 1.305x |
| FP8 scalar | 4 | 4 | 43.587 | 33.626 | 1.296x |
| FP8 scalar | 8 | 8 | 42.986 | 32.915 | 1.306x |
| FP8 scalar | 16 | 4 | 44.582 | 34.435 | 1.295x |
| FP8 token | 1 | 4 | 44.224 | 33.251 | 1.330x |
| FP8 token | 4 | 16 | 43.738 | 32.624 | 1.341x |
| FP8 token | 8 | 16 | 44.614 | 32.931 | 1.355x |
| FP8 token | 16 | 4 | 44.448 | 33.840 | 1.313x |

同一 log 的单步 host median 也呈正收益：BF16 为 1.04–1.09x、FP8 scalar
1.21–1.33x、FP8 token 1.22–1.27x。原因不是“FP8 的 `C=1` 变快”，而是
mode-aware selected 保留 FP8 的可用 split 数，同时 persistent workspace 避免
重复分配；BF16 才通过 `C=1` 合法消除了 merge。CUDA graph replay 改变的是 host
launch 路径，不能替代本表 eager 结论。

## 官方 CUTLASS 核心路径与 vLLM B>=16 门槛

为避免把静态门槛说成测量，本次用
[官方 adapter](harness/official_msa_cutlass_bench.py) 和
[audit launcher](harness/run_official_msa_q1_audit.sh) 在 B300 直接调用
MiniMax-AI/MSA `80434d7f67877c6570ca19cac444b84bc9855dac` 的根 `fmha_sm100`
接口；作业日志同时记录 CUTLASS submodule
`eb61c911471867a5fd2466bfd8f29306cea6ebf8`。冻结 Q=1、K=4096、FP8 E4M3、
`H_q/H_kv/D=64/4/128`、page=128、topk=16、随机物理页表和每 request/head 的
top-k。每个 batch 都先对独立 FP32 selected-logical-page 参考通过
`rtol=.02,atol=.12,cosine>=.999`，才以 warmup 20、单 stream 100 个 per-call
CUDA events 计时：

| B | 官方核心 median (us) | 正确性 |
| ---: | ---: | --- |
| 1 | 26.432 | PASS |
| 4 | 26.272 | PASS |
| 8 | 26.432 | PASS |
| 16 | 26.080 | PASS |

四份机器可读记录是 [B1](experiment_logs/c2_cutlass_q1_b300_b1_job4308.json)、
[B4](experiment_logs/c2_cutlass_q1_b300_b4_job4308.json)、
[B8](experiment_logs/c2_cutlass_q1_b300_b8_job4308.json) 和
[B16](experiment_logs/c2_cutlass_q1_b300_b16_job4308.json)；完整 source pin、
GPU 清卡和同作业的对照在 [job 4308 log](experiment_logs/c2_cutlass_compare_b300_job4308.log)。
在当前 selected Triton 的 FP8 scalar 表中，CUDA event 为约 33 us（B=1/4/8/16
为 33.805/33.626/32.915/34.435 us）。这给出“当前官方核心 kernel 比该 Triton
路线快”的同形状方向性证据，且**并未出现“只在 B=16 才 crossover”**。

这不推翻 vLLM snapshot `d4da0c5` 的静态 `_MIN_CUTLASS_BATCH_SIZE=16`：其
[dispatch/plan 代码](vllm_msa_ref/msa_cutlass_sparse_decode.py) 包含版本、plan
metadata、cache/wrapper 与整合路径的门槛；官方 direct adapter 则是另一个
`80434d7` pin 的核心调用。两套数据不可混称为“同一 full-vLLM pin 的端到端
CUTLASS-vs-Triton scoreboard”。尤其本地仍没有能运行该 `d4da0c5` 完整
`vllm.third_party.fmha_sm100` integration 的 runtime/extension；不能凭 direct
adapter 伪造此端到端表。

负结果也保留：同一官方作业的 Q=8、K=4096、B=16 FP8 sweep 中，官方 dense 为
34.8 us、sparse 为 65.6 us（[CSV](experiment_logs/c2_cutlass_msa_b300_job4308.csv)）。
因此“官方核心路径很快”也不等于每个 Q/稀疏调度都会更快。

## 同 BF16 数据的三路径核心级对照（B300 job 4339）

为消除旧的“FP8 official core 对 BF16 Triton”条件不等价问题，job 4339 用每个
`B∈{1,4,8,16}` 各自固定 seed 生成一份 BF16 Q/K/V、随机物理 `block_table`、排序
`topk` 和 `seq_lens`；**同一 B 的三条路径共享完全相同的数据与 checksum**。三者均
使用同一个独立 FP32 selected-page causal-attention oracle，阈值均为
`rtol=atol=0.03`，caller-owned BF16 输出均在计时外。数据、oracle、output 和
CUDA stream 合同由上述 [JSON](experiment_logs/c2_fair_bf16_crossover_b300_job4339.json)
固定，而非由任一被测实现自证。

本项自身的 [job 4339 日志](experiment_logs/c2_fair_bf16_crossover_b300_job4339.log)
PRE/POST 均清卡且 `FINAL_RC=0`。完整 allocation 的
[外层日志](../../experiment_logs/b300_team_p1_closure_job4339.log) 总返回 `127`，
原因是同 allocation 内先前独立的 C1 子步骤失败；其中 `C2_BF16_FAIR_CROSSOVER`
子段明确为 `RC=0`，故不将外层失败误记为本 C2 三路径实验失败，也不借此省略该
子段自身的 PRE/POST 审计。

| 符号 / 路径 | 计时内工作 | 计时外或版本边界 |
| --- | --- | --- |
| source wrapper | vendored `d4da0c5` source wrapper 的一次 BF16 decode 调用 | 每调用内部分配 `o_partial/lse` |
| prepared Triton | 持久 workspace 的 prepared BF16 decode，selected `C=1` | workspace 已存在，`C=1` 合法地取消 merge |
| official MSA | `80434d7f67877c6570ca19cac444b84bc9855dac` 的 core call，CUTLASS `eb61c911471867a5fd2466bfd8f29306cea6ebf8` | plan/workspace 在计时前建立；与 vendored snapshot 非同一 pin |
| physical ABI bridge | 不在任一每调用计时内 | vLLM `[physical,H_kv,P,2D]` 的 K\|V 分拆成 official K、V `[physical,H_kv,P,D]` contiguous；`block_table` 保持二维，`topk` 由 `[H_kv,Q,K]` 改为 `[Q,H_kv,K]` |

所有 B 的三条正确性结果均 finite、PASS。表中为对同一独立 FP32 oracle 的最大绝对误差
（source / prepared / official）；完整 JSON 同时保留 mean error 与输入 checksum。

| B | source wrapper max abs | prepared `C=1` max abs | official MSA max abs | 三路径 gate |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.0000423621 | 0.0000252873 | 0.0000365302 | PASS / PASS / PASS |
| 4 | 0.0000308855 | 0.0000308855 | 0.0111136669 | PASS / PASS / PASS |
| 8 | 0.0000341311 | 0.0000213720 | 0.0111284507 | PASS / PASS / PASS |
| 16 | 0.0000349693 | 0.0000318224 | 0.0108991843 | PASS / PASS / PASS |

计时均为单一 stream 上 20 次 warmup 后的 100 个 per-call CUDA-event 样本，样本间
不额外同步；数字依次为 `p10 / median / p90`，单位 us。speedup 定义为 official 的
median 相对相应 Triton 路径的倒数（`source_median/official_median` 或
`prepared_median/official_median`）。表中延迟按 0.001 us 列出，原始浮点值保留在
机器可读 JSON；speedup 列到 6 位小数。

| B | source wrapper p10 / median / p90 | prepared `C=1` p10 / median / p90 | official MSA p10 / median / p90 | official/source | official/prepared |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 46.944 / 48.272 / 54.688 | 29.376 / 31.280 / 31.488 | 26.112 / 27.216 / 29.408 | 1.773662x | 1.149324x |
| 4 | 45.792 / 46.896 / 52.672 | 28.128 / 29.152 / 29.216 | 25.088 / 26.160 / 29.184 | 1.792661x | 1.114373x |
| 8 | 46.912 / 48.192 / 70.528 | 28.128 / 29.152 / 29.280 | 26.080 / 27.120 / 29.280 | 1.776991x | 1.074926x |
| 16 | 45.696 / 46.976 / 53.088 | 31.008 / 31.200 / 31.360 | 25.088 / 26.960 / 28.128 | 1.742433x | 1.157270x |

这份表支持的结论很窄：在这份**等 BF16 语义、等输入、等 oracle**的 B300 core-level
合同下，官方 MSA 的 median 比 source wrapper 快 1.742433–1.792661x，比 prepared
`C=1` 快 1.074926–1.157270x。它不支持“同 pin full-vLLM 已 crossover”的说法：
official 是另一 git pin，physical ABI bridge、official plan/workspace 生命周期与
prepared 的 persistent workspace 都在计时外，且 prepared 的 selected `C=1` 同时改变
split/merge 工作量。故不能把差值归因于单一 kernel 指令，也不能将它外推到完整服务端
端到端延迟或所有 FP8/shape。

## 交付自检与证据边界

- [x] 先在 B300 与 5090 profile Triton baseline；两卡 B=1/4/8/16 和三 storage mode 均正确。
- [x] 独立参考、冻结 tolerance、两级间接的随机 page data、源代码与设计文档。
- [x] 当前 mode-aware v2 clean B300 final gate：current hash、三 storage、四 batch，12/12 PASS；final fair benchmark 已归档。
- [x] 官方 MiniMax MSA `80434d7` / CUTLASS `eb61c91` 的 B=1/4/8/16 核心路径、独立参考和 B=16 对照已归档；Q=8 dense/sparse 负例也保留。
- [x] job 4340 真实 TMA 两级 gather：`UTMALDG.2D`、三路 bit-exact 0 mismatch、CUDA event 与 PRE/POST 清卡审计均归档。
- [x] job 4339 同 BF16 数据三路径：四个 B 的 source wrapper / prepared `C=1` / official MSA 都通过独立 FP32 gate，p10/median/p90、pins、hash、ABI bridge 与 workspace 合同均归档。
- [x] B300/5090 baseline profile、独立参考、随机两级页表、FP8 两种 scale 语义和 six-point 分析均有源码/日志链接。

**证据边界：**job 4339 的相等 BF16 语义数据仍只作跨 pin、bridge/plan/workspace
边界公开的核心路径证据。文末续轮已经补上精确 `d4da0c5` wheel、真实
`MiniMaxM3SparseMSAMetadataBuilder.build`、plan cache、`ForwardContext` 与
`MiniMaxM3SparseMSAImpl.forward` 的同 backend-layer 对照；但服务器没有 MiniMax
模型权重，也没有启动 scheduler/service，因此它仍不能冒充完整模型服务 E2E。

## 优化续轮补充：5090 SM120 的冻结 BF16 C=1 no-LSE

本补充与上文 B300 mode-aware final gate **并列记录而不混表**：它只说明 RTX 5090
SM120 上 BF16、`C=1` 的进一步 kernel 专用化，不替代 B300 结论，也不外推到 FP8
scalar/token。

| 变量 | 含义 | 终验值 |
| --- | --- | --- |
| `B` | decode batch | `1,4,8,16` |
| `C` | selected-page chunk | `1` |
| `G` | 一个 KV head 的 Q-head CTA 分片数 | `1` |
| `s` | candidate Triton stage 数 | `3` |
| `t` | 一次完整 decode 的 CUDA-event 延迟 | 每路径 202 样本的中位数 |

冻结配置在 [独立 B=4 freeze gate](challenge_v2/results/5090_sm120_nolse_s1_freeze_gate_job7001)
前固定为 `G=1, warps=4, stages=3, PDL=off, maxnreg=none`。它用 stable online
`(m,l,acc)` 删除 C=1 无消费者的 LSE 更新、LSE store 和 workspace；`G=1` 同时避免
`G=2` 方案把同一 K/V page 重复交给两个 CTA。对照严格为同一 JSON 内 current prepared
BF16 C=1（`warps=4, stages=3, PDL=auto`），输入 seed、caller-owned output、
persistent workspace 生命周期和独立 FP32 selected-page oracle 一致。

终验用 101 个 `ABBA` 对（`control→candidate→candidate→control`），所以控制和候选
各有 202 个单调用 CUDA-event 样本；这避免把探索网格的 min-of-grid 误写成最终速度。
完整 audit 与 source hash 在
[本地证据目录](challenge_v2/results/5090_sm120_nolse_s1_final_20260820)。PRE/POST 为同一
RTX 5090 UUID（SM120）、0 MiB、`compute-apps` 空，Slurm `FINAL_RC=0`；四份 JSON 已用
wrapper 严格重算 config、样本数、speedup 和 strict gate。

| B | control / candidate (us) | 合并 speedup | AB / BA speedup | 独立 FP32 gate |
| ---: | ---: | ---: | ---: | --- |
| 1 | 84.560 / 69.392 | **1.218584x** | 1.2050x / 1.2290x | PASS，`max_abs=6.10e-5` |
| 4 | 116.160 / 90.368 | **1.285411x** | 1.2585x / 1.3216x | PASS，`max_abs=3.05e-5` |
| 8 | 83.744 / 67.392 | **1.242640x** | 1.2218x / 1.2640x | PASS，`max_abs=3.05e-5` |
| 16 | 83.808 / 67.456 | **1.242410x** | 1.2372x / 1.2483x | PASS，`max_abs=6.10e-5` |

早先 `G=2` sweep 的单序最小值未被采用：冻结 AB/BA 后仅约 1.04--1.05x。这个负例
保留在 `challenge_v2/results/5090_sm120_nolse_abba_20260820`，说明上述 1.22--1.29x
不是从多组合中挑选的偶然最低值。该 5090 表本身不替代 B300 复验，也没有把任何
5090 速度数字声称为 B300 或 FP8 成绩；随后得到的 B300 独立复验结果如下。

### B300 clean 复验：保留 current prepared

同一源码、同一冻结配置和 ABBA 合同随后在 clean B300 job 4446 上独立复验。每个 B
均先过独立 FP32 selected-page causal-attention oracle，之后控制与候选各有 202 个
single-call CUDA-event 样本；PRE/POST 为 0 MiB、compute-apps 为空。结果与 5090
方向不同：

| B | current prepared / no-LSE (us) | 合并 speedup | 结论 |
| ---: | ---: | ---: | --- |
| 1 | 30.560 / 28.448 | **1.074241x** | 正收益，但未达 10% |
| 4 | 28.448 / 30.496 | **0.932844x** | 保持 current prepared |
| 8 | 28.480 / 30.528 | **0.932914x** | 保持 current prepared |
| 16 | 30.464 / 32.544 | **0.936087x** | 保持 current prepared |

机器可读结果见 [B300 no-LSE 终验目录](challenge_v2/results/b300_sm103_nolse_s1_job4446)。
该 runner 的 `RC=3` 表示四个 context 未全部通过 strict 10% speedup gate，是预期的
性能判定，而非清卡、执行或 oracle 失败。故此**冻结 no-LSE 配置**是 5090/SM120 的
架构特化选择；当前证据下 B300 继续使用 current prepared，且不把此结论外推 FP8。


## 2026-08-28 B300 续轮：prepared 流水深度与机制闭环

| 符号 | 下标 / 上标含义 | 本轮含义与取值 |
| --- | --- | --- |
| `B` | 无下标；request batch | 终验为 `1,16`；先前 clean compact sweep 另含 `4,8` |
| `C` | 无下标；selected-page split/chunk 数 | `1`，因此 partial 已是全量 softmax，merge bypass |
| `s_3,s_5` | 下标 `3/5` 表示 decode software-pipeline stage 数 | control `s_3=3`，candidate `s_5=5` |
| `t_3,t_5` | 下标同上 | 单调用 CUDA-event 延迟；每路径合并 202 个 AB/BA 样本的中位数 |
| `R_{5/3}` | 下标 `5/3` 表示 stage-5 相对 stage-3 | `R_{5/3}=t_3/t_5`，大于 1 表示 candidate 更快 |
| `D_3,D_5` | 下标同上 | NCU 记录的 DRAM read+write bytes |
| `L_3,L_5` | 下标同上 | NCU `lts__t_bytes.sum`，即本轮比较使用的 L2 traffic 计数 |
| `I_3^{TC},I_5^{TC}` | 上标 `TC` 表示 Tensor pipe；下标同上 | Tensor pipe executed instruction 数 |
| `A_3^{TC},A_5^{TC}` | 上标、下标同上 | Tensor pipe active，占 peak sustained active 的百分比 |
| `S_3^{long},S_5^{long}` | 上标 `long` 表示 long-scoreboard stall；下标同上 | 每 issue-active 的归一化 warp-stall 百分比 |
| `S_3^{wait},S_5^{wait}` | 上标 `wait` 表示 wait stall；下标同上 | 同一 NCU set 的 wait-stall 百分比 |
| `N_{ABBA}` | 下标表示 ABBA pair 数 | `101`，因此 control/candidate 各 `202` 个 event 样本 |

### 为什么本轮优先做这条线

先前 [clean compact sweep](experiment_logs/c2_tuned_bf16_compact_clean_job4420.json) 在未经
冻结 AB/BA 的筛选数据中给出 `B=1` 的 `1.118406x` 和 `B=16` 的
`1.144633x` stage-5 信号，而 `B=4/8` 分别只有 `0.996622x/1.000000x`。
这条线只改变 Triton software-pipeline 深度，能在四小时窗口内同时完成独立 oracle、
冻结 AB/BA、跨 GPU UUID 复现和 NCU 机制检查；相比之下，同 pin full-vLLM extension
重建和 `C>1` cluster/mbarrier 融合都需要跨模块实现，风险明显更高。

冻结配置为 BF16、`C=1`、decode/merge warps 均为 4、PDL auto、maxnreg none；
control/candidate 只有 decode stage `3→5` 不同。由于 `C=1` 已跳过 merge，
候选的 merge stage 固定为 3 且不进入计时路径。每个 context 先过独立 FP32
selected-page causal-attention oracle，再运行 30 次各自 warmup 和 101 个
`control→candidate→candidate→control` 对；探索网格的 min-of-grid 不进入终表。

### 冻结 AB/BA 结果

| Slurm job / GPU UUID 前缀 | B | `t_3 / t_5` (us) | `R_{5/3}` | strict 10% | 独立 FP32 gate |
| --- | ---: | ---: | ---: | --- | --- |
| 9943 / `GPU-3924` | 1 | 30.560 / 27.360 | **1.116959x** | PASS | PASS，max abs `3.05e-5` |
| 10018 / `GPU-7787` | 1 | 30.848 / 27.296 | **1.130129x** | PASS | PASS，max abs `3.05e-5` |
| 9943 / `GPU-3924` | 16 | 30.560 / 28.512 | **1.071829x** | FAIL | PASS，max abs `6.10e-5` |

job 9943 的逐 pair 比值中，`B=1` 有 `193/202` 个大于 1、`158/202`
个达到 1.10；`B=16` 虽有 `199/202` 个大于 1，却只有 `26/202` 个达到
1.10。job 10018 又在另一 B300 UUID 上把 `B=1` 复现为 `1.130129x`。
因此结论不是“stage 5 全局替换 stage 3”，而是：

- B300、BF16、prepared `C=1`、`B=1` 可把这份冻结 stage-5 配置作为 opt-in
  架构/shape 专用策略；
- `B=16` 只支持可重复的小幅正收益，不通过既定 10% 门槛，继续保留 stage 3；
- `B=4/8` 仅有先前 compact clean 的无收益证据，本轮没有把它们伪装成 AB/BA
  终验；保守保持 stage 3；
- 本轮没有测试 FP8 scalar/token，不能把 BF16 策略外推到 FP8。

机器可读 JSON、raw event 样本、source hash 与 PRE/POST 清卡日志位于
[prepared stage-5 ABBA 目录](experiment_logs/prepared_stage5_abba)；两个 job 的
PRE/POST 都是 0 MiB、compute-apps 空。job 9943 因同轮 `B=16` 未过 strict 10%
而按 wrapper 合同预期返回 `RC=3`，但 correctness、raw 样本和清卡 gate 均有效；
job 10018 的 B=1-only strict gate 通过并完成 audit。

### NCU 机制：收益来自 latency hiding，而非减少流量或计算

先用 [section profile job 9986](experiment_logs/prepared_stage5_ncu) 固定
`_gqa_sparse_decode_kernel` 的 stage-3/stage-5 matching launch；其方向性结果显示
eligible-warps 比例从 18.14% 升到 26.01%，active IPC 从 0.72 升到 1.02，而 achieved
occupancy 基本不变（6.24%/6.17%）。stage 5 并非零成本：shared-memory block limit
使 theoretical occupancy 从 18.75% 降到 6.25%；这里只因 B=1 网格很小，实际 occupancy
没有继续下降。随后 job 10100 只收九个显式 counters，raw 宽表强制 base units、
单一 action、每指标恰好一个 finite 值：

| 指标 | stage 3 | stage 5 | 同 set 变化 |
| --- | ---: | ---: | ---: |
| DRAM read bytes | 4,161,536 | 4,165,888 | +0.1046% |
| DRAM write bytes | 0 | 0 | 不变 |
| L2 traffic bytes | 6,624,896 | 6,676,064 | +0.7724% |
| Tensor pipe instructions | 16,384 | 16,384 | 不变 |
| Tensor-active | 13.18% | 18.21% | +5.03 pp；1.3816x |
| long-scoreboard stall | 164.86% | 22.46% | -142.40 pp；相对 -86.38% |
| wait stall | 147.66% | 144.46% | -3.20 pp；相对 -2.17% |
| active warps / cycle-active | 1 | 1 | 不变 |

`per_issue_active.pct` 是归一化 warp-stall 指标，可以大于 100%，不能当作单个
warp 的概率；这里只在相同 B300、相同 metric set、相同 matching-launch 合同内比较。
由“DRAM/L2 基本不变、Tensor 指令完全相同、active warps 不变、long-scoreboard
大幅下降”推断，stage 5 的 B=1 收益主要是更深 software pipeline 隐藏随机页加载的
长延迟，并提高既有 Tensor 工作的活跃占比，而不是少读 KV 或少做矩阵计算。

NCU replay/instrumentation 下 driver JSON 的延迟字段均明确标记
`timing_valid_for_benchmark=false`；本节的性能倍率只来自上面的非 profiler AB/BA。
精确 raw CSV、`.ncu-rep`、driver JSON、filtered metric catalog、SHA256 manifest 和
清卡日志在 [job 10100 目录](experiment_logs/prepared_stage5_ncu_counters_v1)。
job 10049 因第一版 parser 假设了错误的 CSV 形状而在 stage 3 后安全中止，
`FINAL_RC=1`，只保留为 [失败门禁证据](experiment_logs/prepared_stage5_ncu_counters_v1/failed_job10049)，
不参与任何性能结论。汇总值与证据 hash 另见
[机器可读续轮摘要](experiment_logs/prepared_stage5_followup_summary_20260828.json)。

## 2026-08-28 八小时续轮：按优先级连续闭环

先给出本节新增推导和同步协议中的符号，避免把上文同名量用于不同边界：

| 符号 | 下标 / 上标含义 | 本节定义 |
| --- | --- | --- |
| `F` | 无下标；冻结 kernel 的矩阵 FLOP | QK 与 PV 各按一次 multiply-add=2 FLOP 计数 |
| `M_logic` | `logic` 表示逻辑最小流量 | Q/K/V/output、LSE、top-k、block table 与 seq-len 各读写一次 |
| `M_dram^(j,s)` | 上标 `j` 为 Slurm job，`s∈{3,5}` 为 stage | NCU 的 DRAM read+write bytes |
| `AI_logic` / `AI_dram^(j,s)` | 下标表示分母来源 | `F/M_logic` 与 `F/M_dram^(j,s)` |
| `P_dense` | `dense` 表示非稀疏 BF16 峰值 | 由 HGX B300 官方稀疏平台值折算的单 GPU 值 |
| `W_HBM` | HBM 带宽峰值 | B300 单 GPU 官方 “up to 8 TB/s” |
| `rho=P_dense/W_HBM` | 无下标；roofline ridge | 计算屋脊与带宽屋脊的交点 |
| `t_3,t_5` | 下标为 decode stage 数 | 完整 prepared dispatch 的 CUDA-event 中位数 |
| `R_{5/3}=t_3/t_5` | 分子/分母对应 control/candidate | 大于 1 表示 stage 5 更快 |
| `r_0,r_1,r_2,r_3` | 下标为 cluster rank | 两个 producer、一个 merge consumer、一个生命周期参与 CTA |
| `A_mb` | `mb` 表示 mbarrier | producer arrival count，固定为 2 |
| `phi_0` | 下标 0 表示初始 phase | consumer 等待的初始 parity，固定为 0 |
| `N_poll` | `poll` 表示 bounded try-wait 轮询 | 负路径固定为 `2^20` 次 |
| `t_sync,t_mb` | 下标为 data-ready 协议 | `cluster.sync` 与 remote-DSM mbarrier 的 AB/BA 合并中位数 |
| `R_mb=t_sync/t_mb` | `mb` 表示 mbarrier candidate | 大于 1 表示 mbarrier 更快；本轮实质门槛为 1.10 |
| `N_CTA` | 每个 cluster 的 CTA 数 | topology 对照比较 4 与 3 |
| `t_{4CTA},t_{3CTA}` / `S_topo=t_{4CTA}/t_{3CTA}` | 下标为 `N_CTA` | 大于 1 表示真正 3-CTA topology 更快 |
| `t_scalar,t_warp` | 下标为 producer mapping | scalar 与 warp-producer 完整 kernel 的 AB/BA 合并中位数 |
| `R_w=t_scalar/t_warp` | `w` 表示 warp producer | 大于 1 表示 warp mapping 更快 |
| `t_TC` | `TC` 表示 WMMA-QK candidate | 含新增 Q/score shared 与 CTA barrier 的完整 candidate 中位数 |
| `R_TC=t_warp/t_TC` | 分子/分母为 control/candidate | 大于 1 表示真实 BF16 Tensor Core QK 路径更快 |
| `t_{w,b},t_{TC,b}` | 下标 `w/TC` 为 batch control/candidate，`b` 为 batch size | native batch ABI 下两臂完整 kernel 中位数 |
| `R_{TC,b}=t_{w,b}/t_{TC,b}` | 下标 `b` 表示 batch size | 大于 1 表示该 B 上 WMMA-QK 更快 |
| `t_{QK,wPV},t_{QK,TC-PV}` | `wPV/TC-PV` 表示同一 WMMA-QK 后的两种 PV 实现 | 分别为 warp-PV control 与含额外 shared/barrier 的 WMMA-PV candidate 中位数 |
| `R_PV=t_{QK,wPV}/t_{QK,TC-PV}` | 分子/分母来自同一次 PV ABBA | 大于 1 表示 WMMA-PV candidate 更快 |
| `b,L_b` | 下标 `b` 为 request | batch ABI 中 `b∈[0,B)` 及其独立 sequence length |
| `S_{b,kv}` | 下标为 request/KV head | 经 `topk→block_table` 得到的 selected 物理页集合/签名 |

### 1. 精确 `d4da0c5` 的真实 backend-layer 对照

[job 10650 机器可读 JSON](experiment_logs/full_vllm_d4_backend/c2_full_vllm_d4_backend_formal_b300_job10650_20260828T185314Z.json)、
[完整日志](experiment_logs/full_vllm_d4_backend/c2_d4_backend_10650.slurm.log)与
[SHA256 manifest](experiment_logs/full_vllm_d4_backend/c2_full_vllm_d4_backend_job10650.sha256)
固定了以下 provenance：

| 对象 | 精确版本 / 审计结果 |
| --- | --- |
| vLLM checkout / wheel | `d4da0c55af3aa231b6209bf77871f3ed36eab0d2`；wheel SHA256 `91156a7b…47e06` |
| MiniMax-AI/MSA / CUTLASS | `087c161814d4d9c735b46c21212a09e5f8eb92fa` / `eb61c911471867a5fd2466bfd8f29306cea6ebf8` |
| 安装树 | wheel 与 installed fmha 全树 960 文件一致；非 CUTLASS MSA 源树 86 文件一致 |
| 真实 API | `MiniMaxM3SparseMSAMetadataBuilder.build`、plan cache、`ForwardContext`、`MiniMaxM3SparseMSAImpl.forward` |
| 冻结 workload | B=16、Q=1、FP8 per-tensor scalar、K/V scale=0.25/0.5、selected pages=16 |
| 运行环境 | job 10650，B300 `GPU-7787…`，PRE/POST 空卡，`FINAL_RC=0` |

两臂共享同一输入、sorted top-k 与调用方 output 合同；metadata/build、MSA plan、
query FP8 量化和 oracle 都在计时外。50 个
`CUTLASS→Triton→Triton→CUTLASS` 周期给每臂 100 个样本：

| 后端 | p10 / median / p90 (us) | 独立 FP32 oracle | dispatch gate |
| --- | ---: | --- | --- |
| CUTLASS MSA | 38.496 / **40.640** / 46.624 | PASS，max abs `0.0133362` | 只调用 CUTLASS |
| Triton | 52.800 / **55.968** / 62.048 | PASS，max abs `6.1035e-5` | 只调用 Triton |

因此同 backend-layer 的中位数加速为 `55.968/40.640=1.377165x`，CUTLASS 更快。
这里的 `SimpleNamespace` 只充当已安装 config/layer API 所需的最小字段壳；builder、
metadata、plan、forward context 和 backend forward 都是真实 vLLM 对象。服务器没有
MiniMax 模型权重/HF cache，也没有启动 scheduler 或 HTTP service，所以**模型权重加载、
scheduler 与服务 E2E 仍不在证据内**；不能把 1.377165x 写成请求级吞吐或 TTFT 加速。

### 2. 两张 B300 的可审计 kernel roofline

冻结边界是 B=1、BF16、`H_q=64,D=128,L=2048,C=1` 的 prepared decode；
`C=1` 跳过 merge。只计 QK 与 PV 矩阵工作：

`F=2BH_qLD+2BH_qLD=67,108,864 FLOP`。

逐对象逻辑流量为 `M_logic=4,227,716 B`，所以
`AI_logic=F/M_logic=15.8736 FLOP/B`。两张卡、同一 NCU metric set 得到
`M_dram^(j,s)=4,161,536…4,166,144 B`，对应
`AI_dram^(j,s)=16.1081…16.1260 FLOP/B`；实测 DRAM 略小于逻辑流量是 setup
对象由 cache 服务所致，不是少做语义工作。

官方 [HGX 平台页](https://www.nvidia.com/en-us/data-center/hgx/)给出 8-GPU HGX B300
FP16/BF16 36 PFLOP/s（含 sparsity）；按 `36/2/8` 折算单 GPU dense
`P_dense=2.25 PFLOP/s`。官方 [HGX AI Factory 组件表](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)
给出单 B300 `W_HBM=8 TB/s`，于是 `rho=281.25 FLOP/B`。冻结点约
16.1 FLOP/B，位于理想 roofline 的 memory side。

| job / GPU | stage | DRAM bytes | L2 bytes | Tensor active | long-scoreboard |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10100 / `GPU-7787…` | 3 | 4,161,536 | 6,624,896 | 13.18% | 164.86% |
| 10100 / `GPU-7787…` | 5 | 4,165,888 | 6,676,064 | 18.21% | 22.46% |
| 10640 / `GPU-dadf…` | 3 | 4,162,816 | 6,734,080 | 13.15% | 165.88% |
| 10640 / `GPU-dadf…` | 5 | 4,166,144 | 6,678,432 | 18.38% | 26.24% |

有效 AB/BA 的 stage 3/5 吞吐只有 2.175/2.459 TFLOP/s；与 job 10100 同 UUID
配对后，有效 DRAM 带宽是 134.905/152.619 GB/s，即峰值的
1.686%/1.908%，dense BF16 峰值利用率仅 0.0967%/0.1093%。所以准确结论是：

- 理想 roofline 分类在 memory side；
- 但远未饱和 HBM，不能简称“带宽受限”；
- 结合每 cycle-active 仅 1 个 active warp 和很高的 long-scoreboard stall，
  当前瓶颈是随机页加载的长延迟、低并发和低 occupancy；
- stage 5 在两卡上都保持 DRAM 流量与 Tensor 指令数近似不变，同时降低
  long-scoreboard、提高 Tensor active，支持“latency hiding”机制解释。

NCU replay 的 duration 明确不用于性能倍率；延迟只来自独立 AB/BA。完整 raw CSV、
`.ncu-rep`、两卡 manifests 与推导在
[第二卡 NCU 目录](experiment_logs/prepared_stage5_ncu_counters_v2)和
[roofline 摘要](experiment_logs/prepared_stage5_roofline_summary_20260828.json)。

### 3. stage-5 dispatch policy 已覆盖 BF16 B=4/8 与两种 FP8

BF16 B=4/8 的 job 10684 在同一 B300 UUID 上对两个 base seed 各跑 101 个 ABBA
pair；四组均通过独立 FP32 oracle 与 source/raw-sample gate：

| B | seed 20260828 `R_{5/3}` | seed 20260829 `R_{5/3}` | 决策 |
| ---: | ---: | ---: | --- |
| 4 | 1.001653x | 1.001101x | 保留 stage 3 |
| 8 | 1.021300x | 1.024705x | 保留 stage 3 |

证据见 [BF16 policy 目录](experiment_logs/prepared_stage5_policy_abba)与
[机器摘要](experiment_logs/prepared_stage5_policy_summary_20260828.json)。结合既有
B=1 两 UUID 结果和 B=16 结果，BF16 只有 B=1、C=1 的 stage 5 达到预声明
1.10x，可作为窄 opt-in；B=4/8/16 均保持 stage 3。

FP8 另用 job 10711 冻结 scalar/token 各自的真实 scale ABI 与 selected-C 映射，
每个 `storage×B` 两个 seed、每臂 202 个完整 decode+required-merge 样本：

| storage | B=1 两 seed 均值 | B=4 | B=8 | B=16 |
| --- | ---: | ---: | ---: | ---: |
| FP8 scalar | 0.998711x | 1.000000x | 0.991581x | 1.005400x |
| FP8 token | 0.977255x | 0.999576x | 0.955271x | 0.984991x |

全 16 组范围为 0.951670–1.011605x，0 组达到 1.10x；所有 oracle/evidence gate
通过。少数 scalar 单次上下文略高于 1（最高 1.011605x），所以准确表述是
“stage 5 没有稳定或实质收益”，不是“每次都更慢”。FP8 两种模式、B=1/4/8/16
均保留 stage 3。job 的 `FINAL_RC=3` 是“至少一组未达策略门槛”的预期返回码，
不是正确性失败。原始证据、manifest 与汇总见
[FP8 ABBA 目录](experiment_logs/prepared_stage5_fp8_abba)和
[FP8 policy 摘要](experiment_logs/prepared_stage5_fp8_summary_20260828.json)。

### 4. `C=2` 真实 attention 的 DSM/cluster/mbarrier 正确性闭环

实现路径分三步，并且每一步都没有把 host 预计算 partial 塞给 consumer：

1. job 10674 先验证两个 producer 的 BF16 partial/FP32 base-2 LSE 可由 rank 2 通过
   `map_shared_rank` 从 DSM 读取并正确归并；
2. job 10722 把 producer 换成真实 BF16 QK、causal selected-page online softmax，
   用 `cluster.sync` 作 data-ready control；
3. 最终 job 10731 仅把 data-ready handoff 换成 rank 0/1 对 rank-2 remote DSM barrier
   的 release-arrive，以及 rank 2 的 local acquire parity wait。

最终形状为 `B=1,H_kv=4,H_q=64,G=16,D=128,P=128,K_top=16`；每个 cluster
有两个 producer、一个 merge consumer 和一个生命周期参与 CTA，共 4 个 cluster。
producer 各处理 8 个 selected page，只把 BF16 normalized partial 与 FP32 base-2
LSE 放在 CTA-local shared memory；没有 global partial scratch，也没有 atomic。
初始 `cluster.sync` 只保证 rank-2 barrier 初始化/cluster residency，最终
`cluster.sync` 只保护 producer shared lifetime，真正的 producer-ready 条件是
`A_mb=2,phi_0=0` 的 release/acquire mbarrier；该顺序与 NVIDIA
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) 的跨 CTA 示例和
[CUDA Programming Guide barrier 章节](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)
一致。

两个固定场景同时 poison 未选但可见的 page 与最后 page 的 127 个 causal-masked
token：seed 17 为 seq=2049、4 个未选可见 page；seed 2026 为 seq=3969、64 个未选
可见 page。输入层级在 oracle/GPU launch 前验证，独立 oracle 用 FP64 accumulator
和 natural-exp 两遍 softmax；partial 与 caller output 都是 BF16。结果如下：

| job / GPU | data-ready 协议 | max abs | tolerance | 结果 |
| --- | --- | ---: | --- | --- |
| 10722 / `GPU-7787…` | `cluster.sync` control | 0.000158180 | `atol=5e-4, rtol=5e-3` | PASS，`FINAL_RC=0` |
| 10731 / `GPU-3924…` | remote release + local acquire `mbarrier` | 0.000158180 | 同左 | PASS，`FINAL_RC=0` |

job 10731 的 PTX 强制包含
`mbarrier.arrive.release.cluster.shared::cluster.b64` 与
`mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64`；SASS 门禁另强制
`SYNCS.ARRIVE.TRANS64.RED.A1T0` 与 `SYNCS.PHASECHK.TRANS64.TRYWAIT`。bounded poll、120 秒 `timeout --kill-after`、PRE/POST
空卡、source pre/post hash、binary/PTX/SASS hash 和精确 seed 场景都由二次 gate
验证。证据见 [cluster.sync control](experiment_logs/c2_cluster_attention_smoke)与
[mbarrier 正式目录](experiment_logs/c2_cluster_attention_mbarrier_smoke)。

#### 同计算同步 AB/BA：mbarrier 没有实质性能收益

[job 10758 的正式 JSON](experiment_logs/c2_cluster_attention_sync_abba/c2_cluster_attention_sync_abba_clean_20260828T203626Z.json)、
[Slurm 日志](experiment_logs/c2_cluster_attention_sync_abba/c2_sync_abba_10758.slurm.log)与
[manifest](experiment_logs/c2_cluster_attention_sync_abba/c2_cluster_attention_sync_abba_job10758.sha256)
把两臂限制为同一真实 attention data plane、同一输入和 launch shape，只改变 producer-ready
协议。control 在符号范围内有三对 cluster arrive/wait；candidate 有两对 cluster
arrive/wait，并各有一条 mbarrier init、remote release-arrive 与 local acquire-wait。
两臂静态 shared 均为 4,172 B、寄存器均为 166、local memory 均为 0；两个 seed 的
pre-timing 输出及计时后的 final-state 输出都与独立 FP64 oracle allclose，且两臂 BF16
逐 bit 相同。这里没有检查每个中间 timed launch 的输出，报告不作比该证据更强的声明。

| data-ready arm | p10 / median / p90 (us) | AB / BA partition median (us) |
| --- | ---: | ---: |
| `cluster.sync` control | 3894.144 / **3894.656** / 3895.456 | 3894.624 / 3894.688 |
| remote DSM mbarrier | 3890.368 / **3891.264** / 3892.352 | 3891.296 / 3891.264 |

因此 `R_mb=3894.656/3891.264=1.000872x`；AB、BA 分区分别为 1.000855x、
1.000880x，均同向但远低于预声明 1.10x。这个结果支持“mbarrier 是正确可用的精细
handoff primitive”，不支持“在当前标量真实-attention 原型上有实质加速”。按停止条件，
不再扫 polling/count/parity 等纯同步参数，后续只投入会改变并行结构的 topology 或
warp producer。job 10742 的 GPU raw 数据虽完成，但旧 secondary gate 因 C++ float
常量的 JSON 序列化精度而 `FINAL_RC=1`；修复只允许 `1e-9` 绝对序列化误差，并经独立
复核后从头重跑为 job 10758，故性能结论只引用 10758。

#### 缺失 arrival 负路径：有界失败并正常收敛

[job 10741](experiment_logs/c2_cluster_attention_mbarrier_fault_smoke/c2_cluster_attention_mbarrier_fault_smoke_clean_20260828T203236Z.json)
把 rank-2 barrier 的 expected count 保持为 2，但编译产物只有一条、且源码只允许
rank-0/thread-0 执行的 remote arrival；rank 1 故意不 arrival。这不是硬件 arrival
计数器读数，而是 source/PTX/SASS 与控制流共同限定的 fault 配置。rank 2 最多轮询
`N_poll=2^20` 次，然后写 seed 编码的独立 status 与完整 BF16 fault sentinel；所有
CTA 仍无条件进入最终 `cluster.sync`。

seed 17 与 2026 的 4 个 cluster 全部观测到 phase 未完成，status 均匹配各自 seed，
8,192 个输出元素全部是预期 sentinel；单次 fault kernel 分别为 90.319 ms 与
90.307 ms，均低于 5 s 的 liveness 上界，并在外层 45 s watchdog 触发前自行返回。
PRE/POST 是同一 `GPU-7787…`、0 MiB、compute-apps 空，`FINAL_RC=0`。这些时长只作
有界活性检查，绝不是性能数字。完整工件与审计链见
[fault 目录](experiment_logs/c2_cluster_attention_mbarrier_fault_smoke)和
[job 10741 manifest](experiment_logs/c2_cluster_attention_mbarrier_fault_smoke/c2_cluster_attention_mbarrier_fault_job10741.sha256)。

job 10731 关闭的是“真实 attention partial/LSE 能否经 DSM 和真正 mbarrier 安全
handoff、不会在两组对抗输入上死锁或读早”的正确性 prerequisite。以下实验继续沿着
当时声明的停止条件推进性能结构，而不是在旧 scalar 同步参数上反复扫点。

#### Topology ABBA：3-CTA 无实质收益，冻结 scalar topology

[job 10783 正式 JSON](experiment_logs/c2_cluster_attention_topology_abba/c2_cluster_attention_topology_abba_clean_20260828T210431Z_job10783.json)、
[Slurm 日志](experiment_logs/c2_cluster_attention_topology_abba/c2_topology_10783.slurm.log)与
[manifest](experiment_logs/c2_cluster_attention_topology_abba/c2_cluster_attention_topology_abba_job10783.sha256)
比较完整 4-CTA 与真实 3-CTA 实现：后者改变 `clusterDim/grid`、block 到 KV-head 的
映射并删除空闲 `r_3`，不是只测一条 idle-rank 指令。

| topology | p10 / median / p90 (us) | regs / shared / local |
| --- | ---: | ---: |
| 4-CTA | 3912.768 / **3913.872** / 3915.872 | 166 / 4172 B / 0 B |
| 3-CTA | 3902.784 / **3904.928** / 3906.976 | 168 / 4172 B / 0 B |

`S_topo=1.00229043x`，AB/BA 分区为 1.00228637x/1.00228624x，远低于预声明
1.05x 门槛；两个 seed 均通过 FP64 oracle 且两臂 bitwise 相同。因此冻结全部 scalar
topology 与纯同步调参。这里量到的是完整 topology implementation cost，不是纯空闲
rank 的硬件成本，也不是 production/model/server 加速。

#### Warp producer：producer mapping 有 5.139x 强正收益

[job 10790 正式 JSON](experiment_logs/c2_cluster_attention_warp_producer_abba/c2_cluster_attention_warp_producer_abba_clean_20260828T211322Z_job10790.json)、
[Slurm 日志](experiment_logs/c2_cluster_attention_warp_producer_abba/c2_warp_abba_10790.slurm.log)与
[manifest](experiment_logs/c2_cluster_attention_warp_producer_abba/c2_cluster_attention_warp_producer_abba_job10790.sha256)
保持 4-CTA、DSM/mbarrier、merge、output ABI 与 selected causal attention 不变，只把
rank-0/1 producer 从“一线程一 head”改成一整个 warp 协作一个 head：

| producer arm | p10 / median / p90 (us) | regs / shared / local |
| --- | ---: | ---: |
| scalar | 3908.960 / **3911.232** / 3913.280 | 166 / 4172 B / 0 B |
| warp | 757.792 / **761.088** / 764.608 | 32 / 4172 B / 0 B |

于是 `R_w=5.13900084x`，AB/BA 分区为 5.14289296x/5.13360485x。两 seed 的两臂
都独立通过 FP64 oracle；seed 17 跨臂 max abs 为 `3.05175781e-5`、不是 bitwise，
seed 2026 bitwise 相同。PTX/SASS 的 warp candidate 都有 160 条 shuffle-down 与 32 条
shuffle-index，local memory 为 0。这是 native `C=2` correctness prototype 的
producer-mapping 信号，不是 production fusion、吞吐或服务级加速。

#### WMMA-QK：完整 Tensor Core QK candidate 再快 2.329x

[job 10841 正式 JSON](experiment_logs/c2_cluster_attention_tc_qk_abba/c2_cluster_attention_tc_qk_abba_clean_20260828T215623Z_job10841.json)、
[Slurm 日志](experiment_logs/c2_cluster_attention_tc_qk_abba/c2_tc_qk_abba_10841.slurm.log)与
[manifest](experiment_logs/c2_cluster_attention_tc_qk_abba/c2_cluster_attention_tc_qk_abba_job10841.sha256)
在同一 B=1、`C=2`、4-CTA 协议中以 warp producer 为 control。candidate 每页由 8 个
warp 分别计算一个 16-token tile，用 BF16 WMMA `m16n16k16`、FP32 accumulator 完成
QK；PV 仍按原 warp online-softmax 路径处理。

| arm | p10 / median / p90 (us) | regs / shared / local |
| --- | ---: | ---: |
| warp control | 758.592 / **760.512** / 762.656 | 32 / 4172 B / 0 B |
| WMMA-QK | 325.088 / **326.592** / 327.680 | 40 / 16480 B / 0 B |

`R_TC=2.32863012x`，AB/BA 为 2.32944738x/2.32637657x；两个 seed 的两臂都通过
FP64 oracle 且跨臂 BF16 bitwise 相同。符号级门禁确认 control 没有 BF16 MMA/HMMA，
candidate 有 64 条 PTX `mma.sync` 与 128 条 SASS `HMMA.16816.F32.BF16`。candidate
新增 Q/score shared storage 与 CTA barrier，因此这是**完整实现成本**，不能写成孤立
Tensor Core 指令收益；它也尚未接入 production dispatch。

#### Native batch ABI：warp 收益覆盖 B=1/4/8/16

[job 10873 正式 JSON](experiment_logs/c2_cluster_attention_warp_batch_abba/c2_cluster_attention_warp_batch_abba_clean_20260828T222229Z_job10873.json)、
[Slurm 日志](experiment_logs/c2_cluster_attention_warp_batch_abba/c2_warp_batch_10873.slurm.log)与
[manifest](experiment_logs/c2_cluster_attention_warp_batch_abba/c2_cluster_attention_warp_batch_abba_job10873.sha256)
把 ABI 扩为 query/output `[B,H_q,D]`、`seq_lens[B]`、`block_table[B,max_blocks]`、
`topk[B,H_kv,K_top]`，grid 为 `B·H_kv·4`。每个 request 使用互不重叠的物理页池；
两个 seed 都验证每个 `(b,kv)` 的 selected-set 签名、poison 未选但可见页和 causal tail。

| B | scalar median (us) | warp median (us) | `R_w` |
| ---: | ---: | ---: | ---: |
| 1 | 3986.768 | 774.416 | 5.148096x |
| 4 | 3986.768 | 774.560 | 5.147139x |
| 8 | 4075.648 | 798.720 | 5.102725x |
| 16 | 8169.216 | 839.872 | 9.726740x |

四个 B 的 combined 与 AB/BA 分区全部通过 1.10/1.05 门；两臂逐 seed 独立通过
FP64 oracle，max abs 范围 `4.5765e-5…6.2646e-5`，warp local memory 为 0。跨臂
bitwise 只作诊断，不参与 gate。首次 job 10843 的 GPU raw 数据本身四组均通过，但旧
runner 用 Python double 重算 C++ float32 偶数中位数，在 B=1 产生 `0.000125 us`
表示差并以 `FINAL_RC=1` 安全拒绝；它只作为
[失败审计证据](experiment_logs/c2_cluster_attention_warp_batch_abba/failed_job10843)保留，
不贡献性能结论。最终 runner 按 float32 round-trip 严格复算并拒绝错误 JSON 类型，
从头重跑 job 10873 后 `FINAL_RC=0`。本结果仍只是 native correctness-prototype 的
batch ABI 与 producer-mapping 信号，不是吞吐、production、模型或服务器结果。

#### Native batch WMMA-QK：B=1/4/8/16 全部晋级

[job 10886 正式 JSON](experiment_logs/c2_cluster_attention_tc_qk_batch_abba/c2_cluster_attention_tc_qk_batch_abba_clean_20260828T224455Z_job10886.json)、
[Slurm 日志](experiment_logs/c2_cluster_attention_tc_qk_batch_abba/c2_tc_qk_batch_10886.slurm.log)与
[manifest](experiment_logs/c2_cluster_attention_tc_qk_batch_abba/c2_cluster_attention_tc_qk_batch_abba_job10886.sha256)
在上一小节的相同 native batch ABI 上，以 warp-QK 为 control、WMMA-QK 为 candidate：

| B | `t_{w,b}` (us) | `t_{TC,b}` (us) | `R_{TC,b}` |
| ---: | ---: | ---: | ---: |
| 1 | 774.112 | 330.175995 | 2.344543x |
| 4 | 775.167969 | 331.007996 | 2.341841x |
| 8 | 799.183960 | 338.304016 | 2.362325x |
| 16 | 829.375977 | 408.208008 | 2.031749x |

每个 B 都有 51 个 ABBA pair、每臂 102 个样本；combined 与 AB/BA 分区均过
1.10/1.05 门。两个 seed 和计时后 fresh check 在四个 B 上都通过独立 FP64 oracle、
finite、sentinel 与 poison/signature gate，最大绝对误差不超过 `6.27e-5`。control 的
PTX/SASS BF16 MMA/HMMA 为 0/0，candidate 为 64/128；资源从
32 regs/4,172 B shared/0 local 变为 43 regs/16,480 B shared/0 local。这里量到的是
完整 batch candidate 成本，不是 production throughput，也不能与下面 B=1 的 PV
结果拼接后推断 batched PV。

#### WMMA-PV：B=1 完整 candidate 再快 1.451x

[job 10935 正式 JSON](experiment_logs/c2_cluster_attention_tc_qk_pv_abba/c2_tc_qk_pv_abba_clean_20260828T232328Z_job10935.json)、
[Slurm 日志](experiment_logs/c2_cluster_attention_tc_qk_pv_abba/c2_tc_pv_10935.slurm.log)与
[manifest](experiment_logs/c2_cluster_attention_tc_qk_pv_abba/c2_cluster_attention_tc_qk_pv_abba_job10935.sha256)
把 B=1 WMMA-QK candidate 固定为 control，只把后半段 warp PV 换成 BF16 WMMA PV；
两臂都保留 FP32 online-softmax 状态、DSM/mbarrier、merge 与 caller output ABI。
candidate 不写 global score/weight workspace、不启动第二个 kernel，也没有 scalar residual
correction；新增 shared state 与 CTA barrier 全部计入计时：

| arm | p10 / median / p90 (us) | regs / shared / local |
| --- | ---: | ---: |
| WMMA-QK / warp-PV control | 324.832001 / **326.960007** / 328.927979 | 40 / 16480 B / 0 B |
| WMMA-QK / WMMA-PV candidate | 223.903992 / **225.344009** / 226.528000 | 56 / 33632 B / 0 B |

因此 `R_PV=1.45093720x`，AB/BA 分区为 1.45061023x/1.45292945x；101 个
ABBA pair、每臂 202 个样本和 20 次 warmup 全部通过。seed 17/2026 及计时后 fresh
check 的两臂都独立通过 FP64 oracle、finite 与 sentinel gate；candidate 最大绝对误差
分别为 `1.97470188e-4` 和 `1.39700249e-4`。跨臂 BF16 不逐 bit 相同，最大差
`2.44140625e-4`，这里只作诊断，不替代逐臂 oracle。静态门禁从 control 的 64 条 PTX
WMMA / 128 条 SASS HMMA 增至 candidate 的 72/144，且两臂 mbarrier 与 cluster
指令计数保持合同。

同一冻结源码/runner 又完整重跑 job 10938 与 10939；二者 `FINAL_RC=0`、全部正确性和
promotion gate 通过，`R_PV` 分别为 1.45234372x 与 1.45290527x。证据在
[repeat 10938](experiment_logs/c2_cluster_attention_tc_qk_pv_abba/repeat_job10938/c2_tc_qk_pv_abba_clean_20260828T232946Z_job10938.json)及其
[manifest](experiment_logs/c2_cluster_attention_tc_qk_pv_abba/repeat_job10938/c2_cluster_attention_tc_qk_pv_abba_job10938.sha256)，以及
[repeat 10939](experiment_logs/c2_cluster_attention_tc_qk_pv_abba/repeat_job10939/c2_tc_qk_pv_abba_clean_20260828T232956Z_job10939.json)及其
[manifest](experiment_logs/c2_cluster_attention_tc_qk_pv_abba/repeat_job10939/c2_cluster_attention_tc_qk_pv_abba_job10939.sha256)。三次都落在
`GPU-7787…`，所以只支持**同设备重复性**。双 GPU allocation 被
[`QOSMaxGRESPerUser`](experiment_logs/c2_tc_pv_cross_gpu_probe/qos_two_gpu_request.log)拒绝；
合法的 `--gpu-bind=map_gpu:1` 探针仍由 Slurm 分配 `SLURM_JOB_GPUS=0` 和相同 UUID，见
[job 10941 log](experiment_logs/c2_tc_pv_cross_gpu_probe/map1_10941.slurm.log)。因此本轮没有
把串行同卡复验冒充第二 UUID 外部复现。机器可读边界和三次 job 的对应关系见
[external-validity summary](experiment_logs/c2_tc_pv_cross_gpu_probe/c2_tc_pv_external_validity_summary_20260828.json)
与 [manifest](experiment_logs/c2_tc_pv_cross_gpu_probe/c2_tc_pv_external_validity_probe.sha256)。

job 10921 的 GPU payload 虽产生正向 raw 数字，但旧 runner 对 C++ 九位精度 tolerance
字段做 Python exact-equality，最终 `FINAL_RC=1`；它只在
[failed audit 目录](experiment_logs/c2_cluster_attention_tc_qk_pv_abba/failed_job10921)保留，
不贡献上述性能结论。受限修复只允许 `1e-9` 的十进制传输误差并加入反例，之后从头
重跑 job 10935 才形成正式证据。本节边界仍是 native B=1、C=2 完整实现成本，不是
batched PV、production、模型或服务结果。

#### Prepared B=1、C=2 只作支撑性可行锚点

[job 10796 摘要](experiment_logs/c2_prepared_b1_c2_viability_anchor/c2_prepared_b1_c2_viability_anchor_job10796_summary.json)、
[Slurm 日志](experiment_logs/c2_prepared_b1_c2_viability_anchor/c2_prep_c2_anchor_10796.slurm.log)与
[manifest](experiment_logs/c2_prepared_b1_c2_viability_anchor/c2_prepared_b1_c2_viability_anchor_job10796.sha256)
记录 prepared BF16、B=1、`C=2` 的 p10/median/p90 为 36.928/37.984/42.272 us，
max abs `3.0517578e-5`。它证明 prepared 路径在该配置可运行，但输入、布局、实现和
launch 结构都与 native cluster 原型不同；不得把 761.088/37.984 或
326.592/37.984 写成受控 speedup。

### Production-native AOT / wheel：从 ABI 缺口到实验闭环（2026-08-29）

本节使用的性能变量先统一如下；所有比值均来自 CUDA-event clean run，NCU replay 时间不参与
scoreboard：

| 变量 | 含义 | 本节取值 / 下标说明 |
| --- | --- | --- |
| `t_v1` | v1 production-native 内核 median latency | job 11456，`0.293471992 ms` |
| `t_v2` | warp-owned-PV v2 production-native 内核 median latency | job 11468，`0.241536006 ms` |
| `t_{tri,2}` | 与 v2 native 共享输入语义的 Triton median latency | job 11468，`0.029440001 ms`；下标 `2` 为 v2 对照轮 |
| `I_{v2←v1}` | v2 相对 v1 的改善比例，`1-t_v2/t_v1` | `0.176970843`，即 `17.6971%` |
| `L_{v2/tri}` | v2 native 相对 Triton 的 latency ratio，`t_v2/t_{tri,2}` | `8.204347859x`；大于 1 表示 native 更慢 |
| `t^g_{v2},t^g_{v3}` | 上标 `g` 表示 job 11489 direct-stress gate；下标为版本 | `0.241510321/0.242685441 ms` |
| `I^g_{v3←v2}` | v3 相对同门 v2 的改善比例，`1-t^g_{v3}/t^g_{v2}` | `-0.004865717`，负值表示 v3 慢 `0.4866%` |
| `t^g_{v2,4},t^g_{v4}` | 上标 `g` 表示 job 11497 direct gate；下标 `2,4` 表示该轮 v2 reference / v4 candidate | `0.241588961/0.153796961 ms` |
| `I^g_{v4←v2}` | v4 direct gate 相对 v2 的改善，`1-t^g_{v4}/t^g_{v2,4}` | `0.363394088`，即 `36.3394%` |
| `t_v4,t_{tri,4}` | fresh backend 中 v4 / 同输入 Triton median | job 11503，`0.153408006/0.029440001 ms` |
| `I_{v4←v2}` | v4 fresh backend 相对 job 11468 v2 的改善，`1-t_v4/t_v2` | `0.364864857`，即 `36.4865%` |
| `L_{v4/tri}` | v4 native 相对 Triton 的 latency ratio，`t_v4/t_{tri,4}` | `5.210869648x`；仍大于 1 |
| `t^g_{v4,5},t^g_{v5}` | 上标 `g` 表示 job 11762 同 GPU direct ABBA；下标 `4,5` 为冻结 v4 / v5 | `0.153928400/0.116552159 ms` |
| `I^g_{v5←v4}` | v5 在同 GPU direct gate 相对 v4 的改善，`1-t^g_{v5}/t^g_{v4,5}` | `0.242815757`，即 `24.2816%` |
| `t_v5,t_{tri,5}` | job 11775 fresh backend 中 v5 / 同输入 Triton median | `0.117183998/0.029184001 ms`；v4 数字来自冻结 job 11503，不是同 job A/B |
| `I_{v5←v4}` | v5 fresh observation 相对冻结 job 11503 v4 的改善，`1-t_v5/t_v4` | `0.236128535`，即 `23.6129%` |
| `L_{v5/tri}` | v5 native 相对 Triton 的 latency ratio，`t_v5/t_{tri,5}` | `4.015350738x`；仍大于 1 |
| `t^g_{v5,6},t^g_{v6}` | 上标 `g` 表示 job 12314 同 GPU、独立进程 `A-B-B-A` direct gate；下标 `5,6` 为冻结 v5 / v6 | `0.116681599/0.107585361 ms` |
| `I^g_{v6←v5}` | v6 相对同门 v5 的 paired 改善，`1-t^g_{v6}/t^g_{v5,6}` | 点估计 `0.077957779`，即 `7.7958%`；10,000 次 paired bootstrap 的 95% LCB 为 `7.7492%` |
| `t_v6,t_{tri,6}` | job 12385 的 8 个 fresh-process seed median 再取中位数 | `0.108112000/0.029296000 ms` |
| `L_{v6/tri}` | v6 native 相对同输入 Triton 的 8 个逐-seed ratio 的中位数；不是 `t_v6/t_{tri,6}` | `3.685659651x`；仍大于 1，不满足 parity |
| `t^g_{v6,7},t^g_{v7}` | 上标 `g` 表示 job 12513 同一物理 GPU、四个隔离进程 `v6-v7-v7-v6` direct gate；下标 `6,7` 为冻结 v6 / v7 | `0.107525440/0.102334282 ms` |
| `I^g_{v7←v6}` | v7 相对同门 v6 的改善，`1-t^g_{v7}/t^g_{v6,7}` | 点估计 `0.048278421`，即 `4.8278%`；bootstrap 95% LCB 为 `4.7563%`，但点估计未过预设 5% 门 |
| `t^g_{v6,8},t^g_{v8}` | 上标 `g` 表示 job 12557 同一物理 GPU、四个隔离进程 `v6-v8-v8-v6` direct gate；下标 `6,8` 为冻结 v6 / v8 | `0.107806359/0.115448919 ms` |
| `I^g_{v8←v6}` | v8 相对同门 v6 的改善，`1-t^g_{v8}/t^g_{v6,8}` | `-0.070891555`，即慢 `7.0892%`；bootstrap 95% LCB 为 `-7.3237%`，因此拒绝 |
| `w,ℓ,u,h` | `w` 为 warp，`ℓ` 为 lane，`u=ℓ mod 16` 为 subgroup lane，`h=2w+⌊ℓ/16⌋` 为该 subgroup 唯一负责的 GQA head | `w∈[0,7]`、`ℓ∈[0,31]`、`u∈[0,15]`、`h∈[0,15]` |
| `h_6,p_6,i,d_6` | v6 的 register numerator 映射：`h_6=ℓ mod 16`，`p_6=⌊ℓ/16⌋`，`d_6=16w+2i+p_6`；下标 `6` 区分 v5 的 head ownership | `w,i∈[0,7]`、`ℓ∈[0,31]`；恰好覆盖 16 heads × 128 dims，无碰撞或遗漏 |
| `t,ℓ,i,e` | v7 中 `t` 为 V token tile，`ℓ` 为 lane，`i` 为 lane-private FP8 预取槽，`e=ℓ+32i` 为 16×16 tile 的展平元素 | `t,i∈[0,7]`、`ℓ∈[0,31]`；`token=⌊e/16⌋`、`dim=e mod 16` |
| `V^{fp8}_{t,w,ℓ,i}` | v7 的 warp `w`、lane `ℓ` 在 softmax 前预取并跨 softmax 保留的原始 FP8 V 元素 | 只移动 load 时机；BF16 转换、shared stage、TC-PV 与 accumulator 语义不变 |
| `t_8,h,j,s_{buf}` | v8 中 `t_8` 为 token tile，`h` 为 GQA head，`j` 为 16-lane token subgroup 下标，`s_{buf}=t_8 mod 2` 为 metadata 双缓冲槽 | `t_8∈[0,7]`、`h,j∈[0,15]`；源码变量名为 `metadata_slot`，与全局 attention scale `s` 不同 |
| `W^{(8)}_{s_{buf},h,j},A^{(8)}_{s_{buf},h},X^{(8)}_{s_{buf},h}` | v8 双槽 `weights`、`alpha_tile`、`tile_active`；上标 `(8)` 表示版本而非幂 | softmax 写当前槽，PV/update 只读同一槽；另一个槽可被下一 tile 写入 |
| `M^{(8)}_{t_8},B^{(8)}_{t_8}` | `M` 是页内 tile `t_8` 的三组 metadata 逻辑状态，`B` 是其 softmax 写后保留的 CTA publication barrier | 页内 `t_8<7` 时，`B_{t_8+1}` 证明所有 `M_{t_8}` 读取已结束；`t_8=7` 时由下一页 QK/score CTA barrier 提供 quiescence，之后同槽才可覆盖 |
| `R,P,p=P/R` | `R` 为 producer CTA 数，`P` 为 selected pages，`p` 为每 producer 页数 | v2 为 `R=2,p=8`；v4 为 `R=4,p=4`；`P=16` 不变 |
| `N_i,Z_i` | 下标 `i` 表示第 `i` 个 producer；`N_i` 为未归一化 numerator，`Z_i` 为 softmax normalizer | 四路合并保持 `Σ_i N_i/Σ_i Z_i` |
| `B,Q_len` | batch 与单请求 query 长度 | `16,1` |
| `H_q,H_kv,D` | query heads、KV heads、head dimension | `64,4,128` |
| `P,K` | page size 与 sparse top-k | `128,16` |

#### 结论先行：跨 GPU 复现不是当前最高优先级

上一版报告把 production 接入列为第一缺口是正确的；本轮已经把这个缺口推进到**实验性
派生 wheel 的 backend-layer 与单进程生命周期闭环**。精确 d4 checkout 现可独立 AOT 出
原生 plugin，派生 wheel 保持稳定 ABI DSO 逐字节不变，新鲜安装后由固定 loader 注册第三
分支，并在真实 `MiniMaxM3SparseMSAImpl.forward` 内完成 oracle、caller-owned output、
dispatcher、CUDA kernel、8-seed ABBA、并发 loader、1000 次调用和 CUDA Graph replay 门。

最新晋升的 v6 在 job 12314 的同 GPU direct `A-B-B-A` 中比 v5 快 **7.7958%**，又在
job 12385 把 fresh native/Triton ratio 从 v5 的 `4.0154x` 收窄到 `3.6857x`，但仍未具备
性能 parity。本轮又把两个最小假设分别推到裁决：v7 隐藏 V-load latency 的收益稳定但只到
`4.8278%`，v8 以 metadata 双缓冲删去 tile-tail CTA barrier 后反而慢 `7.0892%`。这两条
阴性结果说明剩余主导项不能仅靠这两种局部 overlap/barrier 改写消除，也把短期最高优先级
从“继续盲试候选”转为完成证据闭环、保留 v6，并要求下一性能候选先给出新的机制证据与
单变量假设。第二 UUID 复现会提高外部有效性，却不会消除这个主导性能阻塞；job
12213/12214 的冻结 v5 replay 落在同一 UUID，两 GPU 请求又被 `QOSMaxGRESPerUser` 拒绝，
所以严格跨 GPU 复现仍重要但不占最高优先级。

这里的“production 闭环”有严格边界：它是 exact d4 源码上的独立 AOT plugin、实验性
overlay wheel、新鲜 target install、真实 backend layer 与受控单进程生命周期；**不是**
上游正式发布 wheel、完整模型权重、长期 scheduler/HTTP service、多进程 server、
TTFT/throughput 或任意形状通用性结论。

#### v1：先证明真实 AOT、wheel 与 dispatcher 可以闭环

job 11443 从 commit `d4da0c55af3aa231b6209bf77871f3ed36eab0d2` 独立构建
`_native_c2_msa_decode_plugin.abi3.so`（623,992 B，SHA-256
`f464b2b2…478bf8d`）。编译记录只含一个 `sm_100` cubin、无 PTX，并由 SASS 证明存在
BF16 HMMA；随后它在 compute capability 10.3 的 B300 上真实加载和执行。证据见
[AOT final status](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-aot-artifacts-20260829/job11443/final-status-job11443.txt)、
[compile command](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-aot-artifacts-20260829/job11443/native-c2-compile-command-job11443.json)、
[code object](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-aot-artifacts-20260829/job11443/plugin-cuobjdump-elf-job11443.txt)与
[provenance](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-aot-artifacts-20260829/job11443/plugin-provenance-job11443.sha256)。

取证格式边界：jobs 11443/11464/11487/11496 的 `native-c2-compile-command-*.json` 是从
`compile_commands.json` 原样摘出的单个对象内部三行，缺少外围花括号，因而应按 **JSON
fragment** 而非严格 JSON 解析；为保持历史原始证据字节不变，本轮不重写这些文件。完整命令还
可与相应 configure/compile/build-driver log 交叉核验；后续采集脚本应改用 `jq` 输出完整对象。

job 11449 从原始 d4 wheel 生成 v1 overlay wheel（SHA-256 `42048803…cfb6060`）：

- 只新增 plugin DSO 与 `msa_native_c2_decode.py` adapter；
- 只替换四个批准的 Python dispatch member，不删除成员，也没有未批准变化；
- `vllm/_C_stable_libtorch.abi3.so` 在 baseline/derived 中同为
  `cee888ed…00442`，逐字节一致；
- derived `RECORD` 对所有 payload 与 self row 做全量验证。

对应的 [wheel provenance](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-wheel-artifacts-20260829/job11449/c2-native-plugin-overlay-provenance.json)、
[manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-wheel-artifacts-20260829/job11449/c2-native-plugin-overlay-manifest.json)和
[final status](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-wheel-artifacts-20260829/job11449/final-status-job11449.txt)
把这组成员级断言固定下来。

job 11456 再把 wheel 安装到全新的 target directory。baseline 与 installed tree 在等价的
正常 vLLM imports 后取 dispatcher snapshot；loader 第一次只新增
`_C::native_c2_msa_decode` 且有 CUDA kernel，第二次调用 snapshot 完全不变，所有已有 op
name/schema/dispatch surface 都保留。真实 backend harness 的边界为
`B=16,Q_len=1,H_q/H_kv/D=64/4/128,P=128,K=16`、packed E4M3 KV、scalar q/k/v
scale 与 caller-owned BF16 output；native 最大绝对误差为 `6.103515625e-5`，profiler 恰好
记录一次 dispatcher event 和一次 native CUDA kernel。完整结果见
[job 11456 JSON](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-wheel-runtime-artifacts-20260829/job11456/plugin-wheel-full-backend.json)与
[post-loader surface](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-wheel-runtime-artifacts-20260829/job11456/installed-post-plugin-ops.json)。

v1 的集成正确，但 clean ABBA 得到 native/Triton median 为
`0.293471992/0.029279999 ms`，即 native 慢 `10.022950740x`。因此不能把“可加载”误写成
“性能完成”，也正因如此没有立刻把预算转去跨 GPU。

#### v2 warp-owned PV：两级 5% 门禁均通过

v1 NCU 已显示 shared-memory 访问冲突、long-scoreboard stall 与 barrier/occupancy 压力；
本轮据此只改性能数据路径的 PV ownership/layout，冻结其余 production ABI；另外把
cluster launch 的错误捕获改为直接接收 `cudaLaunchKernelEx` 返回码，不改变成功路径数学。
v2 patch SHA-256 为
`2fa34736…2816`，派生源码 SHA-256 为 `f2aae9c9…4acc4`。job 11464 AOT
`FINAL_RC=0`，DSO 为 628,040 B、SHA-256 `73d769ca…6debb5`，仍只有一个 `sm_100`
cubin、无 PTX。见 [v2 AOT status](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-v2-aot-artifacts-20260829/job11464/final-status-job11464.txt)、
[code object](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-v2-aot-artifacts-20260829/job11464/plugin-cuobjdump-elf-job11464.txt)与
[provenance](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-v2-aot-artifacts-20260829/job11464/plugin-v2-provenance-job11464.sha256)。

第一层 promotion 是 job 11465 的 direct-op stress。它不是在一个 Python 进程内热切库，
而是按 `v1_a → v2_b1 → v2_b2 → v1_a2` 启动四个隔离进程；四臂共享 8 个固定 seed、
30 次 warmup、200 次 iteration、4 次 bitwise stability repeat 与 FP64 oracle。每臂 8 个
seed mean 合并为每版本 16 个 seed mean；v1/v2 中位数为
`0.293656718/0.241527361 ms`，改善 `17.7518%`，超过预先冻结的 5% 门。所有 seed 的
allclose、finite、caller pointer 与重复性门通过，见
[direct-stress decision](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-v2-stress-artifacts-20260829/job11465/v2-vs-v1-direct-stress-decision-job11465.json)。

只有这层 `accepted=true` 后才运行 job 11467。v2 wheel SHA-256 为
`d6dae6d9…eafc388`，provenance 为 `9de9682d…43173c`；它从**原始 d4 baseline wheel**
重新派生，而不是 overlay-on-overlay，且再次通过 stable DSO、member set 与 RECORD 全量
门。证据见 [v2 wheel provenance](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-v2-wheel-artifacts-20260829/job11467/c2-native-plugin-overlay-provenance.json)。

第二层 promotion 是 job 11468 的 fresh-install/full-backend 复验：

| arm | p10 / median / p90 (ms) | 相对结论 |
| --- | ---: | --- |
| v1 native，job 11456 | 0.292448014 / **0.293471992** / 0.295424014 | 冻结 reference |
| v2 native，job 11468 | 0.241375998 / **0.241536006** / 0.243328005 | 比 v1 快 **17.6971%**，通过 5% 门 |
| Triton，job 11468 | 0.028543999 / **0.029440001** / 0.030495999 | v2 native 仍慢 **8.204347859x** |

job 11468 同时重复通过 fresh install、stable DSO、等价 pre-loader surface、幂等 loader、
唯一新增 op、FP32 oracle、caller output、one-dispatcher/one-kernel profiler 与相同 query
ABBA 门；native 最大/平均绝对误差为 `6.103515625e-5/3.529688456e-6`。最终
[promotion decision](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-v2-wheel-runtime-artifacts-20260829/job11468/v2-vs-v1-decision-job11468.json)、
[full-backend JSON](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-v2-wheel-runtime-artifacts-20260829/job11468/plugin-v2-wheel-full-backend-job11468.json)、
[post-loader surface](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-v2-wheel-runtime-artifacts-20260829/job11468/installed-post-plugin-ops-job11468.json)与
[chrome trace](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-v2-wheel-runtime-artifacts-20260829/job11468/plugin-v2-wheel-full-backend-job11468.chrome.json)
共同限制这一结论。

随后 job 11470 对已晋升的同一 v2 DSO 做了与 v1 job 11301 完全同指标的单 kernel NCU。
这组 replay 时间只作机制诊断：v2 的 shared-load wavefront 与 bank conflict 分别下降
`51.64%/62.32%`，register/thread 从 68 降到 62，`launch__occupancy_cluster_pct` 从 `8.78%` 升到
`11.99%`；tensor-pipe instruction 仍为 262,144，DRAM read 与 L2 bytes 基本不变。
这支持“warp-owned PV 的收益主要来自 shared-load/资源压力改善”，不支持“做了更多 Tensor
Core 工作”的说法。下一项诊断假设也更具体：shared-store conflict 上升 `12.42%`，
long-scoreboard stall 指标上升 `36.62%`；但 aggregate NCU 不能把它们定位到单一源码行。
因此可以先做 memory-dependency/shared-store layout 的单变量 A/B，再优化同一 production
ABI 内已经存在的 batched WMMA/TC-PV；不能拿 NCU 的 instrumented duration 代替前述 clean ABBA。原始
[v2 NCU summary](experiment_logs/c2_native_c2_production_evidence/c2-native-v2-ncu-artifacts-20260829/native-c2-ncu-job11470.json)、
[report](experiment_logs/c2_native_c2_production_evidence/c2-native-v2-ncu-artifacts-20260829/native-c2-job11470.ncu-rep)与
[final status](experiment_logs/c2_native_c2_production_evidence/c2-native-v2-ncu-artifacts-20260829/final-status-job11470.txt)
均已归档。

#### v3 shared-accumulator padding：正确性通过，但性能门明确否决

job 11470 的 aggregate NCU 只能提出 shared-store/dependency 假设，不能证明具体源码行是
根因。本轮因此做了一个严格单变量诊断：把 v2 的
`pv_contribution[kWarps][kTile][16]` 最后一维 stride 从 16 pad 到 20，并把匹配的 WMMA
accumulator store leading dimension 从 16 改到 20；共享内存增加 2,048 B，数学、索引语义、
production ABI 与其余数据路径不变。补丁 SHA-256 为 `12fa59d5…b1a22`。

首个 AOT job 11481 在编译前因 GNU patch hunk 不匹配而 `FINAL_RC=1`；它只说明打包脚本有误，
不贡献 CUDA 正确性或性能结论。改成两个精确单行 hunk 后，job 11487 从冻结 v2 源码重新构建并
`FINAL_RC=0`：派生源码 SHA-256 为 `3e01404e…5eeff2`，DSO 为 628,024 B、SHA-256
`5711637b…1d41a`，仍只有一个 `sm_100` cubin 且无 PTX。

job 11489 随后按 `v2_a → v3_b1 → v3_b2 → v2_a2` 启动四个隔离进程，并复用 8 个 seed、
30 次 warmup、200 次 iteration、4 次 bitwise stability repeat、FP64 oracle、caller pointer
与 GPU 污染 fail-closed 门。所有正确性和可比性门都通过，但 16 个 seed mean 的中位数为
`0.241510321/0.242685441 ms`；`I^g_{v3←v2}=-0.004865717`，即 v3 反而慢 `0.4866%`，而
5% 晋升要求 v3 不超过 `0.229434805 ms`。因此 `accepted=false`，job 以预设的拒绝码
`FINAL_RC=3` 结束；**当时不构建 v3 wheel、不做 v3 NCU，并继续保留 v2；后续 v4 已另行晋升**。这项阴性证据只否定
“stride 16→20 padding”在冻结合同下的收益，不外推为所有 WMMA/TC-PV layout 都无效。

原始 [promotion decision](experiment_logs/c2_native_c2_production_evidence/c2-native-v3-padded-pv-evidence-20260829/c2-native-plugin-v3-stress-artifacts-20260829/job11489/v3-vs-v2-direct-stress-decision-job11489.json)、
[AOT status](experiment_logs/c2_native_c2_production_evidence/c2-native-v3-padded-pv-evidence-20260829/c2-native-plugin-v3-aot-artifacts-20260829/job11487/final-status-job11487.txt)、
[patch](experiment_logs/c2_native_c2_production_evidence/c2-native-v3-padded-pv-evidence-20260829/native_c2_v3_padded_pv_20260829.patch)与
[完整归档](experiment_logs/c2_native_c2_production_evidence/c2-native-v3-padded-pv-complete-evidence-20260829.tar.gz)
已固定。

#### v4 four-producer：两级性能门、定向 liveness 与 production wheel 均通过

v2 的四 CTA cluster 只有 rank 0/1 各处理 8 个 selected pages，rank 2 合并而 rank 3 只参与
同步。v4 保持 `clusterDim=4`、shared arrays、QK、现有 batched WMMA/TC-PV、packed-E4M3
ABI 与 caller-owned output 不变，只把 producer partition/merge arity 从 2 改为 4：四个 rank
各处理连续 4 页，rank 0 在自身 producer 完成后等待四个 arrival，再以四路 log-sum-exp
权重合并本地与 rank 1/2/3 DSM partial。补丁 SHA-256 为 `4547dc1b…befab17`，数学上合并
仍是 `ΣN_i/ΣZ_i`；主要数值风险来自 8+8 改成 4+4+4+4 后的 BF16 partial rounding，因此
不能只靠代数证明晋升。

job 11496 从 exact d4 和已晋升 v2 重新独立 AOT，`FINAL_RC=0`；派生源码 SHA-256
`96c0af14…7f4c9`，DSO 为 615,784 B、SHA-256 `e0087708…3763d5`，仍只有一个
`sm_100` cubin、无 PTX。job 11497 随后按 `v2_a → v4_b1 → v4_b2 → v2_a2` 做相同的
四进程、8-seed、FP64、warmup/iteration/stability 与 GPU 污染门；所有正确性/可比性门通过，
`t^g_{v2,4}/t^g_{v4}=0.241588961/0.153796961 ms`，提升 `36.3394%`，显著越过 5% 门。

由于此候选改变 cluster 内 producer/liveness 协议，job 11498 另做四个 rank-directed case 与
all-invalid case。rank 0/1/2/3 各自占主导时，oracle 与预期常数的 `max_abs` 均为 0；
all-invalid 输出有限且全零，所有 caller pointer、bitwise repeat 与逐次同步 watchdog 门通过。

job 11499 才据此从原始 d4 baseline wheel 生成 v4 overlay wheel（SHA-256
`ef8c54ef…7f374ff`；provenance SHA-256 `940f8d7b…24a3a0a`；manifest SHA-256
`2abd0e92…be6deaf`）。稳定 ABI DSO 继续逐字节保持 `cee888ed…00442`，只新增 plugin/adapter，
无删除、无未批准 member 变化，RECORD 全量门通过。首个 fresh-runtime job 11500 的 kernel、
oracle、trace 与 timing 实际全过，但后置验证器把 harness 固定 schema `…backend-v2` 机械写成
`…backend-v4`，因此 `FINAL_RC=1`；这项脚本失败不进入 scoreboard。只修复两个 schema
断言后，job 11503 从 fresh target 全流程重跑并 `FINAL_RC=0`：

| arm | p10 / median / p90 (ms) | 相对结论 |
| --- | ---: | --- |
| v2 native，job 11468 | 0.241375998 / **0.241536006** / 0.243328005 | 冻结 production reference |
| v4 native，job 11503 | 0.153311998 / **0.153408006** / 0.153504000 | 比 v2 快 **36.4865%**，通过 5% 门 |
| Triton，job 11503 | 0.028511999 / **0.029440001** / 0.029600000 | v4 native 仍慢 **5.210869648x** |

job 11503 同时通过 stable member、fresh install、等价 pre-loader surface、幂等 loader、唯一新增
op、caller output 与 one-dispatcher/one-kernel trace 门；native 最大/平均绝对误差为
`3.0517578125e-5/3.533571089e-6`。

最后，job 11506 对同一 v4 DSO 收取与 v2 job 11470 相同的单 kernel NCU 指标，只作机制诊断。
instrumented duration 下降 `35.52%`，与 clean ABBA 方向一致；tensor instruction 仍为
262,144，DRAM read/L2 bytes 只变化 `-1.668%/-0.637%`。`launch__registers_per_thread`
从 62 降到 58，`launch__occupancy_cluster_pct` 保持 `11.99%`，barrier stall 指标下降
`59.15%`、tensor-active 指标上升 `55.70%`，支持收益来自 producer partition/cluster 利用，
而不是增加 Tensor 工作或外部流量。shared load/store wavefront 与 conflict 反而上升约
`3.3–9.6%`，long-scoreboard/wait 指标上升 `17.92%/8.12%`；aggregate NCU 不能把这些变化
定位到单一源码行，但说明下一候选应聚焦剩余 memory dependency、staging 与 softmax/numerator，
不能继续扫描 padding。

关键证据见 [direct decision](experiment_logs/c2_native_c2_production_evidence/c2-native-v4-four-producer-evidence-20260829/c2-native-plugin-v4-stress-artifacts-20260829/job11497/v4-vs-v2-direct-stress-decision-job11497.json)、
[directed result](experiment_logs/c2_native_c2_production_evidence/c2-native-v4-four-producer-evidence-20260829/c2-native-plugin-v4-directed-artifacts-20260829/job11498/four-producer-directed-job11498.json)、
[wheel provenance](experiment_logs/c2_native_c2_production_evidence/c2-native-v4-four-producer-evidence-20260829/c2-native-plugin-v4-wheel-artifacts-20260829/job11499/c2-native-plugin-overlay-provenance.json)、
[fresh-backend result](experiment_logs/c2_native_c2_production_evidence/c2-native-v4-four-producer-evidence-20260829/c2-native-plugin-v4-wheel-runtime-artifacts-20260829/job11503/plugin-v4-wheel-full-backend-job11503.json)、
[promotion decision](experiment_logs/c2_native_c2_production_evidence/c2-native-v4-four-producer-evidence-20260829/c2-native-plugin-v4-wheel-runtime-artifacts-20260829/job11503/v4-vs-v2-decision-job11503.json)与
[v4 NCU](experiment_logs/c2_native_c2_production_evidence/c2-native-v4-four-producer-evidence-20260829/c2-native-v4-ncu-artifacts-20260829/native-c2-ncu-job11506.json)。

#### v5 warp-parallel softmax：同 GPU 强阳性并完成 production 晋升

v4 每个 16-token tile 的 softmax/weight stage 只让 warp 0 的 16 lanes 工作，并让 lane
对应 head、每个 lane 串行遍历 16 tokens；其余 7 个 warp 在 CTA barrier 前等待。v5 只改变
这一 ownership：8 个 warp 各负责两个 16-lane subgroup，subgroup 内 lane 对应 token，分别
写唯一的 `weights[head][token]`。QK、WMMA/TC-PV、四 producer、DSM、mbarrier、shared
allocation 与 ABI 全部不变。补丁 SHA-256 为 `98d8a458…f8710`。

job 11760 从 exact v4 source 重新 AOT，派生 source/DSO SHA-256 为
`6409a546…0f0931` / `9ab755fa…1d2538`，仍只有 `sm_100` cubin、无 PTX；资源为
`REG=64, STACK=0, SHARED=38752, LOCAL=0`。job 11762 随后在同一分配的 B300 上按
`v4_a → v5_b1 → v5_b2 → v4_a2` 运行四个独立进程、8 seeds、FP64 oracle、30 warmup、
200 iterations 与 4 次稳定性复验。所有正确性、finite、caller pointer、return-none、逐 bit
稳定性和清卡门通过：v4/v5 的 16 个 seed-mean 中位数为
`0.153928400/0.116552159 ms`，v5 提升 **24.2816%**，越过预设 5% 门。

softmax 专门门保留了完整的失败—诊断—验收链。初版 job 11764 的四个 producer 与
all-invalid case 全过，但新 V 编码的最终 BF16 舍入使两个 softmax case 超过冻结的
`atol=1e-4, rtol=1e-3`。diagnostic-only job 11766 在独立 v4/v5 进程中得到完全相同的
max/mean error，证明不是 v5 退化。v2 harness 只把 token/head/dim 的 V code 缩小 `1/32`，
不改变 Q/K logits 或控制流；job 11769 仍显示两臂误差完全相同。真正验收 job 11771
随后 `FINAL_RC=0`：rank 0–3、all-invalid、2048-token online-rescale 与 37-token（尾 subgroup
仅 5 个 valid lanes）全部通过，两个 softmax case 最大绝对误差为
`3.57855e-5/6.04120e-5`。

job 11773 生成 v5 overlay wheel（SHA-256 `af221b3c…531f49`；provenance
`5836d6b5…e21a94`；manifest `feee75e5…b385eb`），stable DSO 仍逐字节等于
`cee888ed…00442`，只新增 plugin/adapter，RECORD 全量门通过。job 11775 从 fresh target
安装后通过等价 pre-loader surface、幂等 loader、唯一新增 op、真实
`MiniMaxM3SparseMSAImpl.forward`、caller output 与 one-dispatcher/one-kernel trace：

| arm | p10 / median / p90 (ms) | 相对结论 |
| --- | ---: | --- |
| 冻结 v4 native，job 11503 | 0.153311998 / **0.153408006** / 0.153504000 | frozen reference；与 11775 不同 UUID |
| v5 native，job 11775 | 0.115263999 / **0.117183998** / 0.117376000 | frozen observation 比 v4 快 **23.6129%** |
| Triton，job 11775 | 0.029088000 / **0.029184001** / 0.029344000 | v5 native 仍慢 **4.015350738x** |

这里严格区分证据强度：job 11775 的 v4 数字来自先前冻结 job 11503，二者恰在不同 B300
UUID，因此是 cross-job/cross-UUID observation，不是同 job ABBA；晋升的同 GPU 性能主证据
是 job 11762。两层结果 `24.28%/23.61%` 方向和量级一致。

job 11779 最后复用 v4 job 11506 的 seed、单 kernel filter 与同 14 个 NCU counters，只作
机制解释。tensor instruction 固定 262,144，DRAM read/L2 bytes 仅为 v4 的
`0.99949/0.99373`，cluster occupancy 与 shared bytes 仍为 `11.99%/38752`；寄存器从
58 增至 64，但 residency 未变。与此同时 shared load/store bank conflicts 下降
`37.42%/27.50%`，load/store wavefront 下降 `24.18%/4.80%`，barrier、long-scoreboard、
wait stall 分别下降 `48.43%/15.98%/15.46%`。这与 ownership 假设及 clean ABBA 一致，
但 NCU replay 时间仍不进入 scoreboard。

关键证据见 [direct decision](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-warp-parallel-softmax-evidence-20260829/c2-native-plugin-v5-stress-artifacts-20260829/job11762/v5-vs-v4-direct-stress-decision-job11762.json)、
[directed result](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-warp-parallel-softmax-evidence-20260829/c2-native-plugin-v5-directed-artifacts-20260829/job11771/v5-softmax-directed-job11771.json)、
[wheel provenance](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-warp-parallel-softmax-evidence-20260829/c2-native-plugin-v5-wheel-artifacts-20260829/job11773/c2-native-plugin-overlay-provenance.json)、
[fresh result](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-warp-parallel-softmax-evidence-20260829/c2-native-plugin-v5-wheel-runtime-artifacts-20260829/job11775/plugin-v5-wheel-full-backend-job11775.json)、
[promotion decision](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-warp-parallel-softmax-evidence-20260829/c2-native-plugin-v5-wheel-runtime-artifacts-20260829/job11775/v5-vs-v4-decision-job11775.json)与
[NCU comparison](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-warp-parallel-softmax-evidence-20260829/c2-native-plugin-v5-ncu-artifacts-20260829/job11779/native-c2-v5-v4-mechanism-job11779.json)。

#### v5 生命周期补门：测试范围内关闭 loader/workspace/CUDA Graph

性能晋升之后，job 12295 对冻结 v5 wheel 做了新的 fresh-target 生命周期门。八个线程先在
`threading.Barrier(8)` 同步，再同时调用真实 adapter loader；对真正的
`torch.ops.load_library` 加入会释放 GIL 的 0.1 s 计数延迟后，八个线程全部返回 `True`、无
异常且底层加载调用**恰好一次**。首次加载只新增 `_C::native_c2_msa_decode` 这一 op/schema，
全部既有 op、schema 和完整 dispatch table 不变；生命周期结束后的快照仍与首次加载后逐项一致。

同一 fresh 进程中的运行门如下：

| 门 | job 12295 的硬证据 |
| --- | --- |
| 真实 backend forward | profiler 恰好 1 个 dispatcher + 1 个 native CUDA kernel；caller-owned output pointer、BF16 bitwise repeat 与 FP32 oracle 全过 |
| 稳态 | 1000 次真实 `MiniMaxM3SparseMSAImpl.forward`，每次 pointer/bitwise/oracle 通过 |
| CUDA Graph | 原始 static query 100 次 replay；随后原地修改同一 query storage 再 replay 100 次，输出随输入改变且分别匹配两个 oracle |
| sequence boundary | `2048/2049/4095/4096` 均覆盖最后一个可见 page；每个“少 1 token”错误 oracle 都被拒绝 |
| 动态拒绝 | 8 个不支持 case 均只返回 native selection `False`，没有修改 output，也没有 dispatcher/kernel event；不声称实际执行了 Triton fallback |
| 内存 | synchronized PyTorch allocated/reserved live+peak 与 `cudaMemGetInfo` delta 均在 8 MiB 硬界内 |

因此旧表中的“CUDA Graph 与 loader/workspace 生命周期未测”已经过时；准确边界是：**受控单
进程、单 GPU、固定 contract 的生命周期已关闭**。这些计数器不是 CUDA memory sanitizer，
也没有覆盖多进程加载、长期 server/service、完整模型权重或 scheduler。最终
[lifecycle JSON](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-lifecycle-evidence-20260830/c2-native-plugin-v5-lifecycle-artifacts-20260830/job12295/native-c2-v5-lifecycle-job12295.json)、
[final status](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-lifecycle-evidence-20260830/c2-native-plugin-v5-lifecycle-artifacts-20260830/job12295/final-status-job12295.txt)与
[frozen support manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-lifecycle-evidence-20260830/c2-native-plugin-v5-lifecycle-artifacts-20260830/job12295/reviewed-c2-source-job12295.sha256)
共同固定了结果和五个 oracle/Triton 支持文件。

#### v6 register-resident numerator：7.80% 同 GPU 强阳性并完成实验性 production 晋升

v6 只移除 numerator 的 shared-memory round trip：每个 lane 以
`h_6=ℓ mod 16`、`p_6=⌊ℓ/16⌋`、`d_6=16w+2i+p_6` 保留自身负责的
`(head,dim)` numerator 寄存器值，再直接参与后续写回；softmax、四 producer、TC-PV、
DSM/mbarrier、ABI 与输入合同不变。该映射对 16 heads × 128 dims 是精确双射。补丁
SHA-256 为 `1e0b1740…5b8fd5`；job 12278 从 exact v5 source 构建 source/DSO SHA-256
`c7f05928…117f02` / `0d5492be…0e21a4`，仍只有单一 `sm_100` cubin、无 PTX，资源为
`REG=64, STACK=0, SHARED=30560, LOCAL=0`。

job 12314 在同一 B300 分配、同一冻结 8-seed fixture 上以四个隔离 Python 进程执行
`v5_a → v6_b1 → v6_b2 → v5_a2`，避免两个 DSO 对同一 schema 重复注册。所有 FP64、finite、
caller pointer、return-none、4 次 bitwise repeat、输入 manifest 与单 UUID 门通过：

| arm | 8-seed paired median (ms) | 结论 |
| --- | ---: | --- |
| v5 reference | **0.116681599** | job 11760 frozen DSO |
| v6 candidate | **0.107585361** | 相对 v5 改善 **7.7958%** |

10,000 次 deterministic paired-seed bootstrap 给出的改善 95% LCB 为 `+7.7492%`。预先
冻结的晋升门是“点估计至少 5% 且该 LCB 至少 0”，两项均通过。这里的 LCB 只描述这 8 个
固定 seed 观测的重采样稳定性，不外推未采样 workload/GPU，也不声明总体参数的 95% 统计推断。

结构门 job 12322 又精确覆盖 64 个 head offsets × 128 个 dimension codes、4 个 producer
ranks、2048-token online rescale、37-token tail 与 all-invalid：全部 BF16/FP64 gate 通过，
且 profiler 恰好一个 dispatcher/一个 kernel。job 12331 随后生成 overlay wheel（SHA-256
`3d3205de…9724f`；provenance `2f463afe…d0e21`）：stable DSO 仍为
`cee888ed…00442`，4742 个未修改 baseline payload 逐字节一致，wheel RECORD 的 4748 个
payload 全量验证通过。

job 12385 从全新 target 安装这一 wheel。它分别验证原 wheel RECORD 和安装器重写后的
RECORD：除 RECORD 自身外 4748 个原 payload 字节一致，installed RECORD 精确为 4754 个
成员，只增加 `bin/vllm` 与四个审计过的 `dist-info` 文件；另一个未记录普通文件只能是
`.lock`。支持包不再 `cp -a`，而是只安装五个固定 SHA 的 `.py` 文件。八个 seed 各启一个
fresh Python，并全部经过真实 backend、FP32 oracle、caller output、one-dispatcher/
one-kernel 和同进程 Triton ABBA：

| job 12385 seed-median 聚合 | latency (ms) | 边界 |
| --- | ---: | --- |
| v6 native | **0.108112000** | 8 个 seed median 再取中位数 |
| Triton | **0.029296000** | 同 seed、同进程 ABBA |
| native / Triton | **3.685659651x** | 8 个逐-seed ratio 的中位数，不是相邻两行中位数之商；parity 未达到且不是失败硬门 |

匹配 seed `20260829` 的 v6 fresh median 为 `0.107136004 ms`，相对冻结 job 11775 的 v5
`0.117183998 ms` 方向性改善 `8.5745%`；这是 cross-job fresh observation，不替代 job
12314 的同 GPU paired promotion 主证据。

最后，job 12396 在同一物理 UUID 上分别用独立进程重新采集 v5/v6 的单 native kernel。
shared memory 从 `38752` 降到 `30560` B（`-21.14%`），shared load/store wavefront ratio
为 `0.7912/0.8418`，load/store bank-conflict ratio 为 `0.9189/0.9920`，cycles ratio
`0.9347`；wait stall 降到 `0.9163`，但 barrier/long-scoreboard 反而升到
`1.1290/1.1309`。这说明 shared numerator 流量确实被移除，同时指出下一步仍应盯住
barrier/scoreboard 与 K/V staging。所有 NCU duration/counter **只作机制证据**，不参与性能
晋升或拒绝。

关键证据见 [AOT status](experiment_logs/c2_native_c2_production_evidence/c2-native-v6-register-numerator-evidence-20260830/c2-native-plugin-v6-aot-artifacts-20260830/job12278/final-status-job12278.txt)、
[direct aggregate](experiment_logs/c2_native_c2_production_evidence/c2-native-v6-register-numerator-evidence-20260830/c2-native-plugin-v6-stress-artifacts-20260830/job12314/v6-vs-v5-fixed-fixture-aggregate-job12314.json)、
[promotion decision](experiment_logs/c2_native_c2_production_evidence/c2-native-v6-register-numerator-evidence-20260830/c2-native-plugin-v6-stress-artifacts-20260830/job12314/v6-vs-v5-promotion-decision-job12314.json)、
[directed result](experiment_logs/c2_native_c2_production_evidence/c2-native-v6-register-numerator-evidence-20260830/c2-native-plugin-v6-directed-artifacts-20260830/job12322/v6-register-numerator-directed-job12322.json)、
[wheel provenance](experiment_logs/c2_native_c2_production_evidence/c2-native-v6-register-numerator-evidence-20260830/job12331/c2-native-plugin-overlay-provenance.json)、
[fresh summary](experiment_logs/c2_native_c2_production_evidence/c2-native-v6-register-numerator-evidence-20260830/c2-native-plugin-v6-fresh-backend-artifacts-20260830/job12385/plugin-v6-fresh-backend-summary-job12385.json)与
[NCU mechanism](experiment_logs/c2_native_c2_production_evidence/c2-native-v6-register-numerator-evidence-20260830/c2-native-plugin-v6-ncu-artifacts-20260830/job12396/v6-v5-mechanism-job12396.json)。

#### v7 raw-FP8 V prefetch：正确性成立，但 4.83% 未过 5% 性能门

v7 只改变 V load 的发起时机：每个 lane 在 softmax 前读取 8 个原始
`__nv_fp8_e4m3` 元素，并把它们跨 softmax 保存在寄存器中；softmax、FP8→BF16 转换、
`fp8_stage`、发布 barrier、TC-PV、register numerator、DSM/mbarrier、ABI 与输入合同均保持
不变。补丁 SHA-256 为 `b0c02002…5c66ee`。AOT job 12484 的 source/DSO SHA-256 分别为
`1d638ce6…12e9a3` / `0873af3c…e97db`；DSO 仍只有一个 `sm_100` cubin、无 PTX，资源为
`REG=72, STACK=0, SHARED=30560, LOCAL=0`。因此这轮可以隔离回答“把 V scoreboard
等待藏在 softmax 后面是否足以越过晋升门”，而不是把多个布局或 producer 改动混在一起。

fresh directed job 12501 在 B300 capability 10.3 上通过 4 个 producer rank、2048-token
online rescale、37-token causal tail、all-invalid、64 个 head offsets × 128 个 dimension
codes、FP64/FP32、caller-owned output、4 次 bitwise repeat，以及恰好一个 dispatcher / 一个
native kernel。结果 JSON 仍使用冻结 harness 的 schema
`c2-native-c2-v6-register-numerator-directed-v1`；这表示复用了已审计的 v6 定向合同，不表示
运行了 v6 DSO。candidate path/SHA 与 job12484 v7 DSO 均被脚本和结果固定。

job 12513 随后在同一 UUID
`GPU-dadf9f3b-df58-d3fa-07b0-5fe223423db1` 上，以同一 8-seed CPU fixture 和四个隔离
Python 进程执行 `v6_a → v7_b1 → v7_b2 → v6_a2`。每个 seed 每臂 warmup 30、计时 200、
稳定性重复 4 次；四轮 worker exit status、FP64、finite、pointer、return-none、bitwise、精确
profiler、输入复验与 PRE/POST/FINAL_POST 清卡门全部通过：

| seed | v6 reference (ms) | v7 candidate (ms) | paired improvement |
| ---: | ---: | ---: | ---: |
| 17 | 0.107607600 | 0.102317442 | 4.916157% |
| 23 | 0.107416240 | 0.102331522 | 4.733659% |
| 42 | 0.107663280 | 0.102335442 | 4.948612% |
| 2024 | 0.107468881 | 0.102333122 | 4.778833% |
| 314159 | 0.107458320 | 0.102361282 | 4.743270% |
| 20260801 | 0.107554800 | 0.102359122 | 4.830727% |
| 20260815 | 0.107529680 | 0.102314082 | 4.850381% |
| 20260829 | 0.107521200 | 0.102448402 | 4.717952% |

8-seed paired median 为 `0.107525440/0.102334282 ms`，点估计改善 **4.827842%**；
10,000 次 deterministic paired-seed bootstrap 的 95% LCB 为 **4.756299%**。8 个已观察
seed 的 paired improvement 均为正且固定-seed 重采样下界为正，但这既不替代预先冻结的
“点估计至少 5%”条件，也不外推未采样 workload/GPU。因此 promotion decision 为
`accepted=false`，job 以 `FINAL_RC=3, ORIGINAL_RC=3, FINALIZER_ERROR=0` 正常结束：这是
**正确且可比候选的纯性能拒绝**，不是 correctness、资源或脚本失败。按停止条件不构建 v7
wheel、不做 v7 fresh-backend/NCU，继续保留 v6；也不靠重复运行或改门槛把 4.83% 包装成通过。

关键证据见 [AOT status](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v-prefetch-evidence-20260830/c2-native-plugin-v7-aot-artifacts-20260830/job12484/final-status-job12484.txt)、
[directed result](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-directed-evidence-20260830/c2-native-plugin-v7-directed-artifacts-20260830/job12501/v7-v-prefetch-directed-job12501.json)、
[ABBA aggregate](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v-prefetch-rejection-evidence-20260830/lean-extracted/c2-native-plugin-v7-stress-artifacts-20260830/job12513/v7-v-prefetch-vs-v6-fixed-fixture-aggregate-job12513.json)与
[promotion decision](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v-prefetch-rejection-evidence-20260830/lean-extracted/c2-native-plugin-v7-stress-artifacts-20260830/job12513/v7-v-prefetch-vs-v6-promotion-decision-job12513.json)。AOT、directed、stress lean tar SHA-256 分别为
`90fdac35…c2a92`、`3ad71066…b4679f`、`9d3c869e…205068`；完整 stress 归档留在 B300
`/home/lcpu/85117379/c2-native-v7-stress-job12513-evidence.tar.gz`，SHA-256 为
`103a86a80b9d7f64890644a8347d3a28f3f8d315a56a0dee2b11df97cbd4e7fb`，其完整递归 manifest
SHA-256 为 `dbba9a7b19448a9a147f0fc4e353cfef2fc6c8027d49187a7b527cf4bf54f507`；本地 lean archive
明确省略 68 MB fixed fixture 与 pycache，但保留完整 manifest、fixture metadata 和所有裁决产物。

#### v8 softmax metadata 双缓冲：安全性成立，但同卡慢 7.09%

v8 从冻结 v6 单独派生，不叠加已拒绝的 v7。唯一概念变化是把 `weights`、`alpha_tile`、
`tile_active` 扩成两个 shared-memory 槽，以 `s_buf=t_8 mod 2` 选择当前槽；softmax 后的
publication barrier `B^{(8)}_{t_8}` 保留，而每个 token tile 尾部的 CTA barrier 被删除。
这把 shared allocation 从 30,560 增加到 31,200 B（多 640 B），不改变 ABI、producer
partition、QK/TC-PV、register numerator、DSM/mbarrier 或输入合同。补丁/结果源码 SHA-256
分别为 `40c7d180…04b53` / `9c4317c3…b04f6a1`。

安全性依赖上表的槽生命周期，而不是 timing 假设：tile `t_8` 的全部 softmax store 先完成，
`B^{(8)}_{t_8}` 再把 `M^{(8)}_{t_8}` 发布给所有 PV warp；每个 warp 读完该槽并完成 update
后才会进入下一轮。页内 `t_8<7` 时，下一轮的 `B^{(8)}_{t_8+1}` 必须等所有线程到达，因此
任一线程推进到 `t_8+2`、覆盖同一个 `s_buf` 前，`M^{(8)}_{t_8}` 的全部读取已经结束。
页尾 `t_8=7` 时没有页内 `B_8`：下一页 QK/score 完成后的 CTA barrier 先等齐所有上一页
tile-7 读者，下一页 tile-0 的 publication barrier 又先于 tile-1 覆盖 slot 1，因而跨页同样
闭合。不同槽允许 softmax(`t_8+1`) 与 PV/update(`t_8`) 重叠；同槽不会读写重叠。warp-private `fp8_stage[warp]`
和 `pv_contribution[warp]` 不需要跨 warp 发布，`running_max/normalizer` 则在保留 barrier 前
更新并由该 barrier 继续保护。

AOT job 12534 从精确 v6 source + fuzz-0 patch 重建；DSO SHA-256 为
`2ecfe9b0…b97c8c`，只有一个 `sm_100` cubin、无 PTX，资源为
`REG=56, STACK=0, SHARED=31200, LOCAL=0`。fresh directed job 12548 随后通过四个 producer
rank、2048-token online rescale、37-token causal tail、all-invalid、64×128 head/dimension
编码、FP64/FP32、caller pointer、4 次 bitwise repeat 与恰好一个 dispatcher/一个 native
kernel；其结果 schema 仍是冻结 v6 directed harness 的
`c2-native-c2-v6-register-numerator-directed-v1`，但候选 path/SHA 明确指向 job12534 v8 DSO。

正式 job 12557 在同一 UUID
`GPU-0ff6ec41-3275-1f3d-41d8-8b17413a205a` 上，以同一 8-seed CPU fixture 和四个隔离
Python 进程执行 `v6_a → v8_b1 → v8_b2 → v6_a2`；每臂 warmup 30、计时 200、稳定性重复
4 次。四轮 worker、FP64/finite/pointer/return-none/bitwise、精确 profiler、资源/DSO/source
身份、输入复验和 PRE/POST/FINAL_POST 清卡全部通过，`issues=[]`：

| seed | v6 reference (ms) | v8 candidate (ms) | paired improvement |
| ---: | ---: | ---: | ---: |
| 17 | 0.107870639 | 0.115556159 | -7.124757% |
| 23 | 0.107733199 | 0.115412959 | -7.128499% |
| 42 | 0.107939119 | 0.115763439 | -7.248828% |
| 2024 | 0.107643279 | 0.115186719 | -7.007813% |
| 314159 | 0.107795759 | 0.115363839 | -7.020759% |
| 20260801 | 0.107856959 | 0.116095359 | -7.638265% |
| 20260815 | 0.107757119 | 0.115373519 | -7.068118% |
| 20260829 | 0.107816959 | 0.115484879 | -7.111980% |

paired median 为 `0.107806359/0.115448919 ms`，改善为 **-7.089155%**；10,000 次
deterministic paired bootstrap 的 95% LCB 为 **-7.323718%**。因此
`accepted=false`，job 以 `FINAL_RC=3, ORIGINAL_RC=3, FINALIZER_ERROR=0` 正常完成：这同样是
所有硬门通过后的纯性能拒绝，而非 correctness 或脚本失败。按预设停止条件不构建 v8
wheel、不做 v8 fresh-backend/NCU，继续保留 v6。

关键证据见 [AOT status](experiment_logs/c2_native_c2_production_evidence/c2-native-v8-double-buffer-metadata-evidence-20260830/aot-extracted/c2-native-plugin-v8-aot-artifacts-20260830/job12534/final-status-job12534.txt)、
[directed result](experiment_logs/c2_native_c2_production_evidence/c2-native-v8-double-buffer-metadata-evidence-20260830/directed-extracted/c2-native-plugin-v8-directed-artifacts-20260830/job12548/v8-double-buffer-metadata-directed-job12548.json)、
[ABBA aggregate](experiment_logs/c2_native_c2_production_evidence/c2-native-v8-double-buffer-metadata-evidence-20260830/stress-lean-extracted/c2-native-plugin-v8-stress-artifacts-20260830/job12557/v8-double-buffer-metadata-vs-v6-fixed-fixture-aggregate-job12557.json)与
[promotion decision](experiment_logs/c2_native_c2_production_evidence/c2-native-v8-double-buffer-metadata-evidence-20260830/stress-lean-extracted/c2-native-plugin-v8-stress-artifacts-20260830/job12557/v8-double-buffer-metadata-vs-v6-promotion-decision-job12557.json)。AOT、directed、stress lean tar SHA-256 分别为
`5a8cb103…a7912`、`59457ee2…b1e1`、`d20a8eb0…0df0`。完整 stress 归档保留在 B300
`/home/lcpu/85117379/c2-native-v8-stress-job12557-evidence.tar.gz`，SHA-256 为
`ec3a43c798d16291baab2bd3a851e51cb8f9b33fb51336993f9d6dd85482af77`；其完整递归 manifest
文件的 SHA-256 为 `d0c9f9907dcb54e21b3207e54f39c208b3f64921884e7212063def32cb61a741`。

#### 新 `>3%` 续轮：v7 重新确认、v9 归并并行化、v11 Q fragment 复用

这一节使用用户在本轮单独给出的严格 `>3%` 标准。它是新的实验问题，不能把 job 12513
在预先冻结 5% 门下的合法拒绝追溯改写成“当时已通过”。本轮冻结判定是：全部硬门先通过，
paired-median 点改善严格大于 `3%`，且固定 8-seed 的 deterministic paired-bootstrap 95% LCB
大于 `0`；没有额外要求 LCB 也大于 `3%`。变量先列如下：

| 变量 | 含义 | 范围/定义 |
| --- | --- | --- |
| `h,p,d` | `h` 为 KV 组内 query head，`p` 为 producer CTA rank，`d` 为输出维 | `h∈[0,15]`、`p∈[0,3]`、`d∈[0,127]` |
| `e=128h+d` | v9 merge CTA 中展平的输出元素 | `e∈[0,2047]`；256 个线程按 stride 256 分片 |
| `L_{p,h}` / `P_{p,h,d}` | producer `p` 的局部 base-2 LSE / BF16 partial numerator | 四个 producer 的归并输入 |
| `w_{h,p}` / `D_h` | `w_{h,p}=2^(L_{p,h}-max_q L_{q,h})`，`D_h=sum_p w_{h,p}` | v9 每个 head 只计算一次并在 CTA 内发布 |
| `I^{g,3}_{v7←v6}` | 新 3% 门下、同一物理 GPU 的 v7 相对 v6 改善 | job 12599 的点估计 `4.933390%` |
| `I^{g,3}_{v9←v7}` | 同一物理 GPU、隔离进程 ABBA 中 v9 相对 v7 改善 | job 12776 的点估计 `28.101799%` |
| `I^{g,3}_{v11←v9}` | 同一物理 GPU、隔离进程 ABBA 中 v11 相对 v9 的改善 | job 12905 的点估计 `12.814628%` |
| `LCB^{g,3}_{x←y}` | 对同一比较的 8 个固定 seed 做 10,000 次 deterministic paired bootstrap 的改善 95% 下界 | 接受门为 `LCB^{g,3}_{x←y}>0`；只描述该固定 fixture 的重采样稳定性 |

job 12599 使用冻结 job12484 v7 DSO 与冻结 v6 DSO，按 `v6→v7→v7→v6`、8 个固定
seed、每 seed 30 次 warmup/200 次计时重放。所有 correctness、稳定性、caller pointer、
dispatcher/native-kernel、fixture、环境和单 GPU UUID 门均通过；paired median 改善
**4.933390%**，10,000 次固定 seed paired bootstrap 的 95% LCB 为 **4.717097%**。
因此它通过本轮严格 `>3%` 门；该结论与 job 12513 的旧 5% 拒绝并存，互不改写。

v9 从冻结 v7 只改变最终四路归并。旧实现让 merge CTA 的 16 个线程各串行写 128 维；新实现
先由 16 个线程为每个 head 计算四个 `w_{h,p}` 与 `D_h`，经 CTA barrier 发布，再由全部
256 个线程分片写 2,048 个 `e`。AOT job 12701 的 DSO/source SHA-256 为
`a98a7bee…c592` / `9956b6b6…806ed`，资源为 `REG=64, STACK=0, SHARED=30880,
LOCAL=0`。job 12767 的四个 producer-rank、两个 softmax、all-invalid、head-dimension encoding、
bitwise repeat 和精确 profiler 事件全部通过。第一次 stress job 12772 在计时前因提交了
“内容相同但不是 provenance 固定路径”的源码副本而 `RC=2`；它没有性能数据，证据保留且不
参与裁决。改用 provenance 中精确源码路径后，job 12776 在同一 UUID 内完成
`v7→v9→v9→v7`：所有硬门通过，paired median 改善 **28.101799%**，bootstrap 95% LCB
**27.922466%**，显著越过 `>3%`。v9 因而是本轮新的 direct-plugin 性能基线；这不是
wheel/fresh-backend/full-model 晋升声明。

K-lookahead v10 也完成了可审计止损。AOT job 12702 得到 `REG=92, STACK=0,
SHARED=46944, LOCAL=0`，directed job 12783 全过。stress job 12784 的四个 worker 中，
两个 reference 与第二个 candidate 全过；第一个 candidate 的 seed 42 数值、bitwise repeat、
输出指针和 dispatcher event 都通过，但 profiler 偶发漏采 native-kernel event，故整体按硬门
`RC=2`，没有合法改善点估计。两轮 candidate 的每 seed mean 均约 `0.194 ms`，只能作为
方向性诊断，不能包装成正式 ABBA 裁决；相对约 `0.10 ms` 的 v7 已足以停止 raw-FP8
lookahead。BF16 预转换还会把预期 shared 推至 63,328 B，因此未继续占用 GPU。

v9b 是同一“最终归并”家族的备选分布式实现，而不是第三条独立收益声明：AOT job 12790 与
directed job 12794 成功；stress job 12795 却在 preflight 发现启动器缺失 required directed-sidecar
environment，未进入计时且没有性能数据，故不参与任何 ABBA 或优先级判断。

第三个独立方向 v11 以 accepted v9 为唯一基线。每个 producer CTA 的 Q 只由输入决定、在四个
selected page 上不变；v11 在 page loop 前装入八个 WMMA `matrix_a` Q fragments，随后以
warp-uniform 选择复用，因而减少重复 Q shared-to-WMMA load，而不改变 K/V、online softmax、
barrier、partial、四路 merge 或 ABI。hardened AOT job 12825 成功，v11 DSO/source SHA-256 为
`501b9c…d957` / `0e82b278b7aa44a034a96d2ddd19946a27928d95a9f9d03e8e1fe9f30680c5b4`，资源为
`REG=127, STACK=0, SHARED=30880, LOCAL=0`。首次 directed job 12829 检出提交 provenance
脚本相互矛盾，按 fail-closed 停止；它不提供正确性或性能结论。修复后，job 12904 的 source、DSO、
资源、四 producer-rank、两种 softmax、all-invalid、head/dimension encoding、bitwise repeat、
caller pointer、dispatcher/native-kernel event、GPU identity 与清卡门全部通过。

job 12905 随后在同一 UUID `GPU-778768b4-6c9e-e483-890e-0812760948ae` 执行
`v9→v11→v11→v9`。所有 correctness、身份、资源、provenance、fixture、事件与可比性硬门通过，
paired median improvement 为 **12.814628%**（原始比例 `0.12814628042666487`），10,000 次
deterministic paired bootstrap 的 95% LCB 为 **12.782150%**（原始比例
`0.12782150442086837`），`accepted=true`，aggregate SHA-256 为
`85b9489cc36155a18c239729f4fcea7b746c67c7c3bed8b10944c73ddd7635b9`。故 v11 是第三个严格
超过 `>3%` 的方向。v7 的 `+4.933390%`、v9 的 `+28.101799%` 和 v11 的 `+12.814628%` 分别
来自相邻冻结版本的 paired ABBA；不得把三段百分比相乘、相加或外推为 Triton parity 或 full-model
speedup。

#### v11 overlay-wheel、fresh backend、NCU 与生命周期晋级

这一段不产生新的性能接受率。它把 job12905 已接受的 direct-plugin DSO 放入安装、真实 backend、
机制与生命周期边界。新增记号先列如下：

| 变量 | 含义 | 本轮观察 |
| --- | --- | --- |
| `R` | 每线程寄存器数 | v9/v11 为 `64/127` |
| `S` | kernel static shared memory | v9/v11 均为 `30880 B` |
| `W_ld` | NCU shared-load wavefront 计数 | `W_ld(v11)/W_ld(v9)=0.653017` |
| `C_ld` | NCU shared-load bank-conflict 总事件 | `C_ld(v11)/C_ld(v9)=0.475892` |
| `L_sb` | long-scoreboard stall 计数 | `L_sb(v11)/L_sb(v9)=0.683004` |

job12957 的 overlay packager 先固定 job12825/12904/12905 的 AOT、directed 与接受 decision，再从
相同 distribution version 生成 wheel `278c01a9…1e7f34`。baseline/derived RECORD 分别验证
`4746/4748` 个 payload；`4742` 个既有 baseline payload 保持字节不变，仅新增 plugin 与 adapter，
四个既有 Python dispatch 文件是预先允许的替换项，稳定 DSO `cee888ed…00442` 仍逐字节相同。
[wheel verification](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-wheel-artifacts-20260830/job12957/wheel-driver-verification-job12957.json)
与 [promotion prerequisites](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-wheel-artifacts-20260830/job12957/promotion-prerequisites-job12957.json)
均由最终 manifest 固定。job12956 只证明首版在输入/身份门 fail-closed；现有归档不足以支持更
具体的失败原因，因此不对它作细化推断。

job12960 从一个新 `uv --target` 安装根起，对 8 个 seed 分别启动 fresh Python 并运行真实
`MiniMaxM3SparseMSAImpl.forward`；wheel、RECORD/install、loader surface、稳定 DSO、direct
evidence、8-seed backend 与清卡七组硬门全过。native 与 Triton 的 seed-median 中位数为
`0.1110240035/0.0476640016 ms`，逐 seed native/Triton ratio 中位数 `2.3306479`。这既没有
Triton parity，也没有 v9 fresh-wheel comparator；故 [fresh decision](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-fresh-backend-artifacts-20260830/job12960/plugin-v11-fresh-backend-decision-job12960.json)
只接受安装与 backend integration，不重算 v11-v9 性能。

job12965 在同一输入 manifest、同一 B300 UUID、两个隔离进程和每臂恰好一个 native kernel 下
采集 NCU。`W_ld`、`C_ld` 与 `L_sb` 分别下降 `34.70%`、`52.41%`、`31.70%`，tensor 指令数
保持 `1.0x`；shared-store wavefront/conflict 基本不变并略增，且 `R:64→127`、理论 cluster
occupancy 有代价。[mechanism comparison](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-vs-v9-ncu-artifacts-20260830/job12965/v11-v9-mechanism-job12965.json)
与 Q-fragment reuse 假设一致，但 NCU duration/cycle 不是性能证据；正式裁决仍只来自 job12905。
jobs12961/12963 是 JSON parser 的 fail-closed 诊断，只有 job12965 是有效 NCU closure。

job12977 最后从同一冻结 wheel 做 production lifecycle：8 个同时起跑的 loader 只调用一次真实
`torch.ops.load_library`，随后真实 forward、1000 次 bitwise-repeat 稳态调用、CUDA Graph
原 query 与原地更新 query 各 100 次 replay、native 选择/拒绝、`2048/2049/4095/4096` 精确
序列边界、caller-owned output、dispatcher/kernel event、有界 PyTorch/CUDA memory counter 与
PRE/POST 清卡全部通过。[lifecycle result](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-lifecycle-artifacts-20260830/job12977/native-c2-v11-lifecycle-job12977.json)
明确不覆盖 full model weights、scheduler/server、多进程加载或 CUDA memory sanitizer。job12976
因提交时预期 digest 不匹配，在 artifact 创建前停止且只有空 Slurm log；它没有测试结果，job12977
才是有效生命周期证据。

#### 追加方向 v12：Q shared row padding `128→136`

v11 晋级闭环后，profile 中剩余最直接的可归因假设是：Q fragment 已不再跨页重复装入，但
`128` 个 BF16 的 shared 行距仍让相邻行回到相同 bank phase。计算与判定前先固定记号：

| 变量 | 含义 | v11 / v12 或门槛 |
| --- | --- | --- |
| `S_Q` | read-only Q shared tile 的 leading dimension，单位 BF16 元素 | `128 / 136` |
| `R` | 每线程寄存器数 | `127 / 127` |
| `S` | static shared memory | `30880 / 31136 B` |
| `I_{v12←v11}` | 同卡 8-seed paired-median 中 v12 相对 v11 的改善 | 接受门严格 `>3%` |
| `LCB_{v12←v11}` | 10,000 次 deterministic paired-seed bootstrap 的 95% 下界 | 接受门严格 `>0` |

`S_Q=128` 时行距为 `256 B=64` 个 32-bit bank word；`S_Q=136` 时为
`272 B=68` 个 word，行间 bank phase 旋转 4。v12 只新增 `kQTileStride=136`，把 Q tile
第二维与 8 个 Q `wmma::load_matrix_sync` 的 `ldm` 改为该 stride；有效 Q 写入仍只有
`dim∈[0,127]`。K/V、PV、online softmax、barrier/mbarrier、四路 merge、浮点顺序和 ABI
全部不变，shared 预期只增加 `16×8×2=256 B`。这与早先改变 PV 输出/消费布局且增加约
2 KiB shared 的 v3 padding 不是同一个机制。

两个未执行的提交不计作实验：job12979 在分配前的静态复核发现 AOT 脚本漏声明
`V11_PATCH` 后取消；修正后的 job12980 因 CPU 队列预计在本轮四小时窗口后才启动，也在
allocation 前取消。把相同 AOT 逻辑迁到 GPU partition 后，job12983 从空 root 完成，得到
source/plugin SHA `535d90b8…8aaf` / `064f967a…d7b6`，资源精确为
`REG=127, STACK=0, SHARED=31136, LOCAL=0`，且只有一个 `sm_100` cubin、无 PTX。job12984
随后通过冻结 full oracle、两条 softmax、all-invalid、四个 producer rank、head/dim encoding、
bitwise repeat、caller pointer、精确 profiler event、身份/资源与清卡门。

唯一正式性能 job12985 以同一 B300 UUID
`GPU-dadf9f3b-df58-d3fa-07b0-5fe223423db1` 和四个隔离 Python 进程执行
`v11_A→v12_B1→v12_B2→v11_A2`；每 seed 30 次 warmup、200 次单调用 CUDA-event，8 个
已冻结 seed 的均值如下：

| seed | v11 reference (ms) | v12 candidate (ms) | paired improvement |
| ---: | ---: | ---: | ---: |
| 17 | 0.065893119 | 0.063495599 | 3.638499% |
| 23 | 0.065912720 | 0.063443759 | 3.745803% |
| 42 | 0.065956400 | 0.063699519 | 3.421777% |
| 2024 | 0.065965920 | 0.063456479 | 3.804147% |
| 314159 | 0.065978640 | 0.063612719 | 3.585889% |
| 20260801 | 0.065994320 | 0.063687279 | 3.495818% |
| 20260815 | 0.065915439 | 0.063693919 | 3.370258% |
| 20260829 | 0.065997681 | 0.063433679 | 3.884987% |

所有 identity/correctness/resource/comparability 门通过且 `issues=[]`；paired median 为
`0.065961160/0.063554159 ms`，故 `I_{v12←v11}=3.649118%`，bootstrap
`LCB_{v12←v11}=3.405307%`。两项分别严格越过 `>3%` 与 `>0`，job12985
`FINAL_RC=0, accepted=true`。因此 v12 是在用户原三条显著方向之外、沿当前最高优先级继续得到的
第四个单变量显著候选；它的 [decision](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-q-row-padding-evidence-20260830/c2-native-plugin-v12-stress-3pct-artifacts-20260830/job12985/v12-q-row-padding-vs-v11-3pct-incremental-decision-job12985.json)
已与后续 NCU 和 wheel 证据一起落地。这里的 `3.649118%` 仍只描述该同卡 direct-plugin
fixture，不能与 v7/v9/v11 百分比组合或外推到 Triton/full-model/service。

job12986 在另一张 clean B300 `GPU-3924…bb0a` 上以同一 harness、input manifest、seed、metric
集合及分离 Python 进程各采一个 native kernel action。v12/v11 的 Q shared-load bank-conflict 为
`795732/1253744=0.634685`，shared-load wavefront 为 `2510900/2968912=0.845731`，long-scoreboard
为 `151.56%/155.66%=0.973661`；tensor 指令数不变，寄存器均为 `127`，shared 精确增加
`256 B`。store conflict/wavefront 分别微升至 `1.008852/1.002690`，所以该证据只支持“Q-load
侧冲突与波前下降，并伴随 long-scoreboard 下降”的有限机制解释。NCU duration/cycle 仍不进入
job12985 的性能裁决。

job12987 又构建并逐 RECORD 验证同 distribution-version v12 overlay wheel：baseline/derived
payload 分别为 `4746/4748`，`4742` 个未批准 baseline payload 保持不变；稳定 libtorch DSO
逐字节相同，只新增冻结 plugin 与 adapter，并替换四个已审查 Python dispatch 成员。wheel SHA
为 `b730e561…cc67b`。这是本地实验性 overlay wheel 身份闭环，不等于 upstream/release wheel。

job12991 把该 wheel 安装到节点临时盘，并用 8 个 fresh Python 进程运行真实
`MiniMaxM3SparseMSAImpl.forward`；RECORD/install、loader surface、稳定 DSO、plugin/adapter 身份、
job12985 direct-plugin correctness 与各 arm 内 bitwise-repeat 证据绑定、8/8 seed correctness、
PRE/POST clean GPU 及临时 runtime 清理全部通过。
native/Triton seed-median 的中位数为 `0.064600/0.029248 ms`，ratio 中位数 `2.209910`，仍未达到
parity；没有 v11 fresh-wheel comparator，所以这只是 fresh integration/latency observation，不能
替代 job12985。此前 job12988 在 isolated install 阶段因 home quota fail-closed，未触及 GPU；
job12990 虽跑出功能输出，却因只读 support 使严格 scratch cleanup 失败而整次排除。

v12 lifecycle 已由 job **13334** 闭环。此前 jobs12992/12993/12994/12996 仍保留为 fail-closed
审计；修复后只 allowlist 一个 `state/uv-cache/wheels-v6/url` symlink，并逐项记录其 link path、target、
canonical state-area 边界与 resolved target 目录 manifest。job13334 的 allowlist 项仍在同一受控
state area 内，其他审计项为空。随后 8 线程首加载恰好一次真实 `load_library`、真实 forward/dispatcher/native
kernel、1000 次稳态、Graph `100+100` replay 与 query mutation、支持谓词、动态拒绝、序列边界及
dispatcher persistence 八门全过，且 `FINAL_RC=0, ORIGINAL_RC=0, FINALIZER_ERROR=0`。因此 v12
在 v16 晋级前是完整通过本地受控 lifecycle 的 production candidate；该历史结论仍不覆盖 multi-process、
长期 server、full model、upstream/release 或 Triton parity。[job13334 lifecycle archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-lifecycle-closure-evidence-20260831/c2-v12-lifecycle-job13334-evidence-20260831.tar.gz)
固定了该成功链。

#### v13：distributed merge 的 AOT 成功与 directed 止损

v13 仅改变最终 merge 的 producer 归并布局。job **13428** AOT 成功，资源为
`REG=121, STACK=0, SHARED=30880, LOCAL=0`，并通过唯一 `sm_100` cubin、无 PTX、符号与
SASS 门。可是 job **13439** 在 directed 的首个 head-dim encoding 同步调用即报
`CUDA error: an illegal instruction was encountered`，`FINAL_RC=2`；4-rank/oracle 与任何性能
测量都未执行。按预声明 stop-loss，v13 stress 未运行，不能成为 v12 的性能比较或晋级依据。
[v13 stop-loss archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v13-distributed-merge-stoploss-evidence-20260831/c2-v13-distributed-merge-stoploss-evidence-20260831.tar.gz)
同时保留 AOT 成功与 directed 失败的边界。

#### v14：K/V stage stride `16→24` 的有效性能拒绝

计算前的变量见本文“变量与冻结形状”表。v14 只将 K/V staging leading stride
`S_{KV}:16→24`，存储/读取的 live 16 列、Q、softmax、barrier、merge 与 ABI 均不变。其静态
shared 增量为 `ΔR=W×T×(24−16)×e=8×16×8×2 B=2048 B`，所以冻结基线
`R_{14}=R_{12}+ΔR=31136+2048=33184 B`；AOT job **13487** 实测
`REG=118, STACK=0, SHARED=33184, LOCAL=0`，与资源门一致，job **13513** 的 directed
合同全过。首个 stress job **13518** 在预测量 harness preflight 因 argument-unpack 失败，未产生测量；
修复后从头重跑的 job **13539** 是唯一有效性能裁决，得到 `I_{v14←v12}=+2.40452555%`、
`LCB_{v14←v12}=+2.29466423%`。正确性、身份、资源和可比性门都通过，但点估计未严格超过
`3%`，故 `RC=3` 纯性能拒绝；不把该百分比与 v7/v9/v11/v12 相加或相乘，也不晋升 v14。v14
首轮 job **13518** 的具体边界是预测量 harness 的 argument-unpack 失败，不是性能/正确性
样本；job13539 才是唯一可用于该候选的完整重跑。v14/v15 共同归档已本地验证，见
[lean archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v14-v15-continuation-evidence-20260831/c2-native-v14-v15-continuation-lean-evidence-20260831.tar.gz)、
[lean manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v14-v15-continuation-evidence-20260831/c2-native-v14-v15-continuation-lean-evidence-20260831.manifest.sha256)、
[excluded-fixture list](experiment_logs/c2_native_c2_production_evidence/c2-native-v14-v15-continuation-evidence-20260831/c2-native-v14-v15-continuation-excluded-fixtures-20260831.sha256)、
[full manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v14-v15-continuation-evidence-20260831/c2-native-v14-v15-continuation-evidence-20260831.manifest.sha256)
及 [full sidecar](experiment_logs/c2_native_c2_production_evidence/c2-native-v14-v15-continuation-evidence-20260831/c2-native-v14-v15-continuation-evidence-20260831.tar.gz.sha256)。lean gzip SHA-256 为
`5ae30832ed5345b4d51cb69b378f472774d7792355a6e5ab93c107e7a9671846`，manifest 逐项通过 `161/161`，
tar 共有 162 个安全 regular members；它仅排除两份 fixture payload，二者 SHA 仍由 excluded-fixture
list/full manifest 固定。完整远端包 SHA-256 为
`0a5260556ad189ec0b0b9c405fa0b5f10aaa0f84d98ab4ab9fa0ca0d21301458`；本地验证的对象是 lean 包及其
full manifest/sidecar，而非该 111 MB 完整包。

#### v15：Q-stage stride `136→144` 的有效性能拒绝

v15 从已接受 v12 源仅改变 Q shared tile 的 leading stride `S_Q:136→144`，不叠加 v14；K/V
stage、live 列、softmax、barrier、merge 与 ABI 保持 v12 合同。AOT job **13564** 通过，资源为
`REG=127, STACK=0, SHARED=31392, LOCAL=0`；directed job **13575** 全门通过。job **13576**
在同卡、四隔离进程、8-seed `v12→v15→v15→v12` ABBA 中完成身份、正确性、资源和可比性门，得到
`I_{v15←v12}=-0.09713217%`、`LCB_{v15←v12}=-0.11928154%`。点估计既不大于 `3%`，方向也为负，
故按预声明规则以 `RC=3` 纯性能拒绝；不构建 v15 wheel/fresh/lifecycle，不改写 v12 基线。它与
v14/job13539 是相对同一 v12 的独立 pair，不能相加、连乘或用 NCU timing 重算。

#### v12/v14/v15 NCU：验证 v14 机制、否定 v15 机制，但不重算性能

先保留 job **13662** 的 fail-closed 审计：它在首个 snapshot 前因 Bash `set -u` 下同一 local 声明中
引用尚未赋值的 `tag` 而 `FAILED/1:0`，运行约 1 秒；没有 GPU snapshot、NCU、final-status、正确性或
性能数据。修复后的 profile SHA-256 为 `8314a34199fb61128dd0f52b354b808d495f56890f9c20730c9cf9ab1c237665`。
job **13666** 随即在 `GPU-0c223cf1-4325-822f-1e38-43ae57897edd` 观测为 `COMPLETED/0:0`、43 秒，
`FINAL_RC/ORIGINAL_RC/FINALIZER_ERROR/TEE_RC/RUNTIME_CLEANUP_RC/MANIFEST_RC` 全为 0。它以同 input
manifest、同 harness、15 个 counter、三个独立 Python 进程及每臂一个 logical native kernel action 采集
v12/v14/v15；`timing_valid_for_benchmark=false`，因此以下 `\rho` 仅为机制比率。

| `m` | `\rho_{14/12}^{m}`（相对变化） | `\rho_{15/12}^{m}`（相对变化） |
| --- | ---: | ---: |
| `C_{ld}` | `0.33780898`（`-66.2191%`） | `1.08235542`（`+8.2355%`） |
| `W_{ld}` | `0.79008057`（`-20.9919%`） | `1.02610728`（`+2.6107%`） |
| `S_{long}` | `0.99237394`（`-0.7626%`） | `1.00092039`（`+0.0920%`） |
| `S_{wait}` | `0.96137716`（`-3.8623%`） | `0.99966180`（`-0.0338%`） |
| `N_{TC}` | `1.0` | `1.0` |

故 v14 padding 的 shared-load bank-conflict 机制成立；但该 conflict 降幅与 long-scoreboard 的小幅变化
结合 clean ABBA 只有 `+2.40452555%`，只能说明本次降幅不足以跨过 `3%` 门，不能由一次 NCU 采集认定
唯一或主要剩余瓶颈。v15 的 bank conflict/wavefront 反向或无益，且其余 stall 近乎不变，
与 `-0.09713217%` 一致。NCU replay/cycle/counter 不进入 performance scoreboard，也不提升或重算
v14/v15 的 RC3。

本地 [NCU archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-v14-v15-ncu-evidence-20260831/c2-native-v12-v14-v15-ncu-job13666-evidence-20260831.tar.gz)、
[30-record manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-v14-v15-ncu-evidence-20260831/c2-native-v12-v14-v15-ncu-job13666-evidence-20260831.manifest.sha256)
和 [operator audit](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-v14-v15-ncu-evidence-20260831/ncu-jobs13662-13666-operator-audit-20260831.json)
已固定：tar SHA-256 `8ccfb059901a718511b9ebe38a5b01d3191ca42f90e876731186707fc5843322`，30 条 manifest/31 个
tar member 均在本地复核，且比率由归档 absolute counters 独立重算通过。controller 在归档时已 purge
job13666、slurmdbd 又不可用；故 archive 验证不可独立重建 scheduler 状态，`COMPLETED/0:0` 只来自完成后
即时 operator observation，不写成 archive-contained scheduler attestation。

#### v16：页内 K-chunk raw-FP8 lane-private lookahead 的接受、wheel 与 lifecycle 晋级

本段先固定 v16 专用符号，再给出任何比率或判定；它不重算 v12–v15 的 NCU，也不将历史
direct-plugin 收益叠加。

| 变量 | 含义 | v16 固定值或判定边界 |
| --- | --- | --- |
| `K_c` | 下标 `c` 表示页内 raw-FP8 K lookahead 的 chunk；每条 lane 私有地推进其 K load | 只改变 v12 的页内 K load 排程；跨页语义、PV、softmax、merge 与 ABI 不变 |
| `R_{16}` | 下标 `16` 表示 v16 的 AOT static shared memory | `31136 B` |
| `I_{v16←v12}` | 同 UUID、同输入/harness 的 8-seed paired-median 中 v16 相对 v12 的改善 | `+6.2838716258%`；接受门严格 `>3%` |
| `LCB_{v16←v12}` | `I_{v16←v12}` 的 deterministic paired bootstrap 95% 下界 | `+6.1226889551%`；接受门严格 `>0` |
| `N_seed` / `N_warm` / `N_iter` | 压力验证的 seed 数 / 每臂 warmup / 每 seed 单调用 CUDA-event 次数 | `8 / 30 / 200` |

v16 从冻结 v12 源出发，只把 raw-FP8 K 的页内 lookahead 拆为 lane-private `K_c`。AOT job
**13773** 从干净根完成，冻结 source/plugin SHA-256 分别为
`a01b78c89205a8ded07d65c82cdcf9d2eb5b39df4fc45a2fd992d92200b84180` /
`1da5f731da796656759f0e673e3479392b6b8337c054a61aa4ca1fd0afb4edd4`；资源为
`REG=128, STACK=0, SHARED=31136, LOCAL=0`，并保留唯一 `sm_100` cubin、无 PTX。随后 directed
job **13786** 通过 v12 完整 gate 以及 transition、last 与 stale-shifted 三个新增 case，未使用
monkeypatch，且 PRE/POST/final-post 都在同一 B300 UUID、compute-apps 为空的状态下完成。

正式性能裁决是 job **13789** 的四个隔离 Python 进程 `v12_A→v16_B1→v16_B2→v12_A2`。它在同一
`GPU-dadf9f3b-df58-d3fa-07b0-5fe223423db1` 上执行，所有 8 个 paired seed 的改善均为正，正确性、
身份、资源和可比性门均通过，`accepted=true`。因此 `I_{v16←v12}=+6.2838716258%` 且
`LCB_{v16←v12}=+6.1226889551%`；这是独立的 v16-v12 direct-plugin 比较，既不与 v7/v9/v11/v12
的相邻收益相加或连乘，也不外推为 Triton parity、full-model 或 service 速度。

job **13832** 将该精确 AOT DSO 置入 overlay wheel；wheel SHA-256 为
`3947fab41739c98a30a8fd5486b867347b932f3419def3bfbd846db458ba90a9`，最终
`FINAL_RC=0, CLEANUP_RC=0`，并由 wheel payload/RECORD、稳定成员与 promotion prerequisites 逐项门控。
这仍是实验性 overlay wheel，不是 upstream/release wheel。job **13845** 再从该 wheel 完成 v16
单进程 lifecycle：8 个 loader、1000 次真实 forward、原 query 与原地更新 query 的两组 `100` 次 CUDA
Graph replay、8 MiB 有界内存、选择/拒绝和边界合同均通过；8 个 lifecycle gate 以及 PRE/POST/final-post
同 UUID 且 compute-apps 为空、scratch cleanup 全部通过，最终状态字段均为 0。故 v16 取代 v12，成为
最新 lifecycle-closed candidate；这一状态不认证多进程、长期 server、正式 release、full model 或 Triton
parity。

v16 的本地 [lean archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-k-chunk-lookahead-evidence-20260831/c2-native-plugin-v16-k-chunk-lookahead-lean-evidence-20260831.tar.gz)、
[122-record manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-k-chunk-lookahead-evidence-20260831/c2-native-plugin-v16-k-chunk-lookahead-lean-evidence-20260831.manifest.sha256)与
[sidecar](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-k-chunk-lookahead-evidence-20260831/c2-native-plugin-v16-k-chunk-lookahead-lean-evidence-20260831.tar.gz.sha256)
已逐文件复核：SHA-256 分别为 `a9ce039b9ca9cbf151c49a82f5cdbf5800ff84fa00182c03a1fe59ab80afccd8`、
`be213b6427b173e77930c7881b041e4d2de86750b30949edcdfb5fd356984a59`、
`1b3040b34b59128e786119567b932fc6257928fa7b272928916be223b8b46895`，manifest 与 tar 的 122 个
regular-file member 字节双射，且无 symlink、`.whl` 或 stress fixture。被排除的 `68,219,593 B` fixture
仍由远端 87-file full archive 固定为 SHA-256 `2c571f37c94c744492bed673741930be8a4738b1d35d4d547aa71caab6f1d4a7`；
`313,122,399 B` wheel 由 lifecycle 实际安装链及 SHA-256 `3947fab4…90a9` 固定。远端 full tar SHA-256
为 `784a0d8de98b6ffba3532f55c11e65a3c7c35bb2528b72994f5c4f1496bd104f`，本地未把该 55,584,112 B
完整包误写成已下载对象。

#### v12/v16 matching NCU：支持“等工作量下 cycles 缩短”，但不证明 long-scoreboard 降低

| 变量 | 含义 | 本节取值 |
| --- | --- | --- |
| `C_{x,m}` | 版本 `x` 在 counter `m` 上的绝对 NCU 值 | `x∈{12,16}`；每臂恰一 logical native action |
| `\rho_{16/12}^{m}=C_{16,m}/C_{12,m}` | v16/v12 的同作业 counter 比率 | 只作机制解释；不是 benchmark speedup |
| `m_{cyc},m_{TC},m_{long}` | elapsed cycles / tensor instructions / normalized long-scoreboard stall | job13868 的 15-counter catalog 子集 |

job **13868** 在第二张 B300 `GPU-3924…bb0a` 上，以冻结 v12/v16 DSO、相同 input manifest 与 harness、
两个串行独立 Python/NCU 进程采集；每臂恰一 matching kernel row，15 个 metric 的单位一致，PRE/POST
UUID 相同且 compute-apps 为空。`FINAL_RC/ORIGINAL_RC/FINALIZER_ERROR/TEE_RC/RUNTIME_CLEANUP_RC/
MANIFEST_RC` 全为 0；结果明确写入 `timing_valid_for_benchmark=false` 与
`duration_metric_requested=false`。

| `m` | `\rho_{16/12}^{m}` | 相对变化 |
| --- | ---: | ---: |
| `m_{cyc}` | `0.92172466` | `-7.8275%` |
| tensor active pct | `1.08181818` | `+8.1818%` |
| `m_{TC}` | `1.00000000` | `0%` |
| DRAM read | `1.00000000` | `0%` |
| L2 bytes | `0.99408328` | `-0.5917%` |
| shared-load wavefront | `0.99989367` | `-0.0106%` |
| shared-load bank conflict | `0.99966448` | `-0.0336%` |
| `m_{long}` | `1.11319760` | `+11.3198%` |
| wait stall | `1.04795401` | `+4.7954%` |
| barrier stall | `1.19101124` | `+19.1011%` |

tensor 指令、DRAM 与 shared-load 控制量基本不变，而 instrumented cycles 下降、tensor-active 比例上升；
这与“相同主工作量下排程/重叠更有效”一致。可是 normalized long-scoreboard、wait 与 barrier 比率并未下降，
所以本次 counter **不支持**更窄的“v16 通过降低 long-scoreboard 获益”命题，也不能认定 K-load latency
hiding 是唯一或主要原因。该 stall metric 可超过 `100%`，不能当 wall time；NCU cycles 同样不进入性能
scoreboard。v16 的晋级仍只由 job13789 clean ABBA 决定，job13868 既不重算 `+6.2838716258%`，也不是
第二 UUID 的严格性能复现。结果 JSON SHA-256 为
`37cae5eabac16afbaf097fd3be66cc1a4c6c509e724de0877dc460acfdcbe4aa`，脚本 SHA-256 为
`ab179715d1b218263b66ed100c298574eecc3874cb359ed6bb0d293d9b72cf62`。本地
[NCU archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-v16-ncu-evidence-20260831/c2-native-v12-v16-ncu-job13868-evidence-20260831.tar.gz)、
[25-record manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-v16-ncu-evidence-20260831/c2-native-v12-v16-ncu-job13868-evidence-20260831.manifest.sha256)与
[sidecar](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-v16-ncu-evidence-20260831/c2-native-v12-v16-ncu-job13868-evidence-20260831.tar.gz.sha256)
已完成安全成员与逐字节双射复核；SHA-256 分别为 `9d8e2c6ca0eddec8b2ae6d140005f594593a87c0e549bffa029f0b0f21a7a983`、
`b1583ca0e7bfc937db7e311c4dfd9f471efe30a5d53e4afd94e6874452dae18e`、
`064e7caf27872d8ef209fbc3edd7353bdf389a3240b79823b796278a7d13953b`。

#### v16 fresh wheel / 真实 MiniMax 后端：集成闭环，但仍为 `2.0820x` Triton 延迟

| 变量 | 含义 | job13900 的统计口径 |
| --- | --- | --- |
| `L_{N,s}`、`L_{T,s}` | seed `s` 下 native、Triton 的延迟中位数 | 每个 seed 各 `10` 次 warmup；ABBA=`native,triton,triton,native`，每臂共 `100` 个 sample |
| `R_s=L_{N,s}/L_{T,s}` | seed `s` 的 native/Triton 延迟比 | `R_s≤1` 才表示该 seed 达到 parity |
| `\widetilde L_N`、`\widetilde L_T` | 8 个 seed latency median 的中位数 | 分别为 `0.1009760015 ms`、`0.04848000035 ms` |
| `\widetilde R=median_s(R_s)` | 8 个逐 seed ratio 的中位数 | `2.081999763`；不是 `\widetilde L_N/\widetilde L_T` 的替代定义 |

job **13900** 从 v16 冻结 overlay wheel 新装到 node-local runtime；8 个 seed 各使用独立 Python 进程，
经真实 `MiniMaxM3SparseMSAImpl.forward` 完成 native/Triton ABBA、独立 FP32 oracle、caller output、
dispatcher event 与 native CUDA-kernel trace 验证。8/8 result 的 correctness/integration gate 均通过；PRE、POST、
final-POST 为同一 B300 `GPU-0c223…7edd` 且 compute-apps 为空，scratch、tee、manifest 与 finalizer 的
最终状态字段全为 0。summary、decision、outputs manifest、final-status 与 final sidecar SHA-256 分别为
`a917653daac70115c778228e1aa2fac98c5a0eb37f68b69dfc1cbff6537e0ded`、
`5ac91dc577523bfbb91ee31382a4d6b3f4c5b98e2153828298caf4ef166612ad`、
`e910c840256502555789538c0f339ec9362238f42e14e6b7e188951d72e6b83a`、
`c5c3f88d95662a88398b062b63796a0e5840d590fabd887102c64182182e2c47` 与
`c1cd0bba239de9c08e5cb4b6e5a4afed4e08681514c95c651eb227a034e1f99a`。

所有 `R_s` 都约为 `2.08–2.09`，故 `triton_parity_achieved=false`。这不是 fresh 集成失败：parity
预先声明为 observation 而非硬门；安装、身份、正确性、trace、GPU 清洁和 finalization 均已闭环。
同时，该作业没有安装 v12 fresh-wheel comparator，不能把它与不同 UUID/不同作业的 v12 `2.209910x`
直接相减成 v16/v12 晋升，也不能与 job13789 的 `+6.2838716258%` 相加或连乘。它关闭的是“v16 wheel
在全新进程和真实后端是否可用、离 Triton 还有多远”的讨论，留下的是可量化的约 `2.0820x` kernel gap。

本地 [fresh lean archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-fresh-backend-evidence-20260831/c2-native-v16-k-chunk-lookahead-fresh-backend-job13900-lean-evidence-20260831.tar.gz)、
[42-record manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-fresh-backend-evidence-20260831/c2-native-v16-k-chunk-lookahead-fresh-backend-job13900-lean-evidence-20260831.manifest.sha256)与
[sidecar](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-fresh-backend-evidence-20260831/c2-native-v16-k-chunk-lookahead-fresh-backend-job13900-lean-evidence-20260831.tar.gz.sha256)
的 SHA-256 分别为 `4e93c6902fce7e90da5021f987d1673f86dae549987185ddcb40445d453c9743`、
`a03a029505eda9b3685f5d1a90b30ec3d4328aef9ed3b5dde8a14c0af87c3914`、
`901a8adca66bbf0d0202738484d5da138166b4e6538033b20c49d626970878cd`。本地已把 42 个 safe regular member
解包到 [verified members](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-fresh-backend-evidence-20260831/verified-members/)，
逐项复算并证明 manifest↔tar 字节双射；wheel 与 node-local runtime payload 均未重复打包。

#### 失败与受控拒绝审计：二者不混用

| job | 失败点 / 裁决点 | 原因 | 如何闭环 |
| ---: | --- | --- | --- |
| 11434 | AOT 架构门，`FINAL_RC=1` | 首版把 request `10.3f` 与 toolkit 实际支持目标 `10.0` 混为同一断言；configure 完成但未编译 | 分开记录 request/supported/plugin/compiled arch，并对最终 DSO 强制“唯一 `sm_100` cubin、无 PTX”；11443 从头成功 |
| 11446 | wheel packager，`FINAL_RC=1` | 错把新 adapter 当作 baseline 必须已有成员 | 明确 plugin/adapter 是仅有的两个新增成员；11449 从头成功 |
| 11451 | fresh runtime，`FINAL_RC=1` | baseline 与 installed snapshot 处于不等价 import 状态，断言提前失败 | 两边做等价正常 imports 后再比较 |
| 11452 | dispatcher diagnostic，`FINAL_RC=1` | 正常 `vllm._custom_ops` import 给已有 op 增加预期 Meta kernel，被误判为 plugin mutation | 把 import 增量归一化，之后只审计 loader 自身增量；11456 从头成功 |
| 11481 | v3 AOT patch gate，`FINAL_RC=1` | 首版 patch 的第一个 hunk 与冻结 post-v2 源码不匹配，编译未开始 | 改成两个 `--fuzz=0` 精确单行 hunk；11487 从头 AOT 成功，11489 再独立做 promotion gate |
| 11500 | v4 fresh-runtime post-validator，`FINAL_RC=1` | 真实 backend 结果全过后，脚本错误地期待不存在的 `…backend-v4` schema | 只把两个 schema 断言恢复为 harness 固定的 `…backend-v2`；11503 从 fresh install 全流程重跑成功 |
| 11764 | v5 softmax-directed test design，`FINAL_RC=1` | 初版 V code 的最终 BF16 舍入超出冻结严格 tolerance；11766 证明 v4/v5 误差完全相同 | 只缩小 V code `1/32`，Q/K/control 不变；11769 双臂同值且通过，11771 再做正式验收 |
| 11777 | v5 NCU Python env validator，`FINAL_RC=1` | 错从 resolved interpreter target 反推 venv root，profiler 尚未运行即停止 | 恢复 v4 的 invocation-path 校验；11779 从新目录完成全部采集与 comparison |
| 12216 / step2–4 | v5 lifecycle harness，未作为最终证据 | 初始 baseline-wheel SHA 拼写错误；随后依次出现 harness `NameError`、未初始化 `empty_like` 造成 NaN 假失败，以及依赖冻结前的过早 pass | 每次从新 job-scoped 目录 fail-closed；冻结五个支持文件并修正测试设计后，12295 全流程 hardened pass |
| 12316 | v6 directed 启动前，未进入 GPU 测试 | 上传 harness 文件名缺少 `_20260830` 后缀 | 修正远端文件身份；12318 初步通过，增加精确事件计数后由 12322 最终通过 |
| 12366 | v6 fresh install audit，`FINAL_RC=1` | 错误要求安装后的 RECORD 自身与 wheel 内逐字节一致；`uv` 合法重写 RECORD | 分别验证 wheel/installed RECORD，精确限制 5 个 installer member 与 `.lock`，并要求其余 4748 个 payload 字节一致；12385 从新 target 通过 |
| 12393 | v6 NCU，主动取消且 `FINAL_RC=2` | 提交后独立复核发现 `snapshot()` 可能吞掉 GPU query failure，因而主动取消；取消时仍有 Python 进程，finalizer 以 `ORIGINAL_RC=0, FINALIZER_ERROR=1` fail-closed，任何部分采集均不可依赖 | 不作机制结论；显式传播 selector/nvidia-smi/compute-apps rc、加强清理并重钉脚本，job 12396 从头完整采集通过 |
| 12493 | v7 directed 初步数值通过，但不作最终证据 | 独立复核发现 `snapshot()` 的 compound pipeline 可吞掉前一条 GPU query failure；本次查询实际成功不等于脚本 fail-closed | 显式传播 selector/nvidia-smi/compute-apps rc，并冻结提交脚本快照后重跑 |
| 12500 | v7 directed 数值与 `FINAL_RC=0`，但三份 GPU snapshot 均为 0 字节 | snapshot 先写文件，随后 `tee` 以同一文件为输出、从空 stdin 读取并把它截断 | 改成只读 `cat` 回放；job 12501 从新目录重跑，三份快照为 148/149/155 B 且完整通过 |
| 12513 | v7 valid candidate，`FINAL_RC=3` | correctness/resource/identity/comparability 全过，但点估计改善 `4.827842% < 5%` | 按预设门纯性能拒绝；不构建 wheel/fresh/NCU，继续保留 v6 |
| 12557 | v8 valid candidate，`FINAL_RC=3` | correctness/resource/identity/comparability 全过，但改善为 `-7.089155% < 5%`，bootstrap LCB 也为负 | 按预设门纯性能拒绝；不构建 wheel/fresh/NCU，继续保留 v6 |
| 12772 | v9 stress 计时前，`FINAL_RC=2` | candidate source 内容 SHA 正确，但提交的是 AOT artifact 副本路径；provenance 固定的是构建工作树原路径 | 保留失败证据；换用 provenance 中精确路径并写入新 root，job 12776 从头完成有效 ABBA |
| 12784 | v10 stress 硬门，`FINAL_RC=2` | candidate B1 的 seed 42 数值与稳定性均过，但 profiler 偶发漏采 native-kernel event；四 worker 因而不可聚合 | 不删除失败门或借 B2 单臂给出性能裁决；把约 `0.194 ms` 仅作为止损诊断，停止 raw K-lookahead |
| 12795 | v9b stress preflight，未产生性能数据 | 启动器缺少 required directed-sidecar environment，计时前 fail-closed | 保留日志；不以 v9b 替代 v9 的有效 ABBA，也不把它计为第三条独立性能方向 |
| 12829 | v11 directed provenance gate，fail-closed | 提交的 provenance 脚本彼此矛盾，身份绑定不能成立 | 不引用其数值或性能；修复脚本绑定后由 job 12904 从新 root 完整通过 directed 门 |
| 12956 | v11 wheel 输入/身份门，`FINAL_RC=1` | 在 wheel 与 input manifest 生成前 fail-closed；现存日志不足以证明更具体的断言点 | 不作 wheel 结论；修订 fail-closed 绑定后由 job12957 从新目录完整构建并验证 |
| 12961 | v11/v9 NCU preflight，`FINAL_RC=1` | Torch profiler 输出在 JSON 前带 USDT 文本，whole-file JSON parser 在 NCU 采集前停止 | 改为提取唯一 schema JSON；不把该次作业计为 NCU 证据 |
| 12963 | v11/v9 NCU comparison，`FINAL_RC=1` | 两臂 counter 已局部生成，但 comparison 仍沿用 whole-file parser，最终聚合 fail-closed | 局部 counter 全部排除；job12965 从新目录完整重采并通过 finalizer |
| 12976 | v11 lifecycle 提交输入门，Slurm `ExitCode=1:0`、无 final-status | 操作方提交的三项预期 digest 与脚本内冻结值不匹配；在 artifact 目录与日志重定向前停止，只留下 0 字节 Slurm log | 改用脚本内精确冻结 digest 后，job12977 从新 runtime/目录完成全部生命周期门 |
| 12979 | v12 AOT pre-start 静态复核，未执行 | 仍为 `PENDING(Resources)` 时发现生成脚本引用但未声明 `V11_PATCH` | allocation 前主动取消；无 artifact、无资源/正确性/性能结论，修正声明与 hash 门后另行提交 |
| 12980 | v12 AOT CPU 调度，未执行 | 修正脚本已通过 preflight，但 `squeue --start` 预计 `2026-08-31T02:45:38Z`，超出本轮四小时窗口 | allocation 前主动取消；只把同一 AOT 逻辑移到 GPU partition，job12983 完整通过 |
| 12988 | v12 fresh isolated install，`FINAL_RC=2` | home 用户配额在 `uv --target` 复制稳定 DSO 时耗尽 | 未进入 import/loader/GPU/seed/性能阶段；只把 disposable runtime 迁到 node-local scratch，门禁不变 |
| 12990 | v12 fresh scratch finalizer，`FINAL_RC=2` | 8-seed 功能输出已生成，但只读 support 使受控 runtime 删除失败，`RUNTIME_CLEANUP_RC=1` | 整次功能输出排除；修正写权限恢复与 evidence manifest 位置后，job12991 从全新目录完整重跑并通过 |
| 12992 | v12 lifecycle artifact-root preflight，未生成 final-status | 脚本错误要求尚不存在的全新 artifact root 必须预先存在 | 未分配生命周期证据目录、未执行 GPU/wheel/lifecycle；随后只修正“root 不存在或为空”的 guard |
| 12993 | v12 lifecycle 输入门，无 final-status | 把合法 Python 入口 symlink 与禁止 symlink 的证据文件混在同一门；旧 finalizer 又在 `set -u` 下同一行引用未生效的 local 变量 | 未安装 wheel、未执行 lifecycle；分离可执行入口验证并修复 fail-closed finalizer，本次局部输出排除 |
| 12994 | v12 lifecycle pre-install，`FINAL_RC=2` | 审核 harness 已上传，但脚本冻结的是带日期后缀的另一路径 | finalizer、GPU POST 与 scratch cleanup 正确；没有安装或生命周期结果，后续仅补齐同字节 staging 路径 |
| 12996 | v12 lifecycle post-install symlink gate，`FINAL_RC=2` | wheel 安装成功后，额外的“整个 ephemeral runtime tree 不得有 symlink”门失败；tree 同时含 install target 与 cache/tmp/pycache，删除前未保留链接名或 target，不能归因 wheel payload | accepted-evidence 与 install 之前的门不能外推；保留失败边界。后续以精确 state-area allowlist 与 resolved-target directory manifest 重放，job13334 完成八项 lifecycle 门 |
| 13387 / 13419 | v13 AOT 的 quota / 精确资源门 | 前者依赖复制前遭遇 home quota；后者实际 `SHARED=30880` 而脚本误预期 `30896` | 迁移 disposable build 到 node-local scratch，并按静态布局复算；job13428 从新 root AOT 成功 |
| 13439 | v13 directed 首调用，`FINAL_RC=2` | 首个 head-dim encoding 同步调用报 illegal instruction | 无 oracle/rank/性能数据；按 stop-loss 不调试、不跑 stress，v13 不晋级 |
| 13518 | v14 预测量 harness preflight | argument-unpack 失败在计时前 fail-closed | 无测量；修复 harness 后由 job13539 从头执行完整有效 ABBA |
| 13539 | v14 valid candidate，`RC=3` | 全部硬门通过，但 `I_{v14←v12}=+2.40452555% < 3%`；LCB 为 `+2.29466423%` | 纯性能拒绝，保留 v12；不构建 v14 wheel/fresh/lifecycle，也不合并历史百分比 |
| 13576 | v15 valid candidate，`RC=3` | AOT job13564 与 directed job13575 全过；完整 ABBA 的 `I_{v15←v12}=-0.09713217%`、LCB `-0.11928154%`，未过严格门且方向为负 | 纯性能拒绝，保留 v12；不构建 v15 wheel/fresh/lifecycle，也不合并历史百分比 |
| 13662 | v12/v14/v15 NCU 预测量 snapshot，`FAILED/1:0` | Bash `set -u` 下同一 local 声明引用未赋值 `tag`，运行约 1 秒；无 GPU/NCU/final-status | 不采信任何性能或 counter；拆分声明、固定 profile SHA 后由 job13666 从新 root 采集并闭环 |
| 13767 | v16 AOT，编译前停止 | 外部 Git/submodule 网络在 ref listing/flush 阶段失败，未进入候选编译、资源、正确性或性能阶段 | 这是基础设施失败而非 v16 候选失败；从新根重试的 job13773 完成有效 AOT |
| 13783 | v16 directed preflight，原 Slurm log 为 0 字节 | 原日志本身不含可归因信息；随后以同一操作输入做确定性 preflight replay，才定位到脚本冻结的带日期 harness 路径不存在 | 不把 replay 诊断误写成原日志自带原因；补齐同字节 dated harness 后，job13786 从新目录完成 directed gate |
| 13942 | 第二 UUID replay 的 allocation gate，`FINAL_RC=3` | 调度器仍分配 job13789 原 UUID `GPU-dadf…db1`；预注册脚本在 fixture 复制、A/B worker 与任何性能测量前停止 | 保留为“当前单 GPU Slurm 合同不能选物理 UUID”的直接证据；无 fixture/aggregate/decision/性能数据，不重试择卡，也不计入候选拒绝或成功复现 |

对应原始日志分别为 [11434](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-aot-artifacts-20260829/slurm-11434.log)、
[11446](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-wheel-artifacts-20260829/slurm-11446.log)、
[11451](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-wheel-runtime-artifacts-20260829/slurm-11451.log)、
[11452](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-wheel-runtime-artifacts-20260829/slurm-11452.log)和
[11481](experiment_logs/c2_native_c2_production_evidence/c2-native-v3-padded-pv-evidence-20260829/c2-native-plugin-v3-aot-artifacts-20260829/slurm-11481.log)，以及
[11500](experiment_logs/c2_native_c2_production_evidence/c2-native-v4-four-producer-evidence-20260829/c2-native-plugin-v4-wheel-runtime-artifacts-20260829/slurm-11500.log)、
[11764](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-warp-parallel-softmax-evidence-20260829/c2-native-plugin-v5-directed-artifacts-20260829/slurm-11764.log)与
[11777](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-warp-parallel-softmax-evidence-20260829/c2-native-plugin-v5-ncu-artifacts-20260829/slurm-11777.log)。
2026-08-30 的新增审计链见 [12216 lifecycle log](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-lifecycle-evidence-20260830/c2-native-plugin-v5-lifecycle-artifacts-20260830/job12216/driver-job12216.log)、
[12316 directed log](experiment_logs/c2_native_c2_production_evidence/c2-native-v6-register-numerator-evidence-20260830/c2-native-plugin-v6-directed-artifacts-20260830/slurm-12316.log)、
[12366 fresh log](experiment_logs/c2_native_c2_production_evidence/c2-native-v6-register-numerator-evidence-20260830/c2-native-plugin-v6-fresh-backend-artifacts-20260830/slurm-12366.log)与
[12393 NCU log](experiment_logs/c2_native_c2_production_evidence/c2-native-v6-register-numerator-evidence-20260830/c2-native-plugin-v6-ncu-artifacts-20260830/slurm-12393.log)，以及
[12493](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-directed-evidence-20260830/failure-audit-extracted/c2-native-plugin-v7-directed-artifacts-20260830/slurm-12493.log) / [12500](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-directed-evidence-20260830/failure-audit-extracted/c2-native-plugin-v7-directed-artifacts-20260830/slurm-12500.log) directed audit logs，以及
[12557](experiment_logs/c2_native_c2_production_evidence/c2-native-v8-double-buffer-metadata-evidence-20260830/stress-lean-extracted/c2-native-plugin-v8-stress-artifacts-20260830/slurm-12557.log) v8 performance-rejection log。
工具/脚本失败保留在审计链中但不进入成功 scoreboard；job 11489 的 `FINAL_RC=3` 是正确性
全过后按预设门槛执行的候选拒绝，jobs 12513/12557 同属 valid-candidate performance
rejection。后三者只贡献“候选不晋升”的裁决，不与工具失败混为一类，也不计入已晋升版本的
正向性能 scoreboard。

### 更新后的真实剩余挑战与优先级

| 优先级 | 挑战 | 现在能确认的状态 | 下一步 / 停止条件 |
| ---: | --- | --- | --- |
| 1 | native/Triton parity 与 kernel 差距 | job13900 已关闭 fresh-wheel/真实后端可用性，但 v16 的 8-seed `median(R_s)=2.081999763`，8 个 seed 均未达到 parity | 以 matching native/Triton kernel profile 和 SASS/依赖链为依据，只尝试一个可归因变量；沿用独立 oracle、fresh process 和 `>3%` 门，若正确性/身份/资源失败或收益不超过 `3%` 即止损，不把 job13789 百分比外推到此比率 |
| 2 | 严格跨 GPU / 外部有效性 | v16 性能晋级 job13789 只在 `GPU-dadf…db1`；job13942 以“非原 UUID 才复制同字节 fixture”为预注册门尝试一次，却仍被分配原 UUID，并在复制/测量前 RC3 停止。job13868/13900 的其他 UUID 只覆盖机制/集成 | 获得第二 UUID 的可控调度或更高 GRES quota 后才重开冻结四臂 ABBA；当前资源未变时不再择机重试，也不把 job13942 或跨作业非性能结果冒充严格复现 |
| 3 | 完整模型服务 E2E | 审计过的项目根与常见缓存路径没有 MiniMax checkpoint/config/tokenizer，model match count 为 0；当前证据不含 weights、scheduler、HTTP、质量、TTFT/throughput | 只有提供 B300 可访问的完整 checkpoint 绝对路径及 TP/启动配置后才启动；资产未变时停止重复搜索或构建 |
| 4 | 正式 release、多进程与长期服务生命周期 | v16 wheel 与受控单进程 lifecycle 已闭环，fresh 每 seed 进程也通过；但这不等于 upstream/release、并发多进程首次加载或长期 scheduler/server 认证 | 只有进入发布或服务集成阶段才增加多进程/长稳态/重启恢复门；当前实验性 overlay 不宣称正式 release |
| 5 | 合同外形状与请求分布 | native 明确只选择冻结 `B=16,Q_len=1,H_q/H_kv/D=64/4/128,P/K=128/16` 合同；现有边界/拒绝矩阵不是任意 batch/top-k/page/trace 通用性 | 只有产品合同要求扩大时才逐形状增加 selection、oracle、liveness 与性能门；不把“拒绝 native”写成“已验证 Triton fallback” |
| 6 | scalar sync/topology 与 two-level TMA/no-LSE/head-shard 旧线 | 已有阴性或弱收益，且没有新的 production 证据推翻原判断 | 协议、地址布局或 workload 合同未改变时继续冻结，不再消耗主预算 |

第一优先级的 profile 不能机械复用 job13868 的 one-row native parser：真实 Triton decode 每次 forward
包含 `_gqa_sparse_decode_kernel` 与 `_merge_topk_attn_out_kernel` 两个 kernel，而现有 fresh harness 又固定
运行 `torch.profiler`；直接把它套进 NCU 会形成 Kineto/CUPTI 冲突风险。下一轮应新增专门的 NCU-only
real-backend runner，先用独立 validate 证明 native 1 kernel、Triton 2 kernels 与输入身份一致，再分三进程
采集；bytes/cycles 等可加总量与 tensor-active/stall 等百分比必须分开解释，不能把两个 Triton 百分比直接相加。

因此，v16 的本地实验性 overlay-wheel、真实 backend 与受控单进程 lifecycle 都已完成；它以
`I_{v16←v12}=+6.2838716258%`、`LCB=+6.1226889551%` 取代 v12 成为最新 lifecycle-closed candidate。
这个百分比仅来自 job13789 的同 UUID direct-plugin ABBA，不能和 v7/v9/v11/v12 的相邻收益相加或连乘，
也不能外推为 Triton parity、full-model 或 service E2E。v13/v14/v15 的既有止损/RC3 裁决仍不改变，
v12/v14/v15 的 NCU 也不能重算性能。job13868 已补齐 matching NCU 的有限机制支撑；job13900 已补齐
v16 fresh wheel / 真实后端观察并量出 `2.081999763x` native/Triton gap，因此二者都不再是“待测”项目。
在当前可控资源下，最高可行动优先级变为沿 matching profile 缩小该 kernel gap；严格跨 GPU 复现仍重要，
但 job13942 已直接证明当前单 GPU 合同仍会落回原 UUID，且不能安全择卡。它没有产生 performance datum，
不改变 job13789。正式 release、多进程/长期 service 与 full-model E2E 仍未完成，后者继续受
checkpoint/config 资产缺失阻塞。job13942 的 [operator audit](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-second-uuid-attempt-evidence-20260831/job13942-operator-audit.json)
与 [7-record manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v16-second-uuid-attempt-evidence-20260831/c2-native-v16-second-uuid-attempt-evidence-20260831.sha256)
已本地复核。

历史轮次的机器可读 closure 仍保留在
[2026-08-30 continuation closure](experiment_logs/c2_native_c2_production_closure_20260830.json)及其
[sidecar manifest](experiment_logs/c2_native_c2_production_closure_20260830.sha256)。截至 v15 的本续轮由
[2026-08-31 successor closure](experiment_logs/c2_native_c2_production_closure_20260831.json)、
[successor sidecar](experiment_logs/c2_native_c2_production_closure_20260831.sha256)与
[53-record aggregate](experiment_logs/c2_native_c2_v12_v15_continuation_evidence_aggregate_20260831.sha256)
固定；它们只覆盖 v12–v15。新增且不追溯改写前驱的
[v16 successor closure](experiment_logs/c2_native_c2_production_closure_20260831_v16.json)与
[v16 closure sidecar](experiment_logs/c2_native_c2_production_closure_20260831_v16.sha256)固定 v16、job13868
matching NCU、job13900 fresh 结果及仍未完成边界。该历史续轮的 v12/v13 可复核成功与止损证据见上文链接。v14/v15 的
[continuation evidence root](experiment_logs/c2_native_c2_production_evidence/c2-native-v14-v15-continuation-evidence-20260831/)已本地验证：lean gzip
SHA-256 `5ae30832ed5345b4d51cb69b378f472774d7792355a6e5ab93c107e7a9671846`，161 条 manifest/162 个安全 tar member
均通过，且仅排除两份 fixture（其 SHA 仍在 excluded list/full manifest）。完整远端包固定为 SHA-256
`0a5260556ad189ec0b0b9c405fa0b5f10aaa0f84d98ab4ab9fa0ca0d21301458`，未被表述为本地已下载的完整 tar。
job13666 的本地 [NCU evidence root](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-v14-v15-ncu-evidence-20260831/)也已完成 archive/ratio 复核。完整远端取证归档为 [success evidence archive](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-production-evidence-20260829.tar.gz)
和 [failure-log archive](experiment_logs/c2_native_c2_production_evidence/c2-native-plugin-production-failure-logs-20260829.tar.gz)。
本轮 v3/v4/v5 增量另见 [v3 archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v3-padded-pv-complete-evidence-20260829.tar.gz)、
[v4 lean archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v4-four-producer-production-lean-evidence-20260829.tar.gz)与
[v5 lean archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-warp-parallel-softmax-production-lean-evidence-20260829.tar.gz)；
v6 增量证据位于 [v6 evidence root](experiment_logs/c2_native_c2_production_evidence/c2-native-v6-register-numerator-evidence-20260830/)，
其中 AOT、direct ABBA、directed、wheel、fresh 与 NCU 分别保留原始 JSON/log/report；
[v6 lean manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v6-register-numerator-evidence-20260830/c2-native-v6-register-numerator-evidence-20260830.sha256)
逐文件固定这些产物。v4/v5/v6 wheel 本体均未重复放入 lean archive，其完整 SHA、member 与
RECORD 证据分别由 job 11499/11773/12331 provenance/manifest 固定；v5 archive 内 110 个
文件仍由原 [lean manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-warp-parallel-softmax-evidence-20260829/c2-native-v5-warp-parallel-softmax-lean-evidence-20260829.sha256)固定。
v5 生命周期证据另见 [lifecycle manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v5-lifecycle-evidence-20260830/c2-native-v5-lifecycle-evidence-20260830.sha256)；
冻结 v5 jobs 12213/12214、双 GPU QOS 拒绝、模型目录审计与 exact d4 checkout 见
[external audit](experiment_logs/c2_native_c2_production_evidence/c2-native-external-dependencies-evidence-20260830/c2-native-external-dependencies-audit-20260830.log)、
[replay archive](experiment_logs/c2_native_c2_production_evidence/c2-native-external-dependencies-evidence-20260830/c2-native-v5-replay-jobs12213-12214-lean.tar.gz)与
[external manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-external-dependencies-evidence-20260830/c2-native-external-dependencies-evidence-20260830.sha256)。
该外部审计中的 `sacct` 数据库查询当时不可用，因此成功状态、UUID 与性能值以已归档的 job
日志/JSON 为准，不把失败的 `sacct` 查询本身写成调度器状态证据。
v7 的 AOT、最终 directed、directed failure-audit 与 stress rejection 分别位于
[v7 AOT root](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v-prefetch-evidence-20260830/)、
[v7 directed root](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-directed-evidence-20260830/)和
[v7 rejection root](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v-prefetch-rejection-evidence-20260830/)；
v8 AOT/directed/stress 位于 [v8 evidence root](experiment_logs/c2_native_c2_production_evidence/c2-native-v8-double-buffer-metadata-evidence-20260830/)。两轮新增的 185 个本地文件由
[v7/v8 aggregate manifest](experiment_logs/c2_native_c2_v7_v8_evidence_aggregate_20260830.sha256)
逐文件固定，其 SHA-256 为 `40d210522f78c3f1985020974bc62125710d67b29db65d4759dae26ac1277421`；
上传临时文件不进入该 manifest，远端完整 fixture 的存在与 SHA 则由各自 full manifest 固定。

本次 `>3%` 续轮的 B300 端完整 hardened tar 位于
`/home/lcpu/85117379/c2-native-v7-v11-continuation-hardened-evidence-20260830.tar`，SHA-256 为
`2e3b09aec5d481fc74a12c79088189a2628edbc8f2c2a36481a3f726bd295e15`，其
[full manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-v7-v11-continuation-hardened-evidence-20260830.sha256)
含 311 条文件记录。本地保留并三重验证的是 [hardened lean gzip](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-v7-v11-continuation-hardened-lean-evidence-20260830.tar.gz)：
gzip SHA-256 为 `ed811de01c85b30480a8398dc368cf9b2afd9c0e36fdddbbe145bcb778c3d698`，
解压 tar 流 SHA-256 为 `3b6d2e0a8a1a21398c08f131ffdedbf3b7364418ccabebe6ab4572a67c9d4e12`；
[lean manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-v7-v11-continuation-hardened-lean-evidence-20260830.sha256)
已在本地逐项通过 `309/309`。精简包只省略四份大型 `fixtures-8-seeds.pt` payload；它们的完整
路径与 SHA 仍由 [excluded-fixture list](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-v7-v11-continuation-excluded-fixtures-20260830.sha256)
和 full manifest 双重固定，其余 status、JSON、脚本、DSO/SASS、日志与输出均保留。包含 gzip
本体和 sidecar 在内的 315 个本地有效文件另由
[continuation aggregate manifest](experiment_logs/c2_native_c2_v7_v11_continuation_evidence_aggregate_20260830.sha256)
逐项固定（manifest SHA-256 `893c7b6897f29d8cca959399709f2904c5497599668a775fd7b51e1bbdc5aaed`）；
断线产生的两个 `.incomplete-transfer` 隔离文件和同步器临时文件不属于证据。结构化结论见
[continuation closure](experiment_logs/c2_native_c2_production_closure_20260830.json)，其输入由
[closure sidecar](experiment_logs/c2_native_c2_production_closure_20260830.sha256)固定。以下链接均指向
本地 lean 包已经落地的 top-level root：
[v7 >3% ABBA](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v7-stress-3pct-artifacts-20260830/)、
[v9 AOT](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v9-aot-artifacts-20260830/)、
[v9 directed](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v9-directed-artifacts-20260830/)、
[v9 valid ABBA](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v9-stress-3pct-retry1-artifacts-20260830/)、
[v10 stop](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v10-stress-3pct-artifacts-20260830/)、
[v9b AOT](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v9b-aot-artifacts-20260830/)、
[v9b directed](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v9b-directed-artifacts-20260830/)和
[v9b preflight log](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/slurm-c2-native-plugin-v9b-stress-3pct-12795.log)，以及
[v11 AOT](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-aot-artifacts-20260830/)、
[v11 fail-closed audit](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-directed-failure-audit-job12829-artifacts-20260830/)、
[v11 directed](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-directed-artifacts-20260830/)
与 [v11 accepted ABBA](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-native-plugin-v11-stress-3pct-artifacts-20260830/)。

v11 晋级新增四份本地、逐 tar-listing 验证的归档：[wheel jobs12956/12957 的 closure 记录](experiment_logs/c2_native_c2_production_closure_20260830.json)、
[fresh job12960](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-v11-fresh-evidence-job12960.tar.gz)、
[NCU jobs12961/12963/12965](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-v11-ncu-evidence-jobs12961-12963-12965.tar.gz)与
[lifecycle jobs12976/12977](experiment_logs/c2_native_c2_production_evidence/c2-native-v7-v11-continuation-evidence-20260830/c2-v11-lifecycle-evidence-jobs12976-12977.tar.gz)，
SHA-256 依次为 `3b602a50…aef1`、`b59e0737…df2d`、`f4691d0d…95b3`、
`1fb76199…5f69`。新 [promotion aggregate manifest](experiment_logs/c2_native_c2_v11_promotion_evidence_aggregate_20260830.sha256)
已在本地逐项验证 `147/147`，manifest SHA-256 为
`21ef28f2de6344c0359126a4eeabca594a58c002f5e9029e1b57ce686a7a2d62`；它单独固定
job12960/12965/12977 final-status，并把 job12976 的“无 final-status、仅 0 字节 Slurm log”边界
显式保留。lifecycle job-local outputs manifest 的一个路径指向远端 job-scoped
`c2_lifecycle_core.py`；本地归档以 `native_c2_lifecycle_core_20260830.py` 保存同一
`5ca35d63…a0788` 内容，因此内容身份已固定，但不声称复现该远端 runtime 路径。

v12 的原始性能/wheel/fresh 增量仍位于
[v12 evidence root](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-q-row-padding-evidence-20260830/)。
[core lean archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-q-row-padding-evidence-20260830/v12-q-row-padding-lean-evidence-20260830.tar.gz)
保留 jobs12983/12984/12985/12986 与两个 pre-start 审计；[wheel/fresh addendum manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-q-row-padding-evidence-20260830/c2-native-v12-wheel-fresh-addendum-20260830.sha256)
固定 jobs12988/12990/12991。旧 lifecycle failure audit 仍保存 jobs12992/12993/12994/12996 的边界，
而成功重放独立归档为 [job13334 lifecycle archive](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-lifecycle-closure-evidence-20260831/c2-v12-lifecycle-job13334-evidence-20260831.tar.gz)，
其 [27-record manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v12-lifecycle-closure-evidence-20260831/c2-v12-lifecycle-job13334-evidence-20260831.manifest.sha256)
逐项固定成功证据。v13 的 [76-record stop-loss manifest](experiment_logs/c2_native_c2_production_evidence/c2-native-v13-distributed-merge-stoploss-evidence-20260831/c2-v13-distributed-merge-stoploss-evidence-20260831.manifest.sha256)
与 tar 见上文；v14/v15 lean archive 已以 SHA `5ae30832…`、`161/161` manifest 与 162-member tar
在本地通过验证；job13666 NCU archive 已以 SHA `8ccfb059…`、`30/30` manifest 与 31-member tar 通过验证。
完整 v14/v15 tar 仍仅由远端 SHA 固定，不把它误写成已下载的本地完整包。
