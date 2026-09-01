# C1 `CHUNK` / tcgen05 最小 microbench

## 变量表

| 变量 | 含义 | 取值或下标 |
| --- | --- | --- |
| `C` | 一个 KDA chunk 的 token 数，也是下三角小矩阵边长 | `16, 32, 64` |
| `i` | chunk 内 token 位置 | `0 <= i < C` |
| `g_i` | gate 的对数衰减 | 此处固定最保守安全界 `-5` |
| `r(C)` | 首、尾 token 的相对衰减 | `exp(-5(C-1))` |
| `L` | 严格下三角矩阵 | `C x C`，本 proxy 的随机 FP16 输入 |
| `P_j` | `L` 的二次幂 | `P_j=L^(2^j)` |
| `I_C` | `C x C` 单位矩阵 | 用于 Neumann 逆 |
| `N_C` | doubling Neumann proxy 的逆近似 | `prod_j (I_C + P_j)`，覆盖至 `L^(C-1)` |
| `Q_auto(C)` | 等输入字节口径的一次独立小矩阵 batch 数 | 自动选择，使每个输入矩阵约 128 MiB |
| `Q_fixed` | 同并行度/同 batch 的一次独立小矩阵 batch 数 | `min_C Q_auto(C)`；默认参数下为 16384 |
| `M_t,N_t,K_t` | 作业现有 SM100 tcgen05 单 tile 的逻辑形状 | `128,64,64` |
| `M,N` | HMMA/tcgen05 公平对照的逻辑输出边长 | `128,64` |
| `K_t=64` | 对照中 tcgen05 强制使用的物理归约长度 | 固定 64 |
| `C` | 对照中真正有数值的逻辑归约长度 | `16,32,64`；`[C,K_t)` 全为 BF16 bit-zero |
| `Q` | 一次 event 测量的独立 CTA tile 数 | 默认 4096；两条路径相同 |
| `F=2MNC` | 对照的逻辑有效 FLOP/CTA | 只计相同数学 GEMM 的有效乘加 |
| `F_t=2MNK_t` | tcgen05 实际物理 tile FLOP/CTA | 包含零 padding 消耗的硬件工作 |

## 目的与边界

这是对 `TASK.md` 讨论点 1、2 的**最小、独立、可复核**补充，刻意不修改
pinned FlashKDA 快照：

1. 在 GPU 上将 `r(C)` 转为 BF16，扫描 `C=1..64`，报告 normal、subnormal、
   zero 的实际边界，并单列 `C=16/32/64`。
2. 复刻上游 [`utils.cuh`](../FlashKDA/csrc/smxx/utils.cuh) 中
   `neumann_inv_fused_1warp` 的 *doubling* 代数结构：从 `I+L` 开始，依次构造
   `L^2,L^4,...`，并做 `N <- N + N L^(2^j)`。对 `C=16,32,64`，它分别需要
   `6,8,10` 次完整的稠密 `C x C` GEMM；实际使用大 batch 的 FP16
   `torch.bmm` + CUDA event 计时，避免单个微小 GEMM 被 launch latency 淹没。输出
   有两张严格分开的表：`equal_input_matrix_bytes` 固定每个输入矩阵约 128 MiB，
   用于吞吐效率；`fixed_Q_same_parallelism` 固定 `Q=min(Q_auto)=16384`，才报告
   `time_ratio_vs_C16_same_Q`，用于同工作流的时延/工作量缩放。
3. 从现有 [`m3_tcgen05/02_single_tile.cu`](../../../cuda/m3_tcgen05/02_single_tile.cu)
   的已实现最小 tile `128x64x64` 计算几何/有用 FLOP 利用率；如可执行文件可用，
   额外以多个 seed 跑其严格正确性 gate。该 M3 程序本身**不计时**，所以这里输出的
   `tcgen05_throughput_proxy` 只是 tile 几何利用率，绝不冒充真实 tcgen05 吞吐。

这不是完整 FlashKDA 的 `CHUNK=32/64` 编译变体，也不是 “tcgen05 替换 HMMA” 的
端到端实现：实际 KDA 还包含 TMA、状态递推、布局和同步。它只回答两个可分离的
量化问题：数值下溢发生在哪里，以及若将上游的 16x16 doubling 小矩阵直接扩大，
密集矩阵工作量和同类 Tensor Core proxy 如何缩放。

### 新增：真正同逻辑工作量的 `mma.sync` / tcgen05 对照

旧的 Python proxy 与 M3 单 tile 只能分别说明 Neumann 扩张成本和 tcgen05 的
几何边界，**不能**构成两条指令路径的性能对照。新增
[`hmma_tcgen05_fair.cu`](hmma_tcgen05_fair.cu) 专门补上这个缺口：对每个
`C∈{16,32,64}`，两条路径均计算

`D[M,N] = A[M,C] B[C,N]`，其中 `M=128,N=64`，BF16 输入、FP32 accumulator
与 FP32 输出完全相同。两者读取的是同一块物理 `A[M,64]`/`B[N,64]` 数据；仅
`[0:C)` 含确定性小整数，`[C:64)` 均为 BF16 bit-zero。HMMA 仅发射所需的
`C/16` 个 reduction slice；tcgen05 必须发射完整 `128x64x64` tile。因此这是
**相同数学工作量和相同输入数据**下，对 tcgen05 K-padding 真实成本的比较，而不是
把更多有用计算偷偷放进任意一边。

两 kernel 都是一 CTA、128 threads/4 warps、相同 `Q` 的 grid；分配和 H2D copy
均在 event 外。先用两个 tile 的整数 BF16 输入做逐元素 FP32 **bitwise** 对拍，
再分别以同一套 CUDA event warmup/iterations/repeats 取中位数。脚本还审计 SASS：
参考路径必须有 `HMMA`，BF16 tcgen05 路径必须有 `UTCHMMA`，否则直接失败，避免把 API
名当作硬件实证。

这个对照的动态成本账本如下；其中静态 SASS 行数不能替代动态次数：

| `C` | 逻辑 GEMM | 有效 FLOP `F` | HMMA WMMA 调用 / 预期 `m16n8k16` | tcgen05 MMA 命令 | tcgen05 zero-K | tcgen05 有效 `F/F_t` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | `128x64x16` | 262,144 | 32 / 64 | 4 | 48 | 25.00% |
| 32 | `128x64x32` | 524,288 | 64 / 128 | 4 | 32 | 50.00% |
| 64 | `128x64x64` | 1,048,576 | 128 / 256 | 4 | 0 | 100.00% |

已在 clean B300 job 4340 执行。PRE/POST 显示 0 MiB 且 compute-apps 为空；三组
`C=16/32/64` 的 two-tile FP32 输出均 exact bitwise PASS（各 `0/16384` mismatch）。
编译物的 SASS gate 是 `HMMA=14`、`UTCHMMA=4`、`LDTM=8`。这是**完整 CTA path**：
HMMA 路径的 fragment load/accumulate/store，以及 tcgen05 的 global→swizzled shared、
TMEM alloc/MMA/LDTM/dealloc 都在 event 内；它不是一条孤立 MMA 指令的吞吐测试。

| `C` | HMMA median (ms) | HMMA 逻辑 TFLOP/s | tcgen05 median (ms) | tcgen05 有效 / 物理 TFLOP/s | HMMA/tcgen05 | exact |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 0.056799360 | 18.904118 | 0.242131189 | 4.434546 / 17.738183 | 0.234581 | PASS |
| 32 | 0.088442236 | 24.281200 | 0.242168307 | 8.867732 / 17.735464 | 0.365210 | PASS |
| 64 | 0.152294397 | 28.201742 | 0.242264956 | 17.728389 / 17.728389 | 0.628627 | PASS |

原始 [JSON](../experiment_logs/c1_hmma_tcgen05_same_work_b300_job4340.json)、
[日志](../experiment_logs/c1_hmma_tcgen05_same_work_b300_job4340.log) 与
[SASS](../experiment_logs/c1_hmma_tcgen05_same_work_b300_job4340.sass) 共同构成证据。
tcgen05 的物理吞吐在三种 `C` 下约 17.74 TFLOP/s，而有效吞吐受 `K_t-C` padding
限制；即使 `C=64` 无 padding，HMMA 也在这项完整 CTA 对照中更快。这一结论不外推为
FlashKDA tcgen05 端到端结论。

这里每个 `16x16` WMMA 逻辑 tile 会下沉为两个 `m16n8k16 mma.sync`；每 CTA 始终
有 32 个 `16x16` FP32 accumulator fragment。tcgen05 则固定使用一块
`128x64x64` TMEM 输出（32 KiB）、4 条 `tcgen05.mma` 命令和 32 个 warp-level
`tcgen05.ld.x8` collectives。表中是**动态工作量账本**；实际 SASS 静态指令由运行
脚本保存，不能把静态文本行数误当作动态指令次数。

它仍不是端到端 FlashKDA tcgen05 移植：KDA 还涉及小方阵、TMA、递推状态和
跨 proxy 同步。它的结论范围仅是：在相同 `128x64xC` 数学 GEMM、相同 BF16 输入
与相同 CTA/Q 下，直接采用该 tcgen05 tile 的 latency、有效/物理 TFLOPS 与
padding 损失。

## 纸面检查点

`r(C)=exp(-5(C-1))`。BF16 的最小 normal 为 `2^-126≈1.175e-38`，最小
subnormal 为 `2^-133≈9.184e-41`。因此理论上 `C=18` 仍 normal、`C=19` 为
subnormal、`C=20` 及以后量化为零（round-to-nearest 的边界由程序在设备上再次
验证）。

对 `C=2^p`，upstream doubling 的完整方阵 GEMM 数是
`2(p-1)`：每个 `j=1..p-1` 有一次 `P_j=P_(j-1)P_(j-1)` 和一次
`N<-N+N P_j`。只计 GEMM FMA 的工作量为
`2 * C^3 * 2(p-1)` FLOP。因此相对 `C=16`：

| `C` | dense GEMM 数 | GEMM FLOP 相对 `C=16` |
| ---: | ---: | ---: |
| 16 | 6 | 1.000x |
| 32 | 8 | 10.667x |
| 64 | 10 | 106.667x |

这比只报单个 `C^3` 的 8x/64x 更严格，因为 upstream 实现会随阶数增加更多
doubling 步；它仍是**密集 proxy**，没有声称等于最终已调优 kernel 的时延倍率。

`128x64x64` tcgen05 tile 对一个 `16x16x16` Neumann 子块的单独映射，M/N/K
三个维度的利用率分别是 `1/8,1/4,1/4`，乘积为 **1/128 = 0.78125%**。即使将
32 个相互独立的 `16x16x16` 问题装满 M/N 平面，K 仍只有 16/64 有用，最多为
**25%**；不能把不同独立 GEMM 随意沿 K 拼接，因为 K 是同一输出的归约维。

## 干净 B300 运行

主实验分配好 GPU 后（不要自行申请共享 GPU）运行：

```bash
cd /home/lcpu/85117379/codex-a02-20260819-main/assignment02
PYTHON_BIN=.venv/bin/python \
TCGEN05_BIN=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/cuda/bin/m3_tcgen05/02_single_tile \
team/c1_flashkda/microbench/run_clean_audit.sh
```

脚本在运行前后记录 GPU、driver、CUDA、compute-apps；若开始时已有 compute app
则拒绝运行。默认 JSON 写到 `experiment_logs/c1_chunk_tcgen_microbench_b300.json`
（可用 `OUT_JSON` 覆盖）。JSON schema 为 `c1_chunk_tcgen_microbench/v2`；其中
`neumann_doubling_proxy.fixed_Q_same_parallelism` 对每个 C 保存相同 `Q` 下的
总 GEMM FLOP、median ms、相对 C16 时延和 TFLOPS。JSON 中
`all_assertions_pass=true` 才表示数值边界、代数正确性和可选 M3 strict gate 都通过。

公平 HMMA/tcgen05 对照也只能由主会话在已授权的干净 B300 allocation 中运行：

```bash
cd /home/lcpu/85117379/codex-a02-20260819-main/assignment02
bash team/c1_flashkda/microbench/run_hmma_tcgen05_audit.sh
```

它默认写入 `experiment_logs/c1_hmma_tcgen05_same_work_b300_job${SLURM_JOB_ID}.json`、
同名 `.log` 与 `.sass`；JSON 包含每个 `C` 的逻辑/物理 FLOP、padding、两个路径的
event sample 与 median、有效/物理 TFLOPS，以及 exact gate。若 `HMMA` 或
`UTCHMMA` 的 SASS gate 未通过，结果必须视为无效而非替换文字结论。

FlashKDA K1/K2 的完整 NCU 复核同样不自行申请资源，命令为：

```bash
cd /home/lcpu/85117379/codex-a02-20260819-main/assignment02
bash team/c1_flashkda/harness/run_roofline_ncu_audit.sh \
  /home/lcpu/85117379/flashkda-official \
  /home/lcpu/85117379/fla-ref \
  /home/lcpu/85117379/c1env-cu130/bin/python b300
```

其中前两个绝对路径必须替换为实际 pinned FlashKDA/FLA 工作树。脚本保存 `.ncu-rep`
和 raw CSV，并由 `summarize_ncu_roofline.py` 输出中文证据摘录：Tensor/SM、DRAM/L2、
occupancy/waves、warp stalls 都按 K1/K2 分开；缺失 metric 会明确标记 `MISSING`，
不允许据此宣称 roofline 分类。

本地静态检查（不需要 GPU）：

```bash
python -m compileall -q team/c1_flashkda/microbench/c1_chunk_tcgen_microbench.py
python -m compileall -q team/c1_flashkda/harness/summarize_ncu_roofline.py
bash -n team/c1_flashkda/microbench/run_hmma_tcgen05_audit.sh
bash -n team/c1_flashkda/harness/run_roofline_ncu_audit.sh
git diff --check -- team/c1_flashkda/microbench
```
