# C1 value-shard Phase-6 软件预取探索

本目录是独立实验变体：保留上游 `fwd` 和已冻结的 value-shard P1 `fwd_vshard`，新增 `fwd_vshard_p2`。当前 generator 默认生成实测更优的 P2S3：只把 K2 Phase-6 环形寄存器软件预取深度从 1 改成 2，`kInputStages` 保持 3。显式传入 `--p2-input-stages 2` 才生成 P2S2 消融。P1 始终保持 stage 3；两种 P2 均不改 value-shard 布局、网格、状态 ABI、同步或公共 Python 调用合同。

## 变量表

| 变量 | 含义 |
|---|---|
| `PREFETCH` | Phase-6 软件预取环深度；P1 为 1，P2 为 2 |
| `kInputStages` | K2 输入 pipeline stage 数；P1 和默认/current P2S3 为 3，显式 P2S2 消融为 2 |
| `m` | 当前消费的 K 方向 16 元素块编号 |
| `slot = m % PREFETCH` | 当前消费并随后回填的环槽 |
| `S_M_BLOCKS` | K 方向 16 元素块总数；本特化中 `128/16=8` |
| `warp_id` | MMA warp 编号；每个 warp 独占 value shard 的一个 16 列块 |
| `H` | attention head 数；最终计时合同使用 `H=64` |
| `T` | 序列长度；最终计时合同使用 `T=8192` |

## 实现与不变量

`apply_prefetch2_patch.py` 锁定当前 P1 generator 的 SHA-256，然后从同一份上游 `fwd_kernel2.cuh` 分别生成 P1/P2 header。P2 初始装入 `m=0,1` 两槽；迭代 `m` 消费 `m % 2` 后，把 `m+2` 回填到同一槽。所有索引原本已由 `PREFETCH` 参数化，因此没有改动 `m + PREFETCH`、barrier 或 state store/reload 的先后关系。`static_assert` 只用于阻止非法深度。launch 生成器按 `--p2-input-stages {2,3}` 选择 P2 stage，并分别检查 public baseline、P1 和 P2 的局部片段及全局计数：baseline/P1 必须仍为 stage 3，P2 必须恰好为指定值。

软件预取改动不增加 dynamic shared memory，但每线程多保留一套 A/S fragment 与 gate 标量，可能增加寄存器压力。fresh build 必须从 ptxas 日志逐实例记录资源；最终计时所用的 BF16 fixed-state P2 实例必须 0 spill。FP32 等非计时实例的 spill 作为边界如实记录，不用 BF16 的 0 spill 掩盖。

## 静态与 fresh build

```bash
python -m py_compile challenge_prefetch2/*.py
# 默认/current：P2S3（最佳实测候选）
python challenge_prefetch2/apply_prefetch2_patch.py --source /fresh/FlashKDA-s3 --check-only
python challenge_prefetch2/apply_prefetch2_patch.py --source /fresh/FlashKDA-s3

# 显式消融：必须使用另一份 fresh clone，生成 P2S2
python challenge_prefetch2/apply_prefetch2_patch.py --source /fresh/FlashKDA-s2 --p2-input-stages 2 --check-only
python challenge_prefetch2/apply_prefetch2_patch.py --source /fresh/FlashKDA-s2 --p2-input-stages 2

cd /fresh/FlashKDA-s3
FLASH_KDA_CUDA_ARCHS=103a NVCC_THREADS=8 MAX_JOBS=4 python setup.py build_ext --inplace
```

历史构建使用的 generator 身份与当前统一入口如下。历史 SHA 只标识当时脚本，真正被执行的 variant 还由 build/audit 中的生成源码与 SO SHA 绑定。

| generator SHA-256 | 语义与用途 |
|---|---|
| `23d58a30a43bab13fc0fb76ef414cd51db5e2b4dc550ef464d927834a525e021` | 旧 P2S3-only generator；生成 5090 r1/r2 与 B300 r2 |
| `9a2b255679a4848b02556a0e938099e9585cf3ced5728ebefaedd42cb15b0a93` | 旧 P2S2-only generator；生成 B300 r3；H96 审计时该脚本已位于工具目录，但加载的冻结 r2 source/SO 仍由 source hashes 证明为 P2S3 |
| `f83e3551907ec8f1a5c1f5c3421e94dc1e3d1941e9f35c845d1d982eef38ccb0` | 当前参数化 generator；默认 `3` 可复现 P2S3，显式 `--p2-input-stages 2` 可复现 P2S2 |

终审时从官方仓库建立临时 fresh `1ce47ea3…ffb0b` worktree（未初始化但核对 gitlink 为 CUTLASS `5c149f52…61c6a`），分别执行默认与 `--p2-input-stages 2` 的 `--check-only` 和实际生成。P2S3 的全局 stage-3/stage-2 计数为 `3/0`，P2S2 为 `2/1`；逐片段检查中 baseline/P1 均恰好一个 stage 3，P2 恰好为所选 stage。两棵生成树的 P1 header、P2 header、`fwd.h` 与 binding SHA-256 完全相同，`fwd_launch.cu` 只有 P2 的 `kInputStages = 3`/`2` 一行不同。该复验仅生成文本，没有编译、import extension 或运行 GPU。

5090 只能使用 `FLASH_KDA_CUDA_ARCHS=120a`，其正确性和性能记录必须标为 SM120a，不能拿来宣称 B300 阈值通过。B300 使用 SM103a，且只允许在 PRE/POST compute-app 与显存审计均为空的独占 allocation 中运行。

## 严格 gate 与计时合同

small gate 覆盖 `H=1/2/4` 与 `initial_state=none/BF16/FP32`。P1↔baseline、P2↔P1 的 output 和 final state 必须逐位一致；P1/P2↔独立 pinned `torch_ref` 按仓库已有数值门比较并记录 max-abs：output `rtol=atol=0.02`，state `rtol=atol=0.05`，不得把容差通过写成 bitwise exact。只有 gate 全过才能进入 `H=64,T=8192,BF16`。

计时在同一 allocation、同一输入上交替 AB/BA；每个 CUDA event 恰好包围一次公共 wrapper 调用，包含 workspace allocation。默认 30 轮预热、1000 个独立样本/路径。冻结 P1 B300 中位数为 `0.799616 ms`，满 10% 加速的严格阈值是 `0.799616 / 1.10 = 0.726923636 ms`，验收写作 `P2 median <= 0.726924 ms`。

证据粒度边界：现有 benchmark JSON 只保存每条路径 1000 个 event 的 `mean/median/min/max/count` 汇总，没有保存 1000 个原始 event 数值或逐次 AB/BA 分组。因此可以从 JSON 重算已记录中位数之间的 speedup，但不能离线重建中位数、检查完整分布或量化顺序偏差；历史日志与 JSON 不做事后补写。AB/BA 交替及 full-call event 合同由冻结 runner 源码和审计时 runner SHA 证明。

停止条件：构建失败、最终 BF16 计时实例出现 spill、任一 exact 比较失败、PRE/POST 出现外来进程/显存，立即停止；全实例 spill 映射必须如实单列，不能用 BF16 的 0 spill 掩盖。若 clean 同 allocation 中 P2 中位数不优于 P1，也保留负结果并停止该方向，不筛选样本、不跨卡外推。

## 2026-08-20 SM120a P2S3 fresh 负结果

fresh commit `1ce47ea` 构建成功，extension SHA-256 为 `d01d8bf2…32b3d`。ptxas 对 P1/P2 各解析到 14 个实例：P2 寄存器范围为 53–56，最终 BF16 fixed-state 计时实例为 53 registers、0 spill；但 FP32 fixed-state 实例为 56 registers、8-byte spill stores、16-byte spill loads，FP32 varlen 为 12/20 bytes，所以“全实例 0 spill”未满足。原始构建与完整映射见 [SM120a build log](results/c1_prefetch2_build_5090_r1.log) 和 [SM120a ptxas JSON](results/c1_prefetch2_ptxas_5090_r1.json)。

clean Slurm job 7005 的 PRE/POST 均为 0 MiB、compute-app 空。它在旧的“torch_ref 也必须 bitwise”临时门下停止：P1↔baseline 和 P2↔P1 已逐位一致，但三者与 pinned `torch_ref` 在 SM120a 上有 `7.629395e-06` 的共同一 ULP 差异。见 [job7005 audit](results/c1_prefetch2_5090_sm120a_r1_job7005.log) 和 [机器可读历史摘要](results/c1_prefetch2_5090_job7005_failure.json)；该失败不能归因于 P2。

终门按“P2↔current P1 bitwise，独立 oracle 数值容差”澄清后，clean job 7009 完成全部 `H=1/2/4 × none/BF16/FP32`：P1↔baseline、P2↔P1 的 output/state 全部逐位一致；torch_ref 全部通过，small output max-abs 不超过 `1.525879e-05`，state max-abs 为 0。`H=64,T=8192,BF16` 同 allocation AB/BA 各 1000 样本得到 P1 `2.0363841057 ms`、P2 `2.0368319750 ms`，P1/P2=`0.9997801148×`，即 P2 在 5090 慢约 `0.022%`。PRE、small 后、benchmark 后、POST 均 clean；见 [job7009 audit](results/c1_prefetch2_5090_sm120a_r2_job7009.log)、[small gate JSON](results/c1_prefetch2_5090_sm120a_r2_small_matrix.json) 和 [H64 ABBA JSON](results/c1_prefetch2_5090_sm120a_r2_h64_bf16.json)。这是 SM120a 负结果，不跨架构外推。

## 2026-08-20 SM103a P2S3 clean 正信号但未达严格阈值

在既有 Slurm job 4467 中以 `CUDA_VISIBLE_DEVICES=` 做 CPU-only fresh build，未 import extension、未运行 kernel。r1 只因环境没有显式 Python header 而在 host 编译前失败，见 [r1 环境失败日志](results/c1_prefetch2_build_b300_r1.log)；修正 include 后使用全新 r2。fresh r2 extension SHA-256 为 `d576a7fa…ec87f`。SM103a ptxas 中最终 BF16 fixed-state 的 P1 为 58 registers/0 spill，P2S3 为 54 registers/0 spill；P2S3 的 14 个实例只有 FP32 fixed-state 一项 spill（12-byte stores、20-byte loads）。见 [SM103a r2 build log](results/c1_prefetch2_build_b300_r2.log) 与 [SM103a ptxas JSON](results/c1_prefetch2_ptxas_b300_r2.json)。

获得 clean 窗口后，`H=1/2/4 × none/BF16/FP32` 的 P2S3↔P1 output/state 全部逐位一致，独立 torch_ref 全通过且 max-abs 为 0。`H=64,T=8192,BF16` 同 allocation、各 1000 样本 AB/BA 得到 P1 `0.8063519895 ms`、P2S3 `0.7375999987 ms`，同 allocation 加速 `1.0932103998×`；相对冻结 best 的加速为 `1.0840780930×`。因此它虽比 P1 快 `9.321%`，仍比 `0.726924 ms` 严格目标慢约 `1.47%`，不得宣称 10% 达成。PRE、small 后、benchmark 后、POST 均为 0 MiB/apps 空。证据见 [clean audit](results/c1_prefetch2_b300_sm103a_fresh_r2_envfix2_job4467.log)、[small gate JSON](results/c1_prefetch2_b300_sm103a_fresh_r2_envfix2_small_matrix.json) 和 [H64 ABBA JSON](results/c1_prefetch2_b300_sm103a_fresh_r2_envfix2_h64_bf16.json)。前两次仅环境门失败、未运行 FlashKDA kernel 的日志也完整保留在 `results/`。

同一冻结 P2S3 又按固定第二代表 shape 做了一次 H96 终测，不是扫参。`H=96,T=8192 × none/BF16/FP32` 的 P2S3↔P1 output/state 全部逐位一致，torch_ref 全通过且 max-abs 为 0；1000 样本 AB/BA 得到 P1 `1.0352799892 ms`、P2S3 `1.0031520128 ms`，同 allocation 仅 `1.0320270268×`。相对冻结 H96 best `1.029888 ms` 仅 `1.0266519798×`，严格 10% 目标应为 `0.9362618182 ms`，同样未达。PRE、exact 后、benchmark 后、POST 均 clean。原始 runner 的通用 `frozen_b300_p1_ms` 字段仍是 H64 常量，H96 判定没有使用该字段；未修改的原始数据见 [H96 clean audit](results/c1_prefetch2_b300_sm103a_p2stage3_h96_r2_job4467.log)、[H96 all-state exact JSON](results/c1_prefetch2_b300_sm103a_p2stage3_h96_r2_h96_allstate_exact.json)、[H96 ABBA raw JSON](results/c1_prefetch2_b300_sm103a_p2stage3_h96_r2_h96_bf16.json)，按正确 H96 冻结值重算且绑定原始 JSON SHA-256 的结果见 [H96 shape-specific summary](results/c1_prefetch2_b300_sm103a_p2stage3_h96_r2_derived_summary.json)。

## 2026-08-20 SM103a 显式 P2S2 消融负结果

停止树允许的唯一组合是 P2S3 再把 P2 的 `kInputStages` 从 3 降到 2，P1 继续 stage 3。fresh r3 extension SHA-256 为 `c1b61f4…67a64`，P1/P2 headers 分别为 `af588b32…ae69` 与 `03ba5a26…25c3c`。ptxas 中 BF16 fixed-state P2S2 是 54 registers、0 spill；14 个 P2S2 实例中仍只有 FP32 fixed-state 一项 spill（12-byte stores、20-byte loads）。原始证据见 [r3 build log](results/c1_prefetch2_build_b300_r3_stage2.log) 与 [r3 ptxas JSON](results/c1_prefetch2_stage2_ptxas_b300_r3.json)。

job 4467 的最终 clean gate 中，`H=1/2/4 × none/BF16/FP32` 的 P2S2↔P1 output/state 全部逐位一致，torch_ref 全通过且 max-abs 为 0。H64 各 1000 样本 AB/BA 得到 P1 `0.8053120077 ms`、P2S2 `0.7760320008 ms`，同 allocation 加速仅 `1.0377304117×`；相对冻结 best 仅 `1.0303904983×`。它既未达到 `1.10×`，也未达到 `≤0.726924 ms`，并且明显差于 P2S3，故按停止条件终止这一消融，不再扩展候选、不筛选样本；默认/current generator 保留实测更优的 P2S3。PRE、small 后、benchmark 后、POST 均为 0 MiB/apps 空，审计 `FINAL_RC=0`。证据见 [P2S2 clean audit](results/c1_prefetch2_b300_sm103a_p2stage2_r3_job4467.log)、[small gate JSON](results/c1_prefetch2_b300_sm103a_p2stage2_r3_small_matrix.json) 与 [H64 ABBA JSON](results/c1_prefetch2_b300_sm103a_p2stage2_r3_h64_bf16.json)。
