# C1 value-shard Phase-6 软件预取深度 3 消融

本目录保存最后一个固定候选 P3S3：在 current P2S3 基础上只把 K2 Phase-6 软件环深度从 `PREFETCH=2` 改成 3，P2/P3 的 `kInputStages` 均为 3。extension 同时保留 public baseline、P1、P2S3 与独立 `fwd_vshard_p3` ABI，因而正式性能比较是同一 SO、同一 allocation 的 P2S3↔P3S3。

## 变量表

| 变量 | 含义 |
|---|---|
| `PREFETCH` | Phase-6 软件环深度；P1/P2/P3 分别为 1/2/3 |
| `m` | 当前消费的 K 方向 16 元素块编号 |
| `slot = m % PREFETCH` | 当前消费并随后回填的环槽 |
| `m + PREFETCH` | 消费当前槽后装入该槽的后续块 |
| `S_M_BLOCKS` | K 方向块数；固定特化中为 `128/16=8`，所以 P3 静态合法 |
| `kInputStages` | K2 输入 pipeline 深度；baseline/P1/P2/P3 均为 3 |
| `H,T` | 终门固定为 `H=64,T=8192` |

## 生成与不变量

`apply_prefetch3_patch.py` 绑定 vendored current P2S3 generator SHA-256 `f83e3551…8ccb0`，从 pinned FlashKDA `1ce47ea3…ffb0b` 同时生成 P1/P2/P3 headers。P3 仍使用参数化的初始装环、`slot=m%PREFETCH` 与 `m+PREFETCH` 回填路径；没有改 state store/reload 次序、barrier、value-shard 布局、网格或 Python 调用合同。

fresh 静态生成确认：四条 baseline/P1/P2/P3 launch 都恰好一个 `kInputStages=3`，没有 stage 2；P1/P2/P3 各自恰好一个 `PREFETCH=1/2/3`；binding 同时暴露 `fwd_vshard_p2` 与 `fwd_vshard_p3`。generator SHA-256 为 `9f37c829…9c358`。

## Fresh SM103a build 与 ptxas

在 job 4617 中令 `CUDA_VISIBLE_DEVICES` 为空，仅做 CPU-side fresh clone/build，没有 import extension 或运行 kernel。SO SHA-256 为 `a9139b42…7354a`，P3 header 为 `971d809a…88b3c`。正式 BF16 fixed-state P3 实例为 60 registers、0 spill；P2 同实例为 54 registers、0 spill。P3 的 14 个实例中有 3 个非计时 FP32 边界实例 spill，已在完整映射中保留，不能宣称全实例 0 spill。见 [fresh build log](results/c1_prefetch3_build_b300_r1.log) 与 [ptxas JSON](results/c1_prefetch3_ptxas_b300_r1.json)。

## B300 clean correctness 与性能负结果

GPU 为 NVIDIA B300 SXM6 AC、SM103a，UUID `GPU-dadf9f3b-df58-d3fa-07b0-5fe223423db1`。PRE、all-state exact 后、benchmark 后与 POST 四次审计均为 0 MiB、compute-app 空，`FINAL_RC=0`。

`H=64,T=8192 × initial_state=none/BF16/FP32` 中，P3↔P2 的 output 与 final state 全部逐位一致；P2/P3 对独立 torch_ref 均按仓库容差通过，所有 max-abs 为 0。见 [all-state exact JSON](results/c1_prefetch3_b300_sm103a_p3s3_r1_h64_allstate_exact.json)。

BF16 full-call AB/BA 各 1000 个 event 的结果为：

| 路径 | median ms | mean ms | min–max ms |
|---|---:|---:|---:|
| current P2S3 | 0.734975994 | 0.736125916 | 0.729824007–0.777855992 |
| P3S3 | 0.793200016 | 0.793458819 | 0.787295997–0.808575988 |

同 allocation 的 `P2/P3=0.926596040×`，即 P3 反而慢约 7.92%；P3 也未达到 absolute strict `≤0.726923636 ms`。因此 same-allocation `≥1.10×` 与 absolute strict 两门都失败。JSON 保存全部 1000+1000 个原始 event 值和逐次 AB/BA 顺序，可离线重建中位数、分布与顺序偏差；见 [raw ABBA JSON](results/c1_prefetch3_b300_sm103a_p3s3_r1_h64_bf16_abba.json) 和 [clean audit log](results/c1_prefetch3_b300_sm103a_p3s3_r1_job4617.log)。

## 停止结论

P3 correctness 与 BF16 0-spill 门均通过，但性能显著倒退，表明环深从 2 增至 3 的额外寄存器驻留（BF16 fixed 54→60 registers）没有换来足够的 Phase-6 latency hiding。按预先声明的停止树，本方向到 P3 正式终止：不尝试 P4、不继续扫软件环深、不筛选样本。P2S3 仍是这一系列的最佳候选。
