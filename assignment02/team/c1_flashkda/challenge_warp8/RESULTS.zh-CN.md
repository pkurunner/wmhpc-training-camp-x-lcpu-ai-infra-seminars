# warp8 候选的 B300 实验记录（负结果）

## 固定实验合同

| 项目 | 固定值 |
| --- | --- |
| GPU | NVIDIA B300 SXM6 AC（compute capability 10.3） |
| 输入 | `B=1,T=8192,K=V=128`；分别 `H=64,96` |
| state | BF16 initial/final state |
| 输入 seed | 20260819 |
| 计时 | 每个完整 Python forward 一对 CUDA event；warmup=30，iters=200，repeats=5，共 1000 个样本 |
| 比较 | 同一隔离 extension 内的 upstream `flash_kda.fwd` 与 `fwd_warp8` wrapper；二者都在 event 内分配同 ABI workspace |

## 正确性与性能

| shape | exact gate | baseline median (ms) | warp8 median (ms) | baseline / warp8 | 相对冻结 current-best 的 10% 目标 |
| --- | --- | ---: | ---: | ---: | --- |
| `T=256,H=2` | output/final-state 均逐元素相等 | 0.041744 | 0.043808 | 0.952885x | 仅 small gate，不作性能结论 |
| `T=8192,H=64` | output/final-state 均逐元素相等 | 0.940800 | 0.961344 | 0.978630x | current-best 0.799616 ms；门槛 0.726924 ms；未达标 |
| `T=8192,H=96` | output/final-state 均逐元素相等 | 1.030784 | 1.051184 | 0.980593x | current-best 1.029888 ms；门槛 0.936262 ms；未达标 |

原始 JSON 位于同目录 `results/`：`c1_warp8_small_4420.json`、
`c1_warp8_h64_4420.json`、`c1_warp8_h96_4420.json`。H96 运行前后在 keeper job
的 compute node 上检查 `nvidia-smi`：没有 compute-app，B300 显存均为
`0 / 275040 MiB`。H64 在本分支的单次命令前未单独保存这一快照，因此不把它写作
完整的清卡证据。

结论：实现和 exact correctness 成立，但 320-thread 全 V CTA 在两个目标 shape 都
退化约 2%，故不能计入 C1 优化，也不能作为“current-best”。保留实现、构建日志和
负结果是为了避免后续重复该设计。
