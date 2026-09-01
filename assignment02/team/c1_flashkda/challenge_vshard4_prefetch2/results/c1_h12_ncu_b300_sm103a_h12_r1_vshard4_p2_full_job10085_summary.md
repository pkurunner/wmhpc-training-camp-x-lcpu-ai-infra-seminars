# C1 FlashKDA：NCU full counter / roofline 证据摘录

## 变量表

| 变量 | 含义 | 取值/来源 |
| --- | --- | --- |
| `K1` | workspace prepare kernel | `_flash_kda_fwd_prepare` |
| `K2` | 有状态 recurrence kernel | `_flash_kda_fwd_recurrence` |
| `F` | 目标 kernel 的实际有效 FLOP | 必须由 NCU roofline/FLOP 指标或独立指令计数给出 |
| `B` | 目标 kernel 的实际 DRAM bytes | `dram__bytes*`；不是输入张量大小的猜测 |
| `I=F/B` | operational intensity | 仅在 `F` 与 `B` 同一 kernel 同一 run 都可得时计算 |
| `P_roof` | roofline 上界 | `min(P_peak, I*BW_peak)` |

## 输入与完整性

- 原始 CSV：`/home/lcpu/85117379/codex-a02-20260819-main/assignment02/team/c1_flashkda/challenge_vshard4_prefetch2/results/c1_h12_ncu_b300_sm103a_h12_r1_vshard4_p2_full_job10085.csv`
- CSV 导出布局：long metric rows
- CSV 数据行：197
- 识别到 kernel：K1 prepare, K2 recurrence
- 这份文件只摘录 NCU 实际输出；完整 CSV 和 `.ncu-rep` 才是原始证据。

## 分 kernel 计数

### K1 prepare

- FLOP/instruction evidence: **MISSING**（NCU CSV 未找到匹配 metric；不能把它当作 0）。
- DRAM byte evidence: **MISSING**（NCU CSV 未找到匹配 metric；不能把它当作 0）。
- Tensor/SM:
  - `Compute (SM) Throughput` = `57.81` %
- DRAM/L2:
  - `Memory Throughput` = `59.71` %
  - `DRAM Throughput` = `34.37` %
  - `L2 Cache Throughput` = `59.71` %
  - `Memory Throughput` = `2.63` Tbyte/s
- Occupancy/waves:
  - `Block Size` = `256`
  - `Grid Size` = `6144`
  - `Overall GPU Occupancy` = `0` %
  - `Cluster Occupancy` = `0` %
  - `Theoretical Occupancy` = `100` %
  - `Achieved Occupancy` = `92.04` %
- Warp stall: **MISSING**（NCU CSV 未找到匹配 metric；不能把它当作 0）。

### K2 recurrence

- FLOP/instruction evidence: **MISSING**（NCU CSV 未找到匹配 metric；不能把它当作 0）。
- DRAM byte evidence: **MISSING**（NCU CSV 未找到匹配 metric；不能把它当作 0）。
- Tensor/SM:
  - `Compute (SM) Throughput` = `6.36` %
- DRAM/L2:
  - `Memory Throughput` = `13.65` %
  - `DRAM Throughput` = `3.51` %
  - `L2 Cache Throughput` = `13.65` %
  - `Memory Throughput` = `269.45` Gbyte/s
- Occupancy/waves:
  - `Block Size` = `128`
  - `Grid Size` = `48`
  - `Overall GPU Occupancy` = `0` %
  - `Cluster Occupancy` = `0` %
  - `Theoretical Occupancy` = `25` %
  - `Achieved Occupancy` = `6.26` %
- Warp stall: **MISSING**（NCU CSV 未找到匹配 metric；不能把它当作 0）。

## Roofline 判读纪律

1. 对每个 kernel 单独取同一 profile 的 `F`、`B`、SM/Tensor、DRAM/L2、occupancy/waves/stall；不得把 K1 与 K2 相加后套一个瓶颈标签。
2. 只有 raw report 含有可解释的实际 FLOP 与 DRAM bytes，才能算 `I=F/B`，再与该卡已记录的 `P_peak/BW_peak` 比较。缓存 hit 不能替代 DRAM bytes。
3. 若 K2 的 waves/occupancy 偏低而 DRAM/SM 均非饱和，优先报告“并行度受限”；若 `I` 与 roofline 及 achieved bandwidth/compute 一致，才报告 memory-/compute-bound。

### 自动完整性结论

- **COUNTER_SET_INCOMPLETE**：至少一个 kernel 缺 FLOP/指令、DRAM bytes/sectors 或其余分类证据；报告不得声称完成 roofline 分类。
