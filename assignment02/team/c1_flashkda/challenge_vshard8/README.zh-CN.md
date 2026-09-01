# V8-P1：V8-P2 spill 后的唯一退路

## 变量表

| 变量 | 含义 | 取值 |
|---|---|---:|
| $H$ | 本卡 head 数 | 12 |
| $S_v$ | value shard 数 | 8 |
| $V_s=128/S_v$ | 每 CTA 的 value 列数 | 16 |
| $G_{K2}=H S_v$ | fixed B=1 的 K2 grid | 96 |
| $P$ | phase-6 software prefetch depth | 1 |

V8-P2 的 SM103a 正式 fixed BF16-state 实例在 job 10704 出现 12-byte store / 8-byte
load spill，按预设 gate 未启动 GPU。本目录保留相同 V=16/96-CTA 几何，只把
phase-6 ring 从 P=2 降回 P=1。它是八分片方向的最后一个低成本候选，不是绕过 gate。

新 SO 同时保留 baseline、vshard2-P2、vshard4-P2、V8-P1 和失败的 V8-P2 资源实例。
P1 必须先零 spill，再通过 small `H=1/2/4` 四状态对 baseline/torch-ref 的逐位门；
H12 四状态也必须逐位相同。只有 H12 P50/P95/P99 全胜 vshard4-P2 才扩展 sweep，
否则终止八分片方向且不进入 dispatcher。

## B300 结果：正确，但性能门失败

job 10713 从 clean `1ce47ea` 完成 SM103a fresh build。正式 fixed BF16-state
V8-P1 实例为 58 registers、9 barriers、0 stack、0 spill；同一 SO 中保留的 V8-P2
实例仍复现 8-byte stack、12-byte store spill 与 8-byte load spill。

small `T=256,H=1/2/4` 的 none、BF16-both、FP32-both、FP32-final-only 四种 raw ABI
contract，V8-P1 的 output/final state 对 baseline 和 pinned Torch reference 均逐位相同；
H12 四种 contract 对 baseline 也全部逐位相同。性能门采用 1000 个四路循环 CUDA-event
样本：

| state contract | vshard4-P2 P50 (ms) | V8-P1 P50 (ms) | V8-P1 相对慢 | P50/P95/P99 胜出 |
|---|---:|---:|---:|---|
| none | 0.528640 | 0.599200 | 13.35% | 否/否/否 |
| BF16 both | 0.529856 | 0.578528 | 9.19% | 否/否/否 |
| FP32 both | 0.532192 | 0.595520 | 11.90% | 否/否/否 |
| FP32 final only | 0.523600 | 0.583424 | 11.43% | 否/否/否 |

因此按预先写定的停止门，不做 H1–18 sweep 或 NCU，不把 V8 接入 dispatcher。V8-P1
证明 8 分片几何可以保持现有数值路径，但 96 CTA 的额外并行度不足以抵消更细分片和
P1 数据路径的开销。

可复核证据：

- `results/c1_vshard8_p1_ptxas_b300_r1_job10713.json`
- `results/c1_vshard8_p1_b300_sm103a_h12_r1_small_all_contracts.json`
- `results/c1_vshard8_p1_b300_sm103a_h12_r1_h12_all_contracts.json`
- `results/c1_vshard8_p1_b300_sm103a_h12_r1_job10713.log`
