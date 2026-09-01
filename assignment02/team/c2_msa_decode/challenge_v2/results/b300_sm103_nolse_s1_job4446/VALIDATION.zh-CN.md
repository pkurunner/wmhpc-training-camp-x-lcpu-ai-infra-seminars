# C2 no-LSE B300 独立复验（job 4446）

| 变量 | 含义 | 固定值 |
|---|---|---:|
| `B` | decode batch size | `1,4,8,16` |
| `C` | selected-page split 数 | `1` |
| `G` | 每 KV head 的 GQA 分片数 | `1` |
| `W` | decode warp 数 | `4` |
| `S` | software-pipeline stage 数 | `3` |
| `P` | PDL | off |

本目录保存 NVIDIA B300 SXM6 AC / SM103 上的同源码固定配置复验。PRE/POST 均为
`0 MiB` 且 compute-apps 为空；四个 batch 都通过独立 FP32 oracle 与 source-hash 门。
wrapper 按设计以 `RC=3` 返回，因为 strict 10% 门并未在所有 batch 成立。

| `B` | control / candidate (us) | speedup | 结论 |
|---:|---:|---:|---|
| 1 | 30.560 / 28.448 | 1.074241x | 正收益但不足 10% |
| 4 | 28.448 / 30.496 | 0.932844x | 负优化 |
| 8 | 28.480 / 30.528 | 0.932914x | 负优化 |
| 16 | 30.464 / 32.544 | 0.936087x | 负优化 |

因此该 no-LSE 配置只在 RTX 5090 的现有证据中达到目标；B300 继续选择 current
prepared 实现。原始 JSON、stdout 和完整 audit log 均在本目录中。
