# H12 v4P2 的严格 BF16 tensor-contraction roofline

本页由 `analyze_roofline.py` 直接从同一版候选的 NCU Full raw CSV 生成。
FLOP 口径只包含 NCU 计数的 dense BF16→FP32 HMMA；标量、SFU、控制和地址计算
不被强行换算成 FLOP。因此这是可复核的 tensor-contraction roofline，而不是含糊的
“所有操作总 FLOP”模型。duration 是 NCU replay duration，不替代 full-call event P50。

## 变量表

| 变量 | 含义 | 本次值 |
|---|---|---:|
| $B$ | batch 数 | 1 |
| $T$ | 每个 batch 的 token 数 | 8192 |
| $H$ | 本卡 head 数 | 12 |
| $C$ | chunk 长度 | 16 |
| $D$ | key/value head dimension | 128 |
| $N$ | `(batch, head, chunk)` tile 数 | $B H T/C$ |
| $F_{TC}$ | NCU dense BF16 HMMA tensor FLOP 数 | 分阶段见下 |
| $Q_{DRAM}$ | NCU DRAM read+write bytes | 分阶段见下 |
| $I_{TC}$ | tensor arithmetic intensity | $F_{TC}/Q_{DRAM}$ |
| $P_{TC}$ | 实测 tensor FLOP/s | $F_{TC}/t$ |
| $P_{peak}$ | NCU BF16 HMMA sustained peak | 分阶段回读 |
| $BW_{peak}$ | NCU DRAM sustained peak | 分阶段回读 |
| $I_{ridge}$ | roofline ridge point | $P_{peak}/BW_{peak}$ |

## FLOP 闭合

K1 有两个 `(C,D)×(D,C)` contraction；K2 有三个 D-wide contraction 和两个
C-wide contraction。以 FMA=2 FLOP 计：

$$N=B H T/C,$$

$$F_{TC,K1}=N(4C^2D),$$

$$F_{TC,K2}=N(6CD^2+4C^2D).$$

两式的理论整数必须与 NCU `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_
sparsity_off.sum` 完全相等，否则脚本直接失败。

## 分阶段结果

| phase | $F_{TC}$ | DRAM bytes | $I_{TC}$ (F/B) | achieved (TF/s) | ridge (F/B) | selected roof (TF/s) | efficiency | branch |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| K1_prepare | 805,306,368 | 109,220,864 | 7.373192 | 19.418074 | 75.313428 | 56.489573 | 34.3746% | memory |
| K2_recurrence | 10,468,982,784 | 126,834,432 | 82.540542 | 22.240361 | 75.512476 | 579.304993 | 3.8391% | compute |

K1 位于 memory-roof 分支。K2 的 intensity 越过 ridge、落在 compute-roof 分支，
但这不等于“算力已饱和”：其 roof efficiency 很低，结合小 grid/低并行度 counter，
应解释为延迟/并行度受限的 compute-side 点，而非接近峰值的 compute saturation。

## 可复核性

- raw CSV SHA-256：`dabc28857805b77948774643a8abe838918a05dfc3ceef00b0d25e0c56f3edd6`
- raw CSV：`D:\软微\26春季\科研\Linux_HPC\assignment02\team\c1_flashkda\challenge_vshard4_prefetch2\results\c1_h12_ncu_b300_sm103a_h12_r1_vshard4_p2_full_job10085_raw.csv`
- JSON 保留所有原始量、单位换算、理论整数和派生量。
