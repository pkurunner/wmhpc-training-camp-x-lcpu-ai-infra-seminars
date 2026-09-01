# 严格 BF16 tensor-contraction roofline

这个目录闭合报告中原先尚未定义的“有效 FLOP”口径。分析只统计 NCU 直接给出的
dense BF16→FP32 HMMA FLOP，并使用同一个 kernel replay 的 DRAM read+write bytes 和
duration；不把标量、SFU、控制或地址操作武断地折算成 FLOP。

## 变量表

| 变量 | 含义 |
|---|---|
| $B$ | batch 数 |
| $T$ | 每个 batch 的 token 数 |
| $H$ | 本卡 head 数 |
| $C$ | chunk 长度 |
| $D$ | key/value head dimension |
| $N=BHT/C$ | `(batch, head, chunk)` tile 数 |
| $F_{TC}$ | dense BF16 HMMA tensor FLOP 数 |
| $Q_{DRAM}$ | NCU 实测 DRAM read+write bytes |
| $I_{TC}=F_{TC}/Q_{DRAM}$ | tensor arithmetic intensity |

## 复现

在仓库根目录执行：

```bash
python assignment02/team/c1_flashkda/challenge_roofline/analyze_roofline.py
```

脚本会验证 K1/K2 的源码推导整数与 NCU HMMA counter 完全相等，然后在 `results/`
生成机器可读 JSON 和带变量表、公式、结论边界的 Markdown。输入是同一版 H12
vshard4-P2 的 NCU Full raw CSV；此分析不需要再次占用 GPU。

K1 的理论量为 $N(4C^2D)$；K2 为 $N(6CD^2+4C^2D)$。若任一理论整数不等于
NCU counter，脚本以非零状态退出。

## 解释边界

roofline 的 `compute` 分支只表示 arithmetic intensity 越过 ridge，不自动表示算力
饱和。K2 仍须与小 grid、低 SM 利用率证据联合解释。NCU replay duration 只用于相同
kernel 的 roofline 点，不替代 CUDA-event full-call P50。
