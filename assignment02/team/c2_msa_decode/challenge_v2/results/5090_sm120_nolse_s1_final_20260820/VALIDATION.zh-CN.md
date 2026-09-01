# 5090 SM120：C2 BF16 C=1 no-LSE 冻结终验索引

本目录只记录 RTX 5090 / SM120 的这一次 source-bound 终验；它不代表 B300，也不
代表 FP8 scalar/token。GPU UUID、PRE/POST 显存、`compute-apps` 与源码 SHA256 均在
[`c2_nolse_abba_clean_audit_20260820T015542Z.log`](c2_nolse_abba_clean_audit_20260820T015542Z.log)。
该作业为 Slurm job 7002，`FINAL_RC=0`。

| 变量 | 含义 | 固定值 |
| --- | --- | --- |
| `B` | decode batch | `1,4,8,16` |
| `C` | selected-page split 数 | `1` |
| `G` | 每 KV head 的 Q-head CTA 分片数 | `1` |
| `s` | candidate stage 数 | `3` |
| `t` | 单调用 CUDA-event 延迟 | 控制/候选各 202 样本 |

候选在 B=4 独立 gate 前冻结为 `G=1, warps=4, stages=3, PDL=off,
maxnreg=none`；控制是 current prepared BF16 C=1（`warps=4, stages=3,
PDL=auto`）。二者共享每个 B 的固定 seed、caller-owned output 和计时外 persistent
workspace，且均先通过独立 FP32 selected-page causal-attention oracle。

每一份 JSON 使用 101 个 `ABBA` 对：`control→candidate→candidate→control`，因此
相邻启动顺序不会被隐去。终验结果如下：

| B | control / candidate (us) | 合并 speedup | AB / BA speedup | 最大绝对误差 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 84.560 / 69.392 | 1.2186x | 1.2050x / 1.2290x | `6.10e-5` |
| 4 | 116.160 / 90.368 | 1.2854x | 1.2585x / 1.3216x | `3.05e-5` |
| 8 | 83.744 / 67.392 | 1.2426x | 1.2218x / 1.2640x | `3.05e-5` |
| 16 | 83.808 / 67.456 | 1.2424x | 1.2372x / 1.2483x | `6.10e-5` |

随后在**不运行 GPU**的情况下，运行后加固的离线验证器
[`run_nolse_abba_clean_audit.sh`](../../run_nolse_abba_clean_audit.sh) 对四份 JSON
逐项严格复核：schema、冻结 config、BF16/C=1 合同、source hash 字段、202 样本、
FP32 finite gate、AB/BA 数据、speedup 重算与 `strict_10_percent_target_met=true`
均通过。加固后的 wrapper 哈希与历史运行日志中的 wrapper 哈希不同；测量核心
kernel/CLI 的哈希不变且逐项匹配。核心源码 SHA256 为：

- `c1_no_lse.py`: `7c638d53baad41582756dc2411fa0ede33ddb253b96440bfcb072331855bd9aa`
- `c1_no_lse_abba_cli.py`: `a06e116b05dfcc894a4f33eb503468d58e0fef4e99f195c1269fbfb7d163271a`
- `prepared_tuned.py`: `2989da46d15a1d2c183c18cd1b9a93831a98a21a236653b56847f6fafb01650b`

这些值已在本地、5090 和 B300 main 的最终源文件上逐一比对。配置选择发生在
[job 7001 的 B=4 冻结门](../5090_sm120_nolse_s1_freeze_gate_job7001/VALIDATION.zh-CN.md)
之前；随后 [B300 job 4446](../b300_sm103_nolse_s1_job4446/VALIDATION.zh-CN.md) 已按同一核心源码和
AB/BA 合同独立复验：B1 仅 `1.074241x`，B4/B8/B16 分别为
`0.932844/0.932914/0.936087x`，所以 B300 保留 current prepared 实现。

JSON 保存每组的 p10/median/p90 与 AB/BA 分组统计，没有保存全部 202 个 event 原始
样本；因此可离线重算 speedup 和门槛，但不能从 JSON 独立重建中位数。这是证据粒度
边界，不改变上述 clean、source-bound 结论。
