# C2 Harness 验收协议（挑战实现前）

## 范围与冻结点

本文档是 Team C2 在设计任何自定义 CUDA/Triton kernel 前冻结的验收协议。当前
`harness/` 仅能生成数据、运行 vLLM `d4da0c5` 快照的 Triton decode 基线、给出
独立 FP32 参考并做测量；它不包含挑战 kernel。测量顺序固定为：先在 B300 对
基线跑 `profile`，记录瓶颈，再讨论或实现挑战方案。

## 变量与固定形状

| 符号 | 含义 | 本验收的值/范围 |
| --- | --- | --- |
| \(B\) | 同时 decode 的 request 数（batch） | \(\{1,4,8,16\}\) |
| \(Q_H\) | query attention head 数 | 64 |
| \(K_H\) | KV head 数 | 4 |
| \(G=Q_H/K_H\) | 每个 KV head 服务的 GQA query-head 数 | 16 |
| \(D\) | 每个 head 的通道数 | 128 |
| \(P\) | KV page / sparse block 的 token 数 | 128 |
| \(K\) | 每个 \((K_H,q)\) 选中的逻辑 page 数 | 16 |
| \(L\) | 一个 query 实际参与 softmax 的 token 数 | \(K\cdot P=2048\)（末页可由 causal mask 截断） |
| \(S_b\) | 第 \(b\) 个 request 的当前 KV 序列长度 | 随机整数，\([2048,4096]\) |
| \(q\) | query 张量 | `[B, Q_H, D]`（decode_query_len=1 时） |
| `kv_cache` | 物理 page KV 缓存 | `[num_pages, K_H, P, 2D]`；前 \(D\) 为 K，后 \(D\) 为 V |
| `block_table` | 逻辑 block 到物理 page 的映射 | `[B, ceil(4096/P)]`，随机全局置换 |
| `topk_idx` | 每个 KV head/query 的逻辑 block 选择 | `[K_H, B, K]`，无重复且均可见 |
| \(s=1/\sqrt D\) | attention 缩放 | \(1/\sqrt{128}\) |

`decode_query_len=1` 是测量默认值；CLI 允许增大它做 ABI 回归，但不替代上述
官方小 batch decode 成绩。随机数使用固定基准 seed `20260819`，每个 batch 使用
`seed+B`；所有张量在目标 CUDA 设备上用同一个 `torch.Generator` 生成。KV 的
逻辑-to-物理映射是全局随机且一一对应，避免“恰好物理 page 与逻辑 block 同号”
掩盖两级间接寻址错误。

## 参考实现与精度判定

参考实现不调用 Triton、vLLM 或 SDPA：对每个 request、KV head，将 `topk_idx`
选出的 16 个逻辑 page 经 `block_table` 取回物理 page，组为 \(G\times L\) 的
FP32 QK 矩阵，执行 causal mask、FP32 softmax 和 FP32 PV，再仅在输出时转换为
BF16。它验证的是**稀疏选择语义**，不是错误地把未选择 token 也纳入 dense
attention。

| 存储模式 | `kv_cache` | scale ABI | 基线与参考比较 | 阈值（rtol / atol） |
| --- | --- | --- | --- | --- |
| `bf16` | BF16 | 无 | BF16 输出对 FP32 参考再转 BF16 | 0.03 / 0.03 |
| `fp8-scalar` | `float8_e4m3fn` | K=0.25、V=0.5 标量 | 对同一 FP8 cache 按 scale 反量化后的参考 | 0.03 / 0.03 |
| `fp8-token` | `float8_e4m3fn` | `[K_H, num_pages*P]` 的随机 per-token/head scale | 同上；scale 索引必须使用物理 page | 0.03 / 0.03 |

FP8 不与量化前随机 BF16 KV 作比较，而是与**同一 FP8 表示经相同 scale 反量化**
的参考作比较；这样能隔离 kernel 的 scale/页表/softmax 错误。阈值以 vLLM 上游
`test_sparse_attn_fp8_scale.py` 的 `2e-2` 口径为起点，额外给一次 BF16
split-K/merge 舍入余量。挑战 kernel 不得自行放宽它；若不同，必须报告误差分位数
和归因。

必须覆盖下列组合：`B={1,4,8,16}` × `{bf16, fp8-scalar, fp8-token}`。每个组合
比较 `max_abs` 和 `mean_abs`，并由 `torch.testing.assert_close` 作为硬门。任一
失败即性能结果无效。

## 可复核命令

在 `assignment02/team/c2_msa_decode` 目录、已进入具有 Torch/Triton 的 B300
环境后：

```bash
python -m harness.cli dry-run
python -m harness.cli correctness --all-batches
python -m harness.cli benchmark --all-batches --storage-mode bf16 --warmup 20 --repetitions 100
python -m harness.cli profile --batch 1 --storage-mode bf16 --warmup 20 --profile-steps 10 \
  --trace experiment_logs/c2_baseline_b1_trace.json
```

严格资源纪律下，以上 GPU 命令由主实验调度在干净 B300 allocation 内运行，并将
PRE/POST GPU 审计、完整 stdout/stderr、环境版本和 trace 路径一起保存。`profile`
先对 B=1、4、8、16 都运行（可分别导出 trace）；只有这些基线证据写入报告后，才
可新增挑战 kernel。benchmark 采用 20 次 warmup、100 次逐次同步的 host wall-clock
样本，报告 p10/median/p90；不同实现必须使用相同 problem seed 和同一计时法。

## 适配边界和已知限制

`triton_baseline.py` 在内存中只替换 vendored `sparse_attn.py` 的两条 vLLM import：
`vllm.triton_utils` 改为上游 `triton`/`triton.language`，
`current_platform.is_arch_support_pdl()` 改为 CUDA capability shim（SM90+ 默认
启用，`MSA_BASELINE_PDL=off` 可做回归）。其余 kernel 及 Python launch 逻辑来自
快照原文，未复制到 harness。因此其最小运行依赖是：Python、Torch CUDA、Triton、
CUDA GPU；没有 vLLM 依赖。若未来快照的这两条 import 或其 PDL API 发生变化，
adapter 会明确失败，必须先更新并记录，而不能静默换成自研 kernel。

## P1 补充：跨实现 BF16 核心级公平合同（已执行）

此补充不改写上面的 snapshot 基线验收，而是为“官方 MSA 与 vendored Triton 是否在
相同语义下比较”额外冻结的合同。变量如下；每个 `B` 单独使用固定 seed，但该 `B`
内的三条路径必须共享完全相同的张量及 checksum。

| 变量 / 路径 | 冻结内容 |
| --- | --- |
| `B` | `{1,4,8,16}` |
| 数据 dtype / scale | Q、K、V 都为 BF16；无 FP8 scale |
| 稀疏语义 | 同一随机二维 `block_table`、同一排序 `topk`、同一 `seq_lens`，`H_q/H_kv/D/P/K_top=64/4/128/128/16` |
| three paths | vendored vLLM `d4da0c5` source wrapper、persistent-workspace prepared Triton（BF16 selected `C=1`）、official MiniMax MSA `80434d7` core（CUTLASS `eb61c91`） |
| 正确性 | 三者同对独立 FP32 selected-page causal-attention oracle；`rtol=atol=0.03`，finite 且 PASS 才能计时 |
| 输出与计时 | caller-owned BF16 output；warmup 20，单 stream 100 个 per-call CUDA-event 样本；报告 p10/median/p90 |
| 计时外项 | K/V physical ABI bridge、official MSA plan/workspace 的建立、oracle 与正确性；source wrapper 的调用内 `o_partial/lse` 分配则保留在其调用语义内 |

vLLM 的物理 ABI 为 `[physical,H_kv,P,2D]`（K\|V），official core 的 ABI 是各自
contiguous 的 K、V `[physical,H_kv,P,D]`；因此 bridge 只能在三条路径均不计时的
前提下完成，同时 `block_table` 保持二维、`topk` 从 `[H_kv,Q,K]` 改为
`[Q,H_kv,K]`。这保证相同 attention 语义，却**不**使两侧成为同一 full-vLLM pin
或同一 workspace 生命周期的端到端服务比较。

[job 4339 JSON](../experiment_logs/c2_fair_bf16_crossover_b300_job4339.json) 和
[完整日志](../experiment_logs/c2_fair_bf16_crossover_b300_job4339.log) 满足此合同：
四个 B 的三路径均 PASS。official 的 median 相对 source wrapper 为
1.773662/1.792661/1.776991/1.742433x，相对 prepared `C=1` 为
1.149324/1.114373/1.074926/1.157270x。归因边界必须保留：official 与 snapshot
pin 不同，prepared 的 persistent workspace 与 selected `C=1` 同时改变了
workspace/merge 工作量，故这些数值只能回答等数据的 core-level 对照，不能归因为
单一指令、full-vLLM crossover 或端到端服务延迟。
