# C2 挑战设计：prepared workspace 与 split-K 配置搜索

## 变量表

| 符号 | 含义 | 官方形状/候选 |
| --- | --- | --- |
| \(B\) | decode request 数 | \(\{1,4,8,16\}\) |
| \(H_q\) | query head 数 | 64 |
| \(H_{kv}\) | KV head 数 | 4 |
| \(G=H_q/H_{kv}\) | GQA group 大小 | 16 |
| \(K\) | 被选中的 KV page 数 | 16 |
| \(P\) | 每个 page 的 token 数 | 128 |
| \(C\) | split-K chunk 数 | \(\{1,2,4,8,16\}\) 中的 2 的幂 |
| \(D\) | head dimension | 128 |
| \(O\) | 调用方拥有的最终输出 | `[B, H_q, D]` |
| \(O_c\) | 第 \(c\) 个 chunk 的局部输出 | `[B, H_q, D]` |
| \(\ell_c\) | 第 \(c\) 个 chunk 的 base-2 LSE | `[B, H_q]` |

## 先测量得到的证据

严格干净的 RTX 5090 job 6845 已在实现挑战前完成。每次 profile 运行 10 次，
下面是 CUDA 平均单次时间：

| B | baseline C | decode (us) | merge (us) | merge 占两 kernel 时间 | trace 中 decode grid |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 16 | 3.504 | 1.443 | 29.2% | `[16,4,1]` |
| 4 | 16 | 8.397 | 2.006 | 19.3% | `[64,4,1]` |
| 8 | 8 | 13.171 | 2.720 | 17.1% | `[64,4,1]` |
| 16 | 4 | 21.780 | 2.586 | 10.6% | `[64,4,1]` |

trace 还显示 decode 使用 128 threads、约 74 KiB shared memory、约
213 registers/thread，资源限制项为 shared memory；merge 使用 128 threads，限制项
为 warps。它说明不能凭直觉把 16 个 page 串到一个 CTA：B=1 本来就只有 64 个
decode CTA，若 \(C=1\) 会降到 4 个 CTA，极可能损失更多并行度。因此本挑战不预先
宣称“融合必胜”，而是把 \(C\) 作为必须实测的候选。

另一方面，vLLM production ABI 的 \(O\) 本来就由调用方提供；原 wrapper 只在每步
新建内部 \(O_c\) 与 \(\ell_c\)。早期 harness 的 convenience API 额外创建了一次
\(O\)，所以 job 6845 的约 129--132 us host wall-clock **只能作为探索性诊断，不能
作为最终加速对照**。最终公平 benchmark 已改为两侧都在计时外分配同生命周期
caller-owned \(O\)，挑战只节省 production 真正存在的两块内部 workspace 分配。
两 kernel 的 4.95--24.37 us 与 wrapper host 时间仍说明 CUDA graph replay 值得单独
测量，但不能据旧数据预先宣称收益。

## 实现与不变量

`PreparedSparseDecode` 直接复用 `harness.triton_baseline` 从 pin `d4da0c5`
装载的两个 JIT kernel，不修改 baseline：

1. 调用方在计时外提供 \(O\)；构造时仅一次性分配 \(O_c\)、\(\ell_c\)，后续 decode
   原位更新输入并复用固定地址；这同时适合 CUDA graph capture。
2. \(C\) 显式可选，`auto` 精确复现上游 `TARGET_GRID=256` 策略；CLI 在相同输入
   上搜索 1/2/4/8/16。
3. \(C=1\) 时局部 softmax 已是完整 softmax，令 `o_partial=output[None,...]`
   并旁路 merge；它与 caller output 共用存储，metadata 将独占 partial-output
   workspace 正确记为 0，而不是重复计入；其他 \(C\) 仍使用原 LSE merge。
4. decode/merge warp 数作为独立 launch 参数暴露，默认 4 与 baseline 相同；只有
   GPU correctness 与计时都通过的配置才可进入报告。
5. BF16、FP8 scalar、FP8 per-token/head scale 完全复用原参数 ABI，scale 索引和
   block-table 两级间接寻址不改变。

## 验收与调优命令

在 `assignment02/team/c2_msa_decode` 下、干净 GPU allocation 内运行：

```bash
python -m challenge.cli dry-run
python -m challenge.cli correctness --all-batches --all-modes --chunks auto
python -m challenge.cli benchmark --all-batches --storage-mode bf16 \
  --chunks auto 1 2 4 8 16 --warmup 20 --samples 21 --inner 20
```

这条 `benchmark` 明确标记为 `exploratory_ungated_not_for_final_claim`。baseline 和
prepared 两侧使用相同 problem、计时外预分配且同生命周期的 caller output。
`single_step_host_latency_*` 每个样本只调用一次并立即 device synchronize；
`steady_state_cuda_*` 才是 `inner` 次成组 event 的吞吐指标，二者不可混称。
默认还分别 capture-once/replay-many，并在 replay 后重新检查正确性；若运行时不支持
capture，会记录 `unsupported`，不会伪造数字。`--no-cudagraph` 可只测 eager。

完整公平 B300 sweep 显示 dtype 与 batch 存在交互，最终策略必须按实测表查找：

| 存储模式 | B=1 | B=4 | B=8 | B=16 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 1 | 1 | 1 | 1 |
| FP8 scalar | 16 | 4 | 8 | 4 |
| FP8 per-token/head | 4 | 16 | 16 | 4 |

各格均配合 persistent workspace。CLI 将这张 per-(mode,B) 表命名为 `selected`；
显式 `--chunks 1/2/4/8/16/auto` 仍保留给探索，
但不能用于最终成绩。最终流程必须先生成覆盖 `B={1,4,8,16}` × 三种存储模式的
manifest，再允许最终 benchmark：

```bash
python -m challenge.cli final-gate --chunks selected --decode-warps 4 --merge-warps 4 \
  --manifest challenge/final_gate_manifest.json
python -m challenge.cli final-benchmark --chunks selected --decode-warps 4 --merge-warps 4 \
  --manifest challenge/final_gate_manifest.json --storage-mode bf16
python -m challenge.cli profile --batch <B> --storage-mode <模式> --chunks selected \
  --trace experiment_logs/c2_challenge_b<B>_trace.json
python -m challenge.cli once --batch <B> --storage-mode <模式> --chunks selected
```

`final-gate` 先将 manifest 标为 `in_progress`；任一误差失败则改为 `failed`。
`final-benchmark` 校验 schema、策略、每行 resolved chunks、12 个 PASS 组合、
Torch/Triton/GPU 环境及文件 SHA-256，不匹配就拒绝运行。最后一条 `once` 供
Nsight Systems/Compute 包裹。
正确性仍以独立 FP32 reference、`rtol=atol=0.03` 为硬门。

## 风险与停止条件

- prepared object 绑定 tensor 地址且不可重入；多 CUDA stream 或并发 request 需要
  每个 in-flight decode 独立实例。\(C>1\) 时独占 workspace 是约
  \(C B H_q D\) 个 BF16 元素及 \(C B H_q\) 个 FP32 LSE；\(C=1\) 的 partial output
  与 caller output alias，只独占 LSE scratch。
- \(C=1\) 去掉 merge，但 FP8 的反量化/scale 路径与 batch 对并行度的需求不同；
  不得把 BF16 的选择外推到 FP8，也不得用 baseline auto 近似上述实测表。
- 改变 `num_warps` 会产生新 JIT variant，首次编译时间必须排除在 warmup 外，并且
  必须重新跑三种存储模式正确性。
- 本实现的收益若仅来自 workspace 复用，应明确报告为端到端调度/分配优化，不把它
  伪装成 kernel FLOP 提升。若所有候选在至少一个官方 B 上都没有稳定改善，则保留
  负结果和 trace，不选择性丢弃数据。
