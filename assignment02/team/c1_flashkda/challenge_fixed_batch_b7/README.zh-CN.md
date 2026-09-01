# B=7, H=12, T=2048 fixed-batch discovery

这是一个只读、单 allocation 的补点实验。它只直接调用已固定的 raw ABI wrapper，绝不导入
`auto_dispatch`、修改 production map、生成 release 结论或提交第二个 allocation。其唯一目的
是判断精确形状 `B=7,H=12,T=2048,K=V=128` 是否值得由父流程另行授权复核；这不是 dispatch
发布实验。

| 记号 | 含义 | 本实验取值 |
| --- | --- | --- |
| `B` | fixed-batch 中的序列数 | 7 |
| `H` | attention/KDA 头数 | 12 |
| `T` | 每个序列的 token 数 | 2048 |
| `K,V` | key/value 的通道维度 | 128, 128 |
| `M` | 并行 head-sequence 工作项数 `B×H` | 84 |

## 预注册设计

- 原始 ABI correctness 覆盖四个 state contract：`none`、`bf16_both`、`fp32_both`、
  `fp32_final_only`。每项均验证 baseline、`vshard2_p2`、`vshard4_p2` 的 output，以及所有
  存在的 final state 都逐位一致；三条公开 performance contract 还对 pinned Torch reference
  作逐位核验。
- 每次 wrapper 调用均保存并核验 `q/k/v/g/beta/A_log/dt_bias/cu_seqlens/scale/lower_bound`
  输入不变，并核验 `initial_state` 不变。因此计时不能通过破坏输入 state 换取更低延迟。
- 仅对公开 performance contract `none`、`fp32_final_only`、`fp32_both` 计时三条固定路径：
  `baseline`、`vshard2_p2`、`vshard4_p2`。每个 contract 用 **2 个 fresh Python PID × 每 PID
  2 个 repeat × 每路径每 repeat 1000 个 CUDA-event sample**；每路径另有 100 次 warm-up。
  三路径的首位次按 sample/warm-up 索引循环轮换，1000 samples 下首位计数为
  baseline 334、vshard2 333、vshard4 333。
- stdlib-only analyzer 只从 raw samples 重算 P50/P95/P99、winner 和 runner-up margin，不信任
  runner 的 summary/pass 字段。它还拒绝 PID、GPU UUID、扩展 SO、源码 SHA、contract、样本数、
  形状或 ABI gate 漂移。

## 预注册裁决

每个 public contract 的四个 repeat × 三个分位必须同时满足：

1. 同一条 custom path（`vshard2_p2` 或 `vshard4_p2`）是唯一 winner；
2. 该 winner 相对每个 runner-up 的裕量均不少于 2%；
3. 全部 raw ABI / pinned-reference / 输入和初始 state 不变性 gate 通过。

只有三个 contract 全部通过，machine JSON 才标为
`eligible_for_independent_confirmation_only`。它仍不产生 production map。任何 baseline winner、
分位不一致、裕量小于 2% 或 ABI/identity gate 失败均为完整的 discovery negative：保持 baseline，
停止，不申请第二个 allocation。

## 运行边界

`run_clean_fixed_batch_b7_discovery.sh` 必须由已经取得单 B300 GPU 的父作业调用，且同时满足：

- `C1_FIXED_BATCH_B7_DISCOVERY_GPU_AUTHORIZED=1`；
- 第一个参数为 `--authorized-by-parent`；
- 已设置 `A02_ROOT`、`PATCHED_ROOT`、`REFERENCE_ROOT`、`FLA_ROOT`、
  `C1_PINNED_REFERENCE_HELPER_PATH`、`LABEL`（可另设 `PYTHON_BIN` 与 `RESULTS_DIR`）。

shell 在 PRE、两个 main PID 之间、独立 analyzer 前后、POST 都检查单可见 GPU、无 compute app 和
0 MiB 占用；它固定 patched/reference/FLA commit、预构建 extension、所有 wrapper/harness/helper
和 B7 runner/analyzer 的 SHA256。它不含 Slurm 提交、构建、NVCC、patch generator、git/source
mutation 或第二 allocation 的任何命令。

输出包括：`plan.json`、两个 main raw-sample JSON、独立 audit JSON 和带 PRE/POST 的 clean log。
结果目录在首次授权 GPU 实验前应保持为空；本目录不把计划文件或模拟数据伪装成测量证据。

## B300 实测结论（job 12216）

单个已授权 B300 allocation 已完成。两个 fresh PID 的四种 raw ABI correctness、适用的 pinned
Torch reference、输入与 initial-state 不变性 gate 全部通过；每个 public contract 均取得
`2 PID × 2 repeats × 3 paths × 1000 samples`。独立 analyzer 从 raw samples 重算得到：

| public contract | 四次 repeat 的 P50/P95/P99 共同 winner | winner 裕量范围 | 单格资格 |
| --- | --- | ---: | --- |
| `none` | `vshard2_p2` | 2.285%–3.526% | 通过 |
| `fp32_final_only` | `baseline` | 1.254%–2.209% | 失败 |
| `fp32_both` | `baseline` | 1.715%–3.024% | 失败 |

预注册发布单位是三个 public contract 的完整组，而不是事后挑选 `none`。因此最终机器裁决为
`eligible=false`、`baseline_stop_no_second_allocation`：没有申请第二 allocation，没有修改
dispatcher/production map。若以后研究 `none`，必须另起一个预注册的 none-only 跨 allocation
协议，本次 discovery 不能被追认为 release 证据。该后续协议现已在
[`challenge_fixed_batch_b7_none`](../challenge_fixed_batch_b7_none/README.zh-CN.md) 的 job12570
实际执行，并得到有效 A1 性能负门、按规则不申请 A2；它没有反向改变本目录的历史 discovery
角色。PRE/BETWEEN/POST 均为空卡，日志唯一 `FINAL_RC=0`。

| 工件 | SHA-256 |
| --- | --- |
| [plan](results/c1_fixed_batch_b7_discovery_b300_sm103a_b7_r1.plan.json) | `f7a3718570889a7b91f2360e39e4179d31eaba13c62a8cceb48858c87d8b320d` |
| [main0](results/c1_fixed_batch_b7_discovery_b300_sm103a_b7_r1_main0.json) | `5f0172c8c60c8858fc6607cbc293c346d8984f05643e331623200b98cfaa67fb` |
| [main1](results/c1_fixed_batch_b7_discovery_b300_sm103a_b7_r1_main1.json) | `6b4df1c960411233cfec6d9543f69e117bcfbfd89842afad9488bf1085810708` |
| [independent audit](results/c1_fixed_batch_b7_discovery_b300_sm103a_b7_r1.independent_audit.json) | `b5039f52506ada6d2a14c61a0cb38d35438ca464d5943f6bccd83406dc115283` |
| [clean log](results/c1_fixed_batch_b7_discovery_b300_sm103a_b7_r1_job12216.log) | `ec43f20da384899535a0f8f6e8257f580867415e95d347758ec05c7304fba5af` |

冻结 runner/analyzer SHA-256 分别为
`d36c22917eeecfa8ec23a9abda8d42fd0b87587e07852653343127e302609981` 与
`a96e4cb9ba1954f59854512bb8808691d6ca5cb3b3e164445963daa3a8f32430`。
