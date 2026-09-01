# Packed-varlen 的固定矩阵确认与精确发布

本目录保存 raw-wrapper confirmation、fresh raw release、真实 FLA public `chunk_kda` 集成门，以及后续的 public-overhead 诊断、one-shot handoff candidate 与 r6 production freeze。它们只裁决四个固定 packed-varlen layout 的逐 state-contract cell，不把结果外推到未测 shape；最终发布集合必须取 raw release 与 public-FLA performance gate 的逐 cell 交集，diagnostic/candidate 本身没有发布权限。

## 变量

| 变量 | 含义 |
| --- | --- |
| `B` | packed 表示的 batch；本实验固定为 `1` |
| `N` | `cu_seqlens` 中的逻辑 sequence 数量 |
| `H` | attention head 数；固定为 `12` |
| `T_total` | 所有逻辑 sequence 长度之和 |
| `K`, `V` | key/value 维度；均固定为 `128` |
| `L_i` | 第 `i` 条逻辑 sequence 的长度，`i=0,…,N-1` |
| `o_j` | CPU-authoritative offset，`o_j=sum_{i<j}L_i`；要求 `o_0=0`、`o_N=T_total` 且严格递增 |
| `d_varlen` | 由原始 CPU `int64` offsets tensor 签发的进程内不透明 descriptor；不可 pickle |
| `h_cache` | canonical offsets 的 device-cache 命中标志；hit 必须等待 publication event |
| `e_pub` | offsets H2D 完成后记录的 CUDA publication event；graph capture 下 hit/miss 都 fail-close |
| `P50/P95/P99` | 由一个 variant 的 1000 个 CUDA-event 原始样本线性插值得出的延迟分位数 |
| `m` | 预期 winner 相对同一分位 runner-up 的裕量，`runner_up / winner - 1` |
| `C_i`,`P_i` | 同一个配对下标 `i` 的 C1、pinned public-call CUDA-event 时延 |
| `d_i=C_i-P_i` | 同下标配对差；负值表示 C1 更快 |
| `t_PC1`, `t_DC1` | public registry/direct wrapper 的 C1 单次延迟 |
| `t_PPin`, `t_DPin` | public registry/direct wrapper 的 pinned 单次延迟 |
| `Δ_prep` | C1 特有 public preflight 差分，`(t_PC1-t_DC1)-(t_PPin-t_DPin)` |

其中 `T_total = sum_i L_i`。每个 path 的测量按 baseline、v2、v4 三路径 cyclic rotation 交错；每个 CUDA event 内恰好有一次 wrapper 调用。

## 固定矩阵与门

| case | `L_i` | `N` | `T_total` | `none` | `fp32_final_only` | `fp32_both` |
| --- | --- | ---: | ---: | --- | --- | --- |
| `equal_n2_h12_t2048` | `[2048, 2048]` | 2 | 4096 | baseline（public pinned 更快） | baseline（public pinned 更快） | baseline（public pinned 更快） |
| `equal_n4_h12_t2048` | `[2048, 2048, 2048, 2048]` | 4 | 8192 | baseline（public pinned 更快） | baseline（public pinned 更快） | baseline（raw release 未过门） |
| `mixed_n6_h12_t8192` | `[17, 511, 1024, 1300, 2049, 3291]` | 6 | 8192 | baseline（public pinned 更快） | baseline（public pinned 更快） | record-only baseline |
| `skew_n6_h12_t12288` | `[1, 1, 1, 1, 1, 12283]` | 6 | 12288 | **v2** | **v2** | baseline（handoff candidate r3 的 P95/P99 仍不稳定） |

raw confirmation 阶段共有 11 个正向 cell。基础随机种子固定为 `20260829`；每个正向 cell 都做两个独立 repeat；每个 repeat 固定 warmup=100、每路径 samples=1000。对 P50、P95、P99，预期 winner 必须全部一致，且每个分位的 `m >= 2%`。最终生产资格还必须与 fresh raw release 和完整 public-FLA performance 逐 cell 求交；某 cell 失败只淘汰该 cell，不扩大到其他未测 shape。

`mixed_n6_h12_t8192/fp32_both` 也测两个 repeat 并保存原始样本，但仅是 discovery record，永远不纳入 11-cell gate，未来动作保持 baseline。

## 正确性与身份门

- 四个 case × `none`、`bf16_both`、`fp32_both`、`fp32_final_only`：v2 和 v4 的 output 与（存在时的）final state 必须逐 bit 等于 baseline。
- 四个 case × 三个 FLA public contract：baseline 必须逐 bit 等于 `REFERENCE_ROOT/tests/torch_ref.py` 的 pinned Torch reference。
- 每次 correctness 调用以及每个完整 benchmark repeat 后，`q/k/v/g/beta/A_log/dt_bias/cu_seqlens` 都必须与调用前 snapshot 逐 bit 相同。
- GPU 必须是名称含 B300、compute capability `10.3`、148 SM 的单卡；加载的 extension SHA256 必须为 `8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005`。
- one-shot handoff 只复用 verifier 签发的 descriptor/key；body 先清 TLS plan，再重读 CPU tuple 做 freshness 检查。tensor identity、标量/flag、未知 kwargs 或 CPU 内容任一不匹配都回完整 prepare；GPU structure/cache/capture/extension/symbol/variant 仍由 `auto_dispatch.fwd` 复核。
- handoff 后的 production freeze 必须用非计时 spy 证明 public C1 `_prepare_varlen` delta=1、pinned delta=0，并在任何性能样本前证明没有 instance method shadow。
- patched/reference 两个工作树必须处于固定 commit；reference 必须 clean。patched 树因包含待测 kernel 源码而允许预期的 dirty 状态，但 owned runner、实际 import 的 Python wrapper、共享 runner、reference 与 harness 均以预注册 SHA256 fail-close；shell 与 README 的 SHA 则由作业日志记录并由父流程复核。
- stdlib-only independent analyzer 从 JSON 的 `raw_samples_ms` 重算所有 summary、winner 和 margin，并逐字段交叉核对 runner 汇总；最终 gate 只使用独立重算值。

## 使用

仅规划机描述（不导入 Torch/CUDA）：

```powershell
python assignment02/team/c1_flashkda/challenge_varlen_dispatch/run_varlen_dispatch_confirmation.py --describe --json $env:TEMP/c1_varlen_dispatch_confirmation.describe.json
```

GPU 运行只能由具备调度授权的父流程在空闲单 B300 上执行；该 shell 需要 `A02_ROOT`、`PATCHED_ROOT`、`REFERENCE_ROOT`、`LABEL`，且必须同时设置授权环境变量和参数：

```bash
C1_VARLEN_DISPATCH_GPU_AUTHORIZED=1 \
bash challenge_varlen_dispatch/run_clean_varlen_dispatch_confirmation_audit.sh --authorized-by-parent
```

handoff 后的两项生产冻结使用独立授权入口，并在同一 clean 作业内实际执行 stdlib-only analyzer：

```bash
C1_VARLEN_FLA_INTEGRATION_R6_GPU_AUTHORIZED=1 \
bash challenge_varlen_dispatch/run_clean_varlen_fla_integration_r6_audit.sh --authorized-by-parent
```

confirmation shell 会做初始 PRE、reference-helper 预加载后的第二次 PRE、AFTER、POST clean-GPU 检查；r6 shell 则执行 PRE、runner 内 pre-Torch gate、AFTER-runner、AFTER-analyzer、POST。两者都记录并 pin 所有实际运行依赖/source SHA256；r6 另记录自身 runner/analyzer/shell 与两套 CPU tests，不把 README 当作运行依赖。它们不会重编 FlashKDA 或修改源码；pinned Torch reference 自带的既有 cached CUDA helper 会按固定路径直接加载，其 binary SHA256 必须等于预注册值。runner 仅在加载已锁定的 `torch_ref.py` 时拦截其中唯一一次 `load_inline('sigmoid_ext')` 并返回该 helper，杜绝 Ninja/NVCC。cache 缺失、二进制身份漂移或拦截次数/名称变化时直接 fail-close。

## 当前证据与边界

confirmation job 11212 已在同一 B300、同一审计 SO 上完成：11/11 promotion cell 的两个独立 repeat 均通过 P50/P95/P99 winner 与 2% margin 门；所有 correctness/reference/input-immutability gate 通过，record-only cell 没有获得发布资格。结果 JSON SHA256 为 `447d7f49a624fa5b92adc431b350450f99d53f5b20f3a07a1bf4d2f76a64e51c`，PRE/AFTER/POST 均为 0 MiB，`FINAL_RC=0`。

raw release job 11393 已以新 seed/new allocation 完成，PRE/AFTER/POST 均为 0 MiB、`FINAL_RC=0`。原始 JSON SHA256 为 `338d0b271e153992af8d0ee37ad20dc359cecb97658ca0c23c610541cc6f5838`；独立 analyzer 从 frozen discovery、两次 confirmation 与两次新 repeat 的原始样本重算后发布 10/11 cell（audit SHA256 `9f6174f15a1d06518b6da65d221d9ad9eb42e748d9de8236e4295af2241754a1`）。`equal_n4_h12_t2048/fp32_both` 的两次新轮最小 margin 为 0.91%/1.23%，故已从 dispatcher 删除并显式回 pinned baseline。

public-FLA 失败史全部保留：job 11395 因 Dynamo wrapper 的 sourcefile 身份误判 fail-close；job 11454 因 runner 在 verifier 前未进入 inference mode 停止；job 11461 已持久化 10 个正例与两个 fallback 后，在 negative helper 暴露缺失的局部 `import torch`。这些均为 runner 问题，未进入发布裁决。

修正后的 clean job **11466** 完整 `FINAL_RC=0`，PRE/AFTER/POST 均为 0 MiB。10/10 public C1 correctness cell 与 pinned/reference 逐位一致、输入保持不变；两个预注册 fallback、其余 negative、CPU authority、双 stream、capture cold/hot、hot-sync 与 fixed control 全部通过。原始 JSON SHA256 为 `a608bb83226b7cf6f433c05c77b876e2374cab64e50ac5750f81d209d784119d`；独立审计 SHA256 为 `644007e13ecbe4b14ddb24dd928c9662dfdaef1aef92e00a95cf0982656a16e4`，并从 40,000 个 timed public calls 的 raw samples 重算通过。

public performance 只发布 `skew_n6_h12_t12288/none` 与 `fp32_final_only`，r4 两轮全分位最小 margin 分别为 8.257% 与 12.503%。`skew/fp32_both` 虽由 C1 胜出，但 repeat 0 的 P99 margin 只有 1.9765%，因此回 baseline；其余七个 raw 候选均由完整 pinned public path 更快。证据为 [public r4 JSON](results/c1_varlen_fla_integration_b300_sm103a_public_r4.json) 与 [r4 独立审计](results/c1_varlen_fla_integration_b300_sm103a_public_r4.independent_audit.json)。

dispatcher 收缩后，最终 clean freeze job **11479** 完整 `FINAL_RC=0`，PRE/AFTER/POST 均为 0 MiB。schema 3 runner 验证 2/2 C1 route 和 10/10 pinned fallback；每个 fallback 都有精确 reject reason、pinned verifier/spy、direct-pinned↔public↔Torch-reference 逐位一致、final ABI 和三路径输入不变性。17 个 negative、三类 cache 观察与 fixed control 全通过；两项性能共 8000 个 timed public calls，再次以最小 margin 12.469%/5.719% 发布。JSON SHA256 为 `bebd25650a17aebb38114653d593f7043077dabab575b73ac67a38f7dbdabb41`，独立审计 SHA256 为 `2eb6527497f8bcb7b9b7f5d9f6c9c84794bd86b6f4d7fefe7d494a4490d0fd65`，见 [public r5 JSON](results/c1_varlen_fla_integration_b300_sm103a_public_r5.json) 与 [r5 独立审计](results/c1_varlen_fla_integration_b300_sm103a_public_r5.independent_audit.json)。任意新 offsets/state、其他 GPU 身份、CUDA graph replay，以及没有 opt-in CPU descriptor 的调用仍保持 pre-launch fail-close/fallback，不能解释为通用 varlen dispatch 已完成。

diagnostic job **11493** 用 equal/mixed/skew 三个 `none` layout、两轮、四路径各 1000 样本确认原 public C1/direct C1/public pinned/direct pinned 的 `_prepare_varlen` 次数为 2/1/0/0；独立重算的 `Δ_prep` 均值约为 30–32 μs。JSON SHA256 为 `012c587c1417c20e65171b506d50a0f62bd7dba630d1c4005f2ede9b82f854aa`，审计 SHA256 为 `823321e3e2cecd0d915736d4006f26dd65ad4f6cd56f5fd26b2eb024b7db0e65`，clean log SHA256 为 `6c346ec9fd842fdbbb4183b23aa94e894bc980c2645e1b29316eab0f7552e30c`；见 [overhead JSON](results/c1_varlen_public_overhead_b300_sm103a_diag_r1.json)、[overhead 审计](results/c1_varlen_public_overhead_b300_sm103a_diag_r1.independent_audit.json) 与 [overhead clean 日志](results/c1_varlen_public_overhead_b300_sm103a_diag_r1_job11493.log)。首版 spy restore 后来发现会遗留 bound-method shadow，因此微秒绝对值只用于选择优化方向；candidate/r6 已用 no-shadow gate 修正。

handoff candidate clean job **11538** 的 10-cell correctness、负控、map restoration 和 40,000 个 timed calls 均完成，PRE/AFTER/POST=0 MiB、`FINAL_RC=0`。独立审计只让现有 `none`、`fp32_final_only` 通过；`skew/fp32_both` 的 repeat 0 P95/P99 由 pinned 胜出，故不扩白名单。JSON SHA256 为 `2d50d219c5eb33cde726331cfc7ee613c2a7959918f67675e344e29fead23158`，审计 SHA256 为 `61c352b74049969d5213f520fad9a55194c7192819ed923bf83629f78a2078fd`，clean log SHA256 为 `2da68def909c7d07b8927d67a033b75231934d73404b74a9482fc5d085cf4d71`；见 [candidate r3 JSON](results/c1_varlen_fla_handoff_candidate_b300_sm103a_r3.json)、[candidate r3 审计](results/c1_varlen_fla_handoff_candidate_b300_sm103a_r3.independent_audit.json) 与 [candidate r3 clean 日志](results/c1_varlen_fla_handoff_candidate_b300_sm103a_r3_job11538.log)。

为定位这个失败，stdlib-only [历史尾部重算器](analyze_varlen_fla_fp32_tail_history.py) 固定校验 r1/r3 raw SHA，并独立重算 paired delta、parity、100-sample blocks 与 1.20 ms 阈值计数。r3 repeat 0 的 C1 `>1.20 ms` 共有 108 个，集中于下标 359–514；同轮 pinned `>1.20 ms` 有 132 个，且最重的 block 4 为 C1 64 个、pinned 74 个。r1 两轮与 r3 repeat 1 没有复现该宽簇，因此它证明的是共享时间段尾部而非 handoff/C1 kernel 因果。脚本/结果 SHA256 为 `530eeb70c71fb49e928d522f4becf62930ee7594a53f96101db7260fcaa41b4c` / `a3dc8c50391fd4ec62e1755acbcd29cd5b6c5c6115d67a1d6dc87bea458c85ad`，见 [历史尾部 JSON](results/c1_varlen_fla_fp32_tail_history_r1_r3.json)；该 artifact 仅支持讨论，不具发布权限。

fresh-process v1 job **11679** 虽通过 3 main PID × 2 repeat 的执行完整性、48 项 exact、map 恢复和清卡门，但独立复核发现其 CUDA event 内还包含 route-count dictionary、decision 检查与 spy wrapper；它与 candidate r3“区间内只有真实 `_call`”的计时契约不同。因此 [v1 审计](results/c1_varlen_fla_fp32_tail_diagnosis_b300_sm103a_r1.independent_audit.json) 只能保留为仪器化相对观察，不能支持绝对尾部、内核因果或发布。

修正后的 v2 runner/analyzer/shell SHA256 为 `3b0342af8aca96e85bef75228ae71b1a1e401484373dc42aedc204c2ed533fb0` / `0adb9f93e879d80d5287ff460b9b559935dd04a44853f0dff13363782449c100` / `f339373a54c99e816617d7d40066883f4073978c746507d0d78e576177809402`。它把 path 选择、route accounting 与 event 构造移到区间外，区间内唯一调用为真实 `_call`。job **11767** 的 6/6 repeat 相对门全部通过：C1 在 P50/P95/P99 均胜，最小 `m=15.468%`，六轮 paired `d` P99 都小于 0；然而当次 B300 固定在 graphics/SM 1095 MHz，C1 P99 为 1.747–1.751 ms，6000 个 C1 与 6000 个 pinned 样本全部超过预注册的 1.20 ms 绝对阈值。三个 fresh PID 都只在 repeat 0 的下标 221 出现一个 `C_i>P_i` 联合命中。独立审计分类为 `absolute_or_shared_scale_failure`，并给出严格布尔 `second_allocation_decision.eligible=false`；因此没有第二 allocation，也没有白名单变化。

job 11767 `FINAL_RC=0`，PRE/BETWEEN/POST 均为 0 MiB。远端 [v2 审计](results/c1_varlen_fla_fp32_tail_diagnosis_v2_b300_sm103a_v2_r1.independent_audit.json) SHA256 为 `a8f2e21e8bb8bc224f3cc8895d104f9d27f3067688d7b5d6026c6523aabf506a`；[本地复算](results/c1_varlen_fla_fp32_tail_diagnosis_v2_b300_sm103a_v2_r1.local_recompute.json) 从三份 raw JSON 和 telemetry sidecar 重新生成，除路径字符串外逐字段一致。完整边界见 [v2 clean 日志](results/c1_varlen_fla_fp32_tail_diagnosis_v2_b300_sm103a_v2_r1_job11767.log)。低时钟与两条路径整体抬高相符，但不是因果证明，也不能用于事后放宽已失败的绝对门。

最终 handoff production freeze job **11590** 使用 schema 4，GPU 前精确检查生产 map 仍只有两项；两个正例的 public C1 `_prepare_varlen` delta=1、pinned delta=0。2/2 C1、10/10 fallback、17 negatives 与全部 cache/capture/two-stream/hot-sync/fixed control 通过；8000 个 timed calls 的两轮最小 margin 为 none 21.499%/22.003%、FP32-final-only 13.163%/15.616%。同作业内 stdlib-only analyzer 从 raw samples 重算得到 exact 2 release/0 failed；PRE、AFTER-runner、AFTER-analyzer、POST=0 MiB，`FINAL_RC=0`。JSON/audit/log SHA256 分别为 `e8fc8a09bdf9b7cac547d25ae81f8e9c2ad909e46a9d813a0721d1f800ced9f8`、`dd8bef3919a937a51f0c2c98f0ae175ee5978c56df7491b0cac47f921cef2317`、`fff6d60d6f7b6e628ef5aef205ea82001e5b6072bb8bcad0936afb6967cb4742`，见 [r6 JSON](results/c1_varlen_fla_integration_r6_b300_sm103a_public_r6.json)、[r6 审计](results/c1_varlen_fla_integration_r6_b300_sm103a_public_r6.independent_audit.json) 与 [r6 日志](results/c1_varlen_fla_integration_r6_b300_sm103a_public_r6_job11590.log)。实现 backend/metadata SHA256 为 `8555995c04ecd666a580ddee02eae1d34820ef1a601cbad5d10f9c6b8505974b` / `f89a97ba284c7a24a3df54efca7bb60eb70f9cbda659200bd6cb3e8dfdaf4ccd`；dispatcher SHA256 仍为 `2b817adb7d21d1f223e8df4616eeccd74e34a5b1944492211f0f0254147ba883`。边界仍是 exact B300 tuple/state 子集，不是任意 varlen、跨架构、真实模型或 full TP8 证明。
