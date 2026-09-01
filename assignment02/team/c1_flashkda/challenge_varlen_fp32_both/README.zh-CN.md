# C1 B300：skew packed-varlen FP32-both 的 relative-only 发布协议

本目录只提供实验与证据协议；不改 production dispatcher、wrapper、报告或任何已有
challenge。目标是 public FLA 的 exact packed-varlen cell
`offsets=[0,1,2,3,4,5,12288]`、`B=1,H=12,T=12288`、FP32 initial+final state，临时
让 C1 `vshard4-P2` 与 public pinned FlashKDA 对照。

| 变量 | 含义 | 固定值 |
| --- | --- | --- |
| `B,H,T` | batch、head、总 token 数 | `1,12,12288` |
| `o_i` | CPU-authoritative packed sequence offset | `(0,1,2,3,4,5,12288)` |
| `n` | sequence 数 | `6` |
| `r_q` | pinned latency / C1 latency at percentile `q` | `q∈{P50,P95,P99}` |
| `A` | 独立 Slurm allocation | `A1` 或 `A2` |
| `p` | 每 allocation 内的 fresh Python PID | `p=0,1` |
| `j` | 同一 PID 的 repeat | `j=0,1` |

## 预注册规则

每个 `A` 必须包含 **2 个不同 fresh Python PID**；每个 PID 对两个路径各做 **2 repeats
× 1000** 个原始 CUDA-event 样本。每个 event 内仅有一次真实 public `chunk_kda` 调用，
其中包含对称 backend instrumentation 及其 route-counter increment；path 选择、counter
快照/差分、decision 审计、event 构造、同步、统计均在 event 外。偶数样本 C1→pinned，
奇数样本 pinned→C1。

本协议只用相对门：每个 `A×p×j×q` 都要求
`r_q ≥ 1.02`。没有任何绝对延迟门；不能事后替换分位数或只挑某个 repeat。

每个 PID 在测时前 fail-closed 验证：

- source、SO、B300 GPU、public `fla.ops.kda.chunk_kda`、pinned helper 身份；candidate
  handoff helper 固定 SHA-256 为
  `e07481e7e66551f1f9eaafa7f1fb9db9ef193f1694ff831ee0a07802c8ffbe14`，并验证完整
  public-path runtime-import ledger 的精确 key 集及每项 SHA gate；
- 临时精确 route 的 map object identity、route spy、每样本 chosen variant；
- public C1/pinned output 与适用 final state exact，initial/input immutability；
- 每个 repeat 先在 event 外对真实 public C1 `chunk_kda` 调用临时
  `varlen_metadata.issue_descriptor` spy：恰好一次 issuance、C1 route、canonical offsets、
  descriptor 与 CPU offset object identity、跨 repeat 新鲜、output/final ABI、input
  immutability 与 spy 恢复均需成立；随后 clear cache，才进入 warm-up/timing；
- GPU-offset structural-mismatch 的 pinned takeover fallback；
- map、backend spy 与 `_prepare_varlen` class-descriptor binding 全部恢复。

NVML sidecar 也是发布前置条件而不是性能分数：全部采样必须同一 GPU、`P0`；memory
clock 必须为正且最低值达到本 allocation median 的 95%；SM clock 必须全部为正、median
至少 `1000 MHz`，且至少 80% 的样本达到该 median 的 95%。该预注册占比门允许 fresh-PID
启动/导入阶段出现有界 idle-clock 样本，却拒绝多数时间低频或持续降频；温度须 `<85°C`，功耗
为正且不超过 power limit 的 101%。任一异常把该 allocation 标为 **invalid**，并不把它解释成
性能失败。allocation analyzer 会先写 manifest，再让 `--require-pass` 以非零退出；因此
telemetry invalid 保持 `telemetry_invalid_not_performance_failure` 分类，却会停止后续 A2。
`freeze --require-eligible` 也会先写 decision，再对不 eligible 的结果返回非零。

sidecar 在 `main0` 之前、`main1` 之后分别记录 epoch-ns，并把这两个值同时写入
`*_telemetry_window.json` 与 allocation manifest。collector PID 在两个 fresh PID 前后都必须
存活；CSV 必须覆盖完整窗口（允许 1 秒预注册调度容差）、相邻 timestamp 间隔不得超过 1 秒，
且至少 12 行，不能以少量 NVML 行或稀疏中间空洞伪造覆盖。freeze 和 A2 前置校验都会重新打开 raw runner JSON、CSV、window sidecar，并从
两组 `1000×2` 原始样本重新计算全部 12 个 percentile 门。
所有 JSON 与 telemetry CSV 都只读取一次：对同一份 byte payload 同时做外部 SHA-256
校验和解析，拒绝“先哈希一份、再解析另一份”的替换窗口。

### Runtime ledger authority

candidate handoff helper 的源码 SHA 固定不变；它嵌入的旧 runtime ledger **不作为本协议的
信任根**，因为它早于当前 production registration token。owned runner 以自身固定 SHA 和
当前十项 exact-SHA ledger 取代该旧 ledger，并把
`runtime_import_ledger_authority=owned_runner_current_runtime_import_ledger`、所有实际 source
identity、以及 `candidate_helper_embedded_ledger_trusted=false` 写入 runner/allocation/freeze
artifact；analyzer 会以自己的固定十项 **resolved root/path/SHA** ledger 重验这些字段。
所有 identity、A2→A1 binding 与 stored-vs-recomputed manifest evidence 均用递归严格 JSON
类型相等：object key set、array 长度和每个 scalar 类型/值都必须一致，故 `true` 不能冒充 `1`，
`1` 也不能冒充 `1.0`。
同时固定 `$PATCHED_ROOT/flash_kda/__init__.py` 为
`9cb9bd39186f6993f0067a8ff720cd233ef4b474444f1fd0fcd2bf06cab5fb84`，并固定 patched tree
只能有三条未暂存修改：`csrc/flash_kda.cpp`、`csrc/fwd.h`、`csrc/smxx/fwd_launch.cu`，其 SHA
分别为 `38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4`、
`613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083`、
`a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928`；其它 tracked 修改会拒绝。
source snapshot 前后均记录 wrapper 和该 dirty-set。此安排不修改或 monkey-patch helper，
更不会改变 production。

### Artifact 与 shell 身份

当前 runner SHA-256 是
`9c6ea7a76a5fdc613996d583f6717be16bd0646fa1bd5df922cc7d026ad668ec`；analyzer SHA-256 是
`9f2daedf5d84b436d935bd7884b42332b5158388603725b82bbf060852c4f14b`；本次外部 attestation 的
shell SHA-256 是 `5a60883ee1992f58fa305f8595d30a6bd7678555a6cc38394ebbebb909f998f5`。提交入口的 shell SHA
不内嵌在 shell 本身，必须由提交者以 `EXPECTED_PROTOCOL_SHELL_SHA256` 提供并由 shell 首行
校验，避免自哈希编辑循环。runner、allocation、A2 前置校验及 freeze 都会重新验证该 shell
的 resolved path 和 SHA。

只有 A1 和 A2 都是有效 allocation，且两者的 relative performance 均通过、Slurm job ID
不同，最终 `freeze` 才会写出 `eligible_for_public_freeze=true`。即使如此仍只产生 public
freeze 资格事实；生产 map 保持不变，需主流程另行决定。

## 执行

先计算本次 checkout 的 shell SHA（它必须等于实际提交入口的文件），然后在第一个干净 B300
Slurm allocation 中运行 A1：

```bash
export EXPECTED_PROTOCOL_SHELL_SHA256=5a60883ee1992f58fa305f8595d30a6bd7678555a6cc38394ebbebb909f998f5
export ALLOCATION_ID=A1 LABEL=b300_a1
C1_VARLEN_FP32_BOTH_GPU_AUTHORIZED=1 \
  bash team/c1_flashkda/challenge_varlen_fp32_both/run_clean_varlen_fp32_both_release.sh --authorized-by-parent
```

只有 A1 返回 0 后，才在**不同** Slurm job 的干净 B300 allocation 运行 A2；A2 的入口会在
触碰 GPU 前调用 `verify-allocation`，重新验证 A1 的 raw artifacts、telemetry、runner、
analyzer、helper、ten-key ledger、wrapper，且拒绝 A1 job 与当前 job 相同。allocation 阶段
会再次重开同一 A1 manifest，并将其 resolved path、SHA、A1 job 及 source identity 精确
写入 A2 manifest；freeze 只接受恰好绑定其传入 A1 manifest 的 A2：

```bash
export ALLOCATION_ID=A2 LABEL=b300_a2
export A1_ALLOCATION_MANIFEST=/absolute/path/to/A1_allocation.json
export A1_ALLOCATION_MANIFEST_SHA256="$(sha256sum "$A1_ALLOCATION_MANIFEST" | awk '{print $1}')"
C1_VARLEN_FP32_BOTH_GPU_AUTHORIZED=1 \
  bash team/c1_flashkda/challenge_varlen_fp32_both/run_clean_varlen_fp32_both_release.sh --authorized-by-parent
```

两个 manifest 都完成后，在同一已固定的 source 环境中运行（仍须设置 `A02_ROOT`、
`PATCHED_ROOT`、`REFERENCE_ROOT`、`C1_PINNED_REFERENCE_HELPER_PATH`、runner/analyzer SHA 及
上面的 shell SHA 环境）：

```bash
python team/c1_flashkda/challenge_varlen_fp32_both/analyze_varlen_fp32_both_release.py freeze \
  --allocation-a A1_allocation.json --expected-allocation-a-sha256 "$(sha256sum A1_allocation.json | awk '{print $1}')" \
  --allocation-b A2_allocation.json --expected-allocation-b-sha256 "$(sha256sum A2_allocation.json | awk '{print $1}')" \
  --json public_freeze_decision.json --require-eligible
```

`freeze` 没有任何 production side effect。

## 2026-08-30 实验结果（schema 3）

### schema 2 的无效尝试

job 12528 的两个 fresh PID 均通过 exact correctness 与 relative performance，但这是旧
telemetry 的 all-samples 门：manifest 中 `allocation_telemetry_valid=false`，且
`sm_clock_positive_and_within_95pct_median=false`；因此分类为
`telemetry_invalid_not_performance_failure`，最终不计作 A1，也不追认后续 A2。证据为
[allocation manifest](results_invalid_job12528/c1_varlen_fp32_both_A1_b300_sm103a_skew_fp32_both_release_r4_allocation.json)、
[PID 0](results_invalid_job12528/c1_varlen_fp32_both_A1_b300_sm103a_skew_fp32_both_release_r4_pid0.json)、
[PID 1](results_invalid_job12528/c1_varlen_fp32_both_A1_b300_sm103a_skew_fp32_both_release_r4_pid1.json) 和
[telemetry window](results_invalid_job12528/c1_varlen_fp32_both_A1_b300_sm103a_skew_fp32_both_release_r4_telemetry_window.json)。
job 12551 的 helper pre-gate 也失败；本地没有保留下来的 artifact，故不将其作为实验结果。

### schema 3 的 A1/A2

job 12555（A1）与 job 12556（A2）各包含 2 个 fresh Python PID、每 PID 每条路径
2 repeats × 1000 samples。四个 PID runner 均报告 output/final exact、输入不可变性和
`complete=true`；两个 allocation analyzer 均报告 `allocation_valid=true`、
`relative_performance_pass=true`、`allocation_pass=true`。完整 raw runner、日志、telemetry
与 manifest 见下表：

| allocation | Slurm job | raw runner JSON | 日志 | telemetry | allocation manifest |
| --- | --- | --- | --- | --- | --- |
| A1 | `12555` | [PID 0](results/c1_varlen_fp32_both_A1_b300_sm103a_skew_fp32_both_release_r5c_pid0.json)、[PID 1](results/c1_varlen_fp32_both_A1_b300_sm103a_skew_fp32_both_release_r5c_pid1.json) | [job log](results/c1_varlen_fp32_both_A1_b300_sm103a_skew_fp32_both_release_r5c_job12555.log) | [CSV](results/c1_varlen_fp32_both_A1_b300_sm103a_skew_fp32_both_release_r5c_telemetry.csv)、[window](results/c1_varlen_fp32_both_A1_b300_sm103a_skew_fp32_both_release_r5c_telemetry_window.json) | [A1 manifest](results/c1_varlen_fp32_both_A1_b300_sm103a_skew_fp32_both_release_r5c_allocation.json) |
| A2 | `12556` | [PID 0](results/c1_varlen_fp32_both_A2_b300_sm103a_skew_fp32_both_release_r5c_pid0.json)、[PID 1](results/c1_varlen_fp32_both_A2_b300_sm103a_skew_fp32_both_release_r5c_pid1.json) | [job log](results/c1_varlen_fp32_both_A2_b300_sm103a_skew_fp32_both_release_r5c_job12556.log) | [CSV](results/c1_varlen_fp32_both_A2_b300_sm103a_skew_fp32_both_release_r5c_telemetry.csv)、[window](results/c1_varlen_fp32_both_A2_b300_sm103a_skew_fp32_both_release_r5c_telemetry_window.json) | [A2 manifest](results/c1_varlen_fp32_both_A2_b300_sm103a_skew_fp32_both_release_r5c_allocation.json) |

manifest 中 `minimum_speedups_x` 是每个 PID record 的值；下表取该 allocation 两个
fresh PID record 的逐 percentile 最小值（保留 JSON 的完整精度），三列均高于预注册门
`1.02`：

| allocation | minimum P50 | minimum P95 | minimum P99 | telemetry SM 高频占比 |
| --- | ---: | ---: | ---: | ---: |
| A1 (`12555`) | `1.1052647336652395` | `1.0909242788801983` | `1.053236174702262` | `0.9032258064516129`（约 `0.9032258`） |
| A2 (`12556`) | `1.1049533161883405` | `1.0936929980141237` | `1.0519274586224148` | `0.8629032258064516`（约 `0.8629032`） |

两次 allocation 的 telemetry 均为 valid；同一 GPU 的前后审计均为 `0 MiB`（日志中的
`PRE_MEMORY_USED_MIB=0` 与 `POST_MEMORY_USED_MIB=0`）。随后 [public freeze decision](results/c1_varlen_fp32_both_b300_sm103a_A1_A2_r5c.freeze.json)
写出 `eligible_for_public_freeze=true`，文件 SHA-256 为
`5be6ca7f8533afdd8a643bc6395ecaf69d837fe7bb0fcb92fb70072de88c1520`；A1/A2 manifest
SHA-256 分别为
`3d2e0f96d6b687b50eb684276688df2102ad54620cf737d693aa5b6ef799c40b` 和
`3d0b61b5836d8f41f3adf110188daa1d2f9a62c35a100bdb697053a84f29a1f7`。

该 freeze 仍只证明 public freeze eligibility，`production_action` 保持
`unchanged`；这是生成该 artifact 时的历史事实，不能事后改写 JSON。主流程随后已经把 exact
tuple/state 集成到 v5 dispatcher/backend，但这不会把本目录 test-only freeze 自动升格为
production 证据。真实 production 预试 job12598 的目标 route/exact 通过，之后因协议负控复用
stale handoff 而失败，整 job 不计 A1；修复版协议随后在不同 Slurm job12770/12771 各以两个
fresh PID 完成 A1/A2，并写出 `eligible_for_production_freeze=true` 的最终 freeze。真实生产证据与
工件身份见 [production-freeze README](../challenge_varlen_fp32_both_production_freeze/README.zh-CN.md)；
本目录的历史 test-only freeze 仍不因后续成功而被改写或冒充 production 结果。
