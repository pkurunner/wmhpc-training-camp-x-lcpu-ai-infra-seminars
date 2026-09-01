# B=7 的 none-only public FLA release 协议

本目录从零开始检验一个未发布候选：B300 上 fixed
`B=7,H=12,T=2048,K=V=128,none` 能否通过真实 FLA public API 使用 test-only
`vshard2-P2`，并在两次独立 allocation 都稳定快至少 2%。旧
`challenge_fixed_batch_b7` discovery 不读取、不合并，也不能作为 release evidence。此协议只产生
`eligible_for_public_freeze` 的人工 review 输入，绝不自动改 production。

`job12559` 不能作为性能正、负证据：它在 raw/timing 之前由上游 `torch_ref.py` 的
`load_inline('sigmoid_ext')` 触发了 JIT rebuild，属于基础设施 identity/计时前置条件失效，
而不是候选性能失败。本版 schema 3 在任何 raw correctness 或 warm-up 前直接加载已固定的
`sigmoid_ext.so`，只临时拦截一次同名 `load_inline`，并拒绝所有 build fallback。

| 变量 | 含义 | 固定值 |
| --- | --- | --- |
| `B` | fixed batch 的序列数 | 7 |
| `H` | KDA head 数 | 12 |
| `T` | 每序列 token 数 | 2048 |
| `K,V` | key/value 通道维度 | 128, 128 |
| `s` | raw ABI state contract | `none`、`bf16_both`、`fp32_both`、`fp32_final_only` |
| `a` | clean Slurm allocation | `A1` 或 `A2` |
| `p` | 真实 public API 路径 | `pinned_public` 或 `c1_test_route_public` |
| `h_shell` | 外部提交入口提供的 protocol shell SHA-256 | 下表冻结值 |
| `h_helper` | pinned `sigmoid_ext.so` 的 path/SHA-256 | 固定 path 与 SHA，见下表 |
| `o` | patched FlashKDA tracked overlay | 精确 3-file dirty set 与 SHA map |

## 预注册边界

- test route 仅存在于当次 Python 进程，退出时恢复 FLA backend 属性。它严格拒绝除
  `(B,H,T,K,V)=(7,12,2048,128,128)`、BF16 contiguous 同卡 tensor、FP32 parameter、
  `initial_state=final_state=cu_seqlens=None` 以外的情况；还严格要求
  `scale == 128^-0.5` 与 `lower_bound == -5.0`。
- 注入只会调用审计 SO 的 `fwd_vshard_p2`。SO、符号、输入 ABI、descriptor 或 route decision
  任一不符即 fail closed。C1 decision 必须精确为
  `requested=chosen=vshard2_p2`、`reason=test_only_b7_h12_t2048_none_exact_route`、
  审计 SO hash、`test_only_route=true`、`production_source_mutated=false`。
- 注入前的 production `select_variant` 必须对 B7 和 B8 的四种 state contract 全部给出
  `requested=chosen=baseline` 且
  `reason=fixed_batch_shape_not_whitelisted`。因此 B7 的 `bf16_both`、`fp32_both`、
  `fp32_final_only` 和所有 B8 都仍是 baseline 负控。
- 每个 fresh PID 对四种 raw ABI contract 都运行 pinned baseline、`vshard2-P2` 和 pinned
  `tests/torch_ref.py`。output 和非 `none` 的 final state 必须逐位一致；每条路径的
  `q/k/v/g/beta/A_log/dt_bias/scale/lower_bound` 与 initial state 必须不变。
- `PATCHED_ROOT` 不是 generic clean tree：它必须在 pinned HEAD 上恰好存在工作树修改
  ` M csrc/flash_kda.cpp`、` M csrc/fwd.h`、` M csrc/smxx/fwd_launch.cu`，且三者 SHA 分别为
  `38d5657833946849e6cc9b4bad10c8965702cd7d032a3e2f0c77abd141aae7e4`、
  `613c86a211023ee5284b44a57398dfa2916a5cb272133b1768ce233470b71083`、
  `a4859c8cc365db86ed068849db78190e4199c7d4b645a77695f1200579aae928`；没有其他 tracked modification。shell、runner
  与 analyzer 都重验并持久化这个 exact overlay。
- pinned Torch reference 的 CUDA helper 不是可变缓存：shell、runner 和 analyzer 都要求
  `/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so`，SHA-256
  为 `8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f`。runner 直接 import
  该二进制，并只允许 pinned `torch_ref.py` 恰好一次 `load_inline('sigmoid_ext')` 被替换；path、
  SHA、`intercepted_names=["sigmoid_ext"]`、`no_build=true` 和固定 load contract 都进入 identity。
  缺少、路径漂移、SHA 漂移、零/多次/异名拦截或任何 JIT fallback 都 fail closed。

## 公共路径、计时与门槛

性能只测 `none`，并且两条路径都调用真实 `fla.ops.kda.chunk_kda`：关闭 C1 时要求 pinned spy
`+1`、C1 spy `+0`；打开 C1 时要求 C1 spy `+1`、pinned spy `+0` 与上述精确 decision。

每个 fresh PID 有 2 个 repeat；每个 repeat 每路径 100 次 cyclic warm-up，随后每路径 1000
个 CUDA-event sample。计时样本的固定 contract 为：

```text
CUDA current-stream: prepared environment/context/kwargs/counters/events and
start.record+start.synchronize before interval; interval exactly one public
chunk_kda -> end.record; host-only audit then end.synchronize
```

也就是说，environment/inference context、kwargs、spy snapshot 与 event object 都在 start 前准备；
`start.synchronize()` 完成后，源码路径中紧接着只有一个已准备的真实 public call 与 `end.record`。
audit/decision/counter 读取和 `end.synchronize()` 都在 interval 后。warm-up 结束后才做一次统一 CUDA
synchronize；timed helper 本身没有内部 global synchronize。
循环顺序交替，因此每 repeat 的 `first_path_counts` 固定是 500/500；每个 timed call 也以 host-only
spy/decision proof 计数。stdlib-only analyzer 忽略 runner 给出的 summary/pass，从 raw float samples
重新算 P50/P95/P99，并检查 summary、event 字符串、count、所有 key 和 Python JSON 类型都精确匹配。

每个 repeat 的每个分位必须满足：

```text
latency(pinned_public) / latency(c1_test_route_public) - 1 >= 0.02
```

一个 allocation 内需要 `2 fresh PID × 2 repeat` 全部通过。失败是完整负结果；allocation shell
非零退出，不能以“接近 2%”进入 A2。

## 身份与 A1 → A2 证据链

runner、allocation audit 与 chain 都持久化并重新校验：runner/analyzer/protocol shell、
`auto_dispatch.py`、`fla_backend.py`、harness、pinned `torch_ref.py`、FlashKDA Python、SO、
pinned `sigmoid_ext.so`、patched/reference/FLA Git head、上述 exact patched dirty overlay、全部六个 FLA source entry，以及
每个 main artifact 的 SHA-256 与 content identity。每次 JSON 读取在同一份 `read_bytes()` payload 上完成 SHA、UTF-8 decode 与 parse；content identity、runtime identity 和 repeat assessment 使用递归严格 JSON 类型比较，故 `true/1`、`0/false`、`1/1.0` 都不能冒充相同证据。每个 complete main 还持久化 test-only dispatcher
和两条 backend spy 的 post-restore exact proof。chain 会重新读取两个 audit 所引用的 main JSON、重算
raw gate，并要求两次 allocation 的完整 protocol identity、helper no-build proof 与 A1→A2 full source
binding 相等、Slurm job ID 为不同的 ASCII 正十进制数。schema version 是 `3`；schema 1/2 artifact
一律拒绝，不能与本版 evidence 混合。

shell **不含自引用 hash 常量**。这类常量无法可靠地 self-verify；相反，外层 sbatch 提交入口必须
提供 `EXPECTED_PROTOCOL_SHELL_SHA256`，shell 读取自身实际 SHA 后与这个外部冻结 attestation 比较，
再把 path/SHA 写入 log、plan 和 main/allocation identity。A2 还必须在任何 CUDA workload 前以只读
analyzer `--precondition-a1` 重验 A1 的 audit SHA、`eligible=true`、main SHA/content，以及完整 source/
protocol identity；它还接收并严格校验当前 A2 `SLURM_JOB_ID` 是 ASCII 正十进制且不同于 A1 job。随后 A2
allocation audit **再次**重开相同 A1 raw main/content/source，并持久化
`a1_prerequisite={canonical path, SHA-256, A1 job, full_source_identity}`。A2 audit 自身、后续 reopen
和最终 chain 都要求这四项精确等于传入的 A1 audit；缺少、伪造、失败、同 job 或 source identity 不同的
A1 不能提交/冻结 A2。A1 明确拒绝 A1 precondition 环境变量与 analyzer prerequisite 参数，避免混淆。

冻结 identity：

| 工件 | SHA-256 |
| --- | --- |
| runner | `481462766589ee3ec23c7ab0454a923f2f28aa506826413433fda0450030f534` |
| stdlib analyzer | `2fd71aecd563dc9c5c314de78f52050dddd76539266fcc54d9d93982c2892705` |
| clean shell（外部 attestation 的值） | `26e6a805507f124de9bf4e750ed8d7642441cbe89dc57fa416227dba467e6945` |
| pinned `sigmoid_ext.so` helper | `8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f` |
| production `auto_dispatch.py` | `9cdd460058254016af58723875bdf99ebe74f8e016a4c6027eb7fb38c8e9a88c` |
| production `fla_backend.py` | `206e448abcd3d64826f87a20e7d57c790fef6adacd91e26edcb10a3711b9b656` |

提交前必须在 B300 checkout 上确认 `sha256sum` 的 shell 输出就是表中值；不能用 shell 内部的
`$(sha256sum "$0")` 来填这个环境变量，否则外部 attestation 没有意义。

## 运行

在已取得的单 GPU B300 Slurm allocation 内运行 A1：

```bash
export C1_FIXED_BATCH_B7_NONE_GPU_AUTHORIZED=1
export A02_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02
export PATCHED_ROOT=/home/lcpu/85117379/flashkda-vshard4-prefetch2-1ce47ea-b300-r1
export REFERENCE_ROOT=/home/lcpu/85117379/flashkda-1ce47ea
export FLA_ROOT=/home/lcpu/85117379/fla-a3edffc
export PYTHON_BIN=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/.venv/bin/python
export LABEL=b300_sm103a_b7_none_r3_helper_nobuild
export ALLOCATION_ID=A1
unset A1_AUDIT EXPECTED_A1_AUDIT_SHA256  # A1 会 fail-closed 拒绝遗留 prerequisite
export C1_PINNED_REFERENCE_HELPER_PATH=/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so
export C1_PINNED_REFERENCE_HELPER_SHA256=8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f
export EXPECTED_PROTOCOL_SHELL_SHA256=26e6a805507f124de9bf4e750ed8d7642441cbe89dc57fa416227dba467e6945
bash "$A02_ROOT/team/c1_flashkda/challenge_fixed_batch_b7_none/run_clean_fixed_batch_b7_none_audit.sh" \
  --authorized-by-parent
```

只在 A1 allocation audit 的 `allocation_gate.eligible` 是 exact `true` 时，才在**新的** Slurm
allocation 中运行 A2。外层提交入口必须显式提供 A1 audit 路径及其已冻结 SHA：

```bash
export ALLOCATION_ID=A2
export A1_AUDIT=/absolute/path/to/c1_fixed_batch_b7_none_..._A1.allocation_audit.json
export EXPECTED_A1_AUDIT_SHA256=<A1 audit 的已记录 SHA-256>
export C1_PINNED_REFERENCE_HELPER_PATH=/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so
export C1_PINNED_REFERENCE_HELPER_SHA256=8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f
export EXPECTED_PROTOCOL_SHELL_SHA256=26e6a805507f124de9bf4e750ed8d7642441cbe89dc57fa416227dba467e6945
bash "$A02_ROOT/team/c1_flashkda/challenge_fixed_batch_b7_none/run_clean_fixed_batch_b7_none_audit.sh" \
  --authorized-by-parent
```

两个 audit 均冻结后，父提交方也要在 audit 文件以外的提交/台账中记录 A2 的 SHA；chain 不应在执行时从待验 audit 现场计算“expected”值：

```bash
export EXPECTED_A2_AUDIT_SHA256=<A2 audit 的已记录 SHA-256>
```

CPU-only chain 为：

```bash
python "$A02_ROOT/team/c1_flashkda/challenge_fixed_batch_b7_none/analyze_fixed_batch_b7_none.py" --chain \
  --a1-audit "$A1_AUDIT" --a2-audit "$A2_AUDIT" \
  --expected-a1-sha256 "$EXPECTED_A1_AUDIT_SHA256" \
  --expected-a2-sha256 "$EXPECTED_A2_AUDIT_SHA256" \
  --expected-runner-sha256 481462766589ee3ec23c7ab0454a923f2f28aa506826413433fda0450030f534 \
  --expected-analyzer-sha256 2fd71aecd563dc9c5c314de78f52050dddd76539266fcc54d9d93982c2892705 \
  --expected-protocol-shell-sha256 26e6a805507f124de9bf4e750ed8d7642441cbe89dc57fa416227dba467e6945 \
  --json "$CHAIN_JSON"
```

无 GPU 自检：

```bash
python -m py_compile run_fixed_batch_b7_none.py analyze_fixed_batch_b7_none.py
python run_fixed_batch_b7_none.py --self-test
python analyze_fixed_batch_b7_none.py --self-test
bash -n run_clean_fixed_batch_b7_none_audit.sh
```

## 2026-08-30 schema 3 A1 实测结果（job12570）

A1 在 B300 上完成了预注册的 schema 3 测量。固定形状为
`B=7,H=12,T=2048,K=V=128`、public contract 为 `none`，使用
`vshard2_p2` test-only route。pinned `sigmoid_ext.so` 采用 direct cached binary，
恰好拦截一次 `load_inline('sigmoid_ext')`，`no_build=true`；没有 JIT build 或
fallback。四个 raw ABI contract（`none`、`bf16_both`、`fp32_both`、
`fp32_final_only`）及两次 public 调用的 output/final-state 检查均 exact，且
输入与 initial state 不变。两个 fresh PID 为 `2283616`、`2284814`，GPU UUID
均为 `GPU-dadf9f3b-df58-d3fa-07b0-5fe223423db1`；前置、两 main 之间、后置及
独立审计后的显存读数均为 `0 MiB`。

性能门按 `latency(pinned_public) / latency(c1_test_route_public) - 1 >= 0.02`
逐 repeat、逐 P50/P95/P99 判断。四个 repeat 的 margin（P50/P95/P99）分别为：

| PID / repeat | P50 | P95 | P99 |
| --- | ---: | ---: | ---: |
| 2283616 / 0 | -1.03127% | -1.78295% | -2.73296% |
| 2283616 / 1 | -0.93992% | -1.46609% | -2.52822% |
| 2284814 / 0 | -0.95521% | -3.47353% | -4.14279% |
| 2284814 / 1 | -0.85771% | -1.39858% | -2.28954% |

因此这是一个有效的性能负结果：`eligible=false`、决定为
`STOP_keep_production_baseline`，shell 的 `FINAL_RC=95` 是预期的门拒绝，
不是异常。按预注册协议不申请 A2、不创建 chain，production dispatcher 保持
unchanged，B7 `none -> vshard2_p2` 不得发布。

结果工件及可复核哈希如下：

| 工件 | 相对链接 | SHA-256 |
| --- | --- | --- |
| A1 main0 | [main0](results/c1_fixed_batch_b7_none_b300_sm103a_b7_none_r3_helper_nobuild_A1_main0.json) | `1a591ede1db1798ef0bb0663cb453e414f84d6fdb82b49781e0a4416c86907fc` |
| A1 main1 | [main1](results/c1_fixed_batch_b7_none_b300_sm103a_b7_none_r3_helper_nobuild_A1_main1.json) | `57f2becd4efee4f808ace0211b70979debf4f06d4aebdb896a96169248ac1dfd` |
| A1 allocation audit | [allocation audit](results/c1_fixed_batch_b7_none_b300_sm103a_b7_none_r3_helper_nobuild_A1.allocation_audit.json) | `6312e1838da1a104a512331cdafb254b720df5725e4fbc166cb25729c789e866` |
| A1 plan | [plan](results/c1_fixed_batch_b7_none_b300_sm103a_b7_none_r3_helper_nobuild_A1.plan.json) | `868911541733b59e1122362f04c8970f66d8368da1d5bb76796724ec901aa477` |
| job log | [job12570 log](results/c1_fixed_batch_b7_none_b300_sm103a_b7_none_r3_helper_nobuild_A1_job12570.log) | `eabdb503b9bf4986f9f64c294efc5ebde9ab50888e0d6b9dd7c3750bc8d8ad0c` |

plan 和 job log 仅用于复核协议和执行状态；性能结论以两个 main
与 allocation audit 为准。job12559 仍保留原有边界：它在 raw/timing 之前因
上游 `torch_ref.py` 的 JIT rebuild 失败，属于基础设施前置失效，不能当作性能
结果；本节 job12570 才是 schema 3 的完整 A1 实测负结果。
