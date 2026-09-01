# T=8191 正式生产路径冻结协议

此目录是针对精确形状 `B=1, H=12, T=8191, K=V=128` 的**正式生产路径**复现实验。它不是分派器实现，也不会改写 `auto_dispatch.py` 或 `fla_backend.py`。

| 变量 | 含义 | 本协议取值 |
| --- | --- | --- |
| `B,H,T,K,V` | batch、head、token、key/value 维度 | `1,12,8191,128,128` |
| `A` | 相互独立的 Slurm allocation | `A∈{A1,A2}` |
| `p` | allocation 内的 fresh PID 序号 | `p∈{0,1}` |
| `j` | 同一 PID 内的 repeat 下标 | `j∈{0,1}` |
| `s` | production state contract | `none`、`fp32_final_only` |
| `q` | 延迟分位下标 | `q∈{P50,P95,P99}` |
| `r_{A,p,j,s,q}` | 同一格 pinned/C1 延迟比 | `t_{pinned,A,p,j,s,q}/t_{C1,A,p,j,s,q}` |
| `δ_{A,p,j,s,q}` | C1 相对 pinned 的裕量 | `r_{A,p,j,s,q}-1`；逐格要求至少 2% |

## 边界与验收

- 只有真实且未包 registry spy 的 `fla.ops.kda.chunk_kda` 被计时。临时 spy 只用于计时前的双路径路由证明，随后恢复；环境切换、inference context、kwargs、event 创建和 `start.record+start.synchronize` 都在区间前，区间内是一遍 uninstrumented public call，随后立即 `end.record`，route 审计与 `end.synchronize` 在区间后。`C1_B300_FLASH_KDA=1` 时必须经过已经注册的 `c1_b300_flash_kda`，而 `0` 时必须经过 pinned `flash_kda`。
- 此 runner 不安装模块属性替换、monkeypatch 或 test-only route。C1 调用必须报告 `requested_variant=chosen_variant=vshard4_p2`，并且 reason 精确为 `fixed_single_batch_b1_h12_t8191_{none|fp32_final_only}_whitelist_hit`。
- 本次重新冻结绑定最终 production source：`auto_dispatch.py=9633179c6a5e8ce61b1857a102f81074c6bdfdc63163e13980f40c24d6583c29`、`fla_backend.py=152ac4ba897c1887963553d592456b7f988c89105e6f6cb507a44757058d5fd1`。后者的 registry compatibility token 语义为 `c1-b300-flash-kda-skew-fp32-both-20260830-v5`；二者 SHA 均被 runner、shell 和 allocation/chain audit 的 source ledger 物理锁定。新版 dispatcher 的 exact-symbol 缺失会在 launch 前回退到 baseline；此 T=8191 release 仍要求实际决策为 vshard4，不接受该 fallback 作为通过。
- 生产正合同只有 `none` 和 `fp32_final_only`。`fp32_both`、`bf16_both`，以及 H=11、T=8190、B=2 邻域在两个正合同下都必须是 pre-launch `baseline`，防止策略泛化。
- 每个 fresh PID 先用 raw baseline、raw vshard4 和 pinned `torch_ref` 做逐元素输出/末态精确比较，并检查输入与初态不变；再做 public C1 与 pinned 的逐元素比较。
- 协议 schema 已升为 **4**：不能再让 `torch_ref.py` 自行触发 `load_inline` 的 JIT build。每个 raw main 必须从固定 helper `/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so`（SHA256 `8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f`）直接加载 `sigmoid_ext`；runner 只截获一次同名 `load_inline` 请求，第二次或任何其他名字都会 fail-closed，绝不回退到 builder。无论导入成功或失败，`torch.utils.cpp_extension.load_inline` 与 `sys.modules["sigmoid_ext"]` 都在 `finally` 中恢复。
- helper 的 canonical path/SHA 在 shell 的任何 CUDA workload 前检查，并纳入 allocation 前后 source snapshot。raw identity 的 source ledger、runtime proof 和 artifact content identity 都固定 `path`、`sha256`、load contract、`intercepted_names=["sigmoid_ext"]` 与 `no_build=true`；analyzer 会在 raw、allocation、A1→A2 binding 与 chain reopen 中逐层复核。schema-3 或伪造 helper proof 一律拒绝。
- 每个 allocation 有两个 fresh PID；每 PID、每合同做两轮，每个 public path 每轮 1000 个 CUDA-event 样本，路径先后顺序轮转。A1/A2 的每个 P50/P95/P99 都要求 `pinned / C1 - 1 >= 2%`。
- `analyze_tail8191_production_freeze.py` 仅用标准库从 raw samples 重算全部 summary/margin。它还冻结 B300 SM10.3/148SM、单 GPU UUID、SO、wrapper、harness、reference、六个 FLA 文件、生产源、runner、analyzer、外部 shell attestation、PID 与 ASCII 正十进制 Slurm job ID。allocation audit 会在同一份读取字节上校验 raw main SHA 并解析 JSON，再重开两份 main；按顺序严格交叉绑定 `process_index`、PID、job、GPU UUID、type-preserving content identity 与完整 source identity，拒绝 `true/1` 或 `1/1.0` 这类 JSON 类型替换。patched worktree 固定为恰好三份预期 tracked 修改及其 SHA，而不是错误地要求 clean。A1 有失败则 A2 clean shell 在启动前拒绝；A1/A2 job 必须不同，A2 audit 还必须绑定精确 A1 path/SHA/source identity，且 current A2 job 必须等于它自己的 raw mains。

历史 `challenge_tail8191_dispatch/` 的 A1/A2 是 production 集成前的 **test-only** 探索授权证据，不能当作此目录的 production-freeze 结果，也不会被本 analyzer 读取。

## 执行

本 release 的固定身份为：

- runner：`f4144f5fbdd61396ff907c6290b767b5570e04d19087f8332f9db10e56e7b1dc`
- analyzer：`0e42ff13dce296f83dff8cac8359eebb7ca459caaaef926fd6f6affb284b91dc`
- clean shell（由提交命令外部固定，并对执行中的 canonical `$0` 做同一外部 SHA gate）：`83b9b4cf753f3b411ce026ac610ac49c7c10bbd6f99bb5c0da4f64490eb2e387`

提交 A1 前，父提交命令先独立比较 shell SHA，再把同一固定值传入：

```bash
export C1_TAIL8191_PRODUCTION_FREEZE_GPU_AUTHORIZED=1
export A02_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02
export PATCHED_ROOT=/home/lcpu/85117379/flashkda-vshard4-prefetch2-1ce47ea-b300-r1
export REFERENCE_ROOT=/home/lcpu/85117379/flashkda-1ce47ea
export FLA_ROOT=/home/lcpu/85117379/fla-a3edffc
export PYTHON_BIN=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/.venv/bin/python
export LABEL=b300_sm103a_t8191_prod_v5_helper_r1
export ALLOCATION_ID=A1
unset A1_AUDIT A1_AUDIT_SHA256  # A1 发现任一 prerequisite 参数会 fail-closed 拒绝
export C1_PINNED_REFERENCE_HELPER_PATH=/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so
export C1_PINNED_REFERENCE_HELPER_SHA256=8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f
export EXPECTED_PROTOCOL_SHELL_SHA256=83b9b4cf753f3b411ce026ac610ac49c7c10bbd6f99bb5c0da4f64490eb2e387
test "$(sha256sum team/c1_flashkda/challenge_tail8191_production_freeze/run_clean_tail8191_production_freeze.sh | awk '{print $1}')" = "$EXPECTED_PROTOCOL_SHELL_SHA256"
bash team/c1_flashkda/challenge_tail8191_production_freeze/run_clean_tail8191_production_freeze.sh --authorized-by-parent
```

在 **A1 完成时**，父提交方须把 A1 audit 的路径及 SHA 记录到本次实验外部的提交/台账；不要在随后 A2 或 chain 命令中临时从待验文件重算这个值。在**新 Slurm job**做 A2 前，传入该预先冻结的值：

```bash
export ALLOCATION_ID=A2
export A1_AUDIT=/absolute/path/to/A1.allocation_audit.json
export EXPECTED_A1_AUDIT_SHA256=<parent-recorded-A1-audit-sha256>
export A1_AUDIT_SHA256="$EXPECTED_A1_AUDIT_SHA256"
bash team/c1_flashkda/challenge_tail8191_production_freeze/run_clean_tail8191_production_freeze.sh --authorized-by-parent
```

clean shell 会在任何 CUDA workload 前重开 A1、要求 A1 job 与当前 `$SLURM_JOB_ID` 不同；生成 A2 audit 时再次要求当前 job 与两个 A2 raw mains 的 canonical job 字符串完全相同。A2 完成后，同样由父提交方在外部记录：

```bash
export A2_AUDIT=/absolute/path/to/A2.allocation_audit.json
export EXPECTED_A2_AUDIT_SHA256=<parent-recorded-A2-audit-sha256>
```

最后在同一已冻结 remote 环境链式审计；仅命令成功且 JSON 写出 `production_freeze_passed: true` 时，才可称为 production-freeze 已通过：

```bash
python team/c1_flashkda/challenge_tail8191_production_freeze/analyze_tail8191_production_freeze.py \
  --chain --a1-audit "$A1_AUDIT" --a2-audit "$A2_AUDIT" \
  --expected-a1-sha256 "$EXPECTED_A1_AUDIT_SHA256" \
  --expected-a2-sha256 "$EXPECTED_A2_AUDIT_SHA256" \
  --expected-runner-sha256 f4144f5fbdd61396ff907c6290b767b5570e04d19087f8332f9db10e56e7b1dc \
  --expected-analyzer-sha256 0e42ff13dce296f83dff8cac8359eebb7ca459caaaef926fd6f6affb284b91dc \
  --expected-protocol-shell-sha256 83b9b4cf753f3b411ce026ac610ac49c7c10bbd6f99bb5c0da4f64490eb2e387 \
  --json "$CHAIN_JSON" --require-pass
```

## 2026-08-30 正式生产冻结结果

协议冻结后已在两个不同 Slurm allocation 完成真实 production 路径复验：A1 为
job `12592`，A2 为 job `12593`。每个 allocation 均包含两个 fresh PID；每个 PID
覆盖 `none`、`fp32_final_only` 两个 public contract，各做两轮、每条路径 1000 个
CUDA-event samples。四份 main 的 schema 均为 4，`test_only_route_installed=false`；
raw baseline、raw vshard4、pinned Torch reference 与真实 public C1/pinned 的 output/
final-state 检查逐位一致，输入与 initial state 不变。固定 helper 证据为
`no_build=true`，前后 GPU 清洁度门均通过。

A1/A2 analyzer 都从 raw samples 重算分位数并通过每个 repeat、每个 contract 的
`pinned_public / c1_production_public - 1 >= 2%` 门。最终 chain 重新打开两份
allocation audit 与四份 main，确认不同 Slurm job、source/helper/production-map
身份一致，并写出 exact `production_freeze_passed=true`。因此历史 test-only 授权现已
由真实 production registry 证据闭合；首次使用相对 audit 路径而被 analyzer 拒绝的
chain 命令没有生成证据，不计入结果。

| 工件 | 相对链接 | SHA-256 |
| --- | --- | --- |
| A1 main0 | [main0](results/c1_tail8191_production_freeze_b300_sm103a_t8191_prod_v5_helper_r1_A1_main0.json) | `c1594ab230dc4cb7b17414649b29eef0e37742f9771aa0e49b739939bd2d537b` |
| A1 main1 | [main1](results/c1_tail8191_production_freeze_b300_sm103a_t8191_prod_v5_helper_r1_A1_main1.json) | `59ed89ccde54beaf70ba35544892bcce77e695918764d8eb8cb66ca7b0153a4f` |
| A1 allocation audit | [A1 audit](results/c1_tail8191_production_freeze_b300_sm103a_t8191_prod_v5_helper_r1_A1.allocation_audit.json) | `aff6911cdc665220e988c521b0ebedde73c49e6cc8b84c11254e3094563b25fd` |
| A2 main0 | [main0](results/c1_tail8191_production_freeze_b300_sm103a_t8191_prod_v5_helper_r1_A2_main0.json) | `ee91a050ab2b7334ac996f9d1f3b1f3b6c780e49b024dbc28f2aa379865205b2` |
| A2 main1 | [main1](results/c1_tail8191_production_freeze_b300_sm103a_t8191_prod_v5_helper_r1_A2_main1.json) | `027d77e2018d3dfacd3ccc47ef2c2d303ef4a800751300fa481cfc36a1b34c58` |
| A2 allocation audit | [A2 audit](results/c1_tail8191_production_freeze_b300_sm103a_t8191_prod_v5_helper_r1_A2.allocation_audit.json) | `3922a767dc318d8c133440d9c2b9f790e046d94888652fbd19e4485f40e64aec` |
| A1→A2 chain | [chain](results/c1_tail8191_production_freeze_b300_sm103a_t8191_prod_v5_helper_r1_A1_A2.chain.json) | `e5b2981e2436c0aeed7737605b3e013e577b939f5859bfefaa34364b50a31c33` |
| A1 clean log | [job12592](results/c1_tail8191_production_freeze_b300_sm103a_t8191_prod_v5_helper_r1_A1_job12592.log) | `f14df76797dd6ab77dcfcf2b48d69d814d2e3aa2e59a08ae437fcda856e34c29` |
| A2 clean log | [job12593](results/c1_tail8191_production_freeze_b300_sm103a_t8191_prod_v5_helper_r1_A2_job12593.log) | `e2126cb93cdc0d993d95f8d016a7900ec1803d88d501aaaaea252daa93fc3443` |

上述链接对应的工件现已同步到本地，并再次完成逐文件 `json.load`、SHA-256、A1→A2
链绑定与全部 raw samples 的精确 P50/P95/P99 重算。16 个 PID/contract/repeat 格全部与
artifact summary 一致；全局最小 `r` 为 `1.3549231147729903`（即裕量
`35.49231147729903%`），出现在 A2 PID `3448835` 的 `fp32_final_only`、repeat 1、P99，
仍远高于逐格 2% 门。远端 source/helper 物理路径无法从本地工作区重新打开，故其文件身份
由远端 allocation/chain 的冻结 ledger 与本地 artifact 链绑定共同核对，不把本地解析冒充成
第二次远端 source re-open。
