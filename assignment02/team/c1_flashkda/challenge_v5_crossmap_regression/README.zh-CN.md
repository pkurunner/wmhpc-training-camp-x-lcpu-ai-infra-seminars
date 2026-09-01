# v5 public-registry 跨映射回归协议

本目录是一个**只读的非发布回归门**：在已集成的 v5 production public registry 上，跨越
fixed-B2、fixed-B5、T=8191 与 packed-varlen skew 四个已经发布的精确 cell 做独立 A1/A2
复验。它不会写入或替换 `auto_dispatch.py`、`fla_backend.py`、任何 public map、FLA registry
源码或 backend 实现；也不会调用任何 test-only route/map 安装入口。

| 变量 | 含义 | 本协议取值 |
| --- | --- | --- |
| `B,H,T,K,V` | batch、head、token、key/value 维度 | 固定 cell 见下表；`H=12,K=V=128` |
| `o_i` | 第 `i` 个 CPU-authoritative packed-varlen offset | 正向 `(0,1,2,3,4,5,12288)` |
| `A` | 相互独立的 Slurm allocation | `A∈{A1,A2}`，job 必须不同 |
| `J_A,u_A` | allocation `A` 的 Slurm job 与 GPU UUID | 最终 `J_A=12958/12959`；两次 `u_A` 相同 |
| `p` | allocation 内 fresh Python PID 下标 | `p∈{0,1}` |
| `c` | 一个预注册 cross-map cell | 4 个正向、3 个负控 |
| `q` | 延迟分位 | `q∈{P50,P95,P99}` |
| `r_{A,p,c,q}` | 同一 public cell 的 pinned/C1 延迟比 | 三个 `q` 都要求 `r>1`；非发布回归门 |

## 预注册矩阵

| 角色 | 精确 public cell | 必须的 C1 variant | 备注 |
| --- | --- | --- | --- |
| 正向 | `B=2,H=12,T=2048,fp32_both` | `vshard4_p2` | public/direct C1/pinned/reference exact |
| 正向 | `B=5,H=12,T=2048,fp32_both` | `vshard2_p2` | 同上 |
| 正向 | `B=1,H=12,T=8191,none` | `vshard4_p2` | 同上 |
| 正向 | packed `o=(0,1,2,3,4,5,12288),H=12,fp32_both` | `vshard4_p2` | CPU descriptor 仍由真实 backend/registry 验证 |
| 负控 | `B=7,H=12,T=2048,none` | `baseline` | 真实 public registry：C1 可进入 backend，但 dispatcher 的 direct/public decision 都必须为 baseline，绝不选/启 `vshard2_p2`/`vshard4_p2` |
| 负控 | `B=1,H=12,T=8191,fp32_both` | `baseline` | 同上，且 direct/public/pinned/reference output 与存在的 final state 全 exact |
| 负控 | packed `o=(0,1,2,3,4,6,12288),fp32_both` | `baseline` | 每次清 C1 handoff 与 metadata cache；C1 verifier 必须精确拒绝 `varlen_offsets_not_whitelisted`，public 落 pinned（C1/pinned spy=`+0/+1`） |

每个正向 cell 每个 fresh PID 做 1 轮、每路径 100 个 CUDA-event samples。性能条件是
`pinned/C1>1` 的**回归探测**，不是 `>=2%` 的发布或扩表门；任何 cell 失败都不能引申为
production map 应变化。正向的 correctness 必须同时覆盖真实 `fla.ops.kda.chunk_kda` public
调用、direct C1、direct pinned 与 pinned Torch reference 的 output exact，以及存在时的 FP32
final-state exact。正向 public C1/pinned route spy 必须分别为 `+1/+0` 与 `+0/+1`，且 spy 在 timing
前恢复。B7/T8191 的 baseline 负控也实际经由 public registry：C1 public spy 必须为 `+1/+0`，
但两个 dispatcher decision 均为 baseline。邻接 varlen 负控不读取 `get_last_decision()`（该分支
没有 launch-side decision）；它以当前 `q`、GPU offsets 和 CPU-authoritative offsets 的 verifier/
issuer spy 为拒绝证据。计时区间只有一遍未插桩的真实 public FLA call。

baseline decision 使用独立的 null-provenance ABI：`extension_sha256=null`、
`varlen_cpu_authoritative=false`、`certified_varlen_offsets=null`、`canonical_cache_hit=null`；
extension 的身份只由 source/runtime ledger 证明。正向 fixed decision 则固定为非 varlen provenance
（false/null/null）；正向 skew decision 固定为 CPU-authoritative、精确 offsets，并要求 direct 为
canonical-cache miss、public 为 hit。该 miss→hit 也写入 varlen handoff evidence，不能只靠
`chosen_variant` 推断。

## 不变性与身份门

- 每个 fresh raw PID 的最早 runner gate 是：先用**不依赖 module import** 的 canonical
  `Path.read_bytes()+SHA-256` 固定所有 source/helper/FLA 文件，再检查 `sys.modules` 中不存在
  Torch、FLA 或任一 production helper，随后执行本 runner 的单 UUID、0 MiB、无 compute-app
  `nvidia-smi` gate。只有该 gate 通过后，唯一的重型导入入口才加载 dispatcher、FLA、helper 和
  harness，并逐一将已加载模块的 `__file__` 绑定回刚才的 canonical path。shell 的 `PRE` 是外层
  防线；raw 的 `identity.pre_torch_clean_gpu` 才是可重开、可审计的 earliest pre-Torch 证据。
- 三张 built-in public map 只会在上述导入和 `__file__` identity 通过后、任何 workload 前读取，作
  typed canonical serialization、digest 与 object identity 记录；同一个 raw PID 的 pre/post 必须
  内容、digest、object identity 全同。`object_ids` **绝不**跨 PID 或 allocation 存储/比较；A1/A2
  只携带 entries+重算 digest 的 content-only typed map。
- 两份 production source 的物理路径和 SHA 在前后都必须是
  `auto_dispatch.py=9633179c6a5e8ce61b1857a102f81074c6bdfdc63163e13980f40c24d6583c29` 与
  `fla_backend.py=152ac4ba897c1887963553d592456b7f988c89105e6f6cb507a44757058d5fd1`。
- B300/SM10.3/148SM、单 GPU UUID、0 MiB 且无 compute process、审计过的 extension、patched
  worktree 的三项精确 dirty-set、reference/FLA clean tree 都是 fail-closed gate。
- pinned `torch_ref.py` 必须使用固定的已编译 `sigmoid_ext.so`，只拦截一次同名
  `load_inline` 请求并直接加载，不允许 JIT build 或 fallback builder。
- `schema_version=4` 将 source ledger 分为 `source_pre_torch`（只读路径/SHA，发生在 clean gate
  前）和 `source_pre/source_post`（同一 ledger 加上 clean gate 后的 loaded-module `__file__`
  binding）。loaded-module identity 后，runner 仍只通过同一个 clean-gate protected import path
  取得 canonical `torch` module。它**唯一**显式补全的 runtime global 是本 runner 直接调用的、
  已固定 source 中 `shared_seqcount._make_inputs` 的 TYPE_CHECKING-only `torch`：typed
  `pre_hydration` record 必须证明该 global 缺失；`bound_pre_workload` 和 `post_workload` record
  必须证明同一个 canonical module/function object、同一个 module `__file__`、同一个
  `function.__globals__ is vars(module)`，并把 global 绑定为同一个 `sys.modules['torch']` object。
  这些 object id **只在同一 raw PID 内**比较，绝不跨 PID/allocation 存储或比较。此绑定发生在
  clean gate 之后，且 workload/source/map 复核后重新验证；它不是 source 或 map 写入。
  本工件**不声明**对 confirmation、varlen helper、tail helper、harness 或任意第三方 helper 的
  静态全路径证明，也不会把未完成的静态审计写成证据。它们仍以 source ledger 和
  loaded-module identity 固定：varlen helper `e07481e7…be14`、tail helper `f4144f5f…b1dc`、shared
  `4ba4b262…83f`、confirmation `9445d94a…740b`、harness `5c92ac53…7d52`、`varlen_metadata`、
  pinned `torch_ref.py` `bb037c8b…06a5`，以及 FLA public callable/module 和六个真实 FLA source
  文件。所有这些 SHA 在 workload 前后必须不变。真实 helper path 是否完成，只由本协议已有的
  complete correctness、route-spy、exact、immutability、handoff/cache 与 performance gate 共同证明；
  缺少任何 complete raw 都不能用上述窄绑定替代。
- analyzer 只用标准库：每次 reopen 都是同一次 `read_bytes()` 的 SHA + JSON parse，递归严格
  JSON 类型比较会拒绝 `true` 伪装 `1` 或 `1.0`；它还物理复核 patched dirty-set、reference/FLA
  clean tree、FLA public export 身份，以及所有预注册 exact/verifier/immutability/handoff/cache 字段。
- 每条 exact record 都锁死其 fixed/varlen 的六个比较字段和 output/final ABI；`none` contract
  必须严格没有 final state。每一次真正的 pinned Torch-reference 调用也在 callable boundary 记录
  `q/k/v/g/beta/A_log/dt_bias`、适用时的 GPU/CPU offsets 与 FP32 initial-state 的不可变快照。
- output ABI 还逐 cell 锁定完整 `[B,T,12,128]`：B2=`[2,2048,…]`、B5=`[5,2048,…]`、
  B7=`[7,2048,…]`、T8191=`[1,8191,…]`，两个 packed-varlen cell 都必须是
  `[1,12288,12,128]`（即 total `T`，不是 sequence count）。每条 direct/public/pinned
  immutable record 都必须有该路径适用的**精确** fields 集合，不能只报告“未变”。
- route spy 的 `before`、`after`、`delta` 具有完整且严格的 `c1/pinned` 整数 schema，analyzer
  逐计数重算 `after-before=delta`。正向 varlen handoff 的 prepare 两分支、两份 path immutability
  和五项 cache stats（entries/hits/misses/capture rejections）都锁死 schema；其 miss→hit 的
  cache accounting 是 `entries=1,hits=1,misses=1`、两个 capture-rejection=0。
  邻接-varlen 拒绝控制的 clear 前、clear 后和 cleanup 后三份同 schema cache snapshot 也必须全零，
  因而不能以先前 cell 遗留的 cache/handoff 解释 rejection。
- 性能 record 锁死为 round 0、12 次 warmup、每路径 100 个 event sample、50/50 先发路径计数和
  一次未插桩 public FLA call 的 event contract；先发计数只能是 JSON integer，raw latency、summary
  延迟与 ratios 都只能是有限 JSON float（拒绝 int/bool）。它仍只是 `pinned/C1>1` 的非发布回归门。
- `_varlen_positive` 的 CPU descriptor、verifier、reference、direct backend 和 public registry
  helper 调用都在一层 `torch.inference_mode()` 内；自测同时对抗非 inference-mode 直接 helper 调用
  和 pre/post guard 被绕开的情形。

`schema_version=1/2/3` 或缺少上述 `source_pre_torch`、bootstrap、loaded-module identity、
`shared._make_inputs` typed pre/bound/post identity evidence 的 partial JSON 一律不能通过 analyzer。
因此 job12828 与 retry job12882 都只是“重型 import 早于 runner clean gate”的基础设施失败记录；
job12911 虽通过此前 clean gate，却在任何 workload 前因 `shared._make_inputs` 的 module-global
`torch` 未绑定而 `NameError`。三者都不属于 A1，不能作为性能、correctness 或 freeze 证据，也不得
与本 v4 协议的 future raw 混合。

schema4 的 job12928 曾以修复前 analyzer SHA 成功写出 A1 audit；job12929 的两份 raw、correctness
和 clean-between 都已完成，但 A2 allocation analyzer 在 A1 prerequisite reopen 后因
`obj(a1["allocation"])` 漏传 `label` 抛出 `TypeError`，发生在 A2 audit 写出、A1 binding 与
allocation/chain 证据建立**之前**。其 FINAL 0 MiB 只说明清理，不补足该协议失败。此次 analyzer
修复会改变外部 SHA，因此 job12928 的旧 A1 audit 和 job12929 的 raw 都不得与新 analyzer 混用；
job12929 明确是 A2 audit 前的协议失败，不是 A2 allocation、chain 或 freeze 证据。

## 本地静态自测

```bash
python team/c1_flashkda/challenge_v5_crossmap_regression/run_v5_crossmap_regression.py --self-test
python team/c1_flashkda/challenge_v5_crossmap_regression/analyze_v5_crossmap_regression.py self-test
bash -n team/c1_flashkda/challenge_v5_crossmap_regression/run_clean_v5_crossmap_regression.sh
```

runner 的纯本地自测不导入真实 Torch/FLA/helper；它构造 TYPE_CHECKING-only
`shared_seqcount._make_inputs` 的缺失 global、detached function globals、detached module、
非 canonical/global 或 `sys.modules` `torch`、以及 workload 后 function replacement 等对抗状态，
均必须 fail-closed。
analyzer 自测还拒绝 v1/v2/v3 raw、false 的 module-dict identity、非 canonical module/function name
及 pre-hydration 已出现 `torch` 的伪造 typed record。它还以临时 JSON 的真实 `read_once` SHA/parse
reopen 经过 production `allocation(A2)` 分支的 A1 prerequisite、path/SHA binding、distinct-job 比较
和 audit 输出；controlled raw validators 仅用于避免在本地伪造 B300-complete raw。非 object allocation、
非 string A1 job、相同 Slurm job 与任何泄漏的 `TypeError` 都必须 fail-closed。

本版协议文件的 SHA-256（若任一文件变更，A1/A2 必须重新固定相应值）为：

| 工件 | SHA-256 |
| --- | --- |
| runner | `384758dd4fcb797c79fda12c7e693b92542f294a9ff95b35fa6c6657e1a2e78a` |
| stdlib analyzer | `a807faee74fbe57e0754e7032cdcffd2b18e4d92db40115c7092d1943c7f338a` |
| canonical shell | `2974e679df6502883b7ee34bac492b9a1f704296765db2c21fd8cd0746b435df` |

## B300 A1/A2 提交

父提交方必须在 shell 外部先记录三份协议文件的 SHA，并把它们作为环境变量传入。不要把 shell
内容复制到 `sbatch --wrap` 的 spool 文件中；只允许 canonical shell 路径。A1 成功后，也要在
shell 外部记录 A1 audit 的绝对路径与 SHA，随后在**不同 Slurm job**做 A2。

```bash
export C1_V5_CROSSMAP_GPU_AUTHORIZED=1
export A02_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02
export PATCHED_ROOT=/home/lcpu/85117379/flashkda-vshard4-prefetch2-1ce47ea-b300-r1
export REFERENCE_ROOT=/home/lcpu/85117379/flashkda-1ce47ea
export FLA_ROOT=/home/lcpu/85117379/fla-a3edffc
export PYTHON_BIN="$A02_ROOT/.venv/bin/python"
export LABEL=b300_sm103a_v5_crossmap_r1
export C1_PINNED_REFERENCE_HELPER_PATH=/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so
export C1_PINNED_REFERENCE_HELPER_SHA256=8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f
export C1_V5_CROSSMAP_AUTO_DISPATCH_SHA256=9633179c6a5e8ce61b1857a102f81074c6bdfdc63163e13980f40c24d6583c29
export C1_V5_CROSSMAP_FLA_BACKEND_SHA256=152ac4ba897c1887963553d592456b7f988c89105e6f6cb507a44757058d5fd1
export C1_V5_CROSSMAP_RUNNER_SHA256=<parent-recorded-runner-sha256>
export C1_V5_CROSSMAP_ANALYZER_SHA256=<parent-recorded-analyzer-sha256>
export CANONICAL_PROTOCOL_SHELL="$A02_ROOT/team/c1_flashkda/challenge_v5_crossmap_regression/run_clean_v5_crossmap_regression.sh"
export EXPECTED_PROTOCOL_SHELL_SHA256=<parent-recorded-canonical-shell-sha256>
export ALLOCATION_ID=A1
unset A1_AUDIT EXPECTED_A1_AUDIT_SHA256
sbatch --partition=gpu --qos=gpu1_qos --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:50:00 \
  --export=ALL --wrap='bash "$CANONICAL_PROTOCOL_SHELL" --authorized-by-parent'
```

A1 成功后，从输出中的 `V5_CROSSMAP_AUDIT`/`V5_CROSSMAP_AUDIT_SHA256` 复制为外部冻结值；不要
重新从未审计文件临时计算。以另一个 Slurm job 提交 A2：

```bash
export ALLOCATION_ID=A2
export A1_AUDIT=/absolute/path/to/c1_v5_crossmap_..._A1.allocation_audit.json
export EXPECTED_A1_AUDIT_SHA256=<parent-recorded-a1-audit-sha256>
sbatch --partition=gpu --qos=gpu1_qos --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:50:00 \
  --export=ALL --wrap='bash "$CANONICAL_PROTOCOL_SHELL" --authorized-by-parent'
```

完成 A2 后，冻结命令只写出证据结论，不会修改 production：

```bash
python team/c1_flashkda/challenge_v5_crossmap_regression/analyze_v5_crossmap_regression.py freeze \
  --a1-audit "$A1_AUDIT" --a2-audit "$A2_AUDIT" \
  --expected-a1-sha256 "$EXPECTED_A1_AUDIT_SHA256" \
  --expected-a2-sha256 "$EXPECTED_A2_AUDIT_SHA256" \
  --expected-runner-sha256 "$C1_V5_CROSSMAP_RUNNER_SHA256" \
  --expected-analyzer-sha256 "$C1_V5_CROSSMAP_ANALYZER_SHA256" \
  --expected-protocol-shell-sha256 "$EXPECTED_PROTOCOL_SHELL_SHA256" \
  --json team/c1_flashkda/challenge_v5_crossmap_regression/results/c1_v5_crossmap_${LABEL}_A1_A2.chain.json
```

canonical shell 在 PRE、两个 fresh PID 之间、POST 和 `EXIT` trap 都要求单 UUID、0 MiB、无 compute
app；trap 还会重新计算 protocol/source/helper/FLA SHA。因此 shell 失败或最终清理失败一律不能把
allocation 作为可冻结证据。

## 2026-08-30 B300 r5_a2fix 结果（只读、非发布）

最终协议在 fresh A1 job `12958` 与不同 Slurm job 的 A2 job `12959` 上闭合；每个 allocation
各有两个 fresh PID。两次 allocation 恰好落在同一 GPU UUID
`GPU-3924bd78-8b2f-8e85-3f20-8b0687d0bb0a`，所以这里只证明 job/allocation 隔离，**不**称为
跨 GPU 或跨硬件独立重复。

四个预注册正向 cell 在四个 PID 上均通过真实 public/direct/pinned/reference exact、route spy、
输入/initial-state 不变、source/map 不变与每路径 100 个 CUDA-event samples。A1 的逐格最小
`pinned/C1` 为 `1.046089209043`，A2 为 `1.058466327581`，全局最小出现在 A1/PID1 的
fixed-B5、P99。这个协议预注册的是 `r>1` 的**非发布回归探测**，不是 `>=2%` 的扩表门。
B7-none、T8191/FP32-both 与邻接 offsets 三个负控也都按 baseline/fail-closed 通过；production map
前后 digest 均为 `a4fb43fb4dc98fe8cebacdaa3199600c66676887e6a47447d83d4bab7322117b`，没有任何 map 或
source 变更。

| 工件 | 相对链接 | SHA-256 |
| --- | --- | --- |
| A1 main0 | [main0](results/c1_v5_crossmap_b300_sm103a_v5_crossmap_r5_a2fix_A1_main0.json) | `54e6e8ef30046615810d1f92502b69e623a6c14d3e5f214de3ce6a5efa8b3890` |
| A1 main1 | [main1](results/c1_v5_crossmap_b300_sm103a_v5_crossmap_r5_a2fix_A1_main1.json) | `a4c516ee10221af25434fd927ed01b6922a48907c2530852f887d784a038b4f8` |
| A1 allocation audit | [A1 audit](results/c1_v5_crossmap_b300_sm103a_v5_crossmap_r5_a2fix_A1.allocation_audit.json) | `ef991ec492f18c2d479411eda3c4be6beee7378f98fdce016f50305621aab1db` |
| A2 main0 | [main0](results/c1_v5_crossmap_b300_sm103a_v5_crossmap_r5_a2fix_A2_main0.json) | `ad39e25717423cd528e074b5e1114609ce9eebdeeca2fe0f99311fdd7c44dc96` |
| A2 main1 | [main1](results/c1_v5_crossmap_b300_sm103a_v5_crossmap_r5_a2fix_A2_main1.json) | `452b0a14b1ac2e3b17ff5e666abdf1b7b51216cb87c4ecb9b6a69390a0399e35` |
| A2 allocation audit | [A2 audit](results/c1_v5_crossmap_b300_sm103a_v5_crossmap_r5_a2fix_A2.allocation_audit.json) | `8bc5f4f459cd62b8947ec1709a4b172b3717c38d8de7c12e6faa1e505983fe95` |
| A1→A2 chain | [chain](results/c1_v5_crossmap_b300_sm103a_v5_crossmap_r5_a2fix_A1_A2.chain.json) | `33ff98ed0e2430af144603acc24ef843df89bf079fe9eeeb2ddde39613e55f17` |
| A1 clean log | [job12958](results/c1_v5_crossmap_b300_sm103a_v5_crossmap_r5_a2fix_A1_job12958.log) | `9efbc0371f6b0719fea360a963bcc05cd22fe134d28d97645d420c7bbf04c671` |
| A2 clean log | [job12959](results/c1_v5_crossmap_b300_sm103a_v5_crossmap_r5_a2fix_A2_job12959.log) | `389c6504d12c14e8d2ccd44e0cce5275f2db5a79f1a0e1abe6a98c0f4d6516af` |

最终 [chain](results/c1_v5_crossmap_b300_sm103a_v5_crossmap_r5_a2fix_A1_A2.chain.json) 写出
`production_freeze_passed=true`；这里的 freeze 只冻结回归证据，不修改 production。同步到本地的
7 份最终 JSON 与 2 份日志已经重新解析、逐文件哈希并从 raw samples 复算全部 48 个 ratio，均与
artifact 一致。远端 source/helper/FLA 物理路径无法从当前 Windows 工作区再次打开，因此本地只核对
其冻结 ledger 与 artifact 链绑定，不把它表述为第二次远端 source re-open。
