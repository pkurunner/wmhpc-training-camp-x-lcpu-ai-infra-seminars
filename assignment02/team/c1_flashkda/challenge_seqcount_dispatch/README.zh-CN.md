# sequence-count/head 边界实验

此目录检验一个很窄的假设：在 B300 上，`baseline`、两 CTA/head 的 `vshard2_p2` 和四
CTA/head 的 `vshard4_p2` 的性能胜者，能否只由 sequence–head 工作项数解释。它不修改
`auto_dispatch.py`，也不把实验结果直接推广成 runtime policy。

## 变量表

| 变量 | 含义 |
|---|---|
| $N_{seq}$ | 独立序列数；fixed batch 时等于 $B$，packed varlen 时等于 `len(cu_seqlens)-1` |
| $H$ | 本卡 head 数 |
| $M=N_{seq}H$ | K2 原始 `(sequence, head)` 工作项数；本实验的待检验自变量 |
| $B$ | 输入张量的 batch 维；fixed 为 $N_{seq}$，varlen 为 1 |
| $T$ | 输入张量每个 batch 项的 token 数；主矩阵 fixed 为 2048，必要控制族 fixed 为 257；varlen 为所有序列长度之和 |
| $\ell_i$ | 第 $i$ 个 packed sequence 的长度 |
| $C$ | chunk 长度，固定为 16 |
| $G_{K1}$ | K1 prepare kernel 的 grid；varlen 采用实现中的 upper bound |
| $G_{K2}^{(s)}$ | K2 recurrence kernel 的 grid，$s\in\{1,2,4\}$ 分别表示 baseline/v2/v4 的 value shard 数 |

其中 fixed 与 balanced-varlen 成对匹配：每个逻辑序列均为 2048 token，总 token 数相同；
主矩阵的 skewed-varlen 保持 $N_{seq}$、$H$ 和总 token 数不变，却令前
$N_{seq}-1$ 个序列仅有一个 token，最后一个序列承载其余 token；T=257 必要控制族的
具体长度另见下文。

\[
M=N_{seq}H,\qquad
G_{K2}^{(s)}=(N_{seq},\;sH,\;1),\qquad s\in\{1,2,4\}.
\]

fixed 的 $G_{K1,x}=N_{seq}\lceil T/C\rceil$；主矩阵使用 $T=2048$，必要控制族使用 $T=257$。packed varlen 的实现使用
$G_{K1,x}=\lceil(\sum_i\ell_i)/C\rceil+N_{seq}$ 作为上界。因此本实验也会显式记录每一
case 的实际 tile 数和 launch tile 数，而不是假定两种布局的 K1 完全相同。

## 设计

- 共 54 个 case。完整 fixed 与 balanced-varlen 配对覆盖
  $M=24,36,37,38,39,40,48,72,75,76,96$；每个 $M$ 都至少包含两种不同的
  $(N_{seq},H)$ 分解，避免把“同一分解的表示一致”误报成只由 $M$ 决定。
- 在边界附近额外加入严重 skewed varlen：$M=36,38,72,76$。
- 另有一个必要控制族：fixed `B=37/38,H=1,T=257`；uniform varlen 分别为
  `lengths=[257]*37/38`；skewed varlen 分别为 `[17]*36+[8897]` 与
  `[17]*37+[9137]`。同一 $N_{seq}$ 下三者总 token、每段长度 $\bmod 16$ 和实际 K1 tile
  数都相同，直接检验 $M=37/38$，而不会把这三项混入布局效应。实现的 packed-varlen
  K1 launch upper-bound 仍会单独记录。
- 每个 case 对 `none`、`bf16_both`、`fp32_both`、`fp32_final_only` 四个 raw state ABI
  执行 `vshard2_p2/vshard4_p2` 对 baseline 的 bitwise-exact 检查。
- pinned upstream `tests/torch_ref.py` 只用于 fixed、balanced 和 skew 代表 case；T=257 的
  $N_{seq}=38$ fixed 覆盖全部 raw contract，uniform/skew varlen 至少覆盖 `none` 与
  `fp32_both`，避免只用 shard-to-baseline 一致性推断高序列数的 state 语义。
- `none`、`fp32_final_only` 与 `fp32_both` 分别以三路径 cyclic CUDA-event 预热 100 次、
  采样 1000 次，记录每一路的
  P50/P95/P99、三种分位数的 winner、所有长度、$N_{seq}$/$H$/$M$ 与预期 launch grids。
- 所有参与 $M$ 假设的 case 都完整覆盖 FLA public 的 `none`、`fp32_final_only`、
  `fp32_both`；`bf16_both` 仍只属于 raw ABI 正确性合同。

## 预定义停止/推广门

“分位数一致”专指每个分位数选择的 **winner identity** 相同，而不是不同布局的绝对延迟
必须相等。对任一测量 state contract，如果同一 $M$ 的 fixed、balanced-varlen 或（若已
覆盖）skewed-varlen 在 P50/P95/P99 的 winner vector 不同（包括 T=2048 与 T=257 控制族
同为 $M=38$ 的交叉比较），JSON 将给出
`STOP_do_not_promote_M_only_policy`。此时不得把仅 $M$ 的规则加入 dispatcher。

只有每个 $M$ 都有至少两种 $(N_{seq},H)$ 分解、每个 case/contract 的三个分位数由同一
路径获胜、该路径在每个分位数相对次优路径至少快 2%，且所有 same-$M$ 的形式与分解都
选择同一路径，结果才是 `eligible_for_separate_dispatch_review`。fixed
`B=2/3/4,H=12,T=2048` 另有逐精确形状门禁；即使通过，也仍须做公开 FLA runtime 集成
验证，且不得覆盖既有 fail-closed 条件。

## B300 复现

使用已经构建好的、同时含 `fwd`、`fwd_vshard_p2`、`fwd_vshard4_p2` 的比较 SO；本目录
的 clean audit 不调用 `setup.py`、NVCC 或 patch generator。它会在 PRE/POST 检查单 GPU
空闲，先执行 `py_compile`；runner 还会硬性要求设备名含 `B300`、compute capability 为
10.3、SM 数为 148，并要求已加载 SO 的 SHA256 等于审计值
`8f8cb970...fe3e005`，不匹配即在任何正确性/性能测量前停止：

```bash
export C1_SEQCOUNT_DISPATCH_GPU_AUTHORIZED=1
export A02_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02
export PATCHED_ROOT=/path/to/already-built-comparison-source
export REFERENCE_ROOT=/home/lcpu/85117379/flashkda-1ce47ea
export LABEL=b300_sm103a_r1
bash "$A02_ROOT/team/c1_flashkda/challenge_seqcount_dispatch/run_clean_seqcount_dispatch_audit.sh" \
  --authorized-by-parent
```

在无 GPU 的机器上仅核对矩阵与停止规则：

```bash
python assignment02/team/c1_flashkda/challenge_seqcount_dispatch/run_seqcount_dispatch.py \
  --describe --json /tmp/c1_seqcount_dispatch_plan.json
```

## B300 r2 结果

job 10740 在 `NVIDIA B300 SXM6 AC`（SM103、148 SM）上以审计 SO
`8f8cb970...fe3e005` 完成；PRE/POST 均为 0 MiB、`FINAL_RC=0`。54 个 case 的四种 raw
state contract 中，v2/v4 output 与所有存在的 final state 均逐位等于 baseline；23 个
预先选定的 case-contract 也逐位等于 pinned Torch reference。每个性能 cell 都包含三路径
各 1000 个样本。

只按 $M$ 的通用策略门明确得到 `STOP_do_not_promote_M_only_policy`。最直接的反例是
$M=38$ 的 T=257 控制族：fixed 与 balanced-varlen 的 `none` 三个分位均选 v2，而同总
token、同实际 K1 tile 数的 skewed-varlen 三个分位均选 v4。长度分布会改变负载平衡，不能
从 $N_{seq}H$ 单独推出赢家。M=38/39/40/48/72/75/76/96 还至少有一个公开 state contract
因跨分解不一致、分位赢家不一致或 2% 裕量不足而失败。

fixed `H=12,T=2048` 的逐契约证据如下；“最小裕量”取 P50/P95/P99 中最小的相对次优路径
优势：

| batch / public contract | P50 winner (ms) | 三分位同 winner | 最小裕量 | 当前裁决 |
| --- | ---: | --- | ---: | --- |
| B=2, none | v4 0.161760 | 是 | 8.64% | 进入独立复现 |
| B=2, FP32 final only | v4 0.162208 | 是 | 10.34% | 进入独立复现 |
| B=2, FP32 both | v4 0.166208 | 是 | 14.71% | 进入独立复现 |
| B=3, none | v4 0.170624 | 是 | 8.88% | 进入独立复现 |
| B=3, FP32 final only | v4 0.170592 | 是 | 10.96% | 进入独立复现 |
| B=3, FP32 both | v4 0.174672 | 是 | 14.29% | 进入独立复现 |
| B=4, none | v2 0.195424 | 是 | 3.17% | 进入独立复现 |
| B=4, FP32 final only | v2 0.199888 | 是 | 3.88% | 进入独立复现 |
| B=4, FP32 both | v4 0.209792（P50） | 否；P95 选 v2 | 0.02% | 保持 baseline |

因此预注册的整体 `B=2/3/4` 门也正确返回
`STOP_do_not_expand_fixed_batch_dispatch`；不能事后删掉失败 cell 来宣称原门通过。上表前
8 个明确通过的 cell 将使用新的、事先收窄的独立 repeat 门复现，最后一格固定回退。

原始证据：

- [r2 JSON](results/c1_seqcount_dispatch_b300_sm103a_r2.json)，SHA256
  `46cd27f2...be414f7`；
- [job 10740 clean audit](results/c1_seqcount_dispatch_b300_sm103a_r2_job10740.log)，
  SHA256 `27bff678...c8a0d50`。

## 独立确认与逐 cell 发布门

r2 还把 fixed `B=6/8,H=12,T=2048` 作为 discovery 点：B6 三种公开契约均选 v2，最小
裕量 2.27%；B8 只有 `none` 的 v2 裕量超过 2%，final/both 均接近 baseline。为避免从一次
观测直接发布，[固定批量确认 runner](run_fixed_batch_confirmation.py) 预注册 B2/3/4/6/8
共 12 个正向 cell、两次新 repeat。job 10771 的 11 个 cell 通过；B8 `none` 的一次 P99
裕量只有 1.81%，所以预注册的整体门正确返回 STOP，B8 在本轮当前策略中全部留在 baseline。原始证据：
[confirmation JSON](results/c1_fixed_batch_confirmation_b300_sm103a_r1.json) 与
[clean audit](results/c1_fixed_batch_confirmation_b300_sm103a_r1_job10771.log)。

随后 [逐 cell 发布门](run_fixed_batch_release_gate.py) 以固定 SHA 重新读取 r2 discovery 和
job 10771，从 raw samples 独立重算历史 winner/裕量，再用新 seed 为每个剩余 cell 运行两次
1000-sample repeat。它不使用历史 JSON 的内嵌 pass 布尔值，也不采用全家族 all-or-nothing
晋级：每个精确 shape/state 条目独立裁决。job 10784 中 B6 `fp32_both` 的一次 P99 裕量降到
1.85%，故同样回 baseline；其余 10 个条目取得**进入公开 FLA integration review**的资格：

| fixed shape | none | FP32 final only | FP32 both |
| --- | --- | --- | --- |
| B=2,H=12,T=2048 | v4 candidate | v4 candidate | v4 candidate |
| B=3,H=12,T=2048 | v4 candidate | v4 candidate | v4 candidate |
| B=4,H=12,T=2048 | v2 candidate | v2 candidate | baseline |
| B=6,H=12,T=2048 | v2 candidate | v2 candidate | baseline |
| B=8,H=12,T=2048 | baseline | baseline | baseline |

这里的 `candidate` 仍只表示三轮 raw-wrapper 证据均过门，不等于 FLA registry 已发布；必须
在 dispatcher 精确编码后，由 pinned backend、direct custom backend 和 public
`chunk_kda` 三路逐位对拍并证明 public call 确实经过 custom backend。发布门原始证据为
[release JSON](results/c1_fixed_batch_release_gate_b300_sm103a_r1.json) 与
[job 10784 audit](results/c1_fixed_batch_release_gate_b300_sm103a_r1_job10784.log)。

## public FLA 集成终验

上述 10 个 candidate 已逐项编码进 `challenge_tp8_dispatch/auto_dispatch.py`；B4/B6
`fp32_both`、B8 三种 contract、所有未列入的 `B>1` shape/state 仍保留 baseline。
`challenge_tp8_dispatch/run_fixed_batch_fla_integration.py` 对 B2/3/4/6/8 × 三个 public
contract 共 15 个 cell，分别调用 pinned `FlashKDABackend`、direct custom backend 和真实
`fla.ops.kda.chunk_kda`。每个 public call 前后都检查注册实例的 spy 计数恰好 `+1`，并紧接
读取 decision；因此 baseline 负控也表示“custom backend 被调用后在其内部保守回退”，
不会误读 direct call 留下的旧状态。

最终 clean job 10810 的 15/15 cell 全部逐位等于 pinned 路径，所有存在的 final state 都是
contiguous FP32 `[B,12,128,128]`；10 个正向 cell 和 5 个 baseline 负控均命中预期。
runner 还固定 FLA commit 和 6 份直接源文件 SHA，并断言实际导入的 `fla`、generic
backend、KDA package/registry/backend/chunk 六个模块的 `__file__` 全部位于 `FLA_ROOT`。
PRE/AFTER/POST 均为 0 MiB，`FINAL_RC=0`。最终证据为
[public FLA JSON](../challenge_tp8_dispatch/results/c1_fixed_batch_fla_integration_b300_sm103a_r3.json)
（SHA256 `b3b2fb61b64e03ca77a1b8e41e49bc2c6a4db5b9ceb5c7445dc4c96bcff8657a`）和
[job 10810 audit](../challenge_tp8_dispatch/results/c1_fixed_batch_fla_integration_b300_sm103a_r3_job10810.log)
（SHA256 `4ebadba89f9f484cdd0f9f71ac8eef9d597f61208f3451f7c3cbe12c2a87e90d`）。

因此原始 10 个条目已经从 candidate 晋级为本实验 opt-in public FLA 白名单；这不改变下节
对通用 varlen、其他 batch/shape/state、其他 SO/GPU 和多 rank 的边界。

## B=5 discovery 补点

原矩阵没有 `B=5,H=12,T=2048`；既然本目录的反例已经否决 M-only 与相邻点外推，就必须把
这个 batch 当成独立精确形状测量。续轮 [discovery runner](run_fixed_batch_b5_discovery.py)
和 [独立 analyzer](analyze_fixed_batch_b5_discovery.py) 只读测试 baseline/vshard2-P2/
vshard4-P2，不修改 public dispatcher。clean job 11781 的两个 fresh 进程各做两次
1000-sample repeat；4 个 raw ABI 和 3 个 pinned reference 在每个进程均逐位通过。

| B5 public contract | 四次重复、三个分位的共同 winner | 最小相对次优裕量 | 本阶段权限 |
| --- | --- | ---: | --- |
| none | vshard2-P2 | 6.202% | 仅可进入独立确认 |
| FP32 final only | vshard2-P2 | 5.934% | 仅可进入独立确认 |
| FP32 both | vshard2-P2 | 2.292% | 仅可进入独立确认 |

审计的 `second_allocation_decision.eligible=true` 只授权新 allocation 的 confirmation；尤其
FP32-both 离 2% 门很近，不能从这一次 allocation 直接进入 public mapping。PRE/BETWEEN/POST
均为 0 MiB、`FINAL_RC=0`；证据为 [独立审计](results/c1_fixed_batch_b5_discovery_b300_sm103a_b5_r1.independent_audit.json)、
[本地 raw 复算](results/c1_fixed_batch_b5_discovery_b300_sm103a_b5_r1.local_recompute.json)
和 [clean 日志](results/c1_fixed_batch_b5_discovery_b300_sm103a_b5_r1_job11781.log)。

独立 confirmation job11782 随后在新 allocation、`seed=20260830` 和两个新 PID 中完整复测。
当前 allocation 的 none/FP32-final-only/FP32-both 最小裕量为 6.416%/5.540%/2.227%；与
历史 discovery 合并后，三个 contract 的 8 repeats × P50/P95/P99 均保持 vshard2-P2 且
裕量至少 2%。chain 直接重读当前 raw artifacts，验证 SHA、process index 和 seed 公式，并
要求历史/当前 artifact SHA 集合、Slurm job ID、日志路径不相交。job11782 为
`COMPLETED/0:0`，POST 0 MiB、`FINAL_RC=0`；其 exact-bool
`eligible_for_public_integration_review=true` 只允许人工进入真实 dispatcher/public FLA 终验，
不会自动改表。证据为 [confirmation chain](results/c1_fixed_batch_b5_confirmation_b300_sm103a_b5_confirm_r1.confirmation_chain.json)、
[本地 chain 复算](results/c1_fixed_batch_b5_confirmation_b300_sm103a_b5_confirm_r1.confirmation_chain.local_recompute.json)
和 [job11782 clean 日志](results/c1_fixed_batch_b5_confirmation_b300_sm103a_b5_confirm_r1_job11782.log)。

通过该门后，production candidate 才把 B5 三个精确 public contract 加入 vshard2-P2 映射，
并把既有 public runner 扩为 18 cell = 13 正 + 5 负。clean job11786 的 pinned/direct/public
三路 18/18 全部逐位一致；每个 public registry spy 恰 `+1`，B5 三格 decision 均为
vshard2-P2，旧 15 格没有回归。作业 `COMPLETED/0:0`，PRE/AFTER/POST 0 MiB、
`FINAL_RC=0`。证据为 [18-cell public JSON](../challenge_tp8_dispatch/results/c1_fixed_batch_fla_integration_b300_sm103a_b5_public_r1.json)
（SHA256 `7867854fee67d8632ab09d08d1b0f1b8f0bee2f632f9c12943c4a5da9ba18a1f`）和
[job11786 audit](../challenge_tp8_dispatch/results/c1_fixed_batch_fla_integration_b300_sm103a_b5_public_r1_job11786.log)
（SHA256 `1767cdea2355bd6884f9d555fd0b4812c89f07899e1dc349cf612ea251316e94`）。

fresh-allocation job11787 再用 `seed=20260831` 运行相同 18-cell production source，18/18
逐位通过，`COMPLETED/0:0`、POST 0 MiB、`FINAL_RC=0`。其 raw JSON SHA256 为
`feb6840171f7ed1059d7a2aede0cfc340ff1550900808ce9100427410add3850`，完整日志为
`1ffd36c8c8d2297234db6001456a0a6e131d16f8120a06d7813a799bdab7b259`，均与 job11786
证据独立。初版 freeze analyzer 会接受 shape 中的 JSON float `5.0`，故它在 job11787 生成的
`...b5_prod_freeze_r1.production_freeze.json`（SHA256 `f7f74a7b…960e`）已废弃。
[修正版 analyzer](../challenge_tp8_dispatch/analyze_fixed_batch_b5_production_freeze.py)
逐维强制 exact integer，并在系统类型/结构变异中
零漏过；对 job11787 的离线交叉复核只采信 [远端 strict audit](../challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r1.production_freeze.strict_recompute.json)
（SHA256 `c912f6d38bb937026433a7dacd201d73e572f5c7fd20922ebc1e63b70587c691`）、
[本地 strict audit](../challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r1.production_freeze.strict_local_recompute.json)
和 [job11787 clean 日志](../challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r1_job11787.log)。
修正版 shell 最后在 fresh job11788 中原位完成同一 18-cell GPU run 与严格 freeze gate：
`COMPLETED/0:0`、POST 0 MiB、`PRODUCTION_FREEZE_PASSED=true`、`FINAL_RC=0`。最终主证据为
[freeze JSON](../challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r2_strict.production_freeze.json)
（SHA256 `38bc8bba519a5d9f1f2de875baf28a6f3fb2a097238c7bf958344ef6897bb7f5`）、
[raw public JSON](../challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r2_strict.json)
和 [job11788 clean 日志](../challenge_tp8_dispatch/results/c1_fixed_batch_b5_production_freeze_b300_sm103a_b5_prod_freeze_r2_strict_job11788.log)
（SHA256 `8f037aa9257c30685e38a3fc4f8d31bcde1953c254094cf232af3cb484592294`）。

## 当前边界

$M$ 只刻画 K2 `(sequence, head)` CTA 数；它不包含 varlen 的 K1 upper-bound grid、序列
长度分布、workspace、state ABI、GPU 代际或多-rank contention。CPU-authoritative
descriptor、device cache 与 one-shot verifier→body 可信链已经完成；任意 layout 仍缺真实
调用方对 CPU offsets 生命周期的证明，以及每个新 distribution/state 独立通过的 raw 与
public 性能门。因此未列 cell 的公开 dispatcher 仍必须 fail closed。任何结果都只能说明
该矩阵与该审计 SO 的范围。
