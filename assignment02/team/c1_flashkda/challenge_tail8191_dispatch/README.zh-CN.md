# T=8191 的 state-specific public FLA 路由协议

本目录记录 production 集成前、当时尚未发布的精确表项资格证据：B300 上 fixed
`B=1,H=12,T=8191,K=V=128` 的 `none` 与 `fp32_final_only` 能否经由真实
`fla.ops.kda.chunk_kda` public API 使用测试期 `vshard4-P2` 路由，并在两次独立
allocation 中都稳定快至少 2%。目录中的注入仅修改**当前 Python 进程内** C1 backend
所引用的 module attribute；退出时恢复，绝不编辑 `auto_dispatch.py`、FLA 或 production map。

| 变量 | 含义 | 本协议取值 |
| --- | --- | --- |
| `B` | fixed batch 的序列数 | 1 |
| `H` | KDA head 数 | 12 |
| `T` | 每条序列 token 数 | 8191 |
| `K,V` | key/value 通道维度 | 128, 128 |
| `s` | state contract | `none` 或 `fp32_final_only` |
| `a∈{A1,A2}` | 彼此独立的 clean Slurm allocation | 两次 |
| `p` | 公共 API 路径 | `pinned_public` 或 `c1_test_route_public` |

## 预注册范围与隔离

- test-only route 是精确 predicate：仅 `(B,H,T,K,V)=(1,12,8191,128,128)` 且
  `s∈{none,fp32_final_only}`；它直接选择审计 SO 的 `fwd_vshard4_p2`。任一
  shape、dtype、连续性、state、varlen descriptor、符号或 SO 身份不匹配都会 fail closed，
  不会退化为把不合格输入偷偷送到 benchmark。
- **本协议执行时不**把 `8191` 加到 production `_H12_LENGTHS`。该做法会连带放开
  `bf16_both`/`fp32_both`，超出证据范围。每个 runner 在安装 test route 前调用未改动的
  当时的 production `select_variant`，并要求 T=8191 的这两个 negative contract 均是 pre-launch
  `baseline`。
- raw ABI correctness 对两个正向 contract 都比较 `baseline`、`vshard4-P2` 与固定
  upstream `tests/torch_ref.py`：所以有四个 `path × contract` 的 pinned-reference exact
  检查，并附加 v4-vs-baseline exact 检查。输出和存在的 FP32 final state 均逐位一致。
  每次调用还核验输入与 initial state 不变。
- 性能并非 raw proxy：两个路径都是 FLA registry 的真实 `chunk_kda` public call。每次
  `pinned_public` call 临时关闭 C1 backend，要求 pinned spy `+1`、C1 spy `+0`；每次
  test route call 临时开启 C1，要求 C1 spy `+1`、pinned spy `+0`，并紧接读取
  `chosen_variant=vshard4_p2,test_only_route=true`。

## 固定测量与门槛

一个 allocation `a` 内有两个全新 Python PID；每个 PID 对每个 `s` 独立做两个 repeat。
每一 repeat 对每条 public path 各做 100 次 cyclic warm-up 和 **1000** 个 CUDA-event sample。
两条路径交替首位，恰各 500 次；每个 event 是
`start -> 一个完整 public chunk_kda call -> end -> end.synchronize`，同步不计入 sample。

stdlib-only analyzer 不信任 runner summary/pass 字段：它从 raw samples 重算 P50/P95/P99。
每个 `a,s` 的四个 repeat 在所有三个分位数都必须满足

`latency(pinned_public) / latency(c1_test_route_public) - 1 >= 0.02`。

因此 A1 的通过只允许安排 A2；A2 的通过只允许进入 cross-allocation audit。只有 A1/A2
均通过、且两个 Slurm job ID 不同，chain 才写
`eligible_for_public_freeze=true`。该 machine label 也不会自动修改任何生产文件；实际集成还须
进行单独 code review 与 public correctness 终验。
allocation audit 会先把原始 JSON 写盘；若 `eligible` 不是精确 `true`，clean shell 随即以
非零状态退出，因此父流程不会把失败轮当成可继续 A2 的成功作业。

## 身份和 clean-GPU gate

runner 与 shell 固定 runner/analyzer、C1 dispatcher/backend、harness、FLA 六个入口、pinned
Torch reference、patched/reference/FLA commit、FlashKDA Python wrapper 和 SO SHA。runner 还
要求 B300、SM103、148 SM、一个可见 GPU 和正的 Slurm job ID；analyzer 要求两 PID 不同、同一
allocation 内同一 GPU UUID/SO/reference SHA。shell 在 PRE、两个 PID 之间、独立审计后和 POST
均要求 0 MiB/无 compute app，并且没有 build、NVCC、patch generator 或源文件修改命令。

## 执行

先在已经获得的单卡 B300 Slurm allocation 内运行 A1：

```bash
export C1_TAIL8191_DISPATCH_GPU_AUTHORIZED=1
export A02_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02
export PATCHED_ROOT=/home/lcpu/85117379/flashkda-vshard4-prefetch2-1ce47ea-b300-r1
export REFERENCE_ROOT=/home/lcpu/85117379/flashkda-1ce47ea
export FLA_ROOT=/home/lcpu/85117379/fla-a3edffc
export PYTHON_BIN=/home/lcpu/85117379/codex-a02-20260819-main/assignment02/.venv/bin/python
export LABEL=b300_sm103a_tail8191_r1
export ALLOCATION_ID=A1
bash "$A02_ROOT/team/c1_flashkda/challenge_tail8191_dispatch/run_clean_tail8191_dispatch_audit.sh" \
  --authorized-by-parent
```

仅当 A1 的 allocation audit 中 `allocation_gate.eligible` 是精确 `true`，才在**新的** Slurm
allocation 中把 `ALLOCATION_ID=A2` 后运行相同命令。之后由 CPU-only analyzer 链接两份已哈希
的 allocation audit：

```bash
python "$A02_ROOT/team/c1_flashkda/challenge_tail8191_dispatch/analyze_tail8191_dispatch.py" --chain \
  --a1-audit "$A1_AUDIT" --a2-audit "$A2_AUDIT" \
  --expected-a1-sha256 "$(sha256sum "$A1_AUDIT" | awk '{print $1}')" \
  --expected-a2-sha256 "$(sha256sum "$A2_AUDIT" | awk '{print $1}')" \
  --expected-analyzer-sha256 40297138e2a9fd6c0b58c159bd8801750e0842c2c30f6b6a708e29bfb779594f \
  --json "$CHAIN_JSON"
```

无 GPU 时可做语法/审计自检：

```bash
python -m py_compile run_tail8191_dispatch.py analyze_tail8191_dispatch.py
python analyze_tail8191_dispatch.py --self-test
bash -n run_clean_tail8191_dispatch_audit.sh
```

## 2026-08-30 双 allocation 结果

A1/job12406 与 A2/job12415 是不同 Slurm job，并落在不同 GPU UUID；各自均完成
2 fresh PID × 2 repeat × 2 contract × 2 public path × 1000 samples。raw/reference、
public exact、输入/initial-state不变、registry route、生产负控和 B300/source/SO 身份门
全部通过。

| allocation | `none` 的四 repeat×三分位裕量范围 | `fp32_final_only` 的范围 |
| --- | ---: | ---: |
| A1/job12406 | 38.211%–47.425% | 35.775%–41.963% |
| A2/job12415 | 41.754%–47.920% | 36.504%–41.943% |

每 allocation 24 个、两 allocation 合计 48 个预注册 repeat×分位单元的全局最小裕量为 **35.775%**，远高于 2% 门。独立 chain 的
SHA-256 为 `aef81bab8e53cd730e350d84cfff1b149f57cbee42fdd8a907f2763416ab3ec3`，并写出
`eligible_for_public_freeze=true`。证据入口为
[A1 audit](results/c1_tail8191_dispatch_b300_sm103a_tail8191_r1_A1.allocation_audit.json)、
[A2 audit](results/c1_tail8191_dispatch_b300_sm103a_tail8191_r2_A2.allocation_audit.json) 与
[cross-allocation chain](results/c1_tail8191_dispatch_b300_sm103a_A1_A2.chain.json)。

这两次 allocation 的意义是授权精确 production 集成；它们仍是 test-only route 证据，
不冒充最新 production selector 的冻结结果。正式生产路径随后已由 job12592/12593 与
A1→A2 chain 重新验证并写出 `production_freeze_passed=true`；见
[production-freeze README](../challenge_tail8191_production_freeze/README.zh-CN.md)。两类 evidence
仍分开保存，不能用后来的 production 通过标志回写本 test-only artifact。
