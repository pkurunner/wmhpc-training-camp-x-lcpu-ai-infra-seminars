# C1 B300：skew packed-varlen FP32-both 的真实生产路由冻结

本目录是对已集成生产路由的窄功能/路由复验。它不修改 dispatcher、backend、wrapper 或
任何 production map；也不使用 test-only 标记。一次 A1 预试（job `12598`）在目标真实
production FP32-both 路由与 exact 合同通过后，暴露了相邻-offset 负控错误复用了前一调用的
one-shot backend handoff 的协议隔离缺陷；该 job 不构成 A1 证据、不能进入 chain。修复后的协议
随后在 fresh job `12770`/`12771` 完成 A1/A2，并生成 `eligible_for_production_freeze=true`、
`complete=true` 的最终 freeze；历史失败仍保留，但不进入新证据链。

| 变量 | 含义 | 本协议固定值 |
| --- | --- | --- |
| `B` | packed-varlen 外层 batch | `1` |
| `H` | attention head 数 | `12` |
| `T` | 总 token 数 | `12288` |
| `o_i` | CPU-authoritative 第 `i` 个 sequence offset | `(0,1,2,3,4,5,12288)` |
| `A` | 相互独立的 Slurm allocation | `A1`、`A2` |
| `p` | 一个 allocation 内的 fresh Python PID 序号 | `0`、`1` |

## 冻结事实

每个 raw runner 在任何 CUDA 初始化前读取生产
`_VARLEN_PUBLIC_VARIANTS`，并要求上述 offsets 的三个 state contract 恰为：

| contract | 必须存在的生产 variant |
| --- | --- |
| `none` | `vshard2_p2` |
| `fp32_final_only` | `vshard2_p2` |
| `fp32_both` | `vshard4_p2` |

完整 map 的 typed canonical serialization 与 digest 会在 CUDA 前保存、workload 后重新生成；
两份 serialization/digest 与 map object identity 都必须不变。它不会在协议源码中写死，因为同一最终
production 提交还可能包含其他已经独立审计的 cell。相反，生产 `auto_dispatch.py` 和
`fla_backend.py` 的最终 SHA-256 必须由提交命令以环境变量传入；shell、runner 和 analyzer
均重新核验并绑定它们。这避免把集成前的旧源哈希误当成最终证据。

每个 PID 使用真实 `fla.ops.kda.chunk_kda` registry call，要求：

- 目标 FP32-both 路由的 backend spy 为 C1 `+1`、pinned `+0`，decision 精确为
  `vshard4_p2` / `varlen_skew_n6_h12_t12288_fp32_both_whitelist_hit`；
- public、direct C1、pinned 与 pinned torch reference 的 output 和 FP32 final state 均逐 bit
  exact；输入与 initial state 不可变；
- 在一个 timing 外的真实 public call 中，CPU descriptor issuance 恰一次，issuer 接收的
  CPU offset tensor 与 caller 的 tensor 是同一对象；descriptor 再以该同一 CPU tensor 验证，且
  C1/pinned route 仍为 `+1/+0`；
- `fp32_final_only` 这个相邻已发布 state contract 精确走 `vshard2_p2`，而相邻 offsets
  `(0,1,2,3,4,6,12288)` 必须 fail-closed 到 pinned `baseline`；每个相邻控制前都显式清除
  C1 的 thread-local one-shot handoff 与 `varlen_metadata` cache。随后真实 public registry
  verifier 必须恰好一次接收当前的 CPU/GPU offset tensor，issuer 恰好一次以当前 CPU tensor
  发证并复验 offsets；目标 state 的 verifier 接受，邻接-offset 的 verifier 必须以
  `C1 packed-varlen preflight rejected: varlen_offsets_not_whitelisted` 拒绝。负控不再将
  `auto_dispatch.get_last_decision()`（只在实际 C1 launch 时更新）误用为拒绝路径证据；
- 在 CUDA 前另做一个受限的 v4-symbol-missing 负控：只临时替换私有 extension-loader 的返回
  inventory（绝不替换 `auto_dispatch.fwd`，也不改 map），使其只报告 v2 symbol；真实生产的
  `_choose_available_variant` 必须返回 `baseline`，不能降级为未审计的 v2，随后 loader 原样恢复；
- B300、SM103、148 SM、单 GPU UUID、0 MiB pre-Torch 清洁度、audited extension、patched
  dirty-set、pinned helper、FLA public callable 和完整 source ledger 均须通过。

每个 allocation 有两个 fresh PID。A2 在 GPU workload 前通过标准库 analyzer 单次
`read_bytes()` 重新哈希并解析 A1 manifest/raw JSON，强制 A1 与 A2 为不同的正十进制
Slurm job。最终 `freeze` 再重新读取两个 allocation、两个 PID 的 raw evidence 和 A2→A1
精确 SHA binding。runner、shell 和 analyzer 都只接受 canonical 实际执行文件；递归严格 JSON
类型比较拒绝 `true` 伪造 `1` 或 `1.0`。pinned reference 的 loader evidence 还精确绑定 helper
path/SHA、`load_contract`、唯一 `sigmoid_ext` interception 和 `no_build=true`。

## 静态验证

```bash
python team/c1_flashkda/challenge_varlen_fp32_both_production_freeze/run_varlen_fp32_both_production_freeze.py --self-test
python team/c1_flashkda/challenge_varlen_fp32_both_production_freeze/analyze_varlen_fp32_both_production_freeze.py self-test
bash -n team/c1_flashkda/challenge_varlen_fp32_both_production_freeze/run_clean_varlen_fp32_both_production_freeze.sh
```

## 修复版冻结身份与 A1/A2 入口

隔离修复已经完成独立只读复核，P0/P1/P2 均无。提交时固定以下身份；若任一 production
source 此后变化，必须重新冻结并重审，不能沿用本表：

| 工件 | SHA-256 |
| --- | --- |
| runner | `9f76b876ad0749698a9b4fafae13d88c6877dce55efee07a988e06b9624c3707` |
| stdlib analyzer | `1cdfbfd9014121bf0a44693165541dc278ae1b5083dc73153aedab14222d0873` |
| canonical clean shell | `05b1274dc4ac816047ac599cb0dffa7d3561591bfb6f50cea6b0dd1b11c9db35` |
| production `auto_dispatch.py` | `9633179c6a5e8ce61b1857a102f81074c6bdfdc63163e13980f40c24d6583c29` |
| production `fla_backend.py` | `152ac4ba897c1887963553d592456b7f988c89105e6f6cb507a44757058d5fd1` |
| pinned `sigmoid_ext.so` | `8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f` |

先在一个干净 B300 allocation 运行 A1。A2 必须换用另一个 Slurm job，并传入 A1 manifest
路径及其由父提交方预先记录的 SHA。shell 拒绝任一外部 SHA、source ledger、FLA/patch 状态
或 clean-GPU gate 不匹配的请求。

```bash
export C1_SKEW_PRODUCTION_FREEZE_GPU_AUTHORIZED=1
export A02_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02
export PATCHED_ROOT=/home/lcpu/85117379/flashkda-vshard4-prefetch2-1ce47ea-b300-r1
export REFERENCE_ROOT=/home/lcpu/85117379/flashkda-1ce47ea
export FLA_ROOT=/home/lcpu/85117379/fla-a3edffc
export PYTHON_BIN="$A02_ROOT/.venv/bin/python"
export ALLOCATION_ID=A1 LABEL=b300_sm103a_skew_production_r2_isolation
export C1_PINNED_REFERENCE_HELPER_PATH=/home/lcpu/85117379/.cache/torch_extensions/py312_cu130/sigmoid_ext/sigmoid_ext.so
export C1_PINNED_REFERENCE_HELPER_SHA256=8c524aab9a5bf91e069d7216c3e1879fa6e3b1f61a5affa7ee3eeaa91008622f
export C1_SKEW_PRODUCTION_FREEZE_RUNNER_SHA256=9f76b876ad0749698a9b4fafae13d88c6877dce55efee07a988e06b9624c3707
export C1_SKEW_PRODUCTION_FREEZE_ANALYZER_SHA256=1cdfbfd9014121bf0a44693165541dc278ae1b5083dc73153aedab14222d0873
export EXPECTED_PROTOCOL_SHELL_SHA256=05b1274dc4ac816047ac599cb0dffa7d3561591bfb6f50cea6b0dd1b11c9db35
export C1_SKEW_PRODUCTION_AUTO_DISPATCH_SHA256=9633179c6a5e8ce61b1857a102f81074c6bdfdc63163e13980f40c24d6583c29
export C1_SKEW_PRODUCTION_FLA_BACKEND_SHA256=152ac4ba897c1887963553d592456b7f988c89105e6f6cb507a44757058d5fd1
export CANONICAL_PROTOCOL_SHELL="$A02_ROOT/team/c1_flashkda/challenge_varlen_fp32_both_production_freeze/run_clean_varlen_fp32_both_production_freeze.sh"
sbatch --export=ALL --wrap="bash $CANONICAL_PROTOCOL_SHELL --authorized-by-parent"
```

不要将 shell 内容复制给 `sbatch` 的 spool 文件后执行；shell 会将 `readlink -f
"${BASH_SOURCE[0]}"` 与它自己的 canonical expected path 做严格比较，并对实际执行文件做
外部 SHA gate。A2 同样必须以该 canonical shell 入口提交。

之后调用 analyzer 的 `freeze` 子命令仅生成证据决策；它没有 production side effect。

## 2026-08-30 正式生产冻结结果

修复版 A1 为 Slurm job `12770`，A2 为不同的 job `12771`；每个 allocation 各有两个 fresh
Python PID。四份 raw evidence 都通过精确目标 public `vshard4_p2`、相邻
`fp32_final_only -> vshard2_p2`、相邻 offsets fail-closed baseline、v4-symbol-missing
baseline、public/direct/pinned/reference 逐 bit exact、输入/initial-state 不变、source/map
前后不变，以及 PRE/POST clean-GPU 门。A2 在 CUDA 前重新打开并绑定 A1；最终 freeze 又重新
打开两份 allocation manifest 与四份 raw，确认两个不同 Slurm job 和同一 production identity。

| 工件 | 相对链接 | SHA-256 |
| --- | --- | --- |
| A1 PID 0 | [raw](results/c1_varlen_fp32_both_production_A1_b300_sm103a_skew_production_r2_isolation_pid0.json) | `fb79922c5c0bc560704a8eb9f002f60f72c21c5e3178081d0fabcd68421e5951` |
| A1 PID 1 | [raw](results/c1_varlen_fp32_both_production_A1_b300_sm103a_skew_production_r2_isolation_pid1.json) | `baa32475f01195cb6aa3974e6f1778dcef76613a4749cd27df8710ee67a8d29b` |
| A1 allocation | [manifest](results/c1_varlen_fp32_both_production_A1_b300_sm103a_skew_production_r2_isolation_allocation.json) | `7fc7d86c9927d7f741ceeb5b3f23c9add055de4b62b6656e22adc9d1269a3dd4` |
| A2 PID 0 | [raw](results/c1_varlen_fp32_both_production_A2_b300_sm103a_skew_production_r2_isolation_pid0.json) | `20e70873a27d21bb0f24e453835f05a50bfff32c2a17b5f1e18e278ab85e4359` |
| A2 PID 1 | [raw](results/c1_varlen_fp32_both_production_A2_b300_sm103a_skew_production_r2_isolation_pid1.json) | `b85f6634987187259d96de17eb23a08b3e8aea4b0ceff974b3203cc662f5919e` |
| A2 allocation | [manifest](results/c1_varlen_fp32_both_production_A2_b300_sm103a_skew_production_r2_isolation_allocation.json) | `baf2126daf956e0889d4843c4bda7a070d2804c15ebb769d6d046d46236e2f51` |
| A1→A2 freeze | [freeze](results/c1_varlen_fp32_both_production_b300_sm103a_skew_production_r2_isolation_A1_A2.freeze.json) | `bafa65f83406f301fb8d699fda1451bdb50a5fcd6eb5bbb7ca3e0b5ac158ad12` |
| A1 clean log | [job12770](results/c1_varlen_fp32_both_production_A1_b300_sm103a_skew_production_r2_isolation_job12770.log) | `c1f5d71b268ef9ac4f150b27f6908d33fc5bef3695b1ebd1bb3f4b4d121b1fb1` |
| A2 clean log | [job12771](results/c1_varlen_fp32_both_production_A2_b300_sm103a_skew_production_r2_isolation_job12771.log) | `1791188dd9a765a75e42d4969b35023da1ae6eb9245364a3a38b7e85fa71d47c` |

最终 artifact 的 `production_action` 是
`already-integrated route confirmed; no mutation by freeze`。因此这里闭合的是已集成精确
tuple/state 的真实生产路由，不新增 map，也不把该结果外推到任意 packed-varlen layout。
本地副本已逐文件完成 JSON 解析、SHA-256 重算、四 raw→两 manifest→freeze 的引用绑定与
字段级复核；最终八份 JSON 关系一致。这个协议没有 latency samples 或 performance gate，故其
结论严格限于生产路由、correctness、负控、身份和 clean-GPU 证据，不能据此新增性能声明。
