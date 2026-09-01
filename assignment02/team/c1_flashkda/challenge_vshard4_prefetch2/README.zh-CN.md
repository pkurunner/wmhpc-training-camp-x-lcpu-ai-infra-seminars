# C1 B300：vshard4 + P2S3 组合实验

| 变量 | 含义 | 本挑战取值 |
| --- | --- | --- |
| `T,H,D` | token、每卡 head、特征维 | 正式目标 `8192,12,128`；另做 head sweep |
| `N_SM` | B300 SM 数 | `148` |
| `V_s` | vshard4 单 CTA value 宽度 | `32`，K2 grid=`4H` |
| `P,S` | phase-6 预取距离、input stages | `2,3` |
| P1/P2 | 无/有双环软件预取 | 本目录在同一 SO 比较两者 |

本目录从干净 FlashKDA `1ce47ea` 一次性生成 baseline、vshard2-P1、vshard2-P2、
vshard4-P1、vshard4-P2 五个公开 entry。它不会顺序拼接两个旧 patch；四条计时路径
因此能在同一进程、同一 extension 与同一 allocation 内比较。

## 已验证结论

fresh B300 SM103a build 的 extension SHA-256 为
`8f8cb97077d2496834d065c7cc1e39980e14e1f9b1665cdfd3aa8ae2dfe3e005`。正式
vshard4-P2 fixed BF16 initial+final-state 实例为 59 registers、9 barriers、0 spill。
`H=1/2/4 x none/BF16/FP32` 与正式 H12 均相对 baseline bitwise exact，对 pinned
Torch reference 也通过。

ptxas JSON 的 `log_sha256_at_parse` 绑定 parser 运行时已由 `tee` 写出的日志前缀；随后
打印的 JSON、git status 与结束时间仍会追加到同一日志。完整 build log 的最终 SHA-256
另存于 `results/c1_vshard4_p2_build_b300_r1.log.sha256`，两者作用域不同。

| clean repeat | baseline P50 | vshard2-P2 P50 | vshard4-P1 P50 | vshard4-P2 P50 | vshard2-P2 / vshard4-P2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| job 10005 | 0.792096 ms | 0.595440 ms | 0.593184 ms | **0.529472 ms** | **1.124592x** |
| job 10008 | 0.799968 ms | 0.595136 ms | 0.591744 ms | **0.529472 ms** | **1.124018x** |

job 10173 在同一 B300、同一 SO 上逐整数穷举 `H=1–96`，每个 H 的四条路径各保留
500 个 raw samples；96/96 shape 的 baseline exact 与 Torch reference gate 全部通过。
P50/P95/P99 三种分位数的胜负符号都给出同一个完整分界：H1–37 选择 vshard4-P2，
H38–96 选择 vshard2-P2。独立 1000-sample repeat 中，vshard4-P2 相对 vshard2-P2
在 H37 为 1.116044x，在 H38 为 0.924324x。K2-only NCU 进一步回读：

| `H` | grid | K2 duration | Compute (SM) | DRAM | L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 37 | 148 CTA | 470.43 μs | 19.63% | 11.32% | 43.09% |
| 38 | 152 CTA | 593.63 μs | 15.86% | 9.55% | 34.84% |

因此 H12 是明确正结果；对已经逐整数覆盖的 B300 fixed
`T=8192,D=128,BF16-state,H=1–96` 轴，dispatch 为 `H≤37` 选 vshard4-P2、`H≥38`
选 vshard2-P2。该不等式不能
外推到未测的长度、state、batch 或架构；这些组合仍须走白名单或运行时回退。

全 sweep 的机器可读入口为
[`c1_vshard4_p2_b300_sm103a_hs_full_r1_summary_job10173.csv`](results/c1_vshard4_p2_b300_sm103a_hs_full_r1_summary_job10173.csv)，
每行绑定源 JSON SHA；完整 audit 为
[`c1_vshard4_p2_b300_sm103a_hs_full_r1_head_sweep_job10173.log`](results/c1_vshard4_p2_b300_sm103a_hs_full_r1_head_sweep_job10173.log)。
98-member [原始证据包](results/c1_vshard4_p2_hs_full_r1_job10173.tgz)及其
[SHA-256 sidecar](results/c1_vshard4_p2_hs_full_r1_job10173.tgz.sha256)也保存在 `results/`。

## 文件

- `apply_vshard4_prefetch2_patch.py`：校验三份既有 generator hash，并从 clean tree
  一次性生成同一 SO 的比较 entry。
- `build_fresh_b300_sm103a.sh`、`ptxas_audit.py`：fresh SM103a build 与资源 gate。
- `run_vshard4_prefetch2_final.py`：all-state exact 与四路径循环计时，保留 raw samples。
- `run_clean_vshard4_prefetch2_audit.sh`：H12 正式 clean audit。
- `run_clean_head_sweep_audit.sh`：每个 H 使用独立 Python 进程并做逐 shape 清卡审计。
- `summarize_head_sweep.py`：校验全整数 sweep 的 exact/sample/SO 门并生成 source-bound CSV。
- `ncu_single_variant.py`：每进程只调用一个公开 wrapper 一次。
- `run_clean_h12_ncu_audit.sh`：四路 K2 Basic 与胜者 K1+K2 Full。
- `run_clean_h37_h38_ncu_audit.sh`：dispatch 边界的 K2-only Basic。

## 复现顺序

先在 CPU allocation 中从 pinned clean tree 构建；构建脚本要求显式
`C1_VSHARD4_P2_BUILD_AUTHORIZED=1`。只有 ptxas formal gate 为零 spill 后，才进入 GPU：

```bash
export C1_VSHARD4_P2_GPU_AUTHORIZED=1
export A02_ROOT=/remote/assignment02
export PATCHED_ROOT=/remote/flashkda-vshard4-prefetch2-1ce47ea
export REFERENCE_ROOT=/remote/flashkda-1ce47ea
export PYTHON_BIN=/remote/assignment02/.venv/bin/python
export PYTHON_INCLUDE=/remote/python/include/python3.12
export RESULTS_DIR="$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2/results"
export LABEL=b300_sm103a_h12_repeat
bash "$A02_ROOT/team/c1_flashkda/challenge_vshard4_prefetch2/run_clean_vshard4_prefetch2_audit.sh" \
  --authorized-by-parent
```

head sweep 复用相同环境，另设 `HEADS` 和 `SAMPLES` 后运行
`run_clean_head_sweep_audit.sh`。NCU 脚本必须在 Slurm GPU allocation 内执行，并分别要求
`C1_H12_NCU_AUTHORIZED=1` 或 `C1_BOUNDARY_NCU_AUTHORIZED=1`。

## 证据与边界

机器可读 JSON、CSV、ptxas 账本和 clean logs 均在 [results](results/)。Full NCU 原报告
以 `.ncu-rep.gz` 无损保存；CSV 是可检索 counter，NCU replay duration 不替代
CUDA-event full-call P50。

目录中保留的 job 9996、10099 和 10171 仅是基础设施诊断：9996/10171 都在 Python
header environment gate 以 RC=89 退出，10099 是临时 stdin shell harness 在 NCU 前因
EOF 语法错误退出；9996/10171 只有 POST=0 MiB，10099 的 PRE/POST 均为 0 MiB，且
三者都没有启动被测 kernel。正式结论只取
job 10005/10008/10044/10053/10079/10085/10105/10173。

本挑战只证明 B300 fixed `T=8192,D=128` 的已测 state/heads。尚未完成：真实多卡 TP8
集成与自动 dispatch、varlen/尾 chunk、跨长度/跨架构表、长上下文 BF16-vs-FP32 模型
质量。H64/H96 原 strict +10% 门槛也仍未达到。
