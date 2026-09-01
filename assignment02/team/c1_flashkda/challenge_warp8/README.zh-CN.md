# C1：全 V、8 个 MMA warp 的 K2 候选

本目录新增的是一个与 upstream `flash_kda.fwd` 隔离的 C1 候选：K1、workspace
格式、张量 ABI 和状态 ABI 都保持原状，仅克隆 K2 并通过新的
`flash_kda.fwd_warp8` 入口运行。它不修改本仓库里的 `FlashKDA/` 快照，也不和
`challenge_vshard` 共用补丁目标。

## 变量表

| 变量 | 含义 | 范围/取值 |
| --- | --- | --- |
| `B` | batch 中定长序列数 | `b in [0,B)` |
| `T` | 每条序列 token 数 | 基准使用 8192 |
| `H` | KDA head 数 | 官方比较为 64、96 |
| `D=V` | key/state 行维与 value 列维 | 固定 128 |
| `c` | K2 的 16-token chunk 下标 | `c in [0, ceil(T/16))` |
| `w` | K2 MMA warp 下标 | `w in [0,8)` |
| `j` | `w` 所有的 V 列块 | `[16w,16w+16)` |
| `S_c` | chunk `c` 之后的状态 | 逻辑形状 `[D,V]` |
| `U_c` | chunk 内校正量 | 形状 `[16,V]` |

## 设计与安全边界

对每个 chunk，K2 的关键部分是
`O_c = Q'_c S_(c-1) + M_c U_c` 与
`S_c = diag(exp(g_c)) S_(c-1) + K'_c^T U_c`。
这里的 `V` 列无需跨列归约；原 K2 已让四个 warp 各处理两个 16 列块。本候选把
同一个完整 `V=128` CTA 改为八个 MMA warp，每个 warp 处理恰好一个 16 列块，
并保留一个 TMA-load warp、一个 TMA-store warp。因此
`NumThreads = 8*32 + 32 + 32 = 320`。

这是调度/寄存器并行度候选，而不是 value 分片：一个 `(sequence, head)` 仍只有
一个 CTA，避免 vshard 在 K1 中间量、beta 及状态 TMA 上的双份读取。所有八个 MMA
warp 都会到达 `NamedBarrier(256)`；每个 warp 写入互不重叠的 16 列 state/output
块，之后才由原有 pipeline 提交/释放。补丁固定 `D=128`，任何未验证 shape 不应
被宣称为支持。

## 静态检查、隔离构建与验证

在 B300 的干净专用 clone 中执行；不要在 assignment 快照或既有 vshard clone 上
运行补丁：

```bash
git clone --recurse-submodules https://github.com/MoonshotAI/FlashKDA.git \
  /home/lcpu/85117379/flashkda-warp8-1ce47ea-r1
git -C /home/lcpu/85117379/flashkda-warp8-1ce47ea-r1 checkout 1ce47ea
python assignment02/team/c1_flashkda/challenge_warp8/static_check.py \
  --source /home/lcpu/85117379/flashkda-warp8-1ce47ea-r1
python assignment02/team/c1_flashkda/challenge_warp8/apply_warp8_patch.py \
  --source /home/lcpu/85117379/flashkda-warp8-1ce47ea-r1
cd /home/lcpu/85117379/flashkda-warp8-1ce47ea-r1
CXX=g++ FLASH_KDA_CUDA_ARCHS=103a NVCC_THREADS=8 \
  /home/lcpu/85117379/codex-a02-20260819-main/assignment02/.venv/bin/python \
  setup.py build_ext --inplace
```

构建后先使用现有 harness 的相同输入与 state 配置，让 `flash_kda.fwd`、
`fwd_warp8` 与 torch reference 三者逐元素对拍。只有正确性通过，才能在同一 B300
keeper、清卡、同一 CUDA-event 边界、相同 warmup/iteration 下和“当前 best”比较。
该目录不写入也不伪造性能结论；尤其不把 `H=64` 或 `H=96` 的单点结果外推成全部
shape 的结论。

`run_candidate_bench.py` 是不依赖公共 harness 的最小三方比较 runner。每次进程只
加载一个扩展，避免两个不同 `flash_kda_C` 同名 Python module 相互污染；因此在同一
节点、相同 event 合同下分别运行 warp8 和既有 vshard，再读取两个 JSON 比较：

```bash
# warp8 clone：small exact gate，然后 H64/H96 正式计时。
PYTHONPATH=/home/lcpu/85117379/flashkda-warp8-1ce47ea-r2:$PYTHONPATH \
python assignment02/team/c1_flashkda/challenge_warp8/run_candidate_bench.py \
  --variant warp8 --T 256 --H 2 --state bf16 --warmup 5 --iters 20 --repeats 2 \
  --json /tmp/c1_warp8_small.json
PYTHONPATH=/home/lcpu/85117379/flashkda-warp8-1ce47ea-r2:$PYTHONPATH \
python assignment02/team/c1_flashkda/challenge_warp8/run_candidate_bench.py \
  --variant warp8 --H 96 --state bf16 --json /tmp/c1_warp8_h96.json

# vshard clone：用完全相同的 runner、输入合同和参数生成可直接比较的 JSON。
PYTHONPATH=/home/lcpu/85117379/flashkda-vshard-1ce47ea-r4:$PYTHONPATH \
python assignment02/team/c1_flashkda/challenge_warp8/run_candidate_bench.py \
  --variant vshard --H 96 --state bf16 --json /tmp/c1_vshard_h96_same_contract.json
```

若冻结的 current-best median 为 `m`，严格 10% 门槛为 `m/1.1`。以
`--current-best-ms m --enforce-target` 运行时，runner 将记录门槛，且候选未超过该
门槛会以非零状态退出；正常探索阶段不应加该开关，以便保留所有负结果。

预期收益来自把同一个 CTA 内的八个独立 V 块同时驻留，从而减少每个 warp 的寄存器
工作集并提高同 CTA 的指令级并行度；风险是 320-thread CTA 的寄存器/occupancy
反而恶化。因此它只是待验证候选，不能在正式报告中写作已优化，除非清卡复现实测
相对冻结 current-best 至少快 10%。
