# K2 value 维二分（2 CTA / head）挑战

本目录是对 FlashKDA `fwd_kernel2` 的独立挑战实现；不修改提交快照中的
`FlashKDA/`。目标是把 K2 的 value/output 维 `V=128` 沿列切为两个
`V_s=64` 分片，并为每一个 `(sequence, head, value_shard)` 发射一个 CTA。
K2 的状态递推可以按 value 列独立执行，因此两个 CTA 不需要跨 CTA 通信。
K1 仍按原实现运行一次，且 Python 的基线 `flash_kda.fwd` API 不变；挑战
通过新增的 `flash_kda.fwd_vshard` 入口调用。

## 变量表

| 变量 | 含义 | 取值/下标 |
| --- | --- | --- |
| `B` | batch 中的序列数 | `b in [0, B)` |
| `T` | 每个定长序列的 token 数 | 基准为 8192 |
| `H` | 每卡 KDA head 数 | 官方形状 96；TP8 时 12 |
| `K` | key/state 的行维 | 固定 128 |
| `V` | value/state 的列维 | 固定 128 |
| `s` | value 分片编号 | `s in {0, 1}` |
| `V_s` | 每个 CTA 的 value 列数 | `V / 2 = 64` |
| `c` | K2 递推 chunk 编号 | `c in [0, ceil(T/16))` |
| `S_c` | chunk `c` 后的状态 | 逻辑形状 `[K, V]`；原始 ABI 的内存视图为 `[V, K]` |
| `S_c^(s)` | value 分片 `s` 的状态列 | `[K, V_s]`，相互独立 |
| `U_c` | chunk 内修正量 | `[16, V]`；分片为 `[16, V_s]` |

## 数学与并行安全性

K2 对每个 chunk 的核心更新为：

`O_c = Q'_c S_{c-1} + M_c U_c`，
`S_c = diag(exp(g_c)) S_{c-1} + K'_c^T U_c`。

其中 `diag(exp(g_c))` 只作用于状态行，右侧两个矩阵乘都不在 value 列之间
做归约。因此对 `V = [V_0 | V_1]` 有 `S_c = [S_c^(0) | S_c^(1)]`，两个
分片仅共享只读的 K1 workspace，适合用 2 CTA/head。同一 CTA 中仍使用 4 个
MMA warp，每 warp 负责一个 `16`-列 value block；这正好覆盖 `V_s=64`。

这个结论有两个刻意限定：

1. 只在 `K=V=128`、`value_shards=2` 时启用。变更脚本对其他取值直接拒绝，
   不会悄悄走不安全路径。
2. `initial_state` 与 `final_state` 的全局存储布局保持 upstream 的物理 ABI
   `[V,K]`；每个 CTA 只对其 value 维连续的 64 行发出 TMA load/store。两个
   CTA 写出的区间不重叠。这里不能按最后一个 `K` 维切：`torch_ref.py` 的
   `state_slice.t()` 和 K2 的 `s_acc`/`s_acc_T` 两种视图共同证明该轴会改变
   递推归约，属于错误实现。

## 构建与运行

所有命令都在 B300 上执行，且只对单独的 challenge worktree 操作：

```bash
git clone --recurse-submodules https://github.com/MoonshotAI/FlashKDA.git /home/lcpu/85117379/flashkda-vshard-1ce47ea-r4
git -C /home/lcpu/85117379/flashkda-vshard-1ce47ea-r4 checkout 1ce47ea
git -C /home/lcpu/85117379/flashkda-vshard-1ce47ea-r4 submodule status  # 必须显示 cutlass 5c149f5
python assignment02/team/c1_flashkda/challenge_vshard/apply_vshard_patch.py \\
  --source /home/lcpu/85117379/flashkda-vshard-1ce47ea-r4
cd /home/lcpu/85117379/flashkda-vshard-1ce47ea-r4
srun -p cpu --cpus-per-task=8 --time=30:00 bash -lc '\
  cd /home/lcpu/85117379/flashkda-vshard-1ce47ea-r4 && \
  CXX=g++ FLASH_KDA_CUDA_ARCHS=103a NVCC_THREADS=8 \
  /home/lcpu/85117379/codex-a02-20260819-main/assignment02/.venv/bin/python setup.py build_ext --inplace'
export PYTHONPATH=/home/lcpu/85117379/flashkda-vshard-1ce47ea-r4:$PYTHONPATH
```

首先执行小形状正确性；它会比较 upstream `flash_kda.fwd`、挑战入口和
`FlashKDA/tests/torch_ref.py` 的逐元素结果。随后才运行官方大形状基准：

```bash
python assignment02/team/c1_flashkda/harness/validate_and_bench.py \\
  --reference-root assignment02/team/c1_flashkda/FlashKDA --T 256 --H 2 --states all
python assignment02/team/c1_flashkda/harness/validate_and_bench.py \\
  --reference-root assignment02/team/c1_flashkda/FlashKDA --T 8192 --H 96 \\
  --states bf16 --skip-torch-ref --skip-fla-ref --warmup 30 --iters 200 --repeats 5 \\
  --json /tmp/c1_vshard_b300.json
```

`validate_and_bench.py` 必须先报告 `baseline_vs_vshard` 的 exact 对拍成功，
才会报告时延。真实服务器记录、NCU/SASS 与中文分析由主实验流程保存在公共
`experiment_logs/` 和报告中；本目录不伪造性能结果。

## 预期收益与风险

K2 原始网格为 `[N, H]`，TP8 下 `H=12`，只有 12 个长生命周期 CTA。二分后
为 `[N, 2H]`，增加了可并发 CTA 数并把每 CTA 的状态和 value TMA 流量减半；
但 K1 workspace 的 `k/q/g/INV/Mqk` 读取被两个 CTA 重复，TMA launch 数和
同步也翻倍。因此它不是无条件优化：在 `H=96` 可能因原实现已足够占满而退化，
在 TP8 的 `H=12` 更值得测量。最终结论只能以清卡后的时延、NCU 和 SASS 证据
为准。
