# C1：V=32 的 4 CTA/head K2 实验候选

该候选将原始 `V=128` 的 K2 value/output/state 列划分为四个不重叠的
`V_s=32` 切片。每个 `(sequence, head, shard)` 一个 CTA，CTA 内两个 MMA warp
分别处理 `V` 的一个 16 列块，另有一个 TMA-load warp 与一个 TMA-store warp，
故 `NumThreads=2*32+32+32=128`。

| 变量 | 含义 | 取值 |
| --- | --- | --- |
| `K` | state/key 行维 | 128 |
| `V` | 完整 value/state 列维 | 128 |
| `s` | value shard | `0,1,2,3` |
| `V_s` | 单 CTA 的 value 列数 | 32 |
| `w` | CTA 内 MMA warp | `0,1` |

K2 的递推对 value 列不做归约，四个 CTA 只共享只读 K1 workspace，并各自读写
state/output 的不重叠 32 列切片。该生成器会锁定并校验已有 vshard2 生成器的 SHA256，
再机械地替换 `VDim=32`、grid=`H*4`、compute threads=64 与 launch threads=128；
这样不会悄悄吸收共享代码的未来变动。它只允许在干净的 FlashKDA 1ce47ea worktree 上运行。

## 设计细节与同步/索引安全性

上游 K2 每个 16-token chunk 的 value 相关计算可写为
`O_c = Q'_c S_(c-1) + M_c U_c`、
`S_c = diag(exp(g_c)) S_(c-1) + K'_c^T U_c`。其中 value 列只出现在右侧矩阵的
列维，两个矩阵乘都不沿 value 列归约。因此按
`V=[V^(0)|V^(1)|V^(2)|V^(3)]` 切分时，`S_c`、`U_c`、`O_c` 同样按该列边界独立；
四个 CTA 不需要跨 CTA 通信。

原始 state 的逻辑视图是 `[K,V]`，但 FlashKDA raw ABI 的物理布局为 `[V,K]`。
shard `s` 的 TMA 起点必须是 `value_shard * V_s` 的 **物理 V 行**偏移，形成
`[V_s,K]` 的连续 state slice；initial/final state 以及 out 均按相同切片读写。K1
workspace 的 `k/q/g/INV/Mqk` 保持完整 `K=128`，只读且可安全被四个 CTA 重复加载。
这也是补丁中单独保留 `TMAMMLayout`（全 K workspace）而让 `TMAVOLayout`、state/out
使用 `V_s=32` layout 的原因。

CTA 内 `w=0,1` 两个 MMA warp 各独占一个 `[16,V]` 列块；每个 warp 在 phase 1--6
仅写自己的 output/state 子块。两个计算 warp 都到达原 pipeline 的
`NamedBarrier(kComputeThreads=64)`，TMA load/store warp 不参与该 compute barrier，
与 upstream warp-specialization 角色契约一致。TMA pipeline 的 consumer 计数也改为
64；总 launch block 为 `64 + 32 + 32 = 128` threads。该候选只实例化 `K=128,V_s=32`，
不会对未证明的 shape 静默运行。

`apply_vshard4_patch.py --check-only` 是**对尚未打补丁的干净
`1ce47ea` worktree** 的前置检查；它会刻意拒绝已打补丁的 r4 脏树，避免重复
应用。这不是 r4 构建异常。r4 应通过已生成 `.so` 的 SHA256、四个 patched source
的 SHA256 和 `flash_kda_C.fwd_vshard4` ABI 进行身份核验。

首次 r1 构建因 launch kernel template 仍传 `D/2` 而在静态断言中报告 `VDim=64`；该
失败没有运行 GPU kernel，且已保留在阶段记录中。修正为 `D/4` 后的 r2 已可构建；终审
又发现 launch `sizeof` 应与 kernel 使用相同的 `SharedStorageK2VShard4` 类型，且 generated
header/pybind 文案必须是 four CTA。最终 fresh r4 完成 CUDA 编译、ptxas、链接与 `.so` 生成；
完整构建日志位于 `/home/lcpu/85117379/c1_vshard4_build_4430_r4.log`。这只能证明
C++/CUDA 实例化可编译，不等价于性能有效。

## 正确性门与污染边界

在审计前的 r2 上只运行过一个不含 event timing 的 small gate：`B=1,T=256,H=2,K=V=128`、
BF16 state。baseline vs `fwd_vshard4` 的 output 和 final-state 都逐元素相等，JSON 为
`results/c1_vshard4_small_gate_4420.json`。执行前后均检测到外来训练进程 PID 1462514，
占用 2480 MiB；因此该记录明确标注 `timing: not run (--no-bench)`，仅可用作正确性证据，
绝不能导出或引用性能结论。最终 r4 已在 clean B300 job 4446 复跑 small exact matrix，
且只有该 r4 的结果进入本页最终性能前置证据。专用 `run_vshard4_final.py` 不会导入旧
two-way wrapper：它在每个 full-size 性能测量前先对同一输入做 exact gate，并以交替
AB/BA 顺序用 CUDA event 包围一次完整公开 `fwd` 调用（含两边各自的 workspace 分配）。
`run_clean_vshard4_audit.sh` 会在 small matrix 与 H64/H96 全调用测量前后检查
compute-apps 和显存使用量；查询失败、任何前/后非空或显存非 0 MiB 都会使整轮作废。

## RTX 5090 / SM120a clean 终审：正确但显著负优化

独立 r4 clone 用 `FLASH_KDA_CUDA_ARCHS=120a` 构建，SO SHA256 为
`a85d14f61b64807c06c76774a33e0e87781feda21e6f6a2198fac67b89971163`。
job 6999 在 `gj-5090-1` 上运行，PRE/POST 都为 0 MiB 且 compute-apps 为空，
`FINAL_RC=0`。small matrix 对 `T=256,H=1/2/4` 的 none/BF16/FP32 全部 exact；
H64/H96 的 BF16 同形状 output/final-state 也 exact。每侧保留 1000 个单调用
CUDA-event 样本，调用次序逐样本 AB/BA 交替。

| shape | baseline median | vshard4 median | 本机 speedup | 结论 |
| --- | ---: | ---: | ---: | --- |
| `T=8192,H=64` | 2.227344 ms | 3.276448 ms | 0.679804x | 负优化 |
| `T=8192,H=96` | 2.628704 ms | 4.906656 ms | 0.535742x | 负优化 |

这两行是 RTX 5090/SM120a 的独立结果，不能和 B300 毫秒数混算，更不能当作 B300
`current-best` 的提升。它说明四个小 CTA 带来的额外 launch/TMA/workspace 重复开销已超过
并行列分片的收益；下一类候选应优先调单 CTA 基线的 stage/register/smem/warp 配置，而不是
继续增加 CTA 分片数。证据为 `results/c1_vshard4_5090_sm120a_r2_small_matrix.json`、
`results/c1_vshard4_5090_sm120a_r2_h64.json`、
`results/c1_vshard4_5090_sm120a_r2_h96.json` 与
`results/c1_vshard4_5090_sm120a_r2_job6999.log`。

## B300 最终 clean 性能结果：相对官方正，但不超过 current vshard2

job 4446 在同一 B300 UUID 上运行，PRE/POST 均为 compute-apps 空、显存 0 MiB、
`FINAL_RC=0`；同一 source/SO hash 被写入审计。small matrix 覆盖
`T=256,H=1/2/4` 的 none/BF16/FP32，均 output/state exact；随后 H64/H96 的 BF16
full call 也都 output/final-state exact。每个路径均有 1000 个单调用 CUDA-event 样本，
并逐样本交替 AB/BA：

| shape | 官方 baseline median | vshard4 median | exact gate | relative to frozen current-best | 结论 |
| --- | ---: | ---: | --- | --- | --- |
| `T=8192,H=64` | 0.943168 ms | 0.814592 ms | PASS | 0.981615x（current=0.799616 ms） | 相对官方 1.157841x，低于 current |
| `T=8192,H=96` | 1.033472 ms | 1.149568 ms | PASS | 0.895891x（current=1.029888 ms） | 相对官方 0.899009x，负优化 |

原始 [small matrix JSON](results/c1_vshard4_b300_job4446_small_matrix.json)、
[H64 JSON](results/c1_vshard4_b300_job4446_h64.json)、
[H96 JSON](results/c1_vshard4_b300_job4446_h96.json) 与
[完整审计日志](results/c1_vshard4_b300_job4446_job4446.log) 可复核。冻结 current-best
再快 10% 的严格门槛是 H64≤0.726924 ms、H96≤0.936262 ms，vshard4 两项均未达到，
故不能替换 current vshard2。RTX 5090 job 6999 的 H64/H96 也分别只有 0.679804x/
0.535742x，进一步说明四分片不是跨架构的正优化。

## 未覆盖的输入合同

当前 runner 的 exact/性能合同只覆盖固定长度且 `T` 为 16 的整数倍的调用；small gate
将只检查 `T=256`、`H=1/2/4`、none/BF16/FP32 state。它**没有**覆盖尾 tile（`T % 16 != 0`）
或 `cu_seqlens` varlen 路径，因而不能对这两类输入作正确性或性能声明。补丁保留了 upstream
的相关分支模板，但“代码存在”不等于已验证；若后续需要支持，必须另行写 gate 并记录结果。
