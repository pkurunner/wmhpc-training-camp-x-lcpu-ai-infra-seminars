# V=16 / 8 CTA-per-head 的 K2 P2S3 候选

这是针对 B300 TP8-local `H=12` 低 K2 grid 的隔离实验。它从已验证的 V=32
vshard4-P2 机械派生，只把每个 value/state/output shard 再二分；baseline、vshard2-P2、
vshard4-P2 和 vshard8-P2 保存在同一个扩展中，以便逐位对拍和循环计时。

## 变量表

| 变量 | 含义 | 取值 |
|---|---|---:|
| $H$ | 本卡 KDA head 数 | 12 |
| $V$ | 完整 value/state 列数 | 128 |
| $S_v$ | 每个 head 的独立 value shard 数 | 8 |
| $V_s$ | 每个 CTA 拥有的 value 列数，$V/S_v$ | 16 |
| $N$ | sequence 数 | fixed B=1 时为 1 |
| $G_{K2}$ | K2 CTA 总数，$NHS_v$ | 96 |
| $P$ | phase-6 software prefetch depth | 2 |
| $S$ | K2 TMA input stage 数 | 3 |

## 为什么先做这个方向

现有 vshard4-P2 在 H12 的 K2 只有 48 CTA、约 0.08 waves/SM；同一次 NCU 显示
SM 与 DRAM 都远未饱和。K2 已经由每个 CTA 连续处理一个 head/shard 的全部 time tile，
所以另做 “persistent” 不会减少 launch；multi-head CTA 还会进一步降低 grid。V=16
则把 grid 提升到 96，同时不触碰有严格前后依赖的时间维。V=16 也是当前 16-column
MMA fragment 下的自然分片下限。

value 列不参与跨列归约。每个 CTA 只写自己的 output/state 列切片，K1 workspace
仍是只读全 K 宽，因此预期与 baseline 逐位相同。这个推理不是发布结论；必须通过
下面的 exact、resource 和性能门。

## 验证与停止门

1. 从 clean FlashKDA `1ce47ea` 生成全新的 SM103a 扩展。
2. 固定 BF16-state vshard8-P2 ptxas 实例必须零 spill。
3. `T=256,H=1/2/4` 的 none、BF16 both、FP32 both、FP32-final-only 必须同时与
   baseline 和 torch reference 逐位相同。
4. `T=8192,H=12` 的四种 raw ABI contract 必须与 baseline 逐位相同；四路径各取
   1000 个轮换 CUDA-event 样本。
5. 只有 P50/P95/P99 全部优于 vshard4-P2 才进入 H1–18 sweep 与 NCU；否则保留负
   结果并停止，不接入 dispatcher。

构建和 GPU 脚本都要求显式授权参数，并在独立 fresh clone / clean Slurm allocation
运行。任何 TMA layout、exactness、spill 或 H12 性能门失败都属于有信息量的停止结果。

## B300 结果：在资源门停止

job 10704 从 clean `1ce47ea` 成功生成并编译 SM103a extension；失败不是环境或源码构建
错误。ptxas 对正式 fixed BF16 initial+final-state V8-P2 实例回报 56 registers、9
barriers、8-byte stack、12-byte spill stores 与 8-byte spill loads。审计脚本因此在加载
extension 和启动 GPU kernel 之前按预设门停止，没有生成 correctness 或性能数字。

这项负结果只淘汰 V8-P2 的当前实现，不表示 V8 几何不正确；后续唯一低成本退路
V8-P1 已在相邻 `challenge_vshard8/` 中验证，并因性能门失败而终止整个八分片方向。

可复核证据：

- `results/c1_vshard8_p2_ptxas_b300_r1_job10704.json`
- `results/c1_vshard8_p2_build_b300_r1_job10704.log`
