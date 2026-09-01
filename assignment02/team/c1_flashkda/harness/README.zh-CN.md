# FlashKDA C1 可复现 Harness

`validate_and_bench.py` 是 challenge 与 upstream FlashKDA 的共同验证/计时
入口。它不实现 kernel，而是严格调用已安装的扩展，并在任何计时前检查：

1. GPU 可见且为 CUDA；
2. upstream `flash_kda.fwd` 与 `flash_kda.fwd_vshard` 均可导入；
3. 三种 state 模式（无 state、bf16 state、fp32 state）逐元素对拍；
4. `FlashKDA/tests/torch_ref.py` 对同一随机输入也逐元素对拍；
5. `fla_kda_ref/naive.py` 通过明确的输入/状态布局转换进行独立数值对拍。

变量、数学和价值分片限制见
[`challenge_vshard/README.zh-CN.md`](../challenge_vshard/README.zh-CN.md)。
详细运行命令也在该文档。harness 默认只使用小形状；必须显式传入官方形状
`--T 8192 --H 96` 才会进行正式性能比较。

FLA 朴素参考是顺序递推，不能和大形状吞吐基准混跑。先在小形状保留完整
reference gate；随后大形状计时可明确传入 `--skip-torch-ref --skip-fla-ref`，
但实验记录必须引用前一条成功的 reference gate。

## K1/K2 完整 NCU / roofline 证据

[`run_roofline_ncu_audit.sh`](run_roofline_ncu_audit.sh) 只能在主会话已拿到的
干净 B300 allocation 中执行，不会提交或占用 Slurm 资源。它以正式
`fixed, H=96, D=128` 跑 NCU `--set full`，只 profile
`_flash_kda_fwd_prepare`（K1）与 `_flash_kda_fwd_recurrence`（K2），并留下：

- pre/post GPU 与 compute-app 审计；
- 原始 `.ncu-rep` 与 `--page raw` CSV；
- [`summarize_ncu_roofline.py`](summarize_ncu_roofline.py) 生成的中文摘录。

| 变量 | 含义 | 本次口径 |
| --- | --- | --- |
| `K1`,`K2` | prepare 与有状态 recurrence kernel | `_flash_kda_fwd_prepare`、`_flash_kda_fwd_recurrence` |
| `F`,`B` | 同一 kernel 的有效 FLOP、实际 DRAM bytes | 必须同一 profile；不能从输入 shape 猜测 |
| `I` | operational intensity | `I=F/B` |
| `P_roof` | roofline 上界 | `min(P_peak,I×BW_peak)` |

解析器按 K1/K2 分别列 Tensor/SM、DRAM/L2、occupancy/waves 与 warp stalls；任一类
缺失均写 `MISSING`，不能据此给出 compute-/memory-bound 标签。完整命令与输出路径见
[`../microbench/README.zh-CN.md`](../microbench/README.zh-CN.md)。

## 已完成的 clean B300 证据（job 4339）

正式 `fixed,H=96,D=128` 的 `--set full` 已运行成功：
[运行日志](../experiment_logs/c1_ncu_full_b300_job4339.log) 的 PRE/POST 都显示 0 MiB、
compute-apps 为空、返回码为 0；原始 [CSV](../experiment_logs/c1_ncu_full_b300_job4339.csv)、
[`.ncu-rep`](../experiment_logs/c1_ncu_full_b300_job4339.ncu-rep) 和
[中文逐指标摘录](../experiment_logs/c1_ncu_roofline_b300_job4339.md) 均已归档。

| kernel | compute / memory 实际 metric（elapsed） | parallelism 实际 metric | 保守分类 |
| --- | --- | --- | --- |
| K1 | HMMA 5.633783%；DRAM read/write 28.779275% / 29.829528%；L2 sectors 30.967491% | grid `(512,96,1)`；41.51 waves/SM；active warps 96.625428% | 非并行度受限；SM/DRAM 都未饱和，不能严谨称 compute- 或 memory-bound。 |
| K2 | HMMA 19.390659%；DRAM read/write 15.394061% / 3.243208%；L2 sectors 12.481916% | grid `(1,96,1)`；0.32 waves/SM；active warps 9.365785% | 并行度/小 grid 受限；不是凭低 DRAM 或低 HMMA 把它误称 memory-/compute-bound。 |

虽然 raw report 含 `dram__bytes_*` 和 SASS instruction evidence，但没有可无歧义换算为
同一 kernel **有效 FLOP** 的完整 `F` 口径。因此本次不计算 `I=F/B`、也不声称严格
roofline 交点；以上仅是由饱和度和 waves/occupancy 直接支持的分阶段分类。
