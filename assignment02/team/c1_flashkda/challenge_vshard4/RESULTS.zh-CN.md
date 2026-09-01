# V=32、4 CTA/head 候选的阶段记录

| 阶段 | 结果 | 证据/说明 |
| --- | --- | --- |
| 机械静态检查 | 通过 | `VDim=32`、kernel template `D/4`、grid `H*4`、CTA threads=128 |
| r1 构建 | 失败并保留 | 最初漏改 kernel template 的 `D/2`，nvcc static assertion 报 `VDim=64`；未运行 GPU kernel |
| r2 构建 | 通过但不作为最终产物 | B300 环境完整编译、ptxas、链接并生成独立 `flash_kda_C`；随后审计发现 launch smem 类型漂移 |
| r4 构建 | 通过，最终候选 | launch 改为 `SharedStorageK2VShard4`，four-CTA header/pybind 文案也经 static audit；日志为 `/home/lcpu/85117379/c1_vshard4_build_4430_r4.log` |
| 5090 r4 / SM120a 构建 | 通过 | fresh clone 以 `compute_120a,sm_120a` 编译、ptxas 和链接成功；SO SHA256=`a85d14f61b64807c06c76774a33e0e87781feda21e6f6a2198fac67b89971163` |
| r2 small exact | 通过但受污染，仅历史证据 | `T=256,H=2,bf16`，baseline vs vshard4 的 output/final-state 均逐元素相等 |
| 5090 r4 small exact | 通过、clean | job 6999：`T=256,H=1/2/4` 的 none/BF16/FP32；所有 output exact，BF16/FP32 final-state exact；PRE/POST 0 MiB、apps 空 |
| 5090 r4 H64/H96 | clean 完成，负优化 | 同形状 BF16 exact；AB/BA 各 1000 samples：H64 `2.227344 -> 3.276448 ms`（0.679804x），H96 `2.628704 -> 4.906656 ms`（0.535742x）。仅为 SM120a 结果，不能混入 B300 current-best |
| B300 r4 small exact | 通过、clean | job 4446：`T=256,H=1/2/4` 的 none/BF16/FP32 以及 H64/H96 BF16 full call 均为 output/final-state exact；PRE/POST 0 MiB、apps 空，`FINAL_RC=0` |
| B300 r4 H64/H96 | clean 完成，不能替换 current | AB/BA 各 1000 samples：H64 `0.943168 -> 0.814592 ms`，相对官方为 1.157841x、相对冻结 vshard2 仅 0.981615x；H96 `1.033472 -> 1.149568 ms`，相对官方为 0.899009x、相对冻结 current 为 0.895891x |

历史 r2 small gate JSON 是 `results/c1_vshard4_small_gate_4420.json`。它明确标记
`"timing": "not run (--no-bench)"`，因此不能误读为性能结果。外来进程清退后，最终 r4
已在 job 4446 完成 B300 small exact matrix 与 H64/H96 同合同 AB/BA；原始证据为
`results/c1_vshard4_b300_job4446_{small_matrix,h64,h96}.json` 及
`results/c1_vshard4_b300_job4446_job4446.log`。H64 的严格目标是冻结 current-best
`0.799616 / 1.1 = 0.726924 ms`，B300 与 5090 的 vshard4 均未达到；5090 job 6999 的
clean 证据单独保存在 `results/c1_vshard4_5090_sm120a_r2_*`，不得与 B300 数值混算。
