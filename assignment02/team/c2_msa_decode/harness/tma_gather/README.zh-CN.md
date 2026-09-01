# C2 讨论点 3：两级 page gather 与 TMA 微基准

## 变量表

| 符号 / 变量 | 含义 | 默认值 |
| --- | --- | ---: |
| `B` / `batch` | 同时处理的 request 数 | 4 |
| `K` / `topk` | 每个 request 选中的逻辑 page 数 | 16 |
| `L` / `logical_pages` | 每个 request 的逻辑页表长度 | 64 |
| `R` / `physical_pages` | KV cache 中的物理 page 总数 | 128 |
| `E` / `page_elems` | 单个 page 的 `uint32` 元素数 | 1024 |
| `s=bK+r` / `slot` | request `b` 的第 `r` 个 selected page 输出槽位 | `0..BK-1` |
| `\ell_s` | `topk[b,r]` 给出的运行时逻辑 page | 随机、无重复 |
| `p_s` | `block_table[b,\ell_s]` 给出的运行时物理 page | 随机置换 |
| `X[p,e]` | 物理 KV page payload（本实验使用唯一 `uint32` 模式） | — |
| `Y[s,e]` | 最终连续输出，正确结果为 `X[p_s,e]` | `[BK,E]` |

本实验故意将页内容改为可 bit-exact 比较的 `uint32`，隔离 address generation 与
搬运语义；它不测 attention、softmax 或 FP8 scale。

## 文档考证边界

CUDA 13.0 的 [TMA programming-guide 小节](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma)
规定 multi-dimensional bulk-tensor copy 使用 tensor map；该 map 在 host 端由
`cuTensorMapEncode*` 创建，并以 `const __grid_constant__` kernel 参数传入。官方示例
列出的描述信息是 global base、size、row stride 和 shared-memory box；本程序以
`CUtensorMap` 的同一 ABI 构造 `[E,BK]` 的连续页 map，并用 `{0,s}` 坐标复制。

“不能用单个 map 做这里的两级 gather”是由这组公开 descriptor 字段与下方可运行代码
共同得到的推论，而不是声称 NVIDIA 文档逐字列出了 `topk`：`topk`、`block_table`
均是运行时 global-memory contents，需先被 load 才能形成 `p_s`；它们既不是 map 的
base/shape/stride/box，也不是本次单条 TMA 发射的固定仿射坐标。若 future API 引入
显式 device-side indirect-gather descriptor，本结论必须按新 API 重新验证。

## 被比较的三条路径

每条路径都有两个相同形状的读写腿，端点均为相同的 `Y[BK,E]`，每次迭代的
估计读+写量均为 `4 × sizeof(Y)`：

1. `software_staged`：普通 CUDA 线程先计算
   `logical=topk[b,r]`、`physical=block_table[b,logical]` 并 gather 到连续
   `buffer[s,e]`，再用普通线性 copy 写 `Y`。
2. `gather_then_tma`：第一腿与 1 完全相同；第二腿为真正的
   `cp.async.bulk.tensor.2d.shared::cluster.global.tile`，从连续 `buffer`
   经 shared memory 写 `Y`。
3. `contiguous_then_tma`：以预先按 `s` 排列好的同一 payload 作为第一腿的
   连续输入；第一腿普通 copy 到 `buffer`，第二腿与 2 的 TMA 完全相同。

因此 1 vs 2 只考察第二腿的软件 copy/TMA 替换；2 vs 3 保留相同形状和总字节量，
只考察第一腿是否必须做 data-dependent 两级寻址。程序先将每个输出与独立 host
参考逐元素比对，任一错配即退出，**不会计时**。

## 为什么一个 TMA map 不能替代第一腿

TMA tensor map 的 base、维度、stride、box 和启动坐标共同定义固定仿射访问；本程序
对连续页使用 `coordinates={0, slot}`，其地址是
`base + slot * E * sizeof(uint32_t)`。而 paged KV 的地址是

\[
  addr(s,e)=base+E\cdot block\_table[\lfloor s/K\rfloor,\
  topk[\lfloor s/K\rfloor,s\bmod K]]+e.
\]

它依赖两次内存 load 的值。`CUtensorMap` 的参数里没有 `topk` 或
`block_table` 指针，也不能把这两个 load 的结果用作同一条 TMA 的坐标。因此结论是：
**单个 TMA descriptor 不能表达动态两级 page gather；需要先由线程/warp 读 index
并做 software gather，之后连续 buffer 才可由 TMA 描述。**

## B300 严格运行命令

在主调度已取得独占 B300 allocation 后，运行：

```bash
cd /home/lcpu/85117379/codex-a02-20260819-main/assignment02/team/c2_msa_decode
bash harness/tma_gather/run_tma_two_level_gather_audit.sh "$PWD" b300
```

该脚本在 PRE/POST 打印 GPU UUID、显存、compute-apps；发现其他 compute app 立即
以 `90` 退出，POST 仍有 app 则以 `91` 退出。它记录 source SHA、构建命令、TMA
源码证据、正确性门、CUDA event 计时和 JSON。结果应写入
`experiment_logs/c2_tma_two_level_gather_b300_job<id>.{log,json}`，而不是把任何
静态结论冒充为硬件性能结论。

## B300 已完成结果与解释纪律

| B300 实测项 | `software_staged` | `gather_then_tma` | `contiguous_then_tma` |
| --- | ---: | ---: | ---: |
| bit-exact mismatches | 0 | 0 | 0 |
| CUDA event 平均 ms | 0.007272 | 0.008216 | 0.007186 |
| 估计有效 GB/s（两腿读+写） | 144.200 | 127.621 | 145.921 |

这些是 B300 job 4340 的 clean run：PRE/POST 均为 0 MiB、compute-apps 为空，
`FINAL_RC=0`；可复核的 [JSON](../../experiment_logs/c2_tma_two_level_gather_b300_job4340.json)
保存原始数值，[完整日志](../../experiment_logs/c2_tma_two_level_gather_b300_job4340.log)
保存 source SHA、`nvcc` 命令、三路 gate 及实际 `UTMALDG.2D` SASS。所有路径先与
独立 host `uint32` reference bit-exact 比较；只有三路 0 mismatch 时才启动 CUDA
event 计时。

`gather_then_tma` 并未在本受控尺寸上胜过 software staged：0.008216 ms 高于
0.007272 ms。它额外需要把随机页 materialize 到连续 buffer，不能像单个仿射
TMA 一样直接跳过 index 链。连续输入对照为 0.007186 ms，表明真正的
`CUtensorMap`/TMA 路径已经执行，但不改变两级 data-dependent address generation
的语义限制。即使某个 page size 在未来变快，也只证明这个受控搬运实现，不可外推为
完整 MSA decode 一定更快。
