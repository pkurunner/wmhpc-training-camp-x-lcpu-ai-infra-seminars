# v13 distributed merge 静态并发审计

| 变量 | 含义 | 冻结值或范围 |
|---|---|---|
| `r` | cluster 内 CTA role / rank | `r∈{0,1,2,3}` |
| `t` | CTA 内线程编号 | `t∈[0,255]` |
| `E` | 每个 `(batch, kv_head)` 的输出元素数 | `16×128=2048` |
| `E_r` | role `r` 独占的输出元素半开区间 | `[512r,512(r+1))` |
| `H_r` | role `r` 独占的 query-head 半开区间 | `[4r,4(r+1))` |
| `A` | rank-0 producer-ready mbarrier 的期望 arrival 数 | `4` |
| `P_r` | role `r` 负责的 selected-page 半开区间 | `[4r,4(r+1))` |

## 身份与审计范围

- v12 冻结基线 SHA-256：`535d90b856ed1062aa7b8a105eb2c5f236c450826e65496e646d0d5a27eb8aaf`。
- v13 候选 SHA-256：`def42508c28eef50995c33b70ebddba31a82e932764840c6e4ff1714b9c7f063`。
- 唯一 v12→v13 patch SHA-256：`6efdebfdfde202cac6af66fd008afb32028b31291e276cf32e5566f57ed36016`。
- 本审计只证明源码控制流、所有权与必要的 happens-before 结构；CUDA 编译接受性、资源数值、DSM 运行时可见性和数值正确性仍必须由 B300 AOT 与 directed gate 裁决。

## 静态结论

1. host 端固定 `clusterDim.x=4`，grid 也是 `batch×kv_heads×4`；kernel 内 `role=cluster.block_rank()`，故每个 cluster 恰有 roles `0..3`。设备体在末尾 `cluster.sync()` 前没有提前 `return`。
2. rank 0/thread 0 用 `A=4` 初始化唯一 producer-ready barrier，CTA `__syncthreads()` 后执行首个 `cluster.sync()`；因此四个 producer 开始工作前 barrier 已初始化。
3. `kProducerCtas=kClusterCtas=4`，所以每个 role 的 thread 0 都恰好执行一次 cluster-scope release arrival。随后每个 CTA 的 thread 0 对同一 rank-0 barrier 执行 acquire wait，再用本 CTA 的 `__syncthreads()` 发布 `producer_ready`。任何 producer 的 all-invalid 分支也不会绕过 arrival。
4. arrival 之前存在 CTA `__syncthreads()`，覆盖该 CTA 对 `local_partial` 与 `local_lse` 的全部写入；release-arrive/acquire-wait 是 v12 已接受的 DSM 可见性协议。v13 只把等待者从 rank 0 扩展为四个 CTA。这个 remote-barrier pointer 用法仍须由 AOT 和 watchdog directed 运行验证。
5. 正常与超时路径都只写 `E_r`。对固定 `r`，线程 `t` 写 `512r+t` 与 `512r+t+256`，因此 CTA 内无重叠且覆盖 512 个元素；四个 `E_r` 两两不交并，其并集为 `[0,2048)`。
6. `E_r` 对应 `H_r`。每个 CTA 的 threads `0..3` 分别计算一个本地 head 的四个 producer 权重和 denominator，随后 CTA `__syncthreads()`；其余线程读取 `local_head∈[0,3]`，不存在本 CTA merge metadata 的未初始化读取。
7. 四个 `local_lse` / `local_partial` 指针均显式映射到 ranks `0..3`；`partial_offset=head×128+dim` 使用全局 head，而 `merge_weights` 使用 `local_head=head-4r`，索引域分别为 `[0,2047]` 与 `[0,3]`。
8. 末尾第二个 `cluster.sync()` 保证任何 CTA 都不会在其 shared storage 仍被其他 CTA 读取时退出。输出区间互斥，因此正常路径、NaN 超时路径都没有跨 CTA output data race。
9. Q stride `136`、八个 Q WMMA fragments、QK/PV/softmax、producer page partition、host ABI 与 dispatcher 注册未改变；patch 只改变 merge metadata 尺寸、四 CTA barrier wait 和最终输出所有权。

## 必须保留的动态停止门

- AOT：唯一 `sm_100` cubin、无 PTX、`STACK=LOCAL=0`，记录实际 REG/SHARED；编译器拒绝 DSM-mapped barrier pointer 时立即停止。
- Directed：四个 producer 输入区间、全 invalid、online-rescale、mixed tail、head/dim 唯一编码、bitwise repeat、caller pointer、dispatcher/kernel profiler event 与 watchdog 全部通过。
- ABBA：仅在 directed 全闭环后，使用同卡四隔离进程 `v12_A→v13_B1→v13_B2→v12_A2`、8 seeds；点改善严格 `>3%` 且 deterministic bootstrap 95% LCB `>0` 才接受。
- 任一 liveness、identity、oracle、资源或比较门失败都不得形成性能结论，也不得进入 wheel/NCU。
