# C1 B300：Phase-1 双槽 fragment 预取候选

这是一个隔离的、非生产候选。它从干净 FlashKDA `1ce47ea` one-shot 生成 baseline、
vshard2-P2S3、当前 vshard4-P2S3，以及独立符号
`fwd_vshard4_p2_phase1pf`。不会修改 production wrapper、dispatcher、报告或已有
challenge。

| 变量 | 含义 | 本候选取值 |
| --- | --- | --- |
| `B,T,H` | batch、序列长度、head 数 | 正式门：`1,8192,12` |
| `K,V,V_s` | key/value 宽度、每 CTA value shard 宽度 | `128,128,32` |
| `k` | Phase-1 的 K-dimension `16×16` block 下标 | `0…K/16-1 = 0…7` |
| `a=k mod 2` | 当前活动 fragment slot | `0` 或 `1` |
| `n=(k+1) mod 2` | 下一个非活动 fragment slot | `0` 或 `1` |
| `P` | Phase-6 既有软件预取环深度 | `2`（与 current 相同） |
| `r_q` | current P2S3 latency / candidate latency | `q∈{P50,P95,P99}`，每格门 `r_q≥1.02` |

## 变换与等价条件

当前 P2S3 对每个 `k` 执行 `K@s`、`Q@s`，随后把 `k+1` 的三个 LDSM
fragment 装入同一个 staging slot。候选给 `K`、`Q`、`s_acc` 各分配两个独立寄存器
fragment slot：在活动 slot 完成 `K@s` 后，先向非活动 slot 载入 `k+1`，再执行活动
slot 的 `Q@s`。

这不改变任一输出 accumulator 的算术顺序：`u_acc` 仍严格按 `k=0…7` 做一次
`K@s`，`out_acc` 仍严格按 `k=0…7` 做一次 `Q@s`。三个预取 copy 只读取已经由
`load_pipeline.consumer_wait` 确认完成且本 phase 从不写入的 shared-memory input/state；
它们只写入下一 slot 的私有寄存器。因此当前 slot 的 `Q@s` 不读取、也不等待下一
slot，不添加 block/warp 同步。逐位 output/final-state exact gate 是这一论证的运行时
验证；若编译或 exact 失败，立即停止，不作 dispatch 改动。

## 预注册门

1. fresh SM103a build 的 formal fixed BF16 initial+final-state 实例必须 zero spill，且
   CUBIN 有 candidate shared-memory record；
   runner 还会在分配内硬验恰好一张可见的 B300、capability 10.3 与 148 SM；
2. `H12,T8192,D128` 的四种 raw state contract（none、BF16-both、FP32-both、
   FP32-final-only）都相对 baseline 逐位 exact；
3. 每 contract 作至少两次 cyclic 四路径 repeat；每路径每 repeat 保留恰好 1000 个
   public-wrapper CUDA-event samples（包括 workspace allocation）；
4. 所有 repeat × contract × `{P50,P95,P99}` 都必须 `r_q≥1.02` 才能进入独立确认。

独立 analyzer 从 1000 条 raw sample 重算 summary 与 speedup，不信任 runner 的汇总字段。
任何一格未达标时 clean shell 都以非零退出并停止该方向，禁止事后挑分位数、改门或登记
dispatcher。

## 目标机执行

先在 clean B300 login/build allocation 设置标准环境变量，再运行：

```bash
C1_PHASE1PF_BUILD_AUTHORIZED=1 \
  bash team/c1_flashkda/challenge_phase1_fragment_prefetch/build_fresh_b300_sm103a.sh --authorized-by-parent
```

fresh build 成功后，在新的干净 Slurm GPU allocation 中运行：

```bash
C1_PHASE1PF_GPU_AUTHORIZED=1 \
  bash team/c1_flashkda/challenge_phase1_fragment_prefetch/run_clean_phase1pf_audit.sh --authorized-by-parent
```

脚本会保留 build/CUBIN/PTXAS、extension hash、raw samples、GPU pre/post audit 及
one-allocation analyzer JSON。one allocation 通过也只意味着可做独立确认，不意味着
生产发布。

## 2026-08-30 实测结论

三次 build allocation 都被保留：前两次分别暴露了顶层寄存器 fragment 数组和 CUTE
copy/后续标量引用的编译边界；第三次 job 12401 以显式双槽标量 fragment 完成 fresh
SM103a build。正式 BF16 fixed-both 实例为 **59 registers、9 barriers、0 spill**，14 个
candidate 实例的寄存器范围为 57–59，且 shared-memory record 门通过。

同一 job 的 B300 身份为 SM10.3/148 SM，四种 contract 的 baseline/current/candidate
output 与适用 final state 均逐位 exact。性能门则明确失败：跨所有
repeat × contract 的最小 `current / candidate` 为 P50 **0.981869x**、P95
**0.982007x**、P99 **0.983188x**，即速度比低于 1 达 1.681–1.813 个百分点，而不是快 2%。因此
`eligible_for_independent_confirmation=false`，没有申请第二 allocation，也没有修改生产
dispatcher。完整证据为 [one-allocation gate](results/c1_phase1pf_phase1pf_a1r3_one_allocation_gate.json)、
[raw samples](results/c1_phase1pf_phase1pf_a1r3_h12_all_contracts_two_repeats.json)、
[PTXAS audit](results/c1_phase1pf_ptxas_b300_sm103a_a1r3.json) 与
[clean log](results/c1_phase1pf_phase1pf_a1r3_job12401.log)。
