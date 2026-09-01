# job 10049：预期保留的 parser 失败证据

这不是有效的 stage-3/stage-5 对照。作业在 stage-3 profile 完成后以 `FINAL_RC=1`
安全中止，因为第一版 validator 错把 NCU 2025.3 的显式 `--metrics` Raw 宽表当成
section CSV 纵表；它没有继续执行 stage 5，也没有产生性能结论。

`basecheck.csv` 是从该作业的 stage-3 `.ncu-rep` 用
`--print-units base --print-fp` 重新导出的只读 parser 测试件。修正后的 validator
已在它上面验证列、单位、单一 action 与九个 finite 值；真正闭环的两路径结果位于
父目录、时间戳 `20260828T121313Z`（Slurm job 10100，`FINAL_RC=0`）。
