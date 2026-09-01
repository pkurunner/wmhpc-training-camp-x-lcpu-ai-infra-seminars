# 长上下文 recurrence 质量审计

本目录把真实 BF16 state persistence 与独立 FP32 recurrence oracle 分开比较。正式结论只应
读取 `results/c1_long_context_b300_sm103a_full_r1.json`；它包含 12 组 main run，且
`complete=true`。

`results/c1_long_context_b300_sm103a_preflight_r1.json` 是正式运行前保留的历史 preflight
产物：它只做短正确性校准，`main_runs=[]`。生成该文件的旧 runner 曾错误写入
`complete=true`，因此这个字段不能解释为主实验完成，也不应被结构化消费者用于正式
结论。历史 JSON 和对应日志保持原样以保留审计链；当前 runner 已将完成条件收紧为恰有
12 组 main run。
