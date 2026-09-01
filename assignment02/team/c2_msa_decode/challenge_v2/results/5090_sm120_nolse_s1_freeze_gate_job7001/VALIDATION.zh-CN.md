# C2 no-LSE 配置冻结门（RTX 5090，job 7001）

这组证据记录最终全批次复验之前的单点冻结门，防止事后从 job 7002 的结果反向选参。Slurm 记录为 `7001|c2-nolse-s1s3-abba|lcpu-infra|COMPLETED|0:0|00:00:23|gj-5090-1`。

| 变量 | 含义 | 冻结值 |
|---|---|---:|
| `B` | decode batch size | 4 |
| `G` | 每个 KV head 的 GQA 分片数 | 1 |
| `W` | decode kernel 的 warp 数 | 4 |
| `S` | Triton software-pipeline stage 数 | 3 |
| `P` | programmatic dependent launch 开关 | off |
| `R` | `maxnreg` 限制 | none |

固定候选为 `G=1,W=4,S=3,P=off,R=none`。独立 FP32 oracle 通过；101 个 AB/BA pair（每实现 202 个单调用 CUDA-event 样本）得到 control `79.967998 us`、candidate `63.552000 us`，加速 `1.258308x`。AB 与 BA 两种顺序分别均同向。

审计日志记录同一 RTX 5090 UUID 在 PRE/POST 均为 `0 MiB` 且 compute-apps 为空。随后 job 7002 只复验这一冻结配置在 `B={1,4,8,16}` 的表现，不再扫参。

目录中的 JSON、stdout 与 audit log 是服务器原始文件；不得把本结果外推为 B300 结果。
