# C1 TP8 / H=12 最小可审计实验 harness

这个目录只提供固定长度 `T % 16 == 0` 的验证、计时与审计工具；不包含 kernel
改动、不覆盖 varlen，也不触及尾 token。它服务于 TP8 后每卡 `H=12` 的假设：低 head
数可能重新暴露 K2 grid 的并行度上限。结果必须以本机、本 shape 的 baseline 计算
speedup，不能复用 H64 的 `0.726924 ms` 阈值。

| 名称 | 含义 | 本 harness 的固定值 |
| --- | --- | --- |
| `T` | 固定长度 token 数 | 正式门为 8192；small gate 默认为 256 |
| `H` | 当前卡上的 head 数 | 正式门为 12；small gate 为 1/2/4/12 |
| state | recurrent state 的 API dtype | `none`、BF16、FP32 |
| P1 | 两 CTA/head value-shard wrapper | `fwd_vshard` |
| P2 | P1 加 Phase-6 prefetch=2 的 wrapper | `fwd_vshard_p2` |

## 实验合同

`--variant p2` 会从**同一个 Python 进程、同一个已导入的 `flash_kda_C` SO**调用
baseline `flash_kda.fwd`、P1 和 P2。启动时检查 SO 同时导出 `fwd`、`fwd_vshard` 和
`fwd_vshard_p2`，并把实际导入 SO 的路径与 SHA-256 写入 JSON。

small matrix 为 `H=1/2/4/12 × none/BF16/FP32`。P1 相对 baseline、P2 相对
baseline 和 P1 的 output/final state 必须 bitwise exact。每条路径再对 pinned
`tests/torch_ref.py` 使用仓库既有容差：output `rtol=atol=0.02`；final state
`rtol=atol=0.05`。正式 `T=8192,H=12,BF16` 门仍先做上述三路 exact，但不把耗时很长
的 torch reference 混入性能段；small JSON 已绑定 reference 容差证据。

正式计时每个 CUDA event 恰好包一条完整 public wrapper 调用（含该 wrapper 的
workspace allocation）。三个路径按 `ABC/BCA/CAB` 轮换；vshard4 的两个路径按
`AB/BA` 轮换。JSON 保留每一次 sample 的实际顺序和毫秒数，且对每条路径写出
`P50/P95/P99/mean/min/max`。speedup 始终从**同一 H12 JSON 内 baseline P50**计算。

`--variant vshard4` 是第二个、可比的 runner 配置：它只要求该独立 extension 的
baseline `fwd` 与 `fwd_vshard4`，不假定它同时带 P1/P2。这样不会把不同 extension
混进同一结论，也不会为四分片伪造 P1/P2 对照。

## 直接运行

已在目标机激活含 rebuilt extension 的 Python 环境后，先运行小门：

```bash
export PYTHONPATH=/remote/FlashKDA-p2:/remote/Linux_HPC
python /remote/Linux_HPC/assignment02/team/c1_flashkda/challenge_tp8_h12/run_h12.py \
  --variant p2 --reference-root /remote/FlashKDA-reference \
  --source /remote/FlashKDA-p2/csrc/flash_kda.cpp \
  --source /remote/FlashKDA-p2/csrc/fwd.h \
  --source /remote/FlashKDA-p2/csrc/smxx/fwd_launch.cu \
  --source /remote/FlashKDA-p2/csrc/smxx/fwd_kernel2_vshard.cuh \
  --source /remote/FlashKDA-p2/csrc/smxx/fwd_kernel2_vshard_p2.cuh \
  --small-only --json /remote/results/c1_tp8_h12_p2_small.json
```

接着才运行 H12 正式门（默认 warmup 30、`200 × 5 = 1000` samples/path）：

```bash
python /remote/Linux_HPC/assignment02/team/c1_flashkda/challenge_tp8_h12/run_h12.py \
  --variant p2 --reference-root /remote/FlashKDA-reference \
  --source /remote/FlashKDA-p2/csrc/flash_kda.cpp \
  --source /remote/FlashKDA-p2/csrc/fwd.h \
  --source /remote/FlashKDA-p2/csrc/smxx/fwd_launch.cu \
  --source /remote/FlashKDA-p2/csrc/smxx/fwd_kernel2_vshard.cuh \
  --source /remote/FlashKDA-p2/csrc/smxx/fwd_kernel2_vshard_p2.cuh \
  --official-only --json /remote/results/c1_tp8_h12_p2_h12_bf16.json
```

## 清卡审计示例

`run_clean_h12_audit.sh` 在 PRE、small 后、正式段后和 POST 检查 GPU 显存为 0 MiB
且 compute-apps 为空。它将 runner、审计 shell、指定 generated source 的 SHA-256
写入 log；runner JSON 则记录实际加载 extension 的 SHA-256。为避免误用 GPU，该脚本
需要显式授权变量和参数。

```bash
export C1_H12_AUTHORIZED=1
export C1_H12_WORKSPACE_ROOT=/remote/Linux_HPC
export C1_H12_PATCHED_ROOT=/remote/FlashKDA-p2
export C1_H12_REFERENCE_ROOT=/remote/FlashKDA-reference
export C1_H12_OUTPUT_DIR=/remote/results/tp8_h12
export C1_H12_VARIANT=p2
export C1_H12_LABEL=b300_sm103a_r1
bash /remote/Linux_HPC/assignment02/team/c1_flashkda/challenge_tp8_h12/run_clean_h12_audit.sh --authorized-by-user
```

对 vshard4 只需把 `C1_H12_PATCHED_ROOT` 指向四分片的 rebuilt tree，并设置
`C1_H12_VARIANT=vshard4`。该路径会审计 `fwd_kernel2_vshard4.cuh`，并只执行
baseline/vshard4。

## 4 小时分配与停止线

1. 前 20 分钟：清卡、SO/source/script SHA 与 small matrix。任何 exact/reference/
   audit 失败立即停止，不进入正式段。
2. 接下来 35 分钟：P2 的 H12 正式 BF16 gate + 1000 raw samples/path。若 P2 的
   P50 不优于同次 H12 P1，保留负结果并停止 P2 调参。
3. 之后最多 35 分钟：在**独立** vshard4 extension 上跑同一 H12 harness，判断更多
   CTA 是否反而增加开销。若它不优于 P2，则停止分片数扩展。
4. 余下时间优先做 2--3 次 clean H12 P2 重复，以 raw P95/P99 和跨轮 P50 检验稳定性；
   不扫 CHUNK、varlen/tail 或改变 kernel。只在 P2 仍稳定优于 P1 时，才将最后时间用于
   独立的低风险建模/配置调查。

本目录不定义全局性能门；H12 baseline 每轮单独计入 JSON，读者应据此判断是否值得后续
kernel 方向，而不是把其他 `H` 的历史数字当作门槛。
