# FLA Triton `chunk_kda` 对拍补证

这个目录补齐题面点名的官方 `FlashKDA/tests/test_fwd.py` 中两项 FLA 对拍：

- `test_fwd_vs_fla`（fixed batch）；
- `test_fwd_varlen_vs_fla`（packed varlen）。

它是**正确性/讨论证据**，不是性能测试、不是 release gate，也不会注册 C1 FLA backend 或改动 FlashKDA/FLA 源码。

## 变量与覆盖

| 记号 | 含义 |
| --- | --- |
| `B` | 物理 batch（这两个官方 FLA 测试均为 `1`） |
| `T` | fixed-batch token 长度；官方测试固定为 `8192` |
| `T_total` | packed-varlen 全部逻辑 sequence 的 token 总数 |
| `N` | packed-varlen 的逻辑 sequence 数 |
| `H` | attention heads；官方 FLA 对拍固定为 `1` |
| `D` | key/value 维度；官方 FLA 对拍固定为 `128` |
| `o` | `cu_seqlens`（packed-varlen 的 cumulative offsets） |
| `o_chunk,h_chunk` | FLA Triton `chunk_kda` 返回的 output/final state |
| `o_tri,h_tri` | FLA `fused_recurrent_kda` 的 FP64 gold output/final state |
| `o_fk,h_fk` | candidate `flash_kda.fwd` 写入的 output/final state |

runner 不改变任何张量，也不复制输出内容；只在三条调用边界记录返回/写入的 tensor 元数据。它要求每个官方测试中 `o_fk/h_fk`、`o_tri/h_tri`、`o_chunk/h_chunk` 都真实存在，且三条路径均同时覆盖 fixed 与 packed-varlen。数值判定严格沿用 upstream：candidate output 对 FP64 gold 是 hard `assert_close`（fixed/varlen 阈值 0.005/0.006）；candidate final state 与 Triton chunk final state 使用 `warning=True`，Triton chunk output 只进入 error-ratio 报告和图，不是 hard assertion。

## 防止 backend 截获

shell 在 Python 进程启动前固定设置：

```bash
export FLA_DISABLE_BACKEND_DISPATCH=1
unset C1_B300_FLASH_KDA FLA_FLASH_KDA
```

在 pinned FLA commit `a3edffc...`，该环境变量使 `@dispatch('kda')` 直接返回
`fla.ops.kda.chunk.chunk_kda` 原实现。因此 C1 custom backend、pinned FlashKDA backend 都不能截获这个 `chunk_kda` 调用。Python runner 会 fail-closed 检查 `_DISPATCH_DISABLED`，验证 public `kda.chunk_kda` 与 pinned `fla.ops.kda.chunk` module export 是同一个 callable，并绑定 clean `$FLA_ROOT/fla/ops/kda/chunk.py` 的路径与 SHA。r3 JSON 中旧字段 `chunk_function_class_source` 只是同一 module path 的兼容别名，**不是**对 `ChunkKDAFunction` class 的独立 `inspect`；结论只依赖 module/callable/hash 身份。

candidate `flash_kda` 和其 `.so` 必须来自 `$PATCHED_ROOT`；FP64 fused recurrent 和 Triton chunk 都必须来自 `$FLA_ROOT`。log 和 JSON 都记录这些实际源文件及 SHA256。

## 执行边界与复现

这个入口绝不调用 `sbatch`、`salloc` 或 `srun`。它只能由父调度流程在**已有**的、单张空闲 B300 Slurm allocation 内调用；同时缺失以下任一显式授权都会拒绝运行：

```bash
export C1_FLA_CHUNK_VALIDATION_GPU_AUTHORIZED=1
export A02_ROOT=/home/lcpu/85117379/codex-a02-20260819-main/assignment02
export PATCHED_ROOT=/home/lcpu/85117379/flashkda-vshard4-prefetch2-1ce47ea-b300-r1
export FLA_ROOT=/home/lcpu/85117379/fla-a3edffc
export PYTHON_BIN="$A02_ROOT/.venv/bin/python"
export PYTHON_INCLUDE=/home/lcpu/85117379/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/include/python3.12
export LABEL=b300_sm103a_fla_chunk_repro_r4
bash "$A02_ROOT/team/c1_flashkda/challenge_fla_chunk_validation/run_clean_fla_chunk_validation.sh" --authorized-by-parent
```

`LABEL` 是必填的输出标识。shell fail-closed 绑定 `PATCHED_ROOT` commit
`1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`、FLA commit
`a3edffc39eb5a3d45e9deab5ff9ec4f14f88474d`（如有经过审核的不同 candidate
snapshot，父流程可显式设置对应 `EXPECTED_PATCHED_COMMIT`）；要求 FLA tracked
tree clean、官方 `tests/test_fwd.py` 没有本地 diff、且只能看见一张 B300 SM10.3。
官方 `torch_ref.py` 会通过 PyTorch JIT 构建/加载它自己的 `sigmoid_ext` helper，故
shell 还 fail-closed 要求 venv 内 `ninja`、`$CUDA_HOME/include/cuda_runtime.h` 与
`$PYTHON_INCLUDE/Python.h`，并导出 `CUDA_HOME`、CUDA `PATH` 和 `CPATH`；这不是
FlashKDA 重编译，也不改动 candidate/FLA source tree。

作业会记录：Slurm/GPU 身份、candidate/FLA commit 与 status、runner/test/`torch_ref.py`/candidate Python/SO/FLA chunk、fused-recurrent 及 chunk-fwd 的 SHA256；并在 `PRE`、`AFTER_OFFICIAL_TESTS`、`POST` 三处检查 compute app 为空且显存为 0 MiB。`FINAL_RC=0` 才代表本次 runner 成功，JSON 位于 `results/c1_fla_chunk_validation_<LABEL>.json`，同名 `.log` 是完整审计日志。两张官方测试绘制的 `plot.png`、`plot_varlen.png` 也仅写入这个 `results/` 目录。

## 结论边界

通过时，它支持如下窄结论：在记录的 B300、candidate `.so` 和 pinned FLA source 身份下，官方 fixed/packed-varlen FLA 对拍实际执行了 candidate FlashKDA、FP64 fused recurrent gold 和未被 backend 截获的 FLA Triton `chunk_kda`，三者的 output 与 final state 都进入官方测试流程。只有 candidate output 是 hard tolerance；state 与 Triton output 必须按 upstream 的 warning/error-ratio 口径解读。

它不证明通用 varlen、跨架构、模型质量、全 TP8 并发或性能收益；这些仍须由各自的独立 runner/证据闭合。

## 已完成证据（B300）

single-B300 keeper job **12216** 中的 `r3` 在 B300 SM10.3 上以
`FLA_DISABLE_BACKEND_DISPATCH=1` 完成。官方
`test_fwd_vs_fla` 与 `test_fwd_varlen_vs_fla` 都输出 `Assert results: Success`；两个测试
各有 10 个 gate/bias case，因此 candidate FlashKDA、FP64 fused-recurrent gold、direct
Triton `chunk_kda` 各实际调用 20 次（fixed/varlen 各 10 次），每次的 output 与 final
state 都是非空 CUDA tensor。candidate output 的 upstream hard tolerance 全部通过；日志中
Triton chunk 对 FP64 gold 的最大 output/state error ratio 分别为 fixed
`5.409708e-3/6.832830e-3`、varlen `6.088918e-3/3.617549e-3`，这些是量化报告而非新增
post-hoc hard gate。JSON 还记录 public export 与 pinned FLA `chunk.py`/
`fused_recurrent.py` module identity 一致，且 dispatch 在 import 时已经关闭。

- [审计 JSON](results/c1_fla_chunk_validation_b300_sm103a_fla_chunk_r3.json)：SHA256
  `4564b7e351b093b4e438ac0c9dd2575f2faa38b71c139050a2b17f57395b5cd2`；
- [完整 clean 日志](results/c1_fla_chunk_validation_b300_sm103a_fla_chunk_r3_job12216.log)：SHA256
  `cec980cbc4e82915acee0ac11bb638798940ad3a63571d22b20d5c527a858e75`。
- [fixed error plot](results/plot.png)：2,598,703 bytes，SHA256
  `02c7007ebf5dab03f388d597b6d1386e88ee6c910c1b27b905c5174ba60443ee`；
- [varlen error plot](results/plot_varlen.png)：2,734,546 bytes，SHA256
  `ac86478034b5ff804990306fa2796df086030fef24248da3959c227222d21698`。

日志的 `PRE`、`AFTER_OFFICIAL_TESTS`、`POST` 均是 0 MiB，`FINAL_RC=0`。r1 的 source
identity 包装器误报和 r2 的 JIT 缺 headers 都 fail-closed 且清卡，不能作为此结论的证据。
