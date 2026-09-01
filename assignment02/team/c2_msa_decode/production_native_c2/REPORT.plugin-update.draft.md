### Production-native `C=2`：从整库替换回退到独立 AOT 插件（增补草稿）

下表先固定本节符号；所有未填的动态结果均不得由静态审计、旧 microbenchmark 或 NCU 计数器替代。

| 符号 | 含义 | 本节已冻结/使用的范围 |
| --- | --- | --- |
| `B` | decode 请求批大小 | 生产接入候选合同为 `B=16`；直接算子计时不能代替该合同 |
| `Qlen` | 每个请求本轮参与 decode 的 query 长度 | `Qlen=1` |
| `C` | native cluster 内协作 CTA 数 | `C=2` |
| `H_q`,`H_kv`,`D` | query head 数、KV head 数、head 维度 | `64`,`4`,`128` |
| `P`,`K` | KV page 长度、每个 KV head 选中的稀疏 page 数 | `P=128`,`K=16` |
| `t_native`,`t_triton` | 同一冻结、同一输入合同下 native / Triton 后端 CUDA-event 时间 | 仅未来公平 v2 harness 可形成比值 |
| `R=t_triton/t_native` | 同合同后端速度比 | 只有两臂均通过同一 correctness gate 后才报告 |

#### 已完成的 AOT 与真实后端接入证据

1. **冻结与直接算子闭环。** job 11181 在 B300 上从精确 d4 commit
   `d4da0c55af3aa231b6209bf77871f3ed36eab0d2` 构建 native C2 产物；该 whole-stable
   库的 SHA-256 为
   `beed8c557858da9bdd9e5f4c3681a04c29c8435dd92994f43c9935ce715ed064`。随后 job 11190
   的直接算子检查在两个 seed 上最大绝对误差约 `5.7e-5`、caller output 指针稳定，静态
   检查见到 144 条 SASS HMMA，并保留 cluster/mbarrier 指令。job 11201 在八个 seed 上以
   FP64 oracle（绝对/相对 tolerance 分别 `1e-4`/`1e-3`）通过、重复输出逐 bit 一致，
   该**直接算子 harness** 的 mean 为 `0.507847 ms`。它是 native C2 的独立闭环，既不是
   installed-wheel 结论，也不得同其他 harness 的时间作比例。

2. **真实 backend 首次穿透与公平性修复。** job 11213 已在第二个 B300 UUID 上经
   `MiniMaxM3SparseMSAImpl.forward` 进入 native 路径：没有直接调用算子或 Python
   monkeypatch，并观测到一次 dispatcher 与一次 native kernel。旧 harness 给出 native
   `0.294432 ms`、Triton `0.029344 ms`（约 `10.03x` 慢），但两臂 query 语义不完全相同，
   因此该比例只能说明当时接入可执行，**不能作为公平性能结论**。后续比较统一改用
   [v2 full-backend harness](production_native_c2/native_c2_full_backend_bench.py)，它强制两臂
   使用同一 BF16 query 语义；冻结 SHA-256 为
   `7883edc25df48e3b69a9f4948775a1baafd3ba0b1bd17edcfeac6644aa7b4762`。在该 v2 harness
   重新得到成对的 oracle、dispatch、kernel-count 与 `t_native/t_triton` 前，不报告
   production backend 的速度比。

3. **整块 stable 库 overlay 的构建成功、安装回归失败。** job 11214 成功产生实验性
   overlay wheel；其 provenance 在
   [job11214 JSON](experiment_logs/c2_native_c2_production_aot/c2-native-overlay-provenance-job11214.json)。
   base wheel SHA-256 为
   `91156a7bcfbf729a7213a6ac2a16b64b45c48e36863db30cf7101ddcb5447e06`，派生产物 SHA-256 为
   `22135a9baf42c9418728a2499f509fa521a5d6be11e0819d34c3c101c28a2c6a`；RECORD 和声明 overlay
   成员的构建侧审计均通过。该产物与 base 使用相同 distribution version，只是实验性
   overlay，不能称为发布或部署结论。

   然而 job 11310 在 fresh target 中完成 wheel hash/install 后，dispatcher surface gate
   故意失败（`FINAL_RC=1`，日志见
   [job11310](experiment_logs/c2_native_c2_production_aot/slurm-wheel-runtime-11310.log)）。新库保留了
   schema 名称，却少了 10 个 base CUDA 注册：

   - `_C::allspark_w8a16_gemm`
   - `_C::cutlass_encode_and_reorder_int4b`
   - `_C::cutlass_encode_and_reorder_int4b_grouped`
   - `_C::cutlass_pack_scale_fp8`
   - `_C::cutlass_w4a8_mm`
   - `_C::cutlass_w4a8_moe_mm`
   - `_C::machete_mm`
   - `_C::machete_prepack_B`
   - `_C::machete_supported_schedules`
   - `_C::rearrange_kn_weight_as_n32k16_order`

   对比记录见 [baseline registration audit](experiment_logs/c2_native_c2_production_aot/baseline-stable-ops-job11310.json)。
   根因是 whole-stable 构建只配置 `TORCH_CUDA_ARCH_LIST=10.3f`：精确 d4 的 CMake 对
   AllSpark（8.x）与 Machete（9.0a）有条件编译，而 schema 可被无条件保留，遂形成“有
   schema、无 CUDA 实现”的回归。这是稳定库构建覆盖范围的架构问题，不以放宽 gate 掩盖；
   whole-stable 替换在此停止。

#### 独立插件的受控替代

替代方案是不再改写 `vllm/_C_stable_libtorch.abi3.so`，而是新增一个仅含 C2 source、仅针对
SM103 (`10.3f`) 的 AOT DSO。它在 DSO 内以 `STABLE_TORCH_LIBRARY_FRAGMENT` 定义 schema，
以 `STABLE_TORCH_LIBRARY_IMPL(..., CUDA, ...)` 注册实现；adapter 通过
`torch.ops.load_library` 一次性加载 wheel 内 DSO，若文件、dlopen、schema 或 CUDA 注册任一
步骤不成立即安全回退 Triton。其不是 Python import module，不依赖 `PyInit_*`，不做 JIT，
也不使用 monkeypatch。

| 冻结对象 | SHA-256 / 定位 | 静态审计结论 |
| --- | --- | --- |
| plugin schema patch | [native_c2_plugin_schema.patch](production_native_c2/native_c2_plugin_schema.patch)；`53e7cd4b09a8999010442934c863427c30b019fc549d7f0cf0f3d16ccc1e2e6e` | schema 与 CUDA impl 同置 plugin，避免依赖 whole-stable 的静态注册 |
| plugin CMake patch | [exact_d4_native_c2_plugin_cmake.patch](production_native_c2/exact_d4_native_c2_plugin_cmake.patch)；`5b145f36ce5f8b12dae71ce02e77d65545a29eefcb5a6045f591348bc325ac35` | 单独 target、`USE_SABI 3`、CUDA 13+/`10.3f` gate；不链接/覆盖 stable target |
| plugin loader patch | [exact_d4_native_c2_plugin_python_loader.patch](production_native_c2/exact_d4_native_c2_plugin_python_loader.patch)；`9cb3a22216048083d7a4802ba700fa07fd892436151871dc6e909c77239adb33` | 锁保护一次加载，检查 `_C::native_c2_msa_decode` 的 CUDA kernel；失败保持既有 Triton 路线 |
| NCU 驱动的 v2 候选 | [native_c2_v2_warp_owned_pv.patch](production_native_c2/native_c2_v2_warp_owned_pv.patch)；`2fa34736ba80a50cd6d4a40377ad419709941349eff5c26e745541cfd8862816` | 独立静态质量审查为 GO：每 warp 拥有 16 维 PV slice，热循环访问 32 个 distinct bank，保留最终 CTA barrier 并检查 `cudaLaunchKernelEx` 返回值 |

上述 GO 仅表示静态结构与协议审计通过；独立 plugin 的 B300 build、wheel、fresh install、真实
backend correctness 或性能都**尚未在本段中宣称完成**。未来 plugin wheel 还须逐字节保持 base
stable DSO (`cee888ed2e3a4d6f27564bd615b20d9e49d472ff3db03429b21823ab39800442`) 不变，并以
fresh-target gate 验证 load 前为 base surface、load 后只出现 native C2 的预期 CUDA 注册。

#### NCU 机制证据与下一步优化

job 11301 完成且 `all_gates_pass=true`；机器可读结果见
[job11301 NCU JSON](experiment_logs/c2_native_c2_production_aot/native-c2-ncu-job11301.json)，runner 为
[profile_native_c2_ncu.slurm](production_native_c2/profile_native_c2_ncu.slurm)（SHA-256
`4d5d9ad1d8a99884fd851920d3411f5c31ea1aca4db04673e985c0c562d9b160`）。这个 profile 只作机制诊断，
NCU replay/instrumentation 使 `gpu__time_duration.sum=527,424 ns` **不可作 benchmark 时间**。

| 计数器 | 精确值 | 对 v2 的窄解释 |
| --- | ---: | --- |
| DRAM read / write | `8,611,072 B` / `0 B` | 有读流量，但本轮 profile 不报告写端带宽结论 |
| L2 bytes | `58,413,152 B` | 约为 DRAM read 的 `6.7835x`；保留字节单位，避免数量级误读 |
| tensor active / tensor instructions | `0.71%` / `262,144` | Tensor Core 未成为主导忙碌来源 |
| long-scoreboard / wait stall | `271.61%` / `178.54%` | 应优先降低数据依赖与等待，而非从计数器外推绝对速度 |
| barrier stall ratio | `19.14 inst` | 单位为 inst，不能与百分比直接比较 |
| shared load conflicts / wavefronts | `12,137,140` / `15,534,918` (`0.78128`) | 支持首先试验 warp-owned PV 的冲突降解 |
| shared store conflicts / wavefronts | `991,966` / `3,332,576` (`0.29766`) | 次级热点 |
| active warps / regs / shared per block | `3.58` / `68` / `38,752 B` | 资源与同步都是限制线索 |
| cluster size / GPU cluster occupancy | `4` / `3.29%` | 只说明该 profile 下 cluster 资源压力，不能替代 end-to-end 吞吐 |

因此，在 independent plugin 的动态接入闭环通过后，优先做的 kernel 试验是 warp-owned PV v2：
同一 oracle、same-input ABBA、post-timing fresh check，并要求相对 v1 的真实 backend 时间至少有
预先声明的改进；否则拒绝该分支。四 producer 仅在 v2 已形成可比较的独立 plugin 候选后再试，
不能并行地消耗主验证预算。

#### 优先级、跨 GPU 的边界与待填结果

跨 GPU 复现不是当前最高优先级。当前最小闭环先后是：(1) independent AOT plugin 不改变 base
stable DSO 的 build/wheel/fresh-target 验证；(2) v2 harness 下真实 backend 的同输入 correctness、
dispatch/kernel-count 与性能；(3) 用 job11301 的热点证据验证或拒绝 warp-owned PV。原因是跨 GPU
只能检验一个已经冻结且可安装的 candidate 的外部有效性；目前 candidate 尚缺上述动态闭环。先前
虽然已有一个第二 UUID 的 source-checkout backend 运行，但它不是冻结公平 v2 harness 上的成对复现，
不构成严格跨 GPU 结论。

| 后续动态 gate | 当前状态 | 结果待填字段（通过前保持空白） |
| --- | --- | --- |
| independent plugin AOT build | 未运行/未在此报告 | B300 job、commit/patch SHA、plugin DSO 路径与 SHA、`FINAL_RC` |
| plugin overlay wheel | 未运行/未在此报告 | base/derived wheel SHA、provenance、stable DSO byte equality、RECORD audit |
| fresh-target plugin runtime | 未运行/未在此报告 | install/hash、load 前后注册差、FP64 oracle、fallback 反例、真实 dispatch/kernel count、`t_native`,`t_triton`,`R` |
| warp-owned PV v2 | 仅静态 GO | 两 seed/重复性、post-timing oracle、NCU 差分、预声明门槛、接受或拒绝 |
| 严格跨 GPU 复现 | 未完成 | 两个 UUID、相同 wheel/plugin/harness SHA、相同输入与 tolerance、每卡独立性能和差异说明 |
| 模型服务 E2E | 未完成，外部 checkpoint/config/tokenizer 资产仍缺 | 资产路径、TP/启动配置、TTFT、吞吐及与 backend-layer 的边界 |

本增补应替换旧文中“整块 stable overlay 可作为下一轮主线”的暗示：保留 job11214/11310 作为可复核
失败审计，主线改为独立 plugin；在动态 gate 完成前，不把静态审计、NCU 机制计数或旧不公平时间写成
production 性能、跨 GPU 复现或模型服务结论。
