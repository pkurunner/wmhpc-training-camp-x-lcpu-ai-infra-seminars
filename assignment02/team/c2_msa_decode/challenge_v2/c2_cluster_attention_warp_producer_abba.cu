// Fair C=2 native producer-mapping AB/BA microbenchmark.
//
// The scalar arm is the audited job-10731 mbarrier kernel imported verbatim
// below.  The warp arm retains its 4-CTA cluster shape, rank-2-local mbarrier,
// remote DSM release-arrivals, final lifetime barrier, shared-memory layout,
// inputs, and output ABI.  It changes only rank-0/1 producer arithmetic from
// one thread per GQA head to two serial GQA heads per warp.
//
// Boundary: this is a scalar native correctness-prototype producer mapping
// signal, not a production fusion, throughput result, or model/server claim.

#define main c2_cluster_attention_mbarrier_smoke_embedded_main
#include "c2_cluster_attention_mbarrier_smoke.cu"
#undef main

namespace {

constexpr int kWarmupEach = 20;
constexpr int kAbbapairs = 101;
constexpr int kSamplesPerArm = 2 * kAbbapairs;
constexpr int kTimingSeed = 2026;
constexpr int kWarpsPerBlock = 8;
constexpr int kWarpSize = 32;
constexpr int kHeadsPerWarp = 2;
constexpr int kDimsPerLane = 4;
constexpr unsigned kFullWarpMask = 0xffffffffu;

static_assert(kThreadsPerBlock == kWarpsPerBlock * kWarpSize,
              "the warp producer requires exactly eight full warps");
static_assert(kGqaGroup == kWarpsPerBlock * kHeadsPerWarp,
              "each warp must own exactly two serial GQA heads");
static_assert(kHeadDim == kWarpSize * kDimsPerLane,
              "each lane must own four disjoint head dimensions");
static_assert(kSelectedPages == 2 * kPagesPerProducer && kPagesPerProducer == 8,
              "the two producer CTAs must retain their eight-page split");
static_assert(kNumCtas == 4, "the audited DSM protocol has four CTA roles");

constexpr const char* kWarpAbbaBoundary =
    "scalar native C=2 correctness-prototype producer-mapping signal only; "
    "not a production fusion, throughput result, or vLLM/model/server speedup";
constexpr const char* kWarpProducerDescription =
    "8 warps per producer CTA; each warp serially computes 2 GQA heads; each lane owns "
    "d=lane+{0,32,64,96}, caches Q in registers, reduces four-term QK dots with full-mask "
    "shuffle-down, broadcasts score, and writes disjoint BF16 partial dimensions";

// This kernel is intentionally a structural copy of the imported audited
// mbarrier kernel.  The mbarrier initialization, release-arrive, acquire wait,
// rank-2 merge, output ABI, and lifecycle barriers are deliberately unchanged.
// Only the producer arithmetic inside role 0/1 is warp cooperative.
__global__ void cluster_attention_mbarrier_warp_producer_kernel(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    const int* topk_idx,
    const int* block_table,
    int sequence_length,
    __nv_bfloat16* caller_output) {
  __shared__ __nv_bfloat16 local_partial[kGqaGroup * kHeadDim];
  __shared__ float local_lse[kGqaGroup];
  __shared__ __align__(8) std::uint64_t producer_ready_barrier;
  __shared__ int producer_ready;

  const cg::cluster_group cluster = cg::this_cluster();
  const int role = static_cast<int>(cluster.block_rank());
  const int kv_head = static_cast<int>(blockIdx.x / kNumCtas);
  const int thread = static_cast<int>(threadIdx.x);
  const int query_position = sequence_length - 1;

  if (role == 2 && thread == 0) {
    cuda::ptx::mbarrier_init(&producer_ready_barrier, kMBarrierExpectedArrivals);
    producer_ready = 0;
  }
  __syncthreads();
  cluster.sync();

  if (role == 0 || role == 1) {
    const int lane = thread & (kWarpSize - 1);
    const int warp = thread / kWarpSize;
    const int dim0 = lane;
    const int dim1 = lane + kWarpSize;
    const int dim2 = lane + 2 * kWarpSize;
    const int dim3 = lane + 3 * kWarpSize;
    const int selected_begin = role * kPagesPerProducer;

#pragma unroll
    for (int head_in_warp = 0; head_in_warp < kHeadsPerWarp; ++head_in_warp) {
      const int group_head = warp * kHeadsPerWarp + head_in_warp;
      const int query_head = kv_head * kGqaGroup + group_head;
      const __nv_bfloat16* query_row = query + static_cast<std::size_t>(query_head) * kHeadDim;
      // Each of this lane's four Q elements remains register-resident across
      // all selected tokens for the current GQA head.
      const float q0 = __bfloat162float(query_row[dim0]);
      const float q1 = __bfloat162float(query_row[dim1]);
      const float q2 = __bfloat162float(query_row[dim2]);
      const float q3 = __bfloat162float(query_row[dim3]);
      float acc0 = 0.0f;
      float acc1 = 0.0f;
      float acc2 = 0.0f;
      float acc3 = 0.0f;
      float max_score = -INFINITY;
      float normalizer = 0.0f;

      for (int selected = selected_begin; selected < selected_begin + kPagesPerProducer; ++selected) {
        const int logical_page = topk_idx[kv_head * kSelectedPages + selected];
        const int physical_page = block_table[logical_page];
        for (int token = 0; token < kPageSize; ++token) {
          const int key_position = logical_page * kPageSize + token;
          if (key_position <= query_position && key_position < sequence_length) {
            const std::size_t kv_base = cache_offset(physical_page, kv_head, token, 0);
            float partial_dot = fmaf(q0, __bfloat162float(key_cache[kv_base + dim0]), 0.0f);
            partial_dot = fmaf(q1, __bfloat162float(key_cache[kv_base + dim1]), partial_dot);
            partial_dot = fmaf(q2, __bfloat162float(key_cache[kv_base + dim2]), partial_dot);
            partial_dot = fmaf(q3, __bfloat162float(key_cache[kv_base + dim3]), partial_dot);
#pragma unroll
            for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
              partial_dot += __shfl_down_sync(kFullWarpMask, partial_dot, offset);
            }
            const float score = __shfl_sync(kFullWarpMask, partial_dot, 0) * kScaleLog2e;
            const float next_max = fmaxf(max_score, score);
            const float alpha = isfinite(max_score) ? exp2f(max_score - next_max) : 0.0f;
            const float beta = exp2f(score - next_max);
            acc0 = acc0 * alpha + beta * __bfloat162float(value_cache[kv_base + dim0]);
            acc1 = acc1 * alpha + beta * __bfloat162float(value_cache[kv_base + dim1]);
            acc2 = acc2 * alpha + beta * __bfloat162float(value_cache[kv_base + dim2]);
            acc3 = acc3 * alpha + beta * __bfloat162float(value_cache[kv_base + dim3]);
            normalizer = normalizer * alpha + beta;
            max_score = next_max;
          }
        }
      }

      const std::size_t partial_base = static_cast<std::size_t>(group_head) * kHeadDim;
      if (normalizer > 0.0f) {
        local_partial[partial_base + dim0] = __float2bfloat16_rn(acc0 / normalizer);
        local_partial[partial_base + dim1] = __float2bfloat16_rn(acc1 / normalizer);
        local_partial[partial_base + dim2] = __float2bfloat16_rn(acc2 / normalizer);
        local_partial[partial_base + dim3] = __float2bfloat16_rn(acc3 / normalizer);
        if (lane == 0) {
          local_lse[group_head] = max_score + log2f(normalizer);
        }
      } else {
        local_partial[partial_base + dim0] = __float2bfloat16_rn(0.0f);
        local_partial[partial_base + dim1] = __float2bfloat16_rn(0.0f);
        local_partial[partial_base + dim2] = __float2bfloat16_rn(0.0f);
        local_partial[partial_base + dim3] = __float2bfloat16_rn(0.0f);
        if (lane == 0) {
          local_lse[group_head] = -INFINITY;
        }
      }
    }
  }

  __syncthreads();
  if ((role == 0 || role == 1) && thread == 0) {
    std::uint64_t* remote_rank2_barrier = cluster.map_shared_rank(&producer_ready_barrier, 2);
    cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release,
                               cuda::ptx::scope_cluster,
                               cuda::ptx::space_cluster,
                               remote_rank2_barrier);
  }

  if (role == 2 && thread == 0) {
    bool ready = false;
#pragma unroll 1
    for (int poll = 0; poll < kMBarrierMaxPolls; ++poll) {
      if (cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire,
                                               cuda::ptx::scope_cluster,
                                               &producer_ready_barrier,
                                               kMBarrierInitialParity)) {
        ready = true;
        break;
      }
    }
    producer_ready = ready ? 1 : 0;
  }

  __syncthreads();
  if (role == 2 && thread < kGqaGroup) {
    const std::size_t output_base = static_cast<std::size_t>(kv_head * kGqaGroup + thread) * kHeadDim;
    if (producer_ready == 0) {
      for (int dim = 0; dim < kHeadDim; ++dim) {
        caller_output[output_base + dim] = __float2bfloat16_rn(kSentinel);
      }
    } else {
      const __nv_bfloat16* remote_partial0 = cluster.map_shared_rank(local_partial, 0);
      const __nv_bfloat16* remote_partial1 = cluster.map_shared_rank(local_partial, 1);
      const float* remote_lse0 = cluster.map_shared_rank(local_lse, 0);
      const float* remote_lse1 = cluster.map_shared_rank(local_lse, 1);
      const float lse0 = remote_lse0[thread];
      const float lse1 = remote_lse1[thread];
      const float lse_max = fmaxf(lse0, lse1);
      const float weight0 = isfinite(lse0) ? exp2f(lse0 - lse_max) : 0.0f;
      const float weight1 = isfinite(lse1) ? exp2f(lse1 - lse_max) : 0.0f;
      const float denominator = weight0 + weight1;
      for (int dim = 0; dim < kHeadDim; ++dim) {
        const float partial0 = __bfloat162float(remote_partial0[thread * kHeadDim + dim]);
        const float partial1 = __bfloat162float(remote_partial1[thread * kHeadDim + dim]);
        const float merged = denominator > 0.0f
            ? (partial0 * weight0 + partial1 * weight1) / denominator
            : 0.0f;
        caller_output[output_base + dim] = __float2bfloat16_rn(merged);
      }
    }
  }
  cluster.sync();
}

struct DeviceBuffers {
  __nv_bfloat16* query = nullptr;
  __nv_bfloat16* key_cache = nullptr;
  __nv_bfloat16* value_cache = nullptr;
  int* topk_idx = nullptr;
  int* block_table = nullptr;
  __nv_bfloat16* scalar_output = nullptr;
  __nv_bfloat16* warp_output = nullptr;
  DeviceBuffers(const DeviceBuffers&) = delete;
  DeviceBuffers& operator=(const DeviceBuffers&) = delete;
  DeviceBuffers() = default;
  ~DeviceBuffers() { release(); }
  void release() noexcept {
    cudaFree(warp_output); cudaFree(scalar_output); cudaFree(block_table); cudaFree(topk_idx);
    cudaFree(value_cache); cudaFree(key_cache); cudaFree(query);
    warp_output = nullptr; scalar_output = nullptr; block_table = nullptr; topk_idx = nullptr;
    value_cache = nullptr; key_cache = nullptr; query = nullptr;
  }
};

struct LaunchState {
  cudaLaunchAttribute attribute{};
  cudaLaunchConfig_t config{};
  LaunchState() {
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {kNumCtas, 1, 1};
    config.gridDim = dim3(kKvHeads * kNumCtas, 1, 1);
    config.blockDim = dim3(kThreadsPerBlock, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = nullptr;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }
};

void allocate_and_copy(const AttentionInput& input, DeviceBuffers* buffers) {
  const std::size_t output_bytes = kOutputElements * sizeof(__nv_bfloat16);
  const std::vector<__nv_bfloat16> sentinel_output(kOutputElements, __float2bfloat16_rn(kSentinel));
  CUDA_CHECK(cudaMalloc(&buffers->query, input.query.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers->key_cache, input.key_cache.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers->value_cache, input.value_cache.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers->topk_idx, input.topk_idx.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&buffers->block_table, input.block_table.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&buffers->scalar_output, output_bytes));
  CUDA_CHECK(cudaMalloc(&buffers->warp_output, output_bytes));
  CUDA_CHECK(cudaMemcpy(buffers->query, input.query.data(), input.query.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->key_cache, input.key_cache.data(), input.key_cache.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->value_cache, input.value_cache.data(), input.value_cache.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->topk_idx, input.topk_idx.data(), input.topk_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->block_table, input.block_table.data(), input.block_table.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->scalar_output, sentinel_output.data(), output_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->warp_output, sentinel_output.data(), output_bytes, cudaMemcpyHostToDevice));
}

void launch_scalar(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length) {
  CUDA_CHECK(cudaLaunchKernelEx(&launch.config, cluster_attention_mbarrier_kernel,
                                 buffers.query, buffers.key_cache, buffers.value_cache,
                                 buffers.topk_idx, buffers.block_table, sequence_length, buffers.scalar_output));
  CUDA_CHECK(cudaGetLastError());
}

void launch_warp(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length) {
  CUDA_CHECK(cudaLaunchKernelEx(&launch.config, cluster_attention_mbarrier_warp_producer_kernel,
                                 buffers.query, buffers.key_cache, buffers.value_cache,
                                 buffers.topk_idx, buffers.block_table, sequence_length, buffers.warp_output));
  CUDA_CHECK(cudaGetLastError());
}

struct ArmCorrectness {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  bool oracle_finite = true;
  bool finite = true;
  bool sentinel_clean = true;
  bool allclose = true;
};

struct CrossArmDiagnosis {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  bool bfloat16_bitwise_equal = true;
};

ArmCorrectness validate_output(const std::vector<__nv_bfloat16>& output, const std::vector<float>& oracle) {
  if (output.size() != oracle.size()) throw std::runtime_error("output/oracle size mismatch");
  const __nv_bfloat16 sentinel = __float2bfloat16_rn(kSentinel);
  ArmCorrectness result{};
  for (std::size_t index = 0; index < output.size(); ++index) {
    const float actual = __bfloat162float(output[index]);
    const float expected = oracle[index];
    const float abs_error = std::fabs(actual - expected);
    result.max_abs = std::max(result.max_abs, abs_error);
    result.max_rel = std::max(result.max_rel, abs_error / std::max(std::fabs(expected), 1.0e-7f));
    result.oracle_finite = result.oracle_finite && std::isfinite(expected);
    result.finite = result.finite && std::isfinite(actual);
    result.sentinel_clean = result.sentinel_clean && !same_bfloat16_bits(output[index], sentinel);
    result.allclose = result.allclose && abs_error <= kAtol + kRtol * std::fabs(expected);
  }
  return result;
}

CrossArmDiagnosis diagnose_cross_arm(const std::vector<__nv_bfloat16>& scalar,
                                     const std::vector<__nv_bfloat16>& warp) {
  if (scalar.size() != warp.size()) throw std::runtime_error("scalar/warp output size mismatch");
  CrossArmDiagnosis result{};
  for (std::size_t index = 0; index < scalar.size(); ++index) {
    const float lhs = __bfloat162float(scalar[index]);
    const float rhs = __bfloat162float(warp[index]);
    const float abs_error = std::fabs(lhs - rhs);
    result.max_abs = std::max(result.max_abs, abs_error);
    result.max_rel = std::max(result.max_rel, abs_error / std::max(std::fabs(lhs), 1.0e-7f));
    result.bfloat16_bitwise_equal = result.bfloat16_bitwise_equal && same_bfloat16_bits(scalar[index], warp[index]);
  }
  return result;
}

struct SeedCorrectness {
  int seed = 0;
  int sequence_length = 0;
  bool hierarchy_valid = false;
  int adversarial_unselected_visible_pages = 0;
  int adversarial_masked_tokens = 0;
  ArmCorrectness scalar{};
  ArmCorrectness warp{};
  CrossArmDiagnosis cross_arm{};
};

struct PostTimingCorrectness {
  int seed = 0;
  bool hierarchy_valid = false;
  ArmCorrectness scalar{};
  ArmCorrectness warp{};
  CrossArmDiagnosis cross_arm{};
};

SeedCorrectness check_seed(const AttentionInput& input, const LaunchState& launch,
                           DeviceBuffers* timing_buffers) {
  const bool hierarchy_valid = validate_indirection(input);
  if (!hierarchy_valid) throw std::runtime_error("input indirection failed validation before oracle or GPU launch");
  const std::vector<float> oracle = cpu_fp64_accum_oracle(input);
  DeviceBuffers local_buffers;
  DeviceBuffers* buffers = input.seed == kTimingSeed ? timing_buffers : &local_buffers;
  allocate_and_copy(input, buffers);
  launch_scalar(launch, *buffers, input.sequence_length);
  launch_warp(launch, *buffers, input.sequence_length);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> scalar(kOutputElements);
  std::vector<__nv_bfloat16> warp(kOutputElements);
  CUDA_CHECK(cudaMemcpy(scalar.data(), buffers->scalar_output, scalar.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(warp.data(), buffers->warp_output, warp.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  return SeedCorrectness{input.seed, input.sequence_length, hierarchy_valid,
                         input.adversarial_unselected_visible_pages, input.adversarial_masked_tokens,
                         validate_output(scalar, oracle), validate_output(warp, oracle),
                         diagnose_cross_arm(scalar, warp)};
}

bool correct_arm(const ArmCorrectness& arm) {
  return arm.oracle_finite && arm.finite && arm.sentinel_clean && arm.allclose;
}

void require_correct(const SeedCorrectness& result) {
  if (!result.hierarchy_valid || result.adversarial_unselected_visible_pages <= 0
      || result.adversarial_masked_tokens != kKvHeads * (kPageSize - 1)
      || !correct_arm(result.scalar) || !correct_arm(result.warp)) {
    throw std::runtime_error("scalar/warp correctness gate failed before timing");
  }
}

PostTimingCorrectness revalidate_after_timing(const DeviceBuffers& buffers) {
  // This validates only the deterministic final outputs after all timed launches;
  // no claim is made that every intermediate timed output was copied back.
  AttentionInput input = make_input(kTimingSeed);
  const bool hierarchy_valid = validate_indirection(input);
  if (!hierarchy_valid) throw std::runtime_error("post-timing input indirection validation failed");
  const std::vector<float> oracle = cpu_fp64_accum_oracle(input);
  std::vector<__nv_bfloat16> scalar(kOutputElements);
  std::vector<__nv_bfloat16> warp(kOutputElements);
  CUDA_CHECK(cudaMemcpy(scalar.data(), buffers.scalar_output, scalar.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(warp.data(), buffers.warp_output, warp.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  const PostTimingCorrectness result{input.seed, hierarchy_valid, validate_output(scalar, oracle),
                                     validate_output(warp, oracle), diagnose_cross_arm(scalar, warp)};
  if (!result.hierarchy_valid || !correct_arm(result.scalar) || !correct_arm(result.warp)) {
    throw std::runtime_error("post-timing scalar/warp correctness gate failed");
  }
  return result;
}

float time_scalar_once(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length,
                       cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start));
  launch_scalar(launch, buffers, sequence_length);
  CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
  float milliseconds = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
  return milliseconds * 1000.0f;
}

float time_warp_once(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length,
                     cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start));
  launch_warp(launch, buffers, sequence_length);
  CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
  float milliseconds = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
  return milliseconds * 1000.0f;
}

struct Statistics { double p10_us = 0.0; double median_us = 0.0; double p90_us = 0.0; };

Statistics summarize_us(std::vector<float> values) {
  if (values.empty()) throw std::runtime_error("cannot summarize empty timing series");
  std::sort(values.begin(), values.end());
  const std::size_t count = values.size();
  const std::size_t p10_index = (10 * count + 99) / 100 - 1;
  const std::size_t p90_index = std::min(count - 1, (90 * count + 99) / 100 - 1);
  const double median = count % 2 == 1 ? static_cast<double>(values[count / 2])
      : (static_cast<double>(values[count / 2 - 1]) + static_cast<double>(values[count / 2])) / 2.0;
  return Statistics{static_cast<double>(values[p10_index]), median, static_cast<double>(values[p90_index])};
}

void print_statistics_json(const Statistics& stats) {
  std::cout << "{\"p10_us\":" << stats.p10_us << ",\"median_us\":" << stats.median_us
            << ",\"p90_us\":" << stats.p90_us << '}';
}
void print_samples_json(const std::vector<float>& values) {
  std::cout << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}
void print_arm_json(const ArmCorrectness& arm) {
  std::cout << "{\"max_abs\":" << arm.max_abs << ",\"max_rel\":" << arm.max_rel
            << ",\"oracle_finite\":" << (arm.oracle_finite ? "true" : "false")
            << ",\"finite\":" << (arm.finite ? "true" : "false")
            << ",\"sentinel_clean\":" << (arm.sentinel_clean ? "true" : "false")
            << ",\"allclose\":" << (arm.allclose ? "true" : "false") << '}';
}
void print_cross_json(const CrossArmDiagnosis& cross) {
  std::cout << "{\"max_abs\":" << cross.max_abs << ",\"max_rel\":" << cross.max_rel
            << ",\"bfloat16_bitwise_equal\":" << (cross.bfloat16_bitwise_equal ? "true" : "false") << '}';
}
void print_seed_json(const SeedCorrectness& result) {
  std::cout << "{\"seed\":" << result.seed << ",\"sequence_length\":" << result.sequence_length
            << ",\"hierarchy_valid\":" << (result.hierarchy_valid ? "true" : "false")
            << ",\"adversarial_unselected_visible_pages\":" << result.adversarial_unselected_visible_pages
            << ",\"adversarial_masked_tokens\":" << result.adversarial_masked_tokens << ",\"scalar\":";
  print_arm_json(result.scalar); std::cout << ",\"warp\":"; print_arm_json(result.warp);
  std::cout << ",\"cross_arm\":"; print_cross_json(result.cross_arm); std::cout << '}';
}
void print_post_json(const PostTimingCorrectness& result) {
  std::cout << "{\"seed\":" << result.seed << ",\"hierarchy_valid\":"
            << (result.hierarchy_valid ? "true" : "false") << ",\"scalar\":";
  print_arm_json(result.scalar); std::cout << ",\"warp\":"; print_arm_json(result.warp);
  std::cout << ",\"cross_arm\":"; print_cross_json(result.cross_arm); std::cout << '}';
}

void print_success_json(const cudaDeviceProp& property, const cudaFuncAttributes& scalar_attributes,
                        const cudaFuncAttributes& warp_attributes, int runtime_version, int driver_version,
                        int cluster_launch, const std::vector<SeedCorrectness>& correctness,
                        const PostTimingCorrectness& post, const std::vector<float>& scalar_ab,
                        const std::vector<float>& warp_ab, const std::vector<float>& warp_ba,
                        const std::vector<float>& scalar_ba) {
  std::vector<float> scalar_all = scalar_ab; scalar_all.insert(scalar_all.end(), scalar_ba.begin(), scalar_ba.end());
  std::vector<float> warp_all = warp_ab; warp_all.insert(warp_all.end(), warp_ba.begin(), warp_ba.end());
  const Statistics scalar_all_stats = summarize_us(scalar_all);
  const Statistics warp_all_stats = summarize_us(warp_all);
  const Statistics scalar_ab_stats = summarize_us(scalar_ab);
  const Statistics warp_ab_stats = summarize_us(warp_ab);
  const Statistics scalar_ba_stats = summarize_us(scalar_ba);
  const Statistics warp_ba_stats = summarize_us(warp_ba);
  const double speedup = scalar_all_stats.median_us / warp_all_stats.median_us;
  const double ab_speedup = scalar_ab_stats.median_us / warp_ab_stats.median_us;
  const double ba_speedup = scalar_ba_stats.median_us / warp_ba_stats.median_us;
  const bool promotion = speedup >= 1.10 && ab_speedup > 1.05 && ba_speedup > 1.05
      && warp_attributes.localSizeBytes == 0;

  std::cout << std::setprecision(9)
            << "{\"schema\":\"c2-cluster-attention-warp-producer-abba-v1\",\"status\":\"pass\","
            << "\"boundary\":\"" << json_escape(kWarpAbbaBoundary) << "\","
            << "\"timing_seed\":" << kTimingSeed << ",\"shape\":{\"B\":" << kBatch
            << ",\"Hkv\":" << kKvHeads << ",\"Hq\":" << kQueryHeads << ",\"G\":" << kGqaGroup
            << ",\"D\":" << kHeadDim << ",\"page_size\":" << kPageSize
            << ",\"selected_pages\":" << kSelectedPages << ",\"logical_pages\":" << kLogicalPages << "},"
            << "\"cluster_layout\":{\"num_ctas\":" << kNumCtas << ",\"clusters\":" << kKvHeads
            << ",\"selected_pages_per_producer\":" << kPagesPerProducer
            << ",\"threads_per_block\":" << kThreadsPerBlock << "},"
            << "\"producer_contract\":{\"scalar\":\"imported audited one-thread-per-head producer\","
            << "\"warp\":\"" << json_escape(kWarpProducerDescription) << "\","
            << "\"changed_field\":\"rank-0/1 producer compute mapping only\","
            << "\"same_remote_dsm_mbarrier_protocol\":true,\"same_shared_layout_and_output_abi\":true,"
            << "\"same_launch_shape\":true,\"same_real_selected_causal_attention\":true,"
            << "\"persistent_device_buffers_outside_timing\":true,\"caller_owned_independent_outputs\":true,"
            << "\"single_kernel_launch_per_cuda_event_sample\":true,\"ABBA_interleaved\":true,"
            << "\"initialization_copies_and_oracle_outside_timing\":true,"
            << "\"timed_launch_validation_scope\":\"pre-timing two-seed checks plus post-timing final-state recheck; intermediate timed outputs not inspected\"},"
            << "\"synchronization\":{\"mbarrier_expected_arrivals\":" << kMBarrierExpectedArrivals
            << ",\"mbarrier_wait_parity\":" << kMBarrierInitialParity
            << ",\"mbarrier_max_polls\":" << kMBarrierMaxPolls
            << ",\"producer_ready\":\"two remote release-arrivals then rank-2 local acquire parity wait\","
            << "\"cluster_sync\":\"init plus producer-CTA local partial lifetime only\"},"
            << "\"dtype_contract\":{\"producer_partial\":\"bfloat16\",\"caller_output\":\"bfloat16\","
            << "\"oracle_accumulator\":\"float64\",\"oracle\":\"independent two-pass natural-exp direct selected-page causal attention\","
            << "\"tolerance\":{\"rtol\":" << kRtol << ",\"atol\":" << kAtol << "}},"
            << "\"environment\":{\"device\":\"" << json_escape(property.name) << "\",\"capability\":["
            << property.major << ',' << property.minor << "],\"cuda_runtime\":" << runtime_version
            << ",\"cuda_driver\":" << driver_version << ",\"cluster_launch_supported\":"
            << (cluster_launch ? "true" : "false") << "},"
            << "\"resource_model\":{\"static_shared_equal\":"
            << (scalar_attributes.sharedSizeBytes == warp_attributes.sharedSizeBytes ? "true" : "false")
            << ",\"scalar\":{\"static_shared_bytes\":" << scalar_attributes.sharedSizeBytes
            << ",\"num_regs\":" << scalar_attributes.numRegs << ",\"local_bytes\":" << scalar_attributes.localSizeBytes
            << "},\"warp\":{\"static_shared_bytes\":" << warp_attributes.sharedSizeBytes
            << ",\"num_regs\":" << warp_attributes.numRegs << ",\"local_bytes\":" << warp_attributes.localSizeBytes << "}},"
            << "\"correctness\":[";
  for (std::size_t index = 0; index < correctness.size(); ++index) {
    if (index) std::cout << ',';
    print_seed_json(correctness[index]);
  }
  std::cout << "],\"post_timing_correctness\":"; print_post_json(post);
  std::cout << ",\"timing\":{\"protocol\":\"warmup_each_then_101_scalar_warp_warp_scalar_ABBA_pairs\","
            << "\"warmup_each\":" << kWarmupEach << ",\"abba_pairs\":" << kAbbapairs
            << ",\"samples_per_arm\":" << kSamplesPerArm << ",\"raw_samples_us\":{\"scalar\":{\"AB\":";
  print_samples_json(scalar_ab); std::cout << ",\"BA\":"; print_samples_json(scalar_ba);
  std::cout << "},\"warp\":{\"AB\":"; print_samples_json(warp_ab);
  std::cout << ",\"BA\":"; print_samples_json(warp_ba);
  std::cout << "}},\"scalar\":{\"all\":"; print_statistics_json(scalar_all_stats);
  std::cout << ",\"when_launch_order_is_AB\":"; print_statistics_json(scalar_ab_stats);
  std::cout << ",\"when_launch_order_is_BA\":"; print_statistics_json(scalar_ba_stats);
  std::cout << "},\"warp\":{\"all\":"; print_statistics_json(warp_all_stats);
  std::cout << ",\"when_launch_order_is_AB\":"; print_statistics_json(warp_ab_stats);
  std::cout << ",\"when_launch_order_is_BA\":"; print_statistics_json(warp_ba_stats);
  std::cout << "},\"speedup_scalar_over_warp\":" << speedup
            << ",\"speedup_by_partition\":{\"AB\":" << ab_speedup << ",\"BA\":" << ba_speedup << "},"
            << "\"promotion_gate\":{\"combined_scalar_over_warp_at_least_1_10\":" << (speedup >= 1.10 ? "true" : "false")
            << ",\"AB_scalar_over_warp_greater_than_1_05\":" << (ab_speedup > 1.05 ? "true" : "false")
            << ",\"BA_scalar_over_warp_greater_than_1_05\":" << (ba_speedup > 1.05 ? "true" : "false")
            << ",\"warp_local_size_bytes_zero\":" << (warp_attributes.localSizeBytes == 0 ? "true" : "false")
            << ",\"all_correct\":true,\"promoted\":" << (promotion ? "true" : "false") << "}}}" << std::endl;
}

void print_warp_abba_failure_json(const std::string& error) {
  std::cout << "{\"schema\":\"c2-cluster-attention-warp-producer-abba-v1\",\"status\":\"fail\",\"error\":\""
            << json_escape(error) << "\",\"boundary\":\"" << json_escape(kWarpAbbaBoundary) << "\"}" << std::endl;
}

}  // namespace

int main() {
  try {
    int device = 0; CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp property{}; CUDA_CHECK(cudaGetDeviceProperties(&property, device));
    int cluster_launch = 0; CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device));
    int runtime_version = 0; int driver_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version)); CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    cudaFuncAttributes scalar_attributes{}; cudaFuncAttributes warp_attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&scalar_attributes, cluster_attention_mbarrier_kernel));
    CUDA_CHECK(cudaFuncGetAttributes(&warp_attributes, cluster_attention_mbarrier_warp_producer_kernel));
    if (property.major != 10 || property.minor != 3) throw std::runtime_error("requires B300 compute capability 10.3");
    if (cluster_launch == 0) throw std::runtime_error("cudaDevAttrClusterLaunch is false");
    if (scalar_attributes.sharedSizeBytes > property.sharedMemPerBlock || warp_attributes.sharedSizeBytes > property.sharedMemPerBlock) {
      throw std::runtime_error("per-CTA static shared-memory requirement exceeds device limit");
    }
    if (scalar_attributes.sharedSizeBytes != warp_attributes.sharedSizeBytes) {
      throw std::runtime_error("scalar and warp static shared-memory footprints are not identical");
    }
    const LaunchState launch{};
    DeviceBuffers timing_buffers;
    std::vector<SeedCorrectness> correctness;
    correctness.reserve(2);
    for (const int seed : std::vector<int>{17, kTimingSeed}) {
      correctness.push_back(check_seed(make_input(seed), launch, &timing_buffers));
      require_correct(correctness.back());
    }
    for (int iteration = 0; iteration < kWarmupEach; ++iteration) launch_scalar(launch, timing_buffers, correctness.back().sequence_length);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int iteration = 0; iteration < kWarmupEach; ++iteration) launch_warp(launch, timing_buffers, correctness.back().sequence_length);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr; cudaEvent_t end = nullptr;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> scalar_ab; std::vector<float> warp_ab; std::vector<float> warp_ba; std::vector<float> scalar_ba;
    scalar_ab.reserve(kAbbapairs); warp_ab.reserve(kAbbapairs); warp_ba.reserve(kAbbapairs); scalar_ba.reserve(kAbbapairs);
    try {
      for (int pair = 0; pair < kAbbapairs; ++pair) {
        scalar_ab.push_back(time_scalar_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
        warp_ab.push_back(time_warp_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
        warp_ba.push_back(time_warp_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
        scalar_ba.push_back(time_scalar_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
      }
    } catch (...) { cudaEventDestroy(end); cudaEventDestroy(start); throw; }
    CUDA_CHECK(cudaEventDestroy(end)); CUDA_CHECK(cudaEventDestroy(start));
    if (scalar_ab.size() != kAbbapairs || warp_ab.size() != kAbbapairs || warp_ba.size() != kAbbapairs || scalar_ba.size() != kAbbapairs) {
      throw std::runtime_error("ABBA sample accounting mismatch");
    }
    const PostTimingCorrectness post = revalidate_after_timing(timing_buffers);
    print_success_json(property, scalar_attributes, warp_attributes, runtime_version, driver_version, cluster_launch,
                       correctness, post, scalar_ab, warp_ab, warp_ba, scalar_ba);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    print_warp_abba_failure_json(error.what());
    return EXIT_FAILURE;
  }
}
