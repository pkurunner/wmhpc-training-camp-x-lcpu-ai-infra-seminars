// Fair C=2 native cluster synchronization AB/BA microbenchmark.
//
// The candidate implementation is deliberately imported from the audited
// correctness smoke so this translation unit uses its real BF16 selected-page,
// causal attention producer, BF16 partial/output, and FP64 independent oracle
// unchanged.  The control below is structurally matched to that kernel except
// for the producer-ready handoff: its data-ready phase is cluster.sync(), while
// the candidate uses two remote-DSM mbarrier release arrivals and a local wait.
//
// Boundary: this measures the synchronization cost signal of a scalar
// correctness prototype.  It is neither a throughput implementation nor a
// vLLM/model/server speedup claim.

#define main c2_cluster_attention_mbarrier_smoke_embedded_main
#include "c2_cluster_attention_mbarrier_smoke.cu"
#undef main

namespace {

constexpr int kWarmupEach = 20;
constexpr int kAbbapairs = 101;
constexpr int kSamplesPerArm = 2 * kAbbapairs;
constexpr int kTimingSeed = 2026;

constexpr const char* kAbbaBoundary =
    "scalar native C=2 correctness-prototype synchronization-cost signal only; "
    "not a production fusion, throughput result, or vLLM/model/server speedup";
constexpr const char* kControlDataReady =
    "cooperative_groups::cluster_group::sync after both producers publish CTA-local BF16 partials";
constexpr const char* kCandidateDataReady =
    "two remote DSM mbarrier.arrive.release.cluster calls followed by rank-2 local "
    "mbarrier.try_wait.parity.acquire.cluster";

// This is a line-for-line structural counterpart of
// cluster_attention_mbarrier_kernel's data plane.  The only protocol change is
// the marked producer-ready region.  The init/final cluster barriers preserve
// the identical CTA-local DSM lifetime contract in both arms.
__global__ void cluster_attention_cluster_sync_kernel(const __nv_bfloat16* query,
                                                       const __nv_bfloat16* key_cache,
                                                       const __nv_bfloat16* value_cache,
                                                       const int* topk_idx,
                                                       const int* block_table,
                                                       int sequence_length,
                                                       __nv_bfloat16* caller_output) {
  __shared__ __nv_bfloat16 local_partial[kGqaGroup * kHeadDim];
  __shared__ float local_lse[kGqaGroup];
  // Keep the same shared-memory layout as the candidate's rank-2 mbarrier.
  // In this arm it is ordinary protocol-state padding rather than a barrier.
  __shared__ __align__(8) volatile std::uint64_t producer_ready_barrier;
  __shared__ int producer_ready;

  const cg::cluster_group cluster = cg::this_cluster();
  const int role = static_cast<int>(cluster.block_rank());
  const int kv_head = static_cast<int>(blockIdx.x / kNumCtas);
  const int thread = static_cast<int>(threadIdx.x);
  const int query_position = sequence_length - 1;

  if (role == 2 && thread == 0) {
    producer_ready_barrier = 0;
    producer_ready = 0;
  }
  __syncthreads();
  cluster.sync();

  if ((role == 0 || role == 1) && thread < kGqaGroup) {
    const int group_head = thread;
    const int query_head = kv_head * kGqaGroup + group_head;
    const __nv_bfloat16* query_row = query + static_cast<std::size_t>(query_head) * kHeadDim;
    float accumulator[kHeadDim];
    for (int dim = 0; dim < kHeadDim; ++dim) {
      accumulator[dim] = 0.0f;
    }

    float max_score = -INFINITY;
    float normalizer = 0.0f;
    const int selected_begin = role * kPagesPerProducer;
    for (int selected = selected_begin; selected < selected_begin + kPagesPerProducer; ++selected) {
      const int logical_page = topk_idx[kv_head * kSelectedPages + selected];
      const int physical_page = block_table[logical_page];
      for (int token = 0; token < kPageSize; ++token) {
        const int key_position = logical_page * kPageSize + token;
        if (key_position <= query_position && key_position < sequence_length) {
          const std::size_t kv_base = cache_offset(physical_page, kv_head, token, 0);
          float score = 0.0f;
          for (int dim = 0; dim < kHeadDim; ++dim) {
            score = fmaf(__bfloat162float(query_row[dim]), __bfloat162float(key_cache[kv_base + dim]), score);
          }
          score *= kScaleLog2e;
          const float next_max = fmaxf(max_score, score);
          const float alpha = isfinite(max_score) ? exp2f(max_score - next_max) : 0.0f;
          const float beta = exp2f(score - next_max);
          for (int dim = 0; dim < kHeadDim; ++dim) {
            accumulator[dim] = accumulator[dim] * alpha + beta * __bfloat162float(value_cache[kv_base + dim]);
          }
          normalizer = normalizer * alpha + beta;
          max_score = next_max;
        }
      }
    }

    if (normalizer > 0.0f) {
      for (int dim = 0; dim < kHeadDim; ++dim) {
        local_partial[group_head * kHeadDim + dim] = __float2bfloat16_rn(accumulator[dim] / normalizer);
      }
      local_lse[group_head] = max_score + log2f(normalizer);
    } else {
      for (int dim = 0; dim < kHeadDim; ++dim) {
        local_partial[group_head * kHeadDim + dim] = __float2bfloat16_rn(0.0f);
      }
      local_lse[group_head] = -INFINITY;
    }
  }

  __syncthreads();
  // CONTROL-ONLY data-ready synchronization: all participating CTAs wait for
  // the producer CTA-local shared partials before rank 2 maps DSM.
  cluster.sync();
  if (role == 2 && thread == 0) {
    // Preserve the same rank-2 readiness-state write as the candidate, with
    // cluster.sync supplying the actual data-ready guarantee in this arm.
    producer_ready = producer_ready_barrier == 0 ? 1 : 0;
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
  __nv_bfloat16* control_output = nullptr;
  __nv_bfloat16* candidate_output = nullptr;

  DeviceBuffers(const DeviceBuffers&) = delete;
  DeviceBuffers& operator=(const DeviceBuffers&) = delete;
  DeviceBuffers() = default;
  ~DeviceBuffers() { release(); }

  void release() noexcept {
    cudaFree(candidate_output);
    cudaFree(control_output);
    cudaFree(block_table);
    cudaFree(topk_idx);
    cudaFree(value_cache);
    cudaFree(key_cache);
    cudaFree(query);
    candidate_output = nullptr;
    control_output = nullptr;
    block_table = nullptr;
    topk_idx = nullptr;
    value_cache = nullptr;
    key_cache = nullptr;
    query = nullptr;
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
  const std::vector<__nv_bfloat16> sentinel_output(
      kOutputElements, __float2bfloat16_rn(kSentinel));
  CUDA_CHECK(cudaMalloc(&buffers->query, input.query.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers->key_cache, input.key_cache.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers->value_cache, input.value_cache.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers->topk_idx, input.topk_idx.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&buffers->block_table, input.block_table.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&buffers->control_output, output_bytes));
  CUDA_CHECK(cudaMalloc(&buffers->candidate_output, output_bytes));
  CUDA_CHECK(cudaMemcpy(buffers->query, input.query.data(), input.query.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->key_cache, input.key_cache.data(), input.key_cache.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->value_cache, input.value_cache.data(), input.value_cache.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->topk_idx, input.topk_idx.data(), input.topk_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->block_table, input.block_table.data(), input.block_table.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->control_output, sentinel_output.data(), output_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->candidate_output, sentinel_output.data(), output_bytes, cudaMemcpyHostToDevice));
}

void launch_control(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length) {
  CUDA_CHECK(cudaLaunchKernelEx(&launch.config, cluster_attention_cluster_sync_kernel,
                                 buffers.query, buffers.key_cache, buffers.value_cache,
                                 buffers.topk_idx, buffers.block_table, sequence_length, buffers.control_output));
  CUDA_CHECK(cudaGetLastError());
}

void launch_candidate(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length) {
  CUDA_CHECK(cudaLaunchKernelEx(&launch.config, cluster_attention_mbarrier_kernel,
                                 buffers.query, buffers.key_cache, buffers.value_cache,
                                 buffers.topk_idx, buffers.block_table, sequence_length, buffers.candidate_output));
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

ArmCorrectness validate_output(const std::vector<__nv_bfloat16>& output, const std::vector<float>& oracle) {
  if (output.size() != oracle.size()) {
    throw std::runtime_error("output/oracle size mismatch");
  }
  const __nv_bfloat16 sentinel = __float2bfloat16_rn(kSentinel);
  ArmCorrectness result{};
  for (std::size_t index = 0; index < output.size(); ++index) {
    const float actual = __bfloat162float(output[index]);
    const float expected = oracle[index];
    const float abs_error = std::fabs(actual - expected);
    const float rel_error = abs_error / std::max(std::fabs(expected), 1.0e-7f);
    result.max_abs = std::max(result.max_abs, abs_error);
    result.max_rel = std::max(result.max_rel, rel_error);
    result.oracle_finite = result.oracle_finite && std::isfinite(expected);
    result.finite = result.finite && std::isfinite(actual);
    result.sentinel_clean = result.sentinel_clean && !same_bfloat16_bits(output[index], sentinel);
    result.allclose = result.allclose && abs_error <= kAtol + kRtol * std::fabs(expected);
  }
  return result;
}

struct SeedCorrectness {
  int seed = 0;
  int sequence_length = 0;
  bool hierarchy_valid = false;
  int adversarial_unselected_visible_pages = 0;
  int adversarial_masked_tokens = 0;
  ArmCorrectness control{};
  ArmCorrectness candidate{};
  bool cross_arm_bf16_bitwise_equal = false;
};

struct PostTimingCorrectness {
  int seed = 0;
  bool hierarchy_valid = false;
  ArmCorrectness control{};
  ArmCorrectness candidate{};
  bool cross_arm_bf16_bitwise_equal = false;
};

bool bf16_vectors_bitwise_equal(const std::vector<__nv_bfloat16>& lhs,
                                const std::vector<__nv_bfloat16>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    if (!same_bfloat16_bits(lhs[index], rhs[index])) {
      return false;
    }
  }
  return true;
}

SeedCorrectness check_seed(const AttentionInput& input, const LaunchState& launch, DeviceBuffers* timing_buffers) {
  const bool hierarchy_valid = validate_indirection(input);
  if (!hierarchy_valid) {
    throw std::runtime_error("input indirection failed validation before oracle or GPU launch");
  }
  const std::vector<float> oracle = cpu_fp64_accum_oracle(input);
  DeviceBuffers local_buffers;
  DeviceBuffers* buffers = &local_buffers;
  if (input.seed == kTimingSeed) {
    buffers = timing_buffers;
  }
  allocate_and_copy(input, buffers);
  launch_control(launch, *buffers, input.sequence_length);
  launch_candidate(launch, *buffers, input.sequence_length);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> control(kOutputElements);
  std::vector<__nv_bfloat16> candidate(kOutputElements);
  CUDA_CHECK(cudaMemcpy(control.data(), buffers->control_output, control.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(candidate.data(), buffers->candidate_output, candidate.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  return SeedCorrectness{
      input.seed,
      input.sequence_length,
      hierarchy_valid,
      input.adversarial_unselected_visible_pages,
      input.adversarial_masked_tokens,
      validate_output(control, oracle),
      validate_output(candidate, oracle),
      bf16_vectors_bitwise_equal(control, candidate),
  };
}

void require_correct(const SeedCorrectness& result) {
  const auto valid_arm = [](const ArmCorrectness& arm) {
    return arm.oracle_finite && arm.finite && arm.sentinel_clean && arm.allclose;
  };
  if (!result.hierarchy_valid || result.adversarial_unselected_visible_pages <= 0
      || result.adversarial_masked_tokens != kKvHeads * (kPageSize - 1)
      || !valid_arm(result.control) || !valid_arm(result.candidate)
      || !result.cross_arm_bf16_bitwise_equal) {
    throw std::runtime_error("control/candidate correctness gate failed before timing");
  }
}

PostTimingCorrectness revalidate_after_timing(const DeviceBuffers& buffers) {
  // Reconstruct the deterministic adversarial input and FP64 oracle after all
  // 404 timed launches, then validate the final output left by each arm.  This
  // is intentionally a final-state recheck; intermediate timed outputs are not
  // inspected and no stronger all-launch correctness claim is made.
  AttentionInput input = make_input(kTimingSeed);
  const bool hierarchy_valid = validate_indirection(input);
  if (!hierarchy_valid) {
    throw std::runtime_error("post-timing input indirection validation failed");
  }
  const std::vector<float> oracle = cpu_fp64_accum_oracle(input);
  std::vector<__nv_bfloat16> control(kOutputElements);
  std::vector<__nv_bfloat16> candidate(kOutputElements);
  CUDA_CHECK(cudaMemcpy(control.data(), buffers.control_output, control.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(candidate.data(), buffers.candidate_output, candidate.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  const PostTimingCorrectness result{
      input.seed,
      hierarchy_valid,
      validate_output(control, oracle),
      validate_output(candidate, oracle),
      bf16_vectors_bitwise_equal(control, candidate),
  };
  const auto valid_arm = [](const ArmCorrectness& arm) {
    return arm.oracle_finite && arm.finite && arm.sentinel_clean && arm.allclose;
  };
  if (!result.hierarchy_valid || !valid_arm(result.control) || !valid_arm(result.candidate)
      || !result.cross_arm_bf16_bitwise_equal) {
    throw std::runtime_error("post-timing control/candidate correctness gate failed");
  }
  return result;
}

float time_control_once(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length,
                        cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start));
  launch_control(launch, buffers, sequence_length);
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
  return elapsed_ms * 1000.0f;
}

float time_candidate_once(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length,
                          cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start));
  launch_candidate(launch, buffers, sequence_length);
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
  return elapsed_ms * 1000.0f;
}

struct Statistics {
  double p10_us = 0.0;
  double median_us = 0.0;
  double p90_us = 0.0;
};

Statistics summarize_us(std::vector<float> values) {
  if (values.empty()) {
    throw std::runtime_error("cannot summarize empty timing series");
  }
  std::sort(values.begin(), values.end());
  const std::size_t count = values.size();
  const std::size_t p10_index = (10 * count + 99) / 100 - 1;
  const std::size_t p90_index = std::min(count - 1, (90 * count + 99) / 100 - 1);
  const double median = (count % 2 == 1)
      ? static_cast<double>(values[count / 2])
      : (static_cast<double>(values[count / 2 - 1]) + static_cast<double>(values[count / 2])) / 2.0;
  return Statistics{static_cast<double>(values[p10_index]), median, static_cast<double>(values[p90_index])};
}

void print_statistics_json(const Statistics& statistics) {
  std::cout << "{\"p10_us\":" << statistics.p10_us
            << ",\"median_us\":" << statistics.median_us
            << ",\"p90_us\":" << statistics.p90_us << '}';
}

void print_samples_json(const std::vector<float>& samples) {
  std::cout << '[';
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << samples[index];
  }
  std::cout << ']';
}

void print_arm_correctness_json(const ArmCorrectness& arm) {
  std::cout << "{\"max_abs\":" << arm.max_abs << ",\"max_rel\":" << arm.max_rel
            << ",\"oracle_finite\":" << (arm.oracle_finite ? "true" : "false")
            << ",\"finite\":" << (arm.finite ? "true" : "false")
            << ",\"sentinel_clean\":" << (arm.sentinel_clean ? "true" : "false")
            << ",\"allclose\":" << (arm.allclose ? "true" : "false") << '}';
}

void print_seed_json(const SeedCorrectness& result) {
  std::cout << "{\"seed\":" << result.seed << ",\"sequence_length\":" << result.sequence_length
            << ",\"hierarchy_valid\":" << (result.hierarchy_valid ? "true" : "false")
            << ",\"adversarial_unselected_visible_pages\":" << result.adversarial_unselected_visible_pages
            << ",\"adversarial_masked_tokens\":" << result.adversarial_masked_tokens
            << ",\"control\":";
  print_arm_correctness_json(result.control);
  std::cout << ",\"candidate\":";
  print_arm_correctness_json(result.candidate);
  std::cout << ",\"cross_arm_bf16_bitwise_equal\":"
            << (result.cross_arm_bf16_bitwise_equal ? "true" : "false") << '}';
}

void print_post_timing_json(const PostTimingCorrectness& result) {
  std::cout << "{\"seed\":" << result.seed
            << ",\"hierarchy_valid\":" << (result.hierarchy_valid ? "true" : "false")
            << ",\"control\":";
  print_arm_correctness_json(result.control);
  std::cout << ",\"candidate\":";
  print_arm_correctness_json(result.candidate);
  std::cout << ",\"cross_arm_bf16_bitwise_equal\":"
            << (result.cross_arm_bf16_bitwise_equal ? "true" : "false") << '}';
}

void print_success_json(const cudaDeviceProp& property,
                        const cudaFuncAttributes& control_attributes,
                        const cudaFuncAttributes& candidate_attributes,
                        int runtime_version,
                        int driver_version,
                        int cluster_launch,
                        const std::vector<SeedCorrectness>& correctness,
                        const PostTimingCorrectness& post_timing_correctness,
                        const std::vector<float>& control_ab,
                        const std::vector<float>& candidate_ab,
                        const std::vector<float>& candidate_ba,
                        const std::vector<float>& control_ba) {
  std::vector<float> control_all = control_ab;
  control_all.insert(control_all.end(), control_ba.begin(), control_ba.end());
  std::vector<float> candidate_all = candidate_ab;
  candidate_all.insert(candidate_all.end(), candidate_ba.begin(), candidate_ba.end());
  const Statistics control_all_stats = summarize_us(control_all);
  const Statistics candidate_all_stats = summarize_us(candidate_all);
  const Statistics control_ab_stats = summarize_us(control_ab);
  const Statistics candidate_ab_stats = summarize_us(candidate_ab);
  const Statistics control_ba_stats = summarize_us(control_ba);
  const Statistics candidate_ba_stats = summarize_us(candidate_ba);
  const double speedup = control_all_stats.median_us / candidate_all_stats.median_us;

  std::cout << std::setprecision(9)
            << "{\"schema\":\"c2-cluster-attention-sync-abba-v1\",\"status\":\"pass\","
            << "\"boundary\":\"" << json_escape(kAbbaBoundary) << "\","
            << "\"timing_seed\":" << kTimingSeed << ","
            << "\"shape\":{\"B\":" << kBatch << ",\"Hkv\":" << kKvHeads
            << ",\"Hq\":" << kQueryHeads << ",\"G\":" << kGqaGroup << ",\"D\":" << kHeadDim
            << ",\"page_size\":" << kPageSize << ",\"selected_pages\":" << kSelectedPages
            << ",\"logical_pages\":" << kLogicalPages << "},"
            << "\"cluster_layout\":{\"num_ctas\":" << kNumCtas << ",\"clusters\":" << kKvHeads
            << ",\"selected_pages_per_producer\":" << kPagesPerProducer
            << ",\"threads_per_block\":" << kThreadsPerBlock << "},"
            << "\"input_contract\":{\"input_indirection\":\"topk_idx -> block_table -> physical KV page\","
            << "\"block_table_abi\":\"[B,max_blocks], shared by all KV heads\","
            << "\"adversarial_unselected_visible_pages\":true,\"adversarial_causal_tail\":true,"
            << "\"validated_before_oracle_or_gpu\":true},"
            << "\"fairness_contract\":{\"same_real_selected_causal_attention\":true,"
            << "\"same_launch_shape\":true,\"same_input_device_buffers\":true,"
             << "\"caller_owned_independent_outputs\":true,\"persistent_device_buffers_outside_timing\":true,"
             << "\"single_kernel_launch_per_cuda_event_sample\":true,\"ABBA_interleaved\":true,"
             << "\"initialization_copies_and_oracle_outside_timing\":true,"
             << "\"timed_launch_validation_scope\":\"pre-timing two-seed checks plus post-timing final-state recheck; intermediate timed outputs not inspected\","
             << "\"changed_field\":\"producer-ready synchronization protocol only\"},"
            << "\"synchronization\":{\"control_data_ready\":\"" << json_escape(kControlDataReady) << "\","
            << "\"candidate_data_ready\":\"" << json_escape(kCandidateDataReady) << "\","
            << "\"candidate_mbarrier_expected_arrivals\":" << kMBarrierExpectedArrivals
            << ",\"candidate_mbarrier_wait_parity\":" << kMBarrierInitialParity
            << ",\"candidate_mbarrier_max_polls\":" << kMBarrierMaxPolls
            << ",\"shared_lifetime_sync\":\"cluster.sync in both arms after rank-2 DSM reads\"},"
            << "\"dtype_contract\":{\"producer_partial\":\"bfloat16\",\"caller_output\":\"bfloat16\","
            << "\"oracle_accumulator\":\"float64\","
            << "\"oracle\":\"independent two-pass natural-exp direct selected-page causal attention\","
            << "\"tolerance\":{\"rtol\":" << kRtol << ",\"atol\":" << kAtol << "}},"
            << "\"environment\":{\"device\":\"" << json_escape(property.name) << "\","
            << "\"capability\":[" << property.major << ',' << property.minor << "],"
            << "\"cuda_runtime\":" << runtime_version << ",\"cuda_driver\":" << driver_version
            << ",\"cluster_launch_supported\":" << (cluster_launch != 0 ? "true" : "false") << "},"
             << "\"resource_model\":{\"interpretation\":\"register/local-memory differences are disclosed protocol implementation cost; only static shared bytes are matched\","
             << "\"static_shared_equal\":"
             << (control_attributes.sharedSizeBytes == candidate_attributes.sharedSizeBytes ? "true" : "false")
             << ",\"control\":{\"static_shared_bytes\":" << control_attributes.sharedSizeBytes
             << ",\"num_regs\":" << control_attributes.numRegs
             << ",\"local_bytes\":" << control_attributes.localSizeBytes
             << "},\"candidate\":{\"static_shared_bytes\":" << candidate_attributes.sharedSizeBytes
             << ",\"num_regs\":" << candidate_attributes.numRegs
             << ",\"local_bytes\":" << candidate_attributes.localSizeBytes << "}},"
            << "\"correctness\":[";
  for (std::size_t index = 0; index < correctness.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    print_seed_json(correctness[index]);
  }
  std::cout << "],\"post_timing_correctness\":";
  print_post_timing_json(post_timing_correctness);
  std::cout << ",\"timing\":{\"protocol\":\"warmup_each_then_101_control_candidate_candidate_control_ABBA_pairs\","
            << "\"warmup_each\":" << kWarmupEach << ",\"abba_pairs\":" << kAbbapairs
            << ",\"samples_per_arm\":" << kSamplesPerArm << ",\"raw_samples_us\":{"
            << "\"cluster_sync_control\":{\"AB\":";
  print_samples_json(control_ab);
  std::cout << ",\"BA\":";
  print_samples_json(control_ba);
  std::cout << "},\"remote_dsm_mbarrier_candidate\":{\"AB\":";
  print_samples_json(candidate_ab);
  std::cout << ",\"BA\":";
  print_samples_json(candidate_ba);
  std::cout << "}},\"cluster_sync_control\":{\"all\":";
  print_statistics_json(control_all_stats);
  std::cout << ",\"when_launch_order_is_AB\":";
  print_statistics_json(control_ab_stats);
  std::cout << ",\"when_launch_order_is_BA\":";
  print_statistics_json(control_ba_stats);
  std::cout << "},\"remote_dsm_mbarrier_candidate\":{\"all\":";
  print_statistics_json(candidate_all_stats);
  std::cout << ",\"when_launch_order_is_AB\":";
  print_statistics_json(candidate_ab_stats);
  std::cout << ",\"when_launch_order_is_BA\":";
  print_statistics_json(candidate_ba_stats);
  std::cout << "},\"speedup_control_over_candidate\":" << speedup
            << ",\"strict_10_percent_target_met\":" << (speedup >= 1.10 ? "true" : "false") << "}}" << std::endl;
}

void print_abba_failure_json(const std::string& error) {
  std::cout << "{\"schema\":\"c2-cluster-attention-sync-abba-v1\",\"status\":\"fail\",\"error\":\""
            << json_escape(error) << "\",\"boundary\":\"" << json_escape(kAbbaBoundary) << "\"}" << std::endl;
}

}  // namespace

int main() {
  try {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp property{};
    CUDA_CHECK(cudaGetDeviceProperties(&property, device));
    int cluster_launch = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device));
    int runtime_version = 0;
    int driver_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    cudaFuncAttributes control_attributes{};
    cudaFuncAttributes candidate_attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&control_attributes, cluster_attention_cluster_sync_kernel));
    CUDA_CHECK(cudaFuncGetAttributes(&candidate_attributes, cluster_attention_mbarrier_kernel));
    if (property.major != 10 || property.minor != 3) {
      throw std::runtime_error("requires B300 compute capability 10.3");
    }
    if (cluster_launch == 0) {
      throw std::runtime_error("cudaDevAttrClusterLaunch is false");
    }
    if (control_attributes.sharedSizeBytes > property.sharedMemPerBlock
        || candidate_attributes.sharedSizeBytes > property.sharedMemPerBlock) {
      throw std::runtime_error("per-CTA static shared-memory requirement exceeds the device limit");
    }
    if (control_attributes.sharedSizeBytes != candidate_attributes.sharedSizeBytes) {
      throw std::runtime_error("control and candidate static shared-memory footprints are not matched");
    }

    const LaunchState launch{};
    DeviceBuffers timing_buffers;
    std::vector<SeedCorrectness> correctness;
    correctness.reserve(2);
    for (const int seed : std::vector<int>{17, kTimingSeed}) {
      AttentionInput input = make_input(seed);
      correctness.push_back(check_seed(input, launch, &timing_buffers));
      require_correct(correctness.back());
    }

    for (int iteration = 0; iteration < kWarmupEach; ++iteration) {
      launch_control(launch, timing_buffers, correctness.back().sequence_length);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int iteration = 0; iteration < kWarmupEach; ++iteration) {
      launch_candidate(launch, timing_buffers, correctness.back().sequence_length);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> control_ab;
    std::vector<float> candidate_ab;
    std::vector<float> candidate_ba;
    std::vector<float> control_ba;
    control_ab.reserve(kAbbapairs);
    candidate_ab.reserve(kAbbapairs);
    candidate_ba.reserve(kAbbapairs);
    control_ba.reserve(kAbbapairs);
    try {
      for (int pair = 0; pair < kAbbapairs; ++pair) {
        control_ab.push_back(time_control_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
        candidate_ab.push_back(time_candidate_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
        candidate_ba.push_back(time_candidate_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
        control_ba.push_back(time_control_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
      }
    } catch (...) {
      cudaEventDestroy(end);
      cudaEventDestroy(start);
      throw;
    }
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(start));
    if (control_ab.size() != kAbbapairs || candidate_ab.size() != kAbbapairs
        || candidate_ba.size() != kAbbapairs || control_ba.size() != kAbbapairs) {
      throw std::runtime_error("ABBA sample accounting mismatch");
    }
    const PostTimingCorrectness post_timing_correctness = revalidate_after_timing(timing_buffers);
    print_success_json(property, control_attributes, candidate_attributes, runtime_version, driver_version,
                       cluster_launch, correctness, post_timing_correctness,
                       control_ab, candidate_ab, candidate_ba, control_ba);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    print_abba_failure_json(error.what());
    return EXIT_FAILURE;
  }
}
