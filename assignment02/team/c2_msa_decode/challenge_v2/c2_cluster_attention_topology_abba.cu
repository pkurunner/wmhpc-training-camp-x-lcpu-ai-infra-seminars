// Scalar C=2 topology ABBA: 4-CTA (with an idle rank 3) versus real 3-CTA
// clusters.  Both arms use the same cluster.sync data-ready protocol.
//
// Boundary: this measures the complete 4-CTA-versus-3-CTA topology
// implementation cost, including the required block-to-KV-head mapping.  The
// main structural change is removal of the otherwise idle rank 3, but the
// result is not a pure hardware idle-rank cost.  It is not a production fusion,
// throughput result, or vLLM/model/server speedup.

#define main c2_cluster_attention_mbarrier_smoke_embedded_main
#include "c2_cluster_attention_mbarrier_smoke.cu"
#undef main

namespace {

constexpr int kTopology4Ctas = 4;
constexpr int kTopology3Ctas = 3;
constexpr int kWarmupEach = 20;
constexpr int kAbbapairs = 101;
constexpr int kSamplesPerArm = 2 * kAbbapairs;
constexpr int kTimingSeed = 2026;
constexpr const char* kTopologyBoundary =
    "scalar native C=2 complete 4-CTA-versus-3-CTA topology-implementation cost signal, including required block-to-KV-head mapping; not a pure idle-rank hardware cost or a production/model/server speedup";
constexpr const char* kDataReady =
    "cooperative_groups::cluster_group::sync after both producers publish CTA-local BF16 partials";

// The two global entry points below use this identical data plane and differ
// only in ClusterCtas.  At ClusterCtas=4 rank 3 deliberately does no work but
// must participate in all three cluster barriers; ClusterCtas=3 has only the
// two producer ranks and the merge rank.
template <int ClusterCtas>
__device__ __forceinline__ void topology_cluster_sync_body(
    const __nv_bfloat16* query, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, const int* topk_idx,
    const int* block_table, int sequence_length, __nv_bfloat16* caller_output,
    __nv_bfloat16* local_partial, float* local_lse,
    volatile std::uint64_t* protocol_padding, int* producer_ready) {
  const cg::cluster_group cluster = cg::this_cluster();
  const int role = static_cast<int>(cluster.block_rank());
  const int kv_head = static_cast<int>(blockIdx.x / ClusterCtas);
  const int thread = static_cast<int>(threadIdx.x);
  const int query_position = sequence_length - 1;

  if (role == 2 && thread == 0) {
    *protocol_padding = 0;
    *producer_ready = 0;
  }
  __syncthreads();
  // Cluster residency/initialization barrier.  Every rank participates.
  cluster.sync();

  if ((role == 0 || role == 1) && thread < kGqaGroup) {
    const int group_head = thread;
    const int query_head = kv_head * kGqaGroup + group_head;
    const __nv_bfloat16* query_row = query + static_cast<std::size_t>(query_head) * kHeadDim;
    float accumulator[kHeadDim];
    for (int dim = 0; dim < kHeadDim; ++dim) accumulator[dim] = 0.0f;
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
            score = fmaf(__bfloat162float(query_row[dim]),
                         __bfloat162float(key_cache[kv_base + dim]), score);
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
      for (int dim = 0; dim < kHeadDim; ++dim) local_partial[group_head * kHeadDim + dim] = __float2bfloat16_rn(0.0f);
      local_lse[group_head] = -INFINITY;
    }
  }

  __syncthreads();
  // Data-ready handoff for both arms.  This is the topology experiment's
  // controlled protocol: there is no mbarrier in either entry point.
  cluster.sync();
  if (role == 2 && thread == 0) *producer_ready = (*protocol_padding == 0) ? 1 : 0;
  __syncthreads();

  if (role == 2 && thread < kGqaGroup) {
    const std::size_t output_base = static_cast<std::size_t>(kv_head * kGqaGroup + thread) * kHeadDim;
    if (*producer_ready == 0) {
      for (int dim = 0; dim < kHeadDim; ++dim) caller_output[output_base + dim] = __float2bfloat16_rn(kSentinel);
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
        const float merged = denominator > 0.0f ? (partial0 * weight0 + partial1 * weight1) / denominator : 0.0f;
        caller_output[output_base + dim] = __float2bfloat16_rn(merged);
      }
    }
  }
  // Protect producer CTA-local shared storage through rank-2 DSM reads.
  cluster.sync();
}

extern "C" __global__ void cluster_attention_topology4_cluster_sync_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, const int* topk_idx,
    const int* block_table, int sequence_length, __nv_bfloat16* caller_output) {
  __shared__ __nv_bfloat16 local_partial[kGqaGroup * kHeadDim];
  __shared__ float local_lse[kGqaGroup];
  __shared__ __align__(8) volatile std::uint64_t protocol_padding;
  __shared__ int producer_ready;
  topology_cluster_sync_body<kTopology4Ctas>(query, key_cache, value_cache, topk_idx, block_table,
                                              sequence_length, caller_output, local_partial, local_lse,
                                              &protocol_padding, &producer_ready);
}

extern "C" __global__ void cluster_attention_topology3_cluster_sync_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, const int* topk_idx,
    const int* block_table, int sequence_length, __nv_bfloat16* caller_output) {
  __shared__ __nv_bfloat16 local_partial[kGqaGroup * kHeadDim];
  __shared__ float local_lse[kGqaGroup];
  __shared__ __align__(8) volatile std::uint64_t protocol_padding;
  __shared__ int producer_ready;
  topology_cluster_sync_body<kTopology3Ctas>(query, key_cache, value_cache, topk_idx, block_table,
                                              sequence_length, caller_output, local_partial, local_lse,
                                              &protocol_padding, &producer_ready);
}

struct DeviceBuffers {
  __nv_bfloat16 *query = nullptr, *key_cache = nullptr, *value_cache = nullptr;
  int *topk_idx = nullptr, *block_table = nullptr;
  __nv_bfloat16 *topology4_output = nullptr, *topology3_output = nullptr;
  DeviceBuffers(const DeviceBuffers&) = delete;
  DeviceBuffers& operator=(const DeviceBuffers&) = delete;
  DeviceBuffers() = default;
  ~DeviceBuffers() { release(); }
  void release() noexcept {
    cudaFree(topology3_output); cudaFree(topology4_output); cudaFree(block_table); cudaFree(topk_idx);
    cudaFree(value_cache); cudaFree(key_cache); cudaFree(query);
    topology3_output = topology4_output = query = key_cache = value_cache = nullptr;
    topk_idx = block_table = nullptr;
  }
};

struct LaunchState {
  cudaLaunchAttribute attribute{};
  cudaLaunchConfig_t config{};
  explicit LaunchState(int cluster_ctas) {
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {static_cast<unsigned int>(cluster_ctas), 1U, 1U};
    config.gridDim = dim3(kKvHeads * cluster_ctas, 1, 1);
    config.blockDim = dim3(kThreadsPerBlock, 1, 1);
    config.dynamicSmemBytes = 0; config.stream = nullptr; config.attrs = &attribute; config.numAttrs = 1;
  }
};

void allocate_and_copy(const AttentionInput& input, DeviceBuffers* b) {
  const std::size_t output_bytes = kOutputElements * sizeof(__nv_bfloat16);
  const std::vector<__nv_bfloat16> sentinel(kOutputElements, __float2bfloat16_rn(kSentinel));
  CUDA_CHECK(cudaMalloc(&b->query, input.query.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&b->key_cache, input.key_cache.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&b->value_cache, input.value_cache.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&b->topk_idx, input.topk_idx.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&b->block_table, input.block_table.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&b->topology4_output, output_bytes)); CUDA_CHECK(cudaMalloc(&b->topology3_output, output_bytes));
  CUDA_CHECK(cudaMemcpy(b->query, input.query.data(), input.query.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b->key_cache, input.key_cache.data(), input.key_cache.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b->value_cache, input.value_cache.data(), input.value_cache.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b->topk_idx, input.topk_idx.data(), input.topk_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b->block_table, input.block_table.data(), input.block_table.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b->topology4_output, sentinel.data(), output_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b->topology3_output, sentinel.data(), output_bytes, cudaMemcpyHostToDevice));
}

void launch_topology4(const LaunchState& l, const DeviceBuffers& b, int n) {
  CUDA_CHECK(cudaLaunchKernelEx(&l.config, cluster_attention_topology4_cluster_sync_kernel,
      b.query, b.key_cache, b.value_cache, b.topk_idx, b.block_table, n, b.topology4_output)); CUDA_CHECK(cudaGetLastError());
}
void launch_topology3(const LaunchState& l, const DeviceBuffers& b, int n) {
  CUDA_CHECK(cudaLaunchKernelEx(&l.config, cluster_attention_topology3_cluster_sync_kernel,
      b.query, b.key_cache, b.value_cache, b.topk_idx, b.block_table, n, b.topology3_output)); CUDA_CHECK(cudaGetLastError());
}

struct ArmCorrectness { float max_abs = 0.0f, max_rel = 0.0f; bool oracle_finite = true, finite = true, sentinel_clean = true, allclose = true; };
ArmCorrectness validate_output(const std::vector<__nv_bfloat16>& output, const std::vector<float>& oracle) {
  if (output.size() != oracle.size()) throw std::runtime_error("output/oracle size mismatch");
  const __nv_bfloat16 sentinel = __float2bfloat16_rn(kSentinel); ArmCorrectness r{};
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float actual = __bfloat162float(output[i]), expected = oracle[i];
    const float abs_error = std::fabs(actual - expected), rel_error = abs_error / std::max(std::fabs(expected), 1.0e-7f);
    r.max_abs = std::max(r.max_abs, abs_error); r.max_rel = std::max(r.max_rel, rel_error);
    r.oracle_finite = r.oracle_finite && std::isfinite(expected); r.finite = r.finite && std::isfinite(actual);
    r.sentinel_clean = r.sentinel_clean && !same_bfloat16_bits(output[i], sentinel);
    r.allclose = r.allclose && abs_error <= kAtol + kRtol * std::fabs(expected);
  }
  return r;
}
bool bf16_vectors_bitwise_equal(const std::vector<__nv_bfloat16>& a, const std::vector<__nv_bfloat16>& b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), same_bfloat16_bits);
}
struct SeedCorrectness { int seed = 0, sequence_length = 0; bool hierarchy_valid = false; int adversarial_unselected_visible_pages = 0, adversarial_masked_tokens = 0; ArmCorrectness topology4{}, topology3{}; bool cross_arm_bf16_bitwise_equal = false; };
struct PostTimingCorrectness { int seed = 0; bool hierarchy_valid = false; ArmCorrectness topology4{}, topology3{}; bool cross_arm_bf16_bitwise_equal = false; };

SeedCorrectness check_seed(const AttentionInput& input, const LaunchState& topology4_launch,
                           const LaunchState& topology3_launch, DeviceBuffers* timing_buffers) {
  const bool hierarchy_valid = validate_indirection(input);
  if (!hierarchy_valid) throw std::runtime_error("input indirection failed validation before oracle or GPU launch");
  const std::vector<float> oracle = cpu_fp64_accum_oracle(input);
  DeviceBuffers local; DeviceBuffers* buffers = input.seed == kTimingSeed ? timing_buffers : &local;
  allocate_and_copy(input, buffers);
  launch_topology4(topology4_launch, *buffers, input.sequence_length); launch_topology3(topology3_launch, *buffers, input.sequence_length);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> topology4(kOutputElements), topology3(kOutputElements);
  CUDA_CHECK(cudaMemcpy(topology4.data(), buffers->topology4_output, topology4.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(topology3.data(), buffers->topology3_output, topology3.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  return {input.seed, input.sequence_length, hierarchy_valid, input.adversarial_unselected_visible_pages,
          input.adversarial_masked_tokens, validate_output(topology4, oracle), validate_output(topology3, oracle),
          bf16_vectors_bitwise_equal(topology4, topology3)};
}
bool valid_arm(const ArmCorrectness& a) { return a.oracle_finite && a.finite && a.sentinel_clean && a.allclose; }
void require_correct(const SeedCorrectness& r) {
  if (!r.hierarchy_valid || r.adversarial_unselected_visible_pages <= 0 || r.adversarial_masked_tokens != kKvHeads * (kPageSize - 1)
      || !valid_arm(r.topology4) || !valid_arm(r.topology3) || !r.cross_arm_bf16_bitwise_equal) throw std::runtime_error("topology correctness gate failed before timing");
}
PostTimingCorrectness revalidate_after_timing(const DeviceBuffers& buffers) {
  const AttentionInput input = make_input(kTimingSeed);
  if (!validate_indirection(input)) throw std::runtime_error("post-timing indirection validation failed");
  const std::vector<float> oracle = cpu_fp64_accum_oracle(input);
  std::vector<__nv_bfloat16> topology4(kOutputElements), topology3(kOutputElements);
  CUDA_CHECK(cudaMemcpy(topology4.data(), buffers.topology4_output, topology4.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(topology3.data(), buffers.topology3_output, topology3.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  PostTimingCorrectness r{kTimingSeed, true, validate_output(topology4, oracle), validate_output(topology3, oracle), bf16_vectors_bitwise_equal(topology4, topology3)};
  if (!valid_arm(r.topology4) || !valid_arm(r.topology3) || !r.cross_arm_bf16_bitwise_equal) throw std::runtime_error("post-timing topology correctness gate failed");
  return r;
}

float time_topology4_once(const LaunchState& l, const DeviceBuffers& b, int n, cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start)); launch_topology4(l, b, n); CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end)); float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, end)); return ms * 1000.0f;
}
float time_topology3_once(const LaunchState& l, const DeviceBuffers& b, int n, cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start)); launch_topology3(l, b, n); CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end)); float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start, end)); return ms * 1000.0f;
}
struct Statistics { double p10_us = 0.0, median_us = 0.0, p90_us = 0.0; };
Statistics summarize_us(std::vector<float> v) {
  if (v.empty()) throw std::runtime_error("cannot summarize empty timing series"); std::sort(v.begin(), v.end()); const std::size_t n = v.size();
  return {static_cast<double>(v[(10 * n + 99) / 100 - 1]), n % 2 ? static_cast<double>(v[n / 2]) : (static_cast<double>(v[n / 2 - 1]) + v[n / 2]) / 2.0, static_cast<double>(v[std::min(n - 1, (90 * n + 99) / 100 - 1)])};
}
void print_stats(const Statistics& s) { std::cout << "{\"p10_us\":" << s.p10_us << ",\"median_us\":" << s.median_us << ",\"p90_us\":" << s.p90_us << '}'; }
void print_samples(const std::vector<float>& s) { std::cout << '['; for (std::size_t i = 0; i < s.size(); ++i) { if (i) std::cout << ','; std::cout << s[i]; } std::cout << ']'; }
void print_arm(const ArmCorrectness& a) { std::cout << "{\"max_abs\":" << a.max_abs << ",\"max_rel\":" << a.max_rel << ",\"oracle_finite\":" << (a.oracle_finite ? "true" : "false") << ",\"finite\":" << (a.finite ? "true" : "false") << ",\"sentinel_clean\":" << (a.sentinel_clean ? "true" : "false") << ",\"allclose\":" << (a.allclose ? "true" : "false") << '}'; }
void print_seed(const SeedCorrectness& r) { std::cout << "{\"seed\":" << r.seed << ",\"sequence_length\":" << r.sequence_length << ",\"hierarchy_valid\":" << (r.hierarchy_valid ? "true" : "false") << ",\"adversarial_unselected_visible_pages\":" << r.adversarial_unselected_visible_pages << ",\"adversarial_masked_tokens\":" << r.adversarial_masked_tokens << ",\"topology4\":"; print_arm(r.topology4); std::cout << ",\"topology3\":"; print_arm(r.topology3); std::cout << ",\"cross_arm_bf16_bitwise_equal\":" << (r.cross_arm_bf16_bitwise_equal ? "true" : "false") << '}'; }
void print_post(const PostTimingCorrectness& r) { std::cout << "{\"seed\":" << r.seed << ",\"hierarchy_valid\":" << (r.hierarchy_valid ? "true" : "false") << ",\"topology4\":"; print_arm(r.topology4); std::cout << ",\"topology3\":"; print_arm(r.topology3); std::cout << ",\"cross_arm_bf16_bitwise_equal\":" << (r.cross_arm_bf16_bitwise_equal ? "true" : "false") << '}'; }

void print_success(const cudaDeviceProp& p, const cudaFuncAttributes& a4, const cudaFuncAttributes& a3,
                   int runtime, int driver, int cluster_launch, const std::vector<SeedCorrectness>& correctness,
                   const PostTimingCorrectness& post, const std::vector<float>& t4ab, const std::vector<float>& t3ab,
                   const std::vector<float>& t3ba, const std::vector<float>& t4ba) {
  std::vector<float> t4 = t4ab; t4.insert(t4.end(), t4ba.begin(), t4ba.end()); std::vector<float> t3 = t3ab; t3.insert(t3.end(), t3ba.begin(), t3ba.end());
  const Statistics t4all = summarize_us(t4), t3all = summarize_us(t3), t4abs = summarize_us(t4ab), t3abs = summarize_us(t3ab), t4bas = summarize_us(t4ba), t3bas = summarize_us(t3ba);
  const double s_topo = t4all.median_us / t3all.median_us;
  const bool promoted = s_topo >= 1.05 && t4abs.median_us / t3abs.median_us > 1.0 && t4bas.median_us / t3bas.median_us > 1.0;
  std::cout << std::setprecision(9) << "{\"schema\":\"c2-cluster-attention-topology-abba-v1\",\"status\":\"pass\",\"boundary\":\"" << json_escape(kTopologyBoundary) << "\",\"timing_seed\":" << kTimingSeed
            << ",\"shape\":{\"B\":" << kBatch << ",\"Hkv\":" << kKvHeads << ",\"Hq\":" << kQueryHeads << ",\"G\":" << kGqaGroup << ",\"D\":" << kHeadDim << ",\"page_size\":" << kPageSize << ",\"selected_pages\":" << kSelectedPages << ",\"logical_pages\":" << kLogicalPages << "}"
            << ",\"cluster_layout\":{\"topology4\":{\"ctas_per_cluster\":4,\"clusters\":" << kKvHeads << ",\"idle_rank\":3,\"grid_ctas\":" << kKvHeads * kTopology4Ctas << "},\"topology3\":{\"ctas_per_cluster\":3,\"clusters\":" << kKvHeads << ",\"roles\":\"rank0/rank1 producers plus rank2 merge only\",\"grid_ctas\":" << kKvHeads * kTopology3Ctas << "},\"selected_pages_per_producer\":" << kPagesPerProducer << ",\"threads_per_block\":" << kThreadsPerBlock << "}"
            << ",\"input_contract\":{\"input_indirection\":\"topk_idx -> block_table -> physical KV page\",\"block_table_abi\":\"[B,max_blocks], shared by all KV heads\",\"adversarial_unselected_visible_pages\":true,\"adversarial_causal_tail\":true,\"validated_before_oracle_or_gpu\":true}"
             << ",\"fairness_contract\":{\"same_real_selected_causal_attention\":true,\"same_cluster_sync_data_ready_protocol\":true,\"same_input_device_buffers\":true,\"caller_owned_independent_outputs\":true,\"persistent_device_buffers_outside_timing\":true,\"single_kernel_launch_per_cuda_event_sample\":true,\"ABBA_interleaved\":true,\"initialization_copies_and_oracle_outside_timing\":true,\"timed_launch_validation_scope\":\"pre-timing two-seed checks plus post-timing final-state recheck; intermediate timed outputs not inspected\",\"changed_field\":\"complete cluster topology implementation: 4 CTA including idle rank 3 versus real 3 CTA\",\"attribution_limit\":\"ClusterCtas changes both clusterDim/grid and required blockIdx-to-KV-head mapping; result is not a pure idle-rank hardware cost\"}"
            << ",\"synchronization\":{\"data_ready\":\"" << json_escape(kDataReady) << "\",\"initialization_sync\":\"cluster.sync in both arms\",\"shared_lifetime_sync\":\"cluster.sync in both arms after rank-2 DSM reads\",\"mbarrier_used_by_tested_symbols\":false}"
            << ",\"dtype_contract\":{\"producer_partial\":\"bfloat16\",\"caller_output\":\"bfloat16\",\"oracle_accumulator\":\"float64\",\"oracle\":\"independent two-pass natural-exp direct selected-page causal attention\",\"tolerance\":{\"rtol\":" << kRtol << ",\"atol\":" << kAtol << "}}"
            << ",\"environment\":{\"device\":\"" << json_escape(p.name) << "\",\"capability\":[" << p.major << ',' << p.minor << "],\"cuda_runtime\":" << runtime << ",\"cuda_driver\":" << driver << ",\"cluster_launch_supported\":" << (cluster_launch ? "true" : "false") << "}"
            << ",\"resource_model\":{\"interpretation\":\"register/local-memory differences are disclosed topology implementation cost; only static shared bytes are matched\",\"static_shared_equal\":" << (a4.sharedSizeBytes == a3.sharedSizeBytes ? "true" : "false") << ",\"topology4\":{\"static_shared_bytes\":" << a4.sharedSizeBytes << ",\"num_regs\":" << a4.numRegs << ",\"local_bytes\":" << a4.localSizeBytes << "},\"topology3\":{\"static_shared_bytes\":" << a3.sharedSizeBytes << ",\"num_regs\":" << a3.numRegs << ",\"local_bytes\":" << a3.localSizeBytes << "}}"
            << ",\"correctness\":[";
  for (std::size_t i = 0; i < correctness.size(); ++i) { if (i) std::cout << ','; print_seed(correctness[i]); }
  std::cout << "],\"post_timing_correctness\":"; print_post(post);
  std::cout << ",\"timing\":{\"protocol\":\"warmup_each_then_101_topology4_topology3_topology3_topology4_ABBA_pairs\",\"warmup_each\":" << kWarmupEach << ",\"abba_pairs\":" << kAbbapairs << ",\"samples_per_arm\":" << kSamplesPerArm << ",\"raw_samples_us\":{\"topology4\":{\"AB\":"; print_samples(t4ab); std::cout << ",\"BA\":"; print_samples(t4ba); std::cout << "},\"topology3\":{\"AB\":"; print_samples(t3ab); std::cout << ",\"BA\":"; print_samples(t3ba); std::cout << "}},\"topology4\":{\"all\":"; print_stats(t4all); std::cout << ",\"when_launch_order_is_AB\":"; print_stats(t4abs); std::cout << ",\"when_launch_order_is_BA\":"; print_stats(t4bas); std::cout << "},\"topology3\":{\"all\":"; print_stats(t3all); std::cout << ",\"when_launch_order_is_AB\":"; print_stats(t3abs); std::cout << ",\"when_launch_order_is_BA\":"; print_stats(t3bas); std::cout << "},\"S_topo\":" << s_topo << ",\"S_topo_t4_over_t3\":" << s_topo << ",\"promotion_gate\":{\"combined_median_threshold\":1.05,\"combined_median_met\":" << (s_topo >= 1.05 ? "true" : "false") << ",\"AB_partition_t4_over_t3\":" << t4abs.median_us / t3abs.median_us << ",\"BA_partition_t4_over_t3\":" << t4bas.median_us / t3bas.median_us << ",\"both_partitions_t4_over_t3_gt_1\":" << (t4abs.median_us / t3abs.median_us > 1.0 && t4bas.median_us / t3bas.median_us > 1.0 ? "true" : "false") << ",\"promote_topology_optimization\":" << (promoted ? "true" : "false") << ",\"otherwise\":\"freeze scalar topology tuning\"}}}" << std::endl;
}
void print_failure(const std::string& e) { std::cout << "{\"schema\":\"c2-cluster-attention-topology-abba-v1\",\"status\":\"fail\",\"error\":\"" << json_escape(e) << "\",\"boundary\":\"" << json_escape(kTopologyBoundary) << "\"}" << std::endl; }

}  // namespace

int main() {
  try {
    int device = 0, cluster_launch = 0, runtime = 0, driver = 0; CUDA_CHECK(cudaGetDevice(&device)); cudaDeviceProp property{}; CUDA_CHECK(cudaGetDeviceProperties(&property, device)); CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device)); CUDA_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_CHECK(cudaDriverGetVersion(&driver));
    cudaFuncAttributes attr4{}, attr3{}; CUDA_CHECK(cudaFuncGetAttributes(&attr4, cluster_attention_topology4_cluster_sync_kernel)); CUDA_CHECK(cudaFuncGetAttributes(&attr3, cluster_attention_topology3_cluster_sync_kernel));
    if (property.major != 10 || property.minor != 3) throw std::runtime_error("requires B300 compute capability 10.3");
    if (!cluster_launch) throw std::runtime_error("cudaDevAttrClusterLaunch is false");
    if (attr4.sharedSizeBytes > property.sharedMemPerBlock || attr3.sharedSizeBytes > property.sharedMemPerBlock) throw std::runtime_error("per-CTA static shared-memory requirement exceeds device limit");
    if (attr4.sharedSizeBytes != attr3.sharedSizeBytes) throw std::runtime_error("topology arms static shared-memory footprints differ");
    const LaunchState topology4_launch(kTopology4Ctas), topology3_launch(kTopology3Ctas); DeviceBuffers timing_buffers; std::vector<SeedCorrectness> correctness;
    for (const int seed : std::vector<int>{17, kTimingSeed}) { AttentionInput input = make_input(seed); correctness.push_back(check_seed(input, topology4_launch, topology3_launch, &timing_buffers)); require_correct(correctness.back()); }
    for (int i = 0; i < kWarmupEach; ++i) launch_topology4(topology4_launch, timing_buffers, correctness.back().sequence_length); CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < kWarmupEach; ++i) launch_topology3(topology3_launch, timing_buffers, correctness.back().sequence_length); CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr, end = nullptr; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&end)); std::vector<float> t4ab, t3ab, t3ba, t4ba; t4ab.reserve(kAbbapairs); t3ab.reserve(kAbbapairs); t3ba.reserve(kAbbapairs); t4ba.reserve(kAbbapairs);
    try { for (int i = 0; i < kAbbapairs; ++i) { t4ab.push_back(time_topology4_once(topology4_launch, timing_buffers, correctness.back().sequence_length, start, end)); t3ab.push_back(time_topology3_once(topology3_launch, timing_buffers, correctness.back().sequence_length, start, end)); t3ba.push_back(time_topology3_once(topology3_launch, timing_buffers, correctness.back().sequence_length, start, end)); t4ba.push_back(time_topology4_once(topology4_launch, timing_buffers, correctness.back().sequence_length, start, end)); } } catch (...) { cudaEventDestroy(end); cudaEventDestroy(start); throw; }
    CUDA_CHECK(cudaEventDestroy(end)); CUDA_CHECK(cudaEventDestroy(start)); if (t4ab.size() != kAbbapairs || t3ab.size() != kAbbapairs || t3ba.size() != kAbbapairs || t4ba.size() != kAbbapairs) throw std::runtime_error("ABBA sample accounting mismatch");
    const PostTimingCorrectness post = revalidate_after_timing(timing_buffers); print_success(property, attr4, attr3, runtime, driver, cluster_launch, correctness, post, t4ab, t3ab, t3ba, t4ba); return EXIT_SUCCESS;
  } catch (const std::exception& e) { print_failure(e.what()); return EXIT_FAILURE; }
}
