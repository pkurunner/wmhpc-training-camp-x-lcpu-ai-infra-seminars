// Fair B=1, C=2 AB/BA benchmark: audited warp producer control versus a
// WMMA BF16 QK producer.  The candidate changes only producer-side QK
// arithmetic.  It deliberately retains the four-CTA cluster, the rank-2
// mbarrier / DSM merge, the output ABI, selected-page indirection, causal
// masking, and the final cluster lifetime barrier.
//
// Boundary: this compares complete warp-producer and WMMA-QK-producer
// implementations.  The candidate adds Q and score shared storage plus CTA
// barriers; it is neither an isolated Tensor-Core instruction measurement nor
// a production, throughput, model, or server result.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda/ptx>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// The official warp-control translation unit has its own main.  Enclosing the
// textual import in a named namespace preserves that source byte-for-byte and
// turns its main into an unused namespaced helper, while exposing its audited
// kernel for the global test main below.  Headers are pre-included globally so
// include guards prevent CUDA/standard-library declarations from being nested.
namespace c2_warp_control_import {
#include "c2_cluster_attention_warp_producer_abba.cu"
}  // namespace c2_warp_control_import
using namespace c2_warp_control_import;

namespace c2_tc_qk_candidate {

namespace wmma = nvcuda::wmma;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kTokensPerTile = kWmmaN;
constexpr int kTokenTilesPerPage = kPageSize / kTokensPerTile;

static_assert(kGqaGroup == kWmmaM, "one WMMA row is one GQA head");
static_assert(kHeadDim % kWmmaK == 0, "QK head dimension must be k16 tiled");
static_assert(kPageSize % kTokensPerTile == 0, "page must be token-tiled");
static_assert(kWarpsPerBlock == kTokenTilesPerPage,
              "each full producer warp must own one contiguous 16-token tile");

constexpr const char* kTcQkBoundary =
    "complete B=1 C=2 warp-producer versus WMMA-QK-producer implementation cost; "
    "candidate adds Q/score shared storage and CTA barriers, so this is not an isolated Tensor-Core "
    "instruction measurement, production fusion, throughput result, or model/server result";
constexpr const char* kTcQkDescription =
    "per selected page: cooperatively cache Q[16][128] in shared; 8 full warps each compute one "
    "contiguous 16-token K tile with BF16 WMMA m16n16k16 / FP32 accumulation; store scores[8][16][16]; "
    "then retain the warp producer's serial-head online-softmax/PV data plane in logical token order";

// Candidate producer.  The only intentional protocol difference from the
// imported control is the producer data plane.  In particular, rank 2 still
// owns the local barrier, receives exactly two remote release arrivals, and
// reads the same producer-local partial/LSE DSM addresses.
__global__ void cluster_attention_mbarrier_warp_producer_tc_qk_kernel(
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
  // The candidate-only shared allocation is intentionally disclosed by the
  // runtime resource record. q_tile has A=row-major Q[head][dim]; scores has
  // eight token tiles, then [head][token-within-tile]. WMMA load/store memory
  // pointers need 256-bit alignment: base objects are 32-byte aligned, and
  // both row strides (128 BF16 = 256 B; 16 FP32 = 64 B) preserve it.
  __shared__ __align__(32) __nv_bfloat16 q_tile[kGqaGroup][kHeadDim];
  __shared__ __align__(32) float score_tiles[kTokenTilesPerPage][kGqaGroup][kTokensPerTile];

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

    // A has logical shape MxK = 16 heads x 128 dimensions, row major.
    // This is loaded once per producer CTA and reused for all selected pages.
    for (int element = thread; element < kGqaGroup * kHeadDim; element += kThreadsPerBlock) {
      const int group_head = element / kHeadDim;
      const int dim = element % kHeadDim;
      const int query_head = kv_head * kGqaGroup + group_head;
      q_tile[group_head][dim] = query[static_cast<std::size_t>(query_head) * kHeadDim + dim];
    }
    __syncthreads();

    // Keep exactly the control arm's 8-warps x 2-serial-head PV ownership.
    // The only values supplied by WMMA are scores, in the same selected-page
    // then token-tile order in which the control consumes scalar dots.
    float acc0[kHeadsPerWarp] = {0.0f, 0.0f};
    float acc1[kHeadsPerWarp] = {0.0f, 0.0f};
    float acc2[kHeadsPerWarp] = {0.0f, 0.0f};
    float acc3[kHeadsPerWarp] = {0.0f, 0.0f};
    float max_score[kHeadsPerWarp] = {-INFINITY, -INFINITY};
    float normalizer[kHeadsPerWarp] = {0.0f, 0.0f};

    for (int selected = selected_begin; selected < selected_begin + kPagesPerProducer; ++selected) {
      const int logical_page = topk_idx[kv_head * kSelectedPages + selected];
      const int physical_page = block_table[logical_page];
      const int token_begin = warp * kTokensPerTile;
      const std::size_t tile_base = cache_offset(physical_page, kv_head, token_begin, 0);

      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                     __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                     __nv_bfloat16, wmma::col_major> b;
      wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c;
      wmma::fill_fragment(c, 0.0f);
#pragma unroll
      for (int k_offset = 0; k_offset < kHeadDim; k_offset += kWmmaK) {
        wmma::load_matrix_sync(a, &q_tile[0][k_offset], kHeadDim);
        // Cache tokens are contiguous D-vectors.  Treat [token][dim] as a
        // logical KxN col-major B with ld=128: each N column is one token.
        wmma::load_matrix_sync(b, key_cache + tile_base + k_offset, kHeadDim);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(&score_tiles[warp][0][0], c, kTokensPerTile, wmma::mem_row_major);
      __syncthreads();

#pragma unroll
      for (int head_in_warp = 0; head_in_warp < kHeadsPerWarp; ++head_in_warp) {
        const int group_head = warp * kHeadsPerWarp + head_in_warp;
        for (int tile = 0; tile < kTokenTilesPerPage; ++tile) {
#pragma unroll
          for (int token_in_tile = 0; token_in_tile < kTokensPerTile; ++token_in_tile) {
            const int token = tile * kTokensPerTile + token_in_tile;
            const int key_position = logical_page * kPageSize + token;
            if (key_position <= query_position && key_position < sequence_length) {
              const std::size_t kv_base = cache_offset(physical_page, kv_head, token, 0);
              const float score = score_tiles[tile][group_head][token_in_tile] * kScaleLog2e;
              const float next_max = fmaxf(max_score[head_in_warp], score);
              const float alpha = isfinite(max_score[head_in_warp])
                  ? exp2f(max_score[head_in_warp] - next_max) : 0.0f;
              const float beta = exp2f(score - next_max);
              acc0[head_in_warp] = acc0[head_in_warp] * alpha + beta * __bfloat162float(value_cache[kv_base + dim0]);
              acc1[head_in_warp] = acc1[head_in_warp] * alpha + beta * __bfloat162float(value_cache[kv_base + dim1]);
              acc2[head_in_warp] = acc2[head_in_warp] * alpha + beta * __bfloat162float(value_cache[kv_base + dim2]);
              acc3[head_in_warp] = acc3[head_in_warp] * alpha + beta * __bfloat162float(value_cache[kv_base + dim3]);
              normalizer[head_in_warp] = normalizer[head_in_warp] * alpha + beta;
              max_score[head_in_warp] = next_max;
            }
          }
        }
      }
      // Do not overwrite score_tiles for the next selected page before every
      // warp has consumed its own two serial heads from this page.
      __syncthreads();
    }

#pragma unroll
    for (int head_in_warp = 0; head_in_warp < kHeadsPerWarp; ++head_in_warp) {
      const int group_head = warp * kHeadsPerWarp + head_in_warp;
      const std::size_t partial_base = static_cast<std::size_t>(group_head) * kHeadDim;
      if (normalizer[head_in_warp] > 0.0f) {
        local_partial[partial_base + dim0] = __float2bfloat16_rn(acc0[head_in_warp] / normalizer[head_in_warp]);
        local_partial[partial_base + dim1] = __float2bfloat16_rn(acc1[head_in_warp] / normalizer[head_in_warp]);
        local_partial[partial_base + dim2] = __float2bfloat16_rn(acc2[head_in_warp] / normalizer[head_in_warp]);
        local_partial[partial_base + dim3] = __float2bfloat16_rn(acc3[head_in_warp] / normalizer[head_in_warp]);
        if (lane == 0) local_lse[group_head] = max_score[head_in_warp] + log2f(normalizer[head_in_warp]);
      } else {
        local_partial[partial_base + dim0] = __float2bfloat16_rn(0.0f);
        local_partial[partial_base + dim1] = __float2bfloat16_rn(0.0f);
        local_partial[partial_base + dim2] = __float2bfloat16_rn(0.0f);
        local_partial[partial_base + dim3] = __float2bfloat16_rn(0.0f);
        if (lane == 0) local_lse[group_head] = -INFINITY;
      }
    }
  }

  __syncthreads();
  if ((role == 0 || role == 1) && thread == 0) {
    std::uint64_t* remote_rank2_barrier = cluster.map_shared_rank(&producer_ready_barrier, 2);
    cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release, cuda::ptx::scope_cluster,
                               cuda::ptx::space_cluster, remote_rank2_barrier);
  }
  if (role == 2 && thread == 0) {
    bool ready = false;
#pragma unroll 1
    for (int poll = 0; poll < kMBarrierMaxPolls; ++poll) {
      if (cuda::ptx::mbarrier_try_wait_parity(cuda::ptx::sem_acquire, cuda::ptx::scope_cluster,
                                               &producer_ready_barrier, kMBarrierInitialParity)) {
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
  cluster.sync();
}

void launch_tc_qk(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length) {
  CUDA_CHECK(cudaLaunchKernelEx(&launch.config, cluster_attention_mbarrier_warp_producer_tc_qk_kernel,
                                 buffers.query, buffers.key_cache, buffers.value_cache, buffers.topk_idx,
                                 buffers.block_table, sequence_length, buffers.warp_output));
  CUDA_CHECK(cudaGetLastError());
}

void launch_warp_control(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length) {
  CUDA_CHECK(cudaLaunchKernelEx(&launch.config, cluster_attention_mbarrier_warp_producer_kernel,
                                 buffers.query, buffers.key_cache, buffers.value_cache, buffers.topk_idx,
                                 buffers.block_table, sequence_length, buffers.scalar_output));
  CUDA_CHECK(cudaGetLastError());
}

struct TcSeedCorrectness {
  int seed = 0;
  int sequence_length = 0;
  bool hierarchy_valid = false;
  int adversarial_unselected_visible_pages = 0;
  int adversarial_masked_tokens = 0;
  ArmCorrectness control{};
  ArmCorrectness tc_qk{};
  CrossArmDiagnosis cross_arm{};
};

struct TcPostTimingCorrectness {
  int seed = 0;
  bool hierarchy_valid = false;
  ArmCorrectness control{};
  ArmCorrectness tc_qk{};
  CrossArmDiagnosis cross_arm{};
};

TcSeedCorrectness check_tc_seed(const AttentionInput& input, const LaunchState& launch, DeviceBuffers* timing_buffers) {
  const bool hierarchy_valid = validate_indirection(input);
  if (!hierarchy_valid) throw std::runtime_error("input indirection failed validation before oracle or GPU launch");
  const std::vector<float> oracle = cpu_fp64_accum_oracle(input);
  DeviceBuffers local_buffers;
  DeviceBuffers* buffers = input.seed == kTimingSeed ? timing_buffers : &local_buffers;
  allocate_and_copy(input, buffers);
  launch_warp_control(launch, *buffers, input.sequence_length);
  launch_tc_qk(launch, *buffers, input.sequence_length);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> control(kOutputElements), tc_qk(kOutputElements);
  CUDA_CHECK(cudaMemcpy(control.data(), buffers->scalar_output, control.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(tc_qk.data(), buffers->warp_output, tc_qk.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  return TcSeedCorrectness{input.seed, input.sequence_length, hierarchy_valid,
                           input.adversarial_unselected_visible_pages, input.adversarial_masked_tokens,
                           validate_output(control, oracle), validate_output(tc_qk, oracle), diagnose_cross_arm(control, tc_qk)};
}

void require_tc_correct(const TcSeedCorrectness& result) {
  if (!result.hierarchy_valid || result.adversarial_unselected_visible_pages <= 0
      || result.adversarial_masked_tokens != kKvHeads * (kPageSize - 1)
      || !correct_arm(result.control) || !correct_arm(result.tc_qk)) {
    throw std::runtime_error("control/TC-QK correctness gate failed before timing");
  }
}

TcPostTimingCorrectness revalidate_tc_after_timing(const LaunchState& launch, DeviceBuffers& buffers) {
  const AttentionInput input = make_input(kTimingSeed);
  const bool hierarchy_valid = validate_indirection(input);
  if (!hierarchy_valid) throw std::runtime_error("post-timing input indirection validation failed");
  const std::vector<float> oracle = cpu_fp64_accum_oracle(input);
  // A copy-only check could accidentally accept an output left by an earlier
  // timed launch.  Reset both caller-owned outputs to the BF16 sentinel and
  // perform one fresh, deliberately untimed launch of each arm before reading
  // them back.  The two independent oracle checks therefore still detect a
  // missing writer or stale output after the ABBA sequence.
  const std::vector<__nv_bfloat16> sentinel_output(kOutputElements, __float2bfloat16_rn(kSentinel));
  CUDA_CHECK(cudaMemcpy(buffers.scalar_output, sentinel_output.data(), sentinel_output.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.warp_output, sentinel_output.data(), sentinel_output.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  launch_warp_control(launch, buffers, input.sequence_length);
  launch_tc_qk(launch, buffers, input.sequence_length);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> control(kOutputElements), tc_qk(kOutputElements);
  CUDA_CHECK(cudaMemcpy(control.data(), buffers.scalar_output, control.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(tc_qk.data(), buffers.warp_output, tc_qk.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  const TcPostTimingCorrectness result{input.seed, hierarchy_valid, validate_output(control, oracle),
                                       validate_output(tc_qk, oracle), diagnose_cross_arm(control, tc_qk)};
  if (!result.hierarchy_valid || !correct_arm(result.control) || !correct_arm(result.tc_qk)) {
    throw std::runtime_error("post-timing control/TC-QK correctness gate failed");
  }
  return result;
}

float time_control_once(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length,
                        cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start)); launch_warp_control(launch, buffers, sequence_length);
  CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
  float milliseconds = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
  return milliseconds * 1000.0f;
}

float time_tc_qk_once(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length,
                      cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start)); launch_tc_qk(launch, buffers, sequence_length);
  CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
  float milliseconds = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
  return milliseconds * 1000.0f;
}

void print_tc_arm_json(const ArmCorrectness& arm) { print_arm_json(arm); }
void print_tc_cross_json(const CrossArmDiagnosis& cross) { print_cross_json(cross); }
void print_tc_seed_json(const TcSeedCorrectness& result) {
  std::cout << "{\"seed\":" << result.seed << ",\"sequence_length\":" << result.sequence_length
            << ",\"hierarchy_valid\":" << (result.hierarchy_valid ? "true" : "false")
            << ",\"adversarial_unselected_visible_pages\":" << result.adversarial_unselected_visible_pages
            << ",\"adversarial_masked_tokens\":" << result.adversarial_masked_tokens << ",\"control\":";
  print_tc_arm_json(result.control); std::cout << ",\"tc_qk\":"; print_tc_arm_json(result.tc_qk);
  std::cout << ",\"cross_arm_diagnostic\":"; print_tc_cross_json(result.cross_arm); std::cout << '}';
}
void print_tc_post_json(const TcPostTimingCorrectness& result) {
  std::cout << "{\"seed\":" << result.seed << ",\"hierarchy_valid\":" << (result.hierarchy_valid ? "true" : "false")
            << ",\"control\":"; print_tc_arm_json(result.control); std::cout << ",\"tc_qk\":";
  print_tc_arm_json(result.tc_qk); std::cout << ",\"cross_arm_diagnostic\":"; print_tc_cross_json(result.cross_arm); std::cout << '}';
}

void print_tc_qk_success_json(const cudaDeviceProp& property, const cudaFuncAttributes& control_attributes,
                              const cudaFuncAttributes& tc_attributes, int runtime_version, int driver_version,
                              int cluster_launch, const std::vector<TcSeedCorrectness>& correctness,
                              const TcPostTimingCorrectness& post, const std::vector<float>& control_ab,
                              const std::vector<float>& tc_ab, const std::vector<float>& tc_ba,
                              const std::vector<float>& control_ba) {
  std::vector<float> control_all = control_ab; control_all.insert(control_all.end(), control_ba.begin(), control_ba.end());
  std::vector<float> tc_all = tc_ab; tc_all.insert(tc_all.end(), tc_ba.begin(), tc_ba.end());
  const Statistics control_all_stats = summarize_us(control_all), tc_all_stats = summarize_us(tc_all);
  const Statistics control_ab_stats = summarize_us(control_ab), tc_ab_stats = summarize_us(tc_ab);
  const Statistics control_ba_stats = summarize_us(control_ba), tc_ba_stats = summarize_us(tc_ba);
  const double speedup = control_all_stats.median_us / tc_all_stats.median_us;
  const double ab_speedup = control_ab_stats.median_us / tc_ab_stats.median_us;
  const double ba_speedup = control_ba_stats.median_us / tc_ba_stats.median_us;
  const bool promotion = speedup >= 1.10 && ab_speedup > 1.05 && ba_speedup > 1.05 && tc_attributes.localSizeBytes == 0;
  std::cout << std::setprecision(9)
            << "{\"schema\":\"c2-cluster-attention-tc-qk-abba-v1\",\"status\":\"pass\",\"boundary\":\""
            << json_escape(kTcQkBoundary) << "\",\"timing_seed\":" << kTimingSeed
            << ",\"shape\":{\"B\":" << kBatch << ",\"Hkv\":" << kKvHeads << ",\"Hq\":" << kQueryHeads
            << ",\"G\":" << kGqaGroup << ",\"D\":" << kHeadDim << ",\"page_size\":" << kPageSize
            << ",\"selected_pages\":" << kSelectedPages << ",\"logical_pages\":" << kLogicalPages << "}"
            << ",\"cluster_layout\":{\"num_ctas\":" << kNumCtas << ",\"clusters\":" << kKvHeads
            << ",\"selected_pages_per_producer\":" << kPagesPerProducer << ",\"threads_per_block\":" << kThreadsPerBlock << "}"
            << ",\"producer_contract\":{\"control\":\"imported audited warp-producer kernel\",\"tc_qk\":\""
            << json_escape(kTcQkDescription) << "\",\"changed_field\":\"rank-0/1 producer QK data plane only\""
            << ",\"same_remote_dsm_mbarrier_protocol\":true,\"same_rank2_merge_output_abi_and_lifetime_sync\":true"
            << ",\"same_launch_shape\":true,\"same_real_selected_causal_attention\":true,\"persistent_device_buffers_outside_timing\":true"
            << ",\"caller_owned_independent_outputs\":true,\"single_kernel_launch_per_cuda_event_sample\":true,\"ABBA_interleaved\":true"
            << ",\"initialization_copies_and_oracle_outside_timing\":true,\"post_timing_fresh_sentinel_reset_and_relaunch\":true"
            << ",\"cross_arm_bitwise\":\"diagnostic only; each arm independently gates against FP64 oracle\""
            << ",\"timed_launch_validation_scope\":\"pre-timing two-seed checks plus post-timing sentinel reset and fresh untimed control/TC-QK relaunch; intermediate timed outputs not inspected\"}"
            << ",\"synchronization\":{\"mbarrier_expected_arrivals\":" << kMBarrierExpectedArrivals
            << ",\"mbarrier_wait_parity\":" << kMBarrierInitialParity << ",\"mbarrier_max_polls\":" << kMBarrierMaxPolls
            << ",\"producer_ready\":\"two remote release-arrivals then rank-2 local acquire parity wait\""
            << ",\"cluster_sync\":\"init plus producer-CTA local partial lifetime only\",\"candidate_cta_barriers\":\"Q-load plus score-produce/consume per selected page\"}"
            << ",\"dtype_contract\":{\"control_producer_partial\":\"bfloat16\",\"tc_qk_producer_partial\":\"bfloat16\",\"caller_output\":\"bfloat16\""
            << ",\"qk\":\"WMMA BF16 m16n16k16 with FP32 accumulator\",\"oracle_accumulator\":\"float64\""
            << ",\"oracle\":\"independent two-pass natural-exp direct selected-page causal attention\",\"tolerance\":{\"rtol\":" << kRtol << ",\"atol\":" << kAtol << "}}"
            << ",\"environment\":{\"device\":\"" << json_escape(property.name) << "\",\"capability\":[" << property.major << ',' << property.minor
            << "],\"cuda_runtime\":" << runtime_version << ",\"cuda_driver\":" << driver_version
            << ",\"cluster_launch_supported\":" << (cluster_launch ? "true" : "false") << "}"
            << ",\"resource_model\":{\"static_shared_equal\":" << (control_attributes.sharedSizeBytes == tc_attributes.sharedSizeBytes ? "true" : "false")
            << ",\"candidate_adds_q_and_score_shared\":true,\"control\":{\"static_shared_bytes\":" << control_attributes.sharedSizeBytes
            << ",\"num_regs\":" << control_attributes.numRegs << ",\"local_bytes\":" << control_attributes.localSizeBytes << "}"
            << ",\"tc_qk\":{\"static_shared_bytes\":" << tc_attributes.sharedSizeBytes << ",\"num_regs\":" << tc_attributes.numRegs
            << ",\"local_bytes\":" << tc_attributes.localSizeBytes << "}}"
            << ",\"correctness\":[";
  for (std::size_t index = 0; index < correctness.size(); ++index) { if (index) std::cout << ','; print_tc_seed_json(correctness[index]); }
  std::cout << "],\"post_timing_correctness\":"; print_tc_post_json(post);
  std::cout << ",\"timing\":{\"protocol\":\"warmup_each_then_101_control_tc_tc_control_ABBA_pairs\",\"warmup_each\":" << kWarmupEach
            << ",\"abba_pairs\":" << kAbbapairs << ",\"samples_per_arm\":" << kSamplesPerArm
            << ",\"raw_samples_us\":{\"control\":{\"AB\":"; print_samples_json(control_ab); std::cout << ",\"BA\":"; print_samples_json(control_ba);
  std::cout << "},\"tc_qk\":{\"AB\":"; print_samples_json(tc_ab); std::cout << ",\"BA\":"; print_samples_json(tc_ba);
  std::cout << "}},\"control\":{\"all\":"; print_statistics_json(control_all_stats); std::cout << ",\"when_launch_order_is_AB\":"; print_statistics_json(control_ab_stats);
  std::cout << ",\"when_launch_order_is_BA\":"; print_statistics_json(control_ba_stats); std::cout << "},\"tc_qk\":{\"all\":"; print_statistics_json(tc_all_stats);
  std::cout << ",\"when_launch_order_is_AB\":"; print_statistics_json(tc_ab_stats); std::cout << ",\"when_launch_order_is_BA\":"; print_statistics_json(tc_ba_stats);
  std::cout << "},\"speedup_control_over_tc_qk\":" << speedup << ",\"speedup_by_partition\":{\"AB\":" << ab_speedup << ",\"BA\":" << ba_speedup
            << "},\"promotion_gate\":{\"combined_control_over_tc_qk_at_least_1_10\":" << (speedup >= 1.10 ? "true" : "false")
            << ",\"AB_control_over_tc_qk_greater_than_1_05\":" << (ab_speedup > 1.05 ? "true" : "false")
            << ",\"BA_control_over_tc_qk_greater_than_1_05\":" << (ba_speedup > 1.05 ? "true" : "false")
            << ",\"tc_qk_local_size_bytes_zero\":" << (tc_attributes.localSizeBytes == 0 ? "true" : "false")
            << ",\"all_correct\":true,\"promoted\":" << (promotion ? "true" : "false") << "}}}" << std::endl;
}

void print_tc_qk_failure_json(const std::string& error) {
  std::cout << "{\"schema\":\"c2-cluster-attention-tc-qk-abba-v1\",\"status\":\"fail\",\"error\":\""
            << json_escape(error) << "\",\"boundary\":\"" << json_escape(kTcQkBoundary) << "\"}" << std::endl;
}

}  // namespace c2_tc_qk_candidate
using namespace c2_tc_qk_candidate;

int main() {
  try {
    int device = 0; CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp property{}; CUDA_CHECK(cudaGetDeviceProperties(&property, device));
    int cluster_launch = 0; CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device));
    int runtime_version = 0, driver_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version)); CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    cudaFuncAttributes control_attributes{}, tc_attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&control_attributes, cluster_attention_mbarrier_warp_producer_kernel));
    CUDA_CHECK(cudaFuncGetAttributes(&tc_attributes, cluster_attention_mbarrier_warp_producer_tc_qk_kernel));
    if (property.major != 10 || property.minor != 3) throw std::runtime_error("requires B300 compute capability 10.3");
    if (cluster_launch == 0) throw std::runtime_error("cudaDevAttrClusterLaunch is false");
    if (control_attributes.sharedSizeBytes > property.sharedMemPerBlock || tc_attributes.sharedSizeBytes > property.sharedMemPerBlock) {
      throw std::runtime_error("per-CTA static shared-memory requirement exceeds device limit");
    }
    if (tc_attributes.localSizeBytes != 0) throw std::runtime_error("TC-QK candidate has local-memory spill");
    const LaunchState launch{};
    DeviceBuffers timing_buffers;
    std::vector<TcSeedCorrectness> correctness; correctness.reserve(2);
    for (const int seed : std::vector<int>{17, kTimingSeed}) {
      correctness.push_back(check_tc_seed(make_input(seed), launch, &timing_buffers));
      require_tc_correct(correctness.back());
    }
    for (int iteration = 0; iteration < kWarmupEach; ++iteration) launch_warp_control(launch, timing_buffers, correctness.back().sequence_length);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int iteration = 0; iteration < kWarmupEach; ++iteration) launch_tc_qk(launch, timing_buffers, correctness.back().sequence_length);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr, end = nullptr; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> control_ab, tc_ab, tc_ba, control_ba;
    control_ab.reserve(kAbbapairs); tc_ab.reserve(kAbbapairs); tc_ba.reserve(kAbbapairs); control_ba.reserve(kAbbapairs);
    try {
      for (int pair = 0; pair < kAbbapairs; ++pair) {
        control_ab.push_back(time_control_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
        tc_ab.push_back(time_tc_qk_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
        tc_ba.push_back(time_tc_qk_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
        control_ba.push_back(time_control_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
      }
    } catch (...) { cudaEventDestroy(end); cudaEventDestroy(start); throw; }
    CUDA_CHECK(cudaEventDestroy(end)); CUDA_CHECK(cudaEventDestroy(start));
    if (control_ab.size() != kAbbapairs || tc_ab.size() != kAbbapairs || tc_ba.size() != kAbbapairs || control_ba.size() != kAbbapairs) {
      throw std::runtime_error("ABBA sample accounting mismatch");
    }
    const TcPostTimingCorrectness post = revalidate_tc_after_timing(launch, timing_buffers);
    print_tc_qk_success_json(property, control_attributes, tc_attributes, runtime_version, driver_version, cluster_launch,
                             correctness, post, control_ab, tc_ab, tc_ba, control_ba);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    print_tc_qk_failure_json(error.what());
    return EXIT_FAILURE;
  }
}
