// Strict B=1, C=2 AB/BA feasibility experiment for a fully Tensor-Core
// producer: the control is the reviewed WMMA-QK / warp-PV kernel from the
// TC-QK qualification, while this candidate keeps its QK, cluster shape,
// rank-2 DSM/mbarrier merge, selected-page indirection, causal order, output
// ABI, and lifetime barrier and replaces producer-side PV with BF16 WMMA.
//
// Boundary: this measures the complete QK+PV implementation cost, including
// the candidate's shared state and CTA barriers.  It is neither an isolated
// instruction benchmark nor a production, throughput, model, or server claim.

// The reviewed TC-QK program owns a global main and its nested audited warp
// import deliberately manages the `main` macro itself.  Enclose the complete
// program in a named namespace instead of trying to redefine that macro: the
// original global main becomes an unused namespaced helper, while the exact
// B=1/C=2 QK control source remains byte-for-byte imported.  Headers must be
// pre-included globally so their guarded declarations never enter this scope.
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

namespace c2_tc_qk_control_import {
#include "c2_cluster_attention_tc_qk_abba.cu"
}  // namespace c2_tc_qk_control_import
using namespace c2_tc_qk_control_import;
namespace tcqk_control = c2_tc_qk_control_import::c2_tc_qk_candidate;

namespace c2_tc_qk_pv_candidate {

namespace wmma = nvcuda::wmma;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kTokensPerTile = 16;
constexpr int kTokenTilesPerPage = kPageSize / kTokensPerTile;
constexpr int kDimsPerPvWarp = 16;

static_assert(kGqaGroup == kWmmaM, "one QK/PV WMMA row or column is one GQA head");
static_assert(kHeadDim % kWmmaK == 0 && kHeadDim / kDimsPerPvWarp == kWarpsPerBlock,
              "the eight warps must cover all 128 value dimensions in 16-wide tiles");
static_assert(kPageSize % kTokensPerTile == 0 && kTokenTilesPerPage == kWarpsPerBlock,
              "the existing QK producer and PV token tiles must agree");
static_assert(kWmmaM == kWmmaN && kWmmaN == kWmmaK,
              "this feasibility experiment uses only 16x16x16 BF16 WMMA tiles");

constexpr const char* kTcQkPvBoundary =
    "complete B=1 C=2 WMMA-QK/warp-PV control versus WMMA-QK+WMMA-PV candidate implementation cost; "
    "candidate shared state and CTA barriers are included; not an isolated Tensor-Core, production, throughput, "
    "model, or server result";
constexpr const char* kTcQkPvDescription =
    "for each selected page and logical 16-token tile: BF16 WMMA QK scores; per-head online m/z update and "
    "BF16 weights; eight warps compute V^T[16,16] times W^T[16,16] with BF16 WMMA/FP32 accumulation; shared "
    "N[16,128] is updated as alpha*N+C; entirely masked tiles leave m/z/N unchanged";

// Control is tcqk_control::cluster_attention_mbarrier_warp_producer_tc_qk_kernel,
// imported above.  This kernel deliberately has the same input/output ABI and
// cluster protocol.  The only data-plane change is the PV stage after the
// materialized QK scores.  All state below is CTA-local and timed.
__global__ void cluster_attention_mbarrier_warp_producer_tc_qk_pv_kernel(
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

  // QK materialization is byte-for-byte layout-compatible with the TC-QK
  // control: Q is logical A[head,dim], while cache [token,dim] is logical
  // col-major B[dim,token].  Alignment/strides preserve the WMMA 256-bit rule.
  __shared__ __align__(32) __nv_bfloat16 q_tile[kGqaGroup][kHeadDim];
  __shared__ __align__(32) float score_tiles[kTokenTilesPerPage][kGqaGroup][kTokensPerTile];

  // Online softmax/PV state.  We retain BF16 weights intentionally: the same
  // rounded weights feed the normalizer and W^T, so no scalar residual or
  // hidden correction is present.  N is FP32 and is the only numerator state.
  __shared__ __align__(32) float numerator[kGqaGroup][kHeadDim];
  __shared__ float running_max[kGqaGroup];
  __shared__ float normalizer[kGqaGroup];
  __shared__ float alpha_tile[kGqaGroup];
  __shared__ int tile_active[kGqaGroup];
  __shared__ __align__(32) __nv_bfloat16 bf16_weights[kGqaGroup][kTokensPerTile];
  // One WMMA PV C tile per warp: rows are its 16 dimensions, columns are heads.
  __shared__ __align__(32) float pv_contribution[kWarpsPerBlock][kDimsPerPvWarp][kGqaGroup];

  const cg::cluster_group cluster = cg::this_cluster();
  const int role = static_cast<int>(cluster.block_rank());
  const int kv_head = static_cast<int>(blockIdx.x / kNumCtas);
  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread / kWarpSize;
  const int lane = thread & (kWarpSize - 1);
  const int query_position = sequence_length - 1;

  if (role == 2 && thread == 0) {
    cuda::ptx::mbarrier_init(&producer_ready_barrier, kMBarrierExpectedArrivals);
    producer_ready = 0;
  }
  __syncthreads();
  cluster.sync();

  if (role == 0 || role == 1) {
    const int selected_begin = role * kPagesPerProducer;
    // Cooperative Q cache and complete online state initialization.  These
    // writes are CTA-local and included in each candidate launch.
    for (int element = thread; element < kGqaGroup * kHeadDim; element += kThreadsPerBlock) {
      const int group_head = element / kHeadDim;
      const int dim = element % kHeadDim;
      q_tile[group_head][dim] = query[(kv_head * kGqaGroup + group_head) * kHeadDim + dim];
      numerator[group_head][dim] = 0.0f;
    }
    if (thread < kGqaGroup) {
      running_max[thread] = -INFINITY;
      normalizer[thread] = 0.0f;
      alpha_tile[thread] = 1.0f;
      tile_active[thread] = 0;
    }
    __syncthreads();

    for (int selected = selected_begin; selected < selected_begin + kPagesPerProducer; ++selected) {
      const int logical_page = topk_idx[kv_head * kSelectedPages + selected];
      const int physical_page = block_table[logical_page];
      const int token_begin = warp * kTokensPerTile;
      const std::size_t qk_base = cache_offset(physical_page, kv_head, token_begin, 0);

      // Eight producer warps calculate their own 16-token QK tile.
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                     __nv_bfloat16, wmma::row_major> q_fragment;
      wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                     __nv_bfloat16, wmma::col_major> k_fragment;
      wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> score_fragment;
      wmma::fill_fragment(score_fragment, 0.0f);
#pragma unroll
      for (int k_offset = 0; k_offset < kHeadDim; k_offset += kWmmaK) {
        wmma::load_matrix_sync(q_fragment, &q_tile[0][k_offset], kHeadDim);
        wmma::load_matrix_sync(k_fragment, key_cache + qk_base + k_offset, kHeadDim);
        wmma::mma_sync(score_fragment, q_fragment, k_fragment, score_fragment);
      }
      wmma::store_matrix_sync(&score_tiles[warp][0][0], score_fragment, kTokensPerTile, wmma::mem_row_major);
      __syncthreads();

      // Consume tiles in exactly the preexisting selected-page then token-tile
      // order.  First form W and m/z, then all eight warps form their disjoint
      // dimension block of C=V^T*W^T, then update N cooperatively.
      for (int tile = 0; tile < kTokenTilesPerPage; ++tile) {
        if (warp == 0 && lane < kGqaGroup) {
          const int head = lane;
          float tile_max = -INFINITY;
#pragma unroll
          for (int token_in_tile = 0; token_in_tile < kTokensPerTile; ++token_in_tile) {
            const int token = tile * kTokensPerTile + token_in_tile;
            const int key_position = logical_page * kPageSize + token;
            if (key_position <= query_position && key_position < sequence_length) {
              tile_max = fmaxf(tile_max, score_tiles[tile][head][token_in_tile] * kScaleLog2e);
            }
          }
          if (!isfinite(tile_max)) {
            // No visible keys in this tile: no rounding, multiplication, or
            // reduction touches this head's online state.
            tile_active[head] = 0;
            alpha_tile[head] = 1.0f;
#pragma unroll
            for (int token_in_tile = 0; token_in_tile < kTokensPerTile; ++token_in_tile) {
              bf16_weights[head][token_in_tile] = __float2bfloat16_rn(0.0f);
            }
          } else {
            const float next_max = fmaxf(running_max[head], tile_max);
            const float alpha = isfinite(running_max[head])
                                    ? exp2f(running_max[head] - next_max)
                                    : 0.0f;
            float tile_sum = 0.0f;
#pragma unroll
            for (int token_in_tile = 0; token_in_tile < kTokensPerTile; ++token_in_tile) {
              const int token = tile * kTokensPerTile + token_in_tile;
              const int key_position = logical_page * kPageSize + token;
              const float beta = (key_position <= query_position && key_position < sequence_length)
                                     ? exp2f(score_tiles[tile][head][token_in_tile] * kScaleLog2e - next_max)
                                     : 0.0f;
              const __nv_bfloat16 rounded = __float2bfloat16_rn(beta);
              bf16_weights[head][token_in_tile] = rounded;
              tile_sum += __bfloat162float(rounded);
            }
            running_max[head] = next_max;
            normalizer[head] = alpha * normalizer[head] + tile_sum;
            alpha_tile[head] = alpha;
            tile_active[head] = 1;
          }
        }
        __syncthreads();

        // Warp w owns V^T rows [16*w,16*w+15].  Physical V[token,dim] is
        // logical col-major A[dim,token] with ld=128; stored W[head,token] is
        // logical col-major B[token,head] with ld=16.  The product is C[dim,head].
        const int dim_begin = warp * kDimsPerPvWarp;
        const std::size_t v_base = cache_offset(physical_page, kv_head, tile * kTokensPerTile, dim_begin);
        wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                       __nv_bfloat16, wmma::col_major> v_transpose_fragment;
        wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                       __nv_bfloat16, wmma::col_major> weight_transpose_fragment;
        wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> pv_fragment;
        wmma::fill_fragment(pv_fragment, 0.0f);
        wmma::load_matrix_sync(v_transpose_fragment, value_cache + v_base, kHeadDim);
        wmma::load_matrix_sync(weight_transpose_fragment, &bf16_weights[0][0], kTokensPerTile);
        wmma::mma_sync(pv_fragment, v_transpose_fragment, weight_transpose_fragment, pv_fragment);
        wmma::store_matrix_sync(&pv_contribution[warp][0][0], pv_fragment, kGqaGroup, wmma::mem_row_major);
        __syncthreads();

        for (int element = thread; element < kGqaGroup * kHeadDim; element += kThreadsPerBlock) {
          const int head = element / kHeadDim;
          const int dim = element % kHeadDim;
          if (tile_active[head] != 0) {
            numerator[head][dim] = alpha_tile[head] * numerator[head][dim]
                                  + pv_contribution[dim / kDimsPerPvWarp][dim % kDimsPerPvWarp][head];
          }
        }
        __syncthreads();
      }
    }

    // Preserve the control's 8-warps x 2-head partial ownership for DSM merge.
    const int dim0 = lane;
    const int dim1 = lane + kWarpSize;
    const int dim2 = lane + 2 * kWarpSize;
    const int dim3 = lane + 3 * kWarpSize;
#pragma unroll
    for (int head_in_warp = 0; head_in_warp < kHeadsPerWarp; ++head_in_warp) {
      const int group_head = warp * kHeadsPerWarp + head_in_warp;
      const float z = normalizer[group_head];
      const std::size_t partial_base = static_cast<std::size_t>(group_head) * kHeadDim;
      if (z > 0.0f) {
        local_partial[partial_base + dim0] = __float2bfloat16_rn(numerator[group_head][dim0] / z);
        local_partial[partial_base + dim1] = __float2bfloat16_rn(numerator[group_head][dim1] / z);
        local_partial[partial_base + dim2] = __float2bfloat16_rn(numerator[group_head][dim2] / z);
        local_partial[partial_base + dim3] = __float2bfloat16_rn(numerator[group_head][dim3] / z);
        if (lane == 0) local_lse[group_head] = running_max[group_head] + log2f(z);
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
        const float merged = denominator > 0.0f
                                 ? (partial0 * weight0 + partial1 * weight1) / denominator
                                 : 0.0f;
        caller_output[output_base + dim] = __float2bfloat16_rn(merged);
      }
    }
  }
  cluster.sync();
}

void launch_tc_qk_control(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length) {
  CUDA_CHECK(cudaLaunchKernelEx(&launch.config,
                                 tcqk_control::cluster_attention_mbarrier_warp_producer_tc_qk_kernel,
                                 buffers.query, buffers.key_cache, buffers.value_cache, buffers.topk_idx,
                                 buffers.block_table, sequence_length, buffers.scalar_output));
  CUDA_CHECK(cudaGetLastError());
}

void launch_tc_qk_pv_candidate(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length) {
  CUDA_CHECK(cudaLaunchKernelEx(&launch.config, cluster_attention_mbarrier_warp_producer_tc_qk_pv_kernel,
                                 buffers.query, buffers.key_cache, buffers.value_cache, buffers.topk_idx,
                                 buffers.block_table, sequence_length, buffers.warp_output));
  CUDA_CHECK(cudaGetLastError());
}

struct PvSeedCorrectness {
  int seed = 0;
  int sequence_length = 0;
  bool hierarchy_valid = false;
  int adversarial_unselected_visible_pages = 0;
  int adversarial_masked_tokens = 0;
  ArmCorrectness control{};
  ArmCorrectness candidate{};
  CrossArmDiagnosis cross_arm{};
};

struct PvPostTimingCorrectness {
  int seed = 0;
  bool hierarchy_valid = false;
  ArmCorrectness control{};
  ArmCorrectness candidate{};
  CrossArmDiagnosis cross_arm{};
};

PvSeedCorrectness check_pv_seed(const AttentionInput& input, const LaunchState& launch,
                                DeviceBuffers* timing_buffers) {
  const bool hierarchy_valid = validate_indirection(input);
  if (!hierarchy_valid) throw std::runtime_error("input indirection failed validation before oracle or GPU launch");
  const std::vector<float> oracle = cpu_fp64_accum_oracle(input);
  DeviceBuffers local_buffers;
  DeviceBuffers* buffers = input.seed == kTimingSeed ? timing_buffers : &local_buffers;
  allocate_and_copy(input, buffers);
  launch_tc_qk_control(launch, *buffers, input.sequence_length);
  launch_tc_qk_pv_candidate(launch, *buffers, input.sequence_length);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> control(kOutputElements), candidate(kOutputElements);
  CUDA_CHECK(cudaMemcpy(control.data(), buffers->scalar_output, control.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(candidate.data(), buffers->warp_output, candidate.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  return PvSeedCorrectness{input.seed, input.sequence_length, hierarchy_valid,
                           input.adversarial_unselected_visible_pages, input.adversarial_masked_tokens,
                           validate_output(control, oracle), validate_output(candidate, oracle),
                           diagnose_cross_arm(control, candidate)};
}

void require_pv_correct(const PvSeedCorrectness& result) {
  if (!result.hierarchy_valid || result.adversarial_unselected_visible_pages <= 0
      || result.adversarial_masked_tokens != kKvHeads * (kPageSize - 1)
      || !correct_arm(result.control) || !correct_arm(result.candidate)) {
    throw std::runtime_error("TC-QK control/TC-QK+PV candidate correctness gate failed before timing");
  }
}

PvPostTimingCorrectness revalidate_pv_after_timing(const LaunchState& launch, DeviceBuffers& buffers) {
  const AttentionInput input = make_input(kTimingSeed);
  const bool hierarchy_valid = validate_indirection(input);
  if (!hierarchy_valid) throw std::runtime_error("post-timing input indirection validation failed");
  const std::vector<float> oracle = cpu_fp64_accum_oracle(input);
  const std::vector<__nv_bfloat16> sentinel_output(kOutputElements, __float2bfloat16_rn(kSentinel));
  CUDA_CHECK(cudaMemcpy(buffers.scalar_output, sentinel_output.data(), sentinel_output.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.warp_output, sentinel_output.data(), sentinel_output.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  launch_tc_qk_control(launch, buffers, input.sequence_length);
  launch_tc_qk_pv_candidate(launch, buffers, input.sequence_length);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> control(kOutputElements), candidate(kOutputElements);
  CUDA_CHECK(cudaMemcpy(control.data(), buffers.scalar_output, control.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(candidate.data(), buffers.warp_output, candidate.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  const PvPostTimingCorrectness result{input.seed, hierarchy_valid, validate_output(control, oracle),
                                       validate_output(candidate, oracle), diagnose_cross_arm(control, candidate)};
  if (!result.hierarchy_valid || !correct_arm(result.control) || !correct_arm(result.candidate)) {
    throw std::runtime_error("post-timing TC-QK control/TC-QK+PV candidate correctness gate failed");
  }
  return result;
}

float time_pv_control_once(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length,
                           cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start));
  launch_tc_qk_control(launch, buffers, sequence_length);
  CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
  float milliseconds = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
  return milliseconds * 1000.0f;
}

float time_pv_candidate_once(const LaunchState& launch, const DeviceBuffers& buffers, int sequence_length,
                             cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start));
  launch_tc_qk_pv_candidate(launch, buffers, sequence_length);
  CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
  float milliseconds = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
  return milliseconds * 1000.0f;
}

void print_pv_seed_json(const PvSeedCorrectness& result) {
  std::cout << "{\"seed\":" << result.seed << ",\"sequence_length\":" << result.sequence_length
            << ",\"hierarchy_valid\":" << (result.hierarchy_valid ? "true" : "false")
            << ",\"adversarial_unselected_visible_pages\":" << result.adversarial_unselected_visible_pages
            << ",\"adversarial_masked_tokens\":" << result.adversarial_masked_tokens << ",\"control\":";
  print_arm_json(result.control); std::cout << ",\"candidate\":"; print_arm_json(result.candidate);
  std::cout << ",\"cross_arm_diagnostic\":"; print_cross_json(result.cross_arm); std::cout << '}';
}

void print_pv_post_json(const PvPostTimingCorrectness& result) {
  std::cout << "{\"seed\":" << result.seed << ",\"hierarchy_valid\":" << (result.hierarchy_valid ? "true" : "false")
            << ",\"control\":"; print_arm_json(result.control); std::cout << ",\"candidate\":";
  print_arm_json(result.candidate); std::cout << ",\"cross_arm_diagnostic\":"; print_cross_json(result.cross_arm);
  std::cout << '}';
}

void print_pv_success_json(const cudaDeviceProp& property, const cudaFuncAttributes& control_attributes,
                           const cudaFuncAttributes& candidate_attributes, int runtime_version, int driver_version,
                           int cluster_launch, const std::vector<PvSeedCorrectness>& correctness,
                           const PvPostTimingCorrectness& post, const std::vector<float>& control_ab,
                           const std::vector<float>& candidate_ab, const std::vector<float>& candidate_ba,
                           const std::vector<float>& control_ba) {
  std::vector<float> control_all = control_ab; control_all.insert(control_all.end(), control_ba.begin(), control_ba.end());
  std::vector<float> candidate_all = candidate_ab; candidate_all.insert(candidate_all.end(), candidate_ba.begin(), candidate_ba.end());
  const Statistics control_all_stats = summarize_us(control_all), candidate_all_stats = summarize_us(candidate_all);
  const Statistics control_ab_stats = summarize_us(control_ab), candidate_ab_stats = summarize_us(candidate_ab);
  const Statistics control_ba_stats = summarize_us(control_ba), candidate_ba_stats = summarize_us(candidate_ba);
  const double speedup = control_all_stats.median_us / candidate_all_stats.median_us;
  const double ab_speedup = control_ab_stats.median_us / candidate_ab_stats.median_us;
  const double ba_speedup = control_ba_stats.median_us / candidate_ba_stats.median_us;
  const bool promotion = speedup >= 1.10 && ab_speedup > 1.05 && ba_speedup > 1.05
                      && candidate_attributes.localSizeBytes == 0;
  std::cout << std::setprecision(9)
            << "{\"schema\":\"c2-cluster-attention-tc-qk-pv-abba-v1\",\"status\":\"pass\",\"boundary\":\""
            << json_escape(kTcQkPvBoundary) << "\",\"timing_seed\":" << kTimingSeed
            << ",\"shape\":{\"B\":" << kBatch << ",\"Hkv\":" << kKvHeads << ",\"Hq\":" << kQueryHeads
            << ",\"G\":" << kGqaGroup << ",\"D\":" << kHeadDim << ",\"page_size\":" << kPageSize
            << ",\"selected_pages\":" << kSelectedPages << ",\"logical_pages\":" << kLogicalPages << "}"
            << ",\"cluster_layout\":{\"num_ctas\":" << kNumCtas << ",\"clusters\":" << kKvHeads
            << ",\"selected_pages_per_producer\":" << kPagesPerProducer << ",\"threads_per_block\":" << kThreadsPerBlock << "}"
            << ",\"producer_contract\":{\"control\":\"reviewed B=1/C=2 WMMA-QK plus warp-PV kernel\",\"candidate\":\""
            << json_escape(kTcQkPvDescription) << "\",\"changed_field\":\"rank-0/1 producer PV data plane only after identical WMMA-QK\""
            << ",\"same_wmma_qk\":true,\"same_remote_dsm_mbarrier_protocol\":true,\"same_rank2_merge_output_abi_and_lifetime_sync\":true"
            << ",\"same_launch_shape\":true,\"same_real_selected_causal_attention\":true,\"persistent_device_buffers_outside_timing\":true"
            << ",\"caller_owned_independent_outputs\":true,\"single_kernel_launch_per_cuda_event_sample\":true,\"ABBA_interleaved\":true"
            << ",\"initialization_copies_and_oracle_outside_timing\":true,\"post_timing_fresh_sentinel_reset_and_relaunch\":true"
            << ",\"candidate_extra_shared_and_cta_barriers_included\":true,\"no_global_score_or_weight_workspace\":true,\"no_second_kernel\":true"
            << ",\"no_scalar_residual_correction\":true,\"cross_arm_bitwise\":\"diagnostic only; each arm independently gates against FP64 oracle\"}"
            << ",\"synchronization\":{\"mbarrier_expected_arrivals\":" << kMBarrierExpectedArrivals
            << ",\"mbarrier_wait_parity\":" << kMBarrierInitialParity << ",\"mbarrier_max_polls\":" << kMBarrierMaxPolls
            << ",\"producer_ready\":\"two remote release-arrivals then rank-2 local acquire parity wait\""
            << ",\"cluster_sync\":\"init plus producer-CTA local partial lifetime only\",\"candidate_cta_barriers\":\"Q load, QK materialization, and each online tile's W/PV/N phases\"}"
            << ",\"dtype_contract\":{\"producer_partial\":\"bfloat16\",\"caller_output\":\"bfloat16\""
            << ",\"qk\":\"WMMA BF16 m16n16k16 with FP32 accumulator\",\"pv\":\"WMMA BF16 V^T[16,16] times BF16 W^T[16,16] with FP32 accumulator\""
            << ",\"online_state\":\"FP32 m/z/N; rounded BF16 W feeds both z and PV product\",\"oracle_accumulator\":\"float64\""
            << ",\"oracle\":\"independent two-pass natural-exp direct selected-page causal attention\",\"tolerance\":{\"rtol\":" << kRtol << ",\"atol\":" << kAtol << "}}"
            << ",\"environment\":{\"device\":\"" << json_escape(property.name) << "\",\"capability\":[" << property.major << ',' << property.minor
            << "],\"cuda_runtime\":" << runtime_version << ",\"cuda_driver\":" << driver_version
            << ",\"cluster_launch_supported\":" << (cluster_launch ? "true" : "false") << "}"
            << ",\"resource_model\":{\"control\":{\"static_shared_bytes\":" << control_attributes.sharedSizeBytes
            << ",\"num_regs\":" << control_attributes.numRegs << ",\"local_bytes\":" << control_attributes.localSizeBytes << "}"
            << ",\"candidate\":{\"static_shared_bytes\":" << candidate_attributes.sharedSizeBytes << ",\"num_regs\":" << candidate_attributes.numRegs
            << ",\"local_bytes\":" << candidate_attributes.localSizeBytes << "}}"
            << ",\"correctness\":[";
  for (std::size_t index = 0; index < correctness.size(); ++index) { if (index) std::cout << ','; print_pv_seed_json(correctness[index]); }
  std::cout << "],\"post_timing_correctness\":"; print_pv_post_json(post);
  std::cout << ",\"timing\":{\"protocol\":\"warmup_each_then_101_control_candidate_candidate_control_ABBA_pairs\",\"warmup_each\":" << kWarmupEach
            << ",\"abba_pairs\":" << kAbbapairs << ",\"samples_per_arm\":" << kSamplesPerArm
            << ",\"raw_samples_us\":{\"control\":{\"AB\":"; print_samples_json(control_ab); std::cout << ",\"BA\":"; print_samples_json(control_ba);
  std::cout << "},\"candidate\":{\"AB\":"; print_samples_json(candidate_ab); std::cout << ",\"BA\":"; print_samples_json(candidate_ba);
  std::cout << "}},\"control\":{\"all\":"; print_statistics_json(control_all_stats); std::cout << ",\"when_launch_order_is_AB\":"; print_statistics_json(control_ab_stats);
  std::cout << ",\"when_launch_order_is_BA\":"; print_statistics_json(control_ba_stats); std::cout << "},\"candidate\":{\"all\":"; print_statistics_json(candidate_all_stats);
  std::cout << ",\"when_launch_order_is_AB\":"; print_statistics_json(candidate_ab_stats); std::cout << ",\"when_launch_order_is_BA\":"; print_statistics_json(candidate_ba_stats);
  std::cout << "},\"speedup_control_over_candidate\":" << speedup << ",\"speedup_by_partition\":{\"AB\":" << ab_speedup << ",\"BA\":" << ba_speedup
            << "},\"promotion_gate\":{\"combined_control_over_candidate_at_least_1_10\":" << (speedup >= 1.10 ? "true" : "false")
            << ",\"AB_control_over_candidate_greater_than_1_05\":" << (ab_speedup > 1.05 ? "true" : "false")
            << ",\"BA_control_over_candidate_greater_than_1_05\":" << (ba_speedup > 1.05 ? "true" : "false")
            << ",\"candidate_local_size_bytes_zero\":" << (candidate_attributes.localSizeBytes == 0 ? "true" : "false")
            << ",\"all_correct\":true,\"promoted\":" << (promotion ? "true" : "false") << "}}}" << std::endl;
}

void print_pv_failure_json(const std::string& error) {
  std::cout << "{\"schema\":\"c2-cluster-attention-tc-qk-pv-abba-v1\",\"status\":\"fail\",\"error\":\""
            << json_escape(error) << "\",\"boundary\":\"" << json_escape(kTcQkPvBoundary) << "\"}" << std::endl;
}

}  // namespace c2_tc_qk_pv_candidate

int main() {
  using namespace c2_tc_qk_pv_candidate;
  try {
    int device = 0; CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp property{}; CUDA_CHECK(cudaGetDeviceProperties(&property, device));
    int cluster_launch = 0; CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device));
    int runtime_version = 0, driver_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version)); CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    cudaFuncAttributes control_attributes{}, candidate_attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&control_attributes,
                                     tcqk_control::cluster_attention_mbarrier_warp_producer_tc_qk_kernel));
    CUDA_CHECK(cudaFuncGetAttributes(&candidate_attributes, cluster_attention_mbarrier_warp_producer_tc_qk_pv_kernel));
    if (property.major != 10 || property.minor != 3) throw std::runtime_error("requires B300 compute capability 10.3");
    if (cluster_launch == 0) throw std::runtime_error("cudaDevAttrClusterLaunch is false");
    if (control_attributes.sharedSizeBytes > property.sharedMemPerBlock ||
        candidate_attributes.sharedSizeBytes > property.sharedMemPerBlock) {
      throw std::runtime_error("control or TC-QK+PV candidate static shared-memory requirement exceeds device limit");
    }
    if (candidate_attributes.localSizeBytes != 0) throw std::runtime_error("TC-QK+PV candidate has local-memory spill");
    const LaunchState launch{};
    DeviceBuffers timing_buffers;
    std::vector<PvSeedCorrectness> correctness; correctness.reserve(2);
    for (const int seed : std::vector<int>{17, kTimingSeed}) {
      correctness.push_back(check_pv_seed(make_input(seed), launch, &timing_buffers));
      require_pv_correct(correctness.back());
    }
    for (int iteration = 0; iteration < kWarmupEach; ++iteration)
      launch_tc_qk_control(launch, timing_buffers, correctness.back().sequence_length);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int iteration = 0; iteration < kWarmupEach; ++iteration)
      launch_tc_qk_pv_candidate(launch, timing_buffers, correctness.back().sequence_length);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr, end = nullptr; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> control_ab, candidate_ab, candidate_ba, control_ba;
    control_ab.reserve(kAbbapairs); candidate_ab.reserve(kAbbapairs);
    candidate_ba.reserve(kAbbapairs); control_ba.reserve(kAbbapairs);
    try {
      for (int pair = 0; pair < kAbbapairs; ++pair) {
        control_ab.push_back(time_pv_control_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
        candidate_ab.push_back(time_pv_candidate_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
        candidate_ba.push_back(time_pv_candidate_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
        control_ba.push_back(time_pv_control_once(launch, timing_buffers, correctness.back().sequence_length, start, end));
      }
    } catch (...) { cudaEventDestroy(end); cudaEventDestroy(start); throw; }
    CUDA_CHECK(cudaEventDestroy(end)); CUDA_CHECK(cudaEventDestroy(start));
    if (control_ab.size() != kAbbapairs || candidate_ab.size() != kAbbapairs ||
        candidate_ba.size() != kAbbapairs || control_ba.size() != kAbbapairs) {
      throw std::runtime_error("ABBA sample accounting mismatch");
    }
    const PvPostTimingCorrectness post = revalidate_pv_after_timing(launch, timing_buffers);
    print_pv_success_json(property, control_attributes, candidate_attributes, runtime_version, driver_version, cluster_launch,
                          correctness, post, control_ab, candidate_ab, candidate_ba, control_ba);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    c2_tc_qk_pv_candidate::print_pv_failure_json(error.what());
    return EXIT_FAILURE;
  }
}
