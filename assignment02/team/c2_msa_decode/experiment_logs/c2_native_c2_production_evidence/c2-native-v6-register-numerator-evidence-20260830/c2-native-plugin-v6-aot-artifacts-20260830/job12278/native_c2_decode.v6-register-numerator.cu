// Production-facing MiniMax M3 C2 sparse-decode prototype for exact vLLM d4.
//
// This source is intentionally an AOT stable-Torch operator, not a standalone
// benchmark.  It consumes the production packed E4M3 KV cache directly,
// stages only the current 16x16 tiles to CTA-local BF16 shared memory, and
// writes into caller-owned BF16 output.  The initial eligibility contract is
// deliberately narrow: B=16, Qlen=1, Hq/Hkv/D=64/4/128, page=128, top-k=16,
// scalar q/k/v scales, and a B300-family CUDA device.

#include "libtorch_stable/torch_utils.h"

#include <torch/csrc/stable/library.h>

#include <cooperative_groups.h>
#include <cuda/ptx>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace vllm::native_c2 {

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

constexpr int kBatch = 16;
constexpr int kQueryHeads = 64;
constexpr int kKvHeads = 4;
constexpr int kGqaGroup = kQueryHeads / kKvHeads;
constexpr int kHeadDim = 128;
constexpr int kPageSize = 128;
constexpr int kSelectedPages = 16;
constexpr int kClusterCtas = 4;
constexpr int kProducerCtas = 4;
constexpr int kMergeCta = 0;
constexpr int kPagesPerProducer = kSelectedPages / kProducerCtas;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kHeadsPerWarp = kGqaGroup / kWarps;
constexpr int kTile = 16;
constexpr int kTokenTiles = kPageSize / kTile;
constexpr int kMbarrierArrivals = kProducerCtas;
constexpr int kMbarrierParity = 0;
constexpr int kMbarrierMaxPolls = 1 << 24;
constexpr float kLog2e = 1.4426950408889634f;

static_assert(kGqaGroup == kTile);
static_assert(kTokenTiles == kWarps);
static_assert(kHeadDim / kTile == kWarps);

__device__ __forceinline__ __nv_bfloat16 scaled_fp8(
    const __nv_fp8_e4m3* ptr, std::int64_t offset, float scale) {
  return __float2bfloat16_rn(static_cast<float>(ptr[offset]) * scale);
}

__global__ void native_c2_msa_decode_kernel(
    const __nv_fp8_e4m3* __restrict__ query,
    const __nv_fp8_e4m3* __restrict__ kv_cache,
    const std::int32_t* __restrict__ topk,
    const std::int32_t* __restrict__ block_table,
    const std::int32_t* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ output,
    std::int64_t q_token_stride,
    std::int64_t q_head_stride,
    std::int64_t kv_page_stride,
    std::int64_t kv_head_stride,
    std::int64_t kv_token_stride,
    std::int64_t topk_token_stride,
    std::int64_t topk_head_stride,
    std::int64_t block_table_batch_stride,
    std::int64_t output_token_stride,
    std::int64_t output_head_stride,
    int num_physical_pages,
    int max_logical_pages,
    float scale_log2e,
    float q_scale,
    float k_scale,
    float v_scale) {
  __shared__ __nv_bfloat16 local_partial[kGqaGroup * kHeadDim];
  __shared__ float local_lse[kGqaGroup];
  __shared__ __align__(8) std::uint64_t producer_ready_barrier;
  __shared__ int producer_ready;

  __shared__ __align__(32) __nv_bfloat16 q_tile[kGqaGroup][kHeadDim];
  // Reused for the current K or V 16x16 tile owned by each warp.  Staging is
  // inside the kernel and timed; no split/contiguous/BF16 bridge exists.
  __shared__ __align__(32) __nv_bfloat16 fp8_stage[kWarps][kTile][kTile];
  __shared__ __align__(32) float score_tiles[kTokenTiles][kGqaGroup][kTile];

  __shared__ float running_max[kGqaGroup];
  __shared__ float normalizer[kGqaGroup];
  __shared__ float alpha_tile[kGqaGroup];
  __shared__ int tile_active[kGqaGroup];
  __shared__ __align__(32) __nv_bfloat16 weights[kGqaGroup][kTile];
  __shared__ __align__(32) float pv_contribution[kWarps][kTile][kGqaGroup];

  cg::cluster_group cluster = cg::this_cluster();
  int const role = static_cast<int>(cluster.block_rank());
  int const cluster_index = static_cast<int>(blockIdx.x) / kClusterCtas;
  int const batch = cluster_index / kKvHeads;
  int const kv_head = cluster_index % kKvHeads;
  int const thread = static_cast<int>(threadIdx.x);
  int const warp = thread / kWarpSize;
  int const lane = thread & (kWarpSize - 1);
  int const sequence_length = seq_lens[batch];
  int const query_position = sequence_length - 1;

  if (role == kMergeCta && thread == 0) {
    cuda::ptx::mbarrier_init(&producer_ready_barrier, kMbarrierArrivals);
    producer_ready = 0;
  }
  __syncthreads();
  cluster.sync();

  if (role < kProducerCtas) {
    // Each lane owns one head and eight dimensions in this warp's disjoint
    // 16-dimension slice.  These accumulators persist across all page tiles.
    float numerator_accumulator[kTile / 2] = {};

    int const selected_begin = role * kPagesPerProducer;
    for (int element = thread; element < kGqaGroup * kHeadDim;
         element += kThreads) {
      int const group_head = element / kHeadDim;
      int const dim = element % kHeadDim;
      std::int64_t const q_offset =
          static_cast<std::int64_t>(batch) * q_token_stride +
          static_cast<std::int64_t>(kv_head * kGqaGroup + group_head) *
              q_head_stride +
          dim;
      q_tile[group_head][dim] = scaled_fp8(query, q_offset, q_scale);
    }
    if (thread < kGqaGroup) {
      running_max[thread] = -INFINITY;
      normalizer[thread] = 0.0f;
      alpha_tile[thread] = 1.0f;
      tile_active[thread] = 0;
    }
    __syncthreads();

    for (int selected = selected_begin;
         selected < selected_begin + kPagesPerProducer; ++selected) {
      std::int64_t const topk_offset =
          static_cast<std::int64_t>(batch) * topk_token_stride +
          static_cast<std::int64_t>(kv_head) * topk_head_stride + selected;
      int const logical_page = topk[topk_offset];
      if (logical_page < 0 || logical_page >= max_logical_pages) continue;
      int const physical_page =
          block_table[static_cast<std::int64_t>(batch) *
                          block_table_batch_stride +
                      logical_page];
      if (physical_page < 0 || physical_page >= num_physical_pages) continue;

      int const token_begin = warp * kTile;
      wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __nv_bfloat16,
                     wmma::row_major>
          q_fragment;
      wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __nv_bfloat16,
                     wmma::col_major>
          k_fragment;
      wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
          score_fragment;
      wmma::fill_fragment(score_fragment, 0.0f);

#pragma unroll
      for (int k_offset = 0; k_offset < kHeadDim; k_offset += kTile) {
        for (int element = lane; element < kTile * kTile;
             element += kWarpSize) {
          int const token = element / kTile;
          int const dim = element % kTile;
          std::int64_t const cache_offset =
              static_cast<std::int64_t>(physical_page) * kv_page_stride +
              static_cast<std::int64_t>(kv_head) * kv_head_stride +
              static_cast<std::int64_t>(token_begin + token) *
                  kv_token_stride +
              k_offset + dim;
          fp8_stage[warp][token][dim] =
              scaled_fp8(kv_cache, cache_offset, k_scale);
        }
        __syncwarp();
        wmma::load_matrix_sync(q_fragment, &q_tile[0][k_offset], kHeadDim);
        wmma::load_matrix_sync(k_fragment, &fp8_stage[warp][0][0], kTile);
        wmma::mma_sync(score_fragment, q_fragment, k_fragment,
                       score_fragment);
        __syncwarp();
      }
      wmma::store_matrix_sync(&score_tiles[warp][0][0], score_fragment, kTile,
                              wmma::mem_row_major);
      __syncthreads();

      for (int tile = 0; tile < kTokenTiles; ++tile) {
        // All eight warps participate in softmax/weight generation.  Each
        // warp owns two heads and each 16-lane subgroup owns one token row.
        int const subgroup_lane = lane & (kTile - 1);
        int const head = warp * kHeadsPerWarp + lane / kTile;
        unsigned const subgroup_mask =
            lane < kTile ? 0x0000ffffu : 0xffff0000u;
        int const token = tile * kTile + subgroup_lane;
        int const key_position = logical_page * kPageSize + token;
        bool const valid = key_position <= query_position &&
                           key_position < sequence_length;
        float const scaled_score =
            valid ? score_tiles[tile][head][subgroup_lane] * scale_log2e
                  : -INFINITY;
        float tile_max = scaled_score;
#pragma unroll
        for (int offset = kTile / 2; offset > 0; offset >>= 1) {
          tile_max = fmaxf(
              tile_max,
              __shfl_xor_sync(subgroup_mask, tile_max, offset, kTile));
        }
        if (!isfinite(tile_max)) {
          weights[head][subgroup_lane] = __float2bfloat16_rn(0.0f);
          if (subgroup_lane == 0) {
            tile_active[head] = 0;
            alpha_tile[head] = 1.0f;
          }
        } else {
          float const previous_max = running_max[head];
          float const next_max = fmaxf(previous_max, tile_max);
          float const alpha = isfinite(previous_max)
                                  ? exp2f(previous_max - next_max)
                                  : 0.0f;
          float const beta =
              valid ? exp2f(scaled_score - next_max) : 0.0f;
          __nv_bfloat16 const rounded = __float2bfloat16_rn(beta);
          weights[head][subgroup_lane] = rounded;
          float tile_sum = __bfloat162float(rounded);
#pragma unroll
          for (int offset = kTile / 2; offset > 0; offset >>= 1) {
            tile_sum +=
                __shfl_xor_sync(subgroup_mask, tile_sum, offset, kTile);
          }
          if (subgroup_lane == 0) {
            running_max[head] = next_max;
            normalizer[head] = alpha * normalizer[head] + tile_sum;
            alpha_tile[head] = alpha;
            tile_active[head] = 1;
          }
        }
        __syncthreads();

        int const dim_begin = warp * kTile;
        for (int element = lane; element < kTile * kTile;
             element += kWarpSize) {
          int const token = element / kTile;
          int const dim = element % kTile;
          std::int64_t const cache_offset =
              static_cast<std::int64_t>(physical_page) * kv_page_stride +
              static_cast<std::int64_t>(kv_head) * kv_head_stride +
              static_cast<std::int64_t>(tile * kTile + token) *
                  kv_token_stride +
              kHeadDim + dim_begin + dim;
          fp8_stage[warp][token][dim] =
              scaled_fp8(kv_cache, cache_offset, v_scale);
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __nv_bfloat16,
                       wmma::col_major>
            value_fragment;
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __nv_bfloat16,
                       wmma::col_major>
            weight_fragment;
        wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>
            pv_fragment;
        wmma::fill_fragment(pv_fragment, 0.0f);
        wmma::load_matrix_sync(value_fragment, &fp8_stage[warp][0][0], kTile);
        wmma::load_matrix_sync(weight_fragment, &weights[0][0], kTile);
        wmma::mma_sync(pv_fragment, value_fragment, weight_fragment,
                       pv_fragment);
        wmma::store_matrix_sync(&pv_contribution[warp][0][0], pv_fragment,
                                kGqaGroup, wmma::mem_row_major);
        __syncwarp();

        // Every warp still owns one disjoint 16-dimension slice, but each
        // lane now retains one head's eight dimensions in registers instead
        // of round-tripping the 2,048-element numerator through shared memory.
        int const accumulator_head = lane & (kTile - 1);
        int const accumulator_parity = lane / kTile;
        if (tile_active[accumulator_head] != 0) {
          float const alpha = alpha_tile[accumulator_head];
#pragma unroll
          for (int i = 0; i < kTile / 2; ++i) {
            int const dim_in_warp = 2 * i + accumulator_parity;
            numerator_accumulator[i] =
                alpha * numerator_accumulator[i] +
                pv_contribution[warp][dim_in_warp][accumulator_head];
          }
        }
        __syncthreads();
      }
    }

    int const accumulator_head = lane & (kTile - 1);
    int const accumulator_parity = lane / kTile;
    float const z = normalizer[accumulator_head];
    std::int64_t const partial_base =
        static_cast<std::int64_t>(accumulator_head) * kHeadDim;
#pragma unroll
    for (int i = 0; i < kTile / 2; ++i) {
      int const dim = warp * kTile + 2 * i + accumulator_parity;
      local_partial[partial_base + dim] =
          z > 0.0f
              ? __float2bfloat16_rn(numerator_accumulator[i] / z)
              : __float2bfloat16_rn(0.0f);
    }
    if (warp == 0 && lane < kGqaGroup) {
      local_lse[accumulator_head] =
          z > 0.0f ? running_max[accumulator_head] + log2f(z) : -INFINITY;
    }
  }

  __syncthreads();
  if (role < kProducerCtas && thread == 0) {
    std::uint64_t* remote_merge_barrier =
        cluster.map_shared_rank(&producer_ready_barrier, kMergeCta);
    cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release,
                               cuda::ptx::scope_cluster,
                               cuda::ptx::space_cluster,
                               remote_merge_barrier);
  }
  if (role == kMergeCta && thread == 0) {
    bool ready = false;
#pragma unroll 1
    for (int poll = 0; poll < kMbarrierMaxPolls; ++poll) {
      if (cuda::ptx::mbarrier_try_wait_parity(
              cuda::ptx::sem_acquire, cuda::ptx::scope_cluster,
              &producer_ready_barrier, kMbarrierParity)) {
        ready = true;
        break;
      }
    }
    producer_ready = ready ? 1 : 0;
  }
  __syncthreads();

  if (role == kMergeCta && thread < kGqaGroup) {
    std::int64_t const output_base =
        static_cast<std::int64_t>(batch) * output_token_stride +
        static_cast<std::int64_t>(kv_head * kGqaGroup + thread) *
            output_head_stride;
    if (producer_ready == 0) {
      // A timed-out cluster protocol must be observably invalid; returning a
      // plausible zero tensor would hide a liveness/correctness failure.
      for (int dim = 0; dim < kHeadDim; ++dim) {
        output[output_base + dim] = __float2bfloat16_rn(NAN);
      }
    } else {
      __nv_bfloat16 const* remote_partial1 =
          cluster.map_shared_rank(local_partial, 1);
      __nv_bfloat16 const* remote_partial2 =
          cluster.map_shared_rank(local_partial, 2);
      __nv_bfloat16 const* remote_partial3 =
          cluster.map_shared_rank(local_partial, 3);
      float const* remote_lse1 = cluster.map_shared_rank(local_lse, 1);
      float const* remote_lse2 = cluster.map_shared_rank(local_lse, 2);
      float const* remote_lse3 = cluster.map_shared_rank(local_lse, 3);
      float const lse0 = local_lse[thread];
      float const lse1 = remote_lse1[thread];
      float const lse2 = remote_lse2[thread];
      float const lse3 = remote_lse3[thread];
      float const lse_max =
          fmaxf(fmaxf(lse0, lse1), fmaxf(lse2, lse3));
      float const weight0 = isfinite(lse0) ? exp2f(lse0 - lse_max) : 0.0f;
      float const weight1 = isfinite(lse1) ? exp2f(lse1 - lse_max) : 0.0f;
      float const weight2 = isfinite(lse2) ? exp2f(lse2 - lse_max) : 0.0f;
      float const weight3 = isfinite(lse3) ? exp2f(lse3 - lse_max) : 0.0f;
      float const denominator = weight0 + weight1 + weight2 + weight3;
      for (int dim = 0; dim < kHeadDim; ++dim) {
        float const partial0 =
            __bfloat162float(local_partial[thread * kHeadDim + dim]);
        float const partial1 = __bfloat162float(
            remote_partial1[thread * kHeadDim + dim]);
        float const partial2 = __bfloat162float(
            remote_partial2[thread * kHeadDim + dim]);
        float const partial3 = __bfloat162float(
            remote_partial3[thread * kHeadDim + dim]);
        float const merged = denominator > 0.0f
                                 ? (partial0 * weight0 +
                                    partial1 * weight1 +
                                    partial2 * weight2 +
                                    partial3 * weight3) /
                                       denominator
                                 : 0.0f;
        output[output_base + dim] = __float2bfloat16_rn(merged);
      }
    }
  }
  // Retain every producer's shared-memory lifetime until rank 0 finishes the
  // four-way DSM merge and all threads reach the cluster-wide rendezvous.
  cluster.sync();
}

void native_c2_msa_decode(
    torch::stable::Tensor& output,
    const torch::stable::Tensor& query_fp8,
    const torch::stable::Tensor& kv_cache,
    const torch::stable::Tensor& topk,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens,
    double scale,
    double q_scale,
    double k_scale,
    double v_scale) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(output.is_cuda() && query_fp8.is_cuda() &&
                      kv_cache.is_cuda() && topk.is_cuda() &&
                      block_table.is_cuda() && seq_lens.is_cuda(),
                  "native_c2 tensors must all be CUDA tensors");
  int const device = output.get_device_index();
  STD_TORCH_CHECK(query_fp8.get_device_index() == device &&
                      kv_cache.get_device_index() == device &&
                      topk.get_device_index() == device &&
                      block_table.get_device_index() == device &&
                      seq_lens.get_device_index() == device,
                  "native_c2 tensors must share one CUDA device");
  STD_TORCH_CHECK(query_fp8.scalar_type() == ScalarType::Float8_e4m3fn &&
                      kv_cache.scalar_type() == ScalarType::Float8_e4m3fn,
                  "native_c2 requires E4M3 query and packed KV");
  STD_TORCH_CHECK(output.scalar_type() == ScalarType::BFloat16,
                  "native_c2 output must be bfloat16");
  STD_TORCH_CHECK(topk.scalar_type() == ScalarType::Int &&
                      block_table.scalar_type() == ScalarType::Int &&
                      seq_lens.scalar_type() == ScalarType::Int,
                  "native_c2 metadata must be int32");
  STD_TORCH_CHECK(query_fp8.is_contiguous() && kv_cache.is_contiguous() &&
                      topk.is_contiguous() && block_table.is_contiguous() &&
                      seq_lens.is_contiguous() && output.is_contiguous(),
                  "native_c2 initial contract requires contiguous tensors");
  STD_TORCH_CHECK(query_fp8.dim() == 3 && query_fp8.size(0) == kBatch &&
                      query_fp8.size(1) == kQueryHeads &&
                      query_fp8.size(2) == kHeadDim,
                  "native_c2 query must be [16,64,128]");
  STD_TORCH_CHECK(output.dim() == 3 && output.size(0) == kBatch &&
                      output.size(1) == kQueryHeads &&
                      output.size(2) == kHeadDim,
                  "native_c2 output must be [16,64,128]");
  STD_TORCH_CHECK(kv_cache.dim() == 4 && kv_cache.size(0) > 0 &&
                      kv_cache.size(1) == kKvHeads &&
                      kv_cache.size(2) == kPageSize &&
                      kv_cache.size(3) == 2 * kHeadDim,
                  "native_c2 packed KV must be [pages,4,128,256]");
  STD_TORCH_CHECK(topk.dim() == 3 && topk.size(0) == kBatch &&
                      topk.size(1) == kKvHeads &&
                      topk.size(2) == kSelectedPages,
                  "native_c2 topk must be token-major [16,4,16]");
  STD_TORCH_CHECK(block_table.dim() == 2 &&
                      block_table.size(0) == kBatch &&
                      block_table.size(1) > 0,
                  "native_c2 block_table must be [16,max_blocks]");
  STD_TORCH_CHECK(seq_lens.dim() == 1 && seq_lens.size(0) == kBatch,
                  "native_c2 seq_lens must be [16]");
  STD_TORCH_CHECK(std::isfinite(scale) && scale > 0.0 &&
                      std::isfinite(q_scale) && q_scale > 0.0 &&
                      std::isfinite(k_scale) && k_scale > 0.0 &&
                      std::isfinite(v_scale) && v_scale > 0.0,
                  "native_c2 scales must be finite and positive");

  torch::stable::accelerator::DeviceGuard const device_guard(device);
  cudaDeviceProp property{};
  cudaError_t error = cudaGetDeviceProperties(&property, device);
  STD_TORCH_CHECK(error == cudaSuccess,
                  "native_c2 cudaGetDeviceProperties failed: ",
                  cudaGetErrorString(error));
  STD_TORCH_CHECK(property.major == 10 && property.minor == 3,
                  "native_c2 initial AOT kernel requires B300 compute "
                  "capability 10.3");
  int cluster_supported = 0;
  error = cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch,
                                 device);
  STD_TORCH_CHECK(error == cudaSuccess && cluster_supported != 0,
                  "native_c2 requires CUDA cluster launch support");

  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = kClusterCtas;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kBatch * kKvHeads * kClusterCtas, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = get_current_cuda_stream(device);
  config.attrs = &attribute;
  config.numAttrs = 1;

  error = cudaLaunchKernelEx(
      &config, native_c2_msa_decode_kernel,
      reinterpret_cast<const __nv_fp8_e4m3*>(query_fp8.const_data_ptr()),
      reinterpret_cast<const __nv_fp8_e4m3*>(kv_cache.const_data_ptr()),
      topk.const_data_ptr<std::int32_t>(),
      block_table.const_data_ptr<std::int32_t>(),
      seq_lens.const_data_ptr<std::int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
      query_fp8.stride(0), query_fp8.stride(1), kv_cache.stride(0),
      kv_cache.stride(1), kv_cache.stride(2), topk.stride(0),
      topk.stride(1), block_table.stride(0), output.stride(0),
      output.stride(1), static_cast<int>(kv_cache.size(0)),
      static_cast<int>(block_table.size(1)),
      static_cast<float>(scale) * kLog2e, static_cast<float>(q_scale),
      static_cast<float>(k_scale), static_cast<float>(v_scale));
  STD_TORCH_CHECK(error == cudaSuccess,
                  "native_c2 cluster launch failed: ",
                  cudaGetErrorString(error));
}

}  // namespace vllm::native_c2

// This schema deliberately lives in the standalone plugin DSO rather than
// csrc/libtorch_stable/torch_bindings.cpp.  torch.ops.load_library loads the
// DSO and executes these static registrations without importing it as a
// Python module, so the plugin has no PyInit symbol and does not replace the
// wheel's _C_stable_libtorch.abi3.so.
STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "native_c2_msa_decode(Tensor! output, Tensor query_fp8, Tensor kv_cache,"
      " Tensor topk, Tensor block_table, Tensor seq_lens, float scale,"
      " float q_scale, float k_scale, float v_scale) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("native_c2_msa_decode",
         TORCH_BOX(&vllm::native_c2::native_c2_msa_decode));
}
