// Batched native C=2 Tensor-Core QK producer AB/BA qualification experiment.
//
// This source imports the audited scalar protocol and defines two new
// batched-ABI kernels below.  The reviewed warp-producer source is an audited,
// hash-pinned reference in the runner only: its mapping is derived here, not
// preprocessor-imported or compiled as a dependency.  The new arms use one
// native cluster launch with grid B*Hkv*4 and differ only in the rank-0/1 QK
// producer mapping: warp shuffle control versus BF16 WMMA QK.  The candidate's
// Q/score shared storage and CTA barriers are deliberately part of its complete
// measured implementation cost.  This is a native correctness-prototype
// signal, not a production integration, throughput, model, or server claim.

#define main c2_cluster_attention_mbarrier_batch_embedded_main
#include "c2_cluster_attention_mbarrier_smoke.cu"
#undef main

#include <mma.h>

namespace {

constexpr int kBatchCases[] = {1, 4, 8, 16};
constexpr int kBatchCaseCount = sizeof(kBatchCases) / sizeof(kBatchCases[0]);
constexpr int kBatchTimingSeed = 2026;
constexpr int kBatchWarmupEach = 10;
constexpr int kBatchAbbapairs = 51;
constexpr int kBatchSamplesPerArm = 2 * kBatchAbbapairs;
constexpr int kBatchWarpsPerBlock = 8;
constexpr int kBatchWarpSize = 32;
constexpr int kBatchHeadsPerWarp = 2;
constexpr int kBatchDimsPerLane = 4;
constexpr unsigned kBatchFullWarpMask = 0xffffffffu;
constexpr const char* kBatchBoundary =
    "native C=2 batched ABI: warp-producer QK control versus BF16 WMMA-QK candidate; "
    "candidate Q/score shared storage and CTA barriers are complete implementation cost; "
    "not isolated Tensor-Core instruction, production integration, throughput, model, or server claim";
namespace wmma = nvcuda::wmma;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kTokensPerWmmaTile = kWmmaN;
constexpr int kWmmaTilesPerPage = kPageSize / kTokensPerWmmaTile;

static_assert(kThreadsPerBlock == kBatchWarpsPerBlock * kBatchWarpSize,
              "warp producer requires eight full warps");
static_assert(kGqaGroup == kBatchWarpsPerBlock * kBatchHeadsPerWarp,
              "each full warp owns exactly two serial GQA heads");
static_assert(kHeadDim == kBatchWarpSize * kBatchDimsPerLane,
              "each lane owns d=lane+{0,32,64,96}");
static_assert(kSelectedPages == 2 * kPagesPerProducer && kPagesPerProducer == 8,
              "the two producer CTAs retain the audited eight-page split");
static_assert(kNumCtas == 4 && kMBarrierExpectedArrivals == 2U,
              "this experiment preserves the audited four-CTA, two-arrival protocol");
static_assert(kGqaGroup == kWmmaM && kHeadDim % kWmmaK == 0 &&
                  kPageSize % kTokensPerWmmaTile == 0 &&
                  kBatchWarpsPerBlock == kWmmaTilesPerPage,
              "WMMA QK tile and producer warp geometry must match the B=1 audited design");

// Warp-shuffle QK control: same launch, DSM, mbarrier, merge and ABI as the TC candidate.
// Only producer arithmetic changes to eight warps x two serial heads, each
// lane owning d={lane,lane+32,lane+64,lane+96}.
extern "C" __global__ void c2_batch_warp_control_mbarrier_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, const int* topk_idx,
    const int* block_table, const int* seq_lens, __nv_bfloat16* caller_output) {
  __shared__ __nv_bfloat16 local_partial[kGqaGroup * kHeadDim];
  __shared__ float local_lse[kGqaGroup];
  __shared__ __align__(8) std::uint64_t producer_ready_barrier;
  __shared__ int producer_ready;

  const cg::cluster_group cluster = cg::this_cluster();
  const int role = static_cast<int>(cluster.block_rank());
  const int cluster_index = static_cast<int>(blockIdx.x / kNumCtas);
  const int batch = cluster_index / kKvHeads;
  const int kv_head = cluster_index % kKvHeads;
  const int thread = static_cast<int>(threadIdx.x);
  const int sequence_length = seq_lens[batch];
  const int query_position = sequence_length - 1;
  if (role == 2 && thread == 0) {
    cuda::ptx::mbarrier_init(&producer_ready_barrier, kMBarrierExpectedArrivals);
    producer_ready = 0;
  }
  __syncthreads();
  cluster.sync();

  if (role == 0 || role == 1) {
    const int lane = thread & (kBatchWarpSize - 1);
    const int warp = thread / kBatchWarpSize;
    const int dim0 = lane;
    const int dim1 = lane + kBatchWarpSize;
    const int dim2 = lane + 2 * kBatchWarpSize;
    const int dim3 = lane + 3 * kBatchWarpSize;
    const int selected_begin = role * kPagesPerProducer;
#pragma unroll
    for (int head_in_warp = 0; head_in_warp < kBatchHeadsPerWarp; ++head_in_warp) {
      const int group_head = warp * kBatchHeadsPerWarp + head_in_warp;
      const int query_head = kv_head * kGqaGroup + group_head;
      const __nv_bfloat16* query_row =
          query + (static_cast<std::size_t>(batch) * kQueryHeads + query_head) * kHeadDim;
      const float q0 = __bfloat162float(query_row[dim0]);
      const float q1 = __bfloat162float(query_row[dim1]);
      const float q2 = __bfloat162float(query_row[dim2]);
      const float q3 = __bfloat162float(query_row[dim3]);
      float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
      float max_score = -INFINITY, normalizer = 0.0f;
      for (int selected = selected_begin; selected < selected_begin + kPagesPerProducer; ++selected) {
        const int logical_page = topk_idx[(static_cast<std::size_t>(batch) * kKvHeads + kv_head) *
                                          kSelectedPages + selected];
        const int physical_page = block_table[static_cast<std::size_t>(batch) * kLogicalPages + logical_page];
        for (int token = 0; token < kPageSize; ++token) {
          const int key_position = logical_page * kPageSize + token;
          if (key_position <= query_position && key_position < sequence_length) {
            const std::size_t kv_base = cache_offset(physical_page, kv_head, token, 0);
            float partial_dot = fmaf(q0, __bfloat162float(key_cache[kv_base + dim0]), 0.0f);
            partial_dot = fmaf(q1, __bfloat162float(key_cache[kv_base + dim1]), partial_dot);
            partial_dot = fmaf(q2, __bfloat162float(key_cache[kv_base + dim2]), partial_dot);
            partial_dot = fmaf(q3, __bfloat162float(key_cache[kv_base + dim3]), partial_dot);
#pragma unroll
            for (int offset = kBatchWarpSize / 2; offset > 0; offset /= 2) {
              partial_dot += __shfl_down_sync(kBatchFullWarpMask, partial_dot, offset);
            }
            const float score = __shfl_sync(kBatchFullWarpMask, partial_dot, 0) * kScaleLog2e;
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
        if (lane == 0) local_lse[group_head] = max_score + log2f(normalizer);
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
    const std::size_t output_base =
        (static_cast<std::size_t>(batch) * kQueryHeads + kv_head * kGqaGroup + thread) * kHeadDim;
    if (producer_ready == 0) {
#pragma unroll
      for (int dim = 0; dim < kHeadDim; ++dim) caller_output[output_base + dim] = __float2bfloat16_rn(kSentinel);
    } else {
      const __nv_bfloat16* remote_partial0 = cluster.map_shared_rank(local_partial, 0);
      const __nv_bfloat16* remote_partial1 = cluster.map_shared_rank(local_partial, 1);
      const float* remote_lse0 = cluster.map_shared_rank(local_lse, 0);
      const float* remote_lse1 = cluster.map_shared_rank(local_lse, 1);
      const float lse0 = remote_lse0[thread], lse1 = remote_lse1[thread];
      const float lse_max = fmaxf(lse0, lse1);
      const float weight0 = isfinite(lse0) ? exp2f(lse0 - lse_max) : 0.0f;
      const float weight1 = isfinite(lse1) ? exp2f(lse1 - lse_max) : 0.0f;
      const float denominator = weight0 + weight1;
#pragma unroll
      for (int dim = 0; dim < kHeadDim; ++dim) {
        const float partial0 = __bfloat162float(remote_partial0[thread * kHeadDim + dim]);
        const float partial1 = __bfloat162float(remote_partial1[thread * kHeadDim + dim]);
        caller_output[output_base + dim] = __float2bfloat16_rn(
            denominator > 0.0f ? (partial0 * weight0 + partial1 * weight1) / denominator : 0.0f);
      }
    }
  }
  cluster.sync();
}

// Same native batched ABI, launch geometry, rank-2 mbarrier/DSM merge, output
// ownership, selected-page iteration and causal order as the warp control.
// The only intentional data-plane change is rank-0/1 QK: each producer CTA
// caches Q[16,128], then its eight warps use BF16 WMMA m16n16k16 / FP32
// accumulation for one contiguous 16-token K tile.  Existing serial-head
// online-softmax/PV code consumes the materialized scores in logical page then
// token order.  The candidate-only shared storage and its two CTA barriers per
// selected page remain timed and are reported as complete implementation cost.
extern "C" __global__ void c2_batch_tc_qk_mbarrier_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, const int* topk_idx,
    const int* block_table, const int* seq_lens, __nv_bfloat16* caller_output) {
  __shared__ __nv_bfloat16 local_partial[kGqaGroup * kHeadDim];
  __shared__ float local_lse[kGqaGroup];
  __shared__ __align__(8) std::uint64_t producer_ready_barrier;
  __shared__ int producer_ready;
  // q_tile is matrix A MxK row-major.  The physical cache [token][dim] is
  // viewed by WMMA as logical B KxN col-major with ld=128: each B column is
  // one contiguous token D-vector.  Both allocations/strides preserve WMMA's
  // 256-bit alignment requirement.
  __shared__ __align__(32) __nv_bfloat16 q_tile[kGqaGroup][kHeadDim];
  __shared__ __align__(32) float score_tiles[kWmmaTilesPerPage][kGqaGroup][kTokensPerWmmaTile];

  const cg::cluster_group cluster = cg::this_cluster();
  const int role = static_cast<int>(cluster.block_rank());
  const int cluster_index = static_cast<int>(blockIdx.x / kNumCtas);
  const int batch = cluster_index / kKvHeads;
  const int kv_head = cluster_index % kKvHeads;
  const int thread = static_cast<int>(threadIdx.x);
  const int sequence_length = seq_lens[batch];
  const int query_position = sequence_length - 1;

  if (role == 2 && thread == 0) {
    cuda::ptx::mbarrier_init(&producer_ready_barrier, kMBarrierExpectedArrivals);
    producer_ready = 0;
  }
  __syncthreads();
  cluster.sync();

  if (role == 0 || role == 1) {
    const int lane = thread & (kBatchWarpSize - 1);
    const int warp = thread / kBatchWarpSize;
    const int dim0 = lane;
    const int dim1 = lane + kBatchWarpSize;
    const int dim2 = lane + 2 * kBatchWarpSize;
    const int dim3 = lane + 3 * kBatchWarpSize;
    const int selected_begin = role * kPagesPerProducer;

    for (int element = thread; element < kGqaGroup * kHeadDim; element += kThreadsPerBlock) {
      const int group_head = element / kHeadDim;
      const int dim = element % kHeadDim;
      const int query_head = kv_head * kGqaGroup + group_head;
      q_tile[group_head][dim] =
          query[(static_cast<std::size_t>(batch) * kQueryHeads + query_head) * kHeadDim + dim];
    }
    __syncthreads();

    float acc0[kBatchHeadsPerWarp] = {0.0f, 0.0f};
    float acc1[kBatchHeadsPerWarp] = {0.0f, 0.0f};
    float acc2[kBatchHeadsPerWarp] = {0.0f, 0.0f};
    float acc3[kBatchHeadsPerWarp] = {0.0f, 0.0f};
    float max_score[kBatchHeadsPerWarp] = {-INFINITY, -INFINITY};
    float normalizer[kBatchHeadsPerWarp] = {0.0f, 0.0f};

    for (int selected = selected_begin; selected < selected_begin + kPagesPerProducer; ++selected) {
      const int logical_page = topk_idx[(static_cast<std::size_t>(batch) * kKvHeads + kv_head) *
                                        kSelectedPages + selected];
      const int physical_page = block_table[static_cast<std::size_t>(batch) * kLogicalPages + logical_page];
      const int token_begin = warp * kTokensPerWmmaTile;
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
        wmma::load_matrix_sync(b, key_cache + tile_base + k_offset, kHeadDim);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(&score_tiles[warp][0][0], c, kTokensPerWmmaTile, wmma::mem_row_major);
      __syncthreads();

#pragma unroll
      for (int head_in_warp = 0; head_in_warp < kBatchHeadsPerWarp; ++head_in_warp) {
        const int group_head = warp * kBatchHeadsPerWarp + head_in_warp;
        for (int tile = 0; tile < kWmmaTilesPerPage; ++tile) {
#pragma unroll
          for (int token_in_tile = 0; token_in_tile < kTokensPerWmmaTile; ++token_in_tile) {
            const int token = tile * kTokensPerWmmaTile + token_in_tile;
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
      // score_tiles is overwritten for the next selected page only after all
      // eight warps have consumed this page's rows.
      __syncthreads();
    }

#pragma unroll
    for (int head_in_warp = 0; head_in_warp < kBatchHeadsPerWarp; ++head_in_warp) {
      const int group_head = warp * kBatchHeadsPerWarp + head_in_warp;
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
    const std::size_t output_base =
        (static_cast<std::size_t>(batch) * kQueryHeads + kv_head * kGqaGroup + thread) * kHeadDim;
    if (producer_ready == 0) {
#pragma unroll
      for (int dim = 0; dim < kHeadDim; ++dim) caller_output[output_base + dim] = __float2bfloat16_rn(kSentinel);
    } else {
      const __nv_bfloat16* remote_partial0 = cluster.map_shared_rank(local_partial, 0);
      const __nv_bfloat16* remote_partial1 = cluster.map_shared_rank(local_partial, 1);
      const float* remote_lse0 = cluster.map_shared_rank(local_lse, 0);
      const float* remote_lse1 = cluster.map_shared_rank(local_lse, 1);
      const float lse0 = remote_lse0[thread], lse1 = remote_lse1[thread];
      const float lse_max = fmaxf(lse0, lse1);
      const float weight0 = isfinite(lse0) ? exp2f(lse0 - lse_max) : 0.0f;
      const float weight1 = isfinite(lse1) ? exp2f(lse1 - lse_max) : 0.0f;
      const float denominator = weight0 + weight1;
#pragma unroll
      for (int dim = 0; dim < kHeadDim; ++dim) {
        const float partial0 = __bfloat162float(remote_partial0[thread * kHeadDim + dim]);
        const float partial1 = __bfloat162float(remote_partial1[thread * kHeadDim + dim]);
        caller_output[output_base + dim] = __float2bfloat16_rn(
            denominator > 0.0f ? (partial0 * weight0 + partial1 * weight1) / denominator : 0.0f);
      }
    }
  }
  cluster.sync();
}

struct BatchInput {
  int batch = 0;
  int seed = 0;
  std::vector<__nv_bfloat16> query;
  std::vector<__nv_bfloat16> key_cache;
  std::vector<__nv_bfloat16> value_cache;
  std::vector<int> topk_idx;
  std::vector<int> block_table;
  std::vector<int> seq_lens;
  int adversarial_unselected_visible_pages = 0;
  int adversarial_masked_tokens = 0;
  bool disjoint_page_pools = false;
  bool topk_row_order_differs_from_first = false;
};

__host__ __device__ __forceinline__ std::size_t batch_output_offset(int batch, int query_head, int dim) {
  return (static_cast<std::size_t>(batch) * kQueryHeads + query_head) * kHeadDim + dim;
}

BatchInput make_batch_input(int batch, int seed) {
  BatchInput input{};
  input.batch = batch;
  input.seed = seed;
  const std::size_t cache_elements = static_cast<std::size_t>(batch) * kPhysicalPages * kKvHeads * kPageSize * kHeadDim;
  input.query.resize(static_cast<std::size_t>(batch) * kQueryHeads * kHeadDim);
  input.key_cache.resize(cache_elements);
  input.value_cache.resize(cache_elements);
  input.topk_idx.resize(static_cast<std::size_t>(batch) * kKvHeads * kSelectedPages);
  input.block_table.resize(static_cast<std::size_t>(batch) * kLogicalPages);
  input.seq_lens.resize(batch);
  std::mt19937 generator(static_cast<std::mt19937::result_type>(seed));
  std::uniform_real_distribution<float> values(-0.25f, 0.25f);
  for (auto& item : input.query) item = __float2bfloat16_rn(values(generator));
  for (auto& item : input.key_cache) item = __float2bfloat16_rn(values(generator));
  for (auto& item : input.value_cache) item = __float2bfloat16_rn(values(generator));

  std::vector<int> first_topk(kSelectedPages, -1);
  bool all_pools_disjoint = true;
  bool row_order_differs = batch * kKvHeads > 1;
  for (int b = 0; b < batch; ++b) {
    // As in the audited B=1 case, there are at least 16 fully visible pages
    // plus one final page whose token zero alone is causal.  `visible_pages`
    // below includes that partially visible page.
    const int full_visible_pages = ((seed + 3 * b) & 1) != 0 ? kSelectedPages : kLogicalPages - 1;
    const int visible_pages = full_visible_pages + 1;
    input.seq_lens[b] = full_visible_pages * kPageSize + 1;
    std::vector<int> physical(kPhysicalPages);
    std::iota(physical.begin(), physical.end(), 0);
    std::shuffle(physical.begin(), physical.end(), generator);
    for (int logical = 0; logical < kLogicalPages; ++logical) {
      const int physical_page = b * kPhysicalPages + physical[logical];
      input.block_table[static_cast<std::size_t>(b) * kLogicalPages + logical] = physical_page;
      all_pools_disjoint = all_pools_disjoint && physical_page >= b * kPhysicalPages &&
                           physical_page < (b + 1) * kPhysicalPages;
    }
    for (int kv = 0; kv < kKvHeads; ++kv) {
      std::vector<int> logical(visible_pages);
      std::iota(logical.begin(), logical.end(), 0);
      std::shuffle(logical.begin(), logical.end(), generator);
      // The last causal page must be selected to guarantee an adversarial masked tail.
      const int last = visible_pages - 1;
      if (std::find(logical.begin(), logical.begin() + kSelectedPages, last) == logical.begin() + kSelectedPages) {
        auto where = std::find(logical.begin() + kSelectedPages, logical.end(), last);
        std::iter_swap(logical.begin() + kSelectedPages - 1, where);
      }
      std::vector<bool> selected(kLogicalPages, false);
      for (int selected_index = 0; selected_index < kSelectedPages; ++selected_index) {
        const int logical_page = logical[selected_index];
        input.topk_idx[(static_cast<std::size_t>(b) * kKvHeads + kv) * kSelectedPages + selected_index] = logical_page;
        selected[logical_page] = true;
        if (b == 0 && kv == 0) first_topk[selected_index] = logical_page;
      }
      if (b != 0 || kv != 0) {
        bool differs = false;
        for (int selected_index = 0; selected_index < kSelectedPages; ++selected_index) {
          differs = differs || first_topk[selected_index] !=
              input.topk_idx[(static_cast<std::size_t>(b) * kKvHeads + kv) * kSelectedPages + selected_index];
        }
        row_order_differs = row_order_differs && differs;
      }
      // Poison every visible but unselected logical page.  The paired-seed
      // selected-set-signature gate below makes every (batch, KV-head) row
      // distinguishable across the two correctness passes; poison then makes
      // an exercised wrong-row or cross-batch/head read adversarial rather
      // than claiming one seed alone has globally unique rows.
      for (int logical_page = 0; logical_page < visible_pages; ++logical_page) {
        if (!selected[logical_page]) {
          ++input.adversarial_unselected_visible_pages;
          const int physical_page = input.block_table[static_cast<std::size_t>(b) * kLogicalPages + logical_page];
          for (int token = 0; token < kPageSize; ++token) {
            const std::size_t base = cache_offset(physical_page, kv, token, 0);
            for (int dim = 0; dim < kHeadDim; ++dim) {
              input.key_cache[base + dim] = __float2bfloat16_rn(7.0f + 0.001f * dim);
              input.value_cache[base + dim] = __float2bfloat16_rn(-5.0f + 0.002f * dim);
            }
          }
        }
      }
      const int last_physical = input.block_table[static_cast<std::size_t>(b) * kLogicalPages + last];
      for (int token = 1; token < kPageSize; ++token) {
        const std::size_t base = cache_offset(last_physical, kv, token, 0);
        for (int dim = 0; dim < kHeadDim; ++dim) {
          input.key_cache[base + dim] = __float2bfloat16_rn(9.0f + 0.003f * dim);
          input.value_cache[base + dim] = __float2bfloat16_rn(-8.0f + 0.004f * dim);
        }
        ++input.adversarial_masked_tokens;
      }
    }
  }
  input.disjoint_page_pools = all_pools_disjoint;
  input.topk_row_order_differs_from_first = row_order_differs;
  return input;
}

bool validate_batch_indirection(const BatchInput& input) {
  if (input.batch <= 0 || input.query.size() != static_cast<std::size_t>(input.batch) * kQueryHeads * kHeadDim ||
      input.block_table.size() != static_cast<std::size_t>(input.batch) * kLogicalPages ||
      input.topk_idx.size() != static_cast<std::size_t>(input.batch) * kKvHeads * kSelectedPages ||
      input.seq_lens.size() != static_cast<std::size_t>(input.batch)) return false;
  const std::size_t expected_cache = static_cast<std::size_t>(input.batch) * kPhysicalPages * kKvHeads * kPageSize * kHeadDim;
  if (input.key_cache.size() != expected_cache || input.value_cache.size() != expected_cache) return false;
  for (int b = 0; b < input.batch; ++b) {
    const int visible = (input.seq_lens[b] + kPageSize - 1) / kPageSize;
    if (visible <= kSelectedPages || visible > kLogicalPages || input.seq_lens[b] != (visible - 1) * kPageSize + 1) return false;
    std::vector<bool> physical_seen(kPhysicalPages, false);
    for (int logical = 0; logical < kLogicalPages; ++logical) {
      const int physical = input.block_table[static_cast<std::size_t>(b) * kLogicalPages + logical];
      if (physical < b * kPhysicalPages || physical >= (b + 1) * kPhysicalPages || physical_seen[physical % kPhysicalPages]) return false;
      physical_seen[physical % kPhysicalPages] = true;
    }
    for (int kv = 0; kv < kKvHeads; ++kv) {
      std::vector<bool> selected(kLogicalPages, false);
      for (int index = 0; index < kSelectedPages; ++index) {
        const int logical = input.topk_idx[(static_cast<std::size_t>(b) * kKvHeads + kv) * kSelectedPages + index];
        if (logical < 0 || logical >= visible || selected[logical]) return false;
        selected[logical] = true;
      }
      if (!selected[visible - 1]) return false;
    }
  }
  return input.disjoint_page_pools && input.topk_row_order_differs_from_first &&
         input.adversarial_unselected_visible_pages > 0 &&
         input.adversarial_masked_tokens == input.batch * kKvHeads * (kPageSize - 1);
}

std::string sorted_selected_set_signature(const BatchInput& input, int batch, int kv_head) {
  std::vector<int> pages(kSelectedPages);
  for (int selected = 0; selected < kSelectedPages; ++selected) {
    pages[selected] = input.topk_idx[(static_cast<std::size_t>(batch) * kKvHeads + kv_head) *
                                     kSelectedPages + selected];
  }
  std::sort(pages.begin(), pages.end());
  std::ostringstream stream;
  for (int page : pages) stream << page << ',';
  return stream.str();
}

// A single seed cannot supply unique selected sets for every B=16, Hkv=4
// row: the smallest visible range has only 17 choose 16 possible sets.  The
// correctness contract instead checks the ordered pair of sorted selected
// sets from seed 17 and seed 2026.  A collision is a hard failure before any
// launch, never a weakened coverage claim.
bool paired_seed_selected_set_signatures_unique(const BatchInput& seed17,
                                                const BatchInput& seed2026) {
  if (seed17.batch != seed2026.batch) return false;
  std::vector<std::string> seen;
  seen.reserve(static_cast<std::size_t>(seed17.batch) * kKvHeads);
  for (int batch = 0; batch < seed17.batch; ++batch) {
    for (int kv_head = 0; kv_head < kKvHeads; ++kv_head) {
      const std::string pair_signature = sorted_selected_set_signature(seed17, batch, kv_head) + "|" +
                                         sorted_selected_set_signature(seed2026, batch, kv_head);
      if (std::find(seen.begin(), seen.end(), pair_signature) != seen.end()) return false;
      seen.push_back(pair_signature);
    }
  }
  return true;
}

std::vector<float> batch_fp64_oracle(const BatchInput& input) {
  std::vector<float> output(static_cast<std::size_t>(input.batch) * kQueryHeads * kHeadDim, 0.0f);
  const double inv_sqrt = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  for (int b = 0; b < input.batch; ++b) {
    const int query_position = input.seq_lens[b] - 1;
    for (int kv = 0; kv < kKvHeads; ++kv) for (int group = 0; group < kGqaGroup; ++group) {
      const int query_head = kv * kGqaGroup + group;
      const __nv_bfloat16* query_row =
          input.query.data() + (static_cast<std::size_t>(b) * kQueryHeads + query_head) * kHeadDim;
      double max_score = -std::numeric_limits<double>::infinity();
      for (int selected = 0; selected < kSelectedPages; ++selected) {
        const int logical = input.topk_idx[(static_cast<std::size_t>(b) * kKvHeads + kv) * kSelectedPages + selected];
        const int physical = input.block_table[static_cast<std::size_t>(b) * kLogicalPages + logical];
        for (int token = 0; token < kPageSize; ++token) if (logical * kPageSize + token <= query_position) {
          const std::size_t base = cache_offset(physical, kv, token, 0);
          double score = 0.0;
          for (int dim = 0; dim < kHeadDim; ++dim) {
            score = std::fma(static_cast<double>(__bfloat162float(query_row[dim])),
                             static_cast<double>(__bfloat162float(input.key_cache[base + dim])), score);
          }
          max_score = std::max(max_score, score * inv_sqrt);
        }
      }
      double normalizer = 0.0;
      double accumulator[kHeadDim]{};
      for (int selected = 0; selected < kSelectedPages; ++selected) {
        const int logical = input.topk_idx[(static_cast<std::size_t>(b) * kKvHeads + kv) * kSelectedPages + selected];
        const int physical = input.block_table[static_cast<std::size_t>(b) * kLogicalPages + logical];
        for (int token = 0; token < kPageSize; ++token) if (logical * kPageSize + token <= query_position) {
          const std::size_t base = cache_offset(physical, kv, token, 0);
          double score = 0.0;
          for (int dim = 0; dim < kHeadDim; ++dim) {
            score = std::fma(static_cast<double>(__bfloat162float(query_row[dim])),
                             static_cast<double>(__bfloat162float(input.key_cache[base + dim])), score);
          }
          const double weight = std::exp(score * inv_sqrt - max_score);
          normalizer += weight;
          for (int dim = 0; dim < kHeadDim; ++dim) accumulator[dim] += weight * __bfloat162float(input.value_cache[base + dim]);
        }
      }
      if (!(normalizer > 0.0)) throw std::runtime_error("batch FP64 oracle observed empty selected causal range");
      for (int dim = 0; dim < kHeadDim; ++dim) output[batch_output_offset(b, query_head, dim)] =
          static_cast<float>(accumulator[dim] / normalizer);
    }
  }
  return output;
}

struct BatchBuffers {
  __nv_bfloat16 *query = nullptr, *key_cache = nullptr, *value_cache = nullptr;
  int *topk_idx = nullptr, *block_table = nullptr, *seq_lens = nullptr;
  __nv_bfloat16 *control_output = nullptr, *candidate_output = nullptr;
  BatchBuffers() = default;
  BatchBuffers(const BatchBuffers&) = delete;
  BatchBuffers& operator=(const BatchBuffers&) = delete;
  ~BatchBuffers() { release(); }
  void release() noexcept {
    cudaFree(candidate_output); cudaFree(control_output); cudaFree(seq_lens); cudaFree(block_table); cudaFree(topk_idx);
    cudaFree(value_cache); cudaFree(key_cache); cudaFree(query);
    query = key_cache = value_cache = control_output = candidate_output = nullptr;
    topk_idx = block_table = seq_lens = nullptr;
  }
};

void allocate_batch(const BatchInput& input, BatchBuffers* buffers) {
  const std::size_t output_count = static_cast<std::size_t>(input.batch) * kQueryHeads * kHeadDim;
  std::vector<__nv_bfloat16> sentinel(output_count, __float2bfloat16_rn(kSentinel));
  CUDA_CHECK(cudaMalloc(&buffers->query, input.query.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers->key_cache, input.key_cache.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers->value_cache, input.value_cache.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers->topk_idx, input.topk_idx.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&buffers->block_table, input.block_table.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&buffers->seq_lens, input.seq_lens.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&buffers->control_output, output_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers->candidate_output, output_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemcpy(buffers->query, input.query.data(), input.query.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->key_cache, input.key_cache.data(), input.key_cache.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->value_cache, input.value_cache.data(), input.value_cache.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->topk_idx, input.topk_idx.data(), input.topk_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->block_table, input.block_table.data(), input.block_table.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->seq_lens, input.seq_lens.data(), input.seq_lens.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->control_output, sentinel.data(), output_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->candidate_output, sentinel.data(), output_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
}

void refill_batch_outputs_with_sentinel(const BatchInput& input, const BatchBuffers& buffers) {
  const std::size_t output_count = static_cast<std::size_t>(input.batch) * kQueryHeads * kHeadDim;
  const std::vector<__nv_bfloat16> sentinel(output_count, __float2bfloat16_rn(kSentinel));
  CUDA_CHECK(cudaMemcpy(buffers.control_output, sentinel.data(), output_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.candidate_output, sentinel.data(), output_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
}

struct BatchLaunch {
  cudaLaunchAttribute attr{};
  cudaLaunchConfig_t config{};
  explicit BatchLaunch(int batch) {
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {kNumCtas, 1, 1};
    config.gridDim = dim3(batch * kKvHeads * kNumCtas, 1, 1);
    config.blockDim = dim3(kThreadsPerBlock, 1, 1);
    config.dynamicSmemBytes = 0; config.stream = nullptr; config.attrs = &attr; config.numAttrs = 1;
  }
};

void launch_batch_warp_control(const BatchLaunch& launch, const BatchBuffers& buffers) {
  CUDA_CHECK(cudaLaunchKernelEx(&launch.config, c2_batch_warp_control_mbarrier_kernel, buffers.query, buffers.key_cache,
                                 buffers.value_cache, buffers.topk_idx, buffers.block_table, buffers.seq_lens,
                                 buffers.control_output));
  CUDA_CHECK(cudaGetLastError());
}
void launch_batch_tc_qk(const BatchLaunch& launch, const BatchBuffers& buffers) {
  CUDA_CHECK(cudaLaunchKernelEx(&launch.config, c2_batch_tc_qk_mbarrier_kernel, buffers.query, buffers.key_cache,
                                 buffers.value_cache, buffers.topk_idx, buffers.block_table, buffers.seq_lens,
                                 buffers.candidate_output));
  CUDA_CHECK(cudaGetLastError());
}

struct ArmCheck { float max_abs = 0.0f, max_rel = 0.0f; bool oracle_finite = true, finite = true, sentinel_clean = true, allclose = true; };
struct CrossCheck { float max_abs = 0.0f, max_rel = 0.0f; bool bfloat16_bitwise_equal = true; };
ArmCheck check_arm(const std::vector<__nv_bfloat16>& output, const std::vector<float>& oracle) {
  if (output.size() != oracle.size()) throw std::runtime_error("batch output/oracle shape mismatch");
  const __nv_bfloat16 sentinel = __float2bfloat16_rn(kSentinel);
  ArmCheck result{};
  for (std::size_t i = 0; i < output.size(); ++i) {
    const float actual = __bfloat162float(output[i]), expected = oracle[i];
    const float abs_error = std::fabs(actual - expected);
    result.max_abs = std::max(result.max_abs, abs_error);
    result.max_rel = std::max(result.max_rel, abs_error / std::max(std::fabs(expected), 1.0e-7f));
    result.oracle_finite = result.oracle_finite && std::isfinite(expected);
    result.finite = result.finite && std::isfinite(actual);
    result.sentinel_clean = result.sentinel_clean && !same_bfloat16_bits(output[i], sentinel);
    result.allclose = result.allclose && abs_error <= kAtol + kRtol * std::fabs(expected);
  }
  return result;
}
CrossCheck diagnose_batch_cross(const std::vector<__nv_bfloat16>& control, const std::vector<__nv_bfloat16>& candidate) {
  CrossCheck result{};
  if (control.size() != candidate.size()) throw std::runtime_error("batch control/candidate shape mismatch");
  for (std::size_t i = 0; i < control.size(); ++i) {
    const float lhs = __bfloat162float(control[i]), rhs = __bfloat162float(candidate[i]);
    const float abs_error = std::fabs(lhs - rhs);
    result.max_abs = std::max(result.max_abs, abs_error);
    result.max_rel = std::max(result.max_rel, abs_error / std::max(std::fabs(lhs), 1.0e-7f));
    result.bfloat16_bitwise_equal = result.bfloat16_bitwise_equal && same_bfloat16_bits(control[i], candidate[i]);
  }
  return result;
}
bool good(const ArmCheck& check) { return check.oracle_finite && check.finite && check.sentinel_clean && check.allclose; }

struct SeedCheck { int seed = 0; int batch = 0; bool hierarchy_valid = false; int unselected = 0, masked = 0; ArmCheck control{}, candidate{}; CrossCheck cross{}; };
SeedCheck validate_seed(const BatchInput& input, const BatchLaunch& launch, BatchBuffers* retained) {
  if (!validate_batch_indirection(input)) throw std::runtime_error("batch indirection/poison validation failed");
  const std::vector<float> oracle = batch_fp64_oracle(input);
  BatchBuffers temporary;
  BatchBuffers* buffers = input.seed == kBatchTimingSeed ? retained : &temporary;
  allocate_batch(input, buffers);
  launch_batch_warp_control(launch, *buffers); launch_batch_tc_qk(launch, *buffers); CUDA_CHECK(cudaDeviceSynchronize());
  const std::size_t count = oracle.size();
  std::vector<__nv_bfloat16> control(count), candidate(count);
  CUDA_CHECK(cudaMemcpy(control.data(), buffers->control_output, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(candidate.data(), buffers->candidate_output, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  return SeedCheck{input.seed, input.batch, true, input.adversarial_unselected_visible_pages, input.adversarial_masked_tokens,
                   check_arm(control, oracle), check_arm(candidate, oracle), diagnose_batch_cross(control, candidate)};
}

float time_warp_control(const BatchLaunch& launch, const BatchBuffers& buffers, cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start)); launch_batch_warp_control(launch, buffers); CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
  float milliseconds = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end)); return milliseconds * 1000.0f;
}
float time_tc_qk(const BatchLaunch& launch, const BatchBuffers& buffers, cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start)); launch_batch_tc_qk(launch, buffers); CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
  float milliseconds = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end)); return milliseconds * 1000.0f;
}
struct Stats { float p10_us = 0.0f, median_us = 0.0f, p90_us = 0.0f; };
Stats stats(std::vector<float> values) {
  if (values.empty()) throw std::runtime_error("empty timing samples");
  std::sort(values.begin(), values.end()); const std::size_t n = values.size();
  return {values[std::max<std::size_t>(0, static_cast<std::size_t>(std::ceil(.10 * n)) - 1)],
          n % 2 ? values[n / 2] : 0.5f * (values[n / 2 - 1] + values[n / 2]),
          values[std::min(n - 1, static_cast<std::size_t>(std::ceil(.90 * n)) - 1)]};
}
struct Timing { std::vector<float> control_ab, control_ba, candidate_ab, candidate_ba; Stats control_all{}, control_ab_stats{}, control_ba_stats{}, candidate_all{}, candidate_ab_stats{}, candidate_ba_stats{}; float speedup = 0.0f, ab_speedup = 0.0f, ba_speedup = 0.0f; bool combined = false, ab = false, ba = false, promoted = false; };
Timing benchmark(const BatchLaunch& launch, const BatchBuffers& buffers, bool all_correct, int candidate_local_bytes) {
  cudaEvent_t start = nullptr, end = nullptr; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&end));
  Timing result{};
  try {
    for (int i = 0; i < kBatchWarmupEach; ++i) { launch_batch_warp_control(launch, buffers); launch_batch_tc_qk(launch, buffers); }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < kBatchAbbapairs; ++i) {
      result.control_ab.push_back(time_warp_control(launch, buffers, start, end));
      result.candidate_ab.push_back(time_tc_qk(launch, buffers, start, end));
      result.candidate_ba.push_back(time_tc_qk(launch, buffers, start, end));
      result.control_ba.push_back(time_warp_control(launch, buffers, start, end));
    }
    std::vector<float> control_all = result.control_ab; control_all.insert(control_all.end(), result.control_ba.begin(), result.control_ba.end());
    std::vector<float> candidate_all = result.candidate_ab; candidate_all.insert(candidate_all.end(), result.candidate_ba.begin(), result.candidate_ba.end());
    result.control_all = stats(control_all); result.candidate_all = stats(candidate_all);
    result.control_ab_stats = stats(result.control_ab); result.control_ba_stats = stats(result.control_ba);
    result.candidate_ab_stats = stats(result.candidate_ab); result.candidate_ba_stats = stats(result.candidate_ba);
    result.speedup = result.control_all.median_us / result.candidate_all.median_us;
    result.ab_speedup = result.control_ab_stats.median_us / result.candidate_ab_stats.median_us;
    result.ba_speedup = result.control_ba_stats.median_us / result.candidate_ba_stats.median_us;
    result.combined = result.speedup >= 1.10f; result.ab = result.ab_speedup > 1.05f; result.ba = result.ba_speedup > 1.05f;
    result.promoted = result.combined && result.ab && result.ba && all_correct && candidate_local_bytes == 0;
  } catch (...) { cudaEventDestroy(end); cudaEventDestroy(start); throw; }
  CUDA_CHECK(cudaEventDestroy(end)); CUDA_CHECK(cudaEventDestroy(start)); return result;
}

struct CaseResult {
  int batch = 0;
  bool paired_seed_selected_set_signatures_unique = false;
  SeedCheck seed17{}, seed2026{}, post{};
  Timing timing{};
};
void print_stats(const Stats& value) { std::cout << "{\"p10_us\":" << value.p10_us << ",\"median_us\":" << value.median_us << ",\"p90_us\":" << value.p90_us << '}'; }
void print_samples(const std::vector<float>& values) { std::cout << '['; for (std::size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; } std::cout << ']'; }
void print_arm(const ArmCheck& value) { std::cout << "{\"max_abs\":" << value.max_abs << ",\"max_rel\":" << value.max_rel << ",\"oracle_finite\":" << (value.oracle_finite ? "true" : "false") << ",\"finite\":" << (value.finite ? "true" : "false") << ",\"sentinel_clean\":" << (value.sentinel_clean ? "true" : "false") << ",\"allclose\":" << (value.allclose ? "true" : "false") << '}'; }
void print_seed(const SeedCheck& value) { std::cout << "{\"seed\":" << value.seed << ",\"batch\":" << value.batch << ",\"hierarchy_valid\":" << (value.hierarchy_valid ? "true" : "false") << ",\"adversarial_unselected_visible_pages\":" << value.unselected << ",\"adversarial_masked_tokens\":" << value.masked << ",\"warp_control\":"; print_arm(value.control); std::cout << ",\"tc_qk_candidate\":"; print_arm(value.candidate); std::cout << ",\"cross_arm\":{\"max_abs\":" << value.cross.max_abs << ",\"max_rel\":" << value.cross.max_rel << ",\"bfloat16_bitwise_equal\":" << (value.cross.bfloat16_bitwise_equal ? "true" : "false") << "}}"; }
void print_case(const CaseResult& value) {
  const Timing& t = value.timing;
  std::cout << "{\"B\":" << value.batch << ",\"paired_seed_selected_set_signatures_unique\":" << (value.paired_seed_selected_set_signatures_unique ? "true" : "false") << ",\"correctness\":["; print_seed(value.seed17); std::cout << ','; print_seed(value.seed2026); std::cout << "],\"post_timing_correctness\":"; print_seed(value.post);
  std::cout << ",\"timing\":{\"protocol\":\"warmup_each_then_51_warp_control_tc_qk_tc_qk_warp_control_ABBA_pairs\",\"warmup_each\":" << kBatchWarmupEach << ",\"abba_pairs\":" << kBatchAbbapairs << ",\"samples_per_arm\":" << kBatchSamplesPerArm << ",\"raw_samples_us\":{\"warp_control\":{\"AB\":"; print_samples(t.control_ab); std::cout << ",\"BA\":"; print_samples(t.control_ba); std::cout << "},\"tc_qk_candidate\":{\"AB\":"; print_samples(t.candidate_ab); std::cout << ",\"BA\":"; print_samples(t.candidate_ba); std::cout << "}},\"warp_control\":{\"all\":"; print_stats(t.control_all); std::cout << ",\"when_launch_order_is_AB\":"; print_stats(t.control_ab_stats); std::cout << ",\"when_launch_order_is_BA\":"; print_stats(t.control_ba_stats); std::cout << "},\"tc_qk_candidate\":{\"all\":"; print_stats(t.candidate_all); std::cout << ",\"when_launch_order_is_AB\":"; print_stats(t.candidate_ab_stats); std::cout << ",\"when_launch_order_is_BA\":"; print_stats(t.candidate_ba_stats); std::cout << "},\"speedup_warp_control_over_tc_qk\":" << t.speedup << ",\"speedup_by_partition\":{\"AB\":" << t.ab_speedup << ",\"BA\":" << t.ba_speedup << "},\"promotion_gate\":{\"combined_warp_control_over_tc_qk_at_least_1_10\":" << (t.combined ? "true" : "false") << ",\"AB_warp_control_over_tc_qk_greater_than_1_05\":" << (t.ab ? "true" : "false") << ",\"BA_warp_control_over_tc_qk_greater_than_1_05\":" << (t.ba ? "true" : "false") << ",\"all_correct\":" << ((good(value.seed17.control) && good(value.seed17.candidate) && good(value.seed2026.control) && good(value.seed2026.candidate) && good(value.post.control) && good(value.post.candidate)) ? "true" : "false") << ",\"promoted\":" << (t.promoted ? "true" : "false") << "}}}";
}

int c2_batch_main_impl() {
  try {
    int device = 0; CUDA_CHECK(cudaGetDevice(&device)); cudaDeviceProp property{}; CUDA_CHECK(cudaGetDeviceProperties(&property, device));
    int cluster_launch = 0; CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device));
    if (property.major != 10 || property.minor != 3 || cluster_launch == 0) throw std::runtime_error("requires B300 CC 10.3 cluster launch");
    cudaFuncAttributes control_attr{}, candidate_attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&control_attr, c2_batch_warp_control_mbarrier_kernel));
    CUDA_CHECK(cudaFuncGetAttributes(&candidate_attr, c2_batch_tc_qk_mbarrier_kernel));
    if (control_attr.sharedSizeBytes > property.sharedMemPerBlock ||
        candidate_attr.sharedSizeBytes > property.sharedMemPerBlock) {
      throw std::runtime_error("batch control or TC-QK candidate exceeds static shared-memory fit");
    }
    std::vector<CaseResult> cases; cases.reserve(kBatchCaseCount);
    for (int batch : kBatchCases) {
      BatchLaunch launch(batch); BatchBuffers retained;
      const BatchInput seed17_input = make_batch_input(batch, 17); const BatchInput seed2026_input = make_batch_input(batch, kBatchTimingSeed);
      CaseResult result{}; result.batch = batch;
      result.paired_seed_selected_set_signatures_unique =
          paired_seed_selected_set_signatures_unique(seed17_input, seed2026_input);
      if (!result.paired_seed_selected_set_signatures_unique) {
        throw std::runtime_error("paired seed selected-set signature collision; refusing weakened row-coverage claim");
      }
      result.seed17 = validate_seed(seed17_input, launch, &retained);
      result.seed2026 = validate_seed(seed2026_input, launch, &retained);
      if (!good(result.seed17.control) || !good(result.seed17.candidate) ||
          !good(result.seed2026.control) || !good(result.seed2026.candidate)) {
        throw std::runtime_error("pre-timing batch correctness gate failed");
      }
      result.timing = benchmark(launch, retained, true, candidate_attr.localSizeBytes);
      // Do not accept a copy of either last timed output as the post-timing
      // evidence.  Reinstall the caller-visible BF16 sentinel first, launch
      // each arm afresh once, synchronize, and only then copy for the oracle.
      refill_batch_outputs_with_sentinel(seed2026_input, retained);
      launch_batch_warp_control(launch, retained); CUDA_CHECK(cudaDeviceSynchronize());
      launch_batch_tc_qk(launch, retained); CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<float> oracle = batch_fp64_oracle(seed2026_input);
      std::vector<__nv_bfloat16> control(oracle.size()), candidate(oracle.size());
      CUDA_CHECK(cudaMemcpy(control.data(), retained.control_output, control.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(candidate.data(), retained.candidate_output, candidate.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
      result.post = SeedCheck{kBatchTimingSeed, batch, validate_batch_indirection(seed2026_input), seed2026_input.adversarial_unselected_visible_pages, seed2026_input.adversarial_masked_tokens, check_arm(control, oracle), check_arm(candidate, oracle), diagnose_batch_cross(control, candidate)};
      if (!good(result.post.control) || !good(result.post.candidate)) throw std::runtime_error("post-timing batch correctness gate failed");
      cases.push_back(std::move(result));
    }
    std::cout << std::setprecision(9) << "{\"schema\":\"c2-cluster-attention-tc-qk-batch-abba-v1\",\"status\":\"pass\",\"boundary\":\"" << json_escape(kBatchBoundary) << "\",\"timing_seed\":" << kBatchTimingSeed << ",\"batch_cases\":[1,4,8,16],\"shape\":{\"Hkv\":4,\"Hq\":64,\"G\":16,\"D\":128,\"page_size\":128,\"selected_pages\":16,\"logical_pages\":32,\"physical_pages_per_batch\":32},\"abi\":{\"query\":\"[B,Hq,D]\",\"output\":\"[B,Hq,D]\",\"seq_lens\":\"[B]\",\"block_table\":\"[B,max_blocks]\",\"topk\":\"[B,Hkv,Ktop]\",\"cache\":\"[physical_page,Hkv,P,D]\",\"cluster_mapping\":\"batch=(blockIdx.x/4)/Hkv; kv_head=(blockIdx.x/4)%Hkv\",\"disjoint_physical_page_pool_per_batch\":true,\"topk_row_order_differs_from_first\":true,\"topk_row_order_scope\":\"each nonzero (batch,kv_head) ordered row differs from (0,0); paired sorted-set signatures provide the stronger two-seed row-coverage gate\"},\"provenance\":{\"scalar_protocol\":\"c2_cluster_attention_mbarrier_smoke.cu is preprocessor-included and SHA-pinned\",\"warp_mapping_reference\":\"c2_cluster_attention_warp_producer_abba.cu is SHA-pinned audited reference only; its mapping is derived here and it is not preprocessor-included\",\"tc_qk_design_reference\":\"c2_cluster_attention_tc_qk_abba.cu is SHA-pinned by the runner; its QK mapping is adapted here to the batch ABI\"},\"cluster_layout\":{\"num_ctas\":4,\"grid\":\"B*Hkv*4\",\"selected_pages_per_producer\":8,\"threads_per_block\":256},\"producer_contract\":{\"same_remote_dsm_mbarrier_protocol\":true,\"same_output_abi\":true,\"same_launch_shape\":true,\"same_real_selected_causal_attention\":true,\"persistent_device_buffers_outside_timing\":true,\"caller_owned_independent_outputs\":true,\"single_kernel_launch_per_cuda_event_sample\":true,\"ABBA_interleaved\":true,\"initialization_copies_and_oracle_outside_timing\":true,\"changed_field\":\"rank-0/1 producer QK mapping only: warp shuffle control versus BF16 WMMA QK candidate\",\"candidate_extra_shared_and_cta_barriers_included\":true,\"timed_launch_validation_scope\":\"pre-timing two-seed checks plus post-timing fresh sentinel-reset control/candidate launches and oracle recheck; intermediate timed outputs not inspected\"},\"synchronization\":{\"mbarrier_expected_arrivals\":2,\"mbarrier_wait_parity\":0,\"mbarrier_max_polls\":" << kMBarrierMaxPolls << "},\"dtype_contract\":{\"producer_partial\":\"bfloat16\",\"caller_output\":\"bfloat16\",\"tc_qk_accumulator\":\"float32\",\"oracle_accumulator\":\"float64\",\"oracle_softmax\":\"independent two-pass natural-exp direct selected-page causal attention\",\"tolerance\":{\"rtol\":" << kRtol << ",\"atol\":" << kAtol << "}},\"environment\":{\"device\":\"" << json_escape(property.name) << "\",\"capability\":[" << property.major << ',' << property.minor << "],\"cluster_launch_supported\":true},\"resource_model\":{\"shared_equal\":false,\"warp_control\":{\"static_shared_bytes\":" << control_attr.sharedSizeBytes << ",\"num_regs\":" << control_attr.numRegs << ",\"local_bytes\":" << control_attr.localSizeBytes << "},\"tc_qk_candidate\":{\"static_shared_bytes\":" << candidate_attr.sharedSizeBytes << ",\"num_regs\":" << candidate_attr.numRegs << ",\"local_bytes\":" << candidate_attr.localSizeBytes << "}},\"cases\":[";
    for (std::size_t i = 0; i < cases.size(); ++i) { if (i) std::cout << ','; print_case(cases[i]); }
    std::cout << "]}" << std::endl;
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cout << "{\"schema\":\"c2-cluster-attention-tc-qk-batch-abba-v1\",\"status\":\"fail\",\"boundary\":\"" << json_escape(kBatchBoundary) << "\",\"error\":\"" << json_escape(error.what()) << "\"}" << std::endl;
    return EXIT_FAILURE;
  }
}

}  // namespace

int main() { return c2_batch_main_impl(); }
