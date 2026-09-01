// Batched native C=2 producer-mapping AB/BA qualification experiment.
//
// This source imports the audited scalar protocol and defines two new
// batched-ABI kernels below.  The reviewed warp-producer source is an audited,
// hash-pinned reference in the runner only: its mapping is derived here, not
// preprocessor-imported or compiled as a dependency.  The new arms use one
// native cluster launch with grid B*Hkv*4 and differ only in producer
// arithmetic.  It is a native correctness-prototype batched ABI and producer
// mapping signal; it is not a production integration, throughput, model, or
// server claim.

#define main c2_cluster_attention_mbarrier_batch_embedded_main
#include "c2_cluster_attention_mbarrier_smoke.cu"
#undef main

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
    "native correctness-prototype batched ABI plus producer-mapping signal only; "
    "not production integration, throughput, model, or server claim";

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

// The scalar arm is structurally the audited mbarrier kernel with the sole ABI
// changes required for B>1: batch is decoded from the cluster index and every
// query, top-k row, block-table row, sequence length, and output row is batch
// indexed.  The mbarrier, shared layout, rank-2 merge and lifetime are kept
// equal to the warp arm below.
extern "C" __global__ void c2_batch_scalar_mbarrier_kernel(
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

  if ((role == 0 || role == 1) && thread < kGqaGroup) {
    const int group_head = thread;
    const int query_head = kv_head * kGqaGroup + group_head;
    const __nv_bfloat16* query_row =
        query + (static_cast<std::size_t>(batch) * kQueryHeads + query_head) * kHeadDim;
    float accumulator[kHeadDim];
#pragma unroll
    for (int dim = 0; dim < kHeadDim; ++dim) accumulator[dim] = 0.0f;
    float max_score = -INFINITY;
    float normalizer = 0.0f;
    const int selected_begin = role * kPagesPerProducer;
    for (int selected = selected_begin; selected < selected_begin + kPagesPerProducer; ++selected) {
      const int logical_page = topk_idx[(static_cast<std::size_t>(batch) * kKvHeads + kv_head) *
                                        kSelectedPages + selected];
      const int physical_page = block_table[static_cast<std::size_t>(batch) * kLogicalPages + logical_page];
      for (int token = 0; token < kPageSize; ++token) {
        const int key_position = logical_page * kPageSize + token;
        if (key_position <= query_position && key_position < sequence_length) {
          const std::size_t kv_base = cache_offset(physical_page, kv_head, token, 0);
          float score = 0.0f;
#pragma unroll
          for (int dim = 0; dim < kHeadDim; ++dim) {
            score = fmaf(__bfloat162float(query_row[dim]), __bfloat162float(key_cache[kv_base + dim]), score);
          }
          score *= kScaleLog2e;
          const float next_max = fmaxf(max_score, score);
          const float alpha = isfinite(max_score) ? exp2f(max_score - next_max) : 0.0f;
          const float beta = exp2f(score - next_max);
#pragma unroll
          for (int dim = 0; dim < kHeadDim; ++dim) {
            accumulator[dim] = accumulator[dim] * alpha + beta * __bfloat162float(value_cache[kv_base + dim]);
          }
          normalizer = normalizer * alpha + beta;
          max_score = next_max;
        }
      }
    }
    if (normalizer > 0.0f) {
#pragma unroll
      for (int dim = 0; dim < kHeadDim; ++dim) {
        local_partial[group_head * kHeadDim + dim] = __float2bfloat16_rn(accumulator[dim] / normalizer);
      }
      local_lse[group_head] = max_score + log2f(normalizer);
    } else {
#pragma unroll
      for (int dim = 0; dim < kHeadDim; ++dim) {
        local_partial[group_head * kHeadDim + dim] = __float2bfloat16_rn(0.0f);
      }
      local_lse[group_head] = -INFINITY;
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
      const float lse0 = remote_lse0[thread];
      const float lse1 = remote_lse1[thread];
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

// Same launch, DSM, mbarrier, shared allocation, merge and ABI as scalar.
// Only producer arithmetic changes to eight warps x two serial heads, each
// lane owning d={lane,lane+32,lane+64,lane+96}.
extern "C" __global__ void c2_batch_warp_mbarrier_kernel(
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
  __nv_bfloat16 *scalar_output = nullptr, *warp_output = nullptr;
  BatchBuffers() = default;
  BatchBuffers(const BatchBuffers&) = delete;
  BatchBuffers& operator=(const BatchBuffers&) = delete;
  ~BatchBuffers() { release(); }
  void release() noexcept {
    cudaFree(warp_output); cudaFree(scalar_output); cudaFree(seq_lens); cudaFree(block_table); cudaFree(topk_idx);
    cudaFree(value_cache); cudaFree(key_cache); cudaFree(query);
    query = key_cache = value_cache = scalar_output = warp_output = nullptr;
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
  CUDA_CHECK(cudaMalloc(&buffers->scalar_output, output_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&buffers->warp_output, output_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemcpy(buffers->query, input.query.data(), input.query.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->key_cache, input.key_cache.data(), input.key_cache.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->value_cache, input.value_cache.data(), input.value_cache.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->topk_idx, input.topk_idx.data(), input.topk_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->block_table, input.block_table.data(), input.block_table.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->seq_lens, input.seq_lens.data(), input.seq_lens.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->scalar_output, sentinel.data(), output_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers->warp_output, sentinel.data(), output_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
}

void refill_batch_outputs_with_sentinel(const BatchInput& input, const BatchBuffers& buffers) {
  const std::size_t output_count = static_cast<std::size_t>(input.batch) * kQueryHeads * kHeadDim;
  const std::vector<__nv_bfloat16> sentinel(output_count, __float2bfloat16_rn(kSentinel));
  CUDA_CHECK(cudaMemcpy(buffers.scalar_output, sentinel.data(), output_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buffers.warp_output, sentinel.data(), output_count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
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

void launch_batch_scalar(const BatchLaunch& launch, const BatchBuffers& buffers) {
  CUDA_CHECK(cudaLaunchKernelEx(&launch.config, c2_batch_scalar_mbarrier_kernel, buffers.query, buffers.key_cache,
                                 buffers.value_cache, buffers.topk_idx, buffers.block_table, buffers.seq_lens,
                                 buffers.scalar_output));
  CUDA_CHECK(cudaGetLastError());
}
void launch_batch_warp(const BatchLaunch& launch, const BatchBuffers& buffers) {
  CUDA_CHECK(cudaLaunchKernelEx(&launch.config, c2_batch_warp_mbarrier_kernel, buffers.query, buffers.key_cache,
                                 buffers.value_cache, buffers.topk_idx, buffers.block_table, buffers.seq_lens,
                                 buffers.warp_output));
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
CrossCheck diagnose_batch_cross(const std::vector<__nv_bfloat16>& scalar, const std::vector<__nv_bfloat16>& warp) {
  CrossCheck result{};
  if (scalar.size() != warp.size()) throw std::runtime_error("batch scalar/warp shape mismatch");
  for (std::size_t i = 0; i < scalar.size(); ++i) {
    const float lhs = __bfloat162float(scalar[i]), rhs = __bfloat162float(warp[i]);
    const float abs_error = std::fabs(lhs - rhs);
    result.max_abs = std::max(result.max_abs, abs_error);
    result.max_rel = std::max(result.max_rel, abs_error / std::max(std::fabs(lhs), 1.0e-7f));
    result.bfloat16_bitwise_equal = result.bfloat16_bitwise_equal && same_bfloat16_bits(scalar[i], warp[i]);
  }
  return result;
}
bool good(const ArmCheck& check) { return check.oracle_finite && check.finite && check.sentinel_clean && check.allclose; }

struct SeedCheck { int seed = 0; int batch = 0; bool hierarchy_valid = false; int unselected = 0, masked = 0; ArmCheck scalar{}, warp{}; CrossCheck cross{}; };
SeedCheck validate_seed(const BatchInput& input, const BatchLaunch& launch, BatchBuffers* retained) {
  if (!validate_batch_indirection(input)) throw std::runtime_error("batch indirection/poison validation failed");
  const std::vector<float> oracle = batch_fp64_oracle(input);
  BatchBuffers temporary;
  BatchBuffers* buffers = input.seed == kBatchTimingSeed ? retained : &temporary;
  allocate_batch(input, buffers);
  launch_batch_scalar(launch, *buffers); launch_batch_warp(launch, *buffers); CUDA_CHECK(cudaDeviceSynchronize());
  const std::size_t count = oracle.size();
  std::vector<__nv_bfloat16> scalar(count), warp(count);
  CUDA_CHECK(cudaMemcpy(scalar.data(), buffers->scalar_output, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(warp.data(), buffers->warp_output, count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  return SeedCheck{input.seed, input.batch, true, input.adversarial_unselected_visible_pages, input.adversarial_masked_tokens,
                   check_arm(scalar, oracle), check_arm(warp, oracle), diagnose_batch_cross(scalar, warp)};
}

float time_scalar(const BatchLaunch& launch, const BatchBuffers& buffers, cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start)); launch_batch_scalar(launch, buffers); CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
  float milliseconds = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end)); return milliseconds * 1000.0f;
}
float time_warp(const BatchLaunch& launch, const BatchBuffers& buffers, cudaEvent_t start, cudaEvent_t end) {
  CUDA_CHECK(cudaEventRecord(start)); launch_batch_warp(launch, buffers); CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end));
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
struct Timing { std::vector<float> scalar_ab, scalar_ba, warp_ab, warp_ba; Stats scalar_all{}, scalar_ab_stats{}, scalar_ba_stats{}, warp_all{}, warp_ab_stats{}, warp_ba_stats{}; float speedup = 0.0f, ab_speedup = 0.0f, ba_speedup = 0.0f; bool combined = false, ab = false, ba = false, promoted = false; };
Timing benchmark(const BatchLaunch& launch, const BatchBuffers& buffers, bool all_correct, int warp_local_bytes) {
  cudaEvent_t start = nullptr, end = nullptr; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&end));
  Timing result{};
  try {
    for (int i = 0; i < kBatchWarmupEach; ++i) { launch_batch_scalar(launch, buffers); launch_batch_warp(launch, buffers); }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < kBatchAbbapairs; ++i) {
      result.scalar_ab.push_back(time_scalar(launch, buffers, start, end));
      result.warp_ab.push_back(time_warp(launch, buffers, start, end));
      result.warp_ba.push_back(time_warp(launch, buffers, start, end));
      result.scalar_ba.push_back(time_scalar(launch, buffers, start, end));
    }
    std::vector<float> scalar_all = result.scalar_ab; scalar_all.insert(scalar_all.end(), result.scalar_ba.begin(), result.scalar_ba.end());
    std::vector<float> warp_all = result.warp_ab; warp_all.insert(warp_all.end(), result.warp_ba.begin(), result.warp_ba.end());
    result.scalar_all = stats(scalar_all); result.warp_all = stats(warp_all);
    result.scalar_ab_stats = stats(result.scalar_ab); result.scalar_ba_stats = stats(result.scalar_ba);
    result.warp_ab_stats = stats(result.warp_ab); result.warp_ba_stats = stats(result.warp_ba);
    result.speedup = result.scalar_all.median_us / result.warp_all.median_us;
    result.ab_speedup = result.scalar_ab_stats.median_us / result.warp_ab_stats.median_us;
    result.ba_speedup = result.scalar_ba_stats.median_us / result.warp_ba_stats.median_us;
    result.combined = result.speedup >= 1.10f; result.ab = result.ab_speedup > 1.05f; result.ba = result.ba_speedup > 1.05f;
    result.promoted = result.combined && result.ab && result.ba && all_correct && warp_local_bytes == 0;
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
void print_seed(const SeedCheck& value) { std::cout << "{\"seed\":" << value.seed << ",\"batch\":" << value.batch << ",\"hierarchy_valid\":" << (value.hierarchy_valid ? "true" : "false") << ",\"adversarial_unselected_visible_pages\":" << value.unselected << ",\"adversarial_masked_tokens\":" << value.masked << ",\"scalar\":"; print_arm(value.scalar); std::cout << ",\"warp\":"; print_arm(value.warp); std::cout << ",\"cross_arm\":{\"max_abs\":" << value.cross.max_abs << ",\"max_rel\":" << value.cross.max_rel << ",\"bfloat16_bitwise_equal\":" << (value.cross.bfloat16_bitwise_equal ? "true" : "false") << "}}"; }
void print_case(const CaseResult& value) {
  const Timing& t = value.timing;
  std::cout << "{\"B\":" << value.batch << ",\"paired_seed_selected_set_signatures_unique\":" << (value.paired_seed_selected_set_signatures_unique ? "true" : "false") << ",\"correctness\":["; print_seed(value.seed17); std::cout << ','; print_seed(value.seed2026); std::cout << "],\"post_timing_correctness\":"; print_seed(value.post);
  std::cout << ",\"timing\":{\"protocol\":\"warmup_each_then_51_scalar_warp_warp_scalar_ABBA_pairs\",\"warmup_each\":" << kBatchWarmupEach << ",\"abba_pairs\":" << kBatchAbbapairs << ",\"samples_per_arm\":" << kBatchSamplesPerArm << ",\"raw_samples_us\":{\"scalar\":{\"AB\":"; print_samples(t.scalar_ab); std::cout << ",\"BA\":"; print_samples(t.scalar_ba); std::cout << "},\"warp\":{\"AB\":"; print_samples(t.warp_ab); std::cout << ",\"BA\":"; print_samples(t.warp_ba); std::cout << "}},\"scalar\":{\"all\":"; print_stats(t.scalar_all); std::cout << ",\"when_launch_order_is_AB\":"; print_stats(t.scalar_ab_stats); std::cout << ",\"when_launch_order_is_BA\":"; print_stats(t.scalar_ba_stats); std::cout << "},\"warp\":{\"all\":"; print_stats(t.warp_all); std::cout << ",\"when_launch_order_is_AB\":"; print_stats(t.warp_ab_stats); std::cout << ",\"when_launch_order_is_BA\":"; print_stats(t.warp_ba_stats); std::cout << "},\"speedup_scalar_over_warp\":" << t.speedup << ",\"speedup_by_partition\":{\"AB\":" << t.ab_speedup << ",\"BA\":" << t.ba_speedup << "},\"promotion_gate\":{\"combined_scalar_over_warp_at_least_1_10\":" << (t.combined ? "true" : "false") << ",\"AB_scalar_over_warp_greater_than_1_05\":" << (t.ab ? "true" : "false") << ",\"BA_scalar_over_warp_greater_than_1_05\":" << (t.ba ? "true" : "false") << ",\"all_correct\":" << ((good(value.seed17.scalar) && good(value.seed17.warp) && good(value.seed2026.scalar) && good(value.seed2026.warp) && good(value.post.scalar) && good(value.post.warp)) ? "true" : "false") << ",\"promoted\":" << (t.promoted ? "true" : "false") << "}}}";
}

int c2_batch_main_impl() {
  try {
    int device = 0; CUDA_CHECK(cudaGetDevice(&device)); cudaDeviceProp property{}; CUDA_CHECK(cudaGetDeviceProperties(&property, device));
    int cluster_launch = 0; CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device));
    if (property.major != 10 || property.minor != 3 || cluster_launch == 0) throw std::runtime_error("requires B300 CC 10.3 cluster launch");
    cudaFuncAttributes scalar_attr{}, warp_attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&scalar_attr, c2_batch_scalar_mbarrier_kernel)); CUDA_CHECK(cudaFuncGetAttributes(&warp_attr, c2_batch_warp_mbarrier_kernel));
    if (scalar_attr.sharedSizeBytes != warp_attr.sharedSizeBytes || scalar_attr.sharedSizeBytes > property.sharedMemPerBlock) throw std::runtime_error("batch arms violate static shared-memory equality/fit");
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
      if (!good(result.seed17.scalar) || !good(result.seed17.warp) || !good(result.seed2026.scalar) || !good(result.seed2026.warp)) throw std::runtime_error("pre-timing batch correctness gate failed");
      result.timing = benchmark(launch, retained, true, warp_attr.localSizeBytes);
      // Do not accept a copy of either last timed output as the post-timing
      // evidence.  Reinstall the caller-visible BF16 sentinel first, launch
      // each arm afresh once, synchronize, and only then copy for the oracle.
      refill_batch_outputs_with_sentinel(seed2026_input, retained);
      launch_batch_scalar(launch, retained); CUDA_CHECK(cudaDeviceSynchronize());
      launch_batch_warp(launch, retained); CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<float> oracle = batch_fp64_oracle(seed2026_input);
      std::vector<__nv_bfloat16> scalar(oracle.size()), warp(oracle.size());
      CUDA_CHECK(cudaMemcpy(scalar.data(), retained.scalar_output, scalar.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(warp.data(), retained.warp_output, warp.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
      result.post = SeedCheck{kBatchTimingSeed, batch, validate_batch_indirection(seed2026_input), seed2026_input.adversarial_unselected_visible_pages, seed2026_input.adversarial_masked_tokens, check_arm(scalar, oracle), check_arm(warp, oracle), diagnose_batch_cross(scalar, warp)};
      if (!good(result.post.scalar) || !good(result.post.warp)) throw std::runtime_error("post-timing batch correctness gate failed");
      cases.push_back(std::move(result));
    }
    std::cout << std::setprecision(9) << "{\"schema\":\"c2-cluster-attention-warp-batch-abba-v1\",\"status\":\"pass\",\"boundary\":\"" << json_escape(kBatchBoundary) << "\",\"timing_seed\":" << kBatchTimingSeed << ",\"batch_cases\":[1,4,8,16],\"shape\":{\"Hkv\":4,\"Hq\":64,\"G\":16,\"D\":128,\"page_size\":128,\"selected_pages\":16,\"logical_pages\":32,\"physical_pages_per_batch\":32},\"abi\":{\"query\":\"[B,Hq,D]\",\"output\":\"[B,Hq,D]\",\"seq_lens\":\"[B]\",\"block_table\":\"[B,max_blocks]\",\"topk\":\"[B,Hkv,Ktop]\",\"cache\":\"[physical_page,Hkv,P,D]\",\"cluster_mapping\":\"batch=(blockIdx.x/4)/Hkv; kv_head=(blockIdx.x/4)%Hkv\",\"disjoint_physical_page_pool_per_batch\":true,\"topk_row_order_differs_from_first\":true,\"topk_row_order_scope\":\"each nonzero (batch,kv_head) ordered row differs from (0,0); paired sorted-set signatures provide the stronger two-seed row-coverage gate\"},\"provenance\":{\"scalar_protocol\":\"c2_cluster_attention_mbarrier_smoke.cu is preprocessor-included and SHA-pinned\",\"warp_mapping_reference\":\"c2_cluster_attention_warp_producer_abba.cu is SHA-pinned audited reference only; its mapping is derived here and it is not preprocessor-included\"},\"cluster_layout\":{\"num_ctas\":4,\"grid\":\"B*Hkv*4\",\"selected_pages_per_producer\":8,\"threads_per_block\":256},\"producer_contract\":{\"same_remote_dsm_mbarrier_protocol\":true,\"same_shared_layout_and_output_abi\":true,\"same_launch_shape\":true,\"same_real_selected_causal_attention\":true,\"persistent_device_buffers_outside_timing\":true,\"caller_owned_independent_outputs\":true,\"single_kernel_launch_per_cuda_event_sample\":true,\"ABBA_interleaved\":true,\"initialization_copies_and_oracle_outside_timing\":true,\"changed_field\":\"rank-0/1 producer compute mapping only\",\"timed_launch_validation_scope\":\"pre-timing two-seed checks plus post-timing fresh sentinel-reset scalar/warp launches and oracle recheck; intermediate timed outputs not inspected\"},\"synchronization\":{\"mbarrier_expected_arrivals\":2,\"mbarrier_wait_parity\":0,\"mbarrier_max_polls\":" << kMBarrierMaxPolls << "},\"dtype_contract\":{\"producer_partial\":\"bfloat16\",\"caller_output\":\"bfloat16\",\"oracle_accumulator\":\"float64\",\"oracle_softmax\":\"independent two-pass natural-exp direct selected-page causal attention\",\"tolerance\":{\"rtol\":" << kRtol << ",\"atol\":" << kAtol << "}},\"environment\":{\"device\":\"" << json_escape(property.name) << "\",\"capability\":[" << property.major << ',' << property.minor << "],\"cluster_launch_supported\":true},\"resource_model\":{\"static_shared_equal\":true,\"scalar\":{\"static_shared_bytes\":" << scalar_attr.sharedSizeBytes << ",\"num_regs\":" << scalar_attr.numRegs << ",\"local_bytes\":" << scalar_attr.localSizeBytes << "},\"warp\":{\"static_shared_bytes\":" << warp_attr.sharedSizeBytes << ",\"num_regs\":" << warp_attr.numRegs << ",\"local_bytes\":" << warp_attr.localSizeBytes << "}},\"cases\":[";
    for (std::size_t i = 0; i < cases.size(); ++i) { if (i) std::cout << ','; print_case(cases[i]); }
    std::cout << "]}" << std::endl;
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cout << "{\"schema\":\"c2-cluster-attention-warp-batch-abba-v1\",\"status\":\"fail\",\"boundary\":\"" << json_escape(kBatchBoundary) << "\",\"error\":\"" << json_escape(error.what()) << "\"}" << std::endl;
    return EXIT_FAILURE;
  }
}

}  // namespace

int main() { return c2_batch_main_impl(); }
