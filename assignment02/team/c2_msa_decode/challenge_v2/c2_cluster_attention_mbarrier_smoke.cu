// Native CUDA C=2 correctness prototype with real selected-page attention producers
// and a remote DSM mbarrier readiness phase.
//
// Boundary: this is a cluster communication prerequisite with actual BF16 QK
// causal selected attention.  It is not a performance implementation or a full
// C=2 decode fusion.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda/ptx>

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

namespace cg = cooperative_groups;

namespace {

constexpr int kBatch = 1;
constexpr int kKvHeads = 4;
constexpr int kQueryHeads = 64;
constexpr int kGqaGroup = 16;
constexpr int kHeadDim = 128;
constexpr int kPageSize = 128;
constexpr int kSelectedPages = 16;
constexpr int kPagesPerProducer = 8;
constexpr int kLogicalPages = 32;
constexpr int kPhysicalPages = 32;
constexpr int kNumCtas = 4;
constexpr int kThreadsPerBlock = 256;
constexpr int kOutputElements = kQueryHeads * kHeadDim;
constexpr int kCacheElements = kPhysicalPages * kKvHeads * kPageSize * kHeadDim;
constexpr float kScaleLog2e = 0.127517431f;  // (1 / sqrt(128)) * log2(e)
constexpr float kSentinel = -12345.678f;
// The cluster.sync control's two-seed BF16-output envelope is below 1.6e-4.
// These gates retain BF16 rounding headroom while rejecting merge regressions.
constexpr float kAtol = 5.0e-4f;
constexpr float kRtol = 5.0e-3f;
// Enum constants are visible as immediate operands in both host and device
// compilation, so the JSON evidence and emitted instructions share one source.
enum : std::uint32_t {
  kMBarrierExpectedArrivals = 2U,
  kMBarrierInitialParity = 0U,
};
constexpr int kMBarrierMaxPolls = 1 << 24;

static_assert(kBatch == 1, "this smoke intentionally fixes B=1");
static_assert(kQueryHeads == kKvHeads * kGqaGroup, "GQA dimensions must agree");
static_assert(kSelectedPages == 2 * kPagesPerProducer, "two producers split selected pages evenly");
static_assert(kNumCtas == 4, "roles 0, 1, 2, 3 require exactly four CTAs");
static_assert(kMBarrierExpectedArrivals == 2U, "one remote release-arrive is required from each producer");

constexpr const char* kBoundary =
    "cluster communication prerequisite only; real BF16 QK causal selected attention producers; "
    "not a performance implementation or full C=2 fusion";
constexpr const char* kMBarrierPhase =
    "rank-2 local shared mbarrier starts at parity 0 and completes after two remote DSM release arrivals";
constexpr const char* kProducerReadySync =
    "cuda::ptx::mbarrier_arrive(sem_release, scope_cluster, space_cluster, remote DSM) "
    "-> mbarrier_try_wait_parity(sem_acquire, scope_cluster, local shared, parity=0)";
constexpr const char* kInitSync =
    "cooperative_groups::cluster_group::sync: rank-2 mbarrier initialization and cluster residency only";
constexpr const char* kLifetimeSync =
    "cooperative_groups::cluster_group::sync: producer CTA-local shared partial lifetime after DSM consumer reads";

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::ostringstream stream;
    stream << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(stream.str());
  }
}

#define CUDA_CHECK(operation) check_cuda((operation), #operation)

std::string json_escape(const std::string& value) {
  std::ostringstream stream;
  for (const unsigned char character : value) {
    switch (character) {
      case '\\': stream << "\\\\"; break;
      case '"': stream << "\\\""; break;
      case '\n': stream << "\\n"; break;
      case '\r': stream << "\\r"; break;
      case '\t': stream << "\\t"; break;
      default:
        if (character < 0x20U) {
          stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(character) << std::dec << std::setfill(' ');
        } else {
          stream << static_cast<char>(character);
        }
    }
  }
  return stream.str();
}

bool same_bfloat16_bits(__nv_bfloat16 lhs, __nv_bfloat16 rhs) {
  std::uint16_t lhs_bits = 0;
  std::uint16_t rhs_bits = 0;
  static_assert(sizeof(lhs_bits) == sizeof(lhs), "unexpected bfloat16 size");
  std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
  return lhs_bits == rhs_bits;
}

struct AttentionInput {
  int seed = 0;
  int sequence_length = 0;
  std::vector<__nv_bfloat16> query;
  std::vector<__nv_bfloat16> key_cache;
  std::vector<__nv_bfloat16> value_cache;
  std::vector<int> topk_idx;
  std::vector<int> block_table;
  int adversarial_unselected_visible_pages = 0;
  int adversarial_masked_tokens = 0;
};

__host__ __device__ __forceinline__ std::size_t cache_offset(int physical_page, int kv_head, int token, int dim) {
  return (((static_cast<std::size_t>(physical_page) * kKvHeads + kv_head) * kPageSize + token) * kHeadDim + dim);
}

AttentionInput make_input(int seed) {
  AttentionInput input{};
  input.seed = seed;
  // Both cases expose more than 16 logical pages, and the final visible page
  // contains exactly one valid token.  That simultaneously exercises a true
  // top-k subset and a strong causal-tail mask.
  const int full_visible_pages = (seed & 1) != 0 ? kSelectedPages : kLogicalPages - 1;
  input.sequence_length = full_visible_pages * kPageSize + 1;
  input.query.resize(kOutputElements);
  input.key_cache.resize(kCacheElements);
  input.value_cache.resize(kCacheElements);
  input.topk_idx.resize(kKvHeads * kSelectedPages);
  input.block_table.resize(kLogicalPages);

  std::mt19937 generator(static_cast<std::mt19937::result_type>(seed));
  std::uniform_real_distribution<float> query_distribution(-0.50f, 0.50f);
  std::uniform_real_distribution<float> key_distribution(-0.50f, 0.50f);
  std::uniform_real_distribution<float> value_distribution(-0.75f, 0.75f);
  for (auto& value : input.query) {
    value = __float2bfloat16_rn(query_distribution(generator));
  }
  for (auto& value : input.key_cache) {
    value = __float2bfloat16_rn(key_distribution(generator));
  }
  for (auto& value : input.value_cache) {
    value = __float2bfloat16_rn(value_distribution(generator));
  }

  std::vector<int> physical_pages(kPhysicalPages);
  std::iota(physical_pages.begin(), physical_pages.end(), 0);
  std::shuffle(physical_pages.begin(), physical_pages.end(), generator);
  std::copy_n(physical_pages.begin(), kLogicalPages, input.block_table.begin());

  const int visible_blocks = (input.sequence_length + kPageSize - 1) / kPageSize;
  for (int kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    std::vector<int> logical_pages(visible_blocks);
    std::iota(logical_pages.begin(), logical_pages.end(), 0);
    std::shuffle(logical_pages.begin(), logical_pages.end(), generator);
    // Force the partially visible last page into the selected set so a kernel
    // that drops the causal token mask encounters the adversarial tail below.
    const auto last_page = std::find(logical_pages.begin(), logical_pages.end(), visible_blocks - 1);
    if (last_page >= logical_pages.begin() + kSelectedPages) {
      std::iter_swap(logical_pages.begin(), last_page);
    }
    for (int selected = 0; selected < kSelectedPages; ++selected) {
      input.topk_idx[kv_head * kSelectedPages + selected] = logical_pages[selected];
    }

    std::vector<bool> selected_mask(visible_blocks, false);
    for (int selected = 0; selected < kSelectedPages; ++selected) {
      selected_mask[input.topk_idx[kv_head * kSelectedPages + selected]] = true;
    }
    for (int logical_page = 0; logical_page < visible_blocks; ++logical_page) {
      if (!selected_mask[logical_page]) {
        ++input.adversarial_unselected_visible_pages;
        const int physical_page = input.block_table[logical_page];
        for (int token = 0; token < kPageSize; ++token) {
          for (int dim = 0; dim < kHeadDim; ++dim) {
            const std::size_t offset = cache_offset(physical_page, kv_head, token, dim);
            input.key_cache[offset] = __float2bfloat16_rn(0.0f);
            input.value_cache[offset] = __float2bfloat16_rn(6.0f + static_cast<float>(kv_head));
          }
        }
      }
    }

    const int last_physical_page = input.block_table[visible_blocks - 1];
    for (int token = 1; token < kPageSize; ++token) {
      ++input.adversarial_masked_tokens;
      for (int dim = 0; dim < kHeadDim; ++dim) {
        const std::size_t offset = cache_offset(last_physical_page, kv_head, token, dim);
        input.key_cache[offset] = __float2bfloat16_rn(0.0f);
        input.value_cache[offset] = __float2bfloat16_rn(12.0f + static_cast<float>(kv_head));
      }
    }
  }
  return input;
}

bool validate_indirection(const AttentionInput& input) {
  if (input.block_table.size() != kLogicalPages) {
    return false;
  }
  std::vector<bool> physical_seen(kPhysicalPages, false);
  for (const int physical_page : input.block_table) {
    if (physical_page < 0 || physical_page >= kPhysicalPages || physical_seen[physical_page]) {
      return false;
    }
    physical_seen[physical_page] = true;
  }
  const int visible_blocks = (input.sequence_length + kPageSize - 1) / kPageSize;
  if (visible_blocks <= kSelectedPages || visible_blocks > kLogicalPages) {
    return false;
  }
  for (int kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    std::vector<bool> seen(kLogicalPages, false);
    for (int selected = 0; selected < kSelectedPages; ++selected) {
      const int logical_page = input.topk_idx[kv_head * kSelectedPages + selected];
      if (logical_page < 0 || logical_page >= visible_blocks || seen[logical_page]) {
        return false;
      }
      seen[logical_page] = true;
      const int physical_page = input.block_table[logical_page];
      if (physical_page < 0 || physical_page >= kPhysicalPages) {
        return false;
      }
    }
    if (!seen[visible_blocks - 1]
        || std::all_of(seen.begin(), seen.begin() + visible_blocks, [](bool value) { return value; })) {
      return false;
    }
  }
  return input.adversarial_unselected_visible_pages > 0
      && input.adversarial_masked_tokens == kKvHeads * (kPageSize - 1);
}

// Each producer stores only its BF16 normalized partial O and FP32 base-2 LSE
// in its own CTA-local shared memory. There is no global partial buffer. Rank 2
// owns a local mbarrier; rank 0 and rank 1 release-arrive at its remote DSM
// address after their CTA-local partials are complete.
__global__ void cluster_attention_mbarrier_kernel(const __nv_bfloat16* query,
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

  // This CTA-local object is initialized exactly once in rank 2. The following
  // CTA barrier and initial cluster barrier make it available before rank 0/1
  // map its remote DSM address. This cluster barrier is not the data-ready
  // handoff; the mbarrier below is the producer-to-consumer synchronization.
  if (role == 2 && thread == 0) {
    cuda::ptx::mbarrier_init(&producer_ready_barrier, kMBarrierExpectedArrivals);
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

  // All CTAs execute this CTA barrier. For ranks 0/1, it makes every producer
  // thread's shared partial visible before thread 0 performs the release-arrive.
  __syncthreads();
  if ((role == 0 || role == 1) && thread == 0) {
    std::uint64_t* remote_rank2_barrier = cluster.map_shared_rank(&producer_ready_barrier, 2);
    cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release,
                               cuda::ptx::scope_cluster,
                               cuda::ptx::space_cluster,
                               remote_rank2_barrier);
  }

  // Expected arrival count is two, so a single parity-0 wait observes completion
  // only after both producer CTAs have performed their remote release-arrive.
  // It is deliberately bounded: on a synchronization failure rank 2 writes the
  // caller-owned sentinel and the host finite/sentinel gate rejects the result.
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

  // Rank 2 fans the acquired readiness result to its consumer threads. Every
  // CTA executes this barrier, including role 3, so the later lifetime sync has
  // a uniform lifecycle.
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

  // This final barrier only protects rank 0/1 CTA-local shared partial lifetime
  // until all rank-2 DSM reads have completed.
  cluster.sync();
}

std::vector<float> cpu_fp64_accum_oracle(const AttentionInput& input) {
  // This independent oracle uses an explicit two-pass natural-exp softmax with
  // FP64 accumulators over all
  // selected pages.  It deliberately never materializes or combines producer
  // partials on the host.
  std::vector<float> output(kOutputElements, 0.0f);
  const int query_position = input.sequence_length - 1;
  for (int kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    for (int group_head = 0; group_head < kGqaGroup; ++group_head) {
      const int query_head = kv_head * kGqaGroup + group_head;
      const __nv_bfloat16* query_row = input.query.data() + static_cast<std::size_t>(query_head) * kHeadDim;
      double accumulator[kHeadDim];
      for (int dim = 0; dim < kHeadDim; ++dim) {
        accumulator[dim] = 0.0f;
      }
      double max_score = -std::numeric_limits<double>::infinity();
      const double inverse_sqrt_head_dim = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
      for (int selected = 0; selected < kSelectedPages; ++selected) {
        const int logical_page = input.topk_idx[kv_head * kSelectedPages + selected];
        const int physical_page = input.block_table[logical_page];
        for (int token = 0; token < kPageSize; ++token) {
          const int key_position = logical_page * kPageSize + token;
          if (key_position <= query_position && key_position < input.sequence_length) {
            const std::size_t kv_base =
                (((static_cast<std::size_t>(physical_page) * kKvHeads + kv_head) * kPageSize + token)
                 * kHeadDim);
            double score = 0.0;
            for (int dim = 0; dim < kHeadDim; ++dim) {
              score = std::fma(static_cast<double>(__bfloat162float(query_row[dim])),
                               static_cast<double>(__bfloat162float(input.key_cache[kv_base + dim])), score);
            }
            score *= inverse_sqrt_head_dim;
            max_score = std::max(max_score, score);
          }
        }
      }
      double normalizer = 0.0;
      for (int selected = 0; selected < kSelectedPages; ++selected) {
        const int logical_page = input.topk_idx[kv_head * kSelectedPages + selected];
        const int physical_page = input.block_table[logical_page];
        for (int token = 0; token < kPageSize; ++token) {
          const int key_position = logical_page * kPageSize + token;
          if (key_position <= query_position && key_position < input.sequence_length) {
            const std::size_t kv_base =
                (((static_cast<std::size_t>(physical_page) * kKvHeads + kv_head) * kPageSize + token)
                 * kHeadDim);
            double score = 0.0;
            for (int dim = 0; dim < kHeadDim; ++dim) {
              score = std::fma(static_cast<double>(__bfloat162float(query_row[dim])),
                               static_cast<double>(__bfloat162float(input.key_cache[kv_base + dim])), score);
            }
            score *= inverse_sqrt_head_dim;
            const double weight = std::exp(score - max_score);
            normalizer += weight;
            for (int dim = 0; dim < kHeadDim; ++dim) {
              accumulator[dim] += weight * static_cast<double>(__bfloat162float(input.value_cache[kv_base + dim]));
            }
          }
        }
      }
      if (!(normalizer > 0.0f)) {
        throw std::runtime_error("CPU oracle observed an empty selected-page range");
      }
      const std::size_t output_base = static_cast<std::size_t>(query_head) * kHeadDim;
      for (int dim = 0; dim < kHeadDim; ++dim) {
        output[output_base + dim] = static_cast<float>(accumulator[dim] / normalizer);
      }
    }
  }
  return output;
}

struct SeedResult {
  int seed = 0;
  int sequence_length = 0;
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  bool hierarchy_valid = false;
  int adversarial_unselected_visible_pages = 0;
  int adversarial_masked_tokens = 0;
  bool oracle_finite = true;
  bool finite = true;
  bool sentinel_clean = true;
  bool allclose = true;
};

SeedResult run_seed(int seed) {
  AttentionInput input = make_input(seed);
  const bool hierarchy_valid = validate_indirection(input);
  if (!hierarchy_valid) {
    throw std::runtime_error("input indirection failed validation before oracle or GPU launch");
  }
  const std::vector<float> oracle = cpu_fp64_accum_oracle(input);
  const __nv_bfloat16 sentinel_bf16 = __float2bfloat16_rn(kSentinel);
  std::vector<__nv_bfloat16> output(kOutputElements, sentinel_bf16);

  __nv_bfloat16* device_query = nullptr;
  __nv_bfloat16* device_key_cache = nullptr;
  __nv_bfloat16* device_value_cache = nullptr;
  int* device_topk_idx = nullptr;
  int* device_block_table = nullptr;
  __nv_bfloat16* device_output = nullptr;
  try {
    CUDA_CHECK(cudaMalloc(&device_query, input.query.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&device_key_cache, input.key_cache.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&device_value_cache, input.value_cache.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&device_topk_idx, input.topk_idx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_block_table, input.block_table.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(device_query, input.query.data(), input.query.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_key_cache, input.key_cache.data(), input.key_cache.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_value_cache, input.value_cache.data(), input.value_cache.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_topk_idx, input.topk_idx.data(), input.topk_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_block_table, input.block_table.data(), input.block_table.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_output, output.data(), output.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {kNumCtas, 1, 1};
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = dim3(kKvHeads * kNumCtas, 1, 1);
    launch_config.blockDim = dim3(kThreadsPerBlock, 1, 1);
    launch_config.dynamicSmemBytes = 0;
    launch_config.stream = nullptr;
    launch_config.attrs = &attribute;
    launch_config.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&launch_config, cluster_attention_mbarrier_kernel,
                                  device_query, device_key_cache, device_value_cache,
                                  device_topk_idx, device_block_table, input.sequence_length, device_output));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  } catch (...) {
    cudaFree(device_output);
    cudaFree(device_block_table);
    cudaFree(device_topk_idx);
    cudaFree(device_value_cache);
    cudaFree(device_key_cache);
    cudaFree(device_query);
    throw;
  }
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_block_table));
  CUDA_CHECK(cudaFree(device_topk_idx));
  CUDA_CHECK(cudaFree(device_value_cache));
  CUDA_CHECK(cudaFree(device_key_cache));
  CUDA_CHECK(cudaFree(device_query));

  SeedResult result{};
  result.seed = seed;
  result.sequence_length = input.sequence_length;
  result.hierarchy_valid = hierarchy_valid;
  result.adversarial_unselected_visible_pages = input.adversarial_unselected_visible_pages;
  result.adversarial_masked_tokens = input.adversarial_masked_tokens;
  for (std::size_t index = 0; index < output.size(); ++index) {
    const float actual = __bfloat162float(output[index]);
    const float expected = oracle[index];
    result.oracle_finite = result.oracle_finite && std::isfinite(expected);
    result.finite = result.finite && std::isfinite(actual);
    result.sentinel_clean = result.sentinel_clean && !same_bfloat16_bits(output[index], sentinel_bf16);
    const float absolute_error = std::fabs(actual - expected);
    const float relative_error = absolute_error / std::max(std::fabs(expected), 1.0e-7f);
    result.max_abs = std::max(result.max_abs, absolute_error);
    result.max_rel = std::max(result.max_rel, relative_error);
    result.allclose = result.allclose && absolute_error <= kAtol + kRtol * std::fabs(expected);
  }
  return result;
}

void print_success_json(const cudaDeviceProp& property,
                        const cudaFuncAttributes& attributes,
                        int runtime_version,
                        int driver_version,
                        int cluster_launch,
                        const std::vector<SeedResult>& results) {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  bool hierarchy_valid = true;
  bool oracle_finite = true;
  bool finite = true;
  bool sentinel_clean = true;
  bool allclose = true;
  for (const SeedResult& result : results) {
    max_abs = std::max(max_abs, result.max_abs);
    max_rel = std::max(max_rel, result.max_rel);
    hierarchy_valid = hierarchy_valid && result.hierarchy_valid;
    oracle_finite = oracle_finite && result.oracle_finite;
    finite = finite && result.finite;
    sentinel_clean = sentinel_clean && result.sentinel_clean;
    allclose = allclose && result.allclose;
  }

  std::cout << std::setprecision(9)
            << "{\"schema\":\"c2-cluster-attention-mbarrier-smoke-v1\","
            << "\"status\":\"pass\",\"boundary\":\"" << json_escape(kBoundary) << "\","
            << "\"mbarrier_verified\":true,\"mbarrier_phase\":\"" << json_escape(kMBarrierPhase) << "\","
            << "\"mbarrier_expected_arrivals\":" << kMBarrierExpectedArrivals
            << ",\"mbarrier_wait_parity\":" << kMBarrierInitialParity
            << ",\"mbarrier_max_polls\":" << kMBarrierMaxPolls << ","
            << "\"producer_ready_sync\":\"" << json_escape(kProducerReadySync) << "\","
            << "\"init_sync\":\"" << json_escape(kInitSync) << "\","
            << "\"lifetime_sync\":\"" << json_escape(kLifetimeSync) << "\","
            << "\"sync_api\":\"cooperative_groups::cluster_group::sync (init + lifetime only)\","
            << "\"remote_shared_api\":\"cooperative_groups::cluster_group::map_shared_rank\","
            << "\"global_seed_inputs\":true,\"global_inter_cta_scratch\":false,"
            << "\"caller_owned_output\":true,\"host_precomputed_partials\":false,"
            << "\"partial_dtype\":\"bfloat16\",\"caller_output_dtype\":\"bfloat16\","
            << "\"oracle_accumulator_dtype\":\"float64\","
            << "\"oracle_softmax\":\"independent two-pass natural-exp direct attention\","
            << "\"producer\":\"one-thread-per-head online BF16 QK causal selected attention\","
            << "\"input_indirection\":\"topk_idx -> block_table -> physical KV page\","
            << "\"shape\":{\"B\":" << kBatch << ",\"Hkv\":" << kKvHeads
            << ",\"Hq\":" << kQueryHeads << ",\"G\":" << kGqaGroup << ",\"D\":" << kHeadDim
            << ",\"page_size\":" << kPageSize << ",\"selected_pages\":" << kSelectedPages
            << ",\"logical_pages\":" << kLogicalPages << "},"
            << "\"block_table_abi\":\"[B,max_blocks], shared by all KV heads\","
            << "\"adversarial_unselected_visible_pages\":true,"
            << "\"adversarial_causal_tail\":true,"
            << "\"num_ctas\":" << kNumCtas << ",\"clusters\":" << kKvHeads
            << ",\"selected_pages_per_role\":" << kPagesPerProducer
            << ",\"threads_per_block\":" << kThreadsPerBlock << ","
            << "\"cuda_runtime\":" << runtime_version << ",\"cuda_driver\":" << driver_version << ","
            << "\"cluster_launch_supported\":" << (cluster_launch != 0 ? "true" : "false") << ","
            << "\"device\":\"" << json_escape(property.name) << "\","
            << "\"capability\":[" << property.major << ',' << property.minor << "],"
            << "\"resource_model\":{\"static_shared_bytes\":" << attributes.sharedSizeBytes
            << ",\"num_regs\":" << attributes.numRegs << ",\"shared_mem_per_block\":"
            << property.sharedMemPerBlock << ",\"static_shared_fits\":"
            << (attributes.sharedSizeBytes <= property.sharedMemPerBlock ? "true" : "false") << "},"
            << "\"tolerance\":{\"rtol\":" << kRtol << ",\"atol\":" << kAtol
            << ",\"reason\":\"BF16 producer partial O is merged against a direct FP64 oracle\"},"
            << "\"hierarchy_valid\":" << (hierarchy_valid ? "true" : "false") << ","
            << "\"mbarrier_readiness_observed\":" << (sentinel_clean ? "true" : "false") << ","
            << "\"oracle_finite\":" << (oracle_finite ? "true" : "false") << ","
            << "\"finite\":" << (finite ? "true" : "false") << ","
            << "\"sentinel_clean\":" << (sentinel_clean ? "true" : "false") << ","
            << "\"allclose\":" << (allclose ? "true" : "false") << ","
            << "\"max_abs\":" << max_abs << ",\"max_rel\":" << max_rel << ",\"seeds\":[";
  for (std::size_t index = 0; index < results.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    const SeedResult& result = results[index];
    std::cout << "{\"seed\":" << result.seed << ",\"sequence_length\":" << result.sequence_length
              << ",\"adversarial_unselected_visible_pages\":"
              << result.adversarial_unselected_visible_pages
              << ",\"adversarial_masked_tokens\":" << result.adversarial_masked_tokens
              << ",\"max_abs\":" << result.max_abs << ",\"max_rel\":" << result.max_rel
              << ",\"hierarchy_valid\":" << (result.hierarchy_valid ? "true" : "false")
              << ",\"oracle_finite\":" << (result.oracle_finite ? "true" : "false")
              << ",\"finite\":" << (result.finite ? "true" : "false")
              << ",\"sentinel_clean\":" << (result.sentinel_clean ? "true" : "false")
              << ",\"mbarrier_timeout_sentinel_absent\":" << (result.sentinel_clean ? "true" : "false")
              << ",\"allclose\":" << (result.allclose ? "true" : "false") << '}';
  }
  std::cout << "]}" << std::endl;
}

void print_failure_json(const std::string& error) {
  std::cout << "{\"schema\":\"c2-cluster-attention-mbarrier-smoke-v1\",\"status\":\"fail\",\"error\":\""
            << json_escape(error) << "\",\"boundary\":\"" << json_escape(kBoundary) << "\"}" << std::endl;
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
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, cluster_attention_mbarrier_kernel));
    if (property.major != 10 || property.minor != 3) {
      throw std::runtime_error("requires B300 compute capability 10.3");
    }
    if (cluster_launch == 0) {
      throw std::runtime_error("cudaDevAttrClusterLaunch is false");
    }
    if (attributes.sharedSizeBytes > property.sharedMemPerBlock) {
      throw std::runtime_error("per-CTA static shared-memory requirement exceeds the device limit");
    }

    const std::vector<int> seeds{17, 2026};
    std::vector<SeedResult> results;
    results.reserve(seeds.size());
    for (const int seed : seeds) {
      results.push_back(run_seed(seed));
    }
    for (const SeedResult& result : results) {
      if (!result.hierarchy_valid || result.adversarial_unselected_visible_pages <= 0
          || result.adversarial_masked_tokens != kKvHeads * (kPageSize - 1)
          || !result.oracle_finite || !result.finite || !result.sentinel_clean || !result.allclose) {
        throw std::runtime_error("attention smoke failed hierarchy, finite, sentinel, or allclose validation");
      }
    }
    print_success_json(property, attributes, runtime_version, driver_version, cluster_launch, results);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    print_failure_json(error.what());
    return EXIT_FAILURE;
  }
}
