// Dynamic negative-path validation for the C=2 remote-DSM mbarrier protocol.
//
// This deliberately omits rank 1's arrival.  It proves that rank 2's bounded
// acquire wait expires, writes an independent fault status and a caller-owned
// sentinel, then all four CTAs converge at the final lifetime cluster barrier.
// Boundary: fault-injection only; neither an attention-correctness candidate
// nor a performance measurement.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda/ptx>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

namespace {

constexpr int kClusters = 4;
constexpr int kNumCtas = 4;
constexpr int kThreadsPerBlock = 256;
constexpr int kGroupsPerCluster = 16;
constexpr int kHeadDim = 128;
constexpr int kOutputElementsPerCluster = kGroupsPerCluster * kHeadDim;
constexpr int kOutputElements = kClusters * kOutputElementsPerCluster;
constexpr int kMBarrierMaxPolls = 1 << 20;
constexpr float kFaultKernelUpperBoundMs = 5000.0f;
constexpr float kFaultSentinel = -12352.0f;  // Exactly representable as BF16.
constexpr float kUnexpectedReadyMarker = 23.0f;
constexpr int kFaultStatusExpiredBase = 0x4D420100;
constexpr int kFaultStatusUnexpectedReadyBase = 0x4D420200;

enum : std::uint32_t {
  kMBarrierExpectedArrivals = 2U,
  kMBarrierInitialParity = 0U,
};

static_assert(kClusters >= 4, "fault validation requires at least four clusters");
static_assert(kNumCtas == 4, "roles 0, 1, 2, and 3 are part of the protocol");
static_assert(kMBarrierExpectedArrivals == 2U, "the production protocol has two producers");

constexpr const char* kBoundary =
    "dynamic missing-arrival fault injection only; not an attention correctness or performance candidate";
constexpr const char* kMBarrierPhase =
    "rank-1 deliberately omits its required remote DSM release-arrive; rank-2 bounded parity-0 acquire wait must expire";
constexpr const char* kProducerReadySync =
    "rank-0 only: cuda::ptx::mbarrier_arrive(sem_release, scope_cluster, space_cluster, remote DSM); "
    "rank-1 intentionally does not arrive";
constexpr const char* kWaitSync =
    "rank-2: mbarrier_try_wait_parity(sem_acquire, scope_cluster, local shared, parity=0), bounded";
constexpr const char* kInitSync =
    "cooperative_groups::cluster_group::sync: rank-2 mbarrier initialization and cluster residency only";
constexpr const char* kLifetimeSync =
    "cooperative_groups::cluster_group::sync: every CTA reaches a final lifetime barrier after fault handling";

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

// Role 2 owns the local mbarrier.  Role 0 performs its real remote DSM release
// arrival.  Role 1 deliberately performs no arrival at all: expected count is
// still two, so rank 2 cannot observe readiness.  The final cluster.sync is
// unconditional, including role 3, so a bounded fault cannot strand a CTA.
__global__ void cluster_attention_mbarrier_fault_kernel(int fault_seed,
                                                         __nv_bfloat16* caller_output,
                                                         int* fault_status) {
  __shared__ __align__(8) std::uint64_t producer_ready_barrier;
  __shared__ int local_wait_ready;

  const cg::cluster_group cluster = cg::this_cluster();
  const int role = static_cast<int>(cluster.block_rank());
  const int cluster_index = static_cast<int>(blockIdx.x / kNumCtas);
  const int thread = static_cast<int>(threadIdx.x);
  const int expected_expired_status = kFaultStatusExpiredBase + (fault_seed & 0xff);
  const int unexpected_ready_status = kFaultStatusUnexpectedReadyBase + (fault_seed & 0xff);

  if (role == 2 && thread == 0) {
    cuda::ptx::mbarrier_init(&producer_ready_barrier, kMBarrierExpectedArrivals);
    local_wait_ready = 0;
  }
  __syncthreads();
  cluster.sync();

  // This is intentionally role 0 only.  Rank 1 must not reach an arrive PTX
  // instruction in this fault path, while all CTAs still share the same later
  // barriers and return normally.
  if (role == 0 && thread == 0) {
    std::uint64_t* remote_rank2_barrier = cluster.map_shared_rank(&producer_ready_barrier, 2);
    cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release,
                               cuda::ptx::scope_cluster,
                               cuda::ptx::space_cluster,
                               remote_rank2_barrier);
  }
  __syncthreads();

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
    local_wait_ready = ready ? 1 : 0;
    // Status is a global, caller-observable fault record independent of the
    // output sentinel.  It encodes the launch seed to reject stale writes.
    fault_status[cluster_index] = ready ? unexpected_ready_status : expected_expired_status;
  }
  __syncthreads();

  if (role == 2) {
    const __nv_bfloat16 marker = __float2bfloat16_rn(
        local_wait_ready == 0 ? kFaultSentinel : kUnexpectedReadyMarker);
    const int output_base = cluster_index * kOutputElementsPerCluster;
    for (int index = thread; index < kOutputElementsPerCluster; index += blockDim.x) {
      caller_output[output_base + index] = marker;
    }
  }

  // A single convergent lifetime barrier proves that timeout handling is not
  // relying on the external watchdog to reclaim a deadlocked cluster.
  cluster.sync();
}

struct ClusterResult {
  int cluster = 0;
  int status = 0;
  bool status_expected = false;
};

struct SeedResult {
  int seed = 0;
  int expected_status = 0;
  float fault_kernel_elapsed_ms = 0.0f;
  bool wait_not_ready = false;
  bool sentinel_complete = false;
  bool kernel_within_bound = false;
  std::vector<ClusterResult> clusters;
};

SeedResult run_fault_seed(int seed) {
  std::vector<__nv_bfloat16> output(kOutputElements, __float2bfloat16_rn(0.0f));
  std::vector<int> status(kClusters, -1);
  __nv_bfloat16* device_output = nullptr;
  int* device_status = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  try {
    CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&device_status, status.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(device_output, output.data(), output.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_status, 0xff, status.size() * sizeof(int)));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {kNumCtas, 1, 1};
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim = dim3(kClusters * kNumCtas, 1, 1);
    launch_config.blockDim = dim3(kThreadsPerBlock, 1, 1);
    launch_config.dynamicSmemBytes = 0;
    launch_config.stream = nullptr;
    launch_config.attrs = &attribute;
    launch_config.numAttrs = 1;

    // Timing is explicitly a liveness upper-bound check for this one fault
    // kernel, not a benchmark and never compared with another implementation.
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaLaunchKernelEx(&launch_config, cluster_attention_mbarrier_fault_kernel,
                                  seed, device_output, device_status));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(status.data(), device_status, status.size() * sizeof(int), cudaMemcpyDeviceToHost));

    SeedResult result{};
    result.seed = seed;
    result.expected_status = kFaultStatusExpiredBase + (seed & 0xff);
    result.fault_kernel_elapsed_ms = elapsed_ms;
    result.kernel_within_bound = std::isfinite(elapsed_ms) && elapsed_ms >= 0.0f
        && elapsed_ms < kFaultKernelUpperBoundMs;
    result.wait_not_ready = true;
    for (int cluster = 0; cluster < kClusters; ++cluster) {
      const bool expected = status[cluster] == result.expected_status;
      result.wait_not_ready = result.wait_not_ready && expected;
      result.clusters.push_back(ClusterResult{cluster, status[cluster], expected});
    }
    const __nv_bfloat16 sentinel = __float2bfloat16_rn(kFaultSentinel);
    result.sentinel_complete = true;
    for (const __nv_bfloat16 value : output) {
      result.sentinel_complete = result.sentinel_complete && same_bfloat16_bits(value, sentinel);
    }
    CUDA_CHECK(cudaEventDestroy(stop));
    stop = nullptr;
    CUDA_CHECK(cudaEventDestroy(start));
    start = nullptr;
    CUDA_CHECK(cudaFree(device_status));
    device_status = nullptr;
    CUDA_CHECK(cudaFree(device_output));
    device_output = nullptr;
    return result;
  } catch (...) {
    if (stop != nullptr) cudaEventDestroy(stop);
    if (start != nullptr) cudaEventDestroy(start);
    if (device_status != nullptr) cudaFree(device_status);
    if (device_output != nullptr) cudaFree(device_output);
    throw;
  }
}

void print_success_json(const cudaDeviceProp& property,
                        const cudaFuncAttributes& attributes,
                        int runtime_version,
                        int driver_version,
                        int cluster_launch,
                        const std::vector<SeedResult>& results) {
  std::cout << std::setprecision(9)
            << "{\"schema\":\"c2-cluster-attention-mbarrier-fault-smoke-v1\","
            << "\"status\":\"pass\",\"boundary\":\"" << json_escape(kBoundary) << "\","
            << "\"fault_injection\":{\"omitted_role\":1,\"arrival_role\":0,\"consumer_role\":2,"
            << "\"expected_arrivals\":" << kMBarrierExpectedArrivals
            << ",\"actual_remote_arrivals\":1,\"wait_parity\":" << kMBarrierInitialParity
            << ",\"max_polls\":" << kMBarrierMaxPolls << "},"
            << "\"mbarrier_phase\":\"" << json_escape(kMBarrierPhase) << "\","
            << "\"producer_ready_sync\":\"" << json_escape(kProducerReadySync) << "\","
            << "\"wait_sync\":\"" << json_escape(kWaitSync) << "\","
            << "\"init_sync\":\"" << json_escape(kInitSync) << "\","
            << "\"lifetime_sync\":\"" << json_escape(kLifetimeSync) << "\","
            << "\"sync_api\":\"cooperative_groups::cluster_group::sync (init + final lifetime only)\","
            << "\"remote_shared_api\":\"cooperative_groups::cluster_group::map_shared_rank\","
            << "\"num_ctas\":" << kNumCtas << ",\"clusters\":" << kClusters
            << ",\"threads_per_block\":" << kThreadsPerBlock
            << ",\"output_elements_per_cluster\":" << kOutputElementsPerCluster
            << ",\"fault_sentinel\":" << kFaultSentinel
            << ",\"fault_kernel_upper_bound_ms\":" << kFaultKernelUpperBoundMs << ","
            << "\"cuda_runtime\":" << runtime_version << ",\"cuda_driver\":" << driver_version << ","
            << "\"cluster_launch_supported\":" << (cluster_launch != 0 ? "true" : "false") << ","
            << "\"device\":\"" << json_escape(property.name) << "\","
            << "\"capability\":[" << property.major << ',' << property.minor << "],"
            << "\"resource_model\":{\"static_shared_bytes\":" << attributes.sharedSizeBytes
            << ",\"num_regs\":" << attributes.numRegs << ",\"shared_mem_per_block\":"
            << property.sharedMemPerBlock << ",\"static_shared_fits\":"
            << (attributes.sharedSizeBytes <= property.sharedMemPerBlock ? "true" : "false") << "},"
            << "\"event_timing_scope\":\"single fault kernel liveness bound only; not a performance metric\","
            << "\"seeds\":[";
  for (std::size_t seed_index = 0; seed_index < results.size(); ++seed_index) {
    if (seed_index != 0) std::cout << ',';
    const SeedResult& result = results[seed_index];
    std::cout << "{\"seed\":" << result.seed << ",\"expected_status\":" << result.expected_status
              << ",\"fault_kernel_elapsed_ms\":" << result.fault_kernel_elapsed_ms
              << ",\"wait_not_ready\":" << (result.wait_not_ready ? "true" : "false")
              << ",\"sentinel_complete\":" << (result.sentinel_complete ? "true" : "false")
              << ",\"kernel_within_bound\":" << (result.kernel_within_bound ? "true" : "false")
              << ",\"clusters\":[";
    for (std::size_t cluster_index = 0; cluster_index < result.clusters.size(); ++cluster_index) {
      if (cluster_index != 0) std::cout << ',';
      const ClusterResult& cluster = result.clusters[cluster_index];
      std::cout << "{\"cluster\":" << cluster.cluster << ",\"fault_status\":" << cluster.status
                << ",\"status_expected\":" << (cluster.status_expected ? "true" : "false") << '}';
    }
    std::cout << "]}";
  }
  std::cout << "]}" << std::endl;
}

void print_failure_json(const std::string& error) {
  std::cout << "{\"schema\":\"c2-cluster-attention-mbarrier-fault-smoke-v1\",\"status\":\"fail\",\"error\":\""
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
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, cluster_attention_mbarrier_fault_kernel));
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
      results.push_back(run_fault_seed(seed));
    }
    for (const SeedResult& result : results) {
      if (!result.wait_not_ready || !result.sentinel_complete || !result.kernel_within_bound
          || static_cast<int>(result.clusters.size()) != kClusters) {
        throw std::runtime_error("fault injection did not produce bounded all-cluster timeout sentinel evidence");
      }
      for (const ClusterResult& cluster : result.clusters) {
        if (!cluster.status_expected) {
          throw std::runtime_error("a cluster reported unexpected mbarrier readiness or stale fault status");
        }
      }
    }
    print_success_json(property, attributes, runtime_version, driver_version, cluster_launch, results);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    print_failure_json(error.what());
    return EXIT_FAILURE;
  }
}
