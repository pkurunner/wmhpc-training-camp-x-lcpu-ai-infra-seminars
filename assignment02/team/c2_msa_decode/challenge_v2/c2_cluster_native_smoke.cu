// Native CUDA prerequisite for C=2 decode fusion experiments.
//
// Boundary: cluster communication prerequisite only; producers are synthetic.
// This is deliberately not an attention implementation or a full C=2 fusion.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

namespace {

constexpr int kNumCtas = 4;
constexpr int kClusters = 4;
constexpr int kHGroup = 16;
constexpr int kHeadDim = 128;
constexpr int kElementsPerCluster = kHGroup * kHeadDim;
constexpr int kThreadsPerBlock = 256;
constexpr float kSentinel = -12345.678f;
constexpr float kAtol = 1.0e-3f;
constexpr float kRtol = 1.0e-3f;

constexpr const char* kBoundary = "cluster communication prerequisite only; producers are synthetic";
constexpr const char* kMBarrierPhase =
    "pending/not implemented: this prototype uses no mbarrier; CUDA 13.0 libcu++ exposes no "
    "cuda::thread_scope_cluster barrier; "
    "two cooperative_groups::cluster_group::sync phases are used";

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

float synthetic_partial_value(int seed, int cluster, int head, int dim, int producer) {
  // Integer arithmetic makes every host input deterministic before BF16 rounding.
  const int lane = seed * 17 + cluster * 29 + head * 11 + dim * 5 + producer * 37;
  return static_cast<float>((lane % 257) - 128) / 96.0f;
}

float synthetic_lse_value(int seed, int cluster, int head, int producer) {
  const int lane = seed * 7 + cluster * 13 + head * 3 + producer * 19;
  return -6.0f + static_cast<float>(lane % 113) / 16.0f;
}

bool same_bits(float lhs, float rhs) {
  std::uint32_t lhs_bits = 0;
  std::uint32_t rhs_bits = 0;
  static_assert(sizeof(lhs_bits) == sizeof(lhs), "unexpected float size");
  std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
  return lhs_bits == rhs_bits;
}

// Each physical CTA owns this declaration.  role 0 and role 1 populate their
// own CTA-local allocations; role 2 maps those exact allocations by rank.
__global__ void cluster_reduce_kernel(const __nv_bfloat16* partial0,
                                      const __nv_bfloat16* partial1,
                                      const float* lse0,
                                      const float* lse1,
                                      float* caller_output) {
  __shared__ __nv_bfloat16 local_partial[kElementsPerCluster];
  __shared__ float local_lse[kHGroup];

  const cg::cluster_group cluster = cg::this_cluster();
  const int role = static_cast<int>(cluster.block_rank());
  const int cluster_index = static_cast<int>(blockIdx.x / kNumCtas);
  const int thread = static_cast<int>(threadIdx.x);
  const std::size_t element_base = static_cast<std::size_t>(cluster_index) * kElementsPerCluster;
  const std::size_t lse_base = static_cast<std::size_t>(cluster_index) * kHGroup;

  if (role == 0 || role == 1) {
    const __nv_bfloat16* source_partial = role == 0 ? partial0 + element_base : partial1 + element_base;
    const float* source_lse = role == 0 ? lse0 + lse_base : lse1 + lse_base;
    for (int index = thread; index < kElementsPerCluster; index += blockDim.x) {
      local_partial[index] = source_partial[index];
    }
    if (thread < kHGroup) {
      local_lse[thread] = source_lse[thread];
    }
  }

  // All four CTAs execute the same lifetime: local writes, cluster handoff,
  // remote reads, then a second cluster handoff before any producer may exit.
  __syncthreads();
  cluster.sync();

  if (role == 2) {
    const __nv_bfloat16* remote_partial0 = cluster.map_shared_rank(local_partial, 0);
    const __nv_bfloat16* remote_partial1 = cluster.map_shared_rank(local_partial, 1);
    const float* remote_lse0 = cluster.map_shared_rank(local_lse, 0);
    const float* remote_lse1 = cluster.map_shared_rank(local_lse, 1);

    for (int index = thread; index < kElementsPerCluster; index += blockDim.x) {
      const int head = index / kHeadDim;
      const float lse_a = remote_lse0[head];
      const float lse_b = remote_lse1[head];
      const float lse_max = fmaxf(lse_a, lse_b);
      const float weight_a = exp2f(lse_a - lse_max);
      const float weight_b = exp2f(lse_b - lse_max);
      const float denominator = weight_a + weight_b;
      const float partial_a = __bfloat162float(remote_partial0[index]);
      const float partial_b = __bfloat162float(remote_partial1[index]);
      caller_output[element_base + index] = (partial_a * weight_a + partial_b * weight_b) / denominator;
    }
  }

  cluster.sync();
}

struct SeedResult {
  int seed = 0;
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  bool finite = true;
  bool sentinel_clean = true;
  bool allclose = true;
};

SeedResult run_seed(int seed) {
  const std::size_t output_elements = static_cast<std::size_t>(kClusters) * kElementsPerCluster;
  const std::size_t lse_elements = static_cast<std::size_t>(kClusters) * kHGroup;
  std::vector<__nv_bfloat16> host_partial0(output_elements);
  std::vector<__nv_bfloat16> host_partial1(output_elements);
  std::vector<float> host_lse0(lse_elements);
  std::vector<float> host_lse1(lse_elements);
  std::vector<float> expected(output_elements);
  std::vector<float> output(output_elements, kSentinel);

  for (int cluster = 0; cluster < kClusters; ++cluster) {
    for (int head = 0; head < kHGroup; ++head) {
      const std::size_t lse_index = static_cast<std::size_t>(cluster) * kHGroup + head;
      host_lse0[lse_index] = synthetic_lse_value(seed, cluster, head, 0);
      host_lse1[lse_index] = synthetic_lse_value(seed, cluster, head, 1);
      for (int dim = 0; dim < kHeadDim; ++dim) {
        const std::size_t index = static_cast<std::size_t>(cluster) * kElementsPerCluster +
                                  static_cast<std::size_t>(head) * kHeadDim + dim;
        host_partial0[index] = __float2bfloat16_rn(synthetic_partial_value(seed, cluster, head, dim, 0));
        host_partial1[index] = __float2bfloat16_rn(synthetic_partial_value(seed, cluster, head, dim, 1));

        const float lse_max = std::max(host_lse0[lse_index], host_lse1[lse_index]);
        const float weight0 = std::exp2(host_lse0[lse_index] - lse_max);
        const float weight1 = std::exp2(host_lse1[lse_index] - lse_max);
        expected[index] =
            (__bfloat162float(host_partial0[index]) * weight0 + __bfloat162float(host_partial1[index]) * weight1) /
            (weight0 + weight1);
      }
    }
  }

  __nv_bfloat16* device_partial0 = nullptr;
  __nv_bfloat16* device_partial1 = nullptr;
  float* device_lse0 = nullptr;
  float* device_lse1 = nullptr;
  float* device_output = nullptr;
  try {
    CUDA_CHECK(cudaMalloc(&device_partial0, output_elements * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&device_partial1, output_elements * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&device_lse0, lse_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_lse1, lse_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_output, output_elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_partial0, host_partial0.data(), output_elements * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_partial1, host_partial1.data(), output_elements * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_lse0, host_lse0.data(), lse_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_lse1, host_lse1.data(), lse_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_output, output.data(), output_elements * sizeof(float), cudaMemcpyHostToDevice));

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

    CUDA_CHECK(cudaLaunchKernelEx(&launch_config, cluster_reduce_kernel,
                                  device_partial0, device_partial1, device_lse0, device_lse1, device_output));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, output_elements * sizeof(float), cudaMemcpyDeviceToHost));
  } catch (...) {
    cudaFree(device_output);
    cudaFree(device_lse1);
    cudaFree(device_lse0);
    cudaFree(device_partial1);
    cudaFree(device_partial0);
    throw;
  }
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_lse1));
  CUDA_CHECK(cudaFree(device_lse0));
  CUDA_CHECK(cudaFree(device_partial1));
  CUDA_CHECK(cudaFree(device_partial0));

  SeedResult result{};
  result.seed = seed;
  for (std::size_t index = 0; index < output_elements; ++index) {
    const float actual = output[index];
    const float reference = expected[index];
    result.finite = result.finite && std::isfinite(actual);
    result.sentinel_clean = result.sentinel_clean && !same_bits(actual, kSentinel);
    const float absolute_error = std::fabs(actual - reference);
    const float relative_error = absolute_error / std::max(std::fabs(reference), 1.0e-7f);
    result.max_abs = std::max(result.max_abs, absolute_error);
    result.max_rel = std::max(result.max_rel, relative_error);
    result.allclose = result.allclose && absolute_error <= kAtol + kRtol * std::fabs(reference);
  }
  return result;
}

void print_success_json(const cudaDeviceProp& property,
                        int runtime_version,
                        int driver_version,
                        int cluster_launch,
                        const std::vector<SeedResult>& results) {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  bool finite = true;
  bool sentinel_clean = true;
  bool allclose = true;
  for (const SeedResult& result : results) {
    max_abs = std::max(max_abs, result.max_abs);
    max_rel = std::max(max_rel, result.max_rel);
    finite = finite && result.finite;
    sentinel_clean = sentinel_clean && result.sentinel_clean;
    allclose = allclose && result.allclose;
  }

  std::cout << std::setprecision(9)
            << "{\"schema\":\"c2-cluster-native-smoke-v1\","
            << "\"status\":\"pass\","
            << "\"boundary\":\"" << json_escape(kBoundary) << "\","
            << "\"mbarrier_phase\":\"" << json_escape(kMBarrierPhase) << "\","
            << "\"sync_api\":\"cooperative_groups::cluster_group::sync\","
            << "\"remote_shared_api\":\"cooperative_groups::cluster_group::map_shared_rank\"," 
            << "\"partial_dtype\":\"bfloat16\"," 
            << "\"global_seed_inputs\":true,\"global_inter_cta_scratch\":false,"
            << "\"caller_owned_output\":true,"
            << "\"num_ctas\":" << kNumCtas << ",\"clusters\":" << kClusters << ","
            << "\"hgroup\":" << kHGroup << ",\"head_dim\":" << kHeadDim << ","
            << "\"threads_per_block\":" << kThreadsPerBlock << ","
            << "\"cuda_runtime\":" << runtime_version << ",\"cuda_driver\":" << driver_version << ","
            << "\"cluster_launch_supported\":" << (cluster_launch != 0 ? "true" : "false") << ","
            << "\"device\":\"" << json_escape(property.name) << "\","
            << "\"capability\":[" << property.major << ',' << property.minor << "],"
            << "\"tolerance\":{\"rtol\":" << kRtol << ",\"atol\":" << kAtol << "},"
            << "\"finite\":" << (finite ? "true" : "false") << ","
            << "\"sentinel_clean\":" << (sentinel_clean ? "true" : "false") << ","
            << "\"allclose\":" << (allclose ? "true" : "false") << ","
            << "\"max_abs\":" << max_abs << ",\"max_rel\":" << max_rel << ",\"seeds\":[";
  for (std::size_t index = 0; index < results.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    const SeedResult& result = results[index];
    std::cout << "{\"seed\":" << result.seed << ",\"max_abs\":" << result.max_abs
              << ",\"max_rel\":" << result.max_rel << ",\"finite\":"
              << (result.finite ? "true" : "false") << ",\"sentinel_clean\":"
              << (result.sentinel_clean ? "true" : "false") << ",\"allclose\":"
              << (result.allclose ? "true" : "false") << '}';
  }
  std::cout << "]}" << std::endl;
}

void print_failure_json(const std::string& error) {
  std::cout << "{\"schema\":\"c2-cluster-native-smoke-v1\",\"status\":\"fail\",\"error\":\""
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
    if (property.major != 10 || property.minor != 3) {
      throw std::runtime_error("requires B300 compute capability 10.3");
    }
    if (cluster_launch == 0) {
      throw std::runtime_error("cudaDevAttrClusterLaunch is false");
    }

    const std::vector<int> seeds{17, 2026};
    std::vector<SeedResult> results;
    results.reserve(seeds.size());
    for (const int seed : seeds) {
      results.push_back(run_seed(seed));
    }
    for (const SeedResult& result : results) {
      if (!result.finite || !result.sentinel_clean || !result.allclose) {
        throw std::runtime_error("device output failed finite, sentinel, or allclose validation");
      }
    }
    print_success_json(property, runtime_version, driver_version, cluster_launch, results);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    print_failure_json(error.what());
    return EXIT_FAILURE;
  }
}
