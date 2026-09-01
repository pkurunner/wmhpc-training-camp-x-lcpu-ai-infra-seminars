// C2 discussion-point-3 microbenchmark: a two-level paged gather cannot be
// encoded by one TMA tensor map.  It measures the honest alternative: gather
// with ordinary CUDA loads first, then use TMA only after the result is
// contiguous.  The three timed paths all have two equal-sized transfer legs:
//
//   A) random two-level gather -> ordinary contiguous copy;
//   B) random two-level gather -> TMA contiguous copy;
//   C) already-contiguous input -> ordinary contiguous copy -> TMA copy.
//
// Thus A/B differ only in their second leg, while B/C differ only in whether
// the first leg needs the runtime page-table lookup.  All paths write the same
// [batch * topk, page_elems] uint32 output and are checked against the same
// host reference before timing.  The source intentionally uses real
// cp.async.bulk.tensor.2d with a CUtensorMap; it is not a paper-only model.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",             \
                         cudaGetErrorName(_err), __FILE__, __LINE__,         \
                         cudaGetErrorString(_err));                          \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define CU_CHECK(call)                                                       \
    do {                                                                     \
        CUresult _err = (call);                                              \
        if (_err != CUDA_SUCCESS) {                                          \
            const char* _msg = nullptr;                                     \
            cuGetErrorString(_err, &_msg);                                  \
            std::fprintf(stderr, "CUDA driver error at %s:%d: %s\n",         \
                         __FILE__, __LINE__, _msg ? _msg : "unknown");      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

struct Config {
    int batch = 4;
    int topk = 16;
    int logical_pages = 64;
    int physical_pages = 128;
    int page_elems = 1024;  // 4 KiB uint32 page; inner TMA dimension.
    int warmup = 20;
    int iters = 100;
    std::string json_path;
};

// Keep each TMA box at 256 bytes.  A 4 KiB logical page is copied by sixteen
// boxes under one transaction barrier; this stays within tensor-map box
// limits while preserving the page-sized endpoint used by all three paths.
constexpr int TMA_BOX_ELEMS = 64;

struct Timing {
    float milliseconds = 0.0f;
    double effective_gbps = 0.0;
};

static void usage(const char* program) {
    std::fprintf(
        stderr,
        "usage: %s [--batch N] [--topk N] [--logical-pages N] "
        "[--physical-pages N] [--page-elems N] [--warmup N] [--iters N] "
        "[--json FILE]\n",
        program);
}

static int parse_positive(const char* value, const char* flag) {
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (!value[0] || *end || parsed <= 0 || parsed > INT32_MAX) {
        std::fprintf(stderr, "invalid %s: %s\n", flag, value);
        std::exit(2);
    }
    return static_cast<int>(parsed);
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto next = [&](const char* flag) -> const char* {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[i];
        };
        if (arg == "--batch") cfg.batch = parse_positive(next("--batch"), "--batch");
        else if (arg == "--topk") cfg.topk = parse_positive(next("--topk"), "--topk");
        else if (arg == "--logical-pages") cfg.logical_pages = parse_positive(next("--logical-pages"), "--logical-pages");
        else if (arg == "--physical-pages") cfg.physical_pages = parse_positive(next("--physical-pages"), "--physical-pages");
        else if (arg == "--page-elems") cfg.page_elems = parse_positive(next("--page-elems"), "--page-elems");
        else if (arg == "--warmup") cfg.warmup = parse_positive(next("--warmup"), "--warmup");
        else if (arg == "--iters") cfg.iters = parse_positive(next("--iters"), "--iters");
        else if (arg == "--json") cfg.json_path = next("--json");
        else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            std::exit(2);
        }
    }
    if (cfg.topk > cfg.logical_pages || cfg.physical_pages < cfg.logical_pages) {
        std::fprintf(stderr, "require topk <= logical-pages <= physical-pages\n");
        std::exit(2);
    }
    if (cfg.page_elems % TMA_BOX_ELEMS != 0) {
        std::fprintf(stderr, "page-elems must be a multiple of %d for tiled TMA\n",
                     TMA_BOX_ELEMS);
        std::exit(2);
    }
    return cfg;
}

// topk is [batch, topk] logical page ids.  table is [batch, logical_pages]
// physical page ids.  Each output slot has exactly one selected physical page.
__global__ void gather_two_level_u32(const uint32_t* __restrict__ src,
                                     const int* __restrict__ topk,
                                     const int* __restrict__ table,
                                     uint32_t* __restrict__ gathered,
                                     int topk_count, int logical_pages,
                                     int page_elems) {
    const int slot = blockIdx.x;
    const int request = slot / topk_count;
    const int logical_page = topk[slot];
    const int physical_page = table[request * logical_pages + logical_page];
    const size_t source_base = static_cast<size_t>(physical_page) * page_elems;
    const size_t destination_base = static_cast<size_t>(slot) * page_elems;
    for (int element = threadIdx.x; element < page_elems; element += blockDim.x) {
        gathered[destination_base + element] = src[source_base + element];
    }
}

__global__ void copy_linear_u32(const uint32_t* __restrict__ src,
                                uint32_t* __restrict__ dst, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = src[i];
}

__device__ __forceinline__ void mbarrier_wait(uint32_t barrier, uint32_t phase) {
    uint32_t ready = 0;
    while (!ready) {
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(ready)
            : "r"(barrier), "r"(phase));
    }
}

// One CTA moves one contiguous page.  The tensor map has dimensions
// [page_elems, slots], and this launch always requests coordinates {0, slot}.
// That fixed affine coordinate is exactly the condition that is absent before
// the software two-level gather.
__global__ void tma_contiguous_copy_u32(
    uint32_t* __restrict__ dst, int page_elems,
    const __grid_constant__ CUtensorMap source_map) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(smem_raw) + 127u) &
        ~static_cast<uintptr_t>(127u));
    __shared__ __align__(8) uint64_t full;
    const int tid = threadIdx.x;
    const uint32_t full_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&full));
    const uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem));

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                         "r"(full_addr), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    if (tid == 0) {
        const uint32_t bytes = static_cast<uint32_t>(page_elems * sizeof(uint32_t));
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
                "r"(full_addr), "r"(bytes)
            : "memory");
        for (int chunk = 0; chunk < page_elems; chunk += TMA_BOX_ELEMS) {
            const uint32_t destination =
                smem_addr + static_cast<uint32_t>(chunk * sizeof(uint32_t));
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
                "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::
                    "r"(destination), "l"(&source_map), "r"(chunk),
                    "r"(blockIdx.x), "r"(full_addr)
                : "memory");
        }
    }
    mbarrier_wait(full_addr, 0);
    // TMA writes use the async proxy; the consumer below uses ordinary generic
    // shared loads, so make the proxy write visible before consuming it.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    const uint32_t* page = reinterpret_cast<const uint32_t*>(smem);
    const size_t output_base = static_cast<size_t>(blockIdx.x) * page_elems;
    for (int element = tid; element < page_elems; element += blockDim.x) {
        dst[output_base + element] = page[element];
    }
}

static CUtensorMap make_page_tensor_map(uint32_t* base, int slots, int page_elems) {
    CUtensorMap map = {};
    const cuuint64_t dimensions[2] = {
        static_cast<cuuint64_t>(page_elems), static_cast<cuuint64_t>(slots)};
    const cuuint64_t strides[1] = {
        static_cast<cuuint64_t>(page_elems) * sizeof(uint32_t)};
    const cuuint32_t box[2] = {TMA_BOX_ELEMS, 1};
    const cuuint32_t element_strides[2] = {1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, base, dimensions, strides, box,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

template <typename Launch>
static Timing measure_cuda_events(Launch&& launch, int warmup, int iters,
                                  double bytes_per_iteration) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    Timing result;
    result.milliseconds = total_ms / iters;
    result.effective_gbps = bytes_per_iteration / (result.milliseconds * 1.0e6);
    return result;
}

static long mismatch_count(const std::vector<uint32_t>& got,
                           const std::vector<uint32_t>& expected) {
    long mismatches = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != expected[i]) {
            if (mismatches < 5) {
                std::fprintf(stderr, "mismatch[%zu]: got=0x%08x want=0x%08x\n", i,
                             got[i], expected[i]);
            }
            ++mismatches;
        }
    }
    return mismatches;
}

static void write_json(const std::string& path, const Config& cfg,
                       size_t payload_bytes, long bad_sw, long bad_gather_tma,
                       long bad_contig_tma, const Timing& sw,
                       const Timing& gather_tma, const Timing& contig_tma) {
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "cannot write JSON: %s\n", path.c_str());
        std::exit(1);
    }
    out << "{\n"
        << "  \"schema\": \"c2-tma-two-level-gather-v1\",\n"
        << "  \"batch\": " << cfg.batch << ",\n"
        << "  \"topk\": " << cfg.topk << ",\n"
        << "  \"logical_pages\": " << cfg.logical_pages << ",\n"
        << "  \"physical_pages\": " << cfg.physical_pages << ",\n"
        << "  \"page_elems\": " << cfg.page_elems << ",\n"
        << "  \"payload_bytes\": " << payload_bytes << ",\n"
        << "  \"estimated_read_write_bytes_per_path\": " << (4 * payload_bytes) << ",\n"
        << "  \"correctness\": {\n"
        << "    \"software_staged_mismatches\": " << bad_sw << ",\n"
        << "    \"gather_then_tma_mismatches\": " << bad_gather_tma << ",\n"
        << "    \"contiguous_then_tma_mismatches\": " << bad_contig_tma << "\n"
        << "  },\n"
        << "  \"timing_ms_cuda_event\": {\n"
        << "    \"software_staged\": " << sw.milliseconds << ",\n"
        << "    \"gather_then_tma\": " << gather_tma.milliseconds << ",\n"
        << "    \"contiguous_then_tma\": " << contig_tma.milliseconds << "\n"
        << "  },\n"
        << "  \"effective_gbps_estimated_read_write\": {\n"
        << "    \"software_staged\": " << sw.effective_gbps << ",\n"
        << "    \"gather_then_tma\": " << gather_tma.effective_gbps << ",\n"
        << "    \"contiguous_then_tma\": " << contig_tma.effective_gbps << "\n"
        << "  }\n"
        << "}\n";
}

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);
    const int slots = cfg.batch * cfg.topk;
    const size_t payload_elems = static_cast<size_t>(slots) * cfg.page_elems;
    const size_t payload_bytes = payload_elems * sizeof(uint32_t);
    const size_t source_elems =
        static_cast<size_t>(cfg.physical_pages) * cfg.page_elems;

    std::vector<uint32_t> h_source(source_elems);
    for (int page = 0; page < cfg.physical_pages; ++page) {
        for (int element = 0; element < cfg.page_elems; ++element) {
            h_source[static_cast<size_t>(page) * cfg.page_elems + element] =
                (static_cast<uint32_t>(page) << 20) ^ static_cast<uint32_t>(element);
        }
    }
    std::mt19937 rng(20260819);
    std::vector<int> h_table(static_cast<size_t>(cfg.batch) * cfg.logical_pages);
    std::vector<int> h_topk(slots);
    for (int request = 0; request < cfg.batch; ++request) {
        std::vector<int> page_permutation(cfg.physical_pages);
        std::iota(page_permutation.begin(), page_permutation.end(), 0);
        std::shuffle(page_permutation.begin(), page_permutation.end(), rng);
        for (int logical = 0; logical < cfg.logical_pages; ++logical) {
            h_table[request * cfg.logical_pages + logical] = page_permutation[logical];
        }
        std::vector<int> logical_permutation(cfg.logical_pages);
        std::iota(logical_permutation.begin(), logical_permutation.end(), 0);
        std::shuffle(logical_permutation.begin(), logical_permutation.end(), rng);
        for (int rank = 0; rank < cfg.topk; ++rank) {
            h_topk[request * cfg.topk + rank] = logical_permutation[rank];
        }
    }

    // Independently materialize the selected logical pages on host.  This is
    // the strict expected output for all three endpoint buffers.
    std::vector<uint32_t> h_expected(payload_elems);
    for (int slot = 0; slot < slots; ++slot) {
        const int request = slot / cfg.topk;
        const int logical = h_topk[slot];
        const int physical = h_table[request * cfg.logical_pages + logical];
        std::copy_n(h_source.data() + static_cast<size_t>(physical) * cfg.page_elems,
                    cfg.page_elems,
                    h_expected.data() + static_cast<size_t>(slot) * cfg.page_elems);
    }

    uint32_t *d_source = nullptr, *d_contiguous = nullptr;
    uint32_t *d_buf_sw = nullptr, *d_out_sw = nullptr;
    uint32_t *d_buf_tma = nullptr, *d_out_tma = nullptr;
    uint32_t *d_buf_contig = nullptr, *d_out_contig = nullptr;
    int *d_topk = nullptr, *d_table = nullptr;
    CUDA_CHECK(cudaMalloc(&d_source, source_elems * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_contiguous, payload_bytes));
    CUDA_CHECK(cudaMalloc(&d_topk, h_topk.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_table, h_table.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_buf_sw, payload_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_sw, payload_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf_tma, payload_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_tma, payload_bytes));
    CUDA_CHECK(cudaMalloc(&d_buf_contig, payload_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_contig, payload_bytes));
    CUDA_CHECK(cudaMemcpy(d_source, h_source.data(), source_elems * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_contiguous, h_expected.data(), payload_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_topk, h_topk.data(), h_topk.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table, h_table.data(), h_table.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    // The maps contain only base, dimensions, strides and box sizes.  There is
    // no field for topk/table, which is the concrete API-level reason that one
    // descriptor cannot encode the two load-dependent address calculation.
    const CUtensorMap gathered_map = make_page_tensor_map(d_buf_tma, slots, cfg.page_elems);
    const CUtensorMap contiguous_map =
        make_page_tensor_map(d_buf_contig, slots, cfg.page_elems);
    constexpr int threads = 256;
    const int linear_blocks = static_cast<int>((payload_elems + threads - 1) / threads);
    const size_t tma_smem_bytes = payload_bytes / slots + 128;
    CUDA_CHECK(cudaFuncSetAttribute(tma_contiguous_copy_u32,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(tma_smem_bytes)));

    auto launch_software_staged = [&] {
        gather_two_level_u32<<<slots, threads>>>(d_source, d_topk, d_table, d_buf_sw,
                                                  cfg.topk, cfg.logical_pages,
                                                  cfg.page_elems);
        copy_linear_u32<<<linear_blocks, threads>>>(d_buf_sw, d_out_sw, payload_elems);
    };
    auto launch_gather_then_tma = [&] {
        gather_two_level_u32<<<slots, threads>>>(d_source, d_topk, d_table, d_buf_tma,
                                                  cfg.topk, cfg.logical_pages,
                                                  cfg.page_elems);
        tma_contiguous_copy_u32<<<slots, threads, tma_smem_bytes>>>(
            d_out_tma, cfg.page_elems, gathered_map);
    };
    auto launch_contiguous_then_tma = [&] {
        copy_linear_u32<<<linear_blocks, threads>>>(d_contiguous, d_buf_contig,
                                                     payload_elems);
        tma_contiguous_copy_u32<<<slots, threads, tma_smem_bytes>>>(
            d_out_contig, cfg.page_elems, contiguous_map);
    };

    launch_software_staged();
    launch_gather_then_tma();
    launch_contiguous_then_tma();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> h_out_sw(payload_elems), h_out_tma(payload_elems),
        h_out_contig(payload_elems);
    CUDA_CHECK(cudaMemcpy(h_out_sw.data(), d_out_sw, payload_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_tma.data(), d_out_tma, payload_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_contig.data(), d_out_contig, payload_bytes,
                          cudaMemcpyDeviceToHost));
    const long bad_sw = mismatch_count(h_out_sw, h_expected);
    const long bad_gather_tma = mismatch_count(h_out_tma, h_expected);
    const long bad_contig_tma = mismatch_count(h_out_contig, h_expected);
    if (bad_sw || bad_gather_tma || bad_contig_tma) {
        std::fprintf(stderr,
                     "CORRECTNESS_FAIL software_staged=%ld gather_then_tma=%ld "
                     "contiguous_then_tma=%ld\n",
                     bad_sw, bad_gather_tma, bad_contig_tma);
        return 3;
    }
    std::printf("CORRECTNESS_PASS software_staged=0 gather_then_tma=0 "
                "contiguous_then_tma=0\n");

    // Every timed path has exactly two read+write legs: 4 * payload bytes.
    // It is an effective-throughput convention, not a claim of physical DRAM
    // bytes (cache behavior is intentionally left to a profiler follow-up).
    const double estimated_read_write_bytes = 4.0 * static_cast<double>(payload_bytes);
    const Timing software_staged = measure_cuda_events(
        launch_software_staged, cfg.warmup, cfg.iters, estimated_read_write_bytes);
    const Timing gather_then_tma = measure_cuda_events(
        launch_gather_then_tma, cfg.warmup, cfg.iters, estimated_read_write_bytes);
    const Timing contiguous_then_tma = measure_cuda_events(
        launch_contiguous_then_tma, cfg.warmup, cfg.iters, estimated_read_write_bytes);
    std::printf("CONFIG batch=%d topk=%d logical_pages=%d physical_pages=%d "
                "page_elems=%d slots=%d payload_bytes=%zu staged_rw_bytes=%zu\n",
                cfg.batch, cfg.topk, cfg.logical_pages, cfg.physical_pages,
                cfg.page_elems, slots, payload_bytes, 4 * payload_bytes);
    std::printf("CUDA_EVENT_MS software_staged=%.6f gather_then_tma=%.6f "
                "contiguous_then_tma=%.6f\n",
                software_staged.milliseconds, gather_then_tma.milliseconds,
                contiguous_then_tma.milliseconds);
    std::printf("EFFECTIVE_GBPS_EST software_staged=%.3f gather_then_tma=%.3f "
                "contiguous_then_tma=%.3f\n",
                software_staged.effective_gbps, gather_then_tma.effective_gbps,
                contiguous_then_tma.effective_gbps);
    write_json(cfg.json_path, cfg, payload_bytes, bad_sw, bad_gather_tma,
               bad_contig_tma, software_staged, gather_then_tma,
               contiguous_then_tma);
    std::printf("RESULT=PASS TMA_STATIC_CONCLUSION="
                "single_tensor_map_has_fixed_affine_coordinates_not_two_level_indirection\n");

    CUDA_CHECK(cudaFree(d_source));
    CUDA_CHECK(cudaFree(d_contiguous));
    CUDA_CHECK(cudaFree(d_topk));
    CUDA_CHECK(cudaFree(d_table));
    CUDA_CHECK(cudaFree(d_buf_sw));
    CUDA_CHECK(cudaFree(d_out_sw));
    CUDA_CHECK(cudaFree(d_buf_tma));
    CUDA_CHECK(cudaFree(d_out_tma));
    CUDA_CHECK(cudaFree(d_buf_contig));
    CUDA_CHECK(cudaFree(d_out_contig));
    return 0;
}
