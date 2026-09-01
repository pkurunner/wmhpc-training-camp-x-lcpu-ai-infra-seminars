// C1: a deliberately fair, *logical-work-equivalent* HMMA vs tcgen05 probe.
//
// Variables (also repeated in README.zh-CN.md):
//   C       active reduction length / candidate CHUNK (16, 32, 64)
//   M,N     logical output dimensions (128, 64)
//   K_t     tcgen05 physical reduction tile (64)
//   Q       number of independent CTA tiles in one timed launch
//   A,B,D   BF16 row-major A[M,K_t], BF16 N-major B[N,K_t], FP32 D[M,N]
//
// Both kernels receive exactly the same A/B allocations and contents.  Only
// A[:,0:C] and B[:,0:C] are non-zero; [C:K_t) is initialized to *bit-zero*.
// Hence both implement the identical logical GEMM D[M,N] = A[M,C] B[C,N]
// with BF16 inputs and FP32 accumulation.  tcgen05 necessarily consumes its
// whole 128x64x64 hardware tile, so the zero tail is an intentionally measured
// hardware-padding cost, not extra useful work.
//
// The HMMA path deliberately uses four warps too: each warp serially owns eight
// 16x16 output fragments.  It uses the WMMA API only as a typed spelling of
// `mma.sync` (the audit script rejects a build whose SASS lacks HMMA).  The
// tcgen05 path is the same group::1 protocol and swizzle as assignment M3.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../../cuda/common.h"

namespace {

constexpr int M = 128;
constexpr int N = 64;
constexpr int K_TILE = 64;
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;
namespace wmma = nvcuda::wmma;

struct Samples {
    std::vector<float> ms;
    float median_ms = 0.0f;
};

// Same 128B swizzle as assignment02/cuda/m3_tcgen05/02_single_tile.cu.  Here
// A is MxK and B is logically KxN but stored N-major, so the descriptor sees
// both operands as K-major physical rows.
__host__ __device__ inline int swz128(int row, int col_byte) {
    int atom = row >> 3;
    int r = row & 7;
    int chunk = col_byte >> 4;
    int in16 = col_byte & 15;
    return atom * 1024 + r * 128 + ((chunk ^ r) << 4) + in16;
}

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                             uint32_t sbo, uint32_t layout) {
    uint64_t d = 0;
    d |= static_cast<uint64_t>((saddr >> 4) & 0x3FFF);
    d |= static_cast<uint64_t>((lbo >> 4) & 0x3FFF) << 16;
    d |= static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32;
    d |= static_cast<uint64_t>(1) << 46;  // SM100 descriptor version.
    d |= static_cast<uint64_t>(layout) << 61;
    return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar), "r"(phase));
    }
}

template <int C>
__global__ void hmma_same_work(const __nv_bfloat16* a, const __nv_bfloat16* b,
                               float* d) {
    static_assert(C == 16 || C == 32 || C == 64);
    const int warp = threadIdx.x >> 5;
    const int tile = blockIdx.x;
    const __nv_bfloat16* a_tile = a + static_cast<size_t>(tile) * M * K_TILE;
    const __nv_bfloat16* b_tile = b + static_cast<size_t>(tile) * N * K_TILE;
    float* d_tile = d + static_cast<size_t>(tile) * M * N;

    // Four warps and one CTA match the tcgen05 CTA shape.  There are 8x4
    // independent m16n16 output fragments; each warp owns eight of them.
    for (int mn = warp; mn < (M / 16) * (N / 16); mn += WARPS) {
        const int mt = mn / (N / 16);
        const int nt = mn % (N / 16);
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int kk = 0; kk < C; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                           wmma::row_major>
                a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                           wmma::col_major>
                b_frag;
            // A is [M,K_TILE] row-major.  B is physically [N,K_TILE], which
            // is a [K_TILE,N] col-major matrix with leading dimension K_TILE.
            wmma::load_matrix_sync(a_frag, a_tile + (mt * 16) * K_TILE + kk,
                                   K_TILE);
            wmma::load_matrix_sync(b_frag, b_tile + (nt * 16) * K_TILE + kk,
                                   K_TILE);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        wmma::store_matrix_sync(d_tile + (mt * 16) * N + nt * 16, acc, N,
                                wmma::mem_row_major);
    }
}

// A direct, cta_group::1 tcgen05 path.  It is intentionally kept separate
// from the M3 exercise source so the benchmark cannot change the assignment
// deliverable.  Its physical K is always 64; the host has zero-padded K>C.
__global__ void tcgen05_same_work(const __nv_bfloat16* g_a,
                                  const __nv_bfloat16* g_b, float* g_d) {
    __shared__ __align__(1024) uint8_t s_a[M * K_TILE * 2];
    __shared__ __align__(1024) uint8_t s_b[N * K_TILE * 2];
    __shared__ __align__(8) uint64_t mbar;
    __shared__ uint32_t s_taddr[1];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int tile = blockIdx.x;
    const __nv_bfloat16* a = g_a + static_cast<size_t>(tile) * M * K_TILE;
    const __nv_bfloat16* b = g_b + static_cast<size_t>(tile) * N * K_TILE;
    float* d = g_d + static_cast<size_t>(tile) * M * N;
    const uint32_t mbar_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));

    if (warp == 0) {
        if (lane == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
                             "r"(mbar_u32), "r"(1));
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        const uint32_t dst =
            static_cast<uint32_t>(__cvta_generic_to_shared(s_taddr));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
            "[%0], %1;" ::
            "r"(dst), "r"(64));
        asm volatile(
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    for (int i = tid; i < M * K_TILE; i += blockDim.x) {
        const int row = i / K_TILE;
        const int k = i % K_TILE;
        *reinterpret_cast<__nv_bfloat16*>(&s_a[swz128(row, k * 2)]) = a[i];
    }
    for (int i = tid; i < N * K_TILE; i += blockDim.x) {
        const int row = i / K_TILE;
        const int k = i % K_TILE;
        *reinterpret_cast<__nv_bfloat16*>(&s_b[swz128(row, k * 2)]) = b[i];
    }
    asm volatile("fence.proxy.async.shared::cta;");
    __syncthreads();

    const uint32_t taddr = s_taddr[0];
    uint32_t elected;
    asm volatile(
        "{\n.reg .pred P;\nelect.sync _|P, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, P;\n}"
        : "=r"(elected));
    if (warp == 0 && elected) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        const uint32_t a_base =
            static_cast<uint32_t>(__cvta_generic_to_shared(s_a));
        const uint32_t b_base =
            static_cast<uint32_t>(__cvta_generic_to_shared(s_b));
        // Logical M/N are fixed 128/64.  The descriptor's N/8=8 and M/16=8.
        const uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                               (8u << 17) | (8u << 24);
#pragma unroll
        for (int kk = 0; kk < K_TILE; kk += 16) {
            const uint64_t da = make_desc_sm100(a_base + kk * 2, 0, 1024, 2);
            const uint64_t db = make_desc_sm100(b_base + kk * 2, 0, 1024, 2);
            const uint32_t accum = kk != 0;
            asm volatile(
                "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::
                    "r"(taddr), "l"(da), "l"(db), "r"(idesc), "r"(accum));
        }
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
            ".shared::cluster.b64 [%0];" ::
                "r"(mbar_u32)
            : "memory");
    }
    mbar_wait(mbar_u32, 0);

    asm volatile("tcgen05.fence::after_thread_sync;");
    // 4 warps x 8 N-segments; each collective returns 32 lanes x 8 FP32.
    for (int n = 0; n < N; n += 8) {
        const uint32_t src = taddr + (static_cast<uint32_t>(warp * 32) << 16) + n;
        float r[8];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]),
              "=f"(r[5]), "=f"(r[6]), "=f"(r[7])
            : "r"(src));
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        const int row = warp * 32 + lane;
#pragma unroll
        for (int i = 0; i < 8; ++i) d[row * N + n + i] = r[i];
    }
    asm volatile("tcgen05.fence::before_thread_sync;");
    __syncthreads();
    if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::
                "r"(taddr), "r"(64));
    }
}

template <typename Launch>
Samples event_samples(Launch&& launch, int warmup, int iters, int repeats) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    Samples out;
    out.ms.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        out.ms.push_back(elapsed / iters);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::vector<float> ordered = out.ms;
    std::sort(ordered.begin(), ordered.end());
    out.median_ms = ordered.at(ordered.size() / 2);
    return out;
}

void write_float_array(FILE* f, const std::vector<float>& xs) {
    std::fputc('[', f);
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) std::fputc(',', f);
        std::fprintf(f, "%.9g", xs[i]);
    }
    std::fputc(']', f);
}

template <int C>
void launch_hmma(dim3 grid, const __nv_bfloat16* a, const __nv_bfloat16* b,
                 float* d) {
    hmma_same_work<C><<<grid, THREADS>>>(a, b, d);
}

template <int C>
bool exact_gate_for_c() {
    constexpr int gate_tiles = 2;
    const size_t a_count = static_cast<size_t>(gate_tiles) * M * K_TILE;
    const size_t b_count = static_cast<size_t>(gate_tiles) * N * K_TILE;
    const size_t d_count = static_cast<size_t>(gate_tiles) * M * N;
    std::vector<__nv_bfloat16> h_a(a_count);
    std::vector<__nv_bfloat16> h_b(b_count);
    std::vector<float> h_hmma(d_count), h_tcgen(d_count);
    for (int tile = 0; tile < gate_tiles; ++tile) {
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K_TILE; ++k) {
                const int v = k < C ? ((tile * 19 + m * 7 + k * 3) % 7 - 3) : 0;
                h_a[(static_cast<size_t>(tile) * M + m) * K_TILE + k] =
                    __float2bfloat16(static_cast<float>(v));
            }
        }
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K_TILE; ++k) {
                const int v = k < C ? ((tile * 23 + n * 5 + k * 11) % 7 - 3) : 0;
                h_b[(static_cast<size_t>(tile) * N + n) * K_TILE + k] =
                    __float2bfloat16(static_cast<float>(v));
            }
        }
    }
    __nv_bfloat16 *d_a{}, *d_b{};
    float *d_hmma{}, *d_tcgen{};
    CUDA_CHECK(cudaMalloc(&d_a, a_count * sizeof(*d_a)));
    CUDA_CHECK(cudaMalloc(&d_b, b_count * sizeof(*d_b)));
    CUDA_CHECK(cudaMalloc(&d_hmma, d_count * sizeof(*d_hmma)));
    CUDA_CHECK(cudaMalloc(&d_tcgen, d_count * sizeof(*d_tcgen)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_count * sizeof(*d_a), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_count * sizeof(*d_b), cudaMemcpyHostToDevice));
    launch_hmma<C>(dim3(gate_tiles), d_a, d_b, d_hmma);
    tcgen05_same_work<<<gate_tiles, THREADS>>>(d_a, d_b, d_tcgen);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(h_hmma.data(), d_hmma, d_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tcgen.data(), d_tcgen, d_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_hmma));
    CUDA_CHECK(cudaFree(d_tcgen));
    size_t mismatch = 0;
    for (size_t i = 0; i < d_count; ++i) {
        if (h_hmma[i] != h_tcgen[i]) ++mismatch;
    }
    std::printf("GATE C=%d exact_bitwise=%s mismatches=%zu/%zu\n", C,
                mismatch == 0 ? "PASS" : "FAIL", mismatch, d_count);
    return mismatch == 0;
}

template <int C>
bool bench_c(int tiles, int warmup, int iters, int repeats, FILE* json,
             bool comma_before) {
    const size_t a_count = static_cast<size_t>(tiles) * M * K_TILE;
    const size_t b_count = static_cast<size_t>(tiles) * N * K_TILE;
    const size_t d_count = static_cast<size_t>(tiles) * M * N;
    std::vector<__nv_bfloat16> h_a(a_count);
    std::vector<__nv_bfloat16> h_b(b_count);
    for (int tile = 0; tile < tiles; ++tile) {
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K_TILE; ++k) {
                const int v = k < C ? ((tile * 19 + m * 7 + k * 3) % 7 - 3) : 0;
                h_a[(static_cast<size_t>(tile) * M + m) * K_TILE + k] =
                    __float2bfloat16(static_cast<float>(v));
            }
        }
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K_TILE; ++k) {
                const int v = k < C ? ((tile * 23 + n * 5 + k * 11) % 7 - 3) : 0;
                h_b[(static_cast<size_t>(tile) * N + n) * K_TILE + k] =
                    __float2bfloat16(static_cast<float>(v));
            }
        }
    }
    __nv_bfloat16 *d_a{}, *d_b{};
    float *d_hmma{}, *d_tcgen{};
    CUDA_CHECK(cudaMalloc(&d_a, a_count * sizeof(*d_a)));
    CUDA_CHECK(cudaMalloc(&d_b, b_count * sizeof(*d_b)));
    CUDA_CHECK(cudaMalloc(&d_hmma, d_count * sizeof(*d_hmma)));
    CUDA_CHECK(cudaMalloc(&d_tcgen, d_count * sizeof(*d_tcgen)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_count * sizeof(*d_a), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_count * sizeof(*d_b), cudaMemcpyHostToDevice));

    const dim3 grid(tiles);
    auto hmma_launch = [&]() { launch_hmma<C>(grid, d_a, d_b, d_hmma); };
    auto tcgen_launch = [&]() { tcgen05_same_work<<<grid, THREADS>>>(d_a, d_b, d_tcgen); };
    // Alternate order by C to avoid always handing a warmer GPU to one path;
    // every sample nevertheless contains only one kernel family and one event.
    Samples hmma, tcgen;
    if ((C / 16) % 2 == 1) {
        hmma = event_samples(hmma_launch, warmup, iters, repeats);
        tcgen = event_samples(tcgen_launch, warmup, iters, repeats);
    } else {
        tcgen = event_samples(tcgen_launch, warmup, iters, repeats);
        hmma = event_samples(hmma_launch, warmup, iters, repeats);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const double logical_flop_per_cta = 2.0 * M * N * C;
    const double physical_tcgen_flop_per_cta = 2.0 * M * N * K_TILE;
    const double hmma_tflops = logical_flop_per_cta * tiles / (hmma.median_ms * 1e9);
    const double tcgen_effective_tflops = logical_flop_per_cta * tiles / (tcgen.median_ms * 1e9);
    const double tcgen_physical_tflops = physical_tcgen_flop_per_cta * tiles / (tcgen.median_ms * 1e9);
    const double speedup = hmma.median_ms / tcgen.median_ms;

    std::printf(
        "RESULT C=%d tiles=%d hmma_ms=%.9f tcgen05_ms=%.9f hmma_logical_tflops=%.6f "
        "tcgen05_effective_tflops=%.6f tcgen05_physical_tflops=%.6f hmma_over_tcgen=%.6f "
        "tcgen05_useful_fraction=%.6f%% zero_padded_k=%d\n",
        C, tiles, hmma.median_ms, tcgen.median_ms, hmma_tflops, tcgen_effective_tflops,
        tcgen_physical_tflops, speedup, 100.0 * C / K_TILE, K_TILE - C);

    if (comma_before) std::fputc(',', json);
    std::fprintf(json, "{\n  \"C\": %d,\n", C);
    std::fprintf(json, "  \"logical_gemm_mnk\": [%d,%d,%d],\n", M, N, C);
    std::fprintf(json, "  \"cta_threads\": %d,\n  \"cta_warps\": %d,\n", THREADS, WARPS);
    std::fprintf(json, "  \"same_data_contract\": \"A/B are K_TILE=64 physical arrays; active [0:C) is identical and [C:64) is BF16 zero\",\n");
    std::fprintf(json, "  \"logical_flop_per_cta\": %.0f,\n", logical_flop_per_cta);
    std::fprintf(json, "  \"hmma\": {\n");
    std::fprintf(json, "    \"physical_tile_mnk\": [16,16,16],\n");
    std::fprintf(json, "    \"wmma_mma_calls_per_cta\": %d,\n", 32 * (C / 16));
    std::fprintf(json, "    \"expected_mma_sync_m16n8k16_per_cta\": %d,\n", 64 * (C / 16));
    std::fprintf(json, "    \"accumulator_fragments_16x16_per_cta\": 32,\n");
    std::fprintf(json, "    \"padding_flop_fraction\": 0.0,\n");
    std::fprintf(json, "    \"median_event_ms\": %.9g,\n    \"event_ms_samples\": ", hmma.median_ms);
    write_float_array(json, hmma.ms);
    std::fprintf(json, ",\n    \"logical_tflops\": %.9g\n  },\n", hmma_tflops);
    std::fprintf(json, "  \"tcgen05\": {\n");
    std::fprintf(json, "    \"physical_tile_mnk\": [%d,%d,%d],\n", M, N, K_TILE);
    std::fprintf(json, "    \"tcgen05_mma_commands_per_cta\": 4,\n");
    std::fprintf(json, "    \"tmem_ld_x8_warp_collectives_per_cta\": 32,\n");
    std::fprintf(json, "    \"tmem_output_bytes_per_cta\": %d,\n", M * N * 4);
    std::fprintf(json, "    \"zero_padded_k\": %d,\n", K_TILE - C);
    std::fprintf(json, "    \"hardware_flop_per_cta\": %.0f,\n", physical_tcgen_flop_per_cta);
    std::fprintf(json, "    \"useful_flop_fraction\": %.9g,\n", static_cast<double>(C) / K_TILE);
    std::fprintf(json, "    \"median_event_ms\": %.9g,\n    \"event_ms_samples\": ", tcgen.median_ms);
    write_float_array(json, tcgen.ms);
    std::fprintf(json, ",\n    \"effective_logical_tflops\": %.9g,\n    \"physical_tile_tflops\": %.9g\n  },\n", tcgen_effective_tflops, tcgen_physical_tflops);
    std::fprintf(json, "  \"hmma_over_tcgen_latency_ratio\": %.9g\n}", speedup);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_hmma));
    CUDA_CHECK(cudaFree(d_tcgen));
    return true;
}

struct Args {
    int tiles = 4096;
    int warmup = 20;
    int iters = 50;
    int repeats = 7;
    const char* json_path = nullptr;
};

[[noreturn]] void usage(const char* why) {
    std::fprintf(stderr,
                 "%s\nusage: hmma_tcgen05_fair --json PATH [--tiles N] [--warmup N] "
                 "[--iters N] [--repeats N]\n",
                 why);
    std::exit(2);
}

int positive_int(const char* flag, const char* text) {
    char* end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno || !end || *end || value <= 0 || value > 1'000'000) {
        usage(flag);
    }
    return static_cast<int>(value);
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--json" && i + 1 < argc) {
            args.json_path = argv[++i];
        } else if (flag == "--tiles" && i + 1 < argc) {
            args.tiles = positive_int("--tiles", argv[++i]);
        } else if (flag == "--warmup" && i + 1 < argc) {
            args.warmup = positive_int("--warmup", argv[++i]);
        } else if (flag == "--iters" && i + 1 < argc) {
            args.iters = positive_int("--iters", argv[++i]);
        } else if (flag == "--repeats" && i + 1 < argc) {
            args.repeats = positive_int("--repeats", argv[++i]);
        } else {
            usage("invalid argument");
        }
    }
    if (!args.json_path) usage("--json is required");
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 10) {
        std::fprintf(stderr, "tcgen05 requires SM100+; found %s cc=%d.%d\n", prop.name,
                     prop.major, prop.minor);
        return 3;
    }
    std::printf("DEVICE name=%s cc=%d.%d tiles=%d warmup=%d iters=%d repeats=%d\n",
                prop.name, prop.major, prop.minor, args.tiles, args.warmup,
                args.iters, args.repeats);
    // These use integer BF16 values and FP32 accumulation; every output is
    // exactly representable, so bitwise equality is a real same-work gate.
    const bool gate16 = exact_gate_for_c<16>();
    const bool gate32 = exact_gate_for_c<32>();
    const bool gate64 = exact_gate_for_c<64>();
    if (!(gate16 && gate32 && gate64)) return 4;

    FILE* json = std::fopen(args.json_path, "w");
    if (!json) {
        std::perror("fopen --json");
        return 5;
    }
    std::fprintf(json, "{\n");
    std::fprintf(json, "\"schema\": \"c1_hmma_tcgen05_same_logical_work/v1\",\n");
    std::fprintf(json, "\"device\": {\"name\": \"%s\", \"compute_capability\": [%d,%d]},\n",
                 prop.name, prop.major, prop.minor);
    std::fprintf(json, "\"variables\": {\"C\": \"active K/CHUNK candidate\", \"M\": 128, \"N\": 64, \"K_t\": 64, \"Q\": \"independent CTA tiles\"},\n");
    std::fprintf(json, "\"measurement_contract\": {\"dtype\": \"BF16 inputs, FP32 accumulators and outputs\", \"same_global_data\": true, \"same_cta_threads\": 128, \"same_grid_Q\": %d, \"timing\": \"separate CUDA event medians; no allocation/copy in timed region\", \"correctness\": \"two-tile exact bitwise HMMA-vs-tcgen gate for each C\"},\n", args.tiles);
    std::fprintf(json, "\"results\": [\n");
    bench_c<16>(args.tiles, args.warmup, args.iters, args.repeats, json, false);
    bench_c<32>(args.tiles, args.warmup, args.iters, args.repeats, json, true);
    bench_c<64>(args.tiles, args.warmup, args.iters, args.repeats, json, true);
    std::fprintf(json, "\n],\n\"all_exact_gates_pass\": true\n}\n");
    if (std::fclose(json) != 0) {
        std::perror("fclose --json");
        return 6;
    }
    std::printf("JSON=%s\nALL_GATES=PASS\n", args.json_path);
    return 0;
}
