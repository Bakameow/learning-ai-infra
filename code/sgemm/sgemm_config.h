#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace sgemm_config {
constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 2048;
constexpr unsigned int BLOCK_SIZE = 32;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int WARMUP_ITERS = 1;
constexpr int BENCHMARK_ITERS = 10;

inline void check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line,
                     cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

template <typename Func>
inline float benchmark_sgemm_ms(Func&& func, int warmup_iters = WARMUP_ITERS,
                                int benchmark_iters = BENCHMARK_ITERS) {
    for (int i = 0; i < warmup_iters; ++i) {
        func();
    }
    check_cuda(cudaDeviceSynchronize(), __FILE__, __LINE__);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), __FILE__, __LINE__);
    check_cuda(cudaEventCreate(&stop), __FILE__, __LINE__);

    check_cuda(cudaEventRecord(start), __FILE__, __LINE__);
    for (int i = 0; i < benchmark_iters; ++i) {
        func();
    }
    check_cuda(cudaEventRecord(stop), __FILE__, __LINE__);
    check_cuda(cudaEventSynchronize(stop), __FILE__, __LINE__);

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), __FILE__, __LINE__);

    check_cuda(cudaEventDestroy(start), __FILE__, __LINE__);
    check_cuda(cudaEventDestroy(stop), __FILE__, __LINE__);

    return elapsed_ms / benchmark_iters;
}

inline float check(const float* h_C, const float* h_ref, int M, int N) {
    float max_error = 0.0f;
    int printed = 0;
    constexpr int max_print = 10;
    for (int i = 0; i < M * N; ++i) {
        float error = std::fabs(h_C[i] - h_ref[i]);
        max_error = std::fmax(max_error, error);
        if (error >= 1e-4f && printed < max_print) {
            std::printf("error at index [%d][%d]: got %.6f, expected %.6f, error %.6f\n",
                        i / N, i % N, h_C[i], h_ref[i], error);
            ++printed;
        }
    }
    return max_error;
}
}  // namespace sgemm_config
