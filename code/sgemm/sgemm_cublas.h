#pragma once

#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>

namespace sgemm_config {
inline const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

inline void cublas_sgemm(cublasHandle_t handle, const float* d_A, const float* d_B,
                         float* d_C, int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 对外语义是 row-major 的 C(MxN) = A(MxK) * B(KxN)。
    // cuBLAS 默认按 column-major 解释内存；row-major 的 A/B/C 分别等价于
    // column-major 的 A^T(KxM)、B^T(NxK)、C^T(NxM)。因此这里实际计算：
    // C^T(NxM) = B^T(NxK) * A^T(KxM)，输出内存仍可按 row-major C 读取。
    cublasStatus_t status = cublasSgemm(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        N, M, K,
                                        &alpha,
                                        d_B, N,
                                        d_A, K,
                                        &beta,
                                        d_C, N);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error %s:%d: %s\n", __FILE__, __LINE__,
                     cublas_status_string(status));
        std::exit(EXIT_FAILURE);
    }
}

inline double sgemm_gflops(int M, int N, int K, float elapsed_ms) {
    return 2.0 * M * N * K / (elapsed_ms * 1.0e6);
}
}  // namespace sgemm_config

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                        \
        cublasStatus_t status = (call);                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr, "cuBLAS error %s:%d: %s\n", __FILE__, __LINE__, \
                         sgemm_config::cublas_status_string(status));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
