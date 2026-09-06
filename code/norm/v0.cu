#include <cuda_runtime.h>

#include "norm_config.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                              \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// 最简单的 RMSNorm kernel：每个 block 处理一行。
// X/Y 按 row-major 存储：X: rows x hidden_size, Y: rows x hidden_size。
template <unsigned int BLOCK_SIZE>
__global__ void rmsnorm_kernel(const float* X, const float* weight, float* Y,
                               int rows, int hidden_size, float eps) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float sum_sq_shared[BLOCK_SIZE];

    const float* row_X = X + static_cast<size_t>(row) * hidden_size;
    float* row_Y = Y + static_cast<size_t>(row) * hidden_size;

    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < hidden_size; col += BLOCK_SIZE) {
        float x = row_X[col];
        sum_sq += x * x;
    }
    sum_sq_shared[threadIdx.x] = sum_sq;
    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_sq_shared[threadIdx.x] += sum_sq_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(sum_sq_shared[0] / hidden_size + eps);
    for (int col = threadIdx.x; col < hidden_size; col += BLOCK_SIZE) {
        row_Y[col] = row_X[col] * inv_rms * weight[col];
    }
}

void rmsnorm(const float* d_X, const float* d_weight, float* d_Y,
             int rows, int hidden_size, float eps) {
    dim3 block(norm_config::BLOCK_SIZE);
    dim3 grid(rows);

    rmsnorm_kernel<norm_config::BLOCK_SIZE><<<grid, block>>>(
        d_X, d_weight, d_Y, rows, hidden_size, eps);
    CUDA_CHECK(cudaGetLastError());
}

void cpu_rmsnorm(const float* X, const float* weight, float* Y,
                 int rows, int hidden_size, float eps) {
    for (int row = 0; row < rows; ++row) {
        const float* row_X = X + static_cast<size_t>(row) * hidden_size;
        float* row_Y = Y + static_cast<size_t>(row) * hidden_size;

        float sum_sq = 0.0f;
        for (int col = 0; col < hidden_size; ++col) {
            float x = row_X[col];
            sum_sq += x * x;
        }

        float inv_rms = 1.0f / std::sqrt(sum_sq / hidden_size + eps);
        for (int col = 0; col < hidden_size; ++col) {
            row_Y[col] = row_X[col] * inv_rms * weight[col];
        }
    }
}

int main() {
    const int rows = norm_config::ROWS;
    const int hidden_size = norm_config::HIDDEN_SIZE;
    const float eps = norm_config::EPS;

    const size_t elements = static_cast<size_t>(rows) * hidden_size;
    const size_t bytes_X = elements * sizeof(float);
    const size_t bytes_weight = static_cast<size_t>(hidden_size) * sizeof(float);
    const size_t bytes_Y = elements * sizeof(float);

    float* h_X = static_cast<float*>(std::malloc(bytes_X));
    float* h_weight = static_cast<float*>(std::malloc(bytes_weight));
    float* h_Y = static_cast<float*>(std::malloc(bytes_Y));
    float* h_ref = static_cast<float*>(std::malloc(bytes_Y));

    if (h_X == nullptr || h_weight == nullptr || h_Y == nullptr || h_ref == nullptr) {
        std::fprintf(stderr, "Host malloc failed\n");
        std::free(h_X);
        std::free(h_weight);
        std::free(h_Y);
        std::free(h_ref);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < elements; ++i) {
        h_X[i] = static_cast<float>((i % 200) - 100) / 100.0f;
        h_Y[i] = 0.0f;
        h_ref[i] = 0.0f;
    }
    for (int i = 0; i < hidden_size; ++i) {
        h_weight[i] = 1.0f + static_cast<float>(i % 17) / 100.0f;
    }

    float *d_X = nullptr, *d_weight = nullptr, *d_Y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_weight, bytes_weight));
    CUDA_CHECK(cudaMalloc(&d_Y, bytes_Y));

    CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes_X, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, bytes_weight, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_Y, 0, bytes_Y));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    rmsnorm(d_X, d_weight, d_Y, rows, hidden_size, eps);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, bytes_Y, cudaMemcpyDeviceToHost));

    cpu_rmsnorm(h_X, h_weight, h_ref, rows, hidden_size, eps);

    float max_error = norm_config::check(h_Y, h_ref, rows, hidden_size);

    double bytes = (static_cast<double>(elements) * 3.0 + hidden_size) * sizeof(float);
    double bandwidth = bytes / (elapsed_ms * 1.0e6);
    std::printf("rows=%d hidden_size=%d eps=%.6f\n", rows, hidden_size, eps);
    std::printf("block_size=%u\n", norm_config::BLOCK_SIZE);
    std::printf("bytes: %.2f MB\n", bytes / 1.0e6);
    std::printf("time: %.3f ms\n", elapsed_ms);
    std::printf("bandwidth: %.2f GB/s\n", bandwidth);
    std::printf("max error: %.6f\n", max_error);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_Y));

    std::free(h_X);
    std::free(h_weight);
    std::free(h_Y);
    std::free(h_ref);

    return 0;
}
