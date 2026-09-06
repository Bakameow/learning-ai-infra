#include <cuda_runtime.h>

#include "softmax_config.h"

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

// 最简单的 softmax kernel：每个 block 处理一行。
// X/Y 按 row-major 存储：X: rows x cols, Y: rows x cols。
template <unsigned int BLOCK_SIZE>
__global__ void softmax_kernel(const float* X, float* Y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float shared[BLOCK_SIZE];

    const float* row_X = X + static_cast<size_t>(row) * cols;
    float* row_Y = Y + static_cast<size_t>(row) * cols;

    float max_value = -INFINITY;
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        max_value = fmaxf(max_value, row_X[col]);
    }
    shared[threadIdx.x] = max_value;
    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    max_value = shared[0];

    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        float value = expf(row_X[col] - max_value);
        row_Y[col] = value;
        sum += value;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    sum = shared[0];

    for (int col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        row_Y[col] /= sum;
    }
}

void softmax(const float* d_X, float* d_Y, int rows, int cols) {
    dim3 block(softmax_config::BLOCK_SIZE);
    dim3 grid(rows);

    softmax_kernel<softmax_config::BLOCK_SIZE><<<grid, block>>>(d_X, d_Y, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void cpu_softmax(const float* X, float* Y, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        const float* row_X = X + static_cast<size_t>(row) * cols;
        float* row_Y = Y + static_cast<size_t>(row) * cols;

        float max_value = -INFINITY;
        for (int col = 0; col < cols; ++col) {
            max_value = std::fmax(max_value, row_X[col]);
        }

        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            float value = std::exp(row_X[col] - max_value);
            row_Y[col] = value;
            sum += value;
        }

        for (int col = 0; col < cols; ++col) {
            row_Y[col] /= sum;
        }
    }
}

int main() {
    const int rows = softmax_config::ROWS;
    const int cols = softmax_config::COLS;

    const size_t elements = static_cast<size_t>(rows) * cols;
    const size_t bytes_X = elements * sizeof(float);
    const size_t bytes_Y = elements * sizeof(float);

    float* h_X = static_cast<float*>(std::malloc(bytes_X));
    float* h_Y = static_cast<float*>(std::malloc(bytes_Y));
    float* h_ref = static_cast<float*>(std::malloc(bytes_Y));

    if (h_X == nullptr || h_Y == nullptr || h_ref == nullptr) {
        std::fprintf(stderr, "Host malloc failed\n");
        std::free(h_X);
        std::free(h_Y);
        std::free(h_ref);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < elements; ++i) {
        h_X[i] = static_cast<float>(static_cast<int>(i % 200) - 100) / 100.0f;
        h_Y[i] = 0.0f;
        h_ref[i] = 0.0f;
    }

    float *d_X = nullptr, *d_Y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_Y, bytes_Y));

    CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes_X, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_Y, 0, bytes_Y));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    softmax(d_X, d_Y, rows, cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, bytes_Y, cudaMemcpyDeviceToHost));

    cpu_softmax(h_X, h_ref, rows, cols);

    float max_error = softmax_config::check(h_Y, h_ref, rows, cols);

    double bytes = static_cast<double>(bytes_X) * 2.0 + bytes_Y * 2.0;
    double bandwidth = bytes / (elapsed_ms * 1.0e6);
    std::printf("rows=%d cols=%d\n", rows, cols);
    std::printf("block_size=%u\n", softmax_config::BLOCK_SIZE);
    std::printf("bytes: %.2f MB\n", bytes / 1.0e6);
    std::printf("time: %.3f ms\n", elapsed_ms);
    std::printf("bandwidth: %.2f GB/s\n", bandwidth);
    std::printf("max error: %.6f\n", max_error);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));

    std::free(h_X);
    std::free(h_Y);
    std::free(h_ref);

    return 0;
}
