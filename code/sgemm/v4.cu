#include <cuda_runtime.h>

#include "sgemm_config.h"
#include "sgemm_cublas.h"

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

// 最简单的 SGEMM kernel：每个线程计算 C 的一个元素。
// 矩阵均按 row-major 存储：
// A: M x K, B: K x N, C: M x N
template <unsigned int BLOCK_SIZE>
__global__ void sgemm_kernel(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const float* a_ptr = A + (blockIdx.y * blockDim.y) * K;
    const float* b_ptr = B + blockIdx.x * blockDim.x;
    __shared__ float a_shared[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE * BLOCK_SIZE];

    float sumval = 0.0f;
    for(int s=0; s<K; s+=BLOCK_SIZE){
        a_shared[threadIdx.y * BLOCK_SIZE+threadIdx.x] = a_ptr[threadIdx.y * K + threadIdx.x + s];
        b_shared[threadIdx.y * BLOCK_SIZE+threadIdx.x] = b_ptr[(threadIdx.y + s) * N + threadIdx.x];
        __syncthreads();
        for(int k=0; k<BLOCK_SIZE; ++k)
            sumval += a_shared[threadIdx.y * BLOCK_SIZE +k] * b_shared[k * BLOCK_SIZE + threadIdx.x];
        __syncthreads();

    }
    C[row * N+col]=sumval;
}

void sgemm(const float* d_A, const float* d_B, float* d_C,
           int M, int N, int K) {
    dim3 block(sgemm_config::BLOCK_SIZE, sgemm_config::BLOCK_SIZE);
    // x对应第一个维度N，y对应第二个维度M
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    sgemm_kernel<sgemm_config::BLOCK_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void cpu_sgemm(const float* A, const float* B, float* C,
               int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    const int M = sgemm_config::M;
    const int N = sgemm_config::N;
    const int K = sgemm_config::K;

    const size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

    float* h_A = static_cast<float*>(std::malloc(bytes_A));
    float* h_B = static_cast<float*>(std::malloc(bytes_B));
    float* h_C = static_cast<float*>(std::malloc(bytes_C));
    float* h_ref = static_cast<float*>(std::malloc(bytes_C));

    if (h_A == nullptr || h_B == nullptr || h_C == nullptr || h_ref == nullptr) {
        std::fprintf(stderr, "Host malloc failed\n");
        std::free(h_A);
        std::free(h_B);
        std::free(h_C);
        std::free(h_ref);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>((i % 100) - 50) / 100.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>((i % 100) - 50) / 100.0f;
    }
    for (int i = 0; i < M * N; ++i) {
        h_C[i] = 0.0f;
        h_ref[i] = 0.0f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_ref, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    CUDA_CHECK(cudaEventRecord(start));
    sgemm(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    sgemm_config::cublas_sgemm(cublas_handle, d_A, d_B, d_ref, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    sgemm_config::cublas_sgemm(cublas_handle, d_A, d_B, d_ref, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_ref, d_ref, bytes_C, cudaMemcpyDeviceToHost));
    float max_error = sgemm_config::check(h_C, h_ref, M, N);

    double gflops = sgemm_config::sgemm_gflops(M, N, K, elapsed_ms);
    double cublas_gflops = sgemm_config::sgemm_gflops(M, N, K, cublas_elapsed_ms);
    double cublas_ratio = cublas_gflops > 0.0 ? gflops / cublas_gflops * 100.0 : 0.0;
    std::printf("M=%d N=%d K=%d\n", M, N, K);
    std::printf("custom time: %.3f ms\n", elapsed_ms);
    std::printf("custom performance: %.2f GFLOPS\n", gflops);
    std::printf("cuBLAS time: %.3f ms\n", cublas_elapsed_ms);
    std::printf("cuBLAS performance: %.2f GFLOPS\n", cublas_gflops);
    std::printf("custom/cuBLAS: %.2f%%\n", cublas_ratio);
    std::printf("max error: %.6f\n", max_error);

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_ref));

    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
    std::free(h_ref);

    return 0;
}
