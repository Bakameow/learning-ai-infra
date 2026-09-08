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

template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int r0 = blockIdx.y * BM;
    const int c0 = blockIdx.x * BN;
    // const float* a_ptr = A + blockIdx.y * BM * K;
    // const float* b_ptr = B + blockIdx.x * BN;

    constexpr int A_BLOCK_X = BK;
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;
    const int a_tx = tid % A_BLOCK_X;
    const int a_ty = tid / A_BLOCK_X;


    constexpr int B_BLOCK_Y = BK;
    constexpr int B_BLOCK_X = BLOCK_SIZE / B_BLOCK_Y;
    const int b_tx = tid % B_BLOCK_X;
    const int b_ty = tid / B_BLOCK_X;

    constexpr int C_BLOCK_X = 16;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;
    constexpr int Tm = BM / C_BLOCK_Y;
    constexpr int Tn = BN / C_BLOCK_X;
    const int c_tx = tid % C_BLOCK_X;
    const int c_ty = tid / C_BLOCK_X;

    float c_t[Tm][Tn] = {0.0f};
    __shared__ float a_shared[BM][BK];
    __shared__ float b_shared[BK][BN];

    for(int k=0; k < K; k += BK){
        #pragma unroll
        for (int i=a_ty; i < BM; i+=A_BLOCK_Y){
            int r = r0 + i, c = a_tx + k;
            a_shared[i][a_tx] = (r<M && c<K) ? A[r * K + c]:0.0f;
        }
        #pragma unroll
        for (int j=b_tx; j < BN; j+=B_BLOCK_X){
            int r = b_ty + k, c = c0 + j;
            b_shared[b_ty][j] = (r<K && c<N) ? B[r * N + c]:0.0f;
        }
        __syncthreads();
        // #pragma unroll
        // for(int i=0; i<Tm; ++i){
        //     int row = c_ty + i * C_BLOCK_Y;
        //     for(int j=0; j<Tn; ++j){
        //         int col = c_tx + j * C_BLOCK_X;
        //         for(int p=0;p<BK;++p){
        //             c_t[i][j] += a_shared[row][p] * b_shared[p][col];
        //         }
        //     }
        // }
        for(int p=0;p<BK;++p){
            for(int i=0; i<Tm; ++i){
                int row = c_ty + i * C_BLOCK_Y;
                for(int j=0; j<Tn; ++j){
                    int col = c_tx + j * C_BLOCK_X;
                    c_t[i][j] += a_shared[row][p] * b_shared[p][col];
                }
            }
        }
        __syncthreads();
    }
    
    for(int i=0; i<Tm; ++i){
        int r = r0 + c_ty + i * C_BLOCK_Y;
        for(int j=0; j<Tn; ++j){
            int c = c0 + c_tx + j * C_BLOCK_X;
            if(r<M && c<N)C[r*N +c] = c_t[i][j];
        }
    }
}

void sgemm(const float* d_A, const float* d_B, float* d_C,
           int M, int N, int K) {
    dim3 block(sgemm_config::BLOCK_SIZE, sgemm_config::BLOCK_SIZE);
    // x对应第一个维度N，y对应第二个维度M
    dim3 grid((N + sgemm_config::BN - 1) / sgemm_config::BN,
              (M + sgemm_config::BM - 1) / sgemm_config::BM);

    sgemm_kernel<sgemm_config::BM,sgemm_config::BN,sgemm_config::BK,sgemm_config::BLOCK_SIZE*sgemm_config::BLOCK_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

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

    cpu_sgemm(h_A, h_B, h_ref, M, N, K);

    float max_error = sgemm_config::check(h_C, h_ref, M, N);

    sgemm_config::cublas_sgemm(cublas_handle, d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    sgemm_config::cublas_sgemm(cublas_handle, d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_elapsed_ms, start, stop));

    double gflops = sgemm_config::sgemm_gflops(M, N, K, elapsed_ms);
    double cublas_gflops = sgemm_config::sgemm_gflops(M, N, K, cublas_elapsed_ms);
    std::printf("M=%d N=%d K=%d\n", M, N, K);
    std::printf("custom time: %.3f ms\n", elapsed_ms);
    std::printf("custom performance: %.2f GFLOPS\n", gflops);
    std::printf("cuBLAS time: %.3f ms\n", cublas_elapsed_ms);
    std::printf("cuBLAS performance: %.2f GFLOPS\n", cublas_gflops);
    std::printf("max error: %.6f\n", max_error);

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
    std::free(h_ref);

    return 0;
}
