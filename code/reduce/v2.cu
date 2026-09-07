#include <cuda_runtime.h>

#include "reduce_config.h"

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


__device__ float warpReduceSum(float val){
    for(int offset=16;offset>0;offset>>=1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
__global__ void reduce_sum_kernel(const float* X, float* Y, int n) {
    const float4* X4 = reinterpret_cast<const float4*>(X);
    int tid = threadIdx.x;
    // int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x % 32;
    int wid = tid / 32;
    int n4 = n/4;

    float val = 0.0f;
    for(int i=blockIdx.x * blockDim.x + threadIdx.x; i<n4; i+=gridDim.x * blockDim.x){
        float4 val4 = X4[i];
        val += ( val4.x + val4.y + val4.z +val4.w);
    }
    
    for(int i=n4*4; i<n; i+=gridDim.x * blockDim.x ){
        val += X[i];
    }
    val = warpReduceSum(val);

    __shared__ float shared[32];
    if(lid==0)shared[wid]=val;
    __syncthreads();

    int warp_num = blockDim.x / 32;
    if (wid==0){
        val = (lid<warp_num)?shared[lid]:0.0f;
        val = warpReduceSum(val);
    }

    if (tid == 0) {
        atomicAdd(Y, val);
    }
}

void reduce_sum(const float* d_X, float* d_Y, int n) {
    dim3 block(reduce_config::BLOCK_SIZE);
    dim3 grid((n + block.x * 2 - 1) / (block.x * 2));

    reduce_sum_kernel<<<grid, block>>>(d_X, d_Y, n);
    CUDA_CHECK(cudaGetLastError());
}

float cpu_reduce_sum(const float* X, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += X[i];
    }
    return sum;
}

int main() {
    const int n = reduce_config::N;

    const size_t bytes_X = static_cast<size_t>(n) * sizeof(float);
    const size_t bytes_Y = sizeof(float);

    float* h_X = static_cast<float*>(std::malloc(bytes_X));
    float h_Y = 0.0f;
    float h_ref = 0.0f;

    if (h_X == nullptr) {
        std::fprintf(stderr, "Host malloc failed\n");
        std::free(h_X);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; ++i) {
        h_X[i] = static_cast<float>((i % 100) - 50) / 1024.0f;
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
    reduce_sum(d_X, d_Y, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(&h_Y, d_Y, bytes_Y, cudaMemcpyDeviceToHost));

    h_ref = cpu_reduce_sum(h_X, n);

    float max_error = reduce_config::check(h_Y, h_ref);

    double bytes = static_cast<double>(bytes_X + bytes_Y);
    double bandwidth = bytes / (elapsed_ms * 1.0e6);
    std::printf("n=%d\n", n);
    std::printf("block_size=%u\n", reduce_config::BLOCK_SIZE);
    std::printf("bytes: %.2f MB\n", bytes / 1.0e6);
    std::printf("time: %.3f ms\n", elapsed_ms);
    std::printf("bandwidth: %.2f GB/s\n", bandwidth);
    std::printf("max error: %.6f\n", max_error);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));

    std::free(h_X);

    return 0;
}
