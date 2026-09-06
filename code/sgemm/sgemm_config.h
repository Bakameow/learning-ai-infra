#pragma once

#include <cmath>
#include <cstdio>

namespace sgemm_config {
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 2048;
constexpr unsigned int BLOCK_SIZE = 16;
constexpr unsigned int STRIDE = 2;

inline float check(const float* h_C, const float* h_ref, int M, int N) {
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float error = std::fabs(h_C[i] - h_ref[i]);
        max_error = std::fmax(max_error, error);
        if (error >= 1e-4f) {
            std::printf("error at index [%d][%d]: got %.6f, expected %.6f, error %.6f\n",
                        i / N, i % N, h_C[i], h_ref[i], error);
        }
    }
    return max_error;
}
}  // namespace sgemm_config
