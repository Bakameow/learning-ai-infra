#pragma once

#include <cmath>
#include <cstdio>

namespace sgemm_config {
constexpr int M = 2048;
constexpr int N = 2048;
constexpr int K = 1024;
constexpr unsigned int BLOCK_SIZE = 32;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

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
