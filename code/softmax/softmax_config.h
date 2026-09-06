#pragma once

#include <cmath>
#include <cstdio>

namespace softmax_config {
constexpr int ROWS = 2048;
constexpr int COLS = 4096;
constexpr unsigned int BLOCK_SIZE = 256;

inline float check(const float* h_Y, const float* h_ref,
                   int rows, int cols) {
    float max_error = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        float error = std::fabs(h_Y[i] - h_ref[i]);
        max_error = std::fmax(max_error, error);
        if (error >= 1e-4f) {
            std::printf("error at index [%d][%d]: got %.6f, expected %.6f, error %.6f\n",
                        i / cols, i % cols, h_Y[i], h_ref[i], error);
        }
    }
    return max_error;
}
}  // namespace softmax_config
