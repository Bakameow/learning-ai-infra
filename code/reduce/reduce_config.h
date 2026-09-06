#pragma once

#include <cmath>
#include <cstdio>

namespace reduce_config {
constexpr int ROWS = 2048;
constexpr int COLS = 4096;
constexpr unsigned int BLOCK_SIZE = 256;

inline float check(const float* h_Y, const float* h_ref, int rows) {
    float max_error = 0.0f;
    for (int i = 0; i < rows; ++i) {
        float error = std::fabs(h_Y[i] - h_ref[i]);
        max_error = std::fmax(max_error, error);
        if (error >= 1e-4f) {
            std::printf("error at index [%d]: got %.6f, expected %.6f, error %.6f\n",
                        i, h_Y[i], h_ref[i], error);
        }
    }
    return max_error;
}
}  // namespace reduce_config
