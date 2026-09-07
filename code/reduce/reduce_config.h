#pragma once

#include <cmath>
#include <cstdio>

namespace reduce_config {
constexpr int N = 1 << 22;
constexpr unsigned int BLOCK_SIZE = 256;

inline float check(float h_Y, float h_ref) {
    float error = std::fabs(h_Y - h_ref);
    if (error >= 1e-4f) {
        std::printf("error: got %.6f, expected %.6f, error %.6f\n",
                    h_Y, h_ref, error);
    }
    return error;
}
}  // namespace reduce_config
