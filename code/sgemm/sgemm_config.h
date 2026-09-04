#pragma once

namespace sgemm_config {
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 256;
constexpr unsigned int BLOCK_SIZE = 16;
}  // namespace sgemm_config
