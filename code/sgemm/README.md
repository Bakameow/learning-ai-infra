# SGEMM

这里是一个最简单的 CUDA SGEMM 示例。

当前包含：

- `v0.cu`：naive SGEMM，每个 CUDA 线程计算输出矩阵 `C` 的一个元素
- `CMakeLists.txt`：用于快速编译和运行

矩阵布局为 row-major：

```text
A: M x K
B: K x N
C: M x N
```

计算公式：

```text
C = A x B
```

## 环境要求

需要系统已安装：

- CUDA Toolkit
- CMake 3.18 或更高版本
- 支持 CUDA 的 NVIDIA GPU

可以简单检查：

```bash
nvcc --version
cmake --version
```

## 编译

进入目录：

```bash
cd /mlx_devbox/users/yeliming.void/playground/ai-infra/code/sgemm
```

生成 build 目录：

```bash
cmake -S . -B build
```

编译：

```bash
cmake --build build -j
```

编译完成后，可执行文件位于：

```bash
build/sgemm_v0
```

## 运行

方式一：直接运行可执行文件：

```bash
./build/sgemm_v0
```

方式二：使用 CMake target 运行：

```bash
cmake --build build --target run
```

或者：

```bash
cmake --build build --target run_v0
```

## 指定 GPU 架构

默认 `CMakeLists.txt` 没有固定 CUDA 架构，方便快速编译。

如果需要指定架构，可以在配置时传入 `CMAKE_CUDA_ARCHITECTURES`。

例如 A100 / Ampere：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
./build/v0
```

例如 H100 / Hopper：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
./build/v0
```

如果已经生成过 build 目录，想重新指定架构，可以先删除旧 build：

```bash
rm -rf build
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
```

## 输出示例

运行后会输出类似：

```text
M=1024 N=1024 K=1024
time: 3.000 ms
performance: 715.83 GFLOPS
max error: 0.000000
```

其中：

- `time`：CUDA kernel 执行时间
- `performance`：根据 `2 * M * N * K / time` 估算得到的 GFLOPS
- `max error`：GPU 结果和 CPU reference 的最大误差

## 清理

删除编译产物：

```bash
rm -rf build
```
