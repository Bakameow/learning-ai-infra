# Norm

这里是一个简单的 CUDA Norm 算子示例集合。

当前包含：

- `v0.cu`：naive RMSNorm，每个 CUDA block 处理一行 hidden states
- `norm_config.h`：默认输入尺寸、epsilon、block size 和结果检查函数
- `CMakeLists.txt`：用于快速编译和运行

RMSNorm 输入输出布局为 row-major：

```text
X: rows x hidden_size
weight: hidden_size
Y: rows x hidden_size
```

计算公式：

```text
rms = sqrt(mean(X[row, :]^2) + eps)
Y[row, col] = X[row, col] / rms * weight[col]
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
cd /mlx_devbox/users/yeliming.void/playground/ai-infra/code/norm
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
build/v0
```

## 运行

方式一：直接运行可执行文件：

```bash
./build/v0
```

方式二：使用 CMake target 运行：

```bash
cmake --build build --target run_v0
```

或者运行所有版本：

```bash
cmake --build build --target run_all
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
rows=2048 hidden_size=4096 eps=0.000010
block_size=256
bytes: 100.68 MB
time: 0.120 ms
bandwidth: 1118.48 GB/s
max error: 0.000000
```

其中：

- `time`：CUDA kernel 执行时间
- `bandwidth`：按两次读取 `X`、一次读取 `weight`、一次写入 `Y` 估算的有效带宽
- `max error`：GPU 结果和 CPU reference 的最大误差

## 清理

删除编译产物：

```bash
rm -rf build
```
