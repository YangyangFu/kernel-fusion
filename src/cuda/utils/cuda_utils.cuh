#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

// CUDA kernel launch configuration utilities
struct LaunchConfig {
    dim3 block_size;
    dim3 grid_size;
    int shared_mem_size;
    cudaStream_t stream;
    
    LaunchConfig(int64_t total_elements, int block_size_x = 256, cudaStream_t stream = 0);
};

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA kernel error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

// Template utilities for type dispatch
template<typename T>
struct CudaTypeTraits;

template<>
struct CudaTypeTraits<float> {
    using type = float;
    static constexpr int size = sizeof(float);
};

template<>
struct CudaTypeTraits<double> {
    using type = double;
    static constexpr int size = sizeof(double);
};

template<>
struct CudaTypeTraits<at::Half> {
    using type = __half;
    static constexpr int size = sizeof(__half);
};

// Device utility functions
__device__ __forceinline__ float cuda_gelu(float x) {
    // Optimized GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ double cuda_gelu(double x) {
    return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

__device__ __forceinline__ __half cuda_gelu(__half x) {
    float x_float = __half2float(x);
    float result = 0.5f * x_float * (1.0f + tanhf(0.7978845608028654f * (x_float + 0.044715f * x_float * x_float * x_float)));
    return __float2half(result);
}

__device__ __forceinline__ float cuda_relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ double cuda_relu(double x) {
    return fmax(0.0, x);
}

__device__ __forceinline__ __half cuda_relu(__half x) {
    return __hmax(x, __float2half(0.0f));
}

// Tensor access utilities
template<typename T>
__device__ __forceinline__ T* get_data_ptr(torch::Tensor& tensor) {
    return static_cast<T*>(tensor.data_ptr());
}

template<typename T>
__device__ __forceinline__ const T* get_data_ptr(const torch::Tensor& tensor) {
    return static_cast<const T*>(tensor.data_ptr());
}

// Grid stride loop utility
#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
