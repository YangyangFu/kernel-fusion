#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

// Import type traits and activations
#include "type_traits.cuh"
#include "activations.cuh"

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

// Grid stride loop utility
#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Tensor access utilities
template<typename T>
__device__ __forceinline__ T* get_data_ptr(torch::Tensor& tensor) {
    return static_cast<T*>(tensor.data_ptr());
}

template<typename T>
__device__ __forceinline__ const T* get_data_ptr(const torch::Tensor& tensor) {
    return static_cast<const T*>(tensor.data_ptr());
}
