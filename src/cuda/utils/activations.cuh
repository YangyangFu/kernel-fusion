#pragma once

#include "type_traits.cuh"
#include <cuda_runtime.h>

namespace cuda_activations {

using namespace cuda_type_utils;

// ReLU activation function
template<typename T>
__device__ __forceinline__ T relu(T x) {
    using CudaT = cuda_type_t<T>;
    
    if constexpr (is_single_precision_v<T>) {
        CudaT cuda_x = to_cuda_type(x);
        CudaT result = fmaxf(0.0f, cuda_x);
        return from_cuda_type<CudaT, T>(result);
    } else if constexpr (is_double_precision_v<T>) {
        CudaT cuda_x = to_cuda_type(x);
        CudaT result = fmax(0.0, cuda_x);
        return from_cuda_type<CudaT, T>(result);
    } else if constexpr (is_half_precision_v<T>) {
        CudaT cuda_x = to_cuda_type(x);
        CudaT result = __hmax(cuda_x, __float2half(0.0f));
        return from_cuda_type<CudaT, T>(result);
    } else {
        static_assert(std::is_same_v<T, void>, "Unsupported type for relu");
    }
}

// GELU activation function
template<typename T>
__device__ __forceinline__ T gelu(T x) {
    using CudaT = cuda_type_t<T>;
    
    if constexpr (is_single_precision_v<T>) {
        CudaT cuda_x = to_cuda_type(x);
        CudaT result = 0.5f * cuda_x * (1.0f + tanhf(0.7978845608028654f * (cuda_x + 0.044715f * cuda_x * cuda_x * cuda_x)));
        return from_cuda_type<CudaT, T>(result);
    } else if constexpr (is_double_precision_v<T>) {
        CudaT cuda_x = to_cuda_type(x);
        CudaT result = 0.5 * cuda_x * (1.0 + tanh(0.7978845608028654 * (cuda_x + 0.044715 * cuda_x * cuda_x * cuda_x)));
        return from_cuda_type<CudaT, T>(result);
    } else if constexpr (is_half_precision_v<T>) {
        float x_float = __half2float(to_cuda_type(x));
        float result = 0.5f * x_float * (1.0f + tanhf(0.7978845608028654f * (x_float + 0.044715f * x_float * x_float * x_float)));
        return from_cuda_type<__half, T>(__float2half(result));
    } else {
        static_assert(std::is_same_v<T, void>, "Unsupported type for gelu");
    }
}

// Tanh activation function
template<typename T>
__device__ __forceinline__ T tanh_activation(T x) {
    using CudaT = cuda_type_t<T>;
    
    if constexpr (is_single_precision_v<T>) {
        CudaT cuda_x = to_cuda_type(x);
        CudaT result = tanhf(cuda_x);
        return from_cuda_type<CudaT, T>(result);
    } else if constexpr (is_double_precision_v<T>) {
        CudaT cuda_x = to_cuda_type(x);
        CudaT result = tanh(cuda_x);
        return from_cuda_type<CudaT, T>(result);
    } else if constexpr (is_half_precision_v<T>) {
        float x_float = __half2float(to_cuda_type(x));
        float result = tanhf(x_float);
        return from_cuda_type<__half, T>(__float2half(result));
    } else {
        static_assert(std::is_same_v<T, void>, "Unsupported type for tanh");
    }
}

} // namespace cuda_activations

// Backward compatibility aliases
using cuda_activations::relu;
using cuda_activations::gelu;
using cuda_activations::tanh_activation;

// Legacy function names for existing code
template<typename T>
__device__ __forceinline__ T cuda_relu(T x) { return relu(x); }

template<typename T>
__device__ __forceinline__ T cuda_gelu(T x) { return gelu(x); }

template<typename T>
__device__ __forceinline__ T cuda_tanh(T x) { return tanh_activation(x); }
