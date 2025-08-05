#pragma once

#include "../types.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

namespace kf {
namespace kernels {

// ============================================================================
// Kernel Launch Configuration
// ============================================================================

struct LaunchConfig {
    dim3 block_size;
    dim3 grid_size;
    int shared_memory;
    cudaStream_t stream;
    
    LaunchConfig(int64_t total_elements, cudaStream_t stream = nullptr);
};

// ============================================================================
// Activation Functions (Device)
// ============================================================================

template<typename T>
__device__ __forceinline__ T apply_activation(T x, kf_activation_t activation);

// Specific activation implementations
template<typename T>
__device__ __forceinline__ T relu(T x) { 
    return (x > T(0)) ? x : T(0); 
}

template<typename T>
__device__ __forceinline__ T gelu(T x) {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    T x3 = x * x * x;
    T inner = T(0.7978845608) * (x + T(0.044715) * x3);
    if constexpr (std::is_same_v<T, float>) {
        return T(0.5) * x * (T(1) + tanhf(inner));
    } else {
        return T(0.5) * x * (T(1) + tanh(inner));
    }
}

template<typename T>
__device__ __forceinline__ T silu(T x) { 
    if constexpr (std::is_same_v<T, float>) {
        return x / (T(1) + expf(-x)); 
    } else {
        return x / (T(1) + exp(-x)); 
    }
}

template<typename T>
__device__ __forceinline__ T sigmoid(T x) { 
    if constexpr (std::is_same_v<T, float>) {
        return T(1) / (T(1) + expf(-x)); 
    } else {
        return T(1) / (T(1) + exp(-x)); 
    }
}

// ============================================================================
// Elementwise Kernels
// ============================================================================

namespace elementwise {

template<typename T>
__global__ void add_activation_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ output,
    int64_t numel,
    kf_activation_t activation
);

template<typename T>
__global__ void mul_activation_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ output,
    int64_t numel,
    kf_activation_t activation
);

template<typename T>
__global__ void bias_activation_kernel(
    const T* __restrict__ input,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t numel,
    int64_t bias_size,
    kf_activation_t activation
);

} // namespace elementwise

// ============================================================================
// Linear Algebra Kernels
// ============================================================================

namespace linear {

template<typename T>
__global__ void fused_linear_activation_kernel(
    const T* __restrict__ input,    // [batch_size, in_features]
    const T* __restrict__ weight,   // [out_features, in_features]
    const T* __restrict__ bias,     // [out_features] (optional)
    T* __restrict__ output,         // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features,
    kf_activation_t activation
);

} // namespace linear

// ============================================================================
// Normalization Kernels
// ============================================================================

namespace normalization {

template<typename T>
__global__ void layer_norm_activation_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size,
    int normalized_size,
    float eps,
    kf_activation_t activation
);

template<typename T>
__global__ void batch_norm_activation_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* __restrict__ running_mean,
    const T* __restrict__ running_var,
    T* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size,
    float eps,
    kf_activation_t activation
);

} // namespace normalization

// ============================================================================
// Convolution Kernels
// ============================================================================

namespace convolution {

template<typename T>
__global__ void conv2d_bn_activation_kernel(
    const T* __restrict__ input,      // [N, C_in, H_in, W_in]
    const T* __restrict__ weight,     // [C_out, C_in, K_h, K_w]
    const T* __restrict__ bias,       // [C_out] (optional)
    const T* __restrict__ bn_weight,  // [C_out]
    const T* __restrict__ bn_bias,    // [C_out]
    const T* __restrict__ bn_mean,    // [C_out]
    const T* __restrict__ bn_var,     // [C_out]
    T* __restrict__ output,           // [N, C_out, H_out, W_out]
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    float bn_eps,
    kf_activation_t activation
);

} // namespace convolution

// ============================================================================
// Attention Kernels
// ============================================================================

namespace attention {

template<typename T>
__global__ void multi_head_attention_kernel(
    const T* __restrict__ query,      // [batch, seq_len, embed_dim]
    const T* __restrict__ key,        // [batch, seq_len, embed_dim]
    const T* __restrict__ value,      // [batch, seq_len, embed_dim]
    const T* __restrict__ attn_mask,  // [seq_len, seq_len] (optional)
    T* __restrict__ output,           // [batch, seq_len, embed_dim]
    int batch_size,
    int seq_len,
    int embed_dim,
    int num_heads,
    float scale
);

template<typename T>
__global__ void scaled_dot_product_attention_kernel(
    const T* __restrict__ query,      // [batch, num_heads, seq_len, head_dim]
    const T* __restrict__ key,        // [batch, num_heads, seq_len, head_dim]
    const T* __restrict__ value,      // [batch, num_heads, seq_len, head_dim]
    const T* __restrict__ attn_mask,  // [seq_len, seq_len] (optional)
    T* __restrict__ output,           // [batch, num_heads, seq_len, head_dim]
    T* __restrict__ attn_weights,     // [batch, num_heads, seq_len, seq_len] (optional)
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
);

} // namespace attention

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get optimal launch configuration for a kernel
 */
LaunchConfig get_launch_config(int64_t total_elements, cudaStream_t stream = nullptr);

/**
 * Get optimal launch configuration for 2D kernels
 */
LaunchConfig get_launch_config_2d(int64_t rows, int64_t cols, cudaStream_t stream = nullptr);

} // namespace kernels
} // namespace kf
