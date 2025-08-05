#include "kernel_fusion/kernels/kernels.hpp"

namespace kf {
namespace kernels {
namespace linear {

// ============================================================================
// Fused Linear + Activation Kernel
// ============================================================================

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
) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx < batch_size && out_idx < out_features) {
        T sum = T(0);
        
        // Compute dot product using warp-level reduction
        for (int in_idx = threadIdx.x; in_idx < in_features; in_idx += blockDim.x) {
            sum += input[batch_idx * in_features + in_idx] * 
                   weight[out_idx * in_features + in_idx];
        }
        
        // Warp-level reduction
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // First thread in warp writes result
        if (threadIdx.x == 0) {
            // Add bias if provided
            if (bias) {
                sum += bias[out_idx];
            }
            
            // Apply activation and write output
            output[batch_idx * out_features + out_idx] = apply_activation(sum, activation);
        }
    }
}

// ============================================================================
// Optimized GEMM + Activation (using shared memory)
// ============================================================================

template<typename T>
__global__ void optimized_linear_activation_kernel(
    const T* __restrict__ input,    // [batch_size, in_features]
    const T* __restrict__ weight,   // [out_features, in_features]
    const T* __restrict__ bias,     // [out_features] (optional)
    T* __restrict__ output,         // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features,
    kf_activation_t activation
) {
    // Shared memory for tiling
    __shared__ T shared_input[32][32];
    __shared__ T shared_weight[32][32];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    T sum = T(0);
    
    // Tile across the inner dimension
    for (int tile = 0; tile < (in_features + 31) / 32; ++tile) {
        // Load input tile
        int input_row = row;
        int input_col = tile * 32 + threadIdx.x;
        if (input_row < batch_size && input_col < in_features) {
            shared_input[threadIdx.y][threadIdx.x] = input[input_row * in_features + input_col];
        } else {
            shared_input[threadIdx.y][threadIdx.x] = T(0);
        }
        
        // Load weight tile (transposed)
        int weight_row = tile * 32 + threadIdx.y;
        int weight_col = col;
        if (weight_row < in_features && weight_col < out_features) {
            shared_weight[threadIdx.y][threadIdx.x] = weight[weight_col * in_features + weight_row];
        } else {
            shared_weight[threadIdx.y][threadIdx.x] = T(0);
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < 32; ++k) {
            sum += shared_input[threadIdx.y][k] * shared_weight[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < batch_size && col < out_features) {
        // Add bias if provided
        if (bias) {
            sum += bias[col];
        }
        
        // Apply activation and write
        output[row * out_features + col] = apply_activation(sum, activation);
    }
}

// Explicit template instantiations
template __global__ void fused_linear_activation_kernel<float>(
    const float*, const float*, const float*, float*, int, int, int, kf_activation_t);
template __global__ void fused_linear_activation_kernel<double>(
    const double*, const double*, const double*, double*, int, int, int, kf_activation_t);

template __global__ void optimized_linear_activation_kernel<float>(
    const float*, const float*, const float*, float*, int, int, int, kf_activation_t);
template __global__ void optimized_linear_activation_kernel<double>(
    const double*, const double*, const double*, double*, int, int, int, kf_activation_t);

} // namespace linear
} // namespace kernels
} // namespace kf
