#include "kernel_fusion/kernels/kernels.hpp"

namespace kf {
namespace kernels {
namespace normalization {

// ============================================================================
// Layer Normalization + Activation Kernel
// ============================================================================

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
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const T* x = input + batch_idx * normalized_size;
    T* y = output + batch_idx * normalized_size;
    
    // Shared memory for reduction
    __shared__ T shared_sum[1024];
    
    // Compute mean
    T sum = T(0);
    for (int i = tid; i < normalized_size; i += blockDim.x) {
        sum += x[i];
    }
    
    // Reduce mean across block
    shared_sum[tid] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    T mean = shared_sum[0] / normalized_size;
    __syncthreads();
    
    // Compute variance
    T var_sum = T(0);
    for (int i = tid; i < normalized_size; i += blockDim.x) {
        T diff = x[i] - mean;
        var_sum += diff * diff;
    }
    
    shared_sum[tid] = var_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    T variance = shared_sum[0] / normalized_size;
    T inv_std = T(1) / sqrtf(variance + eps);
    
    // Apply normalization and activation
    for (int i = tid; i < normalized_size; i += blockDim.x) {
        T normalized = (x[i] - mean) * inv_std;
        if (weight) normalized *= weight[i];
        if (bias) normalized += bias[i];
        y[i] = apply_activation(normalized, activation);
    }
}

// ============================================================================
// Batch Normalization + Activation Kernel
// ============================================================================

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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        // Calculate which channel this element belongs to
        int spatial_idx = idx % spatial_size;
        int channel_idx = (idx / spatial_size) % channels;
        int batch_idx = idx / (channels * spatial_size);
        
        T x = input[idx];
        T mean = running_mean[channel_idx];
        T var = running_var[channel_idx];
        T scale = weight ? weight[channel_idx] : T(1);
        T shift = bias ? bias[channel_idx] : T(0);
        
        // Normalize
        T normalized = (x - mean) / sqrtf(var + eps);
        T scaled = normalized * scale + shift;
        
        // Apply activation
        output[idx] = apply_activation(scaled, activation);
    }
}

// Explicit template instantiations
template __global__ void layer_norm_activation_kernel<float>(
    const float*, const float*, const float*, float*, int, int, float, kf_activation_t);
template __global__ void layer_norm_activation_kernel<double>(
    const double*, const double*, const double*, double*, int, int, float, kf_activation_t);

template __global__ void batch_norm_activation_kernel<float>(
    const float*, const float*, const float*, const float*, const float*, float*, int, int, int, float, kf_activation_t);
template __global__ void batch_norm_activation_kernel<double>(
    const double*, const double*, const double*, const double*, const double*, double*, int, int, int, float, kf_activation_t);

} // namespace normalization
} // namespace kernels
} // namespace kf
