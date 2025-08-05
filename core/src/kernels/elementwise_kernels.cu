#include "kernel_fusion/kernels/kernels.hpp"

namespace kf {
namespace kernels {
namespace elementwise {

// ============================================================================
// Elementwise Add + Activation Kernel
// ============================================================================

template<typename T>
__global__ void add_activation_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ output,
    int64_t numel,
    kf_activation_t activation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        T sum = a[idx] + b[idx];
        output[idx] = apply_activation(sum, activation);
    }
}

// ============================================================================
// Elementwise Multiply + Activation Kernel
// ============================================================================

template<typename T>
__global__ void mul_activation_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ output,
    int64_t numel,
    kf_activation_t activation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        T product = a[idx] * b[idx];
        output[idx] = apply_activation(product, activation);
    }
}

// ============================================================================
// Bias + Activation Kernel
// ============================================================================

template<typename T>
__global__ void bias_activation_kernel(
    const T* __restrict__ input,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t numel,
    int64_t bias_size,
    kf_activation_t activation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int bias_idx = idx % bias_size;
        T biased = input[idx] + bias[bias_idx];
        output[idx] = apply_activation(biased, activation);
    }
}

// Explicit template instantiations
template __global__ void add_activation_kernel<float>(
    const float*, const float*, float*, int64_t, kf_activation_t);
template __global__ void add_activation_kernel<double>(
    const double*, const double*, double*, int64_t, kf_activation_t);

template __global__ void mul_activation_kernel<float>(
    const float*, const float*, float*, int64_t, kf_activation_t);
template __global__ void mul_activation_kernel<double>(
    const double*, const double*, double*, int64_t, kf_activation_t);

template __global__ void bias_activation_kernel<float>(
    const float*, const float*, float*, int64_t, int64_t, kf_activation_t);
template __global__ void bias_activation_kernel<double>(
    const double*, const double*, double*, int64_t, int64_t, kf_activation_t);

} // namespace elementwise
} // namespace kernels
} // namespace kf
