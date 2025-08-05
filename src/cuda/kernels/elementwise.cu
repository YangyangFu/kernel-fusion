#include <torch/extension.h>
#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

// Elementwise add + relu kernel
template<typename T>
__global__ void elementwise_add_relu_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ output,
    int64_t numel
) {
    CUDA_KERNEL_LOOP(idx, numel) {
        T sum = a[idx] + b[idx];
        output[idx] = cuda_relu(sum);
    }
}

// Elementwise mul + tanh kernel  
template<typename T>
__global__ void elementwise_mul_tanh_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ output,
    int64_t numel
) {
    CUDA_KERNEL_LOOP(idx, numel) {
        T product = a[idx] * b[idx];
        output[idx] = cuda_tanh(product);
    }
}

// Bias + GELU kernel
template<typename T>
__global__ void fused_bias_gelu_kernel(
    const T* __restrict__ input,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t numel,
    int64_t bias_size
) {
    CUDA_KERNEL_LOOP(idx, numel) {
        int bias_idx = idx % bias_size;
        T biased = input[idx] + bias[bias_idx];
        output[idx] = cuda_gelu(biased);
    }
}

namespace cuda_kernels {

torch::Tensor elementwise_add_relu_cuda(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    
    auto output = torch::empty_like(a);
    int64_t numel = a.numel();
    
    if (numel == 0) return output;
    
    LaunchConfig config(numel);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "elementwise_add_relu_cuda", [&] {
        elementwise_add_relu_kernel<scalar_t><<<config.grid_size, config.block_size, 0, config.stream>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    });
    
    CUDA_KERNEL_CHECK();
    return output;
}

torch::Tensor elementwise_mul_tanh_cuda(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    
    auto output = torch::empty_like(a);
    int64_t numel = a.numel();
    
    if (numel == 0) return output;
    
    LaunchConfig config(numel);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "elementwise_mul_tanh_cuda", [&] {
        elementwise_mul_tanh_kernel<scalar_t><<<config.grid_size, config.block_size, 0, config.stream>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    });
    
    CUDA_KERNEL_CHECK();
    return output;
}

torch::Tensor fused_bias_gelu_cuda(const torch::Tensor& input, const torch::Tensor& bias) {
    TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "Inputs must be CUDA tensors");
    
    auto output = torch::empty_like(input);
    int64_t numel = input.numel();
    int64_t bias_size = bias.numel();
    
    if (numel == 0) return output;
    
    LaunchConfig config(numel);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_bias_gelu_cuda", [&] {
        fused_bias_gelu_kernel<scalar_t><<<config.grid_size, config.block_size, 0, config.stream>>>(
            input.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel,
            bias_size
        );
    });
    
    CUDA_KERNEL_CHECK();
    return output;
}

} // namespace cuda_kernels
