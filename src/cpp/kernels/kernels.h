#pragma once

#include <torch/extension.h>
#include <vector>

// Forward declarations for CUDA kernels
#ifdef USE_CUDA
namespace cuda_kernels {
    // Elementwise operations
    torch::Tensor elementwise_add_relu_cuda(const torch::Tensor& a, const torch::Tensor& b);
    torch::Tensor elementwise_mul_tanh_cuda(const torch::Tensor& a, const torch::Tensor& b);
    torch::Tensor fused_bias_gelu_cuda(const torch::Tensor& input, const torch::Tensor& bias);
    
    // Reduction operations
    torch::Tensor reduce_sum_squared_cuda(const torch::Tensor& input, int64_t dim, bool keepdim);
    torch::Tensor reduce_mean_abs_cuda(const torch::Tensor& input, int64_t dim, bool keepdim);
    
    // Fusion operations
    torch::Tensor fused_layer_norm_relu_cuda(
        const torch::Tensor& input,
        const std::vector<int64_t>& normalized_shape,
        const torch::Tensor& weight,
        const torch::Tensor& bias,
        double eps
    );
    
    torch::Tensor fused_gelu_dropout_cuda(
        const torch::Tensor& input,
        double p,
        bool training
    );
    
    torch::Tensor fused_attention_score_cuda(
        const torch::Tensor& query,
        const torch::Tensor& key,
        double scale
    );
}
#endif

// CPU fallback implementations
namespace cpu_kernels {
    torch::Tensor elementwise_add_relu_cpu(const torch::Tensor& a, const torch::Tensor& b);
    torch::Tensor elementwise_mul_tanh_cpu(const torch::Tensor& a, const torch::Tensor& b);
    torch::Tensor fused_bias_gelu_cpu(const torch::Tensor& input, const torch::Tensor& bias);
    
    torch::Tensor reduce_sum_squared_cpu(const torch::Tensor& input, int64_t dim, bool keepdim);
    torch::Tensor reduce_mean_abs_cpu(const torch::Tensor& input, int64_t dim, bool keepdim);
    
    torch::Tensor fused_layer_norm_relu_cpu(
        const torch::Tensor& input,
        const std::vector<int64_t>& normalized_shape,
        const torch::Tensor& weight,
        const torch::Tensor& bias,
        double eps
    );
    
    torch::Tensor fused_gelu_dropout_cpu(
        const torch::Tensor& input,
        double p,
        bool training
    );
    
    torch::Tensor fused_attention_score_cpu(
        const torch::Tensor& query,
        const torch::Tensor& key,
        double scale
    );
}
