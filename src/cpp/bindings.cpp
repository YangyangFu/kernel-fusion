#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kernels/kernels.h"

// Dispatch functions that choose between CUDA and CPU implementations
torch::Tensor elementwise_add_relu(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.device() == b.device(), "Input tensors must be on the same device");
    
#ifdef USE_CUDA
    if (a.device().is_cuda()) {
        return cuda_kernels::elementwise_add_relu_cuda(a, b);
    }
#endif
    return cpu_kernels::elementwise_add_relu_cpu(a, b);
}

torch::Tensor elementwise_mul_tanh(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.device() == b.device(), "Input tensors must be on the same device");
    
#ifdef USE_CUDA
    if (a.device().is_cuda()) {
        return cuda_kernels::elementwise_mul_tanh_cuda(a, b);
    }
#endif
    return cpu_kernels::elementwise_mul_tanh_cpu(a, b);
}

torch::Tensor fused_bias_gelu(const torch::Tensor& input, const torch::Tensor& bias) {
    TORCH_CHECK(input.device() == bias.device(), "Input tensors must be on the same device");
    
#ifdef USE_CUDA
    if (input.device().is_cuda()) {
        return cuda_kernels::fused_bias_gelu_cuda(input, bias);
    }
#endif
    return cpu_kernels::fused_bias_gelu_cpu(input, bias);
}

torch::Tensor reduce_sum_squared(const torch::Tensor& input, int64_t dim, bool keepdim) {
#ifdef USE_CUDA
    if (input.device().is_cuda()) {
        return cuda_kernels::reduce_sum_squared_cuda(input, dim, keepdim);
    }
#endif
    return cpu_kernels::reduce_sum_squared_cpu(input, dim, keepdim);
}

torch::Tensor reduce_mean_abs(const torch::Tensor& input, int64_t dim, bool keepdim) {
#ifdef USE_CUDA
    if (input.device().is_cuda()) {
        return cuda_kernels::reduce_mean_abs_cuda(input, dim, keepdim);
    }
#endif
    return cpu_kernels::reduce_mean_abs_cpu(input, dim, keepdim);
}

torch::Tensor fused_layer_norm_relu(
    const torch::Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps
) {
#ifdef USE_CUDA
    if (input.device().is_cuda()) {
        return cuda_kernels::fused_layer_norm_relu_cuda(input, normalized_shape, weight, bias, eps);
    }
#endif
    return cpu_kernels::fused_layer_norm_relu_cpu(input, normalized_shape, weight, bias, eps);
}

torch::Tensor fused_gelu_dropout(
    const torch::Tensor& input,
    double p,
    bool training
) {
#ifdef USE_CUDA
    if (input.device().is_cuda()) {
        return cuda_kernels::fused_gelu_dropout_cuda(input, p, training);
    }
#endif
    return cpu_kernels::fused_gelu_dropout_cpu(input, p, training);
}

torch::Tensor fused_attention_score(
    const torch::Tensor& query,
    const torch::Tensor& key,
    double scale
) {
    TORCH_CHECK(query.device() == key.device(), "Query and key must be on the same device");
    
#ifdef USE_CUDA
    if (query.device().is_cuda()) {
        return cuda_kernels::fused_attention_score_cuda(query, key, scale);
    }
#endif
    return cpu_kernels::fused_attention_score_cpu(query, key, scale);
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Kernel Fusion Library - Optimized CUDA kernels for PyTorch";
    
    // Elementwise operations
    m.def("elementwise_add_relu", &elementwise_add_relu, "Fused elementwise addition and ReLU");
    m.def("elementwise_mul_tanh", &elementwise_mul_tanh, "Fused elementwise multiplication and Tanh");
    m.def("fused_bias_gelu", &fused_bias_gelu, "Fused bias addition and GELU");
    
    // Reduction operations
    m.def("reduce_sum_squared", &reduce_sum_squared, "Fused square and sum reduction");
    m.def("reduce_mean_abs", &reduce_mean_abs, "Fused absolute value and mean reduction");
    
    // Fusion operations
    m.def("fused_layer_norm_relu", &fused_layer_norm_relu, "Fused layer normalization and ReLU");
    m.def("fused_gelu_dropout", &fused_gelu_dropout, "Fused GELU and dropout");
    m.def("fused_attention_score", &fused_attention_score, "Fused attention score computation");
}
