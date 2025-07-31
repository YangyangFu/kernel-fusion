#include "kernels.h"
#include <torch/torch.h>
#include <cmath>
#include <random>
#include <omp.h>  // OpenMP for parallelization

namespace cpu_kernels {

// Manual GELU implementation for CPU
template<typename T>
inline T cpu_gelu(T x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    T sqrt_2_over_pi = static_cast<T>(0.7978845608028654);
    T coeff = static_cast<T>(0.044715);
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + 
           std::tanh(sqrt_2_over_pi * (x + coeff * x * x * x)));
}

torch::Tensor elementwise_add_relu_cpu(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    
    auto output = torch::empty_like(a);
    
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "elementwise_add_relu_cpu", [&] {
        auto a_ptr = a.data_ptr<scalar_t>();
        auto b_ptr = b.data_ptr<scalar_t>();
        auto out_ptr = output.data_ptr<scalar_t>();
        
        int64_t numel = a.numel();
        
        // Parallel fused add + relu using OpenMP
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < numel; i++) {
            scalar_t sum = a_ptr[i] + b_ptr[i];
            out_ptr[i] = std::max(scalar_t(0), sum);  // ReLU
        }
    });
    
    return output;
}

torch::Tensor elementwise_mul_tanh_cpu(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    
    auto output = torch::empty_like(a);
    
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "elementwise_mul_tanh_cpu", [&] {
        auto a_ptr = a.data_ptr<scalar_t>();
        auto b_ptr = b.data_ptr<scalar_t>();
        auto out_ptr = output.data_ptr<scalar_t>();
        
        int64_t numel = a.numel();
        
        // Parallel fused mul + tanh using OpenMP
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < numel; i++) {
            scalar_t product = a_ptr[i] * b_ptr[i];
            out_ptr[i] = std::tanh(product);
        }
    });
    
    return output;
}

torch::Tensor fused_bias_gelu_cpu(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    int64_t numel = input.numel();
    int64_t bias_size = bias.numel();
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_bias_gelu_cpu", [&] {
        auto input_ptr = input.data_ptr<scalar_t>();
        auto bias_ptr = bias.data_ptr<scalar_t>();
        auto out_ptr = output.data_ptr<scalar_t>();
        
        // Parallel fused bias addition + GELU using OpenMP
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < numel; i++) {
            int64_t bias_idx = i % bias_size;
            scalar_t biased = input_ptr[i] + bias_ptr[bias_idx];
            out_ptr[i] = cpu_gelu(biased);
        }
    });
    
    return output;
}

torch::Tensor reduce_sum_squared_cpu(const torch::Tensor& input, int64_t dim, bool keepdim) {
    // For simplicity, handle the case where dim is the last dimension
    if (dim == -1 || dim == input.dim() - 1) {
        auto input_shape = input.sizes().vec();
        auto output_shape = input_shape;
        
        if (keepdim) {
            output_shape[dim] = 1;
        } else {
            output_shape.erase(output_shape.begin() + dim);
        }
        
        auto output = torch::zeros(output_shape, input.options());
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "reduce_sum_squared_cpu", [&] {
            auto input_ptr = input.data_ptr<scalar_t>();
            auto out_ptr = output.data_ptr<scalar_t>();
            
            int64_t outer_size = input.numel() / input.size(dim);
            int64_t inner_size = input.size(dim);
            
            // Parallel fused square + sum using OpenMP
            #pragma omp parallel for schedule(static) reduction(+:sum)
            for (int64_t i = 0; i < outer_size; i++) {
                scalar_t local_sum = 0;
                for (int64_t j = 0; j < inner_size; j++) {
                    scalar_t val = input_ptr[i * inner_size + j];
                    local_sum += val * val;  // Square and accumulate
                }
                out_ptr[i] = local_sum;
            }
        });
        
        return output;
    } else {
        // Fallback for other dimensions - this could be optimized further
        auto squared = torch::empty_like(input);
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "reduce_sum_squared_cpu_fallback", [&] {
            auto input_ptr = input.data_ptr<scalar_t>();
            auto squared_ptr = squared.data_ptr<scalar_t>();
            int64_t numel = input.numel();
            
            for (int64_t i = 0; i < numel; i++) {
                squared_ptr[i] = input_ptr[i] * input_ptr[i];
            }
        });
        
        return torch::sum(squared, dim, keepdim);
    }
}

torch::Tensor reduce_mean_abs_cpu(const torch::Tensor& input, int64_t dim, bool keepdim) {
    // For simplicity, handle the case where dim is the last dimension
    if (dim == -1 || dim == input.dim() - 1) {
        auto input_shape = input.sizes().vec();
        auto output_shape = input_shape;
        
        if (keepdim) {
            output_shape[dim] = 1;
        } else {
            output_shape.erase(output_shape.begin() + dim);
        }
        
        auto output = torch::zeros(output_shape, input.options());
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "reduce_mean_abs_cpu", [&] {
            auto input_ptr = input.data_ptr<scalar_t>();
            auto out_ptr = output.data_ptr<scalar_t>();
            
            int64_t outer_size = input.numel() / input.size(dim);
            int64_t inner_size = input.size(dim);
            
            // Fused abs + mean in single loop
            for (int64_t i = 0; i < outer_size; i++) {
                scalar_t sum = 0;
                for (int64_t j = 0; j < inner_size; j++) {
                    scalar_t val = input_ptr[i * inner_size + j];
                    sum += std::abs(val);  // Abs and accumulate
                }
                out_ptr[i] = sum / static_cast<scalar_t>(inner_size);  // Mean
            }
        });
        
        return output;
    } else {
        // Fallback for other dimensions - this could be optimized further
        auto abs_input = torch::empty_like(input);
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "reduce_mean_abs_cpu_fallback", [&] {
            auto input_ptr = input.data_ptr<scalar_t>();
            auto abs_ptr = abs_input.data_ptr<scalar_t>();
            int64_t numel = input.numel();
            
            for (int64_t i = 0; i < numel; i++) {
                abs_ptr[i] = std::abs(input_ptr[i]);
            }
        });
        
        return torch::mean(abs_input, dim, keepdim);
    }
}

torch::Tensor fused_layer_norm_relu_cpu(
    const torch::Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps
) {
    TORCH_CHECK(normalized_shape.size() == 1, "Only 1D layer norm supported");
    
    auto output = torch::empty_like(input);
    int64_t norm_size = normalized_shape[0];
    int64_t batch_size = input.numel() / norm_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_layer_norm_relu_cpu", [&] {
        auto input_ptr = input.data_ptr<scalar_t>();
        auto weight_ptr = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;
        auto bias_ptr = bias.defined() ? bias.data_ptr<scalar_t>() : nullptr;
        auto out_ptr = output.data_ptr<scalar_t>();
        
        scalar_t eps_val = static_cast<scalar_t>(eps);
        
        // Process each batch element
        for (int64_t batch = 0; batch < batch_size; batch++) {
            int64_t offset = batch * norm_size;
            
            // Compute mean
            scalar_t mean = 0;
            for (int64_t i = 0; i < norm_size; i++) {
                mean += input_ptr[offset + i];
            }
            mean /= norm_size;
            
            // Compute variance
            scalar_t var = 0;
            for (int64_t i = 0; i < norm_size; i++) {
                scalar_t diff = input_ptr[offset + i] - mean;
                var += diff * diff;
            }
            var /= norm_size;
            
            scalar_t inv_std = static_cast<scalar_t>(1.0) / std::sqrt(var + eps_val);
            
            // Apply layer norm and ReLU in single pass
            for (int64_t i = 0; i < norm_size; i++) {
                scalar_t normalized = (input_ptr[offset + i] - mean) * inv_std;
                
                if (weight_ptr) normalized *= weight_ptr[i];
                if (bias_ptr) normalized += bias_ptr[i];
                
                // Fused ReLU
                out_ptr[offset + i] = std::max(scalar_t(0), normalized);
            }
        }
    });
    
    return output;
}

torch::Tensor fused_gelu_dropout_cpu(
    const torch::Tensor& input,
    double p,
    bool training
) {
    auto output = torch::empty_like(input);
    
    if (!training || p == 0.0) {
        // No dropout, just GELU
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_gelu_cpu", [&] {
            auto input_ptr = input.data_ptr<scalar_t>();
            auto out_ptr = output.data_ptr<scalar_t>();
            int64_t numel = input.numel();
            
            for (int64_t i = 0; i < numel; i++) {
                out_ptr[i] = cpu_gelu(input_ptr[i]);
            }
        });
        
        return output;
    }
    
    // For training with dropout, we need random number generation
    // This is a simplified version - in practice you'd want proper random state
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_gelu_dropout_cpu", [&] {
        auto input_ptr = input.data_ptr<scalar_t>();
        auto out_ptr = output.data_ptr<scalar_t>();
        int64_t numel = input.numel();
        
        scalar_t dropout_prob = static_cast<scalar_t>(p);
        scalar_t scale = static_cast<scalar_t>(1.0 / (1.0 - p));
        
        // Simple random number generation (not cryptographically secure)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (int64_t i = 0; i < numel; i++) {
            scalar_t gelu_out = cpu_gelu(input_ptr[i]);
            
            // Apply dropout
            if (dis(gen) > dropout_prob) {
                out_ptr[i] = gelu_out * scale;
            } else {
                out_ptr[i] = scalar_t(0);
            }
        }
    });
    
    return output;
}

torch::Tensor fused_attention_score_cpu(
    const torch::Tensor& query,
    const torch::Tensor& key,
    double scale
) {
    TORCH_CHECK(query.dim() == 3 && key.dim() == 3, "Expected 3D tensors");
    TORCH_CHECK(query.size(0) == key.size(0), "Batch sizes must match");
    TORCH_CHECK(query.size(2) == key.size(2), "Hidden dimensions must match");
    
    int64_t batch_size = query.size(0);
    int64_t query_seq_len = query.size(1);
    int64_t key_seq_len = key.size(1);
    int64_t head_dim = query.size(2);
    
    auto output = torch::zeros({batch_size, query_seq_len, key_seq_len}, query.options());
    
    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "fused_attention_score_cpu", [&] {
        auto query_ptr = query.data_ptr<scalar_t>();
        auto key_ptr = key.data_ptr<scalar_t>();
        auto out_ptr = output.data_ptr<scalar_t>();
        
        scalar_t scale_val = static_cast<scalar_t>(scale);
        
        // Fused matrix multiplication and scaling
        for (int64_t b = 0; b < batch_size; b++) {
            for (int64_t q = 0; q < query_seq_len; q++) {
                for (int64_t k = 0; k < key_seq_len; k++) {
                    scalar_t dot_product = 0;
                    
                    // Compute dot product and scale in single loop
                    for (int64_t d = 0; d < head_dim; d++) {
                        int64_t query_idx = b * query_seq_len * head_dim + q * head_dim + d;
                        int64_t key_idx = b * key_seq_len * head_dim + k * head_dim + d;
                        dot_product += query_ptr[query_idx] * key_ptr[key_idx];
                    }
                    
                    int64_t out_idx = b * query_seq_len * key_seq_len + q * key_seq_len + k;
                    out_ptr[out_idx] = dot_product * scale_val;  // Fused scaling
                }
            }
        }
    });
    
    return output;
}

} // namespace cpu_kernels
