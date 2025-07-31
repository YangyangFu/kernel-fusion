"""
CUDA-based LayerNorm implementation
Example demonstrating the complete CUDA algorithm development process
"""

import torch
import math
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Forward pass kernel
__global__ void layernorm_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float eps
) {
    // Each block processes one sequence position
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    // Calculate base offset for this sequence position
    int base_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size;
    
    // Shared memory for reductions
    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float shared_sum_sq[BLOCK_SIZE];
    
    // Thread-local accumulation
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Each thread processes multiple elements
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = input[base_offset + i];
        sum += val;
        sum_sq += val * val;
    }
    
    // Store in shared memory
    shared_sum[threadIdx.x] = sum;
    shared_sum_sq[threadIdx.x] = sum_sq;
    __syncthreads();
    
    // Reduce within block using CUB (or manual reduction)
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    sum = BlockReduce(temp_storage).Sum(sum);
    __syncthreads();
    sum_sq = BlockReduce(temp_storage).Sum(sum_sq);
    
    // Calculate mean and variance
    if (threadIdx.x == 0) {
        float local_mean = sum / hidden_size;
        float local_var = (sum_sq / hidden_size) - (local_mean * local_mean);
        float local_rstd = rsqrtf(local_var + eps);
        
        // Store statistics
        int stats_offset = batch_idx * seq_len + seq_idx;
        mean[stats_offset] = local_mean;
        rstd[stats_offset] = local_rstd;
    }
    __syncthreads();
    
    // Retrieve mean and rstd for normalization
    int stats_offset = batch_idx * seq_len + seq_idx;
    float local_mean = mean[stats_offset];
    float local_rstd = rstd[stats_offset];
    
    // Normalize and apply scale/shift
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = input[base_offset + i];
        float normalized = (val - local_mean) * local_rstd;
        
        // Apply weight and bias
        if (weight != nullptr) {
            normalized *= weight[i];
        }
        if (bias != nullptr) {
            normalized += bias[i];
        }
        
        output[base_offset + i] = normalized;
    }
}

// Backward pass kernel
__global__ void layernorm_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    const int batch_size,
    const int seq_len,
    const int hidden_size
) {
    // Implementation of backward pass
    // Simplified for brevity - full implementation would be more complex
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_size;
    
    if (idx < total_elements) {
        // Calculate gradients (simplified version)
        int batch_idx = idx / (seq_len * hidden_size);
        int seq_idx = (idx % (seq_len * hidden_size)) / hidden_size;
        int hidden_idx = idx % hidden_size;
        
        int stats_offset = batch_idx * seq_len + seq_idx;
        float local_mean = mean[stats_offset];
        float local_rstd = rstd[stats_offset];
        
        float input_val = input[idx];
        float grad_out = grad_output[idx];
        float normalized = (input_val - local_mean) * local_rstd;
        
        // Simplified gradient computation
        grad_input[idx] = grad_out * weight[hidden_idx] * local_rstd;
        
        // Atomic operations for weight and bias gradients
        atomicAdd(&grad_weight[hidden_idx], grad_out * normalized);
        atomicAdd(&grad_bias[hidden_idx], grad_out);
    }
}

std::vector<torch::Tensor> layernorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto hidden_size = input.size(2);
    
    // Allocate output tensors
    auto output = torch::zeros_like(input);
    auto mean = torch::zeros({batch_size, seq_len}, input.options());
    auto rstd = torch::zeros({batch_size, seq_len}, input.options());
    
    // Launch configuration
    dim3 grid(batch_size, seq_len);
    dim3 block(BLOCK_SIZE);
    
    layernorm_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        batch_size,
        seq_len,
        hidden_size,
        eps
    );
    
    cudaDeviceSynchronize();
    
    return {output, mean, rstd};
}

std::vector<torch::Tensor> layernorm_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor mean,
    torch::Tensor rstd
) {
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto hidden_size = input.size(2);
    
    // Allocate gradient tensors
    auto grad_input = torch::zeros_like(input);
    auto grad_weight = torch::zeros_like(weight);
    auto grad_bias = torch::zeros_like(weight);
    
    // Launch configuration
    int total_elements = batch_size * seq_len * hidden_size;
    int grid_size = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    layernorm_backward_kernel<<<grid_size, BLOCK_SIZE>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        grad_bias.data_ptr<float>(),
        batch_size,
        seq_len,
        hidden_size
    );
    
    cudaDeviceSynchronize();
    
    return {grad_input, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward_cuda, "LayerNorm forward pass");
    m.def("backward", &layernorm_backward_cuda, "LayerNorm backward pass");
}
"""

cpp_source = """
std::vector<torch::Tensor> layernorm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
);

std::vector<torch::Tensor> layernorm_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor mean,
    torch::Tensor rstd
);

std::vector<torch::Tensor> layernorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    return layernorm_forward_cuda(input, weight, bias, eps);
}

std::vector<torch::Tensor> layernorm_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor mean,
    torch::Tensor rstd
) {
    return layernorm_backward_cuda(grad_output, input, weight, mean, rstd);
}
"""


class CUDALayerNorm:
    """CUDA-accelerated LayerNorm implementation"""
    
    def __init__(self):
        self.module = None
        self._compile_cuda_kernel()
    
    def _compile_cuda_kernel(self):
        """Compile the CUDA kernel using PyTorch's JIT compilation"""
        try:
            self.module = load_inline(
                name='layernorm_cuda',
                cpp_sources=[cpp_source],
                cuda_sources=[cuda_source],
                functions=['layernorm_forward', 'layernorm_backward'],
                verbose=True,
                extra_cuda_cflags=['-O3', '--use_fast_math', '-lcub'],
                extra_cflags=['-O3']
            )
            print("LayerNorm CUDA kernel compiled successfully!")
        except Exception as e:
            print(f"Failed to compile LayerNorm CUDA kernel: {e}")
            print("Falling back to PyTorch implementation")
            self.module = None
    
    def forward(self, x, weight=None, bias=None, eps=1e-5):
        """
        Forward pass of LayerNorm
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            weight: Optional scale parameter [hidden_size]
            bias: Optional shift parameter [hidden_size]
            eps: Small constant for numerical stability
            
        Returns:
            Tuple of (output, mean, rstd) for backward pass
        """
        if self.module is not None and x.is_cuda:
            # Use compiled CUDA kernel
            if weight is None:
                weight = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
            if bias is None:
                bias = torch.zeros(x.size(-1), device=x.device, dtype=x.dtype)
            
            return self.module.layernorm_forward(
                x.contiguous(), weight.contiguous(), bias.contiguous(), eps
            )
        else:
            # Fallback to PyTorch implementation
            return self._pytorch_layernorm(x, weight, bias, eps)
    
    def backward(self, grad_output, x, weight, mean, rstd):
        """Backward pass of LayerNorm"""
        if self.module is not None and x.is_cuda:
            return self.module.layernorm_backward(
                grad_output.contiguous(),
                x.contiguous(), 
                weight.contiguous(),
                mean.contiguous(),
                rstd.contiguous()
            )
        else:
            # Fallback to CPU implementation
            from .cpu_layernorm import cpu_layernorm_backward
            return cpu_layernorm_backward(grad_output, x, weight, mean, rstd)
    
    def _pytorch_layernorm(self, x, weight=None, bias=None, eps=1e-5):
        """PyTorch fallback implementation"""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + eps)
        
        x_norm = (x - mean) * rstd
        
        if weight is not None:
            x_norm = x_norm * weight
        if bias is not None:
            x_norm = x_norm + bias
        
        return x_norm, mean.squeeze(-1), rstd.squeeze(-1)


def layernorm(x, weight=None, bias=None, eps=1e-5):
    """
    High-level interface for LayerNorm
    
    Args:
        x: Input tensor [batch_size, seq_len, hidden_size]
        weight: Optional scale parameter [hidden_size]
        bias: Optional shift parameter [hidden_size]
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor with same shape as input
    """
    layernorm_op = CUDALayerNorm()
    output, _, _ = layernorm_op.forward(x, weight, bias, eps)
    return output
