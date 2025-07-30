"""
CUDA-based fused attention kernel implementation
"""

import torch
import math
import numpy as np
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32

__global__ void fused_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K, 
    const float* __restrict__ V,
    float* __restrict__ O,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    // Thread and block indices
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_idx = threadIdx.x + blockIdx.z * blockDim.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) {
        return;
    }
    
    // Calculate base offsets for this batch and head
    int qkv_offset = batch_idx * num_heads * seq_len * head_dim + 
                     head_idx * seq_len * head_dim;
    int out_offset = qkv_offset;
    
    // Shared memory for storing attention scores
    extern __shared__ float shared_mem[];
    float* att_scores = shared_mem;
    
    // Initialize output accumulator
    float output_acc[64]; // Assuming head_dim <= 64
    for (int d = 0; d < head_dim; d++) {
        output_acc[d] = 0.0f;
    }
    
    // Load query vector for this sequence position
    float query[64];
    for (int d = 0; d < head_dim; d++) {
        query[d] = Q[qkv_offset + seq_idx * head_dim + d];
    }
    
    // Compute attention scores for all key positions
    float max_score = -INFINITY;
    for (int k_pos = 0; k_pos < seq_len; k_pos++) {
        float score = 0.0f;
        
        // Dot product between query and key
        for (int d = 0; d < head_dim; d++) {
            float key_val = K[qkv_offset + k_pos * head_dim + d];
            score += query[d] * key_val;
        }
        
        score *= scale;
        att_scores[k_pos] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Compute softmax (subtract max for numerical stability)
    float sum_exp = 0.0f;
    for (int k_pos = 0; k_pos < seq_len; k_pos++) {
        att_scores[k_pos] = expf(att_scores[k_pos] - max_score);
        sum_exp += att_scores[k_pos];
    }
    
    // Normalize attention weights
    for (int k_pos = 0; k_pos < seq_len; k_pos++) {
        att_scores[k_pos] /= sum_exp;
    }
    
    // Apply attention weights to values
    for (int v_pos = 0; v_pos < seq_len; v_pos++) {
        float weight = att_scores[v_pos];
        for (int d = 0; d < head_dim; d++) {
            float value_val = V[qkv_offset + v_pos * head_dim + d];
            output_acc[d] += weight * value_val;
        }
    }
    
    // Store output
    for (int d = 0; d < head_dim; d++) {
        O[out_offset + seq_idx * head_dim + d] = output_acc[d];
    }
}

torch::Tensor fused_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    auto batch_size = Q.size(0);
    auto num_heads = Q.size(1);
    auto seq_len = Q.size(2);
    auto head_dim = Q.size(3);
    
    auto O = torch::zeros_like(Q);
    
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    // Launch configuration
    dim3 grid(batch_size, num_heads, (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    
    size_t shared_mem_size = seq_len * sizeof(float);
    
    fused_attention_kernel<<<grid, block, shared_mem_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale
    );
    
    cudaDeviceSynchronize();
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_attention", &fused_attention_cuda, "Fused Attention CUDA");
}
"""

cpp_source = """
torch::Tensor fused_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K, 
    torch::Tensor V
);

torch::Tensor fused_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    return fused_attention_cuda(Q, K, V);
}
"""


class CUDAFusedAttention:
    """CUDA-based fused attention implementation"""
    
    def __init__(self):
        self.module = None
        self._compile_cuda_kernel()
    
    def _compile_cuda_kernel(self):
        """Compile the CUDA kernel using PyTorch's JIT compilation"""
        try:
            self.module = load_inline(
                name='fused_attention_cuda',
                cpp_sources=[cpp_source],
                cuda_sources=[cuda_source],
                functions=['fused_attention'],
                verbose=True,
                extra_cuda_cflags=['-O3'],
                extra_cflags=['-O3']
            )
            print("CUDA kernel compiled successfully!")
        except Exception as e:
            print(f"Failed to compile CUDA kernel: {e}")
            print("Falling back to PyTorch implementation")
            self.module = None
    
    def forward(self, q, k, v, mask=None):
        """
        Forward pass of fused attention
        
        Args:
            q: Query tensor [batch, heads, seq_len, d_head]
            k: Key tensor [batch, heads, seq_len, d_head]  
            v: Value tensor [batch, heads, seq_len, d_head]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, heads, seq_len, d_head]
        """
        if self.module is not None and q.is_cuda:
            # Use compiled CUDA kernel
            return self.module.fused_attention(q.contiguous(), k.contiguous(), v.contiguous())
        else:
            # Fallback to PyTorch implementation
            return self._pytorch_attention(q, k, v, mask)
    
    def _pytorch_attention(self, q, k, v, mask=None):
        """PyTorch fallback implementation"""
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output


def fused_attention(q, k, v, mask=None):
    """
    High-level interface for fused attention
    
    Args:
        q: Query tensor [batch, heads, seq_len, d_head]
        k: Key tensor [batch, heads, seq_len, d_head]
        v: Value tensor [batch, heads, seq_len, d_head]
        mask: Optional attention mask
        
    Returns:
        Output tensor [batch, heads, seq_len, d_head]
    """
    attention_op = CUDAFusedAttention()
    return attention_op.forward(q, k, v, mask)
