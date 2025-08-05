#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include "../utils/cuda_utils.cuh"

// Layer Norm + ReLU fused kernel
template<typename T>
__global__ void fused_layer_norm_relu_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t batch_size,
    int64_t seq_len,
    int64_t hidden_dim,
    T eps
) {
    int64_t batch_idx = blockIdx.x;
    int64_t seq_idx = blockIdx.y;
    int64_t tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    // Shared memory for reduction
    extern __shared__ T shared_mem[];
    T* shared_sum = shared_mem;
    T* shared_var = shared_mem + blockDim.x;
    
    int64_t input_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;
    int64_t output_offset = input_offset;
    
    // Compute mean
    T thread_sum = 0;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        thread_sum += input[input_offset + i];
    }
    shared_sum[tid] = thread_sum;
    __syncthreads();
    
    // Reduce to get mean
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    T mean = shared_sum[0] / hidden_dim;
    __syncthreads();
    
    // Compute variance
    T thread_var = 0;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        T diff = input[input_offset + i] - mean;
        thread_var += diff * diff;
    }
    shared_var[tid] = thread_var;
    __syncthreads();
    
    // Reduce to get variance
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_var[tid] += shared_var[tid + stride];
        }
        __syncthreads();
    }
    T variance = shared_var[0] / hidden_dim;
    T inv_std = rsqrtf(variance + eps);
    __syncthreads();
    
    // Apply layer norm and ReLU in one pass
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        T normalized = (input[input_offset + i] - mean) * inv_std;
        if (weight) normalized *= weight[i];
        if (bias) normalized += bias[i];
        // Fused ReLU
        output[output_offset + i] = cuda_relu(normalized);
    }
}

// GELU + Dropout fused kernel
template<typename T>
__global__ void fused_gelu_dropout_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    curandState* __restrict__ rand_states,
    int64_t numel,
    T dropout_prob,
    T dropout_scale,
    bool training
) {
    CUDA_KERNEL_LOOP(idx, numel) {
        T x = input[idx];
        
        // Apply GELU
        T gelu_out = cuda_gelu(x);
        
        if (training && dropout_prob > 0) {
            // Apply dropout using curand
            curandState local_state = rand_states[idx % 1024]; // Reuse states
            T random_val = curand_uniform(&local_state);
            rand_states[idx % 1024] = local_state;
            
            output[idx] = (random_val > dropout_prob) ? gelu_out * dropout_scale : T(0);
        } else {
            output[idx] = gelu_out;
        }
    }
}

// Optimized attention score kernel using shared memory
template<typename T>
__global__ void fused_attention_score_kernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    T* __restrict__ output,
    int64_t batch_size,
    int64_t seq_len,
    int64_t head_dim,
    T scale
) {
    int64_t batch_idx = blockIdx.x;
    int64_t query_idx = blockIdx.y;
    int64_t key_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || query_idx >= seq_len) return;
    
    extern __shared__ T shared_key[];
    
    // Load key vector into shared memory
    if (key_idx < seq_len && threadIdx.y == 0) {
        for (int d = 0; d < head_dim; d++) {
            int64_t key_offset = batch_idx * seq_len * head_dim + key_idx * head_dim + d;
            shared_key[key_idx * head_dim + d] = key[key_offset];
        }
    }
    __syncthreads();
    
    if (key_idx < seq_len) {
        // Compute dot product between query and key
        T dot_product = 0;
        int64_t query_offset = batch_idx * seq_len * head_dim + query_idx * head_dim;
        
        for (int d = 0; d < head_dim; d++) {
            dot_product += query[query_offset + d] * shared_key[key_idx * head_dim + d];
        }
        
        // Apply scale and store result
        int64_t output_offset = batch_idx * seq_len * seq_len + query_idx * seq_len + key_idx;
        output[output_offset] = dot_product * scale;
    }
}

namespace cuda_kernels {

torch::Tensor fused_layer_norm_relu_cuda(
    const torch::Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(normalized_shape.size() == 1, "Only 1D layer norm supported for now");
    
    auto output = torch::empty_like(input);
    int64_t hidden_dim = normalized_shape[0];
    int64_t total_size = input.numel();
    int64_t batch_size = total_size / hidden_dim;
    int64_t seq_len = batch_size / input.size(0);
    
    // Launch configuration
    dim3 block_size(256);
    dim3 grid_size(input.size(0), seq_len);
    size_t shared_mem_size = 2 * block_size.x * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_layer_norm_relu_cuda", [&] {
        fused_layer_norm_relu_kernel<scalar_t><<<grid_size, block_size, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.defined() ? weight.data_ptr<scalar_t>() : nullptr,
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            input.size(0),
            seq_len,
            hidden_dim,
            static_cast<scalar_t>(eps)
        );
    });
    
    CUDA_KERNEL_CHECK();
    return output;
}

torch::Tensor fused_gelu_dropout_cuda(
    const torch::Tensor& input,
    double p,
    bool training
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    
    auto output = torch::empty_like(input);
    int64_t numel = input.numel();
    
    if (numel == 0) return output;
    
    LaunchConfig config(numel);
    
    // Initialize random states (simplified - in practice you'd want persistent states)
    static curandState* rand_states = nullptr;
    if (!rand_states) {
        CUDA_CHECK(cudaMalloc(&rand_states, 1024 * sizeof(curandState)));
        // Initialize states (simplified)
        // In practice, you'd want a proper initialization kernel
    }
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_gelu_dropout_cuda", [&] {
        scalar_t dropout_scale = training ? (1.0 / (1.0 - p)) : 1.0;
        
        fused_gelu_dropout_kernel<scalar_t><<<config.grid_size, config.block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            rand_states,
            numel,
            static_cast<scalar_t>(p),
            dropout_scale,
            training
        );
    });
    
    CUDA_KERNEL_CHECK();
    return output;
}

torch::Tensor fused_attention_score_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    double scale
) {
    TORCH_CHECK(query.is_cuda() && key.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(query.dim() == 3 && key.dim() == 3, "Expected 3D tensors [batch, seq_len, head_dim]");
    
    int64_t batch_size = query.size(0);
    int64_t seq_len = query.size(1);
    int64_t head_dim = query.size(2);
    
    auto output = torch::empty({batch_size, seq_len, seq_len}, query.options());
    
    // Launch configuration
    dim3 block_size(seq_len, 1);
    dim3 grid_size(batch_size, seq_len);
    size_t shared_mem_size = seq_len * head_dim * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "fused_attention_score_cuda", [&] {
        fused_attention_score_kernel<scalar_t><<<grid_size, block_size, shared_mem_size>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            seq_len,
            head_dim,
            static_cast<scalar_t>(scale)
        );
    });
    
    CUDA_KERNEL_CHECK();
    return output;
}

} // namespace cuda_kernels
