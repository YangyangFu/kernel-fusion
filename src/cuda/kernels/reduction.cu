#include <torch/extension.h>
#include <cuda_runtime.h>
#include "../utils/cuda_utils.cuh"

// Optimized sum of squares reduction kernel
template<typename T>
__global__ void reduce_sum_squared_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t outer_size,
    int64_t inner_size
) {
    int64_t outer_idx = blockIdx.x;
    int64_t tid = threadIdx.x;
    
    if (outer_idx >= outer_size) return;
    
    extern __shared__ T shared_data[];
    
    T thread_sum = 0;
    int64_t input_offset = outer_idx * inner_size;
    
    // Each thread processes multiple elements with stride
    for (int64_t i = tid; i < inner_size; i += blockDim.x) {
        T val = input[input_offset + i];
        thread_sum += val * val;  // Square and accumulate
    }
    
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[outer_idx] = shared_data[0];
    }
}

// Optimized mean absolute value reduction kernel
template<typename T>
__global__ void reduce_mean_abs_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t outer_size,
    int64_t inner_size
) {
    int64_t outer_idx = blockIdx.x;
    int64_t tid = threadIdx.x;
    
    if (outer_idx >= outer_size) return;
    
    extern __shared__ T shared_data[];
    
    T thread_sum = 0;
    int64_t input_offset = outer_idx * inner_size;
    
    // Each thread processes multiple elements with stride
    for (int64_t i = tid; i < inner_size; i += blockDim.x) {
        T val = input[input_offset + i];
        thread_sum += abs(val);  // Absolute value and accumulate
    }
    
    shared_data[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[outer_idx] = shared_data[0] / static_cast<T>(inner_size);  // Mean
    }
}

namespace cuda_kernels {

torch::Tensor reduce_sum_squared_cuda(const torch::Tensor& input, int64_t dim, bool keepdim) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    
    // Handle last dimension case optimally
    if (dim == -1 || dim == input.dim() - 1) {
        auto input_shape = input.sizes().vec();
        auto output_shape = input_shape;
        
        if (keepdim) {
            output_shape[dim] = 1;
        } else {
            output_shape.erase(output_shape.begin() + dim);
        }
        
        auto output = torch::empty(output_shape, input.options());
        
        int64_t outer_size = input.numel() / input.size(dim);
        int64_t inner_size = input.size(dim);
        
        if (outer_size == 0 || inner_size == 0) return output;
        
        // Launch configuration
        dim3 block_size(256);
        dim3 grid_size(outer_size);
        size_t shared_mem_size = block_size.x * sizeof(float);
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "reduce_sum_squared_cuda", [&] {
            reduce_sum_squared_kernel<scalar_t><<<grid_size, block_size, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                inner_size
            );
        });
        
        CUDA_KERNEL_CHECK();
        return output;
    } else {
        // For other dimensions, fall back to PyTorch for now
        // TODO: Implement optimized kernels for other dimensions
        auto squared = torch::empty_like(input);
        
        int64_t numel = input.numel();
        LaunchConfig config(numel);
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "square_kernel", [&] {
            auto square_kernel = [] __device__ (scalar_t* input_ptr, scalar_t* output_ptr, int64_t n) {
                CUDA_KERNEL_LOOP(i, n) {
                    output_ptr[i] = input_ptr[i] * input_ptr[i];
                }
            };
            
            square_kernel<<<config.grid_size, config.block_size>>>(
                input.data_ptr<scalar_t>(),
                squared.data_ptr<scalar_t>(),
                numel
            );
        });
        
        return torch::sum(squared, dim, keepdim);
    }
}

torch::Tensor reduce_mean_abs_cuda(const torch::Tensor& input, int64_t dim, bool keepdim) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    
    // Handle last dimension case optimally
    if (dim == -1 || dim == input.dim() - 1) {
        auto input_shape = input.sizes().vec();
        auto output_shape = input_shape;
        
        if (keepdim) {
            output_shape[dim] = 1;
        } else {
            output_shape.erase(output_shape.begin() + dim);
        }
        
        auto output = torch::empty(output_shape, input.options());
        
        int64_t outer_size = input.numel() / input.size(dim);
        int64_t inner_size = input.size(dim);
        
        if (outer_size == 0 || inner_size == 0) return output;
        
        // Launch configuration
        dim3 block_size(256);
        dim3 grid_size(outer_size);
        size_t shared_mem_size = block_size.x * sizeof(float);
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "reduce_mean_abs_cuda", [&] {
            reduce_mean_abs_kernel<scalar_t><<<grid_size, block_size, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer_size,
                inner_size
            );
        });
        
        CUDA_KERNEL_CHECK();
        return output;
    } else {
        // For other dimensions, fall back to PyTorch for now
        // TODO: Implement optimized kernels for other dimensions  
        auto abs_input = torch::empty_like(input);
        
        int64_t numel = input.numel();
        LaunchConfig config(numel);
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "abs_kernel", [&] {
            auto abs_kernel = [] __device__ (scalar_t* input_ptr, scalar_t* output_ptr, int64_t n) {
                CUDA_KERNEL_LOOP(i, n) {
                    output_ptr[i] = abs(input_ptr[i]);
                }
            };
            
            abs_kernel<<<config.grid_size, config.block_size>>>(
                input.data_ptr<scalar_t>(),
                abs_input.data_ptr<scalar_t>(),
                numel
            );
        });
        
        return torch::mean(abs_input, dim, keepdim);
    }
}

} // namespace cuda_kernels
