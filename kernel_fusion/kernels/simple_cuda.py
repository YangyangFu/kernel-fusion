"""
Simple CUDA kernels for learning and development
These examples show basic CUDA programming patterns
"""

import torch
import math
from torch.utils.cpp_extension import load_inline


# Simple element-wise operations
elementwise_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void gelu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto result = torch::zeros_like(a);
    const int n = a.numel();
    
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    add_kernel<<<grid_size, block_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        n
    );
    
    cudaDeviceSynchronize();
    return result;
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    const int n = input.numel();
    
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    relu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    cudaDeviceSynchronize();
    return output;
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    const int n = input.numel();
    
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    gelu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_cuda, "Element-wise addition");
    m.def("relu", &relu_cuda, "ReLU activation");
    m.def("gelu", &gelu_cuda, "GELU activation");
}
"""

elementwise_cpp_source = """
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor relu_cuda(torch::Tensor input);
torch::Tensor gelu_cuda(torch::Tensor input);

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    return add_cuda(a, b);
}

torch::Tensor relu(torch::Tensor input) {
    return relu_cuda(input);
}

torch::Tensor gelu(torch::Tensor input) {
    return gelu_cuda(input);
}
"""


class CUDAKernels:
    """Collection of simple CUDA kernels for learning"""
    
    def __init__(self):
        self.module = None
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile CUDA kernels"""
        try:
            self.module = load_inline(
                name='simple_cuda_kernels',
                cpp_sources=[elementwise_cpp_source],
                cuda_sources=[elementwise_cuda_source],
                functions=['add', 'relu', 'gelu'],
                verbose=True,
                extra_cuda_cflags=['-O3', '--use_fast_math'],
                extra_cflags=['-O3']
            )
            print("CUDA kernels compiled successfully!")
        except Exception as e:
            print(f"Failed to compile CUDA kernels: {e}")
            print("Using CPU fallback implementations")
            self.module = None
    
    def add(self, a, b):
        """Element-wise addition"""
        if self.module is not None and a.is_cuda and b.is_cuda:
            return self.module.add(a.contiguous(), b.contiguous())
        else:
            return a + b
    
    def relu(self, x):
        """ReLU activation"""
        if self.module is not None and x.is_cuda:
            return self.module.relu(x.contiguous())
        else:
            return torch.relu(x)
    
    def gelu(self, x):
        """GELU activation"""
        if self.module is not None and x.is_cuda:
            return self.module.gelu(x.contiguous())
        else:
            return torch.nn.functional.gelu(x)


def benchmark_kernels():
    """Benchmark CUDA kernels vs PyTorch implementations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmarks on: {device}")
    
    kernels = CUDAKernels()
    sizes = [1000, 10000, 100000, 1000000]
    
    for size in sizes:
        print(f"\nBenchmarking size: {size}")
        
        # Create test data
        a = torch.randn(size, device=device)
        b = torch.randn(size, device=device)
        x = torch.randn(size, device=device)
        
        if device.type == 'cuda':
            # Warmup
            for _ in range(10):
                _ = kernels.add(a, b)
                _ = kernels.relu(x)
                _ = kernels.gelu(x)
            
            torch.cuda.synchronize()
            
            # Benchmark custom kernels
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(100):
                _ = kernels.add(a, b)
            end.record()
            torch.cuda.synchronize()
            custom_add_time = start.elapsed_time(end) / 100
            
            start.record()
            for _ in range(100):
                _ = kernels.relu(x)
            end.record()
            torch.cuda.synchronize()
            custom_relu_time = start.elapsed_time(end) / 100
            
            # Benchmark PyTorch implementations
            start.record()
            for _ in range(100):
                _ = a + b
            end.record()
            torch.cuda.synchronize()
            pytorch_add_time = start.elapsed_time(end) / 100
            
            start.record()
            for _ in range(100):
                _ = torch.relu(x)
            end.record()
            torch.cuda.synchronize()
            pytorch_relu_time = start.elapsed_time(end) / 100
            
            print(f"  Add - Custom: {custom_add_time:.3f}ms, PyTorch: {pytorch_add_time:.3f}ms")
            print(f"  ReLU - Custom: {custom_relu_time:.3f}ms, PyTorch: {pytorch_relu_time:.3f}ms")
        else:
            print("  CUDA not available, skipping GPU benchmarks")


if __name__ == "__main__":
    benchmark_kernels()
