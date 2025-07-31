# CUDA Algorithm Development Guide

## 📁 Workspace Structure

```
kernel-fusion/
├── kernel_fusion/              # Main Python package
│   ├── __init__.py            # Package initialization
│   └── kernels/               # Kernel implementations
│       ├── __init__.py        # Kernel module exports
│       ├── attention.py       # Legacy Triton attention (reference)
│       ├── cpu_kernels.py     # CPU implementations for testing
│       ├── cuda_attention.py  # CUDA attention example
│       └── simple_cuda.py     # Basic CUDA learning examples
├── examples/                  # Jupyter notebooks and examples
│   ├── cuda_development_guide.ipynb
│   ├── development_example.ipynb
│   └── macbook_development.ipynb
├── tests/                     # Test suite
│   └── test_environment.py
├── benchmarks/               # Performance benchmarks (to be created)
├── docker-compose.yml        # GPU development environment
├── docker-compose.cpu.yml    # CPU development environment
├── Dockerfile               # GPU Docker image
├── Dockerfile.cpu          # CPU Docker image
├── requirements.txt         # GPU dependencies
├── requirements-cpu.txt     # CPU dependencies
├── requirements-cuda.txt    # CUDA-specific dependencies
└── setup.py                # Package installation
```

## 🚀 Procedure for Adding a New CUDA Algorithm

### Step 1: Plan Your Algorithm

Before writing any code, plan your implementation:

1. **Understand the algorithm mathematically**
2. **Identify parallelization opportunities**
3. **Consider memory access patterns**
4. **Plan for optimization (shared memory, coalescing, etc.)**

### Step 2: Create CPU Prototype

Start with a CPU implementation to verify correctness:

```python
# In kernel_fusion/kernels/cpu_kernels.py
def cpu_my_algorithm(inputs):
    """CPU reference implementation"""
    # Implement algorithm logic here
    pass
```

### Step 3: Create CUDA Implementation

Follow this template structure:

```python
# Create: kernel_fusion/kernels/cuda_my_algorithm.py

import torch
import math
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code
cuda_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void my_algorithm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    // Your CUDA kernel implementation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx]; // Replace with your logic
    }
}

torch::Tensor my_algorithm_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    const int n = input.numel();
    
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;
    
    my_algorithm_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_algorithm", &my_algorithm_cuda, "My Algorithm CUDA");
}
'''

cpp_source = '''
torch::Tensor my_algorithm_cuda(torch::Tensor input);

torch::Tensor my_algorithm(torch::Tensor input) {
    return my_algorithm_cuda(input);
}
'''

class CUDAMyAlgorithm:
    def __init__(self):
        self.module = None
        self._compile_cuda_kernel()
    
    def _compile_cuda_kernel(self):
        try:
            self.module = load_inline(
                name='my_algorithm_cuda',
                cpp_sources=[cpp_source],
                cuda_sources=[cuda_source],
                functions=['my_algorithm'],
                verbose=True,
                extra_cuda_cflags=['-O3'],
                extra_cflags=['-O3']
            )
            print("CUDA kernel compiled successfully!")
        except Exception as e:
            print(f"Failed to compile CUDA kernel: {e}")
            self.module = None
    
    def forward(self, input):
        if self.module is not None and input.is_cuda:
            return self.module.my_algorithm(input.contiguous())
        else:
            # Fallback to CPU implementation
            return self._cpu_fallback(input)
    
    def _cpu_fallback(self, input):
        # Import and use CPU implementation
        from .cpu_kernels import cpu_my_algorithm
        return cpu_my_algorithm(input)

def my_algorithm(input):
    """High-level interface"""
    algorithm = CUDAMyAlgorithm()
    return algorithm.forward(input)
```

### Step 4: Update Package Exports

Add your new algorithm to the package:

```python
# In kernel_fusion/kernels/__init__.py
try:
    from .cuda_my_algorithm import *
    print("My algorithm CUDA kernels available")
except ImportError:
    print("My algorithm CUDA kernels not available")
```

### Step 5: Create Tests

```python
# Create: tests/test_my_algorithm.py
import torch
import pytest
from kernel_fusion.kernels.cuda_my_algorithm import my_algorithm

def test_my_algorithm_correctness():
    """Test algorithm correctness"""
    input_tensor = torch.randn(1000, device='cuda' if torch.cuda.is_available() else 'cpu')
    output = my_algorithm(input_tensor)
    
    # Add your correctness checks here
    assert output.shape == input_tensor.shape
    # Add more assertions...

def test_my_algorithm_performance():
    """Benchmark performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.randn(100000, device=device)
    
    # Time your implementation
    import time
    start = time.time()
    output = my_algorithm(input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    
    print(f"My algorithm took: {(end-start)*1000:.2f}ms")
```

### Step 6: Create Example Notebook

```python
# Create: examples/my_algorithm_example.ipynb
# Include:
# - Algorithm explanation
# - Usage examples  
# - Performance comparisons
# - Visualization of results
```

### Step 7: Add Benchmarks

```python
# Create: benchmarks/benchmark_my_algorithm.py
import torch
import time
from kernel_fusion.kernels.cuda_my_algorithm import my_algorithm

def benchmark_my_algorithm():
    """Comprehensive benchmark"""
    sizes = [1000, 10000, 100000, 1000000]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for size in sizes:
        input_tensor = torch.randn(size, device=device)
        
        # Warmup
        for _ in range(10):
            _ = my_algorithm(input_tensor)
        
        # Benchmark
        times = []
        for _ in range(100):
            start = time.time()
            _ = my_algorithm(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times) * 1000
        print(f"Size {size}: {avg_time:.3f}ms")

if __name__ == "__main__":
    benchmark_my_algorithm()
```

## 🔧 Development Workflow

### 1. Environment Setup
```bash
# For GPU development
make build
make dev

# For CPU development (MacBook Pro)
make build-mac  
make dev-mac
```

### 2. Development Loop
```bash
# Inside container
cd /workspace

# Edit your kernel
vim kernel_fusion/kernels/cuda_my_algorithm.py

# Test
python -m pytest tests/test_my_algorithm.py -v

# Benchmark
python benchmarks/benchmark_my_algorithm.py

# Interactive development
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 3. Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Run all tests
make test
```

## 🎯 Best Practices

### CUDA Kernel Design
1. **Start simple**: Naive implementation first
2. **Optimize incrementally**: Profile and improve bottlenecks
3. **Memory coalescing**: Access consecutive memory locations
4. **Shared memory**: Cache frequently accessed data
5. **Occupancy**: Use appropriate block sizes (typically 256 threads)

### Code Organization
1. **Separate concerns**: CPU fallback, CUDA implementation, high-level interface
2. **Error handling**: Graceful fallback when CUDA compilation fails
3. **Documentation**: Clear docstrings and examples
4. **Testing**: Both correctness and performance tests

### Performance Considerations
1. **Memory bandwidth**: Often the limiting factor
2. **Compute intensity**: Balance compute vs memory operations
3. **Kernel fusion**: Combine multiple operations
4. **Data types**: Consider fp16 for better performance

## 📊 Profiling and Optimization

### Tools Available in Container
- **nsys**: System-wide profiling
- **ncu**: Kernel-level profiling  
- **PyTorch Profiler**: Python-level profiling
- **cuda-gdb**: Debugging

### Example Profiling Commands
```bash
# Profile with nsys
nsys profile python benchmark_my_algorithm.py

# Profile with ncu
ncu python benchmark_my_algorithm.py

# PyTorch profiler in code
with torch.profiler.profile() as prof:
    output = my_algorithm(input_tensor)
print(prof.key_averages().table())
```

## 🚀 Ready to Start!

Follow this workflow to add any CUDA algorithm to the workspace. The structure provides:
- Clear separation of CPU/GPU code
- Automatic fallbacks
- Comprehensive testing
- Performance benchmarking
- Educational examples

Start with a simple algorithm and gradually increase complexity as you learn!
