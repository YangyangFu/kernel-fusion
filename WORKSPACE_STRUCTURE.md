# 📁 Workspace Structure & CUDA Algorithm Development

## 🏗 **Complete Workspace Structure**

```
kernel-fusion/
├── 📦 kernel_fusion/                    # Main Python package
│   ├── __init__.py                     # Package initialization
│   └── 📂 kernels/                     # Kernel implementations
│       ├── __init__.py                 # Exports all kernels
│       ├── cpu_kernels.py              # CPU reference implementations
│       ├── cpu_layernorm.py            # CPU LayerNorm reference  
│       ├── attention.py                # Legacy Triton attention
│       ├── cuda_attention.py           # CUDA attention kernel
│       ├── cuda_layernorm.py           # CUDA LayerNorm kernel
│       └── simple_cuda.py              # Basic CUDA learning examples
│
├── 📓 examples/                        # Jupyter notebooks & examples
│   ├── cuda_development_guide.ipynb   # Complete CUDA tutorial
│   ├── layernorm_example.ipynb        # LayerNorm implementation example
│   ├── macbook_development.ipynb      # MacBook Pro development
│   └── development_example.ipynb      # General development examples
│
├── 🧪 tests/                          # Test suite
│   ├── test_environment.py            # Environment validation
│   └── test_layernorm.py              # LayerNorm correctness & performance
│
├── 🐳 Docker Configuration             # Development environments
│   ├── Dockerfile                     # GPU development (CUDA 12.1)
│   ├── Dockerfile.cpu                 # CPU development (MacBook Pro)
│   ├── docker-compose.yml             # GPU services
│   └── docker-compose.cpu.yml         # CPU services
│
├── 📋 Requirements                     # Dependencies
│   ├── requirements.txt               # General requirements
│   ├── requirements-cuda.txt          # CUDA-specific packages
│   └── requirements-cpu.txt           # CPU-only packages
│
├── 🛠 Development Tools               # Setup & automation
│   ├── Makefile                       # Common development tasks
│   ├── setup-dev.sh                   # GPU environment setup
│   ├── setup-mac.sh                   # MacBook Pro setup
│   └── setup.py                       # Package installation
│
└── 📚 Documentation
    ├── README.md                       # Project overview
    ├── CUDA_DEVELOPMENT_GUIDE.md       # This comprehensive guide
    └── .gitignore                      # Git configuration
```

## 🚀 **Step-by-Step: Adding a New CUDA Algorithm**

### 1️⃣ **Plan Your Algorithm**
```bash
# Understand the math and parallelization strategy
# Example: For LayerNorm, we parallelize across sequence positions
```

### 2️⃣ **Create CPU Reference**
```python
# File: kernel_fusion/kernels/cpu_my_algorithm.py
def cpu_my_algorithm(input_tensor):
    """CPU reference implementation for correctness testing"""
    # Implement your algorithm here
    return output_tensor
```

### 3️⃣ **Implement CUDA Kernel**
```python
# File: kernel_fusion/kernels/cuda_my_algorithm.py
cuda_source = """
__global__ void my_algorithm_kernel(
    const float* input,
    float* output,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Your CUDA implementation
        output[idx] = process(input[idx]);
    }
}
"""

class CUDAMyAlgorithm:
    def __init__(self):
        self.module = self._compile_cuda_kernel()
    
    def forward(self, input_tensor):
        if self.module and input_tensor.is_cuda:
            return self.module.my_algorithm(input_tensor)
        else:
            return cpu_fallback(input_tensor)
```

### 4️⃣ **Add to Package**
```python
# File: kernel_fusion/kernels/__init__.py
try:
    from .cuda_my_algorithm import *
    print("My algorithm CUDA kernels available")
except ImportError:
    print("My algorithm CUDA kernels not available")
```

### 5️⃣ **Create Tests**
```python
# File: tests/test_my_algorithm.py
def test_my_algorithm_correctness():
    # Test against CPU reference implementation
    
def test_my_algorithm_performance():
    # Benchmark against PyTorch/other implementations
```

### 6️⃣ **Create Example Notebook**
```python
# File: examples/my_algorithm_example.ipynb
# - Algorithm explanation
# - Usage examples
# - Performance analysis
# - Visualization
```

## 🔄 **Development Workflow**

### **Environment Setup**
```bash
# MacBook Pro (CPU development)
make setup          # Auto-detects macOS
make dev-mac         # Start CPU container
make jupyter-mac     # Launch Jupyter

# Linux/Cloud (GPU development)  
make setup           # Build GPU environment
make dev             # Start GPU container
make jupyter         # Launch Jupyter with CUDA
```

### **Development Loop**
```bash
# Inside container
cd /workspace

# 1. Edit kernel
vim kernel_fusion/kernels/cuda_my_algorithm.py

# 2. Test correctness
python -m pytest tests/test_my_algorithm.py -v

# 3. Benchmark performance
python -c "from tests.test_my_algorithm import benchmark_my_algorithm; benchmark_my_algorithm()"

# 4. Interactive development
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### **Code Quality**
```bash
make format          # Black code formatting
make lint            # Flake8 + mypy linting  
make test            # Run all tests
```

## 🎯 **CUDA Kernel Design Patterns**

### **1. Element-wise Operations**
```cuda
__global__ void elementwise_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = process(input[idx]);
    }
}
// Grid: (n/block_size,), Block: (block_size,)
```

### **2. Reduction Operations (LayerNorm, Attention)**
```cuda
__global__ void reduction_kernel(const float* input, float* output, int n) {
    __shared__ float shared_data[BLOCK_SIZE];
    
    // Load data
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    shared_data[threadIdx.x] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Store result
    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}
```

### **3. Matrix Operations (Attention)**
```cuda
__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
// Grid: (N/16, M/16), Block: (16, 16)
```

## 🏆 **Best Practices**

### **Memory Optimization**
- ✅ **Coalescing**: Access consecutive memory locations
- ✅ **Shared Memory**: Cache frequently accessed data
- ✅ **Bank Conflicts**: Avoid shared memory bank conflicts
- ✅ **Occupancy**: Use 256+ threads per block

### **Code Organization**
- ✅ **Separation**: CPU fallback + CUDA implementation + high-level interface
- ✅ **Error Handling**: Graceful fallback when compilation fails
- ✅ **Documentation**: Clear docstrings and examples
- ✅ **Testing**: Both correctness and performance validation

### **Performance Tuning**
- ✅ **Profile First**: Use nsys/ncu to identify bottlenecks
- ✅ **Kernel Fusion**: Combine multiple operations
- ✅ **Data Types**: Consider fp16 for better throughput
- ✅ **Memory vs Compute**: Balance bandwidth vs compute intensity

## 🛠 **Available Tools**

### **Development Environment**
- **PyTorch C++ Extensions**: JIT compilation of CUDA kernels
- **Docker**: Consistent development environment
- **Jupyter**: Interactive development and experimentation

### **Profiling & Debugging**
- **nsys**: System-wide profiling (`nsys profile python script.py`)
- **ncu**: Kernel-level profiling (`ncu python script.py`) 
- **PyTorch Profiler**: Python-level profiling
- **cuda-gdb**: CUDA debugging

### **Testing & Validation**
- **pytest**: Automated testing framework
- **torch.allclose()**: Numerical correctness validation
- **Benchmarking**: Performance comparison utilities

## 📊 **Example: LayerNorm Implementation**

We've created a complete LayerNorm example that demonstrates:

- ✅ **CPU Reference**: `cpu_layernorm.py`
- ✅ **CUDA Implementation**: `cuda_layernorm.py` 
- ✅ **Comprehensive Tests**: `test_layernorm.py`
- ✅ **Interactive Example**: `layernorm_example.ipynb`
- ✅ **Performance Benchmarks**: Included in tests

This serves as a template for adding any new CUDA algorithm!

## 🎉 **Ready to Develop!**

The workspace provides everything needed for CUDA kernel development:

1. **🍎 MacBook Pro**: Start with CPU prototypes and learning
2. **☁️ Cloud GPU**: Move to actual CUDA development when ready  
3. **🔄 Automated Workflow**: Build, test, benchmark, and deploy
4. **📚 Comprehensive Examples**: Learn from working implementations

Follow the LayerNorm example to add your own CUDA algorithms!
