# Kernel Fusion

A collection of optimized fusion kernels for deep learning operations using **direct CUDA programming** and PyTorch.

## 🚀 Features

- **Direct CUDA kernels**: Hand-written CUDA kernels for maximum performance
- **PyTorch integration**: Seamless integration using PyTorch C++ extensions
- **Cross-platform development**: CPU prototyping on MacBook Pro, GPU development on Linux/Cloud
- **Educational focus**: Learn CUDA programming from basics to advanced fusion
- **Comprehensive benchmarking**: Tools to measure and compare kernel performance

## 🎯 Why Direct CUDA?

- **Maximum Performance**: Direct control over GPU resources
- **Educational Value**: Learn fundamental GPU programming concepts
- **Flexibility**: Implement any algorithm without framework limitations
- **Portability**: Works across different CUDA-capable hardware

## 🛠 Development Setup

### Prerequisites

**For GPU Development (Linux/Windows with NVIDIA GPU):**
- Docker and Docker Compose
- NVIDIA Docker runtime
- NVIDIA GPU with CUDA support

**For CPU Development (MacBook Pro/Apple Silicon):**
- Docker Desktop for Mac
- No GPU required - optimized for CPU development

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YangyangFu/kernel-fusion.git
   cd kernel-fusion
   ```

2. **Set up the development environment**:
   ```bash
   # Auto-detects your platform
   make setup
   
   # Or manually:
   # For GPU development (Linux/NVIDIA)
   ./setup-dev.sh
   
   # For CPU development (MacBook Pro)
   ./setup-mac.sh
   ```

3. **Start development**:
   ```bash
   # GPU Development
   make dev        # Interactive container
   make jupyter    # Jupyter Lab
   
   # CPU Development (MacBook Pro)
   make dev-mac    # Interactive container
   make jupyter-mac # Jupyter Lab
   ```

4. **Access services**:
   - Jupyter Lab: http://localhost:8888
   - TensorBoard: http://localhost:6006

### Development Options by Platform

#### 🖥 **GPU Development (Linux/NVIDIA)**
- Full CUDA and Triton support
- Optimal for production kernel development
- Real GPU performance testing

#### 🍎 **MacBook Pro Development**
- CPU-optimized kernel development
- Algorithm prototyping and testing
- PyTorch MPS backend support (Apple Silicon)
- Perfect for learning and initial development

#### ☁️ **Cloud GPU Development**
- Use cloud platforms for GPU kernel testing:
  - Google Colab Pro/Pro+
  - AWS EC2 GPU instances
  - Paperspace Gradient
  - Lambda Labs

### Docker Services

**GPU Development:**
- **kernel-fusion-dev**: Interactive development container with CUDA
- **jupyter**: Jupyter Lab server with GPU support

**CPU Development (MacBook Pro):**
- **kernel-fusion-cpu**: CPU-optimized development container
- **jupyter-cpu**: Jupyter Lab server for CPU development

## 📂 Project Structure

```
kernel-fusion/
├── kernel_fusion/          # Main package
│   ├── __init__.py
│   └── kernels/           # Kernel implementations
│       ├── __init__.py
│       └── attention.py   # Example attention kernel
├── tests/                 # Test suite
├── examples/              # Example notebooks and scripts
├── benchmarks/            # Performance benchmarks
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile            # Docker image definition
├── requirements.txt      # Python dependencies
└── setup.py             # Package setup
```

## 🧪 Testing

Run tests inside the container:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_environment.py

# Run with coverage
python -m pytest --cov=kernel_fusion tests/
```

## 🏃‍♂️ Usage Examples

### MacBook Pro CPU Development

```python
from kernel_fusion.kernels.cpu_kernels import (
    cpu_fused_attention,
    optimized_cpu_attention,
    CPUKernelBenchmark
)

# Create test tensors
q = torch.randn(2, 8, 512, 64)  # [batch, heads, seq_len, d_head]
k = torch.randn(2, 8, 512, 64)
v = torch.randn(2, 8, 512, 64)

# Run CPU-optimized attention
output = cpu_fused_attention(q, k, v)

# Benchmark different implementations
implementations = {
    'standard': cpu_fused_attention,
    'chunked': lambda q, k, v: optimized_cpu_attention(q, k, v, chunk_size=64),
    'pytorch': torch.nn.functional.scaled_dot_product_attention
}
results = CPUKernelBenchmark.compare_implementations(implementations, q, k, v)
```

### GPU Development (Linux/NVIDIA)

```python
import torch
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Kernel implementation
    pass

# Use the kernel
x = torch.randn(1000, device='cuda')
output = my_custom_function(x)
```

### Apple MPS Backend (Apple Silicon)

```python
# Leverage Apple's Metal Performance Shaders
if torch.backends.mps.is_available():
    device = torch.device("mps")
    q = torch.randn(2, 8, 512, 64, device=device)
    k = torch.randn(2, 8, 512, 64, device=device)
    v = torch.randn(2, 8, 512, 64, device=device)
    
    # Use PyTorch's optimized attention
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

## 📊 Benchmarking

The project includes comprehensive benchmarking tools:

```python
from kernel_fusion.benchmarks import benchmark_kernel

# Benchmark your kernel
results = benchmark_kernel(
    your_kernel_function,
    inputs,
    warmup=10,
    repeat=100
)
```

## 🔧 Development Tools

The development environment includes:

- **PyTorch 2.0+** with CUDA support
- **CUDA Toolkit** for direct CUDA kernel development
- **PyCUDA** and **CuPy** for GPU computing
- **PyTorch C++ Extensions** for seamless integration
- **Jupyter Lab** for interactive development
- **Development tools**: black, flake8, mypy, pytest
- **Monitoring**: TensorBoard, Weights & Biases

## 📚 Learning Path

### 1. **CPU Prototyping** (MacBook Pro)
- Understand algorithms and data flow
- Implement CPU versions for correctness
- Learn CUDA concepts without GPU

### 2. **Basic CUDA Kernels**
- Element-wise operations
- Memory access patterns
- Thread and block organization

### 3. **Advanced Fusion Kernels**
- Attention mechanisms
- Layer normalization
- Custom activation functions

### 4. **Optimization Techniques**
- Memory coalescing
- Shared memory usage
- Warp-level primitives

## 🐛 Environment Variables

Customize the container behavior:

```bash
# GPU visibility
export NVIDIA_VISIBLE_DEVICES=0,1

# CUDA capabilities
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Develop your kernel in the Docker environment
4. Add tests and benchmarks
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Support

- Create an issue for bug reports
- Discussions for questions and ideas
- Wiki for detailed documentation

## 🎯 Roadmap

- [ ] Fused attention kernels
- [ ] LayerNorm optimizations
- [ ] Activation function fusions
- [ ] Memory-efficient training kernels
- [ ] Multi-GPU kernel support
- [ ] Automatic kernel tuning
