# Baseline Implementation Notes

This document describes the baseline implementations used for comparison in the kernel fusion benchmarks.

## Overview

To accurately measure the benefits of kernel fusion, we compare against well-optimized baseline implementations:

1. **PyTorch baselines**: Official PyTorch C++ API implementations
2. **Thrust baselines**: NVIDIA Thrust library implementations  
3. **cuBLAS baselines**: NVIDIA cuBLAS optimized kernels where applicable
4. **Custom baselines**: Hand-tuned individual CUDA kernels

## PyTorch Baselines

### Setup and Dependencies

The PyTorch baselines use LibTorch, the C++ frontend of PyTorch:

```cpp
#include <torch/torch.h>
#include <ATen/ATen.h>

// Device setup
torch::Device device(torch::kCUDA, 0);
torch::TensorOptions options = torch::TensorOptions()
    .device(device)
    .dtype(torch::kFloat32);
```

### Activation Function Implementations

#### ReLU
```cpp
torch::Tensor pytorch_relu(const torch::Tensor& input) {
    return torch::relu(input);
}
```

#### Sigmoid  
```cpp
torch::Tensor pytorch_sigmoid(const torch::Tensor& input) {
    return torch::sigmoid(input);
}
```

#### Tanh
```cpp
torch::Tensor pytorch_tanh(const torch::Tensor& input) {
    return torch::tanh(input);
}
```

#### GELU
```cpp
torch::Tensor pytorch_gelu(const torch::Tensor& input) {
    return torch::gelu(input);
}
```

### Operation Chains

For baseline comparison, we execute operations sequentially:

```cpp
torch::Tensor pytorch_activation_chain(const torch::Tensor& input) {
    auto temp1 = torch::relu(input);
    auto temp2 = torch::sigmoid(temp1);
    auto result = torch::tanh(temp2);
    return result;
}
```

### Memory Management

```cpp
class PytorchBaseline {
private:
    torch::Device device_;
    torch::TensorOptions options_;
    
public:
    PytorchBaseline() : device_(torch::kCUDA, 0) {
        options_ = torch::TensorOptions()
            .device(device_)
            .dtype(torch::kFloat32);
    }
    
    torch::Tensor create_tensor(const std::vector<int64_t>& sizes) {
        return torch::empty(sizes, options_);
    }
    
    void copy_from_device(const torch::Tensor& src, float* dst) {
        auto cpu_tensor = src.to(torch::kCPU);
        std::memcpy(dst, cpu_tensor.data_ptr<float>(), 
                   src.numel() * sizeof(float));
    }
};
```

## Thrust Baselines

### Basic Operations

Thrust provides high-level parallel algorithms:

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

// ReLU using Thrust
struct relu_functor {
    __device__ float operator()(float x) const {
        return fmaxf(0.0f, x);
    }
};

void thrust_relu(float* input, float* output, int n) {
    thrust::device_ptr<float> d_input(input);
    thrust::device_ptr<float> d_output(output);
    thrust::transform(d_input, d_input + n, d_output, relu_functor());
}
```

### Sequential Operations

```cpp
void thrust_activation_chain(float* input, float* output, int n) {
    thrust::device_vector<float> temp1(n);
    thrust::device_vector<float> temp2(n);
    
    // Step 1: ReLU
    thrust::transform(thrust::device_ptr<float>(input),
                     thrust::device_ptr<float>(input) + n,
                     temp1.begin(), relu_functor());
                     
    // Step 2: Sigmoid  
    thrust::transform(temp1.begin(), temp1.end(),
                     temp2.begin(), sigmoid_functor());
                     
    // Step 3: Tanh
    thrust::transform(temp2.begin(), temp2.end(),
                     thrust::device_ptr<float>(output),
                     tanh_functor());
}
```

### Custom Functors

```cpp
struct sigmoid_functor {
    __device__ float operator()(float x) const {
        return 1.0f / (1.0f + expf(-x));
    }
};

struct tanh_functor {
    __device__ float operator()(float x) const {
        return tanhf(x);
    }
};

struct gelu_functor {
    __device__ float operator()(float x) const {
        return 0.5f * x * (1.0f + tanhf(0.79788456f * x * (1.0f + 0.044715f * x * x)));
    }
};
```

## cuBLAS Baselines

For operations that can be expressed as BLAS operations:

```cpp
#include <cublas_v2.h>

class CublasBaseline {
private:
    cublasHandle_t handle_;
    
public:
    CublasBaseline() {
        cublasCreate(&handle_);
    }
    
    ~CublasBaseline() {
        cublasDestroy(handle_);
    }
    
    // Element-wise multiplication (DGBMV can be used)
    void elementwise_multiply(float* a, float* b, float* result, int n) {
        const float alpha = 1.0f, beta = 0.0f;
        // Use cuBLAS where appropriate for vectorized operations
        cublasSaxpy(handle_, n, &alpha, a, 1, result, 1);
        // Custom kernel for element-wise multiply
        elementwise_mul_kernel<<<(n+255)/256, 256>>>(a, b, result, n);
    }
};
```

## Custom CUDA Baselines

### Individual Kernel Implementations

For operations without efficient library implementations:

```cpp
__global__ void relu_baseline_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void sigmoid_baseline_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void tanh_baseline_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}
```

### Optimized Individual Kernels

```cpp
// Optimized ReLU with vectorized loads
__global__ void relu_optimized_baseline(const float4* input, float4* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 4 < n) {
        float4 val = input[idx];
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
        output[idx] = val;
    }
}
```

## Baseline Selection Strategy

### Per-Operation Analysis

1. **Library Availability**: Use mature, optimized libraries when available
2. **Operation Complexity**: Simple ops → custom kernels, complex ops → libraries
3. **Memory Pattern**: Choose implementation matching memory access pattern
4. **Performance Characteristics**: Select fastest available baseline

### Implementation Priority

1. **PyTorch**: Primary baseline for activation functions and neural network ops
2. **Thrust**: Secondary baseline for elementwise operations  
3. **cuBLAS**: For linear algebra operations where applicable
4. **Custom**: When library implementations are suboptimal

### Fairness Considerations

#### Optimization Level
- All baselines compiled with `-O3` optimization
- Same compiler flags as fusion implementations
- No unfair optimization differences

#### Memory Allocation
- Pre-allocated memory to exclude allocation overhead
- Same memory layout and alignment
- Cache-warming iterations before timing

#### Measurement Methodology
```cpp
class BaselineBenchmark {
public:
    float benchmark_operation(std::function<void()> op, int warmup_iters = 5, int timing_iters = 100) {
        // Warmup
        for (int i = 0; i < warmup_iters; i++) {
            op();
        }
        cudaDeviceSynchronize();
        
        // Timing
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timing_iters; i++) {
            op();
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / float(timing_iters) / 1000.0f; // ms
    }
};
```

## Performance Characteristics

### Expected Performance Ranges

#### Single Operations (A100, FP32, 64M elements)
- **PyTorch ReLU**: ~1.2 TB/s memory bandwidth
- **PyTorch Sigmoid**: ~1.0 TB/s memory bandwidth  
- **Thrust ReLU**: ~1.1 TB/s memory bandwidth
- **Custom ReLU**: ~1.3 TB/s memory bandwidth

#### Operation Chains
- **2-operation chain**: ~0.6-0.8x single operation bandwidth
- **3-operation chain**: ~0.4-0.6x single operation bandwidth
- **Memory bound**: Limited by global memory bandwidth

### Bottleneck Analysis

#### Memory Bandwidth Limited
Most elementwise operations are memory bandwidth limited:
- **Arithmetic Intensity**: Low (1-2 ops per memory access)
- **Limiting Factor**: Global memory throughput
- **Optimization Target**: Minimize memory transactions

#### Kernel Launch Overhead
Sequential operations suffer from launch overhead:
- **Per-kernel overhead**: ~5-10 μs depending on GPU
- **Impact**: Significant for small workloads
- **Fusion benefit**: Eliminates intermediate launches

## Validation Against Baselines

### Numerical Accuracy
```cpp
bool validate_against_pytorch(float* fusion_result, const torch::Tensor& pytorch_result, int n) {
    std::vector<float> pytorch_cpu(n);
    auto cpu_tensor = pytorch_result.to(torch::kCPU);
    std::memcpy(pytorch_cpu.data(), cpu_tensor.data_ptr<float>(), n * sizeof(float));
    
    std::vector<float> fusion_cpu(n);
    cudaMemcpy(fusion_cpu.data(), fusion_result, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    return validate_results(fusion_cpu.data(), pytorch_cpu.data(), n);
}
```

### Performance Comparison
```cpp
struct BenchmarkResult {
    float pytorch_time_ms;
    float thrust_time_ms; 
    float fusion_time_ms;
    float speedup_vs_pytorch;
    float speedup_vs_thrust;
};

BenchmarkResult compare_implementations(int n) {
    BenchmarkResult result;
    
    // Benchmark PyTorch
    result.pytorch_time_ms = benchmark_pytorch_chain(n);
    
    // Benchmark Thrust  
    result.thrust_time_ms = benchmark_thrust_chain(n);
    
    // Benchmark Fusion
    result.fusion_time_ms = benchmark_fusion_chain(n);
    
    // Calculate speedups
    result.speedup_vs_pytorch = result.pytorch_time_ms / result.fusion_time_ms;
    result.speedup_vs_thrust = result.thrust_time_ms / result.fusion_time_ms;
    
    return result;
}
```

## Limitations and Considerations

### PyTorch Limitations
- **Overhead**: Python→C++ call overhead (minimal in C++ API)
- **Memory Layout**: May use different layouts than custom implementations
- **Version Dependency**: Results may vary between PyTorch versions

### Thrust Limitations  
- **Memory Allocation**: Internal temporary allocations may affect timing
- **Algorithm Selection**: May use different algorithms than expected
- **Precision**: Different numerical characteristics than manual implementations

### Fair Comparison Guidelines
1. **Same Input Data**: Identical input patterns and sizes
2. **Same Output**: Verify numerical equivalence
3. **Same Environment**: Same GPU, CUDA version, driver
4. **Multiple Runs**: Average over multiple iterations
5. **Error Handling**: Consistent error checking overhead

This baseline implementation strategy ensures fair and meaningful comparisons that accurately demonstrate the benefits of kernel fusion optimizations.
