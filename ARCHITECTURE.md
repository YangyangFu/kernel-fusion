# KernelFusion Standalone Architecture

## Overview

KernelFusion has been redesigned as a standalone CUDA library with pluggable frontends, providing maximum flexibility and scalability across different machine learning frameworks.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  PyTorch Frontend  │  TensorFlow Frontend  │  JAX Frontend  │
├─────────────────────────────────────────────────────────────┤
│                    Frontend Abstraction                     │
├─────────────────────────────────────────────────────────────┤
│                   Core C/C++ API                           │
├─────────────────────────────────────────────────────────────┤
│                   CUDA Kernels & Runtime                    │
├─────────────────────────────────────────────────────────────┤
│                     CUDA Driver                            │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Core Library (`core/`)
- **Pure C/C++ API**: Framework-agnostic interface
- **CUDA Kernels**: Optimized kernel implementations
- **Memory Management**: Efficient allocation and pooling
- **Stream Management**: Multi-stream execution support
- **Type System**: Universal tensor and data type definitions

### 2. Frontend System (`frontends/`)
- **PyTorch Bridge**: Native PyTorch tensor integration
- **TensorFlow Ops**: Custom TensorFlow operations (planned)
- **JAX Extensions**: JAX custom call implementations (planned)
- **C API**: Direct C/C++ application interface

## Key Design Principles

### 🔧 **Framework Agnostic Core**
```c
// Pure C API - no framework dependencies
kf_result_t kf_fused_elementwise_add_activation(
    kf_context_t* context,
    const kf_tensor_t* a,
    const kf_tensor_t* b,
    kf_tensor_t* output,
    kf_activation_t activation,
    kf_stream_t* stream
);
```

### 🔌 **Pluggable Frontends**
```cpp
// PyTorch Frontend
torch::Tensor fused_add_relu(const torch::Tensor& a, const torch::Tensor& b) {
    // Bridge PyTorch tensors to core API
    auto context = create_context_from_torch(a);
    return call_core_and_convert_back(context, a, b);
}

// TensorFlow Frontend (planned)
class FusedAddReLUOp : public tensorflow::OpKernel {
    // Bridge TensorFlow tensors to core API
};

// JAX Frontend (planned)  
def fused_add_relu_jax(a, b):
    # Bridge JAX arrays to core API via custom call
    return jax.custom_call("kf_fused_add_relu", ...)
```

### 🚀 **High Performance Core**
- **Zero-copy tensor sharing** between frameworks and core
- **Memory pooling** for efficient allocation
- **Multi-stream execution** with priority scheduling
- **Optimized kernels** with template specialization

## Memory Management Strategy

### Framework Tensor Integration
```cpp
// PyTorch: Share memory, don't copy
kf_tensor_t* tensor_from_torch(const torch::Tensor& tensor) {
    kf_memory_desc_t memory = {
        .data = tensor.data_ptr(),
        .owns_memory = false  // PyTorch owns it
    };
    return kf_tensor_create_from_memory(&memory, &desc);
}
```

### Core Memory Pool
```cpp
class MemoryPool {
    void* allocate(size_t size);     // Reuse freed blocks
    void deallocate(void* ptr);      // Return to pool
    void clear();                    // Release all memory
};
```

## Stream Management

### Multi-Priority Streams
```c
kf_stream_config_t config = {
    .priority = KF_PRIORITY_HIGH,    // 0 = normal, >0 = higher priority
    .flags = KF_STREAM_NON_BLOCKING,
    .device_id = 0
};
kf_stream_t* stream = kf_stream_create(context, &config);
```

### Event Synchronization
```c
// Record event in stream A
kf_event_t* event = kf_event_create(context);
kf_event_record(event, stream_a);

// Wait for event in stream B
kf_stream_wait_event(stream_b, event);
```

## Type System

### Universal Data Types
```c
typedef enum {
    KF_DTYPE_FLOAT32,
    KF_DTYPE_FLOAT16,
    KF_DTYPE_BFLOAT16,
    KF_DTYPE_INT32,
    KF_DTYPE_INT64
} kf_dtype_t;
```

### Device Abstraction
```c
typedef enum {
    KF_DEVICE_CPU,
    KF_DEVICE_CUDA
} kf_device_type_t;
```

### Tensor Descriptors
```c
typedef struct {
    int64_t* shape;
    int64_t* strides;
    int ndim;
    kf_dtype_t dtype;
    kf_device_type_t device_type;
    int device_id;
} kf_tensor_desc_t;
```

## Kernel Interface

### Fused Operations
```c
// Elementwise operations
kf_fused_elementwise_add_activation()
kf_fused_elementwise_mul_activation()

// Linear algebra
kf_fused_linear_activation()
kf_fused_conv2d_bn_activation()

// Normalization
kf_fused_layer_norm_activation()
kf_fused_batch_norm_activation()

// Attention
kf_fused_multi_head_attention()
```

### Activation Functions
```c
typedef enum {
    KF_ACTIVATION_NONE,
    KF_ACTIVATION_RELU,
    KF_ACTIVATION_GELU,
    KF_ACTIVATION_SILU,
    KF_ACTIVATION_TANH
} kf_activation_t;
```

## Build System

### CMake Configuration
```cmake
# Core library (framework-agnostic)
add_library(kernel_fusion_core core/src/core.cpp core/src/kernels.cu)

# PyTorch frontend
if(BUILD_PYTORCH_FRONTEND)
    pybind11_add_module(kernel_fusion_pytorch frontends/pytorch/torch_bridge.cpp)
    target_link_libraries(kernel_fusion_pytorch kernel_fusion_core ${TORCH_LIBRARIES})
endif()

# TensorFlow frontend  
if(BUILD_TENSORFLOW_FRONTEND)
    add_library(kernel_fusion_tensorflow frontends/tensorflow/tf_ops.cpp)
endif()
```

### Installation
```bash
# Build core + all frontends
mkdir build && cd build
cmake .. -DBUILD_PYTORCH_FRONTEND=ON -DBUILD_TENSORFLOW_FRONTEND=ON
make -j$(nproc)

# Install system-wide
sudo make install

# Or install to custom prefix
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/kernel-fusion
make install
```

## Usage Examples

### C API (Direct)
```c
kf_context_t* ctx = kf_context_create(0);
kf_tensor_t* a = kf_tensor_create(ctx, &desc);
kf_tensor_t* b = kf_tensor_create(ctx, &desc);
kf_tensor_t* out = kf_tensor_create(ctx, &desc);

kf_fused_elementwise_add_activation(ctx, a, b, out, KF_ACTIVATION_RELU, stream);
```

### PyTorch (Frontend)
```python
import kernel_fusion_pytorch as kf

# Direct function calls
output = kf.fused_elementwise_add_relu(tensor_a, tensor_b)

# Or with explicit context/stream management
context = kf.Context(device_id=0)
stream = kf.Stream(context, priority=1)
output = kf.fused_elementwise_add_relu(tensor_a, tensor_b, stream=stream)
```

### TensorFlow (Planned)
```python
import kernel_fusion_tensorflow as kf

# Custom op registration
@tf.function
def my_model(x, y):
    return kf.fused_add_relu(x, y)
```

## Benefits of This Architecture

### ✅ **Framework Independence**
- Core library has zero framework dependencies
- Easy to add support for new frameworks
- Consistent performance across all frontends

### ✅ **Memory Efficiency**
- Zero-copy tensor sharing
- Framework memory managers handle allocation
- Efficient memory pooling for internal operations

### ✅ **Performance**
- Direct CUDA kernel execution
- Multi-stream parallelism
- Optimized for each data type

### ✅ **Maintainability**
- Clear separation of concerns
- Framework-specific code isolated in frontends
- Core algorithms implemented once

### ✅ **Extensibility**
- Easy to add new operations to core
- Simple frontend development
- Plugin architecture for specialized use cases

## Migration Path

### From Current PyTorch-only Implementation
1. **Phase 1**: Extract core CUDA kernels → `core/`
2. **Phase 2**: Create C API wrapper → `core/include/`
3. **Phase 3**: Implement PyTorch frontend → `frontends/pytorch/`
4. **Phase 4**: Add other frontends as needed

### Backward Compatibility
- Existing PyTorch code continues to work
- Performance improvements are transparent
- Additional frontends add capability without breaking changes

This architecture provides the foundation for a truly scalable and framework-agnostic kernel fusion library!
