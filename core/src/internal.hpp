#pragma once

#include "kernel_fusion/types.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <unordered_map>

namespace kf {
namespace core {

// Internal tensor implementation
struct Tensor {
    kf_tensor_desc_t desc;
    kf_memory_desc_t memory;
    
    Tensor(const kf_tensor_desc_t& desc, const kf_memory_desc_t& memory)
        : desc(desc), memory(memory) {}
    
    ~Tensor() {
        if (memory.owns_memory && memory.data) {
            if (memory.device_type == KF_DEVICE_CUDA) {
                cudaFree(memory.data);
            } else {
                free(memory.data);
            }
        }
        if (desc.shape) free(desc.shape);
        if (desc.strides) free(desc.strides);
    }
    
    size_t element_count() const {
        size_t count = 1;
        for (int i = 0; i < desc.ndim; ++i) {
            count *= desc.shape[i];
        }
        return count;
    }
    
    size_t element_size() const {
        switch (desc.dtype) {
            case KF_DTYPE_FLOAT32: return 4;
            case KF_DTYPE_FLOAT16: return 2;
            case KF_DTYPE_BFLOAT16: return 2;
            case KF_DTYPE_INT32: return 4;
            case KF_DTYPE_INT64: return 8;
            case KF_DTYPE_UINT8: return 1;
            default: return 0;
        }
    }
    
    size_t size_bytes() const {
        return element_count() * element_size();
    }
};

// Internal stream implementation
struct Stream {
    cudaStream_t cuda_stream;
    kf_stream_config_t config;
    int device_id;
    
    Stream(const kf_stream_config_t& config) : config(config) {
        cudaSetDevice(config.device_id);
        cudaStreamCreateWithPriority(&cuda_stream, config.flags, config.priority);
        device_id = config.device_id;
    }
    
    ~Stream() {
        if (cuda_stream) {
            cudaStreamDestroy(cuda_stream);
        }
    }
};

// Memory pool for efficient allocation
class MemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<MemoryBlock> blocks_;
    kf_device_type_t device_type_;
    int device_id_;
    
public:
    MemoryPool(kf_device_type_t device_type, int device_id)
        : device_type_(device_type), device_id_(device_id) {}
    
    ~MemoryPool() {
        for (auto& block : blocks_) {
            if (device_type_ == KF_DEVICE_CUDA) {
                cudaFree(block.ptr);
            } else {
                free(block.ptr);
            }
        }
    }
    
    void* allocate(size_t size);
    void deallocate(void* ptr);
    void clear();
};

// Internal context implementation
struct Context {
    int device_id;
    kf_device_type_t device_type;
    kf_device_info_t device_info;
    
    // Resource management
    std::unique_ptr<MemoryPool> memory_pool;
    std::vector<std::unique_ptr<Stream>> streams;
    std::vector<std::unique_ptr<Tensor>> tensors;
    
    // Error tracking
    kf_result_t last_error;
    std::string last_error_message;
    
    Context(int device_id);
    ~Context();
    
    kf_result_t initialize();
    Stream* create_stream(const kf_stream_config_t& config);
    Tensor* create_tensor(const kf_tensor_desc_t& desc);
    void set_error(kf_result_t error, const std::string& message);
};

// Kernel launch utilities
struct LaunchConfig {
    dim3 block_size;
    dim3 grid_size;
    int shared_memory;
    cudaStream_t stream;
    
    LaunchConfig(int64_t total_elements, cudaStream_t stream = nullptr);
};

// Type conversion utilities
template<typename T>
struct DTypeTraits {};

template<> struct DTypeTraits<float> { 
    static constexpr kf_dtype_t dtype = KF_DTYPE_FLOAT32; 
};

template<> struct DTypeTraits<int32_t> { 
    static constexpr kf_dtype_t dtype = KF_DTYPE_INT32; 
};

template<> struct DTypeTraits<int64_t> { 
    static constexpr kf_dtype_t dtype = KF_DTYPE_INT64; 
};

// Activation function helpers
template<typename T>
__device__ __forceinline__ T apply_activation(T x, kf_activation_t activation) {
    switch (activation) {
        case KF_ACTIVATION_NONE: return x;
        case KF_ACTIVATION_RELU: return fmaxf(x, T(0));
        case KF_ACTIVATION_GELU: {
            // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            T x3 = x * x * x;
            T inner = T(0.7978845608) * (x + T(0.044715) * x3);
            return T(0.5) * x * (T(1) + tanhf(inner));
        }
        case KF_ACTIVATION_SILU: return x / (T(1) + expf(-x));
        case KF_ACTIVATION_TANH: return tanhf(x);
        case KF_ACTIVATION_SIGMOID: return T(1) / (T(1) + expf(-x));
        default: return x;
    }
}

// Error handling utilities
#define KF_CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            return KF_ERROR_CUDA_ERROR; \
        } \
    } while(0)

#define KF_CHECK_ARG(condition) \
    do { \
        if (!(condition)) { \
            return KF_ERROR_INVALID_ARGUMENT; \
        } \
    } while(0)

} // namespace core
} // namespace kf
