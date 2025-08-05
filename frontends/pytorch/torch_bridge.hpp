#pragma once

#include <torch/extension.h>
#include "kernel_fusion/core.h"
#include "kernel_fusion/types.h"

namespace kf {
namespace pytorch {

/**
 * Convert PyTorch tensor to KernelFusion tensor descriptor
 */
kf_tensor_desc_t tensor_desc_from_torch(const torch::Tensor& tensor);

/**
 * Convert PyTorch dtype to KernelFusion dtype
 */
kf_dtype_t dtype_from_torch(torch::ScalarType scalar_type);

/**
 * Convert PyTorch tensor to KernelFusion memory descriptor
 */
kf_memory_desc_t memory_desc_from_torch(const torch::Tensor& tensor);

/**
 * Create KernelFusion tensor from PyTorch tensor (shares memory)
 */
kf_tensor_t* tensor_from_torch(kf_context_t* context, const torch::Tensor& tensor);

/**
 * Create PyTorch tensor from KernelFusion tensor (shares memory)
 */
torch::Tensor tensor_to_torch(const kf_tensor_t* tensor);

/**
 * RAII wrapper for KernelFusion context
 */
class Context {
private:
    kf_context_t* context_;
    
public:
    explicit Context(int device_id = -1) {
        if (device_id == -1 && torch::cuda::is_available()) {
            device_id = torch::cuda::current_device();
        }
        context_ = kf_context_create(device_id);
        if (!context_) {
            throw std::runtime_error("Failed to create KernelFusion context");
        }
    }
    
    ~Context() {
        if (context_) {
            kf_context_destroy(context_);
        }
    }
    
    // No copy, only move
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    
    Context(Context&& other) noexcept : context_(other.context_) {
        other.context_ = nullptr;
    }
    
    Context& operator=(Context&& other) noexcept {
        if (this != &other) {
            if (context_) {
                kf_context_destroy(context_);
            }
            context_ = other.context_;
            other.context_ = nullptr;
        }
        return *this;
    }
    
    kf_context_t* get() const { return context_; }
    operator kf_context_t*() const { return context_; }
    
    kf_device_info_t get_device_info() const {
        kf_device_info_t info;
        kf_result_t result = kf_context_get_device_info(context_, &info);
        if (result != KF_SUCCESS) {
            throw std::runtime_error("Failed to get device info: " + std::string(kf_get_error_string(result)));
        }
        return info;
    }
};

/**
 * RAII wrapper for KernelFusion stream
 */
class Stream {
private:
    kf_stream_t* stream_;
    
public:
    Stream(Context& context, int priority = 0, unsigned int flags = 0) {
        kf_stream_config_t config;
        config.priority = priority;
        config.flags = flags;
        config.device_id = context.get_device_info().device_id;
        
        stream_ = kf_stream_create(context.get(), &config);
        if (!stream_) {
            throw std::runtime_error("Failed to create KernelFusion stream");
        }
    }
    
    ~Stream() {
        if (stream_) {
            kf_stream_destroy(stream_);
        }
    }
    
    // No copy, only move
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    
    Stream(Stream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                kf_stream_destroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    
    kf_stream_t* get() const { return stream_; }
    operator kf_stream_t*() const { return stream_; }
    
    void synchronize() {
        kf_result_t result = kf_stream_synchronize(stream_);
        if (result != KF_SUCCESS) {
            throw std::runtime_error("Failed to synchronize stream: " + std::string(kf_get_error_string(result)));
        }
    }
};

/**
 * Exception wrapper for KernelFusion errors
 */
class KernelFusionError : public std::runtime_error {
public:
    explicit KernelFusionError(kf_result_t result) 
        : std::runtime_error(kf_get_error_string(result)), result_(result) {}
    
    kf_result_t result() const { return result_; }
    
private:
    kf_result_t result_;
};

#define KF_TORCH_CHECK(result) \
    do { \
        kf_result_t _result = (result); \
        if (_result != KF_SUCCESS) { \
            throw KernelFusionError(_result); \
        } \
    } while(0)

} // namespace pytorch
} // namespace kf
