#include "kernel_fusion/core.h"
#include "internal.hpp"
#include <cstring>
#include <iostream>

using namespace kf::core;

// Global debug flag
static bool g_debug_mode = false;

// ============================================================================
// Context Management
// ============================================================================

Context::Context(int device_id) : device_id(device_id), last_error(KF_SUCCESS) {
    if (device_id >= 0) {
        device_type = KF_DEVICE_CUDA;
    } else {
        device_type = KF_DEVICE_CPU;
    }
}

Context::~Context() {
    // Resources are automatically cleaned up by unique_ptr destructors
}

kf_result_t Context::initialize() {
    if (device_type == KF_DEVICE_CUDA) {
        // Set device
        KF_CHECK_CUDA(cudaSetDevice(device_id));
        
        // Get device properties
        cudaDeviceProp prop;
        KF_CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
        
        device_info.device_id = device_id;
        device_info.device_type = KF_DEVICE_CUDA;
        device_info.total_memory = prop.totalGlobalMem;
        device_info.compute_capability_major = prop.major;
        device_info.compute_capability_minor = prop.minor;
        
        // Get memory info
        size_t free_mem, total_mem;
        KF_CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
        device_info.free_memory = free_mem;
        
        // Create memory pool
        memory_pool = std::make_unique<MemoryPool>(KF_DEVICE_CUDA, device_id);
    } else {
        // CPU device
        device_info.device_id = -1;
        device_info.device_type = KF_DEVICE_CPU;
        device_info.total_memory = 0; // Would need platform-specific code
        device_info.free_memory = 0;
        device_info.compute_capability_major = 0;
        device_info.compute_capability_minor = 0;
        
        memory_pool = std::make_unique<MemoryPool>(KF_DEVICE_CPU, -1);
    }
    
    return KF_SUCCESS;
}

void Context::set_error(kf_result_t error, const std::string& message) {
    last_error = error;
    last_error_message = message;
    if (g_debug_mode) {
        std::cerr << "KernelFusion Error: " << message << std::endl;
    }
}

extern "C" {

kf_context_t* kf_context_create(int device_id) {
    auto context = std::make_unique<Context>(device_id);
    if (context->initialize() != KF_SUCCESS) {
        return nullptr;
    }
    return reinterpret_cast<kf_context_t*>(context.release());
}

void kf_context_destroy(kf_context_t* context) {
    if (context) {
        delete reinterpret_cast<Context*>(context);
    }
}

kf_result_t kf_context_get_device_info(kf_context_t* context, kf_device_info_t* info) {
    KF_CHECK_ARG(context && info);
    
    auto ctx = reinterpret_cast<Context*>(context);
    *info = ctx->device_info;
    return KF_SUCCESS;
}

// ============================================================================
// Stream Management  
// ============================================================================

kf_stream_t* kf_stream_create(kf_context_t* context, const kf_stream_config_t* config) {
    if (!context || !config) return nullptr;
    
    auto ctx = reinterpret_cast<Context*>(context);
    
    try {
        auto stream = std::make_unique<Stream>(*config);
        auto* stream_ptr = stream.get();
        ctx->streams.push_back(std::move(stream));
        return reinterpret_cast<kf_stream_t*>(stream_ptr);
    } catch (const std::exception& e) {
        ctx->set_error(KF_ERROR_CUDA_ERROR, e.what());
        return nullptr;
    }
}

void kf_stream_destroy(kf_stream_t* stream) {
    // Stream will be destroyed when context is destroyed
    // Individual stream destruction would require removing from context's vector
}

kf_result_t kf_stream_synchronize(kf_stream_t* stream) {
    KF_CHECK_ARG(stream);
    
    auto s = reinterpret_cast<Stream*>(stream);
    KF_CHECK_CUDA(cudaStreamSynchronize(s->cuda_stream));
    return KF_SUCCESS;
}

// ============================================================================
// Memory Management
// ============================================================================

kf_result_t kf_memory_alloc(kf_context_t* context, size_t size_bytes, kf_memory_desc_t* memory_desc) {
    KF_CHECK_ARG(context && memory_desc && size_bytes > 0);
    
    auto ctx = reinterpret_cast<Context*>(context);
    
    void* ptr = nullptr;
    if (ctx->device_type == KF_DEVICE_CUDA) {
        KF_CHECK_CUDA(cudaSetDevice(ctx->device_id));
        KF_CHECK_CUDA(cudaMalloc(&ptr, size_bytes));
    } else {
        ptr = malloc(size_bytes);
        if (!ptr) return KF_ERROR_OUT_OF_MEMORY;
    }
    
    memory_desc->data = ptr;
    memory_desc->size_bytes = size_bytes;
    memory_desc->device_type = ctx->device_type;
    memory_desc->device_id = ctx->device_id;
    memory_desc->owns_memory = true;
    
    return KF_SUCCESS;
}

void kf_memory_free(kf_memory_desc_t* memory_desc) {
    if (!memory_desc || !memory_desc->data || !memory_desc->owns_memory) {
        return;
    }
    
    if (memory_desc->device_type == KF_DEVICE_CUDA) {
        cudaFree(memory_desc->data);
    } else {
        free(memory_desc->data);
    }
    
    memory_desc->data = nullptr;
    memory_desc->owns_memory = false;
}

kf_result_t kf_memory_copy(const kf_memory_desc_t* src, kf_memory_desc_t* dst, size_t size_bytes) {
    KF_CHECK_ARG(src && dst && src->data && dst->data);
    
    cudaMemcpyKind kind;
    
    if (src->device_type == KF_DEVICE_CPU && dst->device_type == KF_DEVICE_CPU) {
        memcpy(dst->data, src->data, size_bytes);
        return KF_SUCCESS;
    } else if (src->device_type == KF_DEVICE_CPU && dst->device_type == KF_DEVICE_CUDA) {
        kind = cudaMemcpyHostToDevice;
    } else if (src->device_type == KF_DEVICE_CUDA && dst->device_type == KF_DEVICE_CPU) {
        kind = cudaMemcpyDeviceToHost;
    } else {
        kind = cudaMemcpyDeviceToDevice;
    }
    
    KF_CHECK_CUDA(cudaMemcpy(dst->data, src->data, size_bytes, kind));
    return KF_SUCCESS;
}

// ============================================================================
// Utility Functions
// ============================================================================

const char* kf_get_version(void) {
    return "1.0.0-standalone";
}

const char* kf_get_error_string(kf_result_t result) {
    switch (result) {
        case KF_SUCCESS: return "Success";
        case KF_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case KF_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case KF_ERROR_CUDA_ERROR: return "CUDA error";
        case KF_ERROR_UNSUPPORTED_DTYPE: return "Unsupported data type";
        case KF_ERROR_DIMENSION_MISMATCH: return "Dimension mismatch";
        case KF_ERROR_INVALID_DEVICE: return "Invalid device";
        default: return "Unknown error";
    }
}

void kf_set_debug_mode(bool enabled) {
    g_debug_mode = enabled;
}

} // extern "C"
