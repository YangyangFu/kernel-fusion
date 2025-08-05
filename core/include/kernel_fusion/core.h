#pragma once

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Context Management
// ============================================================================

/**
 * Create a new kernel fusion context
 * @param device_id CUDA device ID (-1 for CPU)
 * @return Context handle or NULL on error
 */
kf_context_t* kf_context_create(int device_id);

/**
 * Destroy a context and free associated resources
 */
void kf_context_destroy(kf_context_t* context);

/**
 * Get device information for a context
 */
kf_result_t kf_context_get_device_info(kf_context_t* context, kf_device_info_t* info);

// ============================================================================
// Stream Management
// ============================================================================

/**
 * Create a CUDA stream with specified configuration
 */
kf_stream_t* kf_stream_create(kf_context_t* context, const kf_stream_config_t* config);

/**
 * Destroy a stream
 */
void kf_stream_destroy(kf_stream_t* stream);

/**
 * Synchronize a stream (wait for all operations to complete)
 */
kf_result_t kf_stream_synchronize(kf_stream_t* stream);

// ============================================================================
// Memory Management
// ============================================================================

/**
 * Allocate device memory
 */
kf_result_t kf_memory_alloc(kf_context_t* context, size_t size_bytes, kf_memory_desc_t* memory_desc);

/**
 * Free device memory
 */
void kf_memory_free(kf_memory_desc_t* memory_desc);

/**
 * Copy memory between host and device
 */
kf_result_t kf_memory_copy(const kf_memory_desc_t* src, kf_memory_desc_t* dst, size_t size_bytes);

// ============================================================================
// Tensor Operations
// ============================================================================

/**
 * Create a tensor from existing memory
 */
kf_tensor_t* kf_tensor_create_from_memory(
    const kf_memory_desc_t* memory,
    const kf_tensor_desc_t* tensor_desc
);

/**
 * Create a new tensor with allocated memory
 */
kf_tensor_t* kf_tensor_create(
    kf_context_t* context,
    const kf_tensor_desc_t* tensor_desc
);

/**
 * Destroy a tensor
 */
void kf_tensor_destroy(kf_tensor_t* tensor);

/**
 * Get tensor descriptor
 */
kf_result_t kf_tensor_get_desc(const kf_tensor_t* tensor, kf_tensor_desc_t* desc);

/**
 * Get tensor memory descriptor
 */
kf_result_t kf_tensor_get_memory(const kf_tensor_t* tensor, kf_memory_desc_t* memory);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get optimal launch configuration for a kernel
 */
kf_result_t kf_get_launch_config(
    kf_context_t* context,
    int64_t total_elements,
    kf_launch_config_t* config
);

/**
 * Get library version information
 */
const char* kf_get_version(void);

/**
 * Get last error message
 */
const char* kf_get_error_string(kf_result_t result);

/**
 * Enable/disable debug mode
 */
void kf_set_debug_mode(bool enabled);

#ifdef __cplusplus
}
#endif
