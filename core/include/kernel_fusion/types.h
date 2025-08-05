#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct kf_tensor kf_tensor_t;
typedef struct kf_context kf_context_t;
typedef struct kf_stream kf_stream_t;

// Data types
typedef enum {
    KF_DTYPE_FLOAT32 = 0,
    KF_DTYPE_FLOAT16 = 1,
    KF_DTYPE_BFLOAT16 = 2,
    KF_DTYPE_INT32 = 3,
    KF_DTYPE_INT64 = 4,
    KF_DTYPE_UINT8 = 5
} kf_dtype_t;

// Device types
typedef enum {
    KF_DEVICE_CPU = 0,
    KF_DEVICE_CUDA = 1
} kf_device_type_t;

// Memory layout
typedef enum {
    KF_LAYOUT_CONTIGUOUS = 0,
    KF_LAYOUT_STRIDED = 1
} kf_layout_t;

// Result codes
typedef enum {
    KF_SUCCESS = 0,
    KF_ERROR_INVALID_ARGUMENT = 1,
    KF_ERROR_OUT_OF_MEMORY = 2,
    KF_ERROR_CUDA_ERROR = 3,
    KF_ERROR_UNSUPPORTED_DTYPE = 4,
    KF_ERROR_DIMENSION_MISMATCH = 5,
    KF_ERROR_INVALID_DEVICE = 6
} kf_result_t;

// Device information
typedef struct {
    int device_id;
    kf_device_type_t device_type;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
} kf_device_info_t;

// Tensor descriptor
typedef struct {
    int64_t* shape;
    int64_t* strides;  
    int ndim;
    kf_dtype_t dtype;
    kf_device_type_t device_type;
    int device_id;
    kf_layout_t layout;
} kf_tensor_desc_t;

// Memory descriptor
typedef struct {
    void* data;
    size_t size_bytes;
    kf_device_type_t device_type;
    int device_id;
    bool owns_memory;  // Whether this descriptor owns the memory
} kf_memory_desc_t;

// Stream configuration
typedef struct {
    int priority;          // Stream priority (higher = more priority)
    unsigned int flags;    // Stream flags (e.g., non-blocking)
    int device_id;        // Target device
} kf_stream_config_t;

// Kernel launch configuration
typedef struct {
    int block_size_x;
    int block_size_y; 
    int block_size_z;
    int grid_size_x;
    int grid_size_y;
    int grid_size_z;
    int shared_memory_bytes;
    kf_stream_t* stream;
} kf_launch_config_t;

// Activation types for fused operations
typedef enum {
    KF_ACTIVATION_NONE = 0,
    KF_ACTIVATION_RELU = 1,
    KF_ACTIVATION_GELU = 2,
    KF_ACTIVATION_SILU = 3,
    KF_ACTIVATION_TANH = 4,
    KF_ACTIVATION_SIGMOID = 5
} kf_activation_t;

#ifdef __cplusplus
}
#endif
