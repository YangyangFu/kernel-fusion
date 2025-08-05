#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "kernel_fusion/core.h"
#include "kernel_fusion/types.h"

int main() {
    printf("=== KernelFusion Standalone C API Demo ===\n");
    
    // 1. Create context
    printf("\n1. Creating CUDA context...\n");
    kf_context_t* context = kf_context_create(0);  // Use GPU 0
    if (!context) {
        printf("Failed to create context. Trying CPU...\n");
        context = kf_context_create(-1);  // Use CPU
        if (!context) {
            printf("Failed to create CPU context. Exiting.\n");
            return 1;
        }
    }
    
    // Get device info
    kf_device_info_t device_info;
    kf_result_t result = kf_context_get_device_info(context, &device_info);
    if (result == KF_SUCCESS) {
        printf("Device ID: %d\n", device_info.device_id);
        printf("Device type: %s\n", device_info.device_type == KF_DEVICE_CUDA ? "CUDA" : "CPU");
        if (device_info.device_type == KF_DEVICE_CUDA) {
            printf("Total memory: %.1f GB\n", device_info.total_memory / (1024.0 * 1024.0 * 1024.0));
            printf("Free memory: %.1f GB\n", device_info.free_memory / (1024.0 * 1024.0 * 1024.0));
            printf("Compute capability: %d.%d\n", 
                   device_info.compute_capability_major, 
                   device_info.compute_capability_minor);
        }
    }
    
    // 2. Create stream
    printf("\n2. Creating stream...\n");
    kf_stream_config_t stream_config = {
        .priority = 0,
        .flags = 0,
        .device_id = device_info.device_id
    };
    kf_stream_t* stream = kf_stream_create(context, &stream_config);
    if (!stream) {
        printf("Failed to create stream\n");
        kf_context_destroy(context);
        return 1;
    }
    
    // 3. Allocate memory for tensors
    printf("\n3. Allocating tensors...\n");
    const int batch_size = 1024;
    const int features = 512;
    const size_t tensor_size = batch_size * features * sizeof(float);
    
    kf_memory_desc_t mem_a, mem_b, mem_output;
    
    result = kf_memory_alloc(context, tensor_size, &mem_a);
    assert(result == KF_SUCCESS);
    
    result = kf_memory_alloc(context, tensor_size, &mem_b);
    assert(result == KF_SUCCESS);
    
    result = kf_memory_alloc(context, tensor_size, &mem_output);
    assert(result == KF_SUCCESS);
    
    printf("Allocated 3 tensors of size %zu bytes each\n", tensor_size);
    
    // 4. Create tensor descriptors
    printf("\n4. Creating tensor descriptors...\n");
    int64_t shape[2] = {batch_size, features};
    int64_t strides[2] = {features, 1};  // Row-major layout
    
    kf_tensor_desc_t tensor_desc = {
        .shape = shape,
        .strides = strides,
        .ndim = 2,
        .dtype = KF_DTYPE_FLOAT32,
        .device_type = device_info.device_type,
        .device_id = device_info.device_id,
        .layout = KF_LAYOUT_CONTIGUOUS
    };
    
    // Create tensors
    kf_tensor_t* tensor_a = kf_tensor_create_from_memory(&mem_a, &tensor_desc);
    kf_tensor_t* tensor_b = kf_tensor_create_from_memory(&mem_b, &tensor_desc);
    kf_tensor_t* tensor_output = kf_tensor_create_from_memory(&mem_output, &tensor_desc);
    
    assert(tensor_a && tensor_b && tensor_output);
    printf("Created tensor descriptors: [%ld, %ld]\n", shape[0], shape[1]);
    
    // 5. Initialize input data (if on CPU)
    if (device_info.device_type == KF_DEVICE_CPU) {
        printf("\n5. Initializing input data...\n");
        float* data_a = (float*)mem_a.data;
        float* data_b = (float*)mem_b.data;
        
        for (int i = 0; i < batch_size * features; ++i) {
            data_a[i] = 0.5f * (float)i / (batch_size * features);
            data_b[i] = 0.3f * (float)(i + 1) / (batch_size * features);
        }
        printf("Initialized input tensors with test data\n");
    } else {
        printf("\n5. Skipping data initialization (GPU tensors)\n");
    }
    
    // 6. Perform fused operation
    printf("\n6. Performing fused elementwise add + ReLU...\n");
    result = kf_fused_elementwise_add_activation(
        context,
        tensor_a,
        tensor_b,
        tensor_output,
        KF_ACTIVATION_RELU,
        stream
    );
    
    if (result == KF_SUCCESS) {
        printf("✓ Fused operation completed successfully!\n");
        
        // Synchronize stream
        result = kf_stream_synchronize(stream);
        if (result == KF_SUCCESS) {
            printf("✓ Stream synchronized\n");
        }
        
        // Check a few output values (if on CPU)
        if (device_info.device_type == KF_DEVICE_CPU) {
            float* output_data = (float*)mem_output.data;
            printf("Sample outputs: %.4f, %.4f, %.4f, %.4f\n",
                   output_data[0], output_data[1], output_data[2], output_data[3]);
        }
    } else {
        printf("✗ Fused operation failed: %s\n", kf_get_error_string(result));
    }
    
    // 7. Test layer normalization + GELU (if supported)
    printf("\n7. Testing layer normalization + GELU...\n");
    
    // Create weight and bias for layer norm
    const int norm_size = features;
    const size_t norm_tensor_size = norm_size * sizeof(float);
    
    kf_memory_desc_t mem_weight, mem_bias;
    result = kf_memory_alloc(context, norm_tensor_size, &mem_weight);
    if (result == KF_SUCCESS) {
        result = kf_memory_alloc(context, norm_tensor_size, &mem_bias);
    }
    
    if (result == KF_SUCCESS) {
        int64_t norm_shape[1] = {norm_size};
        int64_t norm_strides[1] = {1};
        
        kf_tensor_desc_t norm_desc = {
            .shape = norm_shape,
            .strides = norm_strides,
            .ndim = 1,
            .dtype = KF_DTYPE_FLOAT32,
            .device_type = device_info.device_type,
            .device_id = device_info.device_id,
            .layout = KF_LAYOUT_CONTIGUOUS
        };
        
        kf_tensor_t* weight = kf_tensor_create_from_memory(&mem_weight, &norm_desc);
        kf_tensor_t* bias = kf_tensor_create_from_memory(&mem_bias, &norm_desc);
        
        if (weight && bias) {
            // Initialize weights (if on CPU)
            if (device_info.device_type == KF_DEVICE_CPU) {
                float* weight_data = (float*)mem_weight.data;
                float* bias_data = (float*)mem_bias.data;
                for (int i = 0; i < norm_size; ++i) {
                    weight_data[i] = 1.0f;
                    bias_data[i] = 0.0f;
                }
            }
            
            result = kf_fused_layer_norm_activation(
                context,
                tensor_a,  // Use tensor_a as input
                weight,
                bias,
                tensor_output,
                KF_ACTIVATION_GELU,
                1e-5f,     // eps
                stream
            );
            
            if (result == KF_SUCCESS) {
                printf("✓ Layer norm + GELU completed successfully!\n");
                kf_stream_synchronize(stream);
            } else {
                printf("✗ Layer norm + GELU failed: %s\n", kf_get_error_string(result));
            }
            
            kf_tensor_destroy(weight);
            kf_tensor_destroy(bias);
        }
        
        kf_memory_free(&mem_weight);
        kf_memory_free(&mem_bias);
    } else {
        printf("Skipping layer norm test (memory allocation failed)\n");
    }
    
    // 8. Cleanup
    printf("\n8. Cleaning up...\n");
    kf_tensor_destroy(tensor_a);
    kf_tensor_destroy(tensor_b);
    kf_tensor_destroy(tensor_output);
    
    kf_memory_free(&mem_a);
    kf_memory_free(&mem_b);
    kf_memory_free(&mem_output);
    
    kf_stream_destroy(stream);
    kf_context_destroy(context);
    
    printf("✓ Cleanup completed\n");
    
    printf("\n=== Demo completed successfully! ===\n");
    printf("Library version: %s\n", kf_get_version());
    
    return 0;
}
