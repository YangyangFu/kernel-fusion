#include "kernel_fusion/kernels/kernels.hpp"

namespace kf {
namespace kernels {

// ============================================================================
// Activation Function Implementation
// ============================================================================

template<typename T>
__device__ __forceinline__ T apply_activation(T x, kf_activation_t activation) {
    switch (activation) {
        case KF_ACTIVATION_NONE:
            return x;
        case KF_ACTIVATION_RELU:
            return relu(x);
        case KF_ACTIVATION_GELU:
            return gelu(x);
        case KF_ACTIVATION_SILU:
            return silu(x);
        case KF_ACTIVATION_SIGMOID:
            return sigmoid(x);
        default:
            return x;
    }
}

// ============================================================================
// Launch Configuration Implementation
// ============================================================================

LaunchConfig::LaunchConfig(int64_t total_elements, cudaStream_t stream) {
    this->stream = stream;
    
    // Common block size
    block_size = dim3(256, 1, 1);
    
    // Calculate grid size
    int grid_x = (total_elements + block_size.x - 1) / block_size.x;
    grid_size = dim3(min(grid_x, 65535), 1, 1);  // Limit to device capability
    
    shared_memory = 0;
}

LaunchConfig get_launch_config(int64_t total_elements, cudaStream_t stream) {
    return LaunchConfig(total_elements, stream);
}

LaunchConfig get_launch_config_2d(int64_t rows, int64_t cols, cudaStream_t stream) {
    LaunchConfig config(rows * cols, stream);
    
    // For 2D, try to balance block dimensions
    if (cols >= 32) {
        config.block_size = dim3(32, 8, 1);
        config.grid_size = dim3(
            (cols + 31) / 32,
            (rows + 7) / 8,
            1
        );
    } else {
        config.block_size = dim3(cols, min(256 / (int)cols, (int)rows), 1);
        config.grid_size = dim3(
            1,
            (rows + config.block_size.y - 1) / config.block_size.y,
            1
        );
    }
    
    return config;
}

// Explicit template instantiations for apply_activation
template __device__ float apply_activation<float>(float, kf_activation_t);
template __device__ double apply_activation<double>(double, kf_activation_t);

} // namespace kernels
} // namespace kf
