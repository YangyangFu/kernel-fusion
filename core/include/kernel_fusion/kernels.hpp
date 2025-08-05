#pragma once

/**
 * Main header for KernelFusion kernels namespace
 * 
 * This header provides access to all kernel implementations organized
 * in the kf::kernels namespace hierarchy:
 * 
 * - kf::kernels::elementwise    - Element-wise operations
 * - kf::kernels::linear         - Linear algebra operations  
 * - kf::kernels::normalization  - Normalization operations
 * - kf::kernels::convolution    - Convolution operations
 * - kf::kernels::attention      - Attention mechanisms
 */

#include "kernels/kernels.hpp"

namespace kf {
namespace kernels {

// ============================================================================
// Kernel Namespace Documentation
// ============================================================================

/**
 * kf::kernels namespace contains all CUDA kernel implementations
 * organized by operation type:
 * 
 * Elementwise Operations:
 * - add_activation_kernel    - Elementwise addition + activation
 * - mul_activation_kernel    - Elementwise multiplication + activation  
 * - bias_activation_kernel   - Bias addition + activation
 * 
 * Linear Operations:
 * - fused_linear_activation_kernel     - Linear layer + activation
 * - optimized_linear_activation_kernel - Optimized GEMM + activation
 * 
 * Normalization Operations:
 * - layer_norm_activation_kernel - Layer normalization + activation
 * - batch_norm_activation_kernel - Batch normalization + activation
 * 
 * Utility Functions:
 * - apply_activation    - Device function for activation application
 * - get_launch_config   - Optimal kernel launch configuration
 * - relu, gelu, silu    - Individual activation functions
 */

// Re-export commonly used functions for convenience
using elementwise::add_activation_kernel;
using elementwise::mul_activation_kernel;
using elementwise::bias_activation_kernel;

using linear::fused_linear_activation_kernel;
using linear::optimized_linear_activation_kernel;

using normalization::layer_norm_activation_kernel;
using normalization::batch_norm_activation_kernel;

// Utility functions
using ::kf::kernels::get_launch_config;
using ::kf::kernels::get_launch_config_2d;
using ::kf::kernels::apply_activation;

} // namespace kernels
} // namespace kf
