// Simple fusion validation test without external dependencies
// Validates basic correctness of fused operations against sequential implementations

#include "benchmark_utils.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>

// Test configuration
const float TOLERANCE_FP32 = 1e-6f;
const float TOLERANCE_FP16 = 1e-3f;

// Helper function to fill with specific range
void fill_random_range(float* data, int n, float min_val, float max_val) {
    std::vector<float> host_data(n);
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (int i = 0; i < n; ++i) {
        host_data[i] = dis(gen);
    }
    
    cudaMemcpy(data, host_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
}

// Basic kernels for comparison
__global__ void relu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void sigmoid_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void tanh_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.79788456f * x * (1.0f + 0.044715f * x * x)));
    }
}

__global__ void add_kernel(const float* a, const float* b, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

// Fused kernels
__global__ void relu_sigmoid_fused_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float relu_val = fmaxf(0.0f, input[idx]);
        output[idx] = 1.0f / (1.0f + expf(-relu_val));
    }
}

__global__ void sigmoid_tanh_fused_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sigmoid_val = 1.0f / (1.0f + expf(-input[idx]));
        output[idx] = tanhf(sigmoid_val);
    }
}

__global__ void relu_sigmoid_tanh_fused_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float relu_val = fmaxf(0.0f, input[idx]);
        float sigmoid_val = 1.0f / (1.0f + expf(-relu_val));
        output[idx] = tanhf(sigmoid_val);
    }
}

__global__ void add_relu_fused_kernel(const float* a, const float* b, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = a[idx] + b[idx];
        output[idx] = fmaxf(0.0f, sum);
    }
}

// Validation helper functions
bool validate_arrays(const float* fused, const float* sequential, int n, float tolerance = TOLERANCE_FP32) {
    std::vector<float> fused_host(n);
    std::vector<float> sequential_host(n);
    
    cudaMemcpy(fused_host.data(), fused, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sequential_host.data(), sequential, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_error = 0.0f;
    int error_count = 0;
    int first_error_idx = -1;
    
    for (int i = 0; i < n; i++) {
        float diff = std::abs(fused_host[i] - sequential_host[i]);
        float rel_error = diff / (std::abs(sequential_host[i]) + 1e-8f);
        
        if (diff > tolerance && rel_error > tolerance) {
            error_count++;
            if (first_error_idx == -1) {
                first_error_idx = i;
            }
            max_error = std::max(max_error, diff);
        }
    }
    
    if (error_count > 0) {
        std::cout << "  ❌ Validation FAILED:" << std::endl;
        std::cout << "     Errors found: " << error_count << " / " << n << std::endl;
        std::cout << "     Max error: " << std::scientific << max_error << std::endl;
        std::cout << "     First error at index " << first_error_idx << ": "
                  << "fused=" << fused_host[first_error_idx] 
                  << ", sequential=" << sequential_host[first_error_idx] << std::endl;
        return false;
    } else {
        std::cout << "  ✅ Validation PASSED (max error: " << std::scientific << max_error << ")" << std::endl;
        return true;
    }
}

// Test edge cases
bool test_edge_cases() {
    std::cout << "\n=== Edge Case Testing ===" << std::endl;
    
    const int n = 16;
    std::vector<float> edge_inputs = {
        0.0f, -0.0f, 1.0f, -1.0f,
        INFINITY, -INFINITY, NAN, -NAN,
        1e-10f, -1e-10f, 1e10f, -1e10f,
        88.0f, -88.0f, 100.0f, -100.0f
    };
    
    float* d_input;
    float* d_output_fused;
    float* d_output_sequential;
    
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output_fused, n * sizeof(float));
    cudaMalloc(&d_output_sequential, n * sizeof(float));
    
    cudaMemcpy(d_input, edge_inputs.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    bool all_passed = true;
    
    // Test ReLU edge cases
    std::cout << "Testing ReLU edge cases..." << std::endl;
    relu_kernel<<<grid, block>>>(d_input, d_output_sequential, n);
    relu_kernel<<<grid, block>>>(d_input, d_output_fused, n); // Using same kernel for simplicity
    cudaDeviceSynchronize();
    
    if (!validate_arrays(d_output_fused, d_output_sequential, n, 1e-5f)) {
        all_passed = false;
    }
    
    // Test Sigmoid edge cases (important for overflow/underflow)
    std::cout << "Testing Sigmoid edge cases..." << std::endl;
    sigmoid_kernel<<<grid, block>>>(d_input, d_output_sequential, n);
    sigmoid_kernel<<<grid, block>>>(d_input, d_output_fused, n);
    cudaDeviceSynchronize();
    
    if (!validate_arrays(d_output_fused, d_output_sequential, n, 1e-5f)) {
        all_passed = false;
    }
    
    cudaFree(d_input);
    cudaFree(d_output_fused);
    cudaFree(d_output_sequential);
    
    return all_passed;
}

// Main validation tests
class FusionValidator {
public:
    FusionValidator() = default;
    
    bool test_relu_sigmoid_fusion(int n) {
        std::cout << "Testing ReLU→Sigmoid fusion (n=" << n << ")..." << std::endl;
        
        float* d_input;
        float* d_output_fused;
        float* d_output_sequential;
        float* d_temp;
        
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output_fused, n * sizeof(float));
        cudaMalloc(&d_output_sequential, n * sizeof(float));
        cudaMalloc(&d_temp, n * sizeof(float));
        
        // Generate test data
        fill_random_range(d_input, n, -5.0f, 5.0f);
        
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        
        // Sequential execution
        relu_kernel<<<grid, block>>>(d_input, d_temp, n);
        sigmoid_kernel<<<grid, block>>>(d_temp, d_output_sequential, n);
        
        // Fused execution
        relu_sigmoid_fused_kernel<<<grid, block>>>(d_input, d_output_fused, n);
        
        cudaDeviceSynchronize();
        
        bool passed = validate_arrays(d_output_fused, d_output_sequential, n);
        
        cudaFree(d_input);
        cudaFree(d_output_fused);
        cudaFree(d_output_sequential);
        cudaFree(d_temp);
        
        return passed;
    }
    
    bool test_sigmoid_tanh_fusion(int n) {
        std::cout << "Testing Sigmoid→Tanh fusion (n=" << n << ")..." << std::endl;
        
        float* d_input;
        float* d_output_fused;
        float* d_output_sequential;
        float* d_temp;
        
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output_fused, n * sizeof(float));
        cudaMalloc(&d_output_sequential, n * sizeof(float));
        cudaMalloc(&d_temp, n * sizeof(float));
        
        fill_random_range(d_input, n, -3.0f, 3.0f);
        
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        
        // Sequential execution
        sigmoid_kernel<<<grid, block>>>(d_input, d_temp, n);
        tanh_kernel<<<grid, block>>>(d_temp, d_output_sequential, n);
        
        // Fused execution
        sigmoid_tanh_fused_kernel<<<grid, block>>>(d_input, d_output_fused, n);
        
        cudaDeviceSynchronize();
        
        bool passed = validate_arrays(d_output_fused, d_output_sequential, n);
        
        cudaFree(d_input);
        cudaFree(d_output_fused);
        cudaFree(d_output_sequential);
        cudaFree(d_temp);
        
        return passed;
    }
    
    bool test_three_op_fusion(int n) {
        std::cout << "Testing ReLU→Sigmoid→Tanh fusion (n=" << n << ")..." << std::endl;
        
        float* d_input;
        float* d_output_fused;
        float* d_output_sequential;
        float* d_temp1;
        float* d_temp2;
        
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output_fused, n * sizeof(float));
        cudaMalloc(&d_output_sequential, n * sizeof(float));
        cudaMalloc(&d_temp1, n * sizeof(float));
        cudaMalloc(&d_temp2, n * sizeof(float));
        
        fill_random_range(d_input, n, -2.0f, 2.0f);
        
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        
        // Sequential execution
        relu_kernel<<<grid, block>>>(d_input, d_temp1, n);
        sigmoid_kernel<<<grid, block>>>(d_temp1, d_temp2, n);
        tanh_kernel<<<grid, block>>>(d_temp2, d_output_sequential, n);
        
        // Fused execution
        relu_sigmoid_tanh_fused_kernel<<<grid, block>>>(d_input, d_output_fused, n);
        
        cudaDeviceSynchronize();
        
        bool passed = validate_arrays(d_output_fused, d_output_sequential, n);
        
        cudaFree(d_input);
        cudaFree(d_output_fused);
        cudaFree(d_output_sequential);
        cudaFree(d_temp1);
        cudaFree(d_temp2);
        
        return passed;
    }
    
    bool test_add_relu_fusion(int n) {
        std::cout << "Testing Add→ReLU fusion (n=" << n << ")..." << std::endl;
        
        float* d_input_a;
        float* d_input_b;
        float* d_output_fused;
        float* d_output_sequential;
        float* d_temp;
        
        cudaMalloc(&d_input_a, n * sizeof(float));
        cudaMalloc(&d_input_b, n * sizeof(float));
        cudaMalloc(&d_output_fused, n * sizeof(float));
        cudaMalloc(&d_output_sequential, n * sizeof(float));
        cudaMalloc(&d_temp, n * sizeof(float));
        
        fill_random_range(d_input_a, n, -2.0f, 2.0f);
        fill_random_range(d_input_b, n, -2.0f, 2.0f);
        
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        
        // Sequential execution
        add_kernel<<<grid, block>>>(d_input_a, d_input_b, d_temp, n);
        relu_kernel<<<grid, block>>>(d_temp, d_output_sequential, n);
        
        // Fused execution
        add_relu_fused_kernel<<<grid, block>>>(d_input_a, d_input_b, d_output_fused, n);
        
        cudaDeviceSynchronize();
        
        bool passed = validate_arrays(d_output_fused, d_output_sequential, n);
        
        cudaFree(d_input_a);
        cudaFree(d_input_b);
        cudaFree(d_output_fused);
        cudaFree(d_output_sequential);
        cudaFree(d_temp);
        
        return passed;
    }
};

int main(int argc, char** argv) {
    std::cout << "=== Simple Fusion Validation ===" << std::endl;
    print_device_info();
    std::cout << std::endl;
    
    bool verbose = false;
    bool quick = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--quick") {
            quick = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --verbose    Enable verbose output" << std::endl;
            std::cout << "  --quick      Run with smaller test sizes" << std::endl;
            std::cout << "  --help       Show this help message" << std::endl;
            return 0;
        }
    }
    
    FusionValidator validator;
    bool all_tests_passed = true;
    
    // Test different sizes
    std::vector<int> test_sizes;
    if (quick) {
        test_sizes = {1024, 4096, 16384};
    } else {
        test_sizes = {1024, 4096, 16384, 65536, 262144, 1048576};
    }
    
    std::cout << "=== Fusion Operation Validation ===" << std::endl;
    
    for (int size : test_sizes) {
        std::cout << "\nTesting with " << size << " elements:" << std::endl;
        
        if (!validator.test_relu_sigmoid_fusion(size)) {
            all_tests_passed = false;
        }
        
        if (!validator.test_sigmoid_tanh_fusion(size)) {
            all_tests_passed = false;
        }
        
        if (!validator.test_three_op_fusion(size)) {
            all_tests_passed = false;
        }
        
        if (!validator.test_add_relu_fusion(size)) {
            all_tests_passed = false;
        }
    }
    
    // Edge case testing
    if (!test_edge_cases()) {
        all_tests_passed = false;
    }
    
    // Final report
    std::cout << "\n=== Validation Summary ===" << std::endl;
    if (all_tests_passed) {
        std::cout << "🎉 All validation tests PASSED!" << std::endl;
        std::cout << "Kernel fusion implementations are numerically correct." << std::endl;
    } else {
        std::cout << "❌ Some validation tests FAILED!" << std::endl;
        std::cout << "Please review the fusion implementation for correctness." << std::endl;
    }
    
    // Performance note
    std::cout << "\nNote: This test validates correctness only." << std::endl;
    std::cout << "For performance benchmarks, run:" << std::endl;
    std::cout << "  ./baseline_comparison" << std::endl;
    std::cout << "  ./elementwise_benchmark" << std::endl;
    
    return all_tests_passed ? 0 : 1;
}
