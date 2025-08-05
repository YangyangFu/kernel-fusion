#include "kernel_fusion/kernels/kernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <iomanip>

// Simple test framework macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define TEST_ASSERT(condition, message) do { \
    if (!(condition)) { \
        std::cerr << "TEST FAILED: " << message << std::endl; \
        return false; \
    } \
} while(0)

#define TEST_TOLERANCE 1e-10f

// Helper function to check if two arrays are approximately equal
template<typename T>
bool arrays_equal(const T* a, const T* b, size_t n, T tolerance = TEST_TOLERANCE) {
    T max_error = 0;
    size_t error_index = 0;
    int error_count = 0;
    
    for (size_t i = 0; i < n; ++i) {
        T diff = std::abs(a[i] - b[i]);
        
        // Use relative tolerance for better floating-point comparison
        T rel_tolerance = tolerance * std::max(std::abs(a[i]), std::abs(b[i]));
        T abs_tolerance = tolerance;
        T effective_tolerance = std::max(rel_tolerance, abs_tolerance);
        
        if (diff > effective_tolerance) {
            if (error_count < 5) { // Only print first 5 errors
                std::cerr << "Mismatch at index " << i << ": " << std::scientific << std::setprecision(10) 
                         << a[i] << " vs " << b[i] << " (diff: " << diff << ", tolerance: " << effective_tolerance << ")" << std::endl;
            }
            error_count++;
            
            if (diff > max_error) {
                max_error = diff;
                error_index = i;
            }
        }
    }
    
    if (error_count > 0) {
        std::cerr << "Total errors: " << error_count << "/" << n << " elements" << std::endl;
        std::cerr << "Maximum error: " << std::scientific << max_error << " at index " << error_index << std::endl;
        return false;
    }
    
    return true;
}

// CPU reference implementations for validation
template<typename T>
T cpu_apply_activation(T x, kf_activation_t activation) {
    switch (activation) {
        case KF_ACTIVATION_NONE: 
            return x;
        case KF_ACTIVATION_RELU: 
            return (x > T(0)) ? x : T(0);  // Use conditional instead of std::max for consistency
        case KF_ACTIVATION_GELU: {
            // Use the same approximation as GPU code
            T x3 = x * x * x;
            T inner = T(0.7978845608) * (x + T(0.044715) * x3);
            return T(0.5) * x * (T(1) + std::tanh(inner));
        }
        case KF_ACTIVATION_SILU: 
            return x / (T(1) + std::exp(-x));
        case KF_ACTIVATION_SIGMOID: 
            return T(1) / (T(1) + std::exp(-x));
        default: 
            return x;
    }
}

// ============================================================================
// Test Cases
// ============================================================================

bool test_add_activation_kernel() {
    std::cout << "Testing add_activation_kernel..." << std::endl;
    
    const int n = 1024;
    const kf_activation_t activation = KF_ACTIVATION_RELU;
    
    // Allocate host memory
    std::vector<float> h_a(n), h_b(n), h_output(n), h_expected(n);
    
    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
    
    for (int i = 0; i < n; ++i) {
        h_a[i] = dis(gen);
        h_b[i] = dis(gen);
        h_expected[i] = cpu_apply_activation(h_a[i] + h_b[i], activation);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_output;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    kf::kernels::elementwise::add_activation_kernel<float><<<grid_size, block_size>>>(
        d_a, d_b, d_output, n, activation
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Validate
    bool success = arrays_equal(h_output.data(), h_expected.data(), n);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_output));
    
    TEST_ASSERT(success, "add_activation_kernel output mismatch");
    std::cout << "✓ add_activation_kernel passed" << std::endl;
    return true;
}

bool test_mul_activation_kernel() {
    std::cout << "Testing mul_activation_kernel..." << std::endl;
    
    const int n = 1024;
    const kf_activation_t activation = KF_ACTIVATION_GELU;
    
    // Allocate host memory
    std::vector<float> h_a(n), h_b(n), h_output(n), h_expected(n);
    
    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < n; ++i) {
        h_a[i] = dis(gen);
        h_b[i] = dis(gen);
        h_expected[i] = cpu_apply_activation(h_a[i] * h_b[i], activation);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_output;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    kf::kernels::elementwise::mul_activation_kernel<float><<<grid_size, block_size>>>(
        d_a, d_b, d_output, n, activation
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Validate
    bool success = arrays_equal(h_output.data(), h_expected.data(), n, 1e-4f); // GELU needs larger tolerance
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_output));
    
    TEST_ASSERT(success, "mul_activation_kernel output mismatch");
    std::cout << "✓ mul_activation_kernel passed" << std::endl;
    return true;
}

bool test_bias_activation_kernel() {
    std::cout << "Testing bias_activation_kernel..." << std::endl;
    
    const int n = 1024;
    const int bias_size = 64;
    const kf_activation_t activation = KF_ACTIVATION_SILU;
    
    // Allocate host memory
    std::vector<float> h_input(n), h_bias(bias_size), h_output(n), h_expected(n);
    
    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < n; ++i) {
        h_input[i] = dis(gen);
    }
    for (int i = 0; i < bias_size; ++i) {
        h_bias[i] = dis(gen);
    }
    
    // Compute expected results
    for (int i = 0; i < n; ++i) {
        int bias_idx = i % bias_size;
        float biased = h_input[i] + h_bias[bias_idx];
        h_expected[i] = cpu_apply_activation(biased, activation);
    }
    
    // Allocate device memory
    float *d_input, *d_bias, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, bias_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    kf::kernels::elementwise::bias_activation_kernel<float><<<grid_size, block_size>>>(
        d_input, d_bias, d_output, n, bias_size, activation
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Validate
    bool success = arrays_equal(h_output.data(), h_expected.data(), n, 1e-4f);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
    
    TEST_ASSERT(success, "bias_activation_kernel output mismatch");
    std::cout << "✓ bias_activation_kernel passed" << std::endl;
    return true;
}

bool test_different_data_types() {
    std::cout << "Testing with double precision..." << std::endl;
    
    const int n = 512;
    const kf_activation_t activation = KF_ACTIVATION_RELU;
    
    // Test with double precision
    std::vector<double> h_a(n), h_b(n), h_output(n), h_expected(n);
    
    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-2.0, 2.0);
    
    for (int i = 0; i < n; ++i) {
        h_a[i] = dis(gen);
        h_b[i] = dis(gen);
        h_expected[i] = cpu_apply_activation(h_a[i] + h_b[i], activation);
    }
    
    // Allocate device memory
    double *d_a, *d_b, *d_output;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(double)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block_size(256);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    
    kf::kernels::elementwise::add_activation_kernel<double><<<grid_size, block_size>>>(
        d_a, d_b, d_output, n, activation
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Validate with appropriate tolerance for double precision
    // GPU vs CPU can have slight differences due to different math libraries
    bool success = arrays_equal(h_output.data(), h_expected.data(), n, 1e-14);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_output));
    
    TEST_ASSERT(success, "double precision test failed");
    std::cout << "✓ double precision test passed" << std::endl;
    return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "=== CUDA Elementwise Kernels Test Suite ===" << std::endl;
    
    // Check CUDA device availability
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Set device
    CUDA_CHECK(cudaSetDevice(0));
    
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;
    
    // Run tests
    bool all_passed = true;
    
    try {
        all_passed &= test_add_activation_kernel();
        all_passed &= test_mul_activation_kernel();
        all_passed &= test_bias_activation_kernel();
        all_passed &= test_different_data_types();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        all_passed = false;
    }
    
    std::cout << std::endl;
    if (all_passed) {
        std::cout << "🎉 All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some tests FAILED!" << std::endl;
        return 1;
    }
}
