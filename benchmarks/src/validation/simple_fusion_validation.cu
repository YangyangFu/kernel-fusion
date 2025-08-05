#include "kernel_fusion/kernels/kernels.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

class SimpleFusionValidation {
public:
    SimpleFusionValidation() {
        cudaStreamCreate(&stream_);
    }
    
    ~SimpleFusionValidation() {
        cudaStreamDestroy(stream_);
    }
    
    bool validate_all() {
        std::cout << "=== Validating Kernel Fusion Correctness (Simple) ===" << std::endl;
        
        bool all_passed = true;
        all_passed &= validate_add_relu();
        all_passed &= validate_add_gelu();
        all_passed &= validate_mul_silu();
        all_passed &= validate_bias_activation();
        all_passed &= validate_double_precision();
        
        if (all_passed) {
            std::cout << "\n✅ All fusion validations PASSED!" << std::endl;
            std::cout << "Kernel fusion produces identical results to CPU reference implementations." << std::endl;
        } else {
            std::cout << "\n❌ Some fusion validations FAILED!" << std::endl;
        }
        
        return all_passed;
    }

private:
    cudaStream_t stream_;
    
    bool validate_add_relu() {
        std::cout << "\nValidating ADD + RELU fusion..." << std::endl;
        
        const size_t n = 10000;
        
        // Create random test data
        std::vector<float> h_a(n), h_b(n);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-3.0f, 3.0f);
        
        for (size_t i = 0; i < n; ++i) {
            h_a[i] = dis(gen);
            h_b[i] = dis(gen);
        }
        
        // 1. Our fused kernel result
        std::vector<float> fused_result = compute_fused_add_relu(h_a, h_b);
        
        // 2. CPU reference result
        std::vector<float> cpu_result(n);
        for (size_t i = 0; i < n; ++i) {
            cpu_result[i] = std::max(0.0f, h_a[i] + h_b[i]);
        }
        
        // Compare results
        bool success = compare_arrays(fused_result, cpu_result, "Fused vs CPU");
        
        return success;
    }
    
    bool validate_add_gelu() {
        std::cout << "\nValidating ADD + GELU fusion..." << std::endl;
        
        const size_t n = 10000;
        
        std::vector<float> h_a(n), h_b(n);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
        
        for (size_t i = 0; i < n; ++i) {
            h_a[i] = dis(gen);
            h_b[i] = dis(gen);
        }
        
        // Our fused kernel
        std::vector<float> fused_result = compute_fused_add_gelu(h_a, h_b);
        
        // CPU reference (GELU approximation)
        std::vector<float> cpu_result(n);
        for (size_t i = 0; i < n; ++i) {
            float x = h_a[i] + h_b[i];
            float x3 = x * x * x;
            float inner = 0.7978845608f * (x + 0.044715f * x3);
            cpu_result[i] = 0.5f * x * (1.0f + std::tanh(inner));
        }
        
        bool success = compare_arrays(fused_result, cpu_result, "Fused vs CPU", 1e-5f);
        
        return success;
    }
    
    bool validate_mul_silu() {
        std::cout << "\nValidating MUL + SiLU fusion..." << std::endl;
        
        const size_t n = 10000;
        
        std::vector<float> h_a(n), h_b(n);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
        
        for (size_t i = 0; i < n; ++i) {
            h_a[i] = dis(gen);
            h_b[i] = dis(gen);
        }
        
        // Our fused kernel
        std::vector<float> fused_result = compute_fused_mul_silu(h_a, h_b);
        
        // CPU reference
        std::vector<float> cpu_result(n);
        for (size_t i = 0; i < n; ++i) {
            float x = h_a[i] * h_b[i];
            cpu_result[i] = x / (1.0f + std::exp(-x));
        }
        
        bool success = compare_arrays(fused_result, cpu_result, "Fused vs CPU", 1e-6f);
        
        return success;
    }
    
    bool validate_bias_activation() {
        std::cout << "\nValidating BIAS + RELU fusion..." << std::endl;
        
        const size_t n = 10000;
        const size_t bias_size = 128;
        
        std::vector<float> h_input(n), h_bias(bias_size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (size_t i = 0; i < n; ++i) {
            h_input[i] = dis(gen);
        }
        for (size_t i = 0; i < bias_size; ++i) {
            h_bias[i] = dis(gen);
        }
        
        // Our fused kernel
        std::vector<float> fused_result = compute_fused_bias_relu(h_input, h_bias);
        
        // CPU reference
        std::vector<float> cpu_result(n);
        for (size_t i = 0; i < n; ++i) {
            size_t bias_idx = i % bias_size;
            float biased = h_input[i] + h_bias[bias_idx];
            cpu_result[i] = std::max(0.0f, biased);
        }
        
        bool success = compare_arrays(fused_result, cpu_result, "Fused vs CPU");
        
        return success;
    }
    
    bool validate_double_precision() {
        std::cout << "\nValidating double precision..." << std::endl;
        
        const size_t n = 5000;
        
        std::vector<double> h_a(n), h_b(n);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-2.0, 2.0);
        
        for (size_t i = 0; i < n; ++i) {
            h_a[i] = dis(gen);
            h_b[i] = dis(gen);
        }
        
        // Our fused kernel
        std::vector<double> fused_result = compute_fused_add_relu_double(h_a, h_b);
        
        // CPU reference
        std::vector<double> cpu_result(n);
        for (size_t i = 0; i < n; ++i) {
            cpu_result[i] = std::max(0.0, h_a[i] + h_b[i]);
        }
        
        bool success = compare_arrays(fused_result, cpu_result, "Double precision", 1e-14);
        
        return success;
    }
    
    std::vector<float> compute_fused_add_relu(const std::vector<float>& a, const std::vector<float>& b) {
        size_t n = a.size();
        
        float *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        
        cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        kf::kernels::elementwise::add_activation_kernel<float><<<grid_size, block_size, 0, stream_>>>(
            d_a, d_b, d_output, n, KF_ACTIVATION_RELU);
        
        cudaDeviceSynchronize();
        
        std::vector<float> result(n);
        cudaMemcpy(result.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return result;
    }
    
    std::vector<float> compute_fused_add_gelu(const std::vector<float>& a, const std::vector<float>& b) {
        size_t n = a.size();
        
        float *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        
        cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        kf::kernels::elementwise::add_activation_kernel<float><<<grid_size, block_size, 0, stream_>>>(
            d_a, d_b, d_output, n, KF_ACTIVATION_GELU);
        
        cudaDeviceSynchronize();
        
        std::vector<float> result(n);
        cudaMemcpy(result.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return result;
    }
    
    std::vector<float> compute_fused_mul_silu(const std::vector<float>& a, const std::vector<float>& b) {
        size_t n = a.size();
        
        float *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        
        cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        kf::kernels::elementwise::mul_activation_kernel<float><<<grid_size, block_size, 0, stream_>>>(
            d_a, d_b, d_output, n, KF_ACTIVATION_SILU);
        
        cudaDeviceSynchronize();
        
        std::vector<float> result(n);
        cudaMemcpy(result.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return result;
    }
    
    std::vector<float> compute_fused_bias_relu(const std::vector<float>& input, const std::vector<float>& bias) {
        size_t n = input.size();
        size_t bias_size = bias.size();
        
        float *d_input, *d_bias, *d_output;
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_bias, bias_size * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        
        cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, bias.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        kf::kernels::elementwise::bias_activation_kernel<float><<<grid_size, block_size, 0, stream_>>>(
            d_input, d_bias, d_output, n, bias_size, KF_ACTIVATION_RELU);
        
        cudaDeviceSynchronize();
        
        std::vector<float> result(n);
        cudaMemcpy(result.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_input);
        cudaFree(d_bias);
        cudaFree(d_output);
        
        return result;
    }
    
    std::vector<double> compute_fused_add_relu_double(const std::vector<double>& a, const std::vector<double>& b) {
        size_t n = a.size();
        
        double *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(double));
        cudaMalloc(&d_b, n * sizeof(double));
        cudaMalloc(&d_output, n * sizeof(double));
        
        cudaMemcpy(d_a, a.data(), n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice);
        
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        kf::kernels::elementwise::add_activation_kernel<double><<<grid_size, block_size, 0, stream_>>>(
            d_a, d_b, d_output, n, KF_ACTIVATION_RELU);
        
        cudaDeviceSynchronize();
        
        std::vector<double> result(n);
        cudaMemcpy(result.data(), d_output, n * sizeof(double), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return result;
    }
    
    template<typename T>
    bool compare_arrays(const std::vector<T>& a, const std::vector<T>& b, 
                       const std::string& desc, T tolerance = T(1e-6)) {
        if (a.size() != b.size()) {
            std::cout << "  ❌ " << desc << ": Size mismatch" << std::endl;
            return false;
        }
        
        T max_error = T(0);
        size_t error_count = 0;
        size_t n = a.size();
        
        for (size_t i = 0; i < n; ++i) {
            T diff = std::abs(a[i] - b[i]);
            T rel_tolerance = tolerance * std::max(std::abs(a[i]), std::abs(b[i]));
            T effective_tolerance = std::max(rel_tolerance, tolerance);
            
            if (diff > effective_tolerance) {
                error_count++;
                max_error = std::max(max_error, diff);
                
                if (error_count <= 3) {
                    std::cout << "    Error at " << i << ": " << a[i] << " vs " << b[i] 
                              << " (diff: " << diff << ")" << std::endl;
                }
            }
        }
        
        if (error_count > 0) {
            std::cout << "  ❌ " << desc << ": " << error_count << "/" << n 
                      << " mismatches, max error: " << max_error << std::endl;
            return false;
        } else {
            std::cout << "  ✅ " << desc << ": All values match within tolerance" << std::endl;
            return true;
        }
    }
};

int main() {
    try {
        SimpleFusionValidation validator;
        bool success = validator.validate_all();
        
        if (success) {
            std::cout << "\n=== Validation Summary ===" << std::endl;
            std::cout << "✓ All fused kernels produce correct results" << std::endl;
            std::cout << "✓ Multiple activation functions validated" << std::endl;
            std::cout << "✓ Both single and double precision work" << std::endl;
            std::cout << "✓ Bias addition fusion works correctly" << std::endl;
            std::cout << "\nFusion implementation is mathematically correct!" << std::endl;
        }
        
        return success ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Validation failed: " << e.what() << std::endl;
        return 1;
    }
}
