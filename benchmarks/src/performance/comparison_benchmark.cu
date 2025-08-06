#include "../common/benchmark_utils.hpp"
#include "kernel_fusion/kernels/kernels.hpp"
#include <cublas_v2.h>
#include <iostream>
#include <vector>

// Naive separate kernels for comparison
template<typename T>
__global__ void naive_add_kernel(const T* a, const T* b, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

template<typename T>
__global__ void naive_relu_kernel(const T* input, T* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] > T(0)) ? input[idx] : T(0);
    }
}

class ComparisonBenchmark {
public:
    ComparisonBenchmark() {
        cudaSetDevice(0);
        cublasCreate(&cublas_handle_);
        cudaStreamCreate(&stream_);
    }
    
    ~ComparisonBenchmark() {
        cublasDestroy(cublas_handle_);
        cudaStreamDestroy(stream_);
    }
    
    void run_comparison() {
        std::cout << "=== Kernel Fusion vs Separate Kernels Comparison ===" << std::endl;
        print_device_info();
        std::cout << std::endl;
        
        const std::vector<size_t> sizes = {1048576, 4194304, 16777216, 67108864};
        
        for (size_t n : sizes) {
            std::cout << "\n--- Tensor Size: " << n << " elements ---" << std::endl;
            BenchmarkStats::print_header();
            
            // Compare fused vs separate kernels
            auto fused_stats = benchmark_fused_add_relu(n);
            auto separate_stats = benchmark_separate_add_relu(n);
            
            // Calculate speedup
            double speedup = separate_stats.avg_time_ms / fused_stats.avg_time_ms;
            std::cout << "\nSpeedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            
            // Memory traffic comparison
            std::cout << "Memory Traffic Reduction: " 
                      << std::fixed << std::setprecision(1) 
                      << (1.0 - (double)fused_stats.bytes_transferred / separate_stats.bytes_transferred) * 100 
                      << "%" << std::endl;
        }
        
        // Compare different activation functions
        compare_activation_overhead();
    }

private:
    cublasHandle_t cublas_handle_;
    cudaStream_t stream_;
    
    void print_device_info() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device: " << prop.name << std::endl;
    }
    
    BenchmarkStats benchmark_fused_add_relu(size_t n) {
        std::string kernel_name = "fused_add_relu";
        
        // Allocate memory
        float *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        
        // Initialize data
        std::vector<float> h_a(n), h_b(n);
        fill_random(h_a.data(), n, -2.0f, 2.0f);
        fill_random(h_b.data(), n, -2.0f, 2.0f);
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        // Configure launch
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            kf::kernels::elementwise::add_activation_kernel<float><<<grid_size, block_size, 0, stream_>>>(
                d_a, d_b, d_output, n, KF_ACTIVATION_RELU);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < 100; ++i) {
            timer.start(stream_);
            kf::kernels::elementwise::add_activation_kernel<float><<<grid_size, block_size, 0, stream_>>>(
                d_a, d_b, d_output, n, KF_ACTIVATION_RELU);
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        // Calculate statistics
        size_t bytes_per_element = 3 * sizeof(float);  // 2 reads + 1 write
        size_t operations_per_element = 2;             // 1 add + 1 relu
        BenchmarkStats stats = calculate_stats(kernel_name, times, n, bytes_per_element, operations_per_element);
        stats.print();
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return stats;
    }
    
    BenchmarkStats benchmark_separate_add_relu(size_t n) {
        std::string kernel_name = "separate_add_relu";
        
        // Allocate memory
        float *d_a, *d_b, *d_temp, *d_output;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_temp, n * sizeof(float));  // Intermediate result
        cudaMalloc(&d_output, n * sizeof(float));
        
        // Initialize data
        std::vector<float> h_a(n), h_b(n);
        fill_random(h_a.data(), n, -2.0f, 2.0f);
        fill_random(h_b.data(), n, -2.0f, 2.0f);
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        // Configure launch
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            naive_add_kernel<float><<<grid_size, block_size, 0, stream_>>>(d_a, d_b, d_temp, n);
            naive_relu_kernel<float><<<grid_size, block_size, 0, stream_>>>(d_temp, d_output, n);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < 100; ++i) {
            timer.start(stream_);
            naive_add_kernel<float><<<grid_size, block_size, 0, stream_>>>(d_a, d_b, d_temp, n);
            naive_relu_kernel<float><<<grid_size, block_size, 0, stream_>>>(d_temp, d_output, n);
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        // Calculate statistics
        size_t bytes_per_element = 4 * sizeof(float);  // 2 reads + 2 writes (intermediate)
        size_t operations_per_element = 2;             // 1 add + 1 relu
        BenchmarkStats stats = calculate_stats(kernel_name, times, n, bytes_per_element, operations_per_element);
        stats.print();
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_temp);
        cudaFree(d_output);
        
        return stats;
    }
    
    void compare_activation_overhead() {
        std::cout << "\n=== Activation Function Overhead Analysis ===" << std::endl;
        const size_t n = 16777216;  // 16M elements
        
        const std::vector<std::pair<kf_activation_t, std::string>> activations = {
            {KF_ACTIVATION_NONE, "none"},
            {KF_ACTIVATION_RELU, "relu"},
            {KF_ACTIVATION_GELU, "gelu"},
            {KF_ACTIVATION_SILU, "silu"},
            {KF_ACTIVATION_SIGMOID, "sigmoid"}
        };
        
        BenchmarkStats::print_header();
        
        std::vector<BenchmarkStats> results;
        for (const auto& [activation, name] : activations) {
            results.push_back(benchmark_activation_function(n, activation, name));
        }
        
        // Calculate relative overhead
        std::cout << "\nRelative Overhead (vs no activation):" << std::endl;
        double baseline_time = results[0].avg_time_ms;  // NONE activation
        
        for (size_t i = 1; i < results.size(); ++i) {
            double overhead = (results[i].avg_time_ms - baseline_time) / baseline_time * 100;
            std::cout << std::left << std::setw(10) << activations[i].second 
                      << ": +" << std::fixed << std::setprecision(1) << overhead << "%" << std::endl;
        }
    }
    
    BenchmarkStats benchmark_activation_function(size_t n, kf_activation_t activation, const std::string& name) {
        std::string kernel_name = "add_" + name;
        
        // Allocate memory
        float *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        
        // Initialize data
        std::vector<float> h_a(n), h_b(n);
        fill_random(h_a.data(), n, -1.0f, 1.0f);
        fill_random(h_b.data(), n, -1.0f, 1.0f);
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        // Configure launch
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            kf::kernels::elementwise::add_activation_kernel<float><<<grid_size, block_size, 0, stream_>>>(
                d_a, d_b, d_output, n, activation);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < 100; ++i) {
            timer.start(stream_);
            kf::kernels::elementwise::add_activation_kernel<float><<<grid_size, block_size, 0, stream_>>>(
                d_a, d_b, d_output, n, activation);
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        // Calculate statistics
        size_t bytes_per_element = 3 * sizeof(float);
        size_t operations_per_element = (activation == KF_ACTIVATION_NONE) ? 1 : 2;
        BenchmarkStats stats = calculate_stats(kernel_name, times, n, bytes_per_element, operations_per_element);
        stats.print();
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return stats;
    }
};

int main() {
    try {
        ComparisonBenchmark benchmark;
        benchmark.run_comparison();
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
