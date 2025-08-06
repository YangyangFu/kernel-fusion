#include "../common/benchmark_utils.hpp"
#include "kernel_fusion/kernels/kernels.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <type_traits>

using namespace kf::kernels;

class ElementwiseBenchmark {
public:
    ElementwiseBenchmark(const BenchmarkConfig& config) : config_(config) {
        // Initialize CUDA
        cudaSetDevice(0);
        if (config_.use_streams) {
            cudaStreamCreate(&stream_);
        }
    }
    
    ~ElementwiseBenchmark() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    void run_all_benchmarks() {
        std::cout << "=== Elementwise Kernels Benchmark Suite ===" << std::endl;
        print_device_info();
        std::cout << std::endl;
        
        BenchmarkStats::print_header();
        
        // Run benchmarks for different tensor sizes
        for (size_t tensor_size : config_.tensor_sizes) {
            benchmark_add_activation<float>(tensor_size, KF_ACTIVATION_RELU);
            benchmark_add_activation<float>(tensor_size, KF_ACTIVATION_GELU);
            benchmark_mul_activation<float>(tensor_size, KF_ACTIVATION_RELU);
            benchmark_mul_activation<float>(tensor_size, KF_ACTIVATION_SILU);
            benchmark_bias_activation<float>(tensor_size, KF_ACTIVATION_RELU);
            
            // Double precision benchmarks for largest sizes
            if (tensor_size >= 1048576) {
                benchmark_add_activation<double>(tensor_size, KF_ACTIVATION_RELU);
                benchmark_mul_activation<double>(tensor_size, KF_ACTIVATION_GELU);
            }
            
            std::cout << std::endl;
        }
        
        std::cout << "\n=== Activation Function Comparison ===" << std::endl;
        compare_activation_functions();
        
        std::cout << "\n=== Precision Comparison ===" << std::endl;
        compare_precisions();
    }

private:
    BenchmarkConfig config_;
    cudaStream_t stream_ = nullptr;
    
    void print_device_info() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        std::cout << "Device: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
        std::cout << "Memory Clock: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "Peak Memory Bandwidth: " 
                  << (2.0 * prop.memoryClockRate * prop.memoryBusWidth / 8) / 1e6 
                  << " GB/s" << std::endl;
    }
    
    template<typename T>
    BenchmarkStats benchmark_add_activation(size_t n, kf_activation_t activation) {
        std::string activation_name = get_activation_name(activation);
        std::string kernel_name = "add_" + activation_name + "_" + get_type_name<T>();
        
        // Allocate host memory
        std::vector<T> h_a(n), h_b(n);
        fill_random(h_a.data(), n, T(-2), T(2));
        fill_random(h_b.data(), n, T(-2), T(2));
        
        // Allocate device memory
        T *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(T));
        cudaMalloc(&d_b, n * sizeof(T));
        cudaMalloc(&d_output, n * sizeof(T));
        
        cudaMemcpy(d_a, h_a.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        
        // Configure kernel launch
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < config_.warmup_iterations; ++i) {
            elementwise::add_activation_kernel<T><<<grid_size, block_size, 0, stream_>>>(
                d_a, d_b, d_output, n, activation
            );
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < config_.benchmark_iterations; ++i) {
            timer.start(stream_);
            elementwise::add_activation_kernel<T><<<grid_size, block_size, 0, stream_>>>(
                d_a, d_b, d_output, n, activation
            );
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        // Calculate statistics
        size_t bytes_per_element = 3 * sizeof(T);  // 2 inputs + 1 output
        size_t operations_per_element = 2;         // 1 add + 1 activation
        BenchmarkStats stats = calculate_stats(kernel_name, times, n, bytes_per_element, operations_per_element);
        stats.print();
        
        // Cleanup
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return stats;
    }
    
    template<typename T>
    BenchmarkStats benchmark_mul_activation(size_t n, kf_activation_t activation) {
        std::string activation_name = get_activation_name(activation);
        std::string kernel_name = "mul_" + activation_name + "_" + get_type_name<T>();
        
        // Allocate host memory
        std::vector<T> h_a(n), h_b(n);
        fill_random(h_a.data(), n, T(-1), T(1));
        fill_random(h_b.data(), n, T(-1), T(1));
        
        // Allocate device memory
        T *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(T));
        cudaMalloc(&d_b, n * sizeof(T));
        cudaMalloc(&d_output, n * sizeof(T));
        
        cudaMemcpy(d_a, h_a.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        
        // Configure kernel launch
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < config_.warmup_iterations; ++i) {
            elementwise::mul_activation_kernel<T><<<grid_size, block_size, 0, stream_>>>(
                d_a, d_b, d_output, n, activation
            );
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < config_.benchmark_iterations; ++i) {
            timer.start(stream_);
            elementwise::mul_activation_kernel<T><<<grid_size, block_size, 0, stream_>>>(
                d_a, d_b, d_output, n, activation
            );
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        // Calculate statistics
        size_t bytes_per_element = 3 * sizeof(T);
        size_t operations_per_element = 2;
        BenchmarkStats stats = calculate_stats(kernel_name, times, n, bytes_per_element, operations_per_element);
        stats.print();
        
        // Cleanup
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return stats;
    }
    
    template<typename T>
    BenchmarkStats benchmark_bias_activation(size_t n, kf_activation_t activation) {
        std::string activation_name = get_activation_name(activation);
        std::string kernel_name = "bias_" + activation_name + "_" + get_type_name<T>();
        
        size_t bias_size = std::min(n, size_t(1024));  // Reasonable bias size
        
        // Allocate host memory
        std::vector<T> h_input(n), h_bias(bias_size);
        fill_random(h_input.data(), n, T(-1), T(1));
        fill_random(h_bias.data(), bias_size, T(-0.5), T(0.5));
        
        // Allocate device memory
        T *d_input, *d_bias, *d_output;
        cudaMalloc(&d_input, n * sizeof(T));
        cudaMalloc(&d_bias, bias_size * sizeof(T));
        cudaMalloc(&d_output, n * sizeof(T));
        
        cudaMemcpy(d_input, h_input.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, h_bias.data(), bias_size * sizeof(T), cudaMemcpyHostToDevice);
        
        // Configure kernel launch
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < config_.warmup_iterations; ++i) {
            elementwise::bias_activation_kernel<T><<<grid_size, block_size, 0, stream_>>>(
                d_input, d_bias, d_output, n, bias_size, activation
            );
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < config_.benchmark_iterations; ++i) {
            timer.start(stream_);
            elementwise::bias_activation_kernel<T><<<grid_size, block_size, 0, stream_>>>(
                d_input, d_bias, d_output, n, bias_size, activation
            );
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        // Calculate statistics  
        size_t bytes_per_element = 2 * sizeof(T) + sizeof(T) * bias_size / n;  // input + bias + output
        size_t operations_per_element = 2;
        BenchmarkStats stats = calculate_stats(kernel_name, times, n, bytes_per_element, operations_per_element);
        stats.print();
        
        // Cleanup
        cudaFree(d_input);
        cudaFree(d_bias);
        cudaFree(d_output);
        
        return stats;
    }
    
    void compare_activation_functions() {
        const size_t n = 4194304;  // 4M elements
        const std::vector<kf_activation_t> activations = {
            KF_ACTIVATION_NONE, KF_ACTIVATION_RELU, KF_ACTIVATION_GELU, 
            KF_ACTIVATION_SILU, KF_ACTIVATION_SIGMOID
        };
        
        std::cout << "Activation function overhead (4M elements, float32):" << std::endl;
        BenchmarkStats::print_header();
        
        for (auto activation : activations) {
            benchmark_add_activation<float>(n, activation);
        }
    }
    
    void compare_precisions() {
        const size_t n = 4194304;  // 4M elements
        const kf_activation_t activation = KF_ACTIVATION_RELU;
        
        std::cout << "Precision comparison (4M elements, ReLU):" << std::endl;
        BenchmarkStats::print_header();
        
        benchmark_add_activation<float>(n, activation);
        benchmark_add_activation<double>(n, activation);
    }
    
    std::string get_activation_name(kf_activation_t activation) {
        switch (activation) {
            case KF_ACTIVATION_NONE: return "none";
            case KF_ACTIVATION_RELU: return "relu";
            case KF_ACTIVATION_GELU: return "gelu";
            case KF_ACTIVATION_SILU: return "silu";
            case KF_ACTIVATION_SIGMOID: return "sigmoid";
            default: return "unknown";
        }
    }
    
    template<typename T>
    std::string get_type_name() {
        if constexpr (std::is_same_v<T, float>) return "f32";
        else if constexpr (std::is_same_v<T, double>) return "f64";
        else return "unknown";
    }
};

int main(int argc, char** argv) {
    // Parse command line arguments
    BenchmarkConfig config;
    
    if (argc > 1) {
        if (std::string(argv[1]) == "--quick") {
            config.tensor_sizes = {1048576, 16777216};  // 1M and 16M elements only
            config.benchmark_iterations = 50;
        } else if (std::string(argv[1]) == "--extensive") {
            config.benchmark_iterations = 200;
            config.tensor_sizes.insert(config.tensor_sizes.end(), {
                134217728,  // 128M elements
                268435456   // 256M elements  
            });
        }
    }
    
    try {
        ElementwiseBenchmark benchmark(config);
        benchmark.run_all_benchmarks();
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
