#include "../common/benchmark_utils.hpp"
#include "kernel_fusion/kernels/kernels.hpp"
#include <iostream>
#include <vector>
#include <type_traits>

// Simple memory bandwidth test kernels
template<typename T>
__global__ void copy_kernel(const T* __restrict__ input, T* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

template<typename T>
__global__ void add_kernel(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

template<typename T>
__global__ void triad_kernel(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ output, T scalar, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + scalar * b[idx];
    }
}

class MemoryBenchmark {
public:
    MemoryBenchmark(const BenchmarkConfig& config) : config_(config) {
        cudaSetDevice(0);
        if (config_.use_streams) {
            cudaStreamCreate(&stream_);
        }
    }
    
    ~MemoryBenchmark() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    void run_bandwidth_tests() {
        std::cout << "=== Memory Bandwidth Benchmark ===" << std::endl;
        print_device_info();
        std::cout << std::endl;
        
        BenchmarkStats::print_header();
        
        for (size_t tensor_size : config_.tensor_sizes) {
            if (tensor_size >= 4194304) {  // Only test large sizes for bandwidth
                benchmark_copy<float>(tensor_size);
                benchmark_add<float>(tensor_size);
                benchmark_triad<float>(tensor_size);
                benchmark_fused_add_relu<float>(tensor_size);
                std::cout << std::endl;
            }
        }
        
        // Compare theoretical vs achieved bandwidth
        compare_bandwidth_efficiency();
    }

private:
    BenchmarkConfig config_;
    cudaStream_t stream_ = nullptr;
    
    void print_device_info() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        double theoretical_bandwidth = (2.0 * prop.memoryClockRate * prop.memoryBusWidth / 8) / 1e6;
        
        std::cout << "Device: " << prop.name << std::endl;
        std::cout << "Theoretical Peak Bandwidth: " << theoretical_bandwidth << " GB/s" << std::endl;
    }
    
    template<typename T>
    BenchmarkStats benchmark_copy(size_t n) {
        std::string kernel_name = "copy_" + get_type_name<T>();
        
        // Allocate memory
        T *d_input, *d_output;
        cudaMalloc(&d_input, n * sizeof(T));
        cudaMalloc(&d_output, n * sizeof(T));
        
        // Initialize input
        std::vector<T> h_input(n);
        fill_random(h_input.data(), n);
        cudaMemcpy(d_input, h_input.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        
        // Configure launch
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < config_.warmup_iterations; ++i) {
            copy_kernel<T><<<grid_size, block_size, 0, stream_>>>(d_input, d_output, n);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < config_.benchmark_iterations; ++i) {
            timer.start(stream_);
            copy_kernel<T><<<grid_size, block_size, 0, stream_>>>(d_input, d_output, n);
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        // Calculate statistics
        size_t bytes_per_element = 2 * sizeof(T);  // 1 read + 1 write
        size_t operations_per_element = 1;         // 1 copy
        BenchmarkStats stats = calculate_stats(kernel_name, times, n, bytes_per_element, operations_per_element);
        stats.print();
        
        cudaFree(d_input);
        cudaFree(d_output);
        
        return stats;
    }
    
    template<typename T>
    BenchmarkStats benchmark_add(size_t n) {
        std::string kernel_name = "add_" + get_type_name<T>();
        
        // Allocate memory
        T *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(T));
        cudaMalloc(&d_b, n * sizeof(T));
        cudaMalloc(&d_output, n * sizeof(T));
        
        // Initialize inputs
        std::vector<T> h_a(n), h_b(n);
        fill_random(h_a.data(), n);
        fill_random(h_b.data(), n);
        cudaMemcpy(d_a, h_a.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        
        // Configure launch
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < config_.warmup_iterations; ++i) {
            add_kernel<T><<<grid_size, block_size, 0, stream_>>>(d_a, d_b, d_output, n);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < config_.benchmark_iterations; ++i) {
            timer.start(stream_);
            add_kernel<T><<<grid_size, block_size, 0, stream_>>>(d_a, d_b, d_output, n);
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        // Calculate statistics
        size_t bytes_per_element = 3 * sizeof(T);  // 2 reads + 1 write
        size_t operations_per_element = 1;         // 1 add
        BenchmarkStats stats = calculate_stats(kernel_name, times, n, bytes_per_element, operations_per_element);
        stats.print();
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return stats;
    }
    
    template<typename T>
    BenchmarkStats benchmark_triad(size_t n) {
        std::string kernel_name = "triad_" + get_type_name<T>();
        
        // Allocate memory
        T *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(T));
        cudaMalloc(&d_b, n * sizeof(T));
        cudaMalloc(&d_output, n * sizeof(T));
        
        // Initialize inputs
        std::vector<T> h_a(n), h_b(n);
        fill_random(h_a.data(), n);
        fill_random(h_b.data(), n);
        cudaMemcpy(d_a, h_a.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        
        T scalar = T(2.5);
        
        // Configure launch
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < config_.warmup_iterations; ++i) {
            triad_kernel<T><<<grid_size, block_size, 0, stream_>>>(d_a, d_b, d_output, scalar, n);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < config_.benchmark_iterations; ++i) {
            timer.start(stream_);
            triad_kernel<T><<<grid_size, block_size, 0, stream_>>>(d_a, d_b, d_output, scalar, n);
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        // Calculate statistics
        size_t bytes_per_element = 3 * sizeof(T);  // 2 reads + 1 write
        size_t operations_per_element = 2;         // 1 multiply + 1 add
        BenchmarkStats stats = calculate_stats(kernel_name, times, n, bytes_per_element, operations_per_element);
        stats.print();
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return stats;
    }
    
    template<typename T>
    BenchmarkStats benchmark_fused_add_relu(size_t n) {
        std::string kernel_name = "fused_add_relu_" + get_type_name<T>();
        
        // Allocate memory
        T *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(T));
        cudaMalloc(&d_b, n * sizeof(T));
        cudaMalloc(&d_output, n * sizeof(T));
        
        // Initialize inputs
        std::vector<T> h_a(n), h_b(n);
        fill_random(h_a.data(), n);
        fill_random(h_b.data(), n);
        cudaMemcpy(d_a, h_a.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        
        // Configure launch
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < config_.warmup_iterations; ++i) {
            kf::kernels::elementwise::add_activation_kernel<T><<<grid_size, block_size, 0, stream_>>>(
                d_a, d_b, d_output, n, KF_ACTIVATION_RELU);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < config_.benchmark_iterations; ++i) {
            timer.start(stream_);
            kf::kernels::elementwise::add_activation_kernel<T><<<grid_size, block_size, 0, stream_>>>(
                d_a, d_b, d_output, n, KF_ACTIVATION_RELU);
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        // Calculate statistics
        size_t bytes_per_element = 3 * sizeof(T);  // 2 reads + 1 write
        size_t operations_per_element = 2;         // 1 add + 1 relu
        BenchmarkStats stats = calculate_stats(kernel_name, times, n, bytes_per_element, operations_per_element);
        stats.print();
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return stats;
    }
    
    void compare_bandwidth_efficiency() {
        std::cout << "\n=== Bandwidth Efficiency Analysis ===" << std::endl;
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        double theoretical_bandwidth = (2.0 * prop.memoryClockRate * prop.memoryBusWidth / 8) / 1e6;
        
        const size_t n = 67108864;  // 64M elements
        
        auto copy_stats = benchmark_copy<float>(n);
        auto add_stats = benchmark_add<float>(n);
        auto triad_stats = benchmark_triad<float>(n);
        auto fused_stats = benchmark_fused_add_relu<float>(n);
        
        std::cout << "\nBandwidth Efficiency:" << std::endl;
        std::cout << "Copy:      " << std::fixed << std::setprecision(1) 
                  << (copy_stats.bandwidth_gb_s / theoretical_bandwidth * 100) << "%" << std::endl;
        std::cout << "Add:       " << std::fixed << std::setprecision(1) 
                  << (add_stats.bandwidth_gb_s / theoretical_bandwidth * 100) << "%" << std::endl;
        std::cout << "Triad:     " << std::fixed << std::setprecision(1) 
                  << (triad_stats.bandwidth_gb_s / theoretical_bandwidth * 100) << "%" << std::endl;
        std::cout << "Fused:     " << std::fixed << std::setprecision(1) 
                  << (fused_stats.bandwidth_gb_s / theoretical_bandwidth * 100) << "%" << std::endl;
    }
    
    template<typename T>
    std::string get_type_name() {
        if constexpr (std::is_same_v<T, float>) return "f32";
        else if constexpr (std::is_same_v<T, double>) return "f64";
        else return "unknown";
    }
};

int main() {
    BenchmarkConfig config;
    config.tensor_sizes = {4194304, 16777216, 67108864};  // 4M, 16M, 64M elements
    config.benchmark_iterations = 50;
    
    try {
        MemoryBenchmark benchmark(config);
        benchmark.run_bandwidth_tests();
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
