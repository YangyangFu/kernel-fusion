#pragma once

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

// CUDA timing utilities
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start(cudaStream_t stream = nullptr) {
        cudaEventRecord(start_, stream);
    }
    
    void stop(cudaStream_t stream = nullptr) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
    }
    
    float elapsed_ms() const {
        float elapsed;
        cudaEventElapsedTime(&elapsed, start_, stop_);
        return elapsed;
    }

private:
    cudaEvent_t start_, stop_;
};

// CPU timing utilities
class CpuTimer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        stop_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
        return duration.count() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_, stop_;
};

// Benchmark statistics
struct BenchmarkStats {
    std::string name;
    size_t tensor_size;
    size_t bytes_transferred;
    double min_time_ms;
    double max_time_ms;
    double avg_time_ms;
    double std_dev_ms;
    double bandwidth_gb_s;
    double throughput_gops;
    int iterations;
    
    void print() const {
        std::cout << std::left << std::setw(25) << name 
                  << std::right << std::setw(12) << tensor_size
                  << std::setw(10) << std::fixed << std::setprecision(3) << avg_time_ms
                  << std::setw(10) << std::fixed << std::setprecision(2) << bandwidth_gb_s
                  << std::setw(12) << std::fixed << std::setprecision(2) << throughput_gops
                  << std::setw(8) << iterations << std::endl;
    }
    
    static void print_header() {
        std::cout << std::left << std::setw(25) << "Kernel"
                  << std::right << std::setw(12) << "Elements"
                  << std::setw(10) << "Time(ms)"
                  << std::setw(10) << "BW(GB/s)"
                  << std::setw(12) << "GOPS"
                  << std::setw(8) << "Iters" << std::endl;
        std::cout << std::string(77, '-') << std::endl;
    }
};

// Benchmark configuration
struct BenchmarkConfig {
    std::vector<size_t> tensor_sizes = {
        1024,           // 1K elements
        16384,          // 16K elements  
        262144,         // 256K elements
        1048576,        // 1M elements
        4194304,        // 4M elements
        16777216,       // 16M elements
        67108864        // 64M elements
    };
    
    int warmup_iterations = 5;
    int benchmark_iterations = 100;
    bool use_streams = true;
    bool profile_memory = true;
};

// Utility functions
template<typename T>
void fill_random(T* data, size_t n, T min_val = T(-1), T max_val = T(1)) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min_val, max_val);
    
    for (size_t i = 0; i < n; ++i) {
        data[i] = dis(gen);
    }
}

// Memory bandwidth calculation
double calculate_bandwidth_gb_s(size_t bytes, double time_ms) {
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
}

// Throughput calculation (operations per second)
double calculate_throughput_gops(size_t operations, double time_ms) {
    return (operations / 1e9) / (time_ms / 1000.0);
}

// Statistics calculation
BenchmarkStats calculate_stats(const std::string& name, 
                              const std::vector<double>& times_ms,
                              size_t tensor_size,
                              size_t bytes_per_element,
                              size_t operations_per_element) {
    BenchmarkStats stats;
    stats.name = name;
    stats.tensor_size = tensor_size;
    stats.iterations = times_ms.size();
    
    // Calculate min, max, avg
    stats.min_time_ms = *std::min_element(times_ms.begin(), times_ms.end());
    stats.max_time_ms = *std::max_element(times_ms.begin(), times_ms.end());
    stats.avg_time_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
    
    // Calculate standard deviation
    double variance = 0.0;
    for (double time : times_ms) {
        variance += (time - stats.avg_time_ms) * (time - stats.avg_time_ms);
    }
    stats.std_dev_ms = std::sqrt(variance / times_ms.size());
    
    // Calculate performance metrics using minimum time (best case)
    stats.bytes_transferred = tensor_size * bytes_per_element;
    stats.bandwidth_gb_s = calculate_bandwidth_gb_s(stats.bytes_transferred, stats.min_time_ms);
    stats.throughput_gops = calculate_throughput_gops(tensor_size * operations_per_element, stats.min_time_ms);
    
    return stats;
}
