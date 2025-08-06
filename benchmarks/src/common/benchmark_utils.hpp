#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

// Utility class for timing CUDA kernels
class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    CudaTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_event, stream);
    }
    
    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event);
    }
    
    float elapsed_ms() {
        float time;
        cudaEventElapsedTime(&time, start_event, stop_event);
        return time;
    }
};

// Fill array with random values
inline void fill_random(float* data, size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < n; ++i) {
        data[i] = dis(gen);
    }
}

// Calculate memory bandwidth in GB/s
inline double calculate_bandwidth_gb_s(size_t bytes_transferred, double time_ms) {
    return (bytes_transferred * 1e-6) / time_ms;  // GB/s
}

// Check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// Print device information
inline void print_device_info() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    std::cout << "Memory Bandwidth: " << 
        (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6) << " GB/s" << std::endl;
}

// Format results in a table
inline void print_results_header() {
    std::cout << std::left << std::setw(20) << "Operation"
              << std::right << std::setw(12) << "Time(ms)"
              << std::setw(15) << "Bandwidth(GB/s)"
              << std::setw(12) << "GFLOPS" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
}

inline void print_result_row(const std::string& name, double time_ms, 
                            double bandwidth_gb_s, double gflops = 0.0) {
    std::cout << std::left << std::setw(20) << name
              << std::right << std::setw(12) << std::fixed << std::setprecision(3) << time_ms
              << std::setw(15) << std::fixed << std::setprecision(1) << bandwidth_gb_s;
    if (gflops > 0.0) {
        std::cout << std::setw(12) << std::fixed << std::setprecision(1) << gflops;
    }
    std::cout << std::endl;
}
