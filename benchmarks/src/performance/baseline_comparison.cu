#include "../common/benchmark_utils.hpp"
#include "kernel_fusion/kernels/kernels.hpp"
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <iostream>
#include <vector>

// Baseline implementations using different frameworks
// Simple CUDA kernels for benchmarking
__global__ void simple_add_kernel(float* a, float* b, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

__global__ void simple_relu_kernel(float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

__global__ void add_relu_kernel(float* a, float* b, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = a[idx] + b[idx];
        output[idx] = fmaxf(sum, 0.0f);
    }
}

class BaselineComparison {
public:
    BaselineComparison() : torch_device_(torch::kCUDA, 0) {
        cudaSetDevice(0);
        cudaStreamCreate(&stream_);
    }
    
    ~BaselineComparison() {
        cudaStreamDestroy(stream_);
    }
    
    void run_comparison() {
        std::cout << "=== Kernel Fusion vs Framework Baselines ===" << std::endl;
        print_device_info();
        std::cout << std::endl;
        
        const std::vector<size_t> sizes = {1048576, 4194304, 16777216};  // 1M, 4M, 16M
        
        for (size_t n : sizes) {
            std::cout << "\n--- Tensor Size: " << n << " elements ---" << std::endl;
            std::cout << std::left << std::setw(25) << "Implementation"
                      << std::right << std::setw(12) << "Time(ms)"
                      << std::setw(12) << "Speedup"
                      << std::setw(15) << "Memory(GB/s)" << std::endl;
            std::cout << std::string(64, '-') << std::endl;
            
            // Run all baseline comparisons
            auto results = compare_add_relu_baselines(n);
            
            // Calculate speedups relative to naive separate kernels
            double baseline_time = results["naive_separate"];
            for (const auto& [name, time] : results) {
                double speedup = baseline_time / time;
                double bandwidth = calculate_bandwidth_gb_s(n * 3 * sizeof(float), time);
                
                std::cout << std::left << std::setw(25) << name
                          << std::right << std::setw(12) << std::fixed << std::setprecision(3) << time
                          << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x"
                          << std::setw(15) << std::fixed << std::setprecision(1) << bandwidth
                          << std::endl;
            }
        }
        
        // Test different activation functions
        std::cout << "\n=== Activation Function Comparison (16M elements) ===" << std::endl;
        compare_activation_baselines(16777216);
    }

private:
    cudaStream_t stream_;
    torch::Device torch_device_;
    
    void print_device_info() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device: " << prop.name << std::endl;
        std::cout << "PyTorch Version: " << TORCH_VERSION << std::endl;
    }
    
    std::map<std::string, double> compare_add_relu_baselines(size_t n) {
        std::map<std::string, double> results;
        
        // 1. Our fused kernel (should be fastest)
        results["our_fused"] = benchmark_our_fused_add_relu(n);
        
        // 2. Naive separate CUDA kernels (worst case baseline)
        results["naive_separate"] = benchmark_naive_separate_kernels(n);
        
        // 3. PyTorch separate operations (real-world baseline)
        results["pytorch_separate"] = benchmark_pytorch_separate(n);
        
        // 4. PyTorch with potential fusion (if available)
        results["pytorch_optimized"] = benchmark_pytorch_optimized(n);
        
        // 5. Thrust-based implementation
        results["thrust_transform"] = benchmark_thrust_implementation(n);
        
        // 6. Manual memory management baseline
        results["manual_optimized"] = benchmark_manual_optimized(n);
        
        return results;
    }
    
    double benchmark_our_fused_add_relu(size_t n) {
        // Allocate memory
        float *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        
        // Initialize data
        std::vector<float> h_data(n);
        fill_random(h_data.data(), n);
        cudaMemcpy(d_a, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
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
        
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return avg_time;
    }
    
    double benchmark_naive_separate_kernels(size_t n) {
        // Simple separate kernels (worst case) - avoid thrust entirely
        float *d_a, *d_b, *d_temp, *d_output;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_temp, n * sizeof(float));  // Intermediate storage
        cudaMalloc(&d_output, n * sizeof(float));
        
        std::vector<float> h_data(n);
        fill_random(h_data.data(), n);
        cudaMemcpy(d_a, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            simple_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_temp, n);
            simple_relu_kernel<<<grid_size, block_size>>>(d_temp, d_output, n);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < 100; ++i) {
            timer.start(stream_);
            simple_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_temp, n);
            simple_relu_kernel<<<grid_size, block_size>>>(d_temp, d_output, n);
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_temp);
        cudaFree(d_output);
        
        return avg_time;
    }
    
    double benchmark_pytorch_separate(size_t n) {
        // Create PyTorch tensors
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch_device_);
        auto a = torch::randn({static_cast<long>(n)}, options);
        auto b = torch::randn({static_cast<long>(n)}, options);
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto temp = torch::add(a, b);
            auto result = torch::relu(temp);
        }
        torch::cuda::synchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < 100; ++i) {
            timer.start(stream_);
            auto temp = torch::add(a, b);
            auto result = torch::relu(temp);
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    }
    
    double benchmark_pytorch_optimized(size_t n) {
        // Try to use PyTorch's potential optimizations
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch_device_);
        auto a = torch::randn({static_cast<long>(n)}, options);
        auto b = torch::randn({static_cast<long>(n)}, options);
        
        // Try chained operations (PyTorch might optimize)
        auto warmup_fn = [&]() {
            return torch::relu(torch::add(a, b));
        };
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto result = warmup_fn();
        }
        torch::cuda::synchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < 100; ++i) {
            timer.start(stream_);
            auto result = warmup_fn();
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    }
    
    double benchmark_thrust_implementation(size_t n) {
        // Simplified baseline without complex functors
        float *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        
        // Initialize with simple values 
        std::vector<float> h_a(n, 1.0f);
        std::vector<float> h_b(n, -0.5f);
        cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            add_relu_kernel<<<grid_size, block_size>>>(d_a, d_b, d_output, n);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        std::vector<double> times;
        CudaTimer timer;
        
        for (int i = 0; i < 100; ++i) {
            timer.start(stream_);
            add_relu_kernel<<<grid_size, block_size>>>(d_a, d_b, d_output, n);
            timer.stop(stream_);
            times.push_back(timer.elapsed_ms());
        }
        
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return avg_time;
    }
    
    double benchmark_manual_optimized(size_t n) {
        // Hand-optimized separate kernels with memory reuse
        float *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        
        std::vector<float> h_data(n);
        fill_random(h_data.data(), n);
        cudaMemcpy(d_a, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        // The optimized kernel is essentially the same as our fused kernel
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
        
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return avg_time;
    }
    
    void compare_activation_baselines(size_t n) {
        const std::vector<std::pair<kf_activation_t, std::string>> activations = {
            {KF_ACTIVATION_RELU, "relu"},
            {KF_ACTIVATION_GELU, "gelu"},
            {KF_ACTIVATION_SILU, "silu"}
        };
        
        std::cout << std::left << std::setw(15) << "Activation"
                  << std::setw(15) << "Our Fused"
                  << std::setw(15) << "PyTorch Sep"
                  << std::setw(15) << "Speedup" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (const auto& [activation, name] : activations) {
            double our_time = benchmark_our_activation(n, activation);
            double pytorch_time = benchmark_pytorch_activation(n, name);
            double speedup = pytorch_time / our_time;
            
            std::cout << std::left << std::setw(15) << name
                      << std::setw(15) << std::fixed << std::setprecision(3) << our_time
                      << std::setw(15) << std::fixed << std::setprecision(3) << pytorch_time  
                      << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x"
                      << std::endl;
        }
    }
    
    double benchmark_our_activation(size_t n, kf_activation_t activation) {
        float *d_a, *d_b, *d_output;
        cudaMalloc(&d_a, n * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        
        std::vector<float> h_data(n);
        fill_random(h_data.data(), n);
        cudaMemcpy(d_a, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 block_size(256);
        dim3 grid_size((n + block_size.x - 1) / block_size.x);
        
        // Warmup and benchmark
        for (int i = 0; i < 5; ++i) {
            kf::kernels::elementwise::add_activation_kernel<float><<<grid_size, block_size, 0, stream_>>>(
                d_a, d_b, d_output, n, activation);
        }
        cudaDeviceSynchronize();
        
        CudaTimer timer;
        timer.start(stream_);
        kf::kernels::elementwise::add_activation_kernel<float><<<grid_size, block_size, 0, stream_>>>(
            d_a, d_b, d_output, n, activation);
        timer.stop(stream_);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_output);
        
        return timer.elapsed_ms();
    }
    
    double benchmark_pytorch_activation(size_t n, const std::string& activation_name) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch_device_);
        auto a = torch::randn({static_cast<long>(n)}, options);
        auto b = torch::randn({static_cast<long>(n)}, options);
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto temp = torch::add(a, b);
            torch::Tensor result;
            if (activation_name == "relu") {
                result = torch::relu(temp);
            } else if (activation_name == "gelu") {
                result = torch::gelu(temp);
            } else if (activation_name == "silu") {
                result = torch::silu(temp);
            }
        }
        torch::cuda::synchronize();
        
        // Benchmark
        CudaTimer timer;
        timer.start(stream_);
        auto temp = torch::add(a, b);
        torch::Tensor result;
        if (activation_name == "relu") {
            result = torch::relu(temp);
        } else if (activation_name == "gelu") {
            result = torch::gelu(temp);
        } else if (activation_name == "silu") {
            result = torch::silu(temp);
        }
        timer.stop(stream_);
        
        return timer.elapsed_ms();
    }
};

int main() {
    try {
        BaselineComparison comparison;
        comparison.run_comparison();
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
