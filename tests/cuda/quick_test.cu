#include "kernel_fusion/kernels/kernels.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Quick CUDA Kernel Test ===" << std::endl;
    
    // Check CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "CUDA devices found: " << device_count << std::endl;
    
    // Simple test data
    const int n = 8;
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> b = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> result(n);
    
    // GPU memory
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_result, n * sizeof(float));
    
    // Copy to GPU
    cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel (add + ReLU)
    dim3 block(8);
    dim3 grid(1);
    
    kf::kernels::elementwise::add_activation_kernel<float><<<grid, block>>>(
        d_a, d_b, d_result, n, KF_ACTIVATION_RELU
    );
    
    // Copy back
    cudaMemcpy(result.data(), d_result, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "Input A: ";
    for (int i = 0; i < n; ++i) std::cout << a[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Input B: ";
    for (int i = 0; i < n; ++i) std::cout << b[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Output:  ";
    for (int i = 0; i < n; ++i) std::cout << result[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Expected: 2 3 4 5 6 7 8 9" << std::endl;
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    
    std::cout << "✓ Quick test completed!" << std::endl;
    return 0;
}
