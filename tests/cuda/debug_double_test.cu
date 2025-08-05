#include "kernel_fusion/kernels/kernels.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "Debug double precision test..." << std::endl;
    
    // Simple test with known values
    const int n = 4;
    double h_a[n] = {1.0, -0.5, 2.0, -1.0};
    double h_b[n] = {0.5, 1.0, -0.5, 2.0};
    double h_output[n];
    double h_expected[n];
    
    // Compute expected on CPU
    for (int i = 0; i < n; ++i) {
        double sum = h_a[i] + h_b[i];
        h_expected[i] = (sum > 0.0) ? sum : 0.0;  // ReLU
    }
    
    // GPU computation
    double *d_a, *d_b, *d_output;
    cudaMalloc(&d_a, n * sizeof(double));
    cudaMalloc(&d_b, n * sizeof(double));
    cudaMalloc(&d_output, n * sizeof(double));
    
    cudaMemcpy(d_a, h_a, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Launch kernel
    kf::kernels::elementwise::add_activation_kernel<double><<<1, n>>>(
        d_a, d_b, d_output, n, KF_ACTIVATION_RELU
    );
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Compare with high precision
    std::cout << std::scientific << std::setprecision(16) << std::endl;
    std::cout << "Index | Input A        | Input B        | Expected       | GPU Result     | Difference" << std::endl;
    std::cout << "------|----------------|----------------|----------------|----------------|----------------" << std::endl;
    
    bool all_match = true;
    for (int i = 0; i < n; ++i) {
        double diff = std::abs(h_output[i] - h_expected[i]);
        bool match = diff < 1e-15;
        all_match &= match;
        
        std::cout << std::setw(5) << i << " | " 
                 << std::setw(14) << h_a[i] << " | "
                 << std::setw(14) << h_b[i] << " | "
                 << std::setw(14) << h_expected[i] << " | "
                 << std::setw(14) << h_output[i] << " | "
                 << std::setw(14) << diff
                 << (match ? " ✓" : " ✗") << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Test " << (all_match ? "PASSED" : "FAILED") << std::endl;
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_output);
    
    return all_match ? 0 : 1;
}
