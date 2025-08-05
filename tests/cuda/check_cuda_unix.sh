#!/bin/bash

echo "=== CUDA Environment Check ==="

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "❌ nvidia-smi not found"
    echo "Make sure NVIDIA drivers are installed"
fi

echo ""

# Check if nvcc is available
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found"
    nvcc --version | grep "release"
else
    echo "❌ nvcc not found"
    echo "Make sure CUDA toolkit is installed"
fi

echo ""

# Check CUDA environment variables
echo "CUDA Environment Variables:"
echo "CUDA_HOME: ${CUDA_HOME:-'not set'}"
echo "CUDA_PATH: ${CUDA_PATH:-'not set'}"
echo "PATH includes CUDA: $(echo $PATH | grep -o cuda || echo 'not found')"

echo ""

# Try to compile a simple CUDA program
echo "Testing CUDA compilation..."
cat > /tmp/cuda_test.cu << 'EOF'
#include <cuda_runtime.h>
#include <iostream>

__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA devices found: " << deviceCount << std::endl;
    
    if (deviceCount > 0) {
        hello<<<1, 4>>>();
        cudaDeviceSynchronize();
        std::cout << "✓ CUDA test successful!" << std::endl;
    } else {
        std::cout << "❌ No CUDA devices found" << std::endl;
    }
    
    return 0;
}
EOF

if nvcc /tmp/cuda_test.cu -o /tmp/cuda_test 2>/dev/null; then
    echo "✓ CUDA compilation successful"
    if /tmp/cuda_test; then
        echo "✓ CUDA runtime test successful"
    else
        echo "❌ CUDA runtime test failed"
    fi
    rm -f /tmp/cuda_test
else
    echo "❌ CUDA compilation failed"
fi

rm -f /tmp/cuda_test.cu

echo ""
echo "=== Environment Check Complete ==="
