#!/bin/bash
set -e  # Exit on any error

echo "Building and running CUDA kernel tests..."

# Check CUDA environment first
if [ -f "check_cuda.sh" ]; then
    echo "Checking CUDA environment..."
    chmod +x check_cuda.sh
    ./check_cuda.sh
    echo ""
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build
echo "Building..."
cmake --build . --config Release -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Running CUDA kernel tests..."
echo "========================================"

# Check if CUDA is available
nvidia-smi > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Warning: nvidia-smi not found. Make sure CUDA drivers are installed."
fi

# Run individual tests
echo ""
echo "[2/3] Testing Elementwise Kernels..."
./test_elementwise_kernels
if [ $? -ne 0 ]; then
    echo "Elementwise tests failed!"
fi

echo ""
echo "[3/3] Quick Test..."
./quick_test
if [ $? -ne 0 ]; then
    echo "Quick test failed!"
fi

# TODO: Add when implemented
# echo ""
# echo "[3/4] Testing Normalization Kernels..."
# ./test_normalization_kernels
# if [ $? -ne 0 ]; then
#     echo "Normalization tests failed!"
# fi

# echo ""
# echo "[4/4] Testing Linear Kernels..."
# ./test_linear_kernels
# if [ $? -ne 0 ]; then
#     echo "Linear tests failed!"
# fi

echo ""
echo "========================================"
echo "All tests completed!"
echo "========================================"

cd ..
