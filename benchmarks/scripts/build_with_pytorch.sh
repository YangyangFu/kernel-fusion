#!/bin/bash

# Quick script to build benchmarks with proper PyTorch detection

set -e

echo "=== Building Kernel Fusion Benchmarks with PyTorch Support ==="
echo ""

# Navigate to benchmarks directory
cd "$(dirname "$0")/.."

# Check if we're in the right place
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ CMakeLists.txt not found. Make sure you're in the benchmarks directory."
    exit 1
fi

# Check for PyTorch
echo "Checking for PyTorch..."
if command -v python &> /dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not found")
    TORCH_CMAKE_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || echo "")
    
    if [ "$TORCH_VERSION" != "not found" ]; then
        echo "✓ Found PyTorch version: $TORCH_VERSION"
        echo "✓ LibTorch cmake path: $TORCH_CMAKE_PATH"
    else
        echo "❌ PyTorch not available"
        exit 1
    fi
else
    echo "❌ Python not found"
    exit 1
fi

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake using PyTorch path
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PATH"

if [ $? -eq 0 ]; then
    echo "✓ CMake configuration successful"
else
    echo "❌ CMake configuration failed"
    exit 1
fi

# Build
echo ""
echo "Building..."
make -j$(nproc) 2>/dev/null || make -j4

if [ $? -eq 0 ]; then
    echo "✓ Build successful"
else
    echo "❌ Build failed"
    exit 1
fi

# List built executables
echo ""
echo "Built executables:"
for exe in elementwise_benchmark memory_benchmark comparison_benchmark simple_baseline_comparison simple_fusion_validation baseline_comparison fusion_validation; do
    if [ -f "./$exe" ]; then
        echo "  ✓ $exe"
    else
        echo "  ⚠ $exe (not built)"
    fi
done

echo ""
echo "=== Build Complete ==="
echo ""
echo "To run validation:"
echo "  ./simple_fusion_validation"
echo ""
echo "To run baseline comparison:"
if [ -f "./baseline_comparison" ]; then
    echo "  ./baseline_comparison  # Full PyTorch comparison"
fi
echo "  ./simple_baseline_comparison  # Simple comparison"
echo ""
echo "To run all benchmarks:"
echo "  cd .. && ./run_benchmarks.sh"
