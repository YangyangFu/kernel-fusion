#!/bin/bash

# Build script for kernel fusion benchmarks with PyTorch support
# This script downloads and configures LibTorch if needed, then builds all benchmarks

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "=== Kernel Fusion Benchmarks Build Script ==="
echo "Project directory: $PROJECT_DIR"
echo "Build directory: $BUILD_DIR"

# Configuration
LIBTORCH_VERSION="2.1.0"
CUDA_VERSION="12.1"
BUILD_TYPE="Release"
ENABLE_PYTORCH="ON"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-pytorch)
            ENABLE_PYTORCH="OFF"
            echo "Building without PyTorch support"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            echo "Building in debug mode"
            shift
            ;;
        --libtorch-version)
            LIBTORCH_VERSION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --no-pytorch           Build without PyTorch baselines"
            echo "  --debug                Build in debug mode"
            echo "  --libtorch-version VER Set LibTorch version (default: $LIBTORCH_VERSION)"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check prerequisites
echo "=== Checking Prerequisites ==="

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA compiler (nvcc) not found"
    echo "Please install CUDA toolkit and ensure nvcc is in PATH"
    exit 1
fi

CUDA_VERSION_OUTPUT=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "Found CUDA version: $CUDA_VERSION_OUTPUT"

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo "ERROR: CMake not found"
    echo "Please install CMake 3.18 or later"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version //')
echo "Found CMake version: $CMAKE_VERSION"

# Check GPU
echo "=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits | head -1
else
    echo "WARNING: nvidia-smi not found, cannot verify GPU"
fi

# LibTorch setup
if [[ "$ENABLE_PYTORCH" == "ON" ]]; then
    echo "=== LibTorch Setup ==="
    
    # Check if PyTorch is installed in the environment
    if command -v python &> /dev/null; then
        echo "Checking for PyTorch installation in current environment..."
        
        # Try to get LibTorch path from PyTorch installation
        LIBTORCH_DIR=$(python -c "
import torch
import os
torch_dir = os.path.dirname(torch.__file__)
libtorch_dir = os.path.join(torch_dir, 'share', 'cmake')
if os.path.exists(libtorch_dir):
    print(torch_dir)
else:
    # Try alternative locations
    possible_paths = [
        os.path.join(torch_dir, '..', 'torch'),
        os.path.join(torch_dir, 'lib'),
        torch_dir
    ]
    for path in possible_paths:
        cmake_path = os.path.join(path, 'share', 'cmake', 'Torch')
        if os.path.exists(cmake_path):
            print(os.path.abspath(path))
            break
    else:
        print('')
" 2>/dev/null) || LIBTORCH_DIR=""
        
        if [[ -n "$LIBTORCH_DIR" && -d "$LIBTORCH_DIR" ]]; then
            echo "Found PyTorch installation with LibTorch at: $LIBTORCH_DIR"
            
            # Verify that we can find the CMake config
            TORCH_CMAKE_CONFIG=""
            for cmake_config in "$LIBTORCH_DIR/share/cmake/Torch/TorchConfig.cmake" "$LIBTORCH_DIR/share/cmake/torch/TorchConfig.cmake"; do
                if [[ -f "$cmake_config" ]]; then
                    TORCH_CMAKE_CONFIG="$cmake_config"
                    break
                fi
            done
            
            if [[ -n "$TORCH_CMAKE_CONFIG" ]]; then
                echo "Found Torch CMake config at: $TORCH_CMAKE_CONFIG"
                CMAKE_PREFIX_PATH="$LIBTORCH_DIR"
                
                # Get PyTorch version info
                PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
                PYTORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'CPU-only')" 2>/dev/null || echo "unknown")
                echo "PyTorch version: $PYTORCH_VERSION"
                echo "PyTorch CUDA version: $PYTORCH_CUDA_VERSION"
            else
                echo "WARNING: PyTorch found but CMake config not available"
                echo "Falling back to standalone LibTorch download..."
                LIBTORCH_DIR=""
            fi
        else
            echo "PyTorch not found or not properly installed"
            echo "Falling back to standalone LibTorch download..."
        fi
    else
        echo "Python not found, cannot detect PyTorch installation"
        echo "Falling back to standalone LibTorch download..."
    fi
    
    # Fallback to downloading LibTorch if PyTorch detection failed
    if [[ -z "$LIBTORCH_DIR" || -z "$CMAKE_PREFIX_PATH" ]]; then
        echo "=== Downloading standalone LibTorch ==="
        LIBTORCH_DIR="$PROJECT_DIR/libtorch"
        
        if [[ ! -d "$LIBTORCH_DIR" ]]; then
            echo "LibTorch not found, downloading..."
            
            # Determine platform and CUDA version for download URL
            PLATFORM="linux"
            if [[ "$OSTYPE" == "darwin"* ]]; then
                PLATFORM="macos"
            fi
            
            # Use CUDA 11.8 for compatibility (LibTorch naming)
            TORCH_CUDA_VERSION="cu118"
            if [[ "$CUDA_VERSION_OUTPUT" == 12.* ]]; then
                TORCH_CUDA_VERSION="cu121"
            fi
            
            LIBTORCH_URL="https://download.pytorch.org/libtorch/${TORCH_CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${TORCH_CUDA_VERSION}.zip"
            
            echo "Downloading LibTorch from: $LIBTORCH_URL"
            
            cd "$PROJECT_DIR"
            if command -v wget &> /dev/null; then
                wget -O libtorch.zip "$LIBTORCH_URL"
            elif command -v curl &> /dev/null; then
                curl -L -o libtorch.zip "$LIBTORCH_URL"
            else
                echo "ERROR: Neither wget nor curl found"
                echo "Please install one of these tools or manually download LibTorch"
                exit 1
            fi
            
            echo "Extracting LibTorch..."
            unzip -q libtorch.zip
            rm libtorch.zip
            
            echo "LibTorch installed to: $LIBTORCH_DIR"
        else
            echo "Found existing LibTorch installation: $LIBTORCH_DIR"
        fi
        
        # Verify LibTorch installation
        if [[ ! -f "$LIBTORCH_DIR/share/cmake/Torch/TorchConfig.cmake" ]]; then
            echo "ERROR: LibTorch installation appears incomplete"
            echo "Please remove $LIBTORCH_DIR and run this script again"
            exit 1
        fi
        
        CMAKE_PREFIX_PATH="$LIBTORCH_DIR"
    fi
    
    echo "LibTorch ready for build"
else
    echo "=== Building without PyTorch ==="
    CMAKE_PREFIX_PATH=""
fi

# Create build directory
echo "=== Setting up Build Directory ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "=== Configuring with CMake ==="
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

if [[ -n "$CMAKE_PREFIX_PATH" ]]; then
    CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH")
fi

echo "CMake configuration:"
echo "  Build type: $BUILD_TYPE"
echo "  CUDA architectures: 60;70;75;80;86"
if [[ -n "$CMAKE_PREFIX_PATH" ]]; then
    echo "  LibTorch path: $CMAKE_PREFIX_PATH"
fi

cmake "${CMAKE_ARGS[@]}" ..

# Build
echo "=== Building ==="
CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")
echo "Building with $CPU_COUNT parallel jobs..."

make -j"$CPU_COUNT"

# Verify build
echo "=== Build Verification ==="
BUILT_TARGETS=()

# Check which targets were built
for target in elementwise_benchmark memory_benchmark comparison_benchmark simple_fusion_validation simple_baseline_comparison; do
    if [[ -f "$target" ]]; then
        BUILT_TARGETS+=("$target")
        echo "✓ $target"
    else
        echo "✗ $target (not built)"
    fi
done

# PyTorch-dependent targets
if [[ "$ENABLE_PYTORCH" == "ON" ]]; then
    for target in baseline_comparison fusion_validation; do
        if [[ -f "$target" ]]; then
            BUILT_TARGETS+=("$target")
            echo "✓ $target (with PyTorch)"
        else
            echo "✗ $target (PyTorch target not built)"
        fi
    done
fi

if [[ ${#BUILT_TARGETS[@]} -eq 0 ]]; then
    echo "ERROR: No benchmark targets were built successfully"
    exit 1
fi

# Quick test
echo "=== Quick Test ==="
if [[ -f "simple_fusion_validation" ]]; then
    echo "Running quick validation test..."
    if ./simple_fusion_validation; then
        echo "✓ Basic validation passed"
    else
        echo "✗ Basic validation failed"
        exit 1
    fi
else
    echo "WARNING: Cannot run validation test (simple_fusion_validation not built)"
fi

# Create run script
echo "=== Creating run script ==="
cat > run_benchmarks.sh << 'EOF'
#!/bin/bash

# Auto-generated benchmark runner script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Kernel Fusion Benchmark Suite ==="
echo "Build directory: $SCRIPT_DIR"
echo

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    echo
fi

# Run basic benchmarks (no PyTorch required)
echo "=== Basic Benchmarks ==="

if [[ -f "elementwise_benchmark" ]]; then
    echo "Running elementwise benchmark..."
    ./elementwise_benchmark
    echo
fi

if [[ -f "memory_benchmark" ]]; then
    echo "Running memory benchmark..."
    ./memory_benchmark
    echo
fi

if [[ -f "comparison_benchmark" ]]; then
    echo "Running comparison benchmark..."
    ./comparison_benchmark
    echo
fi

# Run validation tests
echo "=== Validation Tests ==="

if [[ -f "simple_fusion_validation" ]]; then
    echo "Running simple fusion validation..."
    ./simple_fusion_validation
    echo
fi

if [[ -f "simple_baseline_comparison" ]]; then
    echo "Running simple baseline comparison..."
    ./simple_baseline_comparison
    echo
fi

# Run PyTorch benchmarks if available
if [[ -f "baseline_comparison" ]]; then
    echo "=== PyTorch Baseline Comparison ==="
    echo "Running PyTorch baseline comparison..."
    ./baseline_comparison
    echo
fi

if [[ -f "fusion_validation" ]]; then
    echo "Running comprehensive fusion validation..."
    ./fusion_validation
    echo
fi

echo "=== Benchmark Suite Complete ==="
EOF

chmod +x run_benchmarks.sh

# Summary
echo "=== Build Complete ==="
echo "Built targets: ${BUILT_TARGETS[*]}"
echo "Build directory: $BUILD_DIR"
echo ""
echo "To run benchmarks:"
echo "  cd $BUILD_DIR"
echo "  ./run_benchmarks.sh"
echo ""
echo "To run individual benchmarks:"
for target in "${BUILT_TARGETS[@]}"; do
    echo "  ./$target"
done
echo ""

if [[ "$ENABLE_PYTORCH" == "ON" ]]; then
    echo "PyTorch baseline comparison enabled"
    echo "LibTorch location: $LIBTORCH_DIR"
else
    echo "Built without PyTorch support"
    echo "To enable PyTorch baselines, run: $0 (without --no-pytorch)"
fi

echo ""
echo "Build successful! 🎉"
