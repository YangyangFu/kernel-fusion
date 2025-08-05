# CUDA Kernel Tests

This directory contains standalone tests for the CUDA kernels in the kernel-fusion library.

## Requirements

- CUDA Toolkit (11.0 or later)
- CMake (3.18 or later)
- C++17 compatible compiler
- NVIDIA GPU with compute capability 6.0 or higher

# CUDA Kernel Tests

This directory contains standalone tests for the CUDA kernels in the kernel-fusion library.

## Requirements

- CUDA Toolkit (11.0 or later)
- CMake (3.18 or later)
- C++17 compatible compiler
- NVIDIA GPU with compute capability 6.0 or higher

## Quick Start

### Option 1: Direct Ubuntu/Linux Build
```bash
cd tests/cuda
chmod +x run_tests.sh
./run_tests.sh
```

### Option 2: Docker (Recommended)
```bash
# Run tests in Docker
cd tests/cuda
docker-compose up cuda-tests

# Or interactive development
docker-compose up -d cuda-dev
docker-compose exec cuda-dev bash
```

### Option 3: Manual Build
```bash
cd tests/cuda
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Run tests
./test_elementwise_kernels
./quick_test
```

## Docker Usage

### Running Tests
```bash
# Build and run all tests
docker-compose up cuda-tests

# Interactive shell for development
docker-compose run --rm cuda-dev bash

# Build custom image
docker build -t kernel-fusion-tests -f Dockerfile ../../
docker run --gpus all -it kernel-fusion-tests
```

### Docker Requirements
- Docker with NVIDIA Container Toolkit
- NVIDIA drivers installed on host
- GPU access enabled (`--gpus all` or `runtime: nvidia`)

## Test Structure

### Available Tests

1. **`test_elementwise_kernels.cu`** - Comprehensive tests for elementwise operations
   - `add_activation_kernel` - Elementwise addition with activation
   - `mul_activation_kernel` - Elementwise multiplication with activation  
   - `bias_activation_kernel` - Bias addition with activation
   - Tests multiple data types (float, double)
   - Tests different activation functions (ReLU, GELU, SiLU, Sigmoid)

2. **`quick_test.cu`** - Simple sanity check
   - Basic functionality test
   - Minimal setup for debugging

3. **`test_normalization_kernels.cu`** - Layer normalization tests (TODO)
4. **`test_linear_kernels.cu`** - Linear algebra tests (TODO)

### Test Features

- **Random Data Generation**: Tests use random inputs for robustness
- **CPU Reference**: Validates GPU results against CPU implementations
- **Multiple Precisions**: Tests both float and double precision
- **Error Handling**: Comprehensive CUDA error checking
- **Performance Info**: Reports device information and timing

## Running Individual Tests

```bash
# Run specific test
./test_elementwise_kernels

# Quick functionality check
./quick_test

# Run with CTest
cd build
ctest --verbose

# Run specific test with CTest
ctest -R ElementwiseKernels --verbose
```

## Adding New Tests

1. Create a new `.cu` file in this directory
2. Follow the pattern from `test_elementwise_kernels.cu`:
   - Include necessary headers
   - Use `CUDA_CHECK` macro for error handling
   - Use `TEST_ASSERT` for validation
   - Implement CPU reference functions
   - Compare GPU vs CPU results

3. Add the test to `CMakeLists.txt`:
```cmake
add_executable(test_new_kernels 
    test_new_kernels.cu
    ${CMAKE_SOURCE_DIR}/../core/src/kernels/new_kernels.cu
    ${CMAKE_SOURCE_DIR}/../core/src/kernels/activation_utils.cu
)
```

## Test Methodology

### Validation Strategy
1. **Random Input Generation**: Use different random seeds and distributions
2. **CPU Reference Implementation**: Implement the same logic on CPU
3. **Numerical Comparison**: Compare with appropriate tolerances
4. **Edge Cases**: Test with zero, negative, and boundary values
5. **Multiple Data Types**: Ensure template instantiations work correctly

### Performance Testing
- Use CUDA events for precise timing
- Test with different grid/block configurations
- Validate memory coalescing patterns
- Check shared memory usage

## Troubleshooting

### Common Issues

**CMake can't find CUDA:**
```
export CUDA_PATH=/usr/local/cuda
# or set CUDA_PATH environment variable on Windows
```

**Compute capability errors:**
Edit `CMakeLists.txt` and adjust the architecture flags:
```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_XX,code=sm_XX")
```

**Memory errors:**
- Use `cuda-memcheck` for detailed analysis
- Check array bounds in kernel code
- Verify memory allocation sizes

**Numerical precision issues:**
- Adjust tolerance in `arrays_equal()` function
- Consider using double precision for reference
- Check for NaN/Inf values in intermediate calculations

## Integration with Main Build

These tests are designed to be standalone but can be integrated into the main CMake build:

```cmake
# In main CMakeLists.txt
option(BUILD_CUDA_TESTS "Build CUDA kernel tests" ON)
if(BUILD_CUDA_TESTS)
    add_subdirectory(tests/cuda)
endif()
```
