# Kernel Fusion Benchmarks

This directory contains comprehensive benchmarks and validation tests for the kernel fusion library. The benchmarks compare the performance of fused kernels against various baseline implementations.

## Directory Structure

```
examples/benchmarks/
├── README.md                    # This file
├── CMakeLists.txt              # Build configuration
├── scripts/                    # Build and execution scripts
│   ├── build_with_pytorch.sh   # Build script with PyTorch support  
│   └── run_benchmarks.sh       # Run benchmarks with various options
├── src/                        # Source files
│   ├── common/                 # Shared utilities
│   │   └── benchmark_utils.hpp # Common benchmarking utilities
│   ├── performance/            # Performance benchmarks
│   │   ├── baseline_comparison.cu    # Compare against PyTorch/Thrust baselines
│   │   ├── elementwise_benchmark.cu  # Elementwise operation benchmarks
│   │   ├── memory_benchmark.cu       # Memory bandwidth benchmarks
│   │   └── comparison_benchmark.cu   # General comparison benchmarks
│   └── validation/             # Correctness validation
│       ├── fusion_validation.cu        # Validate fusion correctness
│       ├── simple_fusion_validation.cu # Simple validation tests
│       └── simple_baseline_comparison.cu # Simple baseline comparisons
├── docs/                       # Documentation
│   ├── FUSION_VALIDATION_GUIDE.md # Guide for validation testing
│   └── README_baseline.md         # Baseline comparison documentation
└── build/                      # Build artifacts (generated)
```

## Quick Start

### 1. Build with PyTorch Support (Recommended)
```bash
cd examples/benchmarks
./scripts/build_with_pytorch.sh
```

### 2. Run All Benchmarks
```bash
./scripts/run_benchmarks.sh
```

### 3. Run Specific Benchmarks
```bash
# Run only the main baseline comparison
./scripts/run_benchmarks.sh --baseline

# Run only validation tests
./scripts/run_benchmarks.sh --validation

# Run only performance benchmarks  
./scripts/run_benchmarks.sh --performance

# List available benchmarks
./scripts/run_benchmarks.sh --list

# Show help
./scripts/run_benchmarks.sh --help
```

### 4. Run Individual Benchmarks Manually
```bash
cd build
./baseline_comparison      # Compare against PyTorch/Thrust baselines
./elementwise_benchmark    # Test elementwise operations
./memory_benchmark         # Test memory bandwidth
./fusion_validation       # Validate correctness
```

## Benchmark Categories

### Performance Benchmarks (`src/performance/`)

- **`baseline_comparison.cu`** - The main benchmark comparing kernel fusion against:
  - PyTorch separate operations
  - PyTorch optimized operations  
  - Thrust-based implementations
  - Naive separate CUDA kernels
  - Manual optimized implementations

- **`elementwise_benchmark.cu`** - Focused benchmarks for elementwise operations with different configurations

- **`memory_benchmark.cu`** - Memory bandwidth and throughput measurements

- **`comparison_benchmark.cu`** - General comparison framework for different implementations

### Validation Tests (`src/validation/`)

- **`fusion_validation.cu`** - Comprehensive correctness validation with PyTorch
- **`simple_fusion_validation.cu`** - Basic validation without external dependencies
- **`simple_baseline_comparison.cu`** - Simple performance comparisons

## Understanding the Results

### Baseline Comparison Output
```
=== Kernel Fusion vs Framework Baselines ===
Device: NVIDIA RTX A1000 Laptop GPU

--- Tensor Size: 1048576 elements ---
Implementation               Time(ms)     Speedup   Memory(GB/s)
----------------------------------------------------------------
manual_optimized                0.108        1.77x          108.5
naive_separate                  0.192        1.00x           61.2
our_fused                       0.111        1.73x          105.6
pytorch_optimized               0.211        0.91x           55.6
pytorch_separate                0.226        0.85x           51.9
thrust_transform                0.126        1.52x           92.8
```

- **our_fused**: Your kernel fusion implementation
- **naive_separate**: Worst-case baseline (separate kernels)
- **pytorch_separate**: Real-world PyTorch baseline
- **pytorch_optimized**: PyTorch with potential optimizations
- **thrust_transform**: Thrust library implementation
- **manual_optimized**: Hand-optimized CUDA implementation

### Performance Metrics
- **Time(ms)**: Execution time in milliseconds
- **Speedup**: Relative to `naive_separate` baseline
- **Memory(GB/s)**: Memory bandwidth utilization

## Build Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compatible compiler
- PyTorch/LibTorch (optional, for baseline comparisons)

## Advanced Usage

### Build Options
```bash
# Build with debug information
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Enable profiling support
cmake .. -DENABLE_NSIGHT=ON

# Enable address sanitizer
cmake .. -DENABLE_ASAN=ON
```

### Running Specific Tests
```bash
# Run main baseline comparison
./scripts/run_benchmarks.sh --baseline

# Run validation tests only
./scripts/run_benchmarks.sh --validation

# List all available benchmarks
./scripts/run_benchmarks.sh --list
```

## Troubleshooting

### Common Issues

1. **PyTorch not found**: Install PyTorch or use simple benchmarks
2. **CUDA out of memory**: Reduce tensor sizes in the source
3. **Compilation errors**: Check CUDA toolkit version compatibility

### Getting Help

- Check the documentation in `docs/` directory
- Review the validation guide: `docs/FUSION_VALIDATION_GUIDE.md`
- Use `./scripts/run_benchmarks.sh --help` for script options

## Contributing

When adding new benchmarks:

1. Place performance benchmarks in `src/performance/`
2. Place validation tests in `src/validation/`
3. Update `CMakeLists.txt` to include new targets
4. Add documentation to `docs/` if needed
5. Update this README with new benchmark descriptions

**Key Metrics:**
- Throughput (GOPS - Giga Operations Per Second)
- Memory bandwidth (GB/s)
- Execution time (milliseconds)

### 2. Memory Bandwidth Benchmark (`memory_benchmark.cu`)
- Measures peak memory bandwidth achievable
- Compares against theoretical GPU limits
- Tests basic operations: copy, add, triad, fused operations
- Analyzes memory access patterns

**Key Metrics:**
- Memory bandwidth efficiency (% of theoretical peak)
- Memory access pattern analysis
- Cache performance characteristics

### 3. Comparison Benchmark (`comparison_benchmark.cu`)
- Compares fused kernels vs separate kernel launches
- Quantifies memory traffic reduction benefits
- Analyzes kernel launch overhead
- Activation function overhead analysis

**Key Metrics:**
- Speedup of fused vs separate kernels
- Memory traffic reduction
- Kernel launch overhead

## Quick Start

### Basic Usage
```bash
cd examples/benchmarks
chmod +x run_benchmarks.sh
./run_benchmarks.sh
```

### Benchmark Modes
```bash
# Quick benchmark (smaller tensor sizes, fewer iterations)
./run_benchmarks.sh --quick

# Extensive benchmark (larger tensors, more iterations) 
./run_benchmarks.sh --extensive

# Save results to file
./run_benchmarks.sh --output results.txt

# Enable profiling
./run_benchmarks.sh --profile
```

### Manual Build and Run
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run individual benchmarks
./elementwise_benchmark
./memory_benchmark  
./comparison_benchmark
```

## Benchmark Configuration

### Tensor Sizes
Default sizes tested:
- 1K elements (1,024)
- 16K elements (16,384)
- 256K elements (262,144)
- 1M elements (1,048,576)
- 4M elements (4,194,304)
- 16M elements (16,777,216)
- 64M elements (67,108,864)

### Activation Functions Tested
- None (baseline)
- ReLU
- GELU
- SiLU (Swish)
- Sigmoid

### Performance Metrics

#### Throughput (GOPS)
```
GOPS = (Elements × Operations_Per_Element) / (Time_in_Seconds × 10^9)
```

#### Memory Bandwidth (GB/s)
```
Bandwidth = (Bytes_Transferred) / (Time_in_Seconds × 10^9)
```

#### Efficiency
```
Efficiency = (Measured_Bandwidth / Theoretical_Peak_Bandwidth) × 100%
```

## Expected Results

### Typical Performance Characteristics

#### Memory Bandwidth Efficiency
- **Copy operations**: 85-95% of peak bandwidth
- **Add operations**: 70-85% of peak bandwidth  
- **Fused operations**: 75-90% of peak bandwidth

#### Kernel Fusion Benefits
- **Memory traffic reduction**: 25-33% less memory traffic
- **Performance improvement**: 1.2-2.0x speedup vs separate kernels
- **Activation overhead**: 5-15% depending on complexity

#### Activation Function Overhead
- **ReLU**: ~5% overhead vs no activation
- **GELU**: ~15% overhead (most complex)
- **SiLU**: ~10% overhead
- **Sigmoid**: ~8% overhead

## Profiling and Analysis

### NVIDIA Nsight Systems
```bash
# Profile with Nsight Systems
nsys profile --output=profile ./elementwise_benchmark --quick

# View in Nsight Systems GUI
nsys-ui profile.nsys-rep
```

### NVIDIA Nsight Compute
```bash
# Detailed kernel analysis
ncu --output=kernel_analysis ./elementwise_benchmark --quick

# View in Nsight Compute GUI
ncu-ui kernel_analysis.ncu-rep
```

### Key Metrics to Monitor
- **Memory throughput**: Should approach theoretical limits
- **Occupancy**: Target >75% for memory-bound kernels
- **Cache hit rates**: L1 and L2 cache efficiency
- **Warp efficiency**: Branch divergence in activation functions

## Optimization Guidelines

### Memory-Bound Optimization
1. **Maximize bandwidth utilization**: Use appropriate block sizes
2. **Coalesce memory accesses**: Ensure contiguous access patterns
3. **Minimize memory traffic**: Kernel fusion reduces intermediate storage

### Compute-Bound Optimization  
1. **Optimize activation functions**: Use lookup tables for complex functions
2. **Minimize divergence**: Avoid branching in activation logic
3. **Use appropriate precision**: Float32 vs Float64 trade-offs

## Hardware-Specific Considerations

### GPU Architecture Differences
- **Turing/Ampere**: Higher memory bandwidth, tensor cores
- **Pascal/Volta**: Different cache hierarchies
- **Compute capability**: Affects available instructions

### Memory Hierarchy
- **Global memory**: Primary bottleneck for elementwise operations
- **L2 cache**: Benefits repeated access patterns
- **Shared memory**: Limited benefit for elementwise operations

## Troubleshooting

### Common Issues

**Low bandwidth efficiency**:
- Check memory access patterns
- Verify optimal block/grid sizes
- Look for memory bank conflicts

**Poor kernel fusion benefits**:
- Verify intermediate results aren't cached
- Check for memory coalescing issues
- Profile memory traffic patterns

**High activation overhead**:
- Profile branch efficiency
- Consider lookup tables for complex functions
- Verify compiler optimizations enabled

### Debug Mode
```bash
# Build with debug information
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON

# Run with memory checking
cuda-memcheck ./elementwise_benchmark --quick
```

## Integration with Main Project

The benchmarks can be integrated into the main project build:

```cmake
# In main CMakeLists.txt
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
if(BUILD_BENCHMARKS)
    add_subdirectory(examples/benchmarks)
endif()
```

```bash
# Build with benchmarks
cmake .. -DBUILD_BENCHMARKS=ON
make benchmarks
```

## Contributing

When adding new kernels or optimizations:

1. **Add benchmark coverage**: Include new operations in benchmark suite
2. **Establish baselines**: Document expected performance characteristics  
3. **Profile regressions**: Ensure optimizations don't hurt other cases
4. **Update documentation**: Keep performance expectations current

## Performance Baseline

Results on NVIDIA RTX 3080 (example):
- **Peak Memory Bandwidth**: ~760 GB/s
- **Fused Add+ReLU**: ~650 GB/s (85% efficiency)
- **Kernel Fusion Speedup**: 1.4x vs separate kernels
- **GELU Overhead**: +12% vs no activation
