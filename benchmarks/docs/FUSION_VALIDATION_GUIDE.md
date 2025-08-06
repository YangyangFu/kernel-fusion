# Kernel Fusion Validation Guide

This guide describes the methodology and best practices for validating kernel fusion implementations.

## Validation Philosophy

Kernel fusion optimizations must maintain **numerical equivalence** with baseline implementations while providing performance improvements. This requires comprehensive testing across multiple dimensions:

1. **Numerical Accuracy**: Fused results must match unfused within acceptable tolerance
2. **Edge Case Handling**: Proper behavior for special values (NaN, Inf, zero)
3. **Gradient Correctness**: Backpropagation must produce correct gradients
4. **Memory Safety**: No buffer overruns or invalid memory access
5. **Performance Consistency**: Speedups must be reproducible across runs

## Validation Levels

### Level 1: Basic Correctness
- **Test Scope**: Individual operations, simple fusion patterns
- **Tolerance**: `1e-6` for FP32, `1e-3` for FP16
- **Coverage**: Common input ranges, basic edge cases
- **Tools**: `simple_fusion_validation.cu`

### Level 2: Comprehensive Validation  
- **Test Scope**: Complex fusion chains, all supported operations
- **Tolerance**: Relative error < 0.01%, absolute error < `1e-5`
- **Coverage**: Full input space, all edge cases, gradient validation
- **Tools**: `fusion_validation.cu`

### Level 3: Production Validation
- **Test Scope**: Real workloads, stress testing, long-running validation
- **Tolerance**: Statistical analysis of error distributions
- **Coverage**: Production data distributions, thermal stress testing
- **Tools**: Extended validation suites, continuous integration

## Test Methodology

### Numerical Comparison

```cpp
bool validate_results(float* fused, float* baseline, int n, 
                     float rel_tol = 1e-5f, float abs_tol = 1e-6f) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(fused[i] - baseline[i]);
        float rel_error = diff / (fabsf(baseline[i]) + 1e-8f);
        
        if (diff > abs_tol && rel_error > rel_tol) {
            printf("Validation failed at index %d: fused=%.8f, baseline=%.8f, "
                   "abs_error=%.8f, rel_error=%.8f\n", 
                   i, fused[i], baseline[i], diff, rel_error);
            return false;
        }
    }
    return true;
}
```

### Edge Case Testing

#### Special Values
- **NaN Propagation**: `NaN` inputs must produce `NaN` outputs
- **Infinity Handling**: Operations with `±Inf` must follow IEEE 754
- **Zero Handling**: Signed zeros, denormal numbers
- **Saturation**: Values at type limits (FP16 overflow to Inf)

#### Boundary Conditions
- **Empty Tensors**: Zero-size inputs should not crash
- **Single Element**: Minimal input sizes
- **Large Tensors**: Memory allocation limits, integer overflow
- **Misaligned Memory**: Non-aligned input pointers

### Input Data Generation

#### Random Testing
```cpp
void generate_random_data(float* data, int n, float min_val = -10.0f, float max_val = 10.0f) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42); // Fixed seed for reproducibility
    curandGenerateUniform(gen, data, n);
    
    // Scale to desired range
    thrust::transform(thrust::device, data, data + n, data,
                     [=] __device__ (float x) { return min_val + x * (max_val - min_val); });
    curandDestroyGenerator(gen);
}
```

#### Structured Testing
- **Linear Sequences**: `[0, 1, 2, ..., n-1]`
- **Geometric Sequences**: `[1, 2, 4, 8, ...]`
- **Alternating Signs**: `[1, -1, 1, -1, ...]`
- **Extreme Values**: Near overflow/underflow thresholds

### Gradient Validation

For operations supporting backpropagation:

```cpp
bool validate_gradients(OpWrapper op, float* input, float* grad_output, int n) {
    // Numerical gradient via finite differences
    float eps = 1e-4f;
    std::vector<float> numerical_grad(n);
    
    for (int i = 0; i < n; i++) {
        input[i] += eps;
        float forward_plus = op.forward(input);
        input[i] -= 2 * eps;  
        float forward_minus = op.forward(input);
        input[i] += eps; // restore
        
        numerical_grad[i] = (forward_plus - forward_minus) / (2 * eps);
    }
    
    // Compare with analytical gradient
    std::vector<float> analytical_grad(n);
    op.backward(grad_output, analytical_grad.data());
    
    return validate_results(analytical_grad.data(), numerical_grad.data(), n);
}
```

## Test Cases

### Activation Functions

#### ReLU Validation
```cpp
void test_relu_validation() {
    const int n = 1024;
    float* input = allocate_gpu_memory<float>(n);
    float* fused_output = allocate_gpu_memory<float>(n);
    float* baseline_output = allocate_gpu_memory<float>(n);
    
    // Test case 1: Random values
    generate_random_data(input, n, -5.0f, 5.0f);
    test_relu_fusion(input, fused_output, n);
    test_relu_baseline(input, baseline_output, n);
    assert(validate_results(fused_output, baseline_output, n));
    
    // Test case 2: Edge cases
    std::vector<float> edge_cases = {0.0f, -0.0f, INFINITY, -INFINITY, NAN};
    test_edge_cases(edge_cases, relu_fusion, relu_baseline);
    
    // Test case 3: Gradient validation
    generate_random_data(input, n, -2.0f, 2.0f);
    assert(validate_gradients(relu_op, input, ones(n), n));
}
```

#### Sigmoid Validation
```cpp
void test_sigmoid_validation() {
    // Test numerical stability for large inputs
    std::vector<float> large_inputs = {50.0f, 100.0f, -50.0f, -100.0f};
    for (float x : large_inputs) {
        float fused = sigmoid_fused(x);
        float baseline = sigmoid_baseline(x);
        
        // Sigmoid should saturate, not overflow
        assert(fused >= 0.0f && fused <= 1.0f);
        assert(baseline >= 0.0f && baseline <= 1.0f);
        assert(fabsf(fused - baseline) < 1e-6f);
    }
}
```

### Fusion Chain Validation

```cpp
void test_activation_chain() {
    // Test: ReLU → Sigmoid → Tanh chain
    const int n = 2048;
    float* input = allocate_gpu_memory<float>(n);
    generate_random_data(input, n, -3.0f, 3.0f);
    
    // Fused implementation
    float* fused_output = allocate_gpu_memory<float>(n);
    relu_sigmoid_tanh_fused<<<blocks, threads>>>(input, fused_output, n);
    
    // Sequential baseline
    float* temp1 = allocate_gpu_memory<float>(n);
    float* temp2 = allocate_gpu_memory<float>(n);
    float* baseline_output = allocate_gpu_memory<float>(n);
    
    relu_kernel<<<blocks, threads>>>(input, temp1, n);
    sigmoid_kernel<<<blocks, threads>>>(temp1, temp2, n);
    tanh_kernel<<<blocks, threads>>>(temp2, baseline_output, n);
    
    assert(validate_results(fused_output, baseline_output, n));
    
    // Performance validation
    assert(measure_speedup(fused_impl, baseline_impl) > 1.5f);
}
```

## Performance Validation

### Speedup Requirements
- **Single Operation**: Minimum 1.2x speedup
- **2-3 Operations**: Minimum 1.5x speedup  
- **4+ Operations**: Minimum 2.0x speedup
- **Complex Chains**: Target 3.0x+ speedup

### Memory Efficiency
```cpp
float measure_memory_efficiency(KernelFunc kernel, int data_size) {
    size_t theoretical_bytes = data_size * sizeof(float) * 2; // read + write
    
    CudaTimer timer;
    timer.start();
    kernel(data, data_size);
    cudaDeviceSynchronize();
    timer.stop();
    
    float bandwidth_gbps = (theoretical_bytes / 1e9f) / (timer.elapsed_ms() / 1000.0f);
    float peak_bandwidth = get_gpu_peak_bandwidth();
    
    return bandwidth_gbps / peak_bandwidth; // Efficiency ratio
}
```

### Consistency Testing
```cpp
void test_performance_consistency() {
    const int num_runs = 100;
    std::vector<float> timings(num_runs);
    
    for (int i = 0; i < num_runs; i++) {
        timings[i] = benchmark_fusion_kernel();
    }
    
    float mean = calculate_mean(timings);
    float stddev = calculate_stddev(timings);
    float cv = stddev / mean; // Coefficient of variation
    
    // Performance should be consistent (CV < 5%)
    assert(cv < 0.05f);
}
```

## Continuous Integration

### Automated Testing
- **Nightly Builds**: Full validation suite on multiple GPUs
- **PR Validation**: Core test suite for code changes
- **Performance Regression**: Track speedup metrics over time
- **Cross-Platform**: Validate on different CUDA versions, drivers

### Test Configuration
```yaml
# CI Pipeline Configuration
validation_levels:
  - quick: basic correctness, ~5 minutes
  - standard: comprehensive validation, ~30 minutes  
  - extended: stress testing, gradient validation, ~2 hours

gpu_targets:
  - V100: CUDA 11.0+
  - A100: CUDA 11.4+  
  - RTX3090: CUDA 11.2+

precision_modes:
  - FP32: Primary validation
  - FP16: Performance mode validation
  - Mixed: Production workload validation
```

## Debugging Failed Validations

### Error Analysis Tools

1. **Element-wise Diff Analysis**
   ```bash
   ./fusion_validation --verbose --save-diffs
   # Saves per-element differences for analysis
   ```

2. **Numerical Precision Investigation**
   ```cpp
   void debug_precision_loss() {
       // Use higher precision reference
       double* double_baseline = compute_reference_double_precision();
       float* float_baseline = compute_reference_float();
       float* fused_result = compute_fused();
       
       // Compare error sources
       analyze_truncation_error(double_baseline, float_baseline);
       analyze_fusion_error(float_baseline, fused_result);
   }
   ```

3. **Memory Access Pattern Validation**
   ```bash
   # Use compute-sanitizer to detect memory errors
   compute-sanitizer --tool memcheck ./fusion_validation
   ```

### Common Failure Patterns

1. **Accumulation Error**: Precision loss in reduction operations
2. **Branch Divergence**: Conditional logic causing incorrect paths
3. **Shared Memory Conflicts**: Race conditions in fused kernels
4. **Alignment Issues**: Misaligned memory access affecting results

## Best Practices

### Implementation Guidelines
- **Maintain Numerical Stability**: Use appropriate precision and algorithms
- **Handle Edge Cases**: Don't assume "normal" inputs
- **Validate Early**: Test individual components before complex fusion
- **Document Assumptions**: Clear contracts for input ranges, precision

### Testing Guidelines  
- **Reproducible Tests**: Fixed random seeds, deterministic execution
- **Comprehensive Coverage**: Test all code paths and edge cases
- **Performance Baselines**: Establish and maintain performance expectations
- **Cross-Validation**: Multiple reference implementations when possible

### Documentation Requirements
- **Error Tolerance**: Document expected precision characteristics
- **Performance Targets**: Specify minimum speedup requirements
- **Edge Case Behavior**: Define handling of special values
- **Usage Constraints**: Input size limits, alignment requirements

This validation methodology ensures that kernel fusion optimizations provide both correctness and performance benefits while maintaining the reliability required for production use.
