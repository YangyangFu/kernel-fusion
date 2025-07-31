# CUDA Debugging Guide for PyTorch Extensions

This guide provides comprehensive strategies for debugging CUDA kernels implemented with PyTorch's JIT compilation.

## 1. Compilation-Time Debugging

### Debug Compilation Flags
```python
# Debug mode compilation
cuda_flags = ['-g', '-G', '-O0', '-lcub', '--ptxas-options=-v']
cpp_flags = ['-g', '-O0']

# Release mode compilation  
cuda_flags = ['-O3', '--use_fast_math', '-lcub']
cpp_flags = ['-O3']
```

**Key flags:**
- `-g`: Generate debug information
- `-G`: Generate device debug information  
- `-O0`: Disable optimizations for debugging
- `--ptxas-options=-v`: Verbose PTX assembler output

## 2. Runtime Error Detection

### CUDA Error Checking
Always synchronize and check for errors after kernel launches:

```python
# After kernel execution
torch.cuda.synchronize()
# Any CUDA errors will be raised as RuntimeError
```

### Memory Access Debugging
Use NVIDIA's compute-sanitizer tool:

```bash
# Install compute-sanitizer (comes with CUDA toolkit)
compute-sanitizer --tool memcheck python your_script.py
```

## 3. Debugging Strategies

### A. Print Debugging in CUDA Kernels

```cuda
__global__ void debug_kernel(...) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Debug prints (be careful with performance!)
    if (tid == 0) {
        printf("Block %d, Thread %d: value = %f\n", blockIdx.x, threadIdx.x, some_value);
    }
}
```

### B. Tensor Validation

```python
def validate_tensor(tensor, name="Tensor"):
    """Comprehensive tensor validation"""
    print(f"{name}: shape={tensor.shape}, device={tensor.device}")
    print(f"  Min={tensor.min():.6f}, Max={tensor.max():.6f}")
    print(f"  Mean={tensor.mean():.6f}, Std={tensor.std():.6f}")
    
    # Check for NaN/Inf
    if torch.isnan(tensor).any():
        print(f"  ⚠️  Contains NaN values!")
    if torch.isinf(tensor).any():
        print(f"  ⚠️  Contains Inf values!")
```

### C. Numerical Accuracy Testing

```python
def compare_implementations(cuda_func, reference_func, inputs, rtol=1e-5):
    """Compare CUDA implementation with reference"""
    cuda_output = cuda_func(*inputs)
    ref_output = reference_func(*inputs)
    
    if torch.allclose(cuda_output, ref_output, rtol=rtol):
        print("✓ Implementations match within tolerance")
    else:
        diff = torch.abs(cuda_output - ref_output)
        print(f"✗ Max difference: {diff.max():.2e}")
        print(f"✗ Mean difference: {diff.mean():.2e}")
```

## 4. Performance Profiling

### A. Basic Timing
```python
def profile_kernel(func, *args, num_runs=100):
    # Warmup
    for _ in range(10):
        func(*args)
    torch.cuda.synchronize()
    
    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(num_runs):
        start.record()
        func(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return np.mean(times), np.std(times)
```

### B. Using PyTorch Profiler
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    result = your_cuda_function(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### C. NVIDIA Nsight Systems
```bash
# Profile your application
nsys profile --stats=true python your_script.py
```

## 5. Memory Debugging

### A. Memory Leaks
```python
def check_memory_leak(func, inputs, iterations=10):
    """Check for memory leaks"""
    initial_memory = torch.cuda.memory_allocated()
    
    for i in range(iterations):
        result = func(*inputs)
        current_memory = torch.cuda.memory_allocated()
        print(f"Iteration {i}: {current_memory - initial_memory} bytes leaked")
        
        # Force garbage collection
        del result
        torch.cuda.empty_cache()
```

### B. Memory Access Patterns
Use compute-sanitizer to detect:
- Out-of-bounds memory access
- Uninitialized memory reads
- Race conditions

## 6. Specific Debugging for LayerNorm

### A. Mathematical Correctness
```python
def debug_layernorm_math(input_tensor, eps=1e-5):
    """Debug LayerNorm mathematical operations"""
    
    # Manual computation for comparison
    mean = input_tensor.mean(dim=-1, keepdim=True)
    var = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    normalized = (input_tensor - mean) / std
    
    print(f"Input stats: mean={input_tensor.mean():.6f}, std={input_tensor.std():.6f}")
    print(f"Computed mean: {mean.mean():.6f}")
    print(f"Computed std: {std.mean():.6f}")
    print(f"Normalized stats: mean={normalized.mean():.6f}, std={normalized.std():.6f}")
    
    return normalized, mean, std
```

### B. Block and Thread Analysis
```python
def analyze_launch_config(batch_size, seq_len, hidden_size, block_size=256):
    """Analyze CUDA launch configuration"""
    
    total_positions = batch_size * seq_len
    print(f"Total sequence positions: {total_positions}")
    print(f"Threads per block: {block_size}")
    print(f"Grid dimensions: ({batch_size}, {seq_len})")
    print(f"Elements per position: {hidden_size}")
    print(f"Threads per position: {min(block_size, hidden_size)}")
    
    if hidden_size > block_size:
        iterations = (hidden_size + block_size - 1) // block_size
        print(f"⚠️  Elements > threads, need {iterations} iterations per thread")
```

## 7. Common Debugging Issues and Solutions

### Issue 1: Compilation Errors
**Symptoms**: Compilation fails with cryptic error messages
**Solutions**:
- Use `-v` flag for verbose compilation
- Check CUDA toolkit version compatibility
- Verify include paths and library linkage

### Issue 2: Runtime Crashes
**Symptoms**: Segmentation faults, CUDA errors
**Solutions**:
- Add bounds checking in kernels
- Use compute-sanitizer
- Validate input tensor properties

### Issue 3: Incorrect Results
**Symptoms**: Output doesn't match reference implementation
**Solutions**:
- Add debug prints in kernel
- Test with simple inputs
- Compare intermediate results
- Check data types and precision

### Issue 4: Performance Issues
**Symptoms**: Kernel slower than expected
**Solutions**:
- Profile with nsys or PyTorch profiler
- Analyze memory access patterns
- Check occupancy and register usage
- Optimize memory coalescing

## 8. Debugging Workflow

1. **Start Simple**: Test with small, known inputs
2. **Validate Compilation**: Ensure kernel compiles without warnings
3. **Check Basic Functionality**: Verify kernel runs without crashes
4. **Validate Correctness**: Compare against reference implementation
5. **Test Edge Cases**: Empty tensors, large values, small values
6. **Profile Performance**: Identify bottlenecks
7. **Stress Test**: Long-running tests to find memory leaks

## 9. Useful Tools Summary

- **compute-sanitizer**: Memory access debugging
- **nsys**: Performance profiling
- **PyTorch Profiler**: High-level profiling
- **nvidia-smi**: Real-time GPU monitoring
- **gdb**: CPU-side debugging
- **cuda-gdb**: GPU-side debugging (limited support in JIT context)

## 10. Environment Setup for Debugging

```bash
# Set environment variables for better debugging
export CUDA_LAUNCH_BLOCKING=1  # Synchronous kernel launches
export TORCH_USE_CUDA_DSA=1    # CUDA device-side assertions
export TORCH_CUDA_DEBUG=1      # Additional CUDA debugging
```

Remember: Debugging CUDA kernels is an iterative process. Start with simple test cases and gradually increase complexity while maintaining validation at each step.
