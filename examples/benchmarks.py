"""
Benchmarking utilities and comprehensive performance tests.
"""

import torch
import time
import gc
from typing import Callable, List, Dict, Any
import kernel_fusion as kf


class BenchmarkRunner:
    """Utility class for running performance benchmarks."""
    
    def __init__(self, warmup_iters: int = 10, bench_iters: int = 100):
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
    
    def benchmark_function(
        self,
        func: Callable,
        *args,
        device: torch.device = None,
        **kwargs
    ) -> Dict[str, float]:
        """Benchmark a function and return timing statistics."""
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Warmup
        for _ in range(self.warmup_iters):
            result = func(*args, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize()
        
        # Clear cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Benchmark
        times = []
        
        for _ in range(self.bench_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": (sum((t - sum(times) / len(times))**2 for t in times) / len(times))**0.5
        }
    
    def compare_functions(
        self,
        functions: Dict[str, Callable],
        *args,
        device: torch.device = None,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple functions and return relative performance."""
        
        results = {}
        
        for name, func in functions.items():
            print(f"Benchmarking {name}...")
            results[name] = self.benchmark_function(func, *args, device=device, **kwargs)
        
        return results


def benchmark_elementwise_operations():
    """Comprehensive benchmark of elementwise operations."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking elementwise operations on {device}")
    
    sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]
    runner = BenchmarkRunner()
    
    for size in sizes:
        print(f"\nTesting size: {size}")
        
        # Create test tensors
        a = torch.randn(size, device=device, dtype=torch.float32)
        b = torch.randn(size, device=device, dtype=torch.float32)
        
        # Define functions to compare
        functions = {
            "fused_add_relu": lambda: kf.ops.elementwise_add_relu(a, b),
            "separate_add_relu": lambda: torch.relu(a + b),
            "fused_mul_tanh": lambda: kf.ops.elementwise_mul_tanh(a, b),
            "separate_mul_tanh": lambda: torch.tanh(a * b)
        }
        
        results = runner.compare_functions(functions, device=device)
        
        # Print results
        for name, stats in results.items():
            print(f"  {name}: {stats['mean_ms']:.3f}ms ± {stats['std_ms']:.3f}ms")
        
        # Calculate speedups
        if "fused_add_relu" in results and "separate_add_relu" in results:
            speedup = results["separate_add_relu"]["mean_ms"] / results["fused_add_relu"]["mean_ms"]
            print(f"  Add+ReLU speedup: {speedup:.2f}x")
        
        if "fused_mul_tanh" in results and "separate_mul_tanh" in results:
            speedup = results["separate_mul_tanh"]["mean_ms"] / results["fused_mul_tanh"]["mean_ms"]
            print(f"  Mul+Tanh speedup: {speedup:.2f}x")


def benchmark_reduction_operations():
    """Benchmark reduction operations."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nBenchmarking reduction operations on {device}")
    
    sizes = [(2048, 2048), (4096, 1024), (1024, 4096)]
    runner = BenchmarkRunner()
    
    for size in sizes:
        print(f"\nTesting size: {size}")
        
        input_tensor = torch.randn(size, device=device, dtype=torch.float32)
        
        functions = {
            "fused_sum_squared": lambda: kf.ops.reduce_sum_squared(input_tensor, dim=-1),
            "separate_sum_squared": lambda: torch.sum(input_tensor * input_tensor, dim=-1),
            "fused_mean_abs": lambda: kf.ops.reduce_mean_abs(input_tensor, dim=-1),
            "separate_mean_abs": lambda: torch.mean(torch.abs(input_tensor), dim=-1)
        }
        
        results = runner.compare_functions(functions, device=device)
        
        for name, stats in results.items():
            print(f"  {name}: {stats['mean_ms']:.3f}ms ± {stats['std_ms']:.3f}ms")


def benchmark_fusion_operations():
    """Benchmark complex fusion operations."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nBenchmarking fusion operations on {device}")
    
    batch_size, seq_len, hidden_dim = 32, 512, 768
    runner = BenchmarkRunner()
    
    # Test layer norm + relu
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    weight = torch.randn(hidden_dim, device=device)
    bias = torch.randn(hidden_dim, device=device)
    
    functions = {
        "fused_layer_norm_relu": lambda: kf.ops.fused_layer_norm_relu(
            input_tensor, (hidden_dim,), weight, bias
        ),
        "separate_layer_norm_relu": lambda: torch.relu(
            torch.nn.functional.layer_norm(input_tensor, (hidden_dim,), weight, bias)
        )
    }
    
    print("Layer Norm + ReLU:")
    results = runner.compare_functions(functions, device=device)
    
    for name, stats in results.items():
        print(f"  {name}: {stats['mean_ms']:.3f}ms ± {stats['std_ms']:.3f}ms")
    
    if len(results) == 2:
        names = list(results.keys())
        speedup = results[names[1]]["mean_ms"] / results[names[0]]["mean_ms"]
        print(f"  Speedup: {speedup:.2f}x")
    
    # Test attention scores
    query = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    key = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    scale = 1.0 / (hidden_dim ** 0.5)
    
    functions = {
        "fused_attention": lambda: kf.ops.fused_attention_score(query, key, scale),
        "separate_attention": lambda: torch.matmul(query, key.transpose(-2, -1)) * scale
    }
    
    print("\nAttention Scores:")
    results = runner.compare_functions(functions, device=device)
    
    for name, stats in results.items():
        print(f"  {name}: {stats['mean_ms']:.3f}ms ± {stats['std_ms']:.3f}ms")


def memory_usage_comparison():
    """Compare memory usage of fused vs separate operations."""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return
    
    device = torch.device("cuda")
    print(f"\nMemory usage comparison on {device}")
    
    size = (4096, 4096)
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    # Measure fused operation memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    result_fused = kf.ops.elementwise_add_relu(a, b)
    fused_memory = torch.cuda.max_memory_allocated()
    
    del result_fused
    torch.cuda.empty_cache()
    
    # Measure separate operations memory
    torch.cuda.reset_peak_memory_stats()
    
    temp = a + b
    result_separate = torch.relu(temp)
    separate_memory = torch.cuda.max_memory_allocated()
    
    print(f"Fused operation memory: {fused_memory / 1024**2:.2f} MB")
    print(f"Separate operations memory: {separate_memory / 1024**2:.2f} MB")
    print(f"Memory savings: {(separate_memory - fused_memory) / 1024**2:.2f} MB")
    print(f"Memory efficiency: {separate_memory / fused_memory:.2f}x")


if __name__ == "__main__":
    print("=== Kernel Fusion Comprehensive Benchmarks ===")
    
    benchmark_elementwise_operations()
    benchmark_reduction_operations()
    benchmark_fusion_operations()
    memory_usage_comparison()
    
    print("\nBenchmark complete!")
