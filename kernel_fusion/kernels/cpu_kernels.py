"""
CPU-optimized kernels and utilities for MacBook Pro development
"""

import torch
import numpy as np
import time
from typing import Tuple, Optional


def cpu_fused_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    CPU-optimized attention implementation for development and testing
    
    Args:
        q: Query tensor [batch, heads, seq_len, d_head]
        k: Key tensor [batch, heads, seq_len, d_head]
        v: Value tensor [batch, heads, seq_len, d_head]
        mask: Optional attention mask
        scale: Optional scaling factor (defaults to 1/sqrt(d_head))
    
    Returns:
        Output tensor [batch, heads, seq_len, d_head]
    """
    batch, n_heads, seq_len, d_head = q.shape
    
    if scale is None:
        scale = 1.0 / (d_head ** 0.5)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, v)
    
    return output


def optimized_cpu_attention(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    chunk_size: int = 64
) -> torch.Tensor:
    """
    Memory-efficient chunked attention for CPU
    
    Args:
        q, k, v: Attention tensors
        chunk_size: Size of chunks to process
    
    Returns:
        Attention output
    """
    batch, n_heads, seq_len, d_head = q.shape
    scale = 1.0 / (d_head ** 0.5)
    
    output = torch.zeros_like(q)
    
    # Process in chunks to reduce memory usage
    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)
        q_chunk = q[:, :, i:end_i, :]
        
        # Compute scores for this chunk
        scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply to values
        output[:, :, i:end_i, :] = torch.matmul(attn_weights, v)
    
    return output


class CPUKernelBenchmark:
    """Benchmarking utilities for CPU kernels"""
    
    @staticmethod
    def benchmark_function(func, *args, warmup=5, repeat=20):
        """Benchmark a function on CPU"""
        # Warmup
        for _ in range(warmup):
            _ = func(*args)
        
        # Benchmark
        times = []
        for _ in range(repeat):
            start = time.time()
            _ = func(*args)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
    
    @staticmethod
    def compare_implementations(implementations, *args):
        """Compare multiple implementations"""
        results = {}
        for name, func in implementations.items():
            results[name] = CPUKernelBenchmark.benchmark_function(func, *args)
        return results


def profile_attention_kernels():
    """Profile different attention implementations"""
    # Test configurations
    configs = [
        (1, 8, 128, 64),   # Small
        (1, 8, 512, 64),   # Medium
        (1, 8, 1024, 64),  # Large
    ]
    
    implementations = {
        'standard': cpu_fused_attention,
        'chunked': lambda q, k, v: optimized_cpu_attention(q, k, v, chunk_size=64),
        'pytorch_sdpa': lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v)
    }
    
    for batch, heads, seq_len, d_head in configs:
        print(f"\nTesting {batch}x{heads}x{seq_len}x{d_head}:")
        
        # Create test data
        q = torch.randn(batch, heads, seq_len, d_head)
        k = torch.randn(batch, heads, seq_len, d_head)
        v = torch.randn(batch, heads, seq_len, d_head)
        
        # Benchmark
        results = CPUKernelBenchmark.compare_implementations(implementations, q, k, v)
        
        for name, stats in results.items():
            print(f"  {name:15s}: {stats['mean']:6.2f}ms ± {stats['std']:5.2f}ms")


if __name__ == "__main__":
    print("Running CPU kernel profiling...")
    profile_attention_kernels()
