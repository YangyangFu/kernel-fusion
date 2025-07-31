"""
Test suite for LayerNorm CUDA implementation
"""

import torch
import pytest
import time
import numpy as np

try:
    from kernel_fusion.kernels.cuda_layernorm import layernorm, CUDALayerNorm
    from kernel_fusion.kernels.cpu_layernorm import cpu_layernorm
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA LayerNorm not available, tests will be skipped")


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA LayerNorm not available")
class TestLayerNorm:
    """Test suite for LayerNorm implementation"""
    
    def test_layernorm_shapes(self):
        """Test that LayerNorm preserves input shapes"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test different input shapes
        shapes = [(2, 10, 128), (1, 5, 64), (4, 20, 256)]
        
        for shape in shapes:
            x = torch.randn(shape, device=device)
            weight = torch.randn(shape[-1], device=device)
            bias = torch.randn(shape[-1], device=device)
            
            output = layernorm(x, weight, bias)
            
            assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
            assert output.device == x.device, "Device mismatch"
            assert output.dtype == x.dtype, "Dtype mismatch"
    
    def test_layernorm_correctness(self):
        """Test LayerNorm correctness against PyTorch reference"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test data
        batch_size, seq_len, hidden_size = 2, 8, 64
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        weight = torch.randn(hidden_size, device=device)
        bias = torch.randn(hidden_size, device=device)
        
        # Our implementation
        our_output = layernorm(x, weight, bias)
        
        # PyTorch reference
        layer_norm = torch.nn.LayerNorm(hidden_size, elementwise_affine=True, device=device)
        layer_norm.weight.data = weight.clone()
        layer_norm.bias.data = bias.clone()
        pytorch_output = layer_norm(x)
        
        # Check correctness
        assert torch.allclose(our_output, pytorch_output, atol=1e-4, rtol=1e-3), \
            f"Max diff: {torch.max(torch.abs(our_output - pytorch_output))}"
    
    def test_layernorm_without_affine(self):
        """Test LayerNorm without weight and bias"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        x = torch.randn(2, 8, 64, device=device)
        output = layernorm(x)  # No weight/bias
        
        # Check that mean is close to 0 and std is close to 1
        mean = output.mean(dim=-1)
        std = output.std(dim=-1, unbiased=False)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), \
            f"Mean not zero: {mean.abs().max()}"
        assert torch.allclose(std, torch.ones_like(std), atol=1e-4), \
            f"Std not one: {(std - 1).abs().max()}"
    
    def test_layernorm_numerical_stability(self):
        """Test LayerNorm with extreme values"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test with large values
        x_large = torch.randn(2, 8, 64, device=device) * 1e6
        output_large = layernorm(x_large)
        assert torch.isfinite(output_large).all(), "Output contains non-finite values"
        
        # Test with small values
        x_small = torch.randn(2, 8, 64, device=device) * 1e-6
        output_small = layernorm(x_small)
        assert torch.isfinite(output_small).all(), "Output contains non-finite values"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_layernorm_performance(self):
        """Benchmark LayerNorm performance"""
        device = torch.device('cuda')
        
        # Test configuration
        batch_size, seq_len, hidden_size = 32, 512, 1024
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        weight = torch.randn(hidden_size, device=device)
        bias = torch.randn(hidden_size, device=device)
        
        # Warmup
        for _ in range(10):
            _ = layernorm(x, weight, bias)
        torch.cuda.synchronize()
        
        # Benchmark our implementation
        start = time.time()
        for _ in range(100):
            _ = layernorm(x, weight, bias)
        torch.cuda.synchronize()
        our_time = (time.time() - start) / 100
        
        # Benchmark PyTorch
        layer_norm = torch.nn.LayerNorm(hidden_size, device=device)
        layer_norm.weight.data = weight.clone()
        layer_norm.bias.data = bias.clone()
        
        start = time.time()
        for _ in range(100):
            _ = layer_norm(x)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / 100
        
        print(f"Our LayerNorm: {our_time*1000:.2f}ms")
        print(f"PyTorch LayerNorm: {pytorch_time*1000:.2f}ms")
        print(f"Speedup: {pytorch_time/our_time:.2f}x")
        
        # Our implementation should be competitive
        # (This assertion might need adjustment based on actual performance)
        assert our_time < pytorch_time * 2, "Our implementation is too slow"


def benchmark_layernorm_sizes():
    """Benchmark LayerNorm across different input sizes"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    device = torch.device('cuda')
    sizes = [
        (16, 128, 512),   # Small
        (32, 256, 768),   # Medium
        (64, 512, 1024),  # Large
        (128, 1024, 2048) # Very large
    ]
    
    print("LayerNorm Performance Benchmark")
    print("=" * 50)
    print(f"{'Size':<20} {'Our Time':<12} {'PyTorch':<12} {'Speedup':<10}")
    print("-" * 50)
    
    for batch_size, seq_len, hidden_size in sizes:
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        weight = torch.randn(hidden_size, device=device)
        bias = torch.randn(hidden_size, device=device)
        
        # Benchmark our implementation
        times = []
        for _ in range(20):
            start = time.time()
            _ = layernorm(x, weight, bias)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        our_time = np.mean(times[5:]) * 1000  # Skip first 5 for warmup
        
        # Benchmark PyTorch
        layer_norm = torch.nn.LayerNorm(hidden_size, device=device)
        times = []
        for _ in range(20):
            start = time.time()
            _ = layer_norm(x)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        pytorch_time = np.mean(times[5:]) * 1000
        
        speedup = pytorch_time / our_time
        size_str = f"{batch_size}x{seq_len}x{hidden_size}"
        
        print(f"{size_str:<20} {our_time:<12.2f} {pytorch_time:<12.2f} {speedup:<10.2f}x")


if __name__ == "__main__":
    # Run benchmarks
    benchmark_layernorm_sizes()
    
    # Run tests
    pytest.main([__file__, "-v"])
