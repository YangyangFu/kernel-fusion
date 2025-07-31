import pytest
import torch
import numpy as np
from kernel_fusion import ops, kernels

class TestElementwiseOperations:
    """Test elementwise fusion operations."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def random_tensors(self, device):
        torch.manual_seed(42)
        a = torch.randn(1024, 512, device=device, dtype=torch.float32)
        b = torch.randn(1024, 512, device=device, dtype=torch.float32)
        return a, b
    
    def test_elementwise_add_relu(self, random_tensors):
        a, b = random_tensors
        
        # Test high-level API
        result_ops = ops.elementwise_add_relu(a, b)
        
        # Compare with PyTorch reference
        reference = torch.relu(a + b)
        
        assert torch.allclose(result_ops, reference, rtol=1e-5, atol=1e-5)
    
    def test_elementwise_mul_tanh(self, random_tensors):
        a, b = random_tensors
        
        # Test high-level API
        result_ops = ops.elementwise_mul_tanh(a, b)
        
        # Compare with PyTorch reference
        reference = torch.tanh(a * b)
        
        assert torch.allclose(result_ops, reference, rtol=1e-5, atol=1e-5)
    
    def test_fused_bias_gelu(self, device):
        torch.manual_seed(42)
        input_tensor = torch.randn(64, 128, device=device, dtype=torch.float32)
        bias = torch.randn(128, device=device, dtype=torch.float32)
        
        # Test high-level API
        result_ops = ops.fused_bias_gelu(input_tensor, bias)
        
        # Compare with PyTorch reference
        reference = torch.nn.functional.gelu(input_tensor + bias)
        
        assert torch.allclose(result_ops, reference, rtol=1e-4, atol=1e-4)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_kernel_direct_access(self, random_tensors):
        """Test direct kernel access for CUDA tensors."""
        a, b = random_tensors
        
        if a.is_cuda:
            # Test direct kernel access
            result_kernel = kernels.elementwise.add_relu(a, b)
            
            # Compare with high-level API
            result_ops = ops.elementwise_add_relu(a, b)
            
            assert torch.allclose(result_kernel, result_ops, rtol=1e-6, atol=1e-6)

class TestReductionOperations:
    """Test reduction fusion operations."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_reduce_sum_squared(self, device):
        torch.manual_seed(42)
        input_tensor = torch.randn(128, 256, device=device, dtype=torch.float32)
        
        # Test different dimensions
        for dim in [None, 0, 1, -1]:
            result = ops.reduce_sum_squared(input_tensor, dim=dim, keepdim=False)
            reference = torch.sum(input_tensor * input_tensor, dim=dim, keepdim=False)
            
            assert torch.allclose(result, reference, rtol=1e-5, atol=1e-5)
    
    def test_reduce_mean_abs(self, device):
        torch.manual_seed(42)
        input_tensor = torch.randn(128, 256, device=device, dtype=torch.float32)
        
        # Test different dimensions
        for dim in [None, 0, 1, -1]:
            result = ops.reduce_mean_abs(input_tensor, dim=dim, keepdim=False)
            reference = torch.mean(torch.abs(input_tensor), dim=dim, keepdim=False)
            
            assert torch.allclose(result, reference, rtol=1e-5, atol=1e-5)

class TestFusionOperations:
    """Test complex fusion operations."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_fused_layer_norm_relu(self, device):
        torch.manual_seed(42)
        input_tensor = torch.randn(32, 128, device=device, dtype=torch.float32)
        weight = torch.randn(128, device=device, dtype=torch.float32)
        bias = torch.randn(128, device=device, dtype=torch.float32)
        
        result = ops.fused_layer_norm_relu(
            input_tensor, 
            normalized_shape=(128,), 
            weight=weight, 
            bias=bias
        )
        
        # Reference implementation
        normalized = torch.nn.functional.layer_norm(input_tensor, (128,), weight, bias)
        reference = torch.relu(normalized)
        
        assert torch.allclose(result, reference, rtol=1e-4, atol=1e-4)
    
    def test_fused_gelu_dropout(self, device):
        torch.manual_seed(42)
        input_tensor = torch.randn(64, 256, device=device, dtype=torch.float32)
        
        # Test in evaluation mode (deterministic)
        torch.manual_seed(42)
        result = ops.fused_gelu_dropout(input_tensor, p=0.0, training=False)
        
        torch.manual_seed(42)
        reference = torch.nn.functional.gelu(input_tensor)
        
        assert torch.allclose(result, reference, rtol=1e-4, atol=1e-4)
    
    def test_fused_attention_score(self, device):
        torch.manual_seed(42)
        batch_size, seq_len, dim = 4, 32, 64
        
        query = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        key = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        scale = 1.0 / (dim ** 0.5)
        
        result = ops.fused_attention_score(query, key, scale)
        
        # Reference implementation
        reference = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        assert torch.allclose(result, reference, rtol=1e-5, atol=1e-5)

class TestPerformance:
    """Performance benchmarking tests."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_performance_comparison(self):
        """Compare performance of fused vs unfused operations."""
        device = torch.device("cuda")
        torch.manual_seed(42)
        
        # Large tensors for meaningful benchmarking
        a = torch.randn(4096, 4096, device=device, dtype=torch.float32)
        b = torch.randn(4096, 4096, device=device, dtype=torch.float32)
        
        # Warmup
        for _ in range(10):
            _ = ops.elementwise_add_relu(a, b)
            torch.cuda.synchronize()
        
        # Benchmark fused operation
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(100):
            result_fused = ops.elementwise_add_relu(a, b)
        end_event.record()
        torch.cuda.synchronize()
        
        fused_time = start_event.elapsed_time(end_event)
        
        # Benchmark unfused operations
        start_event.record()
        for _ in range(100):
            result_unfused = torch.relu(a + b)
        end_event.record()
        torch.cuda.synchronize()
        
        unfused_time = start_event.elapsed_time(end_event)
        
        print(f"Fused time: {fused_time:.2f}ms")
        print(f"Unfused time: {unfused_time:.2f}ms")
        print(f"Speedup: {unfused_time / fused_time:.2f}x")
        
        # Verify correctness
        assert torch.allclose(result_fused, result_unfused, rtol=1e-5, atol=1e-5)
