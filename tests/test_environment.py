"""
Test environment setup
"""

import pytest
import torch


def test_cuda_available():
    """Test that CUDA is available"""
    assert torch.cuda.is_available(), "CUDA should be available in the container"


def test_torch_version():
    """Test PyTorch version"""
    version = torch.__version__
    major, minor = map(int, version.split('.')[:2])
    assert major >= 2 or (major == 1 and minor >= 12), f"PyTorch version {version} is too old"


def test_triton_import():
    """Test that Triton can be imported"""
    try:
        import triton
        import triton.language as tl
        assert True
    except ImportError:
        pytest.fail("Triton should be importable")


def test_basic_gpu_operation():
    """Test basic GPU operations"""
    if torch.cuda.is_available():
        x = torch.randn(10, 10, device='cuda')
        y = torch.randn(10, 10, device='cuda')
        z = torch.matmul(x, y)
        assert z.device.type == 'cuda'
        assert z.shape == (10, 10)


if __name__ == "__main__":
    pytest.main([__file__])
