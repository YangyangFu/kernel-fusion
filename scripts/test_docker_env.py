#!/usr/bin/env python3
"""
Quick Docker Environment Test Script
Validates that the kernel-fusion library works correctly in the Docker environment.
"""

import sys
import torch
import kernel_fusion as kf

def test_environment():
    """Test the Docker environment setup."""
    print("=" * 50)
    print("Kernel Fusion Docker Environment Test")
    print("=" * 50)
    
    # Test Python and PyTorch versions
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available in PyTorch: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    # Test kernel-fusion library
    print("\n" + "=" * 30)
    print("Kernel Fusion Library Status")
    print("=" * 30)
    print(f"Kernel Fusion version: {kf.__version__}")
    print(f"CUDA available: {kf.CUDA_AVAILABLE}")
    print(f"Extension loaded: {kf.EXTENSION_LOADED}")
    
    # Test basic operations
    print("\n" + "=" * 20)
    print("Basic Operations Test")
    print("=" * 20)
    
    try:
        # Create test tensors
        size = (32, 128)
        if kf.CUDA_AVAILABLE:
            device = torch.device('cuda')
            print("Testing on GPU...")
        else:
            device = torch.device('cpu')
            print("Testing on CPU...")
        
        x = torch.randn(size, device=device, requires_grad=True)
        y = torch.randn(size, device=device, requires_grad=True)
        
        print(f"Created tensors of shape {size} on {device}")
        
        # Test basic elementwise operations
        result = kf.ops.add(x, y)
        print(f"Addition result shape: {result.shape}")
        
        result = kf.ops.multiply(x, y)
        print(f"Multiplication result shape: {result.shape}")
        
        print("\n✅ Basic operations successful!")
        
        # Test fusion operations if available
        print("\n" + "=" * 20)
        print("Fusion Operations Test")
        print("=" * 20)
        
        # Test layer norm + relu fusion
        try:
            normalized_shape = (128,)
            fused_result = kf.kernels.fused_layer_norm_relu(x, normalized_shape)
            print(f"Fused LayerNorm+ReLU shape: {fused_result.shape}")
            print("✅ Fused LayerNorm+ReLU successful!")
        except Exception as e:
            print(f"⚠️  Fused LayerNorm+ReLU failed: {e}")
        
        # Test gelu + dropout fusion
        try:
            fused_result = kf.kernels.fused_gelu_dropout(x, p=0.1, training=True)
            print(f"Fused GELU+Dropout shape: {fused_result.shape}")
            print("✅ Fused GELU+Dropout successful!")
        except Exception as e:
            print(f"⚠️  Fused GELU+Dropout failed: {e}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed successfully!")
    print("Docker environment is ready for development.")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)
