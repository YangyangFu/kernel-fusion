"""
Installation verification script for kernel fusion library.
Runs basic tests to ensure everything is working correctly.
"""

import sys
import torch
import numpy as np

def check_basic_import():
    """Check if the library imports correctly."""
    try:
        import kernel_fusion as kf
        print("✓ Kernel fusion library imported successfully")
        print(f"  Version: {kf.__version__}")
        print(f"  CUDA available: {kf.CUDA_AVAILABLE}")
        print(f"  Extension loaded: {kf.EXTENSION_LOADED}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import kernel fusion library: {e}")
        return False

def check_basic_operations():
    """Test basic operations."""
    try:
        import kernel_fusion as kf
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Testing on device: {device}")
        
        # Test elementwise operations
        a = torch.randn(10, 10, device=device)
        b = torch.randn(10, 10, device=device)
        
        result = kf.ops.elementwise_add_relu(a, b)
        reference = torch.relu(a + b)
        
        if torch.allclose(result, reference, rtol=1e-4, atol=1e-4):
            print("✓ Elementwise add_relu operation works correctly")
        else:
            print("✗ Elementwise add_relu operation failed")
            return False
        
        # Test bias + gelu
        input_tensor = torch.randn(5, 8, device=device)
        bias = torch.randn(8, device=device)
        
        result = kf.ops.fused_bias_gelu(input_tensor, bias)
        reference = torch.nn.functional.gelu(input_tensor + bias)
        
        if torch.allclose(result, reference, rtol=1e-4, atol=1e-4):
            print("✓ Fused bias_gelu operation works correctly")
        else:
            print("✗ Fused bias_gelu operation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Basic operations test failed: {e}")
        return False

def check_kernel_access():
    """Test direct kernel access."""
    try:
        import kernel_fusion as kf
        
        if not kf.EXTENSION_LOADED:
            print("  Skipping kernel access test (extension not loaded)")
            return True
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if device.type == "cuda":
            a = torch.randn(5, 5, device=device)
            b = torch.randn(5, 5, device=device)
            
            result = kf.kernels.elementwise.add_relu(a, b)
            reference = torch.relu(a + b)
            
            if torch.allclose(result, reference, rtol=1e-4, atol=1e-4):
                print("✓ Direct kernel access works correctly")
            else:
                print("✗ Direct kernel access failed")
                return False
        else:
            print("  Skipping kernel access test (CUDA not available)")
        
        return True
        
    except Exception as e:
        print(f"✗ Kernel access test failed: {e}")
        return False

def check_performance():
    """Basic performance check."""
    try:
        import kernel_fusion as kf
        import time
        
        if not torch.cuda.is_available():
            print("  Skipping performance test (CUDA not available)")
            return True
        
        device = torch.device("cuda")
        size = (1000, 1000)
        
        a = torch.randn(size, device=device)
        b = torch.randn(size, device=device)
        
        # Warmup
        for _ in range(10):
            _ = kf.ops.elementwise_add_relu(a, b)
            _ = torch.relu(a + b)
        torch.cuda.synchronize()
        
        # Time fused operation
        start = time.time()
        for _ in range(100):
            result_fused = kf.ops.elementwise_add_relu(a, b)
        torch.cuda.synchronize()
        fused_time = time.time() - start
        
        # Time separate operations
        start = time.time()
        for _ in range(100):
            result_separate = torch.relu(a + b)
        torch.cuda.synchronize()
        separate_time = time.time() - start
        
        speedup = separate_time / fused_time
        print(f"✓ Performance test completed")
        print(f"  Fused operation: {fused_time*1000:.2f}ms")
        print(f"  Separate operations: {separate_time*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("Kernel Fusion Library - Installation Verification")
    print("=" * 50)
    
    tests = [
        ("Basic Import", check_basic_import),
        ("Basic Operations", check_basic_operations),
        ("Kernel Access", check_kernel_access),
        ("Performance", check_performance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
