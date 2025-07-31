#!/usr/bin/env python3
"""
Quick debugging script for LayerNorm CUDA kernel
Run this to test and debug your implementation
"""

import os
import sys
import torch
import traceback

# Add the kernel_fusion directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def quick_debug():
    """Quick debugging session for LayerNorm kernel"""
    
    print("="*60)
    print("CUDA LAYERNORM DEBUGGING SESSION")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return False
    
    print(f"✅ CUDA is available")
    print(f"   Device: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # Import debugging utilities
        from kernel_fusion.kernels.cuda_debug_utils import debug_layernorm_kernel, CUDADebugger
        
        # Create simple test case
        print("\n📊 Creating test tensors...")
        batch_size, seq_len, hidden_size = 2, 4, 8
        
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        weight = torch.randn(hidden_size, device='cuda')
        bias = torch.randn(hidden_size, device='cuda')
        
        print(f"   Input shape: {input_tensor.shape}")
        print(f"   Weight shape: {weight.shape}")
        print(f"   Bias shape: {bias.shape}")
        
        # Run debugging
        print("\n🔍 Running debug session...")
        result = debug_layernorm_kernel(input_tensor, weight, bias)
        
        print("\n✅ Debugging completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the cuda_debug_utils.py file is in the kernels directory")
        return False
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        print(f"   Traceback:")
        traceback.print_exc()
        return False

def environment_check():
    """Check debugging environment setup"""
    
    print("\n🔧 ENVIRONMENT CHECK")
    print("-" * 30)
    
    # Check Python and PyTorch versions
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Current device: {torch.cuda.current_device()}")
    
    # Check for debugging tools
    tools_check = {
        'nvidia-smi': 'nvidia-smi --version',
        'compute-sanitizer': 'compute-sanitizer --version', 
        'nsys': 'nsys --version'
    }
    
    print("\n🛠️  Debugging tools availability:")
    for tool, cmd in tools_check.items():
        try:
            import subprocess
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"   ✅ {tool}: Available")
            else:
                print(f"   ❌ {tool}: Not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"   ❌ {tool}: Not found")
        except Exception:
            print(f"   ❌ {tool}: Error checking")

def compilation_test():
    """Test CUDA kernel compilation in both debug and release modes"""
    
    print("\n🔨 COMPILATION TEST")
    print("-" * 30)
    
    try:
        from kernel_fusion.kernels.cuda_layernorm import CUDALayerNorm
        
        print("Testing debug compilation...")
        try:
            debug_ln = CUDALayerNorm(debug=True)
            if debug_ln.module is not None:
                print("   ✅ Debug compilation successful")
            else:
                print("   ❌ Debug compilation failed")
        except Exception as e:
            print(f"   ❌ Debug compilation error: {e}")
        
        print("Testing release compilation...")
        try:
            release_ln = CUDALayerNorm(debug=False)
            if release_ln.module is not None:
                print("   ✅ Release compilation successful")
            else:
                print("   ❌ Release compilation failed")
        except Exception as e:
            print(f"   ❌ Release compilation error: {e}")
            
    except ImportError as e:
        print(f"   ❌ Cannot import CUDALayerNorm: {e}")

def memory_test():
    """Test for memory leaks and usage patterns"""
    
    print("\n🧠 MEMORY TEST")
    print("-" * 30)
    
    if not torch.cuda.is_available():
        print("   ⏭️  Skipping (CUDA not available)")
        return
    
    try:
        from kernel_fusion.kernels.cuda_layernorm import CUDALayerNorm
        
        # Initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        print(f"   Initial memory: {initial_memory / 1024**2:.1f} MB")
        
        # Create layernorm instance
        ln = CUDALayerNorm()
        
        # Run multiple iterations
        input_tensor = torch.randn(8, 32, 64, device='cuda')
        weight = torch.randn(64, device='cuda')
        bias = torch.randn(64, device='cuda')
        
        for i in range(5):
            result = ln.forward(input_tensor, weight, bias)
            current_memory = torch.cuda.memory_allocated()
            print(f"   Iteration {i+1}: {current_memory / 1024**2:.1f} MB")
            
            # Clean up
            del result
        
        # Final cleanup
        del ln, input_tensor, weight, bias
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        print(f"   Final memory: {final_memory / 1024**2:.1f} MB")
        
        if final_memory <= initial_memory + 1024*1024:  # Allow 1MB tolerance
            print("   ✅ No significant memory leak detected")
        else:
            print(f"   ⚠️  Potential memory leak: {(final_memory - initial_memory) / 1024**2:.1f} MB")
            
    except Exception as e:
        print(f"   ❌ Memory test error: {e}")

if __name__ == "__main__":
    print("Starting CUDA LayerNorm debugging session...\n")
    
    # Run all checks
    environment_check()
    compilation_test()
    memory_test()
    
    # Run main debugging
    success = quick_debug()
    
    print("\n" + "="*60)
    if success:
        print("🎉 DEBUGGING SESSION COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Review the output above for any warnings")
        print("2. Run with different tensor sizes")
        print("3. Compare performance with PyTorch implementation")
        print("4. Use profiling tools for detailed analysis")
    else:
        print("❌ DEBUGGING SESSION ENCOUNTERED ISSUES")
        print("\nTroubleshooting:")
        print("1. Check CUDA installation")
        print("2. Verify PyTorch CUDA support")
        print("3. Review compilation errors")
        print("4. Check file paths and imports")
    
    print("="*60)
