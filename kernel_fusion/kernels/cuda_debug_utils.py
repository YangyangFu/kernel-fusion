"""
CUDA Debugging Utilities for Kernel Development
Comprehensive debugging tools for CUDA kernels in PyTorch
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import time
from typing import Dict, List, Optional, Tuple, Any

class CUDADebugger:
    """Comprehensive CUDA debugging utilities"""
    
    def __init__(self):
        self.profiling_enabled = False
        self.memory_tracking = {}
        
    def check_cuda_errors(self, operation_name: str = "CUDA operation"):
        """Check for CUDA errors after kernel execution"""
        try:
            torch.cuda.synchronize()
            print(f"✓ {operation_name} completed without CUDA errors")
        except RuntimeError as e:
            print(f"✗ CUDA Error in {operation_name}: {e}")
            self.print_cuda_info()
            raise
    
    def print_cuda_info(self):
        """Print comprehensive CUDA device information"""
        if not torch.cuda.is_available():
            print("CUDA is not available")
            return
            
        print("\n" + "="*60)
        print("CUDA DEVICE INFORMATION")
        print("="*60)
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"Device {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            print(f"  Max Threads per Block: {props.max_threads_per_block}")
            print(f"  Max Block Dimensions: {props.max_threads_per_multiprocessor}")
            print(f"  Warp Size: {props.warp_size}")
            
        current_device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_cached = torch.cuda.memory_reserved(current_device) / 1024**3
        
        print(f"\nMemory Usage on Device {current_device}:")
        print(f"  Allocated: {memory_allocated:.2f} GB")
        print(f"  Cached: {memory_cached:.2f} GB")
        print("="*60)
    
    def tensor_summary(self, tensor: torch.Tensor, name: str = "Tensor"):
        """Print comprehensive tensor information"""
        print(f"\n{name} Summary:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Device: {tensor.device}")
        print(f"  Requires Grad: {tensor.requires_grad}")
        print(f"  Is Contiguous: {tensor.is_contiguous()}")
        
        if tensor.numel() > 0:
            print(f"  Min: {tensor.min().item():.6f}")
            print(f"  Max: {tensor.max().item():.6f}")
            print(f"  Mean: {tensor.mean().item():.6f}")
            print(f"  Std: {tensor.std().item():.6f}")
            
            # Check for NaN/Inf
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            if nan_count > 0:
                print(f"  ⚠️  NaN values: {nan_count}")
            if inf_count > 0:
                print(f"  ⚠️  Inf values: {inf_count}")
                
    def compare_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor, 
                       name1: str = "Tensor1", name2: str = "Tensor2", 
                       rtol: float = 1e-5, atol: float = 1e-8):
        """Compare two tensors and analyze differences"""
        print(f"\nComparing {name1} vs {name2}:")
        
        if tensor1.shape != tensor2.shape:
            print(f"  ✗ Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
            return False
            
        # Move to same device for comparison
        if tensor1.device != tensor2.device:
            tensor2 = tensor2.to(tensor1.device)
            
        diff = torch.abs(tensor1 - tensor2)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")
        
        # Check if tensors are close
        is_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
        print(f"  Are close (rtol={rtol}, atol={atol}): {is_close}")
        
        if not is_close:
            # Find indices with largest differences
            diff_flat = diff.flatten()
            _, max_indices = torch.topk(diff_flat, min(5, len(diff_flat)))
            print(f"  Top 5 differences at indices:")
            for i, idx in enumerate(max_indices):
                idx_item = idx.item()
                val1 = tensor1.flatten()[idx_item].item()
                val2 = tensor2.flatten()[idx_item].item()
                diff_val = diff_flat[idx_item].item()
                print(f"    {i+1}. Index {idx_item}: {val1:.6f} vs {val2:.6f} (diff: {diff_val:.6f})")
                
        return is_close
    
    def profile_kernel(self, func, *args, warmup_runs: int = 3, 
                      profile_runs: int = 10, **kwargs):
        """Profile kernel execution time"""
        print(f"\nProfiling kernel execution...")
        
        # Warmup runs
        for _ in range(warmup_runs):
            _ = func(*args, **kwargs)
            torch.cuda.synchronize()
        
        # Timing runs
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(profile_runs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(profile_runs)]
        
        for i in range(profile_runs):
            start_events[i].record()
            result = func(*args, **kwargs)
            end_events[i].record()
            
        torch.cuda.synchronize()
        
        # Calculate timing statistics
        times = [start_events[i].elapsed_time(end_events[i]) for i in range(profile_runs)]
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"  Mean time: {mean_time:.3f} ms")
        print(f"  Std time: {std_time:.3f} ms")
        print(f"  Min time: {min_time:.3f} ms")
        print(f"  Max time: {max_time:.3f} ms")
        
        return result, mean_time
    
    def memory_snapshot(self, name: str = ""):
        """Take a snapshot of GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            self.memory_tracking[name] = {
                'allocated': allocated,
                'cached': cached,
                'time': time.time()
            }
            print(f"Memory snapshot '{name}': {allocated:.3f} GB allocated, {cached:.3f} GB cached")
    
    def analyze_memory_usage(self):
        """Analyze memory usage patterns"""
        print("\nMemory Usage Analysis:")
        if not self.memory_tracking:
            print("  No memory snapshots taken")
            return
            
        for name, info in self.memory_tracking.items():
            print(f"  {name}: {info['allocated']:.3f} GB allocated, {info['cached']:.3f} GB cached")
    
    def launch_nvidia_smi(self):
        """Launch nvidia-smi for real-time GPU monitoring"""
        try:
            print("Launching nvidia-smi for GPU monitoring...")
            subprocess.Popen(['nvidia-smi', '-l', '1'])
        except FileNotFoundError:
            print("nvidia-smi not found. Please ensure CUDA toolkit is installed.")
    
    def run_compute_sanitizer(self, python_script: str):
        """Run compute sanitizer on a Python script"""
        try:
            print(f"Running compute-sanitizer on {python_script}...")
            result = subprocess.run([
                'compute-sanitizer', 
                '--tool', 'memcheck',
                'python', python_script
            ], capture_output=True, text=True)
            
            print("Compute Sanitizer Output:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
                
        except FileNotFoundError:
            print("compute-sanitizer not found. Please ensure CUDA toolkit is installed.")

def debug_layernorm_kernel(input_tensor, weight=None, bias=None, eps=1e-5):
    """Debug-specific LayerNorm testing function"""
    debugger = CUDADebugger()
    
    print("="*60)
    print("DEBUGGING LAYERNORM CUDA KERNEL")
    print("="*60)
    
    # Print CUDA info
    debugger.print_cuda_info()
    
    # Analyze input tensors
    debugger.tensor_summary(input_tensor, "Input")
    if weight is not None:
        debugger.tensor_summary(weight, "Weight")
    if bias is not None:
        debugger.tensor_summary(bias, "Bias")
    
    # Memory snapshot before
    debugger.memory_snapshot("Before LayerNorm")
    
    # Import and test the CUDA implementation
    from .cuda_layernorm import CUDALayerNorm
    
    # Test with debug compilation
    print("\nTesting with debug compilation...")
    cuda_ln_debug = CUDALayerNorm(debug=True)
    
    try:
        # Profile the kernel
        cuda_result, exec_time = debugger.profile_kernel(
            cuda_ln_debug.forward, input_tensor, weight, bias, eps
        )
        debugger.check_cuda_errors("CUDA LayerNorm forward")
        
        # Memory snapshot after
        debugger.memory_snapshot("After LayerNorm")
        
        # Compare with PyTorch implementation
        print("\nComparing with PyTorch reference...")
        with torch.no_grad():
            pytorch_result = torch.nn.functional.layer_norm(
                input_tensor, input_tensor.shape[-1:], weight, bias, eps
            )
        
        debugger.compare_tensors(cuda_result[0], pytorch_result, 
                               "CUDA Output", "PyTorch Output")
        
        # Analyze memory usage
        debugger.analyze_memory_usage()
        
        return cuda_result
        
    except Exception as e:
        print(f"Error during kernel execution: {e}")
        debugger.print_cuda_info()
        raise

def create_test_cases():
    """Create various test cases for debugging"""
    test_cases = []
    
    # Test case 1: Small tensor
    test_cases.append({
        'name': 'Small tensor',
        'input': torch.randn(2, 4, 8, device='cuda'),
        'weight': torch.randn(8, device='cuda'),
        'bias': torch.randn(8, device='cuda')
    })
    
    # Test case 2: Large tensor
    test_cases.append({
        'name': 'Large tensor',
        'input': torch.randn(32, 128, 768, device='cuda'),
        'weight': torch.randn(768, device='cuda'),
        'bias': torch.randn(768, device='cuda')
    })
    
    # Test case 3: Edge case - very small values
    test_cases.append({
        'name': 'Small values',
        'input': torch.randn(4, 8, 16, device='cuda') * 1e-6,
        'weight': torch.ones(16, device='cuda'),
        'bias': torch.zeros(16, device='cuda')
    })
    
    # Test case 4: Edge case - large values
    test_cases.append({
        'name': 'Large values',
        'input': torch.randn(4, 8, 16, device='cuda') * 1e6,
        'weight': torch.ones(16, device='cuda'),
        'bias': torch.zeros(16, device='cuda')
    })
    
    return test_cases

if __name__ == "__main__":
    # Run comprehensive debugging tests
    test_cases = create_test_cases()
    
    for test_case in test_cases:
        print(f"\n{'='*20} {test_case['name']} {'='*20}")
        try:
            debug_layernorm_kernel(
                test_case['input'], 
                test_case['weight'], 
                test_case['bias']
            )
        except Exception as e:
            print(f"Test case '{test_case['name']}' failed: {e}")
