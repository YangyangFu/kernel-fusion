"""
Simple example demonstrating basic usage of the kernel fusion library.
This script shows how to use high-level APIs for common operations.
"""

import torch
import time
import kernel_fusion as kf

def main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA extension loaded: {kf.EXTENSION_LOADED}")
    
    # Create sample tensors
    batch_size, seq_len, hidden_dim = 32, 512, 768
    torch.manual_seed(42)
    
    # Example 1: Elementwise operations
    print("\n=== Elementwise Operations ===")
    a = torch.randn(batch_size, hidden_dim, device=device)
    b = torch.randn(batch_size, hidden_dim, device=device)
    
    # Fused add + relu
    result1 = kf.ops.elementwise_add_relu(a, b)
    reference1 = torch.relu(a + b)
    print(f"Add+ReLU - Max difference: {torch.max(torch.abs(result1 - reference1)).item():.8f}")
    
    # Fused mul + tanh  
    result2 = kf.ops.elementwise_mul_tanh(a, b)
    reference2 = torch.tanh(a * b)
    print(f"Mul+Tanh - Max difference: {torch.max(torch.abs(result2 - reference2)).item():.8f}")
    
    # Example 2: Bias + activation fusion
    print("\n=== Bias + Activation Fusion ===")
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    bias = torch.randn(hidden_dim, device=device)
    
    result3 = kf.ops.fused_bias_gelu(input_tensor, bias)
    reference3 = torch.nn.functional.gelu(input_tensor + bias)
    print(f"Bias+GELU - Max difference: {torch.max(torch.abs(result3 - reference3)).item():.8f}")
    
    # Example 3: Reduction operations
    print("\n=== Reduction Operations ===")
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Sum of squares
    result4 = kf.ops.reduce_sum_squared(input_tensor, dim=-1)
    reference4 = torch.sum(input_tensor * input_tensor, dim=-1)
    print(f"Sum Squared - Max difference: {torch.max(torch.abs(result4 - reference4)).item():.8f}")
    
    # Mean absolute value
    result5 = kf.ops.reduce_mean_abs(input_tensor, dim=-1)
    reference5 = torch.mean(torch.abs(input_tensor), dim=-1)
    print(f"Mean Abs - Max difference: {torch.max(torch.abs(result5 - reference5)).item():.8f}")
    
    # Example 4: Complex fusion operations
    print("\n=== Complex Fusion Operations ===")
    
    # Layer norm + ReLU
    weight = torch.randn(hidden_dim, device=device)
    bias = torch.randn(hidden_dim, device=device)
    
    result6 = kf.ops.fused_layer_norm_relu(
        input_tensor, 
        normalized_shape=(hidden_dim,), 
        weight=weight, 
        bias=bias
    )
    reference6 = torch.relu(torch.nn.functional.layer_norm(
        input_tensor, (hidden_dim,), weight, bias
    ))
    print(f"LayerNorm+ReLU - Max difference: {torch.max(torch.abs(result6 - reference6)).item():.8f}")
    
    # Attention scores
    query = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    key = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    scale = 1.0 / (hidden_dim ** 0.5)
    
    result7 = kf.ops.fused_attention_score(query, key, scale)
    reference7 = torch.matmul(query, key.transpose(-2, -1)) * scale
    print(f"Attention Score - Max difference: {torch.max(torch.abs(result7 - reference7)).item():.8f}")

def benchmark_performance():
    """Simple performance benchmark."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping performance benchmark")
        return
        
    device = torch.device("cuda")
    print("\n=== Performance Benchmark ===")
    
    # Large tensors for meaningful timing
    size = (2048, 2048)
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    # Warmup
    for _ in range(10):
        _ = kf.ops.elementwise_add_relu(a, b)
        _ = torch.relu(a + b)
    torch.cuda.synchronize()
    
    # Benchmark fused operation
    start_time = time.time()
    for _ in range(100):
        result_fused = kf.ops.elementwise_add_relu(a, b)
    torch.cuda.synchronize()
    fused_time = time.time() - start_time
    
    # Benchmark separate operations
    start_time = time.time()
    for _ in range(100):
        result_separate = torch.relu(a + b)
    torch.cuda.synchronize()
    separate_time = time.time() - start_time
    
    print(f"Fused Add+ReLU: {fused_time*1000:.2f}ms")
    print(f"Separate ops: {separate_time*1000:.2f}ms")
    print(f"Speedup: {separate_time/fused_time:.2f}x")
    
    # Verify correctness
    max_diff = torch.max(torch.abs(result_fused - result_separate)).item()
    print(f"Max difference: {max_diff:.8f}")

if __name__ == "__main__":
    main()
    benchmark_performance()
