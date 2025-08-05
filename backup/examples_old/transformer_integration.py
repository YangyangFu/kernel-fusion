"""
Advanced example showing integration with PyTorch models.
Demonstrates how to use kernel fusion in a real transformer-like model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import kernel_fusion as kf
import time

class FusedTransformerBlock(nn.Module):
    """Transformer block using fused operations where possible."""
    
    def __init__(self, hidden_dim, num_heads, dropout_prob=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention layers
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Feed-forward layers
        self.ff_linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ff_linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout_prob = dropout_prob
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Self-attention with residual connection
        residual1 = x
        
        # Use fused layer norm + relu for first norm (as example)
        # In practice, you'd want separate layer norm, but this shows the concept
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores using fused operation
        scale = 1.0 / (self.head_dim ** 0.5)
        
        # Flatten heads for fused attention score computation
        q_flat = q.contiguous().view(-1, seq_len, self.head_dim)
        k_flat = k.contiguous().view(-1, seq_len, self.head_dim)
        
        if kf.EXTENSION_LOADED and x.is_cuda:
            scores_flat = kf.ops.fused_attention_score(q_flat, k_flat, scale)
            scores = scores_flat.view(batch_size, self.num_heads, seq_len, seq_len)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply softmax and attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # First residual connection + layer norm
        x = self.norm1(residual1 + attn_output)
        
        # Feed-forward network with fused operations
        residual2 = x
        
        # First FF layer with fused bias + GELU
        ff_out = self.ff_linear1(x)
        if kf.EXTENSION_LOADED and x.is_cuda:
            # Use fused GELU + dropout
            ff_out = kf.ops.fused_gelu_dropout(ff_out, self.dropout_prob, self.training)
        else:
            ff_out = F.gelu(ff_out)
            ff_out = F.dropout(ff_out, self.dropout_prob, self.training)
        
        # Second FF layer
        ff_out = self.ff_linear2(ff_out)
        
        # Second residual connection + layer norm
        x = self.norm2(residual2 + ff_out)
        
        return x

class StandardTransformerBlock(nn.Module):
    """Standard transformer block for comparison."""
    
    def __init__(self, hidden_dim, num_heads, dropout_prob=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention layers
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Feed-forward layers
        self.ff_linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ff_linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout_prob = dropout_prob
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Self-attention with residual connection
        residual1 = x
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply softmax and attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # First residual connection + layer norm
        x = self.norm1(residual1 + attn_output)
        
        # Feed-forward network
        residual2 = x
        ff_out = self.ff_linear1(x)
        ff_out = F.gelu(ff_out)
        ff_out = F.dropout(ff_out, self.dropout_prob, self.training)
        ff_out = self.ff_linear2(ff_out)
        
        # Second residual connection + layer norm
        x = self.norm2(residual2 + ff_out)
        
        return x

def benchmark_transformer_blocks():
    """Compare performance of fused vs standard transformer blocks."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device}")
    
    # Model parameters
    batch_size = 16
    seq_len = 512
    hidden_dim = 768
    num_heads = 12
    
    # Create models
    fused_block = FusedTransformerBlock(hidden_dim, num_heads).to(device)
    standard_block = StandardTransformerBlock(hidden_dim, num_heads).to(device)
    
    # Ensure both models have the same parameters
    standard_block.load_state_dict(fused_block.state_dict())
    
    # Create input
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = fused_block(input_tensor)
            _ = standard_block(input_tensor)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark fused model
    print("Benchmarking fused model...")
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            fused_output = fused_block(input_tensor)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    fused_time = time.time() - start_time
    
    # Benchmark standard model
    print("Benchmarking standard model...")
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            standard_output = standard_block(input_tensor)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    standard_time = time.time() - start_time
    
    # Results
    print(f"\n=== Benchmark Results ===")
    print(f"Fused model: {fused_time*1000:.2f}ms")
    print(f"Standard model: {standard_time*1000:.2f}ms")
    print(f"Speedup: {standard_time/fused_time:.2f}x")
    
    # Verify correctness
    max_diff = torch.max(torch.abs(fused_output - standard_output)).item()
    mean_diff = torch.mean(torch.abs(fused_output - standard_output)).item()
    print(f"Max difference: {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")

def demonstrate_memory_efficiency():
    """Show memory usage benefits of fused operations."""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory efficiency demo")
        return
    
    device = torch.device("cuda")
    print(f"\n=== Memory Efficiency Demo ===")
    
    # Large tensor for memory measurement
    size = (4096, 4096)
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    # Measure memory for fused operation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    result_fused = kf.ops.elementwise_add_relu(a, b)
    fused_memory = torch.cuda.max_memory_allocated()
    
    # Measure memory for separate operations
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    temp = a + b
    result_separate = torch.relu(temp)
    separate_memory = torch.cuda.max_memory_allocated()
    
    print(f"Fused operation peak memory: {fused_memory / 1024**2:.2f} MB")
    print(f"Separate operations peak memory: {separate_memory / 1024**2:.2f} MB")
    print(f"Memory savings: {(separate_memory - fused_memory) / 1024**2:.2f} MB")

if __name__ == "__main__":
    print("=== Advanced Kernel Fusion Example ===")
    benchmark_transformer_blocks()
    demonstrate_memory_efficiency()
