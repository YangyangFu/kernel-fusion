#!/usr/bin/env python3
"""
Method 1: Operation-Level Replacement Example

This example demonstrates how to replace individual PyTorch operations
with kernel fusion equivalents in existing models.

This is the simplest approach - just replace specific operations
without changing the overall model structure.
"""

import torch
import torch.nn as nn
import kernel_fusion as kf

# Example 1: Basic elementwise operations replacement
def basic_operations_example():
    """Replace basic elementwise operations"""
    print("=== Basic Operations Replacement ===")
    
    # Create test tensors
    a = torch.randn(1024, 512, device='cuda')
    b = torch.randn(1024, 512, device='cuda')
    
    print("Before: Using separate PyTorch operations")
    # Standard PyTorch approach
    result_standard = torch.relu(a + b)
    print(f"Standard result shape: {result_standard.shape}")
    
    print("After: Using fused kernel operations")
    # Kernel Fusion approach - drop-in replacement
    result_fused = kf.ops.add_relu(a, b)
    print(f"Fused result shape: {result_fused.shape}")
    
    # Verify results are equivalent
    max_diff = torch.max(torch.abs(result_standard - result_fused))
    print(f"Maximum difference: {max_diff:.2e}")
    print()

# Example 2: Transformer block with operation-level replacements
class StandardTransformerBlock(nn.Module):
    """Standard transformer block using PyTorch operations"""
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        
        # First residual connection + layer norm
        x = self.norm1(x + attn_output)
        
        # Feed-forward network
        ff_output = self.linear1(x)
        ff_output = torch.relu(ff_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)
        
        # Second residual connection + layer norm
        x = self.norm2(x + ff_output)
        
        return x

class OptimizedTransformerBlock(nn.Module):
    """Transformer block with kernel fusion operation replacements"""
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_p = 0.1
        
    def forward(self, x):
        # Self-attention (unchanged)
        attn_output, _ = self.self_attn(x, x, x)
        
        # Fused residual + layer norm
        x = kf.ops.fused_layer_norm_add(
            x, attn_output, 
            self.norm1.weight, self.norm1.bias
        )
        
        # Fused linear + relu + dropout
        ff_output = kf.ops.fused_linear_relu_dropout(
            x, self.linear1.weight, self.linear1.bias,
            p=self.dropout_p, training=self.training
        )
        ff_output = self.linear2(ff_output)
        
        # Fused residual + layer norm
        x = kf.ops.fused_layer_norm_add(
            x, ff_output,
            self.norm2.weight, self.norm2.bias
        )
        
        return x

def transformer_comparison_example():
    """Compare standard vs optimized transformer blocks"""
    print("=== Transformer Block Comparison ===")
    
    d_model, nhead, dim_feedforward = 512, 8, 2048
    batch_size, seq_len = 32, 128
    
    # Create models
    standard_block = StandardTransformerBlock(d_model, nhead, dim_feedforward).cuda()
    optimized_block = OptimizedTransformerBlock(d_model, nhead, dim_feedforward).cuda()
    
    # Copy weights to ensure fair comparison
    optimized_block.load_state_dict(standard_block.state_dict())
    
    # Test input
    x = torch.randn(batch_size, seq_len, d_model, device='cuda')
    
    # Forward pass comparison
    with torch.no_grad():
        standard_output = standard_block(x)
        optimized_output = optimized_block(x)
    
    print(f"Standard output shape: {standard_output.shape}")
    print(f"Optimized output shape: {optimized_output.shape}")
    
    # Check accuracy
    max_diff = torch.max(torch.abs(standard_output - optimized_output))
    print(f"Maximum difference: {max_diff:.2e}")
    print()

# Example 3: CNN block with operation-level replacements
class StandardCNNBlock(nn.Module):
    """Standard CNN block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class OptimizedCNNBlock(nn.Module):
    """CNN block with fused conv+bn+relu"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        # Fused conv + batchnorm + relu
        return kf.ops.fused_conv2d_batchnorm_relu(
            x, self.conv.weight, self.conv.bias,
            self.bn.weight, self.bn.bias, 
            self.bn.running_mean, self.bn.running_var,
            eps=self.bn.eps
        )

def cnn_comparison_example():
    """Compare standard vs optimized CNN blocks"""
    print("=== CNN Block Comparison ===")
    
    in_channels, out_channels = 64, 128
    batch_size, height, width = 16, 32, 32
    
    # Create models
    standard_block = StandardCNNBlock(in_channels, out_channels).cuda()
    optimized_block = OptimizedCNNBlock(in_channels, out_channels).cuda()
    
    # Copy weights
    optimized_block.load_state_dict(standard_block.state_dict())
    
    # Test input
    x = torch.randn(batch_size, in_channels, height, width, device='cuda')
    
    # Forward pass comparison
    standard_block.eval()
    optimized_block.eval()
    
    with torch.no_grad():
        standard_output = standard_block(x)
        optimized_output = optimized_block(x)
    
    print(f"Standard output shape: {standard_output.shape}")
    print(f"Optimized output shape: {optimized_output.shape}")
    
    # Check accuracy
    max_diff = torch.max(torch.abs(standard_output - optimized_output))
    print(f"Maximum difference: {max_diff:.2e}")
    print()

def main():
    """Run all operation-level replacement examples"""
    print("Operation-Level Replacement Examples")
    print("=" * 50)
    print()
    
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available, running CPU examples...")
            # Note: In real implementation, operations would fall back to CPU
        
        # Run examples
        basic_operations_example()
        transformer_comparison_example()
        cnn_comparison_example()
        
        print("✅ All operation-level replacement examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Note: Make sure kernel_fusion is properly installed and CUDA is available")

if __name__ == "__main__":
    main()
