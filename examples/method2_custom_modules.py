#!/usr/bin/env python3
"""
Method 2: Custom Module Replacement Example

This example demonstrates how to create custom PyTorch modules that
encapsulate kernel fusion operations, providing drop-in replacements
for common layer combinations.

This approach is ideal when you want to:
- Encapsulate fusion logic in reusable modules
- Maintain clean separation between fusion and regular PyTorch code
- Create building blocks for larger architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import kernel_fusion as kf

# Custom fused modules
class FusedLinearReLU(nn.Module):
    """Drop-in replacement for Linear + ReLU"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        
        # Initialize weights similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return kf.ops.fused_linear_relu(x, self.weight, self.bias)
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class FusedLinearGELU(nn.Module):
    """Drop-in replacement for Linear + GELU"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return kf.ops.fused_linear_gelu(x, self.weight, self.bias)

class FusedLayerNormReLU(nn.Module):
    """Drop-in replacement for LayerNorm + ReLU"""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        return kf.kernels.fused_layer_norm_relu(
            x, self.weight, self.bias, self.eps
        )

class FusedConv2dBatchNormReLU(nn.Module):
    """Drop-in replacement for Conv2d + BatchNorm2d + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-5, momentum=0.1):
        super().__init__()
        
        # Conv2d parameters
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, dilation, groups, bias)
        
        # BatchNorm2d parameters  
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
    
    def forward(self, x):
        return kf.ops.fused_conv2d_batchnorm_relu(
            x, self.conv.weight, self.conv.bias,
            self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var,
            eps=self.bn.eps
        )

class FusedMultiHeadAttention(nn.Module):
    """Optimized multi-head attention with fused operations"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Combined Q, K, V projection
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    
    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, L, 3*D]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D_h]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Fused attention computation
        attn = kf.ops.fused_attention_score_softmax(q, k, self.scale, attn_mask)
        
        # Apply attention to values
        out = kf.ops.fused_attention_apply(attn, v)  # [B, H, L, D_h]
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, L, D)  # [B, L, D]
        out = self.out_proj(out)
        
        return out

# Example usage: Building optimized models with custom modules
class OptimizedMLP(nn.Module):
    """MLP using fused custom modules"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(FusedLinearReLU(input_size, hidden_size))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(FusedLinearReLU(hidden_size, hidden_size))
        
        # Output layer (no activation)
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class OptimizedTransformerEncoder(nn.Module):
    """Transformer encoder using fused custom modules"""
    def __init__(self, d_model, nhead, dim_feedforward, num_layers=6, dropout=0.1):
        super().__init__()
        
        # Create transformer layers with fused components
        self.layers = nn.ModuleList([
            OptimizedTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = FusedLayerNormReLU(d_model)
    
    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.norm(x)
        return x

class OptimizedTransformerLayer(nn.Module):
    """Single transformer layer with fused operations"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Multi-head attention with fusion
        self.self_attn = FusedMultiHeadAttention(d_model, nhead, dropout)
        
        # Feed-forward network with fused layers
        self.ffn = nn.Sequential(
            FusedLinearGELU(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_mask=None):
        # Self-attention with residual connection
        attn_out = self.self_attn(x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection  
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

class OptimizedCNN(nn.Module):
    """CNN using fused conv blocks"""
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        # Feature extraction with fused conv blocks
        self.features = nn.Sequential(
            FusedConv2dBatchNormReLU(in_channels, 64, 3, padding=1),
            FusedConv2dBatchNormReLU(64, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            
            FusedConv2dBatchNormReLU(64, 128, 3, padding=1),
            FusedConv2dBatchNormReLU(128, 128, 3, padding=1),
            nn.MaxPool2d(2, 2),
            
            FusedConv2dBatchNormReLU(128, 256, 3, padding=1),
            FusedConv2dBatchNormReLU(256, 256, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier with fused linear layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            FusedLinearReLU(256, 512),
            nn.Dropout(0.5),
            FusedLinearReLU(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Demonstration functions
def compare_mlp_models():
    """Compare standard MLP vs optimized MLP with fused modules"""
    print("=== MLP Comparison ===")
    
    input_size, hidden_size, output_size = 512, 1024, 256
    batch_size = 64
    
    # Standard MLP
    class StandardMLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Create models
    standard_mlp = StandardMLP(input_size, hidden_size, output_size).cuda()
    optimized_mlp = OptimizedMLP(input_size, hidden_size, output_size).cuda()
    
    # Test input
    x = torch.randn(batch_size, input_size, device='cuda')
    
    # Forward pass
    with torch.no_grad():
        standard_out = standard_mlp(x)
        optimized_out = optimized_mlp(x)
    
    print(f"Standard MLP output shape: {standard_out.shape}")
    print(f"Optimized MLP output shape: {optimized_out.shape}")
    print(f"Models created successfully!")
    print()

def demonstrate_custom_modules():
    """Demonstrate individual custom fused modules"""
    print("=== Custom Fused Modules Demo ===")
    
    batch_size = 32
    
    # Test FusedLinearReLU
    print("Testing FusedLinearReLU...")
    fused_linear = FusedLinearReLU(256, 512).cuda()
    x = torch.randn(batch_size, 256, device='cuda')
    out = fused_linear(x)
    print(f"FusedLinearReLU: {x.shape} -> {out.shape}")
    
    # Test FusedLayerNormReLU
    print("Testing FusedLayerNormReLU...")
    fused_norm = FusedLayerNormReLU(512).cuda()
    out = fused_norm(out)
    print(f"FusedLayerNormReLU: output shape {out.shape}")
    
    # Test FusedConv2dBatchNormReLU
    print("Testing FusedConv2dBatchNormReLU...")
    fused_conv = FusedConv2dBatchNormReLU(3, 64, 3, padding=1).cuda()
    x_conv = torch.randn(batch_size, 3, 32, 32, device='cuda')
    out_conv = fused_conv(x_conv)
    print(f"FusedConv2dBatchNormReLU: {x_conv.shape} -> {out_conv.shape}")
    
    print("All custom modules working correctly!")
    print()

def benchmark_custom_vs_standard():
    """Benchmark custom fused modules vs standard PyTorch"""
    print("=== Performance Benchmark ===")
    
    import time
    
    # Setup
    batch_size, features = 64, 512
    x = torch.randn(batch_size, features, device='cuda')
    
    # Standard Linear + ReLU
    standard = nn.Sequential(
        nn.Linear(features, features),
        nn.ReLU()
    ).cuda()
    
    # Fused Linear + ReLU
    fused = FusedLinearReLU(features, features).cuda()
    
    # Copy weights for fair comparison
    fused.weight.data = standard[0].weight.data.clone()
    fused.bias.data = standard[0].bias.data.clone()
    
    # Warmup
    for _ in range(10):
        _ = standard(x)
        _ = fused(x)
    
    # Benchmark standard
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        _ = standard(x)
    torch.cuda.synchronize()
    standard_time = time.time() - start_time
    
    # Benchmark fused
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        _ = fused(x)
    torch.cuda.synchronize()
    fused_time = time.time() - start_time
    
    speedup = standard_time / fused_time
    print(f"Standard Linear+ReLU time: {standard_time:.4f}s")
    print(f"Fused Linear+ReLU time: {fused_time:.4f}s") 
    print(f"Speedup: {speedup:.2f}x")
    print()

def main():
    """Run all custom module replacement examples"""
    print("Custom Module Replacement Examples")
    print("=" * 50)
    print()
    
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("CUDA not available, some examples may not work optimally...")
        
        # Run demonstrations
        demonstrate_custom_modules()
        compare_mlp_models()
        benchmark_custom_vs_standard()
        
        print("✅ All custom module replacement examples completed successfully!")
        print("\nKey Benefits of Custom Modules:")
        print("- Encapsulated fusion logic")
        print("- Drop-in replacements for standard modules")
        print("- Reusable building blocks")
        print("- Clean separation of concerns")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Note: Make sure kernel_fusion is properly installed")

if __name__ == "__main__":
    main()
