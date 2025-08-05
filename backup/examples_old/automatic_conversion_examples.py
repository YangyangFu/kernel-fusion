#!/usr/bin/env python3
"""
Automatic Model Conversion Examples

This script demonstrates how to automatically convert popular pretrained models
to use kernel fusion operations while preserving all weights and maintaining
numerical accuracy.

Supported Models:
- ResNet (torchvision)
- BERT (transformers/custom)
- Vision Transformer (custom/timm)
- Custom CNN architectures
- Custom transformer architectures
"""

import torch
import torch.nn as nn
import torchvision.models as models
import time
import sys
import os

# Add parent directory to path for kernel_fusion import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernel_fusion.auto_convert import auto_convert_model, ModelConverter


def benchmark_model(model, input_tensor, num_runs=100, warmup_runs=10):
    """Benchmark model inference time"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Synchronize CUDA
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time * 1000  # Return in milliseconds


def example_resnet_conversion():
    """Example: Convert ResNet50 to use fused kernels"""
    print("=" * 60)
    print("ResNet50 Automatic Conversion Example")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load pretrained ResNet50
    print("Loading pretrained ResNet50...")
    original_model = models.resnet50(pretrained=True)
    original_model = original_model.to(device)
    
    # Create test input
    input_tensor = torch.randn(1, 3, 224, 224, device=device)
    
    # Convert model automatically
    print("\nConverting model to use fused kernels...")
    fused_model = auto_convert_model(
        original_model,
        input_shape=(1, 3, 224, 224),
        device=device,
        validate_accuracy=True,
        accuracy_threshold=1e-5
    )
    
    # Benchmark performance
    print("\nBenchmarking performance...")
    original_time = benchmark_model(original_model, input_tensor)
    fused_time = benchmark_model(fused_model, input_tensor)
    
    speedup = original_time / fused_time
    
    print(f"\nPerformance Results:")
    print(f"Original model: {original_time:.2f} ms")
    print(f"Fused model:    {fused_time:.2f} ms")
    print(f"Speedup:        {speedup:.2f}x")
    
    return fused_model


def example_custom_cnn_conversion():
    """Example: Convert custom CNN architecture"""
    print("\n" + "=" * 60)
    print("Custom CNN Automatic Conversion Example")
    print("=" * 60)
    
    # Define a custom CNN with fusible patterns
    class CustomCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                # Conv + BN + ReLU pattern
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                # Linear + ReLU pattern
                nn.Linear(256 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create and initialize model
    original_model = CustomCNN(num_classes=10)
    original_model = original_model.to(device)
    
    # Initialize weights (simulate pretrained model)
    for m in original_model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    # Convert model
    print("Converting custom CNN to use fused kernels...")
    fused_model = auto_convert_model(
        original_model,
        input_shape=(1, 3, 32, 32),
        device=device,
        validate_accuracy=True
    )
    
    # Test with CIFAR-10 sized input
    input_tensor = torch.randn(1, 3, 32, 32, device=device)
    
    # Benchmark
    original_time = benchmark_model(original_model, input_tensor)
    fused_time = benchmark_model(fused_model, input_tensor)
    speedup = original_time / fused_time
    
    print(f"\nCustom CNN Performance Results:")
    print(f"Original model: {original_time:.2f} ms")
    print(f"Fused model:    {fused_time:.2f} ms")
    print(f"Speedup:        {speedup:.2f}x")
    
    return fused_model


def example_transformer_conversion():
    """Example: Convert custom transformer architecture"""
    print("\n" + "=" * 60)
    print("Custom Transformer Automatic Conversion Example")
    print("=" * 60)
    
    # Define a simple transformer with fusible patterns
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model=512, nhead=8, num_layers=6, vocab_size=10000):
            super().__init__()
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
            
            # Transformer layers with fusible patterns
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'self_attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                    'norm1': nn.LayerNorm(d_model),
                    'ffn': nn.Sequential(
                        # Linear + ReLU pattern
                        nn.Linear(d_model, d_model * 4),
                        nn.ReLU(),
                        nn.Linear(d_model * 4, d_model)
                    ),
                    'norm2': nn.LayerNorm(d_model)
                })
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                # Another Linear + ReLU pattern
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, vocab_size)
            )
        
        def forward(self, x):
            # x shape: (batch_size, seq_len)
            seq_len = x.size(1)
            
            # Embedding + positional encoding
            x = self.embedding(x) + self.pos_encoding[:seq_len]
            
            # Transformer layers
            for layer in self.layers:
                # Self-attention
                attn_out, _ = layer['self_attn'](x, x, x)
                x = layer['norm1'](x + attn_out)
                
                # Feed-forward
                ffn_out = layer['ffn'](x)
                x = layer['norm2'](x + ffn_out)
            
            # Output projection
            x = self.output_proj(x)
            return x
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    original_model = SimpleTransformer(d_model=256, nhead=8, num_layers=4, vocab_size=1000)
    original_model = original_model.to(device)
    
    # Initialize weights
    for m in original_model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    # Convert model
    print("Converting transformer to use fused kernels...")
    fused_model = auto_convert_model(
        original_model,
        input_shape=(1, 50),  # batch_size=1, seq_len=50
        device=device,
        validate_accuracy=True
    )
    
    # Test with sequence input
    input_tensor = torch.randint(0, 1000, (1, 50), device=device)
    
    # Benchmark
    original_time = benchmark_model(original_model, input_tensor)
    fused_time = benchmark_model(fused_model, input_tensor)
    speedup = original_time / fused_time
    
    print(f"\nTransformer Performance Results:")
    print(f"Original model: {original_time:.2f} ms")
    print(f"Fused model:    {fused_time:.2f} ms")
    print(f"Speedup:        {speedup:.2f}x")
    
    return fused_model


def example_detailed_conversion_analysis():
    """Example: Detailed analysis of conversion process"""
    print("\n" + "=" * 60)
    print("Detailed Conversion Analysis Example")
    print("=" * 60)
    
    # Create a model with various fusible patterns
    class AnalysisModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Different types of fusible patterns
            self.conv_block = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            
            self.linear_relu = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU()
            )
            
            self.linear_gelu = nn.Sequential(
                nn.Linear(128, 256),
                nn.GELU()
            )
            
            self.layernorm_relu = nn.Sequential(
                nn.LayerNorm(256),
                nn.ReLU()
            )
            
            self.output = nn.Linear(256, 10)
        
        def forward(self, x):
            # Flatten spatial dimensions for linear layers
            x = self.conv_block(x)
            x = x.mean(dim=[2, 3])  # Global average pooling
            x = self.linear_relu(x)
            x = self.linear_gelu(x)
            x = self.layernorm_relu(x)
            x = self.output(x)
            return x
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = AnalysisModel().to(device)
    
    # Create converter with detailed logging
    converter = ModelConverter(validate_accuracy=True, accuracy_threshold=1e-5)
    
    # Convert model
    print("Performing detailed conversion analysis...")
    converted_model = converter.convert_model(
        model,
        input_shape=(1, 3, 32, 32),
        device=device
    )
    
    # Print detailed module comparison
    print("\n" + "=" * 50)
    print("MODULE STRUCTURE COMPARISON")
    print("=" * 50)
    
    print("\nOriginal Model Structure:")
    print(model)
    
    print("\nConverted Model Structure:")
    print(converted_model)
    
    return converted_model


def save_converted_model(model, save_path):
    """Save converted model with weights"""
    print(f"\nSaving converted model to {save_path}")
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully!")


def load_and_convert_pretrained_model(model_path, model_class, input_shape, device='cuda'):
    """Load pretrained model and convert it"""
    print(f"Loading pretrained model from {model_path}")
    
    # Create model instance
    model = model_class()
    
    # Load pretrained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Convert to use fused kernels
    converted_model = auto_convert_model(
        model,
        input_shape=input_shape,
        device=device,
        validate_accuracy=True
    )
    
    return converted_model


if __name__ == "__main__":
    print("Kernel Fusion - Automatic Model Conversion Examples")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    
    # Run examples
    try:
        # Example 1: ResNet conversion
        resnet_model = example_resnet_conversion()
        
        # Example 2: Custom CNN conversion
        cnn_model = example_custom_cnn_conversion()
        
        # Example 3: Transformer conversion
        transformer_model = example_transformer_conversion()
        
        # Example 4: Detailed analysis
        analysis_model = example_detailed_conversion_analysis()
        
        # Save converted models (optional)
        # save_converted_model(resnet_model, "resnet50_fused.pth")
        # save_converted_model(cnn_model, "custom_cnn_fused.pth")
        
        print("\n" + "=" * 60)
        print("All conversion examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
