#!/usr/bin/env python3
"""
Method 3: Automatic Model Surgery Example

This example demonstrates how to automatically replace compatible operations
in pre-trained models with fused kernels without manual modification.

This approach is ideal for:
- Optimizing existing pre-trained models
- Batch processing multiple models
- Research and experimentation with different fusion strategies
"""

import torch
import torch.nn as nn
import torchvision.models as models
import kernel_fusion as kf
import copy
from typing import Dict, List, Tuple, Any

# Custom fused modules for replacement
class FusedLinearReLU(nn.Module):
    """Fused Linear + ReLU module"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
    
    def forward(self, x):
        return kf.ops.fused_linear_relu(x, self.weight, self.bias)

class FusedConvBNReLU(nn.Module):
    """Fused Conv2d + BatchNorm + ReLU module"""
    def __init__(self, conv_layer, bn_layer):
        super().__init__()
        self.conv_weight = conv_layer.weight
        self.conv_bias = conv_layer.bias
        self.bn_weight = bn_layer.weight
        self.bn_bias = bn_layer.bias
        self.bn_running_mean = bn_layer.running_mean
        self.bn_running_var = bn_layer.running_var
        self.bn_eps = bn_layer.eps
    
    def forward(self, x):
        return kf.ops.fused_conv2d_batchnorm_relu(
            x, self.conv_weight, self.conv_bias,
            self.bn_weight, self.bn_bias,
            self.bn_running_mean, self.bn_running_var,
            eps=self.bn_eps
        )

# Pattern detection functions
def detect_linear_relu_pattern(module: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Detect Linear + ReLU patterns in a model
    Returns list of (name, sequential_module) tuples
    """
    patterns = []
    
    for name, child in module.named_children():
        if isinstance(child, nn.Sequential):
            # Check for Linear + ReLU pattern
            if (len(child) == 2 and 
                isinstance(child[0], nn.Linear) and 
                isinstance(child[1], nn.ReLU)):
                patterns.append((name, child))
        
        # Recursively check children
        patterns.extend([
            (f"{name}.{sub_name}", sub_module) 
            for sub_name, sub_module in detect_linear_relu_pattern(child)
        ])
    
    return patterns

def detect_conv_bn_relu_pattern(module: nn.Module) -> List[Tuple[str, List[nn.Module]]]:
    """
    Detect Conv2d + BatchNorm2d + ReLU patterns
    Returns list of (base_name, [conv, bn, relu]) tuples
    """
    patterns = []
    
    def find_consecutive_pattern(parent_module, parent_name=""):
        children = list(parent_module.named_children())
        
        for i in range(len(children) - 2):
            name1, module1 = children[i]
            name2, module2 = children[i + 1]
            name3, module3 = children[i + 2]
            
            if (isinstance(module1, nn.Conv2d) and 
                isinstance(module2, nn.BatchNorm2d) and 
                isinstance(module3, nn.ReLU)):
                
                base_name = f"{parent_name}.{name1}" if parent_name else name1
                patterns.append((base_name, [module1, module2, module3]))
        
        # Recursively check children
        for name, child in children:
            child_name = f"{parent_name}.{name}" if parent_name else name
            find_consecutive_pattern(child, child_name)
    
    find_consecutive_pattern(module)
    return patterns

# Model surgery functions
def replace_linear_relu_patterns(model: nn.Module) -> nn.Module:
    """
    Automatically replace Linear + ReLU patterns with fused equivalents
    """
    print("Searching for Linear + ReLU patterns...")
    patterns = detect_linear_relu_pattern(model)
    
    replaced_count = 0
    for pattern_name, sequential_module in patterns:
        print(f"Found pattern at: {pattern_name}")
        
        # Extract Linear and ReLU layers
        linear_layer = sequential_module[0]
        relu_layer = sequential_module[1]
        
        # Create fused replacement
        fused_module = FusedLinearReLU(
            linear_layer.in_features,
            linear_layer.out_features,
            bias=linear_layer.bias is not None
        )
        
        # Copy weights and bias
        fused_module.weight.data = linear_layer.weight.data.clone()
        if linear_layer.bias is not None:
            fused_module.bias.data = linear_layer.bias.data.clone()
        
        # Replace in model
        parent = model
        attrs = pattern_name.split('.')
        for attr in attrs[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, attrs[-1], fused_module)
        
        replaced_count += 1
        print(f"Replaced with FusedLinearReLU")
    
    print(f"Total Linear + ReLU patterns replaced: {replaced_count}")
    return model

def replace_conv_bn_relu_patterns(model: nn.Module) -> nn.Module:
    """
    Automatically replace Conv2d + BatchNorm2d + ReLU patterns
    """
    print("Searching for Conv2d + BatchNorm2d + ReLU patterns...")
    # Note: This is more complex due to the sequential nature
    # For now, we'll focus on patterns within nn.Sequential modules
    
    replaced_count = 0
    
    def replace_in_sequential(module, module_name=""):
        nonlocal replaced_count
        
        if isinstance(module, nn.Sequential):
            new_layers = []
            i = 0
            while i < len(module):
                layer = module[i]
                
                # Check for Conv + BN + ReLU pattern
                if (i <= len(module) - 3 and
                    isinstance(layer, nn.Conv2d) and
                    isinstance(module[i + 1], nn.BatchNorm2d) and
                    isinstance(module[i + 2], nn.ReLU)):
                    
                    conv_layer = layer
                    bn_layer = module[i + 1]
                    relu_layer = module[i + 2]
                    
                    print(f"Found Conv+BN+ReLU pattern at {module_name}[{i}:{i+3}]")
                    
                    # Create fused replacement
                    fused_layer = FusedConvBNReLU(conv_layer, bn_layer)
                    new_layers.append(fused_layer)
                    
                    replaced_count += 1
                    i += 3  # Skip the next two layers
                else:
                    new_layers.append(layer)
                    i += 1
            
            # Replace the sequential module's layers
            if replaced_count > 0:
                module._modules.clear()
                for idx, layer in enumerate(new_layers):
                    module.add_module(str(idx), layer)
        
        # Recursively process children
        for name, child in module.named_children():
            child_name = f"{module_name}.{name}" if module_name else name
            replace_in_sequential(child, child_name)
    
    replace_in_sequential(model)
    print(f"Total Conv + BN + ReLU patterns replaced: {replaced_count}")
    return model

def optimize_model_with_fusion(model: nn.Module, 
                              replace_linear_relu: bool = True,
                              replace_conv_bn_relu: bool = True) -> nn.Module:
    """
    Comprehensive model optimization with fusion
    """
    print("=" * 50)
    print("Starting Automatic Model Optimization")
    print("=" * 50)
    
    # Create a deep copy to avoid modifying the original
    optimized_model = copy.deepcopy(model)
    
    if replace_linear_relu:
        optimized_model = replace_linear_relu_patterns(optimized_model)
        print()
    
    if replace_conv_bn_relu:
        optimized_model = replace_conv_bn_relu_patterns(optimized_model)
        print()
    
    print("Model optimization completed!")
    return optimized_model

# Example models for testing
def create_test_mlp():
    """Create a simple MLP for testing"""
    return nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def create_test_cnn():
    """Create a simple CNN for testing"""
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

# Validation functions
def validate_model_equivalence(original_model: nn.Module, 
                             optimized_model: nn.Module,
                             test_input: torch.Tensor,
                             tolerance: float = 1e-5) -> bool:
    """
    Validate that the optimized model produces equivalent results
    """
    original_model.eval()
    optimized_model.eval()
    
    with torch.no_grad():
        original_output = original_model(test_input)
        optimized_output = optimized_model(test_input)
    
    max_diff = torch.max(torch.abs(original_output - optimized_output))
    print(f"Maximum output difference: {max_diff:.2e}")
    
    is_equivalent = max_diff < tolerance
    print(f"Models are equivalent: {is_equivalent}")
    
    return is_equivalent

def benchmark_models(original_model: nn.Module,
                    optimized_model: nn.Module,
                    test_input: torch.Tensor,
                    num_iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark performance of original vs optimized models
    """
    import time
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = original_model(test_input)
            _ = optimized_model(test_input)
    
    # Benchmark original model
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = original_model(test_input)
    torch.cuda.synchronize()
    original_time = time.time() - start_time
    
    # Benchmark optimized model
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = optimized_model(test_input)
    torch.cuda.synchronize()
    optimized_time = time.time() - start_time
    
    speedup = original_time / optimized_time
    
    results = {
        'original_time': original_time,
        'optimized_time': optimized_time,
        'speedup': speedup
    }
    
    print(f"Original model time: {original_time:.4f}s")
    print(f"Optimized model time: {optimized_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    return results

# Demonstration functions
def demo_mlp_optimization():
    """Demonstrate MLP optimization"""
    print("\n=== MLP Optimization Demo ===")
    
    # Create test model
    model = create_test_mlp().cuda()
    print(f"Original model: {model}")
    
    # Optimize model
    optimized_model = optimize_model_with_fusion(model).cuda()
    print(f"Optimized model structure changed")
    
    # Test with sample input
    test_input = torch.randn(64, 784, device='cuda')
    
    # Validate equivalence
    is_valid = validate_model_equivalence(model, optimized_model, test_input)
    
    # Benchmark performance
    if is_valid:
        benchmark_results = benchmark_models(model, optimized_model, test_input)
    
    return optimized_model

def demo_cnn_optimization():
    """Demonstrate CNN optimization"""
    print("\n=== CNN Optimization Demo ===")
    
    # Create test model
    model = create_test_cnn().cuda()
    print(f"Original CNN created")
    
    # Optimize model
    optimized_model = optimize_model_with_fusion(model).cuda()
    print(f"CNN optimization completed")
    
    # Test with sample input
    test_input = torch.randn(16, 3, 32, 32, device='cuda')
    
    # Validate equivalence
    is_valid = validate_model_equivalence(model, optimized_model, test_input)
    
    # Benchmark performance
    if is_valid:
        benchmark_results = benchmark_models(model, optimized_model, test_input)
    
    return optimized_model

def demo_pretrained_model_optimization():
    """Demonstrate optimization of a pre-trained model"""
    print("\n=== Pre-trained Model Optimization Demo ===")
    
    try:
        # Load a pre-trained ResNet18 (modify for smaller model if needed)
        print("Loading pre-trained ResNet18...")
        model = models.resnet18(pretrained=False)  # Set to False to avoid download
        model = model.cuda()
        
        print("Analyzing model structure...")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Optimize the model
        optimized_model = optimize_model_with_fusion(
            model, 
            replace_linear_relu=True,
            replace_conv_bn_relu=True
        ).cuda()
        
        # Test with ImageNet-sized input
        test_input = torch.randn(8, 3, 224, 224, device='cuda')
        
        # Validate equivalence
        is_valid = validate_model_equivalence(model, optimized_model, test_input)
        
        if is_valid:
            benchmark_results = benchmark_models(model, optimized_model, test_input, num_iterations=50)
            
        return optimized_model
        
    except Exception as e:
        print(f"Pre-trained model demo failed: {e}")
        print("This is expected if torchvision models are not available")

def main():
    """Run all automatic model surgery examples"""
    print("Automatic Model Surgery Examples")
    print("=" * 50)
    
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("CUDA not available, performance benchmarks may not be meaningful...")
        
        # Run demonstrations
        mlp_model = demo_mlp_optimization()
        cnn_model = demo_cnn_optimization()
        pretrained_model = demo_pretrained_model_optimization()
        
        print("\n" + "=" * 50)
        print("✅ All automatic model surgery examples completed!")
        print("\nKey Benefits of Automatic Surgery:")
        print("- No manual code modification required")
        print("- Works with pre-trained models")
        print("- Maintains model equivalence")
        print("- Scalable to large models")
        print("- Preserves weights and behavior")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Note: Make sure kernel_fusion is properly installed and CUDA is available")

if __name__ == "__main__":
    main()
