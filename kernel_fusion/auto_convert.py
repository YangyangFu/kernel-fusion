#!/usr/bin/env python3
"""
Automatic Model Conversion for Kernel Fusion

This module provides automatic conversion of pretrained PyTorch models to use
fused kernels while preserving all weights and maintaining numerical accuracy.

Key Features:
- Automatic pattern detection (Linear+ReLU, Conv+BN+ReLU, LayerNorm+activation, etc.)
- Weight preservation and transfer
- Numerical accuracy validation
- Support for popular model architectures (BERT, ResNet, ViT, etc.)
- Comprehensive logging and debugging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import OrderedDict
import warnings

# Import kernel fusion operations
try:
    import kernel_fusion as kf
except ImportError:
    warnings.warn("kernel_fusion module not available. Using mock implementations.")
    # Mock implementation for testing
    class MockOps:
        @staticmethod
        def fused_linear_relu(x, weight, bias=None):
            return F.relu(F.linear(x, weight, bias))
        
        @staticmethod
        def fused_linear_gelu(x, weight, bias=None):
            return F.gelu(F.linear(x, weight, bias))
        
        @staticmethod
        def fused_conv2d_batchnorm_relu(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
            x = F.conv2d(x, conv_weight, conv_bias)
            x = F.batch_norm(x, bn_mean, bn_var, bn_weight, bn_bias, training=False, eps=eps)
            return F.relu(x)
        
        @staticmethod
        def fused_layer_norm_relu(x, weight, bias, eps=1e-5):
            x = F.layer_norm(x, weight.shape, weight, bias, eps)
            return F.relu(x)
    
    class MockKF:
        ops = MockOps()
    
    kf = MockKF()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusedLinearReLU(nn.Module):
    """Fused Linear + ReLU replacement module"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kf.ops.fused_linear_relu(x, self.weight, self.bias)


class FusedLinearGELU(nn.Module):
    """Fused Linear + GELU replacement module"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kf.ops.fused_linear_gelu(x, self.weight, self.bias)


class FusedConv2dBNReLU(nn.Module):
    """Fused Conv2d + BatchNorm2d + ReLU replacement module"""
    
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        super().__init__()
        # Store conv parameters
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode
        
        # Store weights and biases as parameters
        self.conv_weight = nn.Parameter(conv.weight.clone())
        if conv.bias is not None:
            self.conv_bias = nn.Parameter(conv.bias.clone())
        else:
            self.register_parameter('conv_bias', None)
        
        # Store BN parameters
        self.bn_weight = nn.Parameter(bn.weight.clone())
        self.bn_bias = nn.Parameter(bn.bias.clone())
        self.register_buffer('bn_running_mean', bn.running_mean.clone())
        self.register_buffer('bn_running_var', bn.running_var.clone())
        self.bn_eps = bn.eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kf.ops.fused_conv2d_batchnorm_relu(
            x, self.conv_weight, self.conv_bias,
            self.bn_weight, self.bn_bias,
            self.bn_running_mean, self.bn_running_var,
            eps=self.bn_eps
        )


class FusedLayerNormReLU(nn.Module):
    """Fused LayerNorm + ReLU replacement module"""
    
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], 
                 eps: float = 1e-5, elementwise_affine: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kf.ops.fused_layer_norm_relu(x, self.weight, self.bias, self.eps)


class PatternDetector:
    """Detects fusible patterns in PyTorch models"""
    
    @staticmethod
    def detect_linear_activation_patterns(model: nn.Module) -> List[Tuple[str, nn.Module, str]]:
        """
        Detect Linear + Activation patterns
        Returns: [(module_path, sequential_module, activation_type), ...]
        """
        patterns = []
        
        def _detect_in_module(module: nn.Module, prefix: str = ""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, nn.Sequential) and len(child) >= 2:
                    # Check for Linear + Activation
                    if isinstance(child[0], nn.Linear):
                        if isinstance(child[1], nn.ReLU):
                            patterns.append((full_name, child, "relu"))
                        elif isinstance(child[1], nn.GELU):
                            patterns.append((full_name, child, "gelu"))
                
                # Recursively search in children
                _detect_in_module(child, full_name)
        
        _detect_in_module(model)
        return patterns
    
    @staticmethod
    def detect_conv_bn_relu_patterns(model: nn.Module) -> List[Tuple[str, List[nn.Module]]]:
        """
        Detect Conv2d + BatchNorm2d + ReLU patterns
        Returns: [(base_path, [conv, bn, relu]), ...]
        """
        patterns = []
        
        def _detect_in_sequential(seq_module: nn.Module, prefix: str = ""):
            children = list(seq_module.children())
            names = [name for name, _ in seq_module.named_children()]
            
            for i in range(len(children) - 2):
                if (isinstance(children[i], nn.Conv2d) and 
                    isinstance(children[i + 1], nn.BatchNorm2d) and 
                    isinstance(children[i + 2], nn.ReLU)):
                    
                    pattern_name = f"{prefix}.{names[i]}" if prefix else names[i]
                    patterns.append((pattern_name, [children[i], children[i + 1], children[i + 2]]))
        
        def _detect_in_module(module: nn.Module, prefix: str = ""):
            if isinstance(module, nn.Sequential):
                _detect_in_sequential(module, prefix)
            
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                _detect_in_module(child, full_name)
        
        _detect_in_module(model)
        return patterns
    
    @staticmethod
    def detect_layernorm_activation_patterns(model: nn.Module) -> List[Tuple[str, nn.Module, str]]:
        """
        Detect LayerNorm + Activation patterns
        Returns: [(module_path, sequential_module, activation_type), ...]
        """
        patterns = []
        
        def _detect_in_module(module: nn.Module, prefix: str = ""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, nn.Sequential) and len(child) >= 2:
                    # Check for LayerNorm + Activation
                    if isinstance(child[0], nn.LayerNorm):
                        if isinstance(child[1], nn.ReLU):
                            patterns.append((full_name, child, "relu"))
                        elif isinstance(child[1], nn.GELU):
                            patterns.append((full_name, child, "gelu"))
                
                # Recursively search in children
                _detect_in_module(child, full_name)
        
        _detect_in_module(model)
        return patterns


class ModelConverter:
    """Main class for automatic model conversion to fused kernels"""
    
    def __init__(self, validate_accuracy: bool = True, accuracy_threshold: float = 1e-5):
        self.validate_accuracy = validate_accuracy
        self.accuracy_threshold = accuracy_threshold
        self.detector = PatternDetector()
        self.conversion_stats = {
            'linear_relu_replaced': 0,
            'linear_gelu_replaced': 0,
            'conv_bn_relu_replaced': 0,
            'layernorm_relu_replaced': 0,
            'total_replaced': 0
        }
    
    def convert_model(self, model: nn.Module, 
                     input_shape: Optional[Tuple[int, ...]] = None,
                     device: str = 'cuda') -> nn.Module:
        """
        Convert a pretrained model to use fused kernels
        
        Args:
            model: Pretrained PyTorch model
            input_shape: Shape for validation input (batch_size, ...)
            device: Device to use for conversion and validation
            
        Returns:
            Converted model with fused operations
        """
        logger.info("Starting automatic model conversion...")
        
        # Create a deep copy to avoid modifying the original
        converted_model = copy.deepcopy(model)
        original_model = copy.deepcopy(model) if self.validate_accuracy else None
        
        # Move models to device
        converted_model = converted_model.to(device)
        if original_model is not None:
            original_model = original_model.to(device)
        
        # Reset conversion stats
        self.conversion_stats = {k: 0 for k in self.conversion_stats}
        
        # Apply conversions
        self._convert_linear_activation_patterns(converted_model)
        self._convert_conv_bn_relu_patterns(converted_model)
        self._convert_layernorm_activation_patterns(converted_model)
        
        # Update total
        self.conversion_stats['total_replaced'] = sum(
            v for k, v in self.conversion_stats.items() if k != 'total_replaced'
        )
        
        # Validate accuracy if requested
        if self.validate_accuracy and input_shape is not None:
            self._validate_conversion_accuracy(original_model, converted_model, input_shape, device)
        
        # Print conversion summary
        self._print_conversion_summary()
        
        logger.info("Model conversion completed successfully!")
        return converted_model
    
    def _convert_linear_activation_patterns(self, model: nn.Module):
        """Convert Linear + Activation patterns"""
        patterns = self.detector.detect_linear_activation_patterns(model)
        
        for pattern_path, sequential_module, activation_type in patterns:
            logger.info(f"Converting Linear+{activation_type.upper()} at {pattern_path}")
            
            linear_layer = sequential_module[0]
            
            # Create fused replacement
            if activation_type == "relu":
                fused_module = FusedLinearReLU(
                    linear_layer.in_features,
                    linear_layer.out_features,
                    bias=linear_layer.bias is not None,
                    device=linear_layer.weight.device,
                    dtype=linear_layer.weight.dtype
                )
                self.conversion_stats['linear_relu_replaced'] += 1
            elif activation_type == "gelu":
                fused_module = FusedLinearGELU(
                    linear_layer.in_features,
                    linear_layer.out_features,
                    bias=linear_layer.bias is not None,
                    device=linear_layer.weight.device,
                    dtype=linear_layer.weight.dtype
                )
                self.conversion_stats['linear_gelu_replaced'] += 1
            else:
                continue  # Skip unsupported activations
            
            # Transfer weights
            fused_module.weight.data = linear_layer.weight.data.clone()
            if linear_layer.bias is not None:
                fused_module.bias.data = linear_layer.bias.data.clone()
            
            # Replace in model
            self._replace_module_at_path(model, pattern_path, fused_module)
    
    def _convert_conv_bn_relu_patterns(self, model: nn.Module):
        """Convert Conv2d + BatchNorm2d + ReLU patterns"""
        patterns = self.detector.detect_conv_bn_relu_patterns(model)
        
        for pattern_path, modules in patterns:
            logger.info(f"Converting Conv+BN+ReLU at {pattern_path}")
            
            conv_layer, bn_layer, relu_layer = modules
            
            # Create fused replacement
            fused_module = FusedConv2dBNReLU(conv_layer, bn_layer)
            
            # Replace the first module (conv) and remove the others
            parent = self._get_parent_module(model, pattern_path)
            parent_sequential = self._get_module_at_path(model, pattern_path.rsplit('.', 1)[0])
            
            # Find indices to replace
            children_list = list(parent_sequential.children())
            conv_idx = children_list.index(conv_layer)
            
            # Create new sequential without BN and ReLU
            new_children = children_list[:conv_idx] + [fused_module] + children_list[conv_idx + 3:]
            new_sequential = nn.Sequential(*new_children)
            
            # Replace the sequential module
            self._replace_module_at_path(model, pattern_path.rsplit('.', 1)[0], new_sequential)
            self.conversion_stats['conv_bn_relu_replaced'] += 1
    
    def _convert_layernorm_activation_patterns(self, model: nn.Module):
        """Convert LayerNorm + Activation patterns"""
        patterns = self.detector.detect_layernorm_activation_patterns(model)
        
        for pattern_path, sequential_module, activation_type in patterns:
            if activation_type != "relu":
                continue  # Only ReLU supported for now
                
            logger.info(f"Converting LayerNorm+{activation_type.upper()} at {pattern_path}")
            
            layernorm_layer = sequential_module[0]
            
            # Create fused replacement
            fused_module = FusedLayerNormReLU(
                layernorm_layer.normalized_shape,
                eps=layernorm_layer.eps,
                elementwise_affine=layernorm_layer.elementwise_affine,
                device=layernorm_layer.weight.device if layernorm_layer.weight is not None else None,
                dtype=layernorm_layer.weight.dtype if layernorm_layer.weight is not None else None
            )
            
            # Transfer weights
            if layernorm_layer.weight is not None:
                fused_module.weight.data = layernorm_layer.weight.data.clone()
            if layernorm_layer.bias is not None:
                fused_module.bias.data = layernorm_layer.bias.data.clone()
            
            # Replace in model
            self._replace_module_at_path(model, pattern_path, fused_module)
            self.conversion_stats['layernorm_relu_replaced'] += 1
    
    def _get_module_at_path(self, model: nn.Module, path: str) -> nn.Module:
        """Get module at given dot-separated path"""
        if not path:
            return model
        
        module = model
        for attr in path.split('.'):
            module = getattr(module, attr)
        return module
    
    def _get_parent_module(self, model: nn.Module, path: str) -> nn.Module:
        """Get parent module of module at given path"""
        if '.' not in path:
            return model
        
        parent_path = path.rsplit('.', 1)[0]
        return self._get_module_at_path(model, parent_path)
    
    def _replace_module_at_path(self, model: nn.Module, path: str, new_module: nn.Module):
        """Replace module at given path with new module"""
        if '.' not in path:
            setattr(model, path, new_module)
            return
        
        parent_path, attr_name = path.rsplit('.', 1)
        parent_module = self._get_module_at_path(model, parent_path)
        setattr(parent_module, attr_name, new_module)
    
    def _validate_conversion_accuracy(self, original_model: nn.Module, 
                                    converted_model: nn.Module,
                                    input_shape: Tuple[int, ...], 
                                    device: str):
        """Validate that conversion preserves numerical accuracy"""
        logger.info("Validating conversion accuracy...")
        
        # Create test input
        test_input = torch.randn(input_shape, device=device)
        
        # Set models to eval mode
        original_model.eval()
        converted_model.eval()
        
        with torch.no_grad():
            try:
                original_output = original_model(test_input)
                converted_output = converted_model(test_input)
                
                # Handle different output types
                if isinstance(original_output, tuple):
                    max_diff = max(torch.max(torch.abs(orig - conv)).item() 
                                 for orig, conv in zip(original_output, converted_output))
                else:
                    max_diff = torch.max(torch.abs(original_output - converted_output)).item()
                
                logger.info(f"Maximum output difference: {max_diff:.2e}")
                
                if max_diff > self.accuracy_threshold:
                    logger.warning(f"Accuracy difference {max_diff:.2e} exceeds threshold {self.accuracy_threshold:.2e}")
                else:
                    logger.info("✓ Conversion maintains numerical accuracy")
                    
            except Exception as e:
                logger.error(f"Accuracy validation failed: {e}")
    
    def _print_conversion_summary(self):
        """Print summary of conversion statistics"""
        logger.info("\n" + "="*50)
        logger.info("CONVERSION SUMMARY")
        logger.info("="*50)
        logger.info(f"Linear + ReLU replaced:     {self.conversion_stats['linear_relu_replaced']}")
        logger.info(f"Linear + GELU replaced:     {self.conversion_stats['linear_gelu_replaced']}")
        logger.info(f"Conv + BN + ReLU replaced:  {self.conversion_stats['conv_bn_relu_replaced']}")
        logger.info(f"LayerNorm + ReLU replaced:  {self.conversion_stats['layernorm_relu_replaced']}")
        logger.info("-" * 50)
        logger.info(f"Total patterns replaced:    {self.conversion_stats['total_replaced']}")
        logger.info("="*50)


def auto_convert_model(model: nn.Module, 
                      input_shape: Optional[Tuple[int, ...]] = None,
                      device: str = 'cuda',
                      validate_accuracy: bool = True,
                      accuracy_threshold: float = 1e-5) -> nn.Module:
    """
    Convenience function for automatic model conversion
    
    Args:
        model: Pretrained PyTorch model to convert
        input_shape: Input shape for validation (batch_size, ...)
        device: Device to use for conversion and validation
        validate_accuracy: Whether to validate numerical accuracy
        accuracy_threshold: Maximum allowed difference in outputs
        
    Returns:
        Converted model with fused operations
        
    Example:
        >>> import torchvision.models as models
        >>> from kernel_fusion.auto_convert import auto_convert_model
        >>> 
        >>> # Load pretrained ResNet
        >>> model = models.resnet50(pretrained=True)
        >>> 
        >>> # Convert to use fused kernels
        >>> fused_model = auto_convert_model(
        ...     model, 
        ...     input_shape=(1, 3, 224, 224),
        ...     device='cuda'
        ... )
    """
    converter = ModelConverter(validate_accuracy, accuracy_threshold)
    return converter.convert_model(model, input_shape, device)
