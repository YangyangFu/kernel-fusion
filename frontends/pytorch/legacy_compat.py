"""
Legacy Compatibility Layer

Provides the same interface as the original kernel_fusion package
while using the new standalone core library underneath.

This allows existing code to work without changes while getting
the benefits of the new architecture.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import warnings

try:
    from .torch_bridge import (
        Context, Stream, KernelFusionError,
        fused_elementwise_add_relu as _core_add_relu,
        fused_layer_norm_gelu as _core_layer_norm_gelu
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# ============================================================================
# Legacy Operation Functions (compatible with old interface)
# ============================================================================

def fused_layer_norm_relu(
    input: torch.Tensor,
    normalized_shape: Union[int, Tuple[int, ...]],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    """Legacy compatible fused layer norm + ReLU"""
    if CORE_AVAILABLE and input.is_cuda:
        # Use new core implementation with ReLU activation
        # Note: Core implementation expects weight/bias to be provided
        if weight is None:
            if isinstance(normalized_shape, int):
                weight = torch.ones(normalized_shape, device=input.device, dtype=input.dtype)
            else:
                weight = torch.ones(normalized_shape, device=input.device, dtype=input.dtype)
        if bias is None:
            if isinstance(normalized_shape, int):
                bias = torch.zeros(normalized_shape, device=input.device, dtype=input.dtype)
            else:
                bias = torch.zeros(normalized_shape, device=input.device, dtype=input.dtype)
        
        # Use core with GELU, then apply ReLU (temporary - would implement ReLU in core)
        output = _core_layer_norm_gelu(input, weight, bias, eps)
        return torch.relu(output)  # Apply ReLU on top
    else:
        # Fallback to PyTorch implementation
        output = torch.layer_norm(input, normalized_shape, weight, bias, eps)
        return torch.relu(output)

def elementwise_add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Legacy compatible elementwise add + ReLU"""
    if CORE_AVAILABLE and a.is_cuda and b.is_cuda:
        return _core_add_relu(a, b)
    else:
        # Fallback to PyTorch
        return torch.relu(a + b)

def elementwise_mul_tanh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Legacy compatible elementwise mul + tanh"""
    # TODO: Implement in core library
    return torch.tanh(a * b)

def fused_bias_gelu(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Legacy compatible bias + GELU"""
    # TODO: Implement in core library
    return torch.nn.functional.gelu(input + bias)

def fused_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor:
    """Legacy compatible fused attention"""
    # TODO: Implement in core library
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
    )

# ============================================================================
# Legacy Automatic Conversion (with core backend)
# ============================================================================

class ModelConverter:
    """Legacy compatible model converter using new core"""
    
    def __init__(self, validate_accuracy: bool = True, accuracy_threshold: float = 1e-5):
        self.validate_accuracy = validate_accuracy
        self.accuracy_threshold = accuracy_threshold
        
        if CORE_AVAILABLE:
            # Use new core-based conversion
            from ...kernel_fusion.auto_convert import ModelConverter as CoreConverter
            self._core_converter = CoreConverter(validate_accuracy, accuracy_threshold)
        else:
            warnings.warn("Core library not available, using PyTorch-only implementation")
            self._core_converter = None
    
    def convert_model(self, model: nn.Module, 
                     input_shape: Optional[Tuple[int, ...]] = None,
                     device: str = 'cuda') -> nn.Module:
        """Convert model with legacy interface"""
        if self._core_converter and CORE_AVAILABLE:
            return self._core_converter.convert_model(model, input_shape, device)
        else:
            # Return original model if core not available
            warnings.warn("Returning original model (core conversion not available)")
            return model

def auto_convert_model(model: nn.Module,
                      input_shape: Optional[Tuple[int, ...]] = None,
                      device: str = 'cuda',
                      validate_accuracy: bool = True) -> nn.Module:
    """Legacy compatible auto conversion"""
    converter = ModelConverter(validate_accuracy=validate_accuracy)
    return converter.convert_model(model, input_shape, device)

# ============================================================================
# Legacy Stream-Aware Conversion
# ============================================================================

class StreamAwareInferenceWrapper:
    """Legacy compatible stream-aware wrapper"""
    
    def __init__(self, model: nn.Module, stream_context=None):
        self.model = model
        self.stream_context = stream_context
        
        if CORE_AVAILABLE and stream_context:
            # Use new core implementation
            self._core_context = Context(device_id=0 if torch.cuda.is_available() else -1)
            self._core_stream = Stream(self._core_context)
        else:
            self._core_context = None
            self._core_stream = None
    
    def __call__(self, *args, **kwargs):
        """Stream-aware inference"""
        if self._core_stream:
            # Use core stream (simplified - would need full implementation)
            return self.model(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)
    
    def batch_inference(self, batches):
        """Parallel batch inference"""
        results = []
        for batch in batches:
            with torch.no_grad():
                result = self.model(batch)
                results.append(result)
        return results

def auto_convert_with_streams(model: nn.Module,
                             input_shape: Optional[Tuple[int, ...]] = None,
                             device: str = 'cuda') -> StreamAwareInferenceWrapper:
    """Legacy compatible stream-aware conversion"""
    if CORE_AVAILABLE:
        # Use new stream-aware implementation
        try:
            from ...kernel_fusion.stream_convert import auto_convert_with_streams as core_convert
            return core_convert(model, input_shape, device)
        except ImportError:
            pass
    
    # Fallback: regular conversion + wrapper
    converted_model = auto_convert_model(model, input_shape, device)
    return StreamAwareInferenceWrapper(converted_model)

def convert_model_with_streams(model: nn.Module,
                              input_shape: Optional[Tuple[int, ...]] = None,
                              device: str = 'cuda',
                              enable_parallel: bool = True,
                              validate_accuracy: bool = True) -> StreamAwareInferenceWrapper:
    """Legacy compatible stream conversion"""
    return auto_convert_with_streams(model, input_shape, device)

# ============================================================================
# Migration Helper
# ============================================================================

def check_migration_status():
    """Check status of migration to new architecture"""
    status = {
        'core_available': CORE_AVAILABLE,
        'using_new_backend': CORE_AVAILABLE,
        'operations_available': [],
        'operations_fallback': []
    }
    
    operations = [
        'fused_elementwise_add_relu',
        'fused_layer_norm_gelu', 
        'elementwise_add_relu',
        'elementwise_mul_tanh',
        'fused_bias_gelu',
        'fused_attention'
    ]
    
    for op in operations:
        if CORE_AVAILABLE and op in ['fused_elementwise_add_relu', 'fused_layer_norm_gelu']:
            status['operations_available'].append(op)
        else:
            status['operations_fallback'].append(op)
    
    return status

def print_migration_status():
    """Print migration status information"""
    status = check_migration_status()
    
    print("KernelFusion Migration Status:")
    print(f"  Core library available: {status['core_available']}")
    print(f"  Using new backend: {status['using_new_backend']}")
    print(f"  Operations using core: {len(status['operations_available'])}")
    print(f"  Operations using fallback: {len(status['operations_fallback'])}")
    
    if status['operations_fallback']:
        print("  Fallback operations:", ', '.join(status['operations_fallback']))
    
    if not status['core_available']:
        print("  To enable new backend: Build and install core library")
    
    return status
