"""
PyTorch Frontend for KernelFusion Standalone Library

This module provides the same interface as the original kernel_fusion package
but uses the new standalone core library underneath for better performance
and framework independence.

Migration guide:
- All existing imports continue to work
- Performance improvements are automatic
- New features available through core API
"""

# Re-export everything from the new frontend
from .torch_bridge import (
    fused_elementwise_add_relu,
    fused_layer_norm_gelu,
    Context,
    Stream,
    KernelFusionError
)

# Maintain backward compatibility with old interface
from .legacy_compat import (
    # Legacy operation functions
    fused_layer_norm_relu,
    fused_attention,
    elementwise_add_relu,
    elementwise_mul_tanh,
    fused_bias_gelu,
    
    # Legacy classes for automatic conversion
    auto_convert_model,
    ModelConverter,
    
    # Legacy stream-aware conversion
    auto_convert_with_streams,
    convert_model_with_streams
)

# Version and metadata
__version__ = "2.0.0-standalone"
__author__ = "Yangyang Fu"

# Check if core library is available
try:
    import kernel_fusion_core
    CORE_AVAILABLE = True
    CUDA_AVAILABLE = kernel_fusion_core.cuda_available()
    EXTENSION_LOADED = CORE_AVAILABLE and CUDA_AVAILABLE
except ImportError:
    CORE_AVAILABLE = False
    CUDA_AVAILABLE = False
    EXTENSION_LOADED = False
    print("Warning: KernelFusion core library not found. Falling back to PyTorch implementations.")

# Legacy compatibility
def get_device_info():
    """Get device information (legacy compatibility)"""
    if CORE_AVAILABLE:
        context = Context()
        return context.get_device_info()
    else:
        return None

# Export all public symbols
__all__ = [
    # Core operations
    'fused_elementwise_add_relu',
    'fused_layer_norm_gelu',
    'fused_layer_norm_relu',
    'fused_attention',
    'elementwise_add_relu',
    'elementwise_mul_tanh',
    'fused_bias_gelu',
    
    # Automatic conversion
    'auto_convert_model',
    'ModelConverter',
    'auto_convert_with_streams',
    'convert_model_with_streams',
    
    # Core classes
    'Context',
    'Stream',
    'KernelFusionError',
    
    # Metadata
    'CORE_AVAILABLE',
    'CUDA_AVAILABLE', 
    'EXTENSION_LOADED',
    '__version__',
    '__author__'
]
