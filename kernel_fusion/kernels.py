"""
Kernel-level interface providing direct access to individual CUDA kernels.
This module is for advanced users who need fine-grained control over kernel execution.
"""

import torch
from . import EXTENSION_LOADED

class ElementwiseKernels:
    """Collection of elementwise operation kernels."""
    
    @staticmethod
    def add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Direct access to add+relu kernel."""
        if not EXTENSION_LOADED or not a.is_cuda:
            raise RuntimeError("CUDA kernels not available")
        from . import _C
        return _C.elementwise_add_relu(a, b)
    
    @staticmethod
    def mul_tanh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Direct access to mul+tanh kernel."""
        if not EXTENSION_LOADED or not a.is_cuda:
            raise RuntimeError("CUDA kernels not available")
        from . import _C
        return _C.elementwise_mul_tanh(a, b)
    
    @staticmethod
    def bias_gelu(input: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Direct access to bias+gelu kernel."""
        if not EXTENSION_LOADED or not input.is_cuda:
            raise RuntimeError("CUDA kernels not available")
        from . import _C
        return _C.fused_bias_gelu(input, bias)

class ReductionKernels:
    """Collection of reduction operation kernels."""
    
    @staticmethod
    def sum_squared(input: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        """Direct access to sum of squares kernel."""
        if not EXTENSION_LOADED or not input.is_cuda:
            raise RuntimeError("CUDA kernels not available")
        from . import _C
        return _C.reduce_sum_squared(input, dim, keepdim)
    
    @staticmethod
    def mean_abs(input: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        """Direct access to mean absolute value kernel."""
        if not EXTENSION_LOADED or not input.is_cuda:
            raise RuntimeError("CUDA kernels not available")
        from . import _C
        return _C.reduce_mean_abs(input, dim, keepdim)

class FusionKernels:
    """Collection of complex fusion kernels."""
    
    @staticmethod
    def layer_norm_relu(
        input: torch.Tensor,
        normalized_shape,
        weight: torch.Tensor = None,
        bias: torch.Tensor = None,
        eps: float = 1e-5
    ) -> torch.Tensor:
        """Direct access to layer_norm+relu kernel."""
        if not EXTENSION_LOADED or not input.is_cuda:
            raise RuntimeError("CUDA kernels not available")
        from . import _C
        return _C.fused_layer_norm_relu(input, normalized_shape, weight, bias, eps)
    
    @staticmethod
    def gelu_dropout(
        input: torch.Tensor,
        p: float = 0.5,
        training: bool = True
    ) -> torch.Tensor:
        """Direct access to gelu+dropout kernel."""
        if not EXTENSION_LOADED or not input.is_cuda:
            raise RuntimeError("CUDA kernels not available")
        from . import _C
        return _C.fused_gelu_dropout(input, p, training)
    
    @staticmethod
    def attention_score(
        query: torch.Tensor,
        key: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        """Direct access to attention score kernel."""
        if not EXTENSION_LOADED or not query.is_cuda:
            raise RuntimeError("CUDA kernels not available")
        from . import _C
        return _C.fused_attention_score(query, key, scale)

# Convenience access
elementwise = ElementwiseKernels()
reduction = ReductionKernels()
fusion = FusionKernels()
