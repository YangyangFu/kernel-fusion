"""
High-level operation interface for kernel fusion library.
This module provides PyTorch-compatible functions that automatically
dispatch to optimized CUDA kernels when available.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from . import EXTENSION_LOADED

def fused_layer_norm_relu(
    input: torch.Tensor,
    normalized_shape: Union[int, Tuple[int, ...]],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Fused Layer Normalization + ReLU activation.
    
    Args:
        input: Input tensor
        normalized_shape: Shape for normalization
        weight: Optional weight parameter
        bias: Optional bias parameter
        eps: Small constant for numerical stability
        
    Returns:
        Output tensor after layer norm and ReLU
    """
    if EXTENSION_LOADED and input.is_cuda:
        from . import _C
        return _C.fused_layer_norm_relu(input, normalized_shape, weight, bias, eps)
    else:
        # Fallback to PyTorch implementation
        normalized = F.layer_norm(input, normalized_shape, weight, bias, eps)
        return F.relu(normalized)

def fused_gelu_dropout(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True
) -> torch.Tensor:
    """
    Fused GELU + Dropout operation.
    
    Args:
        input: Input tensor
        p: Dropout probability
        training: Whether in training mode
        
    Returns:
        Output tensor after GELU and dropout
    """
    if EXTENSION_LOADED and input.is_cuda:
        from . import _C
        return _C.fused_gelu_dropout(input, p, training)
    else:
        # Fallback implementation
        gelu_out = F.gelu(input)
        return F.dropout(gelu_out, p, training)

def fused_attention_score(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Fused attention score computation with scaling.
    
    Args:
        query: Query tensor [batch, seq_len, dim]
        key: Key tensor [batch, seq_len, dim]
        scale: Optional scaling factor
        
    Returns:
        Attention scores [batch, seq_len, seq_len]
    """
    if scale is None:
        scale = 1.0 / (query.size(-1) ** 0.5)
        
    if EXTENSION_LOADED and query.is_cuda:
        from . import _C
        return _C.fused_attention_score(query, key, scale)
    else:
        # Fallback implementation
        scores = torch.matmul(query, key.transpose(-2, -1))
        return scores * scale

def fused_bias_gelu(
    input: torch.Tensor,
    bias: torch.Tensor
) -> torch.Tensor:
    """
    Fused bias addition + GELU activation.
    
    Args:
        input: Input tensor
        bias: Bias tensor
        
    Returns:
        Output tensor after bias addition and GELU
    """
    if EXTENSION_LOADED and input.is_cuda:
        from . import _C
        return _C.fused_bias_gelu(input, bias)
    else:
        # Fallback implementation
        return F.gelu(input + bias)

# Elementwise operations
def elementwise_add_relu(
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """Fused elementwise addition + ReLU."""
    if EXTENSION_LOADED and a.is_cuda:
        from . import _C
        return _C.elementwise_add_relu(a, b)
    else:
        return F.relu(a + b)

def elementwise_mul_tanh(
    a: torch.Tensor,
    b: torch.Tensor
) -> torch.Tensor:
    """Fused elementwise multiplication + Tanh."""
    if EXTENSION_LOADED and a.is_cuda:
        from . import _C
        return _C.elementwise_mul_tanh(a, b)
    else:
        return torch.tanh(a * b)

# Reduction operations
def reduce_sum_squared(
    input: torch.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False
) -> torch.Tensor:
    """Fused square + sum reduction."""
    if EXTENSION_LOADED and input.is_cuda:
        from . import _C
        return _C.reduce_sum_squared(input, dim, keepdim)
    else:
        return torch.sum(input * input, dim=dim, keepdim=keepdim)

def reduce_mean_abs(
    input: torch.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False
) -> torch.Tensor:
    """Fused absolute value + mean reduction."""
    if EXTENSION_LOADED and input.is_cuda:
        from . import _C
        return _C.reduce_mean_abs(input, dim, keepdim)
    else:
        return torch.mean(torch.abs(input), dim=dim, keepdim=keepdim)
