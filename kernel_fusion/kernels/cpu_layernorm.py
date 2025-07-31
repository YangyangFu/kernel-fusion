"""
CPU implementation of LayerNorm for reference and testing
"""

import torch
import math


def cpu_layernorm(x, weight=None, bias=None, eps=1e-5):
    """
    CPU reference implementation of LayerNorm
    
    Args:
        x: Input tensor [batch_size, seq_len, hidden_size]
        weight: Optional scale parameter [hidden_size]
        bias: Optional shift parameter [hidden_size]
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor with same shape as input
    """
    # Calculate mean and variance over the last dimension
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    
    # Normalize
    x_norm = (x - mean) / torch.sqrt(var + eps)
    
    # Apply scale and shift if provided
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    
    return x_norm


def cpu_layernorm_backward(grad_output, x, weight, mean, rstd):
    """
    CPU implementation of LayerNorm backward pass
    
    Args:
        grad_output: Gradient w.r.t. output [batch_size, seq_len, hidden_size]
        x: Original input [batch_size, seq_len, hidden_size]
        weight: Scale parameter [hidden_size]
        mean: Mean from forward pass [batch_size, seq_len, 1]
        rstd: Reciprocal standard deviation from forward pass [batch_size, seq_len, 1]
    
    Returns:
        Tuple of (grad_input, grad_weight, grad_bias)
    """
    # Get dimensions
    batch_size, seq_len, hidden_size = x.shape
    
    # Normalize input
    x_hat = (x - mean) * rstd
    
    # Gradients w.r.t. weight and bias
    grad_weight = (grad_output * x_hat).sum(dim=(0, 1))
    grad_bias = grad_output.sum(dim=(0, 1))
    
    # Gradient w.r.t. input
    grad_x_hat = grad_output * weight
    grad_var = (grad_x_hat * (x - mean)).sum(dim=-1, keepdim=True) * (-0.5) * (rstd ** 3)
    grad_mean = -grad_x_hat.sum(dim=-1, keepdim=True) * rstd - grad_var * 2 * (x - mean).mean(dim=-1, keepdim=True)
    
    grad_input = grad_x_hat * rstd + grad_var * 2 * (x - mean) / hidden_size + grad_mean / hidden_size
    
    return grad_input, grad_weight, grad_bias
