"""
Example fused attention kernel using Triton
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_attention_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N_CTX, D_HEAD,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Fused attention kernel implementation
    This is a simplified example - real implementation would be more complex
    """
    # Get block indices
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # Compute offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Mask for valid positions
    mask_m = offs_m < N_CTX
    
    # Load Q block
    q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None])
    
    # Initialize output
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K, V blocks
        k_ptrs = K + off_hz * stride_kh + (start_n + offs_n)[None, :] * stride_kn + offs_d[:, None] * stride_kd
        v_ptrs = V + off_hz * stride_vh + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vd
        
        mask_n = (start_n + offs_n) < N_CTX
        k = tl.load(k_ptrs, mask=mask_n[None, :])
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        
        # Compute attention scores
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        
        # Apply softmax (simplified)
        p = tl.exp(qk)
        
        # Update accumulator
        acc += tl.dot(p.to(v.dtype), v)
    
    # Store output
    out_ptrs = Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


def fused_attention(q, k, v):
    """
    Fused attention implementation using Triton
    
    Args:
        q: Query tensor [batch, heads, seq_len, d_head]
        k: Key tensor [batch, heads, seq_len, d_head]
        v: Value tensor [batch, heads, seq_len, d_head]
    
    Returns:
        Output tensor [batch, heads, seq_len, d_head]
    """
    batch, n_heads, seq_len, d_head = q.shape
    
    # Allocate output
    output = torch.empty_like(q)
    
    # Grid and block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = triton.next_power_of_2(d_head)
    
    # Launch kernel
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
    
    fused_attention_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch, n_heads, seq_len, d_head,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return output
