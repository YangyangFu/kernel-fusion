"""
Core kernel implementations

CUDA-based fusion kernels for deep learning operations.
"""

# Import CPU implementations (always available)
try:
    from .cpu_kernels import *
except ImportError:
    pass

# Import CUDA implementations (when available)
try:
    from .cuda_attention import *
    from .simple_cuda import *
    print("CUDA kernels available")
except ImportError:
    print("CUDA kernels not available - using CPU implementations")

# Import any other kernel modules
# from .layernorm import *
# from .activations import *
