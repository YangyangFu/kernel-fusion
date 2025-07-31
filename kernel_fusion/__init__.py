# Python package initialization
from .ops import *
from .kernels import *

__version__ = "0.1.0"
__author__ = "Your Name"

# Automatically check CUDA availability
import torch
CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from . import _C
        EXTENSION_LOADED = True
    except ImportError as e:
        EXTENSION_LOADED = False
        print(f"Warning: CUDA extension not loaded: {e}")
else:
    EXTENSION_LOADED = False
    print("Warning: CUDA not available, falling back to CPU implementations")
