# CUDA utilities and extension loading
import torch

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Try to load CUDA extension
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
