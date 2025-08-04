# Python package initialization

__version__ = "0.1.0"
__author__ = "Yangyang Fu"

# Import CUDA utilities first (to avoid circular imports)
from .cuda_utils import CUDA_AVAILABLE, EXTENSION_LOADED

# Import operations and kernels
from .ops import *
from .kernels import *

# Import automatic conversion utilities
from .auto_convert import auto_convert_model, ModelConverter

# Import stream-aware conversion utilities
from .stream_convert import auto_convert_with_streams, convert_model_with_streams
