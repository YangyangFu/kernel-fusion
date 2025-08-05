from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from torch.utils import cpp_extension
import torch
import os

# Check if CUDA is available
USE_CUDA = torch.cuda.is_available()

# Define include directories
include_dirs = [
    "src/cpp",
    "src/cuda",
    pybind11.get_include(),
]

# Define library directories and libraries
library_dirs = []
libraries = []

# CUDA-specific settings
if USE_CUDA:
    include_dirs.extend([
        "/usr/local/cuda/include",
    ])
    include_dirs.extend(torch.utils.cpp_extension.include_paths())
    
    library_dirs.extend([
        "/usr/local/cuda/lib64",
    ])
    library_dirs.extend(torch.utils.cpp_extension.library_paths())
    
    libraries.extend(["cuda", "cudart", "cublas", "curand"])

# Define source files
cpp_sources = [
    "src/cpp/bindings.cpp",
    #"src/cpp/kernels/fusion_ops.cpp",
    "src/cpp/kernels/elementwise_ops.cpp", 
    #"src/cpp/kernels/reduction_ops.cpp",
    #"src/cpp/utils/tensor_utils.cpp",
]

cuda_sources = []
if USE_CUDA:
    cuda_sources = [
        "src/cuda/kernels/elementwise.cu",
        #"src/cuda/kernels/reduction.cu",
        #"src/cuda/kernels/fusion.cu",
        #"src/cuda/utils/cuda_utils.cu",
    ]

# Compiler flags
extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": ["-O3", "--use_fast_math", "-std=c++17", 
             "-gencode=arch=compute_70,code=sm_70",
             "-gencode=arch=compute_75,code=sm_75",
             "-gencode=arch=compute_80,code=sm_80",
             "-gencode=arch=compute_86,code=sm_86",
             "-gencode=arch=compute_120,code=sm_120"]
}

if USE_CUDA:
    extension = cpp_extension.CUDAExtension(
        name="kernel_fusion._C",
        sources=cpp_sources + cuda_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args
    )
else:
    extension = cpp_extension.CppExtension(
        name="kernel_fusion._C",
        sources=cpp_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args["cxx"]
    )

setup(
    name="kernel-fusion",
    version="0.1.0",
    author="Yangyang Fu",
    author_email="fuyy2008@gmail.com",
    description="Optimized fusion kernels for deep learning operations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=[extension],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
