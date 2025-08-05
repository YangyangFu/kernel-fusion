from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

ext_modules = [
    Pybind11Extension(
        "kernel_fusion_pytorch",
        ["torch_bridge.cpp"],
        include_dirs=[
            "../../core/include",
        ],
        libraries=["kernel_fusion_core"],
        library_dirs=["../../build/core"],
        cxx_std=17,
    ),
]

setup(
    name="kernel_fusion_pytorch",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)