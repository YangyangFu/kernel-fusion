cmake_minimum_required(VERSION 3.18)
project(KernelFusion LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find required packages
find_package(CUDA REQUIRED)

# Options
option(BUILD_PYTORCH_FRONTEND "Build PyTorch frontend" ON)
option(BUILD_TENSORFLOW_FRONTEND "Build TensorFlow frontend" OFF)
option(BUILD_JAX_FRONTEND "Build JAX frontend" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TESTS "Build tests" ON)

# Core library
add_library(kernel_fusion_core
    core/src/core.cpp
    core/src/kernels.cu
    core/src/kernels/kernels_common.cu
    core/src/kernels/elementwise_kernels.cu
    core/src/kernels/normalization_kernels.cu
    core/src/kernels/linear_kernels.cu
)

target_include_directories(kernel_fusion_core
    PUBLIC 
        core/include
    PRIVATE
        core/src
)

target_link_libraries(kernel_fusion_core
    ${CUDA_LIBRARIES}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_curand_LIBRARY}
)

# Set CUDA properties
set_target_properties(kernel_fusion_core PROPERTIES
    CUDA_ARCHITECTURES "75;80;86"  # Adjust based on your target GPUs
    POSITION_INDEPENDENT_CODE ON
)

# PyTorch Frontend
if(BUILD_PYTORCH_FRONTEND)
    find_package(Torch REQUIRED)
    find_package(pybind11 REQUIRED)
    
    pybind11_add_module(kernel_fusion_pytorch
        frontends/pytorch/torch_bridge.cpp
    )
    
    target_include_directories(kernel_fusion_pytorch
        PRIVATE
            core/include
            frontends/pytorch
    )
    
    target_link_libraries(kernel_fusion_pytorch
        PRIVATE
            kernel_fusion_core
            ${TORCH_LIBRARIES}
    )
    
    # Set PyTorch specific properties
    set_target_properties(kernel_fusion_pytorch PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
endif()

# TensorFlow Frontend (placeholder)
if(BUILD_TENSORFLOW_FRONTEND)
    # Would implement TensorFlow custom ops here
    message(STATUS "TensorFlow frontend not implemented yet")
endif()

# JAX Frontend (placeholder)
if(BUILD_JAX_FRONTEND)
    # Would implement JAX custom operations here
    message(STATUS "JAX frontend not implemented yet")
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples/standalone)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS kernel_fusion_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY core/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

if(BUILD_PYTORCH_FRONTEND)
    install(TARGETS kernel_fusion_pytorch
        LIBRARY DESTINATION lib/python
    )
endif()

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/KernelFusionConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/KernelFusionConfig.cmake"
    INSTALL_DESTINATION lib/cmake/KernelFusion
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/KernelFusionConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/KernelFusionConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/KernelFusionConfigVersion.cmake"
    DESTINATION lib/cmake/KernelFusion
)
