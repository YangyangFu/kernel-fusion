cmake_minimum_required(VERSION 3.18)
project(kernel_fusion_cuda_tests LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../core/include)

# CUDA compiler flags for multiple architectures
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Support multiple GPU architectures
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")

# Compiler-specific flags
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
endif()

# Add executable for elementwise kernel tests
add_executable(test_elementwise_kernels 
    test_elementwise_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/kernels/elementwise_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/kernels/activation_utils.cu
)

# Add quick test executable
add_executable(quick_test 
    quick_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/kernels/elementwise_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/kernels/activation_utils.cu
)

# Add debug test for double precision
add_executable(debug_double_test 
    debug_double_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/kernels/elementwise_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/kernels/activation_utils.cu
)

# Set properties for CUDA
set_target_properties(test_elementwise_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(quick_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(debug_double_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA libraries
target_link_libraries(test_elementwise_kernels CUDA::cudart)
target_link_libraries(quick_test CUDA::cudart)
target_link_libraries(debug_double_test CUDA::cudart)

# Add test for normalization kernels (TODO)
# add_executable(test_normalization_kernels 
#     test_normalization_kernels.cu
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/kernels/normalization_kernels.cu
# )

# set_target_properties(test_normalization_kernels PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )

# target_link_libraries(test_normalization_kernels CUDA::cudart)

# Add test for linear kernels (TODO)
# add_executable(test_linear_kernels 
#     test_linear_kernels.cu
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/kernels/linear_kernels.cu
# )

# set_target_properties(test_linear_kernels PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )

# target_link_libraries(test_linear_kernels CUDA::cudart)

# Enable testing
enable_testing()

# Add tests
add_test(NAME ElementwiseKernels COMMAND test_elementwise_kernels)
add_test(NAME QuickTest COMMAND quick_test)
# add_test(NAME NormalizationKernels COMMAND test_normalization_kernels)
# add_test(NAME LinearKernels COMMAND test_linear_kernels)

# Custom target to run all tests
add_custom_target(run_cuda_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_elementwise_kernels quick_test
    COMMENT "Running all CUDA kernel tests"
)
