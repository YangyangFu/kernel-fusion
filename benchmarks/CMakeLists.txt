cmake_minimum_required(VERSION 3.18)
project(kernel_fusion_benchmarks LANGUAGES CXX CUDA)

# Find required packages
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/common)

# CUDA compiler flags for optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Support multiple GPU architectures
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")

# Compiler-specific optimizations
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

# Common source files
set(COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/src/kernels/elementwise_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/src/kernels/activation_utils.cu
)

# Note: Some benchmark files need header compatibility fixes
# For now, building only working benchmarks

# Baseline comparison (requires PyTorch)
find_package(Torch QUIET)
if(Torch_FOUND)
    add_executable(baseline_comparison 
        src/performance/baseline_comparison.cu
        ${COMMON_SOURCES}
    )
    
    set_target_properties(baseline_comparison PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    target_link_libraries(baseline_comparison 
        CUDA::cudart
        CUDA::cublas
        ${TORCH_LIBRARIES}
    )
    
    # Note: fusion_validation.cu not yet created
    # Can be added later when comprehensive validation is needed
    
    message(STATUS "Building with PyTorch baseline support")
else()
    message(WARNING "PyTorch not found - baseline comparison will not be built")
    message(STATUS "To enable baseline comparison, install LibTorch")
endif()

# Simple baseline comparison (works without PyTorch)
# Note: Temporarily disabled due to header compatibility issues
# add_executable(simple_baseline_comparison 
#     src/validation/simple_baseline_comparison.cu
#     ${COMMON_SOURCES}
# )

# Simple fusion validation (works without PyTorch)
add_executable(simple_fusion_validation 
    src/validation/simple_fusion_validation.cu
    ${COMMON_SOURCES}
)

set_target_properties(simple_fusion_validation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(simple_fusion_validation 
    CUDA::cudart
)

# Profile-enabled versions for detailed analysis
# Note: Temporarily disabled due to header compatibility issues
# add_executable(elementwise_benchmark_profile 
#     src/performance/elementwise_benchmark.cu
#     ${COMMON_SOURCES}
# )

# Custom targets for running benchmarks
add_custom_target(run_simple_fusion_validation
    COMMAND simple_fusion_validation
    DEPENDS simple_fusion_validation
    COMMENT "Running simple fusion validation"
)

# Install targets
install(TARGETS simple_fusion_validation
    RUNTIME DESTINATION bin
)

# Optional: Add profiling support
option(ENABLE_NSIGHT "Enable Nsight profiling annotations" OFF)
if(ENABLE_NSIGHT)
    target_link_libraries(simple_fusion_validation CUDA::nvToolsExt)
    target_compile_definitions(simple_fusion_validation PRIVATE ENABLE_NSIGHT=1)
endif()

# Add option for address sanitizer (useful for debugging)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
    target_compile_options(simple_fusion_validation PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
    target_link_options(simple_fusion_validation PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
endif()
