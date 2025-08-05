cmake_minimum_required(VERSION 3.18)
project(kernel_fusion_benchmarks LANGUAGES CXX CUDA)

# Find required packages
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../core/include)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/kernels/elementwise_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core/src/kernels/activation_utils.cu
)

# Elementwise benchmark
add_executable(elementwise_benchmark 
    src/performance/elementwise_benchmark.cu
    ${COMMON_SOURCES}
)

set_target_properties(elementwise_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(elementwise_benchmark 
    CUDA::cudart
    CUDA::curand
)

# Memory bandwidth benchmark
add_executable(memory_benchmark 
    src/performance/memory_benchmark.cu
    ${COMMON_SOURCES}
)

set_target_properties(memory_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(memory_benchmark 
    CUDA::cudart
)

# Comparison benchmark
add_executable(comparison_benchmark 
    src/performance/comparison_benchmark.cu
    ${COMMON_SOURCES}
)

set_target_properties(comparison_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(comparison_benchmark 
    CUDA::cudart
    CUDA::cublas
)

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
    
    # Fusion validation
    add_executable(fusion_validation 
        src/validation/fusion_validation.cu
        ${COMMON_SOURCES}
    )
    
    set_target_properties(fusion_validation PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    target_link_libraries(fusion_validation 
        CUDA::cudart
        ${TORCH_LIBRARIES}
    )
    
    message(STATUS "Building with PyTorch baseline support")
else()
    message(WARNING "PyTorch not found - baseline comparison will not be built")
    message(STATUS "To enable baseline comparison, install LibTorch")
endif()

# Simple baseline comparison (works without PyTorch)
add_executable(simple_baseline_comparison 
    src/validation/simple_baseline_comparison.cu
    ${COMMON_SOURCES}
)

set_target_properties(simple_baseline_comparison PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(simple_baseline_comparison 
    CUDA::cudart
    CUDA::cublas
)

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
add_executable(elementwise_benchmark_profile 
    src/performance/elementwise_benchmark.cu
    ${COMMON_SOURCES}
)

set_target_properties(elementwise_benchmark_profile PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_definitions(elementwise_benchmark_profile PRIVATE ENABLE_PROFILING=1)
target_link_libraries(elementwise_benchmark_profile 
    CUDA::cudart
    CUDA::curand
    CUDA::cupti
)

# Custom targets for running benchmarks
add_custom_target(run_elementwise_benchmark
    COMMAND elementwise_benchmark
    DEPENDS elementwise_benchmark
    COMMENT "Running elementwise kernels benchmark"
)

add_custom_target(run_memory_benchmark
    COMMAND memory_benchmark
    DEPENDS memory_benchmark
    COMMENT "Running memory bandwidth benchmark"
)

add_custom_target(run_comparison_benchmark
    COMMAND comparison_benchmark
    DEPENDS comparison_benchmark
    COMMENT "Running kernel fusion comparison benchmark"
)

add_custom_target(run_all_benchmarks
    COMMAND elementwise_benchmark
    COMMAND memory_benchmark  
    COMMAND comparison_benchmark
    DEPENDS elementwise_benchmark memory_benchmark comparison_benchmark
    COMMENT "Running all benchmarks"
)

add_custom_target(run_quick_benchmarks
    COMMAND elementwise_benchmark --quick
    DEPENDS elementwise_benchmark
    COMMENT "Running quick benchmark suite"
)

# Install targets
install(TARGETS elementwise_benchmark memory_benchmark comparison_benchmark
    RUNTIME DESTINATION bin
)

# Optional: Add profiling support
option(ENABLE_NSIGHT "Enable Nsight profiling annotations" OFF)
if(ENABLE_NSIGHT)
    target_link_libraries(elementwise_benchmark CUDA::nvToolsExt)
    target_link_libraries(memory_benchmark CUDA::nvToolsExt)
    target_link_libraries(comparison_benchmark CUDA::nvToolsExt)
    target_compile_definitions(elementwise_benchmark PRIVATE ENABLE_NSIGHT=1)
    target_compile_definitions(memory_benchmark PRIVATE ENABLE_NSIGHT=1)
    target_compile_definitions(comparison_benchmark PRIVATE ENABLE_NSIGHT=1)
endif()

# Add option for address sanitizer (useful for debugging)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
    target_compile_options(elementwise_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
    target_link_options(elementwise_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>)
endif()
