#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <type_traits>

namespace cuda_type_utils {

// Enhanced template utilities for type dispatch
template<typename T>
struct CudaTypeTraits {
    static_assert(std::is_same_v<T, void>, "Unsupported type for CUDA operations");
};

template<>
struct CudaTypeTraits<float> {
    using cuda_type = float;
    using pytorch_type = float;
    static constexpr int size = sizeof(float);
    static constexpr const char* name = "float";
};

template<>
struct CudaTypeTraits<double> {
    using cuda_type = double;
    using pytorch_type = double;
    static constexpr int size = sizeof(double);
    static constexpr const char* name = "double";
};

template<>
struct CudaTypeTraits<c10::Half> {
    using cuda_type = __half;
    using pytorch_type = c10::Half;
    static constexpr int size = sizeof(__half);
    static constexpr const char* name = "half";
};

// Add support for bfloat16 (future-proofing)
template<>
struct CudaTypeTraits<at::BFloat16> {
    using cuda_type = __nv_bfloat16;
    using pytorch_type = at::BFloat16;
    static constexpr int size = sizeof(__nv_bfloat16);
    static constexpr const char* name = "bfloat16";
};

// Conversion utilities
template<typename PyTorchT>
__device__ __forceinline__ auto to_cuda_type(PyTorchT x) {
    using CudaT = typename CudaTypeTraits<PyTorchT>::cuda_type;
    if constexpr (std::is_same_v<PyTorchT, CudaT>) {
        return x;
    } else {
        return static_cast<CudaT>(x);
    }
}

template<typename CudaT, typename PyTorchT>
__device__ __forceinline__ PyTorchT from_cuda_type(CudaT x) {
    if constexpr (std::is_same_v<CudaT, PyTorchT>) {
        return x;
    } else {
        return static_cast<PyTorchT>(x);
    }
}

// Type checking utilities
template<typename T>
constexpr bool is_half_precision_v = std::is_same_v<typename CudaTypeTraits<T>::cuda_type, __half>;

template<typename T>
constexpr bool is_single_precision_v = std::is_same_v<typename CudaTypeTraits<T>::cuda_type, float>;

template<typename T>
constexpr bool is_double_precision_v = std::is_same_v<typename CudaTypeTraits<T>::cuda_type, double>;

template<typename T>
constexpr bool is_bfloat16_v = std::is_same_v<typename CudaTypeTraits<T>::cuda_type, __nv_bfloat16>;

// Helper aliases for cleaner code
template<typename T>
using cuda_type_t = typename CudaTypeTraits<T>::cuda_type;

template<typename T>
using pytorch_type_t = typename CudaTypeTraits<T>::pytorch_type;

} // namespace cuda_type_utils
