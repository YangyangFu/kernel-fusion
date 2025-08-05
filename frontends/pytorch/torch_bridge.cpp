#include "torch_bridge.hpp"
#include <stdexcept>

namespace kf {
namespace pytorch {

kf_dtype_t dtype_from_torch(torch::ScalarType scalar_type) {
    switch (scalar_type) {
        case torch::kFloat32: return KF_DTYPE_FLOAT32;
        case torch::kFloat16: return KF_DTYPE_FLOAT16;
        case torch::kBFloat16: return KF_DTYPE_BFLOAT16;
        case torch::kInt32: return KF_DTYPE_INT32;
        case torch::kInt64: return KF_DTYPE_INT64;
        case torch::kUInt8: return KF_DTYPE_UINT8;
        default:
            throw std::runtime_error("Unsupported PyTorch dtype");
    }
}

torch::ScalarType dtype_to_torch(kf_dtype_t dtype) {
    switch (dtype) {
        case KF_DTYPE_FLOAT32: return torch::kFloat32;
        case KF_DTYPE_FLOAT16: return torch::kFloat16;
        case KF_DTYPE_BFLOAT16: return torch::kBFloat16;
        case KF_DTYPE_INT32: return torch::kInt32;
        case KF_DTYPE_INT64: return torch::kInt64;
        case KF_DTYPE_UINT8: return torch::kUInt8;
        default:
            throw std::runtime_error("Unsupported KernelFusion dtype");
    }
}

kf_tensor_desc_t tensor_desc_from_torch(const torch::Tensor& tensor) {
    kf_tensor_desc_t desc;
    
    // Allocate and copy shape
    desc.ndim = tensor.dim();
    desc.shape = (int64_t*)malloc(desc.ndim * sizeof(int64_t));
    desc.strides = (int64_t*)malloc(desc.ndim * sizeof(int64_t));
    
    for (int i = 0; i < desc.ndim; ++i) {
        desc.shape[i] = tensor.size(i);
        desc.strides[i] = tensor.stride(i);
    }
    
    desc.dtype = dtype_from_torch(tensor.scalar_type());
    desc.device_type = tensor.is_cuda() ? KF_DEVICE_CUDA : KF_DEVICE_CPU;
    desc.device_id = tensor.is_cuda() ? tensor.device().index() : -1;
    desc.layout = tensor.is_contiguous() ? KF_LAYOUT_CONTIGUOUS : KF_LAYOUT_STRIDED;
    
    return desc;
}

kf_memory_desc_t memory_desc_from_torch(const torch::Tensor& tensor) {
    kf_memory_desc_t memory;
    
    memory.data = tensor.data_ptr();
    memory.size_bytes = tensor.numel() * tensor.element_size();
    memory.device_type = tensor.is_cuda() ? KF_DEVICE_CUDA : KF_DEVICE_CPU;
    memory.device_id = tensor.is_cuda() ? tensor.device().index() : -1;
    memory.owns_memory = false; // PyTorch owns the memory
    
    return memory;
}

kf_tensor_t* tensor_from_torch(kf_context_t* context, const torch::Tensor& tensor) {
    auto desc = tensor_desc_from_torch(tensor);
    auto memory = memory_desc_from_torch(tensor);
    
    return kf_tensor_create_from_memory(&memory, &desc);
}

torch::Tensor tensor_to_torch(const kf_tensor_t* tensor) {
    kf_tensor_desc_t desc;
    kf_memory_desc_t memory;
    
    KF_TORCH_CHECK(kf_tensor_get_desc(tensor, &desc));
    KF_TORCH_CHECK(kf_tensor_get_memory(tensor, &memory));
    
    // Convert shape to torch format
    std::vector<int64_t> sizes(desc.shape, desc.shape + desc.ndim);
    std::vector<int64_t> strides(desc.strides, desc.strides + desc.ndim);
    
    torch::TensorOptions options = torch::TensorOptions()
        .dtype(dtype_to_torch(desc.dtype));
    
    if (desc.device_type == KF_DEVICE_CUDA) {
        options = options.device(torch::Device(torch::kCUDA, desc.device_id));
    } else {
        options = options.device(torch::kCPU);
    }
    
    return torch::from_blob(memory.data, sizes, strides, options);
}

} // namespace pytorch
} // namespace kf

// ============================================================================
// PyTorch Function Wrappers
// ============================================================================

torch::Tensor fused_elementwise_add_relu(const torch::Tensor& a, const torch::Tensor& b) {
    // Validate inputs
    TORCH_CHECK(a.device() == b.device(), "Tensors must be on the same device");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have the same shape");
    
    auto output = torch::empty_like(a);
    
    // Create context
    int device_id = a.is_cuda() ? a.device().index() : -1;
    kf::pytorch::Context context(device_id);
    
    // Convert tensors
    auto kf_a = kf::pytorch::tensor_from_torch(context, a);
    auto kf_b = kf::pytorch::tensor_from_torch(context, b);
    auto kf_output = kf::pytorch::tensor_from_torch(context, output);
    
    // Call kernel
    KF_TORCH_CHECK(kf_fused_elementwise_add_activation(
        context, kf_a, kf_b, kf_output, KF_ACTIVATION_RELU, nullptr
    ));
    
    // Cleanup
    kf_tensor_destroy(kf_a);
    kf_tensor_destroy(kf_b);
    kf_tensor_destroy(kf_output);
    
    return output;
}

torch::Tensor fused_layer_norm_gelu(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps = 1e-5
) {
    auto output = torch::empty_like(input);
    
    // Create context
    int device_id = input.is_cuda() ? input.device().index() : -1;
    kf::pytorch::Context context(device_id);
    
    // Convert tensors
    auto kf_input = kf::pytorch::tensor_from_torch(context, input);
    auto kf_weight = kf::pytorch::tensor_from_torch(context, weight);
    auto kf_bias = kf::pytorch::tensor_from_torch(context, bias);
    auto kf_output = kf::pytorch::tensor_from_torch(context, output);
    
    // Call kernel
    KF_TORCH_CHECK(kf_fused_layer_norm_activation(
        context, kf_input, kf_weight, kf_bias, kf_output, 
        KF_ACTIVATION_GELU, eps, nullptr
    ));
    
    // Cleanup
    kf_tensor_destroy(kf_input);
    kf_tensor_destroy(kf_weight);
    kf_tensor_destroy(kf_bias);
    kf_tensor_destroy(kf_output);
    
    return output;
}

// ============================================================================
// PyTorch Binding
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "KernelFusion PyTorch Frontend";
    
    // Core functions
    m.def("fused_elementwise_add_relu", &fused_elementwise_add_relu, 
          "Fused elementwise add + ReLU");
    m.def("fused_layer_norm_gelu", &fused_layer_norm_gelu,
          "Fused layer normalization + GELU",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
    
    // Context management
    py::class_<kf::pytorch::Context>(m, "Context")
        .def(py::init<int>(), py::arg("device_id") = -1)
        .def("get_device_info", &kf::pytorch::Context::get_device_info);
    
    // Stream management
    py::class_<kf::pytorch::Stream>(m, "Stream")
        .def(py::init<kf::pytorch::Context&, int, unsigned int>(),
             py::arg("context"), py::arg("priority") = 0, py::arg("flags") = 0)
        .def("synchronize", &kf::pytorch::Stream::synchronize);
    
    // Device info
    py::class_<kf_device_info_t>(m, "DeviceInfo")
        .def_readonly("device_id", &kf_device_info_t::device_id)
        .def_readonly("total_memory", &kf_device_info_t::total_memory)
        .def_readonly("free_memory", &kf_device_info_t::free_memory);
}
