#include "tensor_utils.h"
#include <torch/torch.h>

namespace tensor_utils {

bool check_contiguous(const torch::Tensor& tensor) {
    return tensor.is_contiguous();
}

bool check_same_device(const torch::Tensor& a, const torch::Tensor& b) {
    return a.device() == b.device();
}

bool check_same_dtype(const torch::Tensor& a, const torch::Tensor& b) {
    return a.dtype() == b.dtype();
}

torch::Tensor ensure_contiguous(const torch::Tensor& tensor) {
    if (tensor.is_contiguous()) {
        return tensor;
    }
    return tensor.contiguous();
}

std::vector<int64_t> get_broadcast_shape(const torch::Tensor& a, const torch::Tensor& b) {
    auto a_sizes = a.sizes();
    auto b_sizes = b.sizes();
    
    size_t max_dims = std::max(a_sizes.size(), b_sizes.size());
    std::vector<int64_t> result(max_dims);
    
    for (size_t i = 0; i < max_dims; ++i) {
        int64_t a_dim = (i < a_sizes.size()) ? a_sizes[a_sizes.size() - 1 - i] : 1;
        int64_t b_dim = (i < b_sizes.size()) ? b_sizes[b_sizes.size() - 1 - i] : 1;
        
        if (a_dim == 1) {
            result[max_dims - 1 - i] = b_dim;
        } else if (b_dim == 1) {
            result[max_dims - 1 - i] = a_dim;
        } else if (a_dim == b_dim) {
            result[max_dims - 1 - i] = a_dim;
        } else {
            throw std::runtime_error("Cannot broadcast tensors with incompatible shapes");
        }
    }
    
    return result;
}

} // namespace tensor_utils
