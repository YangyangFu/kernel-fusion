#pragma once

#include <torch/extension.h>
#include <vector>

namespace tensor_utils {

// Utility functions for tensor operations
bool check_contiguous(const torch::Tensor& tensor);
bool check_same_device(const torch::Tensor& a, const torch::Tensor& b);
bool check_same_dtype(const torch::Tensor& a, const torch::Tensor& b);

torch::Tensor ensure_contiguous(const torch::Tensor& tensor);
std::vector<int64_t> get_broadcast_shape(const torch::Tensor& a, const torch::Tensor& b);

} // namespace tensor_utils
