#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor relu_custom_impl_npu(const at::Tensor& x) {
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnReluCustom, x, result);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_custom", &relu_custom_impl_npu, "relu operator");
}
