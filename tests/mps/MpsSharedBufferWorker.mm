// Optional Torch adapter built by the selected worker Python, never by GUNDAM.
#include "MpsSharedBuffer.h"

#include <ATen/mps/MPSStream.h>
#include <ATen/ops/from_blob.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

namespace {
  std::vector<at::Tensor> attach(const std::string& path_, std::uint32_t count_) {
    auto mapping = std::make_shared<MpsSharedBuffer>(
        at::mps::getCurrentMPSStream()->device(), path_, count_, false);
    auto view = [&](std::int64_t size_, std::size_t offset_){
      return at::for_blob((void*)mapping->buffer(), {size_})
          .storage_offset(offset_ / sizeof(float))
          .options(at::TensorOptions().device(at::Device(at::kMPS, 0)).dtype(at::kFloat))
          .target_device(at::Device(at::kMPS, 0))
          .deleter([mapping](void*) {})
          .make_tensor();
    };
    return {view(2, 0), view(count_, mapping->inputOffset()), view(count_, mapping->weightOffset())};
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module_) {
  module_.def("attach", &attach);
}
