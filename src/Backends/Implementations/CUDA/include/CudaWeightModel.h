#ifndef GUNDAM_CUDA_WEIGHT_MODEL_H
#define GUNDAM_CUDA_WEIGHT_MODEL_H

#include "EngineView.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Backends {
  // Native-double model. Events are in result-slot order; histogram lists retain
  // the original event traversal order, independently of those result slots.
  struct CudaWeightModel {
    std::vector<BackendEventWeightDescriptor> events{};
    std::vector<BackendDialDescriptor> dials{};
    std::vector<BackendDialInputDescriptor> inputs{};
    std::vector<double> payloads{};
    std::vector<std::uint32_t> eventDialIndices{};
    std::vector<std::size_t> binOffsets{};
    std::vector<std::size_t> binEvents{};
    std::size_t parameterCount{0};
    bool hasReweightCap{false};
    double maxReweight{0};
  };

  // Reject unsupported features and malformed layouts before allocating CUDA
  // buffers. This also makes model validation testable without a toolkit.
  [[nodiscard]] CudaWeightModel makeCudaWeightModel(const EngineView& view_);
}
#endif
