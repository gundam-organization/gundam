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
    std::size_t eventDialStride{1};
    // CSR inverse index: constant Shift dials are initialized by the dense pass.
    std::vector<std::size_t> parameterDialOffsets{};
    std::vector<std::uint32_t> parameterDialIndices{};
    std::vector<std::size_t> binOffsets{};
    std::vector<std::size_t> binEvents{};
    std::size_t parameterCount{0};
    bool hasReweightCap{false};
    double maxReweight{0};
  };

  // Reject unsupported features and malformed layouts before allocating CUDA
  // buffers. This also makes model validation testable without a toolkit.
  [[nodiscard]] CudaWeightModel makeCudaWeightModel(const EngineView& view_, bool interleaveEventDials_ = false);

  inline constexpr std::size_t cudaResponseBlockSize = 256;
  struct CudaResponseUpdate {
    std::size_t firstDial{0};
    std::size_t endDial{0};
    std::size_t firstBlock{0};
  };

  // Reused host-side dispatch storage. Blocks tile changed parameter ranges;
  // no list proportional to the number of changed dials is copied per call.
  struct CudaResponseUpdatePlan {
    std::vector<CudaResponseUpdate> ranges{};
    std::size_t dialCount{0};
    std::size_t blockCount{0};
    void select(const std::vector<std::size_t>& offsets_, const std::vector<double>& previous_,
                const std::vector<double>& current_);
  };
}
#endif
