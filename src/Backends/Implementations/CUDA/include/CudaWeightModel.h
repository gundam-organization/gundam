#ifndef GUNDAM_CUDA_WEIGHT_MODEL_H
#define GUNDAM_CUDA_WEIGHT_MODEL_H

#include "EngineView.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Backends {
  // Disjoint surface response slots sharing one unordered dependency pair.
  // Ordered x/y input descriptors remain attached to each individual dial.
  struct CudaSurfaceGroup {
    std::size_t firstParameter{0}, secondParameter{0};
    std::size_t firstDial{0}, endDial{0};
  };

  struct CudaExternalGroup {
    std::size_t blockIndex{0};
    std::size_t firstDial{0}, endDial{0};
  };

  struct CudaBinningSample {
    int binOffset{0}, binCount{0};
    std::size_t firstBin{0}, endBin{0}, variableCount{0};
    bool isDynamic{false}, readHostVariables{false}, isSmoothed{false};
  };

  // Native-double model. Events are in result-slot order; histogram lists retain
  // the original event traversal order, independently of those result slots.
  struct CudaWeightModel {
    std::vector<BackendEventWeightDescriptor> events{};
    std::vector<BackendDialDescriptor> dials{};
    std::vector<BackendDialInputDescriptor> inputs{};
    std::vector<double> payloads{};
    std::vector<std::uint32_t> eventDialIndices{};
    std::size_t eventDialStride{1};
    // Scalar CSR followed by disjoint surface/external groups in parameterDialIndices.
    // Constant Shift dials are initialized by the dense pass.
    std::vector<std::size_t> parameterDialOffsets{};
    std::vector<std::uint32_t> parameterDialIndices{};
    std::vector<CudaSurfaceGroup> surfaceGroups{};
    // External/table responses have disjoint dependencies on producer blocks.
    std::vector<CudaExternalGroup> externalGroups{};
    std::vector<ExternalWeightBlockDescriptor> externalWeightBlocks{};
    std::vector<BackendTableTerm> tableTerms{};
    std::size_t externalWeightCount{0};
    std::size_t externalDialCount{0};
    // Surface dials form a suffix, allowing a compact cache of second inputs.
    std::size_t surfaceOffset{0};
    std::vector<std::size_t> binOffsets{};
    std::vector<std::size_t> binEvents{};
    bool hasEventState{false};
    bool hasSmoothedBinning{false};
    std::vector<EventView> stateEvents{};
    std::vector<CudaBinningSample> samples{};
    std::vector<BinningBin> bins{};
    std::vector<BinningInterval> intervals{};
    std::vector<double> variableValues{};
    std::vector<BackendVariableResponseDescriptor> variableResponses{};
    std::vector<std::uint32_t> variableDialIndices{};
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

  // Reused host-side dispatch storage. Blocks tile changed dependency ranges;
  // no list proportional to the number of changed dials is copied per call.
  struct CudaResponseUpdatePlan {
    std::vector<CudaResponseUpdate> ranges{};
    std::size_t dialCount{0};
    std::size_t blockCount{0};
    void select(const std::vector<std::size_t>& offsets_, const std::vector<double>& previous_,
                const std::vector<double>& current_, const std::vector<CudaSurfaceGroup>& surfaceGroups_ = {});
    void selectExternal(const std::vector<CudaExternalGroup>& groups_,
                        const std::vector<std::uint64_t>& previous_, const std::vector<std::uint64_t>& current_,
                        bool force_ = false);
  };
}
#endif
