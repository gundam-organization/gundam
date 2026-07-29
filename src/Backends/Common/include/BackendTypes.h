#ifndef GUNDAM_BACKEND_TYPES_H
#define GUNDAM_BACKEND_TYPES_H

#include <cstdint>
#include <string>
#include <vector>

namespace Backends {

  struct PropagationRequest;

  enum class OutputRequest {
    EventWeights,
    Histograms,
    Likelihood,
    BinIndices,
    ObservableValues
  };

  [[nodiscard]] std::string toString(OutputRequest request_);
  [[nodiscard]] std::string toString(const PropagationRequest& request_);

  enum class OutputState {
    NotRequested,
    Scheduled,
    ReadyOnDevice,
    ReadyOnHost,
    Failed
  };

  enum class BackendStatus {
    Unconfigured,
    Ready,
    Running,
    Failed,
    Unavailable
  };

  struct BackendCapabilities {
    bool supportsCpu{false};
    bool supportsGpu{false};
    bool supportsEventWeights{false};
    bool supportsHistograms{false};
    bool supportsLikelihood{false};
    bool supportsDynamicBinning{false};
    bool supportsObservableTransforms{false};
    std::string deviceName{};
  };

  struct PropagationRequest {
    std::vector<OutputRequest> outputs{};
    bool allowAsync{true};

    [[nodiscard]] bool has(OutputRequest request_) const;
  };

  struct PropagationToken {
    std::uint64_t id{0};
    bool isValid{false};
  };

  struct PropagationStatus {
    BackendStatus backend{BackendStatus::Unconfigured};
    OutputState eventWeights{OutputState::NotRequested};
    OutputState histograms{OutputState::NotRequested};
    OutputState likelihood{OutputState::NotRequested};
    OutputState binIndices{OutputState::NotRequested};
    OutputState observableValues{OutputState::NotRequested};

    OutputState& state(OutputRequest request_);
    [[nodiscard]] OutputState state(OutputRequest request_) const;
  };

}

#endif // GUNDAM_BACKEND_TYPES_H
