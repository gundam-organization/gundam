#ifndef GUNDAM_BACKEND_MODEL_H
#define GUNDAM_BACKEND_MODEL_H

#include "BackendEngineDescriptors.h"

#include <functional>
#include <vector>

namespace Backends {

  // Passive, flattened view of the fitter engine state for backend consumption.
  // It owns no engine logic and only exposes backend-friendly event, dial, sample,
  // and likelihood descriptors built upstream by BackendEngineLayout.
  struct BackendEventView {
    int sampleIndex{-1};
    int binIndex{-1};
    int globalBinIndex{-1};
    BackendEventWeightDescriptor weight{};
    std::size_t resultIndex{0};
  };

  struct BackendSampleView {
    int sampleIndex{-1};
    int binOffset{0};
    int binCount{0};
  };

  struct BackendLikelihoodSampleView {
    int binOffset{0};
    std::vector<double> dataSums{};
    std::vector<bool> ignoredBins{};
    std::function<double(double, double, double, int)> evalBin{};
  };

  struct BackendPropagationView {
    std::vector<BackendEventView> events{};
    std::vector<BackendDialDescriptor> eventDials{};
    std::vector<BackendDialInputDescriptor> dialInputs{};
    std::vector<double> dialPayloads{};
    std::vector<BackendSampleView> samples{};
    std::size_t parameterCount{0};
    int totalBins{0};

    void clear() {
      events.clear();
      eventDials.clear();
      dialInputs.clear();
      dialPayloads.clear();
      samples.clear();
      parameterCount = 0;
      totalBins = 0;
    }
    [[nodiscard]] bool empty() const { return events.empty(); }
  };

  struct BackendLikelihoodView {
    std::vector<BackendLikelihoodSampleView> samples{};

    [[nodiscard]] bool empty() const { return samples.empty(); }
  };

  struct BackendEngineView {
    BackendPropagationView propagation{};
    BackendLikelihoodView likelihood{};

    void clear() {
      propagation.clear();
      likelihood.samples.clear();
    }
    [[nodiscard]] bool empty() const { return propagation.empty() and likelihood.empty(); }
  };

}

#endif // GUNDAM_BACKEND_MODEL_H
