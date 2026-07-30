#ifndef GUNDAM_BACKEND_MODEL_H
#define GUNDAM_BACKEND_MODEL_H

#include <cstddef>
#include <functional>
#include <vector>

class DialInterface;
class LikelihoodInterface;
class Parameter;

namespace Backends {

  struct BackendDialInputRef {
    std::size_t parameterIndex{std::size_t(-1)};
    bool useMirror{false};
    double mirrorMin{0};
    double mirrorRange{0};
  };

  struct BackendDialRef {
    const DialInterface* interface{nullptr};
    std::size_t firstInput{0};
    std::size_t inputCount{0};
  };

  struct BackendEventRef {
    int sampleIndex{-1};
    int binIndex{-1};
    int globalBinIndex{-1};
    double baseWeight{1};
    std::size_t firstDial{0};
    std::size_t dialCount{0};
    std::size_t resultIndex{0};
  };

  struct BackendSampleRef {
    int sampleIndex{-1};
    int binOffset{0};
    int binCount{0};
  };

  struct BackendLikelihoodSampleRef {
    int binOffset{0};
    std::vector<double> dataSums{};
    std::vector<bool> ignoredBins{};
    std::function<double(double, double, double, int)> evalBin{};
  };

  struct BackendPropagationView {
    std::vector<BackendEventRef> events{};
    std::vector<BackendDialRef> eventDials{};
    std::vector<BackendDialInputRef> dialInputs{};
    std::vector<BackendSampleRef> samples{};
    std::vector<const Parameter*> parameters{};
    int totalBins{0};

    void clear();
    [[nodiscard]] bool empty() const { return events.empty(); }
  };

  struct BackendLikelihoodView {
    std::vector<BackendLikelihoodSampleRef> samples{};

    [[nodiscard]] bool empty() const { return samples.empty(); }
  };

  struct BackendEngineView {
    BackendPropagationView propagation{};
    BackendLikelihoodView likelihood{};

    void clear();
    void build(LikelihoodInterface& likelihoodInterface_);
    [[nodiscard]] bool empty() const { return propagation.empty() and likelihood.empty(); }
  };

}

#endif // GUNDAM_BACKEND_MODEL_H
