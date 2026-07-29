#ifndef GUNDAM_CPU_BACKEND_H
#define GUNDAM_CPU_BACKEND_H

#include "BackendModel.h"
#include "BackendTypes.h"
#include "IPropagationBackend.h"
#include "ParameterSnapshot.h"

#include <cstdint>
#include <vector>

namespace Backends {

  class CpuBackend : public IPropagationBackend {
  public:
    CpuBackend() = default;

    [[nodiscard]] std::string getName() const override { return "CPU"; }
    [[nodiscard]] BackendCapabilities getCapabilities() const override;
    [[nodiscard]] PropagationStatus getStatus(const PropagationToken& token_) const override;

    void build(const BackendModel& model_) override;
    PropagationToken requestPropagation(
        const ParameterSnapshot& parameters_,
        const PropagationRequest& request_) override;

    bool isReady(const PropagationToken& token_) const override;
    void wait(const PropagationToken& token_) override;
    void materialize(const PropagationToken& token_, OutputRequest output_) override;

  private:
    struct Result {
      PropagationToken token{};
      PropagationStatus status{};
      std::vector<double> eventWeights{};
      std::vector<double> histSums{};
      std::vector<double> histSumSquares{};
    };

    [[nodiscard]] bool isCurrentToken(const PropagationToken& token_) const;
    void applyParameterSnapshot(const ParameterSnapshot& parameters_);
    void resetResult(const PropagationRequest& request_);
    void updateInputBuffers();
    void calculateEventWeights(Result& result_);
    void calculateHistograms(Result& result_);
    void calculateHistogramsFromEvents(Result& result_);
    void materializeEventWeights(Result& result_);
    void materializeHistograms(Result& result_);

    BackendModel _model_{};
    Result _lastResult_{};
    std::uint64_t _nextTokenId_{1};
    bool _isBuilt_{false};
  };

}

#endif // GUNDAM_CPU_BACKEND_H
