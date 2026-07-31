#ifndef GUNDAM_CPU_BACKEND_H
#define GUNDAM_CPU_BACKEND_H

#include "EngineView.h"
#include "BackendTypes.h"
#include "EngineBackend.h"
#include "ParameterSnapshot.h"

#include <cstdint>
#include <vector>

namespace Backends {

  class CpuEngineBackend : public EngineBackend {
  public:
    CpuEngineBackend() = default;

    [[nodiscard]] std::string getName() const override { return "CPU"; }
    [[nodiscard]] BackendCapabilities getCapabilities() const override;
    [[nodiscard]] PropagationStatus getStatus(const PropagationToken& token_) const override;
    [[nodiscard]] const EngineView& getEngineView() const override { return _engineView_; }

    void build(const EngineView& engineView_) override;
    PropagationToken requestPropagation(const ParameterSnapshot& parameters_) override;

    bool isReady(const PropagationToken& token_) const override;
    void wait(const PropagationToken& token_) override;
    void materialize(const PropagationToken& token_, OutputRequest output_) override;
    [[nodiscard]] double getLikelihood(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getEventWeightsHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getHistogramSumsHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getHistogramSumSquaresHostView(const PropagationToken& token_) const override;

  private:
    struct Result {
      PropagationToken token{};
      PropagationStatus status{};
      std::vector<double> eventWeights{};
      std::vector<double> histSums{};
      std::vector<double> histSumSquares{};
      double likelihood{0};
    };

    [[nodiscard]] bool isCurrentToken(const PropagationToken& token_) const;
    void resetResult();
    void calculateEventWeights(Result& result_, const ParameterSnapshot& parameters_);
    void calculateHistograms(Result& result_);
    void calculateHistogramsFromEvents(Result& result_, const ParameterSnapshot& parameters_);
    void calculateLikelihood(Result& result_);

    EngineView _engineView_{};
    Result _lastResult_{};
    std::uint64_t _nextTokenId_{1};
    bool _isBuilt_{false};
  };

}

#endif // GUNDAM_CPU_BACKEND_H
