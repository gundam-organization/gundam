#ifndef GUNDAM_MPS_BACKEND_H
#define GUNDAM_MPS_BACKEND_H

#include "EngineView.h"
#include "BackendTypes.h"
#include "EngineBackend.h"
#include "ParameterSnapshot.h"

#include <memory>
#include <cstdint>

namespace Backends {

  struct MpsEngineBackendImpl;

  class MpsEngineBackend : public EngineBackend {
  public:
    MpsEngineBackend();
    ~MpsEngineBackend() override;

    [[nodiscard]] std::string getName() const override { return "MPS"; }
    [[nodiscard]] BackendCapabilities getCapabilities() const override;
    [[nodiscard]] PropagationStatus getStatus(const PropagationToken& token_) const override;
    [[nodiscard]] const EngineView& getEngineView() const override;

    void build(const EngineView& engineView_) override;
    PropagationToken requestPropagation(const ParameterSnapshot& parameters_) override;

    bool isReady(const PropagationToken& token_) const override;
    void wait(const PropagationToken& token_) override;
    void materialize(const PropagationToken& token_, OutputRequest output_) override;
    [[nodiscard]] double getLikelihood(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getEventWeightsHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getHistogramSumsHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getHistogramSumSquaresHostView(const PropagationToken& token_) const override;
    [[nodiscard]] BackendDeviceView getDeviceView(const PropagationToken& token_) const override;
    [[nodiscard]] BackendTimingSummary getLastTimingSummary() const override;

  private:
    std::unique_ptr<MpsEngineBackendImpl> _impl_{};
  };

}

#endif // GUNDAM_MPS_BACKEND_H
