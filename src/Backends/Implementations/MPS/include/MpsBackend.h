#ifndef GUNDAM_MPS_BACKEND_H
#define GUNDAM_MPS_BACKEND_H

#include "BackendLikelihoodModel.h"
#include "BackendModel.h"
#include "BackendTypes.h"
#include "IPropagationBackend.h"
#include "ParameterSnapshot.h"

#include <memory>
#include <cstdint>

namespace Backends {

  class MpsBackend : public IPropagationBackend {
  public:
    MpsBackend();
    ~MpsBackend() override;

    [[nodiscard]] std::string getName() const override { return "MPS"; }
    [[nodiscard]] BackendCapabilities getCapabilities() const override;
    [[nodiscard]] PropagationStatus getStatus(const PropagationToken& token_) const override;
    [[nodiscard]] const BackendModel& getModel() const override;

    void build(const BackendModel& model_) override;
    void setLikelihoodModel(const BackendLikelihoodModel& likelihoodModel_) override;
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
    struct Impl;
    std::unique_ptr<Impl> _impl_{};
  };

}

#endif // GUNDAM_MPS_BACKEND_H
