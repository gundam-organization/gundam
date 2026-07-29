#ifndef GUNDAM_MPS_BACKEND_H
#define GUNDAM_MPS_BACKEND_H

#include "BackendModel.h"
#include "BackendTypes.h"
#include "CpuBackend.h"
#include "IPropagationBackend.h"
#include "ParameterSnapshot.h"

namespace Backends {

  class MpsBackend : public IPropagationBackend {
  public:
    MpsBackend() = default;

    [[nodiscard]] std::string getName() const override { return "MPS"; }
    [[nodiscard]] BackendCapabilities getCapabilities() const override;
    [[nodiscard]] PropagationStatus getStatus(const PropagationToken& token_) const override;

    void build(const BackendModel& model_) override;
    void setLikelihoodModel(const BackendLikelihoodModel& likelihoodModel_) override;
    PropagationToken requestPropagation(
        const ParameterSnapshot& parameters_,
        const PropagationRequest& request_) override;

    bool isReady(const PropagationToken& token_) const override;
    void wait(const PropagationToken& token_) override;
    void materialize(const PropagationToken& token_, OutputRequest output_) override;
    [[nodiscard]] double getLikelihood(const PropagationToken& token_) const override;

  private:
    CpuBackend _hostBackend_{};
  };

}

#endif // GUNDAM_MPS_BACKEND_H
