#ifndef GUNDAM_I_PROPAGATION_BACKEND_H
#define GUNDAM_I_PROPAGATION_BACKEND_H

#include "BackendModel.h"
#include "BackendLikelihoodModel.h"
#include "BackendTypes.h"
#include "ParameterSnapshot.h"

#include <string>

namespace Backends {

  class IPropagationBackend {
  public:
    virtual ~IPropagationBackend() = default;

    [[nodiscard]] virtual std::string getName() const = 0;
    [[nodiscard]] virtual BackendCapabilities getCapabilities() const = 0;
    [[nodiscard]] virtual PropagationStatus getStatus(const PropagationToken& token_) const = 0;
    [[nodiscard]] virtual const BackendModel& getModel() const = 0;

    virtual void build(const BackendModel& model_) = 0;
    virtual void setLikelihoodModel(const BackendLikelihoodModel& likelihoodModel_) = 0;
    virtual PropagationToken requestPropagation(const ParameterSnapshot& parameters_) = 0;

    virtual bool isReady(const PropagationToken& token_) const = 0;
    virtual void wait(const PropagationToken& token_) = 0;
    virtual void materialize(const PropagationToken& token_, OutputRequest output_) = 0;
    [[nodiscard]] virtual double getLikelihood(const PropagationToken& token_) const = 0;
    [[nodiscard]] virtual const std::vector<double>& getEventWeightsHostView(const PropagationToken& token_) const = 0;
    [[nodiscard]] virtual const std::vector<double>& getHistogramSumsHostView(const PropagationToken& token_) const = 0;
    [[nodiscard]] virtual const std::vector<double>& getHistogramSumSquaresHostView(const PropagationToken& token_) const = 0;
    [[nodiscard]] virtual BackendDeviceView getDeviceView(const PropagationToken&) const { return {}; }
    [[nodiscard]] virtual BackendTimingSummary getLastTimingSummary() const { return {}; }
  };

}

#endif // GUNDAM_I_PROPAGATION_BACKEND_H
