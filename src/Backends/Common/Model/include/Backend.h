#ifndef GUNDAM_I_PROPAGATION_BACKEND_H
#define GUNDAM_I_PROPAGATION_BACKEND_H

#include "EngineView.h"
#include "BackendTypes.h"
#include "PropagationInputs.h"
#include "ConfigUtils.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Backends {

  class Backend : public JsonBaseClass {
  public:
    virtual ~Backend() = default;

    [[nodiscard]] virtual std::string getName() const = 0;
    [[nodiscard]] virtual BackendCapabilities getCapabilities() const = 0;
    [[nodiscard]] virtual PropagationStatus getStatus(const PropagationToken& token_) const = 0;
    [[nodiscard]] virtual const EngineView& getEngineView() const = 0;

    // Gradient is statistical only, in physical model parameter order.
    // Implementations must reject unsupported models and stale propagation tokens.
    virtual void validateLikelihoodGradient() const {
      throw std::runtime_error(getName() + " does not support analytic likelihood gradients.");
    }
    [[nodiscard]] virtual std::vector<double> getLikelihoodGradient(const PropagationToken&) {
      throw std::runtime_error(getName() + " does not support analytic likelihood gradients.");
    }

    virtual void build(const EngineView& engineView_) = 0;
    // Shared immutable model; implementations may retain it or use their own packing.
    virtual void buildShared(std::shared_ptr<const EngineView> engineView_) { build(*engineView_); }
    virtual PropagationToken requestPropagation(const PropagationInputs& inputs_) = 0;

    virtual bool isReady(const PropagationToken& token_) const = 0;
    virtual void wait(const PropagationToken& token_) = 0;
    virtual void materialize(const PropagationToken& token_, OutputRequest output_) = 0;
    [[nodiscard]] virtual double getLikelihood(const PropagationToken& token_) const = 0;
    [[nodiscard]] virtual const std::vector<double>& getEventWeightsHostView(const PropagationToken& token_) const = 0;
    [[nodiscard]] virtual const std::vector<double>& getHistogramSumsHostView(const PropagationToken& token_) const = 0;
    [[nodiscard]] virtual const std::vector<double>& getHistogramSumSquaresHostView(const PropagationToken& token_) const = 0;
    [[nodiscard]] virtual const EventState& getEventStateHostView(const PropagationToken&) const {
      throw std::runtime_error("Backend does not provide event state.");
    }
    [[nodiscard]] virtual const std::vector<double>& getOutsideSumsHostView(const PropagationToken&) const {
      static const std::vector<double> empty;
      return empty;
    }
    [[nodiscard]] virtual const std::vector<double>& getOutsideSumSquaresHostView(const PropagationToken&) const {
      static const std::vector<double> empty;
      return empty;
    }
    [[nodiscard]] virtual BackendDeviceView getDeviceView(const PropagationToken&) const { return {}; }
    [[nodiscard]] virtual BackendResourceUsage getResourceUsage() const { return {}; }
    [[nodiscard]] std::string getMonitorLine() const {
      return formatBackendMonitorLine(getName(), getResourceUsage(), getLastTimingSummary());
    }
    // Additional monitor columns as (title, formatted value). Empty hides the breakdown.
    [[nodiscard]] virtual std::vector<std::pair<std::string, std::string>> getMonitorColumns() const { return {}; }
    [[nodiscard]] virtual BackendTimingSummary getLastTimingSummary() const { return {}; }
    [[nodiscard]] virtual BackendTimingTotals getTimingTotals() const { return {}; }

  protected:
    void configureImpl() override { _config_.printUnusedKeys(); }
  };

}

#endif // GUNDAM_I_PROPAGATION_BACKEND_H
