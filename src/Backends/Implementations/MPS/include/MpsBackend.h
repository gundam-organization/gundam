#ifndef GUNDAM_MPS_BACKEND_H
#define GUNDAM_MPS_BACKEND_H

#include "EngineView.h"
#include "BackendTypes.h"
#include "Backend.h"
#include "PropagationInputs.h"

#include <memory>
#include <cstdint>

namespace Backends {

#define ENUM_NAME MpsPropagationMode
#define ENUM_FIELDS \
ENUM_FIELD( cachedResponses, 0 ) \
ENUM_FIELD( recomputeResponses )
#include "GenericToolbox.MakeEnum.h"

  struct MpsBackendImpl;

  class MpsBackend : public Backend {
  protected:
    void configureImpl() override;

  public:
    MpsBackend();
    ~MpsBackend() override;

    [[nodiscard]] std::string getName() const override;
    [[nodiscard]] BackendCapabilities getCapabilities() const override;
    [[nodiscard]] PropagationStatus getStatus(const PropagationToken& token_) const override;
    [[nodiscard]] const EngineView& getEngineView() const override;

    // Separate diagnostic passes split GPU stages into synchronized commands.
    void setDiagnosticStageProfiling(bool enabled_);
    [[nodiscard]] std::string getDiagnostics() const;
    [[nodiscard]] std::string getDiagnosticEventPropagationState() const;
    [[nodiscard]] std::string getDiagnosticStageGpuSeconds() const;

    void validateLikelihoodGradient() const override;
    [[nodiscard]] std::vector<double> getLikelihoodGradient(const PropagationToken& token_) override;
    void build(const EngineView& engineView_) override;
    PropagationToken requestPropagation(const PropagationInputs& inputs_) override;

    bool isReady(const PropagationToken& token_) const override;
    void wait(const PropagationToken& token_) override;
    void materialize(const PropagationToken& token_, OutputRequest output_) override;
    [[nodiscard]] double getLikelihood(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getEventWeightsHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getHistogramSumsHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getHistogramSumSquaresHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const EventState& getEventStateHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getOutsideSumsHostView(const PropagationToken& token_) const override;
    [[nodiscard]] const std::vector<double>& getOutsideSumSquaresHostView(const PropagationToken& token_) const override;
    [[nodiscard]] BackendDeviceView getDeviceView(const PropagationToken& token_) const override;
    [[nodiscard]] BackendResourceUsage getResourceUsage() const override;
    [[nodiscard]] BackendTimingSummary getLastTimingSummary() const override;

  private:
    std::unique_ptr<MpsBackendImpl> _impl_{};
  };

}

#endif // GUNDAM_MPS_BACKEND_H
