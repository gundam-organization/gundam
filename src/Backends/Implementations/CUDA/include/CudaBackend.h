#ifndef GUNDAM_CUDA_BACKEND_H
#define GUNDAM_CUDA_BACKEND_H

#include "Backend.h"

#include <memory>

namespace Backends {

#define ENUM_NAME CudaPropagationMode
#define ENUM_FIELDS \
ENUM_FIELD( cachedResponses, 0 ) \
ENUM_FIELD( recomputeResponses )
#include "GenericToolbox.MakeEnum.h"

#define ENUM_NAME CudaWeightLayout
#define ENUM_FIELDS \
ENUM_FIELD( eventMajor, 0 ) \
ENUM_FIELD( warpInterleaved )
#include "GenericToolbox.MakeEnum.h"

  struct CudaBackendImpl;

  // Resident native-double executor for weight dials and fixed bins.
  // The public interface deliberately exposes no CUDA headers or runtime types.
  class CudaBackend : public Backend {
  public:
    CudaBackend();
    ~CudaBackend() override;

    [[nodiscard]] static bool isCompiled();
    [[nodiscard]] std::string getName() const override;
    [[nodiscard]] BackendCapabilities getCapabilities() const override;
    [[nodiscard]] PropagationStatus getStatus(const PropagationToken& token_) const override;
    [[nodiscard]] const EngineView& getEngineView() const override;

    void build(const EngineView& engineView_) override;
    void buildShared(std::shared_ptr<const EngineView> engineView_) override;
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
    [[nodiscard]] BackendTimingSummary getLastTimingSummary() const override;
    [[nodiscard]] BackendTimingTotals getTimingTotals() const override;
    [[nodiscard]] std::string getDiagnostics() const;

  protected:
    void configureImpl() override;
    void initializeImpl() override;

  private:
    void initializeDevice();
    [[nodiscard]] bool isCurrentToken(const PropagationToken& token_) const;
    void requireToken(const PropagationToken& token_) const;
    void readWeights() const;
    void readHistograms() const;
    void invalidateResult() const;

    int _deviceIndex_{0};
    CudaPropagationMode _propagationMode_{CudaPropagationMode::cachedResponses};
    CudaWeightLayout _weightLayout_{CudaWeightLayout::warpInterleaved};
    std::uint64_t _nextTokenId_{1};
    std::unique_ptr<CudaBackendImpl> _impl_{};
  };
}

#endif // GUNDAM_CUDA_BACKEND_H
