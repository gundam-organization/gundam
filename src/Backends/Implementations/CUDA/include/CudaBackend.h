#ifndef GUNDAM_CUDA_BACKEND_H
#define GUNDAM_CUDA_BACKEND_H

#include "Backend.h"

#include <memory>

namespace Backends {

  struct CudaBackendImpl;

  // Device lifecycle only. Propagation and all result access remain unsupported.
  // The public interface deliberately exposes no CUDA headers or runtime types.
  class CudaBackend : public Backend {
  public:
    CudaBackend();
    ~CudaBackend() override;

    [[nodiscard]] static bool isCompiled();
    [[nodiscard]] std::string getName() const override { return "CUDA"; }
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

  protected:
    void configureImpl() override;
    void initializeImpl() override;

  private:
    void initializeDevice();
    [[noreturn]] static void throwNoResult();

    int _deviceIndex_{0};
    std::unique_ptr<CudaBackendImpl> _impl_{};
  };
}

#endif // GUNDAM_CUDA_BACKEND_H
