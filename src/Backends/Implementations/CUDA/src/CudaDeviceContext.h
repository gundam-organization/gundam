#ifndef GUNDAM_CUDA_DEVICE_CONTEXT_H
#define GUNDAM_CUDA_DEVICE_CONTEXT_H

#include <memory>
#include <string>
#include <vector>

namespace Backends {
  struct CudaWeightModel;
  struct ExternalWeightBlockView;
  struct BackendTimingSummary;
  struct BackendDeviceView;

  // Private runtime boundary, compiled by nvcc without ROOT headers.
  class CudaDeviceContext {
  public:
    explicit CudaDeviceContext(int deviceIndex_);
    ~CudaDeviceContext();
    CudaDeviceContext(const CudaDeviceContext&) = delete;
    CudaDeviceContext& operator=(const CudaDeviceContext&) = delete;

    [[nodiscard]] const std::string& getDeviceName() const;
    void build(const CudaWeightModel& model_);
    void propagate(const std::vector<double>& parameters_, const std::vector<ExternalWeightBlockView>& externalWeights_,
                   bool recomputeResponses_, BackendTimingSummary& timing_);
    void readWeights(std::vector<double>& weights_);
    void readHistograms(std::vector<double>& sums_, std::vector<double>& sumSquares_);
    void invalidate();
    [[nodiscard]] BackendDeviceView getDeviceView() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> _impl_{};
  };
}

#endif // GUNDAM_CUDA_DEVICE_CONTEXT_H
