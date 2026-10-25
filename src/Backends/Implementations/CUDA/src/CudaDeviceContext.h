#ifndef GUNDAM_CUDA_DEVICE_CONTEXT_H
#define GUNDAM_CUDA_DEVICE_CONTEXT_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace Backends {
  struct CudaWeightModel;
  struct PropagationView;
  struct PropagationInputs;
  struct EventState;
  struct BackendTimingSummary;
  struct BackendResourceUsage;
  struct BackendDeviceView;

  // Private runtime boundary, compiled by nvcc without ROOT headers.
  class CudaDeviceContext {
  public:
    explicit CudaDeviceContext(int deviceIndex_);
    ~CudaDeviceContext();
    CudaDeviceContext(const CudaDeviceContext&) = delete;
    CudaDeviceContext& operator=(const CudaDeviceContext&) = delete;

    [[nodiscard]] const std::string& getDeviceName() const;
    void build(const CudaWeightModel& model_, const PropagationView& original_);
    void propagate(const PropagationInputs& inputs_, bool recomputeResponses_, BackendTimingSummary& timing_);
    void readWeights(std::vector<double>& weights_);
    void readHistograms(std::vector<double>& sums_, std::vector<double>& sumSquares_,
                        std::vector<double>& outsideSums_, std::vector<double>& outsideSquares_);
    std::size_t readEventState(EventState& state_);
    void invalidate();
    [[nodiscard]] BackendDeviceView getDeviceView() const;
    [[nodiscard]] BackendResourceUsage getResourceUsage() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> _impl_{};
  };
}

#endif // GUNDAM_CUDA_DEVICE_CONTEXT_H
