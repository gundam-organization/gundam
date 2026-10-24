#ifndef GUNDAM_CUDA_DEVICE_CONTEXT_H
#define GUNDAM_CUDA_DEVICE_CONTEXT_H

#include <memory>
#include <string>

namespace Backends {

  // Private runtime boundary, compiled by nvcc without GUNDAM/ROOT headers.
  class CudaDeviceContext {
  public:
    explicit CudaDeviceContext(int deviceIndex_);
    ~CudaDeviceContext();
    CudaDeviceContext(const CudaDeviceContext&) = delete;
    CudaDeviceContext& operator=(const CudaDeviceContext&) = delete;

    [[nodiscard]] const std::string& getDeviceName() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> _impl_{};
  };
}

#endif // GUNDAM_CUDA_DEVICE_CONTEXT_H
