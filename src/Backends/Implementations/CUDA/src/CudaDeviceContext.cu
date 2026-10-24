#include "CudaDeviceContext.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace {
  void checkCuda(cudaError_t status_, const char* operation_) {
    if( status_ != cudaSuccess ){
      throw std::runtime_error(std::string("CudaBackend: ") + operation_ + ": " + cudaGetErrorString(status_));
    }
  }

  // CUDA's current device is thread-local. Leave the caller's selection intact.
  class DeviceScope {
  public:
    explicit DeviceScope(int deviceIndex_) {
      checkCuda(cudaGetDevice(&_previousDevice_), "cudaGetDevice");
      checkCuda(cudaSetDevice(deviceIndex_), "cudaSetDevice");
    }
    ~DeviceScope() { cudaSetDevice(_previousDevice_); }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
  private:
    int _previousDevice_{0};
  };
}

struct Backends::CudaDeviceContext::Impl {
  int deviceIndex{0};
  std::string deviceName{};
  cudaStream_t stream{nullptr};

  ~Impl() {
    if( stream == nullptr ){ return; }
    int previousDevice = 0;
    if( cudaGetDevice(&previousDevice) != cudaSuccess ){ return; }
    if( cudaSetDevice(deviceIndex) != cudaSuccess ){ return; }
    cudaStreamDestroy(stream);
    cudaSetDevice(previousDevice);
  }
};

Backends::CudaDeviceContext::CudaDeviceContext(int deviceIndex_) : _impl_(std::make_unique<Impl>()) {
  int deviceCount = 0;
  checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount (run inside a GPU allocation)");
  if( deviceCount == 0 ){
    throw std::runtime_error("CudaBackend: no CUDA device visible. Run inside a GPU allocation.");
  }
  if( deviceIndex_ < 0 or deviceIndex_ >= deviceCount ){
    throw std::runtime_error("CudaBackend: deviceIndex " + std::to_string(deviceIndex_)
                             + " is outside the visible device range [0, " + std::to_string(deviceCount) + ").");
  }
  _impl_->deviceIndex = deviceIndex_;
  DeviceScope scope(deviceIndex_);
  cudaDeviceProp properties{};
  checkCuda(cudaGetDeviceProperties(&properties, deviceIndex_), "cudaGetDeviceProperties");
  _impl_->deviceName = properties.name;
  checkCuda(cudaStreamCreateWithFlags(&_impl_->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
}

Backends::CudaDeviceContext::~CudaDeviceContext() = default;

const std::string& Backends::CudaDeviceContext::getDeviceName() const { return _impl_->deviceName; }
