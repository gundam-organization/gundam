#include "CudaDeviceContext.h"
#include "CudaWeightModel.h"
#include "BackendTypes.h"

#include <cuda_runtime.h>

#include "Semantics/BackendDialSemanticsCore.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
  constexpr unsigned int blockSize = 256;

  void checkCuda(cudaError_t status_, const char* operation_) {
    if( status_ != cudaSuccess ){
      throw std::runtime_error(std::string("CudaBackend: ") + operation_ + ": " + cudaGetErrorString(status_));
    }
  }

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

  // Buffers are destroyed while their owning context's DeviceScope is active.
  template<typename T> struct DeviceBuffer {
    T* data{nullptr};
    std::size_t size{0};
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if( data ){ cudaFree(data); } }
    void allocate(std::size_t size_) {
      if( size_ > std::numeric_limits<std::size_t>::max() / sizeof(T) ){
        throw std::runtime_error("CudaBackend: buffer size overflow.");
      }
      size = size_;
      if( size ){ checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)), "cudaMalloc"); }
    }
    void upload(const std::vector<T>& values_, cudaStream_t stream_) {
      allocate(values_.size());
      if( size ){ checkCuda(cudaMemcpyAsync(data, values_.data(), size * sizeof(T), cudaMemcpyHostToDevice, stream_), "model upload"); }
    }
    void read(std::vector<T>& values_, cudaStream_t stream_) const {
      values_.resize(size);
      if( size ){ checkCuda(cudaMemcpyAsync(values_.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost, stream_), "device readback"); }
    }
  };

  struct WeightBuffers {
    DeviceBuffer<Backends::BackendEventWeightDescriptor> events;
    DeviceBuffer<Backends::BackendDialDescriptor> dials;
    DeviceBuffer<Backends::BackendDialInputDescriptor> inputs;
    DeviceBuffer<double> payloads, parameters, responses, previousInputs, weights, sums, sumSquares;
    DeviceBuffer<std::uint32_t> dialIndices;
    DeviceBuffer<std::size_t> binOffsets, binEvents;
    DeviceBuffer<int> error;
    bool cachePrimed{false};
    bool hasReweightCap{false};
    double maxReweight{0};
  };

  unsigned int gridSize(std::size_t size_) {
    return static_cast<unsigned int>(std::min<std::size_t>((size_ - 1) / blockSize + 1, 65535));
  }

  __global__ void evaluateResponses(const Backends::BackendDialDescriptor* dials_, std::size_t count_,
                                    const Backends::BackendDialInputDescriptor* inputs_, const double* payloads_,
                                    const double* parameters_, double* responses_, double* previousInputs_,
                                    bool reuse_, int* error_) {
    using namespace Backends;
    for( std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count_; i += std::size_t(blockDim.x) * gridDim.x ){
      const auto& dial = dials_[i];
      double input = 0.;
      if( dial.type != BackendDialType::Shift ){
        const auto& descriptor = inputs_[dial.firstInput];
        input = Semantics::transformDialInput(descriptor, parameters_[descriptor.parameterIndex]);
      }
      if( not ::isfinite(input) ){ responses_[i] = 0.; atomicExch(error_, 1); continue; }
      if( reuse_ and previousInputs_[i] == input ){ continue; }
      const double* payload = dial.payloadSize == 0 ? nullptr : payloads_ + dial.payloadOffset;
      // These interpolation routines select a uniform segment through an int.
      if( dial.type == BackendDialType::CompactSpline or dial.type == BackendDialType::MonotonicSpline
          or dial.type == BackendDialType::UniformSpline ){
        const auto knots = dial.type == BackendDialType::UniformSpline ? (dial.payloadSize - 2) / 2 : dial.payloadSize - 2;
        const double x = dial.allowExtrapolation ? input
            : Semantics::clampValue(input, payload[0], payload[0] + payload[1] * double(knots - 1));
        const double position = (x - payload[0]) / payload[1];
        // UniformSpline also forms 2*ix+7 before clamping the segment index.
        const double limit = dial.type == BackendDialType::UniformSpline ? 1073741800. : 2147483000.;
        if( not ::isfinite(position) or ::fabs(position) > limit ){
          responses_[i] = 0.; atomicExch(error_, 1); continue;
        }
      }
      const double response = Semantics::evalDialResponseFromInput(dial, input, payload);
      if( not ::isfinite(response) ){ atomicExch(error_, 1); }
      responses_[i] = ::isfinite(response) ? response : 0.;
      previousInputs_[i] = input;
    }
  }

  __global__ void reweightEvents(const Backends::BackendEventWeightDescriptor* events_, std::size_t count_,
                                 const std::uint32_t* indices_, const double* responses_, double* weights_,
                                 bool capped_, double cap_, int* error_) {
    for( std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count_; i += std::size_t(blockDim.x) * gridDim.x ){
      const auto& event = events_[i];
      double response = 1.;
      for( std::size_t j = 0; j < event.dialCount; ++j ){ response *= responses_[indices_[event.firstDial + j]]; }
      if( capped_ and response > cap_ ){ response = cap_; }
      weights_[i] = event.baseWeight * response;
      if( not ::isfinite(weights_[i]) ){ atomicExch(error_, 1); }
    }
  }

  __global__ void reduceHistograms(const std::size_t* offsets_, const std::size_t* events_,
                                   std::size_t bins_, const double* weights_, double* sums_,
                                   double* squares_, int* error_) {
    __shared__ double sum[blockSize];
    __shared__ double square[blockSize];
    // No floating-point atomics: each bin has one deterministic reduction tree.
    for( std::size_t bin = blockIdx.x; bin < bins_; bin += gridDim.x ){
      double value = 0., valueSquared = 0.;
      for( std::size_t j = offsets_[bin] + threadIdx.x; j < offsets_[bin + 1]; j += blockDim.x ){
        const double weight = weights_[events_[j]];
        value += weight;
        valueSquared += weight * weight;
      }
      sum[threadIdx.x] = value;
      square[threadIdx.x] = valueSquared;
      __syncthreads();
      for( unsigned int stride = blockSize / 2; stride != 0; stride /= 2 ){
        if( threadIdx.x < stride ){
          sum[threadIdx.x] += sum[threadIdx.x + stride];
          square[threadIdx.x] += square[threadIdx.x + stride];
        }
        __syncthreads();
      }
      if( threadIdx.x == 0 ){
        sums_[bin] = sum[0];
        squares_[bin] = square[0];
        if( not ::isfinite(sum[0]) or not ::isfinite(square[0]) ){ atomicExch(error_, 1); }
      }
      __syncthreads();
    }
  }
}

struct Backends::CudaDeviceContext::Impl {
  int deviceIndex{0};
  std::string deviceName{};
  cudaStream_t stream{nullptr};
  std::array<cudaEvent_t, 5> stages{};
  std::unique_ptr<WeightBuffers> buffers{};

  ~Impl() {
    int previousDevice = 0;
    if( cudaGetDevice(&previousDevice) != cudaSuccess or cudaSetDevice(deviceIndex) != cudaSuccess ){ return; }
    if( stream ){ cudaStreamSynchronize(stream); }
    buffers.reset();
    for( auto event : stages ){ if( event ){ cudaEventDestroy(event); } }
    if( stream ){ cudaStreamDestroy(stream); }
    cudaSetDevice(previousDevice);
  }
};

Backends::CudaDeviceContext::CudaDeviceContext(int deviceIndex_) : _impl_(std::make_unique<Impl>()) {
  int deviceCount = 0;
  checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount (run inside a GPU allocation)");
  if( deviceCount == 0 ){ throw std::runtime_error("CudaBackend: no CUDA device visible. Run inside a GPU allocation."); }
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
  for( auto& event : _impl_->stages ){ checkCuda(cudaEventCreate(&event), "cudaEventCreate"); }
}

Backends::CudaDeviceContext::~CudaDeviceContext() = default;
const std::string& Backends::CudaDeviceContext::getDeviceName() const { return _impl_->deviceName; }

void Backends::CudaDeviceContext::build(const CudaWeightModel& model_) {
  DeviceScope scope(_impl_->deviceIndex);
  _impl_->buffers.reset();
  auto buffers = std::make_unique<WeightBuffers>();
  auto& b = *buffers;
  try{
    b.events.upload(model_.events, _impl_->stream);
    b.dials.upload(model_.dials, _impl_->stream);
    b.inputs.upload(model_.inputs, _impl_->stream);
    b.payloads.upload(model_.payloads, _impl_->stream);
    b.dialIndices.upload(model_.eventDialIndices, _impl_->stream);
    b.binOffsets.upload(model_.binOffsets, _impl_->stream);
    b.binEvents.upload(model_.binEvents, _impl_->stream);
    b.parameters.allocate(model_.parameterCount);
    b.responses.allocate(model_.dials.size());
    b.previousInputs.allocate(model_.dials.size());
    b.weights.allocate(model_.events.size());
    b.sums.allocate(model_.binOffsets.size() - 1);
    b.sumSquares.allocate(b.sums.size);
    b.error.allocate(1);
    b.hasReweightCap = model_.hasReweightCap;
    b.maxReweight = model_.maxReweight;
    checkCuda(cudaStreamSynchronize(_impl_->stream), "model upload completion");
  }
  catch(...){ cudaStreamSynchronize(_impl_->stream); throw; }
  _impl_->buffers = std::move(buffers);
}

void Backends::CudaDeviceContext::propagate(const std::vector<double>& parameters_, bool recomputeResponses_,
                                           BackendTimingSummary& timing_) {
  DeviceScope scope(_impl_->deviceIndex);
  if( not _impl_->buffers ){ throw std::runtime_error("CudaBackend device model has not been built."); }
  auto& b = *_impl_->buffers;
  if( parameters_.size() != b.parameters.size ){ throw std::runtime_error("CUDA parameter count mismatch."); }
  int error = 0;
  const auto stream = _impl_->stream;
  auto record = [&](std::size_t i_){ checkCuda(cudaEventRecord(_impl_->stages[i_], stream), "cudaEventRecord"); };
  try{
    record(0);
    checkCuda(cudaMemsetAsync(b.error.data, 0, sizeof(int), stream), "reset execution error");
    if( b.parameters.size ){
      checkCuda(cudaMemcpyAsync(b.parameters.data, parameters_.data(), b.parameters.size * sizeof(double),
                                cudaMemcpyHostToDevice, stream), "parameter upload");
    }
    record(1);
    if( b.dials.size ){
      evaluateResponses<<<gridSize(b.dials.size), blockSize, 0, stream>>>(b.dials.data, b.dials.size,
          b.inputs.data, b.payloads.data, b.parameters.data, b.responses.data, b.previousInputs.data,
          b.cachePrimed and not recomputeResponses_, b.error.data);
      checkCuda(cudaGetLastError(), "dial response kernel launch");
    }
    record(2);
    if( b.events.size ){
      reweightEvents<<<gridSize(b.events.size), blockSize, 0, stream>>>(b.events.data, b.events.size,
          b.dialIndices.data, b.responses.data, b.weights.data, b.hasReweightCap, b.maxReweight, b.error.data);
      checkCuda(cudaGetLastError(), "event weight kernel launch");
    }
    record(3);
    if( b.sums.size ){
      const auto blocks = static_cast<unsigned int>(std::min<std::size_t>(b.sums.size, 65535));
      reduceHistograms<<<blocks, blockSize, 0, stream>>>(b.binOffsets.data, b.binEvents.data,
          b.sums.size, b.weights.data, b.sums.data, b.sumSquares.data, b.error.data);
      checkCuda(cudaGetLastError(), "histogram kernel launch");
    }
    record(4);
    checkCuda(cudaMemcpyAsync(&error, b.error.data, sizeof(int), cudaMemcpyDeviceToHost, stream), "execution error readback");
    checkCuda(cudaStreamSynchronize(stream), "propagation completion");
    if( error ){ throw std::runtime_error("CudaBackend: nonfinite or out-of-range numerical result."); }
    b.cachePrimed = true;
    auto elapsed = [&](std::size_t first_, std::size_t last_){
      float milliseconds = 0.;
      checkCuda(cudaEventElapsedTime(&milliseconds, _impl_->stages[first_], _impl_->stages[last_]), "CUDA stage timing");
      return double(milliseconds) * 1.e-3;
    };
    timing_.parameterUploadSeconds = elapsed(0, 1);
    timing_.cachedDialStageSeconds = elapsed(1, 2);
    timing_.eventWeightsStageSeconds = elapsed(2, 3);
    timing_.histogramStageSeconds = elapsed(3, 4);
    timing_.devicePropagationSeconds = elapsed(0, 4);
  }
  catch(...){ b.cachePrimed = false; cudaStreamSynchronize(stream); throw; }
}

void Backends::CudaDeviceContext::readWeights(std::vector<double>& weights_) {
  DeviceScope scope(_impl_->deviceIndex);
  try{
    _impl_->buffers->weights.read(weights_, _impl_->stream);
    checkCuda(cudaStreamSynchronize(_impl_->stream), "weight readback completion");
  }
  catch(...){ cudaStreamSynchronize(_impl_->stream); throw; }
}

void Backends::CudaDeviceContext::readHistograms(std::vector<double>& sums_, std::vector<double>& sumSquares_) {
  DeviceScope scope(_impl_->deviceIndex);
  try{
    _impl_->buffers->sums.read(sums_, _impl_->stream);
    _impl_->buffers->sumSquares.read(sumSquares_, _impl_->stream);
    checkCuda(cudaStreamSynchronize(_impl_->stream), "histogram readback completion");
  }
  catch(...){ cudaStreamSynchronize(_impl_->stream); throw; }
}

void Backends::CudaDeviceContext::invalidate() { if( _impl_->buffers ){ _impl_->buffers->cachePrimed = false; } }

Backends::BackendDeviceView Backends::CudaDeviceContext::getDeviceView() const {
  BackendDeviceView out;
  out.device = _impl_.get();
  const auto& b = *_impl_->buffers;
  out.eventWeights = b.weights.data;
  out.eventWeightsBytes = b.weights.size * sizeof(double);
  out.histSums = b.sums.data;
  out.histSumSquares = b.sumSquares.data;
  out.histogramBytes = b.sums.size * sizeof(double);
  return out;
}
