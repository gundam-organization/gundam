#include "CudaWorkerBuffer.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

extern "C" void* gundamCudaWorkerOpen(const char*, const char*, std::size_t);

namespace {
  void checkCuda(cudaError_t status_, const char* action_) {
    if( status_ != cudaSuccess ){
      throw std::runtime_error(std::string("CUDA PythonWorker: ") + action_ + ": " + cudaGetErrorString(status_));
    }
  }

  class DeviceScope {
  public:
    explicit DeviceScope(int device_) {
      checkCuda(cudaGetDevice(&_previousDevice_), "get device");
      checkCuda(cudaSetDevice(device_), "select device");
    }
    ~DeviceScope() { cudaSetDevice(_previousDevice_); }
  private:
    int _previousDevice_{0};
  };

  std::string encodeHex(const void* bytes_, std::size_t count_) {
    const auto* bytes = static_cast<const unsigned char*>(bytes_);
    std::string out; out.reserve(count_ * 2);
    for( std::size_t i = 0; i < count_; ++i ){
      out += "0123456789abcdef"[bytes[i] >> 4]; out += "0123456789abcdef"[bytes[i] & 15];
    }
    return out;
  }

  void decodeHex(const char* value_, void* bytes_, std::size_t count_) {
    if( value_ == nullptr or std::strlen(value_) != count_ * 2 ){
      throw std::runtime_error("Invalid CUDA IPC handle length.");
    }
    auto digit = [](char c_){
      if( c_ >= '0' and c_ <= '9' ){ return c_ - '0'; }
      if( c_ >= 'a' and c_ <= 'f' ){ return c_ - 'a' + 10; }
      throw std::runtime_error("Invalid CUDA IPC handle encoding.");
    };
    auto* bytes = static_cast<unsigned char*>(bytes_);
    for( std::size_t i = 0; i < count_; ++i ){ bytes[i] = (digit(value_[2*i]) << 4) | digit(value_[2*i+1]); }
  }

  std::string deviceUuid(int device_) {
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, device_), "get device UUID");
    return encodeHex(properties.uuid.bytes, sizeof(properties.uuid.bytes));
  }

  class CudaWorkerBuffer final : public WorkerBuffer {
  public:
    CudaWorkerBuffer(std::size_t count_, int device_, const char* handle_ = nullptr)
        : _count_(count_), _device_(device_), _imported_(handle_ != nullptr) {
      if( count_ > std::numeric_limits<std::size_t>::max() / sizeof(double) ){
        throw std::overflow_error("CUDA worker buffer size overflow.");
      }
      _bytes_ = std::max<std::size_t>(1, count_) * sizeof(double);
      DeviceScope scope(_device_);
      _uuid_ = deviceUuid(_device_);
      Dl_info library{};
      if( dladdr(reinterpret_cast<void*>(&gundamCudaWorkerOpen), &library) == 0 or library.dli_fname == nullptr ){
        throw std::runtime_error("Cannot locate the CUDA worker transport library.");
      }
      _libraryPath_ = std::filesystem::absolute(library.dli_fname).string();
      if( _imported_ ){
        cudaIpcMemHandle_t handle{};
        decodeHex(handle_, &handle, sizeof(handle));
        _handle_ = handle_;
        checkCuda(cudaIpcOpenMemHandle(&_pointer_, handle, cudaIpcMemLazyEnablePeerAccess), "open IPC memory");
      }
      else{
        checkCuda(cudaMalloc(&_pointer_, _bytes_), "allocate memory");
        cudaIpcMemHandle_t handle{};
        const auto status = cudaIpcGetMemHandle(&handle, _pointer_);
        if( status != cudaSuccess ){ cudaFree(_pointer_); _pointer_ = nullptr; checkCuda(status, "export IPC memory"); }
        _handle_ = encodeHex(&handle, sizeof(handle));
      }
    }
    ~CudaWorkerBuffer() override {
      // Destruction cannot throw, including during CUDA runtime teardown.
      int previous = 0;
      if( cudaGetDevice(&previous) == cudaSuccess and cudaSetDevice(_device_) == cudaSuccess ){
        if( _imported_ ){ cudaIpcCloseMemHandle(_pointer_); }
        else{ cudaFree(_pointer_); }
        cudaSetDevice(previous);
      }
    }
    WorkerBufferDescriptor describe() const override {
      WorkerBufferDescriptor out;
      out.transport = "cuda-ipc"; out.libraryPath = _libraryPath_;
      out.count = _count_; out.byteCount = _bytes_; out.dtype = "float64";
      out.device = "cuda"; out.ipcHandle = _handle_; out.deviceUuid = _uuid_;
      return out;
    }
    void writeValues(const double* values_, std::size_t count_) override {
      validateCount(count_); DeviceScope scope(_device_);
      if( count_ ){
        checkCuda(cudaMemcpy(_pointer_, values_, count_ * sizeof(double), cudaMemcpyHostToDevice), "write buffer");
        // A pageable host upload may return after staging. Complete the copy
        // before notifying a consumer in another process/context.
        checkCuda(cudaStreamSynchronize(nullptr), "publish buffer");
      }
    }
    void readValues(double* values_, std::size_t count_) const override {
      validateCount(count_); DeviceScope scope(_device_);
      if( count_ ){ checkCuda(cudaMemcpy(values_, _pointer_, count_ * sizeof(double), cudaMemcpyDeviceToHost), "read buffer"); }
    }
    void* nativeHandle() const override { return _pointer_; }
    int deviceIndex() const { return _device_; }
    std::size_t size() const { return _count_; }
  private:
    void validateCount(std::size_t count_) const {
      if( count_ != _count_ ){ throw std::invalid_argument("CUDA worker buffer size mismatch."); }
    }
    std::size_t _count_, _bytes_;
    int _device_;
    bool _imported_;
    void* _pointer_{nullptr};
    std::string _handle_, _uuid_, _libraryPath_;
  };

  // DLPack's stable, unversioned C ABI (DLManagedTensor). No dependency on
  // Torch/Python headers or libraries. The tensor owns a shared IPC mapping.
  // Layout: https://github.com/dmlc/dlpack/blob/v1.0/include/dlpack/dlpack.h
  struct DlDevice { std::int32_t type, id; };
  struct DlDataType { std::uint8_t code, bits; std::uint16_t lanes; };
  struct DlTensor {
    void* data;
    DlDevice device;
    std::int32_t ndim;
    DlDataType dtype;
    std::int64_t* shape;
    std::int64_t* strides;
    std::uint64_t byteOffset;
  };
  struct DlManagedTensor {
    DlTensor tensor;
    void* context;
    void (*deleter)(DlManagedTensor*);
  };
  struct WorkerTensor {
    DlManagedTensor managed{};
    std::shared_ptr<CudaWorkerBuffer> buffer;
    std::vector<std::int64_t> shape;
  };
  using Mapping = std::shared_ptr<CudaWorkerBuffer>;
  thread_local std::string lastError;
}

extern "C" const char* gundamCudaWorkerError() { return lastError.c_str(); }
extern "C" void* gundamCudaWorkerOpen(const char* handle_, const char* uuid_, std::size_t count_) {
  try {
    lastError.clear();
    if( handle_ == nullptr or uuid_ == nullptr ){
      throw std::runtime_error("Missing CUDA worker IPC handle or device UUID.");
    }
    int count = 0;
    checkCuda(cudaGetDeviceCount(&count), "enumerate devices");
    for( int device = 0; device < count; ++device ){
      if( deviceUuid(device) == uuid_ ){ return new Mapping(std::make_shared<CudaWorkerBuffer>(count_, device, handle_)); }
    }
    throw std::runtime_error("GUNDAM's CUDA worker GPU is not visible in this Python process.");
  }
  catch( const std::exception& error ){ lastError = error.what(); return nullptr; }
}
extern "C" void gundamCudaWorkerClose(void* handle_) { delete static_cast<Mapping*>(handle_); }
extern "C" void gundamCudaWorkerDeleteTensor(void* tensor_) {
  auto* tensor = static_cast<DlManagedTensor*>(tensor_);
  if( tensor ){ tensor->deleter(tensor); }
}
extern "C" void* gundamCudaWorkerTensor(void* handle_, const std::int64_t* shape_, std::int32_t ndim_) {
  try {
    lastError.clear();
    if( handle_ == nullptr or shape_ == nullptr or ndim_ <= 0 ){
      throw std::runtime_error("Invalid CUDA worker tensor dimensions.");
    }
    auto out = std::make_unique<WorkerTensor>();
    out->buffer = *static_cast<Mapping*>(handle_);
    std::size_t count = 1;
    for( std::int32_t i = 0; i < ndim_; ++i ){
      if( shape_[i] < 0 or (shape_[i] != 0 and count > std::numeric_limits<std::size_t>::max() / std::size_t(shape_[i])) ){
        throw std::runtime_error("CUDA worker tensor shape overflow.");
      }
      count *= std::size_t(shape_[i]); out->shape.push_back(shape_[i]);
    }
    if( count != out->buffer->size() ){ throw std::runtime_error("CUDA worker tensor shape mismatch."); }
    out->managed.tensor = {out->buffer->nativeHandle(), {2, out->buffer->deviceIndex()}, ndim_,
                           {2, 64, 1}, out->shape.data(), nullptr, 0};
    out->managed.context = out.get();
    out->managed.deleter = [](DlManagedTensor* tensor_){ delete static_cast<WorkerTensor*>(tensor_->context); };
    auto* managed = &out->managed;
    out.release();
    return managed;
  }
  catch( const std::exception& error ){ lastError = error.what(); return nullptr; }
}

std::shared_ptr<WorkerBuffer> Backends::makeCudaWorkerBuffer(std::size_t count_, int deviceIndex_) {
  return std::make_shared<CudaWorkerBuffer>(count_, deviceIndex_);
}
