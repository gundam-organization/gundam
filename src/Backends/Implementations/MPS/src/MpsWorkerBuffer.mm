#include "MpsWorkerBuffer.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

extern "C" void* gundamMpsWorkerOpen(const char*, std::size_t);

namespace {
  class MpsWorkerBuffer final : public WorkerBuffer {
  public:
    explicit MpsWorkerBuffer(std::size_t count_, const char* path_ = nullptr) : _count_(count_) {
      const auto page = std::size_t(sysconf(_SC_PAGESIZE));
      if( count_ > (std::numeric_limits<std::size_t>::max() - page) / sizeof(float) ){
        throw std::overflow_error("MPS worker buffer size overflow.");
      }
      _bytes_ = ((std::max<std::size_t>(count_, 1) * sizeof(float) + page - 1) / page) * page;
      try {
        auto device = MTLCreateSystemDefaultDevice();
        if( device == nil or not [device hasUnifiedMemory] ){
          throw std::runtime_error("Device PythonWorker requires unified-memory Metal hardware.");
        }
        if( path_ ){
          _path_ = path_;
          _fd_ = open(path_, O_RDWR);
        }
        else {
          auto pattern = std::filesystem::absolute(std::filesystem::temp_directory_path() / "gundam-worker-XXXXXX").string();
          std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
          _fd_ = mkstemp(name.data());
          if( _fd_ >= 0 ){ _path_ = name.data(); _owner_ = true; }
        }
        if( _fd_ < 0 ){ throw std::runtime_error(std::strerror(errno)); }
        if( fcntl(_fd_, F_SETFD, FD_CLOEXEC) < 0 ){ throw std::runtime_error(std::strerror(errno)); }
        if( _owner_ and ftruncate(_fd_, off_t(_bytes_)) != 0 ){ throw std::runtime_error(std::strerror(errno)); }
        struct stat info{};
        if( fstat(_fd_, &info) != 0 or info.st_size != off_t(_bytes_) ){
          throw std::runtime_error("MPS worker mapping size does not match its descriptor.");
        }
        _mapping_ = mmap(nullptr, _bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, _fd_, 0);
        if( _mapping_ == MAP_FAILED ){ _mapping_ = nullptr; throw std::runtime_error(std::strerror(errno)); }
        _buffer_ = [device newBufferWithBytesNoCopy:_mapping_ length:_bytes_
                                          options:MTLResourceStorageModeShared deallocator:nil];
        if( _buffer_ == nil ){ throw std::runtime_error("Metal rejected the PythonWorker mapping."); }
      }
      catch( ... ){ release(); throw; }
    }
    ~MpsWorkerBuffer() override { release(); }
    WorkerBufferDescriptor describe() const override {
      Dl_info info{};
      if( dladdr((void*)&gundamMpsWorkerOpen, &info) == 0 or info.dli_fname == nullptr ){
        throw std::runtime_error("Cannot locate the compiled MPS worker transport library.");
      }
      return {"mps-shared", _path_, std::filesystem::absolute(info.dli_fname).string(), _count_, _bytes_};
    }
    void writeValues(const double* values_, std::size_t count_) override {
      if( count_ != _count_ ){ throw std::invalid_argument("MPS worker input size mismatch."); }
      if( count_ ){ std::transform(values_, values_ + count_, static_cast<float*>(_mapping_), [](double x_){ return float(x_); }); }
    }
    void readValues(double* values_, std::size_t count_) const override {
      if( count_ != _count_ ){ throw std::invalid_argument("MPS worker output size mismatch."); }
      if( count_ ){ std::copy_n(static_cast<const float*>(_mapping_), count_, values_); }
    }
    void* nativeHandle() const override { return (void*)_buffer_; }
  private:
    void release() {
      [_buffer_ release]; _buffer_ = nil;
      if( _mapping_ ){ munmap(_mapping_, _bytes_); _mapping_ = nullptr; }
      if( _fd_ >= 0 ){ close(_fd_); _fd_ = -1; }
      if( _owner_ ){ unlink(_path_.c_str()); _owner_ = false; }
    }
    std::size_t _count_, _bytes_{0};
    std::string _path_;
    bool _owner_{false};
    int _fd_{-1};
    void* _mapping_{nullptr};
    id<MTLBuffer> _buffer_{nil};
  };
  thread_local std::string lastError;
}

// C ABI only: the worker loads this backend library with ctypes in its own venv.
// No Python, NumPy or Torch headers/libraries are used by this compiled target.
extern "C" void* gundamMpsWorkerOpen(const char* path_, std::size_t count_) {
  @autoreleasepool {
    try { lastError.clear(); return new MpsWorkerBuffer(count_, path_); }
    catch( const std::exception& error ){ lastError = error.what(); return nullptr; }
  }
}
extern "C" const char* gundamMpsWorkerError() { return lastError.c_str(); }
extern "C" void* gundamMpsWorkerMetal(void* handle_) {
  return handle_ ? static_cast<MpsWorkerBuffer*>(handle_)->nativeHandle() : nullptr;
}
extern "C" void gundamMpsWorkerClose(void* handle_) { delete static_cast<MpsWorkerBuffer*>(handle_); }

std::shared_ptr<WorkerBuffer> Backends::makeMpsWorkerBuffer(std::size_t count_) {
  @autoreleasepool { return std::make_shared<MpsWorkerBuffer>(count_); }
}

void Backends::copyMpsWorkerWeights(const WorkerBuffer& source_, void* destination_,
                                   std::size_t offset_, std::size_t count_, std::size_t scalarBytes_) {
  @autoreleasepool {
    auto source = (id<MTLBuffer>)source_.nativeHandle();
    auto destination = (id<MTLBuffer>)destination_;
    if( source == nil or destination == nil or [source device] != [destination device]
        or (scalarBytes_ != sizeof(float) and scalarBytes_ != 2 * sizeof(float))
        or offset_ > [destination length] / scalarBytes_
        or count_ > [destination length] / scalarBytes_ - offset_
        or count_ > [source length] / sizeof(float) ){
      throw std::runtime_error("Invalid MPS PythonWorker weight copy.");
    }
    if( count_ == 0 ){ return; }
    auto queue = [[destination device] newCommandQueue];
    if( queue == nil ){ throw std::runtime_error("Cannot create MPS worker transfer queue."); }
    auto command = [queue commandBuffer];
    if( scalarBytes_ == sizeof(float) ){
      auto encoder = [command blitCommandEncoder];
      [encoder copyFromBuffer:source sourceOffset:0 toBuffer:destination
           destinationOffset:offset_ * scalarBytes_ size:count_ * scalarBytes_];
      [encoder endEncoding];
    }
    else {
      // Observable propagation uses DoubleFloat {hi, lo}; preserve the worker's
      // float32 result as {value, 0} without visiting the numerical data on CPU.
      NSError* error = nil;
      auto library = [[destination device] newLibraryWithSource:
          @"#include <metal_stdlib>\nusing namespace metal;\nkernel void pack(device const float* x[[buffer(0)]], device float2* y[[buffer(1)]], uint i[[thread_position_in_grid]]) { y[i]=float2(x[i],0.f); }"
          options:nil error:&error];
      auto function = [library newFunctionWithName:@"pack"];
      auto pipeline = function ? [[destination device] newComputePipelineStateWithFunction:function error:&error] : nil;
      [function release]; [library release];
      if( pipeline == nil ){ [queue release]; throw std::runtime_error([[error description] UTF8String]); }
      auto encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:source offset:0 atIndex:0];
      [encoder setBuffer:destination offset:offset_ * scalarBytes_ atIndex:1];
      [encoder dispatchThreads:MTLSizeMake(count_, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      [encoder endEncoding]; [pipeline release];
    }
    // First implementation is blocking. The worker synchronizes before replying,
    // and this GPU copy completes before the backend starts consuming weights.
    [command commit]; [command waitUntilCompleted];
    const auto failed = command.status == MTLCommandBufferStatusError;
    const std::string message = failed ? [[[command error] description] UTF8String] : "";
    [queue release];
    if( failed ){ throw std::runtime_error(message); }
  }
}
