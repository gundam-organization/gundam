#include "BackendFactory.h"

#include "BackendManager.h"
#include "CpuBackend.h"
#ifdef GUNDAM_USING_CUDA
#include "CudaBackend.h"
#endif
#ifdef __APPLE__
#include "MpsBackend.h"
#endif

#include "Logger.h"

std::vector<std::string> Backends::getCompiledBackendNames() {
  std::vector<std::string> names{"CPU"};
#ifdef GUNDAM_USING_CUDA
  names.emplace_back("CUDA");
#endif
#ifdef __APPLE__
  names.emplace_back("MPS");
#endif
  return names;
}

std::unique_ptr<Backends::Backend> Backends::makeBackend(const BackendManager& config_) {
  return makeBackend(config_.getType());
}

std::unique_ptr<Backends::Backend> Backends::makeBackend(const std::string& type_) {
  if( type_ == "CPU" or type_ == "cpu" ){
    return std::make_unique<CpuBackend>();
  }
  if( type_ == "CUDA" or type_ == "cuda" ){
#ifdef GUNDAM_USING_CUDA
    return std::make_unique<CudaBackend>();
#else
    LogThrow("CUDA backend was not compiled. Configure GUNDAM with -DWITH_CUDA_LIB=ON and a CUDA toolkit.");
#endif
  }
  if( type_ == "MPS" or type_ == "mps" ){
#ifdef __APPLE__
    return std::make_unique<MpsBackend>();
#else
    LogThrow("MPS backend requires Apple Metal.");
#endif
  }

  LogThrow("Unknown backend type: " << type_);
  return nullptr;
}
