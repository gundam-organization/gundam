#include "BackendFactory.h"

#include "BackendManager.h"
#include "CpuBackend.h"
#include "CudaBackend.h"
#ifdef __APPLE__
#include "MpsBackend.h"
#endif

#include "Logger.h"

std::unique_ptr<Backends::Backend> Backends::makeBackend(const BackendManager& config_) {
  return makeBackend(config_.getType());
}

std::unique_ptr<Backends::Backend> Backends::makeBackend(const std::string& type_) {
  if( type_ == "CPU" or type_ == "cpu" ){
    return std::make_unique<CpuBackend>();
  }
  if( type_ == "CUDA" or type_ == "cuda" ){
    return std::make_unique<CudaBackend>();
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
