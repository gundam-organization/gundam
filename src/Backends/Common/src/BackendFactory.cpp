#include "BackendFactory.h"

#include "BackendManager.h"
#include "CpuBackend.h"
#ifdef __APPLE__
#include "MpsBackend.h"
#endif

#include "Logger.h"

std::unique_ptr<Backends::Backend> Backends::makeBackend(const BackendManager& config_) {
  if( config_.getType() == "CPU" or config_.getType() == "cpu" ){
    return std::make_unique<CpuBackend>();
  }
  if( config_.getType() == "MPS" or config_.getType() == "mps" ){
#ifdef __APPLE__
    return std::make_unique<MpsBackend>();
#else
    LogThrow("MPS backend requires Apple Metal.");
#endif
  }

  LogThrow("Unknown backend type: " << config_.getType());
  return nullptr;
}
