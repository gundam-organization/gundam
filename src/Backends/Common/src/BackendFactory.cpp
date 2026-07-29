#include "BackendFactory.h"

#include "BackendConfig.h"
#include "CpuBackend.h"
#include "MpsBackend.h"

#include "Logger.h"

std::unique_ptr<Backends::IPropagationBackend> Backends::makeBackend(const BackendConfig& config_) {
  if( config_.type == "CPU" or config_.type == "cpu" ){
    return std::make_unique<CpuBackend>();
  }
  if( config_.type == "MPS" or config_.type == "mps" ){
    return std::make_unique<MpsBackend>();
  }

  LogThrow("Unknown backend type: " << config_.type);
  return nullptr;
}
