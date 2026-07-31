#include "BackendFactory.h"

#include "BackendManager.h"
#include "CpuBackend.h"
#include "MpsBackend.h"

#include "Logger.h"

std::unique_ptr<Backends::IPropagationBackend> Backends::makeBackend(const BackendManager& config_) {
  if( config_.getType() == "CPU" or config_.getType() == "cpu" ){
    return std::make_unique<CpuBackend>();
  }
  if( config_.getType() == "MPS" or config_.getType() == "mps" ){
    return std::make_unique<MpsBackend>();
  }

  LogThrow("Unknown backend type: " << config_.getType());
  return nullptr;
}
