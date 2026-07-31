#include "BackendFactory.h"

#include "BackendManager.h"
#include "CpuEngineBackend.h"
#include "MpsEngineBackend.h"

#include "Logger.h"

std::unique_ptr<Backends::EngineBackend> Backends::makeBackend(const BackendManager& config_) {
  if( config_.getType() == "CPU" or config_.getType() == "cpu" ){
    return std::make_unique<CpuEngineBackend>();
  }
  if( config_.getType() == "MPS" or config_.getType() == "mps" ){
    return std::make_unique<MpsEngineBackend>();
  }

  LogThrow("Unknown backend type: " << config_.getType());
  return nullptr;
}
