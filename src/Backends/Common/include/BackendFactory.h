#ifndef GUNDAM_BACKEND_FACTORY_H
#define GUNDAM_BACKEND_FACTORY_H

#include <memory>

namespace Backends {

  class BackendManager;
  class EngineBackend;

  std::unique_ptr<EngineBackend> makeBackend(const BackendManager& config_);

}

#endif // GUNDAM_BACKEND_FACTORY_H
