#ifndef GUNDAM_BACKEND_FACTORY_H
#define GUNDAM_BACKEND_FACTORY_H

#include <memory>

namespace Backends {

  struct BackendConfig;
  class IPropagationBackend;

  std::unique_ptr<IPropagationBackend> makeBackend(const BackendConfig& config_);

}

#endif // GUNDAM_BACKEND_FACTORY_H
