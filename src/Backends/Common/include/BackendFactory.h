#ifndef GUNDAM_BACKEND_FACTORY_H
#define GUNDAM_BACKEND_FACTORY_H

#include <memory>

namespace Backends {

  class BackendsManager;
  class IPropagationBackend;

  std::unique_ptr<IPropagationBackend> makeBackend(const BackendsManager& config_);

}

#endif // GUNDAM_BACKEND_FACTORY_H
