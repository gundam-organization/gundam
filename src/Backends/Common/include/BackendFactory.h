#ifndef GUNDAM_BACKEND_FACTORY_H
#define GUNDAM_BACKEND_FACTORY_H

#include <memory>
#include <string>

namespace Backends {

  class BackendManager;
  class Backend;

  std::unique_ptr<Backend> makeBackend(const BackendManager& config_);
  std::unique_ptr<Backend> makeBackend(const std::string& type_);

}

#endif // GUNDAM_BACKEND_FACTORY_H
