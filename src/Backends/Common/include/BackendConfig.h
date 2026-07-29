#ifndef GUNDAM_BACKEND_CONFIG_H
#define GUNDAM_BACKEND_CONFIG_H

#include "BackendTypes.h"
#include "ConfigUtils.h"

#include <string>
#include <vector>

namespace Backends {

  struct BackendConfig {
    bool isEnabled{false};
    std::string type{"CPU"};
    std::vector<OutputRequest> outputRequests{OutputRequest::Histograms};

    [[nodiscard]] PropagationRequest makePropagationRequest() const;

    static BackendConfig fromConfig(ConfigReader config_);
  };

}

#endif // GUNDAM_BACKEND_CONFIG_H
