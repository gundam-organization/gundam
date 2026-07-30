#ifndef GUNDAM_BACKEND_ENGINE_LAYOUT_H
#define GUNDAM_BACKEND_ENGINE_LAYOUT_H

#include "BackendEngineBindings.h"
#include "BackendEngineView.h"

class LikelihoodInterface;

namespace Backends {

  struct BackendEngineLayout {
    BackendEngineView view{};
    BackendEngineBindings bindings{};

    void clear();
    void build(LikelihoodInterface& likelihoodInterface_);
    [[nodiscard]] bool empty() const { return view.empty() and bindings.empty(); }
  };

}

#endif // GUNDAM_BACKEND_ENGINE_LAYOUT_H
