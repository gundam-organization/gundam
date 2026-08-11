#ifndef GUNDAM_BACKEND_ENGINE_LAYOUT_H
#define GUNDAM_BACKEND_ENGINE_LAYOUT_H

#include "EngineBindings.h"
#include "EngineView.h"

class LikelihoodInterface;

namespace Backends {

  struct EngineLayout {
    EngineView view{};
    EngineBindings bindings{};

    void clear();
    void build(LikelihoodInterface& likelihoodInterface_);
    [[nodiscard]] bool empty() const { return view.empty() and bindings.empty(); }
  };

}

#endif // GUNDAM_BACKEND_ENGINE_LAYOUT_H
