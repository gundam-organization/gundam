#ifndef GUNDAM_BACKEND_RUNTIME_MANAGER_H
#define GUNDAM_BACKEND_RUNTIME_MANAGER_H

#include "BackendEngineView.h"
#include "BackendTypes.h"
#include "IPropagationBackend.h"
#include "ParameterSnapshot.h"

#include <memory>

namespace Backends {

  class BackendRuntimeManager {
  public:
    BackendRuntimeManager() = default;

    void setBackend(std::unique_ptr<IPropagationBackend> backend_);
    [[nodiscard]] bool hasBackend() const { return _backend_ != nullptr; }
    [[nodiscard]] IPropagationBackend* getBackend() { return _backend_.get(); }
    [[nodiscard]] const IPropagationBackend* getBackend() const { return _backend_.get(); }

    void build(const BackendEngineView& engineView_);
    PropagationToken requestPropagation(const ParameterSnapshot& parameters_);
    void wait(const PropagationToken& token_);
    void materialize(const PropagationToken& token_, OutputRequest output_);

  private:
    std::unique_ptr<IPropagationBackend> _backend_{nullptr};
  };

}

#endif // GUNDAM_BACKEND_RUNTIME_MANAGER_H
