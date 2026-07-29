#ifndef GUNDAM_BACKEND_MANAGER_H
#define GUNDAM_BACKEND_MANAGER_H

#include "BackendModel.h"
#include "BackendTypes.h"
#include "IPropagationBackend.h"
#include "ParameterSnapshot.h"

#include <memory>

namespace Backends {

  class BackendManager {
  public:
    BackendManager() = default;

    void setBackend(std::unique_ptr<IPropagationBackend> backend_);
    [[nodiscard]] bool hasBackend() const { return _backend_ != nullptr; }
    [[nodiscard]] IPropagationBackend* getBackend() { return _backend_.get(); }
    [[nodiscard]] const IPropagationBackend* getBackend() const { return _backend_.get(); }

    void build(const BackendModel& model_);
    PropagationToken requestPropagation(
        const ParameterSnapshot& parameters_,
        const PropagationRequest& request_);
    void wait(const PropagationToken& token_);
    void materialize(const PropagationToken& token_, OutputRequest output_);

  private:
    std::unique_ptr<IPropagationBackend> _backend_{nullptr};
  };

}

#endif // GUNDAM_BACKEND_MANAGER_H
