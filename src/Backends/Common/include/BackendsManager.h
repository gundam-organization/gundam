#ifndef GUNDAM_BACKENDS_MANAGER_H
#define GUNDAM_BACKENDS_MANAGER_H

#include "BackendConfig.h"
#include "BackendFactory.h"
#include "BackendLikelihoodModel.h"
#include "BackendModel.h"
#include "BackendRuntimeManager.h"

#include "ConfigUtils.h"

#include <memory>

namespace Backends {

  [[nodiscard]] std::string formatBackendTimingSummary(const BackendTimingSummary& timing_);

  class BackendsManager : public JsonBaseClass {
  public:
    BackendsManager() = default;

    [[nodiscard]] const BackendConfig& getBackendConfig() const { return _backendConfig_; }
    [[nodiscard]] const BackendLikelihoodModel& getLikelihoodModel() const { return _backendLikelihoodModel_; }
    [[nodiscard]] const PropagationRequest& getPropagationRequest() const { return _propagationRequest_; }
    [[nodiscard]] bool isEnabled() const { return _backendConfig_.isEnabled; }
    [[nodiscard]] bool hasBackend() const { return _backendRuntimeManager_ != nullptr and _backendRuntimeManager_->hasBackend(); }
    [[nodiscard]] BackendRuntimeManager* getBackendRuntimeManager() { return _backendRuntimeManager_.get(); }
    [[nodiscard]] const BackendRuntimeManager* getBackendRuntimeManager() const { return _backendRuntimeManager_.get(); }

    void setLikelihoodModel(const BackendLikelihoodModel& likelihoodModel_);
    void initializeBackend(const BackendModel& model_);

  protected:
    void configureImpl() override;

  private:
    BackendConfig _backendConfig_{};
    BackendLikelihoodModel _backendLikelihoodModel_{};
    PropagationRequest _propagationRequest_{};
    std::shared_ptr<BackendRuntimeManager> _backendRuntimeManager_{nullptr};
  };

}

#endif // GUNDAM_BACKENDS_MANAGER_H
