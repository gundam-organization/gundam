#ifndef GUNDAM_BACKENDS_MANAGER_H
#define GUNDAM_BACKENDS_MANAGER_H

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

    [[nodiscard]] const BackendLikelihoodModel& getLikelihoodModel() const { return _backendLikelihoodModel_; }
    [[nodiscard]] const PropagationRequest& getPropagationRequest() const { return _propagationRequest_; }
    [[nodiscard]] bool isEnabled() const { return _isEnabled_; }
    [[nodiscard]] const std::string& getType() const { return _type_; }
    [[nodiscard]] const std::vector<OutputRequest>& getOutputRequests() const { return _outputRequests_; }
    [[nodiscard]] const std::vector<OutputRequest>& getMaterializeOutputRequests() const { return _materializeOutputRequests_; }
    [[nodiscard]] bool hasBackend() const { return _backendRuntimeManager_ != nullptr and _backendRuntimeManager_->hasBackend(); }
    [[nodiscard]] BackendRuntimeManager* getBackendRuntimeManager() { return _backendRuntimeManager_.get(); }
    [[nodiscard]] const BackendRuntimeManager* getBackendRuntimeManager() const { return _backendRuntimeManager_.get(); }

    void setLikelihoodModel(const BackendLikelihoodModel& likelihoodModel_);
    void initializeBackend(const BackendModel& model_);

  protected:
    void configureImpl() override;

  private:
    // configuration
    bool _isEnabled_{false};
    std::string _type_{"CPU"};

    [[nodiscard]] PropagationRequest makePropagationRequest() const;


    std::vector<OutputRequest> _outputRequests_{OutputRequest::Histograms};
    std::vector<OutputRequest> _materializeOutputRequests_{};
    BackendLikelihoodModel _backendLikelihoodModel_{};
    PropagationRequest _propagationRequest_{};
    std::shared_ptr<BackendRuntimeManager> _backendRuntimeManager_{nullptr};
  };

}

#endif // GUNDAM_BACKENDS_MANAGER_H
