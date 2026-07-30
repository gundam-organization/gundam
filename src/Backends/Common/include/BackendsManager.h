#ifndef GUNDAM_BACKENDS_MANAGER_H
#define GUNDAM_BACKENDS_MANAGER_H

#include "BackendFactory.h"
#include "BackendLikelihoodModel.h"
#include "BackendModel.h"
#include "BackendRuntimeManager.h"

#include "ConfigUtils.h"

#include <future>
#include <initializer_list>
#include <memory>

class LikelihoodInterface;
class Propagator;

namespace Backends {

  [[nodiscard]] std::string formatBackendTimingSummary(const BackendTimingSummary& timing_);

  struct BackendPropagationResult {
    bool isValid{false};
    bool hasStatLikelihood{false};
    double statLikelihood{0};
  };

  class BackendsManager : public JsonBaseClass {
  public:
    BackendsManager() = default;

    [[nodiscard]] const BackendLikelihoodModel& getLikelihoodModel() const { return _backendLikelihoodModel_; }
    [[nodiscard]] const PropagationRequest& getPropagationRequest() const { return _propagationRequest_; }
    [[nodiscard]] bool isEnabled() const { return _isEnabled_; }
    [[nodiscard]] bool isAutoMaterializeEnabled() const { return _enableAutoMaterialize_; }
    [[nodiscard]] const std::string& getType() const { return _type_; }
    [[nodiscard]] const std::vector<OutputRequest>& getOutputRequests() const { return _outputRequests_; }
    [[nodiscard]] const std::vector<OutputRequest>& getMaterializeOutputList() const { return _materializeOutputList_; }
    [[nodiscard]] bool hasBackend() const { return _backendRuntimeManager_ != nullptr and _backendRuntimeManager_->hasBackend(); }
    [[nodiscard]] BackendRuntimeManager* getBackendRuntimeManager() { return _backendRuntimeManager_.get(); }
    [[nodiscard]] const BackendRuntimeManager* getBackendRuntimeManager() const { return _backendRuntimeManager_.get(); }

    void setEnableAutoMaterialize(bool enableAutoMaterialize_);
    void setMaterializeOutputList(std::vector<OutputRequest> materializeOutputList_);
    void setMaterializeOutputList(std::initializer_list<OutputRequest> materializeOutputList_);
    void initializeBackend(const LikelihoodInterface& likelihoodInterface_);
    std::future<BackendPropagationResult> propagate(Propagator& propagator_);

  protected:
    void configureImpl() override;

  private:
    // configuration
    bool _isEnabled_{false};
    bool _enableAutoMaterialize_{true};
    std::string _type_{"CPU"};

    [[nodiscard]] PropagationRequest makePropagationRequest() const;

    std::vector<OutputRequest> _outputRequests_{OutputRequest::Likelihood};
    std::vector<OutputRequest> _materializeOutputList_{
        OutputRequest::EventWeights,
        OutputRequest::Histograms,
        OutputRequest::Likelihood
    };
    BackendLikelihoodModel _backendLikelihoodModel_{};
    PropagationRequest _propagationRequest_{};
    std::shared_ptr<BackendRuntimeManager> _backendRuntimeManager_{nullptr};
  };

}

#endif // GUNDAM_BACKENDS_MANAGER_H
