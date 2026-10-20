#ifndef GUNDAM_BACKENDS_MANAGER_H
#define GUNDAM_BACKENDS_MANAGER_H

#include "EngineLayout.h"
#include "BackendFactory.h"
#include "Backend.h"

#include "ConfigUtils.h"

#include <functional>
#include <future>
#include <initializer_list>
#include <memory>

class LikelihoodInterface;
namespace Backends {

  [[nodiscard]] std::string formatBackendTimingSummary(const BackendTimingSummary& timing_);

  struct BackendPropagationResult {
    bool isValid{false};
    bool hasStatLikelihood{false};
    double statLikelihood{0};
  };

  class BackendManager : public JsonBaseClass {

  protected:
    void configureImpl() override;
    void initializeImpl() override;

  public:
    BackendManager() = default;

    void setLikelihoodInterfacePtr(LikelihoodInterface* likelihoodInterfacePtr_){ _likelihoodInterfacePtr_ = likelihoodInterfacePtr_; }

    [[nodiscard]] const EngineLayout& getEngineLayout() const { return *_backendEngineLayout_; }
    [[nodiscard]] const EngineView& getEngineView() const { return _backendEngineLayout_->view; }
    [[nodiscard]] double getLastPreparationSeconds() const { return _lastPreparationSeconds_; }
    [[nodiscard]] bool isEnabled() const { return _isEnabled_; }
    [[nodiscard]] bool isAutoMaterializeEnabled() const { return _enableAutoMaterialize_; }
    [[nodiscard]] const std::string& getType() const { return _type_; }
    [[nodiscard]] const std::vector<OutputRequest>& getMaterializeOutputList() const { return _materializeOutputList_; }
    [[nodiscard]] bool willAutoMaterialize(OutputRequest outputRequest_) const;
    [[nodiscard]] bool hasBackend() const { return _backend_ != nullptr; }
    [[nodiscard]] Backend* getBackend() { return _backend_.get(); }
    [[nodiscard]] const Backend* getBackend() const { return _backend_.get(); }

    void setEnableAutoMaterialize(bool enableAutoMaterialize_){ _enableAutoMaterialize_ = enableAutoMaterialize_; }
    void setMaterializeOutputList(const std::vector<OutputRequest>& materializeOutputList_){ _materializeOutputList_ = materializeOutputList_; }
    void setMaterializeOutputList(std::initializer_list<OutputRequest> materializeOutputList_);
    void materialize(OutputRequest outputRequest_);
    std::future<BackendPropagationResult> propagate();


  private:
    // configuration
    double _lastPreparationSeconds_{0};
    bool _isEnabled_{false};
    bool _enableAutoMaterialize_{true};
    std::string _type_{"CPU"};
    ConfigReader _backendConfig_{};

    [[nodiscard]] bool shouldMaterialize(OutputRequest outputRequest_) const;

    std::vector<OutputRequest> _materializeOutputList_{
        OutputRequest::EventState,
        OutputRequest::EventWeights,
        OutputRequest::Histograms,
        OutputRequest::SampleLikelihoods,
        OutputRequest::StatLikelihood,
    };
    LikelihoodInterface* _likelihoodInterfacePtr_{nullptr};
    std::shared_ptr<EngineLayout> _backendEngineLayout_{std::make_shared<EngineLayout>()};
    std::unique_ptr<Backend> _backend_{nullptr};
    PropagationToken _lastPropagationToken_{};
    std::shared_ptr<bool> _hostBinningChanged_{};
    std::vector<std::shared_ptr<std::function<void()>>> _binningUpdateListeners_{};
    std::uint64_t _lastMaterializedStateRevision_{0};
    std::uint64_t _lastMaterializedVariableRevision_{0};
    std::vector<std::uint64_t> _lastMaterializedSampleRevisions_{};
  };

}

#endif // GUNDAM_BACKENDS_MANAGER_H
