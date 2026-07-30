#include "BackendsManager.h"

#include "Logger.h"

#include <sstream>

std::string Backends::formatBackendTimingSummary(const BackendTimingSummary& timing_) {
  std::stringstream ss;
  ss << "Backend timing summary:"
     << " build[scan=" << timing_.buildCompatibilityScanSeconds
     << "s, parLookup=" << timing_.buildParameterLookupSeconds
     << "s, firstPass=" << timing_.buildFirstPassSeconds
     << "s, secondPass=" << timing_.buildSecondPassSeconds
     << "s, finalFlatten=" << timing_.buildFinalFlattenSeconds
     << "s, histIndex=" << timing_.buildHistogramIndexSeconds
     << "s, bufferUpload=" << timing_.buildBufferUploadSeconds
     << "s]"
     << " device[paramUpload=" << timing_.parameterUploadSeconds
     << "s, cachedDialStage=" << timing_.cachedDialStageSeconds
     << "s, eventWeightsStage=" << timing_.eventWeightsStageSeconds
     << "s, histogramStage=" << timing_.histogramStageSeconds
     << "s, encode=" << timing_.commandEncodeSeconds
     << "s, wait=" << timing_.deviceWaitSeconds
     << "s, histReadback=" << timing_.histogramReadbackSeconds
     << "s/" << timing_.histogramReadbackBytes
     << "B, eventWeightReadback=" << timing_.eventWeightReadbackSeconds
     << "s/" << timing_.eventWeightReadbackBytes
     << "B]"
     << " host[llh=" << timing_.likelihoodHostSeconds
     << "s, eventWeightMaterialize=" << timing_.eventWeightMaterializationSeconds
     << "s, histogramMaterialize=" << timing_.histogramMaterializationSeconds
     << "s]"
     << " counts[uniqueDials=" << timing_.uniqueDialCount
     << ", cachedDials=" << timing_.cachedDialCount
     << ", eventDialIndices=" << timing_.eventDialIndexCount
     << ", splineScalars=" << timing_.splineScalarCount
     << "]";
  return ss.str();
}

void Backends::BackendsManager::configureImpl() {
  _backendConfig_ = BackendConfig::fromConfig(_config_);
  _propagationRequest_ = _backendConfig_.makePropagationRequest();
}

void Backends::BackendsManager::setLikelihoodModel(const BackendLikelihoodModel& likelihoodModel_) {
  _backendLikelihoodModel_ = likelihoodModel_;
  if( _backendRuntimeManager_ != nullptr and _backendRuntimeManager_->hasBackend() ){
    _backendRuntimeManager_->getBackend()->setLikelihoodModel(_backendLikelihoodModel_);
  }
}

void Backends::BackendsManager::initializeBackend(const BackendModel& model_) {
  if( not _backendConfig_.isEnabled ){
    _backendRuntimeManager_ = nullptr;
    return;
  }

  LogInfo << "Initializing propagation backend: " << _backendConfig_.type << std::endl;

  _propagationRequest_ = _backendConfig_.makePropagationRequest();
  if( not _propagationRequest_.has(OutputRequest::Histograms)
      and not _propagationRequest_.has(OutputRequest::Likelihood) ){
    LogWarning << "Adding OutputRequest::Histograms to backendConfig because the current "
               << "LikelihoodInterface consumes CPU histograms." << std::endl;
    _propagationRequest_.outputs.emplace_back(OutputRequest::Histograms);
    if( not _propagationRequest_.materializeOutputs.empty()
        and not _propagationRequest_.shouldMaterialize(OutputRequest::Histograms) ){
      _propagationRequest_.materializeOutputs.emplace_back(OutputRequest::Histograms);
    }
  }
  LogInfo << "Propagation backend enabled: " << _backendConfig_.type
          << " with output requests " << toString(_propagationRequest_)
          << std::endl;
  if( not _propagationRequest_.materializeOutputs.empty() ){
    PropagationRequest materializationRequest;
    materializationRequest.outputs = _propagationRequest_.materializeOutputs;
    LogInfo << "Propagation backend host materialization requests "
            << toString(materializationRequest) << std::endl;
  }

  _backendRuntimeManager_ = std::make_shared<BackendRuntimeManager>();
  _backendRuntimeManager_->setBackend(makeBackend(_backendConfig_));
  _backendRuntimeManager_->build(model_);
  _backendRuntimeManager_->getBackend()->setLikelihoodModel(_backendLikelihoodModel_);
}
