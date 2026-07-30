#include "BackendsManager.h"

#include "ConfigUtils.h"
#include "Logger.h"

#include <sstream>

namespace {
  Backends::OutputRequest parseOutputRequest(const std::string& outputRequest_) {
    if( outputRequest_ == "EventWeights" or outputRequest_ == "eventWeights" ){
      return Backends::OutputRequest::EventWeights;
    }
    if( outputRequest_ == "Histograms" or outputRequest_ == "histograms" ){
      return Backends::OutputRequest::Histograms;
    }
    if( outputRequest_ == "Likelihood" or outputRequest_ == "likelihood" ){
      return Backends::OutputRequest::Likelihood;
    }
    if( outputRequest_ == "BinIndices" or outputRequest_ == "binIndices" ){
      return Backends::OutputRequest::BinIndices;
    }
    if( outputRequest_ == "ObservableValues" or outputRequest_ == "observableValues" ){
      return Backends::OutputRequest::ObservableValues;
    }
    LogThrow("Unknown backend output request: " << outputRequest_);
    return Backends::OutputRequest::Histograms;
  }
}

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
  _config_.defineFields({
    {"isEnabled", {"enabled"}},
    {"type", {"backend", "name"}},
    {"outputRequests", {"outputs"}},
    {"materializeOutputRequests", {"materializeOutputs", "hostOutputs"}},
  });

  std::vector<std::string> outputRequestNames{"Histograms"};
  std::vector<std::string> materializeOutputRequestNames{};
  _config_.fillValue(_isEnabled_, "isEnabled");
  _config_.fillValue(_type_, "type");
  _config_.fillValue(outputRequestNames, "outputRequests");
  _config_.fillValue(materializeOutputRequestNames, "materializeOutputRequests");
  _config_.printUnusedKeys();

  _outputRequests_.clear();
  _outputRequests_.reserve(outputRequestNames.size());
  for( const auto& outputRequestName : outputRequestNames ){
    _outputRequests_.emplace_back(parseOutputRequest(outputRequestName));
  }

  _materializeOutputRequests_.clear();
  _materializeOutputRequests_.reserve(materializeOutputRequestNames.size());
  for( const auto& outputRequestName : materializeOutputRequestNames ){
    _materializeOutputRequests_.emplace_back(parseOutputRequest(outputRequestName));
  }

  _propagationRequest_ = makePropagationRequest();
}

Backends::PropagationRequest Backends::BackendsManager::makePropagationRequest() const {
  PropagationRequest out;
  out.outputs = _outputRequests_;
  if( out.outputs.empty() ){
    out.outputs.emplace_back(OutputRequest::Histograms);
  }
  out.materializeOutputs = _materializeOutputRequests_;
  return out;
}

void Backends::BackendsManager::setLikelihoodModel(const BackendLikelihoodModel& likelihoodModel_) {
  _backendLikelihoodModel_ = likelihoodModel_;
  if( _backendRuntimeManager_ != nullptr and _backendRuntimeManager_->hasBackend() ){
    _backendRuntimeManager_->getBackend()->setLikelihoodModel(_backendLikelihoodModel_);
  }
}

void Backends::BackendsManager::initializeBackend(const BackendModel& model_) {
  if( not _isEnabled_ ){
    _backendRuntimeManager_ = nullptr;
    return;
  }

  LogInfo << "Initializing propagation backend: " << _type_ << std::endl;

  _propagationRequest_ = makePropagationRequest();
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
  LogInfo << "Propagation backend enabled: " << _type_
          << " with output requests " << toString(_propagationRequest_)
          << std::endl;
  if( not _propagationRequest_.materializeOutputs.empty() ){
    PropagationRequest materializationRequest;
    materializationRequest.outputs = _propagationRequest_.materializeOutputs;
    LogInfo << "Propagation backend host materialization requests "
            << toString(materializationRequest) << std::endl;
  }

  _backendRuntimeManager_ = std::make_shared<BackendRuntimeManager>();
  _backendRuntimeManager_->setBackend(makeBackend(*this));
  _backendRuntimeManager_->build(model_);
  _backendRuntimeManager_->getBackend()->setLikelihoodModel(_backendLikelihoodModel_);
}
