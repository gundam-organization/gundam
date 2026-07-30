#include "BackendsManager.h"

#include "BackendModelBuilder.h"
#include "LikelihoodInterface.h"
#include "Logger.h"
#include "Propagator.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace {
  Backends::BackendLikelihoodModel buildBackendLikelihoodModel(const LikelihoodInterface& likelihoodInterface_){
    Backends::BackendLikelihoodModel out;
    out.samples.reserve(likelihoodInterface_.getSamplePairList().size());

    int binOffset{0};
    for( const auto& samplePair : likelihoodInterface_.getSamplePairList() ){
      Backends::BackendLikelihoodSampleRef sampleRef;
      sampleRef.binOffset = binOffset;
      sampleRef.dataSums.reserve(samplePair.data->getHistogram().getNbBins());
      sampleRef.ignoredBins.reserve(samplePair.data->getHistogram().getNbBins());

      const auto& dataBinContentList = samplePair.data->getHistogram().getBinContentList();
      const auto& modelBinContentList = samplePair.model->getHistogram().getBinContentList();
      for( int iBin = 0 ; iBin < samplePair.data->getHistogram().getNbBins() ; iBin++ ){
        sampleRef.dataSums.emplace_back(dataBinContentList[iBin].sumWeights);
        sampleRef.ignoredBins.emplace_back(
            likelihoodInterface_.getJointProbabilityPtr()->isIgnoreBinsWithZeroPredictionAtPrior()
            and modelBinContentList[iBin].sumWeights == 0.
        );
      }

      auto* jointProbability = likelihoodInterface_.getJointProbabilityPtr();
      sampleRef.evalBin = [jointProbability](double data_, double pred_, double err_, int bin_){
        return jointProbability->eval(data_, pred_, err_, bin_);
      };

      out.samples.emplace_back(std::move(sampleRef));
      binOffset += samplePair.model->getHistogram().getNbBins();
    }

    return out;
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
    {"isEnabled"},
    {"type"},
    {"outputRequests"},
  });

  _config_.fillValue(_isEnabled_, "isEnabled");
  _config_.fillValue(_type_, "type");
  if( _config_.hasField("outputRequests") ){
    _outputRequests_.clear();
    for( const auto& outputRequestEntry : _config_.loop("outputRequests") ){
      _outputRequests_.emplace_back(OutputRequest::toEnum(outputRequestEntry.toString(), true));
    }
  }
  _config_.printUnusedKeys();

  _propagationRequest_ = makePropagationRequest();
}

Backends::PropagationRequest Backends::BackendsManager::makePropagationRequest() const {
  PropagationRequest out;
  out.outputs = _outputRequests_;
  if( out.outputs.empty() ){
    out.outputs.emplace_back(OutputRequest::Histograms);
  }

  if( _enableAutoMaterialize_ ){
    out.materializeOutputs.reserve(out.outputs.size());
    for( auto outputRequest : out.outputs ){
      if( std::find(_materializeOutputList_.begin(), _materializeOutputList_.end(), outputRequest) != _materializeOutputList_.end() ){
        out.materializeOutputs.emplace_back(outputRequest);
      }
    }
  }

  return out;
}

void Backends::BackendsManager::setEnableAutoMaterialize(bool enableAutoMaterialize_) {
  _enableAutoMaterialize_ = enableAutoMaterialize_;
  _propagationRequest_ = makePropagationRequest();
}

void Backends::BackendsManager::setMaterializeOutputList(std::vector<OutputRequest> materializeOutputList_) {
  _materializeOutputList_ = std::move(materializeOutputList_);
  _propagationRequest_ = makePropagationRequest();
}

void Backends::BackendsManager::setMaterializeOutputList(std::initializer_list<OutputRequest> materializeOutputList_) {
  setMaterializeOutputList(std::vector<OutputRequest>(materializeOutputList_));
}

void Backends::BackendsManager::initializeBackend(const LikelihoodInterface& likelihoodInterface_) {
  if( not _isEnabled_ ){
    _backendRuntimeManager_ = nullptr;
    return;
  }

  LogInfo << "Initializing propagation backend: " << _type_ << std::endl;

  _backendLikelihoodModel_ = buildBackendLikelihoodModel(likelihoodInterface_);
  _propagationRequest_ = makePropagationRequest();
  if( not _propagationRequest_.has(OutputRequest::Histograms)
      and not _propagationRequest_.has(OutputRequest::Likelihood) ){
    LogWarning << "Adding OutputRequest::Histograms to backend configuration because the standard engine path consumes CPU histograms." << std::endl;
    _propagationRequest_.outputs.emplace_back(OutputRequest::Histograms);
    if( _enableAutoMaterialize_
        and std::find(_materializeOutputList_.begin(), _materializeOutputList_.end(), OutputRequest::Histograms) != _materializeOutputList_.end()
        and not _propagationRequest_.shouldMaterialize(OutputRequest::Histograms) ){
      _propagationRequest_.materializeOutputs.emplace_back(OutputRequest::Histograms);
    }
  }

  LogInfo << "Propagation backend enabled: " << _type_
          << " with output requests " << toString(_propagationRequest_)
          << std::endl;
  if( _enableAutoMaterialize_ and not _propagationRequest_.materializeOutputs.empty() ){
    PropagationRequest materializationRequest;
    materializationRequest.outputs = _propagationRequest_.materializeOutputs;
    LogInfo << "Propagation backend auto materialization requests "
            << toString(materializationRequest) << std::endl;
  }

  _backendRuntimeManager_ = std::make_shared<BackendRuntimeManager>();
  _backendRuntimeManager_->setBackend(makeBackend(*this));
  _backendRuntimeManager_->build(
      BackendModelBuilder::build(
          const_cast<Propagator&>(likelihoodInterface_.getModelPropagator()).getSampleSet(),
          likelihoodInterface_.getModelPropagator().getEventDialCache()
      )
  );
  _backendRuntimeManager_->getBackend()->setLikelihoodModel(_backendLikelihoodModel_);
}

std::future<Backends::BackendPropagationResult> Backends::BackendsManager::propagate(Propagator& propagator_) {
  LogThrowIf(not hasBackend(), "No backend initialized.");

  _propagationRequest_ = makePropagationRequest();

  if( not _propagationRequest_.has(OutputRequest::Histograms)
      and not _propagationRequest_.has(OutputRequest::Likelihood) ){
    LogWarning << "Backend propagation requested without Histograms or Likelihood. The standard engine path may not observe any updated prediction." << std::endl;
  }

  Backends::ParameterSnapshot snapshot;
  auto token = _backendRuntimeManager_->requestPropagation(snapshot, _propagationRequest_);
  if( not token.isValid ){
    return std::async(std::launch::deferred, []{
      return BackendPropagationResult{};
    });
  }

  return std::async(std::launch::deferred, [this, &propagator_, token]{
    BackendPropagationResult result;
    auto* backendRuntimeManager = getBackendRuntimeManager();
    backendRuntimeManager->wait(token);

    auto status = backendRuntimeManager->getBackend()->getStatus(token);
    for( auto outputRequest : _propagationRequest_.outputs ){
      auto outputState = status.state(outputRequest);
      if( outputState == OutputState::Failed ){
        LogWarning << "Requested backend output failed or is not implemented yet. Skipping materialization." << std::endl;
        continue;
      }
      if( outputState != OutputState::ReadyOnDevice and outputState != OutputState::ReadyOnHost ){
        LogWarning << "Requested backend output is not ready. Skipping materialization." << std::endl;
        continue;
      }

      result.isValid = true;
      if( outputRequest == OutputRequest::Likelihood ){
        result.statLikelihood = backendRuntimeManager->getBackend()->getLikelihood(token);
        result.hasStatLikelihood = true;
      }

      if( not _propagationRequest_.shouldMaterialize(outputRequest) ){ continue; }
      backendRuntimeManager->materialize(token, outputRequest);
    }

    if( propagator_.getSampleSet().getSampleList().empty() ){
      result.isValid = false;
    }

    return result;
  });
}
