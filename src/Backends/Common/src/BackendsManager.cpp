#include "BackendsManager.h"

#include "BackendModelBuilder.h"
#include "LikelihoodInterface.h"
#include "Logger.h"

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
  });

  _config_.fillValue(_isEnabled_, "isEnabled");
  _config_.fillValue(_type_, "type");
  _config_.printUnusedKeys();
}

bool Backends::BackendsManager::shouldMaterialize(OutputRequest outputRequest_) const {
  return std::find(_materializeOutputList_.begin(), _materializeOutputList_.end(), outputRequest_) != _materializeOutputList_.end();
}

void Backends::BackendsManager::setEnableAutoMaterialize(bool enableAutoMaterialize_) {
  _enableAutoMaterialize_ = enableAutoMaterialize_;
}

void Backends::BackendsManager::setMaterializeOutputList(std::vector<OutputRequest> materializeOutputList_) {
  _materializeOutputList_ = std::move(materializeOutputList_);
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

  LogInfo << "Propagation backend enabled: " << _type_
          << " with fixed device outputs [EventWeights, Histograms, StatLikelihood]"
          << std::endl;
  if( _enableAutoMaterialize_ and not _materializeOutputList_.empty() ){
    LogInfo << "Propagation backend auto materialization requests "
            << GenericToolbox::toString(_materializeOutputList_) << std::endl;
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

std::future<Backends::BackendPropagationResult> Backends::BackendsManager::propagate(Propagator&) {
  LogThrowIf(not hasBackend(), "No backend initialized.");

  Backends::ParameterSnapshot snapshot;
  auto token = _backendRuntimeManager_->requestPropagation(snapshot);
  if( not token.isValid ){
    return std::async(std::launch::deferred, []{
      return BackendPropagationResult{};
    });
  }

  return std::async(std::launch::deferred, [this, token]{
    BackendPropagationResult result;
    auto* backendRuntimeManager = getBackendRuntimeManager();
    backendRuntimeManager->wait(token);

    auto status = backendRuntimeManager->getBackend()->getStatus(token);
    for( auto outputRequest : {OutputRequest::EventWeights, OutputRequest::Histograms, OutputRequest::SampleLikelihoods, OutputRequest::StatLikelihood} ){
      auto outputState = status.state(outputRequest);
      if( outputState == OutputState::Failed ){ continue; }
      if( outputState != OutputState::ReadyOnDevice and outputState != OutputState::ReadyOnHost ){
        continue;
      }

      result.isValid = true;
      if( outputRequest == OutputRequest::StatLikelihood ){
        result.statLikelihood = backendRuntimeManager->getBackend()->getLikelihood(token);
        result.hasStatLikelihood = true;
      }

      if( not _enableAutoMaterialize_ or not shouldMaterialize(outputRequest) ){ continue; }
      backendRuntimeManager->materialize(token, outputRequest);
    }

    return result;
  });
}
