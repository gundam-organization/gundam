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

bool Backends::BackendsManager::willAutoMaterialize(OutputRequest outputRequest_) const {
  return _enableAutoMaterialize_ and shouldMaterialize(outputRequest_);
}

void Backends::BackendsManager::setMaterializeOutputList(std::initializer_list<OutputRequest> materializeOutputList_) {
  setMaterializeOutputList(std::vector<OutputRequest>(materializeOutputList_));
}

void Backends::BackendsManager::materialize(OutputRequest outputRequest_) {
  LogThrowIf(not hasBackend(), "No backend initialized.");
  LogThrowIf(not _lastPropagationToken_.isValid, "No backend propagation token available for materialization.");
  LogThrowIf(_likelihoodInterfacePtr_ == nullptr, "BackendsManager requires a LikelihoodInterface for materialization.");

  auto* backend = _backendRuntimeManager_->getBackend();
  auto status = backend->getStatus(_lastPropagationToken_);
  auto outputState = status.state(outputRequest_);
  LogThrowIf(outputState != OutputState::ReadyOnDevice and outputState != OutputState::ReadyOnHost,
             "Requested backend output is not ready for materialization: " << outputRequest_.toString());

  switch( outputRequest_.value ){
    case OutputRequest::EventWeights:
    case OutputRequest::Histograms:
      if( outputState == OutputState::ReadyOnDevice ){
        _backendRuntimeManager_->materialize(_lastPropagationToken_, outputRequest_);
      }
      return;

    case OutputRequest::SampleLikelihoods:
      LogThrow("BackendsManager::materialize(OutputRequest::SampleLikelihoods) is not implemented yet: LikelihoodInterface has no destination slot yet.");
      return;

    case OutputRequest::StatLikelihood:
      _likelihoodInterfacePtr_->getBuffer().statLikelihood = backend->getLikelihood(_lastPropagationToken_);
      return;
    default:
      LogError << "Unhandled OutputRequest in BackendsManager::materialize: " << outputRequest_.toString() << std::endl;
  }

  LogThrow("Unhandled OutputRequest in BackendsManager::materialize: " << outputRequest_.toString());
}

void Backends::BackendsManager::initializeImpl() {
  if( not _isEnabled_ ){
    _backendRuntimeManager_ = nullptr;
    return;
  }
  LogThrowIf(_likelihoodInterfacePtr_ == nullptr, "BackendsManager requires a LikelihoodInterface before initialize().");

  LogInfo << "Initializing propagation backend: " << _type_ << std::endl;
  _backendLikelihoodModel_ = buildBackendLikelihoodModel(*_likelihoodInterfacePtr_);

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
          const_cast<Propagator&>(_likelihoodInterfacePtr_->getModelPropagator()).getSampleSet(),
          _likelihoodInterfacePtr_->getModelPropagator().getEventDialCache()
      )
  );
  _backendRuntimeManager_->getBackend()->setLikelihoodModel(_backendLikelihoodModel_);
}

std::future<Backends::BackendPropagationResult> Backends::BackendsManager::propagate() {
  LogThrowIf(not hasBackend(), "No backend initialized.");

  Backends::ParameterSnapshot snapshot;
  auto token = _backendRuntimeManager_->requestPropagation(snapshot);
  _lastPropagationToken_ = token;
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

      if( not willAutoMaterialize(outputRequest) ){ continue; }
      materialize(outputRequest);
    }

    return result;
  });
}
