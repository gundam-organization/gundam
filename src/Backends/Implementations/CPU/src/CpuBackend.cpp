#include "CpuBackend.h"

#include "Semantics/BackendHostPropagation.h"
#include "GundamGlobals.h"
#include "Logger.h"

#include <algorithm>

Backends::BackendCapabilities Backends::CpuBackend::getCapabilities() const {
  BackendCapabilities out;
  out.supportsCpu = true;
  out.supportsEventWeights = true;
  out.supportsHistograms = true;
  out.supportsLikelihood = true;
  out.supportsDynamicBinning = true;
  out.supportsObservableTransforms = true;
  out.deviceName = "host";
  return out;
}

void Backends::CpuBackend::build(const EngineView& engineView_) {
  _engineView_ = engineView_;
  _lastResult_ = Result();
  initializeThreads();
  initializeDialResponseCache();
  const auto& model = _engineView_.propagation;
  _eventState_ = EventState();
  if( model.requiresEventBinning() ){
    _eventState_.variables = model.variableValues;
    _eventState_.sampleRevisions.assign(model.samples.size(), 1);
    _eventState_.bins.resize(model.events.size(), -1);
    _eventState_.contributions.resize(model.events.size());
  }
  _validBinning_.assign(model.events.size(), false);
  _binningIndices_.resize(model.samples.size());
  for( std::size_t i = 0; i < model.samples.size(); ++i ){ _binningIndices_[i].build(model.samples[i].binning); }
  _binningWorkers_.assign(_threadPool_.getNbThreads(), {});
  _workerErrors_.resize(_threadPool_.getNbThreads());
  _variableDialDependencies_.assign(model.dials.size(), false);
  for( auto dial : model.variableDialIndices ){ _variableDialDependencies_.at(dial) = true; }
  for( const auto& target : model.variableResponses ){
    if( target.overrideDial != std::size_t(-1) ){ _variableDialDependencies_.at(target.overrideDial) = true; }
  }
  _lastBinningParameters_.clear();
  _lastExternalGenerations_.clear();
  _lastHostVariables_.clear();
  _lastDialRevisions_.clear();
  _eventStateValid_ = false;
  _cachedContributionCount_ = 0;
  _isBuilt_ = true;
}

Backends::PropagationToken Backends::CpuBackend::requestPropagation(const PropagationInputs& inputs_) {

  LogThrowIf(not _isBuilt_, "CpuBackend has not been built.");
  const auto& model = _engineView_.propagation;
  const auto& likelihoodModel = _engineView_.likelihood;
  inputs_.validate(model);

  resetResult();
  _lastTiming_ = BackendTimingSummary();

  calculateEventWeights(_lastResult_, inputs_);
  _lastResult_.status.eventWeights = OutputState::ReadyOnDevice;

  if( model.requiresEventBinning() ){
    try{ updateEventState(inputs_); }
    catch( ... ){
      // A failed parallel pass may have touched only part of the state.
      // Retry from host/reference values rather than publishing partial caches.
      _eventStateValid_ = false;
      std::fill(_validBinning_.begin(), _validBinning_.end(), false);
      _lastHostVariables_.clear();
      _activeInputs_ = nullptr;
      _lastResult_.status.backend = BackendStatus::Failed;
      _lastResult_.token.isValid = false;
      throw;
    }
    _lastResult_.status.eventState = OutputState::ReadyOnDevice;
  }

  calculateHistograms(_lastResult_);
  _lastResult_.status.histograms = OutputState::ReadyOnDevice;

  _lastResult_.status.sampleLikelihoods = OutputState::Failed;
  if( likelihoodModel.empty() ){
    _lastResult_.status.statLikelihood = OutputState::Failed;
  }
  else{
    calculateLikelihood(_lastResult_);
    _lastResult_.status.statLikelihood = OutputState::ReadyOnDevice;
  }
  _lastResult_.status.backend = BackendStatus::Ready;

  _lastTiming_.cachedDialStageSeconds = _cachedDialStageTimer_.eval().count();
  _lastTiming_.eventWeightsStageSeconds = _eventWeightsStageTimer_.eval().count();
  _lastTiming_.histogramStageSeconds = _histogramStageTimer_.eval().count();
  _lastTiming_.likelihoodHostSeconds = _likelihoodHostTimer_.eval().count();
  _lastTiming_.uniqueDialCount = model.dials.size();
  _lastTiming_.cachedDialCount = std::count(_isCachedDial_.begin(), _isCachedDial_.end(), std::uint8_t(true));
  _lastTiming_.eventDialIndexCount = model.eventDialIndices.size();
  _lastTiming_.splineScalarCount = model.dialPayloads.size();

  return _lastResult_.token;
}

Backends::PropagationStatus Backends::CpuBackend::getStatus(const PropagationToken& token_) const {
  if( not isCurrentToken(token_) ){
    PropagationStatus out;
    out.backend = BackendStatus::Failed;
    return out;
  }
  return _lastResult_.status;
}

bool Backends::CpuBackend::isReady(const PropagationToken& token_) const {
  return isCurrentToken(token_) and _lastResult_.status.backend == BackendStatus::Ready;
}

void Backends::CpuBackend::wait(const PropagationToken& token_) {
  LogThrowIf(not isCurrentToken(token_), "Invalid CpuBackend propagation token.");
}

void Backends::CpuBackend::materialize(const PropagationToken& token_, OutputRequest output_) {
  LogThrowIf(not isCurrentToken(token_), "Invalid CpuBackend propagation token.");
  LogThrowIf(_lastResult_.status.state(output_) != OutputState::ReadyOnDevice
             and _lastResult_.status.state(output_) != OutputState::ReadyOnHost,
             "Requested backend output is not ready.");

  if( output_ == OutputRequest::EventState ){
    _lastResult_.status.eventState = OutputState::ReadyOnHost;
  }
  else if( output_ == OutputRequest::EventWeights ){
    _lastResult_.status.eventWeights = OutputState::ReadyOnHost;
  }
  else if( output_ == OutputRequest::Histograms ){
    _lastResult_.status.histograms = OutputState::ReadyOnHost;
  }
  else if( output_ == OutputRequest::SampleLikelihoods ){
    LogThrow("CpuBackend cannot materialize sample likelihoods yet.");
  }
  else if( output_ == OutputRequest::StatLikelihood ){
    _lastResult_.status.statLikelihood = OutputState::ReadyOnHost;
  }
  else{
    LogThrow("CpuBackend cannot materialize requested output yet.");
  }
}

double Backends::CpuBackend::getLikelihood(const PropagationToken& token_) const {
  LogThrowIf(not isCurrentToken(token_), "Invalid CpuBackend propagation token.");
  LogThrowIf(_lastResult_.status.statLikelihood != OutputState::ReadyOnDevice
             and _lastResult_.status.statLikelihood != OutputState::ReadyOnHost,
             "Backend likelihood is not ready.");
  return _lastResult_.likelihood;
}

const std::vector<double>& Backends::CpuBackend::getEventWeightsHostView(const PropagationToken& token_) const {
  LogThrowIf(not isCurrentToken(token_), "Invalid CpuBackend propagation token.");
  return _lastResult_.eventWeights;
}

const std::vector<double>& Backends::CpuBackend::getHistogramSumsHostView(const PropagationToken& token_) const {
  LogThrowIf(not isCurrentToken(token_), "Invalid CpuBackend propagation token.");
  return _lastResult_.histSums;
}

const std::vector<double>& Backends::CpuBackend::getHistogramSumSquaresHostView(const PropagationToken& token_) const {
  LogThrowIf(not isCurrentToken(token_), "Invalid CpuBackend propagation token.");
  return _lastResult_.histSumSquares;
}

bool Backends::CpuBackend::isCurrentToken(const PropagationToken& token_) const {
  return token_.isValid and _lastResult_.token.isValid and token_.id == _lastResult_.token.id;
}

void Backends::CpuBackend::initializeThreads() {
  if( _threadPool_.getJobPtr("CpuBackend::calculateEventWeights") != nullptr ){ return; }

  _threadPool_.setNThreads(std::max(1, GundamGlobals::getNbCpuThreads()));
  _threadPool_.setCpuTimeSaverIsEnabled(false);
  _threadPool_.addJob(
      "CpuBackend::calculateEventWeights",
      [this](int iThread_){ calculateEventWeightsThread(iThread_); }
  );
  _threadPool_.addJob(
      "CpuBackend::calculateHistograms",
      [this](int iThread_){ calculateHistogramsThread(iThread_); }
  );
  // ParallelWorker does not forward worker exceptions. Capture them so every
  // thread can finish its job before an error is rethrown on the caller.
  _threadPool_.addJob("CpuBackend::updateEventVariables", [this](int iThread_){
    try{ updateEventVariablesThread(iThread_); }
    catch( ... ){ _workerErrors_[iThread_] = std::current_exception(); }
  });
  _threadPool_.addJob("CpuBackend::updateEventBinning", [this](int iThread_){
    try{ updateEventBinningThread(iThread_); }
    catch( ... ){ _workerErrors_[iThread_] = std::current_exception(); }
  });
  _threadPool_.addJob(
      "CpuBackend::updateCachedDialResponses",
      [this](int iThread_){ updateCachedDialResponsesThread(iThread_); }
  );
}

void Backends::CpuBackend::initializeDialResponseCache() {
  const auto& propagation = _engineView_.propagation;
  _cachedDialsByParameter_.assign(propagation.parameterCount, {});
  _cachedDialResponses_.assign(propagation.dials.size(), 1.);
  _cachedDialInputs_.assign(propagation.dials.size(), 0.);
  _isCachedDial_.assign(propagation.dials.size(), false);
  _isCachedDialResponseValid_.assign(propagation.dials.size(), false);
  _lastParameterValues_.clear();
  _isDialResponseCachePrimed_ = false;

  for( std::uint32_t iDial = 0 ; iDial < propagation.dials.size() ; iDial++ ){
    const auto& dial = propagation.dials[iDial];
    const bool canCacheSingleInputResponse = dial.type != BackendDialType::Shift
                                             and dial.type != BackendDialType::ExternalWeight
                                             and dial.inputCount == 1
                                             and dial.firstInput < propagation.dialInputs.size();
    if( not canCacheSingleInputResponse ){
      continue;
    }

    const auto parameterIndex = propagation.dialInputs[dial.firstInput].parameterIndex;
    if( parameterIndex >= propagation.parameterCount ){ continue; }
    _isCachedDial_[iDial] = true;
    _cachedDialsByParameter_[parameterIndex].emplace_back(iDial);
  }
}

void Backends::CpuBackend::resetResult() {
  _lastResult_.token.id = _nextTokenId_++;
  _lastResult_.token.isValid = true;
  _lastResult_.status = PropagationStatus();
  _lastResult_.status.backend = BackendStatus::Running;
  _lastResult_.eventWeights.clear();
  _lastResult_.histSums.clear();
  _lastResult_.histSumSquares.clear();
  _lastResult_.likelihood = 0;
  _lastResult_.status.eventWeights = OutputState::Scheduled;
  _lastResult_.status.histograms = OutputState::Scheduled;
  _lastResult_.status.sampleLikelihoods = OutputState::Scheduled;
  _lastResult_.status.statLikelihood = OutputState::Scheduled;
}

void Backends::CpuBackend::calculateEventWeights(Result& result_, const PropagationInputs& inputs_) {
  const auto& propagation = _engineView_.propagation;
  result_.eventWeights.resize(propagation.events.size());
  updateCachedDialResponses(inputs_);
  _activeResult_ = &result_;
  _activeInputs_ = &inputs_;
  {
    auto timerScope = _eventWeightsStageTimer_.scopeTime();
    _threadPool_.runJob("CpuBackend::calculateEventWeights");
  }
  _activeInputs_ = nullptr;
  _activeResult_ = nullptr;
}

void Backends::CpuBackend::updateCachedDialResponses(const PropagationInputs& inputs_) {
  auto timerScope = _cachedDialStageTimer_.scopeTime();

  _dirtyCachedDialIndices_.clear();
  if( not _isDialResponseCachePrimed_ or _lastParameterValues_.size() != inputs_.parameters.values.size() ){
    for( std::uint32_t iDial = 0 ; iDial < _isCachedDial_.size() ; iDial++ ){
      if( _isCachedDial_[iDial] ){ _dirtyCachedDialIndices_.emplace_back(iDial); }
    }
    _lastParameterValues_ = inputs_.parameters.values;
    _isDialResponseCachePrimed_ = true;
  }
  else{
    for( std::size_t iParameter = 0 ; iParameter < inputs_.parameters.values.size() ; iParameter++ ){
      if( inputs_.parameters.values[iParameter] == _lastParameterValues_[iParameter] ){ continue; }
      if( iParameter < _cachedDialsByParameter_.size() ){
        const auto& cachedDials = _cachedDialsByParameter_[iParameter];
        _dirtyCachedDialIndices_.insert(_dirtyCachedDialIndices_.end(), cachedDials.begin(), cachedDials.end());
      }
      _lastParameterValues_[iParameter] = inputs_.parameters.values[iParameter];
    }
  }

  if( _dirtyCachedDialIndices_.empty() ){ return; }
  _activeInputs_ = &inputs_;
  _threadPool_.runJob("CpuBackend::updateCachedDialResponses");
  _activeInputs_ = nullptr;
}

void Backends::CpuBackend::updateCachedDialResponsesThread(int iThread_) {
  const auto& propagation = _engineView_.propagation;
  const auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, _threadPool_.getNbThreads(), _dirtyCachedDialIndices_.size()
  );
  const double* parameterValues = _activeInputs_->parameters.values.data();
  for( std::size_t iDirtyDial = bounds.beginIndex ; iDirtyDial < bounds.endIndex ; iDirtyDial++ ){
    const auto dialIndex = _dirtyCachedDialIndices_[iDirtyDial];
    const auto& dial = propagation.dials[dialIndex];
    const auto& input = propagation.dialInputs[dial.firstInput];
    const double inputValue = Semantics::transformDialInput(
        input, Semantics::loadParameterValue(input, parameterValues)
    );
    if( _isCachedDialResponseValid_[dialIndex] and inputValue == _cachedDialInputs_[dialIndex] ){
      continue;
    }
    const auto* payload = Semantics::getDialPayload(propagation.dialPayloads.data(), dial);
    _cachedDialResponses_[dialIndex] = Semantics::evalDialResponseFromInput(dial, inputValue, payload);
    _cachedDialInputs_[dialIndex] = inputValue;
    _isCachedDialResponseValid_[dialIndex] = true;
  }
}

void Backends::CpuBackend::calculateEventWeightsThread(int iThread_) {
  const auto& propagation = _engineView_.propagation;
  const auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, _threadPool_.getNbThreads(), propagation.events.size()
  );
  for( std::size_t iEvent = bounds.beginIndex ; iEvent < bounds.endIndex ; iEvent++ ){
    const auto& event = propagation.events[iEvent];
    double weight = 1.;
    for( std::size_t iDial = 0 ; iDial < event.weight.dialCount ; iDial++ ){
      const auto dialIndex = propagation.eventDialIndices[event.weight.firstDial + iDial];
      const auto& dial = propagation.dials[dialIndex];
      weight *= _isCachedDial_[dialIndex]
                ? _cachedDialResponses_[dialIndex]
                : Semantics::evalDialResponse(propagation, dial, *_activeInputs_);
    }
    if( propagation.hasReweightCap ){ weight = std::min(weight, propagation.maxReweight); }
    _activeResult_->eventWeights[event.resultIndex] = event.weight.baseWeight * weight;
  }
}

void Backends::CpuBackend::calculateHistograms(Result& result_) {
  auto timerScope = _histogramStageTimer_.scopeTime();

  const auto& propagation = _engineView_.propagation;
  const int nThreads = _threadPool_.getNbThreads();
  _threadHistogramSums_.resize(nThreads);
  _threadHistogramSumSquares_.resize(nThreads);
  for( int iThread = 0 ; iThread < nThreads ; iThread++ ){
    _threadHistogramSums_[iThread].assign(propagation.totalBins + propagation.samples.size(), 0.);
    _threadHistogramSumSquares_[iThread].assign(propagation.totalBins + propagation.samples.size(), 0.);
  }

  result_.outsideSums.assign(propagation.samples.size(), 0.);
  result_.outsideSumSquares.assign(propagation.samples.size(), 0.);
  result_.histSums.assign(propagation.totalBins, 0.);
  result_.histSumSquares.assign(propagation.totalBins, 0.);
  _activeResult_ = &result_;
  _threadPool_.runJob("CpuBackend::calculateHistograms");
  _activeResult_ = nullptr;

  for( int iBin = 0 ; iBin < propagation.totalBins ; iBin++ ){
    for( int iThread = 0 ; iThread < nThreads ; iThread++ ){
      result_.histSums[iBin] += _threadHistogramSums_[iThread][iBin];
      result_.histSumSquares[iBin] += _threadHistogramSumSquares_[iThread][iBin];
    }
  }
  for( std::size_t iSample = 0; iSample < propagation.samples.size(); ++iSample ){
    for( int iThread = 0; iThread < nThreads; ++iThread ){
      result_.outsideSums[iSample] += _threadHistogramSums_[iThread][propagation.totalBins + iSample];
      result_.outsideSumSquares[iSample] += _threadHistogramSumSquares_[iThread][propagation.totalBins + iSample];
    }
  }
}

void Backends::CpuBackend::calculateHistogramsThread(int iThread_) {
  const auto& propagation = _engineView_.propagation;
  const auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, _threadPool_.getNbThreads(), propagation.events.size()
  );
  auto& histSums = _threadHistogramSums_[iThread_];
  auto& histSumSquares = _threadHistogramSumSquares_[iThread_];
  for( std::size_t iEvent = bounds.beginIndex ; iEvent < bounds.endIndex ; iEvent++ ){
    const auto& event = propagation.events[iEvent];
    const double weight = _activeResult_->eventWeights[event.resultIndex];
    if( event.samplePosition < propagation.samples.size() ){
      const auto& sample = propagation.samples[event.samplePosition];
      if( sample.useDynamicBinning or sample.binning.isSmoothed ){
        for( const auto& contribution : _eventState_.contributions[event.resultIndex] ){
          const std::size_t bin = contribution.binIndex < 0 ? propagation.totalBins + event.samplePosition
              : sample.binOffset + contribution.binIndex;
          double value = weight * contribution.fraction;
          histSums[bin] += value;
          histSumSquares[bin] += value * value;
        }
        continue;
      }
    }
    if( event.globalBinIndex < 0 ){ continue; }
    histSums[event.globalBinIndex] += weight;
    histSumSquares[event.globalBinIndex] += weight * weight;
  }
}

void Backends::CpuBackend::calculateHistogramsFromEvents(Result& result_, const PropagationInputs& inputs_) {
  Semantics::calculateHistograms(
      result_.histSums,
      result_.histSumSquares,
      _engineView_.propagation,
      inputs_
  );
}

void Backends::CpuBackend::calculateLikelihood(Result& result_) {
  auto timerScope = _likelihoodHostTimer_.scopeTime();

  result_.likelihood = Semantics::calculateLikelihood(
      _engineView_.likelihood,
      result_.histSums,
      result_.histSumSquares
  );
}

const Backends::EventState& Backends::CpuBackend::getEventStateHostView(const PropagationToken& token_) const {
  LogThrowIf(not isCurrentToken(token_), "Invalid CpuBackend propagation token.");
  return _eventState_;
}

const std::vector<double>& Backends::CpuBackend::getOutsideSumsHostView(const PropagationToken& token_) const {
  LogThrowIf(not isCurrentToken(token_), "Invalid CpuBackend propagation token.");
  return _lastResult_.outsideSums;
}

const std::vector<double>& Backends::CpuBackend::getOutsideSumSquaresHostView(const PropagationToken& token_) const {
  LogThrowIf(not isCurrentToken(token_), "Invalid CpuBackend propagation token.");
  return _lastResult_.outsideSumSquares;
}
