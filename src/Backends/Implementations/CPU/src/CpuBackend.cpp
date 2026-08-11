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
  out.deviceName = "host";
  return out;
}

void Backends::CpuBackend::build(const EngineView& engineView_) {
  _engineView_ = engineView_;
  _lastResult_ = Result();
  initializeThreads();
  initializeDialResponseCache();
  _isBuilt_ = true;
}

Backends::PropagationToken Backends::CpuBackend::requestPropagation(const ParameterSnapshot& parameters_) {

  LogThrowIf(not _isBuilt_, "CpuBackend has not been built.");
  const auto& model = _engineView_.propagation;
  const auto& likelihoodModel = _engineView_.likelihood;
  LogThrowIf(not parameters_.empty() and parameters_.values.size() != model.parameterCount,
             "ParameterSnapshot size mismatch: " << parameters_.values.size()
                                                 << " != " << model.parameterCount);

  resetResult();

  calculateEventWeights(_lastResult_, parameters_);
  _lastResult_.status.eventWeights = OutputState::ReadyOnDevice;

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

  if( output_ == OutputRequest::EventWeights ){
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
    const bool isComplexDial = dial.type == BackendDialType::CompactSpline
                               or dial.type == BackendDialType::UniformSpline
                               or dial.type == BackendDialType::MonotonicSpline
                               or dial.type == BackendDialType::GeneralSpline
                               or dial.type == BackendDialType::Graph;
    if( not isComplexDial or dial.inputCount != 1 or dial.firstInput >= propagation.dialInputs.size() ){
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

void Backends::CpuBackend::calculateEventWeights(Result& result_, const ParameterSnapshot& parameters_) {
  const auto& propagation = _engineView_.propagation;
  result_.eventWeights.resize(propagation.events.size());
  updateCachedDialResponses(parameters_);
  _activeResult_ = &result_;
  _activeParameters_ = &parameters_;
  _threadPool_.runJob("CpuBackend::calculateEventWeights");
  _activeParameters_ = nullptr;
  _activeResult_ = nullptr;
}

void Backends::CpuBackend::updateCachedDialResponses(const ParameterSnapshot& parameters_) {
  _dirtyCachedDialIndices_.clear();
  if( not _isDialResponseCachePrimed_ or _lastParameterValues_.size() != parameters_.values.size() ){
    for( std::uint32_t iDial = 0 ; iDial < _isCachedDial_.size() ; iDial++ ){
      if( _isCachedDial_[iDial] ){ _dirtyCachedDialIndices_.emplace_back(iDial); }
    }
    _lastParameterValues_ = parameters_.values;
    _isDialResponseCachePrimed_ = true;
  }
  else{
    for( std::size_t iParameter = 0 ; iParameter < parameters_.values.size() ; iParameter++ ){
      if( parameters_.values[iParameter] == _lastParameterValues_[iParameter] ){ continue; }
      if( iParameter < _cachedDialsByParameter_.size() ){
        const auto& cachedDials = _cachedDialsByParameter_[iParameter];
        _dirtyCachedDialIndices_.insert(_dirtyCachedDialIndices_.end(), cachedDials.begin(), cachedDials.end());
      }
      _lastParameterValues_[iParameter] = parameters_.values[iParameter];
    }
  }

  if( _dirtyCachedDialIndices_.empty() ){ return; }
  _activeParameters_ = &parameters_;
  _threadPool_.runJob("CpuBackend::updateCachedDialResponses");
  _activeParameters_ = nullptr;
}

void Backends::CpuBackend::updateCachedDialResponsesThread(int iThread_) {
  const auto& propagation = _engineView_.propagation;
  const auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, _threadPool_.getNbThreads(), _dirtyCachedDialIndices_.size()
  );
  const double* parameterValues = _activeParameters_->values.data();
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
    double weight = event.weight.baseWeight;
    for( std::size_t iDial = 0 ; iDial < event.weight.dialCount ; iDial++ ){
      const auto dialIndex = propagation.eventDialIndices[event.weight.firstDial + iDial];
      const auto& dial = propagation.dials[dialIndex];
      weight *= _isCachedDial_[dialIndex]
                ? _cachedDialResponses_[dialIndex]
                : Semantics::evalDialResponse(propagation, dial, *_activeParameters_);
    }
    _activeResult_->eventWeights[event.resultIndex] = weight;
  }
}

void Backends::CpuBackend::calculateHistograms(Result& result_) {
  LogThrowIf(result_.eventWeights.empty(), "CPU backend histogram build requires event weights.");
  const auto& propagation = _engineView_.propagation;
  const int nThreads = _threadPool_.getNbThreads();
  _threadHistogramSums_.resize(nThreads);
  _threadHistogramSumSquares_.resize(nThreads);
  for( int iThread = 0 ; iThread < nThreads ; iThread++ ){
    _threadHistogramSums_[iThread].assign(propagation.totalBins, 0.);
    _threadHistogramSumSquares_[iThread].assign(propagation.totalBins, 0.);
  }

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
    if( event.globalBinIndex < 0 ){ continue; }
    const double weight = _activeResult_->eventWeights[event.resultIndex];
    histSums[event.globalBinIndex] += weight;
    histSumSquares[event.globalBinIndex] += weight * weight;
  }
}

void Backends::CpuBackend::calculateHistogramsFromEvents(Result& result_, const ParameterSnapshot& parameters_) {
  Semantics::calculateHistograms(
      result_.histSums,
      result_.histSumSquares,
      _engineView_.propagation,
      parameters_
  );
}

void Backends::CpuBackend::calculateLikelihood(Result& result_) {
  result_.likelihood = Semantics::calculateLikelihood(
      _engineView_.likelihood,
      result_.histSums,
      result_.histSumSquares
  );
}
