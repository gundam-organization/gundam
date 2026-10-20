#include "CpuBackend.h"
#include "CpuWeightKernels.h"

#include "Semantics/BackendHostPropagation.h"
#include "GundamGlobals.h"
#include "Logger.h"

#include <algorithm>
#include <chrono>
#include <type_traits>
#include <utility>

void Backends::CpuBackend::configureImpl() {
  _config_.clearFields();
  _config_.defineFields({{"profileOutputPath"}, {"profileSampleInterval"}, {"profileMaxSamples"}});
  _config_.checkConfiguration();
  _config_.fillValue(_profileOutputPath_, "profileOutputPath");
  _config_.fillValue(_profileSampleInterval_, "profileSampleInterval");
  _config_.fillValue(_profileMaxSamples_, "profileMaxSamples");
  LogThrowIf(_profileSampleInterval_ <= 0 or _profileMaxSamples_ <= 0,
             "CPU profiling interval and maximum sample count must be strictly positive.");
}

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
  buildShared(std::make_shared<EngineView>(engineView_));
}

void Backends::CpuBackend::buildShared(std::shared_ptr<const EngineView> engineView_) {
  _isBuilt_ = false;
  LogThrowIf(engineView_ == nullptr, "CpuBackend requires a model.");
  _engineView_ = std::move(engineView_);
  LogThrowIf(_engineView_->propagation.eventWeights.size() != _engineView_->propagation.events.size(),
             "CpuBackend event weight layout size mismatch.");
  _lastResult_ = Result();
  initializeThreads();
  initializeDialResponseCache();
  initializeWeightExecution();
  _timingTotals_ = BackendTimingTotals();
  const auto& model = _engineView_->propagation;
  _eventState_ = EventState();
  _requiresEventBinning_ = model.requiresEventBinning();
  if( _requiresEventBinning_ ){
    _eventState_.variables = model.variableValues;
    _eventState_.sampleRevisions.assign(model.samples.size(), 1);
    _eventState_.bins.resize(model.events.size(), -1);
    _eventState_.contributions.resize(model.events.size());
  }
  _validBinning_.assign(_requiresEventBinning_ ? model.events.size() : 0, false);
  _binningIndices_.resize(model.samples.size());
  for( std::size_t i = 0; i < model.samples.size(); ++i ){ _binningIndices_[i].build(model.samples[i].binning); }
  _binningWorkers_.assign(_threadPool_.getNbThreads(), {});
  _workerErrors_.resize(_threadPool_.getNbThreads());
  _eventStateCache_.build(model);
  _cachedContributionCount_ = 0;
  initializeProfiler();
  _isBuilt_ = true;
}

Backends::PropagationToken Backends::CpuBackend::requestPropagation(const PropagationInputs& inputs_) {

  LogThrowIf(not _isBuilt_, "CpuBackend has not been built.");
  const auto propagationStart = std::chrono::steady_clock::now();
  const auto& model = _engineView_->propagation;
  const auto& likelihoodModel = _engineView_->likelihood;
  inputs_.validate(model);

  if( _profiler_ ){
    _profiler_->begin(_timingTotals_.propagationCount + 1, inputs_.parameters.values,
                      _lastParameterValues_, _isDialResponseCachePrimed_);
  }

  resetResult();
  _lastTiming_ = BackendTimingSummary();

  calculateEventWeights(_lastResult_, inputs_);
  _lastResult_.status.eventWeights = OutputState::ReadyOnDevice;

  if( _requiresEventBinning_ ){
    try{ updateEventState(inputs_); }
    catch( ... ){
      // A failed parallel pass may have touched only part of the state.
      // Retry from host/reference values rather than publishing partial caches.
      _eventStateCache_.invalidate();
      std::fill(_validBinning_.begin(), _validBinning_.end(), false);
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
  _lastTiming_.likelihoodHostSeconds = likelihoodModel.empty() ? 0. : _likelihoodHostTimer_.eval().count();
  _lastTiming_.uniqueDialCount = model.dials.size();
  _lastTiming_.cachedDialCount = _cachedDialCount_;
  _lastTiming_.eventDialIndexCount = model.eventDialIndices.size();
  _lastTiming_.splineScalarCount = model.dialPayloads.size();

  ++_timingTotals_.propagationCount;
  _timingTotals_.propagationSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - propagationStart).count();
  _timingTotals_.cachedDialSeconds += _lastTiming_.cachedDialStageSeconds;
  _timingTotals_.eventWeightsSeconds += _lastTiming_.eventWeightsStageSeconds;
  _timingTotals_.histogramSeconds += _lastTiming_.histogramStageSeconds;
  _timingTotals_.likelihoodSeconds += _lastTiming_.likelihoodHostSeconds;

  if( _profiler_ ){ _profiler_->end(_lastTiming_); }

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
      [this](int iThread_){
        CpuBackendProfiler::WorkerScope probe(_profiler_.get(), CpuBackendProfiler::Stage::Weights, iThread_);
        calculateEventWeightsThread(iThread_);
      }
  );
  _threadPool_.addJob(
      "CpuBackend::calculateHistograms",
      [this](int iThread_){
        CpuBackendProfiler::WorkerScope probe(_profiler_.get(), CpuBackendProfiler::Stage::Histograms, iThread_);
        calculateHistogramsThread(iThread_);
      }
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
  const auto& propagation = _engineView_->propagation;
  _cachedDialsByParameter_.assign(propagation.parameterCount, {});
  _cachedDialResponses_.assign(propagation.dials.size(), 1.);
  _cachedDialInputs_.assign(propagation.dials.size(), 0.);
  _isCachedDial_.assign(propagation.dials.size(), false);
  _isCachedDialResponseValid_.assign(propagation.dials.size(), false);
  _lastParameterValues_.clear();
  _isDialResponseCachePrimed_ = false;

  for( std::uint32_t iDial = 0 ; iDial < propagation.dials.size() ; iDial++ ){
    const auto& dial = propagation.dials[iDial];
    if( dial.type == BackendDialType::Shift ){
      // Shift responses depend only on immutable payload and response bounds.
      // Keep their product position, but evaluate them only once per build.
      LogThrowIf(dial.payloadSize == 0 or dial.payloadOffset >= propagation.dialPayloads.size(),
                 "Invalid CPU constant dial payload.");
      _isCachedDial_[iDial] = true;
      _cachedDialResponses_[iDial] = Semantics::evalDialResponseFromInput(
          dial, 0., Semantics::getDialPayload(propagation.dialPayloads.data(), dial));
      _isCachedDialResponseValid_[iDial] = true;
      continue;
    }
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
  _lastResult_.histSums.clear();
  _lastResult_.histSumSquares.clear();
  _lastResult_.likelihood = 0;
  _lastResult_.status.eventWeights = OutputState::Scheduled;
  _lastResult_.status.histograms = OutputState::Scheduled;
  _lastResult_.status.sampleLikelihoods = OutputState::Scheduled;
  _lastResult_.status.statLikelihood = OutputState::Scheduled;
}

void Backends::CpuBackend::calculateEventWeights(Result& result_, const PropagationInputs& inputs_) {
  const auto& propagation = _engineView_->propagation;
  result_.eventWeights.resize(propagation.events.size());
  if( _profiler_ ){ _profiler_->beginStage(CpuBackendProfiler::Stage::Cache); }
  updateCachedDialResponses(inputs_);
  if( _profiler_ ){ _profiler_->endStage(CpuBackendProfiler::Stage::Cache); }
  _activeResult_ = &result_;
  _activeInputs_ = &inputs_;
  {
    auto timerScope = _eventWeightsStageTimer_.scopeTime();
    if( _profiler_ ){ _profiler_->beginStage(CpuBackendProfiler::Stage::Weights); }
    _threadPool_.runJob("CpuBackend::calculateEventWeights");
    if( _profiler_ ){ _profiler_->endStage(CpuBackendProfiler::Stage::Weights); }
  }
  _activeInputs_ = nullptr;
  _activeResult_ = nullptr;
}

void Backends::CpuBackend::updateCachedDialResponses(const PropagationInputs& inputs_) {
  auto timerScope = _cachedDialStageTimer_.scopeTime();

  _dirtyCachedDialIndices_.clear();
  if( not _isDialResponseCachePrimed_ or _lastParameterValues_.size() != inputs_.parameters.values.size() ){
    for( std::uint32_t iDial = 0 ; iDial < _isCachedDial_.size() ; iDial++ ){
      if( _isCachedDial_[iDial] and not _isCachedDialResponseValid_[iDial] ){
        _dirtyCachedDialIndices_.emplace_back(iDial);
      }
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

  if( _profiler_ ){ _profiler_->cachePrepared(_dirtyCachedDialIndices_.size()); }
  if( _dirtyCachedDialIndices_.empty() ){ return; }
  _activeInputs_ = &inputs_;
  _threadPool_.runJob("CpuBackend::updateCachedDialResponses");
  _activeInputs_ = nullptr;
}

void Backends::CpuBackend::updateCachedDialResponsesThread(int iThread_) {
  CpuBackendProfiler::WorkerScope probe(_profiler_.get(), CpuBackendProfiler::Stage::Cache, iThread_);
  const auto& propagation = _engineView_->propagation;
  const auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, _threadPool_.getNbThreads(), _dirtyCachedDialIndices_.size()
  );
  const double* parameterValues = _activeInputs_->parameters.values.data();
  auto update = [&](auto sampled_){
    for( std::size_t iDirtyDial = bounds.beginIndex ; iDirtyDial < bounds.endIndex ; iDirtyDial++ ){
      const auto dialIndex = _dirtyCachedDialIndices_[iDirtyDial];
      const auto& dial = propagation.dials[dialIndex];
      const auto& input = propagation.dialInputs[dial.firstInput];
      const double inputValue = Semantics::transformDialInput(
          input, Semantics::loadParameterValue(input, parameterValues)
      );
      if( _isCachedDialResponseValid_[dialIndex] and inputValue == _cachedDialInputs_[dialIndex] ){
        if constexpr( decltype(sampled_)::value ){ ++probe.get()->unchangedInputs; }
        continue;
      }
      const auto* payload = Semantics::getDialPayload(propagation.dialPayloads.data(), dial);
      _cachedDialResponses_[dialIndex] = Semantics::evalDialResponseFromInput(dial, inputValue, payload);
      _cachedDialInputs_[dialIndex] = inputValue;
      _isCachedDialResponseValid_[dialIndex] = true;
      if constexpr( decltype(sampled_)::value ){
        ++probe.get()->evaluatedDials;
        ++probe.get()->evaluatedDialTypes[std::size_t(dial.type)];
      }
    }
  };
  if( probe.get() != nullptr ){ update(std::true_type{}); }
  else{ update(std::false_type{}); }
}

void Backends::CpuBackend::initializeWeightExecution() {
  const auto& model = _engineView_->propagation;
  _cachedDialCount_ = std::count(_isCachedDial_.begin(), _isCachedDial_.end(), std::uint8_t(true));
  bool allCached{true};
  // Every result slot is populated directly by the hot descriptor array.
  for( const auto& event : model.events ){
    LogThrowIf(event.resultIndex >= model.eventWeights.size(), "Invalid CPU event result index.");
  }
  for( auto dial : model.eventDialIndices ){
    LogThrowIf(dial >= model.dials.size(), "Invalid CPU event dial index.");
    allCached = allCached and _isCachedDial_[dial];
  }
  std::size_t work{0};
  for( const auto& event : model.eventWeights ){
    LogThrowIf(event.firstDial > model.eventDialIndices.size()
               or event.dialCount > model.eventDialIndices.size() - event.firstDial,
               "Invalid CPU event dial range.");
    work += 1 + event.dialCount;
  }
  const auto threads = std::size_t(_threadPool_.getNbThreads());
  _weightRanges_.assign(threads, {});
  std::size_t cursor{0}, completed{0};
  for( std::size_t thread = 0; thread < threads; ++thread ){
    auto& range = _weightRanges_[thread];
    range.begin = cursor;
    const auto target = (work / threads) * (thread + 1) + (work % threads) * (thread + 1) / threads;
    while( cursor < model.eventWeights.size() and (thread + 1 == threads or completed < target) ){
      completed += 1 + model.eventWeights[cursor++].dialCount;
    }
    range.end = cursor;
  }
  if( allCached ){
    _weightJob_ = model.hasReweightCap ? &CpuBackend::calculateEventWeightsKernel<true, true>
                                     : &CpuBackend::calculateEventWeightsKernel<true, false>;
  }
  else{
    _weightJob_ = model.hasReweightCap ? &CpuBackend::calculateEventWeightsKernel<false, true>
                                     : &CpuBackend::calculateEventWeightsKernel<false, false>;
  }
  LogInfo << "CPU reweight layout: " << model.eventWeights.size() << " events, "
          << model.eventWeights.size() * sizeof(BackendEventWeightDescriptor) << " hot descriptor bytes, "
          << model.eventDialIndices.size() * sizeof(std::uint32_t) << " dial index bytes, "
          << _cachedDialCount_ << "/" << model.dials.size() << " cached dials, "
          << "all event responses cached=" << allCached << ", shared model owners=" << _engineView_.use_count()
          << "." << std::endl;
}

void Backends::CpuBackend::calculateEventWeightsThread(int iThread_) {
  (this->*_weightJob_)(iThread_);
}

template<bool allResponsesCached, bool useReweightCap>
void Backends::CpuBackend::calculateEventWeightsKernel(int iThread_) {
  const auto& model = _engineView_->propagation;
  const auto& range = _weightRanges_[iThread_];
  CpuWeightKernels::reweight<allResponsesCached, useReweightCap>(
      model.eventWeights.data(), model.eventDialIndices.data(), _cachedDialResponses_.data(),
      _activeResult_->eventWeights.data(), range.begin, range.end, model.maxReweight,
      [this, &model](std::uint32_t dial_){
        return _isCachedDial_[dial_] ? _cachedDialResponses_[dial_]
            : Semantics::evalDialResponse(model, model.dials[dial_], *_activeInputs_);
      });
}

void Backends::CpuBackend::calculateHistograms(Result& result_) {
  auto timerScope = _histogramStageTimer_.scopeTime();

  const auto& propagation = _engineView_->propagation;
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
  if( _profiler_ ){ _profiler_->beginStage(CpuBackendProfiler::Stage::Histograms); }
  _threadPool_.runJob("CpuBackend::calculateHistograms");
  if( _profiler_ ){ _profiler_->endStage(CpuBackendProfiler::Stage::Histograms); }
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
  const auto& propagation = _engineView_->propagation;
  const auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, _threadPool_.getNbThreads(), propagation.events.size()
  );
  auto& histSums = _threadHistogramSums_[iThread_];
  auto& histSumSquares = _threadHistogramSumSquares_[iThread_];
  if( not _requiresEventBinning_ ){
    for( std::size_t iEvent = bounds.beginIndex; iEvent < bounds.endIndex; ++iEvent ){
      const auto& event = propagation.events[iEvent];
      if( event.globalBinIndex < 0 ){ continue; }
      const double weight = _activeResult_->eventWeights[event.resultIndex];
      histSums[event.globalBinIndex] += weight;
      histSumSquares[event.globalBinIndex] += weight * weight;
    }
    return;
  }
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
      _engineView_->propagation,
      inputs_
  );
}

void Backends::CpuBackend::calculateLikelihood(Result& result_) {
  auto timerScope = _likelihoodHostTimer_.scopeTime();

  result_.likelihood = Semantics::calculateLikelihood(
      _engineView_->likelihood,
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
