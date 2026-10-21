#include "CpuBackend.h"
#include "CpuWeightKernels.h"

#include "Semantics/BackendHostPropagation.h"
#include "GundamGlobals.h"
#include "Logger.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <unordered_map>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>

void Backends::CpuBackend::configureImpl() {
  _config_.clearFields();
  _config_.defineFields({{"profileOutputPath"}, {"profileSampleInterval"}, {"profileMaxSamples"},
                         {"weightScheduling"}});
  _config_.checkConfiguration();
  _config_.fillValue(_profileOutputPath_, "profileOutputPath");
  _config_.fillValue(_profileSampleInterval_, "profileSampleInterval");
  _config_.fillValue(_profileMaxSamples_, "profileMaxSamples");
  _config_.fillEnum(_weightScheduling_, "weightScheduling");
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
        if( std::size_t(iThread_) >= _weightRanges_.size() ){ return; }
        const auto start = _diagnosticWeightTiming_ ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point{};
        CpuBackendProfiler::WorkerScope probe(_profiler_.get(), CpuBackendProfiler::Stage::Weights, iThread_);
        calculateEventWeightsThread(iThread_);
        if( _diagnosticWeightTiming_ ){
          _weightWorkerStats_[iThread_].wallSeconds =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        if( auto* record = probe.get() ){
          record->weightEvents = _weightWorkerStats_[iThread_].events;
          record->weightBlocks = _weightWorkerStats_[iThread_].blocks;
        }
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
  _parameterDependentCachedDialCount_ = 0;
  _changedCacheParameters_.assign(propagation.parameterCount, false);
  _cachedDialsByParameter_.assign(propagation.parameterCount, {});
  _cachedDialResponses_.assign(propagation.dials.size(), 1.);
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
    ++_parameterDependentCachedDialCount_;
  }
  initializeCachedInputGroups();
}

void Backends::CpuBackend::initializeCachedInputGroups() {
  const auto& model = _engineView_->propagation;
  std::vector<double>{}.swap(_cachedDialInputs_);
  std::vector<CachedInputGroup>{}.swap(_cachedInputGroups_);
  std::vector<std::uint32_t>{}.swap(_cachedDialInputGroups_);
  std::vector<std::vector<std::uint32_t>>{}.swap(_cachedInputGroupsByParameter_);
  if( _parameterDependentCachedDialCount_ == 0 ){ return; }
  auto fallback = [&]{
    std::vector<CachedInputGroup>{}.swap(_cachedInputGroups_);
    std::vector<std::uint32_t>{}.swap(_cachedDialInputGroups_);
    std::vector<std::vector<std::uint32_t>>{}.swap(_cachedInputGroupsByParameter_);
    _cachedDialInputs_.assign(model.dials.size(), 0.);
  };
  if( model.dials.size() < 8 ){ fallback(); return; }
  using Key = std::array<std::uint64_t, 4>;
  struct KeyHash {
    std::size_t operator()(const Key& key_) const {
      std::size_t hash{0};
      for( auto item : key_ ){ hash ^= std::hash<std::uint64_t>{}(item) + 0x9e3779b9 + (hash << 6) + (hash >> 2); }
      return hash;
    }
  };
  // Bitwise keys preserve signed zero and NaN settings conservatively. Unused
  // mirror bounds do not distinguish inputs when mirroring is disabled.
  auto bits = [](double value_){ std::uint64_t result; std::memcpy(&result, &value_, sizeof(result)); return result; };
  std::unordered_map<Key, std::uint32_t, KeyHash> groups;
  _cachedDialInputGroups_.assign(model.dials.size(), std::numeric_limits<std::uint32_t>::max());
  _cachedInputGroupsByParameter_.resize(model.parameterCount);
  for( std::size_t dial = 0; dial < model.dials.size(); ++dial ){
    if( not _isCachedDial_[dial] or model.dials[dial].type == BackendDialType::Shift ){ continue; }
    const auto firstInput = model.dials[dial].firstInput;
    const auto& input = model.dialInputs[firstInput];
    const Key key{std::uint64_t(input.parameterIndex), std::uint64_t(input.useMirror),
                  input.useMirror ? bits(input.mirrorMin) : 0, input.useMirror ? bits(input.mirrorRange) : 0};
    auto found = groups.find(key);
    if( found == groups.end() ){
      // Stop early on models without enough reuse. Also check actual retained
      // capacities below; shared metadata must replace, not augment, the old
      // per-dial last-input buffer at a higher memory cost.
      if( _cachedInputGroups_.size() >= model.dials.size() / 8 ){ groups.clear(); groups.rehash(0); fallback(); return; }
      const auto index = std::uint32_t(_cachedInputGroups_.size());
      found = groups.emplace(key, index).first;
      _cachedInputGroups_.push_back({firstInput});
      _cachedInputGroupsByParameter_[input.parameterIndex].push_back(index);
    }
    _cachedDialInputGroups_[dial] = found->second;
  }
  if( getCachedInputStorageBytes() >= model.dials.size() * sizeof(double) ){
    groups.clear(); groups.rehash(0); fallback();
  }
}

std::size_t Backends::CpuBackend::getCachedInputStorageBytes() const {
  std::size_t bytes = _cachedDialInputs_.capacity() * sizeof(double)
      + _cachedDialInputGroups_.capacity() * sizeof(std::uint32_t)
      + _cachedInputGroups_.capacity() * sizeof(CachedInputGroup)
      + _cachedInputGroupsByParameter_.capacity() * sizeof(std::vector<std::uint32_t>);
  for( const auto& indices : _cachedInputGroupsByParameter_ ){ bytes += indices.capacity() * sizeof(std::uint32_t); }
  return bytes;
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
    if( _usesDynamicWeights_ ){ _nextWeightBlock_.store(0, std::memory_order_relaxed); }
    if( _profiler_ ){ _profiler_->beginStage(CpuBackendProfiler::Stage::Weights); }
    _threadPool_.runJob("CpuBackend::calculateEventWeights");
    if( _profiler_ ){ _profiler_->endStage(CpuBackendProfiler::Stage::Weights); }
  }
  _activeInputs_ = nullptr;
  _activeResult_ = nullptr;
}

void Backends::CpuBackend::setDiagnosticCacheTraversal(CpuCacheTraversal traversal_) {
  LogThrowIf(not _isBuilt_, "Build the CPU backend before changing diagnostic cache traversal.");
  LogThrowIf(traversal_ != CpuCacheTraversal::Auto and traversal_ != CpuCacheTraversal::Indexed,
             "Invalid CPU diagnostic cache traversal.");
  _cacheTraversal_ = traversal_;
  initializeProfiler();
}

void Backends::CpuBackend::setDiagnosticInputEvaluation(CpuInputEvaluation evaluation_) {
  LogThrowIf(not _isBuilt_, "Build the CPU backend before changing input evaluation.");
  LogThrowIf(evaluation_ != CpuInputEvaluation::Shared and evaluation_ != CpuInputEvaluation::PerDial,
             "Invalid CPU diagnostic input evaluation.");
  _inputEvaluation_ = evaluation_;
  initializeProfiler();
}

void Backends::CpuBackend::updateCachedDialResponses(const PropagationInputs& inputs_) {
  auto timerScope = _cachedDialStageTimer_.scopeTime();
  const auto preparationStart = std::chrono::steady_clock::now();
  const auto& values = inputs_.parameters.values;
  const auto& propagation = _engineView_->propagation;
  _dirtyCachedDialIndices_.clear();
  _dirtyCachedDialCount_ = 0;
  std::fill(_changedCacheParameters_.begin(), _changedCacheParameters_.end(), false);
  const bool initial = not _isDialResponseCachePrimed_ or _lastParameterValues_.size() != values.size();
  if( initial ){
    // All parameter-dependent entries start invalid; immutable shifts were
    // already evaluated during build and never enter the update list.
    _dirtyCachedDialCount_ = _parameterDependentCachedDialCount_;
    std::fill(_changedCacheParameters_.begin(), _changedCacheParameters_.end(), true);
  }
  else{
    for( std::size_t parameter = 0; parameter < values.size(); ++parameter ){
      if( values[parameter] != _lastParameterValues_[parameter] ){
        _changedCacheParameters_[parameter] = true;
        _dirtyCachedDialCount_ += _cachedDialsByParameter_[parameter].size();
      }
    }
  }
  // Stream the model when at least half the descriptors need an update.
  // Keep the targeted path otherwise. This avoids scanning a huge
  // external-weight/constant-only model for a handful of changing responses.
  _usesLinearCacheTraversal_ = _cacheTraversal_ == CpuCacheTraversal::Auto
      and _dirtyCachedDialCount_ != 0
      and _dirtyCachedDialCount_ >= propagation.dials.size() - _dirtyCachedDialCount_;
  if( not _usesLinearCacheTraversal_ and _dirtyCachedDialCount_ != 0 ){
    if( initial ){
      for( std::uint32_t dial = 0; dial < _isCachedDial_.size(); ++dial ){
        if( _isCachedDial_[dial] and not _isCachedDialResponseValid_[dial] ){
          _dirtyCachedDialIndices_.push_back(dial);
        }
      }
    }
    else{
      for( std::size_t parameter = 0; parameter < values.size(); ++parameter ){
        if( not _changedCacheParameters_[parameter] ){ continue; }
        const auto& dials = _cachedDialsByParameter_[parameter];
        _dirtyCachedDialIndices_.insert(_dirtyCachedDialIndices_.end(), dials.begin(), dials.end());
      }
    }
  }
  const bool grouped = not _cachedInputGroups_.empty();
  _lastTiming_.sharedCacheInputs = grouped and _inputEvaluation_ == CpuInputEvaluation::Shared;
  if( grouped ){
    for( std::size_t parameter = 0; parameter < values.size(); ++parameter ){
      if( not _changedCacheParameters_[parameter] ){ continue; }
      for( auto index : _cachedInputGroupsByParameter_[parameter] ){
        auto& group = _cachedInputGroups_[index];
        const auto& input = propagation.dialInputs[group.firstInput];
        group.value = Semantics::transformDialInput(input, values[parameter]);
        group.changed = not group.valid or group.value != group.previousValue;
        ++_lastTiming_.cacheInputTransformCount;
      }
    }
  }
  if( not _lastTiming_.sharedCacheInputs ){ _lastTiming_.cacheInputTransformCount += _dirtyCachedDialCount_; }
  _lastParameterValues_ = values;
  _isDialResponseCachePrimed_ = true;
  _lastTiming_.cacheCandidateCount = _dirtyCachedDialCount_;
  _lastTiming_.linearCacheTraversal = _usesLinearCacheTraversal_;
  _lastTiming_.cachePreparationSeconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - preparationStart).count();
  if( _profiler_ ){ _profiler_->cachePrepared(_dirtyCachedDialCount_); }
  if( _dirtyCachedDialCount_ == 0 ){ return; }
  const auto executionStart = std::chrono::steady_clock::now();
  _activeInputs_ = &inputs_;
  _threadPool_.runJob("CpuBackend::updateCachedDialResponses");
  _activeInputs_ = nullptr;
  if( grouped ){
    for( std::size_t parameter = 0; parameter < values.size(); ++parameter ){
      if( not _changedCacheParameters_[parameter] ){ continue; }
      for( auto index : _cachedInputGroupsByParameter_[parameter] ){
        auto& group = _cachedInputGroups_[index];
        group.previousValue = group.value;
        group.valid = true;
        group.changed = false;
      }
    }
  }
  _lastTiming_.cacheExecutionSeconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - executionStart).count();
}

void Backends::CpuBackend::updateCachedDialResponsesThread(int iThread_) {
  CpuBackendProfiler::WorkerScope probe(_profiler_.get(), CpuBackendProfiler::Stage::Cache, iThread_);
  const auto& propagation = _engineView_->propagation;
  const auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, _threadPool_.getNbThreads(),
      _usesLinearCacheTraversal_ ? propagation.dials.size() : _dirtyCachedDialIndices_.size()
  );
  const double* parameterValues = _activeInputs_->parameters.values.data();
  auto update = [&](auto sampled_, auto linear_, auto shared_, auto grouped_){
    for( std::size_t iDirtyDial = bounds.beginIndex ; iDirtyDial < bounds.endIndex ; iDirtyDial++ ){
      const auto dialIndex = [&](){
        if constexpr( decltype(linear_)::value ){ return iDirtyDial; }
        else{ return std::size_t(_dirtyCachedDialIndices_[iDirtyDial]); }
      }();
      if constexpr( decltype(linear_)::value ){
        if( not _isCachedDial_[dialIndex] or propagation.dials[dialIndex].type == BackendDialType::Shift ){ continue; }
      }
      const auto& dial = propagation.dials[dialIndex];
      const auto* group = [&](){
        if constexpr( decltype(grouped_)::value ){ return &_cachedInputGroups_[_cachedDialInputGroups_[dialIndex]]; }
        else{ return static_cast<const CachedInputGroup*>(nullptr); }
      }();
      const auto firstInput = [&](){
        if constexpr( decltype(shared_)::value ){ return group->firstInput; }
        else{ return dial.firstInput; }
      }();
      const auto& input = propagation.dialInputs[firstInput];
      if constexpr( decltype(linear_)::value ){
        if( not _changedCacheParameters_[input.parameterIndex] ){ continue; }
      }
      double inputValue;
      bool unchanged;
      if constexpr( decltype(shared_)::value ){
        inputValue = group->value;
        unchanged = not group->changed;
      }
      else{
        inputValue = Semantics::transformDialInput(input, Semantics::loadParameterValue(input, parameterValues));
        const double previous = [&](){
          if constexpr( decltype(grouped_)::value ){ return group->previousValue; }
          else{ return _cachedDialInputs_[dialIndex]; }
        }();
        unchanged = inputValue == previous;
      }
      if( _isCachedDialResponseValid_[dialIndex] and unchanged ){
        if constexpr( decltype(sampled_)::value ){ ++probe.get()->unchangedInputs; }
        continue;
      }
      const auto* payload = Semantics::getDialPayload(propagation.dialPayloads.data(), dial);
      _cachedDialResponses_[dialIndex] = Semantics::evalDialResponseFromInput(dial, inputValue, payload);
      if constexpr( not decltype(grouped_)::value ){ _cachedDialInputs_[dialIndex] = inputValue; }
      _isCachedDialResponseValid_[dialIndex] = true;
      if constexpr( decltype(sampled_)::value ){
        ++probe.get()->evaluatedDials;
        ++probe.get()->evaluatedDialTypes[std::size_t(dial.type)];
      }
    }
  };
  auto dispatch = [&](auto sampled_, auto shared_, auto grouped_){
    if( _usesLinearCacheTraversal_ ){ update(sampled_, std::true_type{}, shared_, grouped_); }
    else{ update(sampled_, std::false_type{}, shared_, grouped_); }
  };
  auto select = [&](auto sampled_){
    if( _lastTiming_.sharedCacheInputs ){ dispatch(sampled_, std::true_type{}, std::true_type{}); }
    else if( not _cachedInputGroups_.empty() ){ dispatch(sampled_, std::false_type{}, std::true_type{}); }
    else{ dispatch(sampled_, std::false_type{}, std::false_type{}); }
  };
  if( probe.get() != nullptr ){ select(std::true_type{}); }
  else{ select(std::false_type{}); }
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
  _weightBlocksPerThread_ = 2;
  // Diagnostic-only override for controlled benchmarks; normal fits retain
  // the automatic policy without a numeric configuration option.
  if( const auto* diagnostic = std::getenv("GUNDAM_CPU_DIAGNOSTIC_BLOCKS_PER_THREAD") ){
    const std::string value(diagnostic);
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), _weightBlocksPerThread_);
    LogThrowIf(parsed.ec != std::errc{} or parsed.ptr != value.data() + value.size()
               or _weightBlocksPerThread_ == 0
               or _weightBlocksPerThread_ > std::numeric_limits<std::size_t>::max() / threads,
               "GUNDAM_CPU_DIAGNOSTIC_BLOCKS_PER_THREAD must be a positive integer without overflow.");
    LogWarning << "CPU diagnostic block factor override: " << _weightBlocksPerThread_ << std::endl;
  }
  _weightWork_ = work;
  _allWeightResponsesCached_ = allCached;
  initializeWeightScheduling();
}

void Backends::CpuBackend::setDiagnosticWeightScheduling(CpuWeightScheduling scheduling_, std::size_t blocksPerThread_) {
  LogThrowIf(not _isBuilt_, "Build the CPU backend before changing diagnostic scheduling.");
  LogThrowIf(scheduling_ != CpuWeightScheduling::Fixed and scheduling_ != CpuWeightScheduling::Dynamic,
             "Invalid CPU diagnostic scheduling mode.");
  LogThrowIf(blocksPerThread_ == 0
             or blocksPerThread_ > std::numeric_limits<std::size_t>::max() / _threadPool_.getNbThreads(),
             "CPU diagnostic block factor must be a positive integer without overflow.");
  _weightScheduling_ = scheduling_;
  _weightBlocksPerThread_ = blocksPerThread_;
  initializeWeightScheduling();
  // If probes were requested, start a new run with the new layout metadata.
  initializeProfiler();
}

void Backends::CpuBackend::setDiagnosticWeightWorkers(std::size_t workers_, bool timing_) {
  LogThrowIf(not _isBuilt_, "Build the CPU backend before changing weight workers.");
  LogThrowIf(workers_ == 0 or workers_ > std::size_t(_threadPool_.getNbThreads()),
             "Weight workers must be between one and the resident pool size.");
  _diagnosticWeightWorkers_ = workers_;
  _diagnosticWeightTiming_ = timing_;
  initializeWeightScheduling();
}

void Backends::CpuBackend::initializeWeightScheduling() {
  const auto& model = _engineView_->propagation;
  const auto threads = _diagnosticWeightWorkers_ != 0 ? _diagnosticWeightWorkers_
                                                        : std::size_t(_threadPool_.getNbThreads());
  const auto work = _weightWork_;
  const auto blockTarget = _weightBlocksPerThread_ * threads;
  _usesDynamicWeights_ = _weightScheduling_ == CpuWeightScheduling::Dynamic
                        and threads > 1 and model.eventWeights.size() >= blockTarget;
  _weightWorkerStats_.assign(threads, {});
  auto makeRanges = [&](std::size_t count_, bool keepNonempty_){
    std::vector<EventRange> ranges(count_);
    std::size_t cursor{0}, completed{0};
    for( std::size_t part = 0; part < count_; ++part ){
      auto& range = ranges[part];
      range.begin = cursor;
      const auto initialWork = completed;
      const auto target = (work / count_) * (part + 1) + (work % count_) * (part + 1) / count_;
      // A very expensive event cannot be split. Still leave at least one
      // event for every subsequent dynamic block, avoiding empty queue entries.
      const auto limit = model.eventWeights.size() - (keepNonempty_ ? count_ - part - 1 : 0);
      while( cursor < limit and (part + 1 == count_ or completed < target
                                 or (keepNonempty_ and cursor == range.begin)) ){
        completed += 1 + model.eventWeights[cursor++].dialCount;
      }
      range.end = cursor;
      range.dialReferences = completed - initialWork - (range.end - range.begin);
    }
    return ranges;
  };
  _weightRanges_ = makeRanges(threads, false);
  _weightBlocks_ = _usesDynamicWeights_ ? makeRanges(blockTarget, true) : std::vector<EventRange>{};
  switch( _diagnosticWeightInterleave_ ){
    case 2: selectWeightKernel<2>(); break;
    case 4: selectWeightKernel<4>(); break;
    default: selectWeightKernel<1>(); break;
  }
  LogInfo << "CPU reweight layout: " << model.eventWeights.size() << " events, "
          << model.eventWeights.size() * sizeof(BackendEventWeightDescriptor) << " hot descriptor bytes, "
          << model.eventDialIndices.size() * sizeof(std::uint32_t) << " dial index bytes, "
          << _cachedDialCount_ << "/" << model.dials.size() << " cached dials, "
          << "all event responses cached=" << _allWeightResponsesCached_ << ", shared model owners=" << _engineView_.use_count()
          << "." << std::endl;
  LogInfo << "CPU weight scheduling: " << (_usesDynamicWeights_ ? "Dynamic" : "Fixed")
          << " (requested " << _weightScheduling_ << ", " << _weightBlocks_.size()
          << " automatic dynamic blocks)." << std::endl;
}

void Backends::CpuBackend::setDiagnosticWeightInterleave(std::size_t lanes_) {
  LogThrowIf(not _isBuilt_, "Build the CPU backend before selecting a weight kernel.");
  LogThrowIf(lanes_ != 1 and lanes_ != 2 and lanes_ != 4, "Interleave width must be 1, 2 or 4.");
  _diagnosticWeightInterleave_ = lanes_;
  initializeWeightScheduling();
}

const std::vector<double>& Backends::CpuBackend::getDiagnosticEventWeights() const {
  LogThrowIf(not _lastResult_.token.isValid, "Evaluate the CPU backend before inspecting weights.");
  return _lastResult_.eventWeights;
}

template<std::size_t lanes>
void Backends::CpuBackend::selectWeightKernel() {
  const auto& model = _engineView_->propagation;
  if( _allWeightResponsesCached_ ){
    if( _usesDynamicWeights_ ){
      _weightJob_ = model.hasReweightCap ? &CpuBackend::calculateEventWeightsKernel<true, true, true, lanes>
                                       : &CpuBackend::calculateEventWeightsKernel<true, false, true, lanes>;
    }
    else{
      _weightJob_ = model.hasReweightCap ? &CpuBackend::calculateEventWeightsKernel<true, true, false, lanes>
                                       : &CpuBackend::calculateEventWeightsKernel<true, false, false, lanes>;
    }
  }
  else{
    if( _usesDynamicWeights_ ){
      _weightJob_ = model.hasReweightCap ? &CpuBackend::calculateEventWeightsKernel<false, true, true>
                                       : &CpuBackend::calculateEventWeightsKernel<false, false, true>;
    }
    else{
      _weightJob_ = model.hasReweightCap ? &CpuBackend::calculateEventWeightsKernel<false, true, false>
                                       : &CpuBackend::calculateEventWeightsKernel<false, false, false>;
    }
  }
}

void Backends::CpuBackend::calculateEventWeightsThread(int iThread_) {
  (this->*_weightJob_)(iThread_);
}

template<bool allResponsesCached, bool useReweightCap, bool dynamicScheduling, std::size_t lanes>
void Backends::CpuBackend::calculateEventWeightsKernel(int iThread_) {
  const auto& model = _engineView_->propagation;
  auto reweightRange = [&](std::size_t begin_, std::size_t end_){
    if constexpr( lanes > 1 ){
      static_assert(allResponsesCached, "Interleaving requires cached responses.");
      CpuWeightKernels::reweightInterleaved<lanes, useReweightCap>(
          model.eventWeights.data(), model.eventDialIndices.data(), _cachedDialResponses_.data(),
          _activeResult_->eventWeights.data(), begin_, end_, model.maxReweight);
    }
    else{
      CpuWeightKernels::reweight<allResponsesCached, useReweightCap>(
        model.eventWeights.data(), model.eventDialIndices.data(), _cachedDialResponses_.data(),
        _activeResult_->eventWeights.data(), begin_, end_, model.maxReweight,
        [this, &model](std::uint32_t dial_){
          return _isCachedDial_[dial_] ? _cachedDialResponses_[dial_]
              : Semantics::evalDialResponse(model, model.dials[dial_], *_activeInputs_);
        });
    }
  };
  WeightWorkerStats stats;
  if constexpr( dynamicScheduling ){
    while( true ){
      // Only block ownership is shared. Inputs are immutable during the job;
      // workers write disjoint result slots and the pool joins before use.
      const auto block = _nextWeightBlock_.fetch_add(1, std::memory_order_relaxed);
      if( block >= _weightBlocks_.size() ){ break; }
      const auto& range = _weightBlocks_[block];
      reweightRange(range.begin, range.end);
      stats.events += range.end - range.begin;
      stats.dialReferences += range.dialReferences;
      ++stats.blocks;
    }
  }
  else{
    const auto& range = _weightRanges_[iThread_];
    reweightRange(range.begin, range.end);
    stats.events = range.end - range.begin;
    stats.dialReferences = range.dialReferences;
    stats.blocks = stats.events != 0;
  }
  _weightWorkerStats_[iThread_] = stats;
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
