#include "MpsBackend.h"

#include "MpsBackendInternal.h"
#include "Semantics/BackendHostPropagation.h"

Backends::MpsBackend::MpsBackend() : _impl_(std::make_unique<MpsBackendImpl>()) {}
Backends::MpsBackend::~MpsBackend() = default;

void Backends::MpsBackend::configureImpl() {
  _config_.clearFields();
  _config_.defineFields({
      {"propagationMode"},
  });
  _config_.checkConfiguration();
  _config_.fillEnum(_impl_->propagationMode, "propagationMode");
  LogThrowIf(_impl_->propagationMode != MpsPropagationMode::cachedResponses
             and _impl_->propagationMode != MpsPropagationMode::recomputeResponses,
             "Unknown MPS propagationMode. Expected cachedResponses or recomputeResponses.");
  if( _impl_->eventPropagation ){
    _impl_->eventPropagation->setEnableDialCache(
        _impl_->propagationMode == MpsPropagationMode::cachedResponses);
  }
  _config_.printUnusedKeys();
}

std::string Backends::MpsBackend::getName() const {
  return "MPS/" + _impl_->propagationMode.toString();
}

Backends::BackendCapabilities Backends::MpsBackend::getCapabilities() const {
  BackendCapabilities out;
  out.supportsGpu = true;
  out.supportsDynamicBinning = true;
  out.supportsObservableTransforms = true;
  out.supportsEventWeights = true;
  out.supportsHistograms = true;
  out.supportsLikelihood = true;
  // The resident executor preserves float pairs through event products and
  // histogram readback. The packed weight executor still publishes float32.
  out.recommendedFunctionPrecision = _impl_->useEventPropagation ? 1.e-12 : std::numeric_limits<float>::epsilon();
  out.deviceName = _impl_->isAvailable ? [[_impl_->device name] UTF8String] : "Metal unavailable";
  return out;
}

void Backends::MpsBackend::build(const EngineView& engineView_) {
  _impl_->isBuilt = false;
  _impl_->engineView = engineView_;
  _impl_->lastResult = MpsBackendImpl::Result();
  _impl_->buildTiming = BackendTimingSummary();
  _impl_->useEventPropagation = _impl_->propagationMode == MpsPropagationMode::cachedResponses
      or engineView_.propagation.requiresEventBinning() or engineView_.propagation.hasReweightCap;
  for( const auto& dial : engineView_.propagation.dials ){
    LogThrowIf(dial.type == BackendDialType::Composite, "Composite dials are only supported by the CPU backend.");
    LogThrowIf(isTableDial(dial.type), "Tabulated and Kriged dials are not supported by the MPS backend.");
    Semantics::validateSurfaceDial(engineView_.propagation, dial);
    if( dial.type == BackendDialType::ExternalWeight ){ continue; }
    if( dial.inputCount > 1 ){ _impl_->useEventPropagation = true; }
    for( std::size_t i = 0; i < dial.inputCount and dial.firstInput + i < engineView_.propagation.dialInputs.size(); ++i ){
      LogThrowIf(engineView_.propagation.dialInputs[dial.firstInput + i].source != DialInputSource::Parameter,
                 "Event-variable dial inputs require an event-context evaluator.");
    }
  }
  LogInfo << "MPS propagation mode: " << _impl_->propagationMode
          << "; executor: " << (_impl_->useEventPropagation ? "eventPropagation" : "packedWeights") << std::endl;
  if( _impl_->useEventPropagation ){
    LogThrowIf(not _impl_->isAvailable, "Metal is unavailable for event propagation.");
    _impl_->releaseDeviceBuffers();
    _impl_->eventPropagation->build(_impl_->model);
  }
  else{
    if( _impl_->eventPropagation ){ _impl_->eventPropagation->clear(); }
    if( not _impl_->buildWeightModel() ){
      LogWarning << "MPS backend cannot use the GPU propagation path: "
                 << (_impl_->deviceModelFallbackReason.empty() ? "unknown compatibility issue."
                                                               : _impl_->deviceModelFallbackReason)
                 << " Falling back to the standard backend path for unsupported calculations."
                 << std::endl;
    }
  }
  _impl_->isBuilt = true;
}

Backends::PropagationToken Backends::MpsBackend::requestPropagation(const PropagationInputs& inputs_) {
  LogThrowIf(not _impl_->isBuilt, "MpsBackend has not been built.");
  inputs_.validate(_impl_->model);

  _impl_->resetResult();
  const auto propagationStart = std::chrono::steady_clock::now();

  if( not _impl_->isAvailable ){
    _impl_->lastResult.status.backend = BackendStatus::Unavailable;
    _impl_->lastResult.status.eventWeights = OutputState::Failed;
    _impl_->lastResult.status.histograms = OutputState::Failed;
    _impl_->lastResult.status.sampleLikelihoods = OutputState::Failed;
    _impl_->lastResult.status.statLikelihood = OutputState::Failed;
    _impl_->lastResult.token.isValid = false;
    return {};
  }

  if( _impl_->useEventPropagation ){
    try{ _impl_->eventPropagation->propagate(inputs_, _impl_->lastResult, _impl_->lastTiming); }
    catch( ... ){
      _impl_->eventPropagation->invalidate();
      _impl_->lastResult.status.backend = BackendStatus::Failed;
      _impl_->lastResult.token.isValid = false;
      throw;
    }
    auto& status = _impl_->lastResult.status;
    status.eventWeights = OutputState::ReadyOnDevice;
    status.histograms = OutputState::ReadyOnDevice;
    status.eventState = _impl_->model.requiresEventBinning() ? OutputState::ReadyOnDevice : OutputState::NotRequested;
    status.sampleLikelihoods = OutputState::Failed;
    if( _impl_->likelihoodModel.empty() ){ status.statLikelihood = OutputState::Failed; }
    else{ _impl_->calculateLikelihood(); status.statLikelihood = OutputState::ReadyOnHost; }
    status.backend = BackendStatus::Ready;
    _impl_->lastTiming.propagationWallSeconds = secondsSince(propagationStart);
    return _impl_->lastResult.token;
  }

  const bool usedDevicePropagation = _impl_->runWeightPropagation(inputs_);
  if( usedDevicePropagation ){
    _impl_->lastResult.status.eventWeights = OutputState::ReadyOnDevice;
    _impl_->lastResult.status.histograms = OutputState::ReadyOnDevice;
  }

  if( not usedDevicePropagation ){
    Semantics::calculateEventWeights(_impl_->lastResult.eventWeights, _impl_->model, inputs_);
    _impl_->lastResult.status.eventWeights = OutputState::ReadyOnHost;
  }

  if( not usedDevicePropagation ){
    if( not _impl_->calculateHistogramsOnDevice() ){
      _impl_->lastResult.status.backend = BackendStatus::Failed;
      _impl_->lastResult.status.histograms = OutputState::Failed;
      _impl_->lastResult.status.statLikelihood = OutputState::Failed;
      return _impl_->lastResult.token;
    }
    _impl_->lastResult.status.histograms = OutputState::ReadyOnDevice;
  }

  _impl_->lastResult.status.sampleLikelihoods = OutputState::Failed;
  if( _impl_->likelihoodModel.empty() ){
    _impl_->lastResult.status.statLikelihood = OutputState::Failed;
  }
  else{
    _impl_->calculateLikelihood();
    _impl_->lastResult.status.statLikelihood = OutputState::ReadyOnHost;
  }

  _impl_->lastResult.status.backend = BackendStatus::Ready;
  _impl_->lastTiming.propagationWallSeconds = secondsSince(propagationStart);
  return _impl_->lastResult.token;
}

Backends::PropagationStatus Backends::MpsBackend::getStatus(const PropagationToken& token_) const {
  if( not _impl_->isCurrentToken(token_) ){
    PropagationStatus out;
    out.backend = BackendStatus::Failed;
    return out;
  }
  return _impl_->lastResult.status;
}

const Backends::EngineView& Backends::MpsBackend::getEngineView() const {
  return _impl_->engineView;
}

bool Backends::MpsBackend::isReady(const PropagationToken& token_) const {
  return _impl_->isCurrentToken(token_) and _impl_->lastResult.status.backend == BackendStatus::Ready;
}

void Backends::MpsBackend::wait(const PropagationToken& token_) {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
}

void Backends::MpsBackend::materialize(const PropagationToken& token_, OutputRequest output_) {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  LogThrowIf(_impl_->lastResult.status.state(output_) != OutputState::ReadyOnDevice
             and _impl_->lastResult.status.state(output_) != OutputState::ReadyOnHost,
             "Requested backend output is not ready.");

  if( output_ == OutputRequest::EventState ){
    _impl_->eventPropagation->materializeState(_impl_->lastTiming);
    _impl_->lastResult.status.eventState = OutputState::ReadyOnHost;
  }
  else if( output_ == OutputRequest::EventWeights ){
    if( _impl_->useEventPropagation ){ _impl_->eventPropagation->materializeWeights(_impl_->lastResult, _impl_->lastTiming); }
    else{ _impl_->materializeEventWeights(); }
    _impl_->lastResult.status.eventWeights = OutputState::ReadyOnHost;
  }
  else if( output_ == OutputRequest::Histograms ){
    _impl_->materializeHistograms();
    _impl_->lastResult.status.histograms = OutputState::ReadyOnHost;
  }
  else if( output_ == OutputRequest::SampleLikelihoods ){
    LogThrow("MpsBackend cannot materialize sample likelihoods yet.");
  }
  else if( output_ == OutputRequest::StatLikelihood ){
    _impl_->lastResult.status.statLikelihood = OutputState::ReadyOnHost;
  }
  else{
    LogThrow("MpsBackend cannot materialize requested output yet.");
  }
}

double Backends::MpsBackend::getLikelihood(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  LogThrowIf(_impl_->lastResult.status.statLikelihood != OutputState::ReadyOnDevice
             and _impl_->lastResult.status.statLikelihood != OutputState::ReadyOnHost,
             "Backend likelihood is not ready.");
  return _impl_->lastResult.likelihood;
}

const std::vector<double>& Backends::MpsBackend::getEventWeightsHostView(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  return _impl_->lastResult.eventWeights;
}

const std::vector<double>& Backends::MpsBackend::getHistogramSumsHostView(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  return _impl_->lastResult.histSums;
}

const std::vector<double>& Backends::MpsBackend::getHistogramSumSquaresHostView(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  return _impl_->lastResult.histSumSquares;
}

Backends::BackendDeviceView Backends::MpsBackend::getDeviceView(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  if( _impl_->useEventPropagation ){ return _impl_->eventPropagation->getDeviceView(); }
  BackendDeviceView out;
  out.device = _impl_->device;
  out.eventWeights = _impl_->eventWeightsBuffer;
  out.eventWeightsBytes = _impl_->model.events.size() * sizeof(float);
  out.histSums = _impl_->histSumsBuffer;
  out.histSumSquares = _impl_->histSumSquaresBuffer;
  out.histogramBytes = std::size_t(_impl_->model.totalBins) * sizeof(float);
  return out;
}

Backends::BackendTimingSummary Backends::MpsBackend::getLastTimingSummary() const {
  return _impl_->lastTiming;
}

const Backends::EventState& Backends::MpsBackend::getEventStateHostView(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_) or not _impl_->useEventPropagation, "Invalid MPS event state request.");
  return _impl_->eventPropagation->materializeState(_impl_->lastTiming);
}
const std::vector<double>& Backends::MpsBackend::getOutsideSumsHostView(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  return _impl_->lastResult.outsideSums;
}
const std::vector<double>& Backends::MpsBackend::getOutsideSumSquaresHostView(const PropagationToken& token_) const {
  LogThrowIf(not _impl_->isCurrentToken(token_), "Invalid MpsBackend propagation token.");
  return _impl_->lastResult.outsideSumSquares;
}

void Backends::MpsBackend::setDiagnosticStageProfiling(bool enabled_) {
  _impl_->enableDiagnosticStageProfiling = enabled_;
  if( _impl_->eventPropagation ){ _impl_->eventPropagation->setDiagnosticStageProfiling(enabled_); }
}

std::string Backends::MpsBackend::getDiagnosticEventPropagationState() const {
  LogThrowIf(not _impl_->useEventPropagation, "MPS event propagation is not active.");
  return _impl_->eventPropagation->getExecutionState();
}

std::string Backends::MpsBackend::getDiagnostics() const {
  const auto& impl = *_impl_;
  JsonType out;
  out["device"] = getCapabilities().deviceName;
  out["available"] = impl.isAvailable;
  out["unifiedMemory"] = impl.isAvailable && impl.device.hasUnifiedMemory;
  out["stageProfiling"] = impl.enableDiagnosticStageProfiling;
  out["propagationMode"] = impl.propagationMode.toString();
  out["path"] = impl.useEventPropagation ? "eventPropagation" : (impl.isWeightModelSupported ? "packedWeights" : "hostFallback");
  out["fallbackReason"] = impl.deviceModelFallbackReason;
  out["events"] = impl.model.events.size();
  out["bins"] = impl.model.totalBins;
  out["parameters"] = impl.model.parameterCount;
  out["uniqueDials"] = impl.model.dials.size();
  out["weightDialReferences"] = impl.model.eventDialIndices.size();
  out["variableDialReferences"] = impl.model.variableDialIndices.size();
  out["cachedDials"] = impl.cachedDialCount;
  out["splineDescriptorBytes"] = sizeof(MpsSplineDialDescriptor);
  out["normOccurrenceBytes"] = sizeof(MpsNormDialOccurrence);
  out["cachedDialDispatchCount"] = impl.cachedDialCount;
  out["histogramChunkSize"] = impl.histogramChunkSize;
  out["maxHistogramChunksPerBin"] = impl.maxHistogramChunksPerBin;
  out["lastStageGpuSeconds"] = impl.diagnosticStageGpuSeconds;
  std::map<std::string, std::size_t> dialCounts, types;
  for( const auto& event : impl.model.eventWeights ){ ++dialCounts[std::to_string(event.dialCount)]; }
  const char* dialTypeNames[]{"Norm", "Shift", "CompactSpline", "UniformSpline", "MonotonicSpline", "GeneralSpline", "Graph", "ExternalWeight", "Bilinear", "Bicubic"};
  for( const auto& dial : impl.model.dials ){ ++types[dialTypeNames[std::size_t(dial.type)]]; }
  out["eventsByWeightDialCount"] = dialCounts;
  out["uniqueDialsByType"] = types;
  if( impl.useEventPropagation ){
    out["executor"] = JsonType::parse(impl.eventPropagation->getDiagnostics());
    out["lastStageGpuSeconds"] = out["executor"]["lastStageGpuSeconds"];
    // These belong to the packed weight path, not the event propagation path.
    out.erase("cachedDials");
    out.erase("splineDescriptorBytes");
    out.erase("normOccurrenceBytes");
    out.erase("cachedDialDispatchCount");
    out.erase("maxHistogramChunksPerBin");
  }
  else{
    out["buffers"] = JsonType::array();
    auto buffer = [&](const char* name_, id<MTLBuffer> buffer_){
      if( buffer_ == nil ){ return; }
      out["buffers"].push_back({{"name", name_}, {"bytes", buffer_.length},
        {"storage", buffer_.storageMode == MTLStorageModeShared ? "Shared" : "Private"}});
    };
    buffer("eventWeightsBuffer", impl.eventWeightsBuffer);
    buffer("baseWeightsBuffer", impl.baseWeightsBuffer);
    buffer("eventDialRangesBuffer", impl.eventDialRangesBuffer);
    buffer("normDialOccurrencesBuffer", impl.normDialOccurrencesBuffer);
    buffer("compactDialIndicesBuffer", impl.compactDialIndicesBuffer);
    buffer("uniformDialIndicesBuffer", impl.uniformDialIndicesBuffer);
    buffer("monotonicDialIndicesBuffer", impl.monotonicDialIndicesBuffer);
    buffer("generalDialIndicesBuffer", impl.generalDialIndicesBuffer);
    buffer("graphDialIndicesBuffer", impl.graphDialIndicesBuffer);
    buffer("compactDialDescriptorsBuffer", impl.compactDialDescriptorsBuffer);
    buffer("uniformDialDescriptorsBuffer", impl.uniformDialDescriptorsBuffer);
    buffer("monotonicDialDescriptorsBuffer", impl.monotonicDialDescriptorsBuffer);
    buffer("generalDialDescriptorsBuffer", impl.generalDialDescriptorsBuffer);
    buffer("graphDialDescriptorsBuffer", impl.graphDialDescriptorsBuffer);
    buffer("compactCachedResponsesBuffer", impl.compactCachedResponsesBuffer);
    buffer("compactCachedDialIndicesBuffer", impl.compactCachedDialIndicesBuffer);
    buffer("uniformCachedResponsesBuffer", impl.uniformCachedResponsesBuffer);
    buffer("uniformCachedDialIndicesBuffer", impl.uniformCachedDialIndicesBuffer);
    buffer("monotonicCachedResponsesBuffer", impl.monotonicCachedResponsesBuffer);
    buffer("monotonicCachedDialIndicesBuffer", impl.monotonicCachedDialIndicesBuffer);
    buffer("generalCachedResponsesBuffer", impl.generalCachedResponsesBuffer);
    buffer("generalCachedDialIndicesBuffer", impl.generalCachedDialIndicesBuffer);
    buffer("graphCachedResponsesBuffer", impl.graphCachedResponsesBuffer);
    buffer("graphCachedDialIndicesBuffer", impl.graphCachedDialIndicesBuffer);
    buffer("binEventOffsetsBuffer", impl.binEventOffsetsBuffer);
    buffer("binEventIndicesBuffer", impl.binEventIndicesBuffer);
    buffer("splineDataBuffer", impl.splineDataBuffer);
    buffer("responseSmoothingBuffer", impl.responseSmoothingBuffer);
    buffer("parametersBuffer", impl.parametersBuffer);
    buffer("externalWeightsBuffer", impl.externalWeightsBuffer);
    buffer("externalDialOccurrencesBuffer", impl.externalDialOccurrencesBuffer);
    buffer("partialHistSumsBuffer", impl.partialHistSumsBuffer);
    buffer("partialHistSumSquaresBuffer", impl.partialHistSumSquaresBuffer);
    buffer("histSumsBuffer", impl.histSumsBuffer);
    buffer("histSumSquaresBuffer", impl.histSumSquaresBuffer);
    buffer("eventWeightsReadbackBuffer", impl.eventWeightsReadbackBuffer);
    buffer("histSumsReadbackBuffer", impl.histSumsReadbackBuffer);
    buffer("histSumSquaresReadbackBuffer", impl.histSumSquaresReadbackBuffer);
    buffer("nEventsBuffer", impl.nEventsBuffer);
    buffer("totalBinsBuffer", impl.totalBinsBuffer);
    buffer("maxHistogramChunksPerBinBuffer", impl.maxHistogramChunksPerBinBuffer);
    buffer("histogramChunkSizeBuffer", impl.histogramChunkSizeBuffer);
    out["pipelines"] = JsonType::array();
    auto pipeline = [&](const char* name_, id<MTLComputePipelineState> pipeline_){
      if( pipeline_ == nil ){ return; }
      out["pipelines"].push_back({{"name", name_}, {"simdWidth", pipeline_.threadExecutionWidth},
        {"maxThreadsPerThreadgroup", pipeline_.maxTotalThreadsPerThreadgroup},
        {"dispatchThreadsPerThreadgroup", std::min<NSUInteger>(256, pipeline_.maxTotalThreadsPerThreadgroup)},
        {"staticThreadgroupMemoryBytes", pipeline_.staticThreadgroupMemoryLength}});
    };
    pipeline("eventWeightsPipeline", impl.eventWeightsPipeline);
    pipeline("cachedCompactResponsesPipeline", impl.cachedCompactResponsesPipeline);
    pipeline("cachedUniformResponsesPipeline", impl.cachedUniformResponsesPipeline);
    pipeline("cachedMonotonicResponsesPipeline", impl.cachedMonotonicResponsesPipeline);
    pipeline("cachedGeneralResponsesPipeline", impl.cachedGeneralResponsesPipeline);
    pipeline("cachedGraphResponsesPipeline", impl.cachedGraphResponsesPipeline);
    pipeline("histogramPipeline", impl.histogramPipeline);
    pipeline("histogramPartialsPipeline", impl.histogramPartialsPipeline);
    pipeline("histogramFinalizePipeline", impl.histogramFinalizePipeline);
  }
  return out.dump();
}

std::string Backends::MpsBackend::getDiagnosticStageGpuSeconds() const {
  const auto& timings = _impl_->useEventPropagation ? _impl_->eventPropagation->getDiagnosticStageGpuSeconds()
                                                  : _impl_->diagnosticStageGpuSeconds;
  return JsonType(timings).dump();
}

Backends::BackendResourceUsage Backends::MpsBackend::getResourceUsage() const {
  BackendResourceUsage usage;
  if( not _impl_->isAvailable or not _impl_->isBuilt ){ return usage; }
  usage.available = true;
  usage.deviceName = [[_impl_->device name] UTF8String];
  usage.processMemory = true;
  usage.unifiedMemory = _impl_->device.hasUnifiedMemory;
  usage.allocatedBytes = _impl_->device.currentAllocatedSize;
  usage.recommendedDeviceBytes = _impl_->device.recommendedMaxWorkingSetSize;
  return usage;
}
