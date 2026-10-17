#include "MpsBackend.h"

#include "MpsBackendInternal.h"
#include "Semantics/BackendHostPropagation.h"

Backends::MpsBackend::MpsBackend() : _impl_(std::make_unique<MpsBackendImpl>()) {}
Backends::MpsBackend::~MpsBackend() = default;

void Backends::MpsBackend::configureImpl() {
  _config_.defineFields({
      {"enableCachedDialResponses"},
  });
  _config_.fillValue(_impl_->enableCachedDialResponses, "enableCachedDialResponses");
  _config_.printUnusedKeys();
}

Backends::BackendCapabilities Backends::MpsBackend::getCapabilities() const {
  BackendCapabilities out;
  out.supportsGpu = true;
  out.supportsDynamicBinning = true;
  out.supportsObservableTransforms = true;
  out.supportsEventWeights = true;
  out.supportsHistograms = true;
  out.supportsLikelihood = true;
  out.deviceName = _impl_->isAvailable ? [[_impl_->device name] UTF8String] : "Metal unavailable";
  return out;
}

void Backends::MpsBackend::build(const EngineView& engineView_) {
  _impl_->isBuilt = false;
  _impl_->engineView = engineView_;
  _impl_->lastResult = MpsBackendImpl::Result();
  _impl_->buildTiming = BackendTimingSummary();
  _impl_->useEventPropagation = engineView_.propagation.requiresEventBinning() or engineView_.propagation.hasReweightCap;
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
    return _impl_->lastResult.token;
  }

  const bool needsEventWeights = true;
  const bool needsHistograms = true;
  bool usedDevicePropagation = false;

  if( needsEventWeights or needsHistograms ){
    usedDevicePropagation = _impl_->runWeightPropagation(inputs_, needsHistograms);
    if( usedDevicePropagation ){
      _impl_->lastResult.status.eventWeights = OutputState::ReadyOnDevice;
      _impl_->lastResult.status.histograms = OutputState::ReadyOnDevice;
    }
  }

  if( not usedDevicePropagation ){
    Semantics::calculateEventWeights(_impl_->lastResult.eventWeights, _impl_->model, inputs_);
    _impl_->lastResult.status.eventWeights = OutputState::ReadyOnHost;
  }

  if( needsHistograms and not usedDevicePropagation ){
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
