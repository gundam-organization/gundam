#include "MpsBackend.h"

#include "MpsBackendInternal.h"
#include "Semantics/BackendHostPropagation.h"

Backends::MpsBackend::MpsBackend() : _impl_(std::make_unique<MpsBackendImpl>()) {}
Backends::MpsBackend::~MpsBackend() = default;

Backends::BackendCapabilities Backends::MpsBackend::getCapabilities() const {
  BackendCapabilities out;
  out.supportsGpu = true;
  out.supportsEventWeights = true;
  out.supportsHistograms = true;
  out.supportsLikelihood = true;
  out.deviceName = _impl_->isAvailable ? [[_impl_->device name] UTF8String] : "Metal unavailable";
  return out;
}

void Backends::MpsBackend::build(const EngineView& engineView_) {
  _impl_->engineView = engineView_;
  _impl_->lastResult = MpsBackendImpl::Result();
  if( not _impl_->buildDeviceModel() ){
    LogWarning << "MPS backend cannot use the GPU propagation path: "
               << (_impl_->deviceModelFallbackReason.empty() ? "unknown compatibility issue."
                                                             : _impl_->deviceModelFallbackReason)
               << " Falling back to the standard backend path for unsupported calculations."
               << std::endl;
  }
  _impl_->isBuilt = true;
}

Backends::PropagationToken Backends::MpsBackend::requestPropagation(const ParameterSnapshot& parameters_) {
  LogThrowIf(not _impl_->isBuilt, "MpsBackend has not been built.");
  LogThrowIf(not parameters_.empty() and parameters_.values.size() != _impl_->model.parameterCount,
             "ParameterSnapshot size mismatch: " << parameters_.values.size()
                                                 << " != " << _impl_->model.parameterCount);

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

  const bool needsEventWeights = true;
  const bool needsHistograms = true;
  bool usedDevicePropagation = false;

  if( needsEventWeights or needsHistograms ){
    usedDevicePropagation = _impl_->runDevicePropagation(parameters_, needsHistograms);
    if( usedDevicePropagation ){
      _impl_->lastResult.status.eventWeights = OutputState::ReadyOnDevice;
      _impl_->lastResult.status.histograms = OutputState::ReadyOnDevice;
    }
  }

  if( not usedDevicePropagation ){
    Semantics::calculateEventWeights(_impl_->lastResult.eventWeights, _impl_->model, parameters_);
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

  if( output_ == OutputRequest::EventWeights ){
    _impl_->materializeEventWeights();
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
