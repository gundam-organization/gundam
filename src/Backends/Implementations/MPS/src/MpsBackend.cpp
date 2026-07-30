#include "MpsBackend.h"

#include "Logger.h"

struct Backends::MpsBackend::Impl {
  PropagationStatus status{};
};

Backends::MpsBackend::MpsBackend() : _impl_(std::make_unique<Impl>()) {}
Backends::MpsBackend::~MpsBackend() = default;

Backends::BackendCapabilities Backends::MpsBackend::getCapabilities() const {
  BackendCapabilities out;
  out.supportsGpu = true;
  out.deviceName = "Metal Performance Shaders unavailable on this platform";
  return out;
}

void Backends::MpsBackend::build(const BackendModel&) {
  _impl_->status = PropagationStatus();
  _impl_->status.backend = BackendStatus::Unavailable;
}

void Backends::MpsBackend::setLikelihoodModel(const BackendLikelihoodModel&) {
}

Backends::PropagationToken Backends::MpsBackend::requestPropagation(const ParameterSnapshot&) {
  _impl_->status = PropagationStatus();
  _impl_->status.backend = BackendStatus::Unavailable;
  _impl_->status.eventWeights = OutputState::Failed;
  _impl_->status.histograms = OutputState::Failed;
  _impl_->status.sampleLikelihoods = OutputState::Failed;
  _impl_->status.statLikelihood = OutputState::Failed;
  return {};
}

Backends::PropagationStatus Backends::MpsBackend::getStatus(const PropagationToken&) const {
  return _impl_->status;
}

bool Backends::MpsBackend::isReady(const PropagationToken&) const {
  return false;
}

void Backends::MpsBackend::wait(const PropagationToken&) {
}

void Backends::MpsBackend::materialize(const PropagationToken&, OutputRequest) {
  LogThrow("MpsBackend is unavailable on this platform.");
}

double Backends::MpsBackend::getLikelihood(const PropagationToken&) const {
  LogThrow("MpsBackend likelihood is unavailable on this platform.");
  return 0;
}

Backends::BackendDeviceView Backends::MpsBackend::getDeviceView(const PropagationToken&) const {
  return {};
}

Backends::BackendTimingSummary Backends::MpsBackend::getLastTimingSummary() const {
  return {};
}
