#include "MpsBackend.h"

#include "Logger.h"

Backends::BackendCapabilities Backends::MpsBackend::getCapabilities() const {
  BackendCapabilities out;
  out.supportsGpu = true;
  out.deviceName = "Metal Performance Shaders";
  return out;
}

void Backends::MpsBackend::build(const BackendModel& model_) {
  _model_ = model_;
  _status_ = PropagationStatus();
  _status_.backend = BackendStatus::Unavailable;
}

void Backends::MpsBackend::setLikelihoodModel(const BackendLikelihoodModel&) {
}

Backends::PropagationToken Backends::MpsBackend::requestPropagation(
    const ParameterSnapshot&,
    const PropagationRequest& request_) {
  _status_ = PropagationStatus();
  _status_.backend = BackendStatus::Unavailable;
  for( auto request : request_.outputs ){
    _status_.state(request) = OutputState::Failed;
  }
  return {};
}

Backends::PropagationStatus Backends::MpsBackend::getStatus(const PropagationToken&) const {
  return _status_;
}

bool Backends::MpsBackend::isReady(const PropagationToken&) const {
  return false;
}

void Backends::MpsBackend::wait(const PropagationToken&) {
}

void Backends::MpsBackend::materialize(const PropagationToken&, OutputRequest) {
  LogThrow("MpsBackend is declared but device kernels are not implemented yet.");
}

double Backends::MpsBackend::getLikelihood(const PropagationToken&) const {
  LogThrow("MpsBackend likelihood is not implemented yet.");
  return 0;
}
