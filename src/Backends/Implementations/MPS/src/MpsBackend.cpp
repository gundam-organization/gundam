#include "MpsBackend.h"

#include "Logger.h"

Backends::BackendCapabilities Backends::MpsBackend::getCapabilities() const {
  auto out = _hostBackend_.getCapabilities();
  out.supportsGpu = true;
  out.deviceName = "Metal Performance Shaders host bridge";
  return out;
}

void Backends::MpsBackend::build(const BackendModel& model_) {
  _hostBackend_.build(model_);
}

void Backends::MpsBackend::setLikelihoodModel(const BackendLikelihoodModel& likelihoodModel_) {
  _hostBackend_.setLikelihoodModel(likelihoodModel_);
}

Backends::PropagationToken Backends::MpsBackend::requestPropagation(
    const ParameterSnapshot& parameters_,
    const PropagationRequest& request_) {
  return _hostBackend_.requestPropagation(parameters_, request_);
}

Backends::PropagationStatus Backends::MpsBackend::getStatus(const PropagationToken& token_) const {
  return _hostBackend_.getStatus(token_);
}

bool Backends::MpsBackend::isReady(const PropagationToken& token_) const {
  return _hostBackend_.isReady(token_);
}

void Backends::MpsBackend::wait(const PropagationToken& token_) {
  _hostBackend_.wait(token_);
}

void Backends::MpsBackend::materialize(const PropagationToken& token_, OutputRequest output_) {
  _hostBackend_.materialize(token_, output_);
}

double Backends::MpsBackend::getLikelihood(const PropagationToken& token_) const {
  return _hostBackend_.getLikelihood(token_);
}
