#include "BackendManager.h"

#include "Logger.h"

void Backends::BackendManager::setBackend(std::unique_ptr<IPropagationBackend> backend_) {
  _backend_ = std::move(backend_);
}

void Backends::BackendManager::build(const BackendModel& model_) {
  LogThrowIf(_backend_ == nullptr, "No backend selected.");
  _backend_->build(model_);
}

Backends::PropagationToken Backends::BackendManager::requestPropagation(
    const ParameterSnapshot& parameters_,
    const PropagationRequest& request_) {
  LogThrowIf(_backend_ == nullptr, "No backend selected.");
  return _backend_->requestPropagation(parameters_, request_);
}

void Backends::BackendManager::wait(const PropagationToken& token_) {
  LogThrowIf(_backend_ == nullptr, "No backend selected.");
  _backend_->wait(token_);
}

void Backends::BackendManager::materialize(const PropagationToken& token_, OutputRequest output_) {
  LogThrowIf(_backend_ == nullptr, "No backend selected.");
  _backend_->materialize(token_, output_);
}
