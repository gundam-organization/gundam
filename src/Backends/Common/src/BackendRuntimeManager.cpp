#include "BackendRuntimeManager.h"

#include "Logger.h"

void Backends::BackendRuntimeManager::setBackend(std::unique_ptr<IPropagationBackend> backend_) {
  _backend_ = std::move(backend_);
}

void Backends::BackendRuntimeManager::build(const EngineView& engineView_) {
  LogThrowIf(_backend_ == nullptr, "No backend selected.");
  _backend_->build(engineView_);
}

Backends::PropagationToken Backends::BackendRuntimeManager::requestPropagation(const ParameterSnapshot& parameters_) {
  LogThrowIf(_backend_ == nullptr, "No backend selected.");
  return _backend_->requestPropagation(parameters_);
}

void Backends::BackendRuntimeManager::wait(const PropagationToken& token_) {
  LogThrowIf(_backend_ == nullptr, "No backend selected.");
  _backend_->wait(token_);
}

void Backends::BackendRuntimeManager::materialize(const PropagationToken& token_, OutputRequest output_) {
  LogThrowIf(_backend_ == nullptr, "No backend selected.");
  _backend_->materialize(token_, output_);
}
