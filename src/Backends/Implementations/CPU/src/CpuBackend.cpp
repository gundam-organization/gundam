#include "CpuBackend.h"

#include "Semantics/BackendHostPropagation.h"
#include "Logger.h"

Backends::BackendCapabilities Backends::CpuBackend::getCapabilities() const {
  BackendCapabilities out;
  out.supportsCpu = true;
  out.supportsEventWeights = true;
  out.supportsHistograms = true;
  out.supportsLikelihood = true;
  out.deviceName = "host";
  return out;
}

void Backends::CpuBackend::build(const EngineView& engineView_) {
  _engineView_ = engineView_;
  _lastResult_ = Result();
  _isBuilt_ = true;
}

Backends::PropagationToken Backends::CpuBackend::requestPropagation(const ParameterSnapshot& parameters_) {

  LogThrowIf(not _isBuilt_, "CpuBackend has not been built.");
  const auto& model = _engineView_.propagation;
  const auto& likelihoodModel = _engineView_.likelihood;
  LogThrowIf(not parameters_.empty() and parameters_.values.size() != model.parameterCount,
             "ParameterSnapshot size mismatch: " << parameters_.values.size()
                                                 << " != " << model.parameterCount);

  resetResult();

  calculateEventWeights(_lastResult_, parameters_);
  _lastResult_.status.eventWeights = OutputState::ReadyOnDevice;

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

  if( output_ == OutputRequest::EventWeights ){
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

void Backends::CpuBackend::resetResult() {
  _lastResult_.token.id = _nextTokenId_++;
  _lastResult_.token.isValid = true;
  _lastResult_.status = PropagationStatus();
  _lastResult_.status.backend = BackendStatus::Running;
  _lastResult_.eventWeights.clear();
  _lastResult_.histSums.clear();
  _lastResult_.histSumSquares.clear();
  _lastResult_.likelihood = 0;
  _lastResult_.status.eventWeights = OutputState::Scheduled;
  _lastResult_.status.histograms = OutputState::Scheduled;
  _lastResult_.status.sampleLikelihoods = OutputState::Scheduled;
  _lastResult_.status.statLikelihood = OutputState::Scheduled;
}

void Backends::CpuBackend::calculateEventWeights(Result& result_, const ParameterSnapshot& parameters_) {
  Semantics::calculateEventWeights(result_.eventWeights, _engineView_.propagation, parameters_);
}

void Backends::CpuBackend::calculateHistograms(Result& result_) {
  LogThrowIf(result_.eventWeights.empty(), "CPU backend histogram build requires event weights.");
  Semantics::calculateHistogramsFromEventWeights(
      result_.histSums,
      result_.histSumSquares,
      _engineView_.propagation,
      result_.eventWeights
  );
}

void Backends::CpuBackend::calculateHistogramsFromEvents(Result& result_, const ParameterSnapshot& parameters_) {
  Semantics::calculateHistograms(
      result_.histSums,
      result_.histSumSquares,
      _engineView_.propagation,
      parameters_
  );
}

void Backends::CpuBackend::calculateLikelihood(Result& result_) {
  result_.likelihood = Semantics::calculateLikelihood(
      _engineView_.likelihood,
      result_.histSums,
      result_.histSumSquares
  );
}
