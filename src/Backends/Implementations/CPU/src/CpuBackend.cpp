#include "CpuBackend.h"

#include "DialInputBuffer.h"
#include "DialInterface.h"
#include "Event.h"
#include "Histogram.h"
#include "Parameter.h"

#include "Logger.h"

#include <cmath>
#include <algorithm>

Backends::BackendCapabilities Backends::CpuBackend::getCapabilities() const {
  BackendCapabilities out;
  out.supportsCpu = true;
  out.supportsEventWeights = true;
  out.supportsHistograms = true;
  out.supportsLikelihood = true;
  out.deviceName = "host";
  return out;
}

void Backends::CpuBackend::build(const BackendEngineView& engineView_) {
  _engineView_ = engineView_;
  _lastResult_ = Result();
  _isBuilt_ = true;
}

Backends::PropagationToken Backends::CpuBackend::requestPropagation(const ParameterSnapshot& parameters_) {

  LogThrowIf(not _isBuilt_, "CpuBackend has not been built.");
  const auto& model = _engineView_.propagation;
  const auto& likelihoodModel = _engineView_.likelihood;
  LogThrowIf(not parameters_.empty() and parameters_.values.size() != model.parameters.size(),
             "ParameterSnapshot size mismatch: " << parameters_.values.size()
                                                 << " != " << model.parameters.size());

  resetResult();

  applyParameterSnapshot(parameters_);
  updateInputBuffers();

  calculateEventWeights(_lastResult_);
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

void Backends::CpuBackend::applyParameterSnapshot(const ParameterSnapshot& parameters_) {
  if( parameters_.empty() ){ return; }

  for( std::size_t iPar = 0 ; iPar < parameters_.values.size() ; iPar++ ){
    auto* parPtr = const_cast<Parameter*>(_engineView_.propagation.parameters.at(iPar));
    parPtr->setParameterValue(parameters_.values.at(iPar), true);
  }
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

void Backends::CpuBackend::updateInputBuffers() {
  for( const auto* inputBuffer : _engineView_.propagation.inputBuffers ){
    const_cast<DialInputBuffer*>(inputBuffer)->update();
  }
}

void Backends::CpuBackend::calculateEventWeights(Result& result_) {
  const auto& model = _engineView_.propagation;
  result_.eventWeights.resize(model.events.size());

  for( const auto& event : model.events ){
    double weight = event.baseWeight;

    for( std::size_t iDial = 0 ; iDial < event.dialCount ; iDial++ ){
      const auto& dialRef = model.eventDials[event.firstDial + iDial];
      weight *= dialRef.interface->evalResponse();
    }

    result_.eventWeights[event.resultIndex] = weight;
  }
}

void Backends::CpuBackend::calculateHistograms(Result& result_) {
  if( result_.eventWeights.empty() ){
    calculateEventWeights(result_);
  }

  const auto& model = _engineView_.propagation;
  result_.histSums.resize(model.totalBins);
  result_.histSumSquares.resize(model.totalBins);
  std::fill(result_.histSums.begin(), result_.histSums.end(), 0);
  std::fill(result_.histSumSquares.begin(), result_.histSumSquares.end(), 0);

  for( const auto& event : model.events ){
    int globalBin = event.globalBinIndex;
    if( globalBin < 0 ){ continue; }
    double weight = result_.eventWeights[event.resultIndex];
    result_.histSums[globalBin] += weight;
    result_.histSumSquares[globalBin] += weight * weight;
  }
}

void Backends::CpuBackend::calculateHistogramsFromEvents(Result& result_) {
  const auto& model = _engineView_.propagation;
  result_.histSums.resize(model.totalBins);
  result_.histSumSquares.resize(model.totalBins);
  std::fill(result_.histSums.begin(), result_.histSums.end(), 0);
  std::fill(result_.histSumSquares.begin(), result_.histSumSquares.end(), 0);

  for( const auto& event : model.events ){
    int globalBin = event.globalBinIndex;
    if( globalBin < 0 ){ continue; }

    double weight = event.baseWeight;
    for( std::size_t iDial = 0 ; iDial < event.dialCount ; iDial++ ){
      const auto& dialRef = model.eventDials[event.firstDial + iDial];
      weight *= dialRef.interface->evalResponse();
    }

    result_.histSums[globalBin] += weight;
    result_.histSumSquares[globalBin] += weight * weight;
  }
}

void Backends::CpuBackend::calculateLikelihood(Result& result_) {
  const auto& likelihoodModel = _engineView_.likelihood;
  result_.likelihood = 0;

  for( const auto& sample : likelihoodModel.samples ){
    for( int iBin = 0 ; iBin < int(sample.dataSums.size()) ; iBin++ ){
      if( iBin < int(sample.ignoredBins.size()) and sample.ignoredBins[iBin] ){ continue; }
      int globalBin = sample.binOffset + iBin;
      double pred = result_.histSums[globalBin];
      double predErr = std::sqrt(result_.histSumSquares[globalBin]);
      result_.likelihood += sample.evalBin(sample.dataSums[iBin], pred, predErr, iBin);
    }
  }
}
