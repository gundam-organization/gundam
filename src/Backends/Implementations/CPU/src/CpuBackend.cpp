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
  out.deviceName = "host";
  return out;
}

void Backends::CpuBackend::build(const BackendModel& model_) {
  _model_ = model_;
  _lastResult_ = Result();
  _isBuilt_ = true;
}

Backends::PropagationToken Backends::CpuBackend::requestPropagation(
    const ParameterSnapshot& parameters_,
    const PropagationRequest& request_) {

  LogThrowIf(not _isBuilt_, "CpuBackend has not been built.");
  LogThrowIf(not parameters_.empty() and parameters_.values.size() != _model_.parameters.size(),
             "ParameterSnapshot size mismatch: " << parameters_.values.size()
                                                 << " != " << _model_.parameters.size());

  resetResult(request_);

  applyParameterSnapshot(parameters_);
  updateInputBuffers();

  if( request_.has(OutputRequest::EventWeights) ){
    calculateEventWeights(_lastResult_);
    _lastResult_.status.eventWeights = OutputState::ReadyOnDevice;
  }

  if( request_.has(OutputRequest::Histograms) ){
    if( request_.has(OutputRequest::EventWeights) ){
      calculateHistograms(_lastResult_);
    }
    else{
      calculateHistogramsFromEvents(_lastResult_);
    }
    _lastResult_.status.histograms = OutputState::ReadyOnDevice;
  }

  if( request_.has(OutputRequest::Likelihood) ){
    _lastResult_.status.likelihood = OutputState::Failed;
  }
  if( request_.has(OutputRequest::BinIndices) ){
    _lastResult_.status.binIndices = OutputState::Failed;
  }
  if( request_.has(OutputRequest::ObservableValues) ){
    _lastResult_.status.observableValues = OutputState::Failed;
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
    materializeEventWeights(_lastResult_);
    _lastResult_.status.eventWeights = OutputState::ReadyOnHost;
  }
  else if( output_ == OutputRequest::Histograms ){
    materializeHistograms(_lastResult_);
    _lastResult_.status.histograms = OutputState::ReadyOnHost;
  }
  else{
    LogThrow("CpuBackend cannot materialize requested output yet.");
  }
}

bool Backends::CpuBackend::isCurrentToken(const PropagationToken& token_) const {
  return token_.isValid and _lastResult_.token.isValid and token_.id == _lastResult_.token.id;
}

void Backends::CpuBackend::applyParameterSnapshot(const ParameterSnapshot& parameters_) {
  if( parameters_.empty() ){ return; }

  for( std::size_t iPar = 0 ; iPar < parameters_.values.size() ; iPar++ ){
    auto* parPtr = const_cast<Parameter*>(_model_.parameters.at(iPar));
    parPtr->setParameterValue(parameters_.values.at(iPar), true);
  }
}

void Backends::CpuBackend::resetResult(const PropagationRequest& request_) {
  _lastResult_.token.id = _nextTokenId_++;
  _lastResult_.token.isValid = true;
  _lastResult_.status = PropagationStatus();
  _lastResult_.status.backend = BackendStatus::Running;

  for( auto request : request_.outputs ){
    _lastResult_.status.state(request) = OutputState::Scheduled;
  }
}

void Backends::CpuBackend::updateInputBuffers() {
  for( const auto* inputBuffer : _model_.inputBuffers ){
    const_cast<DialInputBuffer*>(inputBuffer)->update();
  }
}

void Backends::CpuBackend::calculateEventWeights(Result& result_) {
  result_.eventWeights.resize(_model_.events.size());

  for( const auto& event : _model_.events ){
    double weight = event.baseWeight;

    for( std::size_t iDial = 0 ; iDial < event.dialCount ; iDial++ ){
      const auto& dialRef = _model_.eventDials[event.firstDial + iDial];
      weight *= dialRef.interface->evalResponse();
    }

    result_.eventWeights[event.resultIndex] = weight;
  }
}

void Backends::CpuBackend::calculateHistograms(Result& result_) {
  if( result_.eventWeights.empty() ){
    calculateEventWeights(result_);
  }

  result_.histSums.resize(_model_.totalBins);
  result_.histSumSquares.resize(_model_.totalBins);
  std::fill(result_.histSums.begin(), result_.histSums.end(), 0);
  std::fill(result_.histSumSquares.begin(), result_.histSumSquares.end(), 0);

  for( const auto& event : _model_.events ){
    int globalBin = event.globalBinIndex;
    if( globalBin < 0 ){ continue; }
    double weight = result_.eventWeights[event.resultIndex];
    result_.histSums[globalBin] += weight;
    result_.histSumSquares[globalBin] += weight * weight;
  }
}

void Backends::CpuBackend::calculateHistogramsFromEvents(Result& result_) {
  result_.histSums.resize(_model_.totalBins);
  result_.histSumSquares.resize(_model_.totalBins);
  std::fill(result_.histSums.begin(), result_.histSums.end(), 0);
  std::fill(result_.histSumSquares.begin(), result_.histSumSquares.end(), 0);

  for( const auto& event : _model_.events ){
    int globalBin = event.globalBinIndex;
    if( globalBin < 0 ){ continue; }

    double weight = event.baseWeight;
    for( std::size_t iDial = 0 ; iDial < event.dialCount ; iDial++ ){
      const auto& dialRef = _model_.eventDials[event.firstDial + iDial];
      weight *= dialRef.interface->evalResponse();
    }

    result_.histSums[globalBin] += weight;
    result_.histSumSquares[globalBin] += weight * weight;
  }
}

void Backends::CpuBackend::materializeEventWeights(Result& result_) {
  LogThrowIf(result_.eventWeights.size() != _model_.events.size());
  for( const auto& event : _model_.events ){
    event.event->getWeights().current = result_.eventWeights[event.resultIndex];
  }
}

void Backends::CpuBackend::materializeHistograms(Result& result_) {
  LogThrowIf(result_.histSums.size() != std::size_t(_model_.totalBins));
  LogThrowIf(result_.histSumSquares.size() != std::size_t(_model_.totalBins));

  for( const auto& sample : _model_.samples ){
    auto& binContentList = sample.histogram->getBinContentList();
    auto& binContextList = sample.histogram->getBinContextList();

    for( auto& binContent : binContentList ){
      binContent.sumWeights = 0;
      binContent.sqrtSumSqWeights = 0;
    }

    for( auto& binContext : binContextList ){
      int globalBin = sample.binOffset + binContext.bin.getIndex();
      auto& binContent = binContentList[binContext.bin.getIndex()];
      binContent.sumWeights = result_.histSums[globalBin];
      binContent.sqrtSumSqWeights = std::sqrt(result_.histSumSquares[globalBin]);
    }
  }
}
