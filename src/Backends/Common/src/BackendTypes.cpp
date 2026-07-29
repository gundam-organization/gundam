#include "BackendTypes.h"

#include <algorithm>

bool Backends::PropagationRequest::has(OutputRequest request_) const {
  return std::find(outputs.begin(), outputs.end(), request_) != outputs.end();
}

Backends::OutputState& Backends::PropagationStatus::state(OutputRequest request_) {
  switch( request_ ){
    case OutputRequest::EventWeights: return eventWeights;
    case OutputRequest::Histograms: return histograms;
    case OutputRequest::Likelihood: return likelihood;
    case OutputRequest::BinIndices: return binIndices;
    case OutputRequest::ObservableValues: return observableValues;
  }
  return histograms;
}

Backends::OutputState Backends::PropagationStatus::state(OutputRequest request_) const {
  switch( request_ ){
    case OutputRequest::EventWeights: return eventWeights;
    case OutputRequest::Histograms: return histograms;
    case OutputRequest::Likelihood: return likelihood;
    case OutputRequest::BinIndices: return binIndices;
    case OutputRequest::ObservableValues: return observableValues;
  }
  return OutputState::Failed;
}
