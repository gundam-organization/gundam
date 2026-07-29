#include "BackendTypes.h"

#include <algorithm>
#include <sstream>

std::string Backends::toString(OutputRequest request_) {
  switch( request_ ){
    case OutputRequest::EventWeights: return "EventWeights";
    case OutputRequest::Histograms: return "Histograms";
    case OutputRequest::Likelihood: return "Likelihood";
    case OutputRequest::BinIndices: return "BinIndices";
    case OutputRequest::ObservableValues: return "ObservableValues";
  }
  return "Unknown";
}

std::string Backends::toString(const PropagationRequest& request_) {
  std::stringstream ss;
  ss << "[";
  for( std::size_t iOutput = 0 ; iOutput < request_.outputs.size() ; iOutput++ ){
    if( iOutput != 0 ){ ss << ", "; }
    ss << toString(request_.outputs[iOutput]);
  }
  ss << "]";
  return ss.str();
}

bool Backends::PropagationRequest::has(OutputRequest request_) const {
  return std::find(outputs.begin(), outputs.end(), request_) != outputs.end();
}

bool Backends::PropagationRequest::shouldMaterialize(OutputRequest request_) const {
  if( materializeOutputs.empty() ){
    return has(request_);
  }
  return std::find(materializeOutputs.begin(), materializeOutputs.end(), request_) != materializeOutputs.end();
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
