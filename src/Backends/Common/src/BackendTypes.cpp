#include "BackendTypes.h"

#include <algorithm>
#include <sstream>

std::string Backends::toString(OutputRequest request_) {
  if( request_ == OutputRequest::EventWeights ){ return "EventWeights"; }
  if( request_ == OutputRequest::Histograms ){ return "Histograms"; }
  if( request_ == OutputRequest::Likelihood ){ return "Likelihood"; }
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
  if( request_ == OutputRequest::EventWeights ){ return eventWeights; }
  if( request_ == OutputRequest::Histograms ){ return histograms; }
  if( request_ == OutputRequest::Likelihood ){ return likelihood; }
  return histograms;
}

Backends::OutputState Backends::PropagationStatus::state(OutputRequest request_) const {
  if( request_ == OutputRequest::EventWeights ){ return eventWeights; }
  if( request_ == OutputRequest::Histograms ){ return histograms; }
  if( request_ == OutputRequest::Likelihood ){ return likelihood; }
  return OutputState::Failed;
}
