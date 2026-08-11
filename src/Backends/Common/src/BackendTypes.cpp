#include "BackendTypes.h"

#include <sstream>

Backends::OutputState& Backends::PropagationStatus::state(OutputRequest request_) {
  if( request_ == OutputRequest::EventWeights ){ return eventWeights; }
  if( request_ == OutputRequest::Histograms ){ return histograms; }
  if( request_ == OutputRequest::SampleLikelihoods ){ return sampleLikelihoods; }
  if( request_ == OutputRequest::StatLikelihood ){ return statLikelihood; }
  return histograms;
}

Backends::OutputState Backends::PropagationStatus::state(OutputRequest request_) const {
  if( request_ == OutputRequest::EventWeights ){ return eventWeights; }
  if( request_ == OutputRequest::Histograms ){ return histograms; }
  if( request_ == OutputRequest::SampleLikelihoods ){ return sampleLikelihoods; }
  if( request_ == OutputRequest::StatLikelihood ){ return statLikelihood; }
  return OutputState::Failed;
}
