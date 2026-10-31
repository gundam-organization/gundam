#include "BackendTypes.h"

#include "GenericToolbox.String.h"

#include <cmath>
#include <iomanip>
#include <sstream>

Backends::OutputState& Backends::PropagationStatus::state(OutputRequest request_) {
  if( request_ == OutputRequest::EventState ){ return eventState; }
  if( request_ == OutputRequest::EventWeights ){ return eventWeights; }
  if( request_ == OutputRequest::Histograms ){ return histograms; }
  if( request_ == OutputRequest::SampleLikelihoods ){ return sampleLikelihoods; }
  if( request_ == OutputRequest::StatLikelihood ){ return statLikelihood; }
  return histograms;
}

Backends::OutputState Backends::PropagationStatus::state(OutputRequest request_) const {
  if( request_ == OutputRequest::EventState ){ return eventState; }
  if( request_ == OutputRequest::EventWeights ){ return eventWeights; }
  if( request_ == OutputRequest::Histograms ){ return histograms; }
  if( request_ == OutputRequest::SampleLikelihoods ){ return sampleLikelihoods; }
  if( request_ == OutputRequest::StatLikelihood ){ return statLikelihood; }
  return OutputState::Failed;
}

std::string Backends::formatBackendMonitorLine(const std::string& backend_,
    const BackendResourceUsage& usage_, const BackendTimingSummary& timing_) {
  if( not usage_.available ){ return {}; }
  std::stringstream line;
  line << "Backend " << backend_.substr(0, backend_.find('/')) << " (" << usage_.deviceName
       << ") / Memory: " << GenericToolbox::parseUnitPrefix(double(usage_.allocatedBytes), 4) << "B";
  if( timing_.propagationWallSeconds > 0 and std::isfinite(timing_.propagationWallSeconds)
      and timing_.devicePropagationSeconds >= 0 and std::isfinite(timing_.devicePropagationSeconds) ){
    // Command/stream intervals include transfers and may include idle gaps;
    // this ratio is not hardware utilization or achieved GPU occupancy.
    const double share = timing_.devicePropagationSeconds / timing_.propagationWallSeconds;
    line << " / On-device time: ";
    if( share <= 1. ){ line << std::fixed << std::setprecision(1) << 100 * share << "%"; }
    else{ line << "n/a"; } // Do not present inconsistent clock intervals as efficiency.
  }
  return line.str();
}
