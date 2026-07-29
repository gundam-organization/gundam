#include "BackendConfig.h"

#include "ConfigUtils.h"
#include "Logger.h"

namespace {
  Backends::OutputRequest parseOutputRequest(const std::string& outputRequest_) {
    if( outputRequest_ == "EventWeights" or outputRequest_ == "eventWeights" ){
      return Backends::OutputRequest::EventWeights;
    }
    if( outputRequest_ == "Histograms" or outputRequest_ == "histograms" ){
      return Backends::OutputRequest::Histograms;
    }
    if( outputRequest_ == "Likelihood" or outputRequest_ == "likelihood" ){
      return Backends::OutputRequest::Likelihood;
    }
    if( outputRequest_ == "BinIndices" or outputRequest_ == "binIndices" ){
      return Backends::OutputRequest::BinIndices;
    }
    if( outputRequest_ == "ObservableValues" or outputRequest_ == "observableValues" ){
      return Backends::OutputRequest::ObservableValues;
    }
    LogThrow("Unknown backend output request: " << outputRequest_);
    return Backends::OutputRequest::Histograms;
  }
}

Backends::PropagationRequest Backends::BackendConfig::makePropagationRequest() const {
  PropagationRequest out;
  out.outputs = outputRequests;
  if( out.outputs.empty() ){
    out.outputs.emplace_back(OutputRequest::Histograms);
  }
  out.materializeOutputs = materializeOutputRequests;
  return out;
}

Backends::BackendConfig Backends::BackendConfig::fromConfig(ConfigReader config_) {
  BackendConfig out;

  config_.defineFields({
    {"isEnabled", {"enabled"}},
    {"type", {"backend", "name"}},
    {"outputRequests", {"outputs"}},
    {"materializeOutputRequests", {"materializeOutputs", "hostOutputs"}},
  });

  std::vector<std::string> outputRequestNames{"Histograms"};
  std::vector<std::string> materializeOutputRequestNames{};
  config_.fillValue(out.isEnabled, "isEnabled");
  config_.fillValue(out.type, "type");
  config_.fillValue(outputRequestNames, "outputRequests");
  config_.fillValue(materializeOutputRequestNames, "materializeOutputRequests");
  config_.printUnusedKeys();

  out.outputRequests.clear();
  out.outputRequests.reserve(outputRequestNames.size());
  for( const auto& outputRequestName : outputRequestNames ){
    out.outputRequests.emplace_back(parseOutputRequest(outputRequestName));
  }
  out.materializeOutputRequests.clear();
  out.materializeOutputRequests.reserve(materializeOutputRequestNames.size());
  for( const auto& outputRequestName : materializeOutputRequestNames ){
    out.materializeOutputRequests.emplace_back(parseOutputRequest(outputRequestName));
  }

  return out;
}
