//
// Created by Nadrino on 22/07/2021.
//


#include "SampleSet.h"

#include "GundamGlobals.h"

#include "GenericToolbox.Json.h"
#include "Logger.h"

#include <TTreeFormulaManager.h>
#include "nlohmann/json.hpp"

#include <memory>


void SampleSet::readConfigImpl(){
  LogWarning << __METHOD_NAME__ << std::endl;
  LogThrowIf(_config_.empty(), "_config_ is not set." << std::endl);

  _showTimeStats_ = GenericToolbox::Json::fetchValue(_config_, "showTimeStats", _showTimeStats_);

  LogInfo << "Reading samples definition..." << std::endl;
  _sampleList_.clear(); // make sure we start from scratch in case readConfig is called twice
  auto fitSampleListConfig = GenericToolbox::Json::fetchValue(_config_, {{"sampleList"}, {"fitSampleList"}}, JsonType());
  for( const auto& fitSampleConfig: fitSampleListConfig ){
    if( not GenericToolbox::Json::fetchValue(fitSampleConfig, "isEnabled", true) ) continue;
    _sampleList_.emplace_back();
    _sampleList_.back().setIndex(int(_sampleList_.size()) - 1);
    _sampleList_.back().setConfig(fitSampleConfig);
    _sampleList_.back().readConfig();
  }
}
void SampleSet::initializeImpl() {
  LogWarning << __METHOD_NAME__ << std::endl;
  LogThrowIf(_sampleList_.empty(), "No sample is defined.");

  for( auto& sample : _sampleList_ ){ sample.initialize(); }
}

void SampleSet::copyMcEventListToDataContainer(){
  for( auto& sample : _sampleList_ ){
    LogInfo << "Copying MC events in sample \"" << sample.getName() << "\"" << std::endl;
    sample.getDataContainer().getEventList().clear();
    sample.getDataContainer().getEventList().reserve(sample.getMcContainer().getEventList().size());
//    sample.getDataContainer().getEventList() = sample.getMcContainer().getEventList();
    sample.getDataContainer().getEventList().insert(
        sample.getDataContainer().getEventList().begin(),
        std::begin(sample.getMcContainer().getEventList()),
        std::end(sample.getMcContainer().getEventList())
    );
  }
}
void SampleSet::clearMcContainers(){
  for( auto& sample : _sampleList_ ){
    LogInfo << "Clearing event list for \"" << sample.getName() << "\"" << std::endl;
    sample.getMcContainer().getEventList().clear();
  }
}

std::vector<std::string> SampleSet::fetchRequestedVariablesForIndexing() const{
  std::vector<std::string> out;
  for (auto &sample: _sampleList_) {
    for (auto &bin: sample.getBinning().getBinList()) {
      for (auto &edges: bin.getEdgesList()) { GenericToolbox::addIfNotInVector(edges.varName, out); }
    }
  }
  return out;
}

