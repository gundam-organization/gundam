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


#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[SampleSet]"); });
#endif


void SampleSet::readConfigImpl(){
  LogWarning << __METHOD_NAME__ << std::endl;
  LogThrowIf(_config_.empty(), "_config_ is not set." << std::endl);

  _showTimeStats_ = GenericToolbox::Json::fetchValue(_config_, "showTimeStats", _showTimeStats_);

  LogInfo << "Reading samples definition..." << std::endl;
  auto sampleListConfig = GenericToolbox::Json::fetchValue(_config_, {{"sampleList"}, {"fitSampleList"}}, JsonType());

  if( _sampleList_.empty() ){
    // alright no problem, it's from scratch
    _sampleList_.resize( sampleListConfig.size() );
  }
  else{
    // for temporary propagators, we want to read the config without removing the content of the samples

    // need to check how many samples are enabled. It should match the list.
    size_t nSamples{0};
    for(const auto & sampleConfig : sampleListConfig){
      if( not GenericToolbox::Json::fetchValue(sampleConfig, "isEnabled", true) ) continue;
      nSamples++;
    }
    LogThrowIf(nSamples != _sampleList_.size(), "Can't reload config with different number of samples");
  }

  for( size_t iSample = 0 ; iSample < sampleListConfig.size() ; iSample++ ){
    if( not GenericToolbox::Json::fetchValue(sampleListConfig[iSample], "isEnabled", true) ) continue;
    _sampleList_[iSample].setIndex( int(iSample) );
    _sampleList_[iSample].readConfig( sampleListConfig[iSample] );
  }
}
void SampleSet::initializeImpl() {
  LogWarning << __METHOD_NAME__ << std::endl;
  LogThrowIf(_sampleList_.empty(), "No sample is defined.");

  for( auto& sample : _sampleList_ ){ sample.initialize(); }
}

void SampleSet::copyMcEventListToDataContainer(std::vector<Sample>& destinationSampleList_){
  LogThrowIf(_sampleList_.size() != destinationSampleList_.size(), "Can't copy the data into mismatching containers.");
  for( size_t iSample = 0 ; iSample < _sampleList_.size() ; iSample++ ){
    LogInfo << "Copying events in sample \"" << _sampleList_[iSample].getName() << "\"" << std::endl;
    destinationSampleList_[iSample].getEventList().reserve(
        destinationSampleList_[iSample].getEventList().size()
        + _sampleList_[iSample].getEventList().size()
    );
    destinationSampleList_[iSample].getEventList().insert(
        destinationSampleList_[iSample].getEventList().end(),
        std::begin(_sampleList_[iSample].getEventList()),
        std::end(_sampleList_[iSample].getEventList())
    );
  }
}
void SampleSet::clearMcContainers(){
  for( auto& sample : _sampleList_ ){
    LogInfo << "Clearing event list for \"" << sample.getName() << "\"" << std::endl;
    sample.getEventList().clear();
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

