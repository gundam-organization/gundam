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


LoggerInit([]{ Logger::setUserHeaderStr("[SampleSet]"); });


void SampleSet::readConfigImpl(){
  LogWarning << __METHOD_NAME__ << std::endl;
  LogThrowIf(_config_.empty(), "_config_ is not set." << std::endl);

  _showTimeStats_ = GenericToolbox::Json::fetchValue(_config_, "showTimeStats", _showTimeStats_);

  LogInfo << "Reading samples definition..." << std::endl;
  auto fitSampleListConfig = GenericToolbox::Json::fetchValue(_config_, "fitSampleList", JsonType());
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

  // Fill the bin index inside each event
  std::function<void(int)> updateSampleEventBinIndexesFct = [this](int iThread){
    LogInfoIf(iThread <= 0) << "Updating event sample bin indices..." << std::endl;
    for( auto& sample : _sampleList_ ){
      sample.getMcContainer().updateEventBinIndexes(iThread);
      sample.getDataContainer().updateEventBinIndexes(iThread);
    }
  };
  GundamGlobals::getParallelWorker().addJob("FitSampleSet::updateSampleEventBinIndexes", updateSampleEventBinIndexesFct);

  // Fill bin event caches
  std::function<void(int)> updateSampleBinEventListFct = [this](int iThread){
    LogInfoIf(iThread <= 0) << "Updating sample per bin event lists..." << std::endl;
    for( auto& sample : _sampleList_ ){
      sample.getMcContainer().updateBinEventList(iThread);
      sample.getDataContainer().updateBinEventList(iThread);
    }
  };
  GundamGlobals::getParallelWorker().addJob("FitSampleSet::updateSampleBinEventList", updateSampleBinEventListFct);


  // Histogram fills
  std::function<void(int)> refillMcHistogramsFct = [this](int iThread){
    for( auto& sample : _sampleList_ ){
      sample.getMcContainer().refillHistogram(iThread);
      sample.getDataContainer().refillHistogram(iThread);
    }
  };
  std::function<void()> rescaleMcHistogramsFct = [this](){
    for( auto& sample : _sampleList_ ){
      sample.getMcContainer().rescaleHistogram();
      sample.getDataContainer().rescaleHistogram();
    }
  };
  GundamGlobals::getParallelWorker().addJob("FitSampleSet::updateSampleHistograms", refillMcHistogramsFct);
  GundamGlobals::getParallelWorker().setPostParallelJob("FitSampleSet::updateSampleHistograms", rescaleMcHistogramsFct);
}

void SampleSet::copyMcEventListToDataContainer(){
  for( auto& sample : _sampleList_ ){
    LogInfo << "Copying MC events in sample \"" << sample.getName() << "\"" << std::endl;
    sample.getDataContainer().eventList.clear();
    sample.getDataContainer().eventList.reserve(sample.getMcContainer().eventList.size());
//    sample.getDataContainer().eventList = sample.getMcContainer().eventList;
    sample.getDataContainer().eventList.insert(
        sample.getDataContainer().eventList.begin(),
        std::begin(sample.getMcContainer().eventList),
        std::end(sample.getMcContainer().eventList)
    );
  }
}
void SampleSet::clearMcContainers(){
  for( auto& sample : _sampleList_ ){
    LogInfo << "Clearing event list for \"" << sample.getName() << "\"" << std::endl;
    sample.getMcContainer().eventList.clear();
  }
}


std::vector<std::string> SampleSet::fetchRequestedVariablesForStorage() const{
  std::vector<std::string> out{_additionalVariablesForStorage_};
  // other for storage?
  return out;
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

void SampleSet::updateSampleEventBinIndexes() const{
  GundamGlobals::getParallelWorker().runJob("FitSampleSet::updateSampleEventBinIndexes");
  LogDebugIf(_showTimeStats_) << __METHOD_NAME__ << " took: " << GundamGlobals::getParallelWorker().getLastJobTimer() << std::endl;
}
void SampleSet::updateSampleBinEventList() const{
  GundamGlobals::getParallelWorker().runJob("FitSampleSet::updateSampleBinEventList");
  LogDebugIf(_showTimeStats_) << __METHOD_NAME__ << " took: " << GundamGlobals::getParallelWorker().getLastJobTimer() << std::endl;
}
void SampleSet::updateSampleHistograms() const {
  GundamGlobals::getParallelWorker().runJob("FitSampleSet::updateSampleHistograms");
  LogDebugIf(_showTimeStats_) << __METHOD_NAME__ << " took: " << GundamGlobals::getParallelWorker().getLastJobTimer() << std::endl;
}
