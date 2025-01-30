//
// Created by Nadrino on 22/07/2021.
//


#include "SampleSet.h"
#include "GundamGlobals.h"

#include "Logger.h"

#include <memory>


void SampleSet::configureImpl(){

  auto sampleListConfig = GenericToolbox::Json::fetchValue(_config_, {{"sampleList"}, {"fitSampleList"}}, JsonType());
  LogDebugIf(GundamGlobals::isDebug()) << sampleListConfig.size() << " samples defined in the config." << std::endl;

  if( _sampleList_.empty() ){
    // from scratch
    _sampleList_.reserve( sampleListConfig.size() );
    int iSample{0};
    for( auto& sampleConfig : sampleListConfig ){
      _sampleList_.emplace_back();
      _sampleList_.back().setIndex( iSample++ );
      _sampleList_.back().configure( sampleConfig );

      LogDebugIf(GundamGlobals::isDebug()) << "Defined sample: " << _sampleList_.back().getName() << std::endl;

      // remove from the list if not enabled
      if( not _sampleList_.back().isEnabled() ){
        LogDebugIf(GundamGlobals::isDebug()) << "-> removing this sample as it is disabled." << std::endl;
        _sampleList_.pop_back(); iSample--;
      }
    }
  }
  else{
    // for temporary config overrides of propagators,
    // we want to read the config without removing the content of the samples

    // need to check how many samples are enabled. It should match the list.
    size_t nSamples{0};
    for(const auto & sampleConfig : sampleListConfig){
      if( not GenericToolbox::Json::fetchValue(sampleConfig, "isEnabled", true) ) continue;
      nSamples++;
    }
    LogThrowIf(nSamples != _sampleList_.size(), "Can't reload config with different number of samples");

    for( size_t iSample = 0 ; iSample < _sampleList_.size() ; iSample++ ){
      if( not GenericToolbox::Json::fetchValue(sampleListConfig[iSample], "isEnabled", true) ) continue;
      _sampleList_[ iSample ].configure( sampleListConfig[iSample] ); // read the config again
    }
  }

  LogDebugIf(GundamGlobals::isDebug()) << sampleListConfig.size() << " samples were defined." << std::endl;
}
void SampleSet::initializeImpl() {
  for( auto& sample : _sampleList_ ){ sample.initialize(); }
}

void SampleSet::clearEventLists(){
  for( auto& sample : _sampleList_ ){ sample.getEventList().clear(); }
}

std::vector<std::string> SampleSet::fetchRequestedVariablesForIndexing() const{
  std::vector<std::string> out;
  for (auto &sample: _sampleList_) {
    for (auto &binContext: sample.getHistogram().getBinContextList()) {
      for (auto &edges: binContext.bin.getEdgesList()) { GenericToolbox::addIfNotInVector(edges.varName, out); }
    }
  }
  return out;
}
void SampleSet::copyEventsFrom(const SampleSet& src_){
  LogThrowIf(
      src_.getSampleList().size() != this->getSampleList().size(),
      "Can't copy events from mismatching sample lists. src(" << src_.getSampleList().size() << ")"
      << "dst(" << this->getSampleList().size() << ")."
  );

  for( size_t iSample = 0 ; iSample < src_.getSampleList().size() ; iSample++ ){
    this->getSampleList()[iSample].getEventList() = src_.getSampleList()[iSample].getEventList();
  }
}
size_t SampleSet::getNbOfEvents() const {
  return std::accumulate(
      _sampleList_.begin(), _sampleList_.end(), size_t(0),
      [](size_t sum_, const Sample& s_){ return sum_ + s_.getEventList().size(); });
}

void SampleSet::printConfiguration() const {

  LogInfo << _sampleList_.size() << " samples defined." << std::endl;
  for( auto& sample : _sampleList_ ){ sample.printConfiguration(); }

}
std::string SampleSet::getSampleBreakdown() const{
  GenericToolbox::TablePrinter t;

  t << "Sample" << GenericToolbox::TablePrinter::NextColumn;
  t << "# of binned event" << GenericToolbox::TablePrinter::NextColumn;
  t << "total rate (weighted)" << GenericToolbox::TablePrinter::NextLine;

  for( auto& sample : _sampleList_ ){
    t << sample.getName() << GenericToolbox::TablePrinter::NextColumn;
    t << sample.getNbBinnedEvents() << GenericToolbox::TablePrinter::NextColumn;
    t << sample.getSumWeights() << GenericToolbox::TablePrinter::NextLine;
  }

  return t.generateTableString();
}
