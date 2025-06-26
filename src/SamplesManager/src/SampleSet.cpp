//
// Created by Nadrino on 22/07/2021.
//


#include "SampleSet.h"
#include "GundamGlobals.h"
#include "GundamBacktrace.h"

#include "Logger.h"

#include <memory>


void SampleSet::configureImpl(){

  _config_.defineFields({
    {"sampleList", {"fitSampleList"}},
    // deprecated
    {FieldFlag::RELOCATED, "llhStatFunction", "likelihoodInterfaceConfig/jointProbabilityConfig/type"},
    {FieldFlag::RELOCATED, "llhConfig", "likelihoodInterfaceConfig/jointProbabilityConfig"},
  });
  _config_.checkConfiguration();

  auto sampleListConfig = _config_.loop("sampleList");
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
    size_t iSample = 0;
    for(auto & sampleConfig : sampleListConfig){
      Sample::prepareConfig(sampleConfig);
      if( sampleConfig.hasField("isEnabled") and not sampleConfig.fetchValue<bool>("isEnabled") ) {
        continue;
      }
      if( iSample >= _sampleList_.size() ){ continue; }
      _sampleList_[ iSample++ ].configure( sampleConfig ); // read the config again
    }
    LogThrowIf(iSample != _sampleList_.size(), "Can't reload config with different number of samples");
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

  LogInfo << _sampleList_.size() << " samples are defined:" << std::endl;
  GenericToolbox::TablePrinter t;
  t << "Name" << GenericToolbox::TablePrinter::NextColumn;
  t << "Selection" << GenericToolbox::TablePrinter::NextColumn;
  t << "Nb of bins" << GenericToolbox::TablePrinter::NextLine;
  for( auto& sample : _sampleList_ ){
    t << sample.getName() << GenericToolbox::TablePrinter::NextColumn;
    t << sample.getSelectionCutsStr() << GenericToolbox::TablePrinter::NextColumn;
    t << sample.getHistogram().getNbBins() << GenericToolbox::TablePrinter::NextLine;
  }
  t.printTable();

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
