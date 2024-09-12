//
// Created by Nadrino on 11/06/2021.
//

#include "Propagator.h"

#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif

#include "ParameterSet.h"
#include "GundamGlobals.h"
#include "ConfigUtils.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Json.h"

#include <memory>
#include <vector>

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[Propagator]"); });
#endif

void Propagator::muteLogger(){ Logger::setIsMuted( true ); }
void Propagator::unmuteLogger(){ Logger::setIsMuted( false ); }

void Propagator::readConfigImpl(){
  LogWarning << __METHOD_NAME__ << std::endl;

  // legacy -- option within propagator -> should be defined elsewhere
  GenericToolbox::Json::deprecatedAction(_config_, "parameterSetListConfig", [&]{
    LogAlert << R"("parameterSetListConfig" should now be set under "parametersManagerConfig/parameterSetList".)" << std::endl;
    auto parameterSetListConfig = GenericToolbox::Json::fetchValue<JsonType>(_config_, "parameterSetListConfig");
    _parManager_.setParameterSetListConfig( ConfigUtils::getForwardedConfig( parameterSetListConfig ) );
  });
  GenericToolbox::Json::deprecatedAction(_config_, "throwToyParametersWithGlobalCov", [&]{
    LogAlert << "Forwarding the option to ParametersManager. Consider moving it into \"parametersManagerConfig:\"" << std::endl;
    _parManager_.setThrowToyParametersWithGlobalCov(GenericToolbox::Json::fetchValue<bool>(_config_, "throwToyParametersWithGlobalCov"));
  });

  // nested objects
  _parManager_.readConfig( GenericToolbox::Json::fetchValue( _config_, "parametersManagerConfig", _parManager_.getConfig()) );

  // Monitoring parameters
  _showEventBreakdown_ = GenericToolbox::Json::fetchValue(_config_, "showEventBreakdown", _showEventBreakdown_);
  _showNbEventParameterBreakdown_ = GenericToolbox::Json::fetchValue(_config_, "showNbEventParameterBreakdown", _showNbEventParameterBreakdown_);
  _showNbEventPerSampleParameterBreakdown_ = GenericToolbox::Json::fetchValue(_config_, "showNbEventPerSampleParameterBreakdown", _showNbEventPerSampleParameterBreakdown_);
  _throwAsimovToyParameters_ = GenericToolbox::Json::fetchValue(_config_, "throwAsimovFitParameters", _throwAsimovToyParameters_);
  _enableStatThrowInToys_ = GenericToolbox::Json::fetchValue(_config_, "enableStatThrowInToys", _enableStatThrowInToys_);
  _gaussStatThrowInToys_ = GenericToolbox::Json::fetchValue(_config_, "gaussStatThrowInToys", _gaussStatThrowInToys_);
  _enableEventMcThrow_ = GenericToolbox::Json::fetchValue(_config_, "enableEventMcThrow", _enableEventMcThrow_);
  _parameterInjectorMc_ = GenericToolbox::Json::fetchValue(_config_, "parameterInjection", _parameterInjectorMc_);

  // debug/dev parameters
  _debugPrintLoadedEvents_ = GenericToolbox::Json::fetchValue(_config_, "debugPrintLoadedEvents", _debugPrintLoadedEvents_);
  _debugPrintLoadedEventsNbPerSample_ = GenericToolbox::Json::fetchValue(_config_, "debugPrintLoadedEventsNbPerSample", _debugPrintLoadedEventsNbPerSample_);
  _devSingleThreadReweight_ = GenericToolbox::Json::fetchValue(_config_, "devSingleThreadReweight", _devSingleThreadReweight_);
  _devSingleThreadHistFill_ = GenericToolbox::Json::fetchValue(_config_, "devSingleThreadHistFill", _devSingleThreadHistFill_);

  // EventDialCache parameters
  if( GenericToolbox::Json::doKeyExist(_config_, "globalEventReweightCap") ){
    _eventDialCache_.getGlobalEventReweightCap().isEnabled = true;
    _eventDialCache_.getGlobalEventReweightCap().maxReweight = GenericToolbox::Json::fetchValue<double>(_config_, "globalEventReweightCap");
  }


  LogInfo << "Reading samples configuration..." << std::endl;
  auto fitSampleSetConfig = GenericToolbox::Json::fetchValue(_config_, {{"sampleSetConfig"}, {"fitSampleSetConfig"}}, JsonType());
  _sampleSet_.setConfig(fitSampleSetConfig);
  _sampleSet_.readConfig();

  LogInfo << "Reading DialCollection configurations..." << std::endl;
  _dialCollectionList_.clear(); // make sure it's empty in case readConfig() is called more than once
  for(size_t iParSet = 0 ; iParSet < _parManager_.getParameterSetsList().size() ; iParSet++ ){
    LogScopeIndent;
    LogInfo << "Reading dial definitions for " << _parManager_.getParameterSetsList()[iParSet].getName() << std::endl;

    if( not _parManager_.getParameterSetsList()[iParSet].isEnabled() ) continue;
    // DEV / DialCollections
    if( not _parManager_.getParameterSetsList()[iParSet].getDialSetDefinitions().empty() ){
      for( auto& dialSetDef : _parManager_.getParameterSetsList()[iParSet].getDialSetDefinitions().get<std::vector<JsonType>>() ){
        _dialCollectionList_.emplace_back(&_parManager_.getParameterSetsList());
        _dialCollectionList_.back().setIndex(int(_dialCollectionList_.size()) - 1);
        _dialCollectionList_.back().setSupervisedParameterSetIndex(int(iParSet) );
        _dialCollectionList_.back().readConfig(dialSetDef );
      }
    }
    else{

      for( auto& par : _parManager_.getParameterSetsList()[iParSet].getParameterList() ){
        if( not par.isEnabled() ) continue;

        // Check if no definition is present -> disable the parameter in that case
        if( par.getDialDefinitionsList().empty() ) {
          LogAlert << "Disabling \"" << par.getFullTitle() << "\": no dial definition." << std::endl;
          par.setIsEnabled(false);
          continue;
        }

        for( const auto& dialDefinitionConfig : par.getDialDefinitionsList() ){
          _dialCollectionList_.emplace_back(&_parManager_.getParameterSetsList());
          _dialCollectionList_.back().setIndex(int(_dialCollectionList_.size()) - 1);
          _dialCollectionList_.back().setSupervisedParameterSetIndex(int(iParSet) );
          _dialCollectionList_.back().setSupervisedParameterIndex(par.getParameterIndex() );
          _dialCollectionList_.back().readConfig(dialDefinitionConfig );
        }
      }
    }
  }

  LogInfo << "Reading config of the Propagator done." << std::endl;
}
void Propagator::initializeImpl(){
  LogWarning << __METHOD_NAME__ << std::endl;

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing parameters...") << std::endl;
  _parManager_.initialize();

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing samples...") << std::endl;
  _sampleSet_.initialize();

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing dials...") << std::endl;
  for( auto& dialCollection : _dialCollectionList_ ){ dialCollection.initialize(); }

  LogInfo << "Initializing propagation threads..." << std::endl;
  initializeThreads();
}

// core
void Propagator::clearContent(){
  LogInfo << "Clearing Propagator content..." << std::endl;

  // clearing events in MC containers
  _sampleSet_.clearMcContainers();

  // also wiping event-by-event dials...
  for( auto& dialCollection: _dialCollectionList_ ) {
    if( not dialCollection.getGlobalDialLeafName().empty() ) { dialCollection.clear(); }

    // clear input buffer cache to trigger the cache eval
    for( auto& dialInput : dialCollection.getDialInputBufferList() ){
      dialInput.invalidateBuffers();
    }
  }
  _eventDialCache_ = EventDialCache();

}
void Propagator::shrinkDialContainers(){
  LogInfo << "Resizing dial containers..." << std::endl;
  for( auto& dialCollection : _dialCollectionList_ ) {
    if( dialCollection.isEventByEvent() ){ dialCollection.resizeContainers(); }
  }
}
void Propagator::buildDialCache(){
  _eventDialCache_.shrinkIndexedCache();
  _eventDialCache_.buildReferenceCache(_sampleSet_, _dialCollectionList_);

  // be extra sure the dial input will request an update
  for( auto& dialCollection : _dialCollectionList_ ){
    for( auto& dialInput : dialCollection.getDialInputBufferList() ){
      dialInput.invalidateBuffers();
    }
  }
}
void Propagator::propagateParameters(){

  if( _enableEigenToOrigInPropagate_ ){
    // Only real parameters are propagated on the spectra -> need to convert the eigen to original
    for( auto& parSet : _parManager_.getParameterSetsList() ){
      if( parSet.isEnableEigenDecomp() ){ parSet.propagateEigenToOriginal(); }
    }
  }

  this->reweightEvents();
  this->refillHistograms();

}
void Propagator::reweightEvents() {
  reweightTimer.start();

  updateDialState();

  bool usedGPU{false};
#ifdef GUNDAM_USING_CACHE_MANAGER
  if( GundamGlobals::getEnableCacheManager() ) {
    if (Cache::Manager::Update(getSampleSet(), getEventDialCache())) {
      usedGPU = Cache::Manager::Fill();
    }
    if (GundamGlobals::getForceDirectCalculation()) usedGPU = false;
  }
#endif
  if( not usedGPU ){
    if( not _devSingleThreadReweight_ ){
      _threadPool_.runJob("Propagator::reweightEvents");
    }
    else{ this->reweightEvents(-1); }
  }

  reweightTimer.stop();
}

// misc
void Propagator::writeEventRates(const GenericToolbox::TFilePath& saveDir_) const {
  for( auto& sample : _sampleSet_.getSampleList() ){ sample.writeEventRates(saveDir_); }
}
void Propagator::printBreakdowns(){

  LogInfo << std::endl << "Breaking down samples..." << std::endl;

  if( _showEventBreakdown_ ){
    LogWarning << "Sample breakdown:" << std::endl;
    std::cout << this->getSampleBreakdownTableStr() << std::endl;
  }

  if( _showNbEventParameterBreakdown_ ){

    struct NbEventBreakdown{
      size_t nbTotal{0};
      std::map<int, int> nbForSample{};
    };

    std::map<const Parameter*, NbEventBreakdown> nbEventForParameter{}; // assuming int = 0 by default
    for( auto& cache: _eventDialCache_.getCache() ){
      for( auto& dial : cache.dialResponseCacheList ){
        for( int iInput = 0 ; iInput < dial.dialInterface.getInputBufferRef()->getInputSize() ; iInput++ ){
          nbEventForParameter[ &dial.dialInterface.getInputBufferRef()->getParameter(iInput) ].nbTotal += 1;

          if( _showNbEventPerSampleParameterBreakdown_ ){
            nbEventForParameter[ &dial.dialInterface.getInputBufferRef()->getParameter(iInput) ]
                .nbForSample[cache.event->getIndices().sample] += 1;
          }
        }
      }
    }

    GenericToolbox::TablePrinter t;
    t << "Parameter";
    t << GenericToolbox::TablePrinter::NextColumn << "All samples";

    if( _showNbEventPerSampleParameterBreakdown_ ){
      for( auto& sample : _sampleSet_.getSampleList() ){
        if( not sample.isEnabled() ){ continue; }
        t << GenericToolbox::TablePrinter::NextColumn << sample.getName();
      }
    }

    t << GenericToolbox::TablePrinter::NextLine;

    for( auto& parSet : _parManager_.getParameterSetsList() ){
      if( not parSet.isEnabled() ){ continue; }
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ){ continue; }

        t.setColorBuffer( GenericToolbox::ColorCodes::resetColor );
        if( nbEventForParameter[ &par ].nbTotal == 0 ){ t.setColorBuffer( GenericToolbox::ColorCodes::redBackground ); }

        t << par.getFullTitle();
        t << GenericToolbox::TablePrinter::NextColumn << nbEventForParameter[ &par ].nbTotal;

        if( _showNbEventPerSampleParameterBreakdown_ ){
          for( auto& sample : _sampleSet_.getSampleList() ){
            if( not sample.isEnabled() ){ continue; }
            t << GenericToolbox::TablePrinter::NextColumn << nbEventForParameter[ &par ].nbForSample[sample.getIndex()];
          }
        }

        t << GenericToolbox::TablePrinter::NextLine;
      }
    }

    LogInfo << "Nb of event affected by parameters:" << std::endl;
    t.printTable();

  }

  if( _debugPrintLoadedEvents_ ){
    LogDebug << "Printing " << _debugPrintLoadedEventsNbPerSample_ << " events..." << std::endl;
    for( int iEvt = 0 ; iEvt < _debugPrintLoadedEventsNbPerSample_ ; iEvt++ ){
      LogDebug << "Event #" << iEvt << "{" << std::endl;
      {
        LogScopeIndent;
        LogDebug << _eventDialCache_.getCache()[iEvt].getSummary() << std::endl;
      }
      LogDebug << "}" << std::endl;
    }
  }
}
std::string Propagator::getSampleBreakdownTableStr() const{
  GenericToolbox::TablePrinter t;

  t << "Sample" << GenericToolbox::TablePrinter::NextColumn;
  t << "MC (# binned event)" << GenericToolbox::TablePrinter::NextColumn;
  t << "Data (# binned event)" << GenericToolbox::TablePrinter::NextColumn;
  t << "MC (weighted)" << GenericToolbox::TablePrinter::NextColumn;
  t << "Data (weighted)" << GenericToolbox::TablePrinter::NextLine;

  for( auto& sample : _sampleSet_.getSampleList() ){
    t << sample.getName() << GenericToolbox::TablePrinter::NextColumn;
    t << sample.getNbBinnedEvents() << GenericToolbox::TablePrinter::NextColumn;
    t << sample.getSumWeights() << GenericToolbox::TablePrinter::NextColumn;
  }

  std::stringstream ss;
  ss << t.generateTableString();
  return ss.str();
}


// Protected
void Propagator::initializeThreads() {

  _threadPool_ = GenericToolbox::ParallelWorker();
  _threadPool_.setNThreads( GundamGlobals::getNumberOfThreads() );

  _threadPool_.addJob(
      "Propagator::reweightEvents",
      [this](int iThread){ this->reweightEvents(iThread); }
  );

  _threadPool_.addJob(
      "Propagator::refillHistograms",
      [this](int iThread){ this->refillHistogramsFct(iThread); }
  );

}

// private
void Propagator::updateDialState(){
  std::for_each(_dialCollectionList_.begin(), _dialCollectionList_.end(),
                [&]( DialCollection& dc_){
                  dc_.updateInputBuffers();
                });
  std::for_each(_dialCollectionList_.begin(), _dialCollectionList_.end(),
                [&]( DialCollection& dc_){
                  dc_.update();
                });
}
void Propagator::refillHistograms(){
  refillHistogramTimer.start();

  if( not _devSingleThreadHistFill_ ){ _threadPool_.runJob("Propagator::refillHistograms"); }
  else{ refillHistogramsFct(-1); }

  refillHistogramTimer.stop();
}

// multithreading
void Propagator::reweightEvents( int iThread_) {

  //! Warning: everything you modify here, may significantly slow down the
  //! fitter

  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, _threadPool_.getNbThreads(),
      int(_eventDialCache_.getCache().size())
  );

  std::for_each(
      _eventDialCache_.getCache().begin() + bounds.beginIndex,
      _eventDialCache_.getCache().begin() + bounds.endIndex,
      [this]( EventDialCache::CacheEntry& cache_){ _eventDialCache_.reweightEntry(cache_); }
  );

}
void Propagator::refillHistogramsFct( int iThread_){
  for( auto& sample : _sampleSet_.getSampleList() ){
    sample.refillHistogram(iThread_);
  }
}

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
