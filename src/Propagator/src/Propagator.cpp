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

LoggerInit([]{
  Logger::setUserHeaderStr("[Propagator]");
});

void Propagator::muteLogger(){ Logger::setIsMuted( true ); }
void Propagator::unmuteLogger(){ Logger::setIsMuted( false ); }

using namespace GenericToolbox::ColorCodes;

void Propagator::readConfigImpl(){
  LogWarning << __METHOD_NAME__ << std::endl;

  // legacy -- option within propagator -> should be defined elsewhere
  GenericToolbox::Json::deprecatedAction(_config_, "parameterSetListConfig", [&]{
    LogAlert << R"("parameterSetListConfig" should now be set under "parametersManagerConfig/parameterSetList".)" << std::endl;
    auto parameterSetListConfig = GenericToolbox::Json::fetchValue<JsonType>(_config_, "parameterSetListConfig");
    _parManager_.setParameterSetListConfig( ConfigUtils::getForwardedConfig( parameterSetListConfig ) );
  });
  GenericToolbox::Json::deprecatedAction(_config_, "reThrowParSetIfOutOfBounds", [&]{
    LogAlert << "Forwarding the option to ParametersManager. Consider moving it into \"parametersManagerConfig:\"" << std::endl;
    _parManager_.setReThrowParSetIfOutOfBounds(GenericToolbox::Json::fetchValue<bool>(_config_, "reThrowParSetIfOutOfBounds"));
  });
  GenericToolbox::Json::deprecatedAction(_config_, "throwToyParametersWithGlobalCov", [&]{
    LogAlert << "Forwarding the option to ParametersManager. Consider moving it into \"parametersManagerConfig:\"" << std::endl;
    _parManager_.setThrowToyParametersWithGlobalCov(GenericToolbox::Json::fetchValue<bool>(_config_, "throwToyParametersWithGlobalCov"));
  });

  // nested objects
  _parManager_.readConfig( GenericToolbox::Json::fetchValue( _config_, "parametersManagerConfig", _parManager_.getConfig()) );

  // Monitoring parameters
  _showEventBreakdown_ = GenericToolbox::Json::fetchValue(_config_, "showEventBreakdown", _showEventBreakdown_);
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

  LogInfo << "Reading PlotGenerator configuration..." << std::endl;
  auto plotGeneratorConfig = ConfigUtils::getForwardedConfig(GenericToolbox::Json::fetchValue(_config_, "plotGeneratorConfig", JsonType()));
  _plotGenerator_.setConfig(plotGeneratorConfig);
  _plotGenerator_.readConfig();

  LogInfo << "Reading DialCollection configurations..." << std::endl;
  _dialCollectionList_.clear(); // make sure it's empty in case readConfig() is called more than once
  for(size_t iParSet = 0 ; iParSet < _parManager_.getParameterSetsList().size() ; iParSet++ ){
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

  // will set it off when the Propagator will be loaded
  GundamGlobals::getParallelWorker().setCpuTimeSaverIsEnabled(true);
}

// Core
void Propagator::propagateParameters(){

  if( _enableEigenToOrigInPropagate_ ){
    // Only real parameters are propagated on the spectra -> need to convert the eigen to original
    for( auto& parSet : _parManager_.getParameterSetsList() ){
      if( parSet.isEnableEigenDecomp() ){ parSet.propagateEigenToOriginal(); }
    }
  }

  this->resetReweight();
  this->reweightMcEvents();
  this->refillSampleHistograms();

}
void Propagator::resetReweight(){
  std::for_each(_dialCollectionList_.begin(), _dialCollectionList_.end(), [&]( DialCollection& dc_){
    dc_.updateInputBuffers();
  });
}
void Propagator::reweightMcEvents() {
  reweightTimer.start();

  bool usedGPU{false};
#ifdef GUNDAM_USING_CACHE_MANAGER
  if( GundamGlobals::getEnableCacheManager() ) {
    Cache::Manager::Update(getSampleSet(), getEventDialCache());
    usedGPU = Cache::Manager::Fill();
  }
#endif
  if( not usedGPU ){
    if( not _devSingleThreadReweight_ ){ GundamGlobals::getParallelWorker().runJob("Propagator::reweightMcEvents"); }
    else{ this->reweightMcEvents(-1); }
  }

  reweightTimer.stop();
}
void Propagator::refillSampleHistograms(){
  refillHistogramTimer.start();

  if( not _devSingleThreadHistFill_ ){
    GundamGlobals::getParallelWorker().runJob("Propagator::refillSampleHistograms");
  }
  else{
    refillSampleHistogramsFct(-1);
    refillSampleHistogramsPostParallelFct();
  }

  refillHistogramTimer.stop();
}

// Misc
std::string Propagator::getSampleBreakdownTableStr() const{
  GenericToolbox::TablePrinter t;

  t << "Sample" << GenericToolbox::TablePrinter::NextColumn;
  t << "MC (# binned event)" << GenericToolbox::TablePrinter::NextColumn;
  t << "Data (# binned event)" << GenericToolbox::TablePrinter::NextColumn;
  t << "MC (weighted)" << GenericToolbox::TablePrinter::NextColumn;
  t << "Data (weighted)" << GenericToolbox::TablePrinter::NextLine;

  for( auto& sample : _sampleSet_.getSampleList() ){
    t << "\"" << sample.getName() << "\"" << GenericToolbox::TablePrinter::NextColumn;
    t << sample.getMcContainer().getNbBinnedEvents() << GenericToolbox::TablePrinter::NextColumn;
    t << sample.getDataContainer().getNbBinnedEvents() << GenericToolbox::TablePrinter::NextColumn;
    t << sample.getMcContainer().getSumWeights() << GenericToolbox::TablePrinter::NextColumn;
    t << sample.getDataContainer().getSumWeights() << GenericToolbox::TablePrinter::NextLine;
  }

  std::stringstream ss;
  ss << t.generateTableString();
  return ss.str();
}

// Protected
void Propagator::initializeThreads() {

  GundamGlobals::getParallelWorker().addJob(
      "Propagator::reweightMcEvents",
      [this](int iThread){ this->reweightMcEvents(iThread); }
  );

  GundamGlobals::getParallelWorker().addJob(
      "Propagator::refillSampleHistograms",
      [this](int iThread){ this->refillSampleHistogramsFct(iThread); }
  );

  GundamGlobals::getParallelWorker().setPostParallelJob(
      "Propagator::refillSampleHistograms",
      [this](){ this->refillSampleHistogramsPostParallelFct(); }
  );

}

// multithreading
void Propagator::reweightMcEvents(int iThread_) {

  //! Warning: everything you modify here, may significantly slow down the
  //! fitter

  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, GundamGlobals::getParallelWorker().getNbThreads(),
      int(_eventDialCache_.getCache().size())
  );

  std::for_each(
      _eventDialCache_.getCache().begin() + bounds.first,
      _eventDialCache_.getCache().begin() + bounds.second,
      [this]( EventDialCache::CacheElem_t& cache_){ _eventDialCache_.reweightEntry(cache_); }
  );

}
void Propagator::refillSampleHistogramsFct(int iThread_){
  for( auto& sample : _sampleSet_.getSampleList() ){
    sample.getMcContainer().refillHistogram(iThread_);
    sample.getDataContainer().refillHistogram(iThread_);
  }
}
void Propagator::refillSampleHistogramsPostParallelFct(){
  for( auto& sample : _sampleSet_.getSampleList() ){
    sample.getMcContainer().rescaleHistogram();
    sample.getDataContainer().rescaleHistogram();
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
