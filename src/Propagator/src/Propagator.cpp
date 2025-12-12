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
#include "GundamBacktrace.h"

#include "GenericToolbox.Utils.h"

#include <memory>
#include <vector>

void Propagator::muteLogger(){ Logger::setIsMuted( true ); }
void Propagator::unmuteLogger(){ Logger::setIsMuted( false ); }

void Propagator::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({
    {"sampleSetConfig", {"fitSampleSetConfig"}},
    {"parametersManagerConfig"},
    {"parameterInjection"},
    {"showNbEventParameterBreakdown"},
    {"showNbEventPerSampleParameterBreakdown"},
    {"debugPrintLoadedEvents"},
    {"debugPrintLoadedEventsNbPerSample"},
    {"devSingleThreadReweight"},
    {"devSingleThreadHistFill"},
    {"globalEventReweightCap"},
    // relocated:
    {FieldFlag::RELOCATED, "parameterSetListConfig", "parametersManagerConfig/parameterSetList"},
    {FieldFlag::RELOCATED, "throwToyParametersWithGlobalCov", "parametersManagerConfig"},
    {FieldFlag::RELOCATED, "enableStatThrowInToys", "likelihoodInterfaceConfig/enableStatThrowInToys"},
    {FieldFlag::RELOCATED, "gaussStatThrowInToys", "likelihoodInterfaceConfig/gaussStatThrowInToys"},
    {FieldFlag::RELOCATED, "enableEventMcThrow", "likelihoodInterfaceConfig/enableEventMcThrow"},
    {FieldFlag::RELOCATED, "plotGeneratorConfig", "likelihoodInterfaceConfig/plotGeneratorConfig"},
    {FieldFlag::RELOCATED, "llhStatFunction", "likelihoodInterfaceConfig/jointProbabilityConfig/type"},
    {FieldFlag::RELOCATED, "llhConfig", "likelihoodInterfaceConfig/jointProbabilityConfig"},
    {FieldFlag::RELOCATED, "scanConfig", "fitterEngineConfig/scanConfig"},
    {FieldFlag::RELOCATED, "eventTreeWriter", "likelihoodInterfaceConfig/eventTreeWriter"},
    {FieldFlag::RELOCATED, "dataSetList", {"fitSampleSetConfig/dataSetList"}, "likelihoodInterfaceConfig/dataSetList"},
  });
  _config_.checkConfiguration();

  // nested objects
  _config_.fillValue(_sampleSet_.getConfig(), "sampleSetConfig");
  _sampleSet_.configure();

  // relocated
  _config_.fillValue(_parManager_.getParameterSetListConfig(), "parameterSetListConfig");
  _config_.fillValue(_parManager_.getThrowToyParametersWithGlobalCov(), "throwToyParametersWithGlobalCov");


  _config_.fillValue(_parManager_.getConfig(), "parametersManagerConfig");
  _parManager_.configure();

  _dialManager_.setParametersManager(&_parManager_);
  _dialManager_.configure();

  // Monitoring parameters
  _config_.fillValue(_showNbEventParameterBreakdown_, "showNbEventParameterBreakdown");
  _config_.fillValue(_showNbEventPerSampleParameterBreakdown_, "showNbEventPerSampleParameterBreakdown");
  _config_.fillValue(_parameterInjectorMc_, "parameterInjection");
  _config_.fillValue(_debugPrintLoadedEvents_, "debugPrintLoadedEvents");
  _config_.fillValue(_debugPrintLoadedEventsNbPerSample_, "debugPrintLoadedEventsNbPerSample");
  _config_.fillValue(_devSingleThreadReweight_, "devSingleThreadReweight");
  _config_.fillValue(_devSingleThreadHistFill_, "devSingleThreadHistFill");
  _config_.fillValue(_eventDialCache_.getGlobalEventReweightCap().maxReweight, "globalEventReweightCap");

}
void Propagator::initializeImpl(){

  _config_.printUnusedKeys();

  _parManager_.initialize();
  _sampleSet_.initialize();
  _dialManager_.initialize();

  initializeThreads();

  _eventDialCache_.getGlobalEventReweightCap().isEnabled = not std::isnan(_eventDialCache_.getGlobalEventReweightCap().maxReweight);
}

// core
void Propagator::clearContent(){

  // clearing events in MC containers
  _sampleSet_.clearEventLists();

  // also wiping event-by-event dials...
  // _dialManager_.clearEventByEventDials();
  _dialManager_.setParametersManager(&_parManager_);
  _dialManager_.configure();
  _dialManager_.initialize();

  // reset the cache
  _eventDialCache_ = EventDialCache();

}
void Propagator::buildDialCache(){
  _eventDialCache_.shrinkIndexedCache();
  _eventDialCache_.buildReferenceCache(_sampleSet_, _dialManager_.getDialCollectionList());
  _dialManager_.invalidateInputBuffers();
}
void Propagator::propagateParameters(){
  std::future<bool> result = applyParameters();
  result.get();
}

std::future<bool> Propagator::applyParameters(){
  // Make sure the dial state is updated before reweighting and filling the
  // histograms.  This has to be done before the GPU and CPU calculations, and
  // should be shared for both.
  if( _enableEigenToOrigInPropagate_ ){ _parManager_.convertEigenToOrig(); }
  _dialManager_.updateDialState();

#ifdef GUNDAM_USING_CACHE_MANAGER
  // Trigger the reweight on the GPU.  This will fill the histograms, but most
  // of the time, leaves the event weights on the GPU.
  std::future<bool> cacheManager = Cache::Manager::Fill(getSampleSet(),getEventDialCache());
  if (cacheManager.valid() and not Cache::Manager::IsForceCpuCalculation()) {
    return cacheManager;  // The cacheManager future could be returned.
  }
#endif

  // Trigger the reweight on the CPU.  Override the dial update inside of
  // reweight event, or the CPU code will decide that the update is already
  // done.
  this->reweightEvents(false);
  this->refillHistograms();

#ifdef GUNDAM_USING_CACHE_MANAGER
  if (cacheManager.valid() and Cache::Manager::IsForceCpuCalculation()) {
    bool valid = Cache::Manager::ValidateHistogramContents();
    if (not valid) {
      LogError << GundamUtils::Backtrace;
      LogError << "Parallel GPU and CPU calculations disagree" << std::endl;
    }
  }
#endif

  // The CPU has already finished filling the data structures by the time we
  // get here, so this could be done with a std::promise<bool> and set the
  // value before returning the future.  It's done with a deferred std::async
  // since the code is a little cleaner to my eye, and the lambda mostly
  // optimizes into oblivion.
  return std::async(std::launch::deferred, []{return true;});
}

void Propagator::reweightEvents(bool updateDials) {

  if (updateDials) {
    // Make sure the dial state is updated before pulling the trigger on the
    // reweight.  will duplicate work when running with GPU
    // isForceCpuCalculation is true.
    if( _enableEigenToOrigInPropagate_ ){ _parManager_.convertEigenToOrig(); }
    _dialManager_.updateDialState();
  }

  // timer start/stop after the dials are updated.
  auto s{reweightTimer.scopeTime()};

  if( not _devSingleThreadReweight_ ){
    _threadPool_.runJob("Propagator::reweightEvents");
  }
  else{ this->reweightEvents(-1); }

}

// misc
void Propagator::writeEventRates(const GenericToolbox::TFilePath& saveDir_) const {
  for( auto& sample : _sampleSet_.getSampleList() ){ sample.writeEventRates(saveDir_); }
}
void Propagator::writeParameterStateTree(const GenericToolbox::TFilePath& saveDir_) const{
  if( saveDir_.getRootDir() == nullptr ) {
    LogError << "TFilePath has no root TDirectory set. Skipping " << __METHOD_NAME__ << std::endl;
    return;
  }

  saveDir_.getDir()->cd();
  auto* tree = new TTree("parStateTree", "Parameter state tree");
  std::list<double> parValueList{};

  for( auto& parSet : _parManager_.getParameterSetsList() ){
    if( not parSet.isEnabled() ){ continue; }

    auto makeBranchFct = [&](const std::vector<Parameter>& parList_){
      for( auto& par : parList_ ) {
        if( not par.isEnabled() ){ continue; }
        parValueList.emplace_back(par.getParameterValue());

        tree->Branch(
          GenericToolbox::generateCleanBranchName(par.getFullTitle()).c_str(),
          &parValueList.back()
        );
      }
    };

    if( parSet.isEnableEigenDecomp() ){
      makeBranchFct(parSet.getEigenParameterList());
    }
    makeBranchFct(parSet.getParameterList());

  }


  tree->Fill();
  tree->Write(tree->GetName(), TObject::kOverwrite);
  delete tree;

}
void Propagator::printConfiguration() const {
  LogInfo << std::endl << "Printing propagator configuration:" << std::endl;

  _sampleSet_.printConfiguration();
  _parManager_.printConfiguration();
  _dialManager_.printSummaryTable();

  LogInfo << std::endl;
}
void Propagator::printBreakdowns() const {

  LogInfo << std::endl << "Breaking down samples..." << std::endl;

  if( _showNbEventParameterBreakdown_ ){

    struct NbEventBreakdown{
      size_t nbTotal{0};
      std::map<int, int> nbForSample{};
    };

    std::map<const Parameter*, NbEventBreakdown> nbEventForParameter{}; // assuming int = 0 by default
    for( auto& cache: _eventDialCache_.getCache() ){
      for( auto& dial : cache.dialResponseCacheList ){
        for( int iInput = 0 ; iInput < dial.dialInterface->getInputBufferRef()->getInputSize() ; iInput++ ){
          nbEventForParameter[ &dial.dialInterface->getInputBufferRef()->getParameter(iInput) ].nbTotal += 1;

          if( _showNbEventPerSampleParameterBreakdown_ ){
            nbEventForParameter[ &dial.dialInterface->getInputBufferRef()->getParameter(iInput) ]
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

    bool hasNoDial{true};

    for( auto& parSet : _parManager_.getParameterSetsList() ){
      if( not parSet.isEnabled() ){ continue; }
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ){ continue; }

        t.setColorBuffer( GenericToolbox::ColorCodes::resetColor );
        if( nbEventForParameter[ &par ].nbTotal == 0 ){ t.setColorBuffer( GenericToolbox::ColorCodes::redBackground ); }

        t << par.getFullTitle();
        t << GenericToolbox::TablePrinter::NextColumn << nbEventForParameter[ &par ].nbTotal;

        LogDebug << "Parameter " << par.getFullTitle() << " has " << nbEventForParameter[ &par ].nbTotal << " events." << std::endl;
        if( hasNoDial and nbEventForParameter[ &par ].nbTotal != 0 ){ hasNoDial = false; }

        if( _showNbEventPerSampleParameterBreakdown_ ){
          for( auto& sample : _sampleSet_.getSampleList() ){
            if( not sample.isEnabled() ){ continue; }
            t << GenericToolbox::TablePrinter::NextColumn << nbEventForParameter[ &par ].nbForSample[sample.getIndex()];
          }
        }

        t << GenericToolbox::TablePrinter::NextLine;
      }
    }

    if( not hasNoDial ){
      LogInfo << "Nb of event affected by parameters:" << std::endl;
      t.printTable();
    }
    else{
      LogInfo << "Events aren't parametrised." << std::endl;
    }

  }

  if( _debugPrintLoadedEvents_ ){
    LogInfo << "Printing " << _debugPrintLoadedEventsNbPerSample_ << " events..." << std::endl;
    for( int iEvt = 0 ; iEvt < std::min(_debugPrintLoadedEventsNbPerSample_, int(_eventDialCache_.getCache().size())) ; iEvt++ ){
      LogInfo << "Event #" << iEvt << "{" << std::endl;
      {
        LogScopeIndent;
        LogDebug << "PTR:" << _eventDialCache_.getCache()[iEvt].event << std::endl;
        LogInfo << _eventDialCache_.getCache()[iEvt].getSummary() << std::endl;
      }
      LogInfo << "}" << std::endl;
    }
  }
}
void Propagator::copyEventsFrom(const Propagator& src_){
  _sampleSet_.copyEventsFrom( src_.getSampleSet() );
  _eventDialCache_.fillCacheEntries( _sampleSet_ );
}
void Propagator::copyHistBinContentFrom(const Propagator& src_){
  _sampleSet_.copyHistBinContentFrom(src_.getSampleSet());
}

#ifdef GUNDAM_USING_CACHE_MANAGER
void Propagator::initializeCacheManager(){
  LogInfo << "Setting up the cache manager..." << std::endl;

  // After all the data has been loaded.  Specifically, this must be after
  // the MC has been copied for the Asimov fit, or the "data" use the MC
  // reweighting cache.  This must also be before the first use of
  // reweightMcEvents that is done using the GPU.
  Cache::Manager::Build(_sampleSet_, _eventDialCache_);

  // By default, make sure every data is copied to the CPU part
  // Some of those part might get disabled for faster calculation
  Cache::Manager::SetIsEventWeightCopyEnabled( true );
  Cache::Manager::SetIsHistContentCopyEnabled( true );
  Cache::Manager::PropagateParameters(_sampleSet_,_eventDialCache_);
}
#endif


// Protected
void Propagator::initializeThreads() {

  _threadPool_ = GenericToolbox::ParallelWorker();
  _threadPool_.setNThreads(GundamGlobals::getNbCpuThreads() );

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
void Propagator::refillHistograms(){
  // timer start/stop in scope
  auto s{refillHistogramTimer.scopeTime()};

  if( not _devSingleThreadHistFill_ ){ _threadPool_.runJob("Propagator::refillHistograms"); }
  else{ refillHistogramsFct(-1); }
}

// multithreading
void Propagator::reweightEvents( int iThread_) {

  //! Warning: everything you modify here, may significantly slow down the
  //! fitter

  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, _threadPool_.getNbThreads(),
      int(_eventDialCache_.getCache().size())
  );


  bool once{true};
  std::for_each(
      _eventDialCache_.getCache().begin() + bounds.beginIndex,
      _eventDialCache_.getCache().begin() + bounds.endIndex,
      [&]( EventDialCache::CacheEntry& cache_){
        if(once) {
          DEBUG_VAR(cache_.dialResponseCacheList.size());
          DEBUG_VAR(cache_.event->getEventWeight());
          LogDebug << cache_.event << std::endl;
          LogDebug << cache_.event->getSummary() << std::endl;
          once = false;
        }
        _eventDialCache_.reweightEntry(cache_);
      }
  );

}
void Propagator::refillHistogramsFct( int iThread_){
  for( auto& sample : _sampleSet_.getSampleList() ){
    sample.getHistogram().refillHistogram(iThread_);
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
// End:
