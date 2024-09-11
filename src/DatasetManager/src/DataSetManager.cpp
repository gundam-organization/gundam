//
// Created by Nadrino on 04/03/2024.
//

#include "DataSetManager.h"

#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif

#include "Logger.h"

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::getUserHeader() << "[DataSetManager]"; });
#endif


void DataSetManager::readConfigImpl(){
  LogInfo << "Reading DataSetManager configurations..." << std::endl;

  _threadPool_.setNThreads( GundamGlobals::getNumberOfThreads() );

  // Propagator config is supposed to be set in the likelihood interface.
  // however in old GUNDAM versions it was set at the FitterEngine level.
  // In that case, the config has already been set, hence "_propagator_.getConfig()"
  _modelPropagator_.setConfig(GenericToolbox::Json::fetchValue(_config_, "propagatorConfig", _modelPropagator_.getConfig() ) );
  _modelPropagator_.readConfig();

  // dataSetList should be present
  JsonType dataSetList;
  dataSetList = GenericToolbox::Json::fetchValue(_config_, "dataSetList", dataSetList);

  // creating the dataSets:
  _dataSetList_.reserve( dataSetList.size() );
  for( const auto& dataSetConfig : dataSetList ){ _dataSetList_.emplace_back(dataSetConfig, int(_dataSetList_.size())); }

  // deprecated config files will already have filled up _treeWriter_.getConfig()
  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getConfig(), "eventTreeWriter", [&]{
    LogAlert << R"("eventTreeWriter" should now be set under "datasetManagerConfig" instead of "propagatorConfig".)" << std::endl;
    _treeWriter_.setConfig( GenericToolbox::Json::fetchValue<JsonType>(_modelPropagator_.getConfig(), "eventTreeWriter") );
  });
  _treeWriter_.readConfig( GenericToolbox::Json::fetchValue(_config_, "eventTreeWriter", _treeWriter_.getConfig()) );
}
void DataSetManager::initializeImpl(){
  LogInfo << "Initializing DataSetManager..." << std::endl;

  _modelPropagator_.initialize();
  for( auto& dataSet : _dataSetList_ ){ dataSet.initialize(); }
  _treeWriter_.initialize();

  _modelPropagator_.getPlotGenerator().setSampleSetPtr(&_modelPropagator_.getSampleSet());
  _modelPropagator_.getPlotGenerator().initialize();

  load();
}

void DataSetManager::loadPropagator( bool isModel_ ){

  // assume we are loading the model
  _reloadModelRequested_ = false;

  std::vector<DataDispenser*> dispenserToLoadList{};
  for( auto& dataSet : _dataSetList_ ){
    LogContinueIf(not dataSet.isEnabled(), "Dataset \"" << dataSet.getName() << "\" is disabled. Skipping");

    if( isModel_ or _modelPropagator_.isLoadAsimovData() ){ dispenserToLoadList.emplace_back(&dataSet.getModelDispenser() ); }
    else{
      dispenserToLoadList.emplace_back( &dataSet.getDataDispenser() );
      if( _modelPropagator_.isThrowAsimovToyParameters() ){ dispenserToLoadList.back() = &dataSet.getToyDataDispenser(); }
    }

    // if we don't load an Asimov-tagged dataset, we'll need to reload the data for building our model.
    if( dispenserToLoadList.back()->getParameters().name != "Asimov" ){ _reloadModelRequested_ = true; }
  }

  // by default, load the model
  Propagator* propagatorPtr{ &_modelPropagator_ };
  std::unique_ptr<Propagator> propagatorUniquePtr{nullptr};

  // use a temporary propagator as some parameter can be edited
  if( _reloadModelRequested_ ){
    // unique ptr will make sure its properly deleted
    propagatorUniquePtr = std::make_unique<Propagator>(_modelPropagator_);
    propagatorPtr = propagatorUniquePtr.get();
  }

  // make sure everything is ready for loading
  propagatorPtr->clearContent();

  // perform the loading
  for( auto* dispenserToLoad : dispenserToLoadList ){
    LogInfo << "Reading dataset: " << dispenserToLoad->getOwner()->getName() << "/" << dispenserToLoad->getParameters().name << std::endl;

    if( not dispenserToLoad->getParameters().overridePropagatorConfig.empty() ){
      LogWarning << "Reload the propagator config with override options" << std::endl;
      ConfigUtils::ConfigHandler configHandler(_modelPropagator_.getConfig() );
      configHandler.override( dispenserToLoad->getParameters().overridePropagatorConfig );
      propagatorPtr->readConfig( configHandler.getConfig() );
      propagatorPtr->initialize();
    }

    // legacy: replacing the parameterSet option "maskForToyGeneration" -> now should use the config override above
    if( not isModel_ ){
      for( auto& parSet : propagatorPtr->getParametersManager().getParameterSetsList() ){
        if( GenericToolbox::Json::fetchValue(parSet.getConfig(), "maskForToyGeneration", false) ){ parSet.nullify(); }
      }
    }

    dispenserToLoad->load( *propagatorPtr );
  }

  LogInfo << "Resizing dial containers..." << std::endl;
  for( auto& dialCollection : propagatorPtr->getDialCollectionList() ) {
    if( dialCollection.isEventByEvent() ){ dialCollection.resizeContainers(); }
  }

  LogInfo << "Build reference cache..." << std::endl;
  propagatorPtr->buildDialCache();

  if( not isModel_ ){
    if( propagatorPtr->isThrowAsimovToyParameters() ){

      if( propagatorPtr->isShowEventBreakdown() ){
        LogInfo << "Propagating prior parameters on the initially loaded events..." << std::endl;
        propagatorPtr->reweightEvents();

        LogInfo << "Sample breakdown prior to the throwing:" << std::endl;
        std::cout << propagatorPtr->getSampleBreakdownTableStr() << std::endl;

        if( propagatorPtr->isDebugPrintLoadedEvents() ){
          LogDebug << "Toy events:" << std::endl;
          LogDebug << GET_VAR_NAME_VALUE(propagatorPtr->getDebugPrintLoadedEventsNbPerSample()) << std::endl;
          int iEvt{0};
          for( auto& entry : propagatorPtr->getEventDialCache().getCache() ) {
            LogDebug << "Event #" << iEvt++ << "{" << std::endl;
            {
              LogScopeIndent;
              LogDebug << entry.getSummary() << std::endl;
            }
            LogDebug << "}" << std::endl;
            if( iEvt >= propagatorPtr->getDebugPrintLoadedEventsNbPerSample() ) break;
          }
        }
      }

      if( _toyParameterInjector_.empty() ){
        LogWarning << "Will throw toy parameters..." << std::endl;
        propagatorPtr->getParametersManager().throwParameters();
      }
      else{
        LogWarning << "Injecting parameters..." << std::endl;
        propagatorPtr->getParametersManager().injectParameterValues( _toyParameterInjector_ );
      }

    } // throw asimov?

    LogInfo << "Propagating parameters on events..." << std::endl;

    // At this point, MC events have been reweighted using their prior
    // but when using eigen decomp, the conversion eigen to original has a small computational error
    // this will make sure the "asimov" data will be reweighted the same way the model is expected to behave
    // while using the eigen decomp
    for( auto& parSet: propagatorPtr->getParametersManager().getParameterSetsList() ) {
      if( not parSet.isEnabled() ){ continue; }
      if( parSet.isEnableEigenDecomp() ) { parSet.propagateEigenToOriginal(); }
    }

    propagatorPtr->reweightEvents();

    // Copies MC events in data container for both Asimov and FakeData event types
    LogWarning << "Copying loaded mc-like event to data container..." << std::endl;

    // first copy the event directly placed in the data container
    if( &_modelPropagator_ != propagatorPtr ){
      for( size_t iSample = 0 ; iSample < _modelPropagator_.getSampleSet().getSampleList().size() ; iSample++ ){
        _modelPropagator_.getSampleSet().getSampleList()[iSample].getDataContainer().getEventList() =
            propagatorPtr->getSampleSet().getSampleList()[iSample].getDataContainer().getEventList();
      }
    }

    // then copy the events that can have been loaded by the MC container
    propagatorPtr->getSampleSet().copyMcEventListToDataContainer(_modelPropagator_.getSampleSet().getSampleList() );

    // back to prior in case the original _propagator_ has been used.
    // typically with `-a --toy` options
    if( propagatorPtr->isThrowAsimovToyParameters() ){
      for( auto& parSet : propagatorPtr->getParametersManager().getParameterSetsList() ){
        if( not parSet.isEnabled() ){ continue; }
        parSet.moveParametersToPrior();
      }
    }
  }

}
void DataSetManager::load(){

  LogInfo << "Loading data into the propagator engine..." << std::endl;
  this->loadPropagator( false );

  // For non-Asimov fits, we need to reload the data.
  if( _reloadModelRequested_ ){
    LogInfo << "Loading the model in the propagator engine..." << std::endl;
    this->loadPropagator( true );
  }

  // Now everything is loaded, create the list
  this->buildSamplePairList();

  // The event reweighting is completely defined!  Now print a breakdown of all
  // the loaded events with all the global reweighting applied, but none of
  // the dials applied.  It needs to be done BEFORE the cache manager is built
  // since this uses a hack to apply the global weights that modifies the
  // reweighting, and then modifies it back. (The reweighting needs to be
  // immutable after the Cache::Manager is created, or it's going to introduce
  // bugs).
  _modelPropagator_.printBreakdowns();

#ifdef GUNDAM_USING_CACHE_MANAGER
  // After all the data has been loaded.  Specifically, this must be after
  // the MC has been copied for the Asimov fit, or the "data" use the MC
  // reweighting cache.  This must also be before the first use of
  // reweightMcEvents that is done using the GPU.
  Cache::Manager::Build(
      _modelPropagator_.getSampleSet(),
      _modelPropagator_.getEventDialCache()
  );
#endif

  LogInfo << "Propagating prior parameters on events..." << std::endl;
  _modelPropagator_.reweightEvents();

  // The histogram bin was assigned to each event by the DataDispenser, now
  // cache the binning results for speed into each of the samples.
  LogInfo << "Filling up sample bin caches..." << std::endl;
  _threadPool_.runJob([this](int iThread){
    LogInfoIf(iThread <= 0) << "Updating sample per bin event lists..." << std::endl;
    for( auto& sample : _modelPropagator_.getSampleSet().getSampleList() ){
      sample.getMcContainer().updateBinEventList(iThread);
      sample.getDataContainer().updateBinEventList(iThread);
    }
  });

  LogInfo << "Filling up sample histograms..." << std::endl;
  _threadPool_.runJob([this](int iThread){
    for( auto& sample : _modelPropagator_.getSampleSet().getSampleList() ){
      sample.getMcContainer().refillHistogram(iThread);
      sample.getDataContainer().refillHistogram(iThread);
    }
  });

  // Throwing stat error on data -> BINNING SHOULD BE SET!!
  if( _modelPropagator_.isThrowAsimovToyParameters() and _modelPropagator_.isEnableStatThrowInToys() ){
    LogInfo << "Throwing statistical error for data container..." << std::endl;

    if( _modelPropagator_.isEnableEventMcThrow() ){
      // Take into account the finite amount of event in MC
      LogInfo << "enableEventMcThrow is enabled: throwing individual MC events" << std::endl;
      for( auto& sample : _modelPropagator_.getSampleSet().getSampleList() ) {
        sample.getDataContainer().throwEventMcError();
      }
    }
    else{
      LogWarning << "enableEventMcThrow is disabled. Not throwing individual MC events" << std::endl;
    }

    LogInfo << "Throwing statistical error on histograms..." << std::endl;
    if( _modelPropagator_.isGaussStatThrowInToys() ) {
      LogWarning << "Using gaussian statistical throws. (caveat: distribution truncated when the bins are close to zero)" << std::endl;
    }
    for( auto& sample : _modelPropagator_.getSampleSet().getSampleList() ){
      // Asimov bin content -> toy data
      sample.getDataContainer().throwStatError(_modelPropagator_.isGaussStatThrowInToys() );
    }
  }

  /// Now caching the event for the plot generator
  _modelPropagator_.getPlotGenerator().defineHistogramHolders();

  /// Propagator needs to be fast, let the workers wait for the signal
  _modelPropagator_.getThreadPool().setCpuTimeSaverIsEnabled(false);

}

void DataSetManager::buildSamplePairList(){

  auto nModelSamples{_modelPropagator_.getSampleSet().getSampleList().size()};
  auto nDataSamples{_dataPropagator_.getSampleSet().getSampleList().size()};
  LogThrowIf(nModelSamples != nDataSamples,
             "Mismatching number of samples for model(" << nModelSamples <<
             ") and data(" << nDataSamples << ") propagators."
 );

  _samplePairList_.clear();
  _samplePairList_.reserve( nModelSamples );
  for( size_t iSample = 0 ; iSample < nModelSamples ; iSample++ ){
    _samplePairList_.emplace_back();
    _samplePairList_.back().model = &_modelPropagator_.getSampleSet().getSampleList()[iSample];
    _samplePairList_.back().data = &_dataPropagator_.getSampleSet().getSampleList()[iSample];
  }

}
