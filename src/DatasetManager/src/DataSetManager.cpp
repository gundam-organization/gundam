//
// Created by Nadrino on 04/03/2024.
//

#include "DataSetManager.h"

#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif

#include "Logger.h"

LoggerInit([]{
  Logger::getUserHeader() << "[DataSetManager]";
});


void DataSetManager::readConfigImpl(){
  LogInfo << "Reading DataSetManager configurations..." << std::endl;

  // Propagator config is supposed to be set in the likelihood interface.
  // however in old GUNDAM versions it was set at the FitterEngine level.
  // In that case, the config has already been set, hence "_propagator_.getConfig()"
  _propagator_.setConfig( GenericToolbox::Json::fetchValue( _config_, "propagatorConfig", _propagator_.getConfig() ) );
  _propagator_.readConfig();

  // dataSetList should be present
  JsonType dataSetList;
  dataSetList = GenericToolbox::Json::fetchValue(_config_, "dataSetList", dataSetList);

  // creating the dataSets:
  _dataSetList_.reserve( dataSetList.size() );
  for( const auto& dataSetConfig : dataSetList ){ _dataSetList_.emplace_back(dataSetConfig, int(_dataSetList_.size())); }

  // deprecated config files will already have filled up _treeWriter_.getConfig()
  GenericToolbox::Json::deprecatedAction(_propagator_.getConfig(), "eventTreeWriter", [&]{
    LogAlert << R"("eventTreeWriter" should now be set under "dataSetManagerConfig" instead of "propagatorConfig".)" << std::endl;
    _treeWriter_.setConfig( GenericToolbox::Json::fetchValue<JsonType>(_config_, "eventTreeWriter") );
  });
  _treeWriter_.readConfig( GenericToolbox::Json::fetchValue(_config_, "eventTreeWriter", _treeWriter_.getConfig()) );
}
void DataSetManager::initializeImpl(){
  LogInfo << "Initializing DataSetManager..." << std::endl;

  _propagator_.initialize();
  for( auto& dataSet : _dataSetList_ ){ dataSet.initialize(); }
  _treeWriter_.initialize();

  _propagator_.getPlotGenerator().setSampleSetPtr(&_propagator_.getSampleSet());
  _propagator_.getPlotGenerator().initialize();

  loadData();
}

void DataSetManager::loadData(){
  LogInfo << "Loading data into the PropagatorEngine..." << std::endl;

  bool cacheManagerState = GundamGlobals::getEnableCacheManager();
  GundamGlobals::setEnableCacheManager(false);

  // First start with the data:
  bool usedMcContainer{false};
  bool allAsimov{true};
  for( auto& dataSet : _dataSetList_ ){
    LogContinueIf(not dataSet.isEnabled(), "Dataset \"" << dataSet.getName() << "\" is disabled. Skipping");

    // selecting the right dispenser
    DataDispenser* dispenser = &dataSet.getSelectedDataDispenser();
    if( _propagator_.isThrowAsimovToyParameters() ) { dispenser = &dataSet.getToyDataDispenser(); }
    if( _propagator_.isLoadAsimovData() ){ dispenser = &dataSet.getDataDispenserDict().at("Asimov"); }

    // checking what we are loading
    if(dispenser->getParameters().name != "Asimov" ){ allAsimov = false; }
    if( dispenser->getParameters().useMcContainer ){ usedMcContainer = true; }

    // loading in the propagator
    LogInfo << "Reading dataset: " << dataSet.getName() << "/" << dispenser->getParameters().name << std::endl;
    dispenser->load( _propagator_ );

    LogInfo << "Resizing dial containers..." << std::endl;
    for( auto& dialCollection : _propagator_.getDialCollectionList() ) {
      if( not dialCollection.isBinned() ){ dialCollection.resizeContainers(); }
    }

    LogInfo << "Build reference cache..." << std::endl;
    _propagator_.buildDialCache();
  }

  // Copy to data container
  if( usedMcContainer ){
    if( _propagator_.isThrowAsimovToyParameters() ){
      LogWarning << "Will throw toy parameters..." << std::endl;

      if( _propagator_.isShowEventBreakdown() ){
        LogInfo << "Propagating prior parameters on the initially loaded events..." << std::endl;
        _propagator_.reweightMcEvents();

        LogInfo << "Sample breakdown prior to the throwing:" << std::endl;
        std::cout << _propagator_.getSampleBreakdownTableStr() << std::endl;

        if( _propagator_.isDebugPrintLoadedEvents() ){
          LogDebug << "Toy events:" << std::endl;
          LogDebug << GET_VAR_NAME_VALUE(_propagator_.getDebugPrintLoadedEventsNbPerSample()) << std::endl;
          int iEvt{0};
          for( auto& entry : _propagator_.getEventDialCache().getCache() ) {
            LogDebug << "Event #" << iEvt++ << "{" << std::endl;
            {
              LogScopeIndent;
              LogDebug << entry.getSummary() << std::endl;
            }
            LogDebug << "}" << std::endl;
            if( iEvt >= _propagator_.getDebugPrintLoadedEventsNbPerSample() ) break;
          }
        }
      }

      _propagator_.getParametersManager().throwParameters();

      // Handling possible masks
      for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
        if( not parSet.isEnabled() ) continue;

        if( parSet.isMaskForToyGeneration() ){
          LogWarning << parSet.getName() << " will be masked for the toy generation." << std::endl;
          parSet.setMaskedForPropagation( true );
        }
      }

    } // throw asimov?

    LogInfo << "Propagating parameters on events..." << std::endl;

    // Make sure before the copy to the data:
    // At this point, MC events have been reweighted using their prior
    // but when using eigen decomp, the conversion eigen -> original has a small computational error
    for( auto& parSet: _propagator_.getParametersManager().getParameterSetsList() ) {
      if( parSet.isEnableEigenDecomp() ) { parSet.propagateEigenToOriginal(); }
    }

    _propagator_.reweightMcEvents();

    // Copies MC events in data container for both Asimov and FakeData event types
    LogWarning << "Copying loaded mc-like event to data container..." << std::endl;
    _propagator_.getSampleSet().copyMcEventListToDataContainer();

    // back to prior
    if( _propagator_.isThrowAsimovToyParameters() ){
      for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){

        if( parSet.isMaskForToyGeneration() ){
          // unmasking
          LogWarning << "Unmasking parSet: " << parSet.getName() << std::endl;
          parSet.setMaskedForPropagation( false );
        }

        parSet.moveParametersToPrior();
      }
    }
  }

  if( not allAsimov ){
    // reload everything
    // Filling the mc containers

    // clearing events in MC containers
    _propagator_.getSampleSet().clearMcContainers();

    // also wiping event-by-event dials...
    LogInfo << "Wiping event-by-event dials..." << std::endl;

    for( auto& dialCollection: _propagator_.getDialCollectionList() ) {
      if( not dialCollection.getGlobalDialLeafName().empty() ) { dialCollection.clear(); }
    }
    _propagator_.getEventDialCache() = EventDialCache();

    for( auto& dataSet : _dataSetList_ ){
      LogContinueIf(not dataSet.isEnabled(), "Dataset \"" << dataSet.getName() << "\" is disabled. Skipping");
      auto& dispenser = dataSet.getMcDispenser();
      dispenser.load( _propagator_ );
    }

    LogInfo << "Resizing dial containers..." << std::endl;
    for( auto& dialCollection : _propagator_.getDialCollectionList() ) {
      if( not dialCollection.isBinned() ){ dialCollection.resizeContainers(); }
    }

    LogInfo << "Build reference cache..." << std::endl;
    _propagator_.buildDialCache();
  }

#ifdef GUNDAM_USING_CACHE_MANAGER
  // After all the data has been loaded.  Specifically, this must be after
  // the MC has been copied for the Asimov fit, or the "data" use the MC
  // reweighting cache.  This must also be before the first use of
  // reweightMcEvents.
  if( cacheManagerState ) {
    Cache::Manager::Build(_propagator_.getSampleSet(), _propagator_.getEventDialCache());
  }
#endif

  LogInfo << "Filling up sample bin caches..." << std::endl;
  GundamGlobals::getParallelWorker().runJob([this](int iThread){
    LogInfoIf(iThread <= 0) << "Updating sample per bin event lists..." << std::endl;
    for( auto& sample : _propagator_.getSampleSet().getSampleList() ){
      sample.getMcContainer().updateBinEventList(iThread);
      sample.getDataContainer().updateBinEventList(iThread);
    }
  });

  LogInfo << "Filling up sample histograms..." << std::endl;
  GundamGlobals::getParallelWorker().runJob([this](int iThread){
    for( auto& sample : _propagator_.getSampleSet().getSampleList() ){
      sample.getMcContainer().refillHistogram(iThread);
      sample.getDataContainer().refillHistogram(iThread);
    }
  });

  // Throwing stat error on data -> BINNING SHOULD BE SET!!
  if( _propagator_.isThrowAsimovToyParameters() and _propagator_.isEnableStatThrowInToys() ){
    LogInfo << "Throwing statistical error for data container..." << std::endl;

    if( _propagator_.isEnableEventMcThrow() ){
      // Take into account the finite amount of event in MC
      LogInfo << "enableEventMcThrow is enabled: throwing individual MC events" << std::endl;
      for( auto& sample : _propagator_.getSampleSet().getSampleList() ) {
        sample.getDataContainer().throwEventMcError();
      }
    }
    else{
      LogWarning << "enableEventMcThrow is disabled. Not throwing individual MC events" << std::endl;
    }

    LogInfo << "Throwing statistical error on histograms..." << std::endl;
    if( _propagator_.isGaussStatThrowInToys() ) {
      LogWarning << "Using gaussian statistical throws. (caveat: distribution truncated when the bins are close to zero)" << std::endl;
    }
    for( auto& sample : _propagator_.getSampleSet().getSampleList() ){
      // Asimov bin content -> toy data
      sample.getDataContainer().throwStatError( _propagator_.isGaussStatThrowInToys() );
    }
  }

  /// Now caching the event for the plot generator
  _propagator_.getPlotGenerator().defineHistogramHolders();

  GundamGlobals::setEnableCacheManager(cacheManagerState);

  /// Propagator needs to be fast, let the workers wait for the signal
  GundamGlobals::getParallelWorker().setCpuTimeSaverIsEnabled(false);
}
