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

#include "GenericToolbox.TablePrinter.h"
#include "GenericToolbox.Json.h"
#include "GenericToolbox.h"

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

  _parManager_.readConfig( GenericToolbox::Json::fetchValue(_config_, "parametersManagerConfig", nlohmann::json()) );

  // legacy -- option within propagator -> should be defined elsewhere
  GenericToolbox::Json::deprecatedAction(_config_, "reThrowParSetIfOutOfBounds", [&]{
    LogAlert << "Forwarding the option to ParametersManager. Consider moving it into \"parametersManagerConfig:\"" << std::endl;
    _parManager_.setReThrowParSetIfOutOfBounds(GenericToolbox::Json::fetchValue<bool>(_config_, "reThrowParSetIfOutOfBounds"));
  });
  GenericToolbox::Json::deprecatedAction(_config_, "throwToyParametersWithGlobalCov", [&]{
    LogAlert << "Forwarding the option to ParametersManager. Consider moving it into \"parametersManagerConfig:\"" << std::endl;
    _parManager_.setThrowToyParametersWithGlobalCov(GenericToolbox::Json::fetchValue<bool>(_config_, "throwToyParametersWithGlobalCov"));
  });

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
  EventDialCache::globalEventReweightCap = GenericToolbox::Json::fetchValue(_config_, "globalEventReweightCap", EventDialCache::globalEventReweightCap);

  LogInfo << "Reading parameter configuration..." << std::endl;
  auto parameterSetListConfig = ConfigUtils::getForwardedConfig(GenericToolbox::Json::fetchValue(_config_, "parameterSetListConfig", nlohmann::json()));
  _parManager_.getParameterSetsList().reserve(parameterSetListConfig.size()); // make sure the objects aren't moved in RAM ( since FitParameter* will be used )
  for( const auto& parameterSetConfig : parameterSetListConfig ){
    _parManager_.getParameterSetsList().emplace_back();
    _parManager_.getParameterSetsList().back().setConfig(parameterSetConfig);
    _parManager_.getParameterSetsList().back().readConfig();
    LogInfo << _parManager_.getParameterSetsList().back().getSummary() << std::endl;
  }

  LogInfo << "Reading samples configuration..." << std::endl;
  auto fitSampleSetConfig = GenericToolbox::Json::fetchValue(_config_, "fitSampleSetConfig", nlohmann::json());
  _fitSampleSet_.setConfig(fitSampleSetConfig);
  _fitSampleSet_.readConfig();

  LogInfo << "Reading PlotGenerator configuration..." << std::endl;
  auto plotGeneratorConfig = ConfigUtils::getForwardedConfig(GenericToolbox::Json::fetchValue(_config_, "plotGeneratorConfig", nlohmann::json()));
  _plotGenerator_.setConfig(plotGeneratorConfig);
  _plotGenerator_.readConfig();

  LogInfo << "Reading datasets configuration..." << std::endl;
  auto dataSetListConfig = ConfigUtils::getForwardedConfig(_config_, "dataSetList");
  if( dataSetListConfig.empty() ){
    // Old config files
    dataSetListConfig = ConfigUtils::getForwardedConfig(_fitSampleSet_.getConfig(), "dataSetList");
    LogAlert << "DEPRECATED CONFIG OPTION: " << "dataSetList should now be located in the Propagator config." << std::endl;
  }
  LogThrowIf(dataSetListConfig.empty(), "No dataSet specified." << std::endl);
  _dataSetList_.reserve(dataSetListConfig.size());
  for( const auto& dataSetConfig : dataSetListConfig ){
    _dataSetList_.emplace_back(dataSetConfig, int(_dataSetList_.size()));
  }

  LogInfo << "Reading ParScanner configuration..." << std::endl;
  _parScanner_.readConfig( GenericToolbox::Json::fetchValue(_config_, "scanConfig", nlohmann::json()) );

  LogInfo << "Reading DialCollection configurations..." << std::endl;
  for(size_t iParSet = 0 ; iParSet < _parManager_.getParameterSetsList().size() ; iParSet++ ){
    if( not _parManager_.getParameterSetsList()[iParSet].isEnabled() ) continue;
    // DEV / DialCollections
    if( not _parManager_.getParameterSetsList()[iParSet].getDialSetDefinitions().empty() ){
      for( auto& dialSetDef : _parManager_.getParameterSetsList()[iParSet].getDialSetDefinitions().get<std::vector<nlohmann::json>>() ){
        if( GenericToolbox::Json::doKeyExist(dialSetDef, "parametersBinningPath") ){
          _dialCollections_.emplace_back(&_parManager_.getParameterSetsList());
          _dialCollections_.back().setIndex(int(_dialCollections_.size())-1);
          _dialCollections_.back().setSupervisedParameterSetIndex( int(iParSet) );
          _dialCollections_.back().readConfig( dialSetDef );
        }
        else{ LogThrow("no parametersBinningPath option?"); }
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
          _dialCollections_.emplace_back(&_parManager_.getParameterSetsList());
          _dialCollections_.back().setIndex(int(_dialCollections_.size())-1);
          _dialCollections_.back().setSupervisedParameterSetIndex( int(iParSet) );
          _dialCollections_.back().setSupervisedParameterIndex( par.getParameterIndex() );
          _dialCollections_.back().readConfig( dialDefinitionConfig );
        }
      }
    }
  }

  LogInfo << "Reading TreeWriter configurations..." << std::endl;
  _treeWriter_.readConfig( GenericToolbox::Json::fetchValue(_config_, "eventTreeWriter", nlohmann::json()) );

  LogInfo << "Reading config of the Propagator done." << std::endl;
}
void Propagator::initializeImpl(){
  LogWarning << __METHOD_NAME__ << std::endl;

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing parameters...") << std::endl;
  _parManager_.initialize();

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing samples...") << std::endl;
  _fitSampleSet_.initialize();

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing " + std::to_string(_dataSetList_.size()) + " datasets...") << std::endl;
  for( auto& dataset : _dataSetList_ ){ dataset.initialize(); }

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing dials...") << std::endl;
  for( auto& dialCollection : _dialCollections_ ){ dialCollection.initialize(); }

  LogInfo << "Initializing propagation threads..." << std::endl;
  initializeThreads();
  GundamGlobals::getParallelWorker().setCpuTimeSaverIsEnabled(true);

  // First start with the data:
  bool usedMcContainer{false};
  bool allAsimov{true};
  for( auto& dataSet : _dataSetList_ ){
    LogContinueIf(not dataSet.isEnabled(), "Dataset \"" << dataSet.getName() << "\" is disabled. Skipping");
    DataDispenser& dispenser = dataSet.getSelectedDataDispenser();
    if( _throwAsimovToyParameters_ ) { dispenser = dataSet.getToyDataDispenser(); }
    if( _loadAsimovData_ ){ dispenser = dataSet.getDataDispenserDict().at("Asimov"); }

    dispenser.getParameters().iThrow = _iThrow_;

    if(dispenser.getParameters().name != "Asimov" ){ allAsimov = false; }
    LogInfo << "Reading dataset: " << dataSet.getName() << "/" << dispenser.getParameters().name << std::endl;

    dispenser.setSampleSetPtrToLoad(&_fitSampleSet_);
    dispenser.setPlotGenPtr(&_plotGenerator_);
    if(dispenser.getParameters().useMcContainer ){
      usedMcContainer = true;
      dispenser.setParSetPtrToLoad(&_parManager_.getParameterSetsList());
      dispenser.setDialCollectionListPtr(&_dialCollections_);
      dispenser.setEventDialCache(&_eventDialCache_);
    }
    dispenser.load();
  }

  LogInfo << "Resizing dial containers..." << std::endl;
  for( auto& dialCollection : _dialCollections_ ) {
    if( not dialCollection.isBinned() ){ dialCollection.resizeContainers(); }
  }

  LogInfo << "Build reference cache..." << std::endl;
  _eventDialCache_.shrinkIndexedCache();
  _eventDialCache_.buildReferenceCache(_fitSampleSet_, _dialCollections_);

  // Copy to data container
  if( usedMcContainer ){
    if( _throwAsimovToyParameters_ ){
      LogWarning << "Will throw toy parameters..." << std::endl;

      if( _showEventBreakdown_ ){
        LogInfo << "Propagating prior parameters on the initially loaded events..." << std::endl;
        bool cacheManagerState = GundamGlobals::getEnableCacheManager();
        GundamGlobals::setEnableCacheManager(false);
        this->resetReweight();
        this->reweightMcEvents();
        GundamGlobals::setEnableCacheManager(cacheManagerState);

        LogInfo << "Sample breakdown prior to the throwing:" << std::endl;
        std::cout << getSampleBreakdownTableStr() << std::endl;

        if( _debugPrintLoadedEvents_ ){
          LogDebug << "Toy events:" << std::endl;
          LogDebug << GET_VAR_NAME_VALUE(_debugPrintLoadedEventsNbPerSample_) << std::endl;
          int iEvt{0};
          for( auto& entry : _eventDialCache_.getCache() ) {
            LogDebug << "Event #" << iEvt++ << "{" << std::endl;
            {
              LogScopeIndent;
              LogDebug << entry.getSummary() << std::endl;
            }
            LogDebug << "}" << std::endl;
            if( iEvt >= _debugPrintLoadedEventsNbPerSample_ ) break;
          }
        }
      }

      _parManager_.throwParameters();

      // Handling possible masks
      for( auto& parSet : _parManager_.getParameterSetsList() ){
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
    for( auto& parSet: _parManager_.getParameterSetsList() ) {
      if( parSet.isUseEigenDecompInFit() ) { parSet.propagateEigenToOriginal(); }
    }

    bool cacheManagerState = GundamGlobals::getEnableCacheManager();
    GundamGlobals::setEnableCacheManager(false);
    this->resetReweight();
    this->reweightMcEvents();
    GundamGlobals::setEnableCacheManager(cacheManagerState);

    // Copies MC events in data container for both Asimov and FakeData event types
    LogWarning << "Copying loaded mc-like event to data container..." << std::endl;
    _fitSampleSet_.copyMcEventListToDataContainer();

    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      sample.getDataContainer().histScale = sample.getMcContainer().histScale;
    }

    // back to prior
    if( _throwAsimovToyParameters_ ){
      for( auto& parSet : _parManager_.getParameterSetsList() ){

        if( parSet.isMaskForToyGeneration() ){
          // unmasking
          LogWarning << "Unmasking parSet: " << parSet.getName() << std::endl;
          parSet.setMaskedForPropagation( false );
        }

        parSet.moveFitParametersToPrior();
      }
    }
  }

  if( not allAsimov ){
    // reload everything
    // Filling the mc containers

    // clearing events in MC containers
    _fitSampleSet_.clearMcContainers();

    // also wiping event-by-event dials...
    LogInfo << "Wiping event-by-event dials..." << std::endl;
    for( auto& dialCollection: _dialCollections_ ) {
      if( not dialCollection.getGlobalDialLeafName().empty() ) {
        dialCollection.clear();
      }
    }
    _eventDialCache_ = EventDialCache();

    for( auto& dataSet : _dataSetList_ ){
      LogContinueIf(not dataSet.isEnabled(), "Dataset \"" << dataSet.getName() << "\" is disabled. Skipping");
      auto& dispenser = dataSet.getMcDispenser();
      dispenser.setSampleSetPtrToLoad(&_fitSampleSet_);
      dispenser.setPlotGenPtr(&_plotGenerator_);
      dispenser.setParSetPtrToLoad(&_parManager_.getParameterSetsList());
      dispenser.setDialCollectionListPtr(&_dialCollections_);
      dispenser.setEventDialCache(&_eventDialCache_);
      dispenser.load();
    }

    LogInfo << "Resizing dial containers..." << std::endl;
    for( auto& dialCollection : _dialCollections_ ) {
      if( not dialCollection.isBinned() ){ dialCollection.resizeContainers(); }
    }

    LogInfo << "Build reference cache..." << std::endl;
    _eventDialCache_.shrinkIndexedCache();
    _eventDialCache_.buildReferenceCache(_fitSampleSet_, _dialCollections_);
  }

#ifdef GUNDAM_USING_CACHE_MANAGER
  // After all the data has been loaded.  Specifically, this must be after
  // the MC has been copied for the Asimov fit, or the "data" use the MC
  // reweighting cache.  This must also be before the first use of
  // reweightMcEvents.
  if(GundamGlobals::getEnableCacheManager()) {
    Cache::Manager::Build(getFitSampleSet(), getEventDialCache());
  }
#endif

  LogInfo << "Propagating prior parameters on events..." << std::endl;
  this->resetReweight();
  this->reweightMcEvents();

  LogInfo << "Set the current MC prior weights as nominal weight..." << std::endl;
  for( auto& sample : _fitSampleSet_.getFitSampleList() ){
    for( auto& event : sample.getMcContainer().eventList ){
      event.setNominalWeight(event.getEventWeight());
    }
  }

  LogInfo << "Filling up sample bin caches..." << std::endl;
  _fitSampleSet_.updateSampleBinEventList();

  LogInfo << "Filling up sample histograms..." << std::endl;
  _fitSampleSet_.updateSampleHistograms();

  // Throwing stat error on data -> BINNING SHOULD BE SET!!
  if( _throwAsimovToyParameters_ and _enableStatThrowInToys_ ){
    LogInfo << "Throwing statistical error for data container..." << std::endl;

    if( _enableEventMcThrow_ ){
      // Take into account the finite amount of event in MC
      LogInfo << "enableEventMcThrow is enabled: throwing individual MC events" << std::endl;
      for( auto& sample : _fitSampleSet_.getFitSampleList() ) {
        sample.getDataContainer().throwEventMcError();
      }
    }
    else{
      LogWarning << "enableEventMcThrow is disabled. Not throwing individual MC events" << std::endl;
    }

    LogInfo << "Throwing statistical error on histograms..." << std::endl;
    if( _gaussStatThrowInToys_ ) {
      LogWarning << "Using gaussian statistical throws. (caveat: distribution truncated when the bins are close to zero)" << std::endl;
    }
    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      // Asimov bin content -> toy data
      sample.getDataContainer().throwStatError(_gaussStatThrowInToys_);
    }
  }

  LogInfo << "Locking data event containers..." << std::endl;
  for( auto& sample : _fitSampleSet_.getFitSampleList() ){
    // Now the data won't be refilled each time
    sample.getDataContainer().isLocked = true;
  }

  if( not _parameterInjectorMc_.empty() ){
    LogWarning << "Injecting parameters on MC samples..." << std::endl;
    _parManager_.injectParameterValues( ConfigUtils::getForwardedConfig(_parameterInjectorMc_) );
    this->resetReweight();
    this->reweightMcEvents();
  }

  //////////////////////////////////////////
  // DON'T MOVE PARAMETERS FROM THIS POINT
  //////////////////////////////////////////

  /// Copy the current state of MC as "nominal" histogram
  LogInfo << "Copy the current state of MC as \"nominal\" histogram..." << std::endl;
  for( auto& sample : _fitSampleSet_.getFitSampleList() ){
    sample.getMcContainer().saveAsHistogramNominal();
  }

  /// Initialise other tools
  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing the plot generator") << std::endl;
  _plotGenerator_.setFitSampleSetPtr(&_fitSampleSet_);
  _plotGenerator_.initialize();

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing the tree writer") << std::endl;
  _treeWriter_.setFitSampleSetPtr( &_fitSampleSet_ );
  _treeWriter_.setParSetListPtr( &_parManager_.getParameterSetsList() );
  _treeWriter_.setEventDialCachePtr( &_eventDialCache_ );

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing the par scanner") << std::endl;
  _parScanner_.initialize();

  /// Printouts for quick monitoring
  if( _showEventBreakdown_ ){

    if(true){
      // STAGED MASK
      LogWarning << "Staged event breakdown:" << std::endl;
      std::vector<std::vector<double>> stageBreakdownList(
          _fitSampleSet_.getFitSampleList().size(),
          std::vector<double>(_parManager_.getParameterSetsList().size() + 1, 0)
      ); // [iSample][iStage]
      std::vector<std::string> stageTitles;
      stageTitles.emplace_back("Sample");
      stageTitles.emplace_back("No reweight");
      for( auto& parSet : _parManager_.getParameterSetsList() ){
        if( not parSet.isEnabled() ){ continue; }
        stageTitles.emplace_back("+ " + parSet.getName());
      }

      int iStage{0};
      std::vector<ParameterSet*> maskedParSetList;
      for( auto& parSet : _parManager_.getParameterSetsList() ){
        if( not parSet.isEnabled() ){ continue; }
        maskedParSetList.emplace_back( &parSet );
        parSet.setMaskedForPropagation( true );
      }

      this->resetReweight();
      this->reweightMcEvents();
      for( size_t iSample = 0 ; iSample < _fitSampleSet_.getFitSampleList().size() ; iSample++ ){
        stageBreakdownList[iSample][iStage] = _fitSampleSet_.getFitSampleList()[iSample].getMcContainer().getSumWeights();
      }

      for( auto* parSetPtr : maskedParSetList ){
        parSetPtr->setMaskedForPropagation(false);
        this->resetReweight();
        this->reweightMcEvents();
        iStage++;
        for( size_t iSample = 0 ; iSample < _fitSampleSet_.getFitSampleList().size() ; iSample++ ){
          stageBreakdownList[iSample][iStage] = _fitSampleSet_.getFitSampleList()[iSample].getMcContainer().getSumWeights();
        }
      }

      GenericToolbox::TablePrinter t;
      t.setColTitles(stageTitles);
      for( size_t iSample = 0 ; iSample < _fitSampleSet_.getFitSampleList().size() ; iSample++ ) {
        std::vector<std::string> tableLine;
        tableLine.emplace_back("\""+_fitSampleSet_.getFitSampleList()[iSample].getName()+"\"");
        for( iStage = 0 ; iStage < stageBreakdownList[iSample].size() ; iStage++ ){
          tableLine.emplace_back( std::to_string(stageBreakdownList[iSample][iStage]) );
        }
        t.addTableLine(tableLine);
      }
      t.printTable();
    }

    LogWarning << "Sample breakdown:" << std::endl;
    std::cout << getSampleBreakdownTableStr() << std::endl;

  }
  if( _debugPrintLoadedEvents_ ){
    LogDebug << GET_VAR_NAME_VALUE(_debugPrintLoadedEventsNbPerSample_) << std::endl;
    int iEvt{0};
    for( auto& entry : _eventDialCache_.getCache() ) {
      LogDebug << "Event #" << iEvt++ << "{" << std::endl;
      {
        LogScopeIndent;
        LogDebug << entry.getSummary() << std::endl;
      }
      LogDebug << "}" << std::endl;
      if( iEvt >= _debugPrintLoadedEventsNbPerSample_ ) break;
    }
  }

  /// Propagator needs to be responsive, let the workers wait for the signal
  GundamGlobals::getParallelWorker().setCpuTimeSaverIsEnabled(false);
}



// Misc getters
std::string Propagator::getLlhBufferSummary() const{
  std::stringstream ss;
  ss << "Total likelihood = " << getLlhBuffer();
  ss << std::endl << "Stat likelihood = " << getLlhStatBuffer();
  ss << " = sum of: " << GenericToolbox::iterableToString(
      _fitSampleSet_.getFitSampleList(), [](const Sample& sample_){
        std::stringstream ssSub;
        ssSub << sample_.getName() << ": ";
        if( sample_.isEnabled() ){ ssSub << sample_.getLlhStatBuffer(); }
        else                     { ssSub << "disabled."; }
        return ssSub.str();
      }
  );
  ss << std::endl << "Penalty likelihood = " << getLlhPenaltyBuffer();
  ss << " = sum of: " << GenericToolbox::iterableToString(
      _parManager_.getParameterSetsList(), [](const ParameterSet& parSet_){
        std::stringstream ssSub;
        ssSub << parSet_.getName() << ": ";
        if( parSet_.isEnabled() ){ ssSub << parSet_.getPenaltyChi2Buffer(); }
        else                     { ssSub << "disabled."; }
        return ssSub.str();
      }
  );
  return ss.str();
}

DatasetLoader* Propagator::getDatasetLoaderPtr(const std::string& name_){
  for( auto& dataSet : _dataSetList_ ){
    if( dataSet.getName() == name_ ){ return &dataSet; }
  }
  return nullptr;
}

// Core
void Propagator::updateLlhCache(){
  double buffer;

  // Propagate on histograms
  this->propagateParametersOnSamples();

  ////////////////////////////////
  // Compute LLH stat
  ////////////////////////////////
  _llhStatBuffer_ = _fitSampleSet_.evalLikelihood();

  ////////////////////////////////
  // Compute the penalty terms
  ////////////////////////////////
  _llhPenaltyBuffer_ = 0;
  for( auto& parSet : _parManager_.getParameterSetsList() ){
    buffer = parSet.getPenaltyChi2();
    LogThrowIf(std::isnan(buffer), parSet.getName() << " penalty chi2 is Nan");
    _llhPenaltyBuffer_ += buffer;
  }

  ////////////////////////////////
  // Compute the regularisation term
  ////////////////////////////////
  _llhRegBuffer_ = 0; // unused

  ////////////////////////////////
  // Total LLH
  ////////////////////////////////
  _llhBuffer_ = _llhStatBuffer_ + _llhPenaltyBuffer_ + _llhRegBuffer_;
}
void Propagator::propagateParametersOnSamples(){

  if( _enableEigenToOrigInPropagate_ ){
    // Only real parameters are propagated on the spectra -> need to convert the eigen to original
    for( auto& parSet : _parManager_.getParameterSetsList() ){
      if( parSet.isUseEigenDecompInFit() ) parSet.propagateEigenToOriginal();
    }
  }

  resetReweight();
  reweightMcEvents();
  refillSampleHistograms();

}
void Propagator::resetReweight(){
  std::for_each(_dialCollections_.begin(), _dialCollections_.end(),[&](DialCollection& dc_){
    dc_.updateInputBuffers();
  });
}
void Propagator::reweightMcEvents() {
  bool usedGPU{false};
#ifdef GUNDAM_USING_CACHE_MANAGER
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  if(GundamGlobals::getEnableCacheManager()) {
    Cache::Manager::Update(getFitSampleSet(), getEventDialCache());
    usedGPU = Cache::Manager::Fill();
  }
#endif
  if( not usedGPU ){
    GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
    if( not _devSingleThreadReweight_ ){ GundamGlobals::getParallelWorker().runJob("Propagator::reweightMcEvents"); }
    else{ this->reweightMcEvents(-1); }
  }
  weightProp.counts++;
  weightProp.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
}
void Propagator::refillSampleHistograms(){
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  if( not _devSingleThreadHistFill_ ){
    GundamGlobals::getParallelWorker().runJob("Propagator::refillSampleHistograms");
  }
  else{
    refillSampleHistogramsFct(-1);
    refillSampleHistogramsPostParallelFct();
  }
  fillProp.counts++; fillProp.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
}

// Misc
std::string Propagator::getSampleBreakdownTableStr() const{
  GenericToolbox::TablePrinter t;

  t << "Sample" << GenericToolbox::TablePrinter::NextColumn;
  t << "MC (# binned event)" << GenericToolbox::TablePrinter::NextColumn;
  t << "Data (# binned event)" << GenericToolbox::TablePrinter::NextColumn;
  t << "MC (weighted)" << GenericToolbox::TablePrinter::NextColumn;
  t << "Data (weighted)" << GenericToolbox::TablePrinter::NextLine;

  for( auto& sample : _fitSampleSet_.getFitSampleList() ){
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
void Propagator::throwParametersFromGlobalCovariance(std::vector<double> &weightsChiSquare){

    // check that weightsChiSquare is an empty vector
    LogThrowIf( not weightsChiSquare.empty(), "ERROR: argument weightsChiSquare is not empty" );

    if( _strippedCovarianceMatrix_ == nullptr ){
        LogInfo << "Creating stripped global covariance matrix..." << std::endl;
        LogThrowIf( _globalCovarianceMatrix_ == nullptr, "Global covariance matrix not set." );
        int nStripped{0};
        for( int iDiag = 0 ; iDiag < _globalCovarianceMatrix_->GetNrows() ; iDiag++ ){
            if( (*_globalCovarianceMatrix_)[iDiag][iDiag] != 0 ){ nStripped++; }
        }

        LogInfo << "Stripped global covariance matrix is " << nStripped << "x" << nStripped << std::endl;
        _strippedCovarianceMatrix_ = std::make_shared<TMatrixD>(nStripped, nStripped);
        int iStrippedBin{-1};
        for( int iBin = 0 ; iBin < _globalCovarianceMatrix_->GetNrows() ; iBin++ ){
            if( (*_globalCovarianceMatrix_)[iBin][iBin] == 0 ){ continue; }
            iStrippedBin++;
            int jStrippedBin{-1};
            for( int jBin = 0 ; jBin < _globalCovarianceMatrix_->GetNrows() ; jBin++ ){
                if( (*_globalCovarianceMatrix_)[jBin][jBin] == 0 ){ continue; }
                jStrippedBin++;
                (*_strippedCovarianceMatrix_)[iStrippedBin][jStrippedBin] = (*_globalCovarianceMatrix_)[iBin][jBin];
            }
        }

        _strippedParameterList_.reserve( nStripped );
        for( auto& parSet : _parameterSetList_ ){
            if( not parSet.isEnabled() ) continue;
            for( auto& par : parSet.getParameterList() ){
                if( not par.isEnabled() ) continue;
                _strippedParameterList_.emplace_back(&par);
            }
        }
        LogThrowIf( _strippedParameterList_.size() != nStripped, "Enabled parameters list don't correspond to the matrix" );
    }

    if( _choleskyMatrix_ == nullptr ){
        LogInfo << "Generating global cholesky matrix" << std::endl;
        _choleskyMatrix_ = std::shared_ptr<TMatrixD>(
                GenericToolbox::getCholeskyMatrix(_strippedCovarianceMatrix_.get())
        );
    }

    bool keepThrowing{true};
//  int throwNb{0};

    while( keepThrowing ){
//    throwNb++;
        bool rethrow{false};
        std::vector<double> throws,weights;
        GenericToolbox::throwCorrelatedParameters(_choleskyMatrix_.get(),throws, weights);
        if(throws.size() != weights.size()){
            LogInfo<<"WARNING: throws.size() != weights.size() "<< throws.size()<<weights.size()<<std::endl;
        }
        for( int iPar = 0 ; iPar < _choleskyMatrix_->GetNrows() ; iPar++ ){
            _strippedParameterList_[iPar]->setParameterValue(
                    _strippedParameterList_[iPar]->getPriorValue()
                    + throws[iPar]
            );
            weightsChiSquare.push_back(weights[iPar]);

            if( _reThrowParSetIfOutOfBounds_ ){
                if( not _strippedParameterList_[iPar]->isValueWithinBounds() ){
                    // re-do the throwing
//          LogDebug << "Not within bounds: " << _strippedParameterList_[iPar]->getSummary() << std::endl;
                    rethrow = true;
                }
            }
        }

        // Making sure eigen decomposed parameters get the conversion done
        for( auto& parSet : _parameterSetList_ ){
            if( not parSet.isEnabled() ) continue;
            if( parSet.isUseEigenDecompInFit() ){
                parSet.propagateOriginalToEigen();

                // also check the bounds of real parameter space
                if( _reThrowParSetIfOutOfBounds_ ){
                    for( auto& par : parSet.getEigenParameterList() ){
                        if( not par.isEnabled() ) continue;
                        if( not par.isValueWithinBounds() ){
                            // re-do the throwing
                            rethrow = true;
                            break;
                        }
                    }
                }
            }
        }


        if( rethrow ){
            // wrap back to the while loop
//      LogDebug << "RE-THROW #" << throwNb << std::endl;
            continue;
        }

        // reached this point: all parameters are within bounds
        keepThrowing = false;
    }



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

//  int nThreads{GundamGlobals::getParallelWorker().getNbThreads()};
//  if( iThread_ == -1 ){ iThread_ = 0 ; nThreads = 1; }
//
//  int iCache{iThread_};
//  int nCache{int(_eventDialCache_.getCache().size())};
//
//  while( iCache < nCache ){
//    EventDialCache::reweightEntry(_eventDialCache_.getCache()[iCache]);
//    iCache += nThreads;
//  }

  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, GundamGlobals::getParallelWorker().getNbThreads(),
      int(_eventDialCache_.getCache().size())
  );

  std::for_each(
      _eventDialCache_.getCache().begin() + bounds.first,
      _eventDialCache_.getCache().begin() + bounds.second,
      &EventDialCache::reweightEntry
  );

}
void Propagator::refillSampleHistogramsFct(int iThread_){
  for( auto& sample : _fitSampleSet_.getFitSampleList() ){
    sample.getMcContainer().refillHistogram(iThread_);
    sample.getDataContainer().refillHistogram(iThread_);
  }
}
void Propagator::refillSampleHistogramsPostParallelFct(){
  for( auto& sample : _fitSampleSet_.getFitSampleList() ){
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
