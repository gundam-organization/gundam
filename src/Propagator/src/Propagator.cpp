//
// Created by Nadrino on 11/06/2021.
//

#include "Propagator.h"

#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif

#include "FitParameterSet.h"
#include "GenericToolbox.Json.h"
#include "GundamGlobals.h"
#include "ConfigUtils.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.TablePrinter.h"
#include "GenericToolbox.ScopedGuard.h"

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

  // Monitoring parameters
  _showEventBreakdown_ = GenericToolbox::Json::fetchValue(_config_, "showEventBreakdown", _showEventBreakdown_);
  _throwAsimovToyParameters_ = GenericToolbox::Json::fetchValue(_config_, "throwAsimovFitParameters", _throwAsimovToyParameters_);
  _throwToyParametersWithGlobalCov_ = GenericToolbox::Json::fetchValue(_config_, "throwToyParametersWithGlobalCov", _throwToyParametersWithGlobalCov_);
  _reThrowParSetIfOutOfBounds_ = GenericToolbox::Json::fetchValue(_config_, "reThrowParSetIfOutOfBounds", _reThrowParSetIfOutOfBounds_);
  _enableStatThrowInToys_ = GenericToolbox::Json::fetchValue(_config_, "enableStatThrowInToys", _enableStatThrowInToys_);
  _gaussStatThrowInToys_ = GenericToolbox::Json::fetchValue(_config_, "gaussStatThrowInToys", _gaussStatThrowInToys_);
  _enableEventMcThrow_ = GenericToolbox::Json::fetchValue(_config_, "enableEventMcThrow", _enableEventMcThrow_);

  // EventDialCache parameters
  EventDialCache::globalEventReweightCap = GenericToolbox::Json::fetchValue(_config_, "globalEventReweightCap", EventDialCache::globalEventReweightCap);

  auto parameterSetListConfig = ConfigUtils::getForwardedConfig(GenericToolbox::Json::fetchValue(_config_, "parameterSetListConfig", nlohmann::json()));
  _parameterSetList_.reserve(parameterSetListConfig.size()); // make sure the objects aren't moved in RAM ( since FitParameter* will be used )
  for( const auto& parameterSetConfig : parameterSetListConfig ){
    _parameterSetList_.emplace_back();
    _parameterSetList_.back().setConfig(parameterSetConfig);
    _parameterSetList_.back().readConfig();
    LogInfo << _parameterSetList_.back().getSummary() << std::endl;
  }

  auto fitSampleSetConfig = GenericToolbox::Json::fetchValue(_config_, "fitSampleSetConfig", nlohmann::json());
  _fitSampleSet_.setConfig(fitSampleSetConfig);
  _fitSampleSet_.readConfig();

  auto plotGeneratorConfig = ConfigUtils::getForwardedConfig(GenericToolbox::Json::fetchValue(_config_, "plotGeneratorConfig", nlohmann::json()));
  _plotGenerator_.setConfig(plotGeneratorConfig);
  _plotGenerator_.readConfig();

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

  _parScanner_.readConfig( GenericToolbox::Json::fetchValue(_config_, "scanConfig", nlohmann::json()) );

  _debugPrintLoadedEvents_ = GenericToolbox::Json::fetchValue(_config_, "debugPrintLoadedEvents", _debugPrintLoadedEvents_);
  _debugPrintLoadedEventsNbPerSample_ = GenericToolbox::Json::fetchValue(_config_, "debugPrintLoadedEventsNbPerSample", _debugPrintLoadedEventsNbPerSample_);

  _devSingleThreadReweight_ = GenericToolbox::Json::fetchValue(_config_, "devSingleThreadReweight", _devSingleThreadReweight_);
  _devSingleThreadHistFill_ = GenericToolbox::Json::fetchValue(_config_, "devSingleThreadHistFill", _devSingleThreadHistFill_);

  for(size_t iParSet = 0 ; iParSet < _parameterSetList_.size() ; iParSet++ ){
    if( not _parameterSetList_[iParSet].isEnabled() ) continue;
    // DEV / DialCollections
    if( not _parameterSetList_[iParSet].getDialSetDefinitions().empty() ){
      for( auto& dialSetDef : _parameterSetList_[iParSet].getDialSetDefinitions().get<std::vector<nlohmann::json>>() ){
        if( GenericToolbox::Json::doKeyExist(dialSetDef, "parametersBinningPath") ){
          _dialCollections_.emplace_back(&_parameterSetList_);
          _dialCollections_.back().setIndex(int(_dialCollections_.size())-1);
          _dialCollections_.back().setSupervisedParameterSetIndex( int(iParSet) );
          _dialCollections_.back().readConfig( dialSetDef );
        }
        else{ LogThrow("no parametersBinningPath option?"); }
      }
    }
    else{
      for( auto& par : _parameterSetList_[iParSet].getParameterList() ){
        if( not par.isEnabled() ) continue;

        // Check if no definition is present -> disable the parameter in that case
        if( par.getDialDefinitionsList().empty() ) {
          LogAlert << "Disabling \"" << par.getFullTitle() << "\": no dial definition." << std::endl;
          par.setIsEnabled(false);
          continue;
        }

        for( const auto& dialDefinitionConfig : par.getDialDefinitionsList() ){
          _dialCollections_.emplace_back(&_parameterSetList_);
          _dialCollections_.back().setIndex(int(_dialCollections_.size())-1);
          _dialCollections_.back().setSupervisedParameterSetIndex( int(iParSet) );
          _dialCollections_.back().setSupervisedParameterIndex( par.getParameterIndex() );
          _dialCollections_.back().readConfig( dialDefinitionConfig );
        }
      }
    }
  }

  _treeWriter_.readConfig( GenericToolbox::Json::fetchValue(_config_, "eventTreeWriter", nlohmann::json()) );

  _parameterInjectorMc_ = GenericToolbox::Json::fetchValue(_config_, "parameterInjection", _parameterInjectorMc_);
  ConfigUtils::forwardConfig(_parameterInjectorMc_);

}
void Propagator::initializeImpl(){
  LogWarning << __METHOD_NAME__ << std::endl;

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing parameters...") << std::endl;
  int nEnabledPars = 0;
  for( auto& parSet : _parameterSetList_ ){
    parSet.initialize();

    int nPars{0};
    for( auto& par : parSet.getParameterList() ){
      if( par.isEnabled() ){ nPars++; }
    }

    nEnabledPars += nPars;
    LogInfo << nPars << " enabled parameters in " << parSet.getName() << std::endl;
  }
  LogInfo << "Total number of parameters: " << nEnabledPars << std::endl;

  LogInfo << "Building global covariance matrix (" << nEnabledPars << "x" << nEnabledPars << ")" << std::endl;
  _globalCovarianceMatrix_ = std::make_shared<TMatrixD>(nEnabledPars, nEnabledPars );
  int parSetOffset = 0;
  for( auto& parSet : _parameterSetList_ ){
    if( parSet.getPriorCovarianceMatrix() != nullptr ){
      int iGlobalOffset{-1};
      bool hasZero{false};
      for(int iCov = 0 ; iCov < parSet.getPriorCovarianceMatrix()->GetNrows() ; iCov++ ){
        if( not parSet.getParameterList()[iCov].isEnabled() ){ continue; }
        iGlobalOffset++;
        _globalCovParList_.emplace_back( &parSet.getParameterList()[iCov] );
        int jGlobalOffset{-1};
        for(int jCov = 0 ; jCov < parSet.getPriorCovarianceMatrix()->GetNcols() ; jCov++ ){
          if( not parSet.getParameterList()[jCov].isEnabled() ){ continue; }
          jGlobalOffset++;
          (*_globalCovarianceMatrix_)[parSetOffset + iGlobalOffset][parSetOffset + jGlobalOffset] = (*parSet.getPriorCovarianceMatrix())[iCov][jCov];
        }
      }
      parSetOffset += (iGlobalOffset+1);
    }
    else{
      // diagonal
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ){ continue; }
        _globalCovParList_.emplace_back(&par);
        if( par.isFree() ){
          (*_globalCovarianceMatrix_)[parSetOffset][parSetOffset] = 0;
        }
        else{
          (*_globalCovarianceMatrix_)[parSetOffset][parSetOffset] = par.getStdDevValue() * par.getStdDevValue();
        }
        parSetOffset++;
      }
    }
  }

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
      dispenser.setParSetPtrToLoad(&_parameterSetList_);
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
      }



      if( not _throwToyParametersWithGlobalCov_ ){
        LogInfo << "Throwing parameter using each parameter sets..." << std::endl;
        for( auto& parSet : _parameterSetList_ ){
          if( not parSet.isEnabled() ) continue;

          LogContinueIf( not parSet.isEnabledThrowToyParameters(), "Toy throw is disabled for " << parSet.getName() );

          if( parSet.getPriorCovarianceMatrix() != nullptr ){
            LogWarning << parSet.getName() << ": throwing correlated parameters..." << std::endl;
            LogScopeIndent;
            parSet.throwFitParameters(_reThrowParSetIfOutOfBounds_);
          } // throw?
          else{
            LogAlert << "No correlation matrix defined for " << parSet.getName() << ". NOT THROWING. (dev: could throw only with sigmas?)" << std::endl;
          }
        } // parSet
      }
      else{
        LogInfo << "Throwing parameter using global covariance matrix..." << std::endl;
        this->throwParametersFromGlobalCovariance(false);
      }

      // Handling possible masks
      for( auto& parSet : _parameterSetList_ ){
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
    for( auto& parSet: _parameterSetList_ ) {
      if( parSet.isUseEigenDecompInFit() ) { parSet.propagateEigenToOriginal(); }
    }

    bool cacheManagerState = GundamGlobals::getEnableCacheManager();
    GundamGlobals::setEnableCacheManager(false);
    this->resetReweight();
    this->reweightMcEvents();
    GundamGlobals::setEnableCacheManager(cacheManagerState);

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
    // Copies MC events in data container for both Asimov and FakeData event types
    LogWarning << "Copying loaded mc-like event to data container..." << std::endl;
    _fitSampleSet_.copyMcEventListToDataContainer();

    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      sample.getDataContainer().histScale = sample.getMcContainer().histScale;
    }

    // back to prior
    if( _throwAsimovToyParameters_ ){
      for( auto& parSet : _parameterSetList_ ){

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
      dispenser.setParSetPtrToLoad(&_parameterSetList_);
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
    this->injectParameterValues(_parameterInjectorMc_);
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
  _treeWriter_.setParSetListPtr( &_parameterSetList_ );
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
          std::vector<double>(_parameterSetList_.size() + 1, 0)
      ); // [iSample][iStage]
      std::vector<std::string> stageTitles;
      stageTitles.emplace_back("Sample");
      stageTitles.emplace_back("No reweight");
      for( auto& parSet : _parameterSetList_ ){
        if( not parSet.isEnabled() ){ continue; }
        stageTitles.emplace_back("+ " + parSet.getName());
      }

      int iStage{0};
      std::vector<FitParameterSet*> maskedParSetList;
      for( auto& parSet : _parameterSetList_ ){
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
      _fitSampleSet_.getFitSampleList(), [](const FitSample& sample_){
        std::stringstream ssSub;
        ssSub << sample_.getName() << ": ";
        if( sample_.isEnabled() ){ ssSub << sample_.getLlhStatBuffer(); }
        else                     { ssSub << "disabled."; }
        return ssSub.str();
      }
  );
  ss << std::endl << "Penalty likelihood = " << getLlhPenaltyBuffer();
  ss << " = sum of: " << GenericToolbox::iterableToString(
      _parameterSetList_, [](const FitParameterSet& parSet_){
        std::stringstream ssSub;
        ssSub << parSet_.getName() << ": ";
        if( parSet_.isEnabled() ){ ssSub << parSet_.getPenaltyChi2Buffer(); }
        else                     { ssSub << "disabled."; }
        return ssSub.str();
      }
  );
  return ss.str();
}
std::string Propagator::getParametersSummary( bool showEigen_ ) const{
  std::stringstream ss;
  for( auto &parSet: getParameterSetsList() ){
    if( not parSet.isEnabled() ){ continue; }
    if( not ss.str().empty() ) ss << std::endl;
    ss << parSet.getName();
    for( auto &par: parSet.getParameterList() ){
      if( not par.isEnabled() ){ continue; }
      ss << std::endl << "  " << par.getTitle() << ": " << par.getParameterValue();
    }
  }
  return ss.str();
}
const FitParameterSet* Propagator::getFitParameterSetPtr(const std::string& name_) const{
  for( auto& parSet : _parameterSetList_ ){
    if( parSet.getName() == name_ ) return &parSet;
  }
  std::vector<std::string> parSetNames{};
  for( auto& parSet : _parameterSetList_ ){ parSetNames.emplace_back(parSet.getName()); }
  LogThrow("Could not find fit parameter set named \"" << name_ << "\" among defined: " << GenericToolbox::parseVectorAsString(parSetNames));
  return nullptr;
}
FitParameterSet* Propagator::getFitParameterSetPtr(const std::string& name_){
  for( auto& parSet : _parameterSetList_ ){
    if( parSet.getName() == name_ ) return &parSet;
  }
  std::vector<std::string> parSetNames{};
  for( auto& parSet : _parameterSetList_ ){ parSetNames.emplace_back(parSet.getName()); }
  LogThrow("Could not find fit parameter set named \"" << name_ << "\" among defined: " << GenericToolbox::parseVectorAsString(parSetNames));
  return nullptr;
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
  for( auto& parSet : _parameterSetList_ ){
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
    for( auto& parSet : _parameterSetList_ ){
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
nlohmann::json Propagator::exportParameterInjectorConfig() const{
  nlohmann::json out;

  std::vector<nlohmann::json> parSetConfig;
  parSetConfig.reserve( _parameterSetList_.size() );
  for( auto& parSet : _parameterSetList_ ){
    if( not parSet.isEnabled() ){ continue; }
    parSetConfig.emplace_back( parSet.exportInjectorConfig() );
  }

  out["parameterSetList"] = parSetConfig;

  out = GenericToolbox::Json::readConfigJsonStr(
      // conversion: json -> str -> json obj (some broken JSON version)
      GenericToolbox::Json::toReadableString(
          out
      )
  );

  return out;
}

void Propagator::injectParameterValues(const nlohmann::json &config_) {
  LogWarning << "Injecting parameters..." << std::endl;

  if( not GenericToolbox::Json::doKeyExist(config_, "parameterSetList") ){
    LogError << "Bad parameter injector config: missing \"parameterSetList\" entry" << std::endl;
    LogError << GenericToolbox::Json::toReadableString( config_ ) << std::endl;
    return;
  }

  for( auto& entryParSet : GenericToolbox::Json::fetchValue<nlohmann::json>( config_, "parameterSetList" ) ){
    auto parSetName = GenericToolbox::Json::fetchValue<std::string>(entryParSet, "name");
    LogInfo << "Reading injection parameters for parSet: " << parSetName << std::endl;

    auto* selectedParSet = this->getFitParameterSetPtr(parSetName );
    LogThrowIf( selectedParSet == nullptr, "Could not find parSet: " << parSetName );

    selectedParSet->injectParameterValues(entryParSet);
  }
}

void Propagator::throwParametersFromGlobalCovariance(bool quietVerbose_){

  if( _strippedCovarianceMatrix_ == nullptr ){
    LogInfo << "Creating stripped global covariance matrix..." << std::endl;
    LogThrowIf( _globalCovarianceMatrix_ == nullptr, "Global covariance matrix not set." );

    _strippedParameterList_.clear();
    for( int iGlobPar = 0 ; iGlobPar < _globalCovarianceMatrix_->GetNrows() ; iGlobPar++ ){
      if( _globalCovParList_[iGlobPar]->isFixed() ){ continue; }
      if( _globalCovParList_[iGlobPar]->isFree() and (*_globalCovarianceMatrix_)[iGlobPar][iGlobPar] == 0 ){ continue; }
      _strippedParameterList_.emplace_back( _globalCovParList_[iGlobPar] );
    }

    int nStripped{int(_strippedParameterList_.size())};
    _strippedCovarianceMatrix_ = std::make_shared<TMatrixD>(nStripped, nStripped);

    for( int iStrippedPar = 0 ; iStrippedPar < nStripped ; iStrippedPar++ ){
      int iGlobPar{GenericToolbox::findElementIndex(_strippedParameterList_[iStrippedPar], _globalCovParList_)};
      for( int jStrippedPar = 0 ; jStrippedPar < nStripped ; jStrippedPar++ ){
        int jGlobPar{GenericToolbox::findElementIndex(_strippedParameterList_[jStrippedPar], _globalCovParList_)};
        (*_strippedCovarianceMatrix_)[iStrippedPar][jStrippedPar] = (*_globalCovarianceMatrix_)[iGlobPar][jGlobPar];
      }
    }
  }

  bool isLoggerAlreadyMuted{Logger::isMuted()};
  GenericToolbox::ScopedGuard g{
      [&](){ if(quietVerbose_ and not isLoggerAlreadyMuted) Logger::setIsMuted(true); },
      [&](){ if(quietVerbose_ and not isLoggerAlreadyMuted) Logger::setIsMuted(false); }
  };

  if(quietVerbose_){
    Logger::setIsMuted(quietVerbose_);
  }

  if( _choleskyMatrix_ == nullptr ){
    LogInfo << "Generating global cholesky matrix" << std::endl;
    _choleskyMatrix_ = std::shared_ptr<TMatrixD>(
        GenericToolbox::getCholeskyMatrix(_strippedCovarianceMatrix_.get())
    );
  }

  bool keepThrowing{true};
  int throwNb{0};

  while( keepThrowing ){
    throwNb++;
    bool rethrow{false};
    auto throws = GenericToolbox::throwCorrelatedParameters(_choleskyMatrix_.get());
    for( int iPar = 0 ; iPar < _choleskyMatrix_->GetNrows() ; iPar++ ){
      auto* parPtr = _strippedParameterList_[iPar];
      parPtr->setParameterValue( parPtr->getPriorValue() + throws[iPar] );
      if( _reThrowParSetIfOutOfBounds_ ){
        if      ( not std::isnan(parPtr->getMinValue()) and parPtr->getParameterValue() < parPtr->getMinValue() ){
          rethrow = true;
          LogAlert << GenericToolbox::ColorCodes::redLightText << "thrown value lower than min bound -> " << GenericToolbox::ColorCodes::resetColor
                   << parPtr->getSummary(true) << std::endl;
        }
        else if( not std::isnan(parPtr->getMaxValue()) and parPtr->getParameterValue() > parPtr->getMaxValue() ){
          rethrow = true;
          LogAlert << GenericToolbox::ColorCodes::redLightText <<"thrown value higher than max bound -> " << GenericToolbox::ColorCodes::resetColor
                   << parPtr->getSummary(true) << std::endl;
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
      LogWarning << "Re-throwing attempt #" << throwNb << std::endl;
      continue;
    }
    else{
      for( auto& parSet : _parameterSetList_ ){
        LogInfo << parSet.getName() << ":" << std::endl;
        for( auto& par : parSet.getParameterList() ){
          LogScopeIndent;
          if( FitParameterSet::isValidCorrelatedParameter(par) ){
            par.setThrowValue( par.getParameterValue() );
            LogInfo << "Thrown par " << par.getFullTitle() << ": " << par.getPriorValue();
            LogInfo << " → " << par.getParameterValue() << std::endl;
          }
        }
        if( parSet.isUseEigenDecompInFit() ){
          LogInfo << "Translated to eigen space:" << std::endl;
          for( auto& eigenPar : parSet.getEigenParameterList() ){
            LogScopeIndent;
            eigenPar.setThrowValue( eigenPar.getParameterValue() );
            LogInfo << "Eigen par " << eigenPar.getFullTitle() << ": " << eigenPar.getPriorValue();
            LogInfo << " → " << eigenPar.getParameterValue() << std::endl;
          }
        }
      }

    }




    // reached this point: all parameters are within bounds
    keepThrowing = false;
  }
}
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

// Protected
void Propagator::initializeThreads() {
  reweightMcEventsFct = [this](int iThread){
    this->reweightMcEvents(iThread);
  };
  GundamGlobals::getParallelWorker().addJob("Propagator::reweightMcEvents", reweightMcEventsFct);

  refillSampleHistogramsFct = [this](int iThread){
    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      sample.getMcContainer().refillHistogram(iThread);
      sample.getDataContainer().refillHistogram(iThread);
    }
  };
  refillSampleHistogramsPostParallelFct = [this](){
    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      sample.getMcContainer().rescaleHistogram();
      sample.getDataContainer().rescaleHistogram();
    }
  };
  GundamGlobals::getParallelWorker().addJob("Propagator::refillSampleHistograms", refillSampleHistogramsFct);
  GundamGlobals::getParallelWorker().setPostParallelJob("Propagator::refillSampleHistograms", refillSampleHistogramsPostParallelFct);
}

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
      &EventDialCache::reweightEntry
  );

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
