//
// Created by Nadrino on 11/06/2021.
//

#include "Propagator.h"

#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif

#include "FitParameterSet.h"
#ifndef USE_NEW_DIALS
#include "Dial.h"
#endif
#include "GenericToolbox.Json.h"
#include "GlobalVariables.h"
#include "ConfigUtils.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.TablePrinter.h"

#include <memory>
#include <vector>

LoggerInit([]{
  Logger::setUserHeaderStr("[Propagator]");
});


void Propagator::readConfigImpl(){
  LogWarning << __METHOD_NAME__ << std::endl;

  // Monitoring parameters
  _showEventBreakdown_ = GenericToolbox::Json::fetchValue(_config_, "showEventBreakdown", _showEventBreakdown_);
  _throwAsimovToyParameters_ = GenericToolbox::Json::fetchValue(_config_, "throwAsimovFitParameters", _throwAsimovToyParameters_);
  _reThrowParSetIfOutOfBounds_ = GenericToolbox::Json::fetchValue(_config_, "reThrowParSetIfOutOfBounds", _reThrowParSetIfOutOfBounds_);
  _enableStatThrowInToys_ = GenericToolbox::Json::fetchValue(_config_, "enableStatThrowInToys", _enableStatThrowInToys_);
  _gaussStatThrowInToys_ = GenericToolbox::Json::fetchValue(_config_, "gaussStatThrowInToys", _gaussStatThrowInToys_);
  _enableEventMcThrow_ = GenericToolbox::Json::fetchValue(_config_, "enableEventMcThrow", _enableEventMcThrow_);

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

#if USE_NEW_DIALS
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
#endif

  _parameterInjectorMc_ = GenericToolbox::Json::fetchValue(_config_, "parameterInjection", _parameterInjectorMc_);
  ConfigUtils::forwardConfig(_parameterInjectorMc_);

}
void Propagator::initializeImpl() {
  LogWarning << __METHOD_NAME__ << std::endl;

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing parameters...") << std::endl;
  int nPars = 0;
  for( auto& parSet : _parameterSetList_ ){
    parSet.initialize();
    nPars += int(parSet.getNbParameters());
  }
  LogInfo << "Total number of parameters: " << nPars << std::endl;

  if( _globalCovarianceMatrix_ == nullptr ){
    LogInfo << "Building global covariance matrix..." << std::endl;
    _globalCovarianceMatrix_ = std::make_shared<TMatrixD>( nPars, nPars );
    int iParOffset = 0;
    for( const auto& parSet : _parameterSetList_ ){
      if( not parSet.isEnabled() ) continue;
      if(parSet.getPriorCovarianceMatrix() != nullptr ){
        for(int iCov = 0 ; iCov < parSet.getPriorCovarianceMatrix()->GetNrows() ; iCov++ ){
          for(int jCov = 0 ; jCov < parSet.getPriorCovarianceMatrix()->GetNcols() ; jCov++ ){
            (*_globalCovarianceMatrix_)[iParOffset+iCov][iParOffset+jCov] = (*parSet.getPriorCovarianceMatrix())[iCov][jCov];
          }
        }
        iParOffset += parSet.getPriorCovarianceMatrix()->GetNrows();
      }
    }
  }
  else{
//    LogInfo << "Global covariance matrix is already set. Checking dimensions..." << std::endl;
//    LogThrowIf(_globalCovarianceMatrix_->GetNrows() != nPars or _globalCovarianceMatrix_->GetNcols() != nPars,
//               "The provided covariance matrix don't have the right size: " << nPars << "x" << nPars
//               << " / " << _globalCovarianceMatrix_->GetNrows() << " x " << _globalCovarianceMatrix_->GetNcols());
  }


  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing samples...") << std::endl;
  _fitSampleSet_.initialize();

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing " + std::to_string(_dataSetList_.size()) + " datasets...") << std::endl;
  for( auto& dataset : _dataSetList_ ){ dataset.initialize(); }

#if USE_NEW_DIALS
  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing dials...") << std::endl;
  for( auto& dialCollection : _dialCollections_ ){ dialCollection.initialize(); }
#endif

  LogInfo << "Initializing propagation threads..." << std::endl;
  initializeThreads();
  GlobalVariables::getParallelWorker().setCpuTimeSaverIsEnabled(true);

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
#if USE_NEW_DIALS
      dispenser.setDialCollectionListPtr(&_dialCollections_);
      dispenser.setEventDialCache(&_eventDialCache_);
#endif
    }
    dispenser.load();
  }

#if USE_NEW_DIALS
  LogInfo << "Resizing dial containers..." << std::endl;
  for( auto& dialCollection : _dialCollections_ ) {
    if( not dialCollection.isBinned() ){ dialCollection.resizeContainers(); }
  }

  LogInfo << "Build reference cache..." << std::endl;
  _eventDialCache_.buildReferenceCache(_fitSampleSet_, _dialCollections_);
#endif


  // Copy to data container
  if( usedMcContainer ){
    if( _throwAsimovToyParameters_ ){
      LogInfo << "Throwing asimov toy parameters..." << std::endl;

      for( auto& parSet : _parameterSetList_ ){
        if( not parSet.isEnabled() ) continue;

        bool keepThrow{false};
        if( parSet.isEnabledThrowToyParameters() and parSet.getPriorCovarianceMatrix() != nullptr ){

          while( not keepThrow ){
            parSet.throwFitParameters();
            keepThrow = true; // keep by default

            if( _reThrowParSetIfOutOfBounds_ ){
              LogInfo << "Checking if the thrown parameters of the set are within bounds..." << std::endl;

              for( auto& par : parSet.getParameterList() ){
                if( not std::isnan(par.getMinValue()) and par.getParameterValue() < par.getMinValue() ){
                  keepThrow = false;
                  LogAlert << par.getFullTitle() << ": thrown value lower than min bound ->" << std::endl;
                  LogAlert << par.getSummary(true) << std::endl;
                }
                else if( not std::isnan(par.getMaxValue()) and par.getParameterValue() > par.getMaxValue() ){
                  keepThrow = false;
                  LogAlert << par.getFullTitle() << ": thrown value higher than max bound ->" << std::endl;
                  LogAlert << par.getSummary(true) << std::endl;
                }
              }

              if( not keepThrow ){
                LogAlert << "Rethrowing \"" << parSet.getName() << "\"..." << std::endl;
              }
              else{
                LogWarning << "Keeping throw..." << std::endl;
              }
            } // check bounds?
          } // keep?
        } // throw?
      } // parSet
    } // throw asimov?

    LogInfo << "Propagating parameters on events..." << std::endl;

    // Make sure before the copy to the data:
    // At this point, MC events have been reweighted using their prior
    // but when using eigen decomp, the conversion eigen -> original has a small computational error
    for( auto& parSet: _parameterSetList_ ) {
      if( parSet.isUseEigenDecompInFit() ) { parSet.propagateEigenToOriginal(); }
    }

    bool cacheManagerState = GlobalVariables::getEnableCacheManager();
    GlobalVariables::setEnableCacheManager(false);
    this->resetReweight();
    this->reweightMcEvents();
    GlobalVariables::setEnableCacheManager(cacheManagerState);

    // Copies MC events in data container for both Asimov and FakeData event types
    LogWarning << "Copying loaded mc-like event to data container..." << std::endl;
    _fitSampleSet_.copyMcEventListToDataContainer();

    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      sample.getDataContainer().histScale = sample.getMcContainer().histScale;
    }

    // back to prior
    if( _throwAsimovToyParameters_ ){
      for( auto& parSet : _parameterSetList_ ){
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
#if USE_NEW_DIALS
    for( auto& dialCollection: _dialCollections_ ) {
      if( not dialCollection.getGlobalDialLeafName().empty() ) {
        dialCollection.clear();
      }
    }
    _eventDialCache_ = EventDialCache();
#else
    for( auto& parSet : _parameterSetList_ ){
      for( auto& par : parSet.getParameterList() ){
        for( auto& dialSet : par.getDialSetList() ){
          if( dialSet.getDialLeafName().empty() ) continue;
          dialSet.getDialList().clear();
        }
      }
    }
#endif

    for( auto& dataSet : _dataSetList_ ){
      LogContinueIf(not dataSet.isEnabled(), "Dataset \"" << dataSet.getName() << "\" is disabled. Skipping");
      auto& dispenser = dataSet.getMcDispenser();
      dispenser.setSampleSetPtrToLoad(&_fitSampleSet_);
      dispenser.setPlotGenPtr(&_plotGenerator_);
      dispenser.setParSetPtrToLoad(&_parameterSetList_);
#if USE_NEW_DIALS
      dispenser.setDialCollectionListPtr(&_dialCollections_);
      dispenser.setEventDialCache(&_eventDialCache_);
#endif
      dispenser.load();
    }

#if USE_NEW_DIALS
    LogInfo << "Resizing dial containers..." << std::endl;
    for( auto& dialCollection : _dialCollections_ ) {
      if( not dialCollection.isBinned() ){ dialCollection.resizeContainers(); }
    }

    LogInfo << "Build reference cache..." << std::endl;
    _eventDialCache_.buildReferenceCache(_fitSampleSet_, _dialCollections_);
#endif
  }

#ifdef GUNDAM_USING_CACHE_MANAGER
  // After all the data has been loaded.  Specifically, this must be after
  // the MC has been copied for the Asimov fit, or the "data" use the MC
  // reweighting cache.  This must also be before the first use of
  // reweightMcEvents.
  if(GlobalVariables::getEnableCacheManager()) {
#ifdef USE_NEW_DIALS
      Cache::Manager::Build(getFitSampleSet(), _eventDialCache_);
#else
      Cache::Manager::Build(getFitSampleSet());
#endif
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
    this->injectParameterOnMcSamples(_parameterInjectorMc_);
    this->resetReweight();
    this->reweightMcEvents();
  }

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing the plot generator") << std::endl;
  _plotGenerator_.setFitSampleSetPtr(&_fitSampleSet_);
  _plotGenerator_.initialize();

  LogInfo << "Saving nominal histograms..." << std::endl;
  for( auto& sample : _fitSampleSet_.getFitSampleList() ){
    sample.getMcContainer().saveAsHistogramNominal();
  }

  _treeWriter_.setFitSampleSetPtr(&_fitSampleSet_);
  _treeWriter_.setParSetListPtr(&_parameterSetList_);

  _parScanner_.initialize();

  if( _showEventBreakdown_ ){

    if(true){
      // STAGED MASK
      LogWarning << "Staged event breakdown:" << std::endl;
#ifndef USE_NEW_DIALS
      Dial::enableMaskCheck = true;
#endif
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
      for( auto& parSet : _parameterSetList_ ){
        if( not parSet.isEnabled() ){ continue; }
        parSet.setMaskedForPropagation(true);
      }
      this->resetReweight();
      this->reweightMcEvents();
      for( size_t iSample = 0 ; iSample < _fitSampleSet_.getFitSampleList().size() ; iSample++ ){
        stageBreakdownList[iSample][iStage] = _fitSampleSet_.getFitSampleList()[iSample].getMcContainer().getSumWeights();
      }

      for( auto& parSet : _parameterSetList_ ){
        if( not parSet.isEnabled() ){ continue; }
        parSet.setMaskedForPropagation(false);
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
#ifndef USE_NEW_DIALS
      Dial::enableMaskCheck = false;
#endif
    }

    LogWarning << "Sample breakdown:" << std::endl;
    GenericToolbox::TablePrinter t;
    t.setColTitles({{"Sample"},{"MC (# binned event)"},{"Data (# binned event)"}, {"MC (weighted)"}, {"Data (weighted)"}});
    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      t.addTableLine({{"\""+sample.getName()+"\""},
                      std::to_string(sample.getMcContainer().getNbBinnedEvents()),
                      std::to_string(sample.getDataContainer().getNbBinnedEvents()),
                      std::to_string(sample.getMcContainer().getSumWeights()),
                      std::to_string(sample.getDataContainer().getSumWeights())
                     });
    }
    t.printTable();
  }

  if( _debugPrintLoadedEvents_ ){

    LogDebug << GET_VAR_NAME_VALUE(_debugPrintLoadedEventsNbPerSample_) << std::endl;
#if USE_NEW_DIALS
    int iEvt{0};
    for( auto& entry : _eventDialCache_.getCache() ) {
      LogDebug << "Event #" << iEvt++ << "{" << std::endl;
      {
        LogScopeIndent;
        LogDebug << entry.event->getSummary() << std::endl;
        LogDebug << "dialCache = {";
        for( auto& dialInterface : entry.dials ) {
#ifndef USE_BREAKDOWN_CACHE
          LogDebug << std::endl << "  - " << dialInterface->getSummary();
#else
          LogDebug << std::endl << "  - " << dialInterface.interface->getSummary();
#endif
        }
        LogDebug << std::endl << "}" << std::endl;
      }
      LogDebug << "}" << std::endl;
      if( iEvt >= _debugPrintLoadedEventsNbPerSample_ ) break;
    }
#else
    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      LogDebug << GenericToolbox::addUpDownBars( sample.getName() ) << std::endl;

      int iEvt=0;
      for( auto& ev : sample.getMcContainer().eventList ){
        if(iEvt++ >= _debugPrintLoadedEventsNbPerSample_) { LogTrace << std::endl; break; }
        LogTrace << iEvt << " -> " << ev.getSummary() << std::endl;
      }
    }
#endif
  }

  _llhPenaltyPerParSet_.resize(_parameterSetList_.size(), 0);

  // Propagator needs to be fast
  GlobalVariables::getParallelWorker().setCpuTimeSaverIsEnabled(false);
}

void Propagator::setShowTimeStats(bool showTimeStats) {
  _showTimeStats_ = showTimeStats;
}
void Propagator::setThrowAsimovToyParameters(bool throwAsimovToyParameters) {
  _throwAsimovToyParameters_ = throwAsimovToyParameters;
}
void Propagator::setEnableEigenToOrigInPropagate(bool enableEigenToOrigInPropagate) {
  _enableEigenToOrigInPropagate_ = enableEigenToOrigInPropagate;
}
void Propagator::setIThrow(int iThrow) {
  _iThrow_ = iThrow;
}
void Propagator::setLoadAsimovData(bool loadAsimovData) {
  _loadAsimovData_ = loadAsimovData;
}
void Propagator::setParameterInjectorConfig(const nlohmann::json &parameterInjector) {
  _parameterInjectorMc_ = parameterInjector;
}
void Propagator::setGlobalCovarianceMatrix(const std::shared_ptr<TMatrixD> &globalCovarianceMatrix) {
  _globalCovarianceMatrix_ = globalCovarianceMatrix;
}

bool Propagator::isThrowAsimovToyParameters() const {
  return _throwAsimovToyParameters_;
}
int Propagator::getIThrow() const {
  return _iThrow_;
}
double Propagator::getLlhBuffer() const {
  return _llhBuffer_;
}
double Propagator::getLlhStatBuffer() const {
  return _llhStatBuffer_;
}
double Propagator::getLlhPenaltyBuffer() const {
  return _llhPenaltyBuffer_;
}
double Propagator::getLlhRegBuffer() const {
  return _llhRegBuffer_;
}
std::vector<double> Propagator::getLlhPenaltyPerParSet() const{
  return _llhPenaltyPerParSet_;
}
const std::shared_ptr<TMatrixD> &Propagator::getGlobalCovarianceMatrix() const {
  return _globalCovarianceMatrix_;
}
const EventTreeWriter &Propagator::getTreeWriter() const {
  return _treeWriter_;
}
const std::vector<DatasetLoader> &Propagator::getDataSetList() const {
  return _dataSetList_;
}
const std::vector<FitParameterSet> &Propagator::getParameterSetsList() const {
  return _parameterSetList_;
}

FitSampleSet &Propagator::getFitSampleSet() {
  return _fitSampleSet_;
}
PlotGenerator &Propagator::getPlotGenerator() {
  return _plotGenerator_;
}
std::vector<FitParameterSet> &Propagator::getParameterSetsList() {
  return _parameterSetList_;
}
std::shared_ptr<TMatrixD> &Propagator::getGlobalCovarianceMatrix(){
  return _globalCovarianceMatrix_;
}
std::vector<DatasetLoader> &Propagator::getDataSetList() {
  return _dataSetList_;
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
  int iParset{-1};
  for( auto& parSet : _parameterSetList_ ){
    iParset++;
    buffer = parSet.getPenaltyChi2();
    LogThrowIf(std::isnan(buffer), parSet.getName() << " penalty chi2 is Nan");
    _llhPenaltyBuffer_ += buffer;
    _llhPenaltyPerParSet_[iParset] = buffer;
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
#if USE_NEW_DIALS
  std::for_each(_dialCollections_.begin(), _dialCollections_.end(),[&](DialCollection& dc_){
    dc_.updateInputBuffers();
  });
#endif
}
void Propagator::reweightMcEvents() {
  bool usedGPU{false};
#ifdef GUNDAM_USING_CACHE_MANAGER
#ifdef DUMP_PARAMETERS
  do {
    static bool printed = false;
    if (printed) break;
    printed = true;
    // This produces a crazy amount of output.
    int iPar = 0;
    for (auto& parSet : _parameterSetsList_) {
      for ( auto& par : parSet.getParameterList()) {
        LogInfo << "DUMP: " << iPar++
                << " " << par.isEnabled()
                << " " << par.getParameterValue()
                << " (" << par.getFullTitle() << ")";
        if (Cache::Manager::ParameterIndex(&par) < 0) {
          LogInfo << " not used";
        }
        LogInfo << std::endl;
      }
    }
  } while (false);
#endif
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  if(GlobalVariables::getEnableCacheManager()) usedGPU = Cache::Manager::Fill();
#endif
  if( not usedGPU ){
    GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
    if( not _devSingleThreadReweight_ ){
      GlobalVariables::getParallelWorker().runJob("Propagator::reweightMcEvents");
    }
    else{
      this->reweightMcEvents(-1);
    }
  }
  weightProp.counts++;
  weightProp.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
}
void Propagator::refillSampleHistograms(){
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  if( not _devSingleThreadHistFill_ ){
    GlobalVariables::getParallelWorker().runJob("Propagator::refillSampleHistograms");
  }
  else{
    refillSampleHistogramsFct(-1);
    refillSampleHistogramsPostParallelFct();
  }
  fillProp.counts++; fillProp.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
}
void Propagator::throwParametersFromGlobalCovariance(){

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

  auto throws = GenericToolbox::throwCorrelatedParameters(_choleskyMatrix_.get());
  for( int iPar = 0 ; iPar < _choleskyMatrix_->GetNrows() ; iPar++ ){
    _strippedParameterList_[iPar]->setParameterValue(
        _strippedParameterList_[iPar]->getPriorValue()
        + throws[iPar]
    );
  }

  // Making sure eigen decomposed parameters get the conversion done
  for( auto& parSet : _parameterSetList_ ){
    if( parSet.isUseEigenDecompInFit() ){ parSet.propagateOriginalToEigen(); }
  }

}
void Propagator::injectParameterOnMcSamples(const nlohmann::json &injectConfig_) {

  nlohmann::json injParsetList;
  injParsetList = GenericToolbox::Json::fetchValue(injectConfig_, "parameterSetList", injParsetList);

  for( auto& entryParset : injParsetList ){
    auto parsetName = GenericToolbox::Json::fetchValue<std::string>(entryParset, "name");
    LogInfo << "Reading injection parameters for parSet: " << parsetName << std::endl;

    auto* selectedParset = this->getFitParameterSetPtr( parsetName );
    LogThrowIf( selectedParset == nullptr, "Could not find parset: " << parsetName );

    auto parValues = GenericToolbox::Json::fetchValue<nlohmann::json>(entryParset, "parameterValues");
    if     ( parValues.empty() ) {
      LogThrow( "" );
    }
    else if( parValues.is_string() ){
      LogInfo << "Reading parameter values from file: " << parValues.get<std::string>() << std::endl;
      auto parList = GenericToolbox::dumpFileAsVectorString( parValues.get<std::string>(), true );
      LogThrowIf( parList.size() != selectedParset->getNbParameters()  ,
                  parList.size() << " parameters provided for " << parsetName << ", expecting " << selectedParset->getNbParameters()
      );

      for( size_t iPar = 0 ; iPar < selectedParset->getNbParameters() ; iPar++ ) {

        if( not selectedParset->getParameterList()[iPar].isEnabled() ){
          LogAlert << "NOT injecting \"" << selectedParset->getParameterList()[iPar].getFullTitle() << "\" as it is disabled." << std::endl;
          continue;
        }

        LogScopeIndent;
        LogInfo << "Injecting \"" << selectedParset->getParameterList()[iPar].getFullTitle() << "\": " << parList[iPar] << std::endl;
        selectedParset->getParameterList()[iPar].setParameterValue( std::stod(parList[iPar]) );
      }
    }
    else{
      LogScopeIndent;
      for( auto& parValueEntry : parValues ){
        if     ( GenericToolbox::Json::doKeyExist(parValueEntry, "name") ) {
          auto parName = GenericToolbox::Json::fetchValue<std::string>(parValueEntry, "name");
          auto* parPtr = selectedParset->getParameterPtr(parName);
          LogThrowIf(parPtr == nullptr, "Could not find " << parName << " among the defined parameters in " << selectedParset->getName());


          if( not parPtr->isEnabled() ){
            LogAlert << "NOT injecting \"" << parPtr->getFullTitle() << "\" as it is disabled." << std::endl;
            continue;
          }

          LogInfo << "Injecting \"" << parPtr->getFullTitle() << "\": " << GenericToolbox::Json::fetchValue<double>(parValueEntry, "value") << std::endl;
          parPtr->setParameterValue( GenericToolbox::Json::fetchValue<double>(parValueEntry, "value") );
        }
        else if( GenericToolbox::Json::doKeyExist(parValueEntry, "title") ){
          auto parTitle = GenericToolbox::Json::fetchValue<std::string>(parValueEntry, "title");
          auto* parPtr = selectedParset->getParameterPtrWithTitle(parTitle);
          LogThrowIf(parPtr == nullptr, "Could not find " << parTitle << " among the defined parameters in " << selectedParset->getName());


          if( not parPtr->isEnabled() ){
            LogAlert << "NOT injecting \"" << parPtr->getFullTitle() << "\" as it is disabled." << std::endl;
            continue;
          }

          LogInfo << "Injecting \"" << parPtr->getFullTitle() << "\": " << GenericToolbox::Json::fetchValue<double>(parValueEntry, "value") << std::endl;
          parPtr->setParameterValue( GenericToolbox::Json::fetchValue<double>(parValueEntry, "value") );
        }
        else {
          LogThrow("Unsupported: " << parValueEntry);
        }
      }


    }

    if( selectedParset->isUseEigenDecompInFit() ){
      LogInfo << "Propagating back to the eigen decomposed parameters for parSet: " << selectedParset->getName() << std::endl;
      selectedParset->propagateOriginalToEigen();
    }

  }

}


// Protected
void Propagator::initializeThreads() {
  reweightMcEventsFct = [this](int iThread){
    this->reweightMcEvents(iThread);
  };
  GlobalVariables::getParallelWorker().addJob("Propagator::reweightMcEvents", reweightMcEventsFct);

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
  GlobalVariables::getParallelWorker().addJob("Propagator::refillSampleHistograms", refillSampleHistogramsFct);
  GlobalVariables::getParallelWorker().setPostParallelJob("Propagator::refillSampleHistograms", refillSampleHistogramsPostParallelFct);
}

void Propagator::reweightMcEvents(int iThread_) {

  //! Warning: everything you modify here, may significantly slow down the fitter

#if USE_NEW_DIALS
//  _eventDialCache_.propagate(iThread_, GlobalVariables::getNbThreads());
  auto start = _eventDialCache_.getCache().begin();
  auto end = _eventDialCache_.getCache().end();

  if( iThread_ != -1 and GlobalVariables::getNbThreads() != 1 ){
    start = _eventDialCache_.getCache().begin() + Long64_t(iThread_)*(Long64_t(_eventDialCache_.getCache().size())/GlobalVariables::getNbThreads());
    if( iThread_+1 != GlobalVariables::getNbThreads() ){
      end = _eventDialCache_.getCache().begin() + (Long64_t(iThread_) + 1) * (Long64_t(_eventDialCache_.getCache().size())/GlobalVariables::getNbThreads());
    }
  }

  std::for_each(start, end, &EventDialCache::reweightEntry);
#else
  int nThreads = GlobalVariables::getNbThreads();
  if(iThread_ == -1){
    // force single thread
    nThreads = 1;
    iThread_ = 0;
  }
  long nToProcess;
  long offset;
  std::vector<PhysicsEvent>* eList;
  std::for_each(
    _fitSampleSet_.getFitSampleList().begin(), _fitSampleSet_.getFitSampleList().end(),
    [&](auto& s){
      if( s.getMcContainer().eventList.empty() ) return;
      nToProcess = long(s.getMcContainer().eventList.size())/nThreads;
      offset = iThread_*nToProcess;
      if( iThread_+1==nThreads ) nToProcess += long(s.getMcContainer().eventList.size())%nThreads;
      std::for_each(
          s.getMcContainer().eventList.begin()+offset,
          s.getMcContainer().eventList.begin()+offset+nToProcess,
          [&](auto& e){ e.reweightUsingDialCache(); }
          );
    }
  );
#endif



}
