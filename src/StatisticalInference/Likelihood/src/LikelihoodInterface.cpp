//
// Created by Clark McGrew 24/1/23
//

#include "LikelihoodInterface.h"
#include "GundamGlobals.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Json.h"
#include "Logger.h"
#include "CacheManager.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[LikelihoodInterface]");
});


void LikelihoodInterface::readConfigImpl(){
  LogWarning << "Configuring LikelihoodInterface..." << std::endl;

  // Propagator config is supposed to be set in the likelihood interface.
  // however in old GUNDAM versions it was set at the FitterEngine level.
  // In that case, the config has already been set, hence "_propagator_.getConfig()"
  _propagator_.setConfig( GenericToolbox::Json::fetchValue( _config_, "propagatorConfig", _propagator_.getConfig() ) );
  _propagator_.readConfig();

  // placeholders
  JsonType configJointProbability;
  std::string jointProbabilityTypeStr;

  GenericToolbox::Json::deprecatedAction(_propagator_.getSampleSet().getConfig(), "llhStatFunction", [&]{
    LogAlert << R"("llhStatFunction" should now be set under "likelihoodInterfaceConfig/jointProbabilityConfig/type".)" << std::endl;
    jointProbabilityTypeStr = GenericToolbox::Json::fetchValue( _propagator_.getSampleSet().getConfig(), "llhStatFunction", jointProbabilityTypeStr );
  });
  GenericToolbox::Json::deprecatedAction(_propagator_.getSampleSet().getConfig(), "llhConfig", [&]{
    LogAlert << R"("llhConfig" should now be set under "likelihoodInterfaceConfig/jointProbabilityConfig".)" << std::endl;
    configJointProbability = GenericToolbox::Json::fetchValue( _propagator_.getSampleSet().getConfig(), "llhConfig", configJointProbability );
  });

  // new config structure
  configJointProbability = GenericToolbox::Json::fetchValue(_config_, "jointProbabilityConfig", configJointProbability);
  jointProbabilityTypeStr = GenericToolbox::Json::fetchValue(configJointProbability, "type", jointProbabilityTypeStr);

  LogInfo << "Using \"" << jointProbabilityTypeStr << "\" JointProbabilityType." << std::endl;
  _jointProbabilityPtr_ = std::shared_ptr<JointProbability::JointProbabilityBase>( JointProbability::makeJointProbability( jointProbabilityTypeStr ) );
  _jointProbabilityPtr_->readConfig( configJointProbability );

  // Now taking care of the DataSetManager
  JsonType dataSetManagerConfig{};

  // handling deprecated configs
  GenericToolbox::Json::deprecatedAction(_propagator_.getSampleSet().getConfig(), "dataSetList", [&]{
    LogAlert << R"("dataSetList" should now be set under "likelihoodInterfaceConfig" instead of "fitSampleSet".)" << std::endl;
    dataSetManagerConfig = _propagator_.getSampleSet().getConfig(); // DataSetManager will look for "dataSetList"
  });
  GenericToolbox::Json::deprecatedAction(_propagator_.getConfig(), "dataSetList", [&]{
    LogAlert << R"("dataSetList" should now be set under "likelihoodInterfaceConfig" instead of "propagatorConfig".)" << std::endl;
    dataSetManagerConfig = _propagator_.getConfig();
  });
  GenericToolbox::Json::deprecatedAction(_propagator_.getConfig(), "eventTreeWriter", [&]{
    LogAlert << R"("eventTreeWriter" should now be set under "dataSetManagerConfig" instead of "propagatorConfig".)" << std::endl;
    _dataSetManager_.getTreeWriter().setConfig( GenericToolbox::Json::fetchValue<JsonType>(_config_, "eventTreeWriter") );
  });

  dataSetManagerConfig = GenericToolbox::Json::fetchValue(_config_, "dataSetManagerConfig", dataSetManagerConfig);
  _dataSetManager_.readConfig( dataSetManagerConfig );

  LogWarning << "LikelihoodInterface configured." << std::endl;
}
void LikelihoodInterface::initializeImpl() {
  LogWarning << "Initializing LikelihoodInterface..." << std::endl;

  _dataSetManager_.initialize();
  _propagator_.initialize();
  _jointProbabilityPtr_->initialize();

  LogInfo << "Fetching the effective number of fit parameters..." << std::endl;
  _nbParameters_ = 0;
  for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
    _nbParameters_ += int( parSet.getNbParameters() );
  }

  LogInfo << "Fetching the number of bins parameters..." << std::endl;
  _nbSampleBins_ = 0;
  for( auto& sample : _propagator_.getSampleSet().getSampleList() ){
    _nbSampleBins_ += int(sample.getBinning().getBinList().size() );
  }

  this->loadData();

  LogInfo << "LikelihoodInterface initialized." << std::endl;
}
void LikelihoodInterface::loadData(){
  LogInfo << "Loading data into the PropagatorEngine..." << std::endl;

  // First start with the data:
  bool usedMcContainer{false};
  bool allAsimov{true};
  for( auto& dataSet : _dataSetManager_.getDataSetList() ){
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
  }

  // Copy to data container
  if( usedMcContainer ){
    if( _propagator_.isThrowAsimovToyParameters() ){
      LogWarning << "Will throw toy parameters..." << std::endl;

      if( _propagator_.isShowEventBreakdown() ){
        LogInfo << "Propagating prior parameters on the initially loaded events..." << std::endl;
        bool cacheManagerState = GundamGlobals::getEnableCacheManager();
        GundamGlobals::setEnableCacheManager(false);
        _propagator_.resetReweight();
        _propagator_.reweightMcEvents();
        GundamGlobals::setEnableCacheManager(cacheManagerState);

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

    bool cacheManagerState = GundamGlobals::getEnableCacheManager();
    GundamGlobals::setEnableCacheManager(false);
    _propagator_.resetReweight();
    _propagator_.reweightMcEvents();
    GundamGlobals::setEnableCacheManager(cacheManagerState);

    // Copies MC events in data container for both Asimov and FakeData event types
    LogWarning << "Copying loaded mc-like event to data container..." << std::endl;
    _propagator_.getSampleSet().copyMcEventListToDataContainer();

    for( auto& sample : _propagator_.getSampleSet().getSampleList() ){
      sample.getDataContainer().histScale = sample.getMcContainer().histScale;
    }

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

    for( auto& dataSet : _dataSetManager_.getDataSetList() ){
      LogContinueIf(not dataSet.isEnabled(), "Dataset \"" << dataSet.getName() << "\" is disabled. Skipping");
      auto& dispenser = dataSet.getMcDispenser();
      dispenser.load( _propagator_ );
    }

    LogInfo << "Resizing dial containers..." << std::endl;
    for( auto& dialCollection : _propagator_.getDialCollectionList() ) {
      if( not dialCollection.isBinned() ){ dialCollection.resizeContainers(); }
    }

    LogInfo << "Build reference cache..." << std::endl;
    _propagator_.getEventDialCache().shrinkIndexedCache();
    _propagator_.getEventDialCache().buildReferenceCache(_propagator_.getSampleSet(), _propagator_.getDialCollectionList());
  }

#ifdef GUNDAM_USING_CACHE_MANAGER
  // After all the data has been loaded.  Specifically, this must be after
  // the MC has been copied for the Asimov fit, or the "data" use the MC
  // reweighting cache.  This must also be before the first use of
  // reweightMcEvents.
  if(GundamGlobals::getEnableCacheManager()) {
    Cache::Manager::Build(_propagator_.getSampleSet(), _propagator_.getEventDialCache());
  }
#endif

  LogInfo << "Propagating prior parameters on events..." << std::endl;
  _propagator_.resetReweight();
  _propagator_.reweightMcEvents();

  LogInfo << "Set the current MC prior weights as nominal weight..." << std::endl;
  for( auto& sample : _propagator_.getSampleSet().getSampleList() ){
    for( auto& event : sample.getMcContainer().eventList ){
      event.setNominalWeight(event.getEventWeight());
    }
  }

  LogInfo << "Filling up sample bin caches..." << std::endl;
  _propagator_.getSampleSet().updateSampleBinEventList();

  LogInfo << "Filling up sample histograms..." << std::endl;
  _propagator_.getSampleSet().updateSampleHistograms();

  // Throwing stat error on data -> BINNING SHOULD BE SET!!
  if( _throwAsimovToyParameters_ and _enableStatThrowInToys_ ){
    LogInfo << "Throwing statistical error for data container..." << std::endl;

    if( _enableEventMcThrow_ ){
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
    if( _gaussStatThrowInToys_ ) {
      LogWarning << "Using gaussian statistical throws. (caveat: distribution truncated when the bins are close to zero)" << std::endl;
    }
    for( auto& sample : _propagator_.getSampleSet().getSampleList() ){
      // Asimov bin content -> toy data
      sample.getDataContainer().throwStatError(_gaussStatThrowInToys_);
    }
  }

  LogInfo << "Locking data event containers..." << std::endl;
  for( auto& sample : _propagator_.getSampleSet().getSampleList() ){
    // Now the data won't be refilled each time
    sample.getDataContainer().isLocked = true;
  }

  if( not _propagator_.getParameterInjectorMc().empty() ){
    LogWarning << "Injecting parameters on MC samples..." << std::endl;
    _propagator_.getParametersManager().injectParameterValues( ConfigUtils::getForwardedConfig(_propagator_.getParameterInjectorMc()) );
    _propagator_.resetReweight();
    _propagator_.reweightMcEvents();
  }

  //////////////////////////////////////////
  // DON'T MOVE PARAMETERS FROM THIS POINT
  //////////////////////////////////////////

  /// Copy the current state of MC as "nominal" histogram
  LogInfo << "Copy the current state of MC as \"nominal\" histogram..." << std::endl;
  for( auto& sample : _propagator_.getSampleSet().getSampleList() ){
    sample.getMcContainer().saveAsHistogramNominal();
  }

  /// Initialise other tools
  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing the plot generator") << std::endl;
  _propagator_.getPlotGenerator().setSampleSetPtr(&_propagator_.getSampleSet());
  _propagator_.getPlotGenerator().initialize(); // TODO: init before and only build the cache here


  _dataSetManager_.getTreeWriter().setSampleSetPtr( &_propagator_.getSampleSet() );
  _dataSetManager_.getTreeWriter().setParSetListPtr( &_propagator_.getParametersManager().getParameterSetsList() );
  _dataSetManager_.getTreeWriter().setEventDialCachePtr( &_propagator_.getEventDialCache() );

  /// Printouts for quick monitoring
  if( _propagator_.isShowEventBreakdown() ){

    if(true){
      // STAGED MASK
      LogWarning << "Staged event breakdown:" << std::endl;
      std::vector<std::vector<double>> stageBreakdownList(
          _propagator_.getSampleSet().getSampleList().size(),
          std::vector<double>(_propagator_.getParametersManager().getParameterSetsList().size() + 1, 0)
      ); // [iSample][iStage]
      std::vector<std::string> stageTitles;
      stageTitles.emplace_back("Sample");
      stageTitles.emplace_back("No reweight");
      for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
        if( not parSet.isEnabled() ){ continue; }
        stageTitles.emplace_back("+ " + parSet.getName());
      }

      int iStage{0};
      std::vector<ParameterSet*> maskedParSetList;
      for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
        if( not parSet.isEnabled() ){ continue; }
        maskedParSetList.emplace_back( &parSet );
        parSet.setMaskedForPropagation( true );
      }

      _propagator_.resetReweight();
      _propagator_.reweightMcEvents();
      for( size_t iSample = 0 ; iSample < _propagator_.getSampleSet().getSampleList().size() ; iSample++ ){
        stageBreakdownList[iSample][iStage] = _propagator_.getSampleSet().getSampleList()[iSample].getMcContainer().getSumWeights();
      }

      for( auto* parSetPtr : maskedParSetList ){
        parSetPtr->setMaskedForPropagation(false);
        _propagator_.resetReweight();
        _propagator_.reweightMcEvents();
        iStage++;
        for( size_t iSample = 0 ; iSample < _propagator_.getSampleSet().getSampleList().size() ; iSample++ ){
          stageBreakdownList[iSample][iStage] = _propagator_.getSampleSet().getSampleList()[iSample].getMcContainer().getSumWeights();
        }
      }

      GenericToolbox::TablePrinter t;
      t.setColTitles(stageTitles);
      for( size_t iSample = 0 ; iSample < _propagator_.getSampleSet().getSampleList().size() ; iSample++ ) {
        std::vector<std::string> tableLine;
        tableLine.emplace_back("\"" + _propagator_.getSampleSet().getSampleList()[iSample].getName() + "\"");
        for( iStage = 0 ; iStage < stageBreakdownList[iSample].size() ; iStage++ ){
          tableLine.emplace_back( std::to_string(stageBreakdownList[iSample][iStage]) );
        }
        t.addTableLine(tableLine);
      }
      t.printTable();
    }

    LogWarning << "Sample breakdown:" << std::endl;
    std::cout << _propagator_.getSampleBreakdownTableStr() << std::endl;

  }
  if( _propagator_.isDebugPrintLoadedEvents() ){
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

  /// Propagator needs to be responsive, let the workers wait for the signal
  GundamGlobals::getParallelWorker().setCpuTimeSaverIsEnabled(false);

}

void LikelihoodInterface::propagateAndEvalLikelihood(){
  _propagator_.propagateParameters();
  this->evalLikelihood();
}

double LikelihoodInterface::evalLikelihood() const {
  this->evalStatLikelihood();
  this->evalPenaltyLikelihood();

  _buffer_.updateTotal();
  return _buffer_.totalLikelihood;
}
double LikelihoodInterface::evalStatLikelihood() const {
  _buffer_.statLikelihood = 0.;
  for( auto &sample: _propagator_.getSampleSet().getSampleList()){
    _buffer_.statLikelihood += this->evalStatLikelihood( sample );
  }
  return _buffer_.statLikelihood;
}
double LikelihoodInterface::evalPenaltyLikelihood() const {
  _buffer_.penaltyLikelihood = 0;
  for( auto& parSet : _propagator_.getParametersManager().getParameterSetsList() ){
    _buffer_.penaltyLikelihood += this->evalPenaltyLikelihood( parSet );
  }
  return _buffer_.penaltyLikelihood;
}
double LikelihoodInterface::evalStatLikelihood(const Sample& sample_) const {
  return _jointProbabilityPtr_->eval( sample_ );
}
double LikelihoodInterface::evalPenaltyLikelihood(const ParameterSet& parSet_) const {
  if( not parSet_.isEnabled() ){ return 0; }

  double buffer = 0;

  if( parSet_.getPriorCovarianceMatrix() != nullptr ){
    if( parSet_.isEnableEigenDecomp() ){
      for( const auto& eigenPar : parSet_.getEigenParameterList() ){
        if( eigenPar.isFixed() ){ continue; }
        buffer += TMath::Sq( (eigenPar.getParameterValue() - eigenPar.getPriorValue()) / eigenPar.getStdDevValue() ) ;
      }
    }
    else{
      // make delta vector
      parSet_.updateDeltaVector();

      // compute penalty term with covariance
      buffer =
          (*parSet_.getDeltaVectorPtr())
          * ( (*parSet_.getInverseStrippedCovarianceMatrix()) * (*parSet_.getDeltaVectorPtr()) );
    }
  }

  return buffer;
}
[[nodiscard]] std::string LikelihoodInterface::getSummary() const {
  std::stringstream ss;

  this->evalLikelihood(); // make sure the buffer is up-to-date

  ss << "Total likelihood = " << _buffer_.totalLikelihood;
  ss << std::endl << "Stat likelihood = " << _buffer_.statLikelihood;
  ss << " = sum of: " << GenericToolbox::toString(
      _propagator_.getSampleSet().getSampleList(), [&]( const Sample& sample_){
        std::stringstream ssSub;
        ssSub << sample_.getName() << ": ";
        if( sample_.isEnabled() ){ ssSub << this->evalStatLikelihood( sample_ ); }
        else                     { ssSub << "disabled."; }
        return ssSub.str();
      }
  );
  ss << std::endl << "Penalty likelihood = " << _buffer_.penaltyLikelihood;
  ss << " = sum of: " << GenericToolbox::toString(
      _propagator_.getParametersManager().getParameterSetsList(), [&](const ParameterSet& parSet_){
        std::stringstream ssSub;
        ssSub << parSet_.getName() << ": ";
        if( parSet_.isEnabled() ){ ssSub << this->evalPenaltyLikelihood( parSet_ ); }
        else                     { ssSub << "disabled."; }
        return ssSub.str();
      }
  );
  return ss.str();
}

// An MIT Style License

// Copyright (c) 2022 GUNDAM DEVELOPERS

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
