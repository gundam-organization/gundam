//
// Created by Clark McGrew 24/1/23
//

#include "LikelihoodInterface.h"
#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif
#include "GundamGlobals.h"

#include "GenericToolbox.Map.h"
#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Json.h"
#include "Logger.h"

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[LikelihoodInterface]"); });
#endif

void LikelihoodInterface::readConfigImpl(){
  LogWarning << "Configuring LikelihoodInterface..." << std::endl;

  _threadPool_.setNThreads( GundamGlobals::getNumberOfThreads() );

  _modelPropagator_.readConfig();
  _dataPropagator_.readConfig( _modelPropagator_.getConfig() );

  // First taking care of the DataSetManager
  JsonType dataSetManagerConfig{};
  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getConfig(), {{"fitSampleSetConfig"}, {"dataSetList"}}, [&]{
    LogAlert << R"("dataSetList" should now be set under "likelihoodInterfaceConfig" instead of "fitSampleSet".)" << std::endl;
    dataSetManagerConfig = GenericToolbox::Json::fetchValue<JsonType>(_modelPropagator_.getConfig(), "fitSampleSetConfig"); // DataSetManager will look for "dataSetList"
  });
  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getConfig(), "dataSetList", [&]{
    LogAlert << R"("dataSetList" should now be set under "likelihoodInterfaceConfig" instead of "propagatorConfig".)" << std::endl;
    dataSetManagerConfig = _modelPropagator_.getConfig();
  });
  dataSetManagerConfig = GenericToolbox::Json::fetchValue(_config_, {{"datasetManagerConfig"}, {"dataSetManagerConfig"}}, dataSetManagerConfig);


  // dataSetList should be present
  JsonType dataSetList;
  dataSetList = GenericToolbox::Json::fetchValue(_config_, "dataSetList", dataSetManagerConfig);

  // creating the dataSets:
  _dataSetList_.reserve( dataSetList.size() );
  for( const auto& dataSetConfig : dataSetList ){
    _dataSetList_.emplace_back(dataSetConfig, int(_dataSetList_.size()));
  }


  JsonType configJointProbability;
  std::string jointProbabilityTypeStr{"PoissonLLH"};

  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getSampleSet().getConfig(), "llhStatFunction", [&]{
    LogAlert << R"("llhStatFunction" should now be set under "likelihoodInterfaceConfig/jointProbabilityConfig/type".)" << std::endl;
    jointProbabilityTypeStr = GenericToolbox::Json::fetchValue( _modelPropagator_.getSampleSet().getConfig(), "llhStatFunction", jointProbabilityTypeStr );
  });
  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getSampleSet().getConfig(), "llhConfig", [&]{
    LogAlert << R"("llhConfig" should now be set under "likelihoodInterfaceConfig/jointProbabilityConfig".)" << std::endl;
    configJointProbability = GenericToolbox::Json::fetchValue( _modelPropagator_.getSampleSet().getConfig(), "llhConfig", configJointProbability );
  });

  // new config structure
  configJointProbability = GenericToolbox::Json::fetchValue(_config_, "jointProbabilityConfig", configJointProbability);
  jointProbabilityTypeStr = GenericToolbox::Json::fetchValue(configJointProbability, "type", jointProbabilityTypeStr);

  LogInfo << "Using \"" << jointProbabilityTypeStr << "\" JointProbabilityType." << std::endl;
  _jointProbabilityPtr_ = std::shared_ptr<JointProbability::JointProbabilityBase>( JointProbability::makeJointProbability( jointProbabilityTypeStr ) );
  _jointProbabilityPtr_->readConfig( configJointProbability );


  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getConfig(), "plotGeneratorConfig", [&]{
    LogAlert << R"("plotGeneratorConfig" should now be set under "likelihoodInterfaceConfig".)" << std::endl;
    _plotGenerator_.setConfig( GenericToolbox::Json::fetchValue( _modelPropagator_.getSampleSet().getConfig(), "plotGeneratorConfig", _plotGenerator_.getConfig() ) );
  });
  _plotGenerator_.setConfig( GenericToolbox::Json::fetchValue( _config_, "plotGeneratorConfig", _plotGenerator_.getConfig() ) );
  _plotGenerator_.readConfig();

  LogWarning << "LikelihoodInterface configured." << std::endl;
}
void LikelihoodInterface::initializeImpl() {
  LogWarning << "Initializing LikelihoodInterface..." << std::endl;

  for( auto& dataSet : _dataSetList_ ){ dataSet.initialize(); }

  _modelPropagator_.initialize();
  _dataPropagator_.initialize(); // TODO: should be copied to avoid duplicated printouts

  _plotGenerator_.setModelSampleSetPtr( &_modelPropagator_.getSampleSet().getSampleList() );
  _plotGenerator_.setDataSampleSetPtr( &_modelPropagator_.getSampleSet().getSampleList() );

  _eventTreeWriter_.initialize();
  _plotGenerator_.initialize();

  // loading the propagators
  this->load();

  _jointProbabilityPtr_->initialize();

  // Loading monitoring values:

  LogInfo << "Fetching the effective number of fit parameters..." << std::endl;
  _nbParameters_ = 0;
  for( auto& parSet : _modelPropagator_.getParametersManager().getParameterSetsList() ){
    _nbParameters_ += int( parSet.getNbParameters() );
  }

  LogInfo << "Fetching the number of bins parameters..." << std::endl;
  _nbSampleBins_ = 0;
  for( auto& sample : _modelPropagator_.getSampleSet().getSampleList() ){
    _nbSampleBins_ += int(sample.getBinning().getBinList().size() );
  }

  LogInfo << "Move back MC parameters to prior..." << std::endl;
  _modelPropagator_.getParametersManager().moveParametersToPrior();

  /// some joint fit probability might need to save the value of the nominal histogram.
  /// here we know every parameter is at its nominal value
  LogInfo << "First evaluation of the LLH at the nominal value..." << std::endl;
  this->propagateAndEvalLikelihood();
  LogInfo << this->getSummary() << std::endl;

  /// move the parameter away from the prior if needed
  if( not _modelPropagator_.getParameterInjectorMc().empty() ){
    LogWarning << "Injecting parameters on MC samples..." << std::endl;
    _modelPropagator_.getParametersManager().injectParameterValues(
        ConfigUtils::getForwardedConfig(_modelPropagator_.getParameterInjectorMc())
    );
    _modelPropagator_.reweightEvents();
  }

  //////////////////////////////////////////
  // DON'T MOVE PARAMETERS FROM THIS POINT
  //////////////////////////////////////////

  LogInfo << "LikelihoodInterface initialized." << std::endl;
}

void LikelihoodInterface::propagateAndEvalLikelihood(){
  _modelPropagator_.propagateParameters();
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
  for( auto &samplePair: _samplePairList_ ){
    _buffer_.statLikelihood += this->evalStatLikelihood( samplePair );
  }
  return _buffer_.statLikelihood;
}
double LikelihoodInterface::evalPenaltyLikelihood() const {
  _buffer_.penaltyLikelihood = 0;
  for( auto& parSet : _modelPropagator_.getParametersManager().getParameterSetsList() ){
    _buffer_.penaltyLikelihood += this->evalPenaltyLikelihood( parSet );
  }
  return _buffer_.penaltyLikelihood;
}
double LikelihoodInterface::evalStatLikelihood(const SamplePair& samplePair_) const {
  return _jointProbabilityPtr_->eval( samplePair_ );
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
std::string LikelihoodInterface::getSummary() const {
  std::stringstream ss;

  this->evalLikelihood(); // make sure the buffer is up-to-date

  ss << "Total likelihood = " << _buffer_.totalLikelihood;
  ss << std::endl << "Stat likelihood = " << _buffer_.statLikelihood;
  ss << " = sum of: " << GenericToolbox::toString(
      _samplePairList_, [&]( const SamplePair& samplePair_){
        std::stringstream ssSub;
        ssSub << samplePair_.model->getName() << ": ";
        if( samplePair_.model->isEnabled() ){ ssSub << this->evalStatLikelihood( samplePair_ ); }
        else                     { ssSub << "disabled."; }
        return ssSub.str();
      }
  );
  ss << std::endl << "Penalty likelihood = " << _buffer_.penaltyLikelihood;
  ss << " = sum of: " << GenericToolbox::toString(
      _modelPropagator_.getParametersManager().getParameterSetsList(), [&](const ParameterSet& parSet_){
        std::stringstream ssSub;
        ssSub << parSet_.getName() << ": ";
        if( parSet_.isEnabled() ){ ssSub << this->evalPenaltyLikelihood( parSet_ ); }
        else                     { ssSub << "disabled."; }
        return ssSub.str();
      }
  );
  return ss.str();
}

void LikelihoodInterface::writeEvents(const GenericToolbox::TFilePath& saveDir_) const {
  LogInfo << "Writing sample events data." << std::endl;

  // TODO: use the EventDialCache for writing the dials?

  LogInfo << "Writing model sample events..." << std::endl;
  for( auto& sample : _modelPropagator_.getSampleSet().getSampleList() ){
    if( not sample.isEnabled() ){ continue; }
    _eventTreeWriter_.writeEvents( saveDir_.getSubDir("model").getSubDir(sample.getName()), sample.getEventList() );
  }

  LogInfo << "Writing data sample events..." << std::endl;
  for( auto& sample : _dataPropagator_.getSampleSet().getSampleList() ){
    if( not sample.isEnabled() ){ continue; }
    _eventTreeWriter_.writeEvents( saveDir_.getSubDir("data").getSubDir(sample.getName()), sample.getEventList() );
  }

}
void LikelihoodInterface::writeEventRates(const GenericToolbox::TFilePath& saveDir_) const {
  LogInfo << "Writing event rates." << std::endl;

  LogInfo << "Writing model event rates..." << std::endl;
  _modelPropagator_.writeEventRates( saveDir_.getSubDir("model") );

  LogInfo << "Writing data event rates..." << std::endl;
  _dataPropagator_.writeEventRates( saveDir_.getSubDir("data") );
}

void LikelihoodInterface::load(){

  loadModelPropagator();
  loadDataPropagator();

  buildSamplePairList();

  /// model propagator needs to be fast, let the workers wait for the signal
  _modelPropagator_.getThreadPool().setCpuTimeSaverIsEnabled( false );

}
void LikelihoodInterface::loadModelPropagator(){

  _modelPropagator_.clearContent();

  for( auto& dataSet : _dataSetList_ ){
    dataSet.getModelDispenser().setPlotGeneratorPtr( &_plotGenerator_ );
    dataSet.getModelDispenser().load( _modelPropagator_ );
  }

  _modelPropagator_.shrinkDialContainers();
  _modelPropagator_.buildDialCache();

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
  LogInfo << "Filling up model sample bin caches..." << std::endl;
  _threadPool_.runJob([this](int iThread){
    LogInfoIf(iThread <= 0) << "Updating sample per bin event lists..." << std::endl;
    for( auto& sample : _modelPropagator_.getSampleSet().getSampleList() ){
      sample.updateBinEventList(iThread);
    }
  });

  LogInfo << "Filling up model sample histograms..." << std::endl;
  _threadPool_.runJob([this](int iThread){
    for( auto& sample : _modelPropagator_.getSampleSet().getSampleList() ){
      sample.refillHistogram(iThread);
    }
  });

  /// Now caching the event for the plot generator
  _plotGenerator_.defineHistogramHolders();

}
void LikelihoodInterface::loadDataPropagator(){

  _dataPropagator_.clearContent();

  if( _dataType_ == DataType::Asimov or _forceAsimovData_ ){
    // copy the events directly from the model
    _dataPropagator_.getSampleSet().copyEventsFrom( _modelPropagator_.getSampleSet() );
  }
  else{
    for( auto& dataSet : _dataSetList_ ){
      // let the llh interface choose witch data entry to load
      auto* dataDispenser{this->getDataDispenser( dataSet )};

      // nothing has been selected, skip (warning message is within getDataDispenser)
      if( dataDispenser == nullptr ){ continue; }

      // handling override of the propagator config
      if( not dataDispenser->getParameters().overridePropagatorConfig.empty() ){
        LogWarning << "Reload the data propagator config with override options..." << std::endl;
        ConfigUtils::ConfigHandler configHandler( _modelPropagator_.getConfig() );
        configHandler.override( dataDispenser->getParameters().overridePropagatorConfig );
        _dataPropagator_.readConfig( configHandler.getConfig() );
        _dataPropagator_.initialize();
      }

      dataDispenser->setPlotGeneratorPtr( &_plotGenerator_ );

      // otherwise load the dataset
      dataDispenser->load( _dataPropagator_ );

      // make sure the config is from scratch each time we read a new dataset
      if( not dataDispenser->getParameters().overridePropagatorConfig.empty() ){
        LogWarning << "Restoring propagator config overrides..." << std::endl;
        _dataPropagator_.readConfig( _modelPropagator_.getConfig() );
        _dataPropagator_.initialize();
      }
    }
  }

  _dataPropagator_.shrinkDialContainers();
  _dataPropagator_.buildDialCache();

  if( _dataPropagator_.isThrowAsimovToyParameters() ){

    if( _dataPropagator_.isShowEventBreakdown() ){
      LogInfo << "Propagating prior parameters on the initially loaded events..." << std::endl;
      _dataPropagator_.reweightEvents();

      LogInfo << "Sample breakdown prior to the throwing:" << std::endl;
      std::cout << _dataPropagator_.getSampleBreakdownTableStr() << std::endl;

      if( _dataPropagator_.isDebugPrintLoadedEvents() ){
        LogDebug << "Toy events:" << std::endl;
        LogDebug << GET_VAR_NAME_VALUE(_dataPropagator_.getDebugPrintLoadedEventsNbPerSample()) << std::endl;
        int iEvt{0};
        for( auto& entry : _dataPropagator_.getEventDialCache().getCache() ) {
          LogDebug << "Event #" << iEvt++ << "{" << std::endl;
          {
            LogScopeIndent;
            LogDebug << entry.getSummary() << std::endl;
          }
          LogDebug << "}" << std::endl;
          if( iEvt >= _dataPropagator_.getDebugPrintLoadedEventsNbPerSample() ) break;
        }
      }
    }

    if( _toyParameterInjector_.empty() ){
      LogWarning << "Will throw toy parameters..." << std::endl;
      _dataPropagator_.getParametersManager().throwParameters();
    }
    else{
      LogWarning << "Injecting parameters..." << std::endl;
      _dataPropagator_.getParametersManager().injectParameterValues( _toyParameterInjector_ );
    }

  } // throw asimov?

  LogInfo << "Propagating parameters on events..." << std::endl;

  // At this point, MC events have been reweighted using their prior
  // but when using eigen decomp, the conversion eigen to original has a small computational error
  // this will make sure the "asimov" data will be reweighted the same way the model is expected to behave
  // while using the eigen decomp
  for( auto& parSet: _dataPropagator_.getParametersManager().getParameterSetsList() ) {
    if( not parSet.isEnabled() ){ continue; }
    if( parSet.isEnableEigenDecomp() ) { parSet.propagateEigenToOriginal(); }
  }

  // re-propagate systematics if applicable
  _dataPropagator_.reweightEvents();

  LogInfo << "Filling up data sample bin caches..." << std::endl;
  _threadPool_.runJob([this](int iThread){
    LogInfoIf(iThread <= 0) << "Updating sample per bin event lists..." << std::endl;
    for( auto& sample : _dataPropagator_.getSampleSet().getSampleList() ){
      sample.updateBinEventList(iThread);
    }
  });

  LogInfo << "Filling up data sample histograms..." << std::endl;
  _threadPool_.runJob([this](int iThread){
    for( auto& sample : _dataPropagator_.getSampleSet().getSampleList() ){
      sample.refillHistogram(iThread);
    }
  });

  // Throwing stat error on data -> BINNING SHOULD BE SET!!
  if( _dataType_ == DataType::Toy and _dataPropagator_.isEnableStatThrowInToys() ){
    LogInfo << "Throwing statistical error for data container..." << std::endl;

    if( _dataPropagator_.isEnableEventMcThrow() ){
      // Take into account the finite amount of event in MC
      LogInfo << "enableEventMcThrow is enabled: throwing individual MC events" << std::endl;
      for( auto& sample : _dataPropagator_.getSampleSet().getSampleList() ) {
        sample.throwEventMcError();
      }
    }
    else{
      LogWarning << "enableEventMcThrow is disabled. Not throwing individual MC events" << std::endl;
    }

    LogInfo << "Throwing statistical error on histograms..." << std::endl;
    if( _dataPropagator_.isGaussStatThrowInToys() ) {
      LogWarning << "Using gaussian statistical throws. (caveat: distribution truncated when the bins are close to zero)" << std::endl;
    }
    for( auto& sample : _dataPropagator_.getSampleSet().getSampleList() ){
      // Asimov bin content -> toy data
      sample.throwStatError( _dataPropagator_.isGaussStatThrowInToys() );
    }
  }


}
void LikelihoodInterface::buildSamplePairList(){

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

DataDispenser* LikelihoodInterface::getDataDispenser( DatasetDefinition& dataset_ ){

  if( _dataType_ == DataType::Asimov ){
    LogInfo << "Fetching asimov entry for dataset: " << dataset_.getName() << std::endl;
    return &dataset_.getModelDispenser();
  }
  else if( _dataType_ == DataType::Toy ){

    auto& selectedToyEntry = dataset_.getSelectedToyEntry();
    if( selectedToyEntry.empty() or selectedToyEntry == "Asimov" ){
      LogInfo << "Fetching asimov toy entry for dataset: " << dataset_.getName() << std::endl;
      return &dataset_.getModelDispenser();
    }

    if( not GenericToolbox::isIn( selectedToyEntry, dataset_.getDataDispenserDict()) ){
      LogWarning << "Could not find toy entry \"" << selectedToyEntry << "\" in dataset: " << dataset_.getName() << std::endl;
      return nullptr;
    }

    LogInfo << "Fetching toy entry \"" << selectedToyEntry << "\" for dataset: " << dataset_.getName() << std::endl;
    return &dataset_.getDataDispenserDict().at(selectedToyEntry);
  }
  else if( _dataType_ == DataType::RealData ){

    auto& selectedDataEntry = dataset_.getSelectedDataEntry();
    if( selectedDataEntry.empty() or selectedDataEntry == "Asimov" ){
      LogInfo << "Fetching asimov data entry for dataset: " << dataset_.getName() << std::endl;
      return &dataset_.getModelDispenser();
    }

    if( not GenericToolbox::isIn( selectedDataEntry, dataset_.getDataDispenserDict()) ){
      LogWarning << "Could not find toy entry \"" << selectedDataEntry << "\" in dataset: " << dataset_.getName() << std::endl;
      return nullptr;
    }

    LogInfo << "Fetching toy entry \"" << selectedDataEntry << "\" for dataset: " << dataset_.getName() << std::endl;
    return &dataset_.getDataDispenserDict().at(selectedDataEntry);
  }

  // invalid
  return nullptr;
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
// End:
