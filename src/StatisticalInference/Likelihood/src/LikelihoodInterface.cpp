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

#include "Logger.h"

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[LikelihoodInterface]"); });
#endif

void LikelihoodInterface::configureImpl(){

  _threadPool_.setNThreads(GundamGlobals::getNbCpuThreads() );

  // reading the configuration of the propagator
  // allows to implement
  GenericToolbox::Json::fillValue(_config_, _modelPropagator_.getConfig(), "propagatorConfig");
  _modelPropagator_.configure();

  JsonType datasetListConfig{};
  JsonType jointProbabilityConfig{};
  std::string jointProbabilityTypeStr{"PoissonLLH"};

  // prior to this version a few parameters were set in the propagator itself
  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getConfig(), "throwAsimovFitParameters", [&]{
    LogAlert << R"("throwAsimovFitParameters" should now be set under "likelihoodInterfaceConfig" instead of "propagatorConfig".)" << std::endl;
    GenericToolbox::Json::fillValue(_modelPropagator_.getConfig(), _throwAsimovToyParameters_, "throwAsimovFitParameters");
  });
  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getConfig(), "enableStatThrowInToys", [&]{
    LogAlert << R"("enableStatThrowInToys" should now be set under "likelihoodInterfaceConfig" instead of "propagatorConfig".)" << std::endl;
    GenericToolbox::Json::fillValue(_modelPropagator_.getConfig(), _enableStatThrowInToys_, "enableStatThrowInToys");
  });
  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getConfig(), "gaussStatThrowInToys", [&]{
    LogAlert << R"("gaussStatThrowInToys" should now be set under "likelihoodInterfaceConfig" instead of "propagatorConfig".)" << std::endl;
    GenericToolbox::Json::fillValue(_modelPropagator_.getConfig(), _gaussStatThrowInToys_, "gaussStatThrowInToys");
  });
  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getConfig(), "enableEventMcThrow", [&]{
    LogAlert << R"("enableEventMcThrow" should now be set under "likelihoodInterfaceConfig" instead of "propagatorConfig".)" << std::endl;
    GenericToolbox::Json::fillValue(_modelPropagator_.getConfig(), _enableEventMcThrow_, "enableEventMcThrow");
  });
  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getConfig(), "plotGeneratorConfig", [&]{
    LogAlert << R"("plotGeneratorConfig" should now be set under "likelihoodInterfaceConfig".)" << std::endl;
    GenericToolbox::Json::fillValue(_modelPropagator_.getConfig(), _plotGenerator_.getConfig(), "plotGeneratorConfig");
  });
  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getConfig(), {{"dataSetList"}, {"fitSampleSetConfig/dataSetList"}}, [&](const std::string& path_){
    LogAlert << "\"" << path_ << R"(" should now be set under "likelihoodInterfaceConfig".)" << std::endl;
    GenericToolbox::Json::fillValue(_modelPropagator_.getConfig(), datasetListConfig, path_);
  });
  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getSampleSet().getConfig(), "llhStatFunction", [&]{
    LogAlert << R"("llhStatFunction" should now be set under "likelihoodInterfaceConfig/jointProbabilityConfig/type".)" << std::endl;
    GenericToolbox::Json::fillValue(_modelPropagator_.getSampleSet().getConfig(), jointProbabilityTypeStr, "llhStatFunction");
  });
  GenericToolbox::Json::deprecatedAction(_modelPropagator_.getSampleSet().getConfig(), "llhConfig", [&]{
    LogAlert << R"("llhConfig" should now be set under "likelihoodInterfaceConfig/jointProbabilityConfig".)" << std::endl;
    GenericToolbox::Json::fillValue(_modelPropagator_.getSampleSet().getConfig(), jointProbabilityConfig, "llhConfig");
  });


  // defining datasets:
  GenericToolbox::Json::fillValue(_config_, datasetListConfig, {{"datasetList"}, {"dataSetList"}});
  _datasetList_.reserve(datasetListConfig.size() );
  for( const auto& dataSetConfig : datasetListConfig ){
    _datasetList_.emplace_back(dataSetConfig, int(_datasetList_.size()));
  }

  // new config structure
  GenericToolbox::Json::fillValue(_config_, jointProbabilityConfig, "jointProbabilityConfig");
  GenericToolbox::Json::fillValue(jointProbabilityConfig, jointProbabilityTypeStr, "type");
  LogDebugIf(GundamGlobals::isDebug()) << "Using \"" << jointProbabilityTypeStr << "\" JointProbabilityType." << std::endl;
  _jointProbabilityPtr_ = std::shared_ptr<JointProbability::JointProbabilityBase>( JointProbability::makeJointProbability( jointProbabilityTypeStr ) );
  _jointProbabilityPtr_->setConfig( jointProbabilityConfig );
  _jointProbabilityPtr_->configure();

  // nested configurations
  GenericToolbox::Json::fillValue(_config_, _plotGenerator_.getConfig(), "plotGeneratorConfig");
  _plotGenerator_.configure();

  // reading local parameters
  GenericToolbox::Json::fillValue(_config_, _throwAsimovToyParameters_, "throwAsimovFitParameters");
  GenericToolbox::Json::fillValue(_config_, _enableStatThrowInToys_, "enableStatThrowInToys");
  GenericToolbox::Json::fillValue(_config_, _gaussStatThrowInToys_, "gaussStatThrowInToys");
  GenericToolbox::Json::fillValue(_config_, _enableEventMcThrow_, "enableEventMcThrow");


  // TODO: move it outside
  _modelPropagator_.printConfiguration();
}
void LikelihoodInterface::initializeImpl() {
  LogWarning << "Initializing LikelihoodInterface..." << std::endl;

  for( auto& dataSet : _datasetList_ ){ dataSet.initialize(); }

  _modelPropagator_.initialize();
  _dataPropagator_ = _modelPropagator_; // avoid tons of printouts

  _plotGenerator_.setModelSampleSetPtr( &_modelPropagator_.getSampleSet().getSampleList() );
  _plotGenerator_.setDataSampleSetPtr( &_dataPropagator_.getSampleSet().getSampleList() );
  _plotGenerator_.initialize();

  _eventTreeWriter_.initialize();

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
    _nbSampleBins_ += sample.getHistogram().getNbBins();
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
    _buffer_.penaltyLikelihood += LikelihoodInterface::evalPenaltyLikelihood( parSet );
  }
  return _buffer_.penaltyLikelihood;
}
double LikelihoodInterface::evalStatLikelihood(const SamplePair& samplePair_) const {
  return _jointProbabilityPtr_->eval( samplePair_ );
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
        if( parSet_.isEnabled() ){ ssSub << LikelihoodInterface::evalPenaltyLikelihood( parSet_ ); }
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

// print
void LikelihoodInterface::printBreakdowns() const{

  LogInfo << "Samples breakdown:" << std::endl;
  std::cout << getSampleBreakdownTable() << std::endl;

}
std::string LikelihoodInterface::getSampleBreakdownTable() const{
  GenericToolbox::TablePrinter t;

  t << "Sample" << GenericToolbox::TablePrinter::NextColumn;
  t << "Model (# binned)" << GenericToolbox::TablePrinter::NextColumn;
  t << "Data (# binned)" << GenericToolbox::TablePrinter::NextColumn;
  t << "Diff." << GenericToolbox::TablePrinter::NextColumn;
  t << "Model (weighted)" << GenericToolbox::TablePrinter::NextColumn;
  t << "Data (weighted)" << GenericToolbox::TablePrinter::NextColumn;
  t << "Diff." << GenericToolbox::TablePrinter::NextLine;

  for( auto& samplePair : _samplePairList_ ){
    t << samplePair.model->getName() << GenericToolbox::TablePrinter::NextColumn;
    t << samplePair.model->getNbBinnedEvents() << GenericToolbox::TablePrinter::NextColumn;
    t << samplePair.data->getNbBinnedEvents() << GenericToolbox::TablePrinter::NextColumn;
    t << long(samplePair.data->getNbBinnedEvents()) - long(samplePair.model->getNbBinnedEvents()) << GenericToolbox::TablePrinter::NextColumn;
    t << samplePair.model->getSumWeights() << GenericToolbox::TablePrinter::NextColumn;
    t << samplePair.data->getSumWeights() << GenericToolbox::TablePrinter::NextColumn;
    t << samplePair.data->getSumWeights() - samplePair.model->getSumWeights() << GenericToolbox::TablePrinter::NextLine;
  }

  return t.generateTableString();
}

// static
double LikelihoodInterface::evalPenaltyLikelihood(const ParameterSet& parSet_) {
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

void LikelihoodInterface::load(){

  LogInfo << std::endl; loadModelPropagator();
  LogInfo << std::endl; loadDataPropagator();
  LogInfo << std::endl;

  /// Now caching the event for the plot generator
  _plotGenerator_.defineHistogramHolders();

  this->buildSamplePairList();
  this->printBreakdowns();

  /// model propagator needs to be fast, let the workers wait for the signal
  _modelPropagator_.getThreadPool().setCpuTimeSaverIsEnabled( false );

}
void LikelihoodInterface::loadModelPropagator(){
  LogInfo << "Loading model..." << std::endl;

  _modelPropagator_.clearContent();

  LogInfo << "Loading datasets..." << std::endl;
  for( auto& dataSet : _datasetList_ ){
    dataSet.getModelDispenser().setPlotGeneratorPtr( &_plotGenerator_ );
    dataSet.getModelDispenser().load( _modelPropagator_ );
  }

  _modelPropagator_.shrinkDialContainers();
  _modelPropagator_.buildDialCache();

  LogInfo << "Propagating prior parameters on events..." << std::endl;
  _modelPropagator_.reweightEvents();


  // The histogram bin was assigned to each event by the DataDispenser, now
  // cache the binning results for speed into each of the samples.
  LogInfo << "Filling up model sample bin caches..." << std::endl;
  _threadPool_.runJob([this](int iThread){
    LogInfoIf(iThread <= 0) << "Updating sample per bin event lists..." << std::endl;
    for( auto& sample : _modelPropagator_.getSampleSet().getSampleList() ){
      sample.indexEventInHistogramBin(iThread);
    }
  });

  LogInfo << "Filling up model sample histograms..." << std::endl;
  _threadPool_.runJob([this](int iThread){
    for( auto& sample : _modelPropagator_.getSampleSet().getSampleList() ){
      sample.getHistogram().refillHistogram(iThread);
    }
  });

#ifdef GUNDAM_USING_CACHE_MANAGER
  if( GundamGlobals::isCacheManagerEnabled() ){
    _modelPropagator_.initializeCacheManager();
  }
#endif

  _modelPropagator_.printBreakdowns();

}
void LikelihoodInterface::loadDataPropagator(){
  LogInfo << "Loading data..." << std::endl;

  _dataPropagator_.clearContent();

  bool isAsimov{_dataType_ == DataType::Asimov or _forceAsimovData_};
  for( auto& dataSet : _datasetList_ ){
    if( _dataType_ == DataType::Toy ){
      if( dataSet.getSelectedToyEntry().empty() or dataSet.getSelectedToyEntry() == "Asimov" ){
        isAsimov = true;
        break;
      }
    }
    if( _dataType_ == DataType::RealData ){
      if( dataSet.getSelectedDataEntry().empty() or dataSet.getSelectedDataEntry() == "Asimov" ){
        isAsimov = true;
        break;
      }
    }
  }

  if( isAsimov ){
    // don't reload, just use the _modelPropagator_
    if( _dataType_ == DataType::Toy and _throwAsimovToyParameters_ ){
      throwToyParameters(_modelPropagator_);
    }

    // copy the events directly from the model
    LogInfo << "Copying events from the model..." << std::endl;
    _dataPropagator_.copyEventsFrom( _modelPropagator_ );
    _dataPropagator_.shrinkDialContainers();
    _dataPropagator_.buildDialCache();

    // move the model back to the prior
    if( _dataType_ == DataType::Toy and _throwAsimovToyParameters_ ){
      _modelPropagator_.getParametersManager().moveParametersToPrior();
      _modelPropagator_.reweightEvents();
    }
  }
  else{
    LogInfo << "Loading datasets..." << std::endl;
    for( auto& dataSet : _datasetList_ ){
      // let the llh interface choose witch data entry to load
      auto* dataDispenser{this->getDataDispenser( dataSet )};

      // nothing has been selected, skip (warning message is within getDataDispenser)
      if( dataDispenser == nullptr ){ continue; }

      // TODO: handle it better? -> data dispenser need to be aware about it for fetching the requested variables
      // Is it better to do the fetching here and provide it to the dispenser?
      dataDispenser->setPlotGeneratorPtr( &_plotGenerator_ );

      // handling override of the propagator config
      if( not dataDispenser->getParameters().overridePropagatorConfig.empty() ){
        LogWarning << "Reload the data propagator config with override options..." << std::endl;
        ConfigUtils::ConfigHandler configHandler( _modelPropagator_.getConfig() );
        configHandler.override( dataDispenser->getParameters().overridePropagatorConfig );
        _dataPropagator_.configure( configHandler.getConfig() );
        _dataPropagator_.initialize();
      }

      // legacy: replacing the parameterSet option "maskForToyGeneration" -> now should use the config override above
      for( auto& parSet : _dataPropagator_.getParametersManager().getParameterSetsList() ){
        if( parSet.isMaskForToyGeneration() ){ parSet.nullify(); }
      }


      // otherwise load the dataset
      dataDispenser->getParameters().isData = true;
      dataDispenser->load( _dataPropagator_ );

      // make sure the config is from scratch each time we read a new dataset
      if( not dataDispenser->getParameters().overridePropagatorConfig.empty() ){
        // TODO: handle multiple datasets loading when editing the configuration
//        LogWarning << "Restoring propagator config overrides..." << std::endl;
//        _dataPropagator_.configure( _modelPropagator_.getConfig() );
//        _dataPropagator_.initialize();
      }
    }

    _dataPropagator_.shrinkDialContainers();
    _dataPropagator_.buildDialCache();

    if( _throwAsimovToyParameters_ ){
      throwToyParameters(_dataPropagator_);
    } // throw asimov?

  }

  LogInfo << "Propagating parameters on events..." << std::endl;
  _dataPropagator_.reweightEvents();

  LogInfo << "Filling up data sample bin caches..." << std::endl;
  _threadPool_.runJob([this](int iThread){
    LogInfoIf(iThread <= 0) << "Updating sample per bin event lists..." << std::endl;
    for( auto& sample : _dataPropagator_.getSampleSet().getSampleList() ){
      sample.indexEventInHistogramBin(iThread);
    }
  });

  LogInfo << "Filling up data sample histograms..." << std::endl;
  _threadPool_.runJob([this](int iThread){
    for( auto& sample : _dataPropagator_.getSampleSet().getSampleList() ){
      sample.getHistogram().refillHistogram(iThread);
    }
  });

  if( _dataType_ == DataType::Toy ){ throwStatErrors(_dataPropagator_); }

  _dataPropagator_.printBreakdowns();

}
void LikelihoodInterface::buildSamplePairList(){

  auto nModelSamples{_modelPropagator_.getSampleSet().getSampleList().size()};
  auto nDataSamples{_dataPropagator_.getSampleSet().getSampleList().size()};
  LogExitIf(nModelSamples != nDataSamples,
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
void LikelihoodInterface::throwToyParameters(Propagator& propagator_){

  LogInfo << "Propagating prior parameters on the initially loaded events..." << std::endl;
  propagator_.reweightEvents();

  LogInfo << "Sample breakdown prior to the throwing:" << std::endl;
  std::cout << propagator_.getSampleSet().getSampleBreakdown() << std::endl;

  if( propagator_.isDebugPrintLoadedEvents() ){
    LogDebug << "Toy events:" << std::endl;
    LogDebug << GET_VAR_NAME_VALUE(propagator_.getDebugPrintLoadedEventsNbPerSample()) << std::endl;
    int iEvt{0};
    for( auto& entry : propagator_.getEventDialCache().getCache() ) {
      LogDebug << "Event #" << iEvt++ << "{" << std::endl;
      {
        LogScopeIndent;
        LogDebug << entry.getSummary() << std::endl;
      }
      LogDebug << "}" << std::endl;
      if( iEvt >= propagator_.getDebugPrintLoadedEventsNbPerSample() ) break;
    }
  }

  if( not _toyParameterInjector_.empty() ){
    LogWarning << "Injecting parameters..." << std::endl;
    propagator_.getParametersManager().injectParameterValues( _toyParameterInjector_ );
  }
  else{
    LogWarning << "Throwing toy parameters according to covariance matrices..." << std::endl;
    propagator_.getParametersManager().throwParameters();
  }

  // reweighting the events accordingly
  propagator_.reweightEvents();

}
void LikelihoodInterface::throwStatErrors(Propagator& propagator_){
  LogInfo << "Throwing statistical error..." << std::endl;

  // TODO: those config parameters should be handled here, not within the propagator...
  if( not _enableStatThrowInToys_ ){
    LogAlert << "Stat error throw is disabled. Skipping..." << std::endl;
    return;
  }

  if( _enableEventMcThrow_ ){
    // Take into account the finite amount of event in MC
    LogInfo << "enableEventMcThrow is enabled: throwing individual MC events" << std::endl;
    for( auto& sample : propagator_.getSampleSet().getSampleList() ) {
      if( sample.isEventMcThrowDisabled() ){
        LogAlert << "MC event throw is disabled for sample: " << sample.getName() << std::endl;
      }

      sample.getHistogram().throwEventMcError();
    }
  }
  else{
    LogWarning << "enableEventMcThrow is disabled. Not throwing individual MC events" << std::endl;
  }

  LogInfo << "Throwing statistical error on histograms..." << std::endl;
  if( _gaussStatThrowInToys_ ) {
    LogWarning << "Using gaussian statistical throws. (caveat: distribution truncated when the bins are close to zero)" << std::endl;
  }
  for( auto& sample : propagator_.getSampleSet().getSampleList() ){
    // Asimov bin content -> toy data
    sample.getHistogram().throwStatError( _gaussStatThrowInToys_ );
  }
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
