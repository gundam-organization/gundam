//
// Created by Nadrino on 11/06/2021.
//

#include "Propagator.h"

#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif

#include "FitParameterSet.h"
#include "Dial.h"
#include "JsonUtils.h"
#include "GlobalVariables.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.TablePrinter.h"

#include <memory>
#include <vector>

LoggerInit([]{
  Logger::setUserHeaderStr("[Propagator]");
});

Propagator::Propagator() { this->reset(); }
Propagator::~Propagator() { this->reset(); }

void Propagator::reset() {
  _isInitialized_ = false;
  _parameterSetsList_.clear();
  _saveDir_ = nullptr;

  std::vector<std::string> jobNameRemoveList;
  for( const auto& jobName : GlobalVariables::getParallelWorker().getJobNameList() ){
    if(jobName == "Propagator::fillEventDialCaches"
       or jobName == "Propagator::reweightMcEvents"
       or jobName == "Propagator::updateDialResponses"
       or jobName == "Propagator::refillSampleHistograms"
       or jobName == "Propagator::applyResponseFunctions"
        ){
      jobNameRemoveList.emplace_back(jobName);
    }
  }
  for( const auto& jobName : jobNameRemoveList ){
    GlobalVariables::getParallelWorker().removeJob(jobName);
  }

  _responseFunctionsSamplesMcHistogram_.clear();
  _nominalSamplesMcHistogram_.clear();
}

void Propagator::setShowTimeStats(bool showTimeStats) {
  _showTimeStats_ = showTimeStats;
}
void Propagator::setConfig(const nlohmann::json &config) {
  _config_ = config;
  JsonUtils::forwardConfig(_config_);
}
void Propagator::setSaveDir(TDirectory *saveDir) {
  _saveDir_ = saveDir;
}
void Propagator::setThrowAsimovToyParameters(bool throwAsimovToyParameters) {
  _throwAsimovToyParameters_ = throwAsimovToyParameters;
}
void Propagator::setIThrow(int iThrow) {
  _iThrow_ = iThrow;
}
void Propagator::setLoadAsimovData(bool loadAsimovData) {
  _loadAsimovData_ = loadAsimovData;
}

void Propagator::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  // Monitoring parameters
  _showEventBreakdown_ = JsonUtils::fetchValue(_config_, "showEventBreakdown", _showEventBreakdown_);

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing parameters...") << std::endl;
  auto parameterSetListConfig = JsonUtils::fetchValue(_config_, "parameterSetListConfig", nlohmann::json());
  if( parameterSetListConfig.is_string() ) parameterSetListConfig = JsonUtils::readConfigFile(parameterSetListConfig.get<std::string>());
  int nPars = 0;
  _parameterSetsList_.reserve(parameterSetListConfig.size()); // make sure the objects aren't moved in RAM ( since FitParameter* will be used )
  for( const auto& parameterSetConfig : parameterSetListConfig ){
    _parameterSetsList_.emplace_back();
    _parameterSetsList_.back().setConfig(parameterSetConfig);
    _parameterSetsList_.back().setSaveDir(GenericToolbox::mkdirTFile(_saveDir_, "ParameterSets"));
    _parameterSetsList_.back().initialize();
    nPars += _parameterSetsList_.back().getNbParameters();
    LogInfo << _parameterSetsList_.back().getSummary() << std::endl;
  }

  _globalCovarianceMatrix_ = std::make_shared<TMatrixD>( nPars, nPars );
  int iParOffset = 0;
  for( const auto& parSet : _parameterSetsList_ ){
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
  if( _saveDir_ != nullptr ){
    _saveDir_->cd();
    _globalCovarianceMatrix_->Write("globalCovarianceMatrix_TMatrixD");
  }

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing samples...") << std::endl;
  auto fitSampleSetConfig = JsonUtils::fetchValue(_config_, "fitSampleSetConfig", nlohmann::json());
  _fitSampleSet_.setConfig(fitSampleSetConfig);
  _fitSampleSet_.initialize();

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Initializing the plot generator") << std::endl;
  auto plotGeneratorConfig = JsonUtils::fetchValue(_config_, "plotGeneratorConfig", nlohmann::json());
  if( plotGeneratorConfig.is_string() ) parameterSetListConfig = JsonUtils::readConfigFile(plotGeneratorConfig.get<std::string>());
  _plotGenerator_.setConfig(plotGeneratorConfig);
  _plotGenerator_.initialize();

  _throwAsimovToyParameters_ = JsonUtils::fetchValue<nlohmann::json>(_config_, "throwAsimovFitParameters", _throwAsimovToyParameters_);
  _enableStatThrowInToys_ = JsonUtils::fetchValue<nlohmann::json>(_config_, "enableStatThrowInToys", _enableStatThrowInToys_);

  LogInfo << std::endl << GenericToolbox::addUpDownBars("Loading datasets...") << std::endl;
  auto dataSetListConfig = JsonUtils::getForwardedConfig(_config_, "dataSetList");
  if( dataSetListConfig.empty() ){
    // Old config files
    dataSetListConfig = JsonUtils::getForwardedConfig(_fitSampleSet_.getConfig(), "dataSetList");
    LogAlert << "DEPRECATED CONFIG OPTION: " << "dataSetList should now be located in the Propagator config." << std::endl;
  }
  LogThrowIf(dataSetListConfig.empty(), "No dataSet specified." << std::endl)
  int iDataSet{0};
  _dataSetList_.reserve(dataSetListConfig.size());
  for( const auto& dataSetConfig : dataSetListConfig ){
    _dataSetList_.emplace_back();
    _dataSetList_.back().setConfig(dataSetConfig);
    _dataSetList_.back().setDataSetIndex(iDataSet++);
    _dataSetList_.back().initialize();
  }

  LogInfo << "Initializing propagation threads..." << std::endl;
  initializeThreads();
  GlobalVariables::getParallelWorker().setCpuTimeSaverIsEnabled(true);

  // First start with the data:
  bool usedMcContainer{false};
  bool allAsimov{true};
  for( auto& dataSet : _dataSetList_ ){
    if( not dataSet.isEnabled() ) continue;
    DataDispenser& dispenser = dataSet.getSelectedDataDispenser();
    if( _throwAsimovToyParameters_ ) { dispenser = dataSet.getToyDataDispenser(); }
    if( _loadAsimovData_ ){ dispenser = dataSet.getDataDispenserDict()["Asimov"]; }

    dispenser.getConfigParameters().iThrow = _iThrow_;

    if( dispenser.getConfigParameters().name != "Asimov" ){ allAsimov = false; }
    LogInfo << "Reading dataset: " << dataSet.getName() << "/" << dispenser.getConfigParameters().name << std::endl;

    dispenser.setSampleSetPtrToLoad(&_fitSampleSet_);
    dispenser.setPlotGenPtr(&_plotGenerator_);
    if( dispenser.getConfigParameters().useMcContainer ){
      usedMcContainer = true;
      dispenser.setParSetPtrToLoad(&_parameterSetsList_);
    }
    dispenser.load();
  }

  if( usedMcContainer ){
    if( _throwAsimovToyParameters_ ){
      for( auto& parSet : _parameterSetsList_ ){
        if( parSet.isEnabledThrowToyParameters() and parSet.getPriorCovarianceMatrix() != nullptr ){
          parSet.throwFitParameters();
        }
      }
    }

    LogInfo << "Propagating prior parameters on events..." << std::endl;
    this->reweightMcEvents();

    // Copies MC events in data container for both Asimov and FakeData event types
    LogWarning << "Copying loaded mc-like event to data container..." << std::endl;
    _fitSampleSet_.copyMcEventListToDataContainer();

    // back to prior
    if( _throwAsimovToyParameters_ ){
      for( auto& parSet : _parameterSetsList_ ){
        parSet.moveFitParametersToPrior();
      }
    }
  }

  if( not allAsimov ){
    // reload everything
    // Filling the mc containers
    _fitSampleSet_.clearMcContainers();
    for( auto& dataSet : _dataSetList_ ){
      if( not dataSet.isEnabled() ) continue;
      auto& dispenser = dataSet.getMcDispenser();
      dispenser.setSampleSetPtrToLoad(&_fitSampleSet_);
      dispenser.setPlotGenPtr(&_plotGenerator_);
      dispenser.setParSetPtrToLoad(&_parameterSetsList_);
      dispenser.load();
    }
  }
//  else{
//    LogDebug << "Check asimov: " << std::endl;
//    for( auto& sample : this->getFitSampleSet().getFitSampleList() ){
//      LogDebug << sample.getName() << std::endl;
//      size_t nDiff{0};
//      for( size_t iEvent = 0 ; iEvent < sample.getMcContainer().eventList.size() ; iEvent++ ){
//        auto& mcEvent = sample.getMcContainer().eventList[iEvent];
//        auto& dataEvent = sample.getDataContainer().eventList[iEvent];
//        if( nDiff<15 and mcEvent.getEventWeight() != dataEvent.getEventWeight() ){
//          nDiff++;
//          LogDebug
//              << mcEvent.getEventWeight() << " => " << dataEvent.getEventWeight()
//              << " / diff: " << mcEvent.getEventWeight() - dataEvent.getEventWeight() << std::endl;
//        }
//      }
//    }
////    LogThrow("debug")
//  }

  LogInfo << "Propagating prior parameters on events..." << std::endl;
  this->reweightMcEvents();

  LogInfo << "Set the current MC prior weights as nominal weight..." << std::endl;
  for( auto& sample : _fitSampleSet_.getFitSampleList() ){
    for( auto& event : sample.getMcContainer().eventList ){
      event.setNominalWeight(event.getEventWeight());
    }
  }

//  if( GlobalVariables::isEnableDevMode() ){
//    LogInfo << "Loading dials stack..." << std::endl;
//    fillDialsStack();
//  }

#ifdef GUNDAM_USING_CACHE_MANAGER
  // After all of the data has been loaded.  Specifically, this must be after
  // the MC has been copied for the Asimov fit, or the "data" use the MC
  // reweighting cache.  This must also be before the first use of
  // reweightMcEvents.
  if(GlobalVariables::getEnableCacheManager()) Cache::Manager::Build(getFitSampleSet());
#endif

  if( _showEventBreakdown_ ){
    {
      // STAGED MASK
      LogWarning << "Staged event breakdown:" << std::endl;
      Dial::enableMaskCheck = true;
      std::vector<std::vector<double>> stageBreakdownList(
          _fitSampleSet_.getFitSampleList().size(),
          std::vector<double>(_parameterSetsList_.size()+1, 0)
      ); // [iSample][iStage]
      std::vector<std::string> stageTitles;
      stageTitles.emplace_back("Sample");
      stageTitles.emplace_back("No reweight");
      for( auto& parSet : _parameterSetsList_ ){
        stageTitles.emplace_back("+ " + parSet.getName());
      }

      int iStage{0};
      for( auto& parSet : _parameterSetsList_ ){ parSet.setMaskedForPropagation(true); }
      this->reweightMcEvents();
      for( size_t iSample = 0 ; iSample < _fitSampleSet_.getFitSampleList().size() ; iSample++ ){
        stageBreakdownList[iSample][iStage] = _fitSampleSet_.getFitSampleList()[iSample].getMcContainer().getSumWeights();
      }

      for( auto& parSet : _parameterSetsList_ ){
        parSet.setMaskedForPropagation(false);
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
      Dial::enableMaskCheck = false;
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

  _plotGenerator_.setFitSampleSetPtr(&_fitSampleSet_);
  _plotGenerator_.defineHistogramHolders();

  LogInfo << "Filling up sample bin caches..." << std::endl;
  _fitSampleSet_.updateSampleBinEventList();

  LogInfo << "Filling up sample histograms..." << std::endl;
  _fitSampleSet_.updateSampleHistograms();

  // Now the data won't be refilled each time
  for( auto& sample : _fitSampleSet_.getFitSampleList() ){
    if( _throwAsimovToyParameters_ and _enableStatThrowInToys_ ){ sample.getDataContainer().throwStatError(); }
    sample.getDataContainer().isLocked = true;
  }

  _useResponseFunctions_ = JsonUtils::fetchValue<nlohmann::json>(_config_, "DEV_useResponseFunctions", false);
  if( _useResponseFunctions_ ){ this->makeResponseFunctions(); }

  if( JsonUtils::fetchValue<nlohmann::json>(_config_, "throwAsimovFitParameters", false) ){
    for( auto& parSet : _parameterSetsList_ ){
      for( auto& par : parSet.getParameterList() ){
        par.setParameterValue( par.getPriorValue() );
      }
    }
    propagateParametersOnSamples();
  }

  _treeWriter_.setFitSampleSetPtr(&_fitSampleSet_);
  _treeWriter_.setParSetListPtr(&_parameterSetsList_);

  // Propagator needs to be fast
  GlobalVariables::getParallelWorker().setCpuTimeSaverIsEnabled(false);

  _isInitialized_ = true;
}

bool Propagator::isUseResponseFunctions() const {
  return _useResponseFunctions_;
}
bool Propagator::isThrowAsimovToyParameters() const {
  return _throwAsimovToyParameters_;
}
FitSampleSet &Propagator::getFitSampleSet() {
  return _fitSampleSet_;
}
std::vector<FitParameterSet> &Propagator::getParameterSetsList() {
  return _parameterSetsList_;
}
const std::vector<FitParameterSet> &Propagator::getParameterSetsList() const {
  return _parameterSetsList_;
}
PlotGenerator &Propagator::getPlotGenerator() {
  return _plotGenerator_;
}
const nlohmann::json &Propagator::getConfig() const {
  return _config_;
}


void Propagator::propagateParametersOnSamples(){

  // Only real parameters are propagated on the specta -> need to convert the eigen to original
  for( auto& parSet : _parameterSetsList_ ){
    if( parSet.isUseEigenDecompInFit() ) parSet.propagateEigenToOriginal();
  }

  if(not _useResponseFunctions_ or not _isRfPropagationEnabled_ ){
//    if(GlobalVariables::isEnableDevMode()) updateDialResponses();
    reweightMcEvents();
    refillSampleHistograms();
  }
  else{
    applyResponseFunctions();
  }

}
void Propagator::updateDialResponses(){
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GlobalVariables::getParallelWorker().runJob("Propagator::updateDialResponses");
  dialUpdate.counts++; dialUpdate.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
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
    GlobalVariables::getParallelWorker().runJob("Propagator::reweightMcEvents");
  }
  weightProp.counts++;
  weightProp.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
}

void Propagator::refillSampleHistograms(){
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GlobalVariables::getParallelWorker().runJob("Propagator::refillSampleHistograms");
  fillProp.counts++; fillProp.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
}

void Propagator::applyResponseFunctions(){
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GlobalVariables::getParallelWorker().runJob("Propagator::applyResponseFunctions");
  applyRf.counts++; applyRf.cumulated += GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
}

void Propagator::preventRfPropagation(){
  if(_isRfPropagationEnabled_){
//    LogInfo << "Parameters propagation using Response Function is now disabled." << std::endl;
    _isRfPropagationEnabled_ = false;
  }
}
void Propagator::allowRfPropagation(){
  if(not _isRfPropagationEnabled_){
//    LogWarning << "Parameters propagation using Response Function is now ENABLED." << std::endl;
    _isRfPropagationEnabled_ = true;
  }
}

void Propagator::fillDialsStack(){
  for( auto& parSet : _parameterSetsList_ ){
    if( not parSet.isUseEigenDecompInFit() ){
      for( auto& par : parSet.getParameterList() ){
        for( auto& dialSet : par.getDialSetList() ){
          if(dialSet.getGlobalDialType() == DialType::Norm){ continue; } // no cache needed
          for( auto& dial : dialSet.getDialList() ){
            if(dial->isReferenced()) _dialsStack_.emplace_back(dial.get());
          } // dial
        } // dialSet
      } // par
    }
  } // parSet
}


// Protected
void Propagator::initializeThreads() {

  std::function<void(int)> reweightMcEventsFct = [this](int iThread){
    this->reweightMcEvents(iThread);
  };
  GlobalVariables::getParallelWorker().addJob("Propagator::reweightMcEvents", reweightMcEventsFct);

  std::function<void(int)> updateDialResponsesFct = [this](int iThread){
    this->updateDialResponses(iThread);
  };
  GlobalVariables::getParallelWorker().addJob("Propagator::updateDialResponses", updateDialResponsesFct);


  std::function<void(int)> refillSampleHistogramsFct = [this](int iThread){
    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      sample.getMcContainer().refillHistogram(iThread);
      sample.getDataContainer().refillHistogram(iThread);
    }
  };
  std::function<void()> refillSampleHistogramsPostParallelFct = [this](){
    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      sample.getMcContainer().rescaleHistogram();
      sample.getDataContainer().rescaleHistogram();
    }
  };
  GlobalVariables::getParallelWorker().addJob("Propagator::refillSampleHistograms", refillSampleHistogramsFct);
  GlobalVariables::getParallelWorker().setPostParallelJob("Propagator::refillSampleHistograms", refillSampleHistogramsPostParallelFct);

  std::function<void(int)> applyResponseFunctionsFct = [this](int iThread){
    this->applyResponseFunctions(iThread);
  };
  GlobalVariables::getParallelWorker().addJob("Propagator::applyResponseFunctions", applyResponseFunctionsFct);
}

void Propagator::makeResponseFunctions(){
  LogWarning << __METHOD_NAME__ << std::endl;

  this->preventRfPropagation(); // make sure, not yet setup

  for( auto& parSet : _parameterSetsList_ ){
    for( auto& par : parSet.getParameterList() ){
      par.setParameterValue(par.getPriorValue());
    }
  }
  this->propagateParametersOnSamples();

  for( auto& sample : _fitSampleSet_.getFitSampleList() ){
    _nominalSamplesMcHistogram_[&sample] = std::shared_ptr<TH1D>((TH1D*) sample.getMcContainer().histogram->Clone());
  }

  for( auto& parSet : _parameterSetsList_ ){
    for( auto& par : parSet.getParameterList() ){
      LogInfo << "Make RF for " << parSet.getName() << "/" << par.getTitle() << std::endl;
      par.setParameterValue(par.getPriorValue() + par.getStdDevValue());

      this->propagateParametersOnSamples();

      for( auto& sample : _fitSampleSet_.getFitSampleList() ){
        _responseFunctionsSamplesMcHistogram_[&sample].emplace_back(std::shared_ptr<TH1D>((TH1D*) sample.getMcContainer().histogram->Clone()) );
        GenericToolbox::transformBinContent(_responseFunctionsSamplesMcHistogram_[&sample].back().get(), [&](TH1D* h_, int b_){
          h_->SetBinContent(
              b_,
              (h_->GetBinContent(b_)/_nominalSamplesMcHistogram_[&sample]->GetBinContent(b_))-1);
          h_->SetBinError(b_,0);
        });
      }

      par.setParameterValue(par.getPriorValue());
    }
  }
  this->propagateParametersOnSamples(); // back to nominal

  // WRITE
  if( _saveDir_ != nullptr ){
    auto* rfDir = GenericToolbox::mkdirTFile(_saveDir_, "RF");
    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      GenericToolbox::mkdirTFile(rfDir, "nominal")->cd();
      _nominalSamplesMcHistogram_[&sample]->Write(Form("nominal_%s", sample.getName().c_str()));

      int iPar = -1;
      auto* devDir = GenericToolbox::mkdirTFile(rfDir, "deviation");
      for( auto& parSet : _parameterSetsList_ ){
        auto* parSetDir = GenericToolbox::mkdirTFile(devDir, parSet.getName());
        for( auto& par : parSet.getParameterList() ){
          iPar++;
          GenericToolbox::mkdirTFile(parSetDir, par.getTitle())->cd();
          _responseFunctionsSamplesMcHistogram_[&sample].at(iPar)->Write(Form("dev_%s", sample.getName().c_str()));
        }
      }
    }
    _saveDir_->cd();
  }

  LogInfo << "RF built" << std::endl;
}

void Propagator::updateDialResponses(int iThread_){
  int nThreads = GlobalVariables::getNbThreads();
  if(iThread_ == -1){
    // force single thread
    nThreads = 1;
    iThread_ = 0;
  }
  int iDial{0};
  int nDials(int(_dialsStack_.size()));

  while( iDial < nDials ){
    _dialsStack_[iDial]->evalResponse();
    iDial += nThreads;
  }

}

void Propagator::reweightMcEvents(int iThread_) {
  int nThreads = GlobalVariables::getNbThreads();
  if(iThread_ == -1){
    // force single thread
    nThreads = 1;
    iThread_ = 0;
  }

  //! Warning: everything you modify here, may significantly slow down the fitter
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
}
void Propagator::applyResponseFunctions(int iThread_){

  TH1D* histBuffer{nullptr};
  TH1D* nominalHistBuffer{nullptr};
  TH1D* rfHistBuffer{nullptr};
  for( auto& sample : _fitSampleSet_.getFitSampleList() ){
    histBuffer = sample.getMcContainer().histogram.get();
    nominalHistBuffer = _nominalSamplesMcHistogram_[&sample].get();
    for( int iBin = 1 ; iBin <= histBuffer->GetNbinsX() ; iBin++ ){
      if( iBin % GlobalVariables::getNbThreads() != iThread_ ) continue;
      histBuffer->SetBinContent(iBin, nominalHistBuffer->GetBinContent(iBin));
    }
  }

  int iPar = -1;
  for( auto& parSet : _parameterSetsList_ ){
    for( auto& par : parSet.getParameterList() ){
      iPar++;
      double xSigmaPar = par.getDistanceFromNominal();
      if( xSigmaPar == 0 ) continue;

      for( auto& sample : _fitSampleSet_.getFitSampleList() ){
        histBuffer = sample.getMcContainer().histogram.get();
        nominalHistBuffer = _nominalSamplesMcHistogram_[&sample].get();
        rfHistBuffer = _responseFunctionsSamplesMcHistogram_[&sample][iPar].get();

        for( int iBin = 1 ; iBin <= histBuffer->GetNbinsX() ; iBin++ ){
          if( iBin % GlobalVariables::getNbThreads() != iThread_ ) continue;
          histBuffer->SetBinContent(
              iBin,
              histBuffer->GetBinContent(iBin) * ( 1 + xSigmaPar * rfHistBuffer->GetBinContent(iBin) )
          );
        }
      }
    }
  }

  for( auto& sample : _fitSampleSet_.getFitSampleList() ){
    histBuffer = sample.getMcContainer().histogram.get();
    nominalHistBuffer = _nominalSamplesMcHistogram_[&sample].get();
    for( int iBin = 1 ; iBin <= histBuffer->GetNbinsX() ; iBin++ ){
      if( iBin % GlobalVariables::getNbThreads() != iThread_ ) continue;
      histBuffer->SetBinError(iBin, TMath::Sqrt(histBuffer->GetBinContent(iBin)));
//      if( iThread_ == 0 ){
//        LogTrace << GET_VAR_NAME_VALUE(iBin)
//        << " / " << GET_VAR_NAME_VALUE(histBuffer->GetBinContent(iBin))
//        << " / " << GET_VAR_NAME_VALUE(nominalHistBuffer->GetBinContent(iBin))
//        << std::endl;
//      }
    }
  }

}

const EventTreeWriter &Propagator::getTreeWriter() const {
  return _treeWriter_;
}

int Propagator::getIThrow() const {
  return _iThrow_;
}

