//
// Created by Nadrino on 11/06/2021.
//

#include <AnaTreeMC.hh>
#include "vector"

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

#include "JsonUtils.h"
#include "Propagator.h"
#include "GlobalVariables.h"
#include "Dial.h"
#include "FitParameterSet.h"

#include "NormalizationDial.h"
#include "SplineDial.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[Propagator]");
})

Propagator::Propagator() { this->reset(); }
Propagator::~Propagator() { this->reset(); }

void Propagator::reset() {
  _isInitialized_ = false;
  _parameterSetsList_.clear();
  _saveDir_ = nullptr;

  std::vector<std::string> jobNameRemoveList;
  for( const auto& jobName : GlobalVariables::getParallelWorker().getJobNameList() ){
    if(jobName == "Propagator::fillEventDialCaches"
    or jobName == "Propagator::reweightSampleEvents"
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
void Propagator::setSaveDir(TDirectory *saveDir) {
  _saveDir_ = saveDir;
}
void Propagator::setConfig(const json &config) {
  _config_ = config;
  while( _config_.is_string() ){
    LogWarning << "Forwarding " << __CLASS_NAME__ << " config: \"" << _config_.get<std::string>() << "\"" << std::endl;
    _config_ = JsonUtils::readConfigFile(_config_.get<std::string>());
  }
}

// To get rid of
void Propagator::setDataTree(TTree *dataTree_) {
  dataTree = dataTree_;
}
void Propagator::setMcFilePath(const std::string &mcFilePath) {
  mc_file_path = mcFilePath;
}

void Propagator::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  LogTrace << "Parameters..." << std::endl;
  auto parameterSetListConfig = JsonUtils::fetchValue<json>(_config_, "parameterSetListConfig");
  if( parameterSetListConfig.is_string() ) parameterSetListConfig = JsonUtils::readConfigFile(parameterSetListConfig.get<std::string>());
  for( const auto& parameterSetConfig : parameterSetListConfig ){
    _parameterSetsList_.emplace_back();
    _parameterSetsList_.back().setJsonConfig(parameterSetConfig);
    _parameterSetsList_.back().initialize();
    LogInfo << _parameterSetsList_.back().getSummary() << std::endl;
  }

  LogDebug << "FitSampleSet..." << std::endl;
  auto fitSampleSetConfig = JsonUtils::fetchValue(_config_, "fitSampleSetConfig", nlohmann::json());
  if( not fitSampleSetConfig.empty() ){
    _fitSampleSet_.setConfig(fitSampleSetConfig);
    _fitSampleSet_.initialize();
  }
  else{
    LogTrace << "Samples..." << std::endl;
    auto samplesConfig = JsonUtils::fetchValue<json>(_config_, "samplesConfig");
    if( samplesConfig.is_string() ) samplesConfig = JsonUtils::readConfigFile(samplesConfig.get<std::string>());
    for( const auto& sampleConfig : samplesConfig ){
      if( JsonUtils::fetchValue(sampleConfig, "isEnabled", true) ){
        _samplesList_.emplace_back();
        _samplesList_.back().setupWithJsonConfig(sampleConfig);
        _samplesList_.back().setDataTree(dataTree);
        _samplesList_.back().Initialize();
      }
    }

    auto* selected_events_AnaTreeMC = new AnaTreeMC(mc_file_path, "selectedEvents"); // trouble while deleting... > need to check
    LogInfo << "Reading and collecting events..." << std::endl;
    std::vector<SignalDef> buf;
    std::vector<AnaSample*> samplePtrList;
    for( auto& sample : _samplesList_ ) samplePtrList.emplace_back(&sample);
    selected_events_AnaTreeMC->GetEvents(samplePtrList, buf, false);
  }


  LogTrace << "Initializing the PlotGenerator" << std::endl;
  auto plotGeneratorConfig = JsonUtils::fetchValue<json>(_config_, "plotGeneratorConfig");
  if( plotGeneratorConfig.is_string() ) parameterSetListConfig = JsonUtils::readConfigFile(plotGeneratorConfig.get<std::string>());
  _plotGenerator_.setConfig(plotGeneratorConfig);
  _plotGenerator_.initialize();

  if( not _fitSampleSet_.empty() ){
    LogInfo << "Polling the requested leaves to load in memory..." << std::endl;
    for( auto& dataSet : _fitSampleSet_.getDataSetList() ){

      // parSet
      for( auto& parSet : _parameterSetsList_ ){
        if( not parSet.isEnabled() ) continue;

        for( auto& par : parSet.getParameterList() ){
          if( not par.isEnabled() ) continue;

          auto* dialSetPtr = par.findDialSet( dataSet.getName() );
          if( dialSetPtr == nullptr ){ continue; }

          if( dialSetPtr->getApplyConditionFormula() != nullptr ){
            for( int iPar = 0 ; iPar < dialSetPtr->getApplyConditionFormula()->GetNpar() ; iPar++ ){
              dataSet.addRequestedLeafName(dialSetPtr->getApplyConditionFormula()->GetParName(iPar));
            }
          }

          for( auto& dial : dialSetPtr->getDialList() ){
            for( auto& var : dial->getApplyConditionBin().getVariableNameList() ){
              dataSet.addRequestedLeafName(var);
            } // var
          } // dial
        } // par
      } // parSet

      // plotGen
      auto varListRequestedByPlotGen = _plotGenerator_.fetchRequestedLeafNames();
      for( const auto& varName : varListRequestedByPlotGen ){
        dataSet.addRequestedLeafName(varName);
      }

    } // dataSets

    _fitSampleSet_.loadPhysicsEvents();
    _plotGenerator_.setFitSampleSetPtr(&_fitSampleSet_);
  }
  else{
    // OLD SAMPLES
    _plotGenerator_.setSampleListPtr( &_samplesList_ );
  }
  _plotGenerator_.defineHistogramHolders();

  initializeThreads();
  initializeCaches();

  fillEventDialCaches();

  if( JsonUtils::fetchValue<json>(_config_, "throwParameters", false) ){
    LogWarning << "Throwing parameters..." << std::endl;
    for( auto& parSet : _parameterSetsList_ ){
      auto thrownPars = GenericToolbox::throwCorrelatedParameters(GenericToolbox::getCholeskyMatrix(parSet.getCovarianceMatrix()));
      for( auto& par : parSet.getParameterList() ){
        par.setParameterValue( par.getPriorValue() + thrownPars.at(par.getParameterIndex()) );
        LogDebug << parSet.getName() << "/" << par.getTitle() << ": thrown = " << par.getParameterValue() << std::endl;
      }
    }
  }

  LogInfo << "Propagating prior parameters on events..." << std::endl;
  reweightSampleEvents();

  if( not fitSampleSetConfig.empty() ){

    LogInfo << "Set the current MC prior weights as nominal weight..." << std::endl;
    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      for( auto& event : sample.getMcContainer().eventList ){
        event.setNominalWeight(event.getEventWeight());
      }
    }

    if( _fitSampleSet_.getDataEventType() == DataEventType::Asimov ){
      LogInfo << "Propagating prior weights on data Asimov events..." << std::endl;
      for( auto& sample : _fitSampleSet_.getFitSampleList() ){
        sample.getDataContainer().histScale = sample.getMcContainer().histScale;
        int nEvents = int(sample.getMcContainer().eventList.size());
        for( int iEvent = 0 ; iEvent < nEvents ; iEvent++ ){
          // Since no reweight is applied on data samples, the nominal weight should be the default one
          sample.getDataContainer().eventList.at(iEvent).setTreeWeight(
            sample.getMcContainer().eventList.at(iEvent).getNominalWeight()
            );
          sample.getDataContainer().eventList.at(iEvent).resetEventWeight();
          sample.getDataContainer().eventList.at(iEvent).setNominalWeight(sample.getDataContainer().eventList.at(iEvent).getEventWeight());
        }
      }
    }

    LogInfo << "Filling up sample bin caches..." << std::endl;
    _fitSampleSet_.updateSampleBinEventList();

    LogInfo << "Filling up sample histograms..." << std::endl;
    _fitSampleSet_.updateSampleHistograms();

    // Now the data won't be refilled each time
    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      sample.getDataContainer().isLocked = true;
    }

    _useResponseFunctions_ = JsonUtils::fetchValue<json>(_config_, "useResponseFunctions", false);
    if( _useResponseFunctions_ ){
      this->makeResponseFunctions();
    }
  }
  else{
    for( auto& sample : _samplesList_ ){
      sample.FillEventHist(
        DataType::kAsimov,
        false
        );
    }
  }

  if( JsonUtils::fetchValue<json>(_config_, "throwParameters", false) ){
    for( auto& parSet : _parameterSetsList_ ){
      for( auto& par : parSet.getParameterList() ){
        par.setParameterValue( par.getPriorValue() );
      }
    }
  }

  _isInitialized_ = true;
}

bool Propagator::isUseResponseFunctions() const {
  return _useResponseFunctions_;
}
FitSampleSet &Propagator::getFitSampleSet() {
  return _fitSampleSet_;
}
std::vector<AnaSample> &Propagator::getSamplesList() {
  return _samplesList_;
}
std::vector<FitParameterSet> &Propagator::getParameterSetsList() {
  return _parameterSetsList_;
}
PlotGenerator &Propagator::getPlotGenerator() {
  return _plotGenerator_;
}


void Propagator::propagateParametersOnSamples(){

  if(not _useResponseFunctions_ or not _isRfPropagationEnabled_ ){
    reweightSampleEvents();
    refillSampleHistograms();
  }
  else{
    applyResponseFunctions();
  }

}
void Propagator::reweightSampleEvents() {
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GlobalVariables::getParallelWorker().runJob("Propagator::reweightSampleEvents");
  weightPropagationTime = GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__);
  if( _showTimeStats_ ) {
    LogDebug << __METHOD_NAME__ << " took: " << weightPropagationTime << std::endl;
  }
}
void Propagator::refillSampleHistograms(){
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GlobalVariables::getParallelWorker().runJob("Propagator::refillSampleHistograms");
  fillPropagationTime = GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__);
  if( _showTimeStats_ ) LogDebug << __METHOD_NAME__ << " took: " << fillPropagationTime << std::endl;
}
void Propagator::applyResponseFunctions(){
  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GlobalVariables::getParallelWorker().runJob("Propagator::applyResponseFunctions");
  applyRfTime = GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__);
  if( _showTimeStats_ ) LogDebug << __METHOD_NAME__ << " took: " << fillPropagationTime << std::endl;
}

void Propagator::preventRfPropagation(){
  if(_isRfPropagationEnabled_){
    LogInfo << "Parameters propagation using Response Function is now disabled." << std::endl;
    _isRfPropagationEnabled_ = false;
  }
}
void Propagator::allowRfPropagation(){
  if(not _isRfPropagationEnabled_){
    LogWarning << "Parameters propagation using Response Function is now ENABLED." << std::endl;
    _isRfPropagationEnabled_ = true;
  }
}


// Protected
void Propagator::initializeThreads() {

  std::function<void(int)> fillEventDialCacheFct = [this](int iThread){
    this->fillEventDialCaches(iThread);
  };
  GlobalVariables::getParallelWorker().addJob("Propagator::fillEventDialCaches", fillEventDialCacheFct);

  std::function<void(int)> reweightSampleEventsFct = [this](int iThread){
    this->reweightSampleEvents(iThread);
  };
  GlobalVariables::getParallelWorker().addJob("Propagator::reweightSampleEvents", reweightSampleEventsFct);

  std::function<void(int)> refillSampleHistogramsFct = [this](int iThread){
    if( _fitSampleSet_.empty() ){
      for( auto& sample : _samplesList_ ){
        sample.FillMcHistograms(GlobalVariables::getNbThreads() == 1 ? -1 : iThread);
      }
    }
    else{
      for( auto& sample : _fitSampleSet_.getFitSampleList() ){
        sample.getMcContainer().refillHistogram(iThread);
        sample.getDataContainer().refillHistogram(iThread);
      }
    }
  };
  std::function<void()> refillSampleHistogramsPostParallelFct = [this](){
    if( _fitSampleSet_.empty() ){
      if(GlobalVariables::getNbThreads() != 1){
        for( auto& sample : _samplesList_ ){
          sample.MergeMcHistogramsThread();
        }
      }
    }
    else{
      for( auto& sample : _fitSampleSet_.getFitSampleList() ){
        sample.getMcContainer().rescaleHistogram();
        sample.getDataContainer().rescaleHistogram();
      }
    }
  };
  GlobalVariables::getParallelWorker().addJob("Propagator::refillSampleHistograms", refillSampleHistogramsFct);
  GlobalVariables::getParallelWorker().setPostParallelJob("Propagator::refillSampleHistograms", refillSampleHistogramsPostParallelFct);

  std::function<void(int)> applyResponseFunctionsFct = [this](int iThread){
    this->applyResponseFunctions(iThread);
  };
  GlobalVariables::getParallelWorker().addJob("Propagator::applyResponseFunctions", applyResponseFunctionsFct);

}
void Propagator::initializeCaches() {
  LogInfo << __METHOD_NAME__ << std::endl;

  for( auto& sample : _samplesList_ ){
    int nEvents = sample.GetN();
    for( int iEvent = 0 ; iEvent < nEvents ; iEvent++ ){
      for( auto& parSet : _parameterSetsList_ ){
        auto* dialCache = sample.GetEvent(iEvent)->getDialCachePtr();
        (*dialCache)[&parSet] = std::vector<Dial*>(parSet.getNbParameters(), nullptr);
      } // parSet
    } // event
  } // sample


  for( auto& sample : _fitSampleSet_.getFitSampleList() ){
    for( auto& event : sample.getMcContainer().eventList ){
      for( auto& parSet : _parameterSetsList_ ){
        auto* dialCache = event.getDialCachePtr();
        (*dialCache)[&parSet] = std::vector<Dial*>(parSet.getNbParameters(), nullptr);
      } // parSet
    } // event
  } // sample

}
void Propagator::fillEventDialCaches(){
  LogInfo << __METHOD_NAME__ << std::endl;
  GlobalVariables::getParallelWorker().runJob("Propagator::fillEventDialCaches");
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

void Propagator::reweightSampleEvents(int iThread_) {
  double weight;

  if( _fitSampleSet_.empty() ){
    AnaEvent* eventPtr;
    for( auto& sample : _samplesList_ ){
      int nEvents = sample.GetN();
      for( int iEvent = 0 ; iEvent < nEvents ; iEvent++ ){

        if( iEvent % GlobalVariables::getNbThreads() != iThread_ ){
          continue;
        }

        eventPtr = sample.GetEvent(iEvent);
        eventPtr->ResetEvWght();

        // Loop over the parSet that are cached (missing ones won't apply on this event anyway)
        for( auto& parSetDialCache : *eventPtr->getDialCachePtr() ){

          weight = 1;
          for( size_t iPar = 0 ; iPar < parSetDialCache.first->getNbParameters() ; iPar++ ){

            Dial* dialPtr = parSetDialCache.second.at(iPar);
            if( dialPtr == nullptr ) continue;

            // No need to recast dialPtr as a NormDial or whatever, it will automatically fetch the right method
            weight *= dialPtr->evalResponse( parSetDialCache.first->getFitParameter(iPar).getParameterValue() );

            // TODO: check if weight cap
            if( weight <= 0 ){
              weight = 0;
              break;
              //            LogError << GET_VAR_NAME_VALUE(iPar) << std::endl;
              //            LogError << GET_VAR_NAME_VALUE(weight) << std::endl;
              //            throw std::runtime_error("<0 weight");
            }

          }

          eventPtr->AddEvWght(weight);

        } // parSetCache

      } // event
    } // sample
  }
  else{
    PhysicsEvent* evPtr;
    Dial* dialPtr;
    int nThreads = GlobalVariables::getNbThreads();
    for( auto& sample : _fitSampleSet_.getFitSampleList() ){
      int nEvents = int(sample.getMcContainer().eventList.size());
      for( int iEvent = 0 ; iEvent < nEvents ; iEvent++ ){
        if( iEvent % nThreads != iThread_ ){
          continue;
        }

        evPtr = &sample.getMcContainer().eventList.at(iEvent);
        evPtr->resetEventWeight();

        // Loop over the parSet that are cached (missing ones won't apply on this event anyway)
        for( auto& parSetDialCache : *evPtr->getDialCachePtr() ){

          weight = 1;
          for( size_t iPar = 0 ; iPar < parSetDialCache.first->getNbParameters() ; iPar++ ){

            dialPtr = parSetDialCache.second.at(iPar);
            if( dialPtr == nullptr ) continue;

            // No need to recast dialPtr as a NormDial or whatever, it will automatically fetch the right method
            weight *= dialPtr->evalResponse( parSetDialCache.first->getFitParameter(iPar).getParameterValue() );

            // TODO: check if weight cap
            if( weight <= 0 ){
              weight = 0;
              break;
            }

          }

          evPtr->addEventWeight(weight);

        } // parSetCache
      } // event
    } // sample
  }
}
void Propagator::fillEventDialCaches(int iThread_){

  DialSet* parameterDialSetPtr;

  if( _fitSampleSet_.empty() ){
    AnaEvent* eventPtr{nullptr};
    if (iThread_ == GlobalVariables::getNbThreads() - 1) LogTrace << "Old samples" << std::endl;
    for( auto& parSet : _parameterSetsList_ ){
      if( not parSet.isEnabled() ){ continue; }
      int iPar = -1;
      for( auto& par : parSet.getParameterList() ){
        iPar++;
        if( not par.isEnabled() ){ continue; }

        for( auto& sample : _samplesList_ ) {
          int nEvents = sample.GetN();
          if (nEvents == 0) continue;

          // selecting the dialSet of the sample
          parameterDialSetPtr = par.findDialSet(sample.GetDetector());
          if (parameterDialSetPtr->getDialList().empty()) {
            continue;
          }

          // Indexing the variables
          eventPtr = sample.GetEvent(0);
          const auto &firstDial = parameterDialSetPtr->getDialList()[0];
          std::vector<int> varIndexList(firstDial->getApplyConditionBin().getVariableNameList().size(), 0);
          std::vector<bool> isIntList(firstDial->getApplyConditionBin().getVariableNameList().size(), true);
          for (size_t iVar = 0; iVar < firstDial->getApplyConditionBin().getVariableNameList().size(); iVar++) {
            varIndexList.at(iVar) = (eventPtr->GetIntIndex(
              firstDial->getApplyConditionBin().getVariableNameList().at(iVar), false));
            if (varIndexList.at(iVar) == -1) {
              isIntList.at(iVar) = false;
              varIndexList.at(iVar) = (eventPtr->GetFloatIndex(
                firstDial->getApplyConditionBin().getVariableNameList().at(iVar), false));
            }
          }

          std::stringstream ss;
          ss << LogWarning.getPrefixString() << "Indexing event dials: " << parSet.getName() << "/" << par.getTitle() << " -> " << sample.GetName();
          if( iThread_ == GlobalVariables::getNbThreads()-1 ){
            GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
          }

          int nbDialsSet = 0;
          for (int iEvent = 0; iEvent < nEvents; iEvent++) {

            if (iEvent % GlobalVariables::getNbThreads() != iThread_) {
              continue;
            }
            if (iThread_ == GlobalVariables::getNbThreads() - 1) {
              GenericToolbox::displayProgressBar(iEvent, nEvents, ss.str());
            }

            eventPtr = sample.GetEvent(iEvent);
            if (eventPtr->getDialCachePtr()->at(&parSet).at(iPar) != nullptr) {
              // already set
              continue;
            }

            if (parameterDialSetPtr->getApplyConditionFormula() != nullptr
            and eventPtr->evalFormula(parameterDialSetPtr->getApplyConditionFormula()) == 0
            ) {
              continue; // SKIP
            }

            for (const auto &dial : parameterDialSetPtr->getDialList()) {
              bool isInBin = true;
              for (size_t iVar = 0; iVar < varIndexList.size(); iVar++) {
                if (isIntList.at(iVar)) {
                  if (
                    not dial->getApplyConditionBin().isBetweenEdges(
                      iVar, eventPtr->GetEventVarInt(varIndexList.at(iVar)))
                      ){
                    isInBin = false;
                    break; // next dial
                  }
                }
                else {
                  if (
                    not dial->getApplyConditionBin().isBetweenEdges(
                      iVar, eventPtr->GetEventVarFloat(varIndexList.at(iVar)))
                      ) {
                    isInBin = false;
                    break; // next dial
                  }
                }
              }
              if (isInBin) {
                eventPtr->getDialCachePtr()->at(&parSet).at(iPar) = dial.get();
                //              eventPtr->Print();
                //              LogDebug << GenericToolbox::parseVectorAsString(firstDial->getApplyConditionBin().getVariableNameList()) << std::endl;
                //              LogDebug << GenericToolbox::parseVectorAsString(varIndexList) << std::endl;
                //              LogDebug << GenericToolbox::parseVectorAsString(isIntList) << std::endl;
                //              exit(0);
                nbDialsSet++;
                break; // found
              }
            } // dial

          } // iEvent

          if (iThread_ == GlobalVariables::getNbThreads() - 1) {
            GenericToolbox::displayProgressBar(nEvents, nEvents, ss.str());
            //          LogTrace << sample.GetName() << ": " << GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__) << " " << GET_VAR_NAME_VALUE(nbDialsSet) << std::endl;
          }
        } // sample

      } // par
    } // parSet
  }
  else{
    PhysicsEvent* evPtr{nullptr};
    if (iThread_ == GlobalVariables::getNbThreads() - 1) LogTrace << "New samples" << std::endl;
    for( auto& parSet : _parameterSetsList_ ){
      if( not parSet.isEnabled() ){ continue; }
      int iPar = -1;
      for( auto& par : parSet.getParameterList() ){
        iPar++;
        if( not par.isEnabled() ){ continue; }

        for( size_t iDataSet = 0 ; iDataSet < _fitSampleSet_.getDataSetList().size() ; iDataSet++ ){
          DataSet* dataSetPtr = &_fitSampleSet_.getDataSetList().at(iDataSet);
          DialSet* dialSet = par.findDialSet(dataSetPtr->getName());
          if( dialSet == nullptr or dialSet->getDialList().empty() ) continue;

          const std::vector<std::string>* activeLeavesList = &dataSetPtr->getMcActiveLeafNameList();


          std::vector<int> varIndexFormulaList;
          if(dialSet->getApplyConditionFormula() != nullptr){
            varIndexFormulaList.resize(dialSet->getApplyConditionFormula()->GetNpar(), -1);
            for( int iFPar = 0 ; iFPar < dialSet->getApplyConditionFormula()->GetNpar() ; iFPar++ ){
              varIndexFormulaList.at(iFPar) = GenericToolbox::findElementIndex(
                dialSet->getApplyConditionFormula()->GetParName(iFPar),
                *activeLeavesList
                );
              LogThrowIf(varIndexFormulaList.at(iFPar) == -1,
                         "Formula parameter \"" << dialSet->getApplyConditionFormula()->GetParName(iFPar)
                         << "\" not found in available leaves of MC dataset \""
                         << dataSetPtr->getName() << "\": "
                         << GenericToolbox::parseVectorAsString(*activeLeavesList)
                         );
            }
          }

          // CAVEAT: assuming every dial share the same ApplyConditionBin variable names!!! -> May need to change this in the future
          const auto &firstDial = dialSet->getDialList().at(0);
          std::vector<int> varIndexList(firstDial->getApplyConditionBin().getVariableNameList().size(), 0);


          for (size_t iVar = 0; iVar < firstDial->getApplyConditionBin().getVariableNameList().size(); iVar++) {
            varIndexList.at(iVar) = GenericToolbox::findElementIndex(
              firstDial->getApplyConditionBin().getVariableNameList().at(iVar),
              *activeLeavesList
              );
            LogThrowIf(varIndexList.at(iVar) == -1,"Could find \""
            << firstDial->getApplyConditionBin().getVariableNameList().at(iVar)
            << "\" in the active leaves list of dataset \"" << dataSetPtr->getName()
            << ": " << GenericToolbox::parseVectorAsString(*activeLeavesList));
          }

          for( auto& sample : _fitSampleSet_.getFitSampleList() ){
            if( not GenericToolbox::doesElementIsInVector(iDataSet, sample.getMcContainer().dataSetIndexList ) ){
              continue;
            }

            std::stringstream ss;
            ss << LogWarning.getPrefixString() << parSet.getName() << "/" << par.getTitle() << " -> " << sample.getName();
            //          ss << "Indexing event dials: " << parSet.getName() << "/" << par.getTitle() << " -> " << sample.getName();

            int nEvents = int(sample.getMcContainer().eventNbList.at(iDataSet));
            for (int iEvent = int(sample.getMcContainer().eventOffSetList.at(iDataSet)); iEvent < nEvents; iEvent++) {

              if (iEvent % GlobalVariables::getNbThreads() != iThread_) {
                continue;
              }
              if (iThread_ == GlobalVariables::getNbThreads() - 1) {
                GenericToolbox::displayProgressBar(iEvent, nEvents, ss.str());
              }

              evPtr = &sample.getMcContainer().eventList.at(iEvent);
              if (evPtr->getDialCachePtr()->at(&parSet).at(iPar) != nullptr) {
                // already set
                continue;
              }

              //            if( evPtr->getEntryIndex() == 6 and par.getName() == "Q2_norm_1" ){
              //              evPtr->print();
              //              if (dialSet->getApplyConditionFormula() != nullptr
              //              and evPtr->evalFormula(dialSet->getApplyConditionFormula(), &varIndexFormulaList) == 0
              //              ) {
              //                LogError << GET_VAR_NAME_VALUE(dialSet->getApplyConditionFormula()->GetExpFormula()) << std::endl;
              ////                LogError << GET_VAR_NAME_VALUE(evPtr->fetchValue<Float_t>("q2_true")) << std::endl;
              //                LogError << "LOL: " << GET_VAR_NAME_VALUE(evPtr->getVarAsDouble("q2_true")) << std::endl;
              //                continue; // SKIP
              //              }
              //            }
              if (dialSet->getApplyConditionFormula() != nullptr
              and evPtr->evalFormula(dialSet->getApplyConditionFormula(), &varIndexFormulaList) == 0
              ) {
                continue; // SKIP
              }

              for (const auto &dial : dialSet->getDialList()) {
                bool isInBin = true;
                for (size_t iVar = 0; iVar < varIndexList.size(); iVar++) {
                  if( not dial->getApplyConditionBin().isBetweenEdges(iVar, evPtr->getVarAsDouble(varIndexList.at(iVar))) ){
                    isInBin = false;
                    break; // next dial
                  }
                }
                if (isInBin) {
                  evPtr->getDialCachePtr()->at(&parSet).at(iPar) = dial.get();
                  //LogTrace << *evPtr << std::endl;
                  break; // found
                }
              } // dial
            }

            if (iThread_ == GlobalVariables::getNbThreads() - 1) {
              GenericToolbox::displayProgressBar(nEvents, nEvents, ss.str());
            }
          }
        }
      } // par
    } // parSet
  }

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
        rfHistBuffer = _responseFunctionsSamplesMcHistogram_[&sample].at(iPar).get();

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
