//
// Created by Adrien BLANCHET on 11/06/2021.
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
    or jobName == "Propagator::propagateParametersOnEvents"
    or jobName == "Propagator::fillSampleHistograms"
      ){
      jobNameRemoveList.emplace_back(jobName);
    }
  }
  for( const auto& jobName : jobNameRemoveList ){
    GlobalVariables::getParallelWorker().removeJob(jobName);
  }

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


  LogTrace << "Other..." << std::endl;
  auto plotGeneratorConfig = JsonUtils::fetchValue<json>(_config_, "plotGeneratorConfig");
  if( plotGeneratorConfig.is_string() ) parameterSetListConfig = JsonUtils::readConfigFile(plotGeneratorConfig.get<std::string>());
  _plotGenerator_.setConfig(plotGeneratorConfig);
  _plotGenerator_.setSampleListPtr( &_samplesList_ );
  _plotGenerator_.initialize();

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


  propagateParametersOnEvents();

  for( auto& sample : _samplesList_ ){
    sample.FillEventHist(
      DataType::kAsimov,
      false
    );
  }

  if( JsonUtils::fetchValue<json>(_config_, "throwParameters", false) ){
    for( auto& parSet : _parameterSetsList_ ){
      for( auto& par : parSet.getParameterList() ){
        par.setParameterValue( par.getPriorValue() );
      }
    }
  }

  _isInitialized_ = true;
  LogTrace << "OK LEAVING INIT" << std::endl;
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


void Propagator::propagateParametersOnEvents() {
  if( _showTimeStats_ ) GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GlobalVariables::getParallelWorker().runJob("Propagator::propagateParametersOnEvents");
  if( _showTimeStats_ ) LogDebug << __METHOD_NAME__ << " took: " << GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__) << std::endl;
}
void Propagator::fillSampleHistograms(){
  if( _showTimeStats_ ) GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GlobalVariables::getParallelWorker().runJob("Propagator::fillSampleHistograms");
  if( _showTimeStats_ ) LogDebug << __METHOD_NAME__ << " took: " << GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__) << std::endl;
}


// Protected
void Propagator::initializeThreads() {

  std::function<void(int)> fillEventDialCacheFct = [this](int iThread){
    this->fillEventDialCaches(iThread);
  };
  GlobalVariables::getParallelWorker().addJob("Propagator::fillEventDialCaches", fillEventDialCacheFct);

  std::function<void(int)> propagateParametersOnSamplesFct = [this](int iThread){
    this->propagateParametersOnSamples(iThread);
  };
  GlobalVariables::getParallelWorker().addJob("Propagator::propagateParametersOnEvents", propagateParametersOnSamplesFct);

  std::function<void(int)> fillSampleHistogramsFct = [this](int iThread){
    for( auto& sample : _samplesList_ ){
      sample.FillMcHistograms(iThread);
    }
  };
  std::function<void()> fillSampleHistogramsPostParallelFct = [this](){
    for( auto& sample : _samplesList_ ){
      sample.MergeMcHistogramsThread();
    }
  };
  GlobalVariables::getParallelWorker().addJob("Propagator::fillSampleHistograms", fillSampleHistogramsFct);
  GlobalVariables::getParallelWorker().setPostParallelJob("Propagator::fillSampleHistograms", fillSampleHistogramsPostParallelFct);

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

}
void Propagator::fillEventDialCaches(){
  LogInfo << __METHOD_NAME__ << std::endl;
  GlobalVariables::getParallelWorker().runJob("Propagator::fillEventDialCaches");
}

void Propagator::fillEventDialCaches(int iThread_){

  DialSet* parameterDialSetPtr;
  AnaEvent* eventPtr;


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
void Propagator::propagateParametersOnSamples(int iThread_) {
  AnaEvent* eventPtr;
  double weight;

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
