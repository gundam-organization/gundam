//
// Created by Adrien BLANCHET on 11/06/2021.
//

#include <AnaTreeMC.hh>
#include "vector"

#include "GenericToolbox.h"
#include "GenericToolboxRootExt.h"

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
  for( const auto& jobName : GlobalVariables::getThreadPool().getJobNameList() ){
    if(jobName == "Propagator::fillEventDialCaches"
    or jobName == "Propagator::propagateParametersOnSamples"
    or jobName == "Propagator::fillSampleHistograms"
      ){
      jobNameRemoveList.emplace_back(jobName);
    }
  }
  for( const auto& jobName : jobNameRemoveList ){
    GlobalVariables::getThreadPool().removeJob(jobName);
  }

}


void Propagator::setSaveDir(TDirectory *saveDir) {
  _saveDir_ = saveDir;
}
void Propagator::setParameterSetConfig(const json &parameterSetConfig) {
  _parameterSetsConfig_ = parameterSetConfig;
  while( _parameterSetsConfig_.is_string() ){
    // forward json definition in external files
    LogDebug << "Forwarding config with file: " << _parameterSetsConfig_.get<std::string>() << std::endl;
    _parameterSetsConfig_ = JsonUtils::readConfigFile(_parameterSetsConfig_.get<std::string>());
  }
}
void Propagator::setSamplesConfig(const json &samplesConfig) {
  _samplesConfig_ = samplesConfig;
  while( _samplesConfig_.is_string() ){
    // forward json definition in external files
    LogDebug << "Forwarding config with file: " << _samplesConfig_.get<std::string>() << std::endl;
    _samplesConfig_ = JsonUtils::readConfigFile(_samplesConfig_.get<std::string>());
  }
}
void Propagator::setSamplePlotGeneratorConfig(const json &samplePlotGeneratorConfig) {
  _samplePlotGeneratorConfig_ = samplePlotGeneratorConfig;
  while( _samplePlotGeneratorConfig_.is_string() ){
    // forward json definition in external files
    LogDebug << "Forwarding config with file: " << _samplesConfig_.get<std::string>() << std::endl;
    _samplePlotGeneratorConfig_ = JsonUtils::readConfigFile(_samplePlotGeneratorConfig_.get<std::string>());
  }
}
void Propagator::setDataTree(TTree *dataTree_) {
  dataTree = dataTree_;
}
void Propagator::setMcFilePath(const std::string &mcFilePath) {
  mc_file_path = mcFilePath;
}

void Propagator::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  LogTrace << "Parameters..." << std::endl;
  for( const auto& parameterSetConfig : _parameterSetsConfig_ ){
    _parameterSetsList_.emplace_back();
    _parameterSetsList_.back().setJsonConfig(parameterSetConfig);
    _parameterSetsList_.back().initialize();
    LogInfo << _parameterSetsList_.back().getSummary() << std::endl;
  }

  LogTrace << "Samples..." << std::endl;
  for( const auto& sampleConfig : _samplesConfig_ ){
    if( JsonUtils::fetchValue(sampleConfig, "isEnabled", true) ){
      _samplesList_.emplace_back();
      _samplesList_.back().setupWithJsonConfig(sampleConfig);
      _samplesList_.back().setDataTree(dataTree);
      _samplesList_.back().Initialize();
    }
  }

  AnaTreeMC* selected_events_AnaTreeMC = new AnaTreeMC(mc_file_path, "selectedEvents"); // trouble while deleting... > need to check
  LogInfo << "Reading and collecting events..." << std::endl;
  std::vector<SignalDef> buf;
  std::vector<AnaSample*> samplePtrList;
  for( auto& sample : _samplesList_ ) samplePtrList.emplace_back(&sample);
  selected_events_AnaTreeMC->GetEvents(samplePtrList, buf, false);


  LogTrace << "Other..." << std::endl;
  _plotGenerator_.setConfig(_samplePlotGeneratorConfig_);
  _plotGenerator_.setSampleListPtr( &_samplesList_ );
  _plotGenerator_.initialize();

  initializeThreads();
  initializeCaches();

  fillEventDialCaches();
  propagateParametersOnSamples();

  for( auto& sample : _samplesList_ ){
    sample.FillEventHist(
      DataType::kAsimov,
      false
    );
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


void Propagator::propagateParametersOnSamples() {
  LogDebug << __METHOD_NAME__ << std::endl;

  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(1);
  GlobalVariables::getThreadPool().runJob("Propagator::propagateParametersOnSamples");

  LogTrace << "Reweight took: " << GenericToolbox::getElapsedTimeSinceLastCallStr(1) << std::endl;
}
void Propagator::fillSampleHistograms(){
  LogDebug << __METHOD_NAME__ << std::endl;

  GenericToolbox::getElapsedTimeSinceLastCallStr(1);
  GlobalVariables::getThreadPool().runJob("Propagator::fillSampleHistograms");
  LogTrace << "Histogram fill took: " << GenericToolbox::getElapsedTimeSinceLastCallStr(1) << std::endl;
}


// Protected
void Propagator::initializeThreads() {

  std::function<void(int)> fillEventDialCacheFct = [this](int iThread){
    this->fillEventDialCaches(iThread);
  };
  GlobalVariables::getThreadPool().addJob("Propagator::fillEventDialCaches", fillEventDialCacheFct);

  std::function<void(int)> propagateParametersOnSamplesFct = [this](int iThread){
    this->propagateParametersOnSamples(iThread);
  };
  GlobalVariables::getThreadPool().addJob("Propagator::propagateParametersOnSamples", propagateParametersOnSamplesFct);

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
  GlobalVariables::getThreadPool().addJob("Propagator::fillSampleHistograms", fillSampleHistogramsFct);
  GlobalVariables::getThreadPool().setPostParallelJob("Propagator::fillSampleHistograms", fillSampleHistogramsPostParallelFct);

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
  GlobalVariables::getThreadPool().runJob("Propagator::fillEventDialCaches");
}

void Propagator::fillEventDialCaches(int iThread_){

  DialSet* parameterDialSetPtr;
  AnaEvent* eventPtr;

  for( auto& sample : _samplesList_ ){

    int nEvents = sample.GetN();
    std::stringstream ss;
    ss << "Filling dial cache for sample: \"" << sample.GetName() << "\"";
    for( int iEvent = 0 ; iEvent < nEvents ; iEvent++ ){
      if( iEvent % GlobalVariables::getNbThreads() != iThread_ ){
        continue;
      }

      if( iThread_ == GlobalVariables::getNbThreads()-1 ){
        GenericToolbox::displayProgressBar(iEvent, nEvents, ss.str());
      }

      eventPtr = sample.GetEvent(iEvent);

      for( auto& parSetPair : *eventPtr->getDialCachePtr() ){
        for( size_t iPar = 0 ; iPar < parSetPair.first->getNbParameters() ; iPar++ ){

          parameterDialSetPtr = parSetPair.first->getFitParameter(iPar).findDialSet(sample.GetDetector());

          // If a formula is defined
          if( parameterDialSetPtr->getApplyConditionFormula() != nullptr
              and eventPtr->evalFormula(parameterDialSetPtr->getApplyConditionFormula()) == 0
          ){
            continue; // SKIP
          }

          for( const auto& dial : parameterDialSetPtr->getDialList() ){
            if( eventPtr->isInBin( dial->getApplyConditionBin() ) ){
              parSetPair.second.at(iPar) = dial.get();
              break; // ok, next parameter
            }
          } // dial

        }
      }

    } // event

  } // sample

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
            LogError << GET_VAR_NAME_VALUE(iPar) << std::endl;
            LogError << GET_VAR_NAME_VALUE(weight) << std::endl;
            throw std::runtime_error("<0 weight");
          }

        }

        eventPtr->AddEvWght(weight);

      } // parSetCache

    } // event
  } // sample

}
