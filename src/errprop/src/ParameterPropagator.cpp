//
// Created by Adrien BLANCHET on 11/06/2021.
//

#include <AnaTreeMC.hh>
#include "vector"

#include "GenericToolbox.h"

#include "JsonUtils.h"
#include "ParameterPropagator.h"
#include "GlobalVariables.h"
#include "Dial.h"
#include "FitParameterSet.h"

#include "NormalizationDial.h"
#include "SplineDial.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[ParameterPropagator]");
})

ParameterPropagator::ParameterPropagator() { this->reset(); }
ParameterPropagator::~ParameterPropagator() { this->reset(); }

void ParameterPropagator::reset() {
  _isInitialized_ = false;
  _parameterSetsList_.clear();
  _nbThreads_ = 1;

  _stopThreads_ = true;
  for( auto& thread : _threadsList_ ){
    thread.get();
  }
  _threadTriggersList_.clear();
  _threadsList_.clear();
  _stopThreads_ = false;
}

void ParameterPropagator::setParameterSetConfig(const json &parameterSetConfig) {
  _parameterSetsConfig_ = parameterSetConfig;
  while( _parameterSetsConfig_.is_string() ){
    // forward json definition in external files
    LogDebug << "Forwarding config with file: " << _parameterSetsConfig_.get<std::string>() << std::endl;
    _parameterSetsConfig_ = JsonUtils::readConfigFile(_parameterSetsConfig_.get<std::string>());
  }
}
void ParameterPropagator::setSamplesConfig(const json &samplesConfig) {
  _samplesConfig_ = samplesConfig;
  while( _samplesConfig_.is_string() ){
    // forward json definition in external files
    LogDebug << "Forwarding config with file: " << _samplesConfig_.get<std::string>() << std::endl;
    _samplesConfig_ = JsonUtils::readConfigFile(_samplesConfig_.get<std::string>());
  }
}
void ParameterPropagator::setDataTree(TTree *dataTree_) {
  dataTree = dataTree_;
}
void ParameterPropagator::setMcFilePath(const std::string &mcFilePath) {
  mc_file_path = mcFilePath;
}

void ParameterPropagator::initialize() {
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

  AnaTreeMC selected_events_AnaTreeMC(mc_file_path, "selectedEvents");
  LogInfo << "Reading and collecting events..." << std::endl;
  std::vector<SignalDef> buf;
  std::vector<AnaSample*> samplePtrList;
  for( auto& sample : _samplesList_ ) samplePtrList.emplace_back(&sample);
  selected_events_AnaTreeMC.GetEvents(samplePtrList, buf, false);

  LogTrace << "Other..." << std::endl;

  initializeThreads();
  initializeCaches();

  fillEventDialCaches();
  propagateParametersOnSamples();

  _isInitialized_ = true;
}

const std::vector<FitParameterSet> &ParameterPropagator::getParameterSetsList() const {
  return _parameterSetsList_;
}

void ParameterPropagator::propagateParametersOnSamples() {
  LogDebug << __METHOD_NAME__ << std::endl;

  LogTrace << "Reweight..." << std::endl;

  GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(1);

  // dispatch the job on each thread
  for( int iThread = 0 ; iThread < _nbThreads_-1 ; iThread++ ){
    _threadTriggersList_.at(iThread).propagateOnSamples = true; // triggering the workers
  }
  // last thread is always this one
  this->propagateParametersOnSamples(_nbThreads_-1);

  for( int iThread = 0 ; iThread < _nbThreads_-1 ; iThread++ ){
    while( _threadTriggersList_.at(iThread).propagateOnSamples ){
      // wait
    }
  }

  LogTrace << "took: " << GenericToolbox::getElapsedTimeSinceLastCallStr(1) << std::endl;
}


// Protected
void ParameterPropagator::initializeThreads() {

  _nbThreads_ = GlobalVariables::getNbThreads();
  if( _nbThreads_ == 1 ){
    return;
  }

  _threadTriggersList_.resize(_nbThreads_-1);

  std::function<void(int)> asyncLoop = [this](int iThread_){
    while(not _stopThreads_){
      // Pending state loop

      if     ( _threadTriggersList_.at(iThread_).propagateOnSamples ){
        this->propagateParametersOnSamples(iThread_);
        _threadTriggersList_.at(iThread_).propagateOnSamples = false; // toggle off the trigger
      }
      else if( _threadTriggersList_.at(iThread_).fillDialCaches ){
        this->fillEventDialCaches(iThread_);
        _threadTriggersList_.at(iThread_).fillDialCaches = false;
      }

      // Add other jobs there
    }
    _propagatorMutex_.lock();
    LogDebug << "Thread " << iThread_ << " will end now." << std::endl;
    _propagatorMutex_.unlock();
  };

  for( int iThread = 0 ; iThread < _nbThreads_-1 ; iThread++ ){
    _threadsList_.emplace_back(
      std::async( std::launch::async, std::bind(asyncLoop, iThread) )
    );
  }

}
void ParameterPropagator::initializeCaches() {
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
void ParameterPropagator::fillEventDialCaches(){
  LogInfo << __METHOD_NAME__ << std::endl;

  // dispatch the job on each thread
  for( int iThread = 0 ; iThread < _nbThreads_-1 ; iThread++ ){
    _threadTriggersList_.at(iThread).fillDialCaches = true; // triggering the workers
  }
  // first thread is always this one
  this->fillEventDialCaches(_nbThreads_-1);

  for( int iThread = 0 ; iThread < _nbThreads_-1 ; iThread++ ){
    while ( _threadTriggersList_.at(iThread).fillDialCaches ){
      // wait triggering the workers
    }
  }

}

void ParameterPropagator::fillEventDialCaches(int iThread_){

  DialSet* currentDialSetPtr;
  AnaEvent* eventPtr;

  for( auto& sample : _samplesList_ ){

    int nEvents = sample.GetN();
    std::stringstream ss;
    ss << "Filling cache: " << sample.GetName();
    GenericToolbox::resetLastDisplayedValue();
    for( int iEvent = 0 ; iEvent < nEvents ; iEvent++ ){
      if( iEvent % _nbThreads_ != iThread_ ){
        continue;
      }
      if( iThread_ == 0 ){
        GenericToolbox::displayProgressBar(iEvent, nEvents, ss.str());
      }

      eventPtr = sample.GetEvent(iEvent);

      for( auto& parSet : _parameterSetsList_ ){
        for( size_t iPar = 0 ; iPar < parSet.getNbParameters() ; iPar++ ){

          currentDialSetPtr = parSet.getFitParameter(iPar).findDialSet(sample.GetDetector());
          if( currentDialSetPtr == nullptr ){
            // This parameter does not apply on this sample
            continue;
          }

          if( currentDialSetPtr->getApplyConditionFormula() != nullptr ){

            if( eventPtr->evalFormula(currentDialSetPtr->getApplyConditionFormula()) == 0 ){
              continue; // SKIP
            }
            else{ /* OK! */ }

          }

          for( auto& dial : currentDialSetPtr->getDialList() ){
            if( eventPtr->isInBin( dial->getApplyConditionBin() ) ){
              eventPtr->getDialCachePtr()->at(&parSet).at(iPar) = dial.get();
              break;
            }
          } // dial

        }// par

      } // parSet



    } // event

  } // sample

}
void ParameterPropagator::propagateParametersOnSamples(int iThread_) {
  AnaEvent* eventPtr;
  double weight;

  for( auto& sample : _samplesList_ ){
    int nEvents = sample.GetN();
    for( int iEvent = 0 ; iEvent < nEvents ; iEvent++ ){

      if( iEvent % _nbThreads_ != iThread_ ){
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
          weight = dialPtr->evalResponse( parSetDialCache.first->getFitParameter(iPar).getParameterValue() );

        }

        // TODO: check if weight cap

        eventPtr->AddEvWght(weight);

      } // parSetCache

    } // event
  } // sample

}

