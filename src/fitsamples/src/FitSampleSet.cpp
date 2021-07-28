//
// Created by Nadrino on 22/07/2021.
//

#include <TROOT.h>
#include "json.hpp"

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

#include "JsonUtils.h"
#include "GlobalVariables.h"
#include "FitSampleSet.h"


LoggerInit([](){
  Logger::setUserHeaderStr("[FitSampleSet]");
})

FitSampleSet::FitSampleSet() { this->reset(); }
FitSampleSet::~FitSampleSet() { this->reset(); }

void FitSampleSet::reset() {

  _config_.clear();
  _isInitialized_ = false;
}

void FitSampleSet::setConfig(const nlohmann::json &config) {
  _config_ = config;
  while( _config_.is_string() ){
    LogWarning << "Forwarding " << __CLASS_NAME__ << " config: \"" << _config_.get<std::string>() << "\"" << std::endl;
    _config_ = JsonUtils::readConfigFile(_config_.get<std::string>());
  }
}

void FitSampleSet::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  LogAssert(not _config_.empty(), "_config_ is not set." << std::endl);

  _dataEventType_ = DataEventTypeEnumNamespace::toEnum(
    JsonUtils::fetchValue<std::string>(_config_, "dataEventType"), true
    );

  LogDebug << "Loading datasets..." << std::endl;
  auto dataSetListConfig = JsonUtils::fetchValue(_config_, "dataSetList", nlohmann::json());
  LogAssert(not dataSetListConfig.empty(), "No dataSet specified." << std::endl);
  for( const auto& dataSetConfig : dataSetListConfig ){
    _dataSetList_.emplace_back();
    _dataSetList_.back().setConfig(dataSetConfig);
    _dataSetList_.back().initialize();
  }

  LogDebug << "Defining samples..." << std::endl;
  auto fitSampleListConfig = JsonUtils::fetchValue(_config_, "fitSampleList", nlohmann::json());
  for( const auto& fitSampleConfig: fitSampleListConfig ){
    _fitSampleList_.emplace_back();
    _fitSampleList_.back().setConfig(fitSampleConfig);
    _fitSampleList_.back().initialize();
  }

  _isInitialized_ = true;
}

std::vector<FitSample> &FitSampleSet::getFitSampleList() {
  return _fitSampleList_;
}
std::vector<DataSet> &FitSampleSet::getDataSetList() {
  return _dataSetList_;
}

void FitSampleSet::loadPhysicsEvents() {
  LogWarning << __METHOD_NAME__ << std::endl;

  LogThrowIf(_dataEventType_ == DataEventType::Unset, "dataEventType not set.");
  LogInfo << "Data events type is set to: " << DataEventTypeEnumNamespace::toString(_dataEventType_) << std::endl;

  for( auto& dataSet : _dataSetList_ ){

    std::vector<FitSample*> samplesToFillList;
    std::vector<TTreeFormula*> sampleCutFormulaList;
    std::vector<std::string> samplesNames;

    if( not dataSet.isEnabled() ){
      continue;
    }

    LogThrowIf(dataSet.getMcChain() == nullptr, "No MC files are available for dataset: " << dataSet.getName());
    LogThrowIf(dataSet.getDataChain() == nullptr and _dataEventType_ == DataEventType::DataFiles,
               "Can't define sample \"" << dataSet.getName() << "\" while in non-Asimov-like fit and no Data files are available" );

    for( auto& sample : _fitSampleList_ ){
      if( not sample.isEnabled() ) continue;
      if( sample.isDataSetValid(dataSet.getName()) ){
        samplesToFillList.emplace_back(&sample);
        samplesNames.emplace_back(sample.getName());
      }
    }
    if( samplesToFillList.empty() ){
      LogAlert << "No sample is set to use dataset \"" << dataSet.getName() << "\"" << std::endl;
    }
    LogInfo << "Dataset \"" << dataSet.getName() << "\" will populate samples: " << GenericToolbox::parseVectorAsString(samplesNames) << std::endl;

    for( bool isData : {false, true} ){
      TChain* chainPtr{nullptr};

      if( isData and _dataEventType_ == DataEventType::Asimov ){ continue; }

      if( isData ){
        LogInfo << "Reading data files..." << std::endl;
        chainPtr = dataSet.getDataChain().get();
      }
      else{
        LogInfo << "Reading MC files..." << std::endl;
        chainPtr = dataSet.getMcChain().get();
      }

      if( chainPtr == nullptr or chainPtr->GetEntries() == 0 ){
        continue;
      }

      LogInfo << "Performing event selection of samples with " << (isData?"data": "mc") << " files..." << std::endl;
      for( auto& sample : samplesToFillList ){
        sampleCutFormulaList.emplace_back(
          new TTreeFormula(
            sample->getName().c_str(),
            sample->getSelectionCutsStr().c_str(),
            chainPtr
          )
        );
        LogThrowIf(sampleCutFormulaList.back()->GetNdim() == 0,
                   "\"" << sample->getSelectionCutsStr() << "\" could not be parsed by the TChain");

        chainPtr->SetNotify(sampleCutFormulaList.back()); // TODO: to be replaced -> may not work when multiple files are loaded
//        sampleCutFormulaList.back()->Notify();
      }

      Long64_t nEvents = chainPtr->GetEntries();
      // for each event, which sample is active?
      std::vector<std::vector<bool>> eventIsInSamplesList(nEvents, std::vector<bool>(samplesToFillList.size(), false));
      std::vector<size_t> sampleNbOfEvents(samplesToFillList.size(), 0);
      std::string progressTitle = LogWarning.getPrefixString() + "Performing event selection";
      for( Long64_t iEvent = 0 ; iEvent < nEvents ; iEvent++ ){
        GenericToolbox::displayProgressBar(iEvent, nEvents, progressTitle);
        chainPtr->GetEntry(iEvent);
        for( size_t iSample = 0 ; iSample < sampleCutFormulaList.size() ; iSample++ ){
          for(int jInstance = 0; jInstance < sampleCutFormulaList.at(iSample)->GetNdata(); jInstance++){
            if( sampleCutFormulaList.at(iSample)->EvalInstance(jInstance) != 0 ){
              eventIsInSamplesList.at(iEvent).at(iSample) = true;
              sampleNbOfEvents.at(iSample)++;
            }
          }

        } // iSample
      } // iEvent

      LogInfo << "Claiming memory for additional events in samples: "
      << GenericToolbox::parseVectorAsString(sampleNbOfEvents) << std::endl;


      // If we don't do this, the events will be resized while being in multithread
      // Because std::vector is insuring continuous memory allocation, a resize sometimes
      // lead to the full moving of a vector memory. This is not thread safe, so better ensure
      // the vector won't have to do this.
      PhysicsEvent eventBuf;
      eventBuf.setLeafNameListPtr(&dataSet.getEnabledLeafNameList());
      eventBuf.hookToTree(chainPtr);
      chainPtr->GetEntry(0);
      // Now the eventBuffer has the right size in memory

      std::vector<size_t> sampleIndexOffsetList(samplesToFillList.size(), 0);
      std::vector< std::vector<PhysicsEvent>* > sampleEventListPtrToFill(samplesToFillList.size(), nullptr);
      for( size_t iSample = 0 ; iSample < sampleNbOfEvents.size() ; iSample++ ){
        if( isData ){
          sampleEventListPtrToFill.at(iSample) = &samplesToFillList.at(iSample)->getDataEventList();
        }
        else{
          sampleEventListPtrToFill.at(iSample) = &samplesToFillList.at(iSample)->getMcEventList();
        }
        sampleIndexOffsetList.at(iSample) = sampleEventListPtrToFill.at(iSample)->size();
        sampleEventListPtrToFill.at(iSample)->resize(sampleIndexOffsetList.at(iSample) + sampleNbOfEvents.at(iSample), eventBuf);
      }

      // Fill function
      ROOT::EnableImplicitMT();
      auto fillFunction = [&](int iThread_){

        TChain* threadChain;
        threadChain = (TChain*) chainPtr->Clone();

        Long64_t nEvents = threadChain->GetEntries();
        PhysicsEvent eventBuffer;
        eventBuffer.setLeafNameListPtr(&dataSet.getEnabledLeafNameList());
        eventBuffer.hookToTree(threadChain);
        GenericToolbox::disableUnhookedBranches(threadChain);

        auto threadSampleIndexOffsetList = sampleIndexOffsetList;

        std::string progressTitle = LogWarning.getPrefixString() + "Reading selected events";
        for( Long64_t iEvent = 0 ; iEvent < nEvents ; iEvent++ ){
          if( iEvent % GlobalVariables::getNbThreads() != iThread_ ){ continue; }
          if( iThread_ == 0 ) GenericToolbox::displayProgressBar(iEvent, nEvents, progressTitle);

          bool skipEvent = true;
          for( bool isInSample : eventIsInSamplesList.at(iEvent) ){
            if( isInSample ){
              skipEvent = false;
              break;
            }
          }
          if( skipEvent ) continue;

          threadChain->GetEntry(iEvent);

          for( size_t iSample = 0 ; iSample < samplesToFillList.size() ; iSample++ ){
            if( eventIsInSamplesList.at(iEvent).at(iSample) ){
              sampleEventListPtrToFill.at(iSample)->at(threadSampleIndexOffsetList.at(iSample)++) = PhysicsEvent(eventBuffer); // copy
            }
          }
        }
        if( iThread_ == 0 ) GenericToolbox::displayProgressBar(nEvents, nEvents, progressTitle);

        delete threadChain;
      };

      LogInfo << "Copying selected events to RAM..." << std::endl;
      GlobalVariables::getParallelWorker().addJob(__METHOD_NAME__, fillFunction);
      GlobalVariables::getParallelWorker().runJob(__METHOD_NAME__);
      GlobalVariables::getParallelWorker().removeJob(__METHOD_NAME__);

      LogInfo << "Events have been loaded for " << ( isData ? "data": "mc" )
      << "with dataset: " << dataSet.getName() << std::endl;

    } // isData

  } // data Set

  if( _dataEventType_ == DataEventType::Asimov ){
    LogWarning << "Asimov data selected: copying MC events..." << std::endl;
    for( auto& sample : _fitSampleList_ ){
      LogInfo << "Copying MC events in sample \"" << sample.getName() << "\"" << std::endl;
      auto& dataEventList = sample.getDataEventList();
      LogThrowIf(not dataEventList.empty(), "Can't fill Asimov data, dataEventList is not empty.");

      auto& mcEventList = sample.getMcEventList();
      dataEventList.resize(mcEventList.size());
      for( size_t iEvent = 0 ; iEvent < dataEventList.size() ; iEvent++ ){
        dataEventList[iEvent] = mcEventList[iEvent];
      }
    }
  }

  for( auto& sample : _fitSampleList_ ){
    LogInfo << "Total events loaded in \"" << sample.getName() << "\": "
    << sample.getMcEventList().size() << "(mc) / "
    << sample.getDataEventList().size() << "(data)" << std::endl;
  }

}

