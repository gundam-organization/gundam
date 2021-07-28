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

DataEventType FitSampleSet::getDataEventType() const {
  return _dataEventType_;
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

  int dataSetIndex = -1;
  for( auto& dataSet : _dataSetList_ ){
    dataSetIndex++;

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

    LogInfo << "Fetching mandatory leaves..." << std::endl;
    for( size_t iSample = 0 ; iSample < samplesToFillList.size() ; iSample++ ){
      // Fit phase space
      for( const auto& bin : samplesToFillList[iSample]->getBinning().getBinsList() ){
        for( const auto& var : bin.getVariableNameList() ){
          dataSet.addRequestedMandatoryLeafName(var);
        }
      }
//        // Cuts // ACTUALLY NOT NECESSARY SINCE THE SELECTION IS DONE INDEPENDENTLY
//        for( int iPar = 0 ; iPar < sampleCutFormulaList.at(iSample)->GetNpar() ; iPar++ ){
//          dataSet.addRequestedMandatoryLeafName(sampleCutFormulaList.at(iSample)->GetParName(iPar));
//        }
    }

    LogInfo << "List of requested leaves: " << GenericToolbox::parseVectorAsString(dataSet.getRequestedLeafNameList()) << std::endl;
    LogInfo << "List of mandatory leaves: " << GenericToolbox::parseVectorAsString(dataSet.getRequestedMandatoryLeafNameList()) << std::endl;


    for( bool isData : {false, true} ){
      TChain* chainPtr{nullptr};
      std::vector<std::string>* activeLeafNameListPtr;

      if( isData and _dataEventType_ == DataEventType::Asimov ){ continue; }

      if( isData ){
        LogInfo << "Reading data files..." << std::endl;
        chainPtr = dataSet.getDataChain().get();
        activeLeafNameListPtr = &dataSet.getDataActiveLeafNameList();
      }
      else{
        LogInfo << "Reading MC files..." << std::endl;
        chainPtr = dataSet.getMcChain().get();
        activeLeafNameListPtr = &dataSet.getMcActiveLeafNameList();
      }

      if( chainPtr == nullptr or chainPtr->GetEntries() == 0 ){
        continue;
      }

      LogInfo << "Checking the availability of requested leaves..." << std::endl;
      for( auto& requestedLeaf : dataSet.getRequestedLeafNameList() ){
        if( not isData or GenericToolbox::doesElementIsInVector(requestedLeaf, dataSet.getRequestedMandatoryLeafNameList()) ){
          LogThrowIf(chainPtr->GetLeaf(requestedLeaf.c_str()) == nullptr,
                     "Could not find leaf \"" << requestedLeaf << "\" in TChain");
        }

        if( chainPtr->GetLeaf(requestedLeaf.c_str()) != nullptr ){
          activeLeafNameListPtr->emplace_back(requestedLeaf);
        }
      }
      LogInfo << "List of leaves which will be loaded in RAM: " << GenericToolbox::parseVectorAsString(*activeLeafNameListPtr) << std::endl;

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
            } // if passes the cut
          } // jInstance
        } // iSample
      } // iEvent

      LogInfo << "Claiming memory for additional events in samples: "
      << GenericToolbox::parseVectorAsString(sampleNbOfEvents) << std::endl;


      // The following lines are necessary since the events might get resized while being in multithread
      // Because std::vector is insuring continuous memory allocation, a resize sometimes
      // lead to the full moving of a vector memory. This is not thread safe, so better ensure
      // the vector won't have to do this by allocating the right event size.
      PhysicsEvent eventBuf;
      eventBuf.setLeafNameListPtr(activeLeafNameListPtr);
      eventBuf.hookToTree(chainPtr, not isData);
      eventBuf.setDataSetIndex(dataSetIndex);
      chainPtr->GetEntry(0);
      // Now the eventBuffer has the right size in memory

      std::vector<size_t> sampleIndexOffsetList(samplesToFillList.size(), 0);
      std::vector< std::vector<PhysicsEvent>* > sampleEventListPtrToFill(samplesToFillList.size(), nullptr);
      for( size_t iSample = 0 ; iSample < sampleNbOfEvents.size() ; iSample++ ){
        if( isData ){
          sampleEventListPtrToFill.at(iSample) = &samplesToFillList.at(iSample)->getDataEventList();
          sampleIndexOffsetList.at(iSample) = sampleEventListPtrToFill.at(iSample)->size();
          samplesToFillList.at(iSample)->reserveMemoryForDataEvents(sampleNbOfEvents.at(iSample), dataSetIndex, eventBuf);
        }
        else{
          sampleEventListPtrToFill.at(iSample) = &samplesToFillList.at(iSample)->getMcEventList();
          sampleIndexOffsetList.at(iSample) = sampleEventListPtrToFill.at(iSample)->size();
          samplesToFillList.at(iSample)->reserveMemoryForMcEvents(sampleNbOfEvents.at(iSample), dataSetIndex, eventBuf);
        }
      }

      // Fill function
      ROOT::EnableImplicitMT();
      auto fillFunction = [&](int iThread_){

        TChain* threadChain;
        threadChain = (TChain*) chainPtr->Clone();

        Long64_t nEvents = threadChain->GetEntries();
        PhysicsEvent eventBufThread(eventBuf);
        eventBufThread.hookToTree(threadChain, not isData);
        GenericToolbox::disableUnhookedBranches(threadChain);

        auto threadSampleIndexOffsetList = sampleIndexOffsetList;

        std::string progressTitle = LogInfo.getPrefixString() + "Reading selected events";
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
          eventBufThread.setEntryIndex(iEvent);

          for( size_t iSample = 0 ; iSample < samplesToFillList.size() ; iSample++ ){
            if( eventIsInSamplesList.at(iEvent).at(iSample) ){
              sampleEventListPtrToFill.at(iSample)->at(threadSampleIndexOffsetList.at(iSample)++) = PhysicsEvent(eventBufThread); // copy
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
      << " with dataset: " << dataSet.getName() << std::endl;

      std::string nominalWeightLeafStr = isData ? dataSet.getDataNominalWeightLeafName(): dataSet.getMcNominalWeightLeafName();
      if( not nominalWeightLeafStr.empty() ){
        LogInfo << "Copying events nominal weight..." << std::endl;
        int iEvt = 0;
        for( auto* eventList : sampleEventListPtrToFill ){
          for( auto& event : *eventList ){
            event.setTreeWeight(event.getVarAsDouble(nominalWeightLeafStr));
            event.setNominalWeight(event.getTreeWeight());
            event.resetEventWeight();
//            if(iEvt++ < 10) LogTrace << event << std::endl;
          }
        }
      }

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
    LogInfo << "Total events loaded in \"" << sample.getName() << "\":" << std::endl
    << "-> mc: " << sample.getMcEventList().size() << "("
    << GenericToolbox::parseSizeUnits(sizeof(sample.getMcEventList()) * sample.getMcEventList().size())
    << ")" << std::endl
    << "-> data: " << sample.getDataEventList().size() << "("
    << GenericToolbox::parseSizeUnits(sizeof(sample.getDataEventList()) * sample.getDataEventList().size())
    << ")" << std::endl;
  }

}


