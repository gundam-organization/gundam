//
// Created by Nadrino on 22/07/2021.
//

#include <TTreeFormulaManager.h>
#include <SplineDial.h>
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "Logger.h"

#include "JsonUtils.h"
#include "DataSetLoader.h"
#include "DialSet.h"
#include "GlobalVariables.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[DataSetLoader]");
})

DataSetLoader::DataSetLoader() { this->reset(); }
DataSetLoader::~DataSetLoader() { this->reset(); }

void DataSetLoader::reset() {
  _isInitialized_ = false;
  _config_.clear();
  _isEnabled_ = false;

  _name_ = "";
  _mcFilePathList_.clear();
  _dataFilePathList_.clear();

  _leavesRequestedForIndexing_.clear();
  _leavesStorageRequestedForData_.clear();
  _leavesStorageRequestedForMc_.clear();

  _mcNominalWeightFormulaStr_ = "1";
}

void DataSetLoader::setConfig(const nlohmann::json &config_) {
  _config_ = config_;
  JsonUtils::forwardConfig(_config_, __CLASS_NAME__);
}
void DataSetLoader::setDataSetIndex(int dataSetIndex) {
  _dataSetIndex_ = dataSetIndex;
}

void DataSetLoader::addLeafRequestedForIndexing(const std::string& leafName_) {
  LogThrowIf(leafName_.empty(), "no leaf name provided.")
  if( not GenericToolbox::doesElementIsInVector(leafName_, _leavesRequestedForIndexing_) ){
    _leavesRequestedForIndexing_.emplace_back(leafName_);
  }
}
void DataSetLoader::addLeafStorageRequestedForData(const std::string& leafName_) {
  LogThrowIf(leafName_.empty(), "no leaf name provided.")
  if( not GenericToolbox::doesElementIsInVector(leafName_, _leavesStorageRequestedForData_) ){
    _leavesStorageRequestedForData_.emplace_back(leafName_);
  }
  this->addLeafRequestedForIndexing(leafName_);
}
void DataSetLoader::addLeafStorageRequestedForMc(const std::string& leafName_){
  LogThrowIf(leafName_.empty(), "no leaf name provided.")
  if( not GenericToolbox::doesElementIsInVector(leafName_, _leavesStorageRequestedForMc_) ){
    _leavesStorageRequestedForMc_.emplace_back(leafName_);
  }
  this->addLeafRequestedForIndexing(leafName_);
}

void DataSetLoader::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  if( _config_.empty() ){
    LogError << "_config_ is not set." << std::endl;
    throw std::logic_error("_config_ is not set.");
  }

  _isEnabled_ = JsonUtils::fetchValue(_config_, "isEnabled", true);
  if( not _isEnabled_ ){
    LogWarning << "\"" << _name_ << "\" is disabled." << std::endl;
    return;
  }

  _name_ = JsonUtils::fetchValue<std::string>(_config_, "name");
  LogDebug << "Initializing dataset: \"" << _name_ << "\"" << std::endl;

  {
    auto mcConfig = JsonUtils::fetchValue(_config_, "mc", nlohmann::json());
    if( not mcConfig.empty() ){
      _mcTreeName_ = JsonUtils::fetchValue<std::string>(mcConfig, "tree");
      auto fileList = JsonUtils::fetchValue(mcConfig, "filePathList", nlohmann::json());
      for( const auto& file: fileList ){
        _mcFilePathList_.emplace_back(file.get<std::string>());
      }
    }

    // override: nominalWeightLeafName is deprecated
    _mcNominalWeightFormulaStr_ = JsonUtils::fetchValue(mcConfig, "nominalWeightLeafName", _mcNominalWeightFormulaStr_);
    _mcNominalWeightFormulaStr_ = JsonUtils::fetchValue(mcConfig, "nominalWeightFormula", _mcNominalWeightFormulaStr_);
  }

  {
    auto dataConfig = JsonUtils::fetchValue(_config_, "data", nlohmann::json());
    if( not dataConfig.empty() ){
      _dataTreeName_ = JsonUtils::fetchValue<std::string>(dataConfig, "tree");
      auto fileList = JsonUtils::fetchValue(dataConfig, "filePathList", nlohmann::json());
      for( const auto& file: fileList ){
        _dataFilePathList_.emplace_back(file.get<std::string>());
      }
    }
  }

  this->print();

  _isInitialized_ = true;
}


bool DataSetLoader::isEnabled() const {
  return _isEnabled_;
}
const std::string &DataSetLoader::getName() const {
  return _name_;
}
std::vector<std::string> &DataSetLoader::getMcActiveLeafNameList() {
  return _mcActiveLeafNameList_;
}
std::vector<std::string> &DataSetLoader::getDataActiveLeafNameList() {
  return _dataActiveLeafNameList_;
}
const std::string &DataSetLoader::getMcNominalWeightFormulaStr() const {
  return _mcNominalWeightFormulaStr_;
}
const std::vector<std::string> &DataSetLoader::getMcFilePathList() const {
  return _mcFilePathList_;
}
const std::vector<std::string> &DataSetLoader::getDataFilePathList() const {
  return _dataFilePathList_;
}

void DataSetLoader::load(FitSampleSet* sampleSetPtr_, std::vector<FitParameterSet>* parSetList_){
  LogThrowIf(not _isInitialized_, "Can't load dataset while not initialized.");

  this->fetchRequestedLeaves(parSetList_);
  this->fetchRequestedLeaves(sampleSetPtr_);

  std::vector<FitSample*> samplesToFillList = this->buildListOfSamplesToFill(sampleSetPtr_);
  if( samplesToFillList.empty() ){ LogWarning << "DataSet \"" << _name_ << "\" isn't used by any defined sample." << std::endl; return; }
  LogInfo << "Dataset \"" << _name_ << "\" will populate samples: {";
  for( auto* sample : samplesToFillList ){
    LogInfo << " \"" << sample->getName() << "\",";
  }
  LogInfo << "}." << std::endl;

  LogInfo << "List of leaves requested for event indexing: " << GenericToolbox::parseVectorAsString(_leavesRequestedForIndexing_) << std::endl;
  LogInfo << "List of leaves requested for MC event storage: " << GenericToolbox::parseVectorAsString(_leavesStorageRequestedForMc_) << std::endl;
  LogInfo << "List of leaves requested for Data event storage: " << GenericToolbox::parseVectorAsString(_leavesStorageRequestedForData_) << std::endl;

  for( bool isData : {false, true} ){

    if( isData and sampleSetPtr_->getDataEventType() == DataEventType::Asimov ){
      // Asimov events will be loaded after the prior weight have been propagated on MC samples
      continue;
    }

    TChain* chainPtr{nullptr};
    std::vector<std::string>* activeLeafNameListPtr;

    LogDebug << GET_VAR_NAME_VALUE(isData) << std::endl;

    isData ? LogInfo << "Reading data files..." << std::endl : LogInfo << "Reading MC files..." << std::endl;
    isData ? chainPtr = this->buildDataChain() : chainPtr = this->buildMcChain();
    isData ? activeLeafNameListPtr = &_dataActiveLeafNameList_ : activeLeafNameListPtr = &this->getMcActiveLeafNameList();

    LogThrowIf(chainPtr == nullptr, "Can't open TChain.");
    LogThrowIf(chainPtr->GetEntries() == 0, "TChain is empty.");

    LogInfo << "Performing event selection of samples with " << (isData? "data": "mc") << " files..." << std::endl;
    std::vector<std::vector<bool>> eventIsInSamplesList = this->makeEventSelection(samplesToFillList, isData);
    std::vector<size_t> sampleNbOfEvents(samplesToFillList.size(), 0);
    for( size_t iEvent = 0 ; iEvent < eventIsInSamplesList.size() ; iEvent++ ){
      for( size_t iSample = 0 ; iSample < samplesToFillList.size() ; iSample++ ){
        if(eventIsInSamplesList[iEvent][iSample]) sampleNbOfEvents[iSample]++;
      }
    }
    LogInfo << "Events passing selection cuts:" << std::endl;
    for( size_t iSample = 0 ; iSample < samplesToFillList.size() ; iSample++ ){
      LogInfo << " - \"" << samplesToFillList.at(iSample)->getName() << "\": " << sampleNbOfEvents[iSample] << std::endl;
    }

    LogInfo << "Claiming memory event storage..." << std::endl;
    // The following lines are necessary since the events might get resized while being in multithread
    // Because std::vector is insuring continuous memory allocation, a resize sometimes
    // lead to the full moving of a vector memory. This is not thread safe, so better ensure
    // the vector won't have to do this by allocating the right event size.
    PhysicsEvent eventTemplate;
    isData ? eventTemplate.setLeafNameListPtr(&_leavesStorageRequestedForData_): eventTemplate.setLeafNameListPtr(&_leavesStorageRequestedForMc_);
    eventTemplate.setDataSetIndex(_dataSetIndex_);
    chainPtr->SetBranchStatus("*", true);
    eventTemplate.hookToTree(chainPtr, true);
//    chainPtr->GetEntry(0); // memory is claimed -> eventBuf has the right size ... necessary?
    // Now the eventBuffer has the right size in memory
//    delete chainPtr; // not used anymore

    size_t dialCacheSize = 0;
    for( auto& parSet : *parSetList_ ){
      if( parSet.isUseOnlyOneParameterPerEvent() ){ dialCacheSize++; }
      else{
        dialCacheSize += parSet.getNbParameters();
      }
    }
    eventTemplate.getRawDialPtrList().resize(dialCacheSize);

    std::vector<size_t> sampleIndexOffsetList(samplesToFillList.size(), 0);
    std::vector< std::vector<PhysicsEvent>* > sampleEventListPtrToFill(samplesToFillList.size(), nullptr);

    for( size_t iSample = 0 ; iSample < sampleNbOfEvents.size() ; iSample++ ){
      if( isData ){
        sampleEventListPtrToFill.at(iSample) = &samplesToFillList.at(iSample)->getDataContainer().eventList;
        sampleIndexOffsetList.at(iSample) = sampleEventListPtrToFill.at(iSample)->size();
        samplesToFillList.at(iSample)->getDataContainer().reserveEventMemory(_dataSetIndex_, sampleNbOfEvents.at(iSample),
                                                                             eventTemplate);
      }
      else{
        sampleEventListPtrToFill.at(iSample) = &samplesToFillList.at(iSample)->getMcContainer().eventList;
        sampleIndexOffsetList.at(iSample) = sampleEventListPtrToFill.at(iSample)->size();
        samplesToFillList.at(iSample)->getMcContainer().reserveEventMemory(_dataSetIndex_, sampleNbOfEvents.at(iSample),
                                                                           eventTemplate);
      }
    }

    // DIALS
    DialSet* dialSetPtr;
    std::map<FitParameterSet*, std::vector<DialSet*>> dialSetPtrMap;
    for( auto& parSet : *parSetList_ ){
      if( not parSet.isEnabled() ){ continue; }
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ){ continue; }
        dialSetPtr = par.findDialSet( _name_ );
        if( dialSetPtr != nullptr and ( not dialSetPtr->getDialList().empty() or not dialSetPtr->getDialLeafName().empty() ) ){
          dialSetPtrMap[&parSet].emplace_back( dialSetPtr );
        }
      }
    }

    // Fill function
    ROOT::EnableImplicitMT();
    std::mutex eventOffSetMutex;
    auto fillFunction = [&](int iThread_){

      TChain* threadChain;
      TTreeFormula* threadNominalWeightFormula{nullptr};

      threadChain = isData ? this->buildDataChain() : this->buildMcChain();
      threadChain->SetBranchStatus("*", false);

      if( not isData and not this->getMcNominalWeightFormulaStr().empty() ){
        threadChain->SetBranchStatus("*", true);
        threadNominalWeightFormula = new TTreeFormula(
            Form("NominalWeightFormula%i", iThread_),
            this->getMcNominalWeightFormulaStr().c_str(),
            threadChain
        );
        threadChain->SetNotify(threadNominalWeightFormula);
        threadChain->SetBranchStatus("*", false);
        for( int iLeaf = 0 ; iLeaf < threadNominalWeightFormula->GetNcodes() ; iLeaf++ ){
          threadChain->SetBranchStatus(threadNominalWeightFormula->GetLeaf(iLeaf)->GetName(), true);
        }
      }

      for( auto& leafName : _leavesRequestedForIndexing_ ){
        threadChain->SetBranchStatus(leafName.c_str(), true);
      }


      Long64_t nEvents = threadChain->GetEntries();
      PhysicsEvent eventBuffer;
      eventBuffer.setLeafNameListPtr(&_leavesRequestedForIndexing_);
      eventBuffer.hookToTree(threadChain, not isData);

      PhysicsEvent* eventPtr{nullptr};

      size_t sampleEventIndex;
      const std::vector<DataBin>* binsListPtr;

      // Loop vars
      int iBin{0};
      size_t iVar{0};
      size_t iSample{0};
      // Dials
      size_t eventDialOffset;
      DialSet* dialSetPtr;
      size_t iDialSet, iDial;
      TGraph* grPtr;
      SplineDial* spDialPtr;
      const DataBin* applyConditionBinPtr;

      std::string progressTitle = LogInfo.getPrefixString() + "Reading selected events";
      for(Long64_t iEntry = 0 ; iEntry < nEvents ; iEntry++ ){
        if(iEntry % GlobalVariables::getNbThreads() != iThread_ ){ continue; }
        if( iThread_ == 0 ) GenericToolbox::displayProgressBar(iEntry, nEvents, progressTitle);

        bool skipEvent = true;
        for( bool isInSample : eventIsInSamplesList.at(iEntry) ){
          if( isInSample ){
            skipEvent = false;
            break;
          }
        }
        if( skipEvent ) continue;

        threadChain->GetEntry(iEntry);

        if( threadNominalWeightFormula != nullptr ){
          eventBuffer.setTreeWeight(threadNominalWeightFormula->EvalInstance());
          if( eventBuffer.getTreeWeight() == 0 ) continue; // skip this event
        }

        for( iSample = 0 ; iSample < samplesToFillList.size() ; iSample++ ){
          if( eventIsInSamplesList.at(iEntry).at(iSample) ){

            // Reset bin index of the buffer
            eventBuffer.setSampleBinIndex(-1);

            // Has valid bin?
            binsListPtr = &samplesToFillList.at(iSample)->getBinning().getBinsList();

            for( iBin = 0 ; iBin < binsListPtr->size() ; iBin++ ){
              auto& bin = binsListPtr->at(iBin);
              bool isInBin = true;
              for( iVar = 0 ; iVar < bin.getVariableNameList().size() ; iVar++ ){
                if( not bin.isBetweenEdges(iVar, eventBuffer.getVarAsDouble(bin.getVariableNameList().at(iVar))) ){
                  isInBin = false;
                  break;
                }
              } // Var
              if( isInBin ){
                eventBuffer.setSampleBinIndex(int(iBin));
                break;
              }
            } // Bin

            if( eventBuffer.getSampleBinIndex() == -1 ) {
              // Invalid bin -> next sample
              break;
            }

            eventOffSetMutex.lock();
            sampleEventIndex = sampleIndexOffsetList.at(iSample)++;
            eventOffSetMutex.unlock();

            eventPtr = &sampleEventListPtrToFill.at(iSample)->at(sampleEventIndex);
            // copy only necessary variables
            eventPtr->copyOnlyExistingLeaves(eventBuffer);

            eventPtr->setEntryIndex(iEntry);
            eventPtr->setSampleBinIndex(eventBuffer.getSampleBinIndex());
            eventPtr->setTreeWeight(eventBuffer.getTreeWeight());
            eventPtr->setNominalWeight(eventPtr->getTreeWeight());
            eventPtr->resetEventWeight();

            eventDialOffset = 0;
            for( auto& dialSetPair : dialSetPtrMap ){
              for( iDialSet = 0 ; iDialSet < dialSetPair.second.size() ; iDialSet++ ){
                dialSetPtr = dialSetPair.second[iDialSet];

                if( dialSetPtr->getApplyConditionFormula() != nullptr ){
                  if( eventBuffer.evalFormula(dialSetPtr->getApplyConditionFormula()) == 0 ){
                    continue;
                  }
                }

                if( not dialSetPtr->getDialLeafName().empty() ){
                  grPtr = (TGraph*) eventBuffer.getVariable<std::shared_ptr<TClonesArray>>(dialSetPtr->getDialLeafName())->At(0);
                  if(grPtr->GetN() > 1){
                    eventOffSetMutex.lock();
                    // DEFINE A NEW DIAL!
                    dialSetPtr->getDialList().emplace_back(new SplineDial());
                    spDialPtr = ((SplineDial*)dialSetPtr->getDialList().back().get());
                    eventOffSetMutex.unlock();
                    spDialPtr->setSplinePtr( new TSpline3(Form("%x", spDialPtr), grPtr) );
                    spDialPtr->setAssociatedParameterReference(dialSetPtr->getAssociatedParameterReference());
                    // Adding dial in the event
                    eventPtr->getRawDialPtrList()[eventDialOffset++] = spDialPtr;
                  }
                  continue;
                }

                bool isInBin = false;
                for( iDial = 0 ; iDial < dialSetPtr->getDialList().size(); iDial++ ){
                  applyConditionBinPtr = &dialSetPtr->getDialList()[iDial]->getApplyConditionBin();
                  isInBin = true;

                  for( iVar = 0 ; iVar < applyConditionBinPtr->getVariableNameList().size() ; iVar++ ){
                    if( not applyConditionBinPtr->isBetweenEdges(iVar, eventBuffer.getVarAsDouble(applyConditionBinPtr->getVariableNameList()[iVar] ) )){
                      isInBin = false;
                      break;
                    }
                  }
                  if( isInBin ){
                    eventPtr->getRawDialPtrList()[eventDialOffset++] = dialSetPtr->getDialList()[iDial].get();
                    break;
                  }
                } // iDial

                if( isInBin and dialSetPair.first->isUseOnlyOneParameterPerEvent() ){
                  break; // leave iDialSet / enabled parameters loop
                }

              } // iDialSet / Enabled-parameter
            } // ParSet / DialSet Pairs

            eventOffSetMutex.lock();
            eventPtr->getRawDialPtrList().resize(eventDialOffset);
            eventPtr->getRawDialPtrList().shrink_to_fit();
            eventOffSetMutex.unlock();

          } // event has passed the selection?
        } // samples
      } // entries
      if( iThread_ == 0 ) GenericToolbox::displayProgressBar(nEvents, nEvents, progressTitle);
      delete threadChain;
      delete threadNominalWeightFormula;
    };

    LogInfo << "Copying selected events to RAM..." << std::endl;
    GlobalVariables::getParallelWorker().addJob(__METHOD_NAME__, fillFunction);
    GlobalVariables::getParallelWorker().runJob(__METHOD_NAME__);
    GlobalVariables::getParallelWorker().removeJob(__METHOD_NAME__);

    LogInfo << "Shrinking event lists..." << std::endl;
    for( size_t iSample = 0 ; iSample < samplesToFillList.size() ; iSample++ ){
      samplesToFillList.at(iSample)->getMcContainer().shrinkEventList(sampleIndexOffsetList.at(iSample));
    }

    LogInfo << "Events have been loaded for " << ( isData ? "data": "mc" )
            << " with dataset: " << this->getName() << std::endl;

  }

}

TChain* DataSetLoader::buildChain(bool isData_){
  LogThrowIf(not _isInitialized_, "Can't do " << __METHOD_NAME__ << " while not init.")
  TChain* out{nullptr};
  if( not isData_ and not _mcFilePathList_.empty() ){
    out = new TChain(_mcTreeName_.c_str());
    for( const auto& file: _mcFilePathList_){
      if( not GenericToolbox::doesTFileIsValid(file, {_mcTreeName_}) ){
        LogError << "Invalid file: " << file << std::endl;
        throw std::runtime_error("Invalid file.");
      }
      out->Add(file.c_str());
    }
  }
  else if( isData_ and not _dataFilePathList_.empty() ){
    out = new TChain(_dataTreeName_.c_str());
    for( const auto& file: _dataFilePathList_){
      if( not GenericToolbox::doesTFileIsValid(file, {_dataTreeName_}) ){
        LogError << "Invalid file: " << file << std::endl;
        throw std::runtime_error("Invalid file.");
      }
      out->Add(file.c_str());
    }
  }
  return out;
}
TChain* DataSetLoader::buildMcChain(){
  return buildChain(false);
}
TChain* DataSetLoader::buildDataChain(){
  return buildChain(true);
}
void DataSetLoader::print() {
  LogInfo << _name_ << std::endl;
  if( _mcFilePathList_.empty() ){
    LogAlert << "No MC files loaded." << std::endl;
  }
  else{
    LogInfo << "List of MC files:" << std::endl;
    for( const auto& filePath : _mcFilePathList_ ){
      LogInfo << filePath << std::endl;
    }
    LogInfo << GET_VAR_NAME_VALUE(_mcNominalWeightFormulaStr_) << std::endl;
  }

  if( _dataFilePathList_.empty() ){
    LogInfo << "No External files loaded." << std::endl;
  }
  else{
    LogInfo << "List of External files:" << std::endl;
    for( const auto& filePath : _dataFilePathList_ ){
      LogInfo << filePath << std::endl;
    }
  }
}

void DataSetLoader::fetchRequestedLeaves(std::vector<FitParameterSet>* parSetList_){

  if( parSetList_ == nullptr ) return;

  // parSet
  for( auto& parSet : *parSetList_ ){
    if( not parSet.isEnabled() ) continue;

    for( auto& par : parSet.getParameterList() ){
      if( not par.isEnabled() ) continue;

      auto* dialSetPtr = par.findDialSet( _name_ );
      if( dialSetPtr == nullptr ){ continue; }

      if( not dialSetPtr->getDialLeafName().empty() ){
        this->addLeafRequestedForIndexing(dialSetPtr->getDialLeafName());
      }
      else{
        if( dialSetPtr->getApplyConditionFormula() != nullptr ){
          for( int iPar = 0 ; iPar < dialSetPtr->getApplyConditionFormula()->GetNpar() ; iPar++ ){
            this->addLeafRequestedForIndexing(dialSetPtr->getApplyConditionFormula()->GetParName(iPar));
          }
        }

        for( auto& dial : dialSetPtr->getDialList() ){
          for( auto& var : dial->getApplyConditionBin().getVariableNameList() ){
            this->addLeafRequestedForIndexing(var);
          } // var
        } // dial
      }

    } // par
  } // parSet

}
void DataSetLoader::fetchRequestedLeaves(FitSampleSet* sampleSetPtr_){

  for( auto& sample: sampleSetPtr_->getFitSampleList() ){
    for( auto& bin: sample.getBinning().getBinsList()){
      for( auto& var: bin.getVariableNameList()){
        this->addLeafRequestedForIndexing(var);
      }
    }
  }

}
void DataSetLoader::fetchRequestedLeaves(PlotGenerator* plotGenPtr_){

  for( auto& var : plotGenPtr_->fetchListOfVarToPlot() ){
    this->addLeafStorageRequestedForMc(var);
    this->addLeafStorageRequestedForData(var);
  }

  for( auto& var : plotGenPtr_->fetchListOfSplitVarNames() ){
    this->addLeafStorageRequestedForMc(var);
  }

}

std::vector<FitSample*> DataSetLoader::buildListOfSamplesToFill(FitSampleSet* sampleSetPtr_){
  std::vector<FitSample*> out;

  for( auto& sample : sampleSetPtr_->getFitSampleList() ){
    if( not sample.isEnabled() ) continue;
    if( sample.isDataSetValid(_name_) ){
      out.emplace_back(&sample);
    }
  }

  return out;
}
std::vector<std::vector<bool>> DataSetLoader::makeEventSelection(std::vector<FitSample*>& samplesToFillList, bool loadData_){

  std::vector<TTreeFormula*> sampleCutFormulaList;

  TChain* chainPtr{nullptr};
  loadData_ ? chainPtr = this->buildDataChain() : chainPtr = this->buildMcChain();
  LogThrowIf(chainPtr == nullptr, "Can't open TChain.");

  chainPtr->SetBranchStatus("*", true); // enabling every branch to define formula
  TTreeFormulaManager formulaManager;
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

    // The TChain will notify the formula that it has to update leaves addresses while swaping TFile
    formulaManager.Add(sampleCutFormulaList.back());
  }
  chainPtr->SetNotify(&formulaManager);

  chainPtr->SetBranchStatus("*", false);
  for( auto* sampleFormula : sampleCutFormulaList ){
    for( int iLeaf = 0 ; iLeaf < sampleFormula->GetNcodes() ; iLeaf++ ){
      chainPtr->SetBranchStatus(sampleFormula->GetLeaf(iLeaf)->GetName(), true);
    }
  }

  Long64_t nEvents = chainPtr->GetEntries();
  // for each event, which sample is active?
  std::vector<std::vector<bool>> eventIsInSamplesList(nEvents, std::vector<bool>(samplesToFillList.size(), true));
  std::string progressTitle = LogWarning.getPrefixString() + "Performing event selection";
  TFile* lastFilePtr{nullptr};
  for( Long64_t iEvent = 0 ; iEvent < nEvents ; iEvent++ ){
    GenericToolbox::displayProgressBar(iEvent, nEvents, progressTitle);
    chainPtr->GetEntry(iEvent);

    for( size_t iSample = 0 ; iSample < sampleCutFormulaList.size() ; iSample++ ){
      for(int jInstance = 0; jInstance < sampleCutFormulaList.at(iSample)->GetNdata(); jInstance++) {
        if (sampleCutFormulaList.at(iSample)->EvalInstance(jInstance) == 0) {
          // if it doesn't pass the cut
          eventIsInSamplesList.at(iEvent).at(iSample) = false;
          break;
        }
      } // Formula Instances
    } // iSample
  } // iEvent

  chainPtr->SetNotify(nullptr);
  delete chainPtr;

  return eventIsInSamplesList;
}
