//
// Created by Nadrino on 22/07/2021.
//

#include <TTreeFormulaManager.h>
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "Logger.h"

#include "JsonUtils.h"
#include "DataSet.h"
#include "GlobalVariables.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[DataSet]");
})

DataSet::DataSet() { this->reset(); }
DataSet::~DataSet() { this->reset(); }

void DataSet::reset() {
  _isInitialized_ = false;
  _config_.clear();
  _isEnabled_ = false;

  _name_ = "";
  _requestedLeafNameList_.clear();
  _mcFilePathList_.clear();
  _dataFilePathList_.clear();

  _mcNominalWeightFormulaStr_ = "1";
}

void DataSet::setConfig(const nlohmann::json &config_) {
  _config_ = config_;
  JsonUtils::forwardConfig(_config_, __CLASS_NAME__);
}
void DataSet::addRequestedLeafName(const std::string& leafName_){
  LogThrowIf(leafName_.empty(), "no leaf name provided.")
  if( not GenericToolbox::doesElementIsInVector(leafName_, _requestedLeafNameList_) ){
    _requestedLeafNameList_.emplace_back(leafName_);
  }
}
void DataSet::addRequestedMandatoryLeafName(const std::string& leafName_){
  if( not leafName_.empty() and not GenericToolbox::doesElementIsInVector(leafName_, _requestedMandatoryLeafNameList_) ){
    _requestedMandatoryLeafNameList_.emplace_back(leafName_);
  }
  this->addRequestedLeafName(leafName_);
}

void DataSet::initialize() {
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


bool DataSet::isEnabled() const {
  return _isEnabled_;
}
const std::string &DataSet::getName() const {
  return _name_;
}
std::vector<std::string> &DataSet::getMcActiveLeafNameList() {
  return _mcActiveLeafNameList_;
}
std::vector<std::string> &DataSet::getDataActiveLeafNameList() {
  return _dataActiveLeafNameList_;
}
const std::string &DataSet::getMcNominalWeightFormulaStr() const {
  return _mcNominalWeightFormulaStr_;
}
const std::vector<std::string> &DataSet::getRequestedLeafNameList() const {
  return _requestedLeafNameList_;
}
const std::vector<std::string> &DataSet::getRequestedMandatoryLeafNameList() const {
  return _requestedMandatoryLeafNameList_;
}
const std::vector<std::string> &DataSet::getMcFilePathList() const {
  return _mcFilePathList_;
}
const std::vector<std::string> &DataSet::getDataFilePathList() const {
  return _dataFilePathList_;
}

void DataSet::load(FitSampleSet* sampleSetPtr_, const std::vector<FitParameterSet>* parSetList_){
  LogThrowIf(not _isInitialized_, "Can't load dataset while not initialized.");

  std::vector<FitSample*> samplesToFillList;
  std::vector<TTreeFormula*> sampleCutFormulaList;
  std::vector<std::string> samplesNames;

  auto* chainBuf = this->buildMcChain();
  LogThrowIf(chainBuf == nullptr, "No MC files are available for dataset: " << this->getName());
  delete chainBuf;

  chainBuf = this->buildDataChain();
  LogThrowIf(chainBuf == nullptr and sampleSetPtr_->getDataEventType() == DataEventType::DataFiles,
             "Can't define sample \"" << _name_ << "\" while in non-Asimov-like fit and no Data files are available" );
  delete chainBuf;

  for( auto& sample : sampleSetPtr_->getFitSampleList() ){
    if( not sample.isEnabled() ) continue;
    if( sample.isDataSetValid(_name_) ){
      samplesToFillList.emplace_back(&sample);
      samplesNames.emplace_back(sample.getName());
    }
  }
  if( samplesToFillList.empty() ){
    LogAlert << "No sample is set to use this dataset: \"" << _name_ << "\"" << std::endl;
    return;
  }
  LogInfo << "Dataset \"" << _name_ << "\" will populate samples: " << GenericToolbox::parseVectorAsString(samplesNames) << std::endl;

  LogInfo << "Fetching mandatory leaves..." << std::endl;
  for(auto & sampleToFill : samplesToFillList){
    // Fit phase space
    for( const auto& bin : sampleToFill->getBinning().getBinsList() ){
      for( const auto& var : bin.getVariableNameList() ){
        this->addRequestedMandatoryLeafName(var);
      }
    }
  }

  LogInfo << "List of requested leaves: " << GenericToolbox::parseVectorAsString(_requestedLeafNameList_) << std::endl;
  LogInfo << "List of mandatory leaves: " << GenericToolbox::parseVectorAsString(_requestedMandatoryLeafNameList_) << std::endl;

  for( bool isData : {false, true} ){

    TChain* chainPtr{nullptr};
    std::vector<std::string>* activeLeafNameListPtr;

    if( isData and sampleSetPtr_->getDataEventType() == DataEventType::Asimov ){ continue; }
    LogDebug << GET_VAR_NAME_VALUE(isData) << std::endl;

    isData ? LogInfo << "Reading data files..." << std::endl : LogInfo << "Reading MC files..." << std::endl;
    isData ? chainPtr = this->buildDataChain() : chainPtr = this->buildMcChain();
    isData ? activeLeafNameListPtr = &_dataActiveLeafNameList_ : activeLeafNameListPtr = &this->getMcActiveLeafNameList();

    if( chainPtr == nullptr or chainPtr->GetEntries() == 0 ){ continue; }

    LogInfo << "Checking the availability of requested leaves..." << std::endl;
    for( auto& requestedLeaf : _requestedLeafNameList_ ){
      if( not isData or GenericToolbox::doesElementIsInVector(requestedLeaf, _requestedMandatoryLeafNameList_) ){
        LogThrowIf(chainPtr->GetLeaf(requestedLeaf.c_str()) == nullptr,
                   "Could not find leaf \"" << requestedLeaf << "\" in TChain");
      }

      if( chainPtr->GetLeaf(requestedLeaf.c_str()) != nullptr ){
        activeLeafNameListPtr->emplace_back(requestedLeaf);
      }
    }
    LogInfo << "List of leaves which will be loaded in buffer: "
            << GenericToolbox::parseVectorAsString(*activeLeafNameListPtr) << std::endl;

    LogInfo << "Performing event selection of samples with " << (isData? "data": "mc") << " files..." << std::endl;
    chainPtr->SetBranchStatus("*", true);
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

    LogDebug << "Enabling only needed branches for sample selection..." << std::endl;
    chainPtr->SetBranchStatus("*", false);
    for( auto* sampleFormula : sampleCutFormulaList ){
      for( int iLeaf = 0 ; iLeaf < sampleFormula->GetNcodes() ; iLeaf++ ){
        chainPtr->SetBranchStatus(sampleFormula->GetLeaf(iLeaf)->GetName(), true);
      }
    }

    Long64_t nEvents = chainPtr->GetEntries();
    // for each event, which sample is active?
    std::vector<std::vector<bool>> eventIsInSamplesList(nEvents, std::vector<bool>(samplesToFillList.size(), true));
    std::vector<size_t> sampleNbOfEvents(samplesToFillList.size(), 0);
    std::string progressTitle = LogWarning.getPrefixString() + "Performing event selection";
    TFile* lastFilePtr{nullptr};
    for( Long64_t iEvent = 0 ; iEvent < nEvents ; iEvent++ ){
      GenericToolbox::displayProgressBar(iEvent, nEvents, progressTitle);
      chainPtr->GetEntry(iEvent);

      for( size_t iSample = 0 ; iSample < sampleCutFormulaList.size() ; iSample++ ){
        for(int jInstance = 0; jInstance < sampleCutFormulaList.at(iSample)->GetNdata(); jInstance++) {
          if (sampleCutFormulaList.at(iSample)->EvalInstance(jInstance) == 0) {
            // if it doesn't passes the cut
            eventIsInSamplesList.at(iEvent).at(iSample) = false;
            break;
          }
        } // Formula Instances
        if( eventIsInSamplesList.at(iEvent).at(iSample) ){ sampleNbOfEvents.at(iSample)++; }
      } // iSample
    } // iEvent

    // The following lines are necessary since the events might get resized while being in multithread
    // Because std::vector is insuring continuous memory allocation, a resize sometimes
    // lead to the full moving of a vector memory. This is not thread safe, so better ensure
    // the vector won't have to do this by allocating the right event size.
    PhysicsEvent eventBuf;
    eventBuf.setLeafNameListPtr(activeLeafNameListPtr);
    eventBuf.setDataSetIndex(_dataSetIndex_);
    chainPtr->SetBranchStatus("*", true);
    eventBuf.hookToTree(chainPtr, not isData);
    chainPtr->GetEntry(0); // memory is claimed -> eventBuf has the right size
    // Now the eventBuffer has the right size in memory
    delete chainPtr; // not used anymore

    LogInfo << "Claiming memory for additional events in samples: "
            << GenericToolbox::parseVectorAsString(sampleNbOfEvents) << std::endl;
    std::vector<size_t> sampleIndexOffsetList(samplesToFillList.size(), 0);
    std::vector< std::vector<PhysicsEvent>* > sampleEventListPtrToFill(samplesToFillList.size(), nullptr);

    for( size_t iSample = 0 ; iSample < sampleNbOfEvents.size() ; iSample++ ){
      LogDebug << "Claiming memory for sample #" << iSample << std::endl;
      if( isData ){
        sampleEventListPtrToFill.at(iSample) = &samplesToFillList.at(iSample)->getDataContainer().eventList;
        sampleIndexOffsetList.at(iSample) = sampleEventListPtrToFill.at(iSample)->size();
        samplesToFillList.at(iSample)->getDataContainer().reserveEventMemory(_dataSetIndex_, sampleNbOfEvents.at(iSample),
                                                                             PhysicsEvent());
      }
      else{
        sampleEventListPtrToFill.at(iSample) = &samplesToFillList.at(iSample)->getMcContainer().eventList;
        sampleIndexOffsetList.at(iSample) = sampleEventListPtrToFill.at(iSample)->size();
        samplesToFillList.at(iSample)->getMcContainer().reserveEventMemory(_dataSetIndex_, sampleNbOfEvents.at(iSample),
                                                                           PhysicsEvent());
      }
    }

    // Fill function
    ROOT::EnableImplicitMT();
    std::mutex eventOffSetMutex;
    auto fillFunction = [&](int iThread_){

      TChain* threadChain;
      TTreeFormula* threadNominalWeightFormula{nullptr};

      threadChain = isData ? this->buildDataChain() : this->buildMcChain();
      threadChain->SetBranchStatus("*", true);

      if( not isData and not this->getMcNominalWeightFormulaStr().empty() ){
        threadNominalWeightFormula = new TTreeFormula(
            Form("NominalWeightFormula%i", iThread_),
            this->getMcNominalWeightFormulaStr().c_str(),
            threadChain
        );
        threadChain->SetNotify(threadNominalWeightFormula);
      }


      Long64_t nEvents = threadChain->GetEntries();
      PhysicsEvent eventBufThread(eventBuf);
      eventBufThread.hookToTree(threadChain, not isData);
      GenericToolbox::disableUnhookedBranches(threadChain);
      if( threadNominalWeightFormula != nullptr ){
        for( int iLeaf = 0 ; iLeaf < threadNominalWeightFormula->GetNcodes() ; iLeaf++ ){
          threadChain->SetBranchStatus(threadNominalWeightFormula->GetLeaf(iLeaf)->GetName(), true);
        }
      }


//        auto threadSampleIndexOffsetList = sampleIndexOffsetList;
      size_t sampleEventIndex;
      const std::vector<DataBin>* binsListPtr;

      // Loop vars
      int iBin{0};
      size_t iVar{0};
      size_t iSample{0};

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

        if( threadNominalWeightFormula != nullptr ){
          eventBufThread.setTreeWeight(threadNominalWeightFormula->EvalInstance());
          if( eventBufThread.getTreeWeight() == 0 ) continue;
          eventBufThread.setNominalWeight(eventBufThread.getTreeWeight());
          eventBufThread.resetEventWeight();
        }

        for( iSample = 0 ; iSample < samplesToFillList.size() ; iSample++ ){
          if( eventIsInSamplesList.at(iEvent).at(iSample) ){

            // Reset bin index of the buffer
            eventBufThread.setSampleBinIndex(-1);

            // Has valid bin?
            binsListPtr = &samplesToFillList.at(iSample)->getBinning().getBinsList();

            for( iBin = 0 ; iBin < binsListPtr->size() ; iBin++ ){
              auto& bin = binsListPtr->at(iBin);
              bool isInBin = true;
              for( iVar = 0 ; iVar < bin.getVariableNameList().size() ; iVar++ ){
                if( not bin.isBetweenEdges(iVar, eventBufThread.getVarAsDouble(bin.getVariableNameList().at(iVar))) ){
                  isInBin = false;
                  break;
                }
              } // Var
              if( isInBin ){
                eventBufThread.setSampleBinIndex(int(iBin));
                break;
              }
            } // Bin

            if( eventBufThread.getSampleBinIndex() == -1 ) {
              // Invalid bin
              break;
            }

//              sampleEventIndex = sampleIndexOffsetList.at(iSample);
//              sampleEventListPtrToFill.at(iSample)->at(sampleEventIndex) = eventBufThread; // copy

            eventOffSetMutex.lock();
            sampleEventIndex = sampleIndexOffsetList.at(iSample)++;
            sampleEventListPtrToFill.at(iSample)->at(sampleEventIndex) = PhysicsEvent(eventBufThread); // copy
            sampleEventListPtrToFill.at(iSample)->at(sampleEventIndex).clonePointerLeaves(); // make sure the pointer leaves aren't pointing toward the TTree basket
            eventOffSetMutex.unlock();
          }
        }
      }
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

TChain* DataSet::buildChain(bool isData_){
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
TChain* DataSet::buildMcChain(){
  return buildChain(false);
}
TChain* DataSet::buildDataChain(){
  return buildChain(true);
}
void DataSet::print() {
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
