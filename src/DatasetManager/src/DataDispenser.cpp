//
// Created by Adrien BLANCHET on 14/05/2022.
//

#include "DataDispenser.h"
#include "GlobalVariables.h"
#include "SplineDial.h"
#include "GraphDial.h"
#include "DatasetLoader.h"
#include "JsonUtils.h"

#include "GenericToolbox.Root.TreeEventBuffer.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.VariablesMonitor.h"
#include "Logger.h"

#include "TTreeFormulaManager.h"
#include "TChain.h"
#include "TChainElement.h"

#include "sstream"

LoggerInit([]{
  Logger::setUserHeaderStr("[DataDispenser]");
});

DataDispenser::DataDispenser() = default;
DataDispenser::~DataDispenser() = default;

void DataDispenser::setConfig(const nlohmann::json &config) {
  _config_ = config;
  JsonUtils::forwardConfig(_config_, __CLASS_NAME__);
}
void DataDispenser::setOwner(DatasetLoader* owner_){
  _owner_ = owner_;
}

void DataDispenser::readConfig(){
  LogThrowIf( _config_.empty(), "Config is not set." )
  LogThrowIf( _owner_==nullptr, "Owner not set.")

  _parameters_.treePath = JsonUtils::fetchValue<std::string>(_config_, "tree", _parameters_.treePath);
  _parameters_.filePathList = JsonUtils::fetchValue<std::vector<std::string>>(_config_, "filePathList", _parameters_.filePathList);
  _parameters_.additionalLeavesStorage = JsonUtils::fetchValue(_config_, "additionalLeavesStorage", _parameters_.additionalLeavesStorage);
  _parameters_.useMcContainer = JsonUtils::fetchValue(_config_, "useMcContainer", _parameters_.useMcContainer);

  _parameters_.selectionCutFormulaStr = JsonUtils::buildFormula(_config_, "selectionCutFormula", "&&", _parameters_.selectionCutFormulaStr);
  _parameters_.nominalWeightFormulaStr = JsonUtils::buildFormula(_config_, "nominalWeightFormula", "*", _parameters_.nominalWeightFormulaStr);

  if( JsonUtils::doKeyExist(_config_, "overrideLeafDict") ){
    _parameters_.overrideLeafDict.clear();
    for( auto& entry : JsonUtils::fetchValue<nlohmann::json>(_config_, "overrideLeafDict") ){
      _parameters_.overrideLeafDict[entry["eventVar"]] = entry["leafVar"];
    }
  }

}
void DataDispenser::initialize(){
  this->readConfig();

  LogWarning << "Initialized data dispenser: " << getTitle() << std::endl;
  _isInitialized_ = true;
}

const DataDispenserParameters &DataDispenser::getConfigParameters() const {
  return _parameters_;
}
DataDispenserParameters &DataDispenser::getConfigParameters() {
  return _parameters_;
}

void DataDispenser::setSampleSetPtrToLoad(FitSampleSet *sampleSetPtrToLoad) {
  _sampleSetPtrToLoad_ = sampleSetPtrToLoad;
}
void DataDispenser::setParSetPtrToLoad(std::vector<FitParameterSet> *parSetListPtrToLoad_) {
  _parSetListPtrToLoad_ = parSetListPtrToLoad_;
}
void DataDispenser::setPlotGenPtr(PlotGenerator *plotGenPtr) {
  _plotGenPtr_ = plotGenPtr;
}

void DataDispenser::load(){
  LogWarning << "Loading dataset: " << getTitle() << std::endl;
  LogThrowIf(not _isInitialized_, "Can't load while not initialized.");
  LogThrowIf(_sampleSetPtrToLoad_==nullptr, "SampleSet not specified.");

  _cache_.clear();

  this->buildSampleToFillList();
  if( _cache_.samplesToFillList.empty() ){
    LogError << "No samples were selected for dataset: " << getTitle() << std::endl;
    return;
  }

  auto replaceToyIndexFct = [&](std::string& formula_){
    if( GenericToolbox::doesStringContainsSubstring(formula_, "<I_TOY>") ){
      LogThrowIf(_parameters_.iThrow==-1, "<I_TOY> not set.");
      GenericToolbox::replaceSubstringInsideInputString(formula_, "<I_TOY>", std::to_string(_parameters_.iThrow));
    }
  };
  auto overrideLeavesNamesFct = [&](std::string& formula_){
    for( auto& replaceEntry : _cache_.leavesToOverrideList ){
      GenericToolbox::replaceSubstringInsideInputString(formula_, replaceEntry, _parameters_.overrideLeafDict[replaceEntry]);
    }
  };

  if( not _parameters_.overrideLeafDict.empty() ){
    for( auto& entryDict : _parameters_.overrideLeafDict ){ replaceToyIndexFct(entryDict.second); }
    LogInfo << "Overriding leaf dict: " << std::endl;
    LogInfo << GenericToolbox::parseMapAsString(_parameters_.overrideLeafDict) << std::endl;

    for( auto& overrideEntry : _parameters_.overrideLeafDict ){
      _cache_.leavesToOverrideList.emplace_back(overrideEntry.first);
    }
    // make sure we process the longest words first: "thisIsATest" variable should be replaced before "thisIs"
    std::function<bool(const std::string&, const std::string&)> aGoesFirst =
				[](const std::string& a_, const std::string& b_){ return a_.size() > b_.size(); };
		GenericToolbox::sortVector(_cache_.leavesToOverrideList, aGoesFirst);
  }

  replaceToyIndexFct(_parameters_.nominalWeightFormulaStr);
  replaceToyIndexFct(_parameters_.selectionCutFormulaStr);

  overrideLeavesNamesFct(_parameters_.nominalWeightFormulaStr);
  overrideLeavesNamesFct(_parameters_.selectionCutFormulaStr);

  LogInfo << "Data will be extracted from: " << GenericToolbox::parseVectorAsString(_parameters_.filePathList, true) << std::endl;
  for( const auto& file: _parameters_.filePathList){ LogThrowIf(not GenericToolbox::doesTFileIsValid(file, {_parameters_.treePath}), "Invalid file: " << file); }

  this->doEventSelection();
  this->fetchRequestedLeaves();
  this->preAllocateMemory();
  this->readAndFill();

  LogWarning << "Loaded " << getTitle() << std::endl;
}
std::string DataDispenser::getTitle(){
  std::stringstream ss;
  if( _owner_ != nullptr ) ss << _owner_->getName();
  ss << "/" << _parameters_.name;
  return ss.str();
}

void DataDispenser::addLeafRequestedForIndexing(const std::string& leafName_) {
  LogThrowIf(leafName_.empty(), "no leaf name provided.")
  if( not GenericToolbox::doesElementIsInVector(leafName_, _cache_.leavesRequestedForIndexing) ){
    _cache_.leavesRequestedForIndexing.emplace_back(leafName_);
  }
}
void DataDispenser::addLeafRequestedForStorage(const std::string& leafName_){
  LogThrowIf(leafName_.empty(), "no leaf name provided.")
  if( not GenericToolbox::doesElementIsInVector(leafName_, _cache_.leavesRequestedForStorage) ){
    _cache_.leavesRequestedForStorage.emplace_back(leafName_);
  }
  this->addLeafRequestedForIndexing(leafName_);
}

void DataDispenser::buildSampleToFillList(){
  LogWarning << "Fetching samples to fill..." << std::endl;

  for( auto& sample : _sampleSetPtrToLoad_->getFitSampleList() ){
    if( not sample.isEnabled() ) continue;
    if(sample.isDatasetValid(_owner_->getName()) ){
      _cache_.samplesToFillList.emplace_back(&sample);
    }
  }

  if( _cache_.samplesToFillList.empty() ){
    LogInfo << "No sample selected." << std::endl;
    return;
  }
}
void DataDispenser::doEventSelection(){
  LogWarning << "Performing event selection..." << std::endl;

  LogInfo << "Opening files..." << std::endl;
  TChain treeChain(_parameters_.treePath.c_str());
  for( const auto& file: _parameters_.filePathList){ treeChain.Add(file.c_str()); }
  LogThrowIf(treeChain.GetEntries() == 0, "TChain is empty.");

  LogInfo << "Defining selection formulas..." << std::endl;
  treeChain.SetBranchStatus("*", true); // enabling every branch to define formula

  TTreeFormula* treeSelectionCutFormula{nullptr};
  std::vector<TTreeFormula*> sampleCutFormulaList(_cache_.samplesToFillList.size(), nullptr);
  TTreeFormulaManager formulaManager; // TTreeFormulaManager handles the notification of multiple TTreeFormula for one TTChain

  if( not _parameters_.selectionCutFormulaStr.empty() ){
    treeSelectionCutFormula = new TTreeFormula("SelectionCutFormula", _parameters_.selectionCutFormulaStr.c_str(), &treeChain);
    LogThrowIf(treeSelectionCutFormula->GetNdim() == 0,
               "\"" << _parameters_.selectionCutFormulaStr << "\" could not be parsed by the TChain");

    // The TChain will notify the formula that it has to update leaves addresses while swaping TFile
    formulaManager.Add(treeSelectionCutFormula);
    LogInfo << "Using tree selection cut: \"" << _parameters_.selectionCutFormulaStr << "\"" << std::endl;
  }

  GenericToolbox::TablePrinter t;
  t.setColTitles({{"Sample"}, {"Selection Cut"}});
  for( size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){

    std::string selectionCut = _cache_.samplesToFillList[iSample]->getSelectionCutsStr();
    for( auto& replaceEntry : _cache_.leavesToOverrideList ){
      GenericToolbox::replaceSubstringInsideInputString(selectionCut, replaceEntry, _parameters_.overrideLeafDict[replaceEntry]);
    }

    t.addTableLine({{"\""+_cache_.samplesToFillList[iSample]->getName()+"\""}, {"\""+selectionCut+"\""}});
    sampleCutFormulaList[iSample] = new TTreeFormula(_cache_.samplesToFillList[iSample]->getName().c_str(), selectionCut.c_str(), &treeChain);
    LogThrowIf(sampleCutFormulaList[iSample]->GetNdim() == 0,
               "\"" << selectionCut << "\" could not be parsed by the TChain");

    // The TChain will notify the formula that it has to update leaves addresses while swaping TFile
    formulaManager.Add(sampleCutFormulaList[iSample]);
  }
  treeChain.SetNotify(&formulaManager);
  t.printTable();

  LogInfo << "Enabling required branches..." << std::endl;
  treeChain.SetBranchStatus("*", false);

  if(treeSelectionCutFormula != nullptr) GenericToolbox::enableSelectedBranches(&treeChain, treeSelectionCutFormula);
  for( auto& sampleFormula : sampleCutFormulaList ){
    GenericToolbox::enableSelectedBranches(&treeChain, sampleFormula);
  }

  LogInfo << "Performing event selection..." << std::endl;
  GenericToolbox::VariableMonitor readSpeed("bytes");
  Long64_t nEvents = treeChain.GetEntries();
  // for each event, which sample is active?
  _cache_.eventIsInSamplesList.resize(nEvents, std::vector<bool>(_cache_.samplesToFillList.size(), true));
  _cache_.sampleNbOfEvents.resize(_cache_.samplesToFillList.size(), 0);
  std::string progressTitle = LogInfo.getPrefixString() + "Reading input dataset";
  std::stringstream ssProgressTitle;
  TFile* lastFilePtr{nullptr};
  for( Long64_t iEvent = 0 ; iEvent < nEvents ; iEvent++ ){
    readSpeed.addQuantity(treeChain.GetEntry(iEvent));
    if( GenericToolbox::showProgressBar(iEvent, nEvents) ){
      ssProgressTitle.str("");
      ssProgressTitle << progressTitle << " - ";
      ssProgressTitle << GenericToolbox::padString(GenericToolbox::parseSizeUnits((unsigned int)(readSpeed.evalTotalGrowthRate())), 8) << "/s";
      ssProgressTitle << " (" << GenericToolbox::padString(std::to_string(int(GenericToolbox::getCpuUsageByProcess())), 3, ' ') << "% CPU efficiency)";
      GenericToolbox::displayProgressBar( iEvent, nEvents, ssProgressTitle.str() );
    }

    if(treeSelectionCutFormula != nullptr and not GenericToolbox::doesEntryPassCut(treeSelectionCutFormula)){
      for( size_t iSample = 0 ; iSample < sampleCutFormulaList.size() ; iSample++ ){ _cache_.eventIsInSamplesList[iEvent][iSample] = false; }
      if( GlobalVariables::isDebugVerbose()){
        LogTrace << "Event #" << treeChain.GetFileNumber() << ":" << treeChain.GetReadEntry()
        << " rejected because of " << treeSelectionCutFormula->GetExpFormula() << std::endl;
      }
      continue;
    }

    for( size_t iSample = 0 ; iSample < sampleCutFormulaList.size() ; iSample++ ){
      if( not GenericToolbox::doesEntryPassCut(sampleCutFormulaList[iSample]) ){
        _cache_.eventIsInSamplesList[iEvent][iSample] = false;
        if( GlobalVariables::isDebugVerbose() ){
          LogTrace << "Event #" << treeChain.GetFileNumber() << ":" << treeChain.GetReadEntry()
          << " rejected as sample " << iSample << " because of "
          << sampleCutFormulaList[iSample]->GetExpFormula() << std::endl;
        }
      }
      else{
        _cache_.sampleNbOfEvents[iSample]++;
        if( GlobalVariables::isDebugVerbose() ){
          LogDebug << "Event #" << treeChain.GetFileNumber() << ":" << treeChain.GetReadEntry()
                   << " included as sample " << iSample << " using "
                   << sampleCutFormulaList[iSample]->GetExpFormula() << std::endl;
        }
      }
    } // iSample

    if( treeChain.GetFileNumber() == 0 and treeChain.GetReadEntry() == 0 ){
      for( size_t iSample = 0 ; iSample < sampleCutFormulaList.size() ; iSample++ ){
        LogDebug << sampleCutFormulaList[iSample]->GetLeaf(0)->GetValue(1) << std::endl;
        LogDebug << sampleCutFormulaList[iSample]->GetExpFormula() << " => " << std::endl;
        for(int jInstance = 0; jInstance < sampleCutFormulaList[iSample]->GetNdata(); jInstance++) {
          LogDebug << sampleCutFormulaList[iSample]->EvalInstance(jInstance) << std::endl;
        }
      }
      LogThrow("debug");
    }
  } // iEvent

  if( _owner_->isShowSelectedEventCount() ){
    LogWarning << "Events passing selection cuts:" << std::endl;
    t.reset();
    t.setColTitles({{"Sample"}, {"# of events"}});
    for(size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
      t.addTableLine({{"\""+_cache_.samplesToFillList[iSample]->getName()+"\""}, std::to_string(_cache_.sampleNbOfEvents[iSample])});
    }
    t.printTable();
  }

}
void DataDispenser::fetchRequestedLeaves(){
  LogWarning << "Fetching requested leaves to extract from the trees..." << std::endl;

  // parSet
  if( _parSetListPtrToLoad_ != nullptr ){
    for( auto& parSet : *_parSetListPtrToLoad_ ){
      if( not parSet.isEnabled() ) continue;

      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ) continue;

        auto* dialSetPtr = par.findDialSet( _owner_->getName() );
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
            if( dial->getApplyConditionBinPtr() != nullptr ){
              for( auto& var : dial->getApplyConditionBinPtr()->getVariableNameList() ){
                this->addLeafRequestedForIndexing(var);
              } // var
            }
          } // dial
        }

      } // par
    } // parSet
  }

  // sample
  for( auto& sample: _sampleSetPtrToLoad_->getFitSampleList() ){
    for( auto& bin: sample.getBinning().getBinsList()){
      for( auto& var: bin.getVariableNameList()){
        this->addLeafRequestedForIndexing(var);
      }
    }
  }

  // plotGen
  if( _plotGenPtr_ != nullptr ){
    for( auto& var : _plotGenPtr_->fetchListOfVarToPlot(not _parameters_.useMcContainer) ){
      this->addLeafRequestedForStorage(var);
    }

    if( _parameters_.useMcContainer ){
      for( auto& var : _plotGenPtr_->fetchListOfSplitVarNames() ){
        this->addLeafRequestedForStorage(var);
      }
    }
  }

  for( auto& additionalLeaf: _parameters_.additionalLeavesStorage ){
    this->addLeafRequestedForStorage(additionalLeaf);
  }

  LogInfo << "Vars requested for indexing: " << GenericToolbox::parseVectorAsString(_cache_.leavesRequestedForIndexing, false) << std::endl;
  LogInfo << "Vars requested for storage: " << GenericToolbox::parseVectorAsString(_cache_.leavesRequestedForStorage, false) << std::endl;
}
void DataDispenser::preAllocateMemory(){
  LogInfo << "Pre-allocating memory..." << std::endl;

  /// \brief The following lines are necessary since the events might get resized while being in multithread
  /// Because std::vector is insuring continuous memory allocation, a resize sometimes
  /// lead to the full moving of a vector memory. This is not thread safe, so better ensure
  /// the vector won't have to do this by allocating the right event size.

  // MEMORY CLAIM?
  TChain treeChain(_parameters_.treePath.c_str());
  for( const auto& file: _parameters_.filePathList){ treeChain.Add(file.c_str()); }
  treeChain.SetBranchStatus("*", false);

  std::vector<std::string> leafVarList;
  for( auto& eventVar : _cache_.leavesRequestedForStorage){
    leafVarList.emplace_back(eventVar);
    if( GenericToolbox::doesKeyIsInMap(eventVar, _parameters_.overrideLeafDict) ){
      leafVarList.back() = _parameters_.overrideLeafDict[eventVar];
      leafVarList.back() = GenericToolbox::stripBracket(leafVarList.back(), '[', ']');
    }
  }
  GenericToolbox::TreeEventBuffer tBuf;
  tBuf.setLeafNameList(leafVarList);
  tBuf.hook(&treeChain);

  PhysicsEvent eventPlaceholder;
  eventPlaceholder.setDataSetIndex(_owner_->getDataSetIndex());
  eventPlaceholder.setCommonLeafNameListPtr(std::make_shared<std::vector<std::string>>(_cache_.leavesRequestedForStorage));
  auto copyDict = eventPlaceholder.generateDict(tBuf, _parameters_.overrideLeafDict);
  eventPlaceholder.copyData(copyDict, true);
  if( _parSetListPtrToLoad_ != nullptr ){
    size_t dialCacheSize = 0;
    for( auto& parSet : *_parSetListPtrToLoad_ ){
      parSet.isUseOnlyOneParameterPerEvent() ? dialCacheSize++: dialCacheSize += parSet.getNbParameters();
    }
    eventPlaceholder.getRawDialPtrList().resize(dialCacheSize);
  }

  _cache_.sampleIndexOffsetList.resize(_cache_.samplesToFillList.size());
  _cache_.sampleEventListPtrToFill.resize(_cache_.samplesToFillList.size());
  for( size_t iSample = 0 ; iSample < _cache_.sampleNbOfEvents.size() ; iSample++ ){
    auto* container = &_cache_.samplesToFillList[iSample]->getDataContainer();
    if(_parameters_.useMcContainer) container = &_cache_.samplesToFillList[iSample]->getMcContainer();

    _cache_.sampleEventListPtrToFill[iSample] = &container->eventList;
    _cache_.sampleIndexOffsetList[iSample] = _cache_.sampleEventListPtrToFill[iSample]->size();
    container->reserveEventMemory(_owner_->getDataSetIndex(), _cache_.sampleNbOfEvents[iSample], eventPlaceholder);
  }

  // DIALS
  DialSet* dialSetPtr;
  if( _parSetListPtrToLoad_ != nullptr ){
    LogInfo << "Claiming memory for additional dials..." << std::endl;
    for( auto& parSet : *_parSetListPtrToLoad_ ){
      if( not parSet.isEnabled() ){ continue; }
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ){ continue; }
        dialSetPtr = par.findDialSet( _owner_->getName() );
        if( dialSetPtr != nullptr and ( not dialSetPtr->getDialList().empty() or not dialSetPtr->getDialLeafName().empty() ) ){

          // Filling var indexes:
          for( auto& dial : dialSetPtr->getDialList() ){
            if( dial->getApplyConditionBinPtr() != nullptr ){
              std::vector<int> varIndexes;
              for( const auto& var : dial->getApplyConditionBinPtr()->getVariableNameList() ){
                varIndexes.emplace_back(GenericToolbox::findElementIndex(var, _cache_.leavesRequestedForIndexing));
              }
              dial->getApplyConditionBinPtr()->setEventVarIndexCache(varIndexes);
            }
          }

          // Reserve memory for additional dials (those on a tree leaf)
          if( not dialSetPtr->getDialLeafName().empty() ){

            auto dialType = dialSetPtr->getGlobalDialType();
            if     ( dialType == DialType::Spline ){
              dialSetPtr->getDialList().resize(treeChain.GetEntries(), DialWrapper(SplineDial()));
            }
            else if( dialType == DialType::Graph ){
              dialSetPtr->getDialList().resize(treeChain.GetEntries(), DialWrapper(GraphDial()));
            }
            else{
              LogThrow("Invalid dial type for event-by-event dial: " << DialType::DialTypeEnumNamespace::toString(dialType))
            }

          }

          // Add the dialSet to the list
          _cache_.dialSetPtrMap[&parSet].emplace_back( dialSetPtr );
        }
      }
    }
  }

  LogInfo << "Current RAM is: " << GenericToolbox::parseSizeUnits(double(GenericToolbox::getProcessMemoryUsage())) << std::endl;
}
void DataDispenser::readAndFill(){
  LogWarning << "Reading dataset and loading..." << std::endl;

  if( not _parameters_.nominalWeightFormulaStr.empty() ){
    LogInfo << "Nominal weight: \"" << _parameters_.nominalWeightFormulaStr << "\"" << std::endl;
  }

  ROOT::EnableThreadSafety();
  auto fillFunction = [&](int iThread_){

    int nThreads = GlobalVariables::getNbThreads();
    if( iThread_ == -1 ){
      iThread_ = 0;
      nThreads = 1;
    }

    TChain treeChain(_parameters_.treePath.c_str());
    for( const auto& file: _parameters_.filePathList){ treeChain.Add(file.c_str()); }

    TTreeFormula* threadNominalWeightFormula{nullptr};
    TList objToNotify;
    treeChain.SetNotify(&objToNotify);

    treeChain.SetBranchStatus("*", false);

    if( not _parameters_.nominalWeightFormulaStr.empty() ){
      treeChain.SetBranchStatus("*", true);
      threadNominalWeightFormula = new TTreeFormula(
          Form("NominalWeightFormula%i", iThread_),
          _parameters_.nominalWeightFormulaStr.c_str(),
          &treeChain
          );
      LogThrowIf(threadNominalWeightFormula->GetNdim() == 0,
                 "\"" <<  _parameters_.nominalWeightFormulaStr << "\" could not be parsed by the TChain");
      objToNotify.Add(threadNominalWeightFormula); // memory handled here!
      treeChain.SetBranchStatus("*", false);
      GenericToolbox::enableSelectedBranches(&treeChain, threadNominalWeightFormula);
    }

    GenericToolbox::TreeEventBuffer tEventBuffer;
    std::vector<std::string> leafVar;
    for( auto& eventVar : _cache_.leavesRequestedForIndexing){
      leafVar.emplace_back(eventVar);
      if( GenericToolbox::doesKeyIsInMap(eventVar, _parameters_.overrideLeafDict) ){
        leafVar.back() = _parameters_.overrideLeafDict[eventVar];
        leafVar.back() = GenericToolbox::stripBracket(leafVar.back(), '[', ']');
      }
    }
    tEventBuffer.setLeafNameList(leafVar);
    tEventBuffer.hook(&treeChain);

    PhysicsEvent eventBuffer;
    eventBuffer.setDataSetIndex(_owner_->getDataSetIndex());
    eventBuffer.setCommonLeafNameListPtr(std::make_shared<std::vector<std::string>>(_cache_.leavesRequestedForIndexing));
    auto copyDict = eventBuffer.generateDict(tEventBuffer, _parameters_.overrideLeafDict);
    eventBuffer.copyData(copyDict, true); // resize array obj
    eventBuffer.resizeVarToDoubleCache();

    PhysicsEvent evStore;
    evStore.setDataSetIndex(_owner_->getDataSetIndex());
    evStore.setCommonLeafNameListPtr(std::make_shared<std::vector<std::string>>(_cache_.leavesRequestedForStorage));
    auto copyStoreDict = evStore.generateDict(tEventBuffer, _parameters_.overrideLeafDict);


    PhysicsEvent* eventPtr{nullptr};

    size_t sampleEventIndex;
    int threadDialIndex;
    const std::vector<DataBin>* binsListPtr;

    // Loop vars
    bool isEventInDialBin{true};
    int iBin{0};
    int lastFailedBinVarIndex{-1}; int lastEventVarIndex{-1};
    const std::pair<double, double>* lastEdges{nullptr};
    size_t iVar{0};
    size_t iSample{0};
    // Dials
    size_t eventDialOffset;
    DialSet* dialSetPtr;
    size_t iDialSet, iDial;
    TGraph* grPtr{nullptr};
    SplineDial* spDialPtr;
    GraphDial* grDialPtr;
    const DataBin* applyConditionBinPtr;

    // Try to read TTree the closest to sequentially possible
    Long64_t nEvents = treeChain.GetEntries();
    Long64_t nEventPerThread = nEvents/Long64_t(nThreads);
    Long64_t iEnd = nEvents;
    Long64_t iStart = Long64_t(iThread_)*nEventPerThread;
    if( iThread_+1 != nThreads ) iEnd = (Long64_t(iThread_)+1)*nEventPerThread;
    Long64_t iGlobal = 0;

    // Load the branches
    treeChain.LoadTree(iStart);

    // IO speed monitor
    GenericToolbox::VariableMonitor readSpeed("bytes");
    Int_t nBytes;

    std::string progressTitle = LogInfo.getPrefixString();
    std::stringstream ssProgressBar;

    for(Long64_t iEntry = iStart ; iEntry < iEnd ; iEntry++ ){

      if( iThread_ == 0 ){
        if( GenericToolbox::showProgressBar(iGlobal, nEvents) ){
          ssProgressBar.str("");
          ssProgressBar << progressTitle;
          ssProgressBar << GenericToolbox::padString(GenericToolbox::parseSizeUnits(nThreads*readSpeed.getTotalAccumulated()), 9);
          ssProgressBar << " (";
          ssProgressBar << GenericToolbox::padString(GenericToolbox::parseSizeUnits(nThreads*readSpeed.evalTotalGrowthRate()), 9);
          ssProgressBar << "/s)";
          double cpuPercent = GenericToolbox::getCpuUsageByProcess();
          ssProgressBar << " "
          << GenericToolbox::padString(std::to_string(int(cpuPercent)), 3, ' ')
          << "% CPU ("
          << GenericToolbox::padString(std::to_string(int(cpuPercent/GlobalVariables::getNbThreads())), 3, ' ') << "% efficiency)";
          GenericToolbox::displayProgressBar( iGlobal, nEvents, ssProgressBar.str() );
        }
        iGlobal += nThreads;
      }

      bool skipEvent = true;
      for( bool isInSample : _cache_.eventIsInSamplesList[iEntry] ){
        if( isInSample ){ skipEvent = false; break; }
      }
      if( skipEvent ) continue;

      nBytes = treeChain.GetEntry(iEntry);
      if( iThread_ == 0 ) readSpeed.addQuantity(nBytes);

      if( threadNominalWeightFormula != nullptr ){
        eventBuffer.setTreeWeight(threadNominalWeightFormula->EvalInstance());

        if( eventBuffer.getTreeWeight() < 0 ){
          LogError << "Negative nominal weight:" << std::endl;

          LogError << "Event buffer is: " << eventBuffer.getSummary() << std::endl;

          LogError << "Formula leaves:" << std::endl;
          for( int iLeaf = 0 ; iLeaf < threadNominalWeightFormula->GetNcodes() ; iLeaf++ ){
            if( threadNominalWeightFormula->GetLeaf(iLeaf) == nullptr ) continue; // for "Entry$" like dummy leaves
            LogError << "Leaf: " << threadNominalWeightFormula->GetLeaf(iLeaf)->GetName() << "[0] = " << threadNominalWeightFormula->GetLeaf(iLeaf)->GetValue(0) << std::endl;
          }

          LogThrow("Negative nominal weight");
        }
        if( eventBuffer.getTreeWeight() == 0 ){
          continue;
        } // skip this event
      }

      for( iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
        if( _cache_.eventIsInSamplesList[iEntry][iSample] ){

          // Reset bin index of the buffer
          eventBuffer.setSampleBinIndex(-1);

          // Getting loaded data in tEventBuffer
          eventBuffer.copyData(copyDict, true);

          // Has valid bin?
          binsListPtr = &_cache_.samplesToFillList[iSample]->getBinning().getBinsList();

          for( iBin = 0 ; iBin < binsListPtr->size() ; iBin++ ){
            auto& bin = (*binsListPtr)[iBin];
            bool isInBin = true;
            for( iVar = 0 ; iVar < bin.getVariableNameList().size() ; iVar++ ){
              if( not bin.isBetweenEdges(iVar, eventBuffer.getVarAsDouble(bin.getVariableNameList()[iVar])) ){
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

          // OK, now we have a valid fit bin. Let's claim an index.
//          eventOffSetMutex.lock();
          sampleEventIndex = _cache_.sampleIndexOffsetList[iSample]++;
//          eventOffSetMutex.unlock();

          eventPtr = &(*_cache_.sampleEventListPtrToFill[iSample])[sampleEventIndex];
          eventPtr->copyData(copyStoreDict, true); // buffer has the right size already

          eventPtr->setEntryIndex(iEntry);
          eventPtr->setSampleBinIndex(eventBuffer.getSampleBinIndex());
          eventPtr->setTreeWeight(eventBuffer.getTreeWeight());
          eventPtr->setNominalWeight(eventBuffer.getTreeWeight());
          eventPtr->resetEventWeight();

          // Now the event is ready. Let's index the dials:
          eventDialOffset = 0;
          // Loop over the parameter Sets (the ones which have a valid dialSet for this dataSet)
          for( auto& dialSetPair : _cache_.dialSetPtrMap ){
            for( iDialSet = 0 ; iDialSet < dialSetPair.second.size() ; iDialSet++ ){
              dialSetPtr = dialSetPair.second[iDialSet];

              if( dialSetPtr->getApplyConditionFormula() != nullptr ){
                if( eventBuffer.evalFormula(dialSetPtr->getApplyConditionFormula()) == 0 ){
                  // next dialSet
                  continue;
                }
              }

              if( not dialSetPtr->getDialLeafName().empty() ){
                // Event-by-event dial?
                if     ( not strcmp(treeChain.GetLeaf(dialSetPtr->getDialLeafName().c_str())->GetTypeName(), "TClonesArray") ){
                  grPtr = (TGraph*) eventBuffer.getVariable<TClonesArray*>(dialSetPtr->getDialLeafName())->At(0);
                  if(grPtr->GetN() > 1){
                    if     ( dialSetPtr->getGlobalDialType() == DialType::Spline ){
                      spDialPtr = (SplineDial*) dialSetPtr->getDialList()[iEntry].get();
                      dialSetPtr->applyGlobalParameters(spDialPtr);
                      spDialPtr->createSpline( grPtr );
                      spDialPtr->initialize();
                      spDialPtr->setIsReferenced(true);
                      // Adding dial in the event
                      eventPtr->getRawDialPtrList()[eventDialOffset++] = spDialPtr;
                    }
                    else if( dialSetPtr->getGlobalDialType() == DialType::Graph ){
                      grDialPtr = (GraphDial*) dialSetPtr->getDialList()[iEntry].get();
                      dialSetPtr->applyGlobalParameters(grDialPtr);
                      grDialPtr->setGraph(*grPtr);
                      grDialPtr->initialize();
                      grDialPtr->setIsReferenced(true);
                      // Adding dial in the event
                      eventPtr->getRawDialPtrList()[eventDialOffset++] = grDialPtr;
                    }
                    else{
                      LogThrow("Unsupported event-by-event dial: " << DialType::DialTypeEnumNamespace::toString(dialSetPtr->getGlobalDialType()))
                    }
                  }
                }
                else if( not strcmp(treeChain.GetLeaf(dialSetPtr->getDialLeafName().c_str())->GetTypeName(), "TGraph") ){
                  grPtr = (TGraph*) eventBuffer.getVariable<TGraph*>(dialSetPtr->getDialLeafName());
                  if     ( dialSetPtr->getGlobalDialType() == DialType::Spline ){
                    spDialPtr = (SplineDial*) dialSetPtr->getDialList()[iEntry].get();
                    dialSetPtr->applyGlobalParameters(spDialPtr);
                    spDialPtr->createSpline(grPtr);
                    spDialPtr->initialize();
                    spDialPtr->setIsReferenced(true);
                    // Adding dial in the event
                    eventPtr->getRawDialPtrList()[eventDialOffset++] = spDialPtr;
                  }
                  else if( dialSetPtr->getGlobalDialType() == DialType::Graph ){
                    grDialPtr = (GraphDial*) dialSetPtr->getDialList()[iEntry].get();
                    dialSetPtr->applyGlobalParameters(grDialPtr);
                    grDialPtr->setGraph(*grPtr);
                    grDialPtr->initialize();
                    grDialPtr->setIsReferenced(true);
                    // Adding dial in the event
                    eventPtr->getRawDialPtrList()[eventDialOffset++] = grDialPtr;
                  }
                  else{
                    LogThrow("Unsupported event-by-event dial: " << DialType::DialTypeEnumNamespace::toString(dialSetPtr->getGlobalDialType()))
                  }
                }
                else{
                  LogThrow("Unsupported event-by-event dial type: " << treeChain.GetLeaf(dialSetPtr->getDialLeafName().c_str())->GetTypeName() )
                }
              }
              else{
                // Binned dial?
                lastFailedBinVarIndex = -1;
                for( iDial = 0 ; iDial < dialSetPtr->getDialList().size(); iDial++ ){
                  // Let's give this dial a chance:
                  isEventInDialBin = true;

                  // ----------> SLOW PART -> check the bin
                  if( (applyConditionBinPtr = dialSetPtr->getDialList()[iDial]->getApplyConditionBinPtr()) != nullptr ){
                    if( lastFailedBinVarIndex != -1 // if the last bin failed, this is not -1
                        and applyConditionBinPtr->getEventVarIndexCache()[lastFailedBinVarIndex] == lastEventVarIndex // make sure this new bin-edges point to the same variable
                    ){
                      if( *lastEdges == applyConditionBinPtr->getEdgesList()[lastFailedBinVarIndex] ){ continue; } // same bin-edges! no need to check again!
                      else{ lastEdges = &applyConditionBinPtr->getEdgesList()[lastFailedBinVarIndex]; }
                      if( not DataBin::isBetweenEdges( *lastEdges, eventBuffer.getVarAsDouble(lastEventVarIndex) )){
                        continue;
                        // NEXT DIAL! Don't check other bin variables
                      }
                    }

                    // Check for the others
                    for( iVar = 0 ; iVar < applyConditionBinPtr->getEdgesList().size() ; iVar++ ){
                      if( iVar == lastFailedBinVarIndex ) continue; // already checked if set
                      lastEventVarIndex = applyConditionBinPtr->getEventVarIndexCache()[iVar];
                      lastEdges = &applyConditionBinPtr->getEdgesList()[iVar];
                      if( not DataBin::isBetweenEdges( *lastEdges,  eventBuffer.getVarAsDouble( lastEventVarIndex ) )){
                        isEventInDialBin = false;
                        lastFailedBinVarIndex = int(iVar);
                        break; // NEXT DIAL! Don't check other bin variables
                      }
                    } // Bin var loop
                  }

                  // <------------------
                  if( isEventInDialBin ) {
                    dialSetPtr->getDialList()[iDial]->setIsReferenced(true);
                    eventPtr->getRawDialPtrList()[eventDialOffset++] = dialSetPtr->getDialList()[iDial].get();
                    break;
                  }
                } // iDial

                if( isEventInDialBin and dialSetPair.first->isUseOnlyOneParameterPerEvent() ){
                  break;
                  // leave iDialSet (ie loop over parameters of the ParSet)
                }
              }

            } // iDialSet / Enabled-parameter
          } // ParSet / DialSet Pairs

          // Resize the dialRef list
          eventPtr->getRawDialPtrList().resize(eventDialOffset);
          eventPtr->getRawDialPtrList().shrink_to_fit();

        } // event has passed the selection?
      } // samples
    } // entries
    if( iThread_ == 0 ) GenericToolbox::displayProgressBar(nEvents, nEvents, progressTitle);
  };

  LogWarning << "Loading and indexing..." << std::endl;
  GlobalVariables::getParallelWorker().addJob(__METHOD_NAME__, fillFunction);
  GlobalVariables::getParallelWorker().runJob(__METHOD_NAME__);
  GlobalVariables::getParallelWorker().removeJob(__METHOD_NAME__);

  LogInfo << "Shrinking event lists..." << std::endl;
  for( size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
    auto* container = &_cache_.samplesToFillList[iSample]->getDataContainer();
    if(_parameters_.useMcContainer) container = &_cache_.samplesToFillList[iSample]->getMcContainer();
    container->shrinkEventList(_cache_.sampleIndexOffsetList[iSample]);
  }
}



