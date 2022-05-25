//
// Created by Adrien BLANCHET on 14/05/2022.
//

#include "DataDispenser.h"
#include "GlobalVariables.h"
#include "SplineDial.h"
#include "GraphDial.h"
#include "GraphDialLin.h"
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
})

DataDispenser::DataDispenser() = default;
DataDispenser::~DataDispenser() = default;

void DataDispenser::setConfig(const json &config) {
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
  _parameters_.nominalWeightFormulaStr = JsonUtils::fetchValue(_config_, "nominalWeightFormula", _parameters_.nominalWeightFormulaStr);
  _parameters_.additionalLeavesStorage = JsonUtils::fetchValue(_config_, "additionalLeavesStorage", _parameters_.additionalLeavesStorage);
  _parameters_.useMcContainer = JsonUtils::fetchValue(_config_, "useMcContainer", _parameters_.useMcContainer);

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
  LogWarning << "Loading data set: " << getTitle() << std::endl;
  LogThrowIf(not _isInitialized_, "Can't load while not initialized.");
  LogThrowIf(_sampleSetPtrToLoad_==nullptr, "SampleSet not specified.");

  _cache_.clear();

  this->buildSampleToFillList();
  if( _cache_.samplesToFillList.empty() ){
    LogError << "No samples were selected for data set: " << getTitle() << std::endl;
    return;
  }

  if( GenericToolbox::doesStringContainsSubstring(_parameters_.nominalWeightFormulaStr, "<I_TOY>") ){
    LogThrowIf(_parameters_.iThrow==-1, "<I_TOY> not set.");
    GenericToolbox::replaceSubstringInsideInputString(
        _parameters_.nominalWeightFormulaStr, "<I_TOY>",
        std::to_string(_parameters_.iThrow)
    );
  }

  if( not _parameters_.overrideLeafDict.empty() ){
    for( auto& entryDict : _parameters_.overrideLeafDict ){
      if( GenericToolbox::doesStringContainsSubstring(entryDict.second, "<I_TOY>") ){
        LogThrowIf(_parameters_.iThrow==-1, "<I_TOY> not set.");
        GenericToolbox::replaceSubstringInsideInputString(
            entryDict.second, "<I_TOY>",
            std::to_string(_parameters_.iThrow)
        );
      }
    }
    LogInfo << "Overriding leaf dict: " << std::endl;
    LogInfo << GenericToolbox::parseMapAsString(_parameters_.overrideLeafDict) << std::endl;
  }

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

TChain *DataDispenser::generateChain() {
  if( _parameters_.filePathList.empty() ) return nullptr;
  TChain* out{new TChain(_parameters_.treePath.c_str())};
  for( const auto& file: _parameters_.filePathList){
    LogThrowIf(not GenericToolbox::doesTFileIsValid(file, {_parameters_.treePath}), "Invalid file: " << file);
    out->Add(file.c_str());
  }
  return out;
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

  LogInfo << "Selected samples are: " << std::endl
          << GenericToolbox::iterableToString(
              _cache_.samplesToFillList,
              [](const FitSample *samplePtr){ return "\""+samplePtr->getName()+"\""; }
              )
          << std::endl;
}
void DataDispenser::doEventSelection(){
  LogWarning << "Performing event selection..." << std::endl;
  TChain* chainPtr{nullptr};

  LogInfo << "Opening files..." << std::endl;
  chainPtr = this->generateChain();
  LogThrowIf(chainPtr == nullptr, "Can't open TChain.");
  LogThrowIf(chainPtr->GetEntries() == 0, "TChain is empty.");

  LogInfo << "Defining selection formulas..." << std::endl;
  TTreeFormulaManager formulaManager; // TTreeFormulaManager handles the notification of multiple TTreeFormula for one TTChain
  std::vector<TTreeFormula*> sampleCutFormulaList;
  chainPtr->SetBranchStatus("*", true); // enabling every branch to define formula

  GenericToolbox::TablePrinter t;
  t.setColTitles({{"Sample"}, {"Selection Cut"}});
  for( auto& sample : _cache_.samplesToFillList ){
    t.addTableLine({{"\""+sample->getName()+"\""}, {"\""+sample->getSelectionCutsStr()+"\""}});
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
  t.printTable();

  LogInfo << "Enabling required branches..." << std::endl;
  chainPtr->SetBranchStatus("*", false);
  for( auto* sampleFormula : sampleCutFormulaList ){
    for( int iLeaf = 0 ; iLeaf < sampleFormula->GetNcodes() ; iLeaf++ ){
      chainPtr->SetBranchStatus(sampleFormula->GetLeaf(iLeaf)->GetBranch()->GetName(), true);
    }
  }

  LogInfo << "Performing event selection..." << std::endl;
  GenericToolbox::VariableMonitor readSpeed("bytes");
  Long64_t nEvents = chainPtr->GetEntries();
  // for each event, which sample is active?
  _cache_.eventIsInSamplesList.resize(nEvents, std::vector<bool>(_cache_.samplesToFillList.size(), true));
  std::string progressTitle = LogInfo.getPrefixString() + "Reading input dataset";
  TFile* lastFilePtr{nullptr};
  for( Long64_t iEvent = 0 ; iEvent < nEvents ; iEvent++ ){
    readSpeed.addQuantity(chainPtr->GetEntry(iEvent));
    if( GenericToolbox::showProgressBar(iEvent, nEvents) ){
      GenericToolbox::displayProgressBar(
          iEvent, nEvents,progressTitle + " - " +
                          GenericToolbox::padString(GenericToolbox::parseSizeUnits(readSpeed.evalTotalGrowthRate()), 8)
                          + "/s");
    }

    for( size_t iSample = 0 ; iSample < sampleCutFormulaList.size() ; iSample++ ){
      for(int jInstance = 0; jInstance < sampleCutFormulaList[iSample]->GetNdata(); jInstance++) {
        if (sampleCutFormulaList[iSample]->EvalInstance(jInstance) == 0) {
          // if it doesn't pass the cut
          _cache_.eventIsInSamplesList[iEvent][iSample] = false;
          break;
        }
      } // Formula Instances
    } // iSample
  } // iEvent

  // detaching the formulas
  chainPtr->SetNotify(nullptr);
  delete chainPtr;

  LogInfo << "Counting requested event slots for each samples..." << std::endl;
  _cache_.sampleNbOfEvents.resize(_cache_.samplesToFillList.size(), 0);
  for( size_t iEvent = 0 ; iEvent < _cache_.eventIsInSamplesList.size() ; iEvent++ ){
    for(size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
      if(_cache_.eventIsInSamplesList[iEvent][iSample]) _cache_.sampleNbOfEvents[iSample]++;
    }
  }

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
    for( auto& var : _plotGenPtr_->fetchListOfVarToPlot() ){
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
  TChain* chainPtr{this->generateChain()};
  chainPtr->SetBranchStatus("*", false);

  std::vector<std::string> leafVar;
  for( auto& eventVar : _cache_.leavesRequestedForStorage){
    leafVar.emplace_back(eventVar);
    if( GenericToolbox::doesKeyIsInMap(eventVar, _parameters_.overrideLeafDict) ){
      leafVar.back() = _parameters_.overrideLeafDict[eventVar];
      leafVar.back() = GenericToolbox::stripBracket(leafVar.back(), '[', ']');
    }
  }
  GenericToolbox::TreeEventBuffer tBuf;
  tBuf.setLeafNameList(leafVar);
  tBuf.hook(chainPtr);

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
              dialSetPtr->getDialList().resize(chainPtr->GetEntries(), DialWrapper(SplineDial()));
            }
            else if( dialType == DialType::Graph ){
              dialSetPtr->getDialList().resize(chainPtr->GetEntries(), DialWrapper(GraphDial()));
            }
            else if( dialType == DialType::GraphLin ){
              dialSetPtr->getDialList().resize(chainPtr->GetEntries(), DialWrapper(GraphDialLin()));
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

  delete chainPtr;
}
void DataDispenser::readAndFill(){
  LogWarning << "Reading data set and loading..." << std::endl;

  ROOT::EnableImplicitMT(GlobalVariables::getNbThreads());
  std::mutex eventOffSetMutex;
  auto fillFunction = [&](int iThread_){

    int nThreads = GlobalVariables::getNbThreads();
    if( iThread_ == -1 ){
      iThread_ = 0;
      nThreads = 1;
    }

    TChain* threadChain{this->generateChain()};
    TTreeFormula* threadNominalWeightFormula{nullptr};
    TList objToNotify;
    threadChain->SetNotify(&objToNotify);

    threadChain->SetBranchStatus("*", false);

    if( not _parameters_.nominalWeightFormulaStr.empty() ){
      threadChain->SetBranchStatus("*", true);
      if(iThread_ == 0) LogInfo << "Nominal weight: \"" << _parameters_.nominalWeightFormulaStr << "\"" << std::endl;
      threadNominalWeightFormula = new TTreeFormula(
          Form("NominalWeightFormula%i", iThread_),
          _parameters_.nominalWeightFormulaStr.c_str(),
          threadChain
          );
      LogThrowIf(threadNominalWeightFormula->GetNdim() == 0,
                 "\"" <<  _parameters_.nominalWeightFormulaStr << "\" could not be parsed by the TChain");
      objToNotify.Add(threadNominalWeightFormula);
      threadChain->SetBranchStatus("*", false);
      // Enabling needed branches for evaluating formulas
      for( int iLeaf = 0 ; iLeaf < threadNominalWeightFormula->GetNcodes() ; iLeaf++ ){
        threadChain->SetBranchStatus(threadNominalWeightFormula->GetLeaf(iLeaf)->GetBranch()->GetName(), true);
      }
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
    eventOffSetMutex.lock();
    tEventBuffer.hook(threadChain);
    eventOffSetMutex.unlock();

    PhysicsEvent eventBuffer;
    eventBuffer.setDataSetIndex(_owner_->getDataSetIndex());
    eventBuffer.setCommonLeafNameListPtr(std::make_shared<std::vector<std::string>>(_cache_.leavesRequestedForIndexing));
    auto copyDict = eventBuffer.generateDict(tEventBuffer, _parameters_.overrideLeafDict);
    eventBuffer.copyData(copyDict, true); // resize array obj

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
    int lastFailedBinVarIndex{-1};
    size_t iVar{0};
    size_t iSample{0};
    // Dials
    size_t eventDialOffset;
    DialSet* dialSetPtr;
    size_t iDialSet, iDial;
    TGraph* grPtr{nullptr};
    SplineDial* spDialPtr;
    GraphDial* grDialPtr;
    GraphDialLin* grLinDialPtr;
    const DataBin* applyConditionBinPtr;

    // Try to read TTree the closest to sequentially possible
    Long64_t nEvents = threadChain->GetEntries();
    Long64_t nEventPerThread = nEvents/Long64_t(nThreads);
    Long64_t iEnd = nEvents;
    Long64_t iStart = Long64_t(iThread_)*nEventPerThread;
    if( iThread_+1 != nThreads ) iEnd = (Long64_t(iThread_)+1)*nEventPerThread;
    Long64_t iGlobal = 0;

    // Load the branches
    threadChain->LoadTree(iStart);

    // IO speed monitor
    GenericToolbox::VariableMonitor readSpeed("bytes");
    Int_t nBytes;

    std::string progressTitle = LogInfo.getPrefixString() + "Loading and indexing";

    for(Long64_t iEntry = iStart ; iEntry < iEnd ; iEntry++ ){

      if( iThread_ == 0 ){
        if( GenericToolbox::showProgressBar(iGlobal, nEvents) ){
          GenericToolbox::displayProgressBar(
              iGlobal, nEvents,
              progressTitle + " - "
              + GenericToolbox::padString(GenericToolbox::parseSizeUnits(double(nThreads)*readSpeed.getTotalAccumulated()), 9)
              + " ("
              + GenericToolbox::padString(GenericToolbox::parseSizeUnits(double(nThreads)*readSpeed.evalTotalGrowthRate()), 9)
              + "/s)"
          );
        }
        iGlobal += nThreads;
      }

      bool skipEvent = true;
      for( bool isInSample : _cache_.eventIsInSamplesList[iEntry] ){
        if( isInSample ){ skipEvent = false; break; }
      }
      if( skipEvent ) continue;

//      threadChain->GetStatus()->Print();
//      for( int iElm=0 ;iElm <  threadChain->GetStatus()->GetEntries() ; iElm++ ){
//        auto* ce = (TChainElement*) threadChain->GetStatus()->At(iElm);
//        LogWarning << ce->GetName() << " -> " << ce->GetStatus() << std::endl;
//      }
//
//      for( int iBranch = 0 ; iBranch < threadChain->GetListOfLeaves()->GetEntries() ; iBranch++ ){
//        auto* l = threadChain->GetLeaf(threadChain->GetListOfLeaves()->At(iBranch)->GetName());
//        if( l == nullptr ) continue;
//        if( threadChain->GetBranchStatus(l->GetBranch()->GetName()) == 1 ){
//          LogTrace << std::string(l->GetFullName()) << " -> " << l->GetValuePointer() << " ? ";
//          if( tEventBuffer.fetchLeafIndex(l->GetName()) == -1 ){
//            LogTrace << "NO" << std::endl;
//          }
//          else{
//            LogTrace << (int*) &(tEventBuffer.getLeafContent(l->GetName()).getByteBuffer()[0]) << std::endl;
//          }
//        }
//      }

//      LogDebug<< "ENTRY?" << std::endl;
      nBytes = threadChain->GetEntry(iEntry);
//      LogDebug << tEventBuffer.getSummary() << std::endl;
//      eventBuffer.copyData(copyDict, true);
//      LogDebug << eventBuffer << std::endl;
//      LogThrow("")
      if( iThread_ == 0 ) readSpeed.addQuantity(nBytes);

      if( threadNominalWeightFormula != nullptr ){
        eventBuffer.setTreeWeight(threadNominalWeightFormula->EvalInstance());
        if( eventBuffer.getTreeWeight() == 0 ){
//          LogTrace << std::endl << "DONT PASS CUT:" << threadNominalWeightFormula->GetExpFormula() << std::endl;
//          for( int iLeaf = 0 ; iLeaf < threadNominalWeightFormula->GetNcodes() ; iLeaf++ ){
//            LogDebug << "branch:" << threadNominalWeightFormula->GetLeaf(iLeaf)->GetFullName();
//            for( int i=0 ; i < threadNominalWeightFormula->GetLeaf(iLeaf)->GetNdata(); i++){
//              LogDebug << " -> " << threadNominalWeightFormula->GetLeaf(iLeaf)->GetValue(i);
//            }
//            LogDebug << std::endl;
//            LogThrowIf(not threadChain->GetBranchStatus(threadNominalWeightFormula->GetLeaf(iLeaf)->GetBranch()->GetName()),
//                       threadNominalWeightFormula->GetLeaf(iLeaf)->GetBranch()->GetName() << "DISABLE?")
////            threadChain->SetBranchStatus(threadNominalWeightFormula->GetLeaf(iLeaf)->GetBranch()->GetName(), true);
//          }
//          LogWarning << threadNominalWeightFormula->EvalInstance() << std::endl;
//          eventBuffer.copyData(copyDict, true);
//          LogThrow(eventBuffer)
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
          eventOffSetMutex.lock();
          sampleEventIndex = _cache_.sampleIndexOffsetList[iSample]++;
          eventOffSetMutex.unlock();

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
                if     ( not strcmp(threadChain->GetLeaf(dialSetPtr->getDialLeafName().c_str())->GetTypeName(), "TClonesArray") ){
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
                    else if( dialSetPtr->getGlobalDialType() == DialType::GraphLin ){
                      grLinDialPtr = (GraphDialLin*) dialSetPtr->getDialList()[iEntry].get();
                      dialSetPtr->applyGlobalParameters(grLinDialPtr);
                      grLinDialPtr->setGraph(*grPtr);
                      grLinDialPtr->initialize();
                      grLinDialPtr->setIsReferenced(true);
                      // Adding dial in the event
                      eventPtr->getRawDialPtrList()[eventDialOffset++] = grLinDialPtr;
                    }
                    else{
                      LogThrow("Unsupported event-by-event dial: " << DialType::DialTypeEnumNamespace::toString(dialSetPtr->getGlobalDialType()))
                    }
                  }
                }
                else if( not strcmp(threadChain->GetLeaf(dialSetPtr->getDialLeafName().c_str())->GetTypeName(), "TGraph") ){
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
                  else if( dialSetPtr->getGlobalDialType() == DialType::GraphLin ){
                    grLinDialPtr = (GraphDialLin*) dialSetPtr->getDialList()[iEntry].get();
                    dialSetPtr->applyGlobalParameters(grLinDialPtr);
                    grLinDialPtr->setGraph(*grPtr);
                    grLinDialPtr->initialize();
                    grLinDialPtr->setIsReferenced(true);
                    // Adding dial in the event
                    eventPtr->getRawDialPtrList()[eventDialOffset++] = grLinDialPtr;
                  }
                  else{
                    LogThrow("Unsupported event-by-event dial: " << DialType::DialTypeEnumNamespace::toString(dialSetPtr->getGlobalDialType()))
                  }
                }
                else{
                  LogThrow("Unsupported event-by-event dial type: " << threadChain->GetLeaf(dialSetPtr->getDialLeafName().c_str())->GetTypeName() )
                }
              }
              else{
                // Binned dial?
                lastFailedBinVarIndex = -1;
                for( iDial = 0 ; iDial < dialSetPtr->getDialList().size(); iDial++ ){
                  // ----------> SLOW PART
                  applyConditionBinPtr = dialSetPtr->getDialList()[iDial]->getApplyConditionBinPtr();

                  if( applyConditionBinPtr != nullptr and lastFailedBinVarIndex != -1 ){
                    if( not applyConditionBinPtr->isBetweenEdges(
                        applyConditionBinPtr->getEdgesList()[lastFailedBinVarIndex],
                        eventBuffer.getVarAsDouble(applyConditionBinPtr->getEventVarIndexCache()[lastFailedBinVarIndex] )
                    )){
                      continue;
                      // NEXT DIAL! Don't check other bin variables
                    }
                  }

                  // Ok, lets give this dial a chance:
                  isEventInDialBin = true;

                  if( applyConditionBinPtr != nullptr ){
                    for( iVar = 0 ; iVar < applyConditionBinPtr->getEdgesList().size() ; iVar++ ){
                      if( iVar == lastFailedBinVarIndex ) continue; // already checked if set
                      if( not applyConditionBinPtr->isBetweenEdges(
                          applyConditionBinPtr->getEdgesList()[iVar],
                          eventBuffer.getVarAsDouble(applyConditionBinPtr->getEventVarIndexCache()[iVar] )
                      )){
                        isEventInDialBin = false;
                        lastFailedBinVarIndex = int(iVar);
                        break;
                        // NEXT DIAL! Don't check other bin variables
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
    delete threadChain;
    delete threadNominalWeightFormula;
  };

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



