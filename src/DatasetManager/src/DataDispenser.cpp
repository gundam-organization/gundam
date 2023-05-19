//
// Created by Adrien BLANCHET on 14/05/2022.
//

#include "EventVarTransform.h"
#include "DataDispenser.h"
#include "GlobalVariables.h"
#include "DatasetLoader.h"
#include "GenericToolbox.Json.h"
#include "Misc.h"

#if USE_NEW_DIALS
#include "DialCollection.h"
#include "DialTypes.h"
#else
#include "SplineDial.h"
#include "GraphDial.h"
#endif

#include "GenericToolbox.Root.TreeEntryBuffer.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.VariablesMonitor.h"
#include "Logger.h"

#include "TTreeFormulaManager.h"
#include "TChain.h"
#include "TChainElement.h"

#include "sstream"
#include "string"
#include "vector"


LoggerInit([]{
  Logger::setUserHeaderStr("[DataDispenser]");
});


DataDispenser::DataDispenser(DatasetLoader* owner_): _owner_(owner_) {}

void DataDispenser::readConfigImpl(){
  LogThrowIf( _config_.empty(), "Config is not set." );

  _parameters_.name = GenericToolbox::Json::fetchValue<std::string>(_config_, "name", _parameters_.name);
  _parameters_.treePath = GenericToolbox::Json::fetchValue<std::string>(_config_, "tree", _parameters_.treePath);
  _parameters_.filePathList = GenericToolbox::Json::fetchValue<std::vector<std::string>>(_config_, "filePathList", _parameters_.filePathList);
  _parameters_.additionalVarsStorage = GenericToolbox::Json::fetchValue(_config_, {{"additionalLeavesStorage"}, {"additionalVarsStorage"}}, _parameters_.additionalVarsStorage);
  _parameters_.dummyVariablesList = GenericToolbox::Json::fetchValue(_config_, "dummyVariablesList", _parameters_.dummyVariablesList);
  _parameters_.useMcContainer = GenericToolbox::Json::fetchValue(_config_, "useMcContainer", _parameters_.useMcContainer);

  _parameters_.selectionCutFormulaStr = GenericToolbox::Json::buildFormula(_config_, "selectionCutFormula", "&&", _parameters_.selectionCutFormulaStr);
  _parameters_.nominalWeightFormulaStr = GenericToolbox::Json::buildFormula(_config_, "nominalWeightFormula", "*", _parameters_.nominalWeightFormulaStr);

  _parameters_.overrideLeafDict.clear();
  for( auto& entry : GenericToolbox::Json::fetchValue(_config_, "overrideLeafDict", nlohmann::json()) ){
    _parameters_.overrideLeafDict[entry["eventVar"]] = entry["leafVar"];
  }
}
void DataDispenser::initializeImpl(){
  // Nothing else to do other than read config?
  LogWarning << "Initialized data dispenser: " << getTitle() << std::endl;
}

const DataDispenserParameters &DataDispenser::getParameters() const {
  return _parameters_;
}
DataDispenserParameters &DataDispenser::getParameters() {
  return _parameters_;
}

void DataDispenser::setSampleSetPtrToLoad(FitSampleSet *sampleSetPtrToLoad) {
  _sampleSetPtrToLoad_ = sampleSetPtrToLoad;
}
void DataDispenser::setParSetPtrToLoad(std::vector<FitParameterSet> *parSetListPtrToLoad_) {
  _parSetListPtrToLoad_ = parSetListPtrToLoad_;
}
void DataDispenser::setDialCollectionListPtr(std::vector<DialCollection> *dialCollectionListPtr) {
  _dialCollectionListPtr_ = dialCollectionListPtr;
}
void DataDispenser::setPlotGenPtr(PlotGenerator *plotGenPtr) {
  _plotGenPtr_ = plotGenPtr;
}
void DataDispenser::setEventDialCache(EventDialCache* eventDialCache_) {
  _eventDialCacheRef_ = eventDialCache_;
}

void DataDispenser::load(){
  LogWarning << "Loading dataset: " << getTitle() << std::endl;
  LogThrowIf(not this->isInitialized(), "Can't load while not initialized.");
  LogThrowIf(_sampleSetPtrToLoad_==nullptr, "SampleSet not specified.");

  if( GlobalVariables::getVerboseLevel() >= MORE_PRINTOUT ){
    LogDebug << "Configuration: " << _parameters_.getSummary() << std::endl;
  }

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
    for( auto& replaceEntry : _cache_.varsToOverrideList ){
      GenericToolbox::replaceSubstringInsideInputString(formula_, replaceEntry, _parameters_.overrideLeafDict[replaceEntry]);
    }
  };

  if( not _parameters_.overrideLeafDict.empty() ){
    for( auto& entryDict : _parameters_.overrideLeafDict ){ replaceToyIndexFct(entryDict.second); }
    LogInfo << "Overriding leaf dict: " << std::endl;
    LogInfo << GenericToolbox::parseMapAsString(_parameters_.overrideLeafDict) << std::endl;

    for( auto& overrideEntry : _parameters_.overrideLeafDict ){
      _cache_.varsToOverrideList.emplace_back(overrideEntry.first);
    }
    // make sure we process the longest words first: "thisIsATest" variable should be replaced before "thisIs"
    std::function<bool(const std::string&, const std::string&)> aGoesFirst =
        [](const std::string& a_, const std::string& b_){ return a_.size() > b_.size(); };
    GenericToolbox::sortVector(_cache_.varsToOverrideList, aGoesFirst);
  }

  if( GenericToolbox::Json::doKeyExist(_config_, "variablesTransform") ){
    // load transformations
    int index{0};
    for( auto& varTransform : GenericToolbox::Json::fetchValue<std::vector<nlohmann::json>>(_config_, "variablesTransform") ){
      _cache_.eventVarTransformList.emplace_back( varTransform );
      _cache_.eventVarTransformList.back().setIndex(index++);
      _cache_.eventVarTransformList.back().initialize();
    }
    // sort them according to their output
    std::function<bool(const EventVarTransformLib&, const EventVarTransformLib&)> aGoesFirst =
        [](const EventVarTransformLib& a_, const EventVarTransformLib& b_){
          // does a_ is a self transformation? -> if yes, don't change the order
          if( GenericToolbox::doesElementIsInVector(a_.getOutputVariableName(), a_.fetchRequestedVars()) ){ return false; }
          // does b_ transformation needs a_ output? -> if yes, a needs to go first
          if( GenericToolbox::doesElementIsInVector(a_.getOutputVariableName(), b_.fetchRequestedVars()) ){ return true; }
          // otherwise keep the order from the declaration
          if( a_.getIndex() < b_.getIndex() ) return true;
          // default -> won't change the order
          return false;
        };
    GenericToolbox::sortVector(_cache_.eventVarTransformList, aGoesFirst);
  }


  replaceToyIndexFct(_parameters_.nominalWeightFormulaStr);
  replaceToyIndexFct(_parameters_.selectionCutFormulaStr);

  overrideLeavesNamesFct(_parameters_.nominalWeightFormulaStr);
  overrideLeavesNamesFct(_parameters_.selectionCutFormulaStr);

  LogInfo << "Data will be extracted from: " << GenericToolbox::parseVectorAsString(_parameters_.filePathList, true) << std::endl;
  for( const auto& file: _parameters_.filePathList){
    std::string path = GenericToolbox::expandEnvironmentVariables(file);
    LogThrowIf(not GenericToolbox::doesTFileIsValid(path, {_parameters_.treePath}), "Invalid file: " << path);
  }

#if USE_NEW_DIALS
  if( _dialCollectionListPtr_ != nullptr ){
    for( auto& dialCollection : *_dialCollectionListPtr_ ){
      if( dialCollection.isDatasetValid( _owner_->getName() ) ){
        _cache_.dialCollectionsRefList.emplace_back( &dialCollection );
      }
    }
  }
#endif

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

  ROOT::EnableThreadSafety();
  int nThreads = GlobalVariables::getNbThreads();
  std::vector<std::vector<std::vector<bool>>> perThreadEventIsInSamplesList(nThreads);
  std::vector<std::vector<size_t>> perThreadSampleNbOfEvents(nThreads);
  auto selectionFct = [&](int iThread_){

    if( iThread_ == -1 ){
      iThread_ = 0;
      nThreads = 1;
    }

    GlobalVariables::getThreadMutex().lock();
    TChain treeChain(_parameters_.treePath.c_str());
    for (const auto &file: _parameters_.filePathList) {
      std::string name = GenericToolbox::expandEnvironmentVariables(file);
      if (name != file and iThread_ == 0) {
        LogWarning << "Filename expanded to: " << name << std::endl;
      }
      treeChain.Add(name.c_str());
    }
    LogThrowIf(treeChain.GetEntries() == 0, "TChain is empty.");

    if( iThread_ == 0 ) LogInfo << "Defining selection formulas..." << std::endl;
    treeChain.SetBranchStatus("*", true); // enabling every branch to define formula

    TTreeFormula *treeSelectionCutFormula{nullptr};
    std::vector<TTreeFormula *> sampleCutFormulaList(_cache_.samplesToFillList.size(), nullptr);
    TTreeFormulaManager formulaManager; // TTreeFormulaManager handles the notification of multiple TTreeFormula for one TTChain

    if ( not _parameters_.selectionCutFormulaStr.empty() ) {
      treeSelectionCutFormula = new TTreeFormula(
          "SelectionCutFormula",
          _parameters_.selectionCutFormulaStr.c_str(),
          &treeChain
      );

      // ROOT Hot fix: https://root-forum.cern.ch/t/ttreeformula-evalinstance-return-0-0/16366/10
      treeSelectionCutFormula->GetNdata();

      LogThrowIf(treeSelectionCutFormula->GetNdim() == 0,
                 "\"" << _parameters_.selectionCutFormulaStr << "\" could not be parsed by the TChain");

      // The TChain will notify the formula that it has to update leaves addresses while swaping TFile
      formulaManager.Add(treeSelectionCutFormula);
      if(iThread_==0) LogInfo << "Using tree selection cut: \"" << _parameters_.selectionCutFormulaStr << "\"" << std::endl;
    }

    GenericToolbox::TablePrinter t;
    t.setColTitles({{"Sample"}, {"Selection Cut"}});
    for (size_t iSample = 0; iSample < _cache_.samplesToFillList.size(); iSample++) {

      std::string selectionCut = _cache_.samplesToFillList[iSample]->getSelectionCutsStr();
      for (auto &replaceEntry: _cache_.varsToOverrideList) {
        GenericToolbox::replaceSubstringInsideInputString(selectionCut, replaceEntry,
                                                          _parameters_.overrideLeafDict[replaceEntry]);
      }

      t.addTableLine({{"\"" + _cache_.samplesToFillList[iSample]->getName() + "\""},
                      {"\"" + selectionCut + "\""}});

      if( selectionCut.empty() ) continue;

      sampleCutFormulaList[iSample] = new TTreeFormula(_cache_.samplesToFillList[iSample]->getName().c_str(),
                                                       selectionCut.c_str(), &treeChain);

      // ROOT Hot fix: https://root-forum.cern.ch/t/ttreeformula-evalinstance-return-0-0/16366/10
      sampleCutFormulaList[iSample]->GetNdata();

      LogThrowIf(sampleCutFormulaList[iSample]->GetNdim() == 0,
                 "\"" << selectionCut << "\" could not be parsed by the TChain");

      // The TChain will notify the formula that it has to update leaves addresses while swaping TFile
      formulaManager.Add(sampleCutFormulaList[iSample]);
    }
    treeChain.SetNotify(&formulaManager);
    if(iThread_==0) t.printTable();


    if(iThread_ == 0) LogInfo << "Enabling required branches..." << std::endl;
    treeChain.SetBranchStatus("*", false);

    if (treeSelectionCutFormula != nullptr) GenericToolbox::enableSelectedBranches(&treeChain, treeSelectionCutFormula);
    for (auto &sampleFormula: sampleCutFormulaList) {
      if( sampleFormula == nullptr ) continue;
      GenericToolbox::enableSelectedBranches(&treeChain, sampleFormula);
    }


    GenericToolbox::VariableMonitor readSpeed("bytes");

    Long64_t nEvents = treeChain.GetEntries();
    Long64_t nEventPerThread = nEvents/Long64_t(nThreads);
    Long64_t iEnd = nEvents;
    Long64_t iStart = Long64_t(iThread_)*nEventPerThread;
    if( iThread_+1 != nThreads ) iEnd = (Long64_t(iThread_)+1)*nEventPerThread;
    Long64_t iGlobal = 0;

    // Load the branches
    treeChain.LoadTree(iStart);

    // for each event, which sample is active?
    perThreadEventIsInSamplesList[iThread_].resize(nEvents, std::vector<bool>(_cache_.samplesToFillList.size(), true));
    perThreadSampleNbOfEvents[iThread_].resize(_cache_.samplesToFillList.size(), 0);
    std::string progressTitle = "Performing event selection on " + this->getTitle() + "...";
    std::stringstream ssProgressTitle;
    TFile *lastFilePtr{nullptr};
    GlobalVariables::getThreadMutex().unlock();
    for ( Long64_t iEntry = iStart ; iEntry < iEnd ; iEntry++ ) {
      if( iThread_ == 0 ){
        readSpeed.addQuantity(treeChain.GetEntry(iEntry)*nThreads);
        if (GenericToolbox::showProgressBar(iGlobal, nEvents)) {
          ssProgressTitle.str("");

          ssProgressTitle << LogInfo.getPrefixString() << "Read from disk: "
                          << GenericToolbox::padString(GenericToolbox::parseSizeUnits(readSpeed.getTotalAccumulated()), 8) << " ("
                          << GenericToolbox::padString(GenericToolbox::parseSizeUnits(readSpeed.evalTotalGrowthRate()), 8) << "/s)";

          int cpuPercent = int(GenericToolbox::getCpuUsageByProcess());
          ssProgressTitle << " / CPU efficiency: " << GenericToolbox::padString(std::to_string(cpuPercent/nThreads), 3,' ')
                          << "%" << std::endl;

          ssProgressTitle << LogInfo.getPrefixString() << progressTitle;
          GenericToolbox::displayProgressBar(iGlobal, nEvents, ssProgressTitle.str());
        }
        iGlobal += nThreads;
      }
      else{
        treeChain.GetEntry(iEntry);
      }

      if (treeSelectionCutFormula != nullptr and not GenericToolbox::doesEntryPassCut(treeSelectionCutFormula)) {
        for (size_t iSample = 0; iSample < sampleCutFormulaList.size(); iSample++) {
          perThreadEventIsInSamplesList[iThread_][iEntry][iSample] = false;
        }
        if (GlobalVariables::getVerboseLevel() == INLOOP_TRACE) {
          LogTrace << "Event #" << treeChain.GetFileNumber() << ":" << treeChain.GetReadEntry()
                   << " rejected because of " << treeSelectionCutFormula->GetExpFormula() << std::endl;
        }
        continue;
      }

      for (size_t iSample = 0; iSample < sampleCutFormulaList.size(); iSample++) {

        if( sampleCutFormulaList[iSample] == nullptr ){
          perThreadSampleNbOfEvents[iThread_][iSample]++;
          if (GlobalVariables::getVerboseLevel() == INLOOP_TRACE) {
            LogDebug << "Event #" << treeChain.GetFileNumber() << ":" << treeChain.GetReadEntry()
                     << " included as sample " << iSample << " (NO SELECTION CUT)" << std::endl;
          }
          continue;
        }

        if ( not GenericToolbox::doesEntryPassCut(sampleCutFormulaList[iSample])) {
          perThreadEventIsInSamplesList[iThread_][iEntry][iSample] = false;
          if (GlobalVariables::getVerboseLevel() == INLOOP_TRACE) {
            LogTrace << "Event #" << treeChain.GetFileNumber() << ":" << treeChain.GetReadEntry()
                     << " rejected as sample " << iSample << " because of "
                     << sampleCutFormulaList[iSample]->GetExpFormula() << std::endl;
          }
        } else {
          perThreadSampleNbOfEvents[iThread_][iSample]++;
          if (GlobalVariables::getVerboseLevel() == INLOOP_TRACE) {
            LogDebug << "Event #" << treeChain.GetFileNumber() << ":" << treeChain.GetReadEntry()
                     << " included as sample " << iSample << " using "
                     << sampleCutFormulaList[iSample]->GetExpFormula() << std::endl;
          }
        }
      } // iSample
    } // iEvent
    if( iThread_ == 0 ){ GenericToolbox::displayProgressBar(nEvents, nEvents, ssProgressTitle.str()); }
  };

  LogInfo << "Event selection..." << std::endl;
  GlobalVariables::getParallelWorker().addJob(__METHOD_NAME__, selectionFct);
  GlobalVariables::getParallelWorker().runJob(__METHOD_NAME__);
  GlobalVariables::getParallelWorker().removeJob(__METHOD_NAME__);

  LogInfo << "Merging thread results" << std::endl;
  _cache_.sampleNbOfEvents.resize(_cache_.samplesToFillList.size(), 0);
  for( size_t iThread = 0 ; iThread < nThreads ; iThread++ ){
    if( _cache_.eventIsInSamplesList.empty() ){
      _cache_.eventIsInSamplesList.resize(perThreadEventIsInSamplesList[iThread].size(), std::vector<bool>(_cache_.samplesToFillList.size(), true));
    }
    for( size_t iEntry = 0 ; iEntry < perThreadEventIsInSamplesList[iThread].size() ; iEntry++ ){
      for( size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
        if(not perThreadEventIsInSamplesList[iThread][iEntry][iSample]) _cache_.eventIsInSamplesList[iEntry][iSample] = false;
      }
    }
    for( size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
      _cache_.sampleNbOfEvents[iSample] += perThreadSampleNbOfEvents[iThread][iSample];
    }
  }

  if( _owner_->isShowSelectedEventCount() ){
    LogWarning << "Events passing selection cuts:" << std::endl;
    GenericToolbox::TablePrinter t;
    t.setColTitles({{"Sample"}, {"# of events"}});
    for(size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
      t.addTableLine({{"\""+_cache_.samplesToFillList[iSample]->getName()+"\""}, std::to_string(_cache_.sampleNbOfEvents[iSample])});
    }
    t.printTable();
  }

}
void DataDispenser::fetchRequestedLeaves(){
  LogWarning << "Poll every objects for requested variables..." << std::endl;

#if USE_NEW_DIALS
  if( not _cache_.dialCollectionsRefList.empty() ) {
    std::vector<std::string> indexRequests;
    for( auto& dialCollection : _cache_.dialCollectionsRefList ) {
      if( dialCollection->getApplyConditionFormula() != nullptr ) {
        for( int iPar = 0 ; iPar < dialCollection->getApplyConditionFormula()->GetNpar() ; iPar++ ){
          GenericToolbox::addIfNotInVector(dialCollection->getApplyConditionFormula()->GetParName(iPar), indexRequests);
        }
      }
      if( not dialCollection->getGlobalDialLeafName().empty() ){
        GenericToolbox::addIfNotInVector(dialCollection->getGlobalDialLeafName(), indexRequests);
      }
      for( auto& bin : dialCollection->getDialBinSet().getBinsList() ) {
        for( auto& var : bin.getVariableNameList() ) {
          GenericToolbox::addIfNotInVector(var, indexRequests);
        }
      }
    }
    LogInfo << "DialCollection requests for indexing: " << GenericToolbox::parseVectorAsString(indexRequests) << std::endl;
    for( auto& var : indexRequests ){ _cache_.addVarRequestedForIndexing(var); }
  }
#else
  // parSet
  if( _parSetListPtrToLoad_ != nullptr ){
    std::vector<std::string> indexRequests;
    for( auto& parSet : *_parSetListPtrToLoad_ ){
      if( not parSet.isEnabled() ) continue;

      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ) continue;

        auto* dialSetPtr = par.findDialSet( _owner_->getName() );
        if( dialSetPtr == nullptr ){ continue; }

        if( not dialSetPtr->getDialLeafName().empty() ){
          GenericToolbox::addIfNotInVector(dialSetPtr->getDialLeafName(), indexRequests);
        }
        else{
          if( dialSetPtr->getApplyConditionFormula() != nullptr ){
            for( int iPar = 0 ; iPar < dialSetPtr->getApplyConditionFormula()->GetNpar() ; iPar++ ){
              GenericToolbox::addIfNotInVector(dialSetPtr->getApplyConditionFormula()->GetParName(iPar), indexRequests);
            }
          }

          for( auto& dial : dialSetPtr->getDialList() ){
            if( dial->getApplyConditionBinPtr() != nullptr ){
              for( auto& var : dial->getApplyConditionBinPtr()->getVariableNameList() ){
                GenericToolbox::addIfNotInVector(var, indexRequests);
              } // var
            }
          } // dial
        }
      } // par
    } // parSet

    LogInfo << "ParameterSets requests for indexing: " << GenericToolbox::parseVectorAsString(indexRequests) << std::endl;
    for( auto& var : indexRequests ){ _cache_.addVarRequestedForIndexing(var); }
  }
#endif

  // sample binning
  if( _sampleSetPtrToLoad_ != nullptr ){
    std::vector<std::string> indexRequests;
    for (auto &sample: _sampleSetPtrToLoad_->getFitSampleList()) {
      for (auto &bin: sample.getBinning().getBinsList()) {
        for (auto &var: bin.getVariableNameList()) {
          GenericToolbox::addIfNotInVector(var, indexRequests);
        }
      }
    }
    LogInfo << "Samples requests for indexing: " << GenericToolbox::parseVectorAsString(indexRequests) << std::endl;
    for( auto& var : indexRequests ){ _cache_.addVarRequestedForIndexing(var); }
  }

  // plotGen
  if( _plotGenPtr_ != nullptr ){
    std::vector<std::string> storeRequests;
    for( auto& var : _plotGenPtr_->fetchListOfVarToPlot(not _parameters_.useMcContainer) ){
      GenericToolbox::addIfNotInVector(var, storeRequests);
    }

    if( _parameters_.useMcContainer ){
      for( auto& var : _plotGenPtr_->fetchListOfSplitVarNames() ){
        GenericToolbox::addIfNotInVector(var, storeRequests);
      }
    }

    LogInfo << "PlotGenerator requests for storage:" << GenericToolbox::parseVectorAsString(storeRequests) << std::endl;
    for (auto &var: storeRequests) { _cache_.addVarRequestedForStorage(var); }
  }

  // storage requested by user
  {
    std::vector<std::string> storeRequests;
    for (auto &additionalLeaf: _parameters_.additionalVarsStorage) {
      GenericToolbox::addIfNotInVector(additionalLeaf, storeRequests);
    }
    LogInfo << "Dataset additional requests for storage:" << GenericToolbox::parseVectorAsString(storeRequests) << std::endl;
    for (auto &var: storeRequests) { _cache_.addVarRequestedForStorage(var); }
  }

  // fit sample set storage requests
  if( _sampleSetPtrToLoad_ != nullptr ){
    std::vector<std::string> storeRequests;
    for (auto &var: _sampleSetPtrToLoad_->getAdditionalVariablesForStorage()) {
      GenericToolbox::addIfNotInVector(var, storeRequests);
    }
    LogInfo << "SampleSet additional request for storage:" << GenericToolbox::parseVectorAsString(storeRequests) << std::endl;
    for (auto &var: storeRequests) { _cache_.addVarRequestedForStorage(var); }
  }

  // transforms inputs
  if( not _cache_.eventVarTransformList.empty() ){
    std::vector<std::string> indexRequests;
    for( int iTrans = int(_cache_.eventVarTransformList.size())-1 ; iTrans >= 0 ; iTrans-- ){
      // in reverse order -> Treat the highest level vars first (they might need lower level variables)
      std::string outVarName = _cache_.eventVarTransformList[iTrans].getOutputVariableName();
      if( GenericToolbox::doesElementIsInVector( outVarName, _cache_.varsRequestedForIndexing )
          or GenericToolbox::doesElementIsInVector( outVarName, indexRequests )
      ){
        // ok it is needed -> activate dependencies
        for( auto& var: _cache_.eventVarTransformList[iTrans].fetchRequestedVars() ){
          GenericToolbox::addIfNotInVector(var, indexRequests);
        }
      }
    }

    LogInfo << "EventVariableTransformation requests for indexing: " << GenericToolbox::parseVectorAsString(indexRequests) << std::endl;
    for( auto& var : indexRequests ){ _cache_.addVarRequestedForIndexing(var); }
  }

  LogInfo << "Vars requested for indexing: " << GenericToolbox::parseVectorAsString(_cache_.varsRequestedForIndexing, false) << std::endl;
  LogInfo << "Vars requested for storage: " << GenericToolbox::parseVectorAsString(_cache_.varsRequestedForStorage, false) << std::endl;

  // Now build the var to leaf translation
  for( auto& var : _cache_.varsRequestedForIndexing ){
    _cache_.varToLeafDict[var].first = var;    // default is the same name
    _cache_.varToLeafDict[var].second = false; // is dummy branch?

    // strip brackets
    _cache_.varToLeafDict[var].first = GenericToolbox::stripBracket(_cache_.varToLeafDict[var].first, '[', ']');

    // look for override requests
    if( GenericToolbox::doesKeyIsInMap(_cache_.varToLeafDict[var].first, _parameters_.overrideLeafDict) ){
      // leafVar will actually be the overrided leaf name while event will keep the original name
      _cache_.varToLeafDict[var].first = _parameters_.overrideLeafDict[_cache_.varToLeafDict[var].first];
      _cache_.varToLeafDict[var].first = GenericToolbox::stripBracket(_cache_.varToLeafDict[var].first, '[', ']');
    }

    // possible dummy ?
    // [OUT] variables only
    // [OUT] not requested by its inputs
    for( auto& varTransform : _cache_.eventVarTransformList ){
      std::string outVarName = varTransform.getOutputVariableName();
      if( outVarName != var ) continue;
      if( GenericToolbox::doesElementIsInVector(outVarName, varTransform.fetchRequestedVars()) ) continue;
      _cache_.varToLeafDict[var].second = true;
      break;
    }
  }

//  GenericToolbox::TablePrinter t;
//  t.setColTitles({"Variable name", "Leaf name", "From VarTransform?"});
//  for( auto& varToLeafDictEntry : _cache_.varToLeafDict ){
//    std::string colorCode{};
//    if( GenericToolbox::doesElementIsInVector(varToLeafDictEntry.first, _cache_.varsRequestedForStorage) ){ colorCode = GenericToolbox::ColorCodes::blueBackground; }
//    t.addTableLine({varToLeafDictEntry.first, varToLeafDictEntry.second.first, std::to_string(varToLeafDictEntry.second.second)}, colorCode);
//  }
//  t.printTable();

}
void DataDispenser::preAllocateMemory(){
  LogInfo << "Pre-allocating memory..." << std::endl;
  /// \brief The following lines are necessary since the events might get resized while being in multithread
  /// Because std::vector is insuring continuous memory allocation, a resize sometimes
  /// lead to the full moving of a vector memory. This is not thread safe, so better ensure
  /// the vector won't have to do this by allocating the right event size.

  // MEMORY CLAIM?
  TChain treeChain(_parameters_.treePath.c_str());
  for( const auto& file: _parameters_.filePathList){
    std::string name = GenericToolbox::expandEnvironmentVariables(file);
    if (name != file) {
      LogWarning << "Filename expanded to: " << name << std::endl;
    }
    treeChain.Add(name.c_str());
  }
  treeChain.SetBranchStatus("*", false);

  // Just a placeholder for creating the dictionary
  auto tBuf = this->generateTreeEventBuffer(&treeChain, _cache_.varsRequestedForStorage);

  PhysicsEvent eventPlaceholder;
  eventPlaceholder.setDataSetIndex(_owner_->getDataSetIndex());
  eventPlaceholder.setCommonLeafNameListPtr(std::make_shared<std::vector<std::string>>(_cache_.varsRequestedForStorage));
  auto copyDict = eventPlaceholder.generateDict(tBuf, _parameters_.overrideLeafDict);
  eventPlaceholder.copyData(copyDict);

#if USE_NEW_DIALS
#else
  if( _parSetListPtrToLoad_ != nullptr ){
    size_t dialCacheSize = 0;
    for( auto& parSet : *_parSetListPtrToLoad_ ){
      parSet.isUseOnlyOneParameterPerEvent() ? dialCacheSize++: dialCacheSize += parSet.getNbParameters();
    }
    eventPlaceholder.getRawDialPtrList().resize(dialCacheSize);
  }
#endif

  _cache_.sampleIndexOffsetList.resize(_cache_.samplesToFillList.size());
  _cache_.sampleEventListPtrToFill.resize(_cache_.samplesToFillList.size());
  for( size_t iSample = 0 ; iSample < _cache_.sampleNbOfEvents.size() ; iSample++ ){
    auto* container = &_cache_.samplesToFillList[iSample]->getDataContainer();
    if(_parameters_.useMcContainer) container = &_cache_.samplesToFillList[iSample]->getMcContainer();

    _cache_.sampleEventListPtrToFill[iSample] = &container->eventList;
    _cache_.sampleIndexOffsetList[iSample] = _cache_.sampleEventListPtrToFill[iSample]->size();
    container->reserveEventMemory(_owner_->getDataSetIndex(), _cache_.sampleNbOfEvents[iSample], eventPlaceholder);
  }

  size_t nEvents = treeChain.GetEntries();
#if USE_NEW_DIALS
  // DEV
  if( _eventDialCacheRef_ != nullptr ){
    if( not _cache_.dialCollectionsRefList.empty() ){
      LogInfo << "Claiming memory for event-by-event dials..." << std::endl;
      double eventByEventDialSize{0};
      size_t nDialsMaxPerEvent{0};
      for( auto& dialCollection : _cache_.dialCollectionsRefList ){
        nDialsMaxPerEvent += 1;
        if( dialCollection->isBinned() ){
          // Filling var indexes for faster eval with PhysicsEvent:
          for( auto& bin : dialCollection->getDialBinSet().getBinsList() ){
            std::vector<int> varIndexes;
            for( auto& var : bin.getVariableNameList() ){
              varIndexes.emplace_back(GenericToolbox::findElementIndex(var, _cache_.varsRequestedForIndexing));
            }
            bin.setEventVarIndexCache(varIndexes);
          }
        }
        else if( not dialCollection->getGlobalDialLeafName().empty() ){
          // Reserve memory for additional dials (those on a tree leaf)
          auto dialType = dialCollection->getGlobalDialType();
          LogInfo << dialCollection->getTitle() << ": creating " << nEvents;
          LogInfo << " " << dialType;

          if     ( dialType == "Spline" ){
            dialCollection->getDialBaseList().clear();
            double dialsSizeInRam{0};
            if(dialCollection->useCachedDials() ){
              dialsSizeInRam = double(nEvents) * sizeof(SplineCache);
              dialCollection->getDialBaseList().resize(nEvents, GenericToolbox::PolymorphicObjectWrapper<DialBase>(SplineCache()));
            }
            else{
              dialsSizeInRam = double(nEvents) * sizeof(Spline);
              dialCollection->getDialBaseList().resize(nEvents, GenericToolbox::PolymorphicObjectWrapper<DialBase>(Spline()));
            }

            eventByEventDialSize += dialsSizeInRam;
            LogInfo << " dials (" << GenericToolbox::parseSizeUnits( dialsSizeInRam ) << ")" << std::endl;

          }
          else if( dialType == "MonotonicSpline" ){
            dialCollection->getDialBaseList().clear();
            double dialsSizeInRam{0};
            if(dialCollection->useCachedDials() ){
              dialsSizeInRam = double(nEvents) * sizeof(MonotonicSplineCache);
              dialCollection->getDialBaseList().resize(nEvents, GenericToolbox::PolymorphicObjectWrapper<DialBase>(MonotonicSplineCache()));
            }
            else{
              dialsSizeInRam = double(nEvents) * sizeof(MonotonicSpline);
              dialCollection->getDialBaseList().resize(nEvents, GenericToolbox::PolymorphicObjectWrapper<DialBase>(MonotonicSpline()));
            }

            eventByEventDialSize += dialsSizeInRam;
            LogInfo << " dials (" << GenericToolbox::parseSizeUnits( dialsSizeInRam ) << ")" << std::endl;

          }
          else if( dialType == "GeneralSpline" ){
            dialCollection->getDialBaseList().clear();
            double dialsSizeInRam{0};
            if(dialCollection->useCachedDials() ){
              dialsSizeInRam = double(nEvents) * sizeof(GeneralSplineCache);
              dialCollection->getDialBaseList().resize(nEvents, GenericToolbox::PolymorphicObjectWrapper<DialBase>(GeneralSplineCache()));
            }
            else{
              dialsSizeInRam = double(nEvents) * sizeof(GeneralSpline);
              dialCollection->getDialBaseList().resize(nEvents, GenericToolbox::PolymorphicObjectWrapper<DialBase>(GeneralSpline()));
            }

            eventByEventDialSize += dialsSizeInRam;
            LogInfo << " dials (" << GenericToolbox::parseSizeUnits( dialsSizeInRam ) << ")" << std::endl;

          }
          else if( dialType == "SimpleSpline" ){
            dialCollection->getDialBaseList().clear();
            double dialsSizeInRam{0};
            if(dialCollection->useCachedDials() ){
              dialsSizeInRam = double(nEvents) * sizeof(SimpleSplineCache);
              dialCollection->getDialBaseList().resize(nEvents, GenericToolbox::PolymorphicObjectWrapper<DialBase>(SimpleSplineCache()));
            }
            else{
              dialsSizeInRam = double(nEvents) * sizeof(SimpleSpline);
              dialCollection->getDialBaseList().resize(nEvents, GenericToolbox::PolymorphicObjectWrapper<DialBase>(SimpleSpline()));
            }

            eventByEventDialSize += dialsSizeInRam;
            LogInfo << " dials (" << GenericToolbox::parseSizeUnits( dialsSizeInRam ) << ")" << std::endl;

          }
          else if( dialType == "Graph" ){
            dialCollection->getDialBaseList().clear();
            double dialsSizeInRam{0};
            if(dialCollection->useCachedDials() ){
              dialsSizeInRam = double(nEvents) * sizeof(GraphCache);
              dialCollection->getDialBaseList().resize(nEvents, GenericToolbox::PolymorphicObjectWrapper<DialBase>(GraphCache()));
            }
            else{
              dialsSizeInRam = double(nEvents) * sizeof(Graph);
              dialCollection->getDialBaseList().resize(nEvents, GenericToolbox::PolymorphicObjectWrapper<DialBase>(Graph()));
            }

            eventByEventDialSize += dialsSizeInRam;
            LogInfo << " dials (" << GenericToolbox::parseSizeUnits( dialsSizeInRam ) << ")" << std::endl;

          }
          else if( dialType == "LightGraph" ){
            dialCollection->getDialBaseList().clear();
            double dialsSizeInRam{0};
            dialsSizeInRam = double(nEvents) * sizeof(LightGraph);
            dialCollection->getDialBaseList().resize(nEvents, GenericToolbox::PolymorphicObjectWrapper<DialBase>(LightGraph()));

            eventByEventDialSize += dialsSizeInRam;
            LogInfo << " dials (" << GenericToolbox::parseSizeUnits( dialsSizeInRam ) << ")" << std::endl;
          }
          else{
            LogInfo << std::endl;
            LogThrow("Invalid dial type for event-by-event dial: " << dialType);
          }

        }
        else{
          LogThrow("DEV ERROR: not binned, not event-by-event?");
        }
      }
      _eventDialCacheRef_->allocateCacheEntries(nEvents, nDialsMaxPerEvent);
      LogInfo << "Event-by-event dials take " << GenericToolbox::parseSizeUnits(eventByEventDialSize) << " in RAM." << std::endl;
    }
  }
#else
  // DIALS
  DialSet* dialSetPtr;
  if( _parSetListPtrToLoad_ != nullptr ){
    LogInfo << "Claiming memory for event-by-event dials..." << std::endl;
    double eventByEventDialSize{0};
    for( auto& parSet : *_parSetListPtrToLoad_ ){
      if( not parSet.isEnabled() ){ continue; }
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ){ continue; }
        dialSetPtr = par.findDialSet( _owner_->getName() );
        if( dialSetPtr != nullptr and ( not dialSetPtr->getDialList().empty() or not dialSetPtr->getDialLeafName().empty() ) ){

          // Filling var indexes for faster eval with PhysicsEvent:
          for( auto& dial : dialSetPtr->getDialList() ){
            if( dial->getApplyConditionBinPtr() != nullptr ){
              std::vector<int> varIndexes;
              for( const auto& var : dial->getApplyConditionBinPtr()->getVariableNameList() ){
                varIndexes.emplace_back(GenericToolbox::findElementIndex(var, _cache_.varsRequestedForIndexing));
              }
              dial->getApplyConditionBinPtr()->setEventVarIndexCache(varIndexes);
            }
          }

          // Reserve memory for additional dials (those on a tree leaf)
          if( not dialSetPtr->getDialLeafName().empty() ){

            auto dialType = dialSetPtr->getGlobalDialType();

            LogInfo << par.getFullTitle() << ": creating " << nEvents;
            LogInfo << " " << DialType::DialTypeEnumNamespace::toString(dialType, true);

            if     ( dialType == DialType::Spline ){
              double dialsSizeInRam = double(nEvents) * sizeof(SplineDial);
              eventByEventDialSize += dialsSizeInRam;
              LogInfo << " dials (" << GenericToolbox::parseSizeUnits( dialsSizeInRam ) << ")" << std::endl;
              dialSetPtr->getDialList().resize(nEvents, DialWrapper(SplineDial(dialSetPtr)));
            }
            else if( dialType == DialType::Graph ){
              double dialsSizeInRam = double(nEvents) * sizeof(GraphDial);
              eventByEventDialSize += dialsSizeInRam;
              LogInfo << " dials (" << GenericToolbox::parseSizeUnits( dialsSizeInRam ) << ")" << std::endl;
              dialSetPtr->getDialList().resize(nEvents, DialWrapper(GraphDial(dialSetPtr)));
            }
            else{
              LogInfo << std::endl;
              LogThrow("Invalid dial type for event-by-event dial: " << DialType::DialTypeEnumNamespace::toString(dialType))
            }

          }

          // Add the dialSet to the list
          _cache_.dialSetPtrMap[&parSet].emplace_back( dialSetPtr );
        }
      }
    }
    LogInfo << "Event-by-event dials take " << GenericToolbox::parseSizeUnits(eventByEventDialSize) << " in RAM." << std::endl;
  }
#endif

}
void DataDispenser::readAndFill(){
  LogWarning << "Reading dataset and loading..." << std::endl;

  if( not _parameters_.nominalWeightFormulaStr.empty() ){
    LogInfo << "Nominal weight: \"" << _parameters_.nominalWeightFormulaStr << "\"" << std::endl;
  }

  LogWarning << "Loading and indexing..." << std::endl;
  if( GlobalVariables::getNbThreads() > 1 ){
    ROOT::EnableThreadSafety();
    std::function<void(int)> f = [&](int iThread_){ this->fillFunction(iThread_); };
    GlobalVariables::getParallelWorker().addJob(__METHOD_NAME__, f);
    GlobalVariables::getParallelWorker().runJob(__METHOD_NAME__);
    GlobalVariables::getParallelWorker().removeJob(__METHOD_NAME__);
  }
  else{
    this->fillFunction(0); // for better debug breakdown
  }

  LogInfo << "Shrinking lists..." << std::endl;
  for( size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
    auto* container = &_cache_.samplesToFillList[iSample]->getDataContainer();
    if(_parameters_.useMcContainer) container = &_cache_.samplesToFillList[iSample]->getMcContainer();
    container->shrinkEventList(_cache_.sampleIndexOffsetList[iSample]);
  }
}
void DataDispenser::fillFunction(int iThread_){

  int nThreads = GlobalVariables::getNbThreads();
  if( iThread_ == -1 ){
    iThread_ = 0;
    nThreads = 1;
  }

  TChain treeChain(_parameters_.treePath.c_str());
  for( const auto& file: _parameters_.filePathList){
    std::string name = GenericToolbox::expandEnvironmentVariables(file);
    if (name != file and iThread_ == 0) {
      LogWarning << "Filename expanded to: " << name << std::endl;
    }
    treeChain.Add(name.c_str());
  }

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

    // ROOT Hot fix: https://root-forum.cern.ch/t/ttreeformula-evalinstance-return-0-0/16366/10
    threadNominalWeightFormula->GetNdata();

    LogThrowIf(threadNominalWeightFormula->GetNdim() == 0,
               "\"" <<  _parameters_.nominalWeightFormulaStr << "\" could not be parsed by the TChain");
    objToNotify.Add(threadNominalWeightFormula); // memory handled here!
    treeChain.SetBranchStatus("*", false);
    GenericToolbox::enableSelectedBranches(&treeChain, threadNominalWeightFormula);
  }

  // TTree data buffer
  auto tEventBuffer = this->generateTreeEventBuffer(&treeChain, _cache_.varsRequestedForIndexing);

  // Event Var Transform
  auto eventVarTransformList = _cache_.eventVarTransformList; // copy for cache
  std::vector<EventVarTransformLib*> varTransformForIndexingList;
  std::vector<EventVarTransformLib*> varTransformForStorageList;
  for( auto& eventVarTransform : eventVarTransformList ){
    if( GenericToolbox::doesElementIsInVector(eventVarTransform.getOutputVariableName(), _cache_.varsRequestedForIndexing) ){
      varTransformForIndexingList.emplace_back(&eventVarTransform);
    }
    if( GenericToolbox::doesElementIsInVector(eventVarTransform.getOutputVariableName(), _cache_.varsRequestedForStorage) ){
      varTransformForStorageList.emplace_back(&eventVarTransform);
    }
  }

  if( iThread_ == 0 ){
    if( not varTransformForIndexingList.empty() ){
      LogInfo << "EventVarTransformLib used for indexing: "
              << GenericToolbox::iterableToString(
                  varTransformForIndexingList,
                  [](const EventVarTransformLib* elm_){ return "\"" + elm_->getTitle() + "\"";}, false)
              << std::endl;
    }
    if( not varTransformForStorageList.empty() ){
      LogInfo << "EventVarTransformLib used for storage: "
              << GenericToolbox::iterableToString(
                  varTransformForStorageList,
                  []( const EventVarTransformLib* elm_){ return "\"" + elm_->getTitle() + "\""; }, false)
              << std::endl;
    }
  }

  // buffer that will store the data for indexing
  PhysicsEvent eventBuffer;
  eventBuffer.setDataSetIndex(_owner_->getDataSetIndex());
  eventBuffer.setCommonLeafNameListPtr(std::make_shared<std::vector<std::string>>(_cache_.varsRequestedForIndexing));
  auto copyDict = eventBuffer.generateDict(tEventBuffer, _parameters_.overrideLeafDict);
  if(iThread_ == 0){
    LogInfo << "Feeding event variables with:" << std::endl;
    GenericToolbox::TablePrinter t;
    t.setColTitles({{"Variable"}, {"Leaf"}, {"Transforms"}});
    for( size_t iVar = 0 ; iVar < eventBuffer.getCommonLeafNameListPtr()->size() ; iVar++ ){
      std::string variableName = (*eventBuffer.getCommonLeafNameListPtr())[iVar];
      t << variableName << std::endl;

      t << copyDict[iVar].first->getLeafFullName();
      if(copyDict[iVar].second != -1) t << "[" << copyDict[iVar].second << "]";
      t << std::endl;

      std::vector<std::string> transformsList;
      for( auto* varTransformForIndexing : varTransformForIndexingList ){
        if( varTransformForIndexing->getOutputVariableName() == variableName ){
          transformsList.emplace_back(varTransformForIndexing->getTitle());
        }
      }
      t << GenericToolbox::parseVectorAsString(transformsList);

      if( GenericToolbox::doesElementIsInVector(variableName, _cache_.varsRequestedForStorage)){
        t.setColorBuffer(GenericToolbox::ColorCodes::blueBackground);
      }
      else if(
          copyDict[iVar].first->getLeafTypeName() == "TClonesArray"
          or copyDict[iVar].first->getLeafTypeName() == "TGraph"
          ){
        t.setColorBuffer(GenericToolbox::ColorCodes::magentaBackground);
      }

      t << std::endl;
    }

    t.printTable();
    LogInfo(Logger::Color::BG_BLUE) << "      " << Logger::getColorEscapeCode(Logger::Color::RESET) << " -> Variables stored in RAM" << std::endl;
    LogInfo(Logger::Color::BG_MAGENTA) << "      " << Logger::getColorEscapeCode(Logger::Color::RESET) << " -> Dials stored in RAM" << std::endl;
  }
  eventBuffer.copyData(copyDict); // resize array obj
  eventBuffer.resizeVarToDoubleCache();

  // only here to create a dictionary to copy data from the TTree that should store in RAM
  PhysicsEvent evStore;
  evStore.setDataSetIndex(_owner_->getDataSetIndex());
  evStore.setCommonLeafNameListPtr(std::make_shared<std::vector<std::string>>(_cache_.varsRequestedForStorage));
  auto copyStoreDict = evStore.generateDict(tEventBuffer, _parameters_.overrideLeafDict);

  // Will keep track of a picked event pointer
  PhysicsEvent* eventPtr{nullptr};

  size_t sampleEventIndex;
  int threadDialIndex;

  // Loop vars
  bool foundValidDialAmongTheSet{true};
  int lastFailedBinVarIndex{-1}; int lastEventVarIndex{-1};
  const std::pair<double, double>* lastEdges{nullptr};
  size_t iVar{0};
  size_t iSample{0}, nSample{_cache_.samplesToFillList.size()};
  size_t iTransform{0}, nTransform{_cache_.eventVarTransformList.size()};

  // Dials
  size_t eventDialOffset;
  size_t iDialSet, iDial;
  size_t nBinEdges;
  TGraph* grPtr{nullptr};

  // Bin searches
  const std::vector<DataBin>* binsListPtr;
  std::vector<DataBin>::const_iterator binFoundItr;
  auto isBinValid = [&](const DataBin& b_){
    for( iVar = 0 ; iVar < b_.getVariableNameList().size() ; iVar++ ){
      if( not b_.isBetweenEdges(iVar, eventBuffer.getVarAsDouble(b_.getVariableNameList()[iVar])) ){
        return false;
      }
    } // Var
    return true;
  };

  // Dial bin search
  DataBin* dataBin{nullptr};
#if USE_NEW_DIALS
  size_t freeSlotDial{0};
  size_t iCollection(-1);
  std::pair<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>>* eventDialCacheEntry;
  std::vector<DataBin>::iterator dial2FoundItr;
  auto isDial2Valid = [&](const DataBin& d_){
    nBinEdges = d_.getEdgesList().size();
    for( iVar = 0 ; iVar < nBinEdges ; iVar++ ){
      if( not DataBin::isBetweenEdges(
          d_.getEdgesList()[iVar],
          eventBuffer.getVarAsDouble( d_.getEventVarIndexCache()[iVar] ) )
          ){
        return false;
      }
    }
    return true;
  };
#else
  SplineDial* spDialPtr{nullptr};
  GraphDial* grDialPtr{nullptr};
  DialSet* dialSetPtr;
  std::vector<DialWrapper>::iterator dialFoundItr;
  auto isDialValid = [&](const DialWrapper& d_){
    dataBin = d_->getApplyConditionBinPtr();
    nBinEdges = dataBin->getEdgesList().size();
    for( iVar = 0 ; iVar < nBinEdges ; iVar++ ){
      if( not DataBin::isBetweenEdges(
          dataBin->getEdgesList()[iVar],
          eventBuffer.getVarAsDouble( dataBin->getEventVarIndexCache()[iVar] ) )
          ){
        return false;
      }
    }
    return true;
  };
#endif

  // Formula
  std::vector<std::shared_ptr<TFormula>> varSelectionFormulaList{};
  for( auto* sample : _cache_.samplesToFillList ){
    varSelectionFormulaList.emplace_back(nullptr);
    if( not sample->getVarSelectionFormulaStr().empty() ){
      varSelectionFormulaList.back() = std::make_shared<TFormula>(
          Form("%s_%i_VarSelectionFormula", sample->getName().c_str(), iThread_),
          sample->getVarSelectionFormulaStr().c_str()
      );
      LogInfo(iThread_==0) << "Var selection formula for " << sample->getName() << ": \"" << sample->getVarSelectionFormulaStr() << "\"" << std::endl;
    }
  }

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

  std::string progressTitle = "Loading and indexing...";
  std::stringstream ssProgressBar;

  for(Long64_t iEntry = iStart ; iEntry < iEnd ; iEntry++ ){

    if( iThread_ == 0 ){
      if( GenericToolbox::showProgressBar(iGlobal, nEvents) ){

        ssProgressBar.str("");

        ssProgressBar << LogInfo.getPrefixString() << "Reading from disk: "
                      << GenericToolbox::padString(GenericToolbox::parseSizeUnits(readSpeed.getTotalAccumulated()), 8) << " ("
                      << GenericToolbox::padString(GenericToolbox::parseSizeUnits(readSpeed.evalTotalGrowthRate()), 8) << "/s)";

        int cpuPercent = int(GenericToolbox::getCpuUsageByProcess());
        ssProgressBar << " / CPU efficiency: " << GenericToolbox::padString(std::to_string(cpuPercent/nThreads), 3,' ')
                      << "% / RAM: " << GenericToolbox::parseSizeUnits( double(GenericToolbox::getProcessMemoryUsage()) ) << std::endl;

        ssProgressBar << LogInfo.getPrefixString() << progressTitle;
        GenericToolbox::displayProgressBar(iGlobal, nEvents, ssProgressBar.str());
      }
      iGlobal += nThreads;
    }

    bool skipEvent = true;
    for( bool isInSample : _cache_.eventIsInSamplesList[iEntry] ){
      if( isInSample ){ skipEvent = false; break; }
    }
    if( skipEvent ) continue;

    nBytes = treeChain.GetEntry(iEntry);

    // monitor
    if( iThread_ == 0 ) readSpeed.addQuantity(nBytes*nThreads);

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

    for( iSample = 0 ; iSample < nSample ; iSample++ ){
      if( _cache_.eventIsInSamplesList[iEntry][iSample] ){

        // Reset bin index of the buffer
        eventBuffer.setSampleBinIndex(-1);

        // Getting loaded data in tEventBuffer
        eventBuffer.copyData(copyDict);

        // Propagate transformations for indexing
        for( auto* varTransformPtr : varTransformForIndexingList ){
          varTransformPtr->evalAndStore(eventBuffer);
        }

        // Sample variable
        if( varSelectionFormulaList[iSample] != nullptr ){
          if( eventBuffer.evalFormula( varSelectionFormulaList[iSample].get() ) == 0 ) break;
        }


        // Has valid bin?
        binsListPtr = &_cache_.samplesToFillList[iSample]->getBinning().getBinsList();
        binFoundItr = std::find_if(
            binsListPtr->begin(),
            binsListPtr->end(),
            isBinValid
        );

        if (binFoundItr == binsListPtr->end()) {
          // Invalid bin -> next sample
          break;
        }
        else {
          // found bin
          eventBuffer.setSampleBinIndex(int(std::distance(binsListPtr->begin(), binFoundItr)));
        }

        // OK, now we have a valid fit bin. Let's claim an index.
        sampleEventIndex = _cache_.sampleIndexOffsetList[iSample]++;

        // Get the next free event in our buffer
        eventPtr = &(*_cache_.sampleEventListPtrToFill[iSample])[sampleEventIndex];
        eventPtr->copyData(copyStoreDict); // buffer has the right size already

        // Propagate transformation for storage -> use the previous results calculated for indexing
        for( auto* varTransformPtr : varTransformForStorageList ){
          varTransformPtr->storeCachedOutput(*eventPtr);
        }

        eventPtr->setEntryIndex(iEntry);
        eventPtr->setSampleBinIndex(eventBuffer.getSampleBinIndex());
        eventPtr->setTreeWeight(eventBuffer.getTreeWeight());
        eventPtr->setNominalWeight(eventBuffer.getTreeWeight());
        eventPtr->resetEventWeight();

        // Now the event is ready. Let's index the dials:
        eventDialOffset = 0;


#if USE_NEW_DIALS
        if( _eventDialCacheRef_ != nullptr ) {
          eventDialCacheEntry = _eventDialCacheRef_->fetchNextCacheEntry();
          eventDialCacheEntry->first = {
              _cache_.samplesToFillList[iSample]->getIndex(),
              sampleEventIndex
          };

          iCollection = -1;
          for( auto* dialCollectionRef : _cache_.dialCollectionsRefList ){
            iCollection = dialCollectionRef->getIndex();

            if( not dialCollectionRef->isEnabled() ){ continue; }

            if( dialCollectionRef->getApplyConditionFormula() != nullptr ){
              if( eventBuffer.evalFormula(dialCollectionRef->getApplyConditionFormula().get()) == 0 ){
                // next dialSet
                continue;
              }
            }

            if( dialCollectionRef->isBinned() ){

              // is only one bin with no condition:
              if( dialCollectionRef->getDialBaseList().size() == 1 ){
                // if is it NOT a DialBinned -> this is the one we are supposed to use
                if( dialCollectionRef->getDialBinSet().isEmpty() ){
                  eventDialCacheEntry->second[eventDialOffset].first = iCollection;
                  eventDialCacheEntry->second[eventDialOffset].second = 0;
                  eventDialOffset++;
                }
              }
              else {
                // -- probably the slowest part of the indexing: ----
                dial2FoundItr = std::find_if(
                    dialCollectionRef->getDialBinSet().getBinsList().begin(),
                    dialCollectionRef->getDialBinSet().getBinsList().end(),
                    isDial2Valid
                );
                // --------------------------------------------------

                if (dial2FoundItr !=  dialCollectionRef->getDialBinSet().getBinsList().end()) {
                  // found DIAL -> get index
                  eventDialCacheEntry->second[eventDialOffset].first = iCollection;
                  eventDialCacheEntry->second[eventDialOffset].second = std::distance(
                      dialCollectionRef->getDialBinSet().getBinsList().begin(), dial2FoundItr
                  );
                  eventDialOffset++;
                }
              }
            }
            else if( not dialCollectionRef->getGlobalDialLeafName().empty() ){
              // Event-by-event dial?
              if     ( not strcmp(treeChain.GetLeaf(dialCollectionRef->getGlobalDialLeafName().c_str())->GetTypeName(), "TClonesArray") ){
                grPtr = (TGraph*) eventBuffer.getVariable<TClonesArray*>(dialCollectionRef->getGlobalDialLeafName())->At(0);
              }
              else if( not strcmp(treeChain.GetLeaf(dialCollectionRef->getGlobalDialLeafName().c_str())->GetTypeName(), "TGraph") ){
                grPtr = (TGraph*) eventBuffer.getVariable<TGraph*>(dialCollectionRef->getGlobalDialLeafName());
              }
              else{
                LogThrow("Unsupported event-by-event dial type: " << treeChain.GetLeaf(dialCollectionRef->getGlobalDialLeafName().c_str())->GetTypeName() )
              }

              // loaded graph is valid?
              if( Misc::isGraphValid(grPtr) ){
                freeSlotDial = dialCollectionRef->getNextDialFreeSlot();
                if      ( dialCollectionRef->getGlobalDialType() == "Spline" ){
                  if(dialCollectionRef->useCachedDials() ) {
                    ( (SplineCache*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->createSpline( grPtr );
                    ( (SplineCache*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setAllowExtrapolation( dialCollectionRef->isAllowDialExtrapolation() );
                  }
                  else {
                    ( (Spline*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->createSpline( grPtr );
                    ( (Spline*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setAllowExtrapolation( dialCollectionRef->isAllowDialExtrapolation() );
                  }
                }
                else if ( dialCollectionRef->getGlobalDialType() == "MonotonicSpline" ){
                  if(dialCollectionRef->useCachedDials() ) {
                    ( (MonotonicSplineCache*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->buildSplineData( *grPtr );
                    ( (MonotonicSplineCache*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setAllowExtrapolation( dialCollectionRef->isAllowDialExtrapolation() );
                  }
                  else {
                    ( (MonotonicSpline*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->buildSplineData( *grPtr );
                    ( (MonotonicSpline*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setAllowExtrapolation( dialCollectionRef->isAllowDialExtrapolation() );
                  }
                }
                else if ( dialCollectionRef->getGlobalDialType() == "GeneralSpline" ){
                  if(dialCollectionRef->useCachedDials() ) {
                    ( (GeneralSplineCache*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->buildSplineData( *grPtr );
                    ( (GeneralSplineCache*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setAllowExtrapolation( dialCollectionRef->isAllowDialExtrapolation() );
                  }
                  else {
                    ( (GeneralSpline*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->buildSplineData( *grPtr );
                    ( (GeneralSpline*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setAllowExtrapolation( dialCollectionRef->isAllowDialExtrapolation() );
                  }
                }
                else if ( dialCollectionRef->getGlobalDialType() == "SimpleSpline" ){
                  if(dialCollectionRef->useCachedDials() ) {
                    ( (SimpleSplineCache*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->buildSplineData( *grPtr );
                    ( (SimpleSplineCache*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setAllowExtrapolation( dialCollectionRef->isAllowDialExtrapolation() );
                  }
                  else {
                    ( (SimpleSpline*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->buildSplineData( *grPtr );
                    ( (SimpleSpline*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setAllowExtrapolation( dialCollectionRef->isAllowDialExtrapolation() );
                  }
                }
                else if ( dialCollectionRef->getGlobalDialType() == "Graph" ){
                  if(dialCollectionRef->useCachedDials() ) {
                    ( (GraphCache*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setGraph( *grPtr );
                    ( (GraphCache*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setAllowExtrapolation( dialCollectionRef->isAllowDialExtrapolation() );
                  }
                  else{
                    ( (Graph*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setGraph( *grPtr );
                    ( (Graph*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setAllowExtrapolation( dialCollectionRef->isAllowDialExtrapolation() );
                  }
                }
                else if ( dialCollectionRef->getGlobalDialType() == "LightGraph" ){
                  ( (LightGraph*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setGraph( *grPtr );
                  ( (LightGraph*) dialCollectionRef->getDialBaseList()[freeSlotDial].get() )->setAllowExtrapolation( dialCollectionRef->isAllowDialExtrapolation() );
                }
                else{
                  LogThrow( "Unsupported event-by-event dial: " << dialCollectionRef->getGlobalDialType() );
                }
                eventDialCacheEntry->second[eventDialOffset].first = iCollection;
                eventDialCacheEntry->second[eventDialOffset].second = freeSlotDial;
                eventDialOffset++;
              }

            }
            else{
              LogThrow("not an event by event dial, nor binned");
            }

          }
        }
#else
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
                    spDialPtr->createSpline( grPtr );
                    spDialPtr->initialize();
                    spDialPtr->setIsReferenced(true);
                    // Adding dial in the event
                    eventPtr->getRawDialPtrList()[eventDialOffset++] = spDialPtr;
                  }
                  else if( dialSetPtr->getGlobalDialType() == DialType::Graph ){
                    grDialPtr = (GraphDial*) dialSetPtr->getDialList()[iEntry].get();
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
                  spDialPtr->createSpline(grPtr);
                  spDialPtr->initialize();
                  spDialPtr->setIsReferenced(true);
                  // Adding dial in the event
                  eventPtr->getRawDialPtrList()[eventDialOffset++] = spDialPtr;
                }
                else if( dialSetPtr->getGlobalDialType() == DialType::Graph ){
                  grDialPtr = (GraphDial*) dialSetPtr->getDialList()[iEntry].get();
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
              // Binned dial:
              foundValidDialAmongTheSet = false;

              if( dialSetPtr->getDialList().size() == 1 and dialSetPtr->getDialList()[0]->getApplyConditionBinPtr() == nullptr ){
                foundValidDialAmongTheSet = true;
                dialSetPtr->getDialList()[0]->setIsReferenced(true);
                eventPtr->getRawDialPtrList()[eventDialOffset++] = dialSetPtr->getDialList()[0].get();
              }
              else {
                // -- probably the slowest part of the indexing: ----
                dialFoundItr = std::find_if(
                    dialSetPtr->getDialList().begin(),
                    dialSetPtr->getDialList().end(),
                    isDialValid
                );
                // --------------------------------------------------

                if (dialFoundItr != dialSetPtr->getDialList().end()) {
                  // found DIAL -> get index
                  iDial = std::distance(dialSetPtr->getDialList().begin(), dialFoundItr);

                  foundValidDialAmongTheSet = true;
                  dialSetPtr->getDialList()[iDial]->setIsReferenced(true);
                  eventPtr->getRawDialPtrList()[eventDialOffset++] = dialSetPtr->getDialList()[iDial].get();
                }
              }

              if( foundValidDialAmongTheSet and dialSetPair.first->isUseOnlyOneParameterPerEvent() ){
                // leave dialSet (corresponding to a given parameter) loop since we explicitly ask for 1 parameter for this parSet
                break;
              }
            } // else (not dialSetPtr->getDialLeafName().empty())

          } // iDialSet / Enabled-parameter
        } // ParSet / DialSet Pairs

        // Resize the dialRef list
        eventPtr->getRawDialPtrList().resize(eventDialOffset);
        eventPtr->getRawDialPtrList().shrink_to_fit();
#endif

      } // event has passed the selection?
    } // samples
  } // entries
  if( iThread_ == 0 ){
    GenericToolbox::displayProgressBar(nEvents, nEvents, ssProgressBar.str());
  }
}

GenericToolbox::TreeEntryBuffer DataDispenser::generateTreeEventBuffer(TChain* treeChain_, const std::vector<std::string>& varsList_){
  GenericToolbox::TreeEntryBuffer tBuf;

  // Gather leaves names list that will have to be hooked to the tree
  std::vector<std::string> leafVarList, isDummyList;
  leafVarList.reserve(varsList_.size());
  isDummyList.reserve(varsList_.size());

  // Find associated leaf according to the dictionary
  for( auto& var : varsList_ ){
    LogThrowIf(
        not GenericToolbox::doesKeyIsInMap(var, _cache_.varToLeafDict),
        "Could not find \"" << var << "\" in " << GenericToolbox::parseMapAsString(_cache_.varToLeafDict)
    );
    if( not GenericToolbox::doesElementIsInVector(_cache_.varToLeafDict[var].first, leafVarList) ){
      leafVarList.emplace_back(_cache_.varToLeafDict[var].first);
    }
  }

  tBuf.setLeafNameList(leafVarList);

  for( auto& var : varsList_ ){
    // don't activate as dummy if the leaf exists
    if( treeChain_->GetLeaf(_cache_.varToLeafDict[var].first.c_str()) != nullptr ) continue;
    tBuf.setIsDummyLeaf(_cache_.varToLeafDict[var].first, _cache_.varToLeafDict[var].second);
  }

  for( auto& dummyVar : _parameters_.dummyVariablesList ){
    tBuf.setIsDummyLeaf(dummyVar, true);
  }

  tBuf.hook(treeChain_);

  return tBuf;
}

void DataDispenserCache::addVarRequestedForIndexing(const std::string& varName_) {
  LogThrowIf(varName_.empty(), "no var name provided.");
  GenericToolbox::addIfNotInVector(varName_, this->varsRequestedForIndexing);
}
void DataDispenserCache::addVarRequestedForStorage(const std::string& varName_){
  LogThrowIf(varName_.empty(), "no var name provided.");
  GenericToolbox::addIfNotInVector(varName_, this->varsRequestedForStorage);
  this->addVarRequestedForIndexing(varName_);
}