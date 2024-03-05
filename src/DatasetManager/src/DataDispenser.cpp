//
// Created by Adrien BLANCHET on 14/05/2022.
//


#include "DataDispenser.h"
#include "DatasetDefinition.h"

#include "EventVarTransform.h"
#include "GundamGlobals.h"
#include "GenericToolbox.Json.h"
#include "ConfigUtils.h"

#include "DialCollection.h"
#include "DialBaseFactory.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Map.h"
#include "Logger.h"

#include "TTreeFormulaManager.h"
#include "TChainElement.h"
#include "TClonesArray.h"
#include "TChain.h"
#include "THn.h"

#include <string>
#include <vector>
#include <sstream>

LoggerInit([]{
  Logger::setUserHeaderStr("[DataDispenser]");
});


void DataDispenser::readConfigImpl(){
  LogThrowIf( _config_.empty(), "Config is not set." );

  _parameters_.name = GenericToolbox::Json::fetchValue<std::string>(_config_, "name", _parameters_.name);

  if( GenericToolbox::Json::doKeyExist( _config_, "fromHistContent" ) ) {
    LogWarning << "Dataset \"" << _parameters_.name << "\" will be defined with histogram data." << std::endl;

    _parameters_.fromHistContent = GenericToolbox::Json::fetchValue<JsonType>( _config_, "fromHistContent" );
    return;
  }

  _parameters_.treePath = GenericToolbox::Json::fetchValue<std::string>(_config_, "tree", _parameters_.treePath);
  _parameters_.filePathList = GenericToolbox::Json::fetchValue<std::vector<std::string>>(_config_, "filePathList", _parameters_.filePathList);
  _parameters_.additionalVarsStorage = GenericToolbox::Json::fetchValue(_config_, {{"additionalLeavesStorage"}, {"additionalVarsStorage"}}, _parameters_.additionalVarsStorage);
  _parameters_.dummyVariablesList = GenericToolbox::Json::fetchValue(_config_, "dummyVariablesList", _parameters_.dummyVariablesList);
  _parameters_.useMcContainer = GenericToolbox::Json::fetchValue(_config_, "useMcContainer", _parameters_.useMcContainer);

  _parameters_.dialIndexFormula = GenericToolbox::Json::fetchValue(_config_, "dialIndexFormula", _parameters_.dialIndexFormula);
  _parameters_.selectionCutFormulaStr = GenericToolbox::Json::buildFormula(_config_, "selectionCutFormula", "&&", _parameters_.selectionCutFormulaStr);
  _parameters_.nominalWeightFormulaStr = GenericToolbox::Json::buildFormula(_config_, "nominalWeightFormula", "*", _parameters_.nominalWeightFormulaStr);

  _parameters_.debugNbMaxEventsToLoad = GenericToolbox::Json::fetchValue(_config_, "debugNbMaxEventsToLoad", _parameters_.debugNbMaxEventsToLoad);

  _parameters_.variableDict.clear();
  for( auto& entry : GenericToolbox::Json::fetchValue(_config_, {{"variableDict"}, {"overrideLeafDict"}}, JsonType()) ){
    auto varName = GenericToolbox::Json::fetchValue<std::string>(entry, {{"name"}, {"eventVar"}});
    auto varExpr = GenericToolbox::Json::fetchValue<std::string>(entry, {{"expr"}, {"expression"}, {"leafVar"}});
    _parameters_.variableDict[ varName ] = varExpr;
  }
}
void DataDispenser::initializeImpl(){
  // Nothing else to do other than read config?
  LogWarning << "Initialized data dispenser: " << getTitle() << std::endl;
}

void DataDispenser::load(Propagator& propagator_){
  LogWarning << "Loading dataset: " << getTitle() << std::endl;
  LogThrowIf(not this->isInitialized(), "Can't load while not initialized.");
  LogThrowIf(not propagator_.isInitialized(), "Can't load while propagator_ is not initialized.");

  if(GundamGlobals::getVerboseLevel() >= VerboseLevel::MORE_PRINTOUT ){
    LogDebug << "Configuration: " << _parameters_.getSummary() << std::endl;
  }

  _cache_.clear();
  _cache_.propagatorPtr = &propagator_;

  this->buildSampleToFillList();

  if( _cache_.samplesToFillList.empty() ){
    LogAlert << "No samples were selected for dataset: " << getTitle() << std::endl;
    return;
  }

  if( not _parameters_.fromHistContent.empty() ){
    this->loadFromHistContent();
    return;
  }

  LogInfo << "Data will be extracted from: " << GenericToolbox::toString(_parameters_.filePathList, true) << std::endl;
  for( const auto& file: _parameters_.filePathList){
    std::string path = GenericToolbox::expandEnvironmentVariables(file);
    LogThrowIf(not GenericToolbox::doesTFileIsValid(path, {_parameters_.treePath}), "Invalid file: " << path);
  }

  this->parseStringParameters();
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

  for( auto& sample : _cache_.propagatorPtr->getSampleSet().getSampleList() ){
    if( not sample.isEnabled() ) continue;
    if( sample.isDatasetValid(_owner_->getName()) ){
      _cache_.samplesToFillList.emplace_back(&sample);
    }
  }

  if( _cache_.samplesToFillList.empty() ){
    LogInfo << "No sample selected." << std::endl;
    return;
  }
}
void DataDispenser::parseStringParameters() {

  auto replaceToyIndexFct = [&](std::string& formula_){
    if( GenericToolbox::hasSubStr(formula_, "<I_TOY>") ){
      LogThrowIf(_cache_.propagatorPtr->getIThrow()==-1, "<I_TOY> not set.");
      GenericToolbox::replaceSubstringInsideInputString(formula_, "<I_TOY>", std::to_string(_cache_.propagatorPtr->getIThrow()));
    }
  };
  auto overrideLeavesNamesFct = [&](std::string& formula_){
    for( auto& replaceEntry : _cache_.varsToOverrideList ){
      GenericToolbox::replaceSubstringInsideInputString(formula_, replaceEntry, _parameters_.variableDict[replaceEntry]);
    }
  };

  if( not _parameters_.variableDict.empty() ){
    for( auto& entryDict : _parameters_.variableDict ){ replaceToyIndexFct(entryDict.second); }
    LogInfo << "Variable dictionary: " << GenericToolbox::toString(_parameters_.variableDict) << std::endl;

    for( auto& overrideEntry : _parameters_.variableDict ){
      _cache_.varsToOverrideList.emplace_back(overrideEntry.first);
    }
    // make sure we process the longest words first: "thisIsATest" variable should be replaced before "thisIs"
    GenericToolbox::sortVector(_cache_.varsToOverrideList, [](const std::string& a_, const std::string& b_){ return a_.size() > b_.size(); });
  }

  if( GenericToolbox::Json::doKeyExist(_config_, "variablesTransform") ){
    // load transformations
    int index{0};
    for( auto& varTransform : GenericToolbox::Json::fetchValue<std::vector<JsonType>>(_config_, "variablesTransform") ){
      _cache_.eventVarTransformList.emplace_back( varTransform );
      _cache_.eventVarTransformList.back().setIndex(index++);
      _cache_.eventVarTransformList.back().readConfig();
      if( not _cache_.eventVarTransformList.back().isEnabled() ){
        _cache_.eventVarTransformList.pop_back();
        continue;
      }
      _cache_.eventVarTransformList.back().initialize();
    }
    // sort them according to their output
    GenericToolbox::sortVector(_cache_.eventVarTransformList, [](const EventVarTransformLib& a_, const EventVarTransformLib& b_){
      // does a_ is a self transformation? -> if yes, don't change the order
      if( GenericToolbox::doesElementIsInVector(a_.getOutputVariableName(), a_.fetchRequestedVars()) ){ return false; }
      // does b_ transformation needs a_ output? -> if yes, a needs to go first
      if( GenericToolbox::doesElementIsInVector(a_.getOutputVariableName(), b_.fetchRequestedVars()) ){ return true; }
      // otherwise keep the order from the declaration
      if( a_.getIndex() < b_.getIndex() ) return true;
      // default -> won't change the order
      return false;
    });
  }

  replaceToyIndexFct(_parameters_.dialIndexFormula);
  replaceToyIndexFct(_parameters_.nominalWeightFormulaStr);
  replaceToyIndexFct(_parameters_.selectionCutFormulaStr);

  overrideLeavesNamesFct(_parameters_.dialIndexFormula);
  overrideLeavesNamesFct(_parameters_.nominalWeightFormulaStr);
  overrideLeavesNamesFct(_parameters_.selectionCutFormulaStr);

  // add surrounding parenthesis to force the LeafForm to treat it as a TFormula
  if(not _parameters_.dialIndexFormula.empty()){ _parameters_.dialIndexFormula = "(" + _parameters_.dialIndexFormula + ")"; }
  if(not _parameters_.nominalWeightFormulaStr.empty()){ _parameters_.nominalWeightFormulaStr = "(" + _parameters_.nominalWeightFormulaStr + ")"; }
  if(not _parameters_.selectionCutFormulaStr.empty()){ _parameters_.selectionCutFormulaStr = "(" + _parameters_.selectionCutFormulaStr + ")"; }
}
void DataDispenser::doEventSelection(){
  LogWarning << "Performing event selection..." << std::endl;

  LogInfo << "Event selection..." << std::endl;

  // Could lead to weird behaviour of ROOT object otherwise:
  ROOT::EnableThreadSafety();

  // how meaning buffers?
  int nThreads{GundamGlobals::getParallelWorker().getNbThreads()};
  if( _owner_->isDevSingleThreadEventSelection() ) { nThreads = 1; }

  Long64_t nEntries{0};
  {
    auto treeChain{this->openChain(true)};
    nEntries = treeChain->GetEntries();
  }
  LogThrowIf(nEntries == 0, "TChain is empty.");
  LogInfo << "Will read " << nEntries << " event entries." << std::endl;

  _cache_.threadSelectionResults.resize(nThreads);
  for( auto& threadResults : _cache_.threadSelectionResults ){
    threadResults.sampleNbOfEvents.resize(_cache_.samplesToFillList.size(), 0);
    threadResults.eventIsInSamplesList.resize(nEntries, std::vector<bool>(_cache_.samplesToFillList.size(), false));
  }

  if( not _owner_->isDevSingleThreadEventSelection() ) {
    GundamGlobals::getParallelWorker().addJob(__METHOD_NAME__, [this](int iThread_){ this->eventSelectionFunction(iThread_); });
    GundamGlobals::getParallelWorker().runJob(__METHOD_NAME__);
    GundamGlobals::getParallelWorker().removeJob(__METHOD_NAME__);
  }
  else {
    this->eventSelectionFunction(-1);
  }

  LogInfo << "Merging thread results..." << std::endl;
  _cache_.sampleNbOfEvents.resize(_cache_.samplesToFillList.size(), 0);
  _cache_.eventIsInSamplesList.resize(nEntries, std::vector<bool>(_cache_.samplesToFillList.size(), false));
  for( auto& threadResults : _cache_.threadSelectionResults ){
    // merging nEvents

    for( int iSample = 0 ; iSample < int(_cache_.sampleNbOfEvents.size()) ; iSample++ ){
      _cache_.sampleNbOfEvents[iSample] += threadResults.sampleNbOfEvents[iSample];
    }

    for( size_t iEntry = 0 ; iEntry < int(_cache_.eventIsInSamplesList.size()) ; iEntry++ ){
      for( size_t iSample = 0 ; iSample < int(_cache_.eventIsInSamplesList[iEntry].size()) ; iSample++ ){
        if( threadResults.eventIsInSamplesList[iEntry][iSample] ){
          _cache_.eventIsInSamplesList[iEntry][iSample] = true;
        }
      }
    }

  }

  LogInfo << "Freeing up thread buffers..." << std::endl;
  _cache_.threadSelectionResults.clear();

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

  if( _parameters_.useMcContainer ){
    LogInfo << "Selecting dial collections..." << std::endl;
    for( auto& dialCollection : _cache_.propagatorPtr->getDialCollectionList() ){
      if( not dialCollection.isEnabled() ){ continue; }
      if( not dialCollection.isDatasetValid( _owner_->getName() ) ){ continue; }
      _cache_.dialCollectionsRefList.emplace_back( &dialCollection );
    }
  }

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
      for( auto& bin : dialCollection->getDialBinSet().getBinList() ) {
        for( auto& edges : bin.getEdgesList() ){
          GenericToolbox::addIfNotInVector(edges.varName, indexRequests);
        }
      }
    }
    LogInfo << "DialCollection requests for indexing: " << GenericToolbox::toString(indexRequests) << std::endl;
    for( auto& var : indexRequests ){ _cache_.addVarRequestedForIndexing(var); }
  }

  std::vector<std::string> varForStorageListBuffer{};

  // sample binning -> indexing only
  {
    std::vector<std::string> varForIndexingListBuffer{};
    varForIndexingListBuffer = _cache_.propagatorPtr->getSampleSet().fetchRequestedVariablesForIndexing();
    LogInfo << "Samples variable request for indexing: " << GenericToolbox::toString(varForIndexingListBuffer) << std::endl;
    for( auto &var: varForIndexingListBuffer ){ _cache_.addVarRequestedForIndexing(var); }
  }

  // plotGen -> for storage as we need those in prefit and postfit
  {
    std::vector<std::string> varForStorageListBuffer{};
    varForStorageListBuffer = _cache_.propagatorPtr->getPlotGenerator().fetchListOfVarToPlot(not _parameters_.useMcContainer);
    if( _parameters_.useMcContainer ){
      for( auto& var : _cache_.propagatorPtr->getPlotGenerator().fetchListOfSplitVarNames() ){
        GenericToolbox::addIfNotInVector(var, varForStorageListBuffer);
      }
    }
    LogInfo << "PlotGenerator variable request for storage: " << GenericToolbox::toString(varForStorageListBuffer) << std::endl;
    for( auto& var : varForStorageListBuffer ){ _cache_.addVarRequestedForStorage(var); }
  }

  // storage requested by user
  {
    std::vector<std::string> varForStorageListBuffer{};
    varForStorageListBuffer = _parameters_.additionalVarsStorage;
    LogInfo << "Additional var requests for storage:" << GenericToolbox::toString(varForStorageListBuffer) << std::endl;
    for (auto &var: varForStorageListBuffer) { _cache_.addVarRequestedForStorage(var); }
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

    LogInfo << "EventVariableTransformation requests for indexing: " << GenericToolbox::toString(indexRequests) << std::endl;
    for( auto& var : indexRequests ){ _cache_.addVarRequestedForIndexing(var); }
  }

  LogInfo << "Vars requested for indexing: " << GenericToolbox::toString(_cache_.varsRequestedForIndexing, false) << std::endl;
  LogInfo << "Vars requested for storage: " << GenericToolbox::toString(_cache_.varsRequestedForStorage, false) << std::endl;

  // Now build the var to leaf translation
  for( auto& var : _cache_.varsRequestedForIndexing ){
    _cache_.varToLeafDict[var].first = var;    // default is the same name
    _cache_.varToLeafDict[var].second = false; // is dummy branch?

    // strip brackets
    _cache_.varToLeafDict[var].first = GenericToolbox::stripBracket(_cache_.varToLeafDict[var].first, '[', ']');

    // look for override requests
    if( GenericToolbox::doesKeyIsInMap(_cache_.varToLeafDict[var].first, _parameters_.variableDict) ){
      // leafVar will actually be the override leaf name while event will keep the original name
      _cache_.varToLeafDict[var].first = _parameters_.variableDict[_cache_.varToLeafDict[var].first];
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

}
void DataDispenser::preAllocateMemory(){
  LogInfo << "Pre-allocating memory..." << std::endl;
  /// \brief The following lines are necessary since the events might get
  /// resized while being in multi-thread Because std::vector is insuring
  /// continuous memory allocation, a resize sometimes lead to the full moving
  /// of a vector memory. This is not thread safe, so better ensure the vector
  /// won't have to do this by allocating the right event size.

  // MEMORY CLAIM?
  TChain treeChain(_parameters_.treePath.c_str());
  for( const auto& file: _parameters_.filePathList){
    std::string name = GenericToolbox::expandEnvironmentVariables(file);
    if (name != file) {
      LogWarning << "Filename expanded to: " << name << std::endl;
    }
    treeChain.Add(name.c_str());
  }

  GenericToolbox::LeafCollection lCollection;
  lCollection.setTreePtr( &treeChain );
  for( auto& var : _cache_.varsRequestedForIndexing ){
    // look for override requests
    lCollection.addLeafExpression(
        GenericToolbox::doesKeyIsInMap(var, _parameters_.variableDict) ?
        _parameters_.variableDict[var] : var
    );
  }
  lCollection.initialize();

  PhysicsEvent eventPlaceholder;
  eventPlaceholder.setDataSetIndex(_owner_->getDataSetIndex());
  eventPlaceholder.setCommonVarNameListPtr(std::make_shared<std::vector<std::string>>(_cache_.varsRequestedForStorage));

  std::vector<const GenericToolbox::LeafForm*> leafFormToVarList{};
  for( auto& storageVar : *eventPlaceholder.getCommonVarNameListPtr() ){
    leafFormToVarList.emplace_back( lCollection.getLeafFormPtr(
        GenericToolbox::doesKeyIsInMap(storageVar, _parameters_.variableDict) ?
        _parameters_.variableDict[storageVar] : storageVar
    ));
  }

  eventPlaceholder.allocateMemory( leafFormToVarList );

  LogInfo << "Reserving event memory..." << std::endl;
  _cache_.sampleIndexOffsetList.resize(_cache_.samplesToFillList.size());
  _cache_.sampleEventListPtrToFill.resize(_cache_.samplesToFillList.size());
  for( size_t iSample = 0 ; iSample < _cache_.sampleNbOfEvents.size() ; iSample++ ){
    auto* container = &_cache_.samplesToFillList[iSample]->getDataContainer();
    if(_parameters_.useMcContainer) container = &_cache_.samplesToFillList[iSample]->getMcContainer();

    _cache_.sampleEventListPtrToFill[iSample] = &container->eventList;
    _cache_.sampleIndexOffsetList[iSample] = _cache_.sampleEventListPtrToFill[iSample]->size();
    container->reserveEventMemory(_owner_->getDataSetIndex(), _cache_.sampleNbOfEvents[iSample], eventPlaceholder);
  }

  LogInfo << "Filling var index cache for bin edges..." << std::endl;
  for( auto* samplePtr : _cache_.samplesToFillList ){
    for( auto& bin : samplePtr->getBinning().getBinList() ){
      for( auto& edges : bin.getEdgesList() ){
        edges.varIndexCache = GenericToolbox::findElementIndex( edges.varName, _cache_.varsRequestedForIndexing );
      }
    }
  }


  size_t nEvents = treeChain.GetEntries();
  if( _parameters_.useMcContainer ){
    if( not _cache_.dialCollectionsRefList.empty() ){
      LogInfo << "Creating slots for event-by-event dials..." << std::endl;
      size_t nDialsMaxPerEvent{0};
      for( auto& dialCollection : _cache_.dialCollectionsRefList ){
        LogScopeIndent;
        nDialsMaxPerEvent += 1;
        if( dialCollection->isBinned() ){
          // Filling var indexes for faster eval with PhysicsEvent:
          for( auto& bin : dialCollection->getDialBinSet().getBinList() ){
            for( auto& edges : bin.getEdgesList() ){
              edges.varIndexCache = GenericToolbox::findElementIndex( edges.varName, _cache_.varsRequestedForIndexing );
            }
          }
        }
        else if( not dialCollection->getGlobalDialLeafName().empty() ){
          // Reserve memory for additional dials (those on a tree leaf)
          auto dialType = dialCollection->getGlobalDialType();
          LogInfo << dialCollection->getTitle() << ": creating " << nEvents;
          LogInfo << " slots for " << dialType << std::endl;

          dialCollection->getDialBaseList().clear();
          dialCollection->getDialBaseList().resize(nEvents);
        }
        else{
          LogThrow("DEV ERROR: not binned, not event-by-event?");
        }
      }
      _cache_.propagatorPtr->getEventDialCache().allocateCacheEntries(nEvents, nDialsMaxPerEvent);
    }
    else{
      // all events should be referenced in the cache even with 0 dial
      _cache_.propagatorPtr->getEventDialCache().allocateCacheEntries(nEvents, 0);
    }
  }
}
void DataDispenser::readAndFill(){
  LogWarning << "Reading dataset and loading..." << std::endl;

  if( not _parameters_.nominalWeightFormulaStr.empty() ){
    LogInfo << "Nominal weight: \"" << _parameters_.nominalWeightFormulaStr << "\"" << std::endl;
  }
  if( not _parameters_.dialIndexFormula.empty() ){
    LogInfo << "Dial index for TClonesArray: \"" << _parameters_.dialIndexFormula << "\"" << std::endl;
  }

  LogWarning << "Loading and indexing..." << std::endl;
  if(not _owner_->isDevSingleThreadEventLoaderAndIndexer() and GundamGlobals::getParallelWorker().getNbThreads() > 1 ){
    ROOT::EnableThreadSafety(); // EXTREMELY IMPORTANT
    GundamGlobals::getParallelWorker().addJob(__METHOD_NAME__, [&](int iThread_){ this->fillFunction(iThread_); });
    GundamGlobals::getParallelWorker().runJob(__METHOD_NAME__);
    GundamGlobals::getParallelWorker().removeJob(__METHOD_NAME__);
  }
  else{
    this->fillFunction(-1); // for better debug breakdown
  }

  LogInfo << "Shrinking lists..." << std::endl;
  for( size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
    auto* container = &_cache_.samplesToFillList[iSample]->getDataContainer();
    if(_parameters_.useMcContainer) container = &_cache_.samplesToFillList[iSample]->getMcContainer();
    container->shrinkEventList( _cache_.sampleIndexOffsetList[iSample] );
  }

}
void DataDispenser::loadFromHistContent(){
  LogWarning << "Creating dummy PhysicsEvent entries for loading hist content" << std::endl;

  // non-trivial as we need to propagate systematics. Need to merge with the original data loader, but not straight forward?
  LogThrowIf( _parameters_.useMcContainer, "Hist loader not implemented for MC containers" );

  // counting events
  _cache_.sampleNbOfEvents.resize(_cache_.samplesToFillList.size());
  _cache_.sampleIndexOffsetList.resize(_cache_.samplesToFillList.size());
  _cache_.sampleEventListPtrToFill.resize(_cache_.samplesToFillList.size());

  PhysicsEvent eventPlaceholder;
  eventPlaceholder.setDataSetIndex(_owner_->getDataSetIndex());
  eventPlaceholder.setEventWeight(0); // default.

  // claiming event memory
  for( size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){

    eventPlaceholder.setCommonVarNameListPtr(
        std::make_shared<std::vector<std::string>>(
            _cache_.samplesToFillList[iSample]->getBinning().buildVariableNameList()
        )
    );
    for( auto& varHolder : eventPlaceholder.getVarHolderList() ){
      varHolder.emplace_back( double(0.) );
    }
    eventPlaceholder.resizeVarToDoubleCache();

    // one event per bin
    _cache_.sampleNbOfEvents[iSample] = _cache_.samplesToFillList[iSample]->getBinning().getBinList().size();

    // fetch event container
    auto* container = &_cache_.samplesToFillList[iSample]->getDataContainer();

    _cache_.sampleEventListPtrToFill[iSample] = &container->eventList;
    _cache_.sampleIndexOffsetList[iSample] = _cache_.sampleEventListPtrToFill[iSample]->size();
    container->reserveEventMemory( _owner_->getDataSetIndex(), _cache_.sampleNbOfEvents[iSample], eventPlaceholder );

    // indexing according to the binning
    for( size_t iEvent=_cache_.sampleIndexOffsetList[iSample] ; iEvent < container->eventList.size() ; iEvent++ ){
      container->eventList[iEvent].setSampleBinIndex( int( iEvent ) );
    }
  }

  LogInfo << "Reading external hist files..." << std::endl;

  // read hist content from file
  TFile* fHist{nullptr};
  LogThrowIf( not GenericToolbox::Json::doKeyExist(_parameters_.fromHistContent, "fromRootFile"), "No root file provided." );
  auto filePath = GenericToolbox::Json::fetchValue<std::string>(_parameters_.fromHistContent, "fromRootFile");

  LogInfo << "Opening: " << filePath << std::endl;

  LogThrowIf( not GenericToolbox::doesTFileIsValid(filePath), "Could not open file: " << filePath );
  fHist = TFile::Open(filePath.c_str());
  LogThrowIf(fHist == nullptr, "Could not open file: " << filePath);

  LogThrowIf( not GenericToolbox::Json::doKeyExist(_parameters_.fromHistContent, "sampleList"), "Could not find samplesList." );
  auto sampleList = GenericToolbox::Json::fetchValue<JsonType>(_parameters_.fromHistContent, "sampleList");
  for( auto& sample : _cache_.samplesToFillList ){
    LogScopeIndent;

    auto entry = GenericToolbox::Json::fetchMatchingEntry( sampleList, "name", sample->getName() );
    LogContinueIf( entry.empty(), "Could not find sample histogram: " << sample->getName() );

    LogThrowIf( not GenericToolbox::Json::doKeyExist( entry, "hist" ), "No hist name provided for " << sample->getName() );
    auto histName = GenericToolbox::Json::fetchValue<std::string>( entry, "hist" );
    LogInfo << "Filling sample \"" << sample->getName() << "\" using hist with name: " << histName << std::endl;

    LogThrowIf( not GenericToolbox::Json::doKeyExist( entry, "axis" ), "No axis names provided for " << sample->getName() );
    auto axisNameList = GenericToolbox::Json::fetchValue<std::vector<std::string>>(entry, "axis");

    auto* hist = fHist->Get<THnD>( histName.c_str() );
    LogThrowIf( hist == nullptr, "Could not find THnD \"" << histName << "\" within " << fHist->GetPath() );

    int nBins = 1;
    for( int iDim = 0 ; iDim < hist->GetNdimensions() ; iDim++ ){
      nBins *= hist->GetAxis(iDim)->GetNbins();
    }

    LogAlertIf( nBins != int( sample->getBinning().getBinList().size() ) ) <<
                                                                           "Mismatching bin number for " << sample->getName() << ":" << std::endl
                                                                           << GET_VAR_NAME_VALUE(nBins) << std::endl
                                                                           << GET_VAR_NAME_VALUE(sample->getBinning().getBinList().size()) << std::endl;

    auto* container = &sample->getDataContainer();
    for( size_t iBin = 0 ; iBin < sample->getBinning().getBinList().size() ; iBin++ ){
      auto target = sample->getBinning().getBinList()[iBin].generateBinTarget( axisNameList );
      auto histBinIndex = hist->GetBin( target.data() ); // bad fetch..?

      container->eventList[iBin].setSampleIndex( sample->getIndex() );
      for( size_t iVar = 0 ; iVar < target.size() ; iVar++ ){
        container->eventList[iBin].setVariable( target[iVar], axisNameList[iVar] );
      }
      container->eventList[iBin].setBaseWeight(hist->GetBinContent(histBinIndex));
      container->eventList[iBin].resetEventWeight();
    }

  }

  fHist->Close();
}

std::unique_ptr<TChain> DataDispenser::openChain(bool verbose_){
  LogInfoIf(verbose_) << "Opening ROOT files containing events..." << std::endl;

  std::unique_ptr<TChain> treeChain(std::make_unique<TChain>(_parameters_.treePath.c_str()));
  for( const auto& file: _parameters_.filePathList){
    std::string name = GenericToolbox::expandEnvironmentVariables(file);
    GenericToolbox::replaceSubstringInsideInputString(name, "//", "/");
    if( verbose_ ){
      LogScopeIndent;
      LogWarning << name << std::endl;
    }
    treeChain->Add(name.c_str());
  }

  return treeChain;
}

void DataDispenser::eventSelectionFunction(int iThread_){

  int nThreads{GundamGlobals::getParallelWorker().getNbThreads()};
  if( iThread_ == -1 ){ iThread_ = 0; nThreads = 1; }

  // Opening ROOT file...
  auto treeChain{this->openChain(false)};

  GenericToolbox::LeafCollection lCollection;
  lCollection.setTreePtr( treeChain.get() );

  LogInfoIf(iThread_ == 0) << "Defining selection formulas..." << std::endl;

  // global cut
  int selectionCutLeafFormIndex{-1};
  if( not _parameters_.selectionCutFormulaStr.empty() ){
    LogInfoIf(iThread_ == 0) << "Global selection cut: \"" << _parameters_.selectionCutFormulaStr << "\"" << std::endl;
    selectionCutLeafFormIndex = lCollection.addLeafExpression( _parameters_.selectionCutFormulaStr );
  }

  // sample cuts
  GenericToolbox::TablePrinter tableSelectionCuts;
  tableSelectionCuts.setColTitles({{"Sample"}, {"Selection Cut"}});

  struct SampleCut{
    int sampleIndex{-1};
    int cutIndex{-1};
  };
  std::vector<SampleCut> sampleCutList;
  sampleCutList.reserve( _cache_.samplesToFillList.size() );

  for( int iSample = 0; iSample < int(_cache_.samplesToFillList.size()) ; iSample++ ){
    auto* samplePtr = _cache_.samplesToFillList[iSample];
    sampleCutList.emplace_back();
    sampleCutList.back().sampleIndex = iSample;

    std::string selectionCut = samplePtr->getSelectionCutsStr();
    for (auto &replaceEntry: _cache_.varsToOverrideList) {
      GenericToolbox::replaceSubstringInsideInputString(
          selectionCut, replaceEntry, _parameters_.variableDict[replaceEntry]
      );
    }

    if( selectionCut.empty() ){ continue; }

    sampleCutList.back().cutIndex = lCollection.addLeafExpression( selectionCut );
    tableSelectionCuts << samplePtr->getName() << GenericToolbox::TablePrinter::Action::NextColumn;
    tableSelectionCuts << selectionCut << GenericToolbox::TablePrinter::Action::NextLine;

  }
  if( iThread_==0 ){ tableSelectionCuts.printTable(); }

  lCollection.initialize();

  GenericToolbox::VariableMonitor readSpeed("bytes");

  // Multi-thread index splitting
  Long64_t nEvents = treeChain->GetEntries();
  Long64_t iGlobal = 0;

  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices( iThread_, nThreads, nEvents );

  // Load the branches
  treeChain->LoadTree( bounds.first );

  // for each event, which sample is active?
  std::string progressTitle = "Performing event selection on " + this->getTitle() + "...";
  std::stringstream ssProgressTitle;
  TFile *lastFilePtr{nullptr};

  for ( Long64_t iEntry = bounds.first ; iEntry < bounds.second ; iEntry++ ) {
    if( iThread_ == 0 ){
      readSpeed.addQuantity(treeChain->GetEntry(iEntry)*nThreads);
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
      treeChain->GetEntry(iEntry);
    }

    if ( selectionCutLeafFormIndex != -1 ){
      if( lCollection.getLeafFormList()[selectionCutLeafFormIndex].evalAsDouble() == 0 ){
        for (size_t iSample = 0; iSample < _cache_.samplesToFillList.size(); iSample++) {
          _cache_.threadSelectionResults[iThread_].eventIsInSamplesList[iEntry][iSample] = false;
        }
        if (GundamGlobals::getVerboseLevel() == VerboseLevel::INLOOP_TRACE) {
          LogTrace << "Event #" << treeChain->GetFileNumber() << ":" << treeChain->GetReadEntry()
                   << " rejected because of " << _parameters_.selectionCutFormulaStr << std::endl;
        }
        continue;
      }
    }

    for( auto& sampleCut : sampleCutList ){

      // no cut?
      if( sampleCut.cutIndex == -1 ){
        _cache_.threadSelectionResults[iThread_].eventIsInSamplesList[iEntry][sampleCut.sampleIndex] = true;
        _cache_.threadSelectionResults[iThread_].sampleNbOfEvents[sampleCut.sampleIndex]++;
        if (GundamGlobals::getVerboseLevel() == VerboseLevel::INLOOP_TRACE) {
          LogDebug << "Event #" << treeChain->GetFileNumber() << ":" << treeChain->GetReadEntry()
                   << " included as sample " << sampleCut.sampleIndex << " (NO SELECTION CUT)" << std::endl;
        }
      }
        // pass cut?
      else if( lCollection.getLeafFormList()[sampleCut.cutIndex].evalAsDouble() != 0 ){
        _cache_.threadSelectionResults[iThread_].eventIsInSamplesList[iEntry][sampleCut.sampleIndex] = true;
        _cache_.threadSelectionResults[iThread_].sampleNbOfEvents[sampleCut.sampleIndex]++;
        if (GundamGlobals::getVerboseLevel() == VerboseLevel::INLOOP_TRACE) {
          LogDebug << "Event #" << treeChain->GetFileNumber() << ":" << treeChain->GetReadEntry()
                   << " included as sample " << sampleCut.sampleIndex << " because of "
                   << lCollection.getLeafFormList()[sampleCut.cutIndex].getSummary() << std::endl;
        }
      }
        // don't pass cut?
      else {
        if (GundamGlobals::getVerboseLevel() == VerboseLevel::INLOOP_TRACE) {
          LogTrace << "Event #" << treeChain->GetFileNumber() << ":" << treeChain->GetReadEntry()
                   << " rejected as sample " << sampleCut.sampleIndex << " because of "
                   << lCollection.getLeafFormList()[sampleCut.cutIndex].getSummary() << std::endl;
        }
      }
    }

  } // iEvent

  if( iThread_ == 0 ){ GenericToolbox::displayProgressBar(nEvents, nEvents, ssProgressTitle.str()); }

}
void DataDispenser::fillFunction(int iThread_){

  int nThreads = GundamGlobals::getParallelWorker().getNbThreads();
  if( iThread_ == -1 ){ iThread_ = 0; nThreads = 1; } // special mode

  auto treeChain = this->openChain();

  GenericToolbox::LeafCollection lCollection;
  lCollection.setTreePtr( treeChain.get() );

  // nominal weight
  TTreeFormula* nominalWeightTreeFormula{nullptr};
  if( not _parameters_.nominalWeightFormulaStr.empty() ){
    auto idx = size_t(lCollection.addLeafExpression( _parameters_.nominalWeightFormulaStr ));
    nominalWeightTreeFormula = (TTreeFormula*) idx; // tweaking types. Ptr will be attributed after init
  }

  // dial array index
  TTreeFormula* dialIndexTreeFormula{nullptr};
  if( not _parameters_.dialIndexFormula.empty() ){
    auto idx = size_t(lCollection.addLeafExpression( _parameters_.dialIndexFormula ));
    dialIndexTreeFormula = (TTreeFormula*) idx; // tweaking types. Ptr will be attributed after init
  }


  // variables definition
  std::vector<const GenericToolbox::LeafForm*> leafFormIndexingList{};
  std::vector<const GenericToolbox::LeafForm*> leafFormStorageList{};
  for( auto& var : _cache_.varsRequestedForIndexing ){
    std::string leafExp{var};
    if( GenericToolbox::doesKeyIsInMap( var, _parameters_.variableDict ) ){
      leafExp = _parameters_.variableDict[leafExp];
    }
    auto idx = size_t(lCollection.addLeafExpression(leafExp));
    leafFormIndexingList.emplace_back( (GenericToolbox::LeafForm*) idx ); // tweaking types
  }
  for( auto& var : _cache_.varsRequestedForStorage ){
    std::string leafExp{var};
    if( GenericToolbox::doesKeyIsInMap( var, _parameters_.variableDict ) ){
      leafExp = _parameters_.variableDict[leafExp];
    }
    auto idx = size_t(lCollection.getLeafExpIndex(leafExp));
    leafFormStorageList.emplace_back( (GenericToolbox::LeafForm*) idx ); // tweaking types
  }

  lCollection.initialize();

  // grab ptr address now
  if( not _parameters_.nominalWeightFormulaStr.empty() ){
    nominalWeightTreeFormula = lCollection.getLeafFormList()[(size_t) nominalWeightTreeFormula].getTreeFormulaPtr().get();
  }
  if( not _parameters_.dialIndexFormula.empty() ){
    dialIndexTreeFormula = lCollection.getLeafFormList()[(size_t) dialIndexTreeFormula].getTreeFormulaPtr().get();
  }
  for( auto& lfInd: leafFormIndexingList ){ lfInd = &(lCollection.getLeafFormList()[(size_t) lfInd]); }
  for( auto& lfSto: leafFormStorageList ){ lfSto = &(lCollection.getLeafFormList()[(size_t) lfSto]); }

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
              << GenericToolbox::toString(
                  varTransformForIndexingList,
                  [](const EventVarTransformLib* elm_){ return "\"" + elm_->getName() + "\"";}, false)
              << std::endl;
    }
    if( not varTransformForStorageList.empty() ){
      LogInfo << "EventVarTransformLib used for storage: "
              << GenericToolbox::toString(
                  varTransformForStorageList,
                  []( const EventVarTransformLib* elm_){ return "\"" + elm_->getName() + "\""; }, false)
              << std::endl;
    }
  }

  // buffer that will store the data for indexing
  PhysicsEvent eventIndexingBuffer;
  eventIndexingBuffer.setDataSetIndex(_owner_->getDataSetIndex());
  eventIndexingBuffer.setCommonVarNameListPtr(std::make_shared<std::vector<std::string>>(_cache_.varsRequestedForIndexing));
  eventIndexingBuffer.allocateMemory(leafFormIndexingList);

  PhysicsEvent eventStorageBuffer;
  eventStorageBuffer.setDataSetIndex(_owner_->getDataSetIndex());
  eventStorageBuffer.setCommonVarNameListPtr(std::make_shared<std::vector<std::string>>(_cache_.varsRequestedForStorage));
  eventStorageBuffer.allocateMemory(leafFormStorageList);

  if(iThread_ == 0){
    LogInfo << "Feeding event variables with:" << std::endl;
    GenericToolbox::TablePrinter table;

    table << "Variable" << GenericToolbox::TablePrinter::NextColumn;
    table << "LeafForm" << GenericToolbox::TablePrinter::NextColumn;
    table << "Transforms" << GenericToolbox::TablePrinter::NextLine;

    for( size_t iVar = 0 ; iVar < _cache_.varsRequestedForIndexing.size() ; iVar++ ){
      std::string var = _cache_.varsRequestedForIndexing[iVar];

      // line color?
      if( GenericToolbox::doesElementIsInVector(var, _cache_.varsRequestedForStorage)){
        table.setColorBuffer(GenericToolbox::ColorCodes::blueBackground);
      }
      else if(
          leafFormIndexingList[iVar]->getLeafTypeName() == "TClonesArray"
          or leafFormIndexingList[iVar]->getLeafTypeName() == "TGraph"
          ){
        table.setColorBuffer( GenericToolbox::ColorCodes::magentaBackground );
      }

      table << var << GenericToolbox::TablePrinter::NextColumn;

      table << leafFormIndexingList[iVar]->getPrimaryExprStr() << "/" << leafFormIndexingList[iVar]->getLeafTypeName();
      table << GenericToolbox::TablePrinter::NextColumn;

      std::vector<std::string> transformsList;
      for( auto* varTransformForIndexing : varTransformForIndexingList ){
        if( varTransformForIndexing->getOutputVariableName() == var ){
          transformsList.emplace_back(varTransformForIndexing->getName());
        }
      }
      table << GenericToolbox::toString(transformsList) << GenericToolbox::TablePrinter::NextColumn;
    }


    table.printTable();

    // printing legend
    LogInfo(Logger::Color::BG_BLUE)    << "      " << Logger::getColorEscapeCode(Logger::Color::RESET) << " -> Variables stored in RAM" << std::endl;
    LogInfo(Logger::Color::BG_MAGENTA) << "      " << Logger::getColorEscapeCode(Logger::Color::RESET) << " -> Dials stored in RAM" << std::endl;

    if( _owner_->isDevSingleThreadEventLoaderAndIndexer() ){
      LogAlert << "Loading data in single thread (devSingleThreadEventLoaderAndIndexer option set to true)" << std::endl;
    }
  }

  // Try to read TTree the closest to sequentially possible
  Long64_t nEvents{treeChain->GetEntries()};

  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices( iThread_, nThreads, nEvents );

  // Load the branches
  treeChain->LoadTree(bounds.first);

  // IO speed monitor
  GenericToolbox::VariableMonitor readSpeed("bytes");

  std::string progressTitle = "Loading and indexing...";
  std::stringstream ssProgressBar;

  for( Long64_t iEntry = bounds.first ; iEntry < bounds.second; iEntry++ ){

    if( iThread_ == 0 ){
      if( GenericToolbox::showProgressBar(iEntry*nThreads, nEvents) ){

        ssProgressBar.str("");

        ssProgressBar << LogInfo.getPrefixString() << "Reading from disk: "
                      << GenericToolbox::padString(GenericToolbox::parseSizeUnits(readSpeed.getTotalAccumulated()), 8) << " ("
                      << GenericToolbox::padString(GenericToolbox::parseSizeUnits(readSpeed.evalTotalGrowthRate()), 8) << "/s)";

        int cpuPercent = int(GenericToolbox::getCpuUsageByProcess());
        ssProgressBar << " / CPU efficiency: " << GenericToolbox::padString(std::to_string(cpuPercent/nThreads), 3,' ')
                      << "% / RAM: " << GenericToolbox::parseSizeUnits( double(GenericToolbox::getProcessMemoryUsage()) ) << std::endl;

        ssProgressBar << LogInfo.getPrefixString() << progressTitle;
        GenericToolbox::displayProgressBar(iEntry*nThreads, nEvents, ssProgressBar.str());
      }
    }

    bool hasSample =
        std::any_of(
            _cache_.eventIsInSamplesList[iEntry].begin(), _cache_.eventIsInSamplesList[iEntry].end(),
            [](bool isInSample_){ return isInSample_; }
        );
    if( not hasSample ){ continue; }

    Int_t nBytes{ treeChain->GetEntry(iEntry) };

    // monitor
    if( iThread_ == 0 ){
      readSpeed.addQuantity(nBytes * nThreads);
    }

    if( nominalWeightTreeFormula != nullptr ){
      eventIndexingBuffer.setBaseWeight(nominalWeightTreeFormula->EvalInstance());
      if(eventIndexingBuffer.getBaseWeight() < 0 ){
        LogError << "Negative nominal weight:" << std::endl;

        LogError << "Event buffer is: " << eventIndexingBuffer.getSummary() << std::endl;

        LogError << "Formula leaves:" << std::endl;
        for( int iLeaf = 0 ; iLeaf < nominalWeightTreeFormula->GetNcodes() ; iLeaf++ ){
          if( nominalWeightTreeFormula->GetLeaf(iLeaf) == nullptr ) continue; // for "Entry$" like dummy leaves
          LogError << "Leaf: " << nominalWeightTreeFormula->GetLeaf(iLeaf)->GetName() << "[0] = " << nominalWeightTreeFormula->GetLeaf(iLeaf)->GetValue(0) << std::endl;
        }

        LogThrow("Negative nominal weight");
      }
      if( eventIndexingBuffer.getBaseWeight() == 0 ){
        continue;
      } // skip this event
    }

    size_t nSample{_cache_.samplesToFillList.size()};
    for( size_t iSample = 0 ; iSample < nSample ; iSample++ ){

      if( not _cache_.eventIsInSamplesList[iEntry][iSample] ){ continue; }

      // Getting loaded data in tEventBuffer
      eventIndexingBuffer.copyData( leafFormIndexingList );

      // Propagate variable transformations for indexing
      for( auto* varTransformPtr : varTransformForIndexingList ){
        varTransformPtr->evalAndStore(eventIndexingBuffer);
      }

      // Has valid bin?
      eventIndexingBuffer.setSampleBinIndex(
          eventIndexingBuffer.findBinIndex(
              _cache_.samplesToFillList[iSample]->getBinning()
          )
      );

      // No bin found -> next sample
      if( eventIndexingBuffer.getSampleBinIndex() == -1){ break; }

      // OK, now we have a valid fit bin. Let's claim an index.
      // Shared index among threads
      size_t sampleEventIndex{};
      EventDialCache::IndexedCacheEntry* eventDialCacheEntry{nullptr};
      {
        std::unique_lock<std::mutex> lock(GundamGlobals::getThreadMutex());
        if( _parameters_.useMcContainer ){

          if( _parameters_.debugNbMaxEventsToLoad != 0 ){
            // check if the limit has been reached
            if( _cache_.propagatorPtr->getEventDialCache().getFillIndex() >= _parameters_.debugNbMaxEventsToLoad ){
              LogAlertIf(iThread_==0) << std::endl << std::endl; // flush pBar
              LogAlertIf(iThread_==0) << "debugNbMaxEventsToLoad: Event number cap reached (";
              LogAlertIf(iThread_==0) << _parameters_.debugNbMaxEventsToLoad << ")" << std::endl;
              return;
            }
          }

          eventDialCacheEntry = _cache_.propagatorPtr->getEventDialCache().fetchNextCacheEntry();
        }
        sampleEventIndex = _cache_.sampleIndexOffsetList[iSample]++;
      }

      // Get the next free event in our buffer
      PhysicsEvent *eventPtr = &(*_cache_.sampleEventListPtrToFill[iSample])[sampleEventIndex];

      // fill meta info
      eventPtr->setEntryIndex( iEntry );
      eventPtr->setBaseWeight( eventIndexingBuffer.getBaseWeight() );
      eventPtr->setSampleIndex( _cache_.samplesToFillList[iSample]->getIndex() );
      eventPtr->setSampleBinIndex( eventIndexingBuffer.getSampleBinIndex() );
      eventPtr->resetEventWeight();

      // drop the content of the leaves
      eventPtr->copyData( leafFormStorageList );

      // Propagate transformation for storage -> use the previous results calculated for indexing
      for( auto *varTransformPtr: varTransformForStorageList ){
        varTransformPtr->storeCachedOutput(*eventPtr);
      }

      // Now the event is ready. Let's index the dials:
      if ( eventDialCacheEntry != nullptr) {
        // there should always be a cache entry even if no dials are applied.
        // This cache is actually used to write MC events with dials in output tree
        eventDialCacheEntry->event.sampleIndex = std::size_t(_cache_.samplesToFillList[iSample]->getIndex());
        eventDialCacheEntry->event.eventIndex = sampleEventIndex;

        auto* dialEntryPtr = &eventDialCacheEntry->dials[0];

        for( auto *dialCollectionRef: _cache_.dialCollectionsRefList ){

          // dial collections may come with a condition formula
          if( dialCollectionRef->getApplyConditionFormula() != nullptr ){
            if( eventIndexingBuffer.evalFormula(dialCollectionRef->getApplyConditionFormula().get()) == 0 ){
              // next dialSet
              continue;
            }
          }

          int iCollection = dialCollectionRef->getIndex();

          if     ( dialCollectionRef->isBinned() ){

            // is only one bin with no condition:
            if( dialCollectionRef->getDialBaseList().size() == 1 and dialCollectionRef->getDialBinSet().getBinList().empty() ){
              // if is it NOT a DialBinned -> this is the one we are
              // supposed to use
              dialEntryPtr->collectionIndex = iCollection;
              dialEntryPtr->interfaceIndex = 0;
              dialEntryPtr++;
            }
            else{
              auto dialBinIdx = eventIndexingBuffer.findBinIndex( dialCollectionRef->getDialBinSet() );
              if( dialBinIdx != -1 ){
                dialEntryPtr->collectionIndex = iCollection;
                dialEntryPtr->interfaceIndex = dialBinIdx;
                dialEntryPtr++;
              }
            }
          }
          else if( not dialCollectionRef->getGlobalDialLeafName().empty() ){
            // Event-by-event dial?
            // grab the dial as a general TObject -> let the factory figure out what to do with it
            auto *dialObjectPtr = (TObject *) *(
                (TObject **) eventIndexingBuffer.getVariableAddress(
                    dialCollectionRef->getGlobalDialLeafName()
                )
            );

            // Extra-step for selecting the right dial with TClonesArray
            if (not strcmp(dialObjectPtr->ClassName(), "TClonesArray")) {
              dialObjectPtr = ((TClonesArray *) dialObjectPtr)->At(
                  (dialIndexTreeFormula == nullptr ? 0 : int(dialIndexTreeFormula->EvalInstance()))
              );
            }

            // Do the unique_ptr dance so that memory gets deleted if
            // there is an exception (being stupidly paranoid).
            DialBaseFactory factory{};
            std::unique_ptr<DialBase> dialBase(
                factory.makeDial(
                    dialCollectionRef->getTitle(),
                    dialCollectionRef->getGlobalDialType(),
                    dialCollectionRef->getGlobalDialSubType(),
                    dialObjectPtr,
                    false
                )
            );

            // dialBase is valid -> store it
            if (dialBase != nullptr) {
              size_t freeSlotDial = dialCollectionRef->getNextDialFreeSlot();
              dialBase->setAllowExtrapolation(dialCollectionRef->isAllowDialExtrapolation());
              dialCollectionRef->getDialBaseList()[freeSlotDial] = DialCollection::DialBaseObject(
                  dialBase.release());

              dialEntryPtr->collectionIndex = iCollection;
              dialEntryPtr->interfaceIndex = freeSlotDial;
              dialEntryPtr++;
            }
          }
          else {
            LogThrow("neither an event by event dial, nor a binned dial");
          }

        } // dial collection loop
      }


    } // samples
  } // entries
  if( iThread_ == 0 ){
    GenericToolbox::displayProgressBar(nEvents, nEvents, ssProgressBar.str());
  }

}

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
