//
// Created by Adrien BLANCHET on 14/05/2022.
//

#include "DataDispenser.h"
#include "DatasetDefinition.h"
#include "LoaderUtils.h"

#include "Propagator.h"
#include "GundamGlobals.h"

#include "ConfigUtils.h"

#include "DialCollection.h"
#include "TabulatedDialFactory.h"

#include "GundamUtils.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Map.h"
#include "Logger.h"

#include "TTreeFormulaManager.h"
#include "TClonesArray.h"
#include "TChain.h"
#include "THn.h"

#include <unordered_map>
#include <string>
#include <vector>
#include <sstream>


void DataDispenser::configureImpl(){

  GenericToolbox::Json::fillValue(_config_, _parameters_.name, "name");
  LogExitIf(_parameters_.name.empty(), "Dataset name not set.");

  // histograms don't need other parameters
  if( GenericToolbox::Json::doKeyExist( _config_, "fromHistContent" ) ) {
    LogDebugIf(GundamGlobals::isDebug()) << "Dataset \"" << _parameters_.name << "\" will be defined with histogram data." << std::endl;
    auto fromHistConfig( GenericToolbox::Json::fetchValue<JsonType>(_config_, "fromHistContent") );

    _parameters_.fromHistContent.isEnabled = true;
    _parameters_.fromHistContent.rootFilePath = GenericToolbox::Json::fetchValue<std::string>(fromHistConfig, "fromRootFile");

    auto sampleListConfig(GenericToolbox::Json::fetchValue<std::vector<JsonType>>(fromHistConfig, "sampleList"));
    _parameters_.fromHistContent.sampleHistList.reserve(sampleListConfig.size());
    for( auto& sampleConfig : sampleListConfig ){

      auto& sampleHist = _parameters_.fromHistContent.addSampleHist(GenericToolbox::Json::fetchValue<std::string>(sampleConfig, "name"));
      GenericToolbox::Json::fillValue(sampleConfig, sampleHist.hist, "hist");
      GenericToolbox::Json::fillValue(sampleConfig, sampleHist.axisList, {{"axisList"},{"axis"}});
    }

    return;
  }

  // nested
  // load transformations
  int index{0};
  for( auto& varTransform : GenericToolbox::Json::fetchValue(_config_, "variablesTransform", std::vector<JsonType>()) ){
    _parameters_.eventVarTransformList.emplace_back( varTransform );
    _parameters_.eventVarTransformList.back().setIndex(index++);
    _parameters_.eventVarTransformList.back().configure();
    if( not _parameters_.eventVarTransformList.back().isEnabled() ){
      _parameters_.eventVarTransformList.pop_back();
      continue;
    }
  }

  _parameters_.variableDict.clear();
  for( auto& entry : GenericToolbox::Json::fetchValue(_config_, {{"variableDict"}, {"overrideLeafDict"}}, JsonType()) ){
    auto varName = GenericToolbox::Json::fetchValue<std::string>(entry, {{"name"}, {"eventVar"}});
    auto varExpr = GenericToolbox::Json::fetchValue<std::string>(entry, {{"expr"}, {"expression"}, {"leafVar"}});
    _parameters_.variableDict[ varName ] = varExpr;
  }

  GenericToolbox::Json::fillValue(_config_, _parameters_.eventVariableAsWeight, "eventVariableAsWeight");

  // options
  GenericToolbox::Json::fillValue(_config_, _parameters_.globalTreePath, "tree");
  GenericToolbox::Json::fillValue(_config_, _parameters_.filePathList, "filePathList");
  GenericToolbox::Json::fillValue(_config_, _parameters_.additionalVarsStorage, {{"additionalLeavesStorage"}, {"additionalVarsStorage"}});
  GenericToolbox::Json::fillValue(_config_, _parameters_.dummyVariablesList, "dummyVariablesList");
  GenericToolbox::Json::fillValue(_config_, _parameters_.useReweightEngine, {{"useReweightEngine"}, {"useMcContainer"}});
  GenericToolbox::Json::fillValue(_config_, _parameters_.debugNbMaxEventsToLoad, "debugNbMaxEventsToLoad");
  GenericToolbox::Json::fillValue(_config_, _parameters_.dialIndexFormula, "dialIndexFormula");
  GenericToolbox::Json::fillValue(_config_, _parameters_.overridePropagatorConfig, "overridePropagatorConfig");

  GenericToolbox::Json::fillFormula(_config_, _parameters_.selectionCutFormulaStr, "selectionCutFormula", "&&");
  GenericToolbox::Json::fillFormula(_config_, _parameters_.nominalWeightFormulaStr, "nominalWeightFormula", "*");

}
void DataDispenser::initializeImpl(){
  // Nothing else to do other than read config?
  LogWarning << "Initialized data dispenser: " << getTitle() << std::endl;

  for( auto& eventVarTransform: _parameters_.eventVarTransformList ){
    eventVarTransform.initialize();
  }
  // sort them according to their output
  GenericToolbox::sortVector(_parameters_.eventVarTransformList, [](const EventVarTransformLib& a_, const EventVarTransformLib& b_){
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

void DataDispenser::load(Propagator& propagator_){
  LogWarning << "Loading dataset: " << getTitle() << std::endl;
  LogThrowIf(not this->isInitialized(), "Can't load while not initialized.");
  LogThrowIf(not propagator_.isInitialized(), "Can't load while propagator_ is not initialized.");

  _cache_.clear();
  _cache_.propagatorPtr = &propagator_;


  if( not _parameters_.overridePropagatorConfig.empty() ){
    LogWarning << "Reload the propagator config with override options" << std::endl;
    ConfigUtils::ConfigHandler configHandler( _cache_.propagatorPtr->getConfig() );
    configHandler.override( _parameters_.overridePropagatorConfig );
    _cache_.propagatorPtr->configure( configHandler.getConfig() );
    _cache_.propagatorPtr->initialize();
  }

  this->buildSampleToFillList();

  if( _cache_.samplesToFillList.empty() ){
    LogAlert << "No samples were selected for dataset: " << getTitle() << std::endl;
    return;
  }

  if( _parameters_.fromHistContent.isEnabled ){
    this->loadFromHistContent();
    return;
  }

  for( const auto& file: _parameters_.filePathList){
    std::string path = GenericToolbox::expandEnvironmentVariables(file);
    LogExitIf(not GenericToolbox::doesTFileIsValid(path, {_parameters_.globalTreePath}), "Invalid file: " << path);
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
      LogExitIf(_cache_.propagatorPtr->getIThrow()==-1, "<I_TOY> not set.");
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
  int nThreads{getNbParallelCpu()};
  if( _owner_->isDevSingleThreadEventSelection() ) { nThreads = 1; }

  Long64_t nEntries{0};
  {
    auto treeChain{this->openChain(true)};
    nEntries = treeChain->GetEntries();
  }
  LogExitIf(nEntries == 0, "TChain is empty.");
  LogInfo << "Will read " << nEntries << " event entries." << std::endl;

  _cache_.threadSelectionResults.resize(nThreads);
  for( auto& threadResults : _cache_.threadSelectionResults ){
    threadResults.sampleNbOfEvents.resize(_cache_.samplesToFillList.size(), 0);
    threadResults.entrySampleIndexList.resize(nEntries, -1);
  }

  if( not _owner_->isDevSingleThreadEventSelection() ) {
    GenericToolbox::ParallelWorker threadPool;
    threadPool.setNThreads( getNbParallelCpu() );
    threadPool.addJob(__METHOD_NAME__, [this](int iThread_){ this->eventSelectionFunction(iThread_); });
    threadPool.runJob(__METHOD_NAME__);
    threadPool.removeJob(__METHOD_NAME__);
  }
  else {
    this->eventSelectionFunction(-1);
  }

  LogInfo << "Merging thread results..." << std::endl;
  _cache_.sampleNbOfEvents.resize(_cache_.samplesToFillList.size(), 0);
  _cache_.entrySampleIndexList.resize(nEntries, -1);
  for( auto& threadResults : _cache_.threadSelectionResults ){
    // merging nEvents

    for( int iSample = 0 ; iSample < int(_cache_.sampleNbOfEvents.size()) ; iSample++ ){
      _cache_.sampleNbOfEvents[iSample] += threadResults.sampleNbOfEvents[iSample];
    }

    for( size_t iEntry = 0 ; iEntry < int(_cache_.entrySampleIndexList.size()) ; iEntry++ ){
      if( threadResults.entrySampleIndexList[iEntry] != -1 ){
        _cache_.entrySampleIndexList[iEntry] = threadResults.entrySampleIndexList[iEntry];
      }
    }

  }

  LogInfo << "Freeing up thread buffers..." << std::endl;
  _cache_.threadSelectionResults.clear();

  // get total amount
  for(size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
    _cache_.totalNbEvents += _cache_.sampleNbOfEvents[iSample];
  }

  if( _owner_->isShowSelectedEventCount() ){
    LogWarning << "Events passing selection cuts:" << std::endl;
    GenericToolbox::TablePrinter t;
    t.setColTitles({{"Sample"}, {"# of events"}});
    for(size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
      t.addTableLine({
                         _cache_.samplesToFillList[iSample]->getName(),
                         std::to_string(_cache_.sampleNbOfEvents[iSample])
                     });
    }
    t.addTableLine({"Total", std::to_string(_cache_.totalNbEvents)});
    t.printTable();
  }

}
void DataDispenser::fetchRequestedLeaves(){
  LogWarning << "Poll every objects for requested variables..." << std::endl;

  if( _parameters_.useReweightEngine ){
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
      if( not dialCollection->getDialLeafName().empty() ){
        GenericToolbox::addIfNotInVector(dialCollection->getDialLeafName(), indexRequests);
      }
      for( auto& bin : dialCollection->getDialBinSet().getBinList() ) {
        for( auto& edges : bin.getEdgesList() ){
          GenericToolbox::addIfNotInVector(edges.varName, indexRequests);
        }
      }
      for( auto& leafName : dialCollection->getExtraLeafNames()) {
        GenericToolbox::addIfNotInVector(leafName, indexRequests);
      }
    }
    LogInfo << "DialCollection requests for indexing: " << GenericToolbox::toString(indexRequests) << std::endl;
    for( auto& var : indexRequests ){ _cache_.addVarRequestedForIndexing(var); }
  }

  // sample binning -> indexing only
  {
    std::vector<std::string> varForIndexingListBuffer{};
    varForIndexingListBuffer = _cache_.propagatorPtr->getSampleSet().fetchRequestedVariablesForIndexing();
    LogInfo << "Samples variable request for indexing: " << GenericToolbox::toString(varForIndexingListBuffer) << std::endl;
    for( auto &var: varForIndexingListBuffer ){ _cache_.addVarRequestedForIndexing(var); }
  }

  // for event weight
  if( not _parameters_.eventVariableAsWeight.empty() ){
    LogInfo << "Variable for event weight: " << _parameters_.eventVariableAsWeight << std::endl;
    _cache_.addVarRequestedForIndexing(_parameters_.eventVariableAsWeight);
  }

  // plotGen -> for storage as we need those in prefit and postfit
  if( _plotGeneratorPtr_ != nullptr ){
    std::vector<std::string> varForStorageListBuffer{};
    varForStorageListBuffer = _plotGeneratorPtr_->fetchListOfVarToPlot(_parameters_.isData);
    if( not _parameters_.isData ){
      for( auto& var : _plotGeneratorPtr_->fetchListOfSplitVarNames() ){
        GenericToolbox::addIfNotInVector(var, varForStorageListBuffer);
      }
    }
    LogInfo << "PlotGenerator variable request for storage: " << GenericToolbox::toString(varForStorageListBuffer) << std::endl;
    for( auto& var : varForStorageListBuffer ) {
      _cache_.addVarRequestedForIndexing(var);
      GenericToolbox::addIfNotInVector(var, _cache_.propagatorPtr->getSampleSet().getEventVariableNameList());
    }
  }

  // storage requested by user
  {
    std::vector<std::string> varForStorageListBuffer{};
    varForStorageListBuffer = _parameters_.additionalVarsStorage;
    LogInfo << "Additional var requests for storage:" << GenericToolbox::toString(varForStorageListBuffer) << std::endl;
    for (auto &var: varForStorageListBuffer) {
      _cache_.addVarRequestedForIndexing(var);
      GenericToolbox::addIfNotInVector(var, _cache_.propagatorPtr->getSampleSet().getEventVariableNameList());
    }
  }

  // transforms inputs
  if( not _parameters_.eventVarTransformList.empty() ){
    std::vector<std::string> indexRequests;
    for( int iTrans = int(_parameters_.eventVarTransformList.size())-1 ; iTrans >= 0 ; iTrans-- ){
      // in reverse order -> Treat the highest level vars first (they might need lower level variables)
      std::string outVarName = _parameters_.eventVarTransformList[iTrans].getOutputVariableName();
      if( GenericToolbox::doesElementIsInVector( outVarName, _cache_.varsRequestedForIndexing )
          or GenericToolbox::doesElementIsInVector( outVarName, indexRequests )
          ){
        // ok it is needed -> activate dependencies
        for( auto& var: _parameters_.eventVarTransformList[iTrans].fetchRequestedVars() ){
          GenericToolbox::addIfNotInVector(var, indexRequests);
        }
      }
    }

    LogInfo << "EventVariableTransformation requests for indexing: " << GenericToolbox::toString(indexRequests) << std::endl;
    for( auto& var : indexRequests ){ _cache_.addVarRequestedForIndexing(var); }
  }

  // LogInfo << "Vars requested for indexing: " << GenericToolbox::toString(_cache_.varsRequestedForIndexing, false) << std::endl;
  LogInfo << "Vars requested for storage: " << GenericToolbox::toString(_cache_.propagatorPtr->getSampleSet().getEventVariableNameList(), false) << std::endl;

  // Now build the var to leaf translation
  for( auto& var : _cache_.varsRequestedForIndexing ){
    _cache_.varToLeafDict[var].first = var;    // default is the same name
    _cache_.varToLeafDict[var].second = false; // is dummy branch?

    // strip brackets
    _cache_.varToLeafDict[var].first = GenericToolbox::stripBracket(_cache_.varToLeafDict[var].first, '[', ']');

    // look for override requests
    if( GenericToolbox::isIn(_cache_.varToLeafDict[var].first, _parameters_.variableDict) ){
      // leafVar will actually be the override leaf name while event will keep the original name
      _cache_.varToLeafDict[var].first = _parameters_.variableDict[_cache_.varToLeafDict[var].first];
      _cache_.varToLeafDict[var].first = GenericToolbox::stripBracket(_cache_.varToLeafDict[var].first, '[', ']');
    }

    // possible dummy ?
    // [OUT] variables only
    // [OUT] not requested by its inputs
    for( auto& varTransform : _parameters_.eventVarTransformList ){
      const std::string& outVarName = varTransform.getOutputVariableName();
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

  auto treeChain = openChain();
  GenericToolbox::TreeBuffer treeBuffer;
  treeBuffer.setTree(treeChain.get());

  for( auto& var : _cache_.varsRequestedForIndexing ){
    treeBuffer.addExpression( getVariableExpression( var ) );
  }
  treeBuffer.initialize();

  Event eventPlaceholder;
  eventPlaceholder.getIndices().dataset = _owner_->getDataSetIndex();
  eventPlaceholder.getVariables().setVarNameList( _cache_.propagatorPtr->getSampleSet().getEventVariableNameList() );

  std::vector<const GenericToolbox::TreeBuffer::ExpressionBuffer*> expList{};
  for( auto& storageVar : *eventPlaceholder.getVariables().getNameListPtr() ){
    expList.emplace_back( treeBuffer.getExpressionBuffer(getVariableExpression( storageVar )) );
  }

  LoaderUtils::copyData(eventPlaceholder, expList);

  LogInfo << "Reserving event memory..." << std::endl;
  {
    GenericToolbox::TablePrinter t;
    t << "Sample" << GenericToolbox::TablePrinter::NextColumn
    << "# of events" << GenericToolbox::TablePrinter::NextColumn
    << "Memory" << GenericToolbox::TablePrinter::NextLine;

    size_t nTotal{0};

    _cache_.sampleIndexOffsetList.resize(_cache_.samplesToFillList.size());
    _cache_.sampleEventListPtrToFill.resize(_cache_.samplesToFillList.size());
    for( size_t iSample = 0 ; iSample < _cache_.sampleNbOfEvents.size() ; iSample++ ){
      _cache_.sampleEventListPtrToFill[iSample] = &_cache_.samplesToFillList[iSample]->getEventList();
      _cache_.sampleIndexOffsetList[iSample] = _cache_.sampleEventListPtrToFill[iSample]->size();
      _cache_.samplesToFillList[iSample]->reserveEventMemory(
          _owner_->getDataSetIndex(), _cache_.sampleNbOfEvents[iSample], eventPlaceholder
      );

      nTotal += _cache_.sampleNbOfEvents[iSample];

      t << _cache_.samplesToFillList[iSample]->getName() << GenericToolbox::TablePrinter::NextColumn
      << _cache_.sampleNbOfEvents[iSample] << GenericToolbox::TablePrinter::NextColumn
      << GenericToolbox::parseSizeUnits(static_cast<double>(eventPlaceholder.getSize() * _cache_.sampleNbOfEvents[iSample]))
      << GenericToolbox::TablePrinter::NextLine;
    }

    t << "Total" << GenericToolbox::TablePrinter::NextColumn
      << nTotal << GenericToolbox::TablePrinter::NextColumn
      << GenericToolbox::parseSizeUnits(static_cast<double>(eventPlaceholder.getSize()) * nTotal)
      << GenericToolbox::TablePrinter::NextLine;

    t.printTable();
  }


  LogInfo << "Filling var index cache for bin edges..." << std::endl;
  for( auto* samplePtr : _cache_.samplesToFillList ){
    for( auto& binContext : samplePtr->getHistogram().getBinContextList() ){
      for( auto& edges : binContext.bin.getEdgesList() ){
        edges.varIndexCache = GenericToolbox::findElementIndex( edges.varName, _cache_.varsRequestedForIndexing );
      }
    }
  }

  GenericToolbox::TablePrinter t;
  t << "Parameter" << GenericToolbox::TablePrinter::NextColumn;
  t << "Dial type" << GenericToolbox::TablePrinter::NextLine;

  size_t nTotalSlots{0};
  size_t nDialsMaxPerEvent{0};
  for( auto& dialCollection : _cache_.dialCollectionsRefList ){
    LogScopeIndent;
    nDialsMaxPerEvent += 1;

    if (dialCollection->isEventByEvent()) {
      // Reserve enough space for all the event-by-event dials
      // that might be added.  This size may be reduced later.
      t << dialCollection->getTitle() << GenericToolbox::TablePrinter::NextColumn;
      t << dialCollection->getDialType() << GenericToolbox::TablePrinter::NextLine;

      // Only increase the size.  It's probably zero before
      // starting, but add the original size... just in case.
      dialCollection->getDialInterfaceList().resize(
          dialCollection->getDialInterfaceList().size()
          + _cache_.totalNbEvents
      );
      nTotalSlots += _cache_.totalNbEvents;
    }
    else {
      // Filling var indexes for faster eval with PhysicsEvent:
      for( auto& bin : dialCollection->getDialBinSet().getBinList() ){
        for( auto& edges : bin.getEdgesList() ){
          edges.varIndexCache = GenericToolbox::findElementIndex( edges.varName, _cache_.varsRequestedForIndexing );
        }
      }
    }
  }

  if( nTotalSlots != 0 ) {
    LogInfo << "Created "  << nTotalSlots << " slots (" << _cache_.totalNbEvents << " per set) for event-by-event dials:" << std::endl;
    t.printTable();
  }


  _cache_.propagatorPtr->getEventDialCache().allocateCacheEntries(_cache_.totalNbEvents, nDialsMaxPerEvent);

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
  if(not _owner_->isDevSingleThreadEventLoaderAndIndexer() and getNbParallelCpu() > 1 ){
    threadSharedDataList.resize(getNbParallelCpu() );
    ROOT::EnableThreadSafety(); // EXTREMELY IMPORTANT
    GenericToolbox::ParallelWorker threadPool;
    threadPool.setNThreads( getNbParallelCpu() );
    threadPool.addJob(__METHOD_NAME__, [&](int iThread_){ this->runEventFillThreads(iThread_); });
    threadPool.runJob(__METHOD_NAME__);
    threadPool.removeJob(__METHOD_NAME__);
  }
  else{
    threadSharedDataList.resize(1);
    this->runEventFillThreads(-1); // for better debug breakdown
  }

  LogInfo << "Shrinking lists..." << std::endl;
  for( size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
    _cache_.samplesToFillList[iSample]->shrinkEventList( _cache_.sampleIndexOffsetList[iSample] );
  }

}
void DataDispenser::loadFromHistContent(){
  LogWarning << "Creating dummy PhysicsEvent entries for loading hist content" << std::endl;

  // non-trivial as we need to propagate systematics. Need to merge with the original data loader, but not straight forward?
  LogThrowIf( _parameters_.useReweightEngine, "Hist loader not implemented for MC containers" );

  // counting events
  _cache_.sampleNbOfEvents.resize(_cache_.samplesToFillList.size());
  _cache_.sampleIndexOffsetList.resize(_cache_.samplesToFillList.size());
  _cache_.sampleEventListPtrToFill.resize(_cache_.samplesToFillList.size());

  Event eventPlaceholder;
  eventPlaceholder.getIndices().dataset = (_owner_->getDataSetIndex());
  eventPlaceholder.getWeights().current = (0); // default.

  // claiming event memory
  for( size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){

    std::vector<std::string> varNameList;
    for( auto& binContext : _cache_.samplesToFillList[iSample]->getHistogram().getBinContextList() ){
      GenericToolbox::mergeInVector(varNameList, binContext.bin.buildVariableNameList(), false);
    }
    _cache_.propagatorPtr->getSampleSet().getEventVariableNameList() = varNameList;

    eventPlaceholder.getVariables().setVarNameList(
      _cache_.propagatorPtr->getSampleSet().getEventVariableNameList()
    );

    // one event per bin
    _cache_.sampleNbOfEvents[iSample] = _cache_.samplesToFillList[iSample]->getHistogram().getNbBins();

    _cache_.sampleEventListPtrToFill[iSample] = &_cache_.samplesToFillList[iSample]->getEventList();
    _cache_.sampleIndexOffsetList[iSample] = _cache_.sampleEventListPtrToFill[iSample]->size();
    _cache_.samplesToFillList[iSample]->reserveEventMemory( _owner_->getDataSetIndex(), _cache_.sampleNbOfEvents[iSample], eventPlaceholder );

    // indexing according to the binning
    for( size_t iEvent=_cache_.sampleIndexOffsetList[iSample] ; iEvent < _cache_.samplesToFillList[iSample]->getEventList().size() ; iEvent++ ){
      _cache_.samplesToFillList[iSample]->getEventList()[iEvent].getIndices().bin = int( iEvent );
    }
  }

  LogInfo << "Reading external hist files..." << std::endl;

  // read hist content from file
  LogInfo << "Opening: " << _parameters_.fromHistContent.rootFilePath << std::endl;
  auto* fHist = GenericToolbox::openExistingTFile(_parameters_.fromHistContent.rootFilePath);

  for( auto& sample : _cache_.samplesToFillList ){
    _cache_.propagatorPtr->getEventDialCache().allocateCacheEntries(sample->getHistogram().getNbBins(), 0);
  }

  for( auto& sample : _cache_.samplesToFillList ){
    LogScopeIndent;

    auto* sampleHistDef = _parameters_.fromHistContent.getSampleHistPtr(sample->getName());
    LogContinueIf(sampleHistDef== nullptr, "Could not find sample histogram: " << sample->getName());

    LogInfo << "Filling sample \"" << sample->getName() << "\" using hist with name: " << sampleHistDef->hist << std::endl;

    auto* histObj = fHist->Get( sampleHistDef->hist.c_str() );
    LogExitIf( histObj == nullptr, "Could not find TObject \"" << sampleHistDef->hist << "\" within " << fHist->GetPath() );

    if( histObj->InheritsFrom("THnD") ){
      auto* hist = (THnD*) histObj;
      int nBins = 1;
      for( int iDim = 0 ; iDim < hist->GetNdimensions() ; iDim++ ){
        nBins *= hist->GetAxis(iDim)->GetNbins();
      }

      LogAlertIf( nBins != sample->getHistogram().getNbBins() )
          << "Mismatching bin number for " << sample->getName() << ":" << std::endl
          << GET_VAR_NAME_VALUE(nBins) << std::endl
          << GET_VAR_NAME_VALUE(sample->getHistogram().getNbBins()) << std::endl;

      for( int iBin = 0 ; iBin < sample->getHistogram().getNbBins() ; iBin++ ){
        auto target = sample->getHistogram().getBinContextList()[iBin].bin.generateBinTarget( sampleHistDef->axisList );
        auto histBinIndex = hist->GetBin( target.data() ); // bad fetch..?

        sample->getEventList()[iBin].getIndices().sample = sample->getIndex();
        for( size_t iVar = 0 ; iVar < target.size() ; iVar++ ){
          sample->getEventList()[iBin].getVariables().fetchVariable(sampleHistDef->axisList[iVar]).set(target[iVar]);
        }
        sample->getEventList()[iBin].getWeights().base = (hist->GetBinContent(histBinIndex));
        sample->getEventList()[iBin].getWeights().resetCurrentWeight();
      }
    }
    else if(histObj->InheritsFrom("TH1D")){
      auto* hist = (TH1D*) histObj;
      int nBins = hist->GetNbinsX();
      LogAlertIf( nBins != sample->getHistogram().getNbBins() )
          << "Mismatching bin number for " << sample->getName() << ":" << std::endl
          << GET_VAR_NAME_VALUE(nBins) << std::endl
          << GET_VAR_NAME_VALUE(sample->getHistogram().getNbBins()) << std::endl;

      for( int iBin = 0 ; iBin < sample->getHistogram().getNbBins() ; iBin++ ){
        sample->getEventList()[iBin].getIndices().sample = sample->getIndex();
        sample->getEventList()[iBin].getWeights().base = (hist->GetBinContent(iBin+1));
        sample->getEventList()[iBin].getWeights().resetCurrentWeight();

        auto* eventDialCacheEntry = _cache_.propagatorPtr->getEventDialCache().fetchNextCacheEntry();
        auto sampleEventIndex = _cache_.sampleIndexOffsetList[sample->getIndex()]++;

        // Get the next free event in our buffer
        Event *eventPtr = &(*_cache_.sampleEventListPtrToFill[sample->getIndex()])[sampleEventIndex];

        // Now the event is ready. Let's index the dials:
        // there should always be a cache entry even if no dials are applied.
        // This cache is actually used to write MC events with dials in output tree
        eventDialCacheEntry->event.sampleIndex = std::size_t(sample->getIndex());
        eventDialCacheEntry->event.eventIndex = sampleEventIndex;
      }
    }

  }

  fHist->Close();
}

int DataDispenser::getNbParallelCpu() const{
  return GundamGlobals::getNbCpuThreads(_owner_->getNbMaxThreadsForLoad());
}
const std::string& DataDispenser::getVariableExpression(const std::string& variable_) const {
  try{ return _parameters_.variableDict.at(variable_); } catch( ... ) {}
  return variable_; // if not found
}
std::shared_ptr<TChain> DataDispenser::openChain(bool verbose_) const{
  LogInfoIf(verbose_) << "Opening ROOT files containing events..." << std::endl;

  std::shared_ptr<TChain> treeChain(std::make_shared<TChain>());
  for( const auto& file: _parameters_.filePathList){
    std::string name = GenericToolbox::expandEnvironmentVariables(file);
    GenericToolbox::replaceSubstringInsideInputString(name, "//", "/");

    if( verbose_ ){
      LogScopeIndent;
      LogWarning << name << std::endl;
    }

    std::string treePath{_parameters_.globalTreePath};
    auto chunks = GenericToolbox::splitString(name, ":", true);
    if( chunks.size() > 1 ){ treePath = chunks[1]; name = chunks[0];  }

    LogExitIf( treePath.empty(), "TTree path not set." );

    LogExitIf( not GenericToolbox::doesTFileIsValid(name, {treePath}), "Could not open TFile: " << name << " with TTree " << treePath);

    Long64_t nMaxEntries{TTree::kMaxEntries};
    if( _parameters_.fractionOfEntries != 1. ){
      std::unique_ptr<TFile> temp{TFile::Open(name.c_str())};
      LogExitIf(temp== nullptr, "Error while opening TFile: " << name);

      auto* tree = temp->Get<TTree>(treePath.c_str());
      LogExitIf(tree== nullptr, "Error while opening TTree: " << treePath << " in " << name);

      nMaxEntries = Long64_t( double(tree->GetEntries()) * _parameters_.fractionOfEntries );
      if( verbose_ ){
        LogScopeIndent;
        LogWarning << "Max entries: " << nMaxEntries << std::endl;
      }

    }
    treeChain->AddFile(name.c_str(), nMaxEntries, treePath.c_str());

  }

  return treeChain;
}

void DataDispenser::eventSelectionFunction(int iThread_){

  int nThreads{getNbParallelCpu()};
  if( iThread_ == -1 ){ iThread_ = 0; nThreads = 1; }

  // Opening ROOT files and make a TChain
  auto treeChain{this->openChain()};

  // Create the memory buffer for the TChain
  GenericToolbox::TreeBuffer tb;
  tb.setTree( treeChain.get() );

  // global cut
  int selectionCutLeafFormIndex{-1};
  if( not _parameters_.selectionCutFormulaStr.empty() ){
    LogInfoIf(iThread_ == 0) << "Global selection cut: \"" << _parameters_.selectionCutFormulaStr << "\"" << std::endl;
    selectionCutLeafFormIndex = tb.addExpression( _parameters_.selectionCutFormulaStr );
  }

  // sample cuts
  struct SampleCut{
    int sampleIndex{-1};
    int cutIndex{-1};
  };
  std::vector<SampleCut> sampleCutList;
  sampleCutList.reserve( _cache_.samplesToFillList.size() );

  GenericToolbox::TablePrinter tCuts;
  tCuts << "Sample" << GenericToolbox::TablePrinter::NextColumn;
  tCuts << "Selection cut" << GenericToolbox::TablePrinter::NextLine;
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

    sampleCutList.back().cutIndex = tb.addExpression( selectionCut );
    tCuts << samplePtr->getName() << GenericToolbox::TablePrinter::NextColumn;
    tCuts << selectionCut << GenericToolbox::TablePrinter::NextColumn;
  }

  if(iThread_ == 0){ tCuts.printTable(); }
  tb.initialize();

  GenericToolbox::VariableMonitor readSpeed("bytes");

  // Multi-thread index splitting
  Long64_t nEvents = treeChain->GetEntries();
  Long64_t iGlobal = 0;

  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices( iThread_, nThreads, nEvents );

  // Load the branches
  treeChain->LoadTree( bounds.beginIndex );

  // for each event, which sample is active?
  std::string progressTitle = "Performing event selection on " + this->getTitle() + "...";
  std::stringstream ssProgressTitle;
  TFile *lastFilePtr{nullptr};

  auto& threadSelectionResults = _cache_.threadSelectionResults[iThread_];

  for ( Long64_t iEntry = bounds.beginIndex ; iEntry < bounds.endIndex ; iEntry++ ) {
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
    tb.saveExpressions();

    if ( selectionCutLeafFormIndex != -1 ){
      if( tb.getExpressionBufferList()[selectionCutLeafFormIndex]->getBuffer().getValueAsDouble() == 0 ){
        for (size_t iSample = 0; iSample < _cache_.samplesToFillList.size(); iSample++) {
          threadSelectionResults.entrySampleIndexList[iEntry] = -1;
        }
        continue;
      }
    }

    bool sampleHasBeenFound{false};
    for( auto& sampleCut : sampleCutList ){


      if(  sampleCut.cutIndex == -1  // no cut?
           or tb.getExpressionBufferList()[sampleCut.cutIndex]->getBuffer().getValueAsDouble() != 0 // pass cut?
          ){
        if( sampleHasBeenFound ){
          LogError << "Entry #" << iEntry << "already has a sample." << std::endl;
          LogThrow("Multi-sample event isn't handled yet by GUNDAM.");
        }
        sampleHasBeenFound = true;
        threadSelectionResults.entrySampleIndexList[iEntry] = sampleCut.sampleIndex;
        threadSelectionResults.sampleNbOfEvents[sampleCut.sampleIndex]++;
      }
      else{
        // don't pass cut?
//          LogTrace << "Event #" << treeChain->GetFileNumber() << ":" << treeChain->GetReadEntry()
//                   << " rejected as sample " << sampleCut.sampleIndex << " because of "
//                   << lCollection.getLeafFormList()[sampleCut.cutIndex].getSummary() << std::endl;
      }
    }

  } // iEvent

  if( iThread_ == 0 ){ GenericToolbox::displayProgressBar(nEvents, nEvents, ssProgressTitle.str()); }

}

void DataDispenser::runEventFillThreads(int iThread_){

  int nThreads = getNbParallelCpu();
  if( iThread_ == -1 ){ iThread_ = 0; nThreads = 1; } // special mode

  // init shared data
  auto& threadSharedData = threadSharedDataList[iThread_];
  threadSharedData = ThreadSharedData(); // force reinitialization

  // open the TChain now
  threadSharedData.treeChain = this->openChain();
  threadSharedData.nbEntries = threadSharedData.treeChain->GetEntries();

  threadSharedData.treeBuffer.setTree(threadSharedData.treeChain.get());

  // nominal weight
  if( not _parameters_.nominalWeightFormulaStr.empty() ){
    ThreadSharedData::VariableBuffer::storeTempIndex(
          threadSharedData.buffer.nominalWeight,
          threadSharedData.treeBuffer.addExpression(_parameters_.nominalWeightFormulaStr)
        );
  }

  // dial array index
  if( not _parameters_.dialIndexFormula.empty() ){
    ThreadSharedData::VariableBuffer::storeTempIndex(
          threadSharedData.buffer.dialIndex,
          threadSharedData.treeBuffer.addExpression(_parameters_.dialIndexFormula)
        );
  }

  // variables definition
  for( auto& var : _cache_.varsRequestedForIndexing ){
    threadSharedData.buffer.varIndexingList.emplace_back();
    ThreadSharedData::VariableBuffer::storeTempIndex(
      threadSharedData.buffer.varIndexingList.back(),
      threadSharedData.treeBuffer.addExpression(getVariableExpression(var))
    );
  }
  for( auto& var : _cache_.propagatorPtr->getSampleSet().getEventVariableNameList() ){
    threadSharedData.buffer.varStorageList.emplace_back();
    ThreadSharedData::VariableBuffer::storeTempIndex(
      threadSharedData.buffer.varStorageList.back(),
      threadSharedData.treeBuffer.addExpression(getVariableExpression(var))
    );
  }

  threadSharedData.treeBuffer.initialize();

  // grab ptr address now
  if( not _parameters_.nominalWeightFormulaStr.empty() ){ ThreadSharedData::VariableBuffer::unfoldTempIndex(threadSharedData.buffer.nominalWeight, threadSharedData.treeBuffer.getExpressionBufferList()); }
  if( not _parameters_.dialIndexFormula.empty() ){ ThreadSharedData::VariableBuffer::unfoldTempIndex(threadSharedData.buffer.dialIndex, threadSharedData.treeBuffer.getExpressionBufferList()); }
  for( auto& varInd: threadSharedData.buffer.varIndexingList ){ ThreadSharedData::VariableBuffer::unfoldTempIndex(varInd, threadSharedData.treeBuffer.getExpressionBufferList()); }
  for( auto& varSto: threadSharedData.buffer.varStorageList ){ ThreadSharedData::VariableBuffer::unfoldTempIndex(varSto, threadSharedData.treeBuffer.getExpressionBufferList()); }

  // event variable as weight
  if( not _parameters_.eventVariableAsWeight.empty() ){
    for( size_t iVar = 0 ; iVar < _cache_.varsRequestedForIndexing.size() ; iVar++ ){
      if( _cache_.varsRequestedForIndexing[iVar] == _parameters_.eventVariableAsWeight ) {
        threadSharedData.buffer.eventVarAsWeight = threadSharedData.buffer.varIndexingList[iVar];
        break;
      }
    }

    LogThrowIf(threadSharedData.buffer.eventVarAsWeight==nullptr, "Could not find variable: " << _parameters_.eventVariableAsWeight);
  }

  // start event filler
  // create thread
  std::shared_ptr<std::future<void>> eventFillerThread{nullptr};
  eventFillerThread = std::make_shared<std::future<void>>(
      std::async( std::launch::async, [this, iThread_]{ this->loadEvent( iThread_ ); } )
  );


  // start TChain reader
  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices( iThread_, nThreads, threadSharedData.nbEntries );

  // IO speed monitor
  GenericToolbox::VariableMonitor readSpeed("bytes");
  std::string progressTitle = "Loading and indexing...";
  std::stringstream ssProgressBar;

  // make sure we're ready to start the loop
  threadSharedData.isEventFillerReady.waitUntilEqual( true );

  // Load the first TTree / need to wait for the event filler to finish hooking branches
  threadSharedData.treeChain->LoadTree(bounds.beginIndex);

  for( Long64_t iEntry = bounds.beginIndex ; iEntry < bounds.endIndex ; iEntry++ ){

    // before load, check if it has a sample
    bool hasSample = _cache_.entrySampleIndexList[iEntry] != -1;
    if( not hasSample ){ continue; }

    Int_t nBytes{ threadSharedData.treeChain->GetEntry(iEntry) };
    threadSharedData.treeBuffer.saveExpressions();

    threadSharedData.isEntryBufferReady.setValue(true); // loaded! -> let the other thread get everything it needs

    if( iThread_ == 0 ){
      readSpeed.addQuantity(nBytes * nThreads);

      if( GenericToolbox::showProgressBar(iEntry*nThreads, threadSharedData.nbEntries) ){

        ssProgressBar.str("");

        ssProgressBar << LogInfo.getPrefixString() << "Reading from disk: "
                      << GenericToolbox::padString(GenericToolbox::parseSizeUnits(readSpeed.getTotalAccumulated()), 8) << " ("
                      << GenericToolbox::padString(GenericToolbox::parseSizeUnits(readSpeed.evalTotalGrowthRate()), 8) << "/s)";

        int cpuPercent = int(GenericToolbox::getCpuUsageByProcess());
        ssProgressBar << " / CPU efficiency: " << GenericToolbox::padString(std::to_string(cpuPercent/nThreads), 3,' ')
                      << "% / RAM: " << GenericToolbox::parseSizeUnits( double(GenericToolbox::getProcessMemoryUsage()) ) << std::endl;

        ssProgressBar << LogInfo.getPrefixString() << "Data size per entry: " << GenericToolbox::parseSizeUnits(readSpeed.getLastValue());
        ssProgressBar << " / Using " << nThreads << " threads" << std::endl;

        ssProgressBar << LogInfo.getPrefixString() << progressTitle;
        GenericToolbox::displayProgressBar(
            iEntry*nThreads,
            threadSharedData.nbEntries,
            ssProgressBar.str()
        );
      }
    }

    // make sure the event filler thread has received the signal for the last entry
    threadSharedData.isEntryBufferReady.waitUntilEqual( false );

    // make sure currentEntry don't get updated while it hasn't been read by the other thread
    threadSharedData.requestReadNextEntry.waitUntilEqualThenToggle( true );

    // was the event loader stopped?
    if( not threadSharedData.isEventFillerReady.getValue() ){ break; }

  }

  threadSharedData.isDoneReading.setValue( true ); // trigger the loop break
  threadSharedData.isEntryBufferReady.setValue(true ); // release

  // wait for the event filler threads to stop
  eventFillerThread->get();

  // printout last p-bar
  if( iThread_ == 0 ){
    GenericToolbox::displayProgressBar(
        threadSharedData.nbEntries,
        threadSharedData.nbEntries,
        ssProgressBar.str()
    );
  }

}
void DataDispenser::loadEvent(int iThread_){

  // shared
  auto& threadSharedData = threadSharedDataList[iThread_];

  // local
  Event eventIndexingBuffer;
  eventIndexingBuffer.getIndices().dataset = _owner_->getDataSetIndex();

  eventIndexingBuffer.getVariables().setVarNameList(_cache_.varsRequestedForIndexing);

  auto eventVarTransformList = _parameters_.eventVarTransformList; // copy for cache
  std::vector<EventVarTransformLib*> varTransformForIndexingList;
  for( auto& eventVarTransform : eventVarTransformList ){
    if( GenericToolbox::doesElementIsInVector(eventVarTransform.getOutputVariableName(), _cache_.varsRequestedForIndexing) ){
      varTransformForIndexingList.emplace_back(&eventVarTransform);
    }
  }

  std::vector<DialBase*> eventByEventDialBuffer{};
  eventByEventDialBuffer.resize(_cache_.dialCollectionsRefList.size(), nullptr);

  if(iThread_ == 0){

    if( not varTransformForIndexingList.empty() ){
      LogInfo << "EventVarTransformLib used: "
              << GenericToolbox::toString(
                  varTransformForIndexingList,
                  [](const EventVarTransformLib* elm_){ return "\"" + elm_->getName() + "\"";}, false)
              << std::endl;
    }

    LogInfo << "Feeding event variables with:" << std::endl;
    GenericToolbox::TablePrinter table;

    table << "Variable" ;
    table << GenericToolbox::TablePrinter::NextColumn << "Expression";
    if(not varTransformForIndexingList.empty()){
      table << GenericToolbox::TablePrinter::NextColumn << "Transforms";
    }
    table << GenericToolbox::TablePrinter::NextLine;

    struct VarDisplay{
      std::string varName{};

      std::string leafName{};
      std::string leafTypeName{};

      std::string transformStr{};

      std::string lineColor{};

      int priorityIndex{-1};
    };
    std::vector<VarDisplay> varDisplayList{};

    bool hasEventDials{false};

    varDisplayList.reserve( _cache_.varsRequestedForIndexing.size() );
    for( size_t iVar = 0 ; iVar < _cache_.varsRequestedForIndexing.size() ; iVar++ ){
      varDisplayList.emplace_back();

      varDisplayList.back().varName = _cache_.varsRequestedForIndexing[iVar];

      varDisplayList.back().leafName = threadSharedData.buffer.varIndexingList[iVar]->getExpression();
      varDisplayList.back().leafTypeName = GenericToolbox::findOriginalVariableType(threadSharedData.buffer.varIndexingList[iVar]->getBuffer());

      std::vector<std::string> transformsList;
      for( auto* varTransformForIndexing : varTransformForIndexingList ){
        if( varTransformForIndexing->getOutputVariableName() == _cache_.varsRequestedForIndexing[iVar] ){
          transformsList.emplace_back(varTransformForIndexing->getName());
        }
      }
      varDisplayList.back().transformStr = GenericToolbox::toString(transformsList);
      varDisplayList.back().priorityIndex = 999;
      if( varDisplayList.back().leafTypeName != "\xFF" ){
        varDisplayList.back().priorityIndex = int( threadSharedData.buffer.varIndexingList[iVar]->getBuffer().getStoredSize() );
      }

      // line color?
      if( GenericToolbox::doesElementIsInVector(_cache_.varsRequestedForIndexing[iVar], _cache_.propagatorPtr->getSampleSet().getEventVariableNameList())){
        varDisplayList.back().lineColor = GenericToolbox::ColorCodes::blueBackground;
      }
      else if( varDisplayList.back().leafTypeName == "\xFF" ){
        varDisplayList.back().leafTypeName = "p";
        hasEventDials = true;
        varDisplayList.back().lineColor =  GenericToolbox::ColorCodes::magentaBackground;
      }
    }

    GenericToolbox::sortVector( varDisplayList, [](const VarDisplay& a_, const VarDisplay& b_){
      if( a_.priorityIndex < b_.priorityIndex ){ return true; }
      if( a_.priorityIndex > b_.priorityIndex ){ return false; }
      if( a_.leafTypeName.size() < b_.leafTypeName.size() ){ return true; }
      if( a_.leafTypeName.size() > b_.leafTypeName.size() ){ return false; }
      if( a_.varName < b_.varName ){ return true; }
      return false;
    } );

    for( auto& varDisplay : varDisplayList ){
      if( not varDisplay.lineColor.empty() ){ table.setColorBuffer( varDisplay.lineColor ); }
      table << varDisplay.varName << GenericToolbox::TablePrinter::NextColumn;
      table << varDisplay.leafName << "/" << varDisplay.leafTypeName << GenericToolbox::TablePrinter::NextColumn;

      if(not varTransformForIndexingList.empty()){
        table << varDisplay.transformStr << GenericToolbox::TablePrinter::NextColumn;
      }
    }

    table.printTable();

    // printing legend
    LogInfoIf(not _cache_.propagatorPtr->getSampleSet().getEventVariableNameList().empty()) << LOGGER_STR_COLOR_BLUE_BG    << "      " << LOGGER_STR_COLOR_RESET << " -> Variables stored in RAM" << std::endl;
    LogInfoIf(hasEventDials) << LOGGER_STR_COLOR_MAGENTA_BG << "      " << LOGGER_STR_COLOR_RESET << " -> Dials stored in RAM" << std::endl;

    if( _owner_->isDevSingleThreadEventLoaderAndIndexer() ){
      LogAlert << "Loading data in single thread (devSingleThreadEventLoaderAndIndexer option set to true)" << std::endl;
    }
  }


  // buffers
  int iSample{-1};
  size_t sampleEventIndex{};

  // make sure isEventFillerReady flag is true in this scope
  GenericToolbox::ScopedGuard g{
      [&]{ threadSharedData.isEventFillerReady.setValue( true ); },
      [&]{ threadSharedData.isEventFillerReady.setValue( false ); }
  };

  std::unordered_map<int, const TObject**> dialAddressMap;

  while( true ){

    {
      // make sure we request a new entry and wait once we go for another loop
      GenericToolbox::ScopedGuard readerGuard{
        [&]{ threadSharedData.isEntryBufferReady.waitUntilEqual( true ); threadSharedData.isEntryBufferReady.setValue( false ); },
        [&]{ threadSharedData.requestReadNextEntry.setValue( true ); }
      };

      // ***** from this point, the TChain reader is waiting *****

      // check if the reader has nothing else to do / end of the loop
      if( threadSharedData.isDoneReading.getValue() ){ break; }

      // leafFormIndexingList is modified by the TChain reader
      LoaderUtils::copyData(eventIndexingBuffer, threadSharedData.buffer.varIndexingList);

      // Propagate variable transformations for indexing
      LoaderUtils::applyVarTransforms(eventIndexingBuffer, varTransformForIndexingList);

      // nominalWeightTreeFormula is attached to the TChain
      if( threadSharedData.buffer.nominalWeight != nullptr ){
        eventIndexingBuffer.getWeights().base = threadSharedData.buffer.nominalWeight->getBuffer().getValueAsDouble();
      }

      // additional weight with an event variable
      if( threadSharedData.buffer.eventVarAsWeight != nullptr ){
        eventIndexingBuffer.getWeights().base *= threadSharedData.buffer.eventVarAsWeight->getBuffer().getValueAsDouble();
      }


      // skip this event if 0
      if( eventIndexingBuffer.getWeights().base == 0 ){ continue; }
      // no negative weights -> error
      if( eventIndexingBuffer.getWeights().base  < 0 ){
        LogError << "Negative nominal weight:" << std::endl;
        LogError << "Event buffer is: " << eventIndexingBuffer.getSummary() << std::endl;
        LogThrow("Negative nominal weight");
      }

      // grab data from TChain
      eventIndexingBuffer.getIndices().entry     = threadSharedData.treeChain->GetReadEntry();
      eventIndexingBuffer.getIndices().treeFile      = threadSharedData.treeChain->GetTreeNumber();
      eventIndexingBuffer.getIndices().treeEntry = threadSharedData.treeChain->GetTree()->GetReadEntry();

      // get sample index / all -1 samples have been ruled out by the chain reader
      iSample = _cache_.entrySampleIndexList[eventIndexingBuffer.getIndices().entry];
      Sample& eventSample{*_cache_.samplesToFillList[iSample]};

      eventIndexingBuffer.getIndices().sample = eventSample.getIndex();

      // look for the bin index
      LoaderUtils::fillBinIndex(eventIndexingBuffer, eventSample.getHistogram().getBinContextList());

      // No bin found -> next sample
      if( eventIndexingBuffer.getIndices().bin == -1 ){ continue; }

      // dialIndexTreeFormula is modified by the TChain reader
      int dialCloneArrayIndex{0};
      if( threadSharedData.buffer.dialIndex != nullptr ){
        dialCloneArrayIndex = static_cast<int>(threadSharedData.buffer.dialIndex->getBuffer().getValueAsLong());
      }

      // only load event-by-event dials, binned dials etc. will be processed later
      for( auto *dialCollectionRef: _cache_.dialCollectionsRefList ){

        eventByEventDialBuffer[dialCollectionRef->getIndex()] = nullptr;

        // if not event-by-event dial -> leave
        if( dialCollectionRef->getDialLeafName().empty() ){ continue; }

        // dial collections may come with a condition formula
        if( dialCollectionRef->getApplyConditionFormula() != nullptr ){
          if( LoaderUtils::evalFormula(eventIndexingBuffer, dialCollectionRef->getApplyConditionFormula().get()) == 0 ){
            // next dialSet
            continue;
          }
        }

        // grab as a general TObject, then let the factory figure out what to do with it
        try {
          dialAddressMap.at(dialCollectionRef->getIndex());
        }
        catch( ... ) {
          auto* dialExpression = threadSharedData.treeBuffer.getExpressionBuffer( dialCollectionRef->getDialLeafName() );
          LogThrowIf( dialExpression == nullptr );
          dialAddressMap[dialCollectionRef->getIndex()] = (const TObject**) dialExpression->getBuffer().getPlaceHolderPtr()->getVariableAddress();
        }

        const TObject* dialObjectPtr = *dialAddressMap[dialCollectionRef->getIndex()];

        // Extra-step for selecting the right dial with TClonesArray
        if( not strcmp(dialObjectPtr->ClassName(), "TClonesArray")){
          dialObjectPtr = ((const TClonesArray *) dialObjectPtr)->At(dialCloneArrayIndex);
        }

        auto dial = dialCollectionRef->makeDial(dialObjectPtr);
        eventByEventDialBuffer[dialCollectionRef->getIndex()] = dial.release();
      }

      if( _parameters_.debugNbMaxEventsToLoad != 0 ){
        // check if the limit has been reached
        std::unique_lock<std::mutex> lock(_mutex_);
        if( _cache_.propagatorPtr->getEventDialCache().getFillIndex() >= _parameters_.debugNbMaxEventsToLoad ){
          LogAlertIf(iThread_ == 0) << std::endl << std::endl; // flush pBar
          LogAlertIf(iThread_ == 0) << "debugNbMaxEventsToLoad: Event number cap reached (";
          LogAlertIf(iThread_ == 0) << _parameters_.debugNbMaxEventsToLoad << ")" << std::endl;
          threadSharedData.isDoneReading.setValue( true );
          return;
        }
      }
    }

    // ***** from this point, the TChain reader is released *****

    // Let's claim an index. Indices are shared among threads
    EventDialCache::IndexedCacheEntry *eventDialCacheEntry{nullptr};
    {
      std::unique_lock<std::mutex> lock(_mutex_);
      eventDialCacheEntry = _cache_.propagatorPtr->getEventDialCache().fetchNextCacheEntry();
      sampleEventIndex = _cache_.sampleIndexOffsetList[iSample]++;
    }

    // Get the next free event in our buffer
    Event *eventPtr = &(*_cache_.sampleEventListPtrToFill[iSample])[sampleEventIndex];

    // copy from the event indexing buffer
    LoaderUtils::copyData(eventIndexingBuffer, *eventPtr);

    // Now the event is ready. Let's index the dials:
    // there should always be a cache entry even if no dials are applied.
    // This cache is actually used to write MC events with dials in output tree
    eventDialCacheEntry->event.sampleIndex = std::size_t(eventIndexingBuffer.getIndices().sample);
    eventDialCacheEntry->event.eventIndex = sampleEventIndex;

    auto *dialEntryPtr = &eventDialCacheEntry->dials[0];
    for( auto *dialCollectionRef: _cache_.dialCollectionsRefList ){

      // leave if event-by-event -> already loaded
      if( not dialCollectionRef->getDialLeafName().empty() ){

        // dialBase is valid -> store it
        if( eventByEventDialBuffer[dialCollectionRef->getIndex()] != nullptr ){
          size_t freeSlotDial = dialCollectionRef->getNextDialFreeSlot();
          dialCollectionRef->getDialInterfaceList()[freeSlotDial].getDial().dialPtr
            = std::unique_ptr<DialBase>(eventByEventDialBuffer[dialCollectionRef->getIndex()]);

          dialEntryPtr->collectionIndex = dialCollectionRef->getIndex();
          dialEntryPtr->interfaceIndex = freeSlotDial;
          dialEntryPtr++;
        }

        continue; // skip the rest
      }

      // dial collections may come with a condition formula
      if( dialCollectionRef->getApplyConditionFormula() != nullptr ){
        if( LoaderUtils::evalFormula(eventIndexingBuffer, dialCollectionRef->getApplyConditionFormula().get()) == 0 ){
          // next dialSet
          continue;
        }
      }

      int iCollection = dialCollectionRef->getIndex();

      if( dialCollectionRef->getDialType() == DialCollection::DialType::Tabulated ){
        // Event-by-event dial for a precalculated table.  The table
        // can hold things like oscillation weights and is filled before
        // the event weighting is done.

        std::unique_ptr<DialBase> dialBase(
            dialCollectionRef->getCollectionData<TabulatedDialFactory>(0)
                ->makeDial(eventIndexingBuffer));

        // dialBase is valid -> store it
        if( dialBase != nullptr ){
          size_t freeSlotDial = dialCollectionRef->getNextDialFreeSlot();
          dialCollectionRef->getDialInterfaceList()[freeSlotDial].getDial().dialPtr =
            std::unique_ptr<DialBase>(dialBase.release());

          dialEntryPtr->collectionIndex = iCollection;
          dialEntryPtr->interfaceIndex = freeSlotDial;
          dialEntryPtr++;
        }
      }
      else{

        if( dialCollectionRef->getDialInterfaceList().size() == 1
            and dialCollectionRef->getDialBinSet().getBinList().empty()){
          // There isn't any binning, and there is only one dial.
          // In this case we don't need to check if the dial is in
          // a bin.
          dialEntryPtr->collectionIndex = iCollection;
          dialEntryPtr->interfaceIndex = 0;
          dialEntryPtr++;
        }
        else{
          // There are multiple dials, or there is a list of bins
          // to apply the dial to.  Check if the event falls into
          // a bin, and apply the correct binning.  Some events
          // may not be in any bin.
          auto dialBinIdx = eventIndexingBuffer.getVariables().findBinIndex(
              dialCollectionRef->getDialBinSet().getBinList());
          if( dialBinIdx != -1 ){
            dialEntryPtr->collectionIndex = iCollection;
            dialEntryPtr->interfaceIndex = dialBinIdx;
            dialEntryPtr++;
          }
        }

      }

    } // dial collection loop

  } // while ok

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
// End:
