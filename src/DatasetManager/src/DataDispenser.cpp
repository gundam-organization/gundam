//
// Created by Adrien BLANCHET on 14/05/2022.
//

#include "DataDispenser.h"
#include "DatasetDefinition.h"
#include "LoaderUtils.h"

#include "Propagator.h"
#include "GundamGlobals.h"

#include "ConfigUtils.h"
#include "FormulaUtils.h"

#include "DialCollection.h"
#include "TabulatedDialFactory.h"
#include "KrigedDialFactory.h"

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
#include <map>
#include <set>
#include <string>
#include <vector>
#include <sstream>

namespace {

  bool hasFormulaReferences(const std::string& formula_){
    return not FormulaUtils::extractFormulaReferenceNames(formula_).empty();
  }

  bool parseBracketToken(
      const std::string& formula_,
      size_t openingBracketPos_,
      size_t& closingBracketPos_,
      std::string& token_
  ){
    if( openingBracketPos_ >= formula_.size() or formula_[openingBracketPos_] != '[' ){ return false; }
    closingBracketPos_ = formula_.find(']', openingBracketPos_ + 1);
    LogExitIf(
        closingBracketPos_ == std::string::npos,
        "Invalid formula reference in \"" << formula_ << "\": missing closing ']'."
    );
    token_ = formula_.substr(openingBracketPos_ + 1, closingBracketPos_ - openingBracketPos_ - 1);
    return true;
  }

  std::string registerTreeExpressionAlias(
      DataDispenserCache& cache_,
      const std::string& treeExpression_
  ){
    auto existingAlias = cache_.eventFormulaTreeExpressionAliases.find(treeExpression_);
    if( existingAlias != cache_.eventFormulaTreeExpressionAliases.end() ){ return existingAlias->second; }

    std::string alias = "__gundam_formula_tree_expr_" + std::to_string(cache_.eventFormulaTreeExpressionAliases.size());
    cache_.eventFormulaTreeExpressionAliases[treeExpression_] = alias;

    cache_.addVarRequestedForIndexing(alias);
    cache_.variableDictEvalList.emplace_back();
    cache_.variableDictEvalList.back().name = alias;
    cache_.variableDictEvalList.back().expr = treeExpression_;
    cache_.variableDictEvalList.back().backend = DataDispenserCache::VariableDictEntry::TreeBufferExpression;

    return alias;
  }

  bool isGeneratedTreeExpressionAlias(const DataDispenserCache& cache_, const std::string& name_){
    return std::any_of(
        cache_.eventFormulaTreeExpressionAliases.begin(),
        cache_.eventFormulaTreeExpressionAliases.end(),
        [&](const auto& entry_){ return entry_.second == name_; }
    );
  }

  std::vector<std::string> filterDisplayedVariableNames(
      const DataDispenserCache& cache_,
      const std::vector<std::string>& variableNameList_
  ){
    std::vector<std::string> out;
    out.reserve(variableNameList_.size());
    for( const auto& varName : variableNameList_ ){
      if( isGeneratedTreeExpressionAlias(cache_, varName) ){ continue; }
      out.emplace_back(varName);
    }
    return out;
  }

  std::string resolveTreeArrayIndexToken(
      const std::string& token_,
      const std::map<std::string, std::string>& variableDict_
  ){
    auto dictEntry = variableDict_.find(token_);
    if( dictEntry == variableDict_.end() ){ return token_; }
    return FormulaUtils::resolveFormulaReferences(
        dictEntry->second,
        variableDict_,
        FormulaUtils::FormulaResolutionMode::StrictVariableDictOnly
    );
  }

  std::string buildTreeArrayExpression(
      const std::string& branchName_,
      const std::string& formula_,
      size_t firstIndexOpeningBracketPos_,
      size_t& expressionEnd_,
      const std::map<std::string, std::string>& variableDict_
  ){
    std::string output = branchName_;
    expressionEnd_ = firstIndexOpeningBracketPos_;

    while( expressionEnd_ < formula_.size() and formula_[expressionEnd_] == '[' ){
      size_t closingBracketPos{0};
      std::string token;
      parseBracketToken(formula_, expressionEnd_, closingBracketPos, token);
      output += "[" + resolveTreeArrayIndexToken(token, variableDict_) + "]";
      expressionEnd_ = closingBracketPos + 1;
    }

    return output;
  }

  std::string replaceAliasedTreeExpressions(
      const std::string& formula_,
      const std::map<std::string, std::string>& treeExpressionAliases_
  ){
    std::string output = formula_;
    for( const auto& entry : treeExpressionAliases_ ){
      auto firstArrayBracketPos = entry.first.find('[');
      auto bracketExpression = (
          firstArrayBracketPos == std::string::npos ?
          "[" + entry.first + "]" :
          "[" + entry.first.substr(0, firstArrayBracketPos) + "]" + entry.first.substr(firstArrayBracketPos)
      );

      GenericToolbox::replaceSubstringInsideInputString(output, bracketExpression, "[" + entry.second + "]");
    }
    return output;
  }

  std::string registerAndReplaceTreeArrayReferences(
      DataDispenserCache& cache_,
      const std::string& formula_,
      const std::map<std::string, std::string>& variableDict_
  ){
    std::string output;
    output.reserve(formula_.size());

    size_t cursor{0};
    while( cursor < formula_.size() ){
      if( std::isalpha(static_cast<unsigned char>(formula_[cursor])) or formula_[cursor] == '_' ){
        auto identifierBegin = cursor;
        cursor++;
        while(
            cursor < formula_.size()
            and (std::isalnum(static_cast<unsigned char>(formula_[cursor])) or formula_[cursor] == '_')
        ){
          cursor++;
        }

        if( cursor < formula_.size() and formula_[cursor] == '[' ){
          size_t expressionEnd = cursor;
          auto treeExpression = buildTreeArrayExpression(
              formula_.substr(identifierBegin, cursor - identifierBegin),
              formula_,
              cursor,
              expressionEnd,
              variableDict_
          );
          auto alias = registerTreeExpressionAlias(cache_, treeExpression);
          output += "[" + alias + "]";
          cursor = expressionEnd;
          continue;
        }

        output.append(formula_, identifierBegin, cursor - identifierBegin);
        continue;
      }

      if( formula_[cursor] != '[' ){
        output += formula_[cursor];
        cursor++;
        continue;
      }

      auto openingBracketPos = cursor;
      if( openingBracketPos == std::string::npos ){
        output.append(formula_, cursor, std::string::npos);
        break;
      }

      size_t firstClosingBracketPos{0};
      std::string firstToken;
      parseBracketToken(formula_, openingBracketPos, firstClosingBracketPos, firstToken);

      if(
          not FormulaUtils::extractFormulaReferenceNames("[" + firstToken + "]").empty()
          and variableDict_.find(firstToken) != variableDict_.end()
      ){
        output.append(formula_, cursor, firstClosingBracketPos - cursor + 1);
        cursor = firstClosingBracketPos + 1;
        continue;
      }

      if(
          FormulaUtils::extractFormulaReferenceNames("[" + firstToken + "]").empty()
          or firstClosingBracketPos + 1 >= formula_.size()
          or formula_[firstClosingBracketPos + 1] != '['
      ){
        output.append(formula_, cursor, firstClosingBracketPos - cursor + 1);
        cursor = firstClosingBracketPos + 1;
        continue;
      }

      size_t expressionEnd = firstClosingBracketPos + 1;
      auto treeExpression = buildTreeArrayExpression(
          firstToken,
          formula_,
          firstClosingBracketPos + 1,
          expressionEnd,
          variableDict_
      );
      auto alias = registerTreeExpressionAlias(cache_, treeExpression);

      output.append(formula_, cursor, openingBracketPos - cursor);
      output += "[" + alias + "]";
      cursor = expressionEnd;
    }

    return output;
  }

  bool isActiveVariableDictName(const DataDispenserCache& cache_, const std::string& name_){
    for( const auto& entry : cache_.variableDictEvalList ){
      if( entry.name == name_ ){ return true; }
    }
    return false;
  }

  const DataDispenserCache::VariableDictEntry* getActiveVariableDictEntry(const DataDispenserCache& cache_, const std::string& name_){
    for( const auto& entry : cache_.variableDictEvalList ){
      if( entry.name == name_ ){ return &entry; }
    }
    return nullptr;
  }

  bool doesVariableDictEntryNeedTreeValue(const DataDispenserCache::VariableDictEntry* entry_){
    return (
        entry_ != nullptr
        and entry_->backend == DataDispenserCache::VariableDictEntry::LibraryTransform
        and GenericToolbox::doesElementIsInVector(entry_->name, entry_->transformPtr->fetchRequestedVars())
    );
  }

  bool isEventBufferOnlyVariable(const DataDispenserCache& cache_, const std::string& name_){
    auto* entry = getActiveVariableDictEntry(cache_, name_);
    return (
        entry != nullptr
        and (
            entry->backend == DataDispenserCache::VariableDictEntry::EventBufferFormula
            or (
                entry->backend == DataDispenserCache::VariableDictEntry::LibraryTransform
                and not doesVariableDictEntryNeedTreeValue(entry)
            )
        )
    );
  }

  const std::string& getVariableExpression(
      const DataDispenserCache& cache_,
      const std::map<std::string, std::string>& variableDict_,
      const std::string& variable_
  ){
    auto* activeEntry = getActiveVariableDictEntry(cache_, variable_);
    if( activeEntry != nullptr ){
      if( activeEntry->backend == DataDispenserCache::VariableDictEntry::TreeBufferExpression ){
        return activeEntry->expr;
      }
      return variable_;
    }
    try{ return variableDict_.at(variable_); } catch( ... ) {}
    return variable_;
  }

  std::string getVariableDisplayExpression(
      const DataDispenserCache::VariableDictEntry* entry_
  ){
    if( entry_ == nullptr ){ return ""; }
    if( entry_->backend != DataDispenserCache::VariableDictEntry::LibraryTransform ){ return entry_->expr; }

    const auto& transform = *entry_->transformPtr;
    auto libraryFileName = GenericToolbox::getFileName(transform.getLibraryFile(), true);
    if( transform.getName().empty() or transform.getName() == transform.getOutputVariableName() ){
      return "evalFromLib(\"" + libraryFileName + "\")";
    }

    return "evalFromLib(\"" + transform.getName() + "\", \"" + libraryFileName + "\")";
  }

  void addVariablesRequestedByFormula(
      DataDispenserCache& cache_,
      const std::string& formula_,
      const std::map<std::string, std::string>& variableDict_,
      const std::map<std::string, EventVarTransformLib>& variableDictTransform_,
      bool strictFormulaReferences_,
      std::set<std::string>& variableDictStack_
  );

  void addVariableDictRequestedByName(
      DataDispenserCache& cache_,
      const std::string& name_,
      const std::map<std::string, std::string>& variableDict_,
      const std::map<std::string, EventVarTransformLib>& variableDictTransform_,
      std::set<std::string>& variableDictStack_
  ){
    auto dictEntry = variableDict_.find(name_);
    auto transformEntry = variableDictTransform_.find(name_);
    if( dictEntry == variableDict_.end() and transformEntry == variableDictTransform_.end() ){
      cache_.addVarRequestedForIndexing(name_);
      return;
    }

    if( isActiveVariableDictName(cache_, name_) ){ return; }
    LogExitIf(
        variableDictStack_.count(name_) != 0,
        "Cyclic variableDict reference detected while preparing variable \"" << name_ << "\"."
    );

    variableDictStack_.insert(name_);
    if( dictEntry != variableDict_.end() ){
      addVariablesRequestedByFormula(cache_, dictEntry->second, variableDict_, variableDictTransform_, true, variableDictStack_);
    }
    else{
      for( const auto& inputVarName : transformEntry->second.fetchRequestedVars() ){
        if( inputVarName == name_ ){
          cache_.addVarRequestedForIndexing(inputVarName);
          continue;
        }
        addVariableDictRequestedByName(cache_, inputVarName, variableDict_, variableDictTransform_, variableDictStack_);
      }
    }
    variableDictStack_.erase(name_);

    cache_.addVarRequestedForIndexing(name_);
    cache_.variableDictEvalList.emplace_back();
    cache_.variableDictEvalList.back().name = name_;
    if( dictEntry != variableDict_.end() ){
      cache_.variableDictEvalList.back().expr = dictEntry->second;
      cache_.variableDictEvalList.back().backend = (
          hasFormulaReferences(dictEntry->second) ?
          DataDispenserCache::VariableDictEntry::EventBufferFormula :
          DataDispenserCache::VariableDictEntry::TreeBufferExpression
      );
    }
    else{
      cache_.variableDictEvalList.back().backend = DataDispenserCache::VariableDictEntry::LibraryTransform;
      cache_.variableDictEvalList.back().transformPtr = &transformEntry->second;
    }
  }

  void addVariablesRequestedByFormula(
      DataDispenserCache& cache_,
      const std::string& formula_,
      const std::map<std::string, std::string>& variableDict_,
      const std::map<std::string, EventVarTransformLib>& variableDictTransform_,
      bool strictFormulaReferences_,
      std::set<std::string>& variableDictStack_
  ){
    if( formula_.empty() ){ return; }

    auto formula = registerAndReplaceTreeArrayReferences(cache_, formula_, variableDict_);

    for( const auto& referenceName : FormulaUtils::extractFormulaReferenceNames(formula) ){
      auto dictEntry = variableDict_.find(referenceName);
      if( dictEntry != variableDict_.end() ){
        addVariableDictRequestedByName(cache_, referenceName, variableDict_, variableDictTransform_, variableDictStack_);
      }
      else if( variableDictTransform_.find(referenceName) != variableDictTransform_.end() ){
        addVariableDictRequestedByName(cache_, referenceName, variableDict_, variableDictTransform_, variableDictStack_);
      }
      else if( isActiveVariableDictName(cache_, referenceName) ){
        cache_.addVarRequestedForIndexing(referenceName);
      }
      else{
        LogExitIf(
            strictFormulaReferences_,
            "Unknown variableDict reference [" << referenceName << "] in formula \"" << formula_ << "\"."
        );
        cache_.addVarRequestedForIndexing(referenceName);
      }
    }

    for( const auto& bareVarName : FormulaUtils::extractBareVariableNames(formula) ){
      cache_.addVarRequestedForIndexing(bareVarName);
    }
  }

  void addVariablesRequestedByFormula(
      DataDispenserCache& cache_,
      const std::string& formula_,
      const std::map<std::string, std::string>& variableDict_,
      const std::map<std::string, EventVarTransformLib>& variableDictTransform_,
      bool strictFormulaReferences_
  ){
    std::set<std::string> variableDictStack;
    addVariablesRequestedByFormula(cache_, formula_, variableDict_, variableDictTransform_, strictFormulaReferences_, variableDictStack);
  }

  void compileEventFormula(
      ThreadSharedData::VariableBuffer::EventFormula& eventFormula_,
      const std::string& formulaStr_,
      const std::vector<std::string>& eventVariableNameList_
  ){
    eventFormula_.expr = FormulaUtils::convertBareVariablesToFormulaParameters(formulaStr_);
    eventFormula_.formula = TFormula(eventFormula_.expr.c_str(), eventFormula_.expr.c_str());
    LogExitIf(not eventFormula_.formula.IsValid(), "\"" << formulaStr_ << "\" -> \"" << eventFormula_.expr << "\": could not be parsed as event-buffer formula.");

    eventFormula_.varIndexList.clear();
    eventFormula_.varIndexList.reserve(eventFormula_.formula.GetNpar());
    for( int iPar = 0 ; iPar < eventFormula_.formula.GetNpar() ; iPar++ ){
      auto varIndex = GenericToolbox::findElementIndex(eventFormula_.formula.GetParName(iPar), eventVariableNameList_);
      LogExitIf(
          varIndex == -1,
          "Formula \"" << formulaStr_ << "\" requires event variable \"" << eventFormula_.formula.GetParName(iPar)
                       << "\", but it is not available in the event buffer."
      );
      eventFormula_.varIndexList.emplace_back(varIndex);
    }
  }

  void compileRuntimeFormula(
      ThreadSharedData::VariableBuffer::RuntimeFormula& formula_,
      GenericToolbox::TreeBuffer& treeBuffer_,
      const std::string& formulaStr_,
      const std::vector<std::string>& eventVariableNameList_,
      const std::map<std::string, std::string>& treeExpressionAliases_
  ){
    formula_ = ThreadSharedData::VariableBuffer::RuntimeFormula();
    if( formulaStr_.empty() ){ return; }

    auto formulaStr = replaceAliasedTreeExpressions(formulaStr_, treeExpressionAliases_);

    if( hasFormulaReferences(formulaStr) ){
      formula_.backend = ThreadSharedData::VariableBuffer::RuntimeFormula::EventBufferFormula;
      compileEventFormula(formula_.eventFormula, formulaStr, eventVariableNameList_);
      return;
    }

    if( FormulaUtils::extractBareVariableNames(formulaStr).empty() ){
      formula_.backend = ThreadSharedData::VariableBuffer::RuntimeFormula::EventBufferFormula;
      compileEventFormula(formula_.eventFormula, formulaStr, eventVariableNameList_);
      return;
    }

    formula_.backend = ThreadSharedData::VariableBuffer::RuntimeFormula::TreeBufferExpression;
    ThreadSharedData::VariableBuffer::storeTempIndex(
        formula_.treeExpression,
        treeBuffer_.addExpression(formulaStr)
    );
  }

  void unfoldRuntimeFormula(
      ThreadSharedData::VariableBuffer::RuntimeFormula& formula_,
      const std::vector<std::shared_ptr<GenericToolbox::TreeBuffer::ExpressionBuffer>>& expressionBufferList_
  ){
    if( formula_.backend != ThreadSharedData::VariableBuffer::RuntimeFormula::TreeBufferExpression ){ return; }
    ThreadSharedData::VariableBuffer::unfoldTempIndex(formula_.treeExpression, expressionBufferList_);
  }

  void evalVariableDict(Event& event_, std::vector<ThreadSharedData::VariableBuffer::VariableDictBuffer>& variableDictEvalList_){
    for( auto& entry : variableDictEvalList_ ){
      if( entry.isLibraryTransform ){
        entry.transform.evalAndStore(event_);
      }
      else{
        event_.getVariables().getVarList()[entry.outputVarIndex].set(entry.formula.eval(event_));
      }
    }
  }

}


void DataDispenser::prepareConfig(ConfigReader &config_){
  config_.clearFields();
  config_.defineFields({
    {"name"},
    {"tree"},
    {"filePathList"},
    {"debugNbMaxEventsToLoad"},
    {"fromHistContent"},
    {"dummyVariablesList"},
    {"useReweightEngine", {"useMcContainer"}},
    {"variablesTransform"},
    {"eventVariableAsWeight"},
    {"additionalLeavesStorage"},
    {"dialIndexFormula"},
    {"overridePropagatorConfig"},
    {"selectionCutFormula"},
    {"allowMultipleSamplesPerEntry"},
    {"nominalWeightFormula", {"nominalTreeWeightFormula"}},
    {"variableDict", {"overrideLeafDict"}},
    {"fromModel", {"fromMc"}},
    {"evalModelAt"},
  });
  config_.checkConfiguration();
}
void DataDispenser::configureImpl(){
  DataDispenser::prepareConfig(_config_);

  _config_.fillValue(_parameters_.name, "name");

  // histograms don't need other parameters
  if( _config_.hasField("fromHistContent" ) ) {
    LogDebugIf(GundamGlobals::isDebug()) << "Dataset \"" << _parameters_.name << "\" will be defined with histogram data." << std::endl;
    auto fromHistConfig(_config_.fetchValue<ConfigReader>("fromHistContent"));

    fromHistConfig.defineFields({
      {"fromRootFile"},
      {"sampleList"},
    });

    _parameters_.fromHistContent.isEnabled = true;
    _parameters_.fromHistContent.rootFilePath = fromHistConfig.fetchValue<std::string>("fromRootFile");

    auto sampleListConfig(fromHistConfig.loop("sampleList"));
    _parameters_.fromHistContent.sampleHistList.reserve(sampleListConfig.size());
    for( auto& sampleConfig : sampleListConfig ){
      sampleConfig.defineFields({
        {FieldFlag::MANDATORY, "name"},
        {FieldFlag::MANDATORY, "hist"},
        {"axisList", {"axis"}},
      });

      auto& sampleHist = _parameters_.fromHistContent.addSampleHist(sampleConfig.fetchValue<std::string>("name"));
      sampleConfig.fillValue(sampleHist.hist, "hist");
      sampleConfig.fillValue(sampleHist.axisList, "axisList");
    }

    return;
  }

  _parameters_.variableDict.clear();
  _parameters_.variableDictTransform.clear();
  for( auto& entry : _config_.loop("variableDict") ){
    entry.defineFields({
      {FieldFlag::MANDATORY, "name", {"eventVar"}},
      {"expr", {"expression", "leafVar"}},
      {"evalFromLib"},
    });
    entry.checkConfiguration();
    auto varName = entry.fetchValue<std::string>("name");
    bool hasExpr = entry.hasField("expr");
    bool hasEvalFromLib = entry.hasField("evalFromLib");
    LogExitIf(hasExpr == hasEvalFromLib, "variableDict entry \"" << varName << "\" must define exactly one of: expr, evalFromLib.");
    LogExitIf(
        _parameters_.variableDict.count(varName) != 0 or _parameters_.variableDictTransform.count(varName) != 0,
        "Duplicate variableDict entry: " << varName
    );

    if( hasExpr ){
      _parameters_.variableDict[ varName ] = entry.fetchValue<std::string>("expr");
    }
    else{
      auto evalFromLibConfig = entry.fetchValue<ConfigReader>("evalFromLib");
      auto& transform = _parameters_.variableDictTransform[varName];
      transform.configureFromVariableDict(varName, evalFromLibConfig);
    }
  }

  int index{0};
  for( auto& varTransform : _config_.loop("variablesTransform") ){
    EventVarTransformLib transform;
    transform.configure(varTransform);
    transform.setIndex(index++);
    if( not transform.isEnabled() ){ continue; }

    auto varName = transform.getOutputVariableName();
    LogExitIf(varName.empty(), "variablesTransform entry has an empty outputVariableName.");
    LogExitIf(
        _parameters_.variableDict.count(varName) != 0 or _parameters_.variableDictTransform.count(varName) != 0,
        "Deprecated variablesTransform output \"" << varName << "\" collides with an existing variableDict entry."
    );
    LogWarning << "Deprecated config field \"variablesTransform\" defines \"" << varName
               << "\". Please use variableDict/evalFromLib instead." << std::endl;
    _parameters_.variableDictTransform[varName] = transform;
  }

  _config_.fillValue(_parameters_.eventVariableAsWeight, "eventVariableAsWeight");

  // options
  _config_.fillValue(_parameters_.globalTreePath, "tree");
  _config_.fillValue(_parameters_.filePathList, "filePathList");
  _config_.fillValue(_parameters_.additionalVarsStorage, "additionalLeavesStorage");
  _config_.fillValue(_parameters_.dummyVariablesList, "dummyVariablesList");
  _config_.fillValue(_parameters_.useReweightEngine, "useReweightEngine");
  _config_.fillValue(_parameters_.debugNbMaxEventsToLoad, "debugNbMaxEventsToLoad");
  _config_.fillValue(_parameters_.dialIndexFormula, "dialIndexFormula");
  _config_.fillValue(_parameters_.overridePropagatorConfig, "overridePropagatorConfig");
  _config_.fillValue(_parameters_.evalModelAt, "evalModelAt");
  _config_.fillValue(_parameters_.allowMultipleSamplesPerEntry, "allowMultipleSamplesPerEntry");

  _config_.fillFormula(_parameters_.selectionCutFormulaStr, "selectionCutFormula", "&&");
  _config_.fillFormula(_parameters_.nominalWeightFormulaStr, "nominalWeightFormula", "*");

}
void DataDispenser::initializeImpl(){

  _config_.printUnusedKeys();

  for( auto& entry: _parameters_.variableDictTransform ){
    entry.second.initialize();
  }

}

void DataDispenser::load(Propagator& propagator_){
  LogInfo << "Loading dataset: " << getTitle() << std::endl;
  LogExitIf(not this->isInitialized(), "Can't load while not initialized.");
  LogExitIf(not propagator_.isInitialized(), "Can't load while propagator_ is not initialized.");

  _cache_.clear();
  _cache_.propagatorPtr = &propagator_;

  if( not _parameters_.overridePropagatorConfig.empty() ){
    LogWarning << "Reload the propagator config with override options" << std::endl;
    ConfigUtils::ConfigBuilder configHandler( _cache_.propagatorPtr->getConfig().getConfig() );
    configHandler.override( _parameters_.overridePropagatorConfig );

    ConfigReader ch( configHandler.getConfig() );
    ch.setParentPath(_cache_.propagatorPtr->getConfig().getParentPath());
    _cache_.propagatorPtr->setConfig(ch);

    LogWarning << "Re-configuring the propagator with overriden parameters..." << std::endl;
    _cache_.propagatorPtr->configure();
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
  this->fetchRequestedLeaves();
  this->doEventSelection();
  this->preAllocateMemory();
  this->readAndFill();

  LogInfo << "Loaded " << getTitle() << std::endl;
  if (this->_unbinnedEvents_.getValue() > 0) {
#ifdef GUNDAM_EXIT_ON_UNBINNED_EVENTS
    // Exit since the fitted results will be "wrong".
    LogError << "Invalid event selection or likelihood definition"
             << std::endl
             << "Events selected, but not included in a likelihood bin: "
             << this->_unbinnedEvents_.getValue()
             << std::endl;
    LogExit("Incorrect event selection or likelihood binning");
#else
    LogWarning << "Mismatch between event selection and likelihood definitions"
             << std::endl
             << "Events selected, but not included in likelihood: "
             << this->_unbinnedEvents_.getValue()
             << std::endl;
#endif
 }

}
std::string DataDispenser::getTitle(){
  std::stringstream ss;
  if( _owner_ != nullptr ) ss << _owner_->getName();
  ss << "/" << _parameters_.name;
  return ss.str();
}

void DataDispenser::buildSampleToFillList(){
  LogInfo << "Fetching samples to fill..." << std::endl;

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

  if( not _parameters_.variableDict.empty() ){
    for( auto& entryDict : _parameters_.variableDict ){ replaceToyIndexFct(entryDict.second); }
    LogInfo << "Variable dictionary: " << GenericToolbox::toString(_parameters_.variableDict) << std::endl;
  }

  replaceToyIndexFct(_parameters_.dialIndexFormula);
  replaceToyIndexFct(_parameters_.nominalWeightFormulaStr);
  replaceToyIndexFct(_parameters_.selectionCutFormulaStr);

  // add surrounding parenthesis to force the LeafForm to treat it as a TFormula
  if(not _parameters_.dialIndexFormula.empty()){ _parameters_.dialIndexFormula = "(" + _parameters_.dialIndexFormula + ")"; }
  if(not _parameters_.nominalWeightFormulaStr.empty()){ _parameters_.nominalWeightFormulaStr = "(" + _parameters_.nominalWeightFormulaStr + ")"; }
  if(not _parameters_.selectionCutFormulaStr.empty()){ _parameters_.selectionCutFormulaStr = "(" + _parameters_.selectionCutFormulaStr + ")"; }
}
void DataDispenser::doEventSelection(){
  LogInfo << "Performing event selection..." << std::endl;

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
    threadResults.entrySampleIndexList.reserve(nEntries);
    // for (auto& sampleIdxList : threadResults.entrySampleIndexList){ sampleIdxList.clear(); sampleIdxList.reserve(1); }
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

  // get minimum overhead with low capacity
  // _cache_.entrySampleIndexList.reserve(nEntries);
  // for (auto& sampleIdxList : _cache_.entrySampleIndexList){ sampleIdxList.clear(); sampleIdxList.reserve(1); }

  for( auto& threadResults : _cache_.threadSelectionResults ){
    // merging nEvents

    for( int iSample = 0 ; iSample < int(_cache_.sampleNbOfEvents.size()) ; iSample++ ){
      _cache_.sampleNbOfEvents[iSample] += threadResults.sampleNbOfEvents[iSample];
    }

    _cache_.entrySampleIndexList.append_move(std::move(threadResults.entrySampleIndexList));

  }

  LogInfo << "Freeing up thread buffers..." << std::endl;
  _cache_.threadSelectionResults.clear();

  // get total amount
  for(size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
    _cache_.totalNbEvents += _cache_.sampleNbOfEvents[iSample];
  }

  if( _owner_->isShowSelectedEventCount() ){
    LogInfo << "Events passing selection cuts:" << std::endl;
    GenericToolbox::TablePrinter t;
    t << "Sample" << GenericToolbox::TablePrinter::NextColumn;
    t << "Selection" << GenericToolbox::TablePrinter::NextColumn;
    t << "# of events" << GenericToolbox::TablePrinter::NextLine;
    for(size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
      t << _cache_.samplesToFillList[iSample]->getName() << GenericToolbox::TablePrinter::NextColumn;
      t << _cache_.samplesToFillList[iSample]->getSelectionCutsStr() << GenericToolbox::TablePrinter::NextColumn;
      t << _cache_.sampleNbOfEvents[iSample] << GenericToolbox::TablePrinter::NextLine;
    }
    t.addSeparatorLine();
    t << "Total" << GenericToolbox::TablePrinter::NextColumn;
    t << "" << GenericToolbox::TablePrinter::NextColumn;
    t << _cache_.totalNbEvents << GenericToolbox::TablePrinter::NextLine;
    t.printTable();
  }

}
void DataDispenser::fetchRequestedLeaves(){
  LogInfo << "Poll all objects for requested variables..." << std::endl;

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
      auto applyConditionFormulaStr = dialCollection->getApplyConditionStr();
      if( not applyConditionFormulaStr.empty() ) {
        addVariablesRequestedByFormula(_cache_, applyConditionFormulaStr, _parameters_.variableDict, _parameters_.variableDictTransform, false);
        LogInfo << "DialCollection \"" << dialCollection->getTitle()
                << "\" applyCondition: \"" << applyConditionFormulaStr << "\"" << std::endl;
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
    for( auto& var : indexRequests ){
      std::set<std::string> variableDictStack;
      addVariableDictRequestedByName(_cache_, var, _parameters_.variableDict, _parameters_.variableDictTransform, variableDictStack);
    }
  }

  // sample binning -> indexing only
  {
    std::vector<std::string> varForIndexingListBuffer{};
    for (const auto& sample : _cache_.propagatorPtr->getSampleSet().getSampleList()) {
      for (const auto& binContext : sample.getHistogram().getBinContextList()) {
        for (const auto& edges : binContext.bin.getEdgesList()) {
          GenericToolbox::addIfNotInVector(edges.varName, varForIndexingListBuffer);
        }
      }
      if( not sample.getSelectionCutsStr().empty() ){
        addVariablesRequestedByFormula(_cache_, sample.getSelectionCutsStr(), _parameters_.variableDict, _parameters_.variableDictTransform, false);
      }
      auto sampleWeightFormulaStr = sample.getSampleWeightFormulaStr();
      if( not sampleWeightFormulaStr.empty() ){
        addVariablesRequestedByFormula(_cache_, sampleWeightFormulaStr, _parameters_.variableDict, _parameters_.variableDictTransform, false);
        LogInfo << "Sample \"" << sample.getName() << "\" weight formula: \"" << sampleWeightFormulaStr << "\"" << std::endl;
      }
    }
    LogInfo << "Samples variable request for indexing: " << GenericToolbox::toString(varForIndexingListBuffer) << std::endl;
    for( auto &var: varForIndexingListBuffer ){
      std::set<std::string> variableDictStack;
      addVariableDictRequestedByName(_cache_, var, _parameters_.variableDict, _parameters_.variableDictTransform, variableDictStack);
    }
  }

  // for event weight
  if( not _parameters_.eventVariableAsWeight.empty() ){
    LogInfo << "Variable for event weight: " << _parameters_.eventVariableAsWeight << std::endl;
    std::set<std::string> variableDictStack;
    addVariableDictRequestedByName(_cache_, _parameters_.eventVariableAsWeight, _parameters_.variableDict, _parameters_.variableDictTransform, variableDictStack);
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
      std::set<std::string> variableDictStack;
      addVariableDictRequestedByName(_cache_, var, _parameters_.variableDict, _parameters_.variableDictTransform, variableDictStack);
      GenericToolbox::addIfNotInVector(var, _cache_.propagatorPtr->getSampleSet().getEventVariableNameList());
    }
  }

  // storage requested by user
  {
    std::vector<std::string> varForStorageListBuffer{};
    varForStorageListBuffer = _parameters_.additionalVarsStorage;
    LogInfo << "Additional var requests for storage:" << GenericToolbox::toString(varForStorageListBuffer) << std::endl;
    for (auto &var: varForStorageListBuffer) {
      std::set<std::string> variableDictStack;
      addVariableDictRequestedByName(_cache_, var, _parameters_.variableDict, _parameters_.variableDictTransform, variableDictStack);
      GenericToolbox::addIfNotInVector(var, _cache_.propagatorPtr->getSampleSet().getEventVariableNameList());
    }
  }

  addVariablesRequestedByFormula(_cache_, _parameters_.selectionCutFormulaStr, _parameters_.variableDict, _parameters_.variableDictTransform, false);
  addVariablesRequestedByFormula(_cache_, _parameters_.nominalWeightFormulaStr, _parameters_.variableDict, _parameters_.variableDictTransform, false);
  addVariablesRequestedByFormula(_cache_, _parameters_.dialIndexFormula, _parameters_.variableDict, _parameters_.variableDictTransform, false);

  // LogInfo << "Vars requested for indexing: " << GenericToolbox::toString(_cache_.varsRequestedForIndexing, false) << std::endl;
  LogInfo << "Vars requested for storage: " << GenericToolbox::toString(_cache_.propagatorPtr->getSampleSet().getEventVariableNameList(), false) << std::endl;

  // Now build the var to leaf translation
  for( auto& var : _cache_.varsRequestedForIndexing ){
    _cache_.varToLeafDict[var].first = var;    // default is the same name
    _cache_.varToLeafDict[var].second = false; // is dummy branch?

    // strip brackets
    _cache_.varToLeafDict[var].first = GenericToolbox::stripBracket(_cache_.varToLeafDict[var].first, '[', ']');

    // look for override requests
    if(
        GenericToolbox::isIn(_cache_.varToLeafDict[var].first, _parameters_.variableDict)
        and not isActiveVariableDictName(_cache_, var)
    ){
      // leafVar will actually be the override leaf name while event will keep the original name
      _cache_.varToLeafDict[var].first = _parameters_.variableDict[_cache_.varToLeafDict[var].first];
      _cache_.varToLeafDict[var].first = GenericToolbox::stripBracket(_cache_.varToLeafDict[var].first, '[', ']');
    }

    if( isActiveVariableDictName(_cache_, var) and not doesVariableDictEntryNeedTreeValue(getActiveVariableDictEntry(_cache_, var)) ){
      _cache_.varToLeafDict[var].second = true;
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
    if( isEventBufferOnlyVariable(_cache_, var) ){ continue; }
    treeBuffer.addExpression( getVariableExpression( var ) );
  }
  treeBuffer.initialize();

  Event eventPlaceholder;
  eventPlaceholder.getIndices().dataset = _owner_->getDataSetIndex();
  eventPlaceholder.getVariables().setVarNameList( _cache_.propagatorPtr->getSampleSet().getEventVariableNameList() );

  std::vector<const GenericToolbox::TreeBuffer::ExpressionBuffer*> expList{};
  for( auto& storageVar : *eventPlaceholder.getVariables().getNameListPtr() ){
    if( isEventBufferOnlyVariable(_cache_, storageVar) ){
      expList.emplace_back(nullptr);
      continue;
    }
    expList.emplace_back( treeBuffer.getExpressionBuffer(getVariableExpression( storageVar )) );
  }

  for( size_t iExp = 0 ; iExp < expList.size() ; iExp++ ){
    if( expList[iExp] == nullptr ){ eventPlaceholder.getVariables().getVarList()[iExp].set(0.); }
    else{ eventPlaceholder.getVariables().getVarList()[iExp].set(expList[iExp]->getBuffer()); }
  }

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
    t.addSeparatorLine();
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

  size_t nTotalSlots{0};
  size_t nDialsMaxPerEvent{0};
  for( auto& dialCollection : _cache_.dialCollectionsRefList ){
    LogScopeIndent;
    nDialsMaxPerEvent += 1;

    if (dialCollection->isEventByEvent()) {
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
  }

  _cache_.propagatorPtr->getEventDialCache().allocateCacheEntries(_cache_.totalNbEvents, nDialsMaxPerEvent);
}
void DataDispenser::readAndFill(){
  LogInfo << "Reading dataset and loading..." << std::endl;
  this->_unbinnedEvents_.setValue(0);

  if( not _parameters_.nominalWeightFormulaStr.empty() ){
    LogInfo << "Nominal weight: \"" << _parameters_.nominalWeightFormulaStr << "\"" << std::endl;
  }
  if( not _parameters_.dialIndexFormula.empty() ){
    LogInfo << "Dial index for TClonesArray: \"" << _parameters_.dialIndexFormula << "\"" << std::endl;
  }

  LogInfo << "Loading and indexing..." << std::endl;
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
  LogExitIf( _parameters_.useReweightEngine, "Hist loader not implemented for MC containers" );

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
  return ::getVariableExpression(_cache_, _parameters_.variableDict, variable_);
}
std::shared_ptr<TChain> DataDispenser::openChain(bool verbose_) const{
  LogInfoIf(verbose_) << "Opening ROOT files containing events..." << std::endl;

  std::shared_ptr<TChain> treeChain(std::make_shared<TChain>());
  for( const auto& file: _parameters_.filePathList){
    std::string name = GenericToolbox::expandEnvironmentVariables(file);
    GenericToolbox::replaceSubstringInsideInputString(name, "//", "/");

    if( verbose_ ){
      LogScopeIndent;
      LogInfo << name << std::endl;
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

  DataDispenserCache selectionCache;
  addVariablesRequestedByFormula(
      selectionCache,
      _parameters_.selectionCutFormulaStr,
      _parameters_.variableDict,
      _parameters_.variableDictTransform,
      false
  );
  for( auto* samplePtr : _cache_.samplesToFillList ){
    addVariablesRequestedByFormula(
        selectionCache,
        samplePtr->getSelectionCutsStr(),
        _parameters_.variableDict,
        _parameters_.variableDictTransform,
        false
    );
  }
  LogInfoIf(iThread_ == 0) << "Selection variable requests: "
                           << GenericToolbox::toString(filterDisplayedVariableNames(selectionCache, selectionCache.varsRequestedForIndexing), false)
                           << std::endl;

  Event eventSelectionBuffer;
  eventSelectionBuffer.getIndices().dataset = _owner_->getDataSetIndex();
  eventSelectionBuffer.getVariables().setVarNameList(selectionCache.varsRequestedForIndexing);

  std::vector<const GenericToolbox::TreeBuffer::ExpressionBuffer*> varIndexingList;
  varIndexingList.reserve(selectionCache.varsRequestedForIndexing.size());

  // global cut
  ThreadSharedData::VariableBuffer::RuntimeFormula selectionCutFormula;
  LogInfoIf(iThread_ == 0 and not _parameters_.selectionCutFormulaStr.empty()) << "Global selection cut: \"" << _parameters_.selectionCutFormulaStr << "\"" << std::endl;
  compileRuntimeFormula(
      selectionCutFormula,
      tb,
      _parameters_.selectionCutFormulaStr,
      selectionCache.varsRequestedForIndexing,
      selectionCache.eventFormulaTreeExpressionAliases
  );

  // sample cuts
  struct SampleCut{
    int sampleIndex{-1};
    ThreadSharedData::VariableBuffer::RuntimeFormula formula{};
  };
  std::vector<SampleCut> sampleCutList;
  sampleCutList.reserve( _cache_.samplesToFillList.size() );

  for( int iSample = 0; iSample < int(_cache_.samplesToFillList.size()) ; iSample++ ){
    auto* samplePtr = _cache_.samplesToFillList[iSample];
    sampleCutList.emplace_back();
    sampleCutList.back().sampleIndex = iSample;

    std::string selectionCut = samplePtr->getSelectionCutsStr();
    if( selectionCut.empty() ){ continue; }
    compileRuntimeFormula(
        sampleCutList.back().formula,
        tb,
        selectionCut,
        selectionCache.varsRequestedForIndexing,
        selectionCache.eventFormulaTreeExpressionAliases
    );
  }

  std::vector<ThreadSharedData::VariableBuffer::VariableDictBuffer> variableDictEvalList;
  variableDictEvalList.reserve(selectionCache.variableDictEvalList.size());
  for( const auto& variableDictEntry : selectionCache.variableDictEvalList ){
    variableDictEvalList.emplace_back();
    variableDictEvalList.back().name = variableDictEntry.name;
    variableDictEvalList.back().outputVarIndex = GenericToolbox::findElementIndex(
        variableDictEntry.name,
        selectionCache.varsRequestedForIndexing
    );
    LogExitIf(variableDictEvalList.back().outputVarIndex == -1, "Missing variableDict output variable: " << variableDictEntry.name);
    if( variableDictEntry.backend == DataDispenserCache::VariableDictEntry::LibraryTransform ){
      variableDictEvalList.back().isLibraryTransform = true;
      variableDictEvalList.back().transform = *variableDictEntry.transformPtr;
    }
    else{
      compileRuntimeFormula(
          variableDictEvalList.back().formula,
          tb,
          variableDictEntry.expr,
          selectionCache.varsRequestedForIndexing,
          selectionCache.eventFormulaTreeExpressionAliases
      );
    }
  }

  for( auto& var : selectionCache.varsRequestedForIndexing ){
    varIndexingList.emplace_back();
    if( isActiveVariableDictName(selectionCache, var) and not doesVariableDictEntryNeedTreeValue(getActiveVariableDictEntry(selectionCache, var)) ){
      ThreadSharedData::VariableBuffer::storeTempIndex(varIndexingList.back(), -1);
      continue;
    }
    ThreadSharedData::VariableBuffer::storeTempIndex(
        varIndexingList.back(),
        tb.addExpression(::getVariableExpression(selectionCache, _parameters_.variableDict, var))
    );
  }

  tb.initialize();

  unfoldRuntimeFormula(selectionCutFormula, tb.getExpressionBufferList());
  for( auto& sampleCut : sampleCutList ){ unfoldRuntimeFormula(sampleCut.formula, tb.getExpressionBufferList()); }
  for( auto& variableDictEntry : variableDictEvalList ){
    if( variableDictEntry.isLibraryTransform ){ continue; }
    unfoldRuntimeFormula(variableDictEntry.formula, tb.getExpressionBufferList());
  }
  for( auto& varInd : varIndexingList ){ ThreadSharedData::VariableBuffer::unfoldTempIndex(varInd, tb.getExpressionBufferList()); }

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
    auto row = threadSelectionResults.entrySampleIndexList.emplace_back();
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

    for( size_t iVar = 0 ; iVar < varIndexingList.size() ; iVar++ ){
      if( varIndexingList[iVar] == nullptr ){ continue; }
      eventSelectionBuffer.getVariables().getVarList()[iVar].set(varIndexingList[iVar]->getBuffer());
    }
    evalVariableDict(eventSelectionBuffer, variableDictEvalList);

    if ( selectionCutFormula.isEnabled() ){
      if( selectionCutFormula.eval(eventSelectionBuffer) == 0 ){
        // skip it
        continue;
      }
    }

    bool sampleHasBeenFound{false};
    for( auto& sampleCut : sampleCutList ){

      if(  not sampleCut.formula.isEnabled()  // no cut?
           or sampleCut.formula.eval(eventSelectionBuffer) != 0 // pass cut?
          ){
        if( not _parameters_.allowMultipleSamplesPerEntry and sampleHasBeenFound ){
          LogError << "Entry #" << iEntry << "already has a sample." << std::endl;
          LogExit("Multi-sample event isn't enabled. By default, `allowMultipleSamplesPerEntry: false` by default.");
        }
        sampleHasBeenFound = true;
        row.emplace_back(sampleCut.sampleIndex);
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
  compileRuntimeFormula(
      threadSharedData.buffer.nominalWeightFormula,
      threadSharedData.treeBuffer,
      _parameters_.nominalWeightFormulaStr,
      _cache_.varsRequestedForIndexing,
      _cache_.eventFormulaTreeExpressionAliases
  );

  // dial array index
  compileRuntimeFormula(
      threadSharedData.buffer.dialIndexFormula,
      threadSharedData.treeBuffer,
      _parameters_.dialIndexFormula,
      _cache_.varsRequestedForIndexing,
      _cache_.eventFormulaTreeExpressionAliases
  );

  threadSharedData.buffer.dialApplyConditionFormulaList.resize(_cache_.dialCollectionsRefList.size());
  for( size_t iDialCollection = 0 ; iDialCollection < _cache_.dialCollectionsRefList.size() ; iDialCollection++ ){
    auto applyConditionFormulaStr = _cache_.dialCollectionsRefList[iDialCollection]->getApplyConditionStr();
    if( applyConditionFormulaStr.empty() ){ continue; }
    compileRuntimeFormula(
        threadSharedData.buffer.dialApplyConditionFormulaList[iDialCollection],
        threadSharedData.treeBuffer,
        applyConditionFormulaStr,
        _cache_.varsRequestedForIndexing,
        _cache_.eventFormulaTreeExpressionAliases
    );
  }

  threadSharedData.buffer.sampleWeightFormulaList.resize(_cache_.samplesToFillList.size());
  for( size_t iSample = 0 ; iSample < _cache_.samplesToFillList.size() ; iSample++ ){
    auto sampleWeightFormulaStr = _cache_.samplesToFillList[iSample]->getSampleWeightFormulaStr();
    if( sampleWeightFormulaStr.empty() ){ continue; }
    compileRuntimeFormula(
        threadSharedData.buffer.sampleWeightFormulaList[iSample],
        threadSharedData.treeBuffer,
        sampleWeightFormulaStr,
        _cache_.varsRequestedForIndexing,
        _cache_.eventFormulaTreeExpressionAliases
    );
  }

  threadSharedData.buffer.variableDictEvalList.reserve(_cache_.variableDictEvalList.size());
  for( const auto& variableDictEntry : _cache_.variableDictEvalList ){
    threadSharedData.buffer.variableDictEvalList.emplace_back();
    threadSharedData.buffer.variableDictEvalList.back().name = variableDictEntry.name;
    threadSharedData.buffer.variableDictEvalList.back().outputVarIndex = GenericToolbox::findElementIndex(
        variableDictEntry.name,
        _cache_.varsRequestedForIndexing
    );
    LogExitIf(threadSharedData.buffer.variableDictEvalList.back().outputVarIndex == -1, "Missing variableDict output variable: " << variableDictEntry.name);
    if( variableDictEntry.backend == DataDispenserCache::VariableDictEntry::LibraryTransform ){
      threadSharedData.buffer.variableDictEvalList.back().isLibraryTransform = true;
      threadSharedData.buffer.variableDictEvalList.back().transform = *variableDictEntry.transformPtr;
    }
    else{
      compileRuntimeFormula(
          threadSharedData.buffer.variableDictEvalList.back().formula,
          threadSharedData.treeBuffer,
          variableDictEntry.expr,
          _cache_.varsRequestedForIndexing,
          _cache_.eventFormulaTreeExpressionAliases
      );
    }
  }

  // variables definition
  for( auto& var : _cache_.varsRequestedForIndexing ){
    threadSharedData.buffer.varIndexingList.emplace_back();
    if( isActiveVariableDictName(_cache_, var) and not doesVariableDictEntryNeedTreeValue(getActiveVariableDictEntry(_cache_, var)) ){
      ThreadSharedData::VariableBuffer::storeTempIndex(threadSharedData.buffer.varIndexingList.back(), -1);
      continue;
    }
    ThreadSharedData::VariableBuffer::storeTempIndex(
      threadSharedData.buffer.varIndexingList.back(),
      threadSharedData.treeBuffer.addExpression(getVariableExpression(var))
    );
  }
  for( auto& var : _cache_.propagatorPtr->getSampleSet().getEventVariableNameList() ){
    threadSharedData.buffer.varStorageList.emplace_back();
    if( isActiveVariableDictName(_cache_, var) and not doesVariableDictEntryNeedTreeValue(getActiveVariableDictEntry(_cache_, var)) ){
      ThreadSharedData::VariableBuffer::storeTempIndex(threadSharedData.buffer.varStorageList.back(), -1);
      continue;
    }
    ThreadSharedData::VariableBuffer::storeTempIndex(
      threadSharedData.buffer.varStorageList.back(),
      threadSharedData.treeBuffer.addExpression(getVariableExpression(var))
    );
  }

  threadSharedData.treeBuffer.initialize();

  // grab ptr address now
  unfoldRuntimeFormula(threadSharedData.buffer.nominalWeightFormula, threadSharedData.treeBuffer.getExpressionBufferList());
  unfoldRuntimeFormula(threadSharedData.buffer.dialIndexFormula, threadSharedData.treeBuffer.getExpressionBufferList());
  for( auto& dialApplyCondition : threadSharedData.buffer.dialApplyConditionFormulaList ){ unfoldRuntimeFormula(dialApplyCondition, threadSharedData.treeBuffer.getExpressionBufferList()); }
  for( auto& sampleWeight : threadSharedData.buffer.sampleWeightFormulaList ){ unfoldRuntimeFormula(sampleWeight, threadSharedData.treeBuffer.getExpressionBufferList()); }
  for( auto& variableDictEntry : threadSharedData.buffer.variableDictEvalList ){
    if( variableDictEntry.isLibraryTransform ){ continue; }
    unfoldRuntimeFormula(variableDictEntry.formula, threadSharedData.treeBuffer.getExpressionBufferList());
  }
  for( auto& varInd: threadSharedData.buffer.varIndexingList ){ ThreadSharedData::VariableBuffer::unfoldTempIndex(varInd, threadSharedData.treeBuffer.getExpressionBufferList()); }
  for( auto& varSto: threadSharedData.buffer.varStorageList ){ ThreadSharedData::VariableBuffer::unfoldTempIndex(varSto, threadSharedData.treeBuffer.getExpressionBufferList()); }

  // event variable as weight
  if( not _parameters_.eventVariableAsWeight.empty() ){
    for( size_t iVar = 0 ; iVar < _cache_.varsRequestedForIndexing.size() ; iVar++ ){
      if( _cache_.varsRequestedForIndexing[iVar] == _parameters_.eventVariableAsWeight ) {
        threadSharedData.buffer.eventVarAsWeightIndex = int(iVar);
        break;
      }
    }

    LogExitIf(threadSharedData.buffer.eventVarAsWeightIndex == -1, "Could not find variable: " << _parameters_.eventVariableAsWeight);
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
    bool hasSample = not _cache_.entrySampleIndexList[iEntry].empty();
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

  std::unordered_map<int, DialBase*> eventByEventDialBuffer{};
  eventByEventDialBuffer.reserve(_cache_.dialCollectionsRefList.size());

  if(iThread_ == 0){

    LogInfo << "Feeding event variables with:" << std::endl;
    GenericToolbox::TablePrinter table;

    table << "Variable" ;
    table << GenericToolbox::TablePrinter::NextColumn << "Expression";
    table << GenericToolbox::TablePrinter::NextLine;

    struct VarDisplay{
      std::string varName{};

      std::string leafName{};
      std::string leafTypeName{};

      std::string lineColor{};

      int priorityIndex{-1};
    };
    std::vector<VarDisplay> varDisplayList{};

    bool hasEventDials{false};

    varDisplayList.reserve( _cache_.varsRequestedForIndexing.size() );
    for( size_t iVar = 0 ; iVar < _cache_.varsRequestedForIndexing.size() ; iVar++ ){
      if( isGeneratedTreeExpressionAlias(_cache_, _cache_.varsRequestedForIndexing[iVar]) ){ continue; }
      varDisplayList.emplace_back();

      varDisplayList.back().varName = _cache_.varsRequestedForIndexing[iVar];

      if( threadSharedData.buffer.varIndexingList[iVar] != nullptr ){
        varDisplayList.back().leafName = threadSharedData.buffer.varIndexingList[iVar]->getExpression();
        varDisplayList.back().leafTypeName = GenericToolbox::findOriginalVariableType(threadSharedData.buffer.varIndexingList[iVar]->getBuffer());
      }
      else{
        auto* variableDictEntry = getActiveVariableDictEntry(_cache_, _cache_.varsRequestedForIndexing[iVar]);
        varDisplayList.back().leafName = getVariableDisplayExpression(variableDictEntry);
        varDisplayList.back().leafTypeName = "formula";
      }

      varDisplayList.back().priorityIndex = 999;
      if( threadSharedData.buffer.varIndexingList[iVar] != nullptr and varDisplayList.back().leafTypeName != "\xFF" ){
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
      table.setColorBuffer( varDisplay.lineColor );
      table << varDisplay.varName << GenericToolbox::TablePrinter::NextColumn;
      table << varDisplay.leafName << "/" << varDisplay.leafTypeName << GenericToolbox::TablePrinter::NextColumn;
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
  size_t sampleEventIndex{};

  // make sure isEventFillerReady flag is true in this scope
  GenericToolbox::ScopedGuard g{
      [&]{ threadSharedData.isEventFillerReady.setValue( true ); },
      [&]{ threadSharedData.isEventFillerReady.setValue( false ); }
  };

  std::unordered_map<int, const TObject**> dialAddressMap;
  // std::vector<int> sampleIdxList;
  std::vector<int> sampleBinIdxList;
  std::vector<double> sampleWeightList;

  while( true ){

    // VERY IMPORTANT
    eventByEventDialBuffer.clear();

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
      for( size_t iVar = 0 ; iVar < threadSharedData.buffer.varIndexingList.size() ; iVar++ ){
        if( threadSharedData.buffer.varIndexingList[iVar] == nullptr ){ continue; }
        eventIndexingBuffer.getVariables().getVarList()[iVar].set(threadSharedData.buffer.varIndexingList[iVar]->getBuffer());
      }

      evalVariableDict(eventIndexingBuffer, threadSharedData.buffer.variableDictEvalList);

      // Default behavior for empty or omitted weight formulas must be neutral.
      eventIndexingBuffer.getWeights().base = 1;

      // nominalWeightTreeFormula is attached to the TChain
      if( threadSharedData.buffer.nominalWeightFormula.isEnabled() ){
        eventIndexingBuffer.getWeights().base = threadSharedData.buffer.nominalWeightFormula.eval(eventIndexingBuffer);
      }

      // additional weight with an event variable
      if( threadSharedData.buffer.eventVarAsWeightIndex != -1 ){
        eventIndexingBuffer.getWeights().base *= eventIndexingBuffer.getVariables().getVarList()[threadSharedData.buffer.eventVarAsWeightIndex].getVarAsDouble();
      }

      // skip this event if 0
      if( eventIndexingBuffer.getWeights().base == 0 ){ continue; }
      // no negative weights -> error
      if( eventIndexingBuffer.getWeights().base  < 0 ){
        LogError << "Negative nominal weight:" << std::endl;
        LogError << "Event buffer is: " << eventIndexingBuffer.getSummary() << std::endl;
        LogExit("Negative nominal weight");
      }

      // grab data from TChain
      eventIndexingBuffer.getIndices().entry     = threadSharedData.treeChain->GetReadEntry();
      eventIndexingBuffer.getIndices().treeFile      = threadSharedData.treeChain->GetTreeNumber();
      eventIndexingBuffer.getIndices().treeEntry = threadSharedData.treeChain->GetTree()->GetReadEntry();

      // get sample index / all -1 samples have been ruled out by the chain reader
      const auto& sampleIdxList = _cache_.entrySampleIndexList[eventIndexingBuffer.getIndices().entry];
      sampleBinIdxList.clear(); sampleBinIdxList.reserve(sampleIdxList.size());
      sampleWeightList.clear(); sampleWeightList.reserve(sampleIdxList.size());


      bool hasValidBin{false};
      for( auto& sampleIdx : sampleIdxList ) {
        Sample& eventSample{*_cache_.samplesToFillList[sampleIdx]};
        // look for the bin index
        LoaderUtils::fillBinIndex(eventIndexingBuffer, eventSample.getHistogram().getBinContextList());
        sampleBinIdxList.emplace_back(eventIndexingBuffer.getIndices().bin);

        // No bin found -> warning
        if( sampleBinIdxList.back() == -1 ){
          const int unbinnedEventThrottle = 5;
          if( this->_unbinnedEvents_++ < unbinnedEventThrottle ){

            // grab relevant variables
            auto varNames = LoaderUtils::getRelevantVarNames(eventSample.getHistogram().getBinContextList());

            LogAlert <<  "Selected event not in a likelihood histogram bin: " << std::endl;
            LogScopeIndent;
            LogAlert << eventIndexingBuffer.getSummary(false) << std::endl;
            LogAlert << "Variables used in binning{ ";
            for( auto& varName : varNames ) {
              LogAlert << varName << ": " << eventIndexingBuffer.getVariables().fetchVariable(varName).getVarAsDouble() << ", ";
            }
            LogAlert << "}" << std::endl;


            if ( this->_unbinnedEvents_.getValue() == unbinnedEventThrottle ) {
              LogAlert <<  "Further unbinned event warnings will be skipped."
                       << std::endl;
            }
          }
        }

        sampleWeightList.emplace_back(1);
        if( threadSharedData.buffer.sampleWeightFormulaList[sampleIdx].isEnabled() ){
          double sampleWeight = threadSharedData.buffer.sampleWeightFormulaList[sampleIdx].eval(eventIndexingBuffer);
          if( sampleWeight < 0 ) {
            LogError << "Negative sampleWeight:" << sampleWeight << std::endl;
            LogError << "sampleWeight buffer is: " << eventIndexingBuffer.getSummary() << std::endl;
            LogExit("Negative sampleWeight");
          }
          sampleWeightList.back() *= sampleWeight;
        }

        hasValidBin = hasValidBin or ( sampleBinIdxList.back() != -1 and sampleWeightList.back() != 0 );
      }

      if( not hasValidBin ){ continue; }

      // now we are sure the entry should be read till the end

      // dialIndexTreeFormula is modified by the TChain reader
      int dialCloneArrayIndex{0};
      if( threadSharedData.buffer.dialIndexFormula.isEnabled() ){
        dialCloneArrayIndex = static_cast<int>(threadSharedData.buffer.dialIndexFormula.eval(eventIndexingBuffer));
      }

      // only load event-by-event dials, binned dials etc. will be processed later
      for( size_t iDialCollection = 0 ; iDialCollection < _cache_.dialCollectionsRefList.size() ; iDialCollection++ ){
        auto *dialCollectionRef = _cache_.dialCollectionsRefList[iDialCollection];

        // if not event-by-event dial -> leave
        if( dialCollectionRef->getDialLeafName().empty() ){ continue; }

        if( threadSharedData.buffer.dialApplyConditionFormulaList[iDialCollection].isEnabled() ){
          if( threadSharedData.buffer.dialApplyConditionFormulaList[iDialCollection].eval(eventIndexingBuffer) == 0 ){
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

    }

    // ***** from this point, the TChain reader is released *****

    const auto& sampleIdxList = _cache_.entrySampleIndexList[eventIndexingBuffer.getIndices().entry];
    for( int iSample = 0 ; iSample < int(sampleIdxList.size()) ; iSample++ ){
      if( sampleBinIdxList[iSample] == -1 or sampleWeightList[iSample] == 0. ){ continue; }

      // update for this specific sample
      eventIndexingBuffer.getIndices().sample = _cache_.samplesToFillList[sampleIdxList[iSample]]->getIndex();
      eventIndexingBuffer.getIndices().bin = sampleBinIdxList[iSample];
      eventIndexingBuffer.getWeights().base *= sampleWeightList[iSample];

      // Let's claim an index. Indices are shared among threads
      EventDialCache::IndexedCacheEntry *eventDialCacheEntry{nullptr};
      {
        std::unique_lock<std::mutex> lock(_mutex_);
        eventDialCacheEntry = _cache_.propagatorPtr->getEventDialCache().fetchNextCacheEntry();
        sampleEventIndex = _cache_.sampleIndexOffsetList[sampleIdxList[iSample]]++;
      }

      // Get the next free event in our buffer
      Event *eventPtr = &(*_cache_.sampleEventListPtrToFill[sampleIdxList[iSample]])[sampleEventIndex];

      // copy from the event indexing buffer
      LoaderUtils::copyData(eventIndexingBuffer, *eventPtr);

      // Now the event is ready. Let's index the dials:
      // there should always be a cache entry even if no dials are applied.
      // This cache is actually used to write MC events with dials in output tree
      eventDialCacheEntry->event.sampleIndex = std::size_t(eventIndexingBuffer.getIndices().sample);
      eventDialCacheEntry->event.eventIndex = sampleEventIndex;

      auto *dialEntryPtr = eventDialCacheEntry->dials.data();
      for( size_t iDialCollection = 0 ; iDialCollection < _cache_.dialCollectionsRefList.size() ; iDialCollection++ ){
        auto *dialCollectionRef = _cache_.dialCollectionsRefList[iDialCollection];

        // leave if event-by-event -> already loaded
        if( not dialCollectionRef->getDialLeafName().empty() ){

          // dialBase is valid -> store it
          if( eventByEventDialBuffer[dialCollectionRef->getIndex()] != nullptr ){
            size_t freeSlotDial = dialCollectionRef->getNextDialFreeSlot();

            if( iSample == 0 ) {
              dialCollectionRef->getDialInterfaceList()[freeSlotDial].getDial().dialPtr
                = std::unique_ptr<DialBase>(eventByEventDialBuffer[dialCollectionRef->getIndex()]);
            }
            else {
              // cloning the dial ptr, otherwise the ownership will be lost
              dialCollectionRef->getDialInterfaceList()[freeSlotDial].getDial().dialPtr
                = std::unique_ptr<DialBase>(eventByEventDialBuffer[dialCollectionRef->getIndex()]->clone());
            }

            dialEntryPtr->collectionIndex = dialCollectionRef->getIndex();
            dialEntryPtr->interfaceIndex = freeSlotDial;
            dialEntryPtr++;
          }

          continue; // skip the rest
        }

        if( threadSharedData.buffer.dialApplyConditionFormulaList[iDialCollection].isEnabled() ){
          if( threadSharedData.buffer.dialApplyConditionFormulaList[iDialCollection].eval(eventIndexingBuffer) == 0 ){
            // next dialSet
            continue;
          }
        }

        int iCollection = dialCollectionRef->getIndex();

        if( dialCollectionRef->getDialType() == DialCollection::DialType::Tabulated
            or dialCollectionRef->getDialType() == DialCollection::DialType::Kriged ) {
          // Event-by-event dial with a factory.

          std::unique_ptr<DialBase> dialBase(
              dialCollectionRef->getCollectionData<DialFactoryBase>(0)
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
