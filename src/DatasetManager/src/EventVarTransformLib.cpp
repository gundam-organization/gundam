//
// Created by Adrien BLANCHET on 11/11/2022.
//

#include "EventVarTransformLib.h"
#include "LoaderUtils.h"
#include "ConfigUtils.h"
#include "FormulaUtils.h"

#include "Logger.h"

#include <dlfcn.h>
#include <cctype>


namespace {

  bool isIdentifierStart(char c_){
    return std::isalpha(static_cast<unsigned char>(c_)) or c_ == '_';
  }

  bool isIdentifierChar(char c_){
    return std::isalnum(static_cast<unsigned char>(c_)) or c_ == '_';
  }

  bool parseBracketToken(
      const std::string& formula_,
      size_t openingBracketPos_,
      size_t& closingBracketPos_,
      std::string& token_
  ){
    if( openingBracketPos_ >= formula_.size() or formula_[openingBracketPos_] != '[' ){ return false; }
    closingBracketPos_ = formula_.find(']', openingBracketPos_ + 1);
    LogThrowIf(
        closingBracketPos_ == std::string::npos,
        "Invalid formula reference in \"" << formula_ << "\": missing closing ']'."
    );
    token_ = formula_.substr(openingBracketPos_ + 1, closingBracketPos_ - openingBracketPos_ - 1);
    return true;
  }

}


void EventVarTransformLib::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({
      {FieldFlag::MANDATORY, "name", {"title"}},
      {FieldFlag::MANDATORY, "outputVariableName"},
      {"isEnabled"},
      {"inputList"},
      {"messageOnError"},
      {"libraryFile"},
  });
  _config_.checkConfiguration();

  _config_.fillValue(_name_, "name");
  _config_.fillValue(_isEnabled_, "isEnabled");
  _config_.fillValue(_messageOnError_, "messageOnError");
  _config_.fillValue(_outputVariableName_, "outputVariableName");
  _inputFormulaStrList_ = ConfigUtils::readFormulaExprList(_config_, "inputList");
  _config_.fillValue(_libraryFile_, "libraryFile");
}
void EventVarTransformLib::configureFromVariableDict(const std::string& outputVariableName_, ConfigReader& config_){
  config_.defineFields({
      {FieldFlag::MANDATORY, "libraryFile"},
      {FieldFlag::MANDATORY, "inputList"},
      {"messageOnError"},
      {"title", {"name"}},
      {"isEnabled"},
  });
  config_.checkConfiguration();

  this->setOutputVariableName(outputVariableName_);
  this->setName(outputVariableName_);
  config_.fillValue(_name_, "title");
  config_.fillValue(_isEnabled_, "isEnabled");
  config_.fillValue(_messageOnError_, "messageOnError");
  config_.fillValue(_libraryFile_, "libraryFile");
  this->setInputFormulaStrList(ConfigUtils::readFormulaExprList(config_, "inputList"));
  config_.printUnusedKeys();
}
void EventVarTransformLib::initializeImpl(){

  _config_.printUnusedKeys();

  LogThrowIf(_outputVariableName_.empty(), "output variable name not set.");

  this->reload();
}

void EventVarTransformLib::reload(){
  this->loadLibrary();
  this->initInputFormulas();
}

void EventVarTransformLib::loadLibrary(){
  LogThrowIf(not GenericToolbox::isFile(_libraryFile_), "Could not find lib file: " << _libraryFile_ << std::endl << _messageOnError_);
  _loadedLibrary_ = dlopen(_libraryFile_.c_str(), RTLD_LAZY );
  LogThrowIf(_loadedLibrary_ == nullptr, "Cannot open library: " << dlerror() << std::endl << _messageOnError_);
  _evalVariable_ = (dlsym(_loadedLibrary_, "evalVariable"));
  LogThrowIf(_evalVariable_ == nullptr, "Cannot open evalFcn" << std::endl << _messageOnError_);
}
std::string EventVarTransformLib::registerInputFormulaArrayAlias(const std::string& sourceExpression_, size_t formulaIndex_){
  std::string alias = "__gundam_evtlib_expr_" + std::to_string(formulaIndex_) + "_" + std::to_string(getInputFormulaParameterSourceCount());
  registerInputFormulaParameterSource(alias, sourceExpression_);
  return alias;
}
std::string EventVarTransformLib::rewriteInputFormulaTreeArrayExpressions(const std::string& formula_, size_t formulaIndex_){
  std::string output;
  output.reserve(formula_.size());

  size_t cursor{0};
  while( cursor < formula_.size() ){
    if( isIdentifierStart(formula_[cursor]) ){
      auto identifierBegin = cursor;
      cursor++;
      while( cursor < formula_.size() and isIdentifierChar(formula_[cursor]) ){ cursor++; }

      if( cursor < formula_.size() and formula_[cursor] == '[' ){
        auto sourceExpression = formula_.substr(identifierBegin, cursor - identifierBegin);
        auto expressionEnd = cursor;
        while( expressionEnd < formula_.size() and formula_[expressionEnd] == '[' ){
          size_t closingBracketPos{0};
          std::string token;
          parseBracketToken(formula_, expressionEnd, closingBracketPos, token);
          sourceExpression += "[" + token + "]";
          expressionEnd = closingBracketPos + 1;
        }
        output += registerInputFormulaArrayAlias(sourceExpression, formulaIndex_);
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
    size_t firstClosingBracketPos{0};
    std::string firstToken;
    parseBracketToken(formula_, openingBracketPos, firstClosingBracketPos, firstToken);

    if(
        not firstToken.empty()
        and isIdentifierStart(firstToken[0])
        and std::all_of(firstToken.begin() + 1, firstToken.end(), [](char c_){ return isIdentifierChar(c_); })
        and firstClosingBracketPos + 1 < formula_.size()
        and formula_[firstClosingBracketPos + 1] == '['
    ){
      auto sourceExpression = firstToken;
      auto expressionEnd = firstClosingBracketPos + 1;
      while( expressionEnd < formula_.size() and formula_[expressionEnd] == '[' ){
        size_t closingBracketPos{0};
        std::string token;
        parseBracketToken(formula_, expressionEnd, closingBracketPos, token);
        sourceExpression += "[" + token + "]";
        expressionEnd = closingBracketPos + 1;
      }
      output += "[" + registerInputFormulaArrayAlias(sourceExpression, formulaIndex_) + "]";
      cursor = expressionEnd;
      continue;
    }

    output.append(formula_, openingBracketPos, firstClosingBracketPos - openingBracketPos + 1);
    cursor = firstClosingBracketPos + 1;
  }

  return output;
}
void EventVarTransformLib::initInputFormulas(){
  _inputFormulaList_.clear();
  _inputFormulaParameterSourceDict_.clear();
  _requestedLeavesForEvalCache_.clear();
  for( size_t iFormula = 0 ; iFormula < _inputFormulaStrList_.size() ; iFormula++ ){
    auto rewrittenFormulaStr = rewriteInputFormulaTreeArrayExpressions(_inputFormulaStrList_[iFormula], iFormula);
    auto formulaStr = FormulaUtils::convertBareVariablesToFormulaParameters(rewrittenFormulaStr);
    _inputFormulaList_.emplace_back( formulaStr.c_str(), formulaStr.c_str() );
    LogThrowIf(not _inputFormulaList_.back().IsValid(), "\"" << _inputFormulaStrList_[iFormula] << "\" -> \"" << formulaStr << "\": could not be parsed as formula expression.")
  }
  _inputBuffer_.resize(_inputFormulaList_.size(), std::nan("unset"));
}
double EventVarTransformLib::evalTransformation( const Event& event_, std::vector<double>& inputBuffer_) const{
  std::lock_guard<std::mutex> guard(GundamGlobals::getGlobalMutEx());
  // Eval the requested variables
  size_t nFormula{_inputFormulaList_.size()};
  for( size_t iFormula = 0 ; iFormula < nFormula ; iFormula++ ){
    std::vector<double> parArray(_inputFormulaList_[iFormula].GetNpar());
    for( int iPar = 0 ; iPar < _inputFormulaList_[iFormula].GetNpar() ; iPar++ ){
      parArray[iPar] = event_.getVariables().fetchVariable(getInputFormulaParameterSource(_inputFormulaList_[iFormula].GetParName(iPar))).getVarAsDouble();
    }
    inputBuffer_[iFormula] = _inputFormulaList_[iFormula].EvalPar(nullptr, parArray.empty() ? nullptr : parArray.data());
  }
  // Eval with dynamic function
  return reinterpret_cast<double(*)(double*)>(_evalVariable_)(&inputBuffer_[0]);
}
