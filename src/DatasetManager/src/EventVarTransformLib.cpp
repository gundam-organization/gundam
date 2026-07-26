//
// Created by Adrien BLANCHET on 11/11/2022.
//

#include "EventVarTransformLib.h"
#include "LoaderUtils.h"
#include "ConfigUtils.h"
#include "FormulaUtils.h"

#include "Logger.h"

#include <dlfcn.h>


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

  LogInfo << "Loading variable transformation: " << _name_ << std::endl;
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
void EventVarTransformLib::initInputFormulas(){
  _inputFormulaList_.clear();
  for( auto& inputFormulaStr : _inputFormulaStrList_ ){
    auto formulaStr = FormulaUtils::convertBareVariablesToFormulaParameters(inputFormulaStr);
    _inputFormulaList_.emplace_back( formulaStr.c_str(), formulaStr.c_str() );
    LogThrowIf(not _inputFormulaList_.back().IsValid(), "\"" << inputFormulaStr << "\" -> \"" << formulaStr << "\": could not be parsed as formula expression.")
  }
  _inputBuffer_.resize(_inputFormulaList_.size(), std::nan("unset"));
}
double EventVarTransformLib::evalTransformation( const Event& event_, std::vector<double>& inputBuffer_) const{
  std::lock_guard<std::mutex> guard(GundamGlobals::getGlobalMutEx());
  // Eval the requested variables
  size_t nFormula{_inputFormulaList_.size()};
  for( size_t iFormula = 0 ; iFormula < nFormula ; iFormula++ ){
    inputBuffer_[iFormula] = LoaderUtils::evalFormula(event_, &(_inputFormulaList_[iFormula]));
  }
  // Eval with dynamic function
  return reinterpret_cast<double(*)(double*)>(_evalVariable_)(&inputBuffer_[0]);
}
