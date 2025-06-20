//
// Created by Adrien BLANCHET on 11/11/2022.
//

#include "EventVarTransformLib.h"
#include "LoaderUtils.h"
#include "ConfigUtils.h"

#include "Logger.h"

#include <dlfcn.h>


void EventVarTransformLib::configureImpl(){
  _config_.clearFields();
  this->EventVarTransform::configureImpl();
  _config_.defineFields({{"libraryFile"}});
  _config_.fillValue(_libraryFile_, "libraryFile");
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
  LogInfo << "Loading shared lib: " << _libraryFile_ << std::endl;
  LogThrowIf(not GenericToolbox::isFile(_libraryFile_), "Could not find lib file: " << _libraryFile_ << std::endl << _messageOnError_);
  _loadedLibrary_ = dlopen(_libraryFile_.c_str(), RTLD_LAZY );
  LogThrowIf(_loadedLibrary_ == nullptr, "Cannot open library: " << dlerror() << std::endl << _messageOnError_);
  _evalVariable_ = (dlsym(_loadedLibrary_, "evalVariable"));
  LogThrowIf(_evalVariable_ == nullptr, "Cannot open evalFcn" << std::endl << _messageOnError_);
}
void EventVarTransformLib::initInputFormulas(){
  _inputFormulaList_.clear();
  for( auto& inputFormulaStr : _inputFormulaStrList_ ){
    _inputFormulaList_.emplace_back( inputFormulaStr.c_str(), inputFormulaStr.c_str() );
    LogThrowIf(not _inputFormulaList_.back().IsValid(), "\"" << inputFormulaStr << "\": could not be parsed as formula expression.")
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
