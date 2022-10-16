//
// Created by Adrien BLANCHET on 13/10/2022.
//

#include "EventVarTransform.h"
#include "JsonUtils.h"

#include "GenericToolbox.h"
#include "Logger.h"

#include <dlfcn.h>


LoggerInit([]{
  Logger::setUserHeaderStr("[EventVarTransform]");
});


EventVarTransform::EventVarTransform(const nlohmann::json& config_){ this->initialize(config_); }

void EventVarTransform::initialize(const nlohmann::json& config_){
  this->readConfig(config_);

  LogInfo << "Loading variable transformation: " << _title_ << std::endl;

  this->loadLibrary();
  this->initInputFormulas();
}

void EventVarTransform::setIndex(int index_){ _index_ = index_; }
void EventVarTransform::setUseCache(bool useCache_) { _useCache_ = useCache_; }

int EventVarTransform::getIndex() const { return _index_; }
bool EventVarTransform::useCache() const { return _useCache_; }
const std::string &EventVarTransform::getTitle() const { return _title_; }
const std::string &EventVarTransform::getOutputVariableName() const { return _outputVariableName_; }

const std::vector<std::string>& EventVarTransform::fetchRequestedVars() const {
  if( _requestedLeavesForEvalCache_.empty() ){
    for( auto& formula : _inputFormulaList_ ){
      for( int iPar = 0 ; iPar < formula.GetNpar() ; iPar++ ){
        if( not GenericToolbox::doesElementIsInVector(formula.GetParName(iPar), _requestedLeavesForEvalCache_) ){
          _requestedLeavesForEvalCache_.emplace_back( formula.GetParName(iPar) );
        }
      }
    }
  }
  return _requestedLeavesForEvalCache_;
}

double EventVarTransform::eval(const PhysicsEvent& event_){
  if( _useCache_ ){ return ( _outputCache_ = this->evalTransformation(event_, _inputBuffer_) ); }
  return this->evalTransformation(event_);
}
void EventVarTransform::storeCachedOutput(PhysicsEvent& event_){
  this->storeOutput(_outputCache_, event_);
}
void EventVarTransform::evalAndStore(PhysicsEvent& event_){
  this->storeOutput(this->eval(event_), event_);
}
void EventVarTransform::evalAndStore(const PhysicsEvent& evalEvent_, PhysicsEvent& storeEvent_){
  this->storeOutput(this->eval(evalEvent_), storeEvent_);
}

void EventVarTransform::readConfig(const nlohmann::json& config_){
  _title_ = JsonUtils::fetchValue<std::string>(config_, "title");
  _libraryFile_ = JsonUtils::fetchValue<std::string>(config_, "libraryFile");
  _messageOnError_ = JsonUtils::fetchValue<std::string>(config_, "messageOnError", "");
  _outputVariableName_ = JsonUtils::fetchValue<std::string>(config_, "outputVariableName");
  _inputFormulaStrList_ = JsonUtils::fetchValue(config_, "inputList", std::vector<std::string>());
}
void EventVarTransform::loadLibrary(){
  LogInfo << "Loading shared lib: " << _libraryFile_ << std::endl;
  _loadedLibrary_ = dlopen(_libraryFile_.c_str(), RTLD_LAZY );
  LogThrowIf(_loadedLibrary_ == nullptr, "Cannot open library: " << dlerror() << std::endl << _messageOnError_);
  _evalVariable_ = (dlsym(_loadedLibrary_, "evalVariable"));
  LogThrowIf(_evalVariable_ == nullptr, "Cannot open evalFcn" << std::endl << _messageOnError_);
}
void EventVarTransform::initInputFormulas(){
  _inputFormulaList_.clear();
  for( auto& inputFormulaStr : _inputFormulaStrList_ ){
    _inputFormulaList_.emplace_back( inputFormulaStr.c_str(), inputFormulaStr.c_str() );
    LogThrowIf(not _inputFormulaList_.back().IsValid(), "\"" << inputFormulaStr << "\": could not be parsed as formula expression.")
  }
  _inputBuffer_.resize(_inputFormulaList_.size(), std::nan("unset"));
}

double EventVarTransform::evalTransformation(const PhysicsEvent& event_) const {
  std::vector<double> buff(_inputFormulaList_.size());
  return this->evalTransformation(event_, buff);
}
double EventVarTransform::evalTransformation(const PhysicsEvent& event_, std::vector<double>& inputBuffer_) const{
  LogThrowIf(_evalVariable_ == nullptr, "Library not loaded properly.");
  // Eval the requested variables
  size_t nFormula{_inputFormulaList_.size()};
  for( size_t iFormula = 0 ; iFormula < nFormula ; iFormula++ ){
    inputBuffer_[iFormula] = event_.evalFormula(&(_inputFormulaList_[iFormula]));
  }
  // Eval with dynamic function
  return reinterpret_cast<double(*)(double*)>(_evalVariable_)(&inputBuffer_[0]);
}
void EventVarTransform::storeOutput(double output_, PhysicsEvent& storeEvent_) const{
  storeEvent_.getVariable<double>(this->getOutputVariableName()) = output_;
}

