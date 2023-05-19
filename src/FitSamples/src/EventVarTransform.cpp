//
// Created by Adrien BLANCHET on 13/10/2022.
//

#include "EventVarTransform.h"
#include "GenericToolbox.Json.h"

#include "GenericToolbox.h"
#include "Logger.h"



LoggerInit([]{
  Logger::setUserHeaderStr("[EventVarTransform]");
});

void EventVarTransform::readConfigImpl(){
  _title_ = GenericToolbox::Json::fetchValue(_config_, "title", _title_);
  _messageOnError_ = GenericToolbox::Json::fetchValue(_config_, "messageOnError", _messageOnError_);
  _outputVariableName_ = GenericToolbox::Json::fetchValue(_config_, "outputVariableName", _outputVariableName_);
  _inputFormulaStrList_ = GenericToolbox::Json::fetchValue(_config_, "inputList", _inputFormulaStrList_);
}
void EventVarTransform::initializeImpl(){
  LogInfo << "Loading variable transformation: " << _title_ << std::endl;
  LogThrowIf(_outputVariableName_.empty(), "output variable name not set.");
}


EventVarTransform::EventVarTransform(const nlohmann::json& config_){ this->readConfig(config_); }

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
        GenericToolbox::addIfNotInVector(formula.GetParName(iPar), _requestedLeavesForEvalCache_);
      }
    }
  }
  return _requestedLeavesForEvalCache_;
}

double EventVarTransform::eval(const PhysicsEvent& event_){
  if( not _useCache_ ){ return this->evalTransformation(event_); }
  _outputCache_ = this->evalTransformation(event_, _inputBuffer_);
  return _outputCache_;
}
void EventVarTransform::storeCachedOutput(PhysicsEvent& event_) const{
  this->storeOutput(_outputCache_, event_);
}
void EventVarTransform::evalAndStore(PhysicsEvent& event_){
  this->storeOutput(this->eval(event_), event_);
}
void EventVarTransform::evalAndStore(const PhysicsEvent& evalEvent_, PhysicsEvent& storeEvent_){
  this->storeOutput(this->eval(evalEvent_), storeEvent_);
}



double EventVarTransform::evalTransformation(const PhysicsEvent& event_) const {
  std::vector<double> buff(_inputFormulaList_.size());
  return this->evalTransformation(event_, buff);
}
double EventVarTransform::evalTransformation(const PhysicsEvent& event_, std::vector<double>& inputBuffer_) const{
  return std::nan("defaultEvalTransformOutput");
}
void EventVarTransform::storeOutput(double output_, PhysicsEvent& storeEvent_) const{
  storeEvent_.setVariable(output_, this->getOutputVariableName());
}

