//
// Created by Adrien BLANCHET on 13/10/2022.
//

#include "EventVarTransform.h"
#include "ConfigUtils.h"

#include "Logger.h"


void EventVarTransform::configureImpl(){
  GenericToolbox::Json::fillValue(_config_, _name_, {{"name"}, {"title"}});
  GenericToolbox::Json::fillValue(_config_, _isEnabled_, "isEnabled");
  GenericToolbox::Json::fillValue(_config_, _messageOnError_, "messageOnError");
  GenericToolbox::Json::fillValue(_config_, _outputVariableName_, "outputVariableName");
  GenericToolbox::Json::fillValue(_config_, _inputFormulaStrList_, "inputList");
}
void EventVarTransform::initializeImpl(){
  LogInfo << "Loading variable transformation: " << _name_ << std::endl;
  LogThrowIf(_outputVariableName_.empty(), "output variable name not set.");
}


EventVarTransform::EventVarTransform(const JsonType& config_){ this->configure(config_); }

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

double EventVarTransform::eval(const Event& event_){
  if( not _useCache_ ){ return this->evalTransformation(event_); }
  _outputCache_ = this->evalTransformation(event_, _inputBuffer_);
  return _outputCache_;
}
void EventVarTransform::storeCachedOutput( Event& event_) const{
  this->storeOutput(_outputCache_, event_);
}
void EventVarTransform::evalAndStore( Event& event_){
  this->storeOutput(this->eval(event_), event_);
}
void EventVarTransform::evalAndStore( const Event& evalEvent_, Event& storeEvent_){
  this->storeOutput(this->eval(evalEvent_), storeEvent_);
}



double EventVarTransform::evalTransformation(const Event& event_) const {
  std::vector<double> buff(_inputFormulaList_.size());
  return this->evalTransformation(event_, buff);
}
double EventVarTransform::evalTransformation( const Event& event_, std::vector<double>& inputBuffer_) const{
  return std::nan("defaultEvalTransformOutput");
}
void EventVarTransform::storeOutput( double output_, Event& storeEvent_ ) const{
  auto& variable = storeEvent_.getVariables().fetchVariable(this->getOutputVariableName());
  variable.set( output_ );
}

