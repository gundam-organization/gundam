//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialInterface.h"
#include "GlobalVariables.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[DialInterface]");
});

void DialInterface::setDialBaseRef(DialBase *dialBasePtr) {
  _dialBaseRef_ = dialBasePtr;
}
void DialInterface::setInputBufferRef(DialInputBuffer *inputBufferRef) {
  _inputBufferRef_ = inputBufferRef;
}
void DialInterface::setResponseSupervisorRef(const DialResponseSupervisor *responseSupervisorRef) {
  _responseSupervisorRef_ = responseSupervisorRef;
}

double DialInterface::evalResponse(){
  if( _inputBufferRef_->isMasked() ) return 1;

  double response = _dialBaseRef_->evalResponse( *_inputBufferRef_ );

  if(_responseSupervisorRef_ != nullptr ){
    _responseSupervisorRef_->process(response);
  }

  return response;
}
std::string DialInterface::getSummary(bool shallow_) const {
  std::stringstream ss;
  ss << "DialInterface:";
  if( _dialBaseRef_ != nullptr ) ss << " DialBase(" << _dialBaseRef_ << ")";
  if( _inputBufferRef_ != nullptr ) ss << " nInputs=" << _inputBufferRef_->getBufferSize() << "(" << _inputBufferRef_->getBuffer() << ")";
//  if( _responseSupervisorRef_ != nullptr ) ss << " nInputs=" << _responseSupervisorRef_->;
  if( not shallow_){

  }
  return ss.str();
}
