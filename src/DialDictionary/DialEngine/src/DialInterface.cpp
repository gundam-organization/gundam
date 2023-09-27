//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialInterface.h"
#include "GundamGlobals.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[DialInterface]");
});


double DialInterface::evalResponse() const {
  return DialInterface::evalResponse(_inputBufferRef_, _dialBaseRef_, _responseSupervisorRef_);
}
std::string DialInterface::getSummary(bool shallow_) const {
  std::stringstream ss;
  ss << _dialBaseRef_->getDialTypeName() << ":";

  if( _inputBufferRef_ != nullptr ){

    ss << " ";

    if( _inputBufferRef_->isMasked() ){
      ss << GenericToolbox::ColorCodes::redBackground;
    }

    ss << "input(" << _inputBufferRef_->getSummary() << ")";

    if( _inputBufferRef_->isMasked() ){
      ss << "/MASKED" << GenericToolbox::ColorCodes::resetColor;
    }
  }

  // apply on?
  if( _dialBinRef_ != nullptr ){
    ss << " applyOn(" << _dialBinRef_->getSummary() << ")";
  }

  if( _responseSupervisorRef_ != nullptr ){
    ss << " supervisor(" << _responseSupervisorRef_->getSummary() << ")";
  }

  ss << " response=" << this->evalResponse();

  if( not shallow_ ){
    ss << std::endl;
    ss << _dialBaseRef_->getSummary();
  }

  return ss.str();
}

double DialInterface::evalResponse(
    DialInputBuffer *inputBufferPtr_, DialBase *dialBaseRef_,
    const DialResponseSupervisor *responseSupervisorRef_
    ) {
  if( inputBufferPtr_->isMasked() ){ return 1; }
  return responseSupervisorRef_->process( dialBaseRef_->evalResponse( *inputBufferPtr_ ) );
}
