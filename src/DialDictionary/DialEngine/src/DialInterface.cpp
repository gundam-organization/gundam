//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialInterface.h"
#include "GundamGlobals.h"

#include "Logger.h"

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[DialInterface]"); });
#endif


double DialInterface::evalResponse() const {
  return DialInterface::evalResponse(_inputBufferRef_, _dialBaseRef_, _responseSupervisorRef_);
}
std::string DialInterface::getSummary(bool shallow_) const {
  std::stringstream ss;
  ss << _dialBaseRef_->getDialTypeName() << ":";

  if( _inputBufferRef_ != nullptr ){
    ss << " input(" << _inputBufferRef_->getSummary() << ")";
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
  return responseSupervisorRef_->process( dialBaseRef_->evalResponse( *inputBufferPtr_ ) );
}
