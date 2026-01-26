//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialInterface.h"
#include "GundamGlobals.h"

#include "Logger.h"


std::string DialInterface::getSummary(bool shallow_) const {
  std::stringstream ss;
  ss << _dial_->getDialTypeName() << ":";

  if( _inputBufferPtr_ != nullptr ){
    ss << " input(" << _inputBufferPtr_->getSummary(shallow_) << ")";
  }

  // apply on?
  if( _dialBinRef_ != nullptr ){
    ss << " applyOn(" << _dialBinRef_->getSummary(shallow_) << ")";
  }

  if( _responseSupervisorPtr_ != nullptr ){
    ss << " supervisor(" << _responseSupervisorPtr_->getSummary(shallow_) << ")";
  }

  ss << " response=" << this->evalResponse();

  if( not shallow_ ){
    ss << std::endl;
    ss << _dial_->getSummary();
  }

  return ss.str();
}
