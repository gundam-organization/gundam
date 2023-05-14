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
void DialInterface::setDialBinRef(const DataBin *dialBinRef) {
  _dialBinRef_ = dialBinRef;
}

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

void DialInterface::fillWithDialResponse(TGraph *graphBuffer_) const {
  LogThrowIf(graphBuffer_ == nullptr, "no buffer provided for " << std::endl << this->getSummary( false ));
  DialInputBuffer inputBuf{*_inputBufferRef_};
  for( int iPt = 0 ; iPt < graphBuffer_->GetN() ; iPt++ ){
    LogThrowIf(inputBuf.getBufferSize() != 1, "multi-dim dial not supported yet.");
    inputBuf.getBufferVector()[0] = graphBuffer_->GetX()[iPt];
    graphBuffer_->GetY()[iPt] = DialInterface::evalResponse(&inputBuf, _dialBaseRef_, _responseSupervisorRef_);
  }
}

double DialInterface::evalResponse(
    DialInputBuffer *inputBufferPtr_, DialBase *dialBaseRef_,
    const DialResponseSupervisor *responseSupervisorRef_
    ) {
  if( inputBufferPtr_->isMasked() ){ return 1; }
  return responseSupervisorRef_->process( dialBaseRef_->evalResponse( *inputBufferPtr_ ) );
}
