//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialInterface.h"
#include "GlobalVariables.h"

void DialInterface::setDialBaseRef(DialBase *dialBasePtr) {
  _dialBaseRef_ = dialBasePtr;
}
void DialInterface::setInputBufferRef(const DialInputBuffer *inputBufferRef) {
  _inputBufferRef_ = inputBufferRef;
}
void DialInterface::setResponseSupervisorRef(const DialResponseSupervisor *responseSupervisorRef) {
  _responseSupervisorRef_ = responseSupervisorRef;
}

double DialInterface::evalResponse(){
  double response = _dialBaseRef_->evalResponse(*_inputBufferRef_ );

  if(_responseSupervisorRef_ != nullptr ){
    _responseSupervisorRef_->process(response);
  }

  return response;
}
void DialInterface::propagateToTargets(int iThread_){
  if( _responseTargetsList_.empty() ) return;

  double cachedResponse{ this->evalResponse() };

  auto beginPtr = _responseTargetsList_.begin();
  auto endPtr = _responseTargetsList_.end();

  if( iThread_ != -1 ){
    Long64_t nEventPerThread = Long64_t(_responseTargetsList_.size())/GlobalVariables::getNbThreads();
    beginPtr = _responseTargetsList_.begin() + Long64_t(iThread_)*nEventPerThread;
    if( iThread_+1 != GlobalVariables::getNbThreads() ){
      endPtr = _responseTargetsList_.begin() + (Long64_t(iThread_) + 1) * nEventPerThread;
    }
  }

  std::for_each(beginPtr, endPtr, [&](PhysicsEvent* event_){
    // thread safe:
    event_->multiplyEventWeight(cachedResponse );
  });

}

