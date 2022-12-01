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
void DialInterface::setInputBufferRef(const DialInputBuffer *inputBufferRef) {
  _inputBufferRef_ = inputBufferRef;
}
void DialInterface::setResponseSupervisorRef(const DialResponseSupervisor *responseSupervisorRef) {
  _responseSupervisorRef_ = responseSupervisorRef;
}

std::vector<PhysicsEvent *> &DialInterface::getTargetEventList() {
  return _targetEventList_;
}

double DialInterface::evalResponse(){
  double response = _dialBaseRef_->evalResponse(*_inputBufferRef_ );

  if(_responseSupervisorRef_ != nullptr ){
    _responseSupervisorRef_->process(response);
  }

  return response;
}
void DialInterface::propagateToTargets(int iThread_){
  if( _targetEventList_.empty() ) return;

  double cachedResponse{ this->evalResponse() };

  auto beginPtr = _targetEventList_.begin();
  auto endPtr = _targetEventList_.end();

  if( iThread_ != -1 ){
    Long64_t nEventPerThread = Long64_t(_targetEventList_.size()) / GlobalVariables::getNbThreads();
    beginPtr = _targetEventList_.begin() + Long64_t(iThread_) * nEventPerThread;
    if( iThread_+1 != GlobalVariables::getNbThreads() ){
      endPtr = _targetEventList_.begin() + (Long64_t(iThread_) + 1) * nEventPerThread;
    }
  }

  std::for_each(beginPtr, endPtr, [&](PhysicsEvent* event_){
    // thread safe:
    LogDebug << GET_VAR_NAME_VALUE(event_->getSummary()) << std::endl;
    if( event_->getLeafContentList().empty() ) return;
    event_->multiplyEventWeight( cachedResponse );
  });

}
void DialInterface::addTargetEvent(PhysicsEvent* event_){
  // lock -> only one at a time pass this point
#if __cplusplus >= 201703L
  // https://stackoverflow.com/questions/26089319/is-there-a-standard-definition-for-cplusplus-in-c14
  std::scoped_lock<std::mutex> g(_mutex_);
#else
  std::lock_guard<std::mutex> g(_mutex_);
#endif
  _targetEventList_.emplace_back(event_);
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
