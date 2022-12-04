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
  double response = _dialBaseRef_->evalResponse( *_inputBufferRef_ );

  LogThrowIf(
      std::isnan(response),
      "invalid dial response: "
      << GET_VAR_NAME_VALUE(_inputBufferRef_->getBufferSize())
      << GET_VAR_NAME_VALUE(_inputBufferRef_->getBuffer()[0])
      << GET_VAR_NAME_VALUE(_inputBufferRef_->getSummary())
      );

  if(_responseSupervisorRef_ != nullptr ){
    _responseSupervisorRef_->process(response);
  }

  return response;
}
void DialInterface::addTarget(const std::pair<size_t, size_t>& sampleEventIndices_){
  // lock -> only one at a time pass this point
#if __cplusplus >= 201703L
  // https://stackoverflow.com/questions/26089319/is-there-a-standard-definition-for-cplusplus-in-c14
  std::scoped_lock<std::mutex> g(_mutex_);
#else
  std::lock_guard<std::mutex> g(_mutex_);
#endif
  _targetSampleEventIndicesList_.emplace_back(sampleEventIndices_);
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
