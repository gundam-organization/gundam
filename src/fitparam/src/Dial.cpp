//
// Created by Nadrino on 21/05/2021.
//

#include "sstream"

#include "Logger.h"

#include "Dial.h"


LoggerInit([](){
  Logger::setUserHeaderStr("[Dial]");
  Logger::setMaxLogLevel(LogDebug);
} )



DialType::DialType DialType::toDialType(const std::string& dialStr_){
  int enumIndex = DialTypeEnumNamespace::toEnumInt("DialType::" + dialStr_);
  if( enumIndex == DialTypeEnumNamespace::enumOffSet - 1 ){
    LogError << "\"" << dialStr_ << "\" unrecognized  dial type. " << std::endl;
    LogError << "Expecting: { " << DialTypeEnumNamespace::enumNamesAgregate << " }" << std::endl;
    throw std::runtime_error("Unrecognized  dial type.");
  }
  return static_cast<DialType>(enumIndex);
}

Dial::Dial() {
  this->reset();
}
Dial::~Dial() {
  this->reset();
}

void Dial::reset() {
  _dialResponseCache_ = std::nan("Unset");
  _dialParameterCache_ = std::nan("Unset");
  _applyConditionBin_ = DataBin();
  _dialType_ = DialType::Invalid;
  _mutexPtr_ = std::make_shared<std::mutex>();
}

void Dial::setApplyConditionBin(const DataBin &applyConditionBin) {
  _applyConditionBin_ = applyConditionBin;
}

void Dial::initialize() {
  if( _dialType_ == DialType::Invalid ){
    LogError << "_dialType_ is not set." << std::endl;
    throw std::logic_error("_dialType_ is not set.");
  }
}

std::string Dial::getSummary(){
  std::stringstream ss;
  ss << DialType::DialTypeEnumNamespace::toString(_dialType_);
  if( not _applyConditionBin_.getEdgesList().empty() ) ss << ": " << _applyConditionBin_.getSummary();
  return ss.str();
}
double Dial::evalResponse(const double &parameterValue_) {

  _mutexPtr_->lock(); // don't fetch the cache at the same time
  if( _dialParameterCache_ == parameterValue_ ){
    _mutexPtr_->unlock();
    return _dialResponseCache_;
  }

//  LogTrace << this << " < " << _dialParameterCache_ << " -> " << _dialResponseCache_ << std::endl;
  _dialParameterCache_ = parameterValue_;
  this->updateResponseCache(parameterValue_); // specified in the corresponding dial class
//  LogTrace << this << " > " << _dialParameterCache_ << " -> " << _dialResponseCache_ << std::endl;
  _mutexPtr_->unlock();

  return _dialResponseCache_;
}

const DataBin &Dial::getApplyConditionBin() const {
  return _applyConditionBin_;
}
DialType::DialType Dial::getDialType() const {
  return _dialType_;
}

