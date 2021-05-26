//
// Created by Adrien BLANCHET on 21/05/2021.
//

#include "sstream"

#include "Logger.h"

#include "Dial.h"

Dial::Dial() {
  Logger::setUserHeaderStr("[Dial]");
  this->reset();
}
Dial::~Dial() {
  this->reset();
}

void Dial::reset() {
  _dialResponseCache_ = std::nan("Unset");
  _dialParameterCache_ = std::nan("Unset");
  _applyConditionBin_ = DataBin();
  _dialType_   = DialType::Invalid;
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
  ss << DialType::DialTypeEnumNamespace::toString(_dialType_) << ": " << _applyConditionBin_.generateSummary();
  return ss.str();
}
double Dial::evalResponse(const double &parameterValue_) {
  if( _dialParameterCache_ == parameterValue_ ){
    return _dialResponseCache_;
  }

  _dialParameterCache_ = parameterValue_;
  this->updateResponseCache(parameterValue_); // specified in the corresponding dial class

  return _dialResponseCache_;
}

