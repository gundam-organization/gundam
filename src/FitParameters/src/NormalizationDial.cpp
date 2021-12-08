//
// Created by Nadrino on 26/05/2021.
//

#include "Logger.h"

#include "FitParameter.h"
#include "NormalizationDial.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[NormalizationDial]");
})

NormalizationDial::NormalizationDial() {
  this->reset();
}

void NormalizationDial::fillResponseCache() {
  // Normalization dial: y = x
  _dialResponseCache_ = _dialParameterCache_;
}

void NormalizationDial::reset() {
  Dial::reset();
  _dialType_ = DialType::Normalization;
}

void NormalizationDial::initialize() {
  Dial::initialize();
  LogThrowIf(_associatedParameterReference_ == nullptr, "Par reference not set.")
  _priorValue_ = ( (FitParameter*) _associatedParameterReference_ )->getPriorValue();
  _isInitialized_ = true;
}

double NormalizationDial::evalResponse(double parameterValue_){
  return parameterValue_ * _priorValue_; // NO CACHE NEEDED ?
}

