//
// Created by Nadrino on 26/05/2021.
//

#include "Logger.h"

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
}

double NormalizationDial::evalResponse(const double& parameterValue_){
  return parameterValue_; // NO CACHE NEEDED :)
}

