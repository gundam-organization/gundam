//
// Created by Nadrino on 26/05/2021.
//

#include "Logger.h"

#include "FitParameter.h"
#include "NormalizationDial.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[NormalizationDial]");
});

NormalizationDial::NormalizationDial() : Dial(DialType::Normalization) {
  this->NormalizationDial::reset();
}

void NormalizationDial::fillResponseCache() {
  // Normalization dial: y = x
  // Don't use _effectiveDialParameterValue_ since it doesn't make sense
  _dialResponseCache_ = _dialParameterCache_;
}

void NormalizationDial::reset() {
  Dial::reset();
}

void NormalizationDial::initialize() {
  Dial::initialize();
}

double NormalizationDial::evalResponse(double parameterValue_){
  return parameterValue_; // NO CACHE NEEDED ?
}

