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
  this->NormalizationDial::reset();
}

void NormalizationDial::fillResponseCache() {
  // Normalization dial: y = x
  // Don't use _effectiveDialParameterValue_ since it doesn't make sense
  _dialResponseCache_ = _dialParameterCache_;
}

void NormalizationDial::reset() {
  Dial::reset();
  _dialType_ = DialType::Normalization;
}

void NormalizationDial::initialize() {
  Dial::initialize();
  LogThrowIf(
      _dialType_!=DialType::Normalization,
      "_dialType_ is not Normalization: " << DialType::DialTypeEnumNamespace::toString(_dialType_)
      )
  _isInitialized_ = true;
}

double NormalizationDial::evalResponse(double parameterValue_){
  return parameterValue_; // NO CACHE NEEDED ?
}

