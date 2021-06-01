//
// Created by Adrien BLANCHET on 26/05/2021.
//

#include "Logger.h"

#include "NormalizationDial.h"

NormalizationDial::NormalizationDial() {
  this->reset();
}

void NormalizationDial::updateResponseCache(const double &parameterValue_) {
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


