//
// Created by Adrien BLANCHET on 21/05/2021.
//

#include "Dial.h"

Dial::Dial() {
  this->reset();
}
Dial::~Dial() {
  this->reset();
}

void Dial::reset() {
  _lastEvalDial_ = std::nan("Unset");
  _lastEvalParameter_ = std::nan("Unset");
}

double Dial::evalDial(const double &parameterValue_) {

  if( _lastEvalParameter_ != _lastEvalParameter_ or _lastEvalParameter_ != parameterValue_ ){
    _lastEvalParameter_ = parameterValue_;
    _lastEvalDial_ = 0; // update the cache
  }

  return _lastEvalDial_;
}
