//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialResponseSupervisor.h"


void DialResponseSupervisor::setMinResponse(double minResponse) {
  _minResponse_ = minResponse;
}
void DialResponseSupervisor::setMaxResponse(double maxResponse) {
  _maxResponse_ = maxResponse;
}

void DialResponseSupervisor::process(double& output_) const {
  // apply cap?
  if( not std::isnan(_minResponse_) and output_ < _minResponse_ ){ output_ = _minResponse_; }
  else if( not std::isnan(_maxResponse_) and output_ > _maxResponse_ ){ output_ = _maxResponse_; }
}
