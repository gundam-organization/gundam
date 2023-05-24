//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialResponseSupervisor.h"

#include <sstream>


void DialResponseSupervisor::setMinResponse(double minResponse) {
  _minResponse_ = minResponse;
}
void DialResponseSupervisor::setMaxResponse(double maxResponse) {
  _maxResponse_ = maxResponse;
}

double DialResponseSupervisor::process(double reponse_) const {
  // apply cap?
  if     ( not std::isnan(_minResponse_) and reponse_ < _minResponse_ ){ return _minResponse_; }
  else if( not std::isnan(_maxResponse_) and reponse_ > _maxResponse_ ){ return _maxResponse_; }

  // else?

  return reponse_;
}

std::string DialResponseSupervisor::getSummary() const{
  std::stringstream ss;

  if(not std::isnan(_minResponse_)){
    ss << "minResponse=" << _minResponse_;
  }

  if(not std::isnan(_maxResponse_)){
    if( not ss.str().empty() ) ss << ", ";
    ss << "maxResponse=" << _maxResponse_;
  }

  return ss.str();
}