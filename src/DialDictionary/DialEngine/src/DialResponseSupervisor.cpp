//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialResponseSupervisor.h"
#include "Logger.h"

#include <sstream>


double DialResponseSupervisor::process(double reponse_) const {
  // apply cap?
  // print out info if out of range
  if     ( not std::isnan(_minResponse_) and reponse_ < _minResponse_ ){
      LogWarning << "Response " << reponse_ << " is below the minimum response " << _minResponse_ << std::endl;
      return _minResponse_;
  }
  else if( not std::isnan(_maxResponse_) and reponse_ > _maxResponse_ ){
      LogWarning << "Response " << reponse_ << " is above the maximum response " << _maxResponse_ << std::endl;
      return _maxResponse_;
  }

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
