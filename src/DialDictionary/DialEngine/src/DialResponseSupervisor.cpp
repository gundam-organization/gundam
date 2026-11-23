//
// Created by Adrien Blanchet on 29/11/2022.
//

/// @file DialResponseSupervisor.cpp
/// @brief Implements DialResponseSupervisor: clamping a dial response and
///        describing the bounds.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in
/// DialResponseSupervisor.h. The comments here describe how each method does
/// its job.

#include "DialResponseSupervisor.h"

#include <sstream>


/// @par Implementation
/// The lower bound is tested first. Each test also requires the bound to be
/// set, since any comparison with NaN is false; for the same reason a NaN
/// response passes both tests and is returned as is.
double DialResponseSupervisor::process(double reponse_) const {
  // apply cap?
  if     ( not std::isnan(_minResponse_) and reponse_ < _minResponse_ ){ return _minResponse_; }
  else if( not std::isnan(_maxResponse_) and reponse_ > _maxResponse_ ){ return _maxResponse_; }

  // else?

  return reponse_;
}
/// @par Implementation
/// The separator is only written when the minimum was printed before.
std::string DialResponseSupervisor::getSummary(bool shallow_ ) const{
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
