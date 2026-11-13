//
// Created by Adrien Blanchet on 29/11/2022.
//

#include "DialResponseSupervisor.h"

#include "GundamUtils.h"

#include "Logger.h"

#include <sstream>


void DialResponseSupervisor::setSmoothingWidths(double minWidth_, double maxWidth_){
  auto checkWidth = [](double width_){
    LogThrowIf(not std::isfinite(width_) or width_ <= 0.,
               "dialResponseSmoothingWidth must be finite and strictly positive. "
               "Set enableResponseSmoothing to false to disable smoothing.");
  };
  if( std::isfinite(_minResponse_) ){ checkWidth(minWidth_); }
  if( std::isfinite(_maxResponse_) ){ checkWidth(maxWidth_); }
  LogThrowIf(std::isfinite(_minResponse_) and std::isfinite(_maxResponse_)
             and minWidth_ > (_maxResponse_ - _minResponse_) - maxWidth_,
             "dialResponseSmoothingWidth: lower and upper transitions overlap");
  _minSmoothingWidth_ = std::isfinite(_minResponse_) ? minWidth_ : 0.;
  _maxSmoothingWidth_ = std::isfinite(_maxResponse_) ? maxWidth_ : 0.;
  updateTransitions();
  LogThrowIf((_minSmoothingWidth_ > 0.
              and (not std::isfinite(_minTransitionStart_) or not std::isfinite(_minResponse_ - _minSmoothingWidth_)
                   or not std::isfinite(_minInverseTransitionWidth_) or _minTransitionStart_ <= _minResponse_))
             or (_maxSmoothingWidth_ > 0.
                 and (not std::isfinite(_maxTransitionStart_) or not std::isfinite(_maxResponse_ + _maxSmoothingWidth_)
                      or not std::isfinite(_maxInverseTransitionWidth_) or _maxTransitionStart_ >= _maxResponse_)),
             "dialResponseSmoothingWidth cannot be represented at the configured response limits");
}

void DialResponseSupervisor::updateTransitions(){
  _minTransitionStart_ = _minResponse_ + _minSmoothingWidth_;
  _maxTransitionStart_ = _maxResponse_ - _maxSmoothingWidth_;
  _minInverseTransitionWidth_ = _minSmoothingWidth_ > 0. ? 0.5 / _minSmoothingWidth_ : 0.;
  _maxInverseTransitionWidth_ = _maxSmoothingWidth_ > 0. ? 0.5 / _maxSmoothingWidth_ : 0.;
}

double DialResponseSupervisor::process(double response_) const {
  if( not std::isnan(_minTransitionStart_) and response_ < _minTransitionStart_ ) GUNDAM_UNLIKELY_COMPILER_FLAG {
    if( response_ <= _minResponse_ - _minSmoothingWidth_ ){ return _minResponse_; }
    const double t = 0.5 + (response_ - _minResponse_) * _minInverseTransitionWidth_;
    return _minResponse_ + _minSmoothingWidth_ * (t * t * t * (2. - t));
  }
  else if( not std::isnan(_maxTransitionStart_) and response_ > _maxTransitionStart_ ) GUNDAM_UNLIKELY_COMPILER_FLAG {
    if( response_ >= _maxResponse_ + _maxSmoothingWidth_ ){ return _maxResponse_; }
    const double t = 0.5 + (_maxResponse_ - response_) * _maxInverseTransitionWidth_;
    return _maxResponse_ - _maxSmoothingWidth_ * (t * t * t * (2. - t));
  }
  return response_;
}
std::string DialResponseSupervisor::getSummary(bool shallow_ ) const{
  std::stringstream ss;

  if(not std::isnan(_minResponse_)){
    ss << "minResponse=" << _minResponse_;
  }

  if(not std::isnan(_maxResponse_)){
    if( not ss.str().empty() ) ss << ", ";
    ss << "maxResponse=" << _maxResponse_;
  }

  if( _minSmoothingWidth_ > 0. or _maxSmoothingWidth_ > 0. ){
    ss << ", smoothingWidths=[" << _minSmoothingWidth_ << ", " << _maxSmoothingWidth_ << "]";
  }

  return ss.str();
}
