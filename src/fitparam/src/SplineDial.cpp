//
// Created by Nadrino on 26/05/2021.
//

#include "SplineDial.h"

#include "Logger.h"

LoggerInit([](){ Logger::setUserHeaderStr("[SplineDial]"); } )

SplineDial::SplineDial() {
  this->reset();
}

void SplineDial::reset() {
  Dial::reset();
  _dialType_ = DialType::Spline;
  _splinePtr_ = nullptr;
}

void SplineDial::initialize() {
  Dial::initialize();
  if( _splinePtr_ == nullptr ){
    LogError << "_splinePtr_ is not set." << std::endl;
    throw std::logic_error("_splinePtr_ is not set.");
  }
}

std::string SplineDial::getSummary() {
  std::stringstream ss;
  ss << Dial::getSummary();
  ss << ": spline(" << _splinePtr_ << ")";
  return ss.str();
}
void SplineDial::fillResponseCache() {
  if     ( _dialParameterCache_ < _splinePtr_->GetXmin() ) _dialResponseCache_ = _splinePtr_->Eval(_splinePtr_->GetXmin());
  else if( _dialParameterCache_ > _splinePtr_->GetXmax() ) _dialResponseCache_ = _splinePtr_->Eval(_splinePtr_->GetXmax());
  else                                                     _dialResponseCache_ = _splinePtr_->Eval(_dialParameterCache_);
}

void SplineDial::setSplinePtr(TSpline3 *splinePtr) {
  _splinePtr_ = splinePtr;
}
