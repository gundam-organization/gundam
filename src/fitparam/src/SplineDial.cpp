//
// Created by Nadrino on 26/05/2021.
//

#include <FitParameter.h>
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

  // check if prior is out of bounds:
  if( _associatedParameterReference_ != nullptr ){
    if(  static_cast<FitParameter*>(_associatedParameterReference_)->getPriorValue() < _splinePtr_->GetXmin()
      or static_cast<FitParameter*>(_associatedParameterReference_)->getPriorValue() > _splinePtr_->GetXmax()
    ){
      LogError << "Prior value of parameter \""
      << static_cast<FitParameter*>(_associatedParameterReference_)->getTitle()
      << "\" = " << static_cast<FitParameter*>(_associatedParameterReference_)->getPriorValue()
      << " is out of the spline bounds: " <<  _splinePtr_->GetXmin() << " < X < " << _splinePtr_->GetXmax()
      << std::endl;
      throw std::logic_error("Prior is out of the spline bounds.");
    }

    _dialParameterCache_ = static_cast<FitParameter*>(_associatedParameterReference_)->getPriorValue();
    try{ fillResponseCache(); }
    catch(...){
      LogError << "Negative spline response evaluated at the prior value: " << _dialParameterCache_ << " -> " << _dialResponseCache_ << std::endl;
      throw std::logic_error("negative spline response");
    }
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
  if( _dialResponseCache_ <= 0 ){
    LogError << this << "(" << _dialParameterCache_ << ") = " << _dialResponseCache_ << std::endl;
    throw std::runtime_error("negative dial response");
  }
}

void SplineDial::setSplinePtr(TSpline3 *splinePtr) {
  _splinePtr_ = splinePtr;
}
