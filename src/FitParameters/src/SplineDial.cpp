//
// Created by Nadrino on 26/05/2021.
//

#include "TFile.h"

#include <FitParameter.h>
#include "SplineDial.h"

#include "Logger.h"

LoggerInit([](){ Logger::setUserHeaderStr("[SplineDial]"); } )


SplineDial::SplineDial() {
  SplineDial::reset();
}

void SplineDial::reset() {
  Dial::reset();
  _dialType_ = DialType::Spline;
  _spline_ = TSpline3();
//  _splinePtr_ = nullptr;
}

void SplineDial::initialize() {
  Dial::initialize();
//  if( _splinePtr_ == nullptr ){
  LogThrowIf(_spline_.GetXmin() == _spline_.GetXmax(), "Spline is not valid.")

  // check if prior is out of bounds:
  if( _associatedParameterReference_ != nullptr ){
    if(
//        static_cast<FitParameter*>(_associatedParameterReference_)->getPriorValue() < _splinePtr_->GetXmin()
        static_cast<FitParameter*>(_associatedParameterReference_)->getPriorValue() < _spline_.GetXmin()
//      or static_cast<FitParameter*>(_associatedParameterReference_)->getPriorValue() > _splinePtr_->GetXmax()
      or static_cast<FitParameter*>(_associatedParameterReference_)->getPriorValue() > _spline_.GetXmax()
    ){
      LogError << "Prior value of parameter \""
      << static_cast<FitParameter*>(_associatedParameterReference_)->getTitle()
      << "\" = " << static_cast<FitParameter*>(_associatedParameterReference_)->getPriorValue()
//      << " is out of the spline bounds: " <<  _splinePtr_->GetXmin() << " < X < " << _splinePtr_->GetXmax()
      << " is out of the spline bounds: " <<  _spline_.GetXmin() << " < X < " << _spline_.GetXmax()
      << std::endl;
      throw std::logic_error("Prior is out of the spline bounds.");
    }

    _dialParameterCache_ = static_cast<FitParameter*>(_associatedParameterReference_)->getPriorValue();
    try{ fillResponseCache(); }
    catch(...){
      LogError << "Error while evaluating spline response at the prior value: d(" << _dialParameterCache_ << ") = " << _dialResponseCache_ << std::endl;
      throw std::logic_error("error eval spline response");
    }
  }
  _isInitialized_ = true;
}

std::string SplineDial::getSummary() {
  std::stringstream ss;
  ss << Dial::getSummary();
//  ss << ": spline(" << _splinePtr_ << ")";
  return ss.str();
}
void SplineDial::fillResponseCache() {
  LogThrowIf(_spline_.GetXmin() == _spline_.GetXmax(), "Spline is not valid.")

//  if     ( _dialParameterCache_ < _splinePtr_->GetXmin() ) _dialResponseCache_ = _splinePtr_->Eval(_splinePtr_->GetXmin());
  if     ( _dialParameterCache_ < _spline_.GetXmin() ) _dialResponseCache_ = _spline_.Eval(_spline_.GetXmin());
//  else if( _dialParameterCache_ > _splinePtr_->GetXmax() ) _dialResponseCache_ = _splinePtr_->Eval(_splinePtr_->GetXmax());
  else if( _dialParameterCache_ > _spline_.GetXmax() ) _dialResponseCache_ = _spline_.Eval(_spline_.GetXmax());
//  else   { _dialResponseCache_ = _splinePtr_->Eval(_dialParameterCache_); }
  else   { _dialResponseCache_ = _spline_.Eval(_dialParameterCache_); }

  // Checks
  if( _minimumSplineResponse_ == _minimumSplineResponse_ and _dialResponseCache_ < _minimumSplineResponse_ ){
    _dialResponseCache_ = _minimumSplineResponse_;
  }

  if( _throwIfResponseIsNegative_ and _dialResponseCache_ < 0 ){

    this->writeSpline();

    LogThrow(
      "Negative dial response: dial(" << _dialParameterCache_ << ") = " << _dialResponseCache_
//      << std::endl << "Dial is defined in between: [" << _splinePtr_->GetXmin() << ", " << _splinePtr_->GetXmax() << "]" << std::endl
      << std::endl << "Dial is defined in between: [" << _spline_.GetXmin() << ", " << _spline_.GetXmax() << "]" << std::endl
      << ( _associatedParameterReference_ != nullptr ? "Parameter: " + static_cast<FitParameter *>(_associatedParameterReference_)->getName() : "" )
      )
  }
}

void SplineDial::copySpline(const TSpline3* splinePtr_){
//  LogThrowIf(_splinePtr_ != nullptr, "Spline already set")
//  _splinePtr_ = std::make_shared<TSpline3>(*splinePtr_);
  LogThrowIf(_spline_.GetXmin() != _spline_.GetXmax(), "Spline already set")
  _spline_ = *splinePtr_;
}
void SplineDial::createSpline(TGraph* grPtr_){
//  LogThrowIf(_splinePtr_ != nullptr, "Spline already set")
//  _splinePtr_ = std::make_shared<TSpline3>(TSpline3(grPtr_->GetName(), grPtr_));
  LogThrowIf(_spline_.GetXmin() != _spline_.GetXmax(), "Spline already set")
  _spline_ = TSpline3(grPtr_->GetName(), grPtr_);
}
void SplineDial::setMinimumSplineResponse(double minimumSplineResponse) {
  _minimumSplineResponse_ = minimumSplineResponse;
}

const TSpline3* SplineDial::getSplinePtr() const {
//  return _splinePtr_.get();
  return &_spline_;
}

void SplineDial::writeSpline(const std::string &fileName_) const{

  TFile* f;
  if(fileName_.empty()) f = TFile::Open(Form("badDial_%x.root", this), "RECREATE");
  else                  f = TFile::Open(fileName_.c_str(), "RECREATE");

//  f->WriteObject(_splinePtr_.get(), _splinePtr_->GetName());
  f->WriteObject(&_spline_, _spline_.GetName());
  f->Close();
}

