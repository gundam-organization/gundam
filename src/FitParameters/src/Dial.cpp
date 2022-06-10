//
// Created by Nadrino on 21/05/2021.
//

#include "Dial.h"
#include "DialSet.h"
#include "FitParameterSet.h"

#include "Logger.h"

#include "sstream"

LoggerInit([]{
  Logger::setUserHeaderStr("[Dial]");
});



bool Dial::enableMaskCheck{false};
bool Dial::disableDialCache{false};
bool Dial::throwIfResponseIsNegative{true};

Dial::Dial(DialType::DialType dialType_) : _dialType_{dialType_} {}
Dial::~Dial() = default;

void Dial::reset() {
  _dialResponseCache_ = std::nan("Unset");
  _dialParameterCache_ = std::nan("Unset");
  _applyConditionBin_ = nullptr;
}

void Dial::setApplyConditionBin(DataBin *applyConditionBin) {
  _applyConditionBin_ = applyConditionBin;
}
void Dial::setIsReferenced(bool isReferenced) {
  _isReferenced_ = isReferenced;
}
void Dial::setOwner(const DialSet* dialSetPtr) {
  _owner_ = dialSetPtr;
}

void Dial::initialize() {
  LogThrowIf( _dialType_ == DialType::Invalid, "_dialType_ is not set." )
  LogThrowIf(_owner_ == nullptr, "Owner not set.")
}

bool Dial::isReferenced() const {
  return _isReferenced_;
}
bool Dial::isMasked() const{
  return (_owner_->getOwner()->getOwner()->isMaskedForPropagation());
}
double Dial::getDialResponseCache() const{
  return _dialResponseCache_;
}
double Dial::getAssociatedParameter() const {
  return _owner_->getOwner()->getParameterValue();
}
const DialSet* Dial::getOwner() const {
  LogThrowIf(!_owner_, "Invalid owning DialSet")
  return _owner_;
}
const DataBin* Dial::getApplyConditionBinPtr() const{ return _applyConditionBin_; }

DataBin* Dial::getApplyConditionBinPtr(){ return _applyConditionBin_; }
DialType::DialType Dial::getDialType() const {
  return _dialType_;
}

double Dial::getEffectiveDialParameter(double parameterValue_){
  if( _owner_->useMirrorDial() ){
    parameterValue_ = std::abs(std::fmod(
        parameterValue_ - _owner_->getMirrorLowEdge(),
        2 * _owner_->getMirrorRange()
    ));

    if(parameterValue_ > _owner_->getMirrorRange() ){
      // odd pattern  -> mirrored -> decreasing effective X while increasing parameter
      parameterValue_ -= 2 * _owner_->getMirrorRange();
      parameterValue_ = -parameterValue_;
    }

    // re-apply the offset
    parameterValue_ += _owner_->getMirrorLowEdge();
  }
  return parameterValue_;
}
double Dial::capDialResponse(double response_){
  // Cap checks
  if     (_owner_->getMinDialResponse() == _owner_->getMinDialResponse() and response_ < _owner_->getMinDialResponse() ){ response_=_owner_->getMinDialResponse(); }
  else if(_owner_->getMaxDialResponse() == _owner_->getMaxDialResponse() and response_ > _owner_->getMaxDialResponse() ){ response_=_owner_->getMaxDialResponse(); }

  LogThrowIf( response_ != response_, "NaN response returned:" << std::endl << this->getSummary());
  if( Dial::throwIfResponseIsNegative and response_ < 0 ){
    this->writeSpline("");
    LogError << this->getSummary() << std::endl;
    LogThrow("Negative response.");
  }

  return response_;
}
double Dial::evalResponse(){
  return this->evalResponse( _owner_->getOwner()->getParameterValue() );
}

// Virtual
double Dial::evalResponse(double parameterValue_) {
  if( Dial::disableDialCache ){
    return this->capDialResponse(this->calcDial(this->getEffectiveDialParameter(parameterValue_)));
  }

  // Check if all is already up-to-date
  if( _dialParameterCache_ == parameterValue_ ){ return _dialResponseCache_; }

  // If we reach this point, we either need to compute the response or wait for another thread to make the update.
#if __cplusplus >= 201703L // https://stackoverflow.com/questions/26089319/is-there-a-standard-definition-for-cplusplus-in-c14
  std::scoped_lock<std::mutex> g(_evalDialLock_); // There can be only one.
#else
  std::lock_guard<std::mutex> g(_evalDialLock_); // There can be only one.
#endif
  if( _dialParameterCache_ == parameterValue_ ) return _dialResponseCache_; // stop if already updated by another threads

  // Edit the cache
  _dialResponseCache_ = this->capDialResponse(this->calcDial(this->getEffectiveDialParameter(parameterValue_)));
  _dialParameterCache_ = parameterValue_;

  return _dialResponseCache_;
}
std::string Dial::getSummary(){
  std::stringstream ss;
  ss << _owner_->getOwner()->getOwner()->getName(); // parSet name
  ss << "/" << _owner_->getOwner()->getTitle();
  ss << "(" << _owner_->getOwner()->getParameterValue() << ")";
  ss << "/";
  ss << DialType::DialTypeEnumNamespace::toString(_dialType_, true);
  if( _applyConditionBin_ != nullptr and not _applyConditionBin_->getEdgesList().empty() ) ss << ":b{" << _applyConditionBin_->getSummary() << "}";
  ss << " dial(" << _dialParameterCache_ << ") = " << _dialResponseCache_;
  return ss.str();
}

//void Dial::copySplineCache(TSpline3& splineBuffer_){
//  if( _responseSplineCache_ == nullptr ) this->buildResponseSplineCache();
//  splineBuffer_ = *_responseSplineCache_;
//}
//void Dial::buildResponseSplineCache(){
//  // overridable
//  double xmin = static_cast<FitParameter *>(_associatedParameterReference_)->getMinValue();
//  double xmax = static_cast<FitParameter *>(_associatedParameterReference_)->getMaxValue();
//  double prior = static_cast<FitParameter *>(_associatedParameterReference_)->getPriorValue();
//  double sigma = static_cast<FitParameter *>(_associatedParameterReference_)->getStdDevValue();
//
//  std::vector<double> xSigmaSteps = {-5, -3, -2, -1, -0.5,  0,  0.5,  1,  2,  3,  5};
//  for( size_t iStep = xSigmaSteps.size() ; iStep > 0 ; iStep-- ){
//    if( xmin == xmin and (prior + xSigmaSteps[iStep-1]*sigma) < xmin ){
//      xSigmaSteps.erase(xSigmaSteps.begin() + int(iStep)-1);
//    }
//    if( xmax == xmax and (prior + xSigmaSteps[iStep-1]*sigma) > xmax ){
//      xSigmaSteps.erase(xSigmaSteps.begin() + int(iStep)-1);
//    }
//  }
//
//  std::vector<double> yResponse(xSigmaSteps.size(), 0);
//
//  for( size_t iStep = 0 ; iStep < xSigmaSteps.size() ; iStep++ ){
//    yResponse[iStep] = this->evalResponse(prior + xSigmaSteps[iStep]*sigma);
//  }
//  _responseSplineCache_ = std::shared_ptr<TSpline3>(
//      new TSpline3(Form("%p", this), &xSigmaSteps[0], &yResponse[0], int(xSigmaSteps.size()))
//  );
//}



