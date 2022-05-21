//
// Created by Nadrino on 21/05/2021.
//

#include "Dial.h"
#include "DialSet.h"
#include "FitParameter.h"
#include "FitParameterSet.h"
#include "GlobalVariables.h"

#include "Logger.h"

#include "sstream"

LoggerInit([](){
  Logger::setUserHeaderStr("[Dial]");
  Logger::setMaxLogLevel(LogDebug);
} )


Dial::Dial(DialType::DialType dialType_)
    : _dialType_{dialType_}, _evalDialLock_{std::make_shared<std::mutex>()} {}
Dial::~Dial() = default;

void Dial::reset() {
  _dialResponseCache_ = std::nan("Unset");
  _dialParameterCache_ = std::nan("Unset");
  _effectiveDialParameterValue_ = std::nan("Unset");
  _applyConditionBin_ = nullptr;
}

void Dial::setApplyConditionBin(const DataBin &applyConditionBin) {
  _applyConditionBin_ = std::make_shared<DataBin>(applyConditionBin);
}
void Dial::setIsReferenced(bool isReferenced) {
  _isReferenced_ = isReferenced;
}
void Dial::setOwner(const DialSet* dialSetPtr) {
  _ownerDialSet_ = dialSetPtr;
}
const DialSet* Dial::getOwner() const {
  LogThrowIf(!_ownerDialSet_, "Invalid owning DialSet");
  return _ownerDialSet_;
}
void Dial::initialize() {
  LogThrowIf( _dialType_ == DialType::Invalid, "_dialType_ is not set." );
  LogThrowIf(_ownerDialSet_ == nullptr, "Owner not set.")
}

bool Dial::isReferenced() const {
  return _isReferenced_;
}
double Dial::getDialResponseCache() const{
  return _dialResponseCache_;
}
const DataBin &Dial::getApplyConditionBin() const {
  LogThrowIf(_applyConditionBin_ == nullptr, "_applyConditionBin_ not set.")
  return *_applyConditionBin_;
}
DataBin &Dial::getApplyConditionBin() {
  LogThrowIf(_applyConditionBin_ == nullptr, "_applyConditionBin_ not set.")
  return *_applyConditionBin_;
}
DialType::DialType Dial::getDialType() const {
  return _dialType_;
}
double Dial::getAssociatedParameter() const {
  return _ownerDialSet_->getOwnerFitParameter()->getParameterValue();
}

void Dial::updateEffectiveDialParameter(){
  _effectiveDialParameterValue_ = _dialParameterCache_;
  if( _ownerDialSet_->useMirrorDial() ){
    _effectiveDialParameterValue_ = std::abs(std::fmod(
        _dialParameterCache_ - _ownerDialSet_->getMirrorLowEdge(),
        2 * _ownerDialSet_->getMirrorRange()
    ));

    if(_effectiveDialParameterValue_ > _ownerDialSet_->getMirrorRange() ){
      // odd pattern  -> mirrored -> decreasing effective X while increasing parameter
      _effectiveDialParameterValue_ -= 2 * _ownerDialSet_->getMirrorRange();
      _effectiveDialParameterValue_ = -_effectiveDialParameterValue_;
    }

    // re-apply the offset
    _effectiveDialParameterValue_ += _ownerDialSet_->getMirrorLowEdge();
  }
}
double Dial::evalResponse(){
  return this->evalResponse(_ownerDialSet_->getOwnerFitParameter()->getParameterValue() );
}
//void Dial::copySplineCache(TSpline3& splineBuffer_){
//  if( _responseSplineCache_ == nullptr ) this->buildResponseSplineCache();
//  splineBuffer_ = *_responseSplineCache_;
//}

// Virtual
double Dial::evalResponse(double parameterValue_) {

  // Check if all is already up-to-date
  if( not _isEditingCache_ and _dialParameterCache_ == parameterValue_ ){
    return _dialResponseCache_;
  }

  // If we reach this point, we either need to compute the response or wait for another thread to make the update.
  std::lock_guard<std::mutex> g(*_evalDialLock_); // There can be only one.
  if( _dialParameterCache_ == parameterValue_ ) return _dialResponseCache_; // stop if already updated by another threads

  // Edit the cache
  _dialParameterCache_ = parameterValue_;
  this->updateEffectiveDialParameter();
  this->fillResponseCache(); // specified in the corresponding dial class
  if     (_ownerDialSet_->getMinDialResponse() == _ownerDialSet_->getMinDialResponse() and _dialResponseCache_ < _ownerDialSet_->getMinDialResponse() ){ _dialResponseCache_=_ownerDialSet_->getMinDialResponse(); }
  else if(_ownerDialSet_->getMaxDialResponse() == _ownerDialSet_->getMaxDialResponse() and _dialResponseCache_ > _ownerDialSet_->getMaxDialResponse() ){ _dialResponseCache_=_ownerDialSet_->getMaxDialResponse(); }

  return _dialResponseCache_;
}
std::string Dial::getSummary(){
  std::stringstream ss;
  ss << ((FitParameterSet*) _ownerDialSet_->getOwnerFitParameter()->getParSetRef())->getName();
  ss << "/" << _ownerDialSet_->getOwnerFitParameter()->getTitle();
  ss << "(" << _ownerDialSet_->getOwnerFitParameter()->getParameterValue() << ")";
  ss << "/";
  ss << DialType::DialTypeEnumNamespace::toString(_dialType_, true);
  if( _applyConditionBin_ != nullptr and not _applyConditionBin_->getEdgesList().empty() ) ss << ":b{" << _applyConditionBin_->getSummary() << "}";
  return ss.str();
}
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
