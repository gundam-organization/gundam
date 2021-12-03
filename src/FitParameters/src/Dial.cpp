//
// Created by Nadrino on 21/05/2021.
//

#include "sstream"

#include "Logger.h"

#include "Dial.h"
#include "FitParameter.h"
#include "FitParameterSet.h"
#include "GlobalVariables.h"



LoggerInit([](){
  Logger::setUserHeaderStr("[Dial]");
  Logger::setMaxLogLevel(LogDebug);
} )



DialType::DialType DialType::toDialType(const std::string& dialStr_){
  int enumIndex = DialTypeEnumNamespace::toEnumInt("DialType::" + dialStr_);
  if( enumIndex == DialTypeEnumNamespace::enumOffSet - 1 ){
    LogError << "\"" << dialStr_ << "\" unrecognized  dial type. " << std::endl;
    LogError << "Expecting: { " << DialTypeEnumNamespace::enumNamesAgregate << " }" << std::endl;
    throw std::runtime_error("Unrecognized  dial type.");
  }
  return static_cast<DialType>(enumIndex);
}

Dial::Dial() {
  this->reset();
}
Dial::~Dial() {
  this->reset();
}

void Dial::reset() {
  _isInitialized_ = false;
  _dialResponseCache_ = std::nan("Unset");
  _dialParameterCache_ = std::nan("Unset");
  _applyConditionBin_ = DataBin();
  _dialType_ = DialType::Invalid;
  _associatedParameterReference_ = nullptr;
//  dialMutex = std::make_unique<std::mutex>();
}

void Dial::setApplyConditionBin(const DataBin &applyConditionBin) {
  _applyConditionBin_ = applyConditionBin;
}
void Dial::setAssociatedParameterReference(void *associatedParameterReference) {
  _associatedParameterReference_ = associatedParameterReference;
}
void Dial::setIsReferenced(bool isReferenced) {
  _isReferenced_ = isReferenced;
}

void Dial::initialize() {
  LogThrowIf( _dialType_ == DialType::Invalid, "_dialType_ is not set." );
  LogThrowIf( _associatedParameterReference_ == nullptr,  "Parameter not set");
}

bool Dial::isInitialized() const {
  return _isInitialized_;
}
std::string Dial::getSummary(){
  std::stringstream ss;
  if( _associatedParameterReference_ != nullptr ){
    ss << ((FitParameterSet*)((FitParameter*) _associatedParameterReference_)->getParSetRef())->getName();
    ss << "/" << ((FitParameter*) _associatedParameterReference_)->getTitle();
    ss << "(" << ((FitParameter*) _associatedParameterReference_)->getParameterValue() << ")";
    ss << "/";
  }
  ss << DialType::DialTypeEnumNamespace::toString(_dialType_, true);
  if( not _applyConditionBin_.getEdgesList().empty() ) ss << ":b{" << _applyConditionBin_.getSummary() << "}";
  return ss.str();
}
bool Dial::isReferenced() const {
  return _isReferenced_;
}

double Dial::evalResponse(double parameterValue_) {

  if( _dialParameterCache_ == parameterValue_ ){
    while( _isEditingCache_.atomicValue ){ }
    return _dialResponseCache_;
  }
  _isEditingCache_.atomicValue = true;
  _dialParameterCache_ = parameterValue_;
  this->fillResponseCache(); // specified in the corresponding dial class
  _isEditingCache_.atomicValue = false;

  return _dialResponseCache_;
}
double Dial::evalResponse(){
  if( _associatedParameterReference_ == nullptr ){
    LogError << "_associatedParameterReference_ is not set." << std::endl;
    throw std::logic_error("_associatedParameterReference_ is not set.");
  }
  return this->evalResponse( static_cast<FitParameter *>(_associatedParameterReference_)->getParameterValue() );
}

void Dial::copySplineCache(TSpline3& splineBuffer_){
  if( _responseSplineCache_ == nullptr ) this->buildResponseSplineCache();
  splineBuffer_ = *_responseSplineCache_;
}
void Dial::buildResponseSplineCache(){
  // overridable
  double xmin = static_cast<FitParameter *>(_associatedParameterReference_)->getMinValue();
  double xmax = static_cast<FitParameter *>(_associatedParameterReference_)->getMaxValue();
  double prior = static_cast<FitParameter *>(_associatedParameterReference_)->getPriorValue();
  double sigma = static_cast<FitParameter *>(_associatedParameterReference_)->getStdDevValue();

  std::vector<double> xSigmaSteps = {-5, -3, -2, -1, -0.5,  0,  0.5,  1,  2,  3,  5};
  for( size_t iStep = xSigmaSteps.size() ; iStep > 0 ; iStep-- ){
    if( xmin == xmin and (prior + xSigmaSteps[iStep-1]*sigma) < xmin ){
      xSigmaSteps.erase(xSigmaSteps.begin() + int(iStep)-1);
    }
    if( xmax == xmax and (prior + xSigmaSteps[iStep-1]*sigma) > xmax ){
      xSigmaSteps.erase(xSigmaSteps.begin() + int(iStep)-1);
    }
  }

  std::vector<double> yResponse(xSigmaSteps.size(), 0);

  for( size_t iStep = 0 ; iStep < xSigmaSteps.size() ; iStep++ ){
    yResponse[iStep] = this->evalResponse(prior + xSigmaSteps[iStep]*sigma);
  }
  _responseSplineCache_ = std::shared_ptr<TSpline3>(
      new TSpline3(Form("%p", this), &xSigmaSteps[0], &yResponse[0], int(xSigmaSteps.size()))
  );
}

double Dial::getDialResponseCache() const{
  return _dialResponseCache_;
}
const DataBin &Dial::getApplyConditionBin() const {
  return _applyConditionBin_;
}
DataBin &Dial::getApplyConditionBin() {
  return _applyConditionBin_;
}
DialType::DialType Dial::getDialType() const {
  return _dialType_;
}
void *Dial::getAssociatedParameterReference() const {
  return _associatedParameterReference_;
}


