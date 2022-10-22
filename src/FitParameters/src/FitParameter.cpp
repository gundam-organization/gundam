//
// Created by Nadrino on 21/05/2021.
//

#include "FitParameter.h"
#include "FitParameterSet.h"
#include "JsonUtils.h"

#include "Logger.h"

#include "sstream"


LoggerInit([]{ Logger::setUserHeaderStr("[FitParameter]"); });


FitParameter::FitParameter() {
  this->reset();
}
FitParameter::~FitParameter() {
  this->reset();
}

void FitParameter::reset() {
  _dialSetList_.clear();
  _dialDefinitionsList_ = nlohmann::json();
  _parameterIndex_ = -1;
  _parameterValue_ = std::numeric_limits<double>::quiet_NaN();
  _priorValue_     = std::numeric_limits<double>::quiet_NaN();
  _stdDevValue_    = std::numeric_limits<double>::quiet_NaN();
  _dialsWorkingDirectory_ = ".";
  _isEnabled_ = true;
  _isFixed_ = false;
  _owner_ = nullptr;
  _priorType_ = PriorType::Gaussian;
}

void FitParameter::setIsEnabled(bool isEnabled){
  _isEnabled_ = isEnabled;
}
void FitParameter::setIsFixed(bool isFixed) {
  _isFixed_ = isFixed;
}
void FitParameter::setIsEigen(bool isEigen) {
  _isEigen_ = isEigen;
}
void FitParameter::setIsFree(bool isFree) {
  _isFree_ = isFree;
}
void FitParameter::setDialSetConfig(const nlohmann::json &jsonConfig_) {
  auto jsonConfig = jsonConfig_;
  while( jsonConfig.is_string() ){
    LogWarning << "Forwarding FitParameterSet config to: \"" << jsonConfig.get<std::string>() << "\"..." << std::endl;
    jsonConfig = JsonUtils::readConfigFile(jsonConfig.get<std::string>());
  }
  _dialDefinitionsList_ = jsonConfig.get<std::vector<nlohmann::json>>();
}
void FitParameter::setParameterDefinitionConfig(const nlohmann::json &config_){
  _parameterConfig_ = config_;
  JsonUtils::forwardConfig(_parameterConfig_);
}
void FitParameter::setParameterIndex(int parameterIndex) {
  _parameterIndex_ = parameterIndex;
}
void FitParameter::setName(const std::string &name) {
  _name_ = name;
}
void FitParameter::setParameterValue(double parameterValue) {
  _parameterValue_ = parameterValue;
}
void FitParameter::setPriorValue(double priorValue) {
  _priorValue_ = priorValue;
}
void FitParameter::setThrowValue(double throwValue){
  _throwValue_ = throwValue;
}
void FitParameter::setStdDevValue(double stdDevValue) {
  _stdDevValue_ = stdDevValue;
}
void FitParameter::setEnableDialSetsSummary(bool enableDialSetsSummary) {
  _enableDialSetsSummary_ = enableDialSetsSummary;
}
void FitParameter::setMinValue(double minValue) {
  _minValue_ = minValue;
}
void FitParameter::setMaxValue(double maxValue) {
  _maxValue_ = maxValue;
}
void FitParameter::setStepSize(double stepSize) {
  _stepSize_ = stepSize;
}
void FitParameter::setOwner(const FitParameterSet* owner_) {
  _owner_ = owner_;
}
void FitParameter::setPriorType(PriorType::PriorType priorType) {
  _priorType_ = priorType;
}

void FitParameter::setValueAtPrior(){
  _parameterValue_ = _priorValue_;
}
void FitParameter::setCurrentValueAsPrior(){
  _priorValue_ = _parameterValue_;
}

void FitParameter::initialize() {

  LogThrowIf(_parameterIndex_ == -1, "Parameter index is not set.")
  LogThrowIf(_priorValue_     == std::numeric_limits<double>::quiet_NaN(), "Prior value is not set.")
  LogThrowIf(_stdDevValue_    == std::numeric_limits<double>::quiet_NaN(), "Std dev value is not set.")
  LogThrowIf(_parameterValue_ == std::numeric_limits<double>::quiet_NaN(), "Parameter value is not set.")
  LogThrowIf(_owner_ == nullptr, "Parameter set ref is not set.")

  if( not _parameterConfig_.empty() ){
    _isEnabled_ = JsonUtils::fetchValue(_parameterConfig_, "isEnabled", true);
    if( not _isEnabled_ ) { return; }

    auto priorTypeStr = JsonUtils::fetchValue(_parameterConfig_, "priorType", "");
    if( not priorTypeStr.empty() ){
      _priorType_ = PriorType::PriorTypeEnumNamespace::toEnum(priorTypeStr);
     if( _priorType_ == PriorType::Flat ){ _isFree_ = true; }
    }

    if( JsonUtils::doKeyExist(_parameterConfig_, "priorValue") ){
      _priorValue_ = JsonUtils::fetchValue(_parameterConfig_, "priorValue", _priorValue_);
      LogWarning << this->getTitle() << ": prior value override -> " << _priorValue_ << std::endl;
      this->setParameterValue(_priorValue_);
    }

    if( JsonUtils::doKeyExist(_parameterConfig_, "parameterLimits") ){
      std::pair<double, double> limits{std::nan(""), std::nan("")};
      limits = JsonUtils::fetchValue(_parameterConfig_, "parameterLimits", limits);
      LogWarning << "Overriding parameter limits: [" << limits.first << ", " << limits.second << "]." << std::endl;
      this->setMinValue(limits.first);
      this->setMaxValue(limits.second);
    }

    _dialDefinitionsList_ = JsonUtils::fetchValue(_parameterConfig_, "dialSetDefinitions", _dialDefinitionsList_);
  }

  _dialSetList_.reserve(_dialDefinitionsList_.size());
  for( const auto& dialDefinitionConfig : _dialDefinitionsList_ ){
    _dialSetList_.emplace_back();
    _dialSetList_.back().setOwner(this);
    _dialSetList_.back().setConfig(dialDefinitionConfig);
    _dialSetList_.back().initialize();
  }

  // Check if no dials is actually defined -> disable the parameter in that case
  bool dialSetAreAllDisabled = true;
  for( const auto& dialSet : _dialSetList_ ){
    if( dialSet.isEnabled() ){
      dialSetAreAllDisabled = false;
      break;
    }
  }
  if( dialSetAreAllDisabled ){
    LogError << "Parameter " << getTitle() << " has no dials: disabled." << std::endl;
    _isEnabled_ = false;
  }

}

bool FitParameter::isEnabled() const {
  return _isEnabled_;
}
bool FitParameter::isFixed() const {
  return _isFixed_;
}
bool FitParameter::isEigen() const {
  return _isEigen_;
}
bool FitParameter::isFree() const {
  return _isFree_;
}
const std::string &FitParameter::getName() const {
  return _name_;
}
double FitParameter::getParameterValue() const {
  return _parameterValue_;
}
int FitParameter::getParameterIndex() const {
  return _parameterIndex_;
}
double FitParameter::getStdDevValue() const {
  return _stdDevValue_;
}
double FitParameter::getPriorValue() const {
  return _priorValue_;
}
double FitParameter::getThrowValue() const{
  return _throwValue_;
}
PriorType::PriorType FitParameter::getPriorType() const {
  return _priorType_;
}
std::vector<DialSet> &FitParameter::getDialSetList() {
  return _dialSetList_;
}
double FitParameter::getMinValue() const {
  return _minValue_;
}
double FitParameter::getMaxValue() const {
  return _maxValue_;
}
double FitParameter::getStepSize() const {
  return _stepSize_;
}
const FitParameterSet *FitParameter::getOwner() const {
  return _owner_;
}


double FitParameter::getDistanceFromNominal() const{
  return (_parameterValue_ - _priorValue_) / _stdDevValue_;
}
DialSet* FitParameter::findDialSet(const std::string& dataSetName_){
  for( auto& dialSet : _dialSetList_ ){
    if( GenericToolbox::doesElementIsInVector(dataSetName_, dialSet.getDataSetNameList()) ){
      return &dialSet;
    }
  }

  // If not found, find general dialSet
  for( auto& dialSet : _dialSetList_ ){
    if( GenericToolbox::doesElementIsInVector("", dialSet.getDataSetNameList())
        or GenericToolbox::doesElementIsInVector("*", dialSet.getDataSetNameList())
      ){
      return &dialSet;
    }
  }

  // If no general dialSet found, this parameter does not apply on this dataSet
  return nullptr;
}

std::string FitParameter::getSummary() const {
  std::stringstream ss;

  ss << "#" << _parameterIndex_;
  if( not _name_.empty() ) ss << " (" << _name_ << ")";
  ss << ": value=" << _parameterValue_;
  ss << ", prior=" << _priorValue_;
  ss << ", stdDev=" << _stdDevValue_;
  ss << ", isEnabled=" << _isEnabled_;

  if( _enableDialSetsSummary_ ){
    ss << ":";
    for( const auto& dialSet : _dialSetList_ ){
      ss << std::endl << GenericToolbox::indentString(dialSet.getSummary(), 2);
    }
  }

  return ss.str();
}
std::string FitParameter::getTitle() const {
  std::stringstream ss;
  ss << "#" << _parameterIndex_;
  if( not _name_.empty() ) ss << "_" << _name_;
  return ss.str();
}
std::string FitParameter::getFullTitle() const{
  return ((FitParameterSet*) _owner_)->getName() + "/" + this->getTitle();
}
