//
// Created by Nadrino on 21/05/2021.
//

#include "FitParameter.h"
#include "FitParameterSet.h"
#include "JsonUtils.h"

#include "Logger.h"

#include "sstream"


LoggerInit([]{ Logger::setUserHeaderStr("[FitParameter]"); });


FitParameter::FitParameter(const FitParameterSet* owner_): _owner_(owner_) {}

void FitParameter::readConfigImpl(){
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
    _dialSetList_.emplace_back(this);
    _dialSetList_.back().readConfig(dialDefinitionConfig);
  }
}
void FitParameter::initializeImpl() {
  LogThrowIf(_owner_ == nullptr, "Parameter set ref is not set.");
  LogThrowIf(_parameterIndex_ == -1, "Parameter index is not set.");
  LogThrowIf(std::isnan(_priorValue_), "Prior value is not set.");
  LogThrowIf(std::isnan(_stdDevValue_), "Std dev value is not set.");
  LogThrowIf(std::isnan(_parameterValue_), "Parameter value is not set.");

  if( not _isEnabled_ ) { return; }

  for( auto& dialSet : _dialSetList_ ){ dialSet.initialize(); }

  // Check if no dials is actually defined -> disable the parameter in that case
  bool dialSetAreAllDisabled = true;
  for( const auto& dialSet : _dialSetList_ ){ if( dialSet.isEnabled() ){ dialSetAreAllDisabled = false; break; } }
  if( dialSetAreAllDisabled ){
    LogWarning << "Parameter " << getTitle() << " has no dials: disabled." << std::endl;
    _isEnabled_ = false;
  }
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
  LogThrowIf( std::isnan(parameterValue), "Attempting to set NaN value for par:" << std::endl << this->getSummary() );
  if( _parameterValue_ != parameterValue ){
    _gotUpdated_ = true;
    _parameterValue_ = parameterValue;
  }
  else{ _gotUpdated_ = false; }
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

std::string FitParameter::getSummary(bool shallow_) const {
  std::stringstream ss;

  ss << "#" << _parameterIndex_;
  if( not _name_.empty() ) ss << " (" << _name_ << ")";
  ss << ", isEnabled=" << _isEnabled_;
  ss << ": value=" << _parameterValue_;
  ss << ", prior=" << _priorValue_;
  ss << ", stdDev=" << _stdDevValue_;
  ss << ", bounds=[ ";
  if( std::isnan(_minValue_) ) ss << "-inf";
  else ss << _minValue_;
  ss << ", ";
  if( std::isnan(_maxValue_) ) ss << "+inf";
  else ss << _maxValue_;
  ss << " ]";

  if( not shallow_ ){
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
