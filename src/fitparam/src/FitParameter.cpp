//
// Created by Nadrino on 21/05/2021.
//

#include "sstream"

#include "Logger.h"

#include "JsonUtils.h"
#include "FitParameter.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[FitParameter]");
} )

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
}

void FitParameter::setIsFixed(bool isFixed) {
  _isFixed_ = isFixed;
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
//  if( _isFixed_ and _parameterValue_ != parameterValue ){
//    LogDebug << "CHANGING FIXED " << getTitle() << ": " << _parameterValue_ << " -> " << parameterValue << std::endl;
//  }
  _parameterValue_ = parameterValue;
}
void FitParameter::setPriorValue(double priorValue) {
  _priorValue_ = priorValue;
}
void FitParameter::setStdDevValue(double stdDevValue) {
  _stdDevValue_ = stdDevValue;
}
void FitParameter::setEnableDialSetsSummary(bool enableDialSetsSummary) {
  _enableDialSetsSummary_ = enableDialSetsSummary;
}
void FitParameter::setDialsWorkingDirectory(const std::string &dialsWorkingDirectory) {
  _dialsWorkingDirectory_ = dialsWorkingDirectory;
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

void FitParameter::initialize() {

  LogTrace << GET_VAR_NAME_VALUE(this) << std::endl;

  LogThrowIf(_priorValue_ == std::numeric_limits<double>::quiet_NaN(), "Prior value is not set.");
  LogThrowIf(_stdDevValue_ == std::numeric_limits<double>::quiet_NaN(), "Std dev value is not set.");
  LogThrowIf(_parameterValue_ == std::numeric_limits<double>::quiet_NaN(), "Parameter value is not set.");
  LogThrowIf(_parameterIndex_ == -1, "Parameter index is not set.");

  LogDebug << "Initializing Parameter " << getTitle() << std::endl;

  _stepSize_ = _stdDevValue_ * 0.01; // default

  if( not _parameterConfig_.empty() ){
    _isEnabled_ = JsonUtils::fetchValue(_parameterConfig_, "isEnabled", true);
    if( not _isEnabled_ ) {
      LogDebug << getTitle() << " is marked as not Enabled." << std::endl;
      return;
    }
    _dialDefinitionsList_ = JsonUtils::fetchValue(_parameterConfig_, "dialSetDefinitions", _dialDefinitionsList_);
  }

  LogDebug << "Defining associated dials..." << std::endl;
  _dialSetList_.reserve(_dialDefinitionsList_.size());
  for( const auto& dialDefinitionConfig : _dialDefinitionsList_ ){
    _dialSetList_.emplace_back();
    _dialSetList_.back().setParameterIndex(_parameterIndex_);
    _dialSetList_.back().setParameterName(_name_);
    _dialSetList_.back().setDialSetConfig(dialDefinitionConfig);
    _dialSetList_.back().setWorkingDirectory(_dialsWorkingDirectory_);
    _dialSetList_.back().setAssociatedParameterReference(this);
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
    LogWarning << "Parameter " << getTitle() << " has no dials: disabled." << std::endl;
    _isEnabled_ = false;
  }

}

bool FitParameter::isEnabled() const {
  return _isEnabled_;
}
bool FitParameter::isFixed() const {
  return _isFixed_;
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

