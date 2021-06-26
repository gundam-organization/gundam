//
// Created by Adrien BLANCHET on 21/05/2021.
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
void FitParameter::setStdDevValue(double stdDevValue) {
  _stdDevValue_ = stdDevValue;
}
void FitParameter::setEnableDialSetsSummary(bool enableDialSetsSummary) {
  _enableDialSetsSummary_ = enableDialSetsSummary;
}
void FitParameter::setDialsWorkingDirectory(const std::string &dialsWorkingDirectory) {
  _dialsWorkingDirectory_ = dialsWorkingDirectory;
}

void FitParameter::initialize() {

  if     ( _priorValue_ == std::numeric_limits<double>::quiet_NaN() ){
    LogError << "_priorValue_ is not set." << std::endl;
    throw std::logic_error("_priorValue_ is not set.");
  }
  else if( _stdDevValue_ == std::numeric_limits<double>::quiet_NaN() ){
    LogError << "_stdDevValue_ is not set." << std::endl;
    throw std::logic_error("_stdDevValue_ is not set.");
  }
  else if( _parameterValue_ == std::numeric_limits<double>::quiet_NaN() ){
    LogError << "_parameterValue_ is not set." << std::endl;
    throw std::logic_error("_parameterValue_ is not set.");
  }
  else if( _parameterIndex_ == -1 ){
    LogError << "_parameterIndex_ is not set." << std::endl;
    throw std::logic_error("_parameterIndex_ is not set.");
  }
  else if( _dialDefinitionsList_.empty() ){
    LogError << "_dialDefinitionsList_ is not set." << std::endl;
    throw std::logic_error("_dialDefinitionsList_ is not set.");
  }

  LogDebug << "Initializing Parameter " << getTitle() << std::endl;

  for( const auto& dialDefinitionConfig : _dialDefinitionsList_ ){
    _dialSetList_.emplace_back();
    _dialSetList_.back().setParameterIndex(_parameterIndex_);
    _dialSetList_.back().setParameterName(_name_);
    _dialSetList_.back().setDialSetConfig(dialDefinitionConfig);
    _dialSetList_.back().setWorkingDirectory(_dialsWorkingDirectory_);
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
const std::vector<DialSet> &FitParameter::getDialSetList() const {
  return _dialSetList_;
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

