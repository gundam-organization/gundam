//
// Created by Nadrino on 21/05/2021.
//

#include "Parameter.h"
#include "ParameterSet.h"
#include "ConfigUtils.h"
#include "GundamBacktrace.h"

#include "Logger.h"

#include <sstream>


void Parameter::configureImpl(){
  ConfigUtils::checkFields(_config_,
                           "/fitterEngineConfig/likelihoodInterfaceConfig"
                           "/propagatorConfig/parametersManagerConfig"
                           "/parameterDefinitions/(parameter)",
                           // Allowed fields (don't need to list fields in
                           // expected, or deprecated).
                           {{"parameterName"}, // handled in parameter set.
                            {"isEnabled"},
                            {"isFixed"},
                            {"priorValue"},
                            {"isThrown"},
                            {"parameterStepSize"},
                            {"parameterLimits"},
                            {"throwLimits"},
                            {"mirrorRange"},
                            {"dialSetDefinitions"},
                            {"priorType"},
                           },
                           // Expected fields (must be present)
                           {
                           },
                           // Deprecated fields (allowed, but cause a warning)
                           {
                           },
                           // Replaced field names (allowed, but warn)
                           {
                             {{"name"}, {"parameterName"}},
                           }
);

  GenericToolbox::Json::fillValue(_config_, _isEnabled_, "isEnabled");
  if( not _isEnabled_ ) { return; }

  if( GenericToolbox::Json::doKeyExist(_config_, "priorValue") ){
    auto priorValue = GenericToolbox::Json::fetchValue<double>(_config_, "priorValue");
    if( not std::isnan(priorValue) and priorValue != _priorValue_ ){
      _priorValue_ = priorValue;
      _parameterValue_ = _priorValue_;
    }
  }

  GenericToolbox::Json::fillValue(_config_, _isFixed_, "isFixed");
  GenericToolbox::Json::fillValue(_config_, _isThrown_, "isThrown");
  GenericToolbox::Json::fillValue(_config_, _stepSize_, "parameterStepSize");
  GenericToolbox::Json::fillValue(_config_, _parameterLimits_, "parameterLimits");
  GenericToolbox::Json::fillValue(_config_, _physicalLimits_, "physicalLimits");
  GenericToolbox::Json::fillValue(_config_, _throwLimits_, "throwLimits");
  GenericToolbox::Json::fillValue(_config_, _mirrorRange_, "mirrorRange");
  GenericToolbox::Json::fillValue(_config_, _dialDefinitionsList_, "dialSetDefinitions");

  GenericToolbox::Json::fillEnum(_config_, _priorType_, "priorType");
  if( _priorType_ == PriorType::Flat ){ _isFree_ = true; }

}
void Parameter::initializeImpl() {
  LogThrowIf(_owner_ == nullptr, "Parameter set ref is not set.");
  LogThrowIf(_parameterIndex_ == -1, "Parameter index is not set.");

  if( not _isEnabled_ ) { return; }
  LogThrowIf(std::isnan(_priorValue_), "Prior value is not set: " << getFullTitle());
  LogThrowIf(std::isnan(_stdDevValue_), "Std dev value is not set: " << getFullTitle());
  LogThrowIf(std::isnan(_parameterValue_), "Parameter value is not set: " << getFullTitle());

  if( _priorValue_ == _parameterLimits_.min or _priorValue_ == _parameterLimits_.max ) {
    // the user should know. This will prevent Asimov fits to converge
    LogAlert << "Prior value of \"" << getFullTitle() << "\" is set on the defined limits: " << _priorValue_ << " -> " << _parameterLimits_ << std::endl;
  }

  // make sure the throws will always give parameters in bounds
  _throwLimits_.fillMostConstrainingBounds(_parameterLimits_);
  _throwLimits_.fillMostConstrainingBounds(_physicalLimits_);
}

void Parameter::setMinMirror(double minMirror) {
  if (std::isfinite(_mirrorRange_.min) and std::abs(_mirrorRange_.min-minMirror) > 1E-6) {
    LogWarning << "Minimum mirror bound changed for " << getFullTitle()
               << " old: " << _mirrorRange_.min
               << " new: " << minMirror
               << std::endl;
  }
  _mirrorRange_.min = minMirror;
}
void Parameter::setMaxMirror(double maxMirror) {
  if (std::isfinite(_mirrorRange_.max) and std::abs(_mirrorRange_.max-maxMirror) > 1E-6) {
    LogWarning << "Maximum mirror bound changed for " << getFullTitle()
               << " old: " << _mirrorRange_.max
               << " new: " << maxMirror
               << std::endl;
  }
  _mirrorRange_.max = maxMirror;
}
void Parameter::setParameterValue(double parameterValue, bool force) {
  if (not isInDomain(parameterValue, true)) {
    LogError << "New parameter value is not in domain: " << parameterValue
             << std::endl;
    if (not force) {
      LogError << GundamUtils::Backtrace;
      std::exit(EXIT_FAILURE);
    }
    else {
#ifdef DEBUG_BUILD
      LogDebug << GundamUtils::Backtrace;
#endif
      LogAlert << "Forced continuation with invalid parameter" << std::endl;
    }
  }
  if( _parameterValue_ != parameterValue ){
    _gotUpdated_ = true;
    _parameterValue_ = parameterValue;
  }
  else{ _gotUpdated_ = false; }
}
double Parameter::getParameterValue() const {
  if ( isEnabled() and not isValueWithinBounds() ) {
    LogWarning << "Getting out of bounds parameter: "
               << getSummary() << std::endl;
#ifdef DEBUG_BUILD
    LogDebug << GundamUtils::Backtrace;
#endif
  }
  return _parameterValue_;
}
void Parameter::setDialSetConfig(const JsonType &jsonConfig_) {
  auto jsonConfig = jsonConfig_;
  while( jsonConfig.is_string() ){
    LogWarning << "Forwarding FitParameterSet config to: \"" << jsonConfig.get<std::string>() << "\"..." << std::endl;
    jsonConfig = ConfigUtils::readConfigFile(jsonConfig.get<std::string>());
  }
  _dialDefinitionsList_ = jsonConfig.get<std::vector<JsonType>>();
}

void Parameter::setValueAtPrior(){
  setParameterValue(getPriorValue());
}
void Parameter::setCurrentValueAsPrior(){
  setPriorValue(getParameterValue());
}
bool Parameter::isInDomain(double value_, bool verbose_) const {
  if( std::isnan(value_) ) {
    if (verbose_) {
      LogError << "NaN value is not in parameter domain" << std::endl;
      LogError << "Summary: " << getSummary() << std::endl;
    }
    return false;
  }
  if ( not std::isnan(_parameterLimits_.min) and value_ < _parameterLimits_.min ) {
    if (verbose_) {
      LogError << "Value is below minimum: " << value_ << std::endl;
      LogError << "Summary: " << getSummary() << std::endl;
    }
    return false;
  }
  if ( not std::isnan(_parameterLimits_.max) and value_ > _parameterLimits_.max ) {
    if (verbose_) {
      LogError << "Attempting to set parameter above the maximum"
               << " -- New value: " << value_
               << std::endl;
      LogError << "Summary: " << getSummary() << std::endl;
    }
    return false;
  }
  return true;
}
bool Parameter::isPhysical(double value_) const {
  if( not isInDomain(value_) ){ return false; }
  if( not _parameterLimits_.isInBounds(value_) ){ return false; }
  return true;
}
bool Parameter::isMirrored(double value_) const {
  if( not _mirrorRange_.isInBounds(value_) ){ return true; }
  return false;
}
bool Parameter::isValidValue(double value) const {
  if ((_validFlags_ & 0b0001)!=0 and (not isInDomain(value))) return false;
  if ((_validFlags_ & 0b0010)!=0 and (isMirrored(value))) return false;
  if ((_validFlags_ & 0b0100)!=0 and (not isPhysical(value))) return false;
  return true;
}
void Parameter::setValidity(const std::string& validity) {
  if (validity.find("noran") != std::string::npos) _validFlags_ &= ~0b0001;
  else if (validity.find("ran") != std::string::npos) _validFlags_ |= 0b0001;
  if (validity.find("nomir") != std::string::npos) _validFlags_ &= ~0b0010;
  else if (validity.find("mir") != std::string::npos) _validFlags_ |= 0b0010;
  if (validity.find("nophy") != std::string::npos) _validFlags_ &= ~0b0100;
  else if (validity.find("phy") != std::string::npos) _validFlags_ |= 0b0100;
}
bool Parameter::isValueWithinBounds() const{
  return isInDomain(_parameterValue_);
}
double Parameter::getDistanceFromNominal() const{
  return (getParameterValue() - getPriorValue()) / _stdDevValue_;
}
std::string Parameter::getTitle() const {
  std::stringstream ss;
  ss << "#" << _parameterIndex_;
  if( not _name_.empty() ) ss << "_" << _name_;
  return ss.str();
}
std::string Parameter::getFullTitle() const{
  return _owner_->getName() + "/" + this->getTitle();
}
std::string Parameter::getSummary() const {
  std::stringstream ss;

  ss << this->getFullTitle();

  ss << ", ";
  if( not _isEnabled_ ){ ss << GenericToolbox::ColorCodes::redBackground; }
  ss << "isEnabled=" << _isEnabled_;
  if( not _isEnabled_ ){ ss << GenericToolbox::ColorCodes::resetColor; return ss.str(); }

  ss << ": value=" << _parameterValue_;
  ss << ", prior=" << _priorValue_;
  ss << ", stdDev=" << _stdDevValue_;
  ss << ", bounds=" << _parameterLimits_;
  if (not isValueWithinBounds()){
    ss << GenericToolbox::ColorCodes::redBackground << " out of bounds" << GenericToolbox::ColorCodes::resetColor;
  }

  return ss.str();
}

void Parameter::printConfiguration() const {
  LogInfo << getSummary() << std::endl;
}

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// End:
