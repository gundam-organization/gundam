//
// Created by Nadrino on 21/05/2021.
//

#include "Parameter.h"
#include "ParameterSet.h"
#include "ConfigUtils.h"
#include "GundamBacktrace.h"

#include "Logger.h"

#include <sstream>


void Parameter::prepareConfig(ConfigReader& config_){
  config_.clearFields();
  config_.defineFields({
    {"name", {"parameterName"}},
    {"isEnabled"},
    {"priorValue"},
    {"isFixed"},
    {"isThrown"},
    {"parameterStepSize"},
    {"parameterIndex"},
    {"parameterLimits"},
    {"physicalLimits"},
    {"throwLimits"},
    {"mirrorRange"},
    {"cyclicRange"},
    {"dialSetDefinitions"},
    {"priorType"},
  });
  config_.checkConfiguration();
}
void Parameter::configureImpl(){

  prepareConfig(_config_);

  _config_.fillValue(_name_, "name");
  _config_.fillValue(_isEnabled_, "isEnabled");
  if( not _isEnabled_ ) { return; }

  if( _config_.hasField("priorValue") ){
    auto priorValue = _config_.fetchValue<double>("priorValue");
    if( not std::isnan(priorValue) and priorValue != _priorValue_ ){
      _priorValue_ = priorValue;
      _parameterValue_ = _priorValue_;
    }
  }

  _config_.fillValue(_isFixed_, "isFixed");
  _config_.fillValue(_isThrown_, "isThrown");
  _config_.fillValue(_stepSize_, "parameterStepSize");
  _config_.fillValue(_physicalLimits_, "physicalLimits");
  _config_.fillValue(_throwLimits_, "throwLimits");
  _config_.fillValue(_mirrorRange_, "mirrorRange");
  _config_.fillValue(_cyclicRange_, "cyclicRange");
  _config_.fillValue(_dialDefinitionsList_, "dialSetDefinitions");

  _config_.fillValue(_parameterLimits_, "parameterLimits");

  _config_.fillEnum(_priorType_, "priorType");
  if( _priorType_ == PriorType::Flat ){ _isFree_ = true; }

}
void Parameter::initializeImpl() {

  _config_.printUnusedKeys();

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

bool Parameter::isCyclic() const {
  if (!std::isfinite(_cyclicRange_.min)) return false;
  if (!std::isfinite(_cyclicRange_.max)) return false;
  if (_cyclicRange_.max <= _cyclicRange_.min) return false;
  return true;
}

void Parameter::setMinCyclic(double minCyclic) {
  if (std::isfinite(_cyclicRange_.min) and std::abs(_cyclicRange_.min-minCyclic) > 1E-6) {
    LogWarning << "Minimum cyclic bound changed for " << getFullTitle()
               << " old: " << _cyclicRange_.min
               << " new: " << minCyclic
               << std::endl;
  }
  _cyclicRange_.min = minCyclic;
}
void Parameter::setMaxCyclic(double maxCyclic) {
  if (std::isfinite(_cyclicRange_.max) and std::abs(_cyclicRange_.max-maxCyclic) > 1E-6) {
    LogWarning << "Maximum cyclic bound changed for " << getFullTitle()
               << " old: " << _cyclicRange_.max
               << " new: " << maxCyclic
               << std::endl;
  }
  _cyclicRange_.max = maxCyclic;
}
void Parameter::setParameterValue(double parameterValue, bool force) {
  // update and flag parameter
  if( _parameterValue_ != parameterValue ){
    _gotUpdated_ = true;

#ifdef DEBUG_BUILD
    if (not isInDomain(parameterValue, true)) {
#else
    if (not isInDomain(parameterValue, false)) {
#endif
      LogError << getFullTitle() << ": setting new value out of domain. " << parameterValue << " is not in " << _parameterLimits_ << std::endl;

      static bool once{false};
      if( not once and _owner_->isEnableEigenDecomp() and not this->isEigen() ){
        once = true;
        LogAlert << "Not in domain error will appears since par limits can't directly be applied when using eigen decomp." << std::endl;
      }

      if( not force ){ LogError << GundamUtils::Backtrace; std::exit(EXIT_FAILURE); }
#ifdef DEBUG_BUILD
      LogDebug << GundamUtils::Backtrace;
      LogAlert << "Forced continuation with invalid parameter" << std::endl;
#endif
    }

    // setting the parameter
    _parameterValue_ = parameterValue;
  }
  else{ _gotUpdated_ = false; }
}
double Parameter::getParameterValue() const {
#ifdef DEBUG_BUILD
  if ( isEnabled() and not isValueWithinBounds() ) {
    LogAlert << "Getting out of bounds parameter: " << getSummary() << std::endl;
    LogDebug << GundamUtils::Backtrace;
  }
#endif
  return _parameterValue_;
}
double Parameter::getRegularizedValue() const {
  double v = getParameterValue();
  if (isCyclic()) {
    double r = _cyclicRange_.max - _cyclicRange_.min;
    while (v < _cyclicRange_.min) v += r;
    while (v >= _cyclicRange_.max) v -= r;
  }
  if (_mirrorRange_.hasBound()) {
    while (not _mirrorRange_.isInBounds(v)) {
      if (_mirrorRange_.hasLowerBound() and v < _mirrorRange_.min) {
        v = 2.0*_mirrorRange_.min - v;
      }
      if (_mirrorRange_.hasUpperBound() and v > _mirrorRange_.max) {
        v = 2.0*_mirrorRange_.max - v;
      }
    }
  }
  return v;
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

  if( not _parameterLimits_.hasBound() ){ return true; }

  if( _parameterLimits_.isBelowMin(value_) ){
    if (verbose_) {
      LogError << "Value is below minimum: " << value_ << std::endl;
      LogError << "Summary: " << getSummary() << std::endl;
    }
    return false;
  }
  if ( _parameterLimits_.isAboveMax(value_) ) {
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
