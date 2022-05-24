//
// Created by Adrien BLANCHET on 07/04/2022.
//

#include "ScanConfig.h"
#include "JsonUtils.h"

#include <utility>

ScanConfig::ScanConfig() = default;
ScanConfig::ScanConfig(nlohmann::json config_) : _config_(std::move(config_)) { this->readConfig(); }
ScanConfig::~ScanConfig() = default;


void ScanConfig::readConfig() {
  if( _config_.empty() ) return;

  _useParameterLimits_ = JsonUtils::fetchValue(_config_, "useParameterLimits", _useParameterLimits_);
  _nbPoints_ = JsonUtils::fetchValue(_config_, "nbPoints", _nbPoints_);
  _parameterSigmaRange_ = JsonUtils::fetchValue(_config_, "parameterSigmaRange", _parameterSigmaRange_);

  _varsConfig_ = JsonUtils::fetchValue(_config_, "varsConfig", nlohmann::json());
}

int ScanConfig::getNbPoints() const {
  return _nbPoints_;
}
const std::pair<double, double> &ScanConfig::getParameterSigmaRange() const {
  return _parameterSigmaRange_;
}
bool ScanConfig::isUseParameterLimits() const {
  return _useParameterLimits_;
}

const nlohmann::json &ScanConfig::getVarsConfig() const {
  return _varsConfig_;
}


