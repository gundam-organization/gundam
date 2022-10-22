//
// Created by Adrien BLANCHET on 23/10/2022.
//

#include "ConfigBasedClass.h"

#include "JsonUtils.h"


ConfigBasedClass::ConfigBasedClass(const nlohmann::json& config_){ this->readConfig(config_); }

void ConfigBasedClass::setConfig(const nlohmann::json &config_) { _config_ = config_; JsonUtils::forwardConfig(_config_); }

void ConfigBasedClass::readConfig() {
  _isConfigReadDone_ = true;
  this->readConfigImpl();
}
void ConfigBasedClass::readConfig(const nlohmann::json& config_){
  this->setConfig(config_);
  this->readConfig();
}

void ConfigBasedClass::initialize() {
  if( not _isConfigReadDone_ ) this->readConfig();
  this->initializeImpl();
  _isInitialized_ = true;
}

bool ConfigBasedClass::isConfigReadDone() const { return _isConfigReadDone_; }
bool ConfigBasedClass::isInitialized() const { return _isInitialized_; }
const nlohmann::json &ConfigBasedClass::getConfig() const { return _config_; }
