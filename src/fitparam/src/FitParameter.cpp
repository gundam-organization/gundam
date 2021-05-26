//
// Created by Adrien BLANCHET on 21/05/2021.
//

#include "FitParameter.h"

#include "Logger.h"

FitParameter::FitParameter() {
  Logger::setUserHeaderStr("[FitParameter]");
  this->reset();
}
FitParameter::~FitParameter() {
  this->reset();
}

void FitParameter::reset() {
  _dialSetList_.clear();
  _parameterDefinitionsConfig_ = nlohmann::json();
  _parameterValue_ = 0;
  _parameterIndex_ = -1;
}

void FitParameter::initialize() {

}

void FitParameter::setParameterDefinitionsConfig(const nlohmann::json &jsonConfig) {
  _parameterDefinitionsConfig_ = jsonConfig;
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

const std::string &FitParameter::getName() const {
  return _name_;
}
double FitParameter::getParameterValue() const {
  return _parameterValue_;
}





