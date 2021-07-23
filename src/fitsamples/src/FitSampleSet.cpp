//
// Created by Adrien BLANCHET on 22/07/2021.
//

#include "json.hpp"

#include "Logger.h"
#include "GenericToolbox.h"

#include "JsonUtils.h"
#include "FitSampleSet.h"


LoggerInit([](){
  Logger::setUserHeaderStr("[FitSampleSet]");
})

FitSampleSet::FitSampleSet() { this->reset(); }
FitSampleSet::~FitSampleSet() { this->reset(); }

void FitSampleSet::reset() {

  _config_.clear();

}

void FitSampleSet::setConfig(const nlohmann::json &config) {
  _config_ = config;
  while( _config_.is_string() ){
    LogWarning << "Forwarding " << __CLASS_NAME__ << " config: \"" << _config_.get<std::string>() << "\"" << std::endl;
    _config_ = JsonUtils::readConfigFile(_config_.get<std::string>());
  }
}

void FitSampleSet::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  if( _config_.empty() ){
    LogError << "_config_ is not set." << std::endl;
    throw std::logic_error("_config_ is not set.");
  }

  LogDebug << "Loading datasets..." << std::endl;
  auto dataSetListConfig = JsonUtils::fetchValue(_config_, "dataSetList", nlohmann::json());
  if( dataSetListConfig.empty() ){
    LogError << "No dataSet specified." << std::endl;
    throw std::runtime_error("No dataSet specified.");
  }
  for( const auto& dataSetConfig : dataSetListConfig ){
    _dataSetList_.emplace_back();
    _dataSetList_.back().setConfig(dataSetConfig);
    _dataSetList_.back().initialize();
  }

  LogDebug << "Defining samples..." << std::endl;
  auto fitSampleListConfig = JsonUtils::fetchValue(_config_, "fitSampleList", nlohmann::json());
  for( const auto& fitSampleConfig: fitSampleListConfig ){
    _fitSampleList_.emplace_back();
    _fitSampleList_.back().setConfig(fitSampleConfig);
    _fitSampleList_.back().initialize();
  }

}

const std::vector<FitSample> &FitSampleSet::getFitSampleList() const {
  return _fitSampleList_;
}
std::vector<DataSet> &FitSampleSet::getDataSetList() {
  return _dataSetList_;
}
