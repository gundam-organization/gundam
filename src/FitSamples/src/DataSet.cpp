//
// Created by Nadrino on 22/07/2021.
//

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "Logger.h"

#include "JsonUtils.h"
#include "DataSet.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[DataSet]");
})

DataSet::DataSet() { this->reset(); }
DataSet::~DataSet() { this->reset(); }

void DataSet::reset() {
  _isInitialized_ = false;
  _config_.clear();
  _isEnabled_ = false;

  _name_ = "";
  _requestedLeafNameList_.clear();
  _mcFilePathList_.clear();
  _dataFilePathList_.clear();

  _mcNominalWeightFormulaStr_ = "1";
}

void DataSet::setConfig(const nlohmann::json &config_) {
  _config_ = config_;
  JsonUtils::forwardConfig(_config_, __CLASS_NAME__);
}
void DataSet::addRequestedLeafName(const std::string& leafName_){
  LogThrowIf(leafName_.empty(), "no leaf name provided.")
  if( not GenericToolbox::doesElementIsInVector(leafName_, _requestedLeafNameList_) ){
    _requestedLeafNameList_.emplace_back(leafName_);
  }
}
void DataSet::addRequestedMandatoryLeafName(const std::string& leafName_){
  if( not leafName_.empty() and not GenericToolbox::doesElementIsInVector(leafName_, _requestedMandatoryLeafNameList_) ){
    _requestedMandatoryLeafNameList_.emplace_back(leafName_);
  }
  this->addRequestedLeafName(leafName_);
}

void DataSet::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  if( _config_.empty() ){
    LogError << "_config_ is not set." << std::endl;
    throw std::logic_error("_config_ is not set.");
  }

  _isEnabled_ = JsonUtils::fetchValue(_config_, "isEnabled", true);
  if( not _isEnabled_ ){
    LogWarning << "\"" << _name_ << "\" is disabled." << std::endl;
    return;
  }

  _name_ = JsonUtils::fetchValue<std::string>(_config_, "name");
  LogDebug << "Initializing dataset: \"" << _name_ << "\"" << std::endl;

  {
    auto mcConfig = JsonUtils::fetchValue(_config_, "mc", nlohmann::json());
    if( not mcConfig.empty() ){
      _mcTreeName_ = JsonUtils::fetchValue<std::string>(mcConfig, "tree");
      auto fileList = JsonUtils::fetchValue(mcConfig, "filePathList", nlohmann::json());
      for( const auto& file: fileList ){
        _mcFilePathList_.emplace_back(file.get<std::string>());
      }
    }

    // override: nominalWeightLeafName is deprecated
    _mcNominalWeightFormulaStr_ = JsonUtils::fetchValue(mcConfig, "nominalWeightLeafName", _mcNominalWeightFormulaStr_);
    _mcNominalWeightFormulaStr_ = JsonUtils::fetchValue(mcConfig, "nominalWeightFormula", _mcNominalWeightFormulaStr_);
  }

  {
    auto dataConfig = JsonUtils::fetchValue(_config_, "data", nlohmann::json());
    if( not dataConfig.empty() ){
      _dataTreeName_ = JsonUtils::fetchValue<std::string>(dataConfig, "tree");
      auto fileList = JsonUtils::fetchValue(dataConfig, "filePathList", nlohmann::json());
      for( const auto& file: fileList ){
        _dataFilePathList_.emplace_back(file.get<std::string>());
      }
    }
  }

  this->print();

  _isInitialized_ = true;
}


bool DataSet::isEnabled() const {
  return _isEnabled_;
}
const std::string &DataSet::getName() const {
  return _name_;
}
std::vector<std::string> &DataSet::getMcActiveLeafNameList() {
  return _mcActiveLeafNameList_;
}
std::vector<std::string> &DataSet::getDataActiveLeafNameList() {
  return _dataActiveLeafNameList_;
}
const std::string &DataSet::getMcNominalWeightFormulaStr() const {
  return _mcNominalWeightFormulaStr_;
}
const std::vector<std::string> &DataSet::getRequestedLeafNameList() const {
  return _requestedLeafNameList_;
}
const std::vector<std::string> &DataSet::getRequestedMandatoryLeafNameList() const {
  return _requestedMandatoryLeafNameList_;
}
const std::vector<std::string> &DataSet::getMcFilePathList() const {
  return _mcFilePathList_;
}
const std::vector<std::string> &DataSet::getDataFilePathList() const {
  return _dataFilePathList_;
}

TChain* DataSet::buildChain(bool isData_){
  LogThrowIf(not _isInitialized_, "Can't do " << __METHOD_NAME__ << " while not init.")
  TChain* out{nullptr};
  if( not isData_ and not _mcFilePathList_.empty() ){
    out = new TChain(_mcTreeName_.c_str());
    for( const auto& file: _mcFilePathList_){
      if( not GenericToolbox::doesTFileIsValid(file, {_mcTreeName_}) ){
        LogError << "Invalid file: " << file << std::endl;
        throw std::runtime_error("Invalid file.");
      }
      out->Add(file.c_str());
    }
  }
  else if( isData_ and not _dataFilePathList_.empty() ){
    out = new TChain(_dataTreeName_.c_str());
    for( const auto& file: _dataFilePathList_){
      if( not GenericToolbox::doesTFileIsValid(file, {_dataTreeName_}) ){
        LogError << "Invalid file: " << file << std::endl;
        throw std::runtime_error("Invalid file.");
      }
      out->Add(file.c_str());
    }
  }
  return out;
}
TChain* DataSet::buildMcChain(){
  return buildChain(false);
}
TChain* DataSet::buildDataChain(){
  return buildChain(true);
}
void DataSet::print() {
  LogInfo << _name_ << std::endl;
  if( _mcFilePathList_.empty() ){
    LogAlert << "No MC files loaded." << std::endl;
  }
  else{
    LogInfo << "List of MC files:" << std::endl;
    for( const auto& filePath : _mcFilePathList_ ){
      LogInfo << filePath << std::endl;
    }
    LogInfo << GET_VAR_NAME_VALUE(_mcNominalWeightFormulaStr_) << std::endl;
  }

  if( _dataFilePathList_.empty() ){
    LogInfo << "No External files loaded." << std::endl;
  }
  else{
    LogInfo << "List of External files:" << std::endl;
    for( const auto& filePath : _dataFilePathList_ ){
      LogInfo << filePath << std::endl;
    }
  }
}
