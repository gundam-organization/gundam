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
  _config_.clear();
  _isEnabled_ = false;

  _name_ = "";
  _enabledLeafNameList_.clear();
  _mcFilePathList_.clear();
  _dataFilePathList_.clear();
}

void DataSet::setConfig(const nlohmann::json &config_) {
  _config_ = config_;
  JsonUtils::forwardConfig(_config_, __CLASS_NAME__);
}
void DataSet::addRequestedLeafName(const std::string& leafName_){
  if( not leafName_.empty() and not GenericToolbox::doesElementIsInVector(leafName_, _enabledLeafNameList_) ){
    LogThrowIf( _mcChain_->GetLeaf(leafName_.c_str()) == nullptr,
                "\"" << leafName_ << "\" not defined in the TChain of dataSet: " << _name_ );
    _enabledLeafNameList_.emplace_back(leafName_);
  }
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

  auto mcConfig = JsonUtils::fetchValue(_config_, "mc", nlohmann::json());
  if( not mcConfig.empty() ){
    _mcTreeName_ = JsonUtils::fetchValue<std::string>(mcConfig, "tree");
    auto fileList = JsonUtils::fetchValue(mcConfig, "filePathList", nlohmann::json());
    for( const auto& file: fileList ){
      _mcFilePathList_.emplace_back(file.get<std::string>());
    }
  }

  auto dataConfig = JsonUtils::fetchValue(_config_, "data", nlohmann::json());
  if( not dataConfig.empty() ){
    _dataTreeName_ = JsonUtils::fetchValue<std::string>(dataConfig, "tree");
    auto fileList = JsonUtils::fetchValue(dataConfig, "filePathList", nlohmann::json());
    for( const auto& file: fileList ){
      _dataFilePathList_.emplace_back(file.get<std::string>());
    }
  }

  this->initializeChains();
  this->print();
}


bool DataSet::isEnabled() const {
  return _isEnabled_;
}
const std::string &DataSet::getName() const {
  return _name_;
}
std::shared_ptr<TChain> &DataSet::getMcChain() {
  return _mcChain_;
}
std::shared_ptr<TChain> &DataSet::getDataChain() {
  return _dataChain_;
}
const std::vector<std::string> &DataSet::getEnabledLeafNameList() const {
  return _enabledLeafNameList_;
}
const std::vector<std::string> &DataSet::getMcFilePathList() const {
  return _mcFilePathList_;
}
const std::vector<std::string> &DataSet::getDataFilePathList() const {
  return _dataFilePathList_;
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

void DataSet::initializeChains() {

  if( not _mcFilePathList_.empty() ){
    _mcChain_ = std::shared_ptr<TChain>(new TChain(_mcTreeName_.c_str()));
    for( const auto& file: _mcFilePathList_){
      if( not GenericToolbox::doesTFileIsValid(file, {_mcTreeName_}) ){
        LogError << "Invalid file: " << file << std::endl;
        throw std::runtime_error("Invalid file.");
      }
      _mcChain_->Add(file.c_str());
    }
  }

  if( not _dataFilePathList_.empty() ){
    _dataChain_ = std::shared_ptr<TChain>(new TChain(_dataTreeName_.c_str()));
    for( const auto& file: _dataFilePathList_){
      if( not GenericToolbox::doesTFileIsValid(file, {_dataTreeName_}) ){
        LogError << "Invalid file: " << file << std::endl;
        throw std::runtime_error("Invalid file.");
      }
      _dataChain_->Add(file.c_str());
    }
  }

}


