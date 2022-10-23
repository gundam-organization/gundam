//
// Created by Nadrino on 22/07/2021.
//

#include "DatasetLoader.h"

#include "DialSet.h"
#include "GlobalVariables.h"
#include "GraphDial.h"
#include "SplineDial.h"
#include "JsonUtils.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.VariablesMonitor.h"
#include "Logger.h"

#include <TTreeFormulaManager.h>
#include "TTree.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[DataSetLoader]");
});


void DatasetLoader::readConfigImpl() {
  LogThrowIf(_config_.empty(), "Config not set.");
  _name_ = JsonUtils::fetchValue<std::string>(_config_, "name");
  LogInfo << "Reading config for dataset: \"" << _name_ << "\"" << std::endl;

  _isEnabled_ = JsonUtils::fetchValue(_config_, "isEnabled", true);
  LogReturnIf(not _isEnabled_, "\"" << _name_ << "\" is disabled.");

  _selectedDataEntry_ = JsonUtils::fetchValue<std::string>(_config_, "selectedDataEntry", "Asimov");
  _selectedToyEntry_ = JsonUtils::fetchValue<std::string>(_config_, "selectedToyEntry", "Asimov");

  _showSelectedEventCount_ = JsonUtils::fetchValue(_config_, "showSelectedEventCount", _showSelectedEventCount_);

  _mcDispenser_ = DataDispenser(JsonUtils::fetchValue<nlohmann::json>(_config_, "mc"), this);
  _mcDispenser_.getParameters().name = "Asimov";
  _mcDispenser_.getParameters().useMcContainer = true;

  // Always loaded by default
  _dataDispenserDict_["Asimov"] = _mcDispenser_;

  for( auto& dataEntry : JsonUtils::fetchValue(_config_, "data", nlohmann::json()) ){
    std::string name = JsonUtils::fetchValue(dataEntry, "name", "data");
    LogThrowIf( GenericToolbox::doesKeyIsInMap(name, _dataDispenserDict_),
                "\"" << name << "\" already taken, please use another name." )

    if( JsonUtils::fetchValue(dataEntry, "fromMc", false) ){ _dataDispenserDict_[name] = _mcDispenser_; }
    else{ _dataDispenserDict_[name] = DataDispenser(dataEntry, this); }
    _dataDispenserDict_[name].getParameters().name = name;
  }
}
void DatasetLoader::initializeImpl() {
  if( not _isEnabled_ ) return;
  LogInfo << "Initializing dataset: \"" << _name_ << "\"" << std::endl;

  _mcDispenser_.initialize();
  for( auto& dataDispenser : _dataDispenserDict_ ){ dataDispenser.second.initialize(); }

  if( not GenericToolbox::doesKeyIsInMap(_selectedDataEntry_, _dataDispenserDict_) ){
    LogThrow("selectedDataEntry could not be find in available data: "
                 << GenericToolbox::iterableToString(_dataDispenserDict_, [](const std::pair<std::string, DataDispenser>& elm){ return elm.first; })
                 << std::endl);
  }
}

DatasetLoader::DatasetLoader(const nlohmann::json& config_, int datasetIndex_): _dataSetIndex_(datasetIndex_) {
  this->readConfig(config_);
}

void DatasetLoader::setDataSetIndex(int dataSetIndex) {
  _dataSetIndex_ = dataSetIndex;
}

bool DatasetLoader::isEnabled() const {
  return _isEnabled_;
}
const std::string &DatasetLoader::getName() const {
  return _name_;
}
int DatasetLoader::getDataSetIndex() const {
  return _dataSetIndex_;
}

DataDispenser &DatasetLoader::getMcDispenser() {
  return _mcDispenser_;
}
DataDispenser &DatasetLoader::getSelectedDataDispenser(){
  return _dataDispenserDict_[_selectedDataEntry_];
}
DataDispenser &DatasetLoader::getToyDataDispenser(){
  return _dataDispenserDict_[_selectedToyEntry_];
}
std::map<std::string, DataDispenser> &DatasetLoader::getDataDispenserDict() {
  return _dataDispenserDict_;
}

const std::string &DatasetLoader::getSelectedDataEntry() const {
  return _selectedDataEntry_;
}
const std::string &DatasetLoader::getToyDataEntry() const {
  return _selectedToyEntry_;
}

bool DatasetLoader::isShowSelectedEventCount() const {
  return _showSelectedEventCount_;
}
