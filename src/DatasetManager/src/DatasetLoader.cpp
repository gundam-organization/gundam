//
// Created by Nadrino on 22/07/2021.
//

#include "DatasetLoader.h"

#include "GlobalVariables.h"
#include "GenericToolbox.Json.h"

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
  _name_ = GenericToolbox::Json::fetchValue<std::string>(_config_, "name");
  LogInfo << "Reading config for dataset: \"" << _name_ << "\"" << std::endl;

  _isEnabled_ = GenericToolbox::Json::fetchValue(_config_, "isEnabled", true);
  LogReturnIf(not _isEnabled_, "\"" << _name_ << "\" is disabled.");

  _selectedDataEntry_ = GenericToolbox::Json::fetchValue<std::string>(_config_, "selectedDataEntry", "Asimov");
  _selectedToyEntry_ = GenericToolbox::Json::fetchValue<std::string>(_config_, "selectedToyEntry", "Asimov");

  _showSelectedEventCount_ = GenericToolbox::Json::fetchValue(_config_, "showSelectedEventCount", _showSelectedEventCount_);

  _mcDispenser_ = DataDispenser(this);
  _mcDispenser_.readConfig(GenericToolbox::Json::fetchValue<nlohmann::json>(_config_, "mc"));
  _mcDispenser_.getParameters().name = "Asimov";
  _mcDispenser_.getParameters().useMcContainer = true;

  // Always loaded by default
  _dataDispenserDict_.emplace("Asimov", DataDispenser(_mcDispenser_));

  for( auto& dataEntry : GenericToolbox::Json::fetchValue(_config_, "data", nlohmann::json()) ){
    std::string name = GenericToolbox::Json::fetchValue(dataEntry, "name", "data");
    LogThrowIf( GenericToolbox::doesKeyIsInMap(name, _dataDispenserDict_),
                "\"" << name << "\" already taken, please use another name." )

    if( GenericToolbox::Json::fetchValue(dataEntry, "fromMc", false) ){ _dataDispenserDict_.emplace(name, _mcDispenser_); }
    else{ _dataDispenserDict_.emplace(name, DataDispenser(this)); }
    _dataDispenserDict_.at(name).readConfig(dataEntry);
  }

  _devSingleThreadEventLoaderAndIndexer_ = GenericToolbox::Json::fetchValue(_config_, "devSingleThreadEventLoaderAndIndexer", _devSingleThreadEventLoaderAndIndexer_);
  _devSingleThreadEventSelection_ = GenericToolbox::Json::fetchValue(_config_, "devSingleThreadEventSelection", _devSingleThreadEventSelection_);
  _sortLoadedEvents_ = GenericToolbox::Json::fetchValue(_config_, "sortLoadedEvents", _sortLoadedEvents_);

}
void DatasetLoader::initializeImpl() {
  if( not _isEnabled_ ) return;
  LogInfo << "Initializing dataset: \"" << _name_ << "\"" << std::endl;

  _mcDispenser_.initialize();
  for( auto& dataDispenser : _dataDispenserDict_ ){
    dataDispenser.second.initialize();
  }

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
bool DatasetLoader::isShowSelectedEventCount() const {
  return _showSelectedEventCount_;
}
int DatasetLoader::getDataSetIndex() const {
  return _dataSetIndex_;
}
const std::string &DatasetLoader::getName() const {
  return _name_;
}
const std::string &DatasetLoader::getSelectedDataEntry() const {
  return _selectedDataEntry_;
}
const std::string &DatasetLoader::getToyDataEntry() const {
  return _selectedToyEntry_;
}
bool DatasetLoader::isDevSingleThreadEventLoaderAndIndexer() const {
  return _devSingleThreadEventLoaderAndIndexer_;
}
bool DatasetLoader::isDevSingleThreadEventSelection() const {
  return _devSingleThreadEventSelection_;
}
bool DatasetLoader::isSortLoadedEvents() const {
  return _sortLoadedEvents_;
}

DataDispenser &DatasetLoader::getMcDispenser() {
  return _mcDispenser_;
}
DataDispenser &DatasetLoader::getSelectedDataDispenser(){
  return _dataDispenserDict_.at(_selectedDataEntry_);
}
DataDispenser &DatasetLoader::getToyDataDispenser(){
  return _dataDispenserDict_.at(_selectedToyEntry_);
}
std::map<std::string, DataDispenser> &DatasetLoader::getDataDispenserDict() {
  return _dataDispenserDict_;
}

void DatasetLoader::updateDispenserOwnership(){
  _mcDispenser_.setOwner(this);
  for( auto& dispenser : _dataDispenserDict_ ){
    dispenser.second.setOwner(this);
  }
}
