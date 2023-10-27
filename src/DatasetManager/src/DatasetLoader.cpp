//
// Created by Nadrino on 22/07/2021.
//

#include "DatasetLoader.h"

#include "GundamGlobals.h"
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

  LogDebug << __LINE__ << std::endl;

  _selectedDataEntry_ = GenericToolbox::Json::fetchValue<std::string>(_config_, "selectedDataEntry", "Asimov");
  _selectedToyEntry_ = GenericToolbox::Json::fetchValue<std::string>(_config_, "selectedToyEntry", "Asimov");

  LogDebug << __LINE__ << std::endl;
  _showSelectedEventCount_ = GenericToolbox::Json::fetchValue(_config_, "showSelectedEventCount", _showSelectedEventCount_);

  LogDebug << __LINE__ << std::endl;
  _mcDispenser_ = DataDispenser(this);
  _mcDispenser_.readConfig(GenericToolbox::Json::fetchValue<nlohmann::json>(_config_, "mc"));
  _mcDispenser_.getParameters().name = "Asimov";
  _mcDispenser_.getParameters().useMcContainer = true;

  LogDebug << __LINE__ << std::endl;
  // Always loaded by default
  _dataDispenserDict_.emplace("Asimov", DataDispenser(_mcDispenser_));

  LogDebug << __LINE__ << std::endl;
  for( auto& dataEntry : GenericToolbox::Json::fetchValue(_config_, "data", nlohmann::json()) ){
    std::string name = GenericToolbox::Json::fetchValue(dataEntry, "name", "data");
    LogThrowIf( GenericToolbox::doesKeyIsInMap(name, _dataDispenserDict_),
                "\"" << name << "\" already taken, please use another name." )

    if( GenericToolbox::Json::fetchValue(dataEntry, "fromMc", false) ){ _dataDispenserDict_.emplace(name, _mcDispenser_); }
    else{ _dataDispenserDict_.emplace(name, DataDispenser(this)); }
    _dataDispenserDict_.at(name).readConfig(dataEntry);
  }

  LogDebug << __LINE__ << std::endl;
  _devSingleThreadEventLoaderAndIndexer_ = GenericToolbox::Json::fetchValue(_config_, "devSingleThreadEventLoaderAndIndexer", _devSingleThreadEventLoaderAndIndexer_);
  _devSingleThreadEventSelection_ = GenericToolbox::Json::fetchValue(_config_, "devSingleThreadEventSelection", _devSingleThreadEventSelection_);
  _sortLoadedEvents_ = GenericToolbox::Json::fetchValue(_config_, "sortLoadedEvents", _sortLoadedEvents_);

  LogDebug << __LINE__ << std::endl;
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

void DatasetLoader::updateDispenserOwnership(){
  _mcDispenser_.setOwner(this);
  for( auto& dispenser : _dataDispenserDict_ ){
    dispenser.second.setOwner(this);
  }
}
