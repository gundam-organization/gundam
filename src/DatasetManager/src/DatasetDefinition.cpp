//
// Created by Nadrino on 22/07/2021.
//

#include "DatasetDefinition.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Json.h"
#include "GenericToolbox.Map.h"
#include "Logger.h"

#include "TTreeFormulaManager.h"


#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[DataSetLoader]"); });
#endif


void DatasetDefinition::readConfigImpl() {
  LogThrowIf(_config_.empty(), "Config not set.");
  _name_ = GenericToolbox::Json::fetchValue<std::string>(_config_, "name");
  LogInfo << "Reading config for dataset: \"" << _name_ << "\"" << std::endl;

  _isEnabled_ = GenericToolbox::Json::fetchValue(_config_, "isEnabled", bool(true));
  LogReturnIf(not _isEnabled_, "\"" << _name_ << "\" is disabled.");

  _selectedDataEntry_ = GenericToolbox::Json::fetchValue<std::string>(_config_, "selectedDataEntry", "Asimov");
  _selectedToyEntry_ = GenericToolbox::Json::fetchValue<std::string>(_config_, "selectedToyEntry", "Asimov");

  _showSelectedEventCount_ = GenericToolbox::Json::fetchValue(_config_, "showSelectedEventCount", _showSelectedEventCount_);

  _mcDispenser_ = DataDispenser(this);
  _mcDispenser_.readConfig(GenericToolbox::Json::fetchValue<JsonType>(_config_, {{"model"}, {"mc"}}));
  _mcDispenser_.getParameters().name = "Asimov";
  _mcDispenser_.getParameters().useMcContainer = true;

  // Always loaded by default
  _dataDispenserDict_.emplace("Asimov", DataDispenser(_mcDispenser_));

  for( auto& dataEntry : GenericToolbox::Json::fetchValue(_config_, "data", JsonType()) ){
    std::string name = GenericToolbox::Json::fetchValue(dataEntry, "name", "data");
    LogThrowIf( GenericToolbox::isIn(name, _dataDispenserDict_),
                "\"" << name << "\" already taken, please use another name." )

    if( GenericToolbox::Json::fetchValue(dataEntry, "fromMc", bool(false)) ){ _dataDispenserDict_.emplace(name, _mcDispenser_); }
    else{ _dataDispenserDict_.emplace(name, DataDispenser(this)); }
    _dataDispenserDict_.at(name).readConfig(dataEntry);
  }

  _devSingleThreadEventLoaderAndIndexer_ = GenericToolbox::Json::fetchValue(_config_, "devSingleThreadEventLoaderAndIndexer", _devSingleThreadEventLoaderAndIndexer_);
  _devSingleThreadEventSelection_ = GenericToolbox::Json::fetchValue(_config_, "devSingleThreadEventSelection", _devSingleThreadEventSelection_);
  _sortLoadedEvents_ = GenericToolbox::Json::fetchValue(_config_, "sortLoadedEvents", _sortLoadedEvents_);

}
void DatasetDefinition::initializeImpl() {
  if( not _isEnabled_ ) return;
  LogInfo << "Initializing dataset: \"" << _name_ << "\"" << std::endl;

  _mcDispenser_.initialize();
  for( auto& dataDispenser : _dataDispenserDict_ ){
    dataDispenser.second.initialize();
  }

  if( not GenericToolbox::isIn(_selectedDataEntry_, _dataDispenserDict_) ){
    LogThrow("selectedDataEntry could not be find in available data: "
                 << GenericToolbox::toString(_dataDispenserDict_, [](const std::pair<std::string, DataDispenser>& elm){ return elm.first; })
                 << std::endl);
  }
}

void DatasetDefinition::updateDispenserOwnership(){
  _mcDispenser_.setOwner(this);
  for( auto& dispenser : _dataDispenserDict_ ){
    dispenser.second.setOwner(this);
  }
}
