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

  // mandatory
  _name_ = GenericToolbox::Json::fetchValue<std::string>(_config_, "name");

  // optional
  GenericToolbox::Json::fillValue(_config_, "isEnabled", _isEnabled_);
  LogReturnIf(not _isEnabled_, "\"" << _name_ << "\" is disabled.");

  _modelDispenser_ = DataDispenser(this);
  GenericToolbox::Json::fillValue<JsonType>(_config_, {{"model"}, {"mc"}}, _modelDispenser_.getConfig());
  _modelDispenser_.readConfig();
  _modelDispenser_.getParameters().name = "Asimov";
  _modelDispenser_.getParameters().useReweightEngine = true;

  // Always put the Asimov as a data entry
  _dataDispenserDict_.emplace("Asimov", DataDispenser(_modelDispenser_));

  for( auto& dataEntry : GenericToolbox::Json::fetchValue(_config_, "data", JsonType()) ){
    auto name = GenericToolbox::Json::fetchValue<std::string>(dataEntry, "name");
    LogThrowIf( GenericToolbox::isIn(name, _dataDispenserDict_), "\"" << name << "\" already taken, please use another name." )

    if( GenericToolbox::Json::fetchValue(dataEntry, "fromMc", bool(false)) ){ _dataDispenserDict_.emplace(name, _modelDispenser_); }
    else{ _dataDispenserDict_.emplace(name, DataDispenser(this)); }
    _dataDispenserDict_.at(name).readConfig(dataEntry);
    _dataDispenserDict_.at(name).getParameters().isData = true;
  }

  GenericToolbox::Json::fillValue(_config_, "selectedDataEntry", _selectedDataEntry_);
  GenericToolbox::Json::fillValue(_config_, "selectedToyEntry", _selectedToyEntry_);
  GenericToolbox::Json::fillValue(_config_, "showSelectedEventCount", _showSelectedEventCount_);
  GenericToolbox::Json::fillValue(_config_, "devSingleThreadEventLoaderAndIndexer", _devSingleThreadEventLoaderAndIndexer_);
  GenericToolbox::Json::fillValue(_config_, "devSingleThreadEventSelection", _devSingleThreadEventSelection_);
  GenericToolbox::Json::fillValue(_config_, "sortLoadedEvents", _sortLoadedEvents_);

}
void DatasetDefinition::initializeImpl() {
  if( not _isEnabled_ ) return;
  LogInfo << "Initializing dataset: \"" << _name_ << "\"" << std::endl;

  _modelDispenser_.initialize();
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
  _modelDispenser_.setOwner(this);
  for( auto& dispenser : _dataDispenserDict_ ){
    dispenser.second.setOwner(this);
  }
}
