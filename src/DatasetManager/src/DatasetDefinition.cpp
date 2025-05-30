//
// Created by Nadrino on 22/07/2021.
//

#include "DatasetDefinition.h"

#include "GenericToolbox.Utils.h"

#include "GenericToolbox.Map.h"
#include "Logger.h"

#include "TTreeFormulaManager.h"


void DatasetDefinition::configureImpl() {

  _config_.defineFields({
    {"name", true},
    {"isEnabled"},
    {"model", true, {"mc"}},
    {"data"},
    {"selectedDataEntry"},
    {"selectedToyEntry"},
    {"showSelectedEventCount"},
    {"devSingleThreadEventLoaderAndIndexer"},
    {"devSingleThreadEventSelection"},
    {"sortLoadedEvents"},
    {"nbMaxThreadsForLoad"},
  });
  _config_.checkConfiguration();

  // mandatory
  _name_ = _config_.fetchValue<std::string>("name");

  // optional
  _config_.fillValue(_isEnabled_, "isEnabled");
  LogReturnIf(not _isEnabled_, "\"" << _name_ << "\" is disabled.");

  _modelDispenser_ = DataDispenser(this);
  _modelDispenser_.getParameters().name = "Asimov";
  _modelDispenser_.getParameters().useReweightEngine = true;
  _config_.fillValue(_modelDispenser_.getConfig(), "model");
  _modelDispenser_.configure();

  // Always put the Asimov as a data entry
  _dataDispenserDict_.emplace("Asimov", DataDispenser(_modelDispenser_));

  for( auto& dataEntry : _config_.loop("data") ){
    DataDispenser::prepareConfig(dataEntry);

    LogThrowIf(not dataEntry.hasField("name"), "name of a dataset is mandatory for \"data\".");
    auto name = dataEntry.fetchValue<std::string>("name");
    LogThrowIf( GenericToolbox::isIn(name, _dataDispenserDict_), "\"" << name << "\" already taken, please use another name." )

    _dataDispenserDict_.emplace(name, DataDispenser(this));
    _dataDispenserDict_.at(name).getParameters().isData = true;

    if( dataEntry.fetchValue("fromMc", false) ){
      _dataDispenserDict_.at(name).setConfig( _modelDispenser_.getConfig() );
    }

    // use override
    GenericToolbox::Json::applyOverrides(
      _dataDispenserDict_.at(name).getConfig().getConfig(),
      dataEntry.getConfig()
    );
    _dataDispenserDict_.at(name).configure();
  }

  _config_.fillValue(_selectedDataEntry_, "selectedDataEntry");
  _config_.fillValue(_selectedToyEntry_, "selectedToyEntry");
  _config_.fillValue(_showSelectedEventCount_, "showSelectedEventCount");
  _config_.fillValue(_devSingleThreadEventLoaderAndIndexer_, "devSingleThreadEventLoaderAndIndexer");
  _config_.fillValue(_devSingleThreadEventSelection_, "devSingleThreadEventSelection");
  _config_.fillValue(_sortLoadedEvents_, "sortLoadedEvents");
  _config_.fillValue(_nbMaxThreadsForLoad_, "nbMaxThreadsForLoad");

}
void DatasetDefinition::initializeImpl() {
  if( not _isEnabled_ ) return;

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
