//
// Created by Nadrino on 22/07/2021.
//

#include "DatasetDefinition.h"

#include "GenericToolbox.Utils.h"

#include "GenericToolbox.Map.h"
#include "Logger.h"

#include "TTreeFormulaManager.h"


void DatasetDefinition::configureImpl() {

  // All of the fields that should (or may) be at this level in the YAML.
  // This provides a rudimentary syntax check for user inputs.
  ConfigUtils::checkFields(_config_,
                           "/fitterEngineConfig/likelihoodInterfaceConfig"
                           "/datasetList/(dataset)",
                           // Allowed fields (don't need to list fields in
                           // expected, or deprecated).
                           {
                             {"isEnabled"},
                             {"selectedDataEntry"},
                             {"selectedToyEntry"},
                             {"showSelectedEventCount"},
                             {"devSingleThreadEventLoaderAndIndexer"},
                             {"devSingleThreadEventSelection"},
                             {"sortLoadedEvents"},
                             {"nbMaxThreadsForLoad"},
                           },
                           // Expected fields (must be present)
                           {
                             {"name"},
                             {"model"},
                             {"data"},
                           },
                           // Deprecated fields (allowed, but cause a warning)
                           {},
                           // Replaced fields (allowed, but cause a warning)
                           {
                             {{"mc"}, {"model"}}
                           });

  // mandatory
  _name_ = GenericToolbox::Json::fetchValue<std::string>(_config_, "name");

  // optional
  GenericToolbox::Json::fillValue(_config_, _isEnabled_, "isEnabled");
  LogReturnIf(not _isEnabled_, "\"" << _name_ << "\" is disabled.");

  _modelDispenser_ = DataDispenser(this);
  _modelDispenser_.getParameters().name = "Asimov";
  _modelDispenser_.getParameters().useReweightEngine = true;
  GenericToolbox::Json::fillValue<JsonType>(_config_, _modelDispenser_.getConfig(), {{"model"},{"mc"}});
  _modelDispenser_.configure();

  // Always put the Asimov as a data entry
  _dataDispenserDict_.emplace("Asimov", DataDispenser(_modelDispenser_));

  for( auto& dataEntry : GenericToolbox::Json::fetchValue(_config_, "data", JsonType()) ){
    auto name = GenericToolbox::Json::fetchValue<std::string>(dataEntry, "name");
    LogThrowIf( GenericToolbox::isIn(name, _dataDispenserDict_), "\"" << name << "\" already taken, please use another name." )

    _dataDispenserDict_.emplace(name, DataDispenser(this));
    _dataDispenserDict_.at(name).getParameters().isData = true;

    if( GenericToolbox::Json::fetchValue(dataEntry, "fromMc", bool(false)) ){
      _dataDispenserDict_.at(name).setConfig( _modelDispenser_.getConfig() );
    }

    // use override
    GenericToolbox::Json::applyOverrides( _dataDispenserDict_.at(name).getConfig(), dataEntry );
    _dataDispenserDict_.at(name).configure();
  }

  GenericToolbox::Json::fillValue(_config_, _selectedDataEntry_, "selectedDataEntry");
  GenericToolbox::Json::fillValue(_config_, _selectedToyEntry_, "selectedToyEntry");
  GenericToolbox::Json::fillValue(_config_, _showSelectedEventCount_, "showSelectedEventCount");
  GenericToolbox::Json::fillValue(_config_, _devSingleThreadEventLoaderAndIndexer_, "devSingleThreadEventLoaderAndIndexer");
  GenericToolbox::Json::fillValue(_config_, _devSingleThreadEventSelection_, "devSingleThreadEventSelection");
  GenericToolbox::Json::fillValue(_config_, _sortLoadedEvents_, "sortLoadedEvents");
  GenericToolbox::Json::fillValue(_config_, _nbMaxThreadsForLoad_, "nbMaxThreadsForLoad");

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
