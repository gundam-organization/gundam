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
    {FieldFlag::MANDATORY, "name"},
    {FieldFlag::MANDATORY, "model", {"mc"}},
    {"isEnabled"},
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

    ConfigUtils::ConfigBuilder dataConfigBuilder;

    // use model as a base if "fromModel" is set
    if( dataEntry.fetchValue("fromModel", false) ){
      LogInfo << name << " dataset will inherit from Model definition." << std::endl;
      dataConfigBuilder.setConfig( _modelDispenser_.getConfig().getConfig() );
      LogDebug << "Base config:" << GenericToolbox::Json::toReadableString(dataConfigBuilder.getConfig()) << std::endl;
    }

    LogDebug << "Override config:" << GenericToolbox::Json::toReadableString(dataEntry.getConfig()) << std::endl;


    // use override
    dataConfigBuilder.override( dataEntry.getConfig() );
    ConfigReader cr(dataConfigBuilder.getConfig());
    cr.setParentPath(dataEntry.getParentPath());
    _dataDispenserDict_.at(name).setConfig( cr );

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
