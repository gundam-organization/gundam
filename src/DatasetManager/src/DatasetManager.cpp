#include "DatasetManager.h"

void DatasetManager::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({
    {"name"},
    {"datasetList"},
  });
  _config_.checkConfiguration();

  ConfigReader datasetListConfig{};
  _config_.fillValue(datasetListConfig, "datasetList");
  _datasetList_.clear();
  _datasetList_.reserve(datasetListConfig.getConfig().size());
  for( const auto& datasetConfig : datasetListConfig.loop() ){
    _datasetList_.emplace_back(datasetConfig, int(_datasetList_.size()));
  }
}

void DatasetManager::initializeImpl(){
  for( auto& dataset : _datasetList_ ){ dataset.initialize(); }
}
