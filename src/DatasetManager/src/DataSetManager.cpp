//
// Created by Nadrino on 04/03/2024.
//

#include "DataSetManager.h"

#include "Logger.h"

LoggerInit([]{
  Logger::getUserHeader() << "[DataSetManager]";
});


void DataSetManager::readConfigImpl(){
  LogInfo << "Reading DataSetManager configurations..." << std::endl;

  // check if config is pointing to another file
  GenericToolbox::Json::forwardConfig(_config_);

  // dataSetList should be present
  JsonType dataSetList{ GenericToolbox::Json::fetchValue<JsonType>(_config_, "dataSetList") };

  // creating the dataSets:
  _dataSetList_.reserve( dataSetList.size() );
  for( const auto& dataSetConfig : dataSetList ){
    _dataSetList_.emplace_back(dataSetConfig, int(_dataSetList_.size()));
  }

  // deprecated config files will already have filled up _treeWriter_.getConfig()
  _treeWriter_.readConfig( GenericToolbox::Json::fetchValue(_config_, "eventTreeWriter", _treeWriter_.getConfig()) );
}
void DataSetManager::initializeImpl(){
  LogInfo << "Initializing DataSetManager..." << std::endl;

  for( auto& dataSet : _dataSetList_ ){ dataSet.initialize(); }
  _treeWriter_.initialize();
}

