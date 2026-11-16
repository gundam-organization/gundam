#include "DatasetManager.h"

#include "Logger.h"

#include <map>

void DatasetManager::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({
    {"name"},
    {"datasetList"},
    {"datasetTemplateList"},
  });
  _config_.checkConfiguration();

  _datasetTemplateList_.clear();
  std::map<std::string, size_t> templateIndexDict;
  for( const auto& templateConfig : _config_.loop("datasetTemplateList") ){
    _datasetTemplateList_.emplace_back();
    _datasetTemplateList_.back().configure(templateConfig);
    const auto& name = _datasetTemplateList_.back().getName();
    LogThrowIf(not templateIndexDict.emplace(name, _datasetTemplateList_.size()-1).second,
               "Duplicate dataset template name: " << name);
  }

  ConfigReader datasetListConfig{};
  _config_.fillValue(datasetListConfig, "datasetList");
  _datasetList_.clear();
  _datasetList_.reserve(datasetListConfig.getConfig().size());
  for( const auto& datasetConfig : datasetListConfig.loop() ){
    ConfigReader datasetReader(datasetConfig);
    datasetReader.defineFields({{"useTemplates"}});
    ConfigUtils::ConfigBuilder configBuilder;
    for( auto& templateReference : datasetReader.loop("useTemplates") ){
      templateReference.clearFields();
      templateReference.defineFields({
        {FieldFlag::MANDATORY, "name"},
        {"isEnabled"},
      });
      templateReference.checkConfiguration();
      if( not templateReference.fetchValue("isEnabled", true) ){ continue; }
      const auto name = templateReference.fetchValue<std::string>("name");
      auto templateIndex = templateIndexDict.find(name);
      LogThrowIf(templateIndex == templateIndexDict.end(), "Unknown dataset template: " << name);
      auto templateConfig = _datasetTemplateList_.at(templateIndex->second).getConfig().getConfig();
      templateConfig.erase("name");
      configBuilder.override(templateConfig);
    }
    auto ownConfig = datasetConfig.getConfig();
    ownConfig.erase("useTemplates");
    configBuilder.override(ownConfig);
    ConfigReader effectiveConfig(configBuilder.getConfig());
    effectiveConfig.setParentPath(datasetConfig.getParentPath());
    _datasetList_.emplace_back(effectiveConfig, int(_datasetList_.size()));
  }
}

void DatasetManager::initializeImpl(){
  for( auto& dataset : _datasetList_ ){ dataset.initialize(); }
}
