#include "DatasetManager.h"

#include "Logger.h"

#include <cmath>
#include <map>

void DatasetManager::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({
    {"name"},
    {"datasetList"},
    {"datasetTemplateList"},
    {"debugLoadDatasetFraction"},
  });
  _config_.checkConfiguration();

  _debugLoadDatasetFraction_.reset();
  if( _config_.hasField("debugLoadDatasetFraction") ){
    setDebugLoadDatasetFraction(_config_.fetchValue<double>("debugLoadDatasetFraction"));
  }

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
    ConfigReader effectiveConfig(datasetConfig);
    effectiveConfig.defineFields({{"model", {"mc"}}, {"data"}});
    if( effectiveConfig.hasField("model") ){
      auto modelConfig = effectiveConfig.fetchValue<ConfigReader>("model");
      DataDispenser::applyTemplates(modelConfig, _datasetTemplateList_);
      effectiveConfig.getConfig().erase("mc");
      effectiveConfig.getConfig()["model"] = modelConfig.getConfig();
    }
    if( effectiveConfig.hasField("data") ){
      auto dataConfig = effectiveConfig.fetchValue<ConfigReader>("data");
      for( auto& entry : dataConfig.getConfig() ){
        ConfigReader dispenserConfig(entry);
        DataDispenser::applyTemplates(dispenserConfig, _datasetTemplateList_);
        entry = dispenserConfig.getConfig();
      }
      effectiveConfig.getConfig()["data"] = dataConfig.getConfig();
    }
    effectiveConfig.clearFields();
    _datasetList_.emplace_back(effectiveConfig, int(_datasetList_.size()));
  }
}

void DatasetManager::initializeImpl(){
  for( auto& dataset : _datasetList_ ){
    if( _debugLoadDatasetFraction_.has_value() ){
      dataset.setFractionOfEntriesToLoad(*_debugLoadDatasetFraction_);
    }
    dataset.initialize();
  }
}

void DatasetManager::setDebugLoadDatasetFraction(double fraction_){
  LogThrowIf(not std::isfinite(fraction_) or fraction_ < 0. or fraction_ > 1.,
             "debugLoadDatasetFraction must be finite and between 0 and 1");
  _debugLoadDatasetFraction_ = fraction_;
  LogInfo << "Will read " << fraction_ * 100. << "% of the entries in each dataset input tree." << std::endl;
}
