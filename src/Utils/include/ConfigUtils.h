//
// Created by Adrien Blanchet on 28/02/2023.
//

#ifndef GUNDAM_CONFIG_UTILS_H
#define GUNDAM_CONFIG_UTILS_H


#include "GenericToolbox.Json.h"

#include "yaml-cpp/yaml.h"

#include <string>


// shortcuts
typedef GenericToolbox::Json::JsonType JsonType;
typedef GenericToolbox::Json::ConfigBaseClass JsonBaseClass;


namespace ConfigUtils {

  // could be YAML or JSON
  JsonType readConfigFile(const std::string& configFilePath_);
  JsonType convertYamlToJson(const std::string& configFilePath_);
  JsonType convertYamlToJson(const YAML::Node& yamlConfig_);

  // make sure both YAML and JSON are supported
  JsonType getForwardedConfig(const JsonType& config_);
  JsonType getForwardedConfig(const JsonType& config_, const std::string& keyName_);
  void forwardConfig(JsonType& config_);
  void unfoldConfig(JsonType& config_);

  class ConfigHandler{
    JsonType config{};

  public:
    explicit ConfigHandler(const std::string& filePath_);
    explicit ConfigHandler(JsonType  config_);

    // const-getters
    [[nodiscard]] std::string toString() const;
    [[nodiscard]] const JsonType &getConfig() const;

    // non-const getters
    JsonType &getConfig();


    // config actions
    void override( const JsonType& overrideConfig_ );
    void override( const std::string& filePath_ );
    void override( const std::vector<std::string>& filesList_ );

    void flatOverride( const std::string& flattenEntry_ );
    void flatOverride( const std::vector<std::string>& flattenEntryList_ );

    void exportToJsonFile( const std::string& filePath_ ) const;

  };

}

#endif //GUNDAM_CONFIG_UTILS_H
