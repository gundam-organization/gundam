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

  // read JSON/YAML
  JsonType readConfigFile(const std::string& configFilePath_);

  // converting YAML to JSON
  JsonType convertYamlToJson(const std::string& configFilePath_);
  JsonType convertYamlToJson(const YAML::Node& yamlConfig_);

  // unfolding JSON/YAML
  JsonType getForwardedConfig(const JsonType& config_);
  JsonType getForwardedConfig(const JsonType& config_, const std::string& keyName_);
  void forwardConfig(JsonType& config_);
  void unfoldConfig(JsonType& config_);


  // handle all the hard work for us
  class ConfigHandler{

  public:
    explicit ConfigHandler(const std::string& filePath_);
    explicit ConfigHandler(JsonType  config_): config(std::move(config_)) {}

    // const-getters
    [[nodiscard]] std::string toString() const{ return GenericToolbox::Json::toReadableString( config ); }
    [[nodiscard]] const JsonType &getConfig() const{ return config; }

    // mutable getters
    JsonType &getConfig(){ return config; }

    // core
    void override( const JsonType& overrideConfig_ );
    void override( const std::string& filePath_ );
    void override( const std::vector<std::string>& filesList_ );
    void flatOverride( const std::string& flattenEntry_ );
    void flatOverride( const std::vector<std::string>& flattenEntryList_ );
    void exportToJsonFile( const std::string& filePath_ ) const;

  private:
    JsonType config{};

  };

}

#endif //GUNDAM_CONFIG_UTILS_H
