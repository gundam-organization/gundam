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
  void forwardConfig(JsonType& config_);
  void unfoldConfig(JsonType& config_);

  /// Check that the config only contains fields in the allowed_ vector, and
  /// has all of the fields in the expected_ vector.  Fields that are in the
  /// "deprecated_ vector will generate an warning, but are still considered
  /// valid.
  bool checkFields(JsonType& config_,
                   std::string parent_,
                   std::vector<std::string> allowed_,
                   std::vector<std::string> expected_ = {},
                   std::vector<std::string> deprecated_ = {},
                   std::vector<std::pair<std::string,std::string>>
                   replaced_ = {});

  // handle all the hard work for us
  class ConfigHandler{

  public:
    ConfigHandler() = default;
    explicit ConfigHandler(const std::string& filePath_){ setConfig(filePath_); }
    explicit ConfigHandler(const JsonType& config_): config(config_) {}

    // setters
    void setConfig(const std::string& filePath_);
    void setConfig(const JsonType& config_){ config = config_; }

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
