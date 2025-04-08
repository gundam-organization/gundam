//
// Created by Adrien Blanchet on 28/02/2023.
//

#ifndef GUNDAM_CONFIG_UTILS_H
#define GUNDAM_CONFIG_UTILS_H

#include <Logger.h>

#include "GenericToolbox.Json.h"

#include "yaml-cpp/yaml.h"

#include <string>
#include <utility>

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
    explicit ConfigHandler(JsonType config_): _config_(std::move(config_)) {}

    // setters
    void setConfig(const std::string& filePath_);
    void setConfig(const JsonType& config_){ _config_ = config_; }

    // const-getters
    [[nodiscard]] std::string toString() const{ return GenericToolbox::Json::toReadableString( _config_ ); }
    [[nodiscard]] const JsonType &getConfig() const{ return _config_; }

    // mutable getters
    JsonType &getConfig(){ return _config_; }

    // core
    void override( const JsonType& overrideConfig_ );
    void override( const std::string& filePath_ );
    void override( const std::vector<std::string>& filesList_ );
    void flatOverride( const std::string& flattenEntry_ );
    void flatOverride( const std::vector<std::string>& flattenEntryList_ );
    void exportToJsonFile( const std::string& filePath_ ) const;

    // read options
    template<typename T> void fillValue(T& object_, const std::string& keyPath_);
    template<typename T> void fillValue(T& object_, const std::vector<std::string> &keyPathList_);

    void printUnusedOptions() const;

  private:
    JsonType _config_{};

    // keep track of fields that have been red
    std::vector<std::string> _usedKeyList_{};

  };


  template<typename T> void ConfigHandler::fillValue(T& object_, const std::string& keyPath_){
    if( GenericToolbox::Json::doKeyExist(_config_, keyPath_) ) {
      object_ = GenericToolbox::Json::fetchValue<T>(_config_, keyPath_);
      _usedKeyList_.emplace_back(keyPath_);
    }
  }
  template<typename T> void ConfigHandler::fillValue(T& object_, const std::vector<std::string> &keyPathList_){

    // keyPathList_ has all the possible names for a given option
    // the first one is the official one, when others are set a message will appear telling the user it's deprecated

    bool alreadyFound{false};
    for( auto& keyPath : keyPathList_ ) {
      if( GenericToolbox::Json::doKeyExist(_config_, keyPath) ) {
        if( keyPath != keyPathList_.front() ) {
          LogAlert << "\"" << keyPath << "\" is a deprecated field name, use \"" << keyPathList_.front() << "\" instead." << std::endl;
        }

        auto temp = GenericToolbox::Json::fetchValue<T>(_config_, keyPath);
        _usedKeyList_.emplace_back(keyPath);

        if( alreadyFound and temp != object_ ){
          LogError << "\"" << keyPath << "\" returned: " << temp << std::endl;
          LogError << "while it has been already set with: " << _usedKeyList_[_usedKeyList_.size()-2] << " -> " << object_ << std::endl;
          LogExit("Two config options with different values.");
        }

        alreadyFound = true;
      }
    }
  }

}

#endif //GUNDAM_CONFIG_UTILS_H
