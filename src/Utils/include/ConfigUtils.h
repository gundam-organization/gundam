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
// typedef GenericToolbox::Json::ConfigBaseClass JsonBaseClass;

namespace ConfigUtils{ class ConfigReader; }
typedef GenericToolbox::ConfigClass<ConfigUtils::ConfigReader> JsonBaseClass;


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

  // handle all the hard work for us
  class ConfigBuilder{

  public:
    ConfigBuilder() = default;
    explicit ConfigBuilder(const std::string& filePath_){ setConfig(filePath_); }
    explicit ConfigBuilder(JsonType config_): _config_(std::move(config_)) {}

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

  private:
    JsonType _config_{};

  };

  class ConfigReader{

  public:
    ConfigReader() = default;
    explicit ConfigReader(JsonType config_): _config_(std::move(config_)) {}

    // setters
    void setConfig(const JsonType& config_){ _config_ = config_; }

    // const getters
    [[nodiscard]] std::string toString() const{ return GenericToolbox::Json::toReadableString( _config_ ); }
    [[nodiscard]] const JsonType &getConfig() const{ return _config_; }

    // mutable getters
    JsonType &getConfig(){ return _config_; }

    // read options
    [[nodiscard]] bool hasKey(const std::string& keyPath_) const{ return GenericToolbox::Json::doKeyExist(_config_, keyPath_); }
    ConfigReader fetchSubConfig(const std::string& keyPath_) const;
    [[nodiscard]] std::string getUnusedOptionsMessage() const;


    // templates
    template<typename T> T fetchValue(const std::vector<std::string>& keyPathList_) const; // source

    // nested template
    template<typename T> T fetchValue(const std::vector<std::string>& keyPathList_, const T& defaultValue_) const{ try{ return fetchValue<T>(keyPathList_); } catch( ... ) { return defaultValue_; } }
    template<typename T> void fillValue(T& object_, const std::vector<std::string> &keyPathList_) const{ try{ object_ = this->fetchValue<T>(keyPathList_); } catch(...){} }
    template<typename T> void fillEnum(T& enum_, const std::vector<std::string>& keyPath_) const;

    // nested template (string to vector<string>)
    template<typename T> T fetchValue(const std::string& keyPath_) const{ return this->fetchValue<T>(std::vector<std::string>({keyPath_})); }
    template<typename T> T fetchValue(const std::string& keyPath_, const T& defaultValue_) const{ return fetchValue(std::vector<std::string>({keyPath_}), defaultValue_); }
    template<typename T> void fillValue(T& object_, const std::string& keyPath_) const{ fillValue(object_, std::vector<std::string>({keyPath_})); }
    template<typename T> void fillEnum(T& enum_, const std::string& keyPath_) const{ fillEnum(enum_, std::vector<std::string>({keyPath_})); }

    friend std::ostream& operator <<( std::ostream& o, const ConfigReader& this_ ){ o << this_.toString(); return o; }

  private:
    JsonType _config_{};

    // keep track of fields that have been red
    mutable std::vector<std::string> _usedKeyList_{};

  };

  template<typename T> T ConfigReader::fetchValue(const std::vector<std::string>& keyPathList_) const{
    // keyPathList_ has all the possible names for a given option
    // the first one is the official one, when others are set a message will appear telling the user it's deprecated
    T out;
    bool hasBeenFound{false};
    for( auto& keyPath : keyPathList_ ){
      try{
        T temp = GenericToolbox::Json::fetchValue<T>(_config_, keyPath);
        // pass this point it won't return an error

        // tag the found option
        GenericToolbox::addIfNotInVector(keyPath, _usedKeyList_);

        // already found?
        if( hasBeenFound ){
          // check if the two values are matching
          if(out != temp){
            LogError << "\"" << keyPath << "\" has a different value than \"" << _usedKeyList_[_usedKeyList_.size()-2] << "\"" << std::endl;
            LogError << this->toString() << std::endl;
            LogExit("Two config options with different values.");
          }
        }
        else{
          if( keyPath != keyPathList_.front() ) {
            // printing the deprecation only if not already found -> could be an old option present for compatibility
            LogAlert << "\"" << keyPath << "\" is a deprecated field name, use \"" << keyPathList_.front() << "\" instead." << std::endl;
          }

          out = temp;
          hasBeenFound = true;
        }


      }
      catch(...){}
    }

    if( not hasBeenFound ) {
      // let this one return the error
      GenericToolbox::Json::fetchValue<T>(_config_, keyPathList_);
      throw std::logic_error("SHOULD NOT GET THERE. CALL A DEV");
    }

    return out;
  }
  template<typename T> void ConfigReader::fillEnum(T& enum_, const std::vector<std::string>& keyPathList_) const{
    std::string enumName;
    this->fillValue(enumName, keyPathList_);
    if( enumName.empty() ){ return; }
    enum_ = enum_.toEnum( enumName, true );
  }

}

#endif //GUNDAM_CONFIG_UTILS_H
