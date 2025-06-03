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
    struct FieldDefinition{
      enum Flag: uint8_t{
        DEFAULT    = 0b00000000,
        MANDATORY  = 0b00000001,
        RELOCATED  = 0b00000010,
        DEPRECATED = 0b00000100,
      };

      uint8_t flags{0x0};
      std::string name{};
      std::string message{};
      std::vector<std::string> altNameList{};

      FieldDefinition() = default;
      FieldDefinition(uint8_t flags_, std::string name_, std::vector<std::string> altList_ = {}, std::string message_={})
        : flags(flags_), name(std::move(name_)), message(std::move(message_)), altNameList(std::move(altList_)) {}
      FieldDefinition(uint8_t flags_, std::string name_, std::initializer_list<const char*> altList_, std::string message_={})
        : flags(flags_), name(std::move(name_)), message(std::move(message_)){
        altNameList.reserve(altList_.size());
        for(const char* s : altList_){ altNameList.emplace_back(s); }
      }
      FieldDefinition(uint8_t flags_, std::string name_, std::string message_)
        : flags(flags_), name(std::move(name_)), message(std::move(message_)){}
      FieldDefinition(std::string name_, std::vector<std::string> altList_ = {}, std::string message_={})
        : name(std::move(name_)), message(std::move(message_)), altNameList(std::move(altList_)){}

      [[nodiscard]] bool isMandatory() const  { return (flags & Flag::MANDATORY)  != 0; }
      [[nodiscard]] bool isRelocated() const  { return (flags & Flag::RELOCATED)  != 0; }
      [[nodiscard]] bool isDeprecated() const { return (flags & Flag::DEPRECATED) != 0; }

      friend std::ostream& operator<< (std::ostream& stream, const FieldDefinition& obj_){ stream << obj_.toString(); return stream; }
      [[nodiscard]] std::string toString() const;
    };

    ConfigReader() = default;
    explicit ConfigReader(JsonType config_): _config_(std::move(config_)) {}

    // setters
    void setConfig(const JsonType& config_){ _config_ = config_; }
    void setParentPath(const std::string& parentPath_){ _parentPath_ = parentPath_; }

    // const getters
    [[nodiscard]] const JsonType &getConfig() const{ return _config_; }
    [[nodiscard]] const std::string& getParentPath() const{ return _parentPath_; }

    // mutable getters
    JsonType &getConfig(){ return _config_; }

    // define fields
    void clearFields(){ _fieldDefinitionList_.clear(); _definedFieldNameList_.clear(); }
    void defineField(const FieldDefinition& fieldDefinition_);
    void defineFields(const std::vector<FieldDefinition>& fieldDefinition_);
    void checkConfiguration() const;
    const FieldDefinition& getFieldDefinition(const std::string& fieldName_) const;
    std::pair<std::string, const JsonType*> getConfigEntry(const FieldDefinition& field_) const;
    std::pair<std::string, const JsonType*> getConfigEntry(const std::string& fieldName_) const;

    // read options
    [[nodiscard]] bool empty() const{ return _config_.empty(); }
    [[nodiscard]] bool hasField(const std::string& fieldName_) const;
    const JsonType* getJsonEntry(const std::string& key_) const; // perform key registration if exists + case insensitive
    [[nodiscard]] std::string toString(bool shallow_=false) const{ return GenericToolbox::Json::toReadableString( _config_, shallow_ ); }

    void printUnusedKeys() const;

    // for loops
    std::vector<ConfigReader> loop() const;
    std::vector<ConfigReader> loop(const std::string& fieldName_) const;

    friend std::ostream& operator <<( std::ostream& o, const ConfigReader& this_ ){ o << this_.toString(); return o; }

    // new templates
    template<typename T> T fetchValue(const std::string& fieldName_) const;
    template<typename T> T fetchValue(const std::string& fieldName_, const T& default_) const;
    template<typename T> void fillValue(T& object_, const std::string& fieldName_) const;
    template<typename T> void fillEnum(T& enum_, const std::string& fieldName_) const;
    template<typename F> void deprecatedAction(const std::string& fieldName_, const F& action_) const;

    void fillFormula(std::string& formulaToFill_, const std::string& fieldName_, const std::string& joinStr_) const;

  protected:
    std::string getStrippedParentPath() const;
    bool doShowWarning(const std::string& key_) const;

  private:
    std::string _parentPath_{"/"};
    JsonType _config_;

    // defining fields before reading the config
    std::vector<FieldDefinition> _fieldDefinitionList_{};

    // check for field name collisions
    std::unordered_set<std::string> _definedFieldNameList_{};

    // keep track of fields that have been red in runtime
    mutable std::unordered_set<std::string> _usedKeyList_{};

    // handling printing only once
    static std::vector<std::string> _deprecatedList_;

  };

  // inline definitions
  template<typename T> T ConfigReader::fetchValue(const std::string& fieldName_) const{
    auto* jsonField = getConfigEntry(fieldName_).second;
    if(jsonField == nullptr){
      throw std::runtime_error("Could not get field value \"" + fieldName_ + "\" in config " + toString());
      return {};
    }
    return GenericToolbox::Json::get<T>(*jsonField);
  }
  template<typename T> T ConfigReader::fetchValue(const std::string& fieldName_, const T& default_) const{
    auto out = default_;
    fillValue(out, fieldName_);
    return out;
  }
  template<> inline ConfigReader ConfigReader::fetchValue<ConfigReader>(const std::string& fieldName_) const{
    auto keyValuePair = getConfigEntry(fieldName_);
    if(keyValuePair.second == nullptr){
      throw std::runtime_error("Could not get field value \"" + fieldName_ + "\" in config " + toString());
      return {};
    }
    auto out = ConfigReader(GenericToolbox::Json::get<JsonType>(*keyValuePair.second));
    // using a config key so users can retrieve the path in their config
    out.setParentPath(GenericToolbox::joinPath(_parentPath_, keyValuePair.first));
    return out;
  }
  template<> inline std::vector<ConfigReader> ConfigReader::fetchValue<std::vector<ConfigReader>>(const std::string& fieldName_) const{ return fetchValue<ConfigReader>(fieldName_).loop(); }
  template<typename T> void ConfigReader::fillValue(T& object_, const std::string& fieldName_) const{
    try{ object_ = fetchValue<T>(fieldName_); }
    catch(...){}
  }
  template<typename T> void ConfigReader::fillEnum(T& enum_, const std::string& fieldName_) const{
    std::string enumName;
    this->fillValue(enumName, fieldName_);
    if( enumName.empty() ){ return; }
    enum_ = enum_.toEnum( enumName, true );
  }

  template<typename F> void ConfigReader::deprecatedAction(const std::string& fieldName_, const F& action_) const {
    auto& field = getFieldDefinition(fieldName_);
    auto entry = getConfigEntry(field);
    if( entry.second != nullptr ) {
      if( doShowWarning(fieldName_) ){
        LogAlert << _parentPath_ << ": \"" << fieldName_ << "\" should be set under \"" << field.message << "\"" << std::endl;
      }
      action_();
    }
  }

}


typedef ConfigUtils::ConfigReader ConfigReader;
typedef ConfigUtils::ConfigReader::FieldDefinition::Flag FieldFlag;
typedef GenericToolbox::ConfigClass<ConfigReader> JsonBaseClass;


#endif //GUNDAM_CONFIG_UTILS_H
