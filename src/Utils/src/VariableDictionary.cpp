//
// Created by Adrien BLANCHET on 10/11/2022.
//

#include "VariableDictionary.h"
#include "ConfigUtils.h"

#include "Logger.h"

#include "GenericToolbox.Json.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[VariableDictionary]");
});

VariableDictEntry::VariableDictEntry(const nlohmann::json& config_){
  this->readConfig(config_);
}

void VariableDictEntry::readConfig(const nlohmann::json& config_){
  auto config = ConfigUtils::getForwardedConfig(config_);

  // Mandatory
  this->name = GenericToolbox::Json::fetchValue<std::string>(config, "name");

  // Defaulted
  this->displayName = this->name;

  // Optionals
  this->displayName = GenericToolbox::Json::fetchValue(config, "displayName", this->displayName);
  this->unit = GenericToolbox::Json::fetchValue(config, "unit", this->unit);
  this->description = GenericToolbox::Json::fetchValue(config, "description", this->description);
}


void VariableDictionary::fillDictionary(const nlohmann::json& config_, bool overrideIfDefined_){
  auto config = ConfigUtils::getForwardedConfig(config_);

  auto entryList = config.get<std::vector<nlohmann::json>>();
  dictionary.reserve( dictionary.size() + entryList.size() ); // max size

  for( auto& dictEntry : entryList ){
    if( this->isVariableDefined( GenericToolbox::Json::fetchValue<std::string>(dictEntry, "name") ) and overrideIfDefined_ ){
      this->getEntry( GenericToolbox::Json::fetchValue<std::string>(dictEntry, "name") ).readConfig(dictEntry);
    }
    else{
      dictionary.emplace_back( dictEntry );
    }
  }
}

bool VariableDictionary::isVariableDefined(const std::string& variableName_) const{
  auto it = std::find_if(dictionary.begin(), dictionary.end(), [&](const VariableDictEntry& v_){ return v_.name == variableName_; });
  return it != dictionary.end();
}
const VariableDictEntry& VariableDictionary::getEntry(const std::string& variableName_) const{
  auto it = std::find_if(dictionary.begin(), dictionary.end(), [&](const VariableDictEntry& v_){ return v_.name == variableName_; });
  LogThrowIf( it == dictionary.end(), "Could not find " << variableName_ << " in the dictionary." );
  return *it;
}
VariableDictEntry& VariableDictionary::getEntry(const std::string& variableName_){
  auto it = std::find_if(dictionary.begin(), dictionary.end(), [&](const VariableDictEntry& v_){ return v_.name == variableName_; });
  LogThrowIf( it == dictionary.end(), "Could not find " << variableName_ << " in the dictionary." );
  return *it;
}