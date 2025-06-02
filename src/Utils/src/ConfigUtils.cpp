//
// Created by Adrien Blanchet on 28/02/2023.
//

#include "ConfigUtils.h"
#include "GundamUtils.h"

#include "GenericToolbox.Root.h"
#include "GenericToolbox.Yaml.h"
#include "Logger.h"

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <map>

namespace ConfigUtils {

  // static
  std::vector<std::string> ConfigReader::_deprecatedList_{};

  // open file
  JsonType readConfigFile(const std::string& configFilePath_){
    if( not GenericToolbox::isFile(configFilePath_) ){
      LogError << "\"" << configFilePath_ << "\" could not be found." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    JsonType output;

    try{
      if( GenericToolbox::hasExtension(configFilePath_, {{"yaml"}, {"yml"}}) ){
        output = ConfigUtils::convertYamlToJson( configFilePath_ );
      }
      else{
        output = GenericToolbox::Json::readConfigFile(configFilePath_);
      }
    }
    catch(...){
        LogError << "Error while reading config file: " << configFilePath_
                 << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // resolve subreferences to other config files
    ConfigUtils::unfoldConfig( output );

    return output;
  }

  // YAML to JSON converting
  JsonType convertYamlToJson(const std::string& configFilePath_){
    return ConfigUtils::convertYamlToJson(GenericToolbox::Yaml::readConfigFile(configFilePath_));
  }
  JsonType convertYamlToJson(const YAML::Node& yaml){
    JsonType output = JsonType::parse(GenericToolbox::Yaml::toJsonString(yaml));

    auto is_number = [](const std::string& s){
      return !s.empty() && std::find_if(s.begin(),
                                        s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
    };
    auto is_numeric = [](std::string const & str){
      auto result = double();
      auto i = std::istringstream(str);
      i >> result;
      return !i.fail() && i.eof();
    };

    std::function<void(JsonType&)> recursiveFix;
    recursiveFix = [&recursiveFix, is_number, is_numeric](JsonType& jsonEntry_){

      if( jsonEntry_.is_null() ){
        return;
      }
      else if(jsonEntry_.is_array() or jsonEntry_.is_structured()){
        for( auto &jsonSubEntry : jsonEntry_ ){
          recursiveFix(jsonSubEntry);
        }
      }
      else if(jsonEntry_.is_string()){

        auto value = jsonEntry_.get<std::string>();
        if( value == "true" ){
          jsonEntry_ = true;
        }
        else if( value == "false" ) {
          jsonEntry_ = false;
        }
        else if( is_number(value) ){
          jsonEntry_ = std::stoi(value);
        }
        else if( is_numeric(value) ){
          jsonEntry_ = std::stod(value);
        }

      }
    };

    recursiveFix(output);
    return output;
  }

  // unfolding
  JsonType getForwardedConfig(const JsonType& config_){
    JsonType out = config_;
    while( out.is_string() ){
      out = ConfigUtils::readConfigFile(out.get<std::string>());
    }
    return out;
  }

  void forwardConfig(JsonType& config_){
    while( config_.is_string() and
         ( GenericToolbox::endsWith(config_.get<std::string>(), ".yaml", true)
        or GenericToolbox::endsWith(config_.get<std::string>(), ".json", true) )
        ){
      auto name = config_.get<std::string>();
      std::string expand = GenericToolbox::expandEnvironmentVariables(name);
      config_ = ConfigUtils::readConfigFile(expand);
    }
  }

  void unfoldConfig(JsonType& config_){

    std::function<void(JsonType&)> unfoldRecursive = [&](JsonType& outEntry_){
      for( auto& entry : config_ ){
        if( entry.is_string() and (
               GenericToolbox::endsWith(entry.get<std::string>(), ".yaml", true)
            or GenericToolbox::endsWith(entry.get<std::string>(), ".json", true)
        ) ){
          ConfigUtils::forwardConfig( entry );
          ConfigUtils::unfoldConfig( config_ ); // remake the loop on the unfolder config
          break; // don't touch anymore
        }

        if( entry.is_structured() ){ ConfigUtils::unfoldConfig( entry ); }
      }
    };
    unfoldRecursive(config_);

  }

  // class impl
  void ConfigBuilder::setConfig(const std::string& filePath_){
    if( GenericToolbox::hasExtension( filePath_, "root" ) ){
      LogInfo << "Extracting config file for fitter file: " << filePath_ << std::endl;
      if (not GenericToolbox::doesTFileIsValid(filePath_)) {
        LogError << "Invalid root file: " << filePath_ << std::endl;
        std::exit(EXIT_FAILURE);
      }
      auto fitFile = std::shared_ptr<TFile>( GenericToolbox::openExistingTFile( filePath_ ) );

      auto* conf = fitFile->Get<TNamed>("gundam/config_TNamed");
      if( conf == nullptr ){
        // legacy
        conf = fitFile->Get<TNamed>("gundamFitter/unfoldedConfig_TNamed");
      }
      if (conf == nullptr) {
        LogError << "No config in ROOT file " << filePath_ << std::endl;
        std::exit(EXIT_FAILURE);
      }
      _config_ = GenericToolbox::Json::readConfigJsonStr( conf->GetTitle() );
      fitFile->Close();
    }
    else{
      LogInfo << "Reading config file: " << filePath_ << std::endl;
      _config_ = ConfigUtils::readConfigFile(filePath_ ); // works with yaml
    }
  }

  void ConfigBuilder::override( const JsonType& overrideConfig_ ){
    LogScopeIndent;
    LogWarning << GenericToolbox::Json::applyOverrides(_config_, overrideConfig_);
  }
  void ConfigBuilder::override( const std::string& filePath_ ){
    LogInfo << "Overriding config with \"" << filePath_ << "\"" << std::endl;
    if (not GenericToolbox::isFile(filePath_)) {
      LogError << "Could not find " << filePath_ << std::endl;
      std::exit(EXIT_FAILURE);
    }
    this->override( ConfigUtils::readConfigFile(filePath_) );
  }
  void ConfigBuilder::override( const std::vector<std::string>& filesList_ ){
    for( auto& file : filesList_ ){ this->override( file ); }
  }
  void ConfigBuilder::flatOverride( const std::string& flattenEntry_ ){
    // Override the configuration values.  If the old value was a string then
    // replace with the new string. Otherwise, the input value is parsed.  The
    // configuration value are references like path names
    // (e.g. /fitterEngineConfig/mcmcConfig/steps to change the MCMC interface
    // "steps" value.)  This is intended to make minor changes to the behavior,
    // so for sanity's sake, the key must already exist in the configuration
    // files (if the key does not exist an exception will be thrown).  The
    // command line syntax to change the number of mcmc steps to 1000 per cycle
    // would be
    //
    // gundamFitter.exe -O /fitterEngineConfig/mcmcConfig/steps=1000 ...
    //

    std::vector<std::string> split = GenericToolbox::splitString( flattenEntry_,"=" );
    LogWarning << "Override " << split[0] << " with " << split[1]
               << std::endl;
    JsonType flat = _config_.flatten();
    LogWarning << "    Original value: " << flat.at(split[0])
               << std::endl;
    if (flat.at(split[0]).is_string()) flat.at(split[0]) = split[1];
    else flat.at(split[0]) = JsonType::parse(split[1]);
    LogWarning << "         New value: " << flat.at(split[0])
               << std::endl;
    _config_ = flat.unflatten();
  }
  void ConfigBuilder::flatOverride( const std::vector<std::string>& flattenEntryList_ ){
    for( auto& flattenEntry : flattenEntryList_ ){ this->flatOverride( flattenEntry ); }
  }
  void ConfigBuilder::exportToJsonFile(const std::string &filePath_) const {
    auto outPath{filePath_};

    if( not GenericToolbox::endsWith(outPath, ".json") ){
      // add extension if missing
      outPath += ".json";
    }

    LogInfo << "Writing as: " << outPath << std::endl;
    GenericToolbox::dumpStringInFile(outPath, this->toString());
    LogInfo << "Unfolded config written as: " << outPath << std::endl;
  }

  std::string ConfigReader::FieldDefinition::toString() const{
    std::stringstream ss;
    ss << "name=" << name;
    if(isMandatory()) ss << ", isMandatory";
    if(isDeprecated()) ss << ", isDeprecated";
    if(isRelocated()) ss << ", isRelocated";
    if(not altNameList.empty()) ss << ", altNameList=" << GenericToolbox::toString(altNameList);
    return ss.str();
  }

  void ConfigReader::defineField(const FieldDefinition& fieldDefinition_){
    // Check collision on name
    LogThrowIf(
      not _definedFieldNameList_.insert(GenericToolbox::toLowerCase(fieldDefinition_.name)).second,
      "[DEV] Collision on name: " << fieldDefinition_.name
    );
    for( auto& altName : fieldDefinition_.altNameList ){
      LogThrowIf(
        not _definedFieldNameList_.insert(GenericToolbox::toLowerCase(altName)).second,
        "[DEV] Collision on altname: " << altName
      );
    }
    _fieldDefinitionList_.emplace_back(fieldDefinition_);
  }
  void ConfigReader::defineFields(const std::vector<FieldDefinition>& fieldDefinition_){
    for(auto& field : fieldDefinition_){ defineField(field); }
  }
  void ConfigReader::checkConfiguration() const{

    // 1 - check for missing compulsory fields
    std::vector<std::string> missingFieldList{};
    for( auto& field : _fieldDefinitionList_ ){
      if( not field.isMandatory ){ continue; }
      auto* entry = getConfigEntry(field).second;
      if(entry == nullptr){ missingFieldList.emplace_back(field.name); }
    }
    if( not missingFieldList.empty() ){
      LogError << _parentPath_ << ": found " << missingFieldList.size() << " missing compulsory fields." << std::endl;
      for( auto& missingField : missingFieldList ){
        LogError << "  > \"" << missingField << "\" is a mandatory field." << std::endl;
      }
      LogError << toString() << std::endl;
      LogExit("Invalid configuration");
    }

    // 2 - check if some config keys have some collisions
    // 2.1 - if they do, do they carry the same value?
    // -> should allow collisions since some configs are meant to be handled by older versions of GUNDAM as well
    std::map<const FieldDefinition*, std::vector<std::string>> collisionDict{};
    for( auto& field : _fieldDefinitionList_ ){
      std::vector<std::string> keyCollisionList{};
      if( _config_.contains(field.name) ){ keyCollisionList.emplace_back(field.name); }
      for( auto& altFieldName : field.altNameList ){
        if( _config_.contains(altFieldName) ){ keyCollisionList.emplace_back(altFieldName); }
      }
      if( keyCollisionList.size() >= 2 ){
        collisionDict[&field] = keyCollisionList;
      }
    }
    if( not collisionDict.empty() ){
      // check if they carry the same value? -> if NOT -> ERROR
      bool unmatchingCollisionFound = false;
      for( const auto& collision : collisionDict ){
        auto val = _config_.at(collision.second[0]);
        for( auto& key : collision.second ){
          if(_config_.at(key) != val){
            unmatchingCollisionFound = true;
            LogError << _parentPath_ << ": found unmatching values for field \"" << collision.first->name << "\". Make sure they have the same value." << std::endl;
          }
          else{
            LogAlertIf(doShowWarning(collision.first->name)) << _parentPath_ << ": field \"" << collision.first->name << "\" has collisions with different keys: " << GenericToolbox::toString(collision.second) << std::endl;
          }
        }
      }
      if( unmatchingCollisionFound ){ LogExit("Invalid configuration"); }
    }

    // 3 - look for invalid key names
    // just a warning
    std::vector<std::string> invalidKeyList{};
    for(auto it = _config_.begin(); it != _config_.end(); ++it){
      if(not GenericToolbox::isIn(GenericToolbox::toLowerCase(it.key()), _definedFieldNameList_)){
        // already printed out? regardless of indexed path
        if( not doShowWarning(it.key()) ){ continue; }
        invalidKeyList.emplace_back(it.key());
      }
    }
    if( not invalidKeyList.empty() ){
      for( auto& invalidKey : invalidKeyList ){
        LogAlert << _parentPath_ << ": key \"" << invalidKey << "\" has an invalid name. It won't be recognized by GUNDAM." << std::endl;
      }
    }

  }
  const ConfigReader::FieldDefinition& ConfigReader::getFieldDefinition(const std::string& fieldName_) const{
    for(const auto& field : _fieldDefinitionList_){
      if( field.name == fieldName_ ){ return field; }
    }

    LogError << "[DEV] (" << _parentPath_ << ") Unknown field name \"" << fieldName_ << "\" among list: " << GenericToolbox::toString(_fieldDefinitionList_) << std::endl;
    exit(EXIT_FAILURE);
  }
  std::pair<std::string, const JsonType*> ConfigReader::getConfigEntry(const FieldDefinition& field_) const{
    auto temp = getJsonEntry(field_.name);
    if( temp != nullptr ){ return {field_.name, temp}; }
    for( auto& altKeyName : field_.altNameList ){
      temp = getJsonEntry(altKeyName);
      if( temp != nullptr ){
        printDeprecatedMessage(altKeyName, field_.name);
        return {altKeyName, temp};
      }
    }
    return {"", nullptr};
  }
  std::pair<std::string, const JsonType*> ConfigReader::getConfigEntry(const std::string& fieldName_) const{
    auto& field = getFieldDefinition(fieldName_);
    return getConfigEntry(field);
  }

  const JsonType* ConfigReader::getJsonEntry(const std::string& key_) const{
    const JsonType* current = &_config_;
    auto keys = GenericToolbox::splitString(key_, "/");

    for(const auto& subKey : keys){
      if(not current->is_object()){ return nullptr; }

      bool found = false;
      for(auto it = current->begin(); it != current->end(); ++it){
        if(strcasecmp(it.key().c_str(), subKey.c_str()) == 0){
          current = &it.value();
          found = true;
          break;
        }
      }
      if(not found){ return nullptr; }
    }

    _usedKeyList_.insert(GenericToolbox::toLowerCase(key_));
    return current;
  }
  bool ConfigReader::hasField(const std::string& fieldName_) const{
    return getConfigEntry(fieldName_).second != nullptr;
  }
  void ConfigReader::fillFormula(std::string& formulaToFill_, const std::string& fieldName_, const std::string& joinStr_) const{
    if( not hasField(fieldName_) ){ return; }
    formulaToFill_ = GenericToolbox::joinVectorString(
      fetchValue<std::vector<std::string>>(fieldName_),
      joinStr_
    );
  }
  void ConfigReader::printUnusedKeys() const{
    // for context dependent options
    std::vector<std::string> unusedKeyList{};
    for (auto it = _config_.begin(); it != _config_.end(); ++it) {
      if( GenericToolbox::isIn(GenericToolbox::toLowerCase(it.key()), _usedKeyList_) ){ continue; }

      // already printed out?
      if( doShowWarning(it.key()) ){ unusedKeyList.emplace_back(it.key()); }
    }

    if( not unusedKeyList.empty() ){
      for( auto& unusedKey : unusedKeyList ){
        LogAlert << _parentPath_ << ": key \"" << unusedKey << "\" was ignored while parsing config. Is is context dependent option?" << std::endl;
      }
    }
  }
  std::vector<ConfigReader> ConfigReader::loop() const {
    std::vector<ConfigReader> out;
    out.reserve( _config_.size() );
    for( auto& entry : _config_ ) {
      out.emplace_back(entry);
      out.back().setParentPath(GenericToolbox::joinPath(_parentPath_, out.size()-1));
    }
    return out;
  }
  std::vector<ConfigReader> ConfigReader::loop(const std::string& fieldName_) const{
    ConfigReader c;
    fillValue(c, fieldName_);
    return c.loop();
  }

  std::string ConfigReader::getStrippedParentPath() const{
    std::string out{_parentPath_};
    out.erase(
      std::remove_if(
        out.begin(), out.end(),
        [](char c) { return std::isdigit(static_cast<unsigned char>(c)); }
        ),
      out.end()
    );

    GenericToolbox::removeRepeatedCharInsideInputStr(out, "/");
    return out;
  }
  void ConfigReader::printDeprecatedMessage(const std::string& oldKey_, const std::string& newKey_) const {
    // only print it once
    if( doShowWarning(oldKey_) ) {
      LogWarning << _parentPath_ << ": key \"" << oldKey_ << "\" is deprecated. Use \"" << newKey_ << "\" instead." << std::endl;
    }
  }
  bool ConfigReader::doShowWarning(const std::string& key_) const{
    std::string referenceStr{GenericToolbox::joinPath(getStrippedParentPath(), key_)};
    if( not GenericToolbox::isIn(referenceStr, _deprecatedList_) ){
      _deprecatedList_.emplace_back(referenceStr);
      return true;
    }
    return false;
  }

}
