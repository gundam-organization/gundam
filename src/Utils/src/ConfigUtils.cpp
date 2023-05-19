//
// Created by Adrien Blanchet on 28/02/2023.
//

#include "ConfigUtils.h"

#include "GenericToolbox.Json.h"
#include "GenericToolbox.Yaml.h"
#include "Logger.h"

#include "nlohmann/json.hpp"


LoggerInit([]{
  Logger::setUserHeaderStr("[ConfigUtils]");
} );


namespace ConfigUtils {

  nlohmann::json readConfigFile(const std::string& configFilePath_){
    if( not GenericToolbox::doesPathIsFile(configFilePath_) ){
      LogError << "\"" << configFilePath_ << "\" could not be found." << std::endl;
      throw std::runtime_error("file not found.");
    }

    nlohmann::json output;

    if( GenericToolbox::doesFilePathHasExtension(configFilePath_, "yml")
        or GenericToolbox::doesFilePathHasExtension(configFilePath_,"yaml")
        ){
      output = ConfigUtils::convertYamlToJson(configFilePath_);
    }
    else{
      output = GenericToolbox::Json::readConfigFile(configFilePath_);
    }

    return output;
  }

  nlohmann::json convertYamlToJson(const std::string& configFilePath_){
    return ConfigUtils::convertYamlToJson(GenericToolbox::Yaml::readConfigFile(configFilePath_));
  }
  nlohmann::json convertYamlToJson(const YAML::Node& yaml){
    nlohmann::json output = nlohmann::json::parse(GenericToolbox::Yaml::toJsonString(yaml));

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

    std::function<void(nlohmann::json&)> recursiveFix;
    recursiveFix = [&recursiveFix, is_number, is_numeric](nlohmann::json& jsonEntry_){

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

  nlohmann::json getForwardedConfig(const nlohmann::json& config_){
    nlohmann::json out = config_;
    while( out.is_string() ){
      out = ConfigUtils::readConfigFile(out.get<std::string>());
    }
    return out;
  }
  nlohmann::json getForwardedConfig(const nlohmann::json& config_, const std::string& keyName_){
    return ConfigUtils::getForwardedConfig(GenericToolbox::Json::fetchValue<nlohmann::json>(config_, keyName_));
  }
  void forwardConfig(nlohmann::json& config_, const std::string& className_){
    while( config_.is_string() ){
//      LogDebug << "Forwarding " << (className_.empty()? "": className_ + " ") << "config: \"" << config_.get<std::string>() << "\"" << std::endl;
      auto name = config_.get<std::string>();
      std::string expand = GenericToolbox::expandEnvironmentVariables(name);
      config_ = ConfigUtils::readConfigFile(expand);
    }
  }
  void unfoldConfig(nlohmann::json& config_){
    for( auto& entry : config_ ){
      if( entry.is_string() and (
          GenericToolbox::doesStringEndsWithSubstring(entry.get<std::string>(), ".yaml", true)
          or GenericToolbox::doesStringEndsWithSubstring(entry.get<std::string>(), ".json", true)
      ) ){
        ConfigUtils::forwardConfig(entry);
        ConfigUtils::unfoldConfig(config_); // remake the loop on the unfolder config
        break; // don't touch anymore
      }

      if( entry.is_structured() ){
        ConfigUtils::unfoldConfig(entry);
      }
    }
  }

}
