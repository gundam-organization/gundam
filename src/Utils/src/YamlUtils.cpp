//
// Created by Nadrino on 17/06/2021.
//

#include "Logger.h"

#include "GenericToolbox.h"
#include "YamlUtils.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[YamlUtils]");
});


YAML::Node YamlUtils::readConfigFile(const std::string &configFilePath_) {

  if( not GenericToolbox::doesPathIsFile(configFilePath_) ){
    LogError << "\"" << configFilePath_ << "\" could not be found." << std::endl;
    throw std::runtime_error("file not found.");
  }

  return YAML::LoadFile(configFilePath_);
}
nlohmann::json YamlUtils::toJson(const YAML::Node& yamlConfig_){
  YAML::Emitter emitter;
  emitter << YAML::DoubleQuoted << YAML::Flow << YAML::BeginSeq << yamlConfig_;
  std::string asJsonStr = std::string(emitter.c_str() + 1);

  auto output = nlohmann::json::parse(asJsonStr);

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

      std::string value = jsonEntry_.get<std::string>();
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
